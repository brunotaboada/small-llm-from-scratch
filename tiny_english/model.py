"""
TinyEnglishGPT — decoder-only transformer with course-matching weight names.

Parameter names mirror the algo.monster NumPy loader:
  token_embed.weight, pos_embed.weight,
  blocks.{i}.attn.W_{q,k,v,o}.weight,
  blocks.{i}.ff.linear{1,2}.{weight,bias},
  blocks.{i}.norm{1,2}.{weight,bias},
  ln_f.{weight,bias}, lm_head.weight
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from tiny_english.vocab import PAD_ID, VOCAB_SIZE


@dataclass
class TinyConfig:
    vocab_size: int = VOCAB_SIZE
    max_seq_len: int = 16
    n_layers: int = 2
    n_heads: int = 4
    d_model: int = 32
    d_ff: int = 128
    dropout: float = 0.1
    pad_token_id: int = PAD_ID


class MultiHeadAttention(nn.Module):
    """Multi-head causal self-attention with separate Q/K/V/O projections."""

    def __init__(self, cfg: TinyConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.d_model = cfg.d_model

        # Named to match the course .npz export / NumPy loader
        self.W_q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_k = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_v = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_o = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        mask = torch.triu(
            torch.ones(cfg.max_seq_len, cfg.max_seq_len), diagonal=1
        ).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        scale = math.sqrt(self.d_head)
        scores = (q @ k.transpose(-2, -1)) / scale
        scores = scores.masked_fill(
            self.causal_mask[:T, :T].unsqueeze(0).unsqueeze(0), float("-inf")
        )
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.W_o(out)
        return self.resid_dropout(out)


class FeedForward(nn.Module):
    def __init__(self, cfg: TinyConfig):
        super().__init__()
        self.linear1 = nn.Linear(cfg.d_model, cfg.d_ff)
        self.linear2 = nn.Linear(cfg.d_ff, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.linear2(F.gelu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block (LayerNorm → sublayer → residual)."""

    def __init__(self, cfg: TinyConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ff = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class TinyEnglishGPT(nn.Module):
    """
    Minimal GPT matching the course NumPy inference interface.

    Modules are nested so state_dict keys look like:
      token_embed.weight
      blocks.0.attn.W_q.weight
      blocks.0.ff.linear1.weight
      ln_f.weight
      lm_head.weight
    """

    def __init__(self, cfg: TinyConfig | None = None):
        super().__init__()
        self.cfg = cfg or TinyConfig()
        c = self.cfg

        self.token_embed = nn.Embedding(
            c.vocab_size, c.d_model, padding_idx=c.pad_token_id
        )
        self.pos_embed = nn.Embedding(c.max_seq_len, c.d_model)
        self.drop = nn.Dropout(c.dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(c) for _ in range(c.n_layers)]
        )
        self.ln_f = nn.LayerNorm(c.d_model)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)

        # Weight tying (optional but helpful for tiny models)
        self.lm_head.weight = self.token_embed.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def forward(
        self, input_ids: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            input_ids: (B, T) token ids
            targets:   (B, T) next-token ids (PAD ignored in loss)

        Returns:
            logits: (B, T, vocab_size)
            loss:   scalar or None
        """
        B, T = input_ids.shape
        assert T <= self.cfg.max_seq_len

        positions = torch.arange(T, device=input_ids.device)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=self.cfg.pad_token_id,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 3,
        temperature: float = 0.0,
        end_id: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive generation. temperature=0 → greedy."""
        self.eval()
        for _ in range(max_new_tokens):
            # Crop to context window
            idx_cond = input_ids[:, -self.cfg.max_seq_len :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # (B, vocab)

            if temperature <= 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_id], dim=1)
            if end_id is not None and (next_id == end_id).all():
                break
        return input_ids

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
