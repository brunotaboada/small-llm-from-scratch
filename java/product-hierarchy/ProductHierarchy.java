import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public final class ProductHierarchy {

    private ProductHierarchy() {}

    /**
     * Builds a forest of product trees from a flat list. Two passes, no sorting.
     * List order is preserved among siblings (append order when linking children).
     */
    public static List<Product> buildForest(List<Product> products) {
        Map<Integer, Product> byId = new HashMap<>();
        for (Product product : products) {
            byId.put(product.getId(), product);
        }

        List<Product> roots = new ArrayList<>();
        for (Product product : products) {
            Integer parentId = product.getParentId();
            if (parentId == null) {
                roots.add(product);
                continue;
            }
            Product parent = byId.get(parentId);
            if (parent == null) {
                throw new IllegalArgumentException(
                        "Unknown parent id " + parentId + " for product " + product.getId());
            }
            parent.getChildren().add(product);
        }
        return roots;
    }

    public static void printTree(Product node, int depth) {
        System.out.println("  ".repeat(depth) + node.getName());
        for (Product child : node.getChildren()) {
            printTree(child, depth + 1);
        }
    }
}
