import java.util.ArrayList;
import java.util.List;

public class Product {
    private final int id;
    private final String name;
    private final Integer parentId;
    private final List<Product> children = new ArrayList<>();

    public Product(int id, String name, Integer parentId) {
        this.id = id;
        this.name = name;
        this.parentId = parentId;
    }

    public int getId() {
        return id;
    }

    public String getName() {
        return name;
    }

    public Integer getParentId() {
        return parentId;
    }

    public List<Product> getChildren() {
        return children;
    }

    @Override
    public String toString() {
        return name;
    }
}
