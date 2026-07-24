import java.util.Arrays;
import java.util.List;

public class Main {

    public static void main(String[] args) {
        List<Product> products = Arrays.asList(
                new Product(1, "Food", null),
                new Product(3, "Bread", 1),
                new Product(4, "Ciabatta", 3),
                new Product(5, "Alcohol", 2),
                new Product(6, "White Wine", 5),
                new Product(7, "Red Wine", 5),
                new Product(8, "Non-alcohol", 2),
                new Product(9, "Coca-cola", 8),
                new Product(10, "Fanta", 8),
                new Product(11, "Baguette", 3),
                new Product(12, "Personal Care", null),
                new Product(13, "Shampoo", 12),
                new Product(14, "Nivea", 13),
                new Product(2, "Drinks", null));

        List<Product> roots = ProductHierarchy.buildForest(products);

        for (Product root : roots) {
            ProductHierarchy.printTree(root, 0);
            System.out.println();
        }
    }
}
