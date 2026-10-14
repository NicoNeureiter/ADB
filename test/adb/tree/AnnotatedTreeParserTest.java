package adb.tree;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

public class AnnotatedTreeParserTest {

    static final String TREE_1 = "(1[&type=0]:1.0,2[&type=0]:1.0)[&type=0]:0.5;";
    static final String TREE_2 = "((1[&type=0]:1.0)[&type=0]:0.5,2[&type=0]:1.5)[&type=0]:0.5;";

    @TempDir
    Path dir;

    private String file() throws IOException {
        // trees spread over several lines with indentation, as in files written by other programs
        Path file = dir.resolve("trees.newick");
        Files.writeString(file, TREE_1 + "\n  " + TREE_2.substring(0, 10) + "\n  " + TREE_2.substring(10) + "\n");
        return file.toString();
    }

    private AnnotatedTreeParser parse(Object... inputs) {
        AnnotatedTreeParser parser = new AnnotatedTreeParser();
        parser.initByName(inputs);
        return parser;
    }

    @Test
    public void testNewick() {
        AnnotatedTreeParser parser = parse("newick", TREE_1);
        assertEquals(2, parser.getLeafNodeCount());
        assertEquals(1.0, parser.getRoot().getHeight(), 1e-12);
    }

    @Test
    public void testFileSelectsTreeByIndex() throws IOException {
        String file = file();
        assertEquals(2, parse("fileName", file).getLeafNodeCount());
        AnnotatedTreeParser second = parse("fileName", file, "treeIndex", 2);
        assertEquals(1.5, second.getRoot().getHeight(), 1e-12);
        // the hidden event on the branch above tip 1 is kept
        assertEquals(2, ((AnnotatedNode) second.getNode(0)).getEventCount()
                + ((AnnotatedNode) second.getNode(1)).getEventCount() - 1);
    }

    @Test
    public void testNeitherOrBothInputsFail() throws IOException {
        assertThrows(RuntimeException.class, () -> parse());
        assertThrows(RuntimeException.class, () -> parse("newick", TREE_1, "fileName", file()));
    }

    @Test
    public void testInvalidTreeIndexFails() throws IOException {
        String file = file();
        assertThrows(IllegalArgumentException.class, () -> AnnotatedTreeParser.readTreeFromFile(file, 0));
        assertThrows(IllegalArgumentException.class, () -> AnnotatedTreeParser.readTreeFromFile(file, 3));
        // initByName wraps the exception
        assertThrows(RuntimeException.class, () -> parse("fileName", file, "treeIndex", 0));
    }

    @Test
    public void testMissingFileFails() {
        assertThrows(RuntimeException.class, () -> parse("fileName", dir.resolve("none.newick").toString()));
    }

    @Test
    public void testFileInputIsNotCopiedToNewick() throws IOException {
        assertNull(parse("fileName", file()).newickInput.get());
    }
}
