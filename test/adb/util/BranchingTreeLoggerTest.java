package adb.util;

import adb.tree.AnnotatedNode;
import adb.tree.AnnotatedTree;
import adb.tree.AnnotatedTreeParser;
import beast.base.evolution.tree.Node;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;

import static org.junit.jupiter.api.Assertions.*;

public class BranchingTreeLoggerTest {

    private String log(AnnotatedTree tree, boolean metaData) {
        BranchingTreeLogger logger = new BranchingTreeLogger();
        logger.initByName("annotatedTree", tree, "printMetaData", metaData);
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        logger.log(7, new PrintStream(bytes));
        return bytes.toString();
    }

    // logging must not touch the live tree: same checksum
    @Test
    public void testLoggingLeavesTreeUnchanged() {
        AnnotatedTreeParser tree = new AnnotatedTreeParser();
        tree.initByName("newick", "((1[&type=0]:0.4,2[&type=1]:0.4)[&type=0]:0.3,3[&type=1]:0.7)[&type=0]:1;");
        int checksum = tree.getChecksum();

        String out = log(tree, true);

        assertTrue(out.startsWith("tree STATE_7 = ("), out);
        assertTrue(out.contains("type=0") && out.contains("type=1"), out);
        assertEquals(checksum, tree.getChecksum());
        assertEquals(3, out.split(",").length);
    }

    @Test
    public void testNoMetaData() {
        AnnotatedTreeParser tree = new AnnotatedTreeParser();
        tree.initByName("newick", "((1:0.4,2:0.4):0.3,3:0.7):1;");
        assertFalse(log(tree, false).contains("type"));
    }

    // hidden events (here a type change on the branch above tip 2) are dropped: only branching nodes remain,
    // the branch keeps its full length, and root and tip types are logged
    @Test
    public void testHiddenEventsDropped() {
        AnnotatedTreeParser tree = new AnnotatedTreeParser();
        tree.initByName("newick", "((1[&type=0]:0.4,(2[&type=1]:0.2)[&type=0]:0.2)[&type=0]:0.3,3[&type=1]:0.7)[&type=0]:1;");
        assertEquals(2, ((AnnotatedNode) tree.getNode(1)).getEventCount()); // tip 2 has the hidden type change

        String out = log(tree, true);
        assertEquals(2, out.chars().filter(c -> c == '(').count(), out); // two divisions, no single-child node
        assertEquals(5, out.split("type=").length - 1, out); // 3 tips, 1 internal node, root

        AnnotatedTreeParser logged = new AnnotatedTreeParser();
        logged.initByName("newick", out.substring(out.indexOf('=') + 1, out.lastIndexOf(';')).trim());
        assertEquals(5, logged.getNodeCount());
        for (Node node : logged.getNodesAsArray()) {
            assertEquals(1, ((AnnotatedNode) node).getEventCount());
        }
        assertEquals(0, ((AnnotatedNode) logged.getRoot()).getType());
        assertEquals(1, ((AnnotatedNode) logged.getNode(1)).getType()); // tip 2 keeps its own type
        assertEquals(0.4, logged.getNode(1).getParent().getHeight(), 1e-9); // full branch length
    }

    // each node is labelled with its own type (the type at the node, not the type above a hidden change)
    @Test
    public void testLabelMatchesNodeType() {
        AnnotatedTreeParser tree = new AnnotatedTreeParser();
        tree.initByName("newick", "((1[&type=0]:0.4,(2[&type=1]:0.2)[&type=0]:0.2)[&type=0]:0.3,3[&type=1]:0.7)[&type=0]:1;");

        log(tree, true);

        for (Node node : tree.getNodesAsArray()) {
            assertEquals("type=" + AnnotatedNode.getTypeOf(node), node.metaDataString, "node " + node.getNr());
        }
        assertEquals("type=1", tree.getNode(1).metaDataString); // tip 2 is type 1 although its branch ends in type 0
    }
}
