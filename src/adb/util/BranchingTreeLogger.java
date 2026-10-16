package adb.util;

import adb.tree.AnnotatedNode;
import adb.tree.AnnotatedTree;
import beast.base.core.BEASTObject;
import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.core.Loggable;
import beast.base.evolution.tree.Node;

import java.io.PrintStream;

/**
 * Logs the tree without the hidden events, one Newick line per sample. The node type is written as
 * metadata "type" when printMetaData is true.
 */
@Description("Logs an annotated tree as a standard branching tree, without hidden events.")
public class BranchingTreeLogger extends BEASTObject implements Loggable {

    public Input<AnnotatedTree> annotatedTreeInput =
            new Input<>("annotatedTree", "annotated tree to log as standard branching tree", Input.Validate.REQUIRED);

    public Input<Boolean> printMetaDataInput =
            new Input<>("printMetaData", "print node metadata (including type) (default true)", true);

    AnnotatedTree tree;
    boolean printMetaData;

    @Override
    public void initAndValidate() {
        tree = annotatedTreeInput.get();
        printMetaData = printMetaDataInput.get();
    }

    @Override
    public void init(PrintStream out) {
        tree.init(out);
    }

    @Override
    public void log(long nSample, PrintStream out) {

        if (printMetaData) {
            // set metadata string
            for (Node node : tree.getNodesAsArray()) {
                node.metaDataString = String.format("type=%d", ((AnnotatedNode)node).getType());
            }
        }
        out.print("tree STATE_" + nSample + " = ");
        out.print(tree.getRoot().toSortedNewick(new int[1], printMetaData));
        out.print(";");
    }

    @Override
    public void close(PrintStream out) {
        tree.close(out);
    }

}
