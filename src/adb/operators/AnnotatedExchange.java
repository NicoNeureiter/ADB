package adb.operators;

import adb.tree.AnnotatedNode;
import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.util.InputUtil;
import beast.base.util.Randomizer;

/**
 * Adapted from the Exchange operator of BEAST. The narrow version swaps a random node with its uncle,
 * the wide version swaps two random nodes whose branches overlap in time. Every node keeps the events of its own branch.
 * Because the parent heights are exchanged, the branches change length: their type changes are mapped onto
 * the new length (with their Jacobian), and the hidden events between them are resampled.
 * The type of the lineage at the top of a branch is not changed, so a move can make the division at one of the
 * two parents impossible. Such moves have zero likelihood and are rejected right away.
 */
@Description("Exchange move for annotated trees: swaps subtrees while every node keeps its own branch events")
public class AnnotatedExchange extends AnnotatedTreeOperator {

    public Input<Boolean> isNarrowInput = new Input<>("isNarrow", "if true (default) a narrow exchange is performed, otherwise a wide exchange", true);

    @Override
    public double proposal() {

        final Tree tree = (Tree) InputUtil.get(treeInput, this);

        double logHR;

        if (isNarrowInput.get()) {
            logHR = narrow(tree);
        } else {
            logHR = wide(tree);
        }

        return logHR;
    }

    private int isg(final Node n) {
        return (n.getLeft().isLeaf() && n.getRight().isLeaf()) ? 0 : 1;
    }

    private int sisg(final Node n) {
        return n.isLeaf() ? 0 : isg(n);
    }


    public double narrow(final Tree tree) {

        double logHR = 0;

        final int internalNodes = tree.getInternalNodeCount();
        if (internalNodes <= 1) {
            return Double.NEGATIVE_INFINITY;
        }

        Node grandParent = tree.getNode(internalNodes + 1 + Randomizer.nextInt(internalNodes));
        while (grandParent.getLeft().isLeaf() && grandParent.getRight().isLeaf()) {
            grandParent = tree.getNode(internalNodes + 1 + Randomizer.nextInt(internalNodes));
        }

        Node parentIndex = grandParent.getLeft();
        Node uncle = grandParent.getRight();
        if (parentIndex.getHeight() < uncle.getHeight()) {
            parentIndex = grandParent.getRight();
            uncle = grandParent.getLeft();
        }

        int validGP = 0;
        {
            for (int i = internalNodes + 1; i < 1 + 2*internalNodes; ++i) {
                validGP += isg(tree.getNode(i));
            }
        }

        if (uncle.getHeight() >= parentIndex.getHeight()) {
            return Double.NEGATIVE_INFINITY; // uncle would get a branch of length 0
        }

        int c2 = sisg(parentIndex) + sisg(uncle);

        Node i = (Randomizer.nextBoolean() ? parentIndex.getLeft() : parentIndex.getRight());

        logHR += exchangeBranches(i, uncle, parentIndex, grandParent);

        int validGPafter = validGP - c2 + sisg(parentIndex) + sisg(uncle);
        logHR += Math.log((float)validGP/validGPafter);

        return logHR;
    }


    public double wide(final Tree tree) {

        double logHR = 0;

        final int nodeCount = tree.getNodeCount();
        Node i = tree.getRoot();
        while (i.isRoot()) {
            i = tree.getNode(Randomizer.nextInt(nodeCount));
        }

        Node j = i;
        while (j.getNr() == i.getNr() || j.isRoot()) {
            j = tree.getNode(Randomizer.nextInt(nodeCount));
        }

        final Node p = i.getParent();
        final Node jP = j.getParent();

        if ((p != jP) && (i != jP) && (j != p)
                && (j.getHeight() < p.getHeight())
                && (i.getHeight() < jP.getHeight())) {

            logHR += exchangeBranches(i, j, p, jP);

            // all the nodes on the path from i/j to the common ancestor of i/j parents had a topology change,
            // so they need to be marked FILTHY.
            if (markCladesInput.get()) {
                Node iup = p;
                Node jup = jP;
                while (iup != jup) {
                    if( iup.getHeight() < jup.getHeight() ) {
                        assert !iup.isRoot();
                        iup = iup.getParent();
                        iup.makeDirty(Tree.IS_FILTHY);
                    } else {
                        assert !jup.isRoot();
                        jup = jup.getParent();
                        jup.makeDirty(Tree.IS_FILTHY);
                    }
                }
            }

            return logHR;
        }

        // randomly selected nodes i and j are not valid candidates for a wide exchange.
        // reject instead of counting (like we do for narrow)
        return Double.NEGATIVE_INFINITY;
    }


    /**
     * Exchanges the subtrees i (child of p) and j (child of jP). Each subtree keeps the events on its own branch:
     * the type changes are mapped onto the new branch length, and the hidden events between them are resampled.
     * Returns the log of the Jacobian and of the ratio of the branch resampling densities.
     */
    private double exchangeBranches(Node i, Node j, Node p, Node jP) {

        // after the exchange p has children j and its other child, jP has children i and its other child
        if (!isDivisionPossible(p, j, getOtherChild(p, i)) || !isDivisionPossible(jP, i, getOtherChild(jP, j))) {
            return Double.NEGATIVE_INFINITY;
        }

        double iOldBranchProb = getBranchProbability((AnnotatedNode)i, true);
        double jOldBranchProb = getBranchProbability((AnnotatedNode)j, true);
        double iOldTop = p.getHeight();
        double jOldTop = jP.getHeight();

        // precondition p -> i & jP -> j
        replace(p, i, j);
        replace(jP, j, i);
        // postcondition p -> j & jP -> i

        double logHR = scaleEvents((AnnotatedNode)i, iOldTop, jOldTop) + scaleEvents((AnnotatedNode)j, jOldTop, iOldTop);
        logHR += iOldBranchProb - resampleEvents((AnnotatedNode)i, true);
        logHR += jOldBranchProb - resampleEvents((AnnotatedNode)j, true);
        return logHR;
    }

}
