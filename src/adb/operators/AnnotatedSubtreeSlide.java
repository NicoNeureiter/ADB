package adb.operators;

import adb.tree.AnnotatedNode;
import adb.tree.AnnotatedNode.EventNode;
import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.util.InputUtil;
import beast.base.util.Randomizer;

import java.util.ArrayList;
import java.util.List;

/**
 * Adapted from the SubtreeSlide operator of BEAST. The parent p of a random node i slides along the hidden events
 * of the tree: by a random number of events (the window size) up, or down into the other child's branch,
 * possibly into a different branch, which changes the topology. The events passed on the way move
 * between the branches of p and its other child, so the ancestral type histories are kept.
 * The branch above i changes length: its type changes are mapped onto the new length (with their Jacobian),
 * and the hidden events between them are resampled.
 * Along the way a type change of the sliding lineage can become the type of the division at p, or the other way round.
 * Moves that make this division impossible have zero likelihood and are rejected right away.
 */
@Description("Subtree slide for annotated trees: moves the parent of a subtree along the hidden events of the tree")
public class AnnotatedSubtreeSlide extends AnnotatedTreeOperator {

    public Input<Integer> sizeInput = new Input<>("size", "the size of the sliding window up and down, default 1", 1);

    protected int size;


    @Override
    public void initAndValidate() {
        super.initAndValidate();
        size = sizeInput.get();
    }


    @Override
    public double proposal() {

        Tree tree = (Tree) InputUtil.get(treeInput, this);
        final boolean markClades = markCladesInput.get();

        double newHeight, oldHeight, oldBranchProb, newBranchProb, logHR = 0.0;

        Node i;
        // 1. choose a random node avoiding root
        final int nodeCount = tree.getNodeCount();
        if (nodeCount == 1) {
            // test for degenerate case
            return Double.NEGATIVE_INFINITY;
        }
        do {
            i = tree.getNode(Randomizer.nextInt(nodeCount));
        } while (i.isRoot());

        final Node p = i.getParent();
        final Node CiP = getOtherChild(p, i);
        final Node PiP = p.getParent();

        // 2. choose magnitude and direction of the move
        int delta;
        do {
            delta = Randomizer.nextInt(2 * size + 1) - size;
        } while (delta == 0);


        oldBranchProb = getBranchProbability((AnnotatedNode)i, true);
        oldHeight = p.getHeight();

        // store events from the old parent branch
        List<EventNode> pEvents = ((AnnotatedNode)p).getEvents();

        // 3. perform move
        // 3.1 if the move is up
        if (delta > 0) {

            if (delta == ((AnnotatedNode)p).getEventCount()) {
                // no shift possible
                return Double.NEGATIVE_INFINITY;

            // 3.1.1 if topology changes
            } else if (delta > ((AnnotatedNode)p).getEventCount()) {

                if (PiP == null) {
                    // no shift possible
                    return Double.NEGATIVE_INFINITY;
                }

                // find new parent
                Node newParent = PiP;
                Node newChild = p;

                int x = 0;
                for (int k = 0; k < delta; k++) {
                    x++;
                    if (x >= ((AnnotatedNode)newChild).getEventCount()) {
                        x = 0; // restore counter
                        newChild = newParent;
                        if (newChild == null) {
                            // no shift possible (no events remain)
                            return Double.NEGATIVE_INFINITY;
                        }
                        if (markClades) newParent.makeDirty(Tree.IS_FILTHY);
                        newParent = newParent.getParent();
                    }
                }

                if (x == 0) {
                    // no shift possible (target event is branching node)
                    return Double.NEGATIVE_INFINITY;
                }

                // get target event
                newHeight = ((AnnotatedNode)newChild).getEvent(x).getHeight();
                p.setHeight(newHeight);

                boolean isRoot = newChild.isRoot();

                // update topology and event lists
                replace(p, CiP, newChild);
                replace(PiP, p, CiP);
                ((AnnotatedNode)CiP).addEvents(new ArrayList<>(pEvents), true);
                List<EventNode> events = ((AnnotatedNode)newChild).getEvents().subList(x, ((AnnotatedNode)newChild).getEventCount());
                ((AnnotatedNode)p).setEvents(new ArrayList<>(events));
                events.clear();

                // creating a new root
                if (isRoot) {
                    p.setParent(null);
                    tree.setRoot(p);
                }

                // no new root
                else {
                    replace(newParent, newChild, p);
                }

                // count the hypothetical sources of this destination
                int possibleSources = intersectingEdges(newChild, delta, null);
                logHR += -Math.log(possibleSources);


            // 3.1.2 if topology does not change
            } else {
                newHeight = ((AnnotatedNode)p).getEvent(delta).getHeight();
                p.setHeight(newHeight);
                // copy events from parent branch to sibling
                List<EventNode> events = ((AnnotatedNode)p).getEvents().subList(0, delta);
                ((AnnotatedNode)CiP).addEvents(new ArrayList<>(events), true);
                events.clear();
            }


        // 3.2 if the move is down
        } else {

            if (Math.abs(delta) == ((AnnotatedNode)CiP).getEventCount()) {
                // no shift possible
                return Double.NEGATIVE_INFINITY;

            // 3.2.1 if topology changes
            } else if (Math.abs(delta) > ((AnnotatedNode)CiP).getEventCount()) {

                List<Node> newChildren = new ArrayList<>();
                int possibleDestinations = intersectingEdges(CiP, Math.abs(delta), newChildren);

                // if no valid destinations then return a failure
                if (newChildren.size() == 0) {
                    return Double.NEGATIVE_INFINITY;
                }

                // pick a random parent/child destination edge uniformly from options
                int childIndex = Randomizer.nextInt(newChildren.size());
                Node newChild = newChildren.get(childIndex);
                Node newParent = newChild.getParent();
                int eventIndex = eventIndexForDownMove(CiP, newChild, Math.abs(delta));
                newHeight = ((AnnotatedNode)newChild).getEvent(eventIndex).getHeight();

                // TODO: potentially also add some margin around i's height
                if (newHeight <= i.getHeight()) {
                    return Double.NEGATIVE_INFINITY;
                }

                // if p was root
                if (p.isRoot()) {
                    // new root is CiP
                    replace(p, CiP, newChild);
                    replace(newParent, newChild, p);

                    CiP.setParent(null);
                    tree.setRoot(CiP);

                } else {
                    replace(p, CiP, newChild);
                    replace(PiP, p, CiP);
                    replace(newParent, newChild, p);
                }

                // event lists
                ((AnnotatedNode)CiP).addEvents(new ArrayList<>(pEvents), true);
                List<EventNode> events = ((AnnotatedNode)newChild).getEvents().subList(eventIndex, ((AnnotatedNode)newChild).getEventCount());
                ((AnnotatedNode)p).setEvents(new ArrayList<>(events));
                events.clear();

                p.setHeight(newHeight);
                if (markClades) {
                    // make dirty the path from the (down) moved node back up to former parent.
                    Node n = p;
                    while(n != CiP) {
                        n.makeDirty(Tree.IS_FILTHY);
                        n = n.getParent();
                    }
                }

                logHR += Math.log(possibleDestinations);


            // 3.2.2 if topology does not change
            } else {
                int ix = ((AnnotatedNode)CiP).getEventCount() + delta; // delta is negative!
                newHeight = ((AnnotatedNode)CiP).getEvents().get(ix).getHeight();
                if (newHeight <= i.getHeight()) {
                    return Double.NEGATIVE_INFINITY;
                }
                p.setHeight(newHeight);

                // copy events from slide to parent branch, remove events from children
                List<EventNode> events = ((AnnotatedNode)CiP).getEvents().subList(ix, ((AnnotatedNode)CiP).getEventCount());
                ((AnnotatedNode)p).addEvents(new ArrayList<>(events), 0);
                events.clear();
            }
        }

        // the division at p has new daughters (the types at the top of both branches are unchanged)
        if (!isDivisionPossible(p, p.getLeft(), p.getRight())) {
            return Double.NEGATIVE_INFINITY;
        }

        // map the type changes on the i-branch to the new branch length, then resample the hidden events between them
        logHR += scaleEvents((AnnotatedNode)i, oldHeight, newHeight);
        newBranchProb = resampleEvents((AnnotatedNode)i, true);
        logHR += (oldBranchProb - newBranchProb);

        return logHR;
    }


    private int intersectingEdges(Node node, int eventCount, List<Node> directChildren) {
        if (node.getParent() == null) {
            return 0;
        }

        int nodeEventCount = ((AnnotatedNode)node).getEventCount();
        if (eventCount < nodeEventCount) {
            if (directChildren != null) directChildren.add(node);
            return 1;
        }

        if (eventCount == nodeEventCount || node.isLeaf()) {
            return 0;
        }

        int remainingEventCount = eventCount - nodeEventCount;
        return intersectingEdges(node.getLeft(), remainingEventCount, directChildren) +
                intersectingEdges(node.getRight(), remainingEventCount, directChildren);
    }


    private int eventIndexForDownMove(Node source, Node destination, int eventCount) {
        Node node = destination;
        int eventsBeforeDestination = ((AnnotatedNode)source).getEventCount();

        while (node.getParent() != source) {
            node = node.getParent();
            eventsBeforeDestination += ((AnnotatedNode)node).getEventCount();
        }

        int eventsOnDestination = eventCount - eventsBeforeDestination;
        return ((AnnotatedNode)destination).getEventCount() - eventsOnDestination;
    }

}
