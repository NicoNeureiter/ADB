package adb.operators;

import adb.tree.AnnotatedNode;
import adb.tree.AnnotatedNode.EventNode;
import beast.base.core.Description;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.util.InputUtil;
import beast.base.util.Randomizer;

import java.util.ArrayList;
import java.util.List;

/**
 * Adapted from the Wilson-Balding operator of BEAST. It cuts the branch above a random node i and
 * re-attaches it at a hidden event of another branch j that is older than i. The new parent of i takes over
 * the part of branch j above the attachment event, and the old parent is replaced by its other child.
 * Its events are merged into that child's branch, so the ancestral type histories are kept.
 * The Hastings ratio counts the available attachment events in both directions.
 * The branch above i changes length: its type changes are mapped onto the new length (with their Jacobian),
 * and the hidden events between them are resampled.
 * The move is rejected if the division at the attachment point is impossible for the model.
 */
@Description("Wilson-Balding move for annotated trees: re-attaches a subtree at a hidden event of another branch")
public class AnnotatedWilsonBalding extends AnnotatedTreeOperator {

    @Override
    public double proposal() {

        Tree tree = (Tree) InputUtil.get(treeInput, this);

        double oldHeight, newHeight, oldMinAge, newMinAge, oldBranchProb, newBranchProb, logHR;
        int oldEventsCount, newEventsCount, oldMinNr, newMinNr, newEventNr;

        // choose a random node avoiding root
        final int nodeCount = tree.getNodeCount();
        Node i;
        do {
            i = tree.getNode(Randomizer.nextInt(nodeCount));
        } while (i.isRoot());
        final Node p = i.getParent();

        // choose another random node to insert i above
        Node j;
        Node jP;
        // make sure that the target branch <k, j> is above the subtree being moved
        do {
            j = tree.getNode(Randomizer.nextInt(nodeCount));
            jP = j.getParent();
        } while ((jP != null && jP.getHeight() <= i.getHeight()) || (i.getNr() == j.getNr()));

        // disallow moves that change the root
        if (j.isRoot() || p.isRoot()) {
            return Double.NEGATIVE_INFINITY;
        }

        assert jP != null;  // j != root tested above
        final int pnr = p.getNr();
        final int jPnr = jP.getNr();
        if (jPnr == pnr || j.getNr() == pnr || jPnr == i.getNr()) {
            return Double.NEGATIVE_INFINITY;
        }

        final Node CiP = getOtherChild(p, i);

        // forward move: count available positions
        newMinAge = Math.max(i.getHeight(), j.getHeight());
        newMinNr = ((AnnotatedNode)j).findEventAbove(newMinAge);
        newEventsCount = ((AnnotatedNode)j).getEventCount() - newMinNr;

        // backward move: count available positions (assuming the forward move has happened)
        oldMinAge = Math.max(i.getHeight(), CiP.getHeight());
        oldMinNr = ((AnnotatedNode)CiP).findEventAbove(oldMinAge);
        oldEventsCount = ((AnnotatedNode)CiP).getEventCount() - oldMinNr + ((AnnotatedNode)p).getEventCount();

        if (newEventsCount == 0 || oldEventsCount == 0) {
            // no move
            return Double.NEGATIVE_INFINITY;
        }

        logHR = Math.log((double) newEventsCount / oldEventsCount);

        // calculate current branch probability
        oldBranchProb = getBranchProbability((AnnotatedNode)i, true);
        // record height and type of event
        oldHeight = p.getHeight();
        int iEvent = ((AnnotatedNode)i).getInitialType();

        // sample event to attach to
        newEventNr = newMinNr + Randomizer.nextInt(newEventsCount);
        newHeight = ((AnnotatedNode)j).getEvents().get(newEventNr).getHeight();
        int jEvent = ((AnnotatedNode)j).getEvents().get(newEventNr).getType();

        if (parameterization.getNTypes() > 1) {
            // the division at the attachment point must be possible: parent type jEvent,
            // daughters j (type below the attachment point) and i
            // TODO: alternative - resample i-branch conditional on j-event
            int jBelow = ((AnnotatedNode)j).getEvents().get(newEventNr - 1).getType();
            if (parameterization.getDivisionFactor(jEvent, jBelow, iEvent) == 0.0) {
                return Double.NEGATIVE_INFINITY;
            }
        }

        // disconnect p
        final Node pP = p.getParent();
        replace(pP, p, CiP);
        // re-attach, first child node to p
        replace(p, CiP, j);
        // then parent node of j to p
        replace(jP, j, p);

        // mark paths to common ancestor as changed
        if (markCladesInput.get()) {
            Node iup = pP;
            Node jup = p;
            while (iup != jup) {
                if (iup.getHeight() < jup.getHeight()) {
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

        // update height
        p.setHeight(newHeight);

        // update events
        List<EventNode> pEvents = ((AnnotatedNode)p).getEvents();
        ((AnnotatedNode)CiP).addEvents(new ArrayList<>(pEvents), true);
        // keep events below cut for node j and transfer events above cut to node p
        List<EventNode> events = ((AnnotatedNode)j).getEvents().subList(newEventNr, ((AnnotatedNode)j).getEventCount());
        ((AnnotatedNode)p).setEvents(new ArrayList<>(events));
        events.clear();

        // map the events on the i-branch to the new branch length, then resample the hidden events between type changes
        logHR += scaleEvents((AnnotatedNode)i, oldHeight, newHeight);

        // resample events on the branch
        newBranchProb = resampleEvents((AnnotatedNode)i, true);
        logHR += (oldBranchProb - newBranchProb);

        return logHR;
    }
}
