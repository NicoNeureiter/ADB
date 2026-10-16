package adb.operators;

import adb.distribution.Parameterization;
import adb.tree.AnnotatedNode;
import adb.tree.AnnotatedNode.EventNode;
import adb.tree.AnnotatedTree;
import adb.util.Utils;
import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.operator.TreeOperator;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.parameter.RealParameter;
import beast.base.util.Randomizer;
import org.apache.commons.math.special.Gamma;
import org.apache.commons.math3.distribution.GammaDistribution;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


/**
 * Base class for operators that change an AnnotatedTree together with its hidden events.
 * Along a branch the events are the node itself, the type changes and the hidden events of the same type.
 * The topology operators keep the type changes of a branch (mapped to the new branch length, see scaleEvents)
 * and redraw the hidden events between them with resampleEvents. The number of hidden events per type segment
 * is approximated by a discretised normal distribution (central limit theorem for renewal processes),
 * and their positions follow a Dirichlet distribution on the segment, with the shape of the lifetime distribution
 * as concentration. A leaf sampled at time 0 is drawn exactly backwards in time from the lifetime distribution.
 * resampleEvents returns the log density of the draw and getBranchProbability that of the current events,
 * so the difference of the two enters the log Hastings ratio.
 */
@Description("Base class of operators that propose changes to an annotated tree and its hidden events")
public abstract class AnnotatedTreeOperator extends TreeOperator {

    public Input<Parameterization> parameterizationInput =
            new Input<>("parameterization", "ADB parameterization", Input.Validate.REQUIRED);

    protected Parameterization parameterization;


    @Override
    public void initAndValidate() {

        Tree tree = treeInput.get();
        if (!(tree instanceof AnnotatedTree)) {
            throw new IllegalArgumentException("Attempted to initialise annotated tree operator with regular tree.");
        }

        parameterization = parameterizationInput.get();
    }


    /**
     * Whether the model allows the division of the parent into the two daughters, given the type at the parent
     * and the type at the bottom of each daughter's branch. Moves that create an impossible division have zero
     * likelihood, so rejecting them early only saves the cost of resampling and of the likelihood.
     */
    protected boolean isDivisionPossible(Node parent, Node left, Node right) {
        return parameterization.getDivisionFactor(AnnotatedNode.getTypeOf(parent),
                ((AnnotatedNode) left).getInitialType(), ((AnnotatedNode) right).getInitialType()) > 0.0;
    }

    /**
     * Redraws the hidden events above the node and returns the log proposal density of this draw.
     * The node itself and the type changes stay where they are, and the hidden events of each type segment
     * are drawn between them: the segments end at the next type change or at the parent
     * (at the origin for the root, and for the root without origin nothing is drawn).
     * If drawEventCount is false the number of events per segment is kept, otherwise it is drawn as well.
     * The branch must have a positive length and its events must be sorted.
     */
    protected double resampleEvents(AnnotatedNode node, boolean drawEventCount) {

        double logp = 0;

        double tMax;
        if (node.isRoot()) {
            if (parameterization.getOriginTime() == null) {
                node.getEvent(0).setHeight(node.getHeight()); // keep the node event in sync with the node height
                return 0.0; // no stem: the root has no hidden events, so there is nothing to draw
            } else {
                tMax = parameterization.getOriginTime();
            }
        } else {
            tMax = node.getParent().getHeight();
        }

        int nTotal = node.getEventCount();
        double start = node.getHeight();
        int type = node.getType();
        ArrayList<EventNode> events = new ArrayList<>();
        events.add(new EventNode(type, start));

        int ix = 1;
        int nEvents = 1;
        double end;
        while (ix < nTotal) {
            if (node.getEvent(ix).getType() == type) { // no type change
                nEvents++;
            } else {
                end = node.getEvent(ix).getHeight();
                logp += sampleEventsPerType(node, start, end, type, nEvents, drawEventCount, events);
                // reset counters
                type = node.getEvent(ix).getType();
                start = node.getEvent(ix).getHeight();
                nEvents = 1;
            }
            ix++;
        }
        logp += sampleEventsPerType(node, start, tMax, type, nEvents, drawEventCount, events);

        node.setEvents(events);

        // return probability
        return logp;
    }


    // Draws the hidden events of one type segment [start, end) into events and returns the log density of the draw.
    // A leaf starting at height 0 draws waiting times backwards from end, and the remaining time contributes the survival probability.
    // Otherwise the event count comes from the normal approximation and the positions from the Dirichlet distribution.
    private double sampleEventsPerType(AnnotatedNode node, double start, double end, int type, int nEvents, boolean drawEventCount, ArrayList<EventNode> events) {

        double logp = 0;

        double lifetime = parameterization.getLifetime(type);
        double shape = parameterization.getShape(type);
        double scale = lifetime / shape;

        if (node.isLeaf() && start == 0.0) {
            GammaDistribution gammaDist = new GammaDistribution(null, shape, scale);
            double t = end;
            double waitingTime;
            while (t > start) {
                // determine the next hidden event
                waitingTime = Randomizer.nextGamma(shape, 1/scale);
                if (t - waitingTime > start) {
                    t -= waitingTime;
                    // add new hidden event
                    events.add(1, new EventNode(type, t));
                    logp += gammaDist.logDensity(waitingTime);
                } else {
                    // calculate probability of remaining time
                    logp += Math.log(1 - gammaDist.cumulativeProbability(t));
                    break;
                }
            }

        } else {
            if (start > node.getHeight()) {
                events.add(new EventNode(type, start)); // copy initial event
            }

            double length = end - start;
            int eventCount;
            if (drawEventCount) {
                // use CLT for renewal processes: approximate number of events with discretized normal distribution
                double mean = length / lifetime;
                double variance = ((shape * (scale * scale)) * length) / Math.pow(lifetime, 3);
                double sd = Math.sqrt(variance);
                eventCount = Utils.sampleDiscretizedNormal(mean, sd);
                logp += Utils.logProbabilityDiscretizedNormal(eventCount, mean, sd);
            } else {
                eventCount = nEvents;
            }

            if (eventCount > 1) {
                double[] segmentLengths = new double[eventCount];
                Arrays.fill(segmentLengths, length/eventCount);
                // distribute
                logp += Utils.distributeDirichletProbability(segmentLengths, shape);
                double t = start;
                for (int i = 0; i < eventCount - 1; i++) { // last segment corresponds to time until next branching event
                    t += segmentLengths[i];
                    events.add(new EventNode(type, t));
                }
            }
        }

        return logp;
    }


    /**
     * Log density with which resampleEvents would draw the current hidden events above the node
     * (the same segments and the same distributions, but evaluated instead of sampled).
     */
    protected double getBranchProbability(AnnotatedNode node, boolean drawEventCount) {

        double logp = 0;

        double tMax;
        if (node.isRoot()) {
            if (parameterization.getOriginTime() == null) {
                return 0.0;
            } else {
                tMax = parameterization.getOriginTime();
            }
        } else {
            tMax = node.getParent().getHeight();
        }

        int nTotal = node.getEventCount();
        double start = node.getHeight();
        int type = node.getType();

        int ix = 1;
        int nEvents = 1;
        double end;
        int ixTypeChange = 0;
        while (ix < nTotal) {
            if (node.getEvent(ix).getType() == type) { // no type change
                nEvents++;
            } else {
                end = node.getEvent(ix).getHeight();
                logp += getProbabilityPerType(node, ixTypeChange, start, end, type, nEvents, drawEventCount);
                // reset counters
                type = node.getEvent(ix).getType();
                start = node.getEvent(ix).getHeight();
                nEvents = 1;
                ixTypeChange = ix;
            }
            ix++;
        }
        logp += getProbabilityPerType(node, ixTypeChange, start, tMax, type, nEvents, drawEventCount);

        // return probability
        return logp;
    }


    // Log density of the hidden events of one type segment, starting at event index ix, mirrors sampleEventsPerType
    private double getProbabilityPerType(AnnotatedNode node, int ix, double start, double end, int type, int eventCount, boolean drawEventCount) {

        double logp = 0;

        double lifetime = parameterization.getLifetime(type);
        double shape = parameterization.getShape(type);
        double scale = lifetime / shape;

        double waitingTime;

        if (node.isLeaf() && start == 0.0) {
            GammaDistribution gammaDist = new GammaDistribution(null, shape, scale);
            if (eventCount == 1) {
                waitingTime = end;
            } else {
                for (int i = 2; i < eventCount; i++) {
                    waitingTime = node.getEvent(i).getHeight() - node.getEvent(i - 1).getHeight();
                    logp += gammaDist.logDensity(waitingTime);
                }
                waitingTime = end - node.getEvent(eventCount - 1).getHeight();
                logp += gammaDist.logDensity(waitingTime);
                // remainder
                waitingTime = node.getEvent(1).getHeight();
            }
            logp += Math.log(1 - gammaDist.cumulativeProbability(waitingTime));

        } else {
            double length = end - start;
            if (drawEventCount) {
                double mean = length / lifetime;
                double variance = ((shape * (scale * scale)) * length) / Math.pow(lifetime, 3);
                double sd = Math.sqrt(variance);
                logp += Utils.logProbabilityDiscretizedNormal(eventCount, mean, sd);
            }
            double logDirichlet = 0;
            for (int i = ix + 1; i < ix + eventCount; i++) {
                waitingTime = node.getEvent(i).getHeight() - node.getEvent(i - 1).getHeight();
                logDirichlet += Math.log(waitingTime / length);
            }
            waitingTime = end - node.getEvent(ix + eventCount - 1).getHeight();
            logDirichlet += Math.log(waitingTime / length);
            logp += -(eventCount - 1) * Math.log(length) +
                    Gamma.logGamma(eventCount * shape) - eventCount * Gamma.logGamma(shape) + (shape - 1) * logDirichlet;
        }

        return logp;
    }


    /**
     * Maps the events above node i from the branch [i.height, oldTop] affinely onto [i.height, newTop],
     * keeping the node itself fixed. Used by the topology operators when the parent of i changes height.
     * Only the type changes need to be kept (the other hidden events are resampled afterwards), so the return value
     * is the log Jacobian of this deterministic map for the m type changes: m * log(scale).
     * It is 0 for a single type, and the branch is then fully described by resampleEvents.
     */
    protected static double scaleEvents(AnnotatedNode i, double oldTop, double newTop) {
        return scaleEvents(i, i.getHeight(), oldTop, i.getHeight(), newTop, false);
    }

    /**
     * Maps the events above node i affinely from the branch [oldBase, oldTop] onto [newBase, newTop], where newBase
     * is the new height of i, so the order of the events and their branch are kept. Returns the log Jacobian
     * m * log(scale) of this deterministic map, where m counts the type changes, or all events above the node if
     * allEvents is true (when none of them is resampled afterwards).
     */
    protected static double scaleEvents(AnnotatedNode i, double oldBase, double oldTop, double newBase, double newTop,
                                        boolean allEvents) {
        double scale = (newTop - newBase) / (oldTop - oldBase);
        i.getEvent(0).setHeight(newBase);

        int mapped = 0;
        for (int e = 1; e < i.getEventCount(); e++) {
            EventNode event = i.getEvent(e);
            event.setHeight(newBase + (event.getHeight() - oldBase) * scale);
            if (allEvents || event.getType() != i.getEvent(e - 1).getType()) {
                mapped++;
            }
        }
        return mapped == 0 ? 0.0 : mapped * Math.log(scale);
    }

    /**
     * Multiplies entry index of every parameter by factor, or every entry if index is -1. Returns the share of the
     * log Hastings ratio of a scale move with a kernel symmetric on the log scale, log(factor) per scaled entry,
     * or minus infinity if a new value is outside the bounds of its parameter.
     */
    protected static double scaleParameters(List<RealParameter> parameters, int index, double factor) {
        int count = 0;
        for (RealParameter param : parameters) {
            int from = index < 0 ? 0 : index;
            int to = index < 0 ? param.getDimension() : index + 1;
            for (int i = from; i < to; i++) {
                double newValue = param.getValue(i) * factor;
                if (newValue < param.getLower() || newValue > param.getUpper()) return Double.NEGATIVE_INFINITY;
                param.setValue(i, newValue);
                count++;
            }
        }
        return count * Math.log(factor);
    }

}
