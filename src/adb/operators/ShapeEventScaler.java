package adb.operators;

import adb.tree.AnnotatedNode;
import adb.tree.AnnotatedNode.EventNode;
import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.operator.kernel.KernelDistribution;
import beast.base.inference.parameter.RealParameter;
import beast.base.inference.util.InputUtil;
import beast.base.util.Randomizer;

import java.util.ArrayList;
import java.util.List;

/**
 * Scales the shape k of the lifetime distribution (must be a RealParameter in parameterization)
 * by f and scales the hidden events so that they stay typical under the new shape:
 * the lifetimes have a coefficient of variation of 1/sqrt(k), so every waiting time w of an event moves
 * towards or away from the regular length by c = sqrt(k/k') = 1/sqrt(f), w' = regular + c (w - regular).
 * Node heights, type changes, event counts, and the truncated lifetimes at tips stay the same.
 * If a waiting time becomes negative (only possible when k decreases), the move fails.
 * For multi-type models, one type j is chosen uniformly, and only entry j of the shape and the segments of type j move.
 * The map is affine with factor c on the m free event positions of every segment, and its inverse is the same move
 * with f replaced by 1/f, which the Bactrian kernel (symmetric on the log scale) proposes with the same density.
 * With M the number of moved hidden events, the log Hastings ratio is log f + M log c = (1 - M/2) log f.
 */
@Description("Scales the shape of the lifetime distribution and moves the hidden events towards or away from regular spacing, so that they stay typical under the new shape")
public class ShapeEventScaler extends AnnotatedTreeOperator {

    public Input<Double> scaleFactorInput = new Input<>("scaleFactor",
            "window of the log scale factor: the scale factor is exp(scaleFactor * x) with x from the Bactrian kernel (default 0.1)", 0.1);

    public Input<Boolean> optimiseInput = new Input<>("optimise",
            "flag to indicate that the scale factor is automatically changed in order to achieve a good acceptance rate (default true)", true);


    // limits of the tuned scale factor, as in UpDownEventSampler
    static final double MIN_SCALE_FACTOR = 1e-8, MAX_SCALE_FACTOR = 10.0;

    final KernelDistribution kernel = new KernelDistribution.Bactrian(KernelDistribution.Bactrian.mode.bactrian_normal);
    RealParameter shape;
    double scaleFactor;


    @Override
    public void initAndValidate() {
        super.initAndValidate();
        scaleFactor = scaleFactorInput.get();
        // the operator scales the shape that the model reads, so that the events and the shape move together
        if (!(parameterization.getShapeParameter() instanceof RealParameter)) {
            throw new IllegalArgumentException("ShapeEventScaler needs a real shape in the parameterization, but the shape "
                    + "is an integer parameter, which cannot be scaled continuously");
        }
        shape = (RealParameter) parameterization.getShapeParameter();
    }


    @Override
    public double proposal() {

        Tree tree = (Tree) InputUtil.get(treeInput, this);
        int nTypes = parameterization.getNTypes();
        int type = nTypes > 1 ? Randomizer.nextInt(nTypes) : 0;

        // scale the shape of the chosen type, symmetric on the log scale
        double f = kernel.getScaler(0, Double.NaN, scaleFactor);
        double logHR = scaleParameters(List.of(shape), type, f);
        if (logHR == Double.NEGATIVE_INFINITY) return logHR;

        // move the hidden events of the segments of this type
        double c = 1.0 / Math.sqrt(f);
        int moved = 0;
        for (Node node : tree.getNodesAsArray()) {
            int m = scaleDeviations((AnnotatedNode) node, type, c);
            if (m < 0) return Double.NEGATIVE_INFINITY;
            moved += m;
        }
        return logHR + moved * Math.log(c);
    }


    /**
     * Maps the waiting times of every segment of the given type above the node to regular + c (w - regular),
     * where regular is the length of the segment divided by the number of its waiting times.
     * Returns the number of moved hidden events, or -1 if a waiting time would not be positive.
     */
    int scaleDeviations(AnnotatedNode node, int type, double c) {

        double top;
        if (node.isRoot()) {
            if (parameterization.getOriginTime() == null) return 0; // no stem
            top = parameterization.getOriginTime();
        } else {
            top = node.getParent().getHeight();
        }

        List<EventNode> events = node.getEvents();
        int n = events.size();
        List<EventNode> newEvents = new ArrayList<>(n);
        for (EventNode event : events) newEvents.add(event.copy());

        // a segment starts at the node (index 0) or at a type change, and ends at the next type change or at the top
        int moved = 0;
        int start = 0;
        while (start < n) {
            int end = start + 1;
            while (end < n && events.get(end).getType() == events.get(end - 1).getType()) end++;
            // the age of a cell sampled at time 0 is a truncated lifetime: the first hidden event above it stays
            int first = (start == 0 && node.isLeaf() && node.getHeight() == 0.0 && end > 1) ? 1 : 0;
            int m = end - (start + first) - 1; // hidden events that move

            if (m > 0 && events.get(start).getType() == type) {
                int base = start + first;
                double from = events.get(base).getHeight();
                double to = end < n ? events.get(end).getHeight() : top;
                double regular = (to - from) / (m + 1);
                double previous = from;
                for (int i = 1; i <= m; i++) {
                    double grid = from + i * regular;
                    double height = grid + c * (events.get(base + i).getHeight() - grid);
                    if (height <= previous) return -1;
                    newEvents.get(base + i).setHeight(height);
                    previous = height;
                }
                if (to <= previous) return -1;
                moved += m;
            }
            start = end;
        }

        if (moved > 0) node.setEvents(newEvents);
        return moved;
    }


    /**
     * Automatic parameter tuning on the log scale
     */
    @Override
    public void optimize(double logAlpha) {
        if (optimiseInput.get()) {
            double delta = calcDelta(logAlpha);
            delta += Math.log(scaleFactor);
            setCoercableParameterValue(Math.exp(delta));
        }
    }

    @Override
    public double getCoercableParameterValue() {
        return scaleFactor;
    }

    @Override
    public void setCoercableParameterValue(double value) {
        scaleFactor = Math.max(Math.min(value, MAX_SCALE_FACTOR), MIN_SCALE_FACTOR);
    }

    @Override
    public double getTargetAcceptanceProbability() {
        return 0.3;
    }
}
