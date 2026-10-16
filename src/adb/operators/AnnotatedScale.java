package adb.operators;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.operator.kernel.KernelDistribution;
import beast.base.inference.parameter.RealParameter;
import beast.base.inference.util.InputUtil;
import adb.tree.AnnotatedNode;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;

/**
 * Node heights are multiplied by f (the leaves must be at 0) and the events of every branch are mapped affinely
 * onto the new branch, so type changes stay on their branch; the stem [root, origin] goes to [new root, origin].
 * The hidden events of the stem, or with drawEvents of all branches, are redrawn between the type changes.
 * The Bactrian kernel is symmetric on the log scale, so the log Hastings ratio is
 * the log Jacobian of the deterministic map plus, per redrawn branch, the old minus the new event density.
 */
@Description("Scales an annotated tree and its events, optionally together with parameters (forward or inversely), and redraws the hidden events")
public class AnnotatedScale extends AnnotatedTreeOperator {

    // Note: The lifetime is a time, so scaling it with the tree (parameter) follows the time-scale ridge, along which the
    // gamma densities of the branch segments change little (the fixed origin breaks the invariance only slightly).
    // Rates, such as a clock rate when there is an alignment, go inversely (parameterInverse). Probabilities such as
    // death or sampling have no time unit and do not belong in this move: the ridge between lifetime and death with
    // the tree fixed is the move of UpDownEventSampler (lifetime up, death down).

    public Input<List<RealParameter>> parametersInput = new Input<>("parameter",
            "parameters scaled by the same factor as the tree", new ArrayList<>());

    public Input<List<RealParameter>> parametersInverseInput = new Input<>("parameterInverse",
            "parameters scaled by the inverse factor", new ArrayList<>());

    public Input<Double> scaleFactorInput = new Input<>("scaleFactor",
            "window of the log scale factor: the scale factor is exp(scaleFactor * x) with x from the Bactrian kernel (default 0.1)", 0.1);

    public Input<Boolean> optimiseInput = new Input<>("optimise",
            "flag to indicate that the scale factor is automatically changed in order to achieve a good acceptance rate (default true)", true);

    public Input<Boolean> drawEventsInput = new Input<>("drawEvents",
            "redraw the hidden events on all branches; otherwise they are scaled with the tree and only the stem (if there is an origin) is redrawn (default true)", true);

    // limits of the tuned scale factor, as in BEAST's Bactrian operators, but positive: 0 would make the move a no-op
    static final double MIN_SCALE_FACTOR = 1e-8, MAX_SCALE_FACTOR = 10.0;

    final KernelDistribution kernel = new KernelDistribution.Bactrian(KernelDistribution.Bactrian.mode.bactrian_normal);
    boolean drawEvents;
    double scaleFactor;

    @Override
    public void initAndValidate() {

        super.initAndValidate();
        scaleFactor = scaleFactorInput.get();
        drawEvents = drawEventsInput.get();

        for (Node leaf : treeInput.get().getExternalNodes()) {
            if (leaf.getHeight() != 0.0) {
                throw new IllegalArgumentException("AnnotatedScale requires all leaves at height 0, but leaf "
                        + leaf.getID() + " is at " + leaf.getHeight());
            }
        }
    }


    @Override
    public double proposal() {

        Tree tree = (Tree) InputUtil.get(treeInput, this);

        // choose scale factor, symmetric on the log scale
        double f = kernel.getScaler(0, Double.NaN, scaleFactor);
        double logf = Math.log(f);
        double logHR = 0;

        // the root cannot be scaled above the origin
        Double origin = parameterization.getOriginTime();
        if (origin != null && tree.getRoot().getHeight() * f >= origin) {
            return Double.NEGATIVE_INFINITY;
        }

        // the hidden events are redrawn on all branches with drawEvents, otherwise only on the stem, whose top
        // (the origin) does not move, so that its events are not squeezed by a different factor than the tree
        // (without origin the root has no stem, and its density is 0); evaluate their current events under the old parameters
        Node[] nodes = tree.getNodesAsArray();
        for (Node node : nodes) {
            if (drawEvents || node.isRoot()) {
                logHR += getBranchProbability((AnnotatedNode) node, true);
            }
        }

        // scale parameters forward and inversely
        logHR += scaleParameters(parametersInput.get(), -1, f);
        if (logHR == Double.NEGATIVE_INFINITY) return logHR;
        logHR += scaleParameters(parametersInverseInput.get(), -1, 1.0 / f);
        if (logHR == Double.NEGATIVE_INFINITY) return logHR;

        // scale node heights (before the events, so that the tree is stored before they change);
        // the internal nodes count in the Jacobian, the leaves stay at 0
        double[] oldHeights = new double[nodes.length];
        for (Node node : nodes) {
            oldHeights[node.getNr()] = node.getHeight();
            node.setHeight(node.getHeight() * f);
            if (!node.isLeaf()) {
                logHR += logf;
            }
        }

        // map the events of every branch onto its new interval: by f below the root, from [root, origin] onto
        // [new root, origin] on the stem; events that are not redrawn count in the Jacobian, otherwise the type changes;
        // then redraw the hidden events between the kept type changes, under the new parameters
        for (Node node : nodes) {
            AnnotatedNode annotated = (AnnotatedNode) node;
            boolean redraw = drawEvents || node.isRoot();
            if (!node.isRoot() || origin != null) { // without origin the root has no events above it
                double oldTop = node.isRoot() ? origin : oldHeights[node.getParent().getNr()];
                double newTop = node.isRoot() ? origin : node.getParent().getHeight();
                logHR += scaleEvents(annotated, oldHeights[node.getNr()], oldTop, node.getHeight(), newTop, !redraw);
            }
            if (redraw) {
                logHR -= resampleEvents(annotated, true); // also syncs the root event without origin
            }
        }

        // return Hastings ratio
        return logHR;
    }


    /**
     * Automatic parameter tuning on the log scale, as in BEAST's Bactrian operators
     */
    @Override
    public void optimize(final double logAlpha) {
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
    public void setCoercableParameterValue(final double value) {
        scaleFactor = Math.max(Math.min(value, MAX_SCALE_FACTOR), MIN_SCALE_FACTOR);
    }

    @Override
    public double getTargetAcceptanceProbability() {
        return 0.3;
    }

    @Override
    public String getPerformanceSuggestion() {
        final double prob = m_nNrAccepted / (m_nNrAccepted + m_nNrRejected + 0.0);
        final double targetProb = getTargetAcceptanceProbability();

        double ratio = prob / targetProb;
        if (ratio > 2.0) ratio = 2.0;
        if (ratio < 0.5) ratio = 0.5;

        // new scale factor
        final double sf = scaleFactor * ratio;

        final DecimalFormat formatter = new DecimalFormat("#.###");
        if (prob < 0.10 || prob > 0.40) {
            return "Try setting scaleFactor to about " + formatter.format(sf);
        } else return "";
    }
}
