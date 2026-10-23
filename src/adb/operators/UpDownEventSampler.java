package adb.operators;

import adb.tree.AnnotatedNode;
import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.operator.kernel.KernelDistribution;
import beast.base.inference.parameter.RealParameter;
import beast.base.inference.util.InputUtil;
import beast.base.util.Randomizer;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;

/**
 * Scales the up parameters by f and the down parameters by 1/f, as BEAST's BactrianUpDownOperator does,
 * and redraws the hidden events on a random subset of branches under the new parameters.
 * The log scale factor is scaleFactor times a draw from the Bactrian kernel, which is symmetric around 0,
 * so the scaling contributes log f for every up element and -log f for every down element to the log Hastings ratio.
 * With elementWise, for multi-type models, a type j is chosen uniformly and only entry j of every parameter is scaled,
 * so every parameter needs one entry per type. All branches that carry type j are redrawn (and no others), since only
 * their densities depend on the parameters of type j. This set is fixed by the type changes, which the move keeps,
 * so the reverse move redraws the same branches; proportionBranches is then not used.
 * Redrawing only helps for parameters that the event proposal uses: the lifetime and the shape, and the death and
 * sampling probability through the hidden-event factor (1 - d) 2 P0 (except in the discretized normal event count).
 */
@Description("Scales parameters up and down with a Bactrian kernel and redraws the hidden events on a subset of branches (random, or those carrying one type with elementWise)")
public class UpDownEventSampler extends AnnotatedTreeOperator {

    public Input<List<RealParameter>> upInput = new Input<>("up",
            "parameters to scale upwards", new ArrayList<>());

    public Input<List<RealParameter>> downInput = new Input<>("down",
            "parameters to scale downwards", new ArrayList<>());

    public Input<Boolean> elementWiseInput = new Input<>("elementWise",
            "scale the entry of one random type in every parameter and redraw the branches that carry this type (default false)", false);

    public Input<Double> scaleFactorInput = new Input<>("scaleFactor",
            "window of the log scale factor: the scale factor is exp(scaleFactor * x) with x from the Bactrian kernel (default 0.1)", 0.1);

    public Input<Double> proportionBranchesInput = new Input<>("proportionBranches",
            "proportion of branches to sample events on, not used with elementWise (default 0.1)", 0.1);

    public Input<Boolean> optimiseInput = new Input<>("optimise",
            "flag to indicate that the scale factor is automatically changed in order to achieve a good acceptance rate (default true)", true);


    // limits of the tuned scale factor, as in BactrianUpDownOperator, but positive: 0 would make the move a no-op
    static final double MIN_SCALE_FACTOR = 1e-8, MAX_SCALE_FACTOR = 10.0;

    final KernelDistribution kernel = new KernelDistribution.Bactrian(KernelDistribution.Bactrian.mode.bactrian_normal);
    double scaleFactor;
    double proportionBranches;
    boolean elementWise;


    @Override
    public void initAndValidate() {
        super.initAndValidate();
        scaleFactor = scaleFactorInput.get();
        proportionBranches = proportionBranchesInput.get();
        elementWise = elementWiseInput.get();
        if (upInput.get().size() + downInput.get().size() == 0) {
            throw new IllegalArgumentException("At least one up or down parameter must be specified");
        }
        if (elementWise) {
            List<RealParameter> parameters = new ArrayList<>(upInput.get());
            parameters.addAll(downInput.get());
            for (RealParameter parameter : parameters) {
                if (parameter.getDimension() != parameterization.getNTypes()) {
                    throw new IllegalArgumentException("With elementWise, every parameter needs one entry per type, but "
                            + parameter.getID() + " has dimension " + parameter.getDimension());
                }
            }
        }
        if (!(proportionBranches > 0.0 && proportionBranches <= 1.0)) {
            throw new IllegalArgumentException("proportionBranches must be in (0, 1], but is " + proportionBranches);
        }
    }


    @Override
    public double proposal() {

        Tree tree = (Tree) InputUtil.get(treeInput, this);

        // with elementWise, choose the type to scale (-1: scale all entries)
        int type = elementWise ? Randomizer.nextInt(parameterization.getNTypes()) : -1;

        // choose the branches to redraw, and evaluate their current events under the old parameters
        double logHR = 0;
        List<AnnotatedNode> chosen = new ArrayList<>();
        for (Node node : tree.getNodesAsArray()) {
            // with elementWise: the branches that carry the type, at the node or after a type change
            boolean redraw = elementWise ? ((AnnotatedNode) node).getEvents().stream().anyMatch(e -> e.getType() == type)
                    : Randomizer.nextDouble() < proportionBranches;
            if (redraw) {
                chosen.add((AnnotatedNode) node);
                logHR += getBranchProbability((AnnotatedNode) node, true);
            }
        }

        // choose scale factor, symmetric on the log scale
        double f = kernel.getScaler(0, Double.NaN, scaleFactor);

        // scale parameters upwards and downwards
        logHR += scaleParameters(upInput.get(), type, f);
        if (logHR == Double.NEGATIVE_INFINITY) return logHR;
        logHR += scaleParameters(downInput.get(), type, 1.0 / f);
        if (logHR == Double.NEGATIVE_INFINITY) return logHR;

        // redraw the events of the chosen branches under the new parameters
        for (AnnotatedNode node : chosen) {
            logHR -= resampleEvents(node, true);
        }
        return logHR;
    }


    /**
     * Automatic parameter tuning on the log scale, as in BactrianUpDownOperator
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
