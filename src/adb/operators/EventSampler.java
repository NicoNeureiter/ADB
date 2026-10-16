package adb.operators;

import adb.tree.AnnotatedNode;
import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.util.InputUtil;
import beast.base.util.Randomizer;

import java.text.DecimalFormat;

/**
 * Each branch is chosen independently with probability proportionBranches, and its hidden events are redrawn with
 * resampleEvents. The node heights and the type changes stay fixed, so the move works for any number of types.
 * If drawEventCount is false, the number of hidden events per type segment is kept,
 * except on the segment of a leaf sampled at time 0, which is always drawn exactly from the lifetime distribution.
 */
@Description("Redraws the hidden events on a random subset of branches of an annotated tree, keeping the nodes and type changes fixed")
public class EventSampler extends AnnotatedTreeOperator {

    public Input<Boolean> drawEventCountInput = new Input<>("drawEventCount",
            "draw or fix the number of events along internal branches (default true)", true);

    public Input<Double> proportionBranchesInput = new Input<>("proportionBranches",
            "proportion of branches to sample events on (default 0.1)", 0.1);

    public Input<Boolean> optimiseInput = new Input<>("optimise",
            "flag to indicate that proportionBranches is automatically changed in order to achieve a good acceptance rate (default true)", true);

    boolean drawEventCount;
    double proportionBranches;

    @Override
    public void initAndValidate() {
        super.initAndValidate();
        drawEventCount = drawEventCountInput.get();
        proportionBranches = proportionBranchesInput.get();
        if (!(proportionBranches > 0.0 && proportionBranches <= 1.0)) {
            throw new IllegalArgumentException("proportionBranches must be in (0, 1], but is " + proportionBranches);
        }
    }


    @Override
    public double proposal() {

        Tree tree = (Tree) InputUtil.get(treeInput, this);
        double oldBranchProb, newBranchProb;
        double logHR = 0;

        double u;
        for (Node node : tree.getNodesAsArray()) {
            u = Randomizer.nextDouble();
            if (u < proportionBranches) {
                oldBranchProb = getBranchProbability((AnnotatedNode)node, drawEventCount);
                newBranchProb = resampleEvents((AnnotatedNode)node, drawEventCount);
                logHR += (oldBranchProb - newBranchProb);
            }
        }

        return logHR;
    }

    /**
     * Tune branch proportion on logit scale
     */
    @Override
    public void optimize(double logAlpha) {
        if (optimiseInput.get()) {
            double delta = calcDelta(logAlpha);
            delta += Math.log(proportionBranches / (1.0 - proportionBranches));
            setCoercableParameterValue(1.0 / (1.0 + Math.exp(-delta)));
        }
    }

    @Override
    public double getCoercableParameterValue() {
        return proportionBranches;
    }

    @Override
    public void setCoercableParameterValue(double value) {
        proportionBranches = Math.max(Math.min(value, 1.0 - 1e-8), 1e-8);
    }

    @Override
    public String getPerformanceSuggestion() {
        final double prob = m_nNrAccepted / (m_nNrAccepted + m_nNrRejected + 0.0);
        final double targetProb = getTargetAcceptanceProbability();

        double ratio = prob / targetProb;
        if (ratio > 2.0) ratio = 2.0;
        if (ratio < 0.5) ratio = 0.5;

        double logit = Math.log(proportionBranches / (1.0 - proportionBranches));
        logit += Math.log(ratio);
        double newProportionBranches = 1.0 / (1.0 + Math.exp(-logit));

        final DecimalFormat formatter = new DecimalFormat("#.###");
        if (prob < 0.10) {
            return "Try setting proportionBranches to about " + formatter.format(newProportionBranches);
        } else if (prob > 0.40) {
            return "Try setting proportionBranches to about " + formatter.format(newProportionBranches);
        } else return "";
    }

}
