package adb.operators;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.inference.Operator;
import beast.base.inference.StateNode;
import beast.base.inference.parameter.RealParameter;
import beast.base.util.Randomizer;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;


/**
 * Adapted from the delta exchange operator of BEAST. Each parameter is a square matrix stored row by row, for example
 * the symmetric and the asymmetric division probabilities. The operator picks a row and two entries of that row, which
 * may belong to different matrices, and moves a random amount of at most delta from one to the other. The sum of the
 * row over all matrices is therefore kept. Entries that are 0 or 1 when the operator is created never move, so
 * transitions that are impossible stay impossible. Values are kept in [0, 1].
 */
@Description("Operator for row-wise sum-constrained parameters.")
public class MatrixDeltaExchangeOperator extends Operator {

    public Input<List<RealParameter>> parameterInput =
            new Input<>("parameter", "One or more parameters to operate on", new ArrayList<>());

    public Input<Double> deltaInput =
            new Input<>("delta", "Magnitude of change for two randomly picked values.", 0.1);

    public final Input<Boolean> autoOptimizeInput =
            new Input<>("autoOptimize", "Adjust delta during the MCMC run to improve mixing (default true).", true);


    private List<RealParameter> parameters;
    private double delta;
    private boolean autoOptimize;

    private int nparams;
    private int ndims;
    private int[][][] movable; // columns of the entries that can move, per parameter and row


    public void initAndValidate() {
        parameters = parameterInput.get();
        delta = deltaInput.get();
        autoOptimize = autoOptimizeInput.get();

        nparams = parameters.size();
        if (nparams == 0) {
            throw new IllegalArgumentException("Please provide at least one parameter for this operator.");
        }

        int dim = parameters.get(0).getDimension();
        ndims = (int) Math.round(Math.sqrt(dim));
        if (dim != ndims * ndims) {
            throw new IllegalArgumentException("Provided parameters cannot be converted into quadratic matrices.");
        }
        for (RealParameter param : parameters) {
            if (param.getDimension() != dim) {
                throw new IllegalArgumentException("Parameter dimensions do not match.");
            }
        }

        // collect the entries strictly between 0 and 1, row-wise
        movable = new int[nparams][ndims][];
        boolean valid = false;
        for (int i = 0; i < ndims; i++) {
            int rowCount = 0;
            for (int p = 0; p < nparams; p++) {
                List<Integer> columns = new ArrayList<>();
                for (int j = 0; j < ndims; j++) {
                    double v = parameters.get(p).getValue(i * ndims + j);
                    if (v != 0.0 && v != 1.0) columns.add(j);
                }
                movable[p][i] = columns.stream().mapToInt(Integer::intValue).toArray();
                rowCount += columns.size();
            }
            if (rowCount > 1) valid = true; // at least one row can be modified with this operator
        }
        if (!valid) {
            throw new IllegalArgumentException("Provided matrices are too sparse for this operator.");
        }
    }


    @Override
    public final double proposal() {

        // select the matrices, then a row where both have entries that can move
        int m1 = Randomizer.nextInt(nparams);
        int m2 = Randomizer.nextInt(nparams);
        int[] rows = new int[ndims];
        int nrows = 0;
        for (int i = 0; i < ndims; i++) {
            if (movable[m1][i].length > 0 && movable[m2][i].length > 0) rows[nrows++] = i;
        }
        if (nrows == 0) return Double.NEGATIVE_INFINITY; // no move possible, the state is unchanged
        int i = rows[Randomizer.nextInt(nrows)];

        int dim1 = i * ndims + movable[m1][i][Randomizer.nextInt(movable[m1][i].length)]; // entry 1
        int dim2 = i * ndims + movable[m2][i][Randomizer.nextInt(movable[m2][i].length)]; // entry 2
        if (m1 == m2 && dim1 == dim2) {
            return Double.NEGATIVE_INFINITY; // no move (self), the state is unchanged
        }

        // update values
        RealParameter param1 = parameters.get(m1);
        RealParameter param2 = parameters.get(m2);
        double d = Randomizer.nextDouble() * delta; // d ~ Uniform(0, delta)
        double value1 = param1.getValue(dim1) - d / 2;
        double value2 = param2.getValue(dim2) + d / 2;

        if (value1 < Math.max(0.0, param1.getLower()) || value2 > Math.min(1.0, param2.getUpper())) {
            return Double.NEGATIVE_INFINITY;
        }
        param1.setValue(dim1, value1);
        param2.setValue(dim2, value2);

        return 0.0; // symmetric move
    }


    @Override
    public double getCoercableParameterValue() {
        return delta;
    }


    @Override
    public void setCoercableParameterValue(double value) {
        delta = value;
    }


    @Override
    public void optimize(double logAlpha) {
        if (autoOptimize) {
            double _delta = calcDelta(logAlpha);
            _delta += Math.log(delta);
            delta = Math.exp(_delta);
        }
    }


    @Override
    public final String getPerformanceSuggestion() {
        final double prob = m_nNrAccepted / (m_nNrAccepted + m_nNrRejected + 0.0);
        final double targetProb = getTargetAcceptanceProbability();

        double ratio = prob / targetProb;
        if (ratio > 2.0) ratio = 2.0;
        if (ratio < 0.5) ratio = 0.5;

        // new magnitude of change
        double newDelta = delta * ratio;
        final DecimalFormat formatter = new DecimalFormat("#.###");
        if (prob < 0.10) {
            return "Try setting delta to about " + formatter.format(newDelta);
        } else if (prob > 0.40) {
            return "Try setting delta to about " + formatter.format(newDelta);
        } else return "";
    }


    @Override
    public List<StateNode> listStateNodes() {
        return new ArrayList<>(parameterInput.get());
    }
}
