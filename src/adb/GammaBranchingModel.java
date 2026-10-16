package adb;

import beast.base.core.*;
import beast.base.evolution.speciation.SpeciesTreeDistribution;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeInterface;
import beast.base.evolution.tree.TreeUtils;
import beast.base.inference.parameter.IntegerParameter;
import beast.base.inference.parameter.RealParameter;

import java.util.Arrays;

import static org.apache.commons.math3.special.Gamma.logGamma;

@Deprecated
@Description("This model implements an Age-Dependent Branching Process " +
        "with Gamma (or Erlang)-distributed lifetimes, some death probability and extant sampling.")
public class GammaBranchingModel extends SpeciesTreeDistribution {

    // parametrization
    final public Input<RealParameter> lifetimeParameterInput =
            new Input<>("lifetime", "expected lifetime", Input.Validate.REQUIRED);
    final public Input<IntegerParameter> shapeIntegerParameterInput =
            new Input<>("shapeInteger", "integer shape parameter of the Erlang distribution");
    final public Input<RealParameter> shapeRealParameterInput =
            new Input<>("shapeReal", "real shape parameter of the Gamma distribution");
    final public Input<RealParameter> deathParameterInput =
            new Input<>("deathprob", "probability of death at branching times (default 0)", new RealParameter("0.0"));
    final public Input<RealParameter> rhoParameterInput =
            new Input<>("rho", "sampling probability at the end of the process (default 1)", new RealParameter("1.0"));
    final public Input<RealParameter> originInput =
            new Input<>("origin", "time of origin of the process");

    // options
    public Input<Integer> maxIterationsInput =
            new Input<>("maxIterations", "maximum number of iterations for numerical integration",100);
    public Input<Double> tolerancePInput =
            new Input<>("toleranceP", "tolerance for numerical integration of P0 and P1",1e-12);
    public Input<Double> toleranceBInput =
            new Input<>("toleranceB", "tolerance for numerical integration of B (branch probabilities)",1e-6);
    public Input<Integer> stepSizePInput =
            new Input<>("stepSizeP", "number of time steps for FFT for P0 and P1", (int)Math.pow(2, 14));
    public Input<Integer> stepSizeBInput =
            new Input<>("stepSizeB", "number of time steps for FFT for B", (int)Math.pow(2, 12));
    public Input<Boolean> approxInput =
            new Input<>("approx", "approximate branch probabilities (default true)", true);
    public Input<Boolean> conditionOnRootInput =
            new Input<>("conditionOnRoot", "condition on the root height otherwise on origin (default false)", false);
    public Input<Boolean> useAnalyticalBDSolutionInput =
            new Input<>("useAnalyticalBDSolution", "use analytical solution if shape is 1 (default true)", true);

    @Override
    public void initAndValidate() {
        super.initAndValidate();

        // make sure that all tips are at the same height
        TreeInterface tree = treeInput.get();
        if (!TreeUtils.isUltrametric(tree)) {
            Log.warning.println("WARNING: This model (tree prior) cannot handle dated tips.");
        }

        // check type of shape
        if (shapeIntegerParameterInput.get() == null && shapeRealParameterInput.get() == null) {
            throw new IllegalArgumentException("Please specify a shape parameter");
        } else if (shapeIntegerParameterInput.get() != null && shapeRealParameterInput.get() != null) {
            throw new IllegalArgumentException("Please specify either an integer or real shape parameter, not both");
        } else if (shapeRealParameterInput.get() != null) {
            if (approxInput.get()) {
                throw new IllegalArgumentException("The approximation only works for an integer shape parameter!");
            }
        }

        // check that origin is given if conditioning on survival (and not root)
        if (originInput.get() == null & !conditionOnRootInput.get()) {
            throw new IllegalArgumentException("Please provide the time of origin of the process, or use conditionOnRoot");
        }

        // check that step sizes are power of 2 (required for FFT)
        if (!isPowerOfTwo(stepSizePInput.get())) {
            throw new IllegalArgumentException("stepSizeP must be a power of 2");
        }

        if (!isPowerOfTwo(stepSizeBInput.get())) {
            throw new IllegalArgumentException("stepSizeB must be a power of 2");
        }
    }


    // Helper method to check if number is a power if two
    private boolean isPowerOfTwo(int x) {
        return (x > 0) && ((x & (x - 1)) == 0); // https://stackoverflow.com/questions/600293/how-to-check-if-a-number-is-a-power-of-2
    }


    @Override
    public double calculateTreeLogLikelihood(final TreeInterface tree) {

        // parameters
        double C = lifetimeParameterInput.get().getValue(); // l
        Number b = null; // k
        if (shapeIntegerParameterInput.get() != null) {
            b = shapeIntegerParameterInput.get().getValue();
        } else if (shapeRealParameterInput.get() != null) {
            b = shapeRealParameterInput.get().getValue();
        }
        double d = deathParameterInput.get().getValue();
        double rho = rhoParameterInput.get().getValue();

        double origin = 0; // initialize and update with provided origin
        if (originInput.get() != null) {
            origin = originInput.get().getValue();

            // stop if tree origin is smaller than root height
            if (tree.getRoot().getHeight() >= origin) {
                return Double.NEGATIVE_INFINITY;
            }
        }

        // options
        int maxIt = maxIterationsInput.get();
        double tolP = tolerancePInput.get();
        double tolB = toleranceBInput.get();
        int mP = stepSizePInput.get();
        int mB = stepSizeBInput.get();
        boolean approx = approxInput.get();
        boolean conditionOnRoot = conditionOnRootInput.get();
        boolean useBD = useAnalyticalBDSolutionInput.get();

        // calculate tree factor
        int nTips = tree.getLeafNodeCount();
        double treeFactor = (nTips - 1) * Math.log(2) - logGamma(nTips + 1); // 2^(n-1)/n!

        // get scale parameter
        double a = C / b.doubleValue(); // theta

        // calculate tree likelihood based on branching times
        double logL;
        if (conditionOnRoot) { // L(T|t_mrca = t_mrca) = L(T_L|t_or = t_mrca, S) * L(T_R|t_or = t_mrca, S)

            // get root and subtrees
            Node root = tree.getRoot();
            double rootHeight = root.getHeight();
            Node childLeft = root.getChild(0);
            Node childRight = root.getChild(1);
            Node[] nodesLeft = childLeft.getAllChildNodesAndSelf().toArray(new Node[0]); // left subtree
            Node[] nodesRight = childRight.getAllChildNodesAndSelf().toArray(new Node[0]); // right subtree

            // collect branching times per subtree
            BranchingTimes leftSubtree = traverseTree(nodesLeft, rootHeight);
            BranchingTimes rightSubtree = traverseTree(nodesRight, rootHeight);

            // use BD solution if indicated (and possible)
            if (useBD && b.doubleValue() == 1 && d != 0.5) {
                logL = GammaLogLikelihood.calcBDLogLikelihood(a, d, rho, rootHeight, leftSubtree.intS, leftSubtree.extE) +
                        GammaLogLikelihood.calcBDLogLikelihood(a, d, rho, rootHeight, rightSubtree.intS, rightSubtree.extE);
            } else {

                // get densities, P0 and P1
                GammaLogLikelihood.Densities densities = GammaLogLikelihood.getDensities(a, b, rootHeight, mP);
                double[] P0 = GammaLogLikelihood.calcP0(densities.pdfFFT, densities.cdf, d, rho, densities.dx, maxIt, tolP);
                double[] P1 = GammaLogLikelihood.calcP1(densities.pdfFFT, densities.cdf, P0, d, rho, densities.dx, maxIt, tolP);

                // calculate likelihood
                logL = GammaLogLikelihood.calcLogLikelihood(a, b, d, rho, densities, P0, P1, leftSubtree.intS, leftSubtree.intE, leftSubtree.extE, maxIt, tolB, mB, approx) +
                        GammaLogLikelihood.calcLogLikelihood(a, b, d, rho, densities, P0, P1, rightSubtree.intS, rightSubtree.intE, rightSubtree.extE, maxIt, tolB, mB, approx);

            }

        } else {
            assert origin > 0;

            // collect branching times of the tree
            Node[] nodes = tree.getNodesAsArray();
            BranchingTimes branches = traverseTree(nodes, origin);

            // use BD solution if indicated (and possible)
            if (useBD && b.doubleValue() == 1 && d != 0.5) {
                logL = GammaLogLikelihood.calcBDLogLikelihood(a, d, rho, origin, branches.intS, branches.extE);
            } else {

                // get densities, P0 and P1
                GammaLogLikelihood.Densities densities = GammaLogLikelihood.getDensities(a, b, origin, mP);
                double[] P0 = GammaLogLikelihood.calcP0(densities.pdfFFT, densities.cdf, d, rho, densities.dx, maxIt, tolP);
                double[] P1 = GammaLogLikelihood.calcP1(densities.pdfFFT, densities.cdf, P0, d, rho, densities.dx, maxIt, tolP);

                // calculate likelihood
                logL = GammaLogLikelihood.calcLogLikelihood(a, b, d, rho, densities, P0, P1, branches.intS, branches.intE, branches.extE, maxIt, tolB, mB, approx);

            }
        }

        return treeFactor + logL;
    }


    // Small class for storing branching times
    protected static class BranchingTimes{
        double[] intS;
        double[] intE;
        double[] extE;
        protected BranchingTimes(double[] intS, double[] intE, double[] extE) {
            this.intS = intS;
            this.intE = intE;
            this.extE = extE;
        }
    }


    // Function for collecting branching times from tree
    protected BranchingTimes traverseTree(final Node[] nodes, double origin) {

        // create empty arrays to store start and end times of branches
        double[] intS = new double[0];
        double[] intE = new double[0];
        double[] extE = new double[0];

        // traverse all nodes and compute start/end times of all branches (backwards in time)
        for (Node node : nodes) {

            // start time is the node height (or divergence time of the node)
            double startTime = node.getHeight();
            double endTime;
            if (node.isRoot()) {
                // end time is the tree origin
                endTime = origin;
            } else {
                // end time is the parent node height (i.e., the divergence time of the parent)
                endTime = node.getParent().getHeight();
            }

            // if node is a tip, add times to external branches, otherwise, add to internal branches
            if (node.isLeaf()) {
                assert startTime == 0; // tips should have height 0
                extE = Arrays.copyOf(extE, extE.length + 1);
                extE[extE.length - 1] = endTime;
            } else {
                intS = Arrays.copyOf(intS, intS.length + 1);
                intE = Arrays.copyOf(intE, intE.length + 1);
                intS[intS.length - 1] = startTime;
                intE[intE.length - 1] = endTime;
            }
        }

        return new BranchingTimes(intS, intE, extE);
    }


    @Override
    protected boolean requiresRecalculation() {
        return true;
    }


    @Override
    public boolean canHandleTipDates() {
        return false;
    }
}
