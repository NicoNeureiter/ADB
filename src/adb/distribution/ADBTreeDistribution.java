package adb.distribution;

import adb.tree.AnnotatedNode;
import adb.tree.AnnotatedNode.EventNode;
import adb.tree.AnnotatedTree;
import adb.util.Utils;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.core.Log;
import beast.base.evolution.speciation.SpeciesTreeDistribution;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeInterface;
import beast.base.evolution.tree.TreeUtils;

import org.apache.commons.math3.analysis.UnivariateFunction;
import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.distribution.GammaDistribution;
import org.apache.commons.math3.util.Pair;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;

import static adb.util.Utils.TRANSFORM_FORWARD;
import static adb.tree.AnnotatedNode.getTypeOf;
import static org.apache.commons.math3.special.Gamma.logGamma;


/**
 * The age-dependent multi-type binary branching process:
 * a lineage of type i lives for a gamma-distributed time (mean lifetime l, shape k), then dies with probability d or
 * divides into two daughters whose types follow the symmetric and asymmetric transition probabilities.
 * At time 0 each lineage of type i is sampled with probability rho. The process starts at the origin time and type.
 *
 * The log density includes the tree factor 2^(n-1)/n! for n tips. Heights are measured backwards from the present.
 * With conditionOnOrigin, the density is conditioned on at least one sampled descendant of the origin, otherwise on
 * the root.
 *
 * For an AnnotatedTree, the hidden division events and type changes are given, and the density is a product of gamma
 * segment densities and event terms (the first event of a branch is the node itself). For any other tree, the hidden
 * events are integrated out: P1 is used on tip branches, and an internal branch density is solved by Picard iteration
 * with FFT, or by a series with approx (one type, integer shape). P0 and P1 are solved on a uniform grid of nSteps
 * points (a power of 2) up to the origin time, and only recomputed when a parameter changes.
 *
 * Limits (work in progress): with conditionOnOrigin false, the grid and the conditioning use the initial root height.
 * For annotated trees the stem type is not enforced to be the origin type. Dated tips are not supported,
 * and useAnalyticalBDSolution is not implemented.
 */
@Description("Probability of a tree under the age-dependent multi-type branching process, either with the hidden " +
        "division events given (annotated tree) or integrated out (other trees).")
public class ADBTreeDistribution extends SpeciesTreeDistribution {

    public Input<Parameterization> parameterizationInput =
            new Input<>("parameterization", "ADB parameterization", Input.Validate.REQUIRED);

    // computing options
    public Input<Integer> maxIterationsInput =
            new Input<>("maxIterations", "maximum number of iterations of the fixed-point iteration for P0, P1 and the branch densities (default 100)",100);
    public Input<Double> toleranceInput =
            new Input<>("tolerance", "stopping tolerance of the fixed-point iteration (default 1e-12)",1e-12);
    public Input<Integer> nStepsInput =
            new Input<>("nSteps", "number of time steps of the grid between 0 and the origin time, must be a power of 2 (default 2^12)", (int)Math.pow(2, 12));
    public Input<Boolean> conditionOnOriginInput =
            new Input<>("conditionOnOrigin", "condition on the origin time and type, otherwise on the root height (default true)", true);
    public Input<Boolean> approxInput =
            new Input<>("approx", "approximate the branch densities by a series, only for one type and an integer shape (default false)", false);
    public Input<Boolean> useAnalyticalBDSolutionInput =
            new Input<>("useAnalyticalBDSolution", "use the analytical birth-death solution if the shape is 1 (default false)", false);


    Parameterization parameterization;
    TreeInterface tree;

    // options
    int maxIt;
    double tol;
    int nSteps;
    boolean conditionOnOrigin;
    boolean approx;
    boolean useBD;

    // time array
    double[] timeArray;
    double timeStep;

    int nTypes;

    // calculation nodes
    HashMap<Integer, GammaDistribution> gammaDistributions;
    LifetimeDistributions lifetimeDistributions;
    P0System P0System;
    HashMap<Integer, UnivariateFunction> P0Map;

    // optional
    P1System P1System;
    HashMap<Pair<Integer,Integer>, UnivariateFunction> P1Map;

    // for approximation:
    // use cashing to avoid creating a new GammaDistribution object with shape i*b for i=1,2,...
    // create a map to store the distributions for each i (once needed) and re-use them
    // use a thread-safe and dynamic map
    ConcurrentHashMap<Integer, GammaDistribution> gammaCache = new ConcurrentHashMap<>();


    @Override
    public void initAndValidate() {
        super.initAndValidate();

        parameterization = parameterizationInput.get();
        nTypes = parameterization.getNTypes();
        tree = treeInput.get();

        // make sure that all tips are at the same height
        if (!TreeUtils.isUltrametric(tree)) {
            Log.warning("WARNING: This model (tree prior) cannot handle dated tips.");
        }

        // make sure that all tips have valid types
        for (Node tip : tree.getExternalNodes()) {
            int type = getTypeOf(tip);
            if (type < 0 || type >= nTypes) {
                throw new IllegalArgumentException("Not all types at tips are valid, please check the input.");
            }
        }

        maxIt = maxIterationsInput.get();
        tol = toleranceInput.get();
        nSteps = nStepsInput.get();
        conditionOnOrigin = conditionOnOriginInput.get();
        approx = approxInput.get();
        useBD = useAnalyticalBDSolutionInput.get();

        if (approx) {
            if (nTypes > 1) {
                throw new IllegalArgumentException("The approximation only works for a single-type branching process!");
            }
            // check whether the shape parameter is an integer
            if (!parameterization.shapeIsInteger()) {
                throw new IllegalArgumentException("The approximation only works for an integer shape parameter!");
            }
        }

        if (useBD && nTypes > 1) {
            throw new IllegalArgumentException("This model only uses the BD analytical solution for a single-type branching process.");
        }

        // check that step size is a power of 2 (required for FFT)
        if (!Utils.isPowerOfTwo(nSteps)) {
            throw new IllegalArgumentException("nSteps must be a power of 2!");
        }

        // create time array for computation
        if (conditionOnOrigin) {
            if (parameterization.getOriginTime() == null) {
                throw new IllegalArgumentException("Please provide the time of origin of the process, or set conditionOnOrigin to false.");
            } else {
                timeStep = parameterization.getOriginTime() / nSteps;
                timeArray = Utils.linSpace(0, parameterization.getOriginTime(), nSteps);
            }
        } else {
            timeStep = tree.getRoot().getHeight() / nSteps;
            timeArray = Utils.linSpace(0, tree.getRoot().getHeight(), nSteps);
        }

        // initialize internal calculations
        lifetimeDistributions = new LifetimeDistributions(
                parameterization.getLifetimeParameter(),
                parameterization.getShapeParameter(),
                timeArray
        );

        P0System = new P0System(
                parameterization,
                lifetimeDistributions,
                maxIt, tol, timeArray, timeStep
        );

        if (!(tree instanceof AnnotatedTree)) { // TODO: or xml input tag?
            P1System = new P1System(
                    parameterization,
                    lifetimeDistributions,
                    P0System,
                    maxIt, tol, timeArray, timeStep
            );
        }

    }


    @Override
    public double calculateTreeLogLikelihood(TreeInterface tree) {

        Double originTime = parameterization.getOriginTime();
        int originType = parameterization.getOriginType();

        // get root
        Node root = tree.getRoot();
        double rootHeight = tree.getRoot().getHeight();

        // stop if tree origin is smaller than root height
        if (originTime != null) {
            if (rootHeight >= originTime) {
                return Double.NEGATIVE_INFINITY;
            }
        }

        // clear cache
        if (approx) {
            gammaCache.clear();
        }

        // TODO: add single-type BD analytical solution

        // stop if extinction is certain
        double[][] P0 = P0System.getP0();
        if (P0[originType][nSteps - 1] >= 1.0) {
            return Double.NEGATIVE_INFINITY;
        }

        // get distributions for various calculations
        gammaDistributions = new HashMap<>();
        for (int i = 0; i < nTypes; i++) {
            double lifetime = parameterization.getLifetime(i);
            double shape = parameterization.getShape(i);
            double scale = lifetime / shape;
            gammaDistributions.put(i, new GammaDistribution(null, shape, scale));
        }

        // extend time array to calculate probabilities of tiny branches
        double[] extTimeArray = new double[nSteps + 1];
        extTimeArray[0] = 0;
        System.arraycopy(timeArray, 0, extTimeArray, 1, nSteps);

        // extend and interpolate P0
        P0Map = new HashMap<>();
        for (int i = 0; i < nTypes; i++) {
            double[] extP0 = new double[nSteps + 1];
            extP0[0] = 1 - parameterization.getSampling(i);
            System.arraycopy(P0[i], 0, extP0, 1, nSteps);
            UnivariateFunction function = Utils.interpolator.interpolate(extTimeArray, extP0);
            P0Map.put(i, function);
        }

        if (!(tree instanceof AnnotatedTree)) {
            double[][][] P1 = P1System.getP1();

            // extend and interpolate P1
            P1Map = new HashMap<>();
            for (int i = 0; i < nTypes; i++) {
                for (int j = 0; j < nTypes; j++) {
                    double[] extP1 = new double[nSteps + 1];
                    if (i == j) { extP1[0] = parameterization.getSampling(i); } else { extP1[0] = 0; }
                    System.arraycopy(P1[i][j], 0, extP1, 1, nSteps);
                    UnivariateFunction function = Utils.interpolator.interpolate(extTimeArray, extP1);
                    P1Map.put(new Pair<>(i, j), function);
                }
            }
        }

        // calculate tree factor
        int nTips = tree.getLeafNodeCount();
        double treeFactor = (nTips - 1) * Math.log(2) - logGamma(nTips + 1); // 2^(n-1)/n!

        double conditionFactor;
        double logL;

        if (conditionOnOrigin) {
            conditionFactor = -Math.log1p(-P0[originType][nSteps - 1]);

            if (tree instanceof AnnotatedTree) {
                double[] subtreeLikelihood = new double[tree.getNodeCount()];
                logL = calculateAnnotatedSubtreeLikelihood((AnnotatedNode) root, rootHeight, originTime, subtreeLikelihood);
            } else {
                double[][] subtreeLikelihood = new double[tree.getNodeCount()][nTypes];
                logL = Math.log(calculateSubtreeLikelihood(root, rootHeight, originTime, subtreeLikelihood)[originType]);
            }

        } else {
            // TODO: account for type transition at root
            conditionFactor = -2 * Math.log1p(-P0[getTypeOf(root)][nSteps - 1]);

            Node leftSubtree = root.getLeft();
            Node rightSubtree = root.getRight();
            if (tree instanceof AnnotatedTree) {
                double[] subtreeLikelihood = new double[tree.getNodeCount()];
                logL = calculateAnnotatedSubtreeLikelihood((AnnotatedNode) leftSubtree, leftSubtree.getHeight(), rootHeight, subtreeLikelihood) +
                        calculateAnnotatedSubtreeLikelihood((AnnotatedNode) rightSubtree, rightSubtree.getHeight(), rootHeight, subtreeLikelihood);
            } else {
                double[][] subtreeLikelihood = new double[tree.getNodeCount()][nTypes];
                logL = Math.log(calculateSubtreeLikelihood(leftSubtree, leftSubtree.getHeight(), rootHeight, subtreeLikelihood)[getTypeOf(root)]) +
                        Math.log(calculateSubtreeLikelihood(rightSubtree, rightSubtree.getHeight(), rootHeight, subtreeLikelihood)[getTypeOf(root)]);
            }
        }

        logL = treeFactor + conditionFactor + logL;
        //Tree flatTree = ((AnnotatedTree)tree).convertAnnotatedTree(false);
        //System.out.println(flatTree.getRoot().toNewick());
        return logL;
    }


    private double calculateAnnotatedSubtreeLikelihood(AnnotatedNode node, double start, double end, double[] subtreeLikelihood) {

        double likelihood;

        // upstream branch
        double branchDensity = calculateAnnotatedNodeLikelihood(node, start, end);

        // at tips
        if (node.isLeaf()) {
            likelihood = branchDensity;

        // recursion bottom-up
        } else {
            int nodeType = node.getType();
            List<EventNode> children = node.getChildEvents(0);
            EventNode left = children.get(0);
            EventNode right = children.get(1);
            int leftType = left.getType();
            int rightType = right.getType();

            AnnotatedNode leftSubtree = (AnnotatedNode) node.getLeft();
            AnnotatedNode rightSubtree = (AnnotatedNode) node.getRight();

            double factor = parameterization.getDivisionFactor(nodeType, leftType, rightType);
            if (factor == 0.0) { // impossible division
                return Double.NEGATIVE_INFINITY;
            }

            likelihood = branchDensity + Math.log(factor) +
                    calculateAnnotatedSubtreeLikelihood(leftSubtree, leftSubtree.getHeight(), node.getHeight(), subtreeLikelihood) +
                    calculateAnnotatedSubtreeLikelihood(rightSubtree, rightSubtree.getHeight(), node.getHeight(), subtreeLikelihood);
        }

        subtreeLikelihood[node.getNr()] = likelihood;
        return likelihood;
    }


    // calculate the density of the branch upstream of an annotated node
    private double calculateAnnotatedNodeLikelihood(AnnotatedNode node, double start, double end) {

        List<EventNode> events = node.getEvents();
        assert events.get(0).getHeight() == start;

        // initial event
        int ix = events.size() - 1;
        EventNode event = events.get(ix);
        double likelihood = calculateSegmentDensity(event.getType(), event.getHeight(), end);

        while (ix > 0) {
            ix -= 1;
            EventNode next = events.get(ix);
            double e = event.getHeight();
            double s = next.getHeight();
            int i = event.getType();
            int j = next.getType();

            double nextLik = calculateSegmentDensity(j, s, e);
            if (i == j) {
                double sum = 2 * parameterization.getSymTransition(i, i) * P0Map.get(i).value(e);
                for (int k = 0; k < nTypes; k++) {
                    sum += parameterization.getAsymTransition(i, k) * P0Map.get(k).value(e);
                }
                likelihood += Math.log(sum) + nextLik;
            } else {
                likelihood += Math.log(2 * parameterization.getSymTransition(i, j) * P0Map.get(j).value(e) +
                        parameterization.getAsymTransition(i, j) * P0Map.get(i).value(e)) +
                        nextLik;
            }

            event = next;
        }

        return likelihood;
    }


    private double calculateSegmentDensity(int type, double start, double end) {
        double density;
        if (start == 0.0) {
            density = Math.log(parameterization.getSampling(type)) + Math.log1p(-gammaDistributions.get(type).cumulativeProbability(end));
        } else {
            density = Math.log1p(-parameterization.getDeath(type)) + gammaDistributions.get(type).logDensity(end - start);
        }
        return density;
    }


    // track time backwards (start < end)
    private double[] calculateSubtreeLikelihood(Node node, double start, double end, double[][] subtreeLikelihood) {

        double[] likelihood = new double[nTypes];
        int type = getTypeOf(node);

        // upstream branch
        double[][] branchDensity = calculateNodeLikelihood(node, start, end);

        // at tips
        if (node.isLeaf()) {
            for (int i = 0; i < nTypes; i++) {
                likelihood[i] = branchDensity[i][type];
            }

            // recursion bottom-up
        } else {
            Node leftChild = node.getLeft();
            Node rightChild = node.getRight();
            double[] leftSubtreeLik = calculateSubtreeLikelihood(leftChild, leftChild.getHeight(), start, subtreeLikelihood);
            double[] rightSubtreeLik = calculateSubtreeLikelihood(rightChild, rightChild.getHeight(), start, subtreeLikelihood);

            for (int i = 0; i < nTypes; i++) {
                double lik = 0;
                for (int j = 0; j < nTypes; j++) {
                    double sum = 0;
                    for (int k = 0; k < nTypes; k++) {
                        sum += parameterization.getSymTransition(j, k) * leftSubtreeLik[k] * rightSubtreeLik[k] +
                                0.5 * parameterization.getAsymTransition(j, k) *
                                        (leftSubtreeLik[j] * rightSubtreeLik[k] + leftSubtreeLik[k] * rightSubtreeLik[j]);
                    }
                    lik += branchDensity[i][j] * sum;
                }
                likelihood[i] = lik;
            }
        }

        System.arraycopy(likelihood, 0, subtreeLikelihood[node.getNr()], 0, nTypes);
        return likelihood;
    }


    // calculate the density of the branch upstream of node
    private double[][] calculateNodeLikelihood(Node node, double start, double end) {

        double[][] likelihood = new double[nTypes][nTypes];

        if (node.isLeaf()) { // tip - sampling event
            assert start == 0.0;
            for (int i = 0; i < nTypes; i++) {
                for (int j = 0; j < nTypes; j++) {
                    likelihood[i][j] = P1Map.get(new Pair<>(i, j)).value(end);
                }
            }

        } else {
            if (approx) {
                likelihood[0][0] = approximateBranchDensity(start, end);
            } else {
                likelihood = calculateBranchDensity(start, end);
            }
        }

        return likelihood;
    }


    // calculate the density of an internal branch (s,e) with types i -> j
    // requires solving a system of integral equations; very slow
    private double[][] calculateBranchDensity(double start, double end) {

        double[][] B = new double[nTypes][nTypes];

        // generate linearly spaced values between start and end
        double[] seq = new double[nSteps];
        double[] ageSeq = new double[nSteps];
        double dx = (end - start) / nSteps;
        for (int w = 0; w < nSteps; w++) {
            seq[w] = start + dx * (w + 1);
            ageSeq[w] = seq[w] - start;
        }

        // calculate the FFT from PDF of the gamma distribution and interpolate P0 per type, initialize matrix
        Complex[][] pdfFFT = new Complex[nTypes][nSteps * 2];
        double[][] P0 = new double[nTypes][nSteps];
        double[][][] X0 = new double[nTypes][nTypes][nSteps];
        for (int i = 0; i < nTypes; i++) {
            GammaDistribution gammaDist = gammaDistributions.get(i);
            double[] pdf = new double[nSteps];
            for (int w = 0; w < nSteps; w++) {
                pdf[w] = Math.exp(gammaDist.logDensity(ageSeq[w])); // get density
                P0[i][w] = P0Map.get(i).value(seq[w]); // extrapolate P0
                X0[i][i][w] = (1 - parameterization.getDeath(i)) * pdf[w]; // initialize matrix
            }

            Complex[] Ft = Utils.fft.transform(Utils.padZeros(pdf), TRANSFORM_FORWARD);
            for (int w = 0; w < nSteps * 2; w++) {
                pdfFFT[i][w] = Ft[w];
            }
        }

        // set up iteration
        double err = 1;
        int it = 0;
        double[][][] X = X0;

        // iterate
        while (err > tol && it < maxIt) {
            double[][][] Xn = new double[nTypes][nTypes][nSteps];

            for (int i = 0; i < nTypes; i++) {
                for (int j = 0; j < nTypes; j++) {
                    // get vectors for convolution
                    double[] y = new double[nSteps];
                    for (int w = 0; w < nSteps; w++) { // multiply elementwise on times
                        for (int k = 0; k < nTypes; k++) { // sum over all types k
                            y[w] += parameterization.getSymTransition(i, k) * P0[k][w] * X[k][j][w] +
                                    0.5 * parameterization.getAsymTransition(i, k) * (P0[i][w] * X[k][j][w] + P0[k][w] * X[i][j][w]);
                        }
                    }

                    // extract column from the pdf matrix
                    Complex[] Ft = pdfFFT[i];

                    // partially convolve
                    double[] I = Utils.convolveFFT(Ft, y, nSteps, dx);

                    // sum
                    for (int w = 0; w < nSteps; w++) {
                        Xn[i][j][w] = X0[i][j][w] + 2 * (1 - parameterization.getDeath(i)) * I[w];
                    }
                }
            }

            // compute error
            err = Utils.getError(X, Xn);

            // update
            X = Xn;
            it++;
        }

        if (it == maxIt) {
            System.err.printf("calculateBranchDensity Warning: max iterations reached with error: %.2f%n", err);
        }

        // take final time slice only
        for (int i = 0; i < nTypes; i++) {
            for (int j = 0; j < nTypes; j++) {
                B[i][j] = X[i][j][nSteps - 1];
            }
        }

        return B;
    }


    // single-type approximation
    private double approximateBranchDensity(double start, double end) {

        assert nTypes == 1;
        double[] P0 = P0System.getP0()[0];

        // get average P0 over branch
        // use binary search to find the closest indices to the branch lengths in time array (t0 is sorted per definition!)
        double[] P0Slice = Arrays.copyOfRange(P0, Utils.findClosestIndex(timeArray, start), Utils.findClosestIndex(timeArray, end) + 1);
        double P0M = Utils.getMean(P0Slice); // TODO: take extrapolated values using P0Map

        // initialize the approximation
        double lifetime = parameterization.getLifetime(0);
        double shape = parameterization.getShape(0);
        double scale = lifetime / shape;
        int k = (int) ((end - start) / (shape * scale)); // around this k, the term b_k will be maximal // TODO: update maximum (factor)
        double density = 0;

        int i = 0; // increase (start with 0 and treat k as minimum number of terms)
        double term = 1;
        while (term > tol || i <= k) { // only stops when both term <= tol and i > k!
            GammaDistribution gammaDist;
            // check if GammaDistribution with the given shape is already in the cache
            if (gammaCache.containsKey(i + 1)) {
                gammaDist = gammaCache.get(i + 1);
            } else { // otherwise, add it
                gammaDist = new GammaDistribution(null, (i + 1) * shape, scale);
                gammaCache.put(i + 1, gammaDist);
            }
            term = Math.pow(2, i) * Math.pow(1 - parameterization.getDeath(0), i + 1) * Math.pow(P0M, i) * Math.exp(gammaDist.logDensity(end - start));
            density += term;
            i++;
        }

        return density;
    }


    @Override
    public boolean requiresRecalculation() {
        lifetimeDistributions.findDirty();
        P0System.isDirty();
        if (!(tree instanceof AnnotatedTree)) {
            P1System.isDirty();
        }
        return true;
    }


    @Override
    public void store() {
        lifetimeDistributions.store();
        P0System.store();
        if (!(tree instanceof AnnotatedTree)) {
            P1System.store();
        }
        super.store();
    }


    @Override
    public void restore() {
        lifetimeDistributions.restore();
        P0System.restore();
        if (!(tree instanceof AnnotatedTree)) {
            P1System.restore();
        }
        super.restore();
    }

    @Override
    public void accept() {
        lifetimeDistributions.accept();
        P0System.accept();
        if (!(tree instanceof AnnotatedTree)) {
            P1System.accept();
        }
        super.accept();
    }

}
