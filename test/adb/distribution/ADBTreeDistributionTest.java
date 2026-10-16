package adb.distribution;

import adb.tree.AnnotatedTreeParser;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeParser;
import beast.base.inference.parameter.IntegerParameter;
import beast.base.inference.parameter.RealParameter;
import org.apache.commons.math3.distribution.GammaDistribution;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.function.Executable;

import static adb.distribution.ReferenceModels.*;
import static org.junit.jupiter.api.Assertions.*;


/**
 * Checks of the tree density of ADBTreeDistribution:
 * the annotated density against a hand-computed history,
 * the annotated density summed over all hidden histories against the integrated density,
 * the series approximation against the integrated density,
 * and the cases in which the density is zero or the input is rejected.
 */
public class ADBTreeDistributionTest {

    private static final String FOUR_TIP_TREE = "((D:5.0,C:5.0):6.0,(A:8.0,B:8.0):3.0):0.0;";


    private ADBTreeDistribution distribution(Tree tree, Parameterization model, boolean approx, int nSteps) {
        ADBTreeDistribution distribution = new ADBTreeDistribution();
        distribution.initByName("tree", tree,
                "parameterization", model,
                "approx", approx,
                "conditionOnOrigin", true,
                "nSteps", nSteps,
                "maxIterations", 500,
                "tolerance", 1e-12);
        return distribution;
    }


    // ---- hand-computed history ----

    @Test
    public void testTwoTipAnnotatedLogLikelihoodAgainstManualHistory() throws Exception {
        double originTime = 2.0;
        double rootHeight = 1.0;
        double tipEventHeight = 0.35;
        double stemEventHeight = 1.45;

        Parameterization model = singleType(1.0, 2, 0.2, 0.4, originTime);

        AnnotatedTreeParser tree = new AnnotatedTreeParser();
        tree.initByName(
                "newick", "(((1:0.35):0.65,2:1.0):0.45);",
                "adjustTipHeights", true);

        ADBTreeDistribution distribution = distribution(tree, model, false, 1 << 14);

        double actual = distribution.calculateTreeLogLikelihood(tree);
        double expected = expectedTwoTipAnnotatedLogLikelihood(
                distribution, model, rootHeight, tipEventHeight, stemEventHeight, originTime);

        assertTrue(Double.isFinite(actual));
        assertEquals(expected, actual, 1e-10);
    }



    // ---- sum over hidden histories ----

    @Test
    public void testSingleTypeHistoriesSumToIntegratedLikelihood() throws Exception {
        double originTime = 0.2;
        Parameterization model = singleType(1.0, 1, 0.2, 0.4, originTime);

        Tree integratedTree = new TreeParser("1;", true);
        ADBTreeDistribution integrated = distribution(integratedTree, model, false, 4096);
        ADBTreeDistribution annotated = distribution(singleTipTree(new int[]{0}, new double[0]), model, false, 4096);

        double integratedLogL = integrated.calculateTreeLogLikelihood(integratedTree);
        double noEventLogL = annotated.calculateTreeLogLikelihood(singleTipTree(new int[]{0}, new double[0]));
        double summedLogL = logSumTipHistories(annotated, model, 0, 2, 100);

        assertTrue(Double.isFinite(integratedLogL));
        assertTrue(summedLogL > noEventLogL, "hidden events add probability");
        assertEquals(integratedLogL, summedLogL, 5e-3);
    }


    @Test
    public void testTwoTypeHistoriesSumToIntegratedLikelihood() throws Exception {
        // a tip of type 1 and an origin of type 0: the lineage has to change its type, directly or in several steps
        double originTime = 0.3;
        Parameterization model = model(new double[]{0.6, 0.9}, new int[]{2, 1}, new double[]{0.1, 0.2},
                new double[]{0.5, 0.4}, new double[][]{{0.2, 0}, {0, 0.6}}, new double[][]{{0, 0.8}, {0.4, 0}}, originTime);

        for (int tipType = 0; tipType < 2; tipType++) {
            Tree integratedTree = new TreeParser("1[&type=" + tipType + "];", true);
            ADBTreeDistribution integrated = distribution(integratedTree, model, false, 4096);
            ADBTreeDistribution annotated = distribution(
                    singleTipTree(new int[]{tipType}, new double[0]), model, false, 4096);

            double integratedLogL = integrated.calculateTreeLogLikelihood(integratedTree);
            double twoEvents = logSumTipHistories(annotated, model, tipType, 2, 20);
            double threeEvents = logSumTipHistories(annotated, model, tipType, 3, 20);

            // histories with up to 3 hidden events are most, but not all of the probability
            assertTrue(Double.isFinite(integratedLogL));
            assertTrue(twoEvents < threeEvents, "tip type " + tipType);
            assertTrue(threeEvents < integratedLogL + 1e-3, "tip type " + tipType);
            assertEquals(integratedLogL, threeEvents, 1.5e-2, "tip type " + tipType);
        }
    }


    // ---- approximation ----

    @Test
    public void testApproximationIsCloseToIntegratedLikelihood() throws Exception {
        for (int shape : new int[]{2, 5}) {
            Parameterization model = singleType(5.0, shape, 0.1, 0.1, 15.0);
            Tree tree = new TreeParser(FOUR_TIP_TREE, true);

            double integrated = distribution(tree, model, false, 1 << 12).calculateTreeLogLikelihood(tree);
            double approximated = distribution(tree, model, true, 1 << 12).calculateTreeLogLikelihood(tree);

            assertTrue(Double.isFinite(integrated));
            assertEquals(integrated, approximated, 5e-3, "shape " + shape);
        }
    }


    // ---- impossible trees and invalid input ----

    @Test
    public void testRootAboveOriginHasZeroDensity() throws Exception {
        Parameterization model = singleType(5.0, 2, 0.1, 0.1, 10.0);
        Tree tree = new TreeParser("((A:12.0,B:12.0):0.0);", true);

        assertEquals(Double.NEGATIVE_INFINITY, distribution(tree, model, false, 1 << 10).calculateTreeLogLikelihood(tree));
    }


    @Test
    public void testCertainExtinctionHasZeroDensityTwoTypes() throws Exception {
        // nothing is sampled, so a tree with sampled tips is impossible
        Parameterization model = model(new double[]{2, 5}, new int[]{2, 3}, new double[]{0.1, 0.2}, new double[]{0, 0},
                new double[][]{{0.2, 0}, {0, 0.6}}, new double[][]{{0, 0.8}, {0.4, 0}}, 10.0);
        Tree tree = new TreeParser("(A[&type=0]:3.0,B[&type=1]:3.0):0.0;", true);

        assertEquals(Double.NEGATIVE_INFINITY, distribution(tree, model, false, 1 << 10).calculateTreeLogLikelihood(tree));
    }


    @Test
    public void testCertainExtinctionHasZeroDensitySingleType() throws Exception {
        Parameterization model = singleType(5.0, 2, 0.1, 0.0, 10.0);
        Tree tree = new TreeParser("(A:3.0,B:3.0):0.0;", true);

        assertEquals(Double.NEGATIVE_INFINITY, distribution(tree, model, false, 1 << 10).calculateTreeLogLikelihood(tree));
    }


    @Test
    public void testInvalidInputIsRejected() throws Exception {
        Parameterization twoTypes = twoTypes(1, 1, 10.0);
        Parameterization oneType = singleType(5.0, 2, 0.1, 0.1, 10.0);
        Tree tree = new TreeParser("(A:3.0,B:3.0):0.0;", true);

        // a tip type that does not exist
        Tree badTypes = new TreeParser("(A[&type=2]:3.0,B[&type=0]:3.0):0.0;", true);
        assertRejected(() -> distribution(badTypes, twoTypes, false, 1 << 10), "tips are valid");

        // the number of steps has to be a power of 2
        assertRejected(() -> distribution(tree, oneType, false, 1000), "power of 2");

        // the series is only for one type
        assertRejected(() -> distribution(tree, twoTypes, true, 1 << 10), "single-type");

        // conditioning on the origin needs an origin time
        Parameterization noOrigin = new Parameterization();
        noOrigin.initByName("nTypes", 1,
                "lifetime", new RealParameter("5"),
                "shape", new IntegerParameter("2"),
                "death", new RealParameter("0.1"),
                "sampling", new RealParameter("0.1"));
        assertRejected(() -> distribution(tree, noOrigin, false, 1 << 10), "time of origin");
    }


    // initByName wraps the exception of initAndValidate in a RuntimeException that keeps only the message
    private void assertRejected(Executable construction, String messagePart) {
        RuntimeException thrown = assertThrows(RuntimeException.class, construction);
        assertTrue(thrown.getMessage().contains(messagePart), "unexpected message: " + thrown.getMessage());
    }


    // ---- helpers ----

    private double expectedTwoTipAnnotatedLogLikelihood(ADBTreeDistribution distribution, Parameterization model,
                                                       double rootHeight, double tipEventHeight,
                                                       double stemEventHeight, double originTime) {
        double conditionFactor = -Math.log1p(-distribution.P0System.getP0()[0][distribution.nSteps - 1]);
        double hiddenTipEvent = Math.log(2.0 * distribution.P0Map.get(0).value(tipEventHeight));
        double hiddenStemEvent = Math.log(2.0 * distribution.P0Map.get(0).value(stemEventHeight));

        double stem = internalSegmentLogDensity(model, stemEventHeight, originTime) +
                hiddenStemEvent +
                internalSegmentLogDensity(model, rootHeight, stemEventHeight);
        double leftTip = internalSegmentLogDensity(model, tipEventHeight, rootHeight) +
                hiddenTipEvent +
                tipSegmentLogDensity(model, tipEventHeight);
        double rightTip = tipSegmentLogDensity(model, rootHeight);

        return conditionFactor + stem + leftTip + rightTip;
    }


    private double internalSegmentLogDensity(Parameterization model, double start, double end) {
        return Math.log((1.0 - model.getDeath(0)) * gammaDistribution(model).density(end - start));
    }


    private double tipSegmentLogDensity(Parameterization model, double end) {
        return Math.log(model.getSampling(0)) + Math.log1p(-gammaDistribution(model).cumulativeProbability(end));
    }


    private GammaDistribution gammaDistribution(Parameterization model) {
        double shape = model.getShape(0);
        double scale = model.getLifetime(0) / shape;
        return new GammaDistribution(null, shape, scale);
    }


    /**
     * Log of the sum of the annotated densities over all histories of a single tip with at most maxEvents hidden
     * events: the event heights on a grid with gridSize cells (midpoints, each event with the weight of its cell),
     * and all types of the segments between the tip type and the origin type. The origin type is the one of the model.
     */
    private double logSumTipHistories(ADBTreeDistribution distribution, Parameterization model, int tipType,
                                      int maxEvents, int gridSize) {
        double dx = model.getOriginTime() / gridSize;
        int originType = model.getOriginType();

        // no hidden event: the tip lineage has to be of the origin type
        double logSum = Double.NEGATIVE_INFINITY;
        if (tipType == originType) {
            logSum = distribution.calculateTreeLogLikelihood(singleTipTree(new int[]{tipType}, new double[0]));
        }

        for (int eventCount = 1; eventCount <= maxEvents; eventCount++) {
            logSum = addHeights(distribution, model, tipType, eventCount, new double[eventCount], 0, 0,
                    gridSize, dx, eventCount * Math.log(dx), logSum);
        }
        return logSum;
    }


    private double addHeights(ADBTreeDistribution distribution, Parameterization model, int tipType, int eventCount,
                              double[] heights, int depth, int startIndex, int gridSize, double dx,
                              double logWeight, double logSum) {
        if (depth == eventCount) {
            int[] types = new int[eventCount + 1];
            types[0] = tipType;
            types[eventCount] = model.getOriginType();
            return addTypes(distribution, model, types, heights, 1, logWeight, logSum);
        }

        int remainingEvents = eventCount - depth;
        for (int i = startIndex; i <= gridSize - remainingEvents; i++) {
            heights[depth] = (i + 0.5) * dx;
            logSum = addHeights(distribution, model, tipType, eventCount, heights, depth + 1, i + 1,
                    gridSize, dx, logWeight, logSum);
        }
        return logSum;
    }


    // the types of the segments above the events 1, ..., m - 1 are free, the one above the last event is the origin type
    private double addTypes(ADBTreeDistribution distribution, Parameterization model, int[] types, double[] heights,
                            int position, double logWeight, double logSum) {
        if (position == types.length - 1) {
            double logLikelihood = distribution.calculateTreeLogLikelihood(singleTipTree(types, heights));
            return logSumExp(logSum, logLikelihood + logWeight);
        }
        for (int type = 0; type < model.getNTypes(); type++) {
            types[position] = type;
            logSum = addTypes(distribution, model, types, heights, position + 1, logWeight, logSum);
        }
        return logSum;
    }


    /**
     * A tree with a single tip and hidden events at the given heights. The type of the tip is types[0], and
     * types[k] is the type of the lineage above the event k.
     */
    private AnnotatedTreeParser singleTipTree(int[] types, double[] eventHeights) {
        AnnotatedTreeParser tree = new AnnotatedTreeParser();
        tree.initByName(
                "newick", singleTipNewick(types, eventHeights),
                "adjustTipHeights", true);
        return tree;
    }


    private String singleTipNewick(int[] types, double[] eventHeights) {
        String newick = "1[&type=" + types[0] + "]";
        double previousHeight = 0.0;
        for (int k = 0; k < eventHeights.length; k++) {
            newick = String.format("(%s:%.16f)[&type=%d]", newick, eventHeights[k] - previousHeight, types[k + 1]);
            previousHeight = eventHeights[k];
        }
        return newick + ";";
    }


    private double logSumExp(double x, double y) {
        if (x == Double.NEGATIVE_INFINITY) {
            return y;
        }
        if (y == Double.NEGATIVE_INFINITY) {
            return x;
        }
        if (y > x) {
            double tmp = x;
            x = y;
            y = tmp;
        }
        return x + Math.log1p(Math.exp(y - x));
    }

}
