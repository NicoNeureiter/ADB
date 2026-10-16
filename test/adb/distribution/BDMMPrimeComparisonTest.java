package adb.distribution;

import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeParser;
import beast.base.util.Randomizer;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import static adb.distribution.ReferenceModels.model;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;


/**
 * Compares the integrated density of ADBTreeDistribution with the density of BDMM-Prime for shape 1, where the age-dependent
 * process is a multi-type birth-death process (see BDMMPrimeReference). Both give the density of a tree with
 * labelled tips (factor 2^(n-1)/n!) conditioned on the survival of the process from the origin, so the log densities
 * are equal, apart from the discretisation error of ADBTreeDistribution.
 *
 * The trees are simulated from the model by BDMM-Prime (with a fixed seed), so they are typical for the model, with
 * random shapes and tip types, from a single tip upwards. The models are mostly complex: up to 5 types, with death
 * and incomplete sampling, hidden type changes, and an origin several lifetimes long. The trees have internal nodes,
 * so the division of internal nodes, with symmetric and asymmetric parts, and the integrated branch density are
 * tested, which the sums over hidden histories in ADBTreeDistributionTest cannot reach.
 *
 * The discretisation error is of the first order in 1 / nSteps, so a quarter of the step gives a quarter of the error.
 * A model or formula error does not shrink with nSteps, and it remains after the extrapolation (4 fine - coarse) / 3,
 * which removes the first-order error: for every tree that value has to be small compared with the coarse error.
 * The total error (the sum over the trees of ADB minus BDMM-Prime, in log density) at nSteps 2^10, 2^12 and 2^14 is
 * printed for every model.
 *
 * Tagged slow: exclude with --exclude-tag slow.
 */
@Tag("slow")
public class BDMMPrimeComparisonTest {

    private static final int[] STEPS = {1 << 10, 1 << 12, 1 << 14};


    private static Parameterization fourTypes(double[] death, double[] sampling, double originTime) {
        return model(new double[]{0.5, 1, 2, 3}, new int[]{1, 1, 1, 1}, death, sampling,
                new double[][]{{0.4, 0, 0, 0}, {0, 0.2, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}},
                new double[][]{{0, 0.6, 0, 0}, {0, 0, 0.5, 0.3}, {0, 0, 0, 0}, {0, 0, 0, 0}}, originTime);
    }


    /** A hierarchy of five types with different lifetimes, death and sampling; type 4 only divides symmetrically. */
    private static Parameterization fiveTypes(double originTime) {
        return model(new double[]{0.5, 1, 1.5, 2, 3}, new int[]{1, 1, 1, 1, 1},
                new double[]{0.05, 0.2, 0.1, 0.1, 0.15}, new double[]{0.4, 0.5, 0.3, 0.6, 0.5},
                new double[][]{{0.3, 0, 0, 0, 0}, {0, 0.4, 0, 0, 0}, {0, 0, 0.5, 0, 0}, {0, 0, 0, 0.7, 0}, {0, 0, 0, 0, 1}},
                new double[][]{{0, 0.4, 0.3, 0, 0}, {0, 0, 0.3, 0.3, 0}, {0, 0, 0, 0.25, 0.25}, {0, 0, 0, 0, 0.3}, {0, 0, 0, 0, 0}},
                originTime);
    }


    private static double adbLogDensity(Parameterization model, Tree tree, int nSteps) {
        ADBTreeDistribution distribution = new ADBTreeDistribution();
        distribution.initByName("tree", tree,
                "parameterization", model,
                "approx", false,
                "conditionOnOrigin", true,
                "nSteps", nSteps);
        return distribution.calculateTreeLogLikelihood(tree);
    }


    /** Compares the densities of nTrees simulated trees and prints the errors. */
    private static void compareOnSimulatedTrees(String modelName, Parameterization model, int nTrees, long seed) {
        Randomizer.setSeed(seed);
        long start = System.nanoTime();

        double[] totalError = new double[STEPS.length];
        double[] maxError = new double[STEPS.length];
        int[] tips = new int[nTrees];

        for (int k = 0; k < nTrees; k++) {
            Tree tree = BDMMPrimeReference.simulateTree(model);
            tips[k] = tree.getLeafNodeCount();
            String name = "tree " + k + " with " + tips[k] + " tips: " + tree.getRoot().toNewick();

            double reference = BDMMPrimeReference.logDensity(model, tree);
            assertTrue(Double.isFinite(reference), name);

            double[] error = new double[STEPS.length];
            for (int s = 0; s < STEPS.length; s++) {
                error[s] = adbLogDensity(model, tree, STEPS[s]) - reference;
                totalError[s] += error[s];
                maxError[s] = Math.max(maxError[s], Math.abs(error[s]));
            }

            // the two finest grids: less than 5% of the error on the coarser one is left, or less than 1e-4 in total
            double extrapolated = (4 * error[2] - error[1]) / 3;
            assertEquals(0.0, extrapolated, 0.05 * Math.abs(error[1]) + 1e-4,
                    name + ": errors " + java.util.Arrays.toString(error));
        }

        java.util.Arrays.sort(tips);
        System.out.printf("%s: %d trees, tips %d / %d / %d (min / median / max), %.0f s%n", modelName, nTrees,
                tips[0], tips[nTrees / 2], tips[nTrees - 1], (System.nanoTime() - start) / 1e9);
        for (int s = 0; s < STEPS.length; s++) {
            System.out.printf("    nSteps 2^%d: total error %.3e, mean %.3e, largest |error| %.3e%n",
                    Integer.numberOfTrailingZeros(STEPS[s]), totalError[s], totalError[s] / nTrees, maxError[s]);
        }
        assertTrue(tips[nTrees - 1] >= 2, "the simulated trees should not all be a single tip");
    }


    @Test
    public void testOneTypeWithDeathAndIncompleteSampling() {
        compareOnSimulatedTrees("1 type", ReferenceModels.singleType(1.0, 1, 0.4, 0.1, 12.0), 15, 12);
    }


    @Test
    public void testTwoTypesWithBackTransitions() {
        // type 0 and 1 give rise to each other
        compareOnSimulatedTrees("2 types", ReferenceModels.twoTypes(1, 1, 6.0), 20, 13);
    }


    @Test
    public void testThreeTypes() {
        compareOnSimulatedTrees("3 types", ReferenceModels.threeTypes(1, 1, 1, 4.0), 15, 14);
    }


    @Test
    public void testFourTypes() {
        compareOnSimulatedTrees("4 types",
                fourTypes(new double[]{0.1, 0.2, 0.1, 0.1}, new double[]{0.5, 0.5, 0.5, 0.5}, 3.0), 12, 15);
    }


    @Test
    public void testFiveTypes() {
        compareOnSimulatedTrees("5 types", fiveTypes(3.0), 10, 17);
    }


    @Test
    public void testImpossibleTreeHasZeroDensityInBoth() {
        // without death and with full sampling nothing is hidden, so the tips C and D cannot be of the types 2 and 3
        // (their parent can only give rise to 1 and 2, or 1 and 3)
        Parameterization model = fourTypes(new double[]{0, 0, 0, 0}, new double[]{1, 1, 1, 1}, 5.0);
        Tree tree = new TreeParser(
                "((A[&type=0]:1.0,B[&type=1]:1.0):1.5,(C[&type=2]:1.2,D[&type=3]:1.2):1.3):0.0;", true);

        assertEquals(Double.NEGATIVE_INFINITY, BDMMPrimeReference.logDensity(model, tree));
        assertEquals(Double.NEGATIVE_INFINITY, adbLogDensity(model, tree, STEPS[0]));
    }


    @Test
    public void testModelsWithoutBDMMPrimeCounterpart() {
        // a symmetric division of type 0 into two lineages of type 1 is not a BDMM-Prime event
        Parameterization symmetricChange = model(new double[]{1, 1}, new int[]{1, 1}, new double[]{0.1, 0.1},
                new double[]{0.5, 0.5}, new double[][]{{0.5, 0.5}, {0, 1}}, new double[][]{{0, 0}, {0, 0}}, 5.0);
        assertFalse(BDMMPrimeReference.isRepresentable(symmetricChange));

        Parameterization shape2 = ReferenceModels.singleType(1.0, 2, 0.1, 0.5, 5.0);
        assertFalse(BDMMPrimeReference.isRepresentable(shape2));

        assertTrue(BDMMPrimeReference.isRepresentable(ReferenceModels.twoTypes(1, 1, 5.0)));
    }
}
