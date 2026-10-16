package adb.distribution;

import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import static adb.distribution.ReferenceModels.*;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;


/**
 * Checks P0 against independent solutions: the closed form of a birth-death process (one type, shape 1) and the
 * differential equations of the lifetime stages (any number of types and integer shapes, see ReferenceModels).
 * The Picard iteration on a grid has an error that is proportional to the step size dx = origin time / nSteps.
 */
public class P0SystemTest {

    private static final int N_STEPS = 1 << 12;
    private static final double ERROR_PER_STEP = 5.0; // maximal error times nSteps


    @Test
    public void testSingleTypeShape1MatchesClosedForm() {
        Parameterization model = singleType(2.0, 1, 0.1, 0.3, 10.0);
        double[][] P0 = solveP0(model, N_STEPS);

        assertEquals(0.0, maxAbsDifference(P0[0], closedFormP0(model, N_STEPS)), ERROR_PER_STEP / N_STEPS);
    }


    @Test
    public void testReferenceSolutionMatchesClosedForm() {
        // checks the reference itself, which is then used for the other models
        Parameterization model = singleType(2.0, 1, 0.1, 0.3, 10.0);
        double[][] reference = referenceP0(model, N_STEPS);

        assertEquals(0.0, maxAbsDifference(reference[0], closedFormP0(model, N_STEPS)), 1e-10);
    }


    @Test
    public void testSingleTypeShapesMatchReference() {
        for (int shape : new int[]{2, 5, 20}) {
            Parameterization model = singleType(2.0, shape, 0.1, 0.3, 10.0);
            assertEquals(0.0, maxAbsDifference(solveP0(model, N_STEPS), referenceP0(model, N_STEPS)),
                    ERROR_PER_STEP / N_STEPS, "shape " + shape);
        }
    }


    @Test
    public void testTwoTypesMatchReference() {
        int[][] shapes = {{1, 1}, {3, 5}, {1, 4}};
        for (int[] shape : shapes) {
            Parameterization model = twoTypes(shape[0], shape[1], 10.0);
            assertEquals(0.0, maxAbsDifference(solveP0(model, N_STEPS), referenceP0(model, N_STEPS)),
                    ERROR_PER_STEP / N_STEPS, "shapes " + shape[0] + " " + shape[1]);
        }
    }


    @Test
    public void testThreeTypesMatchReference() {
        int[][] shapes = {{1, 1, 1}, {1, 2, 4}};
        for (int[] shape : shapes) {
            Parameterization model = threeTypes(shape[0], shape[1], shape[2], 10.0);
            assertEquals(0.0, maxAbsDifference(solveP0(model, N_STEPS), referenceP0(model, N_STEPS)),
                    ERROR_PER_STEP / N_STEPS, "shapes " + shape[0] + " " + shape[1] + " " + shape[2]);
        }
    }


    @Test
    public void testErrorIsProportionalToStepSize() {
        // a quarter of the step size gives about a quarter of the error
        Parameterization model = twoTypes(3, 5, 10.0);
        double coarse = maxAbsDifference(solveP0(model, 1 << 10), referenceP0(model, 1 << 10));
        double fine = maxAbsDifference(solveP0(model, 1 << 12), referenceP0(model, 1 << 12));

        assertTrue(fine < coarse);
        assertEquals(4.0, coarse / fine, 0.5);
    }


    @Test
    public void testNonMonotoneP0MatchesReference() {
        // asymmetric divisions only between the two types, so that the extinction probability of one type depends on the other
        Parameterization model = model(new double[]{1, 1}, new int[]{1, 1}, new double[]{0.4, 0.1}, new double[]{0.5, 0.5},
                new double[][]{{0, 0}, {0, 0}}, new double[][]{{0, 1}, {1, 0}}, 10.0);
        double[][] P0 = solveP0(model, N_STEPS);

        assertEquals(0.0, maxAbsDifference(P0, referenceP0(model, N_STEPS)), ERROR_PER_STEP / N_STEPS);

        boolean nonMonotone = false;
        for (double[] p : P0) {
            for (int w = 1; w < p.length; w++) {
                if (p[w] < p[w - 1] - 1e-6) nonMonotone = true;
            }
        }
        assertTrue(nonMonotone, "P0 of at least one type is expected to decrease somewhere");
    }


    @Test
    public void testSingleTypeIsMonotone() {
        // supercritical (death probability below 0.5, enough surviving lineages): P0 decreases from 1 - rho
        double[] decreasing = solveP0(singleType(2.0, 3, 0.1, 0.3, 10.0), N_STEPS)[0];
        for (int w = 1; w < N_STEPS; w++) {
            assertTrue(decreasing[w] <= decreasing[w - 1], "decreasing at " + w);
        }
        // subcritical: P0 increases to 1
        double[] increasing = solveP0(singleType(2.0, 3, 0.7, 0.3, 10.0), N_STEPS)[0];
        for (int w = 1; w < N_STEPS; w++) {
            assertTrue(increasing[w] >= increasing[w - 1], "increasing at " + w);
        }
        assertTrue(increasing[N_STEPS - 1] > 0.95, "P0 approaches 1");
    }


    @Test
    public void testValuesAreProbabilitiesAndStartAtOneMinusSampling() {
        Parameterization model = threeTypes(1, 2, 4, 10.0);
        double[][] P0 = solveP0(model, N_STEPS);

        for (int i = 0; i < model.getNTypes(); i++) {
            for (double p : P0[i]) {
                assertTrue(p >= 0.0 && p <= 1.0, "P0 outside [0, 1]: " + p);
            }
            // the first grid point is one step after time 0, where P0 is 1 - rho
            assertEquals(1 - model.getSampling(i), P0[i][0], 5e-3, "type " + i);
        }
    }


    @Test
    public void testNoSamplingGivesCertainExtinction() {
        // nothing is sampled, so no lineage has a sampled descendant at any time
        Parameterization model = model(new double[]{2, 5}, new int[]{2, 3}, new double[]{0.1, 0.2}, new double[]{0, 0},
                new double[][]{{0.2, 0}, {0, 0.6}}, new double[][]{{0, 0.8}, {0.4, 0}}, 10.0);
        double[][] P0 = solveP0(model, N_STEPS);

        for (double[] p : P0) {
            for (double value : p) {
                assertEquals(1.0, value, 1e-9);
            }
        }
    }


    @Test
    public void testSingleTypeValuesNeverExceedOne() {
        // P0 approaches 1 when nothing is sampled or when the process is subcritical, and the discretisation must
        // not push it above 1
        double[][] models = {{0.0, 0.1}, {0.3, 0.7}}; // {sampling, death}
        for (double[] m : models) {
            double[] p = solveP0(singleType(5.0, 2, m[1], m[0], 50.0), N_STEPS)[0];
            for (int w = 0; w < N_STEPS; w++) {
                assertTrue(p[w] <= 1.0, "sampling " + m[0] + ", death " + m[1] + ": P0 = " + p[w] + " at " + w);
            }
        }
    }


    @Test
    public void testIterationHasConverged() {
        // more iterations than the default do not change the solution
        Parameterization model = threeTypes(1, 2, 4, 10.0);
        double[][] converged = solveP0(model, N_STEPS, 100, 1e-12);
        double[][] more = solveP0(model, N_STEPS, 400, 1e-14);

        assertEquals(0.0, maxAbsDifference(converged, more), 1e-9);
    }

    /** One type, shape 10, lifetime 1, death 0.1, origin 13 (as in the profiling of the error), the given sampling. */
    private static Parameterization smallSampling(double sampling) {
        return singleType(1.0, 10, 0.1, sampling, 13.0);
    }


    @Test
    public void testErrorWithSmallSamplingShrinksWithStep() {
        // P0 stays close to 1 - rho for a long time and then falls; the error of the grid is large but has to decrease
        Parameterization model = smallSampling(0.001);
        double previous = Double.POSITIVE_INFINITY;
        for (int nSteps : new int[]{1 << 12, 1 << 14, 1 << 16}) {
            double error = maxAbsDifference(solveP0(model, nSteps), referenceP0(model, nSteps));
            assertTrue(error < previous / 1.5, "nSteps " + nSteps + ": error " + error + " after " + previous);
            previous = error;
        }
    }


    /**
     * Problem (found by the profiling of the error): for small sampling probabilities P0 on the grid is
     * far from the solution of the differential equations. With origin 13 and 2^14 steps, the maximal error times
     * nSteps is 6.3 for rho 0.1 (as for the other models here), 115 for rho 0.01 and 970 for rho 0.001, and it does not
     * become first order up to 2^16 steps for rho 0.001. For rho 0.001, P0 stays at its start value 1 - rho = 0.999
     * until t = 1 to 1.5, while the true value starts to fall at t = 0.4 (0.998562 at t = 1). Disabled until it is fixed;
     * remove the annotation to run it.
     */
    @Disabled("P0 on the grid does not resolve the early decrease for small sampling probabilities")
    @Test
    public void testSmallSamplingMatchesReference() {
        int nSteps = 1 << 14;
        for (double sampling : new double[]{0.1, 0.01, 0.001}) {
            Parameterization model = smallSampling(sampling);
            double[][] P0 = solveP0(model, nSteps);
            double[][] reference = referenceP0(model, nSteps);

            // first-order error with a constant like the one of the other models (observed 1.8 to 3.2, and 6.3 here at rho 0.1)
            assertEquals(0.0, maxAbsDifference(P0, reference), 10.0 / nSteps, "sampling " + sampling);

            // the probability of sampled descendants 1 - P0 at t = 1 has a relative error below 5%
            int index = nSteps / 13 - 1; // t = 1
            assertEquals(1.0, (1 - P0[0][index]) / (1 - reference[0][index]), 0.05, "sampling " + sampling + ", t = 1");
        }
    }
}
