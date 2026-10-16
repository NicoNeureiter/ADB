package adb.distribution;

import adb.tree.AnnotatedTree;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.function.DoubleFunction;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;


/**
 * Profile likelihood curves: the log likelihood summed over many trees simulated by scTreeSim (SimulatedTrees), as a
 * function of one parameter with the others at their true values, should peak at the truth. This is a check of the
 * likelihood on its own, without a second implementation: a wrong formula, a missing factor or a wrong conditioning
 * moves the peak.
 */
@Tag("slow")
public class ProfileLikelihoodTest {

    /** The log likelihood of the truth may be below the maximum by this much: half the square of 3 standard errors. */
    private static final double MAX_DROP = 4.5;

    /** The grids have 15 points, the truth is the eighth. */
    private static final int N_GRID = 15;
    private static final int TRUTH = 7;


    /** One profile: the grid, the pooled log likelihood on it and the position of the truth. */
    private static final class Curve {
        final String name;
        final double[] x;
        final double[] logL;
        final int truth;

        Curve(String name, double[] x, double[] logL, int truth) {
            this.name = name;
            this.x = x;
            this.logL = logL;
            this.truth = truth;
        }

        int argmax() {
            int best = 0;
            for (int i = 1; i < x.length; i++) if (logL[i] > logL[best]) best = i;
            return best;
        }

        double drop() {
            return logL[argmax()] - logL[truth];
        }

        /** Leading coefficient of the parabola through the maximum and its two neighbours (NaN at the edge). */
        private double parabola() {
            int m = argmax();
            if (m == 0 || m == x.length - 1) return Double.NaN;
            double d01 = (logL[m] - logL[m - 1]) / (x[m] - x[m - 1]);
            double d12 = (logL[m + 1] - logL[m]) / (x[m + 1] - x[m]);
            return (d12 - d01) / (x[m + 1] - x[m - 1]);
        }

        /** Standard error from the curvature of the parabola at the maximum. */
        double standardError() {
            return 1 / Math.sqrt(-2 * parabola());
        }

        /** Position of the maximum of the parabola at the maximum (NaN at the edge of the grid). */
        double vertex() {
            int m = argmax();
            if (m == 0 || m == x.length - 1) return Double.NaN;
            double a = parabola();
            double d01 = (logL[m] - logL[m - 1]) / (x[m] - x[m - 1]);
            double b = d01 - a * (x[m] + x[m - 1]);
            return -b / (2 * a);
        }
    }


    /** The truth times 1 + k * step, for k from -7 to 7. The lifetime, death and sampling curves are over a factor on the true values. */
    private static double[] scaled(double truth, double step) {
        double[] x = new double[N_GRID];
        for (int i = 0; i < N_GRID; i++) x[i] = truth * (1 + (i - TRUTH) * step);
        return x;
    }

    /** The truth plus k * step, for k from -7 to 7. */
    private static double[] shifted(double truth, double step) {
        double[] x = new double[N_GRID];
        for (int i = 0; i < N_GRID; i++) x[i] = truth + (i - TRUTH) * step;
        return x;
    }


    /** Pooled log likelihood of a model, over a fixed set of trees. */
    private interface Pooled {
        double logLikelihood(Parameterization model);
    }

    /** Annotated trees (hidden divisions given): the complete-data likelihood. */
    private static Pooled annotated(SimulatedTrees data, int nTrees, int nSteps) {
        AnnotatedTree[] trees = new AnnotatedTree[nTrees];
        for (int k = 0; k < nTrees; k++) trees[k] = data.annotated(k);
        return model -> {
            ADBTreeDistribution distribution = new ADBTreeDistribution();
            distribution.initByName("tree", trees[0], "parameterization", model, "nSteps", nSteps);
            double sum = 0;
            for (AnnotatedTree tree : trees) sum += distribution.calculateTreeLogLikelihood(tree);
            return sum;
        };
    }

    /**
     * The same trees without their hidden divisions: the hidden divisions are integrated out, with the branch densities
     * solved by Picard iteration, or with the series approximation (one type, integer shape).
     */
    private static Pooled integrated(SimulatedTrees data, int nTrees, int nSteps, boolean approx) {
        Tree[] trees = new Tree[nTrees];
        for (int k = 0; k < nTrees; k++) trees[k] = data.collapsed(k);
        return model -> {
            ADBTreeDistribution distribution = new ADBTreeDistribution();
            distribution.initByName("tree", trees[0], "parameterization", model, "nSteps", nSteps, "approx", approx);
            double sum = 0;
            for (Tree tree : trees) sum += distribution.calculateTreeLogLikelihood(tree);
            return sum;
        };
    }

    private static Curve curve(String name, double[] grid, int truth, DoubleFunction<Parameterization> model, Pooled pooled) {
        double[] logL = new double[grid.length];
        for (int i = 0; i < grid.length; i++) {
            logL[i] = pooled.logLikelihood(model.apply(grid[i]));
        }
        return new Curve(name, grid, logL, truth);
    }


    /** The shapes 1, ..., 7 if the true shape is 1, otherwise the truth times 1 + k * width, for k from -3 to 3. */
    private static Curve shapeCurve(SimulatedTrees data, Pooled pooled, double width) {
        double[] grid = new double[7];
        for (int i = 0; i < grid.length; i++) {
            grid[i] = data.shape == 1 ? i + 1 : Math.round(data.shape * (1 + (i - 3) * width));
        }
        return curve("shape", grid, data.shape == 1 ? 0 : 3,
                x -> data.model(1.0, (int) Math.round(x), 1.0, 1.0, data.transition), pooled);
    }


    private static void report(SimulatedTrees data, String variant, Curve curve) {
        System.out.printf(Locale.ROOT, "%s %s %s: truth %.4g, maximum at %.4g (%.2f standard errors of %.3g), drop at the truth %.2f%n",
                data.config, variant, curve.name, curve.x[curve.truth], curve.vertex(),
                (curve.vertex() - curve.x[curve.truth]) / curve.standardError(), curve.standardError(), curve.drop());
    }

    /** All curves of a configuration and variant in one file, with the parameter as a column. */
    private static void writeCsv(SimulatedTrees data, String variant, List<Curve> curves) {
        File directory = new File("profiling/results/likelihood_profiles");
        File file = new File(directory, "profile_" + data.config + "_" + variant + ".csv");
        if (file.exists() && !Boolean.getBoolean("profiling.overwrite")) {
            System.out.println(file + " exists, not overwritten (-Dprofiling.overwrite=true)");
            return;
        }
        directory.mkdirs();
        try (PrintWriter out = new PrintWriter(file)) {
            out.println("parameter,value,logL,truth");
            for (Curve curve : curves) {
                for (int i = 0; i < curve.x.length; i++) {
                    out.println(curve.name + "," + curve.x[i] + "," + curve.logL[i] + "," + (i == curve.truth));
                }
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }


    private static void check(SimulatedTrees data, String variant, Curve curve) {
        String where = data.config + " " + variant + " " + curve.name + ": ";
        assertTrue(curve.drop() <= MAX_DROP, where + "the log likelihood at the truth is " + curve.drop() + " below the maximum");
        int m = curve.argmax();
        if (!curve.name.equals("shape")) {
            assertTrue(m > 0 && m < curve.x.length - 1, where + "the maximum is at the edge of the grid");
        }
    }


    /** Profiles of all parameters; the step of the grids is the given multiple of the steps used for the annotated trees. */
    private static void profiles(SimulatedTrees data, String variant, Pooled pooled, double stepFactor) {
        List<Curve> curves = new ArrayList<>();
        // the standard error of the lifetime shrinks with the shape like the spread of the lifetimes, 1 / sqrt(shape)
        curves.add(curve("lifetime", scaled(1.0, 0.0125 * stepFactor / Math.sqrt(data.shape)), TRUTH,
                x -> data.model(x, data.shape, 1.0, 1.0, data.transition), pooled));
        curves.add(curve("death", scaled(1.0, 0.03 * stepFactor), TRUTH,
                x -> data.model(1.0, data.shape, x, 1.0, data.transition), pooled));
        curves.add(curve("sampling", scaled(1.0, 0.02 * stepFactor), TRUTH,
                x -> data.model(1.0, data.shape, 1.0, x, data.transition), pooled));
        curves.add(shapeCurve(data, pooled, 0.1 * Math.sqrt(stepFactor)));
        if (data.nTypes == 2) {
            curves.add(curve("transition", shifted(data.transition, 0.01 * stepFactor), TRUTH,
                    x -> data.model(1.0, data.shape, 1.0, 1.0, x), pooled));
        }

        writeCsv(data, variant, curves);
        for (Curve curve : curves) report(data, variant, curve);
        for (Curve curve : curves) check(data, variant, curve);
    }

    private static void annotatedProfiles(String config) {
        SimulatedTrees data = new SimulatedTrees(config);
        profiles(data, "annotated", annotated(data, data.size(), 1 << 12), 1.0);
    }

    private static void integratedProfiles(String config, int nTrees, double stepFactor) {
        SimulatedTrees data = new SimulatedTrees(config);
        profiles(data, "integrated", integrated(data, nTrees, 1 << 10, false), stepFactor);
        if (data.nTypes == 1) {
            profiles(data, "approx", integrated(data, nTrees, 1 << 10, true), stepFactor);
        }
    }


    @Test
    public void testAnnotatedOneTypeShape1() { annotatedProfiles("adb_1type_shape1"); }

    @Test
    public void testAnnotatedOneTypeShape10() { annotatedProfiles("adb_1type_shape10"); }

    @Test
    public void testAnnotatedOneTypeShape100() { annotatedProfiles("adb_1type_shape100"); }

    @Test
    public void testAnnotatedTwoTypesShape1() { annotatedProfiles("adb_2type_shape1"); }

    @Test
    public void testAnnotatedTwoTypesShape10() { annotatedProfiles("adb_2type_shape10"); }

    @Test
    public void testAnnotatedTwoTypesShape100() { annotatedProfiles("adb_2type_shape100"); }

    @Test
    public void testCollapsedTreesAreTheAnnotatedTreesWithoutHiddenDivisions() {
        SimulatedTrees data = new SimulatedTrees("adb_2type_shape10");
        for (int k = 0; k < data.size(); k++) {
            AnnotatedTree annotated = data.annotated(k);
            Tree collapsed = data.collapsed(k);
            String where = "tree " + k;
            assertEquals(annotated.getLeafNodeCount(), collapsed.getLeafNodeCount(), where);
            assertEquals(annotated.getRoot().getHeight(), collapsed.getRoot().getHeight(), 1e-6, where);
            // the heights of the divisions with sampled descendants on both sides, and the tip types
            double[] heights = new double[annotated.getInternalNodeCount()];
            double[] collapsedHeights = new double[heights.length];
            int i = 0;
            for (Node node : annotated.getInternalNodes()) heights[i++] = node.getHeight();
            i = 0;
            for (Node node : collapsed.getInternalNodes()) collapsedHeights[i++] = node.getHeight();
            Arrays.sort(heights);
            Arrays.sort(collapsedHeights);
            assertArrayEquals(heights, collapsedHeights, 1e-6, where);
        }
    }

    @Test
    public void testIntegratedOneTypeShape1() { integratedProfiles("adb_1type_shape1", 15, 3.0); }

    @Test
    public void testIntegratedOneTypeShape10() { integratedProfiles("adb_1type_shape10", 15, 3.0); }

    @Test
    public void testIntegratedOneTypeShape100() { integratedProfiles("adb_1type_shape100", 15, 3.0); }

    @Test
    public void testIntegratedTwoTypesShape1() { integratedProfiles("adb_2type_shape1", 15, 3.0); }

    @Test
    public void testIntegratedTwoTypesShape10() { integratedProfiles("adb_2type_shape10", 15, 3.0); }

    @Test
    public void testIntegratedTwoTypesShape100() { integratedProfiles("adb_2type_shape100", 15, 3.0); }

}
