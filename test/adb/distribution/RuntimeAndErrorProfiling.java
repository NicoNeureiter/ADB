package adb.distribution;

import adb.tree.AnnotatedTreeParser;
import adb.util.Utils;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeInterface;
import beast.base.evolution.tree.TreeParser;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.function.Supplier;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;


/**
 * Run times of the likelihood calculations of ADBTreeDistribution for different numbers of steps, tree sizes, numbers
 * of types and shapes, on trees simulated by scripts/simulate_profiling_trees.R (profiling/trees). Writes
 * profiling/results/runtime_and_error/runtime_<experiment>.csv, one row per tree, mode, component and number of steps.
 *
 * The grid is always the whole interval from 0 to the origin time of the tree. The components (seconds, median of
 * repeated runs after a warm-up) are:
 *   lifetimes   the gamma distributions of the lifetimes on the grid (all modes)
 *   P0          solving P0 on the grid (all modes)
 *   P1          solving P1 on the grid (integrated and approx only; the approx path constructs it but does not use it)
 *   tree        calculateTreeLogLikelihood with P0 and P1 solved before, which includes the interpolation of both
 *   full        lifetimes + P0 + (P1) + tree: what an MCMC step pays if a model parameter changed
 * The modes are integrated (hidden divisions integrated out, Picard iteration and FFT), approx (series
 * approximation of the branch density, one type, integer shape) and annotated (the tree with its hidden divisions).
 * The full time of the annotated mode has no P1: lifetimes + P0 + tree.
 *
 * Time budget: a measurement whose predicted time (ProfilingSeries, from the previous cells of the series) exceeds the
 * budget of -Dprofiling.budget seconds (default 60) is not run; its value is the prediction and extrapolated is true.
 *
 * -Dprofiling.configs=tips10,tips100 restricts the experiments to these configurations of the trees files.
 *
 * An experiment whose output file exists is skipped before anything is measured, so that results that took long are
 * not lost: delete the file or run with -Dprofiling.overwrite=true to repeat it.
 *
 * Tagged profiling: not part of the normal runs, run with --include-tag profiling or --select-method.
 */
@Tag("profiling")
public class RuntimeAndErrorProfiling {

    private static final double BUDGET = Double.parseDouble(System.getProperty("profiling.budget", "60"));
    private static final java.util.Set<String> CONFIGS = java.util.Set.of(
            System.getProperty("profiling.configs", "").split(",")).stream().filter(c -> !c.isEmpty())
            .collect(java.util.stream.Collectors.toSet());
    private static final File TREES = new File("profiling/trees");
    private static final File RESULTS = new File("profiling/results/runtime_and_error");
    private static final String TREES_HEADER = "config\ttree\tntypes\tgraph\tshape\tlifetime\tdeath\trho\torigin_time"
            + "\ttips\tnewick_annotated\tnewick_collapsed";
    private static final boolean OVERWRITE = Boolean.getBoolean("profiling.overwrite");
    private static final String HEADER =
            "experiment,config,tree,mode,component,types,graph,shape,rho,origin,tips,nSteps,seconds,extrapolated,logL";

    private static final int[] STEPS_ALL = {1 << 10, 1 << 12, 1 << 14};


    /** Skips the test if the output file exists, unless overwriting is allowed. */
    static File protectedOutput(File file) {
        Assumptions.assumeTrue(OVERWRITE || !file.exists(),
                file + " exists, delete it or run with -Dprofiling.overwrite=true to repeat the experiment");
        return file;
    }

    /** A row of a trees file: the model parameters and the tree, annotated and with the hidden divisions collapsed. */
    record Scenario(String config, int tree, int nTypes, String graph, int shape, double lifetime, double death,
                    double rho, double origin, int tips, String annotated, String collapsed) {

        /** The model of scripts/simulate_profiling_trees.R: equal types, 'chain' or 'dense' transition graph. */
        Parameterization model() {
            double[] lifetimes = new double[nTypes], deaths = new double[nTypes], sampling = new double[nTypes];
            int[] shapes = new int[nTypes];
            double[][] sym = new double[nTypes][nTypes], asym = new double[nTypes][nTypes];
            for (int i = 0; i < nTypes; i++) {
                lifetimes[i] = lifetime;
                deaths[i] = death;
                sampling[i] = rho;
                shapes[i] = shape;
                sym[i][i] = 1.0;
                if (nTypes > 1 && graph.equals("chain") && i < nTypes - 1) {
                    asym[i][i + 1] = 0.2;
                    sym[i][i] = 0.8;
                } else if (nTypes > 1 && graph.equals("dense")) {
                    Arrays.fill(asym[i], 0.2 / (nTypes - 1));
                    asym[i][i] = 0.0;
                    sym[i][i] = 0.8;
                }
            }
            return ReferenceModels.model(lifetimes, shapes, deaths, sampling, sym, asym, origin);
        }

        Tree collapsedTree() {
            return new TreeParser(collapsed, true);
        }

        TreeInterface annotatedTree() {
            AnnotatedTreeParser parser = new AnnotatedTreeParser();
            parser.initByName("newick", annotated);
            return parser;
        }
    }


    static List<Scenario> read(String experiment) {
        File file = new File(TREES, "trees_" + experiment + ".tsv");
        Assumptions.assumeTrue(file.exists(), file + " is missing, run scripts/simulate_profiling_trees.R");
        List<Scenario> scenarios = new ArrayList<>();
        try {
            List<String> lines = Files.readAllLines(file.toPath());
            assertEquals(TREES_HEADER, lines.get(0), file + " does not have the columns of scripts/simulate_profiling_trees.R");
            for (String line : lines.subList(1, lines.size())) {
                String[] v = line.split("\t");
                if (!CONFIGS.isEmpty() && !CONFIGS.contains(v[0])) continue;
                scenarios.add(new Scenario(v[0], Integer.parseInt(v[1]), Integer.parseInt(v[2]), v[3],
                        Integer.parseInt(v[4]), Double.parseDouble(v[5]), Double.parseDouble(v[6]),
                        Double.parseDouble(v[7]), Double.parseDouble(v[8]), Integer.parseInt(v[9]), v[10], v[11]));
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return scenarios;
    }


    // ------------------------------------------------------------------ timing

    /** Seconds of one call: the first call is the warm-up and the measurement if it is long, else the median of repeats. */
    static double time(Runnable run) {
        double first = once(run);
        if (first > 2.0) return first;
        once(run);
        int repeats = first < 0.05 ? 7 : first < 0.5 ? 5 : 3;
        double[] t = new double[repeats];
        for (int i = 0; i < repeats; i++) t[i] = once(run);
        Arrays.sort(t);
        return t[repeats / 2];
    }

    private static double once(Runnable run) {
        long start = System.nanoTime();
        run.run();
        return (System.nanoTime() - start) / 1e9;
    }

    /**
     * The measured run times of one quantity along one axis (number of steps, tips, types), used to predict the time of a
     * cell that is too expensive to measure. The prediction is a power law fitted by least squares in log-log space to the
     * last few measured points, so linear scaling in the number of steps or tips gives a slope of 1. With a single point the
     * time is taken as proportional to the axis value, and without a point there is no prediction (NaN).
     */
    static final class ProfilingSeries {

    private static final int POINTS_USED = 4;

    private final List<double[]> points = new ArrayList<>();


    void add(double x, double seconds) {
        if (x > 0 && seconds > 0) points.add(new double[]{x, seconds});
    }

    boolean isEmpty() {
        return points.isEmpty();
    }

    /** The predicted time at axis value x. */
    double predict(double x) {
        if (points.isEmpty()) return Double.NaN;
        List<double[]> used = points.subList(Math.max(0, points.size() - POINTS_USED), points.size());

        int n = used.size();
        double meanX = 0, meanY = 0;
        for (double[] p : used) {
            meanX += Math.log(p[0]) / n;
            meanY += Math.log(p[1]) / n;
        }
        double sxx = 0, sxy = 0;
        for (double[] p : used) {
            sxx += (Math.log(p[0]) - meanX) * (Math.log(p[0]) - meanX);
            sxy += (Math.log(p[0]) - meanX) * (Math.log(p[1]) - meanY);
        }

        if (sxx < 1e-12) { // all points at one axis value: proportional to x from their mean
            return Math.exp(meanY) * x / Math.exp(meanX);
        }
        double slope = sxy / sxx;
        return Math.exp(meanY + slope * (Math.log(x) - meanX));
    }
    }


    /** The CSV rows of one experiment and the series for the predictions of the cells that are over the budget. */
    static class Recorder {
        final String experiment;
        final File file;
        final List<String> rows = new ArrayList<>();
        final Map<String, ProfilingSeries> series = new HashMap<>();
        boolean extrapolate = true;

        Recorder(String experiment) {
            this.experiment = experiment;
            this.file = protectedOutput(new File(RESULTS, "runtime_" + experiment + ".csv"));
        }

        ProfilingSeries series(String key) {
            return series.computeIfAbsent(key, k -> new ProfilingSeries());
        }

        /** Whether the prediction for this key at x is over the budget. */
        boolean tooExpensive(String key, double x) {
            ProfilingSeries s = series(key);
            return extrapolate && !s.isEmpty() && s.predict(x) > BUDGET;
        }

        void write(Scenario s, String mode, String component, int nSteps, double seconds, boolean extrapolated,
                   double logL) {
            rows.add(String.format(Locale.ROOT, "%s,%s,%d,%s,%s,%d,%s,%d,%s,%.6f,%d,%d,%.6g,%b,%s", experiment, s.config(),
                    s.tree(), mode, component, s.nTypes(), s.graph(), s.shape(), s.rho(), s.origin(), s.tips(), nSteps,
                    seconds, extrapolated, Double.isNaN(logL) ? "NA" : String.format(Locale.ROOT, "%.10f", logL)));
        }

        void save() {
            RESULTS.mkdirs();
            try (PrintWriter out = new PrintWriter(file)) {
                out.println(HEADER);
                rows.forEach(out::println);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
            System.out.println("wrote " + file + " (" + rows.size() + " rows)");
        }
    }


    /** The modes of a scenario: approx only for one type. */
    static List<String> modes(Scenario s) {
        return s.nTypes() == 1 ? List.of("integrated", "approx", "annotated") : List.of("integrated", "annotated");
    }


    static ADBTreeDistribution distribution(Parameterization model, TreeInterface tree, boolean approx, int nSteps) {
        ADBTreeDistribution d = new ADBTreeDistribution();
        d.initByName("tree", tree, "parameterization", model, "approx", approx, "conditionOnOrigin", true,
                "nSteps", nSteps);
        return d;
    }


    /**
     * Measures (or predicts) all components of one scenario at one number of steps. setupKey / setupX is the series of the
     * components that do not depend on the tree (axis value setupX), treeKey / treeX the one of the tree calculation.
     */
    static void runCell(Recorder rec, Scenario s, int nSteps, String setupKey, double setupX, String treeKey, double treeX) {
        Parameterization model = s.model();
        double[] timeArray = Utils.linSpace(0, s.origin(), nSteps);
        double timeStep = s.origin() / nSteps;
        ADBTreeDistribution defaults = new ADBTreeDistribution();
        int maxIt = defaults.maxIterationsInput.get();
        double tol = defaults.toleranceInput.get();

        String[] setupNames = {"lifetimes", "P0", "P1"};
        double[] setup = new double[3];
        boolean setupPredicted = false;
        for (String name : setupNames) {
            if (rec.tooExpensive(setupKey + "|" + name, setupX)) setupPredicted = true;
        }

        if (setupPredicted) {
            for (int c = 0; c < 3; c++) setup[c] = rec.series(setupKey + "|" + setupNames[c]).predict(setupX);
        } else {
            Supplier<LifetimeDistributions> lifetimes = () -> new LifetimeDistributions(
                    model.getLifetimeParameter(), model.getShapeParameter(), timeArray);
            LifetimeDistributions ld = lifetimes.get();
            P0System p0 = new P0System(model, ld, maxIt, tol, timeArray, timeStep);
            setup[0] = time(lifetimes::get);
            setup[1] = time(() -> new P0System(model, ld, maxIt, tol, timeArray, timeStep));
            setup[2] = time(() -> new P1System(model, ld, p0, maxIt, tol, timeArray, timeStep));
            for (int c = 0; c < 3; c++) rec.series(setupKey + "|" + setupNames[c]).add(setupX, setup[c]);
        }
        rec.write(s, "all", "lifetimes", nSteps, setup[0], setupPredicted, Double.NaN);
        rec.write(s, "all", "P0", nSteps, setup[1], setupPredicted, Double.NaN);
        rec.write(s, "integrated", "P1", nSteps, setup[2], setupPredicted, Double.NaN);
        rec.write(s, "approx", "P1", nSteps, setup[2], setupPredicted, Double.NaN);

        for (String mode : modes(s)) {
            boolean annotated = mode.equals("annotated");
            String key = treeKey + "|" + mode;
            boolean treePredicted = setupPredicted || rec.tooExpensive(key, treeX);
            double tree;
            double logL = Double.NaN;
            // without a series there is nothing to predict from: the first cell of a series is always measured
            if (treePredicted && !rec.series(key).isEmpty()) {
                tree = rec.series(key).predict(treeX);
            } else {
                TreeInterface t = annotated ? s.annotatedTree() : s.collapsedTree();
                ADBTreeDistribution d = distribution(model, t, mode.equals("approx"), nSteps);
                logL = d.calculateTreeLogLikelihood(t);
                tree = time(() -> d.calculateTreeLogLikelihood(t));
                rec.series(key).add(treeX, tree);
                treePredicted = false;
            }
            double full = setup[0] + setup[1] + (annotated ? 0 : setup[2]) + tree;
            rec.write(s, mode, "tree", nSteps, tree, treePredicted, logL);
            rec.write(s, mode, "full", nSteps, full, treePredicted || setupPredicted, Double.NaN);
            System.out.printf(Locale.ROOT, "%s %s tree %d %s nSteps 2^%d: P0 %.3g s, P1 %.3g s, tree %.3g s%s%n",
                    rec.experiment, s.config(), s.tree(), mode, Integer.numberOfTrailingZeros(nSteps), setup[1],
                    setup[2], tree, treePredicted || setupPredicted ? " (predicted)" : "");
        }
    }


    /**
     * Saves the rows and checks them: times are positive, and the log likelihood of a measured annotated tree is finite.
     * The integrated log likelihood of a tree with some hundreds of tips is -Infinity (the product of the subtree
     * likelihoods underflows, bug 13 of the roadmap), so it is counted and not asserted.
     */
    private static void check(Recorder rec) {
        rec.save();
        assertTrue(!rec.rows.isEmpty(), "no scenarios, check -Dprofiling.configs");
        int underflow = 0;
        for (String row : rec.rows) {
            String[] v = row.split(",");
            assertTrue(Double.parseDouble(v[12]) > 0, row);
            if (v[13].equals("false") && v[4].equals("tree")) {
                if (v[3].equals("annotated")) assertTrue(Double.isFinite(Double.parseDouble(v[14])), row);
                else if (v[14].equals("-Infinity")) underflow++;
            }
        }
        System.out.println(rec.experiment + ": " + underflow + " integrated or approx evaluations with log likelihood -Infinity");
    }


    // ------------------------------------------------------------------ error

    private static final String ERROR_HEADER =
            "experiment,config,tree,mode,tips,types,rho,origin,nSteps,seconds,logL,reference,finestGrid,ratio,error";
    private static final int FIRST_GRID = 10, LAST_GRID_CHEAP = 20, LAST_GRID_INTEGRATED = 16;
    /** The grid (two steps of 2 after the first) at which the error is printed and must be below the one at the first. */
    private static final int CHECK_GRID = FIRST_GRID + 4;

    /**
     * The log likelihood of the tree for the numbers of steps 2^10, 2^12, 2^14, ... (a factor 4 apart), for every mode
     * (integrated, approx for one type, annotated), and its error against a reference. The error is of the first order
     * in the step, so with the two finest grids G - 2 and G the reference is the extrapolated value
     * (4 f(G) - f(G - 2)) / 3. The grids go up to 2^20 for the cheap modes (annotated, approx) and up to 2^16 for the
     * integrated mode, whose cost is linear in tips and steps, and stop when the next grid is predicted (4 times the
     * time of the last) to exceed the budget. The column ratio is (f(G - 2) - f(G - 4)) / (f(G) - f(G - 2)): about 4 if
     * the error is in the first-order regime, so the reference is reliable; well below 4, it is not (the grid is too
     * coarse for the tree, which happens for small sampling proportions): the error is then a lower bound, since the
     * reference itself is still too far from the limit. If the log likelihood is -Infinity at 2^10 (underflow, bug 13 of the
     * roadmap) or not finite, the other grids are not evaluated and reference, ratio and error are NA. Every grid is
     * evaluated once, and all of them are in the CSV.
     */
    static void runError(String experiment) {
        File file = protectedOutput(new File(RESULTS, "error_" + experiment + ".csv"));
        List<String> rows = new ArrayList<>();
        for (Scenario s : read(experiment)) {
            for (String mode : modes(s)) {
                boolean annotated = mode.equals("annotated");
                Parameterization model = s.model();
                TreeInterface tree = annotated ? s.annotatedTree() : s.collapsedTree();

                int lastGrid = mode.equals("integrated") ? LAST_GRID_INTEGRATED : LAST_GRID_CHEAP;
                double[] logL = new double[lastGrid + 1];
                double[] seconds = new double[lastGrid + 1];
                int finest = FIRST_GRID;
                for (int g = FIRST_GRID; g <= lastGrid; g += 2) {
                    if (g > FIRST_GRID && (seconds[g - 2] * 4 > BUDGET || !Double.isFinite(logL[FIRST_GRID]))) break;
                    long start = System.nanoTime();
                    ADBTreeDistribution d = distribution(model, tree, mode.equals("approx"), 1 << g);
                    logL[g] = d.calculateTreeLogLikelihood(tree);
                    seconds[g] = (System.nanoTime() - start) / 1e9;
                    finest = g;
                }

                boolean finite = Double.isFinite(logL[FIRST_GRID]) && finest >= FIRST_GRID + 4;
                double reference = finite ? (4 * logL[finest] - logL[finest - 2]) / 3 : Double.NaN;
                double ratio = finite ? (logL[finest - 2] - logL[finest - 4]) / (logL[finest] - logL[finest - 2]) : Double.NaN;
                for (int g = FIRST_GRID; g <= finest; g += 2) {
                    rows.add(String.format(Locale.ROOT, "%s,%s,%d,%s,%d,%d,%s,%.6f,%d,%.4g,%s,%s,%d,%s,%s", experiment,
                            s.config(), s.tree(), mode, s.tips(), s.nTypes(), s.rho(), s.origin(), 1 << g, seconds[g],
                            number(logL[g]), number(reference), finest, number(ratio), number(logL[g] - reference)));
                }
                System.out.printf(Locale.ROOT, "%s %s tree %d %s (%d tips): finest grid 2^%d, error at 2^%d %s, at 2^%d %s, ratio %s%n",
                        experiment, s.config(), s.tree(), mode, s.tips(), finest, FIRST_GRID, number(logL[FIRST_GRID] - reference), CHECK_GRID,
                        finest >= CHECK_GRID ? number(logL[CHECK_GRID] - reference) : "-", number(ratio));

                // the error shrinks with the grid: at 2^14 it is smaller than at 2^10, if it was noticeable at 2^10
                if (finite && Math.abs(logL[FIRST_GRID] - reference) > 1e-6) {
                    assertTrue(Math.abs(logL[CHECK_GRID] - reference) < Math.abs(logL[FIRST_GRID] - reference),
                            experiment + " " + s.config() + " tree " + s.tree() + " " + mode);
                }
            }
        }

        RESULTS.mkdirs();
        try (PrintWriter out = new PrintWriter(file)) {
            out.println(ERROR_HEADER);
            rows.forEach(out::println);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        System.out.println("wrote " + file + " (" + rows.size() + " rows)");
    }

    private static String number(double x) {
        return Double.isNaN(x) ? "NA" : Double.isInfinite(x) ? (x > 0 ? "Infinity" : "-Infinity") : String.format(Locale.ROOT, "%.10f", x);
    }


    // ------------------------------------------------------------------ experiments

    /** E1: number of steps, base model (one type, shape 10, 100 tips), 2^10 to 2^14. */
    @Test
    public void testRuntimeVsNSteps() {
        Recorder rec = new Recorder("nsteps");
        for (Scenario s : read("size")) {
            if (!s.config().equals("tips100")) continue;
            for (int nSteps = 1 << 10; nSteps <= 1 << 14; nSteps <<= 1) {
                String key = "nsteps|" + s.tree();
                runCell(rec, s, nSteps, key, nSteps, key, nSteps);
            }
        }
        check(rec);
    }


    /** E2: tree size, 10 to 3000 tips (base model), at 2^10, 2^12 and 2^14 steps. */
    @Test
    public void testRuntimeVsTreeSize() {
        Recorder rec = new Recorder("size");
        List<Scenario> scenarios = new ArrayList<>(read("size"));
        scenarios.sort(java.util.Comparator.comparingInt(Scenario::tips));
        for (int nSteps : STEPS_ALL) {
            for (Scenario s : scenarios) {
                runCell(rec, s, nSteps, "size|" + nSteps, s.origin(), "size|" + nSteps, s.tips());
            }
        }
        check(rec);
    }


    /** E3: number of types (1 to 10, chain and dense graph), 100 tips, 2^12 steps. */
    @Test
    public void testRuntimeVsTypes() {
        Recorder rec = new Recorder("ntypes");
        List<Scenario> scenarios = new ArrayList<>(read("ntypes"));
        scenarios.sort(java.util.Comparator.comparingInt(Scenario::nTypes));
        for (Scenario s : scenarios) {
            String key = "ntypes|" + s.graph();
            runCell(rec, s, 1 << 12, key, s.nTypes(), key, s.nTypes());
        }
        check(rec);
    }


    /** E4: shape 1 to 100, one type, 100 tips, 2^12 steps (no extrapolation: the cost is not monotone in the shape). */
    @Test
    public void testRuntimeVsShape() {
        Recorder rec = new Recorder("shape");
        rec.extrapolate = false;
        List<Scenario> scenarios = new ArrayList<>(read("shape"));
        scenarios.sort(java.util.Comparator.comparingInt(Scenario::shape));
        for (Scenario s : scenarios) {
            runCell(rec, s, 1 << 12, "shape", s.shape(), "shape", s.shape());
        }
        check(rec);
    }

    /** E5A: error against the sampling proportion 1, 0.1, 0.01, 0.001 at a fixed origin (the tree size follows rho). */
    @Test
    public void testErrorVsSamplingFixedOrigin() {
        runError("rho_origin");
    }


    /** E5B: error against the sampling proportion at a fixed number of tips (100; the origin follows rho). */
    @Test
    public void testErrorVsSamplingFixedTaxa() {
        runError("rho_taxa");
    }
}
