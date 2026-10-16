package adb.distribution;

import adb.tree.AnnotatedTree;
import adb.tree.AnnotatedTreeParser;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeParser;
import org.junit.jupiter.api.Assumptions;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


/**
 * Trees simulated from the model by scTreeSim, and the parameters they were simulated with. A configuration is a set
 * of files in test_data made by scripts/simulate_likelihood_test_trees.R: the annotated trees (hidden divisions
 * included) and the same trees with the hidden divisions collapsed, one Newick string per line, and a text file with
 * the parameters. There is no window on the number of tips, so the trees are draws from the model conditioned on at
 * least two sampled tips.
 *
 * A configuration has one or two types with the same shape and, for two types, different lifetime, death and sampling
 * probability; type 0 gives rise to type 1 with the probability transition (and divides into 0 and 0 otherwise), and
 * type 1 divides into 1 and 1. The model can be rebuilt with the lifetimes, death or sampling probabilities multiplied
 * by a factor, or with another shape or transition probability, for the profile likelihood.
 */
final class SimulatedTrees {

    private static final String DIRECTORY = "test_data/";

    final String config;
    final int nTypes;
    final double[] lifetime;
    final int shape;
    final double[] death;
    final double[] sampling;
    final double transition;
    final double originTime;
    private final List<String> newicks;
    private final List<String> collapsedNewicks;


    /** Skips the test if the files of the configuration are missing. */
    SimulatedTrees(String config) {
        this.config = config;
        File trees = new File(DIRECTORY + config + ".newick");
        File collapsed = new File(DIRECTORY + config + "_collapsed.newick");
        File parameters = new File(DIRECTORY + config + ".txt");
        Assumptions.assumeTrue(trees.exists() && collapsed.exists() && parameters.exists(),
                config + " is missing in test_data, run scripts/simulate_likelihood_test_trees.R");

        Map<String, String> values = new HashMap<>();
        try {
            newicks = Files.readAllLines(trees.toPath());
            collapsedNewicks = Files.readAllLines(collapsed.toPath());
            for (String line : Files.readAllLines(parameters.toPath())) {
                int colon = line.indexOf(':');
                values.put(line.substring(0, colon).replaceAll(" \\(.*\\)", "").trim(), line.substring(colon + 1).trim());
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        String[] asym = values.get("asym_trans_prob").split(",");
        nTypes = (int) Math.round(Math.sqrt(asym.length));
        lifetime = vector(values.get("mean_lifetime"), nTypes);
        shape = Integer.parseInt(values.get("shape"));
        death = vector(values.get("death_prob"), nTypes);
        sampling = vector(values.get("sampling_prob"), nTypes);
        transition = nTypes == 2 ? Double.parseDouble(asym[1]) : 0.0;
        originTime = Double.parseDouble(values.get("origin_time"));
    }

    /** Values per type, separated by commas; a single value is used for all types. */
    private static double[] vector(String commaSeparated, int nTypes) {
        double[] x = java.util.Arrays.stream(commaSeparated.split(",")).mapToDouble(Double::parseDouble).toArray();
        if (x.length == 1) {
            double[] all = new double[nTypes];
            java.util.Arrays.fill(all, x[0]);
            return all;
        }
        return x;
    }

    int size() {
        return newicks.size();
    }

    /** The model with the parameters of the simulation. */
    Parameterization truth() {
        return model(1.0, shape, 1.0, 1.0, transition);
    }

    /**
     * The model with the lifetimes, death and sampling probabilities of the simulation multiplied by the given factors,
     * the given shape (the same for all types) and transition probability, and the origin time of the simulation.
     */
    Parameterization model(double lifetimeFactor, int shape, double deathFactor, double samplingFactor, double transition) {
        int[] shapes = new int[nTypes];
        java.util.Arrays.fill(shapes, shape);
        double[][] sym = new double[nTypes][nTypes];
        double[][] asym = new double[nTypes][nTypes];
        if (nTypes == 1) {
            sym[0][0] = 1.0;
        } else {
            sym[0][0] = 1.0 - transition;
            sym[1][1] = 1.0;
            asym[0][1] = transition;
        }
        return ReferenceModels.model(scaled(lifetime, lifetimeFactor), shapes, scaled(death, deathFactor),
                scaled(sampling, samplingFactor), sym, asym, originTime);
    }

    private static double[] scaled(double[] x, double factor) {
        double[] y = x.clone();
        for (int i = 0; i < y.length; i++) y[i] *= factor;
        return y;
    }

    /** Annotated tree k (starting at 0). */
    AnnotatedTree annotated(int k) {
        AnnotatedTreeParser parser = new AnnotatedTreeParser();
        parser.initByName("newick", newicks.get(k));
        return parser;
    }

    /** Tree k with the hidden divisions collapsed (only the divisions with sampled descendants on both sides), with tip types. */
    Tree collapsed(int k) {
        return new TreeParser(collapsedNewicks.get(k), true);
    }
}
