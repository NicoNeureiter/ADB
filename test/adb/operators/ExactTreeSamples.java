package adb.operators;

import adb.distribution.ADBTreeDistribution;
import adb.distribution.Parameterization;
import adb.tree.AnnotatedTree;
import adb.tree.AnnotatedTreeParser;
import beast.base.inference.Operator;
import beast.base.inference.parameter.IntegerParameter;
import beast.base.inference.parameter.RealParameter;
import beast.base.util.Randomizer;
import org.junit.jupiter.api.Assumptions;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

/**
 * Exact draws from the ADB density, for the stationarity and flow tests of the tree operators.
 * A configuration is a pair of files in test_data made by scripts/simulate_operator_test_trees.R:
 * the trees, one Newick string per line, and a text file with the parameters of the model,
 * which are read from there so that the model always matches the trees.
 *
 * The model, the density and the tree that the operators work on are created once, and load(k) copies tree k into
 * that tree. Tree moves do not change any parameter, so P0 is solved only once, which is much faster than
 * building a new density for every tree.
 *
 * The step methods count the proposals, so that the tests can report how often an operator proposed something
 * valid and how often it was accepted.
 */
class ExactTreeSamples {

    static final String DIRECTORY = "test_data/";

    final String config;
    final Parameterization model;
    final List<String> newicks;
    /** The tree that the operators change. */
    final AnnotatedTree tree = new AnnotatedTree();
    final ADBTreeDistribution density = new ADBTreeDistribution();

    /** Log density of the tree in its current state. */
    private double logP;

    // counts since the last resetCounts()
    int proposals;
    int validProposals;
    /** Number of accepted Metropolis-Hastings steps, or the sum of the acceptance probabilities. */
    double accepted;


    /** Skips the test if the files of the configuration are missing. */
    ExactTreeSamples(String config) {
        this.config = config;
        File trees = new File(DIRECTORY + config + ".newick");
        File parameters = new File(DIRECTORY + config + ".txt");
        Assumptions.assumeTrue(trees.exists() && parameters.exists(),
                config + " is missing in test_data, run scripts/simulate_operator_test_trees.R");
        try {
            newicks = Files.readAllLines(trees.toPath());
            model = readModel(Files.readAllLines(parameters.toPath()));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        tree.assignFrom(read(0));
        density.initByName("tree", tree, "parameterization", model);
        logP = density.calculateLogP();
    }

    int size() {
        return newicks.size();
    }

    /** A new tree from line k (starting at 0). */
    AnnotatedTree read(int k) {
        AnnotatedTreeParser parser = new AnnotatedTreeParser();
        parser.initByName("newick", newicks.get(k));
        return parser;
    }

    /** Copies tree k into the tree of the operators. */
    void load(int k) {
        tree.assignFrom(read(k));
        logP = density.calculateLogP();
    }

    double logP() {
        return logP;
    }

    void resetCounts() {
        proposals = 0;
        validProposals = 0;
        accepted = 0;
    }


    /**
     * One Metropolis-Hastings step with the ADB density as target. The proposed tree is kept if it is accepted and
     * the old tree is restored otherwise.
     */
    boolean metropolisStep(Operator operator) {
        AnnotatedTree backup = new AnnotatedTree();
        backup.assignFrom(tree);

        double a = propose(operator);
        boolean accept = Math.log(Randomizer.nextDouble()) < Math.log(a);
        if (accept) {
            accepted++;
            logP = density.calculateLogP();
        } else {
            tree.assignFrom(backup);
        }
        return accept;
    }

    /**
     * The acceptance probability min(1, pi(y)/pi(x) * Hastings ratio) of one proposal from the current tree x, or 0
     * for an invalid proposal. The tree is restored afterwards.
     */
    double acceptanceProbability(Operator operator) {
        return acceptanceProbability(operator, proposed -> { });
    }

    /**
     * As above. Before the tree is restored, the observer sees the proposed tree y; it is not called for an invalid
     * proposal.
     */
    double acceptanceProbability(Operator operator, Consumer<AnnotatedTree> observer) {
        AnnotatedTree backup = new AnnotatedTree();
        backup.assignFrom(tree);

        int valid = validProposals;
        double a = propose(operator);
        accepted += a;
        if (validProposals > valid) observer.accept(tree);
        tree.assignFrom(backup);
        return a;
    }

    /** Makes a proposal and returns its acceptance probability. The tree is left in the proposed state. */
    private double propose(Operator operator) {
        proposals++;
        double logHR = operator.proposal();
        if (logHR == Double.NEGATIVE_INFINITY || Double.isNaN(logHR)) return 0.0;
        validProposals++;
        double newLogP = density.calculateLogP();
        return Math.min(1.0, Math.exp(newLogP - logP + logHR));
    }


    /** The model from the parameters in the text file of a configuration. */
    static Parameterization readModel(List<String> lines) {
        Map<String, String> values = new HashMap<>();
        for (String line : lines) {
            int colon = line.indexOf(':');
            String key = line.substring(0, colon).replaceAll(" \\(.*\\)", "").trim();
            values.put(key, line.substring(colon + 1).trim());
        }
        String sym = values.get("sym_trans_prob").replace(',', ' ');
        int nTypes = (int) Math.round(Math.sqrt(sym.split(" ").length));

        Parameterization model = new Parameterization();
        model.initByName("nTypes", nTypes,
                "lifetime", new RealParameter(repeat(values.get("mean_lifetime"), nTypes)),
                "shape", new IntegerParameter(repeat(values.get("shape"), nTypes)),
                "death", new RealParameter(repeat(values.get("death_prob"), nTypes)),
                "symTransitions", new RealParameter(sym),
                "asymTransitions", new RealParameter(values.get("asym_trans_prob").replace(',', ' ')),
                "sampling", new RealParameter(repeat(values.get("sampling_prob"), nTypes)),
                "originTime", Double.parseDouble(values.get("origin_time")));
        return model;
    }

    private static String repeat(String value, int n) {
        return String.join(" ", java.util.Collections.nCopies(n, value));
    }
}
