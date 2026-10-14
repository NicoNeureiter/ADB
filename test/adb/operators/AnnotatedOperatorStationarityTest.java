package adb.operators;

import adb.distribution.ADBTreeDistribution;
import adb.distribution.Parameterization;
import adb.tree.AnnotatedNode;
import adb.tree.AnnotatedNode.EventNode;
import adb.tree.AnnotatedTree;
import adb.tree.AnnotatedTreeParser;
import beast.base.evolution.tree.Node;
import beast.base.inference.Operator;
import beast.base.inference.parameter.IntegerParameter;
import beast.base.inference.parameter.RealParameter;
import beast.base.util.Randomizer;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.util.List;
import java.util.function.BiFunction;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Stationarity test of the Metropolis-Hastings kernel of single tree operators, see dev/detailed-balance-tests.md.
 * The trees in test_data are simulated from the ADB density (conditioned on the number of tips and the tip types).
 * If an operator has the correct Hastings ratio, a few MH steps with the ADB density as target leave their
 * distribution unchanged, so the paired differences of tree statistics before and after must have mean 0.
 * Generate the trees with scripts/simulate_test_trees.R.
 * Tagged slow (about 15 minutes): exclude with --exclude-tag slow.
 */
@Tag("slow")
public class AnnotatedOperatorStationarityTest {

    static final int REPLICATES = 2000;
    static final int STEPS = 10;
    static final double MAX_Z = 4.0;
    static final String[] STATISTICS = {"rootHeight", "treeLength", "hiddenEvents", "stemEvents", "typeChanges", "typeChangeHeights"};

    // the same parameters as in scripts/simulate_test_trees.R
    static Parameterization model(boolean twoTypes) {
        Parameterization model = new Parameterization();
        if (twoTypes) {
            model.initByName("nTypes", 2,
                    "lifetime", new RealParameter("1 1"),
                    "shape", new IntegerParameter("3 3"),
                    "death", new RealParameter("0.1 0.1"),
                    "symTransitions", new RealParameter("0.7 0 0 1"),
                    "asymTransitions", new RealParameter("0 0.3 0 0"),
                    "sampling", new RealParameter("0.5 0.5"),
                    "originTime", 3.0);
        } else {
            model.initByName("nTypes", 1,
                    "lifetime", new RealParameter("1"),
                    "shape", new IntegerParameter("3"),
                    "death", new RealParameter("0.1"),
                    "sampling", new RealParameter("0.5"),
                    "originTime", 3.0);
        }
        return model;
    }

    static AnnotatedTree readTree(String config, int index) {
        File file = new File("test_data/" + config + ".newick");
        Assumptions.assumeTrue(file.exists(), file + " is missing, run scripts/simulate_test_trees.R");
        AnnotatedTreeParser tree = new AnnotatedTreeParser();
        tree.initByName("fileName", file.getPath(), "treeIndex", index);
        return tree;
    }

    /** Values of STATISTICS for a tree. Hidden events include the type changes. */
    static double[] statistics(AnnotatedTree tree) {
        double length = 0, hidden = 0, changes = 0, changeHeights = 0;
        for (Node n : tree.getNodesAsArray()) {
            AnnotatedNode node = (AnnotatedNode) n;
            if (!node.isRoot()) length += node.getParent().getHeight() - node.getHeight();
            List<EventNode> events = node.getEvents();
            hidden += events.size() - 1;
            for (int e = 1; e < events.size(); e++) {
                if (events.get(e).getType() != events.get(e - 1).getType()) {
                    changes++;
                    changeHeights += events.get(e).getHeight();
                }
            }
        }
        AnnotatedNode root = (AnnotatedNode) tree.getRoot();
        return new double[]{root.getHeight(), length, hidden, root.getEventCount() - 1, changes, changeHeights};
    }

    /** Runs STEPS MH steps with the ADB density as target. The tree is changed in place. */
    static void runChain(AnnotatedTree tree, ADBTreeDistribution density, Operator operator) {
        double logP = density.calculateLogP();
        for (int step = 0; step < STEPS; step++) {
            AnnotatedTree backup = new AnnotatedTree();
            backup.assignFrom(tree);

            double logHR = operator.proposal();
            boolean accept = false;
            double newLogP = Double.NEGATIVE_INFINITY;
            if (logHR != Double.NEGATIVE_INFINITY) {
                newLogP = density.calculateLogP();
                accept = Math.log(Randomizer.nextDouble()) < newLogP - logP + logHR;
            }
            if (accept) {
                logP = newLogP;
            } else {
                tree.assignFrom(backup);
            }
        }
    }

    /** z-scores of the mean paired differences (after - before) per statistic. */
    static double[] stationarityZScores(String config, boolean twoTypes, BiFunction<AnnotatedTree, Parameterization, Operator> create) {
        Randomizer.setSeed(12345);
        int nStats = STATISTICS.length;
        double[] sum = new double[nStats], sumSquares = new double[nStats];
        for (int i = 1; i <= REPLICATES; i++) {
            AnnotatedTree tree = readTree(config, i);
            Parameterization model = model(twoTypes);
            ADBTreeDistribution density = new ADBTreeDistribution();
            density.initByName("tree", tree, "parameterization", model);
            Operator operator = create.apply(tree, model);

            double[] before = statistics(tree);
            runChain(tree, density, operator);
            double[] after = statistics(tree);
            for (int s = 0; s < nStats; s++) {
                double diff = after[s] - before[s];
                sum[s] += diff;
                sumSquares[s] += diff * diff;
            }
        }
        double[] z = new double[nStats];
        for (int s = 0; s < nStats; s++) {
            double mean = sum[s] / REPLICATES;
            double variance = (sumSquares[s] - REPLICATES * mean * mean) / (REPLICATES - 1);
            z[s] = variance > 0 ? mean / Math.sqrt(variance / REPLICATES) : 0.0;
            System.out.printf("%s %-18s mean difference %9.5f  z = %6.2f%n", config, STATISTICS[s], mean, z[s]);
        }
        return z;
    }

    static void assertStationary(double[] z) {
        for (int s = 0; s < z.length; s++) {
            assertTrue(Math.abs(z[s]) < MAX_Z, STATISTICS[s] + " drifts, z = " + z[s]);
        }
    }

    static boolean drifts(double[] z) {
        for (double zs : z) {
            if (Math.abs(zs) >= MAX_Z) return true;
        }
        return false;
    }

    static Operator eventSampler(AnnotatedTree tree, Parameterization model) {
        EventSampler operator = new EventSampler();
        operator.initByName("tree", tree, "parameterization", model, "drawEventCount", true,
                "proportionBranches", 0.5, "optimise", false, "weight", 1.0);
        return operator;
    }

    static Operator wilsonBalding(AnnotatedTree tree, Parameterization model) {
        AnnotatedWilsonBalding operator = new AnnotatedWilsonBalding();
        operator.initByName("tree", tree, "parameterization", model, "weight", 1.0);
        return operator;
    }

    static Operator exchange(AnnotatedTree tree, Parameterization model, boolean isNarrow) {
        AnnotatedExchange operator = new AnnotatedExchange();
        operator.initByName("tree", tree, "parameterization", model, "isNarrow", isNarrow, "weight", 1.0);
        return operator;
    }

    static Operator subtreeSlide(AnnotatedTree tree, Parameterization model) {
        AnnotatedSubtreeSlide operator = new AnnotatedSubtreeSlide();
        operator.initByName("tree", tree, "parameterization", model, "size", 1, "weight", 1.0);
        return operator;
    }

    /** The operator with a wrong Hastings ratio (log 2 too large), used to check that the test can fail. */
    static Operator biased(Operator operator) {
        return new Operator() {
            @Override
            public void initAndValidate() { }

            @Override
            public double proposal() {
                return operator.proposal() + Math.log(2);
            }
        };
    }


    @Test
    public void testSimulatedTreesAreValid() {
        for (String config : new String[]{"adb_1type_n4", "adb_2type_n4"}) {
            boolean twoTypes = config.contains("2type");
            Parameterization model = model(twoTypes);
            for (int i = 1; i <= REPLICATES; i++) {
                AnnotatedTree tree = readTree(config, i);
                tree.validate(model);
                assertEquals(4, tree.getLeafNodeCount());
            }
        }
    }

    @Test
    public void testEventSamplerOneType() {
        assertStationary(stationarityZScores("adb_1type_n4", false, AnnotatedOperatorStationarityTest::eventSampler));
    }

    @Test
    public void testControlBiasedEventSamplerFails() {
        double[] z = stationarityZScores("adb_1type_n4", false,
                (tree, model) -> biased(eventSampler(tree, model)));
        assertTrue(drifts(z), "a Hastings ratio that is off by log 2 must be detected");
    }

    @Test
    public void testWilsonBaldingOneType() {
        assertStationary(stationarityZScores("adb_1type_n4", false, AnnotatedOperatorStationarityTest::wilsonBalding));
    }

    @Test
    public void testNarrowExchangeOneType() {
        assertStationary(stationarityZScores("adb_1type_n4", false, (tree, model) -> exchange(tree, model, true)));
    }

    @Test
    public void testWideExchangeOneType() {
        assertStationary(stationarityZScores("adb_1type_n4", false, (tree, model) -> exchange(tree, model, false)));
    }

    @Test
    public void testSubtreeSlideOneType() {
        assertStationary(stationarityZScores("adb_1type_n4", false, AnnotatedOperatorStationarityTest::subtreeSlide));
    }

    @Test
    public void testEventSamplerTwoTypes() {
        assertStationary(stationarityZScores("adb_2type_n4", true, AnnotatedOperatorStationarityTest::eventSampler));
    }

    @Test
    public void testWilsonBaldingTwoTypes() {
        assertStationary(stationarityZScores("adb_2type_n4", true, AnnotatedOperatorStationarityTest::wilsonBalding));
    }

    @Test
    public void testNarrowExchangeTwoTypes() {
        assertStationary(stationarityZScores("adb_2type_n4", true, (tree, model) -> exchange(tree, model, true)));
    }

    @Test
    public void testWideExchangeTwoTypes() {
        assertStationary(stationarityZScores("adb_2type_n4", true, (tree, model) -> exchange(tree, model, false)));
    }

    @Test
    public void testSubtreeSlideTwoTypes() {
        assertStationary(stationarityZScores("adb_2type_n4", true, AnnotatedOperatorStationarityTest::subtreeSlide));
    }
}
