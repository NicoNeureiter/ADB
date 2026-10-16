package adb.operators;

import adb.tree.AnnotatedNode;
import adb.tree.AnnotatedNode.EventNode;
import adb.tree.AnnotatedTree;
import beast.base.evolution.tree.Node;
import beast.base.inference.Operator;
import beast.base.util.Randomizer;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Stationarity test of the Metropolis-Hastings kernel of single tree operators.
 * The trees in test_data are exact draws from the ADB density (see ExactTreeSamples). If an operator has the correct
 * Hastings ratio, a few MH steps with the ADB density as target leave their distribution unchanged, so the paired
 * differences of tree statistics before and after must have mean 0. A report with the acceptance rate and the
 * z-score of every statistic is printed for each test. A control with a wrong Hastings ratio must fail.
 * Tagged slow: exclude with --exclude-tag slow.
 */
@Tag("slow")
public class StationarityTest {

    static final int STEPS = 10;
    static final double MAX_Z = 4.0;
    static final double MIN_ACCEPTANCE = 0.01;

    static final String ONE = "adb_1type_win";
    static final String TWO = "adb_2type_win";
    static final String THREE = "adb_3type_win";


    /** The names of the statistics of a tree, in the order of statistics(). */
    static List<String> statisticNames(int nTypes) {
        List<String> names = new ArrayList<>(List.of("rootHeight", "treeLength", "hiddenEvents", "stemEvents",
                "cherries", "smallerRootSubtree"));
        if (nTypes > 1) {
            names.addAll(List.of("typeChanges", "typeChangeHeights", "stemChanges", "pendantChanges",
                    "internalChanges", "changesBelowSmallerRoot", "rootChildrenMinType", "rootChildrenMaxType"));
            for (int from = 0; from < nTypes; from++) {
                for (int to = 0; to < nTypes; to++) {
                    if (from != to) names.add("changes " + from + ">" + to);
                }
            }
        }
        return names;
    }

    /** Values of statisticNames(). Hidden events include the type changes. */
    static double[] statistics(AnnotatedTree tree, int nTypes) {
        double length = 0, hidden = 0, cherries = 0;
        double changes = 0, changeHeights = 0, pendant = 0, internal = 0;
        double[][] kinds = new double[nTypes][nTypes];
        for (Node n : tree.getNodesAsArray()) {
            AnnotatedNode node = (AnnotatedNode) n;
            if (!node.isRoot()) {
                length += node.getParent().getHeight() - node.getHeight();
                if (!node.isLeaf() && node.getLeft().isLeaf() && node.getRight().isLeaf()) cherries++;
            }
            List<EventNode> events = node.getEvents();
            hidden += events.size() - 1;
            for (int e = 1; e < events.size(); e++) {
                int to = events.get(e - 1).getType(), from = events.get(e).getType();
                if (from != to) {
                    changes++;
                    changeHeights += events.get(e).getHeight();
                    kinds[from][to]++;
                    if (node.isLeaf()) pendant++;
                    else if (!node.isRoot()) internal++;
                }
            }
        }
        AnnotatedNode root = (AnnotatedNode) tree.getRoot();
        AnnotatedNode left = (AnnotatedNode) root.getLeft(), right = (AnnotatedNode) root.getRight();
        AnnotatedNode smaller = left.getLeafNodeCount() <= right.getLeafNodeCount() ? left : right;
        double stem = root.getEventCount() - 1;
        double stemChanges = typeChangesAbove(root);

        List<Double> values = new ArrayList<>(List.of(root.getHeight(), length, hidden, stem,
                cherries, (double) smaller.getLeafNodeCount()));
        if (nTypes > 1) {
            values.addAll(List.of(changes, changeHeights, stemChanges, pendant, internal, typeChangesBelow(smaller),
                    (double) Math.min(left.getType(), right.getType()), (double) Math.max(left.getType(), right.getType())));
            for (int from = 0; from < nTypes; from++) {
                for (int to = 0; to < nTypes; to++) {
                    if (from != to) values.add(kinds[from][to]);
                }
            }
        }
        return values.stream().mapToDouble(Double::doubleValue).toArray();
    }

    /** Number of type changes on the branch above the node. */
    static double typeChangesAbove(AnnotatedNode node) {
        double count = 0;
        List<EventNode> events = node.getEvents();
        for (int e = 1; e < events.size(); e++) {
            if (events.get(e).getType() != events.get(e - 1).getType()) count++;
        }
        return count;
    }

    /** Number of type changes on the branch above the node and in its subtree. */
    static double typeChangesBelow(AnnotatedNode node) {
        double count = typeChangesAbove(node);
        if (!node.isLeaf()) {
            count += typeChangesBelow((AnnotatedNode) node.getLeft()) + typeChangesBelow((AnnotatedNode) node.getRight());
        }
        return count;
    }


    /** The z-scores of the mean paired differences (after - before) of the statistics, and the acceptance rate. */
    record Result(List<String> names, double[] z, double acceptanceRate) { }

    /** Runs STEPS MH steps from every tree of the configuration and prints the report. */
    static Result stationarity(String config, String operatorName, Function<ExactTreeSamples, Operator> create) {
        Randomizer.setSeed(12345);
        ExactTreeSamples samples = new ExactTreeSamples(config);
        Operator operator = create.apply(samples);
        int nTypes = samples.model.getNTypes();
        List<String> names = statisticNames(nTypes);
        int nStats = names.size(), nTrees = samples.size();
        double[] sum = new double[nStats], sumSquares = new double[nStats];

        samples.resetCounts();
        for (int k = 0; k < nTrees; k++) {
            samples.load(k);
            double[] before = statistics(samples.tree, nTypes);
            for (int step = 0; step < STEPS; step++) samples.metropolisStep(operator);
            double[] after = statistics(samples.tree, nTypes);
            for (int s = 0; s < nStats; s++) {
                double diff = after[s] - before[s];
                sum[s] += diff;
                sumSquares[s] += diff * diff;
            }
        }

        System.out.printf("%nproposals: %d, valid %.1f%%%n", samples.proposals,
                100.0 * samples.validProposals / samples.proposals);
        return report(config, operatorName, names, sum, sumSquares, nTrees, STEPS, samples.accepted / samples.proposals);
    }

    /** Prints the z-scores of the mean paired differences, given their sums and sums of squares over the trees. */
    static Result report(String config, String operatorName, List<String> names, double[] sum, double[] sumSquares,
                         int nTrees, int steps, double acceptance) {
        int nStats = names.size();
        double[] z = new double[nStats];
        System.out.printf("== %s, %s: %d trees, %d MH steps each, accepted %.1f%% ==%n", operatorName, config, nTrees,
                steps, 100.0 * acceptance);
        System.out.printf("%-26s %12s %10s %8s%n", "statistic", "mean diff", "SE", "z");
        double largest = 0;
        String largestName = "";
        for (int s = 0; s < nStats; s++) {
            double mean = sum[s] / nTrees;
            double variance = (sumSquares[s] - nTrees * mean * mean) / (nTrees - 1);
            double se = Math.sqrt(Math.max(variance, 0) / nTrees);
            if (se > 1e-12) z[s] = mean / se;
            else z[s] = Math.abs(mean) > 1e-9 ? Math.signum(mean) * Double.POSITIVE_INFINITY : 0.0; // constant change
            System.out.printf("%-26s %12.5f %10.5f %8.2f%n", names.get(s), mean, se, z[s]);
            if (Math.abs(z[s]) > largest) {
                largest = Math.abs(z[s]);
                largestName = names.get(s);
            }
        }
        System.out.printf("largest |z| = %.2f (%s), limit %.1f -> %s%n", largest, largestName, MAX_Z,
                largest < MAX_Z && acceptance >= MIN_ACCEPTANCE ? "PASS" : "FAIL");
        return new Result(names, z, acceptance);
    }

    static void assertStationary(Result result) {
        assertTrue(result.acceptanceRate() >= MIN_ACCEPTANCE,
                "the operator is almost never accepted, acceptance rate " + result.acceptanceRate());
        for (int s = 0; s < result.z().length; s++) {
            assertTrue(Math.abs(result.z()[s]) < MAX_Z, result.names().get(s) + " drifts, z = " + result.z()[s]);
        }
    }

    static boolean drifts(Result result) {
        for (double z : result.z()) {
            if (Math.abs(z) >= MAX_Z) return true;
        }
        return false;
    }


    static Operator eventSampler(ExactTreeSamples samples) {
        EventSampler operator = new EventSampler();
        operator.initByName("tree", samples.tree, "parameterization", samples.model, "drawEventCount", true,
                "proportionBranches", 0.5, "optimise", false, "weight", 1.0);
        return operator;
    }

    static Operator wilsonBalding(ExactTreeSamples samples) {
        AnnotatedWilsonBalding operator = new AnnotatedWilsonBalding();
        operator.initByName("tree", samples.tree, "parameterization", samples.model, "weight", 1.0);
        return operator;
    }

    static Operator exchange(ExactTreeSamples samples, boolean isNarrow) {
        AnnotatedExchange operator = new AnnotatedExchange();
        operator.initByName("tree", samples.tree, "parameterization", samples.model, "isNarrow", isNarrow, "weight", 1.0);
        return operator;
    }

    static Operator subtreeSlide(ExactTreeSamples samples) {
        AnnotatedSubtreeSlide operator = new AnnotatedSubtreeSlide();
        operator.initByName("tree", samples.tree, "parameterization", samples.model, "size", 3, "weight", 1.0);
        return operator;
    }

    static Operator scale(ExactTreeSamples samples, boolean drawEvents) {
        AnnotatedScale operator = new AnnotatedScale();
        operator.initByName("tree", samples.tree, "parameterization", samples.model, "scaleFactor", 0.1,
                "drawEvents", drawEvents, "optimise", false, "weight", 1.0);
        return operator;
    }

    /** The operator with a wrong Hastings ratio (log 2 too large), used to check that the test can fail. */
    static Operator biased(Operator operator) {
        return biased(operator, 2.0);
    }

    /** The operator with a Hastings ratio that is too large by the given factor. */
    static Operator biased(Operator operator, double factor) {
        return new Operator() {
            @Override
            public void initAndValidate() { }

            @Override
            public double proposal() {
                return operator.proposal() + Math.log(factor);
            }
        };
    }


    @Test
    public void testControlBiasedEventSamplerFails() {
        Result result = stationarity(ONE, "EventSampler with Hastings ratio + log 2",
                samples -> biased(eventSampler(samples)));
        assertTrue(drifts(result), "a Hastings ratio that is off by log 2 must be detected");
    }

    @Test
    public void testEventSamplerOneType() {
        assertStationary(stationarity(ONE, "EventSampler", StationarityTest::eventSampler));
    }

    @Test
    public void testEventSamplerTwoTypes() {
        assertStationary(stationarity(TWO, "EventSampler", StationarityTest::eventSampler));
    }

    @Test
    public void testEventSamplerThreeTypes() {
        assertStationary(stationarity(THREE, "EventSampler", StationarityTest::eventSampler));
    }

    @Test
    public void testWilsonBaldingOneType() {
        assertStationary(stationarity(ONE, "Wilson-Balding", StationarityTest::wilsonBalding));
    }

    @Test
    public void testWilsonBaldingTwoTypes() {
        assertStationary(stationarity(TWO, "Wilson-Balding", StationarityTest::wilsonBalding));
    }

    @Test
    public void testWilsonBaldingThreeTypes() {
        assertStationary(stationarity(THREE, "Wilson-Balding", StationarityTest::wilsonBalding));
    }

    @Test
    public void testNarrowExchangeOneType() {
        assertStationary(stationarity(ONE, "narrow exchange", samples -> exchange(samples, true)));
    }

    @Test
    public void testNarrowExchangeTwoTypes() {
        assertStationary(stationarity(TWO, "narrow exchange", samples -> exchange(samples, true)));
    }

    @Test
    public void testNarrowExchangeThreeTypes() {
        assertStationary(stationarity(THREE, "narrow exchange", samples -> exchange(samples, true)));
    }

    @Test
    public void testWideExchangeOneType() {
        assertStationary(stationarity(ONE, "wide exchange", samples -> exchange(samples, false)));
    }

    @Test
    public void testWideExchangeTwoTypes() {
        assertStationary(stationarity(TWO, "wide exchange", samples -> exchange(samples, false)));
    }

    @Test
    public void testWideExchangeThreeTypes() {
        assertStationary(stationarity(THREE, "wide exchange", samples -> exchange(samples, false)));
    }

    @Test
    public void testSubtreeSlideOneType() {
        assertStationary(stationarity(ONE, "subtree slide", StationarityTest::subtreeSlide));
    }

    @Test
    public void testSubtreeSlideTwoTypes() {
        assertStationary(stationarity(TWO, "subtree slide", StationarityTest::subtreeSlide));
    }

    @Test
    public void testSubtreeSlideThreeTypes() {
        assertStationary(stationarity(THREE, "subtree slide", StationarityTest::subtreeSlide));
    }

    @Test
    public void testScaleOneType() {
        assertStationary(stationarity(ONE, "scale", samples -> scale(samples, false)));
    }

    @Test
    public void testScaleTwoTypes() {
        assertStationary(stationarity(TWO, "scale", samples -> scale(samples, false)));
    }

    @Test
    public void testScaleThreeTypes() {
        assertStationary(stationarity(THREE, "scale", samples -> scale(samples, false)));
    }

    @Test
    public void testScaleDrawEventsOneType() {
        assertStationary(stationarity(ONE, "scale, drawEvents", samples -> scale(samples, true)));
    }

    @Test
    public void testScaleDrawEventsTwoTypes() {
        assertStationary(stationarity(TWO, "scale, drawEvents", samples -> scale(samples, true)));
    }

    @Test
    public void testScaleDrawEventsThreeTypes() {
        assertStationary(stationarity(THREE, "scale, drawEvents", samples -> scale(samples, true)));
    }
}
