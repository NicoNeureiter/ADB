package adb.operators;

import beast.base.inference.Operator;
import beast.base.util.Randomizer;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;
import java.util.function.ToIntFunction;

import static adb.operators.StationarityTest.*;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Detailed-balance test of annotated tree operators.
 * The trees in test_data are exact draws from the ADB density (see ExactTreeSamples), so from these trees the flow
 * under the target distribution can be measured: the acceptance probability of proposals from a tree in class i to
 * a tree in class j. If the operator is reversible with respect to the ADB density, the flow from i to j equals the
 * flow from j to i for every pair of classes. For each pair, the difference of the two flows per tree must have
 * mean 0 (a paired z-test over the trees). The classes are defined by tree statistics, for example the number of
 * cherries. A report with the class sizes and the flows is printed for each test. A control with a wrong Hastings
 * ratio must fail.
 * Tagged slow: exclude with --exclude-tag slow.
 */
@Tag("slow")
public class FlowTest {

    static final int PROPOSALS = 50; // per tree
    static final double MAX_Z = 4.5;
    static final int MIN_TREES = 30; // trees with a flow in either direction for a pair to be checked
    static final int MIN_PAIRS = 3;  // checked pairs of classes in a test


    /** A partition of the trees into classes, given by a function of the statistics of a tree. */
    record Mapper(String name, List<String> labels, ToIntFunction<double[]> classOf, boolean ordinal) {
        Mapper(String name, List<String> labels, ToIntFunction<double[]> classOf) {
            this(name, labels, classOf, false);
        }
    }

    /** Classes 0, 1, ..., max (the last one is "max+" if the value can be larger). */
    static Mapper capped(String name, String statistic, int min, int max, List<String> names) {
        int index = names.indexOf(statistic);
        List<String> labels = new ArrayList<>();
        for (int v = min; v < max; v++) labels.add(String.valueOf(v));
        labels.add(max + "+");
        return new Mapper(name, labels, x -> Math.min((int) Math.round(x[index]), max) - min, true);
    }

    /** Three classes of about the same size, with the cut points taken from the trees. */
    static Mapper terciles(String name, String statistic, List<String> names, double[][] stats) {
        int index = names.indexOf(statistic);
        double[] sorted = Arrays.stream(stats).mapToDouble(x -> x[index]).sorted().toArray();
        double c1 = sorted[sorted.length / 3], c2 = sorted[2 * sorted.length / 3];
        List<String> labels = List.of(String.format("<%.3g", c1), String.format("%.3g to <%.3g", c1, c2),
                String.format(">=%.3g", c2));
        return new Mapper(name, labels, x -> x[index] < c1 ? 0 : x[index] < c2 ? 1 : 2, true);
    }

    static List<Mapper> mappers(int nTypes, List<String> names, double[][] stats) {
        List<Mapper> mappers = new ArrayList<>();
        mappers.add(capped("root split", "smallerRootSubtree", 1, 4, names));
        mappers.add(capped("cherries", "cherries", 1, 4, names));
        mappers.add(terciles("tree length", "treeLength", names, stats));
        mappers.add(terciles("hidden events", "hiddenEvents", names, stats));

        int cherries = names.indexOf("cherries"), length = names.indexOf("treeLength");
        Mapper lengthClasses = terciles("tree length", "treeLength", names, stats);
        List<String> joint = new ArrayList<>();
        for (int c = 1; c <= 3; c++) {
            for (String l : lengthClasses.labels()) joint.add("cherries " + c + (c == 3 ? "+" : "") + ", length " + l);
        }
        mappers.add(new Mapper("cherries and length", joint,
                x -> (Math.min((int) Math.round(x[cherries]), 3) - 1) * 3 + lengthClasses.classOf().applyAsInt(x)));

        if (nTypes > 1) {
            mappers.add(capped("type changes", "typeChanges", 0, 2, names));

            int min = names.indexOf("rootChildrenMinType"), max = names.indexOf("rootChildrenMaxType");
            List<String> pairs = new ArrayList<>();
            for (int a = 0; a < nTypes; a++) {
                for (int b = 0; b < nTypes; b++) pairs.add(a + "|" + b); // a <= b occur
            }
            mappers.add(new Mapper("types of the root children", pairs,
                    x -> (int) Math.round(x[min]) * nTypes + (int) Math.round(x[max])));

            int pendant = names.indexOf("pendantChanges"), internal = names.indexOf("internalChanges");
            mappers.add(new Mapper("type changes by branch kind", List.of("none", "pendant", "internal", "both"),
                    x -> (x[pendant] > 0 ? 1 : 0) + (x[internal] > 0 ? 2 : 0)));
        }
        if (nTypes > 2) {
            List<Integer> kinds = new ArrayList<>();
            List<String> kindNames = new ArrayList<>();
            for (int from = 0; from < nTypes; from++) {
                for (int to = from + 1; to < nTypes; to++) {
                    kinds.add(names.indexOf("changes " + from + ">" + to));
                    kindNames.add(from + ">" + to);
                }
            }
            List<String> labels = new ArrayList<>();
            for (int mask = 0; mask < 1 << kinds.size(); mask++) {
                List<String> present = new ArrayList<>();
                for (int i = 0; i < kinds.size(); i++) if ((mask >> i & 1) == 1) present.add(kindNames.get(i));
                labels.add(present.isEmpty() ? "none" : String.join(" ", present));
            }
            mappers.add(new Mapper("kinds of type changes", labels, x -> {
                int mask = 0;
                for (int i = 0; i < kinds.size(); i++) if (x[kinds.get(i)] > 0) mask |= 1 << i;
                return mask;
            }));
        }
        return mappers;
    }


    /** The outcome of a test: the number of checked pairs, the largest |z| and the acceptance rate. */
    record Result(int pairsChecked, double largestZ, String largestPair, double acceptanceRate) { }

    /** Measures the flows from every tree of the configuration, prints the report and checks every pair. */
    static Result flow(String config, String operatorName, Function<ExactTreeSamples, Operator> create) {
        Randomizer.setSeed(12345);
        ExactTreeSamples samples = new ExactTreeSamples(config);
        Operator operator = create.apply(samples);
        int nTypes = samples.model.getNTypes(), nTrees = samples.size();
        List<String> names = statisticNames(nTypes);

        double[][] stats = new double[nTrees][];
        for (int k = 0; k < nTrees; k++) {
            samples.load(k);
            stats[k] = statistics(samples.tree, nTypes);
        }
        List<Mapper> mappers = mappers(nTypes, names, stats);

        // flow[mapper][tree][from][to]: acceptance probability of the proposals of this tree
        double[][][][] flow = new double[mappers.size()][nTrees][][];
        for (int m = 0; m < mappers.size(); m++) {
            int nClasses = mappers.get(m).labels().size();
            for (int k = 0; k < nTrees; k++) flow[m][k] = new double[nClasses][nClasses];
        }

        samples.resetCounts();
        for (int k = 0; k < nTrees; k++) {
            samples.load(k);
            int[] from = new int[mappers.size()];
            for (int m = 0; m < mappers.size(); m++) from[m] = mappers.get(m).classOf().applyAsInt(stats[k]);

            for (int p = 0; p < PROPOSALS; p++) {
                double[][] proposed = new double[1][];
                double a = samples.acceptanceProbability(operator, tree -> proposed[0] = statistics(tree, nTypes));
                if (a == 0.0) continue;
                for (int m = 0; m < mappers.size(); m++) {
                    flow[m][k][from[m]][mappers.get(m).classOf().applyAsInt(proposed[0])] += a / PROPOSALS;
                }
            }
        }

        System.out.printf("%n== Flow, %s, %s: %d trees x %d proposals ==%n", operatorName, config, nTrees, PROPOSALS);
        System.out.printf("proposals: %d, valid %.1f%%, mean acceptance probability %.1f%%%n", samples.proposals,
                100.0 * samples.validProposals / samples.proposals, 100.0 * samples.accepted / samples.proposals);

        int pairsChecked = 0;
        double largest = 0;
        String largestPair = "";
        for (int m = 0; m < mappers.size(); m++) {
            Mapper mapper = mappers.get(m);
            int nClasses = mapper.labels().size();
            int[] sizes = new int[nClasses];
            for (int k = 0; k < nTrees; k++) sizes[mapper.classOf().applyAsInt(stats[k])]++;
            StringBuilder line = new StringBuilder();
            for (int c = 0; c < nClasses; c++) {
                if (sizes[c] > 0) line.append(" [").append(mapper.labels().get(c)).append("] ").append(sizes[c]);
            }
            System.out.printf("mapper \"%s\", class sizes:%s%n", mapper.name(), line);

            int skipped = 0;
            for (int i = 0; i < nClasses; i++) {
                for (int j = i + 1; j < nClasses; j++) {
                    int contributing = 0;
                    double sum = 0, sumSquares = 0, forward = 0, backward = 0;
                    for (int k = 0; k < nTrees; k++) {
                        double f = flow[m][k][i][j], b = flow[m][k][j][i];
                        if (f != 0.0 || b != 0.0) contributing++;
                        forward += f;
                        backward += b;
                        sum += f - b;
                        sumSquares += (f - b) * (f - b);
                    }
                    if (contributing == 0) continue;
                    if (contributing < MIN_TREES) {
                        skipped++;
                        continue;
                    }
                    double mean = sum / nTrees;
                    double se = Math.sqrt(Math.max(0, (sumSquares - nTrees * mean * mean) / (nTrees - 1)) / nTrees);
                    double z = se > 1e-12 ? mean / se : (Math.abs(mean) > 1e-9 ? Math.signum(mean) * Double.POSITIVE_INFINITY : 0.0);
                    String pair = mapper.labels().get(i) + " <-> " + mapper.labels().get(j);
                    System.out.printf("  %-44s trees %5d  flow %9.5f -> %9.5f  z = %6.2f%n", pair, contributing,
                            forward / nTrees, backward / nTrees, z);
                    pairsChecked++;
                    if (Math.abs(z) > largest) {
                        largest = Math.abs(z);
                        largestPair = mapper.name() + ": " + pair;
                    }
                }
            }
            if (skipped > 0) System.out.printf("  skipped %d pairs with fewer than %d trees%n", skipped, MIN_TREES);

            if (mapper.ordinal()) { // all pairs together: net flow towards higher classes
                double sum = 0, sumSquares = 0;
                boolean moves = false;
                for (int k = 0; k < nTrees; k++) {
                    double net = 0;
                    for (int i = 0; i < nClasses; i++) {
                        for (int j = i + 1; j < nClasses; j++) {
                            net += flow[m][k][i][j] - flow[m][k][j][i];
                            if (flow[m][k][i][j] != 0.0 || flow[m][k][j][i] != 0.0) moves = true;
                        }
                    }
                    sum += net;
                    sumSquares += net * net;
                }
                if (moves) {
                    double mean = sum / nTrees;
                    double se = Math.sqrt(Math.max(0, (sumSquares - nTrees * mean * mean) / (nTrees - 1)) / nTrees);
                    double z = se > 1e-12 ? mean / se : (Math.abs(mean) > 1e-9 ? Math.signum(mean) * Double.POSITIVE_INFINITY : 0.0);
                    System.out.printf("  %-44s net flow upwards %9.5f  z = %6.2f%n", "all pairs together", mean, z);
                    pairsChecked++;
                    if (Math.abs(z) > largest) {
                        largest = Math.abs(z);
                        largestPair = mapper.name() + ": all pairs together";
                    }
                }
            }
        }
        double acceptance = (double) samples.accepted / samples.proposals;
        System.out.printf("checked %d pairs and pooled tests, largest |z| = %.2f (%s), limit %.1f -> %s%n", pairsChecked, largest,
                largestPair, MAX_Z, largest < MAX_Z && pairsChecked >= MIN_PAIRS && acceptance >= MIN_ACCEPTANCE ? "PASS" : "FAIL");
        return new Result(pairsChecked, largest, largestPair, acceptance);
    }

    static void assertBalanced(Result result) {
        assertTrue(result.acceptanceRate() >= MIN_ACCEPTANCE,
                "the operator is almost never accepted, acceptance " + result.acceptanceRate());
        assertTrue(result.pairsChecked() >= MIN_PAIRS, "only " + result.pairsChecked() + " pairs of classes checked");
        assertTrue(result.largestZ() < MAX_Z, "the flow is not symmetric: " + result.largestPair() + ", |z| = " + result.largestZ());
    }


    @Test
    public void testControlBiasedWilsonBaldingFails() {
        Result result = flow(ONE, "Wilson-Balding with Hastings ratio + log 3", samples -> biased(wilsonBalding(samples), 3.0));
        assertTrue(result.largestZ() >= MAX_Z, "a Hastings ratio that is off by log 3 must be detected");
    }

    @Test
    public void testEventSamplerOneType() {
        assertBalanced(flow(ONE, "EventSampler", StationarityTest::eventSampler));
    }

    @Test
    public void testEventSamplerTwoTypes() {
        assertBalanced(flow(TWO, "EventSampler", StationarityTest::eventSampler));
    }

    @Test
    public void testEventSamplerThreeTypes() {
        assertBalanced(flow(THREE, "EventSampler", StationarityTest::eventSampler));
    }

    @Test
    public void testWilsonBaldingOneType() {
        assertBalanced(flow(ONE, "Wilson-Balding", StationarityTest::wilsonBalding));
    }

    @Test
    public void testWilsonBaldingTwoTypes() {
        assertBalanced(flow(TWO, "Wilson-Balding", StationarityTest::wilsonBalding));
    }

    @Test
    public void testWilsonBaldingThreeTypes() {
        assertBalanced(flow(THREE, "Wilson-Balding", StationarityTest::wilsonBalding));
    }

    @Test
    public void testNarrowExchangeOneType() {
        assertBalanced(flow(ONE, "narrow exchange", samples -> exchange(samples, true)));
    }

    @Test
    public void testNarrowExchangeTwoTypes() {
        assertBalanced(flow(TWO, "narrow exchange", samples -> exchange(samples, true)));
    }

    @Test
    public void testNarrowExchangeThreeTypes() {
        assertBalanced(flow(THREE, "narrow exchange", samples -> exchange(samples, true)));
    }

    @Test
    public void testWideExchangeOneType() {
        assertBalanced(flow(ONE, "wide exchange", samples -> exchange(samples, false)));
    }

    @Test
    public void testWideExchangeTwoTypes() {
        assertBalanced(flow(TWO, "wide exchange", samples -> exchange(samples, false)));
    }

    @Test
    public void testWideExchangeThreeTypes() {
        assertBalanced(flow(THREE, "wide exchange", samples -> exchange(samples, false)));
    }

    @Test
    public void testSubtreeSlideOneType() {
        assertBalanced(flow(ONE, "subtree slide", StationarityTest::subtreeSlide));
    }

    @Test
    public void testSubtreeSlideTwoTypes() {
        assertBalanced(flow(TWO, "subtree slide", StationarityTest::subtreeSlide));
    }

    @Test
    public void testSubtreeSlideThreeTypes() {
        assertBalanced(flow(THREE, "subtree slide", StationarityTest::subtreeSlide));
    }
}
