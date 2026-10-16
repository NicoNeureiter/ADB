package adb.operators;

import adb.distribution.Parameterization;
import adb.tree.AnnotatedNode;
import adb.tree.AnnotatedTreeParser;
import beast.base.evolution.tree.Node;
import beast.base.inference.parameter.IntegerParameter;
import beast.base.inference.parameter.RealParameter;
import beast.base.util.Randomizer;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/** Operators that resample events or scale the tree, on a single-type tree. */
public class AnnotatedTreeOperatorTest {

    static final String NEWICK = "(((((((5:0.704651623):0.9815616586):0.9724532301):1.076834454):1.249319317,((((2:0.7965609946):1.052640805):0.9526174057):1.110375181,(((3:0.7945840806):1.02083498):1.106252391):0.9905229342):1.072625896):0.904531321,(((((4:0.7883689605):1.024284023):0.9996663794,((1:0.9184522562):0.9222398679):0.9716272388):0.9727550889):1.034266685):1.070010466):1.036790172):0.9334925396;";

    AnnotatedTreeParser tree;
    RealParameter lifetime, death, sampling;
    IntegerParameter shape;
    Parameterization model;

    @BeforeEach
    public void setUp() {
        Randomizer.setSeed(1);
        tree = new AnnotatedTreeParser();
        tree.initByName("newick", NEWICK);

        lifetime = new RealParameter("1.0");
        lifetime.setBounds(0.0, Double.POSITIVE_INFINITY);
        shape = new IntegerParameter("100");
        shape.setBounds(1, 1000);
        death = new RealParameter("0.1");
        death.setBounds(0.0, 1.0);
        sampling = new RealParameter("0.1");
        sampling.setBounds(0.0, 1.0);

        model = new Parameterization();
        model.initByName("nTypes", 1,
                "lifetime", lifetime,
                "shape", shape,
                "death", death,
                "sampling", sampling,
                "originTime", 7.859634);
    }

    private int totalEvents() {
        int n = 0;
        for (Node node : tree.getNodesAsArray()) {
            n += ((AnnotatedNode) node).getEventCount();
        }
        return n;
    }

    @Test
    public void testEventSampler() {
        int before = totalEvents();
        EventSampler operator = new EventSampler();
        operator.initByName("tree", tree, "parameterization", model, "drawEventCount", true, "proportionBranches", 1.0, "weight", 1.0);

        double logHR = operator.proposal();

        assertTrue(Double.isFinite(logHR));
        assertNotEquals(before, totalEvents(), "the number of hidden events is redrawn");
        assertDoesNotThrow(() -> tree.validate(model));
    }

    // lifetime up and sampling down by the same factor leaves their product unchanged
    @Test
    public void testUpDownEventSampler() {
        UpDownEventSampler operator = new UpDownEventSampler();
        operator.initByName("tree", tree, "parameterization", model, "up", lifetime, "down", sampling, "scaleFactor", 0.2, "weight", 1.0);

        double logHR = operator.proposal();

        assertTrue(Double.isFinite(logHR));
        assertNotEquals(1.0, lifetime.getValue(), 1e-12);
        assertEquals(0.1, lifetime.getValue() * sampling.getValue(), 1e-12);
        assertDoesNotThrow(() -> tree.validate(model));
    }

    // the Bactrian kernel is symmetric on the log scale, so the log Hastings ratio is log f for the scaled lifetime
    // plus, for every redrawn branch, the density of its old events under the old lifetime minus that of the new ones;
    // resampleEvents always sets a new event list, so the redrawn branches are the ones whose list was replaced
    @Test
    public void testUpDownEventSamplerHastingsRatio() {
        for (double proportion : new double[]{1.0, 0.5}) {
            UpDownEventSampler operator = new UpDownEventSampler();
            operator.initByName("tree", tree, "parameterization", model, "up", lifetime, "proportionBranches", proportion,
                    "optimise", false, "weight", 1.0);
            Node[] nodes = tree.getNodesAsArray();
            for (int proposal = 0; proposal < 20; proposal++) {
                double oldLifetime = lifetime.getValue();
                double[] oldDensities = new double[nodes.length];
                List<List<AnnotatedNode.EventNode>> oldEvents = new ArrayList<>();
                for (int i = 0; i < nodes.length; i++) {
                    oldDensities[i] = operator.getBranchProbability((AnnotatedNode) nodes[i], true);
                    oldEvents.add(((AnnotatedNode) nodes[i]).getEvents());
                }

                double logHR = operator.proposal();

                double expected = Math.log(lifetime.getValue() / oldLifetime);
                for (int i = 0; i < nodes.length; i++) {
                    if (oldEvents.get(i) != ((AnnotatedNode) nodes[i]).getEvents()) {
                        expected += oldDensities[i] - operator.getBranchProbability((AnnotatedNode) nodes[i], true);
                    }
                }
                assertEquals(expected, logHR, 1e-9, "proportion " + proportion + ", proposal " + proposal);
            }
        }
    }

    @Test
    public void testUpDownEventSamplerRejectsInvalidInputs() {
        UpDownEventSampler withoutParameters = new UpDownEventSampler();
        assertThrows(RuntimeException.class, () -> withoutParameters.initByName("tree", tree, "parameterization", model,
                "weight", 1.0), "no up or down parameter");
        for (double proportion : new double[]{0.0, 1.5}) {
            UpDownEventSampler operator = new UpDownEventSampler();
            assertThrows(RuntimeException.class, () -> operator.initByName("tree", tree, "parameterization", model,
                    "up", lifetime, "proportionBranches", proportion, "weight", 1.0), "proportion " + proportion);
        }
    }

    // the log scale factor is scaleFactor times a draw from the Bactrian kernel, which has mean 0 and variance 1;
    // the default scaleFactor is 0.1
    @Test
    public void testUpDownEventSamplerScaleDistribution() {
        UpDownEventSampler operator = new UpDownEventSampler();
        operator.initByName("tree", tree, "parameterization", model, "up", lifetime, "optimise", false, "weight", 1.0);
        int draws = 5000;
        double sum = 0, sumSquares = 0;
        for (int draw = 0; draw < draws; draw++) {
            double oldLifetime = lifetime.getValue();
            operator.proposal();
            double logF = Math.log(lifetime.getValue() / oldLifetime) / 0.1;
            sum += logF;
            sumSquares += logF * logF;
        }
        assertEquals(0.0, sum / draws, 0.05);
        assertEquals(1.0, sumSquares / draws, 0.05);
    }

    // the tree and the lifetime are scaled by f (the time-scale ridge), a rate by 1/f
    @Test
    public void testScale() {
        double rootHeight = tree.getRoot().getHeight();
        RealParameter rate = new RealParameter("1.0");
        rate.setBounds(0.0, Double.POSITIVE_INFINITY);
        AnnotatedScale operator = new AnnotatedScale();
        operator.initByName("tree", tree, "parameterization", model, "parameter", lifetime, "parameterInverse", rate,
                "scaleFactor", 0.1, "drawEvents", true, "weight", 1.0);

        double logHR = operator.proposal();

        assertTrue(Double.isFinite(logHR));
        double f = tree.getRoot().getHeight() / rootHeight;
        assertNotEquals(1.0, f, 1e-12);
        assertEquals(f, lifetime.getValue() / 1.0, 1e-9);
        assertEquals(1.0 / f, rate.getValue(), 1e-9);
        assertDoesNotThrow(() -> tree.validate(model));
    }

    private Parameterization noOriginModel() {
        Parameterization noOrigin = new Parameterization();
        noOrigin.initByName("nTypes", 1, "lifetime", lifetime, "shape", shape, "death", death, "sampling", sampling);
        return noOrigin;
    }

    // without origin and without redraws the move is deterministic given f, and the Bactrian kernel is symmetric on
    // the log scale: the log Hastings ratio is log f for every internal node and every event above a node
    // (without origin the root has no stem, so the event of the single-child node on top of the Newick root is removed)
    @Test
    public void testScaleHastingsRatioWithoutRedraws() {
        ((AnnotatedNode) tree.getRoot()).clearEvents();
        AnnotatedScale operator = new AnnotatedScale();
        operator.initByName("tree", tree, "parameterization", noOriginModel(), "drawEvents", false, "optimise", false,
                "weight", 1.0);
        int scaled = 0;
        for (Node node : tree.getNodesAsArray()) {
            if (!node.isLeaf()) scaled++;
            scaled += ((AnnotatedNode) node).getEventCount() - 1;
        }
        for (int proposal = 0; proposal < 20; proposal++) {
            double oldRootHeight = tree.getRoot().getHeight();
            double logHR = operator.proposal();
            assertEquals(scaled * Math.log(tree.getRoot().getHeight() / oldRootHeight), logHR, 1e-9, "proposal " + proposal);
        }
    }

    // the leaves are data: a scale move must not change them, so it requires them at height 0
    @Test
    public void testScaleRequiresLeavesAtZero() {
        AnnotatedTreeParser dated = new AnnotatedTreeParser();
        dated.initByName("newick", "((1:1,2:2):1,3:3);", "adjustTipHeights", false);
        AnnotatedScale operator = new AnnotatedScale();
        assertThrows(RuntimeException.class, () -> operator.initByName("tree", dated, "parameterization", model,
                "weight", 1.0));
    }

    // without an origin, the root event must stay in sync with the root height
    @Test
    public void testRootEventFollowsRootHeight() {
        EventSampler operator = new EventSampler();
        operator.initByName("tree", tree, "parameterization", noOriginModel(), "weight", 1.0);

        AnnotatedNode root = (AnnotatedNode) tree.getRoot();
        root.setHeight(root.getHeight() * 1.1);
        operator.resampleEvents(root, true);

        assertEquals(root.getHeight(), root.getEvent(0).getHeight(), 0.0);
    }

    // the density returned by resampleEvents is the density getBranchProbability evaluates for the same events,
    // which is what makes the Hastings ratio of the event moves correct
    @Test
    public void testDrawDensityMatchesEvaluatedDensity() {
        EventSampler operator = new EventSampler();
        operator.initByName("tree", tree, "parameterization", model, "weight", 1.0);

        for (boolean drawEventCount : new boolean[]{true, false}) {
            for (int draw = 0; draw < 20; draw++) {
                for (beast.base.evolution.tree.Node node : tree.getNodesAsArray()) {
                    double drawn = operator.resampleEvents((AnnotatedNode) node, drawEventCount);
                    double evaluated = operator.getBranchProbability((AnnotatedNode) node, drawEventCount);
                    assertEquals(drawn, evaluated, 1e-9, "node " + node.getNr());
                }
            }
        }
    }

    // with a fixed number of events n on an internal branch of length L, the positions follow a symmetric Dirichlet
    // distribution, so the j-th hidden event has mean height start + j * L / n
    @Test
    public void testHiddenEventPositions() {
        AnnotatedNode node = (AnnotatedNode) tree.getRoot().getLeft(); // an internal node
        node.clearEvents();
        int n = 5;
        double start = node.getHeight();
        double length = node.getParent().getHeight() - start;
        for (int j = 1; j < n; j++) {
            node.addEvent(new AnnotatedNode.EventNode(0, start + j * length / n), true);
        }
        EventSampler operator = new EventSampler();
        operator.initByName("tree", tree, "parameterization", model, "weight", 1.0);

        int draws = 20_000;
        double[] sum = new double[n], sumSquares = new double[n];
        for (int draw = 0; draw < draws; draw++) {
            operator.resampleEvents(node, false);
            assertEquals(n, node.getEventCount());
            for (int j = 1; j < n; j++) {
                double h = node.getEvent(j).getHeight();
                sum[j] += h;
                sumSquares[j] += h * h;
            }
        }
        for (int j = 1; j < n; j++) {
            double mean = sum[j] / draws;
            double standardError = Math.sqrt((sumSquares[j] / draws - mean * mean) / draws);
            assertEquals(start + j * length / n, mean, 4 * standardError, "event " + j);
        }
    }

    // on a long branch the number of hidden events follows the renewal rate: about length / lifetime (minus the final partial interval)
    @Test
    public void testHiddenEventCount() {
        AnnotatedNode node = (AnnotatedNode) tree.getRoot().getLeft();
        double length = node.getParent().getHeight() - node.getHeight();
        EventSampler operator = new EventSampler();
        operator.initByName("tree", tree, "parameterization", model, "weight", 1.0);

        int draws = 5_000;
        double sum = 0;
        for (int draw = 0; draw < draws; draw++) {
            operator.resampleEvents(node, true);
            sum += node.getEventCount() - 1; // hidden events
        }
        // the drawn count n includes the last interval, so there are n - 1 hidden events
        assertEquals(length / lifetime.getValue() - 1, sum / draws, 0.1);
    }
}
