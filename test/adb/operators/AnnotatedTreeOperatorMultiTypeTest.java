package adb.operators;

import adb.distribution.Parameterization;
import adb.tree.AnnotatedNode;
import adb.tree.AnnotatedNode.EventNode;
import adb.tree.AnnotatedTreeParser;
import beast.base.evolution.tree.Node;
import beast.base.inference.parameter.IntegerParameter;
import beast.base.inference.parameter.RealParameter;
import beast.base.util.Randomizer;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

import static org.junit.jupiter.api.Assertions.*;

/** Operators on a four-type tree: after every proposal the tree must still be a valid history under the model. */
public class AnnotatedTreeOperatorMultiTypeTest {

    static final String NEWICK = "((((1[&type=3]:0.5050647894)[&type=3]:2.629032343,(((3[&type=3]:0.02658926371)[&type=1]:1.167256789)[&type=1]:0.9582704094)[&type=1]:0.9819806705)[&type=1]:0.8807615117,((((2[&type=1]:0.3540064826)[&type=1]:0.9311262134)[&type=1]:1.069878825)[&type=1]:1.123534175)[&type=0]:0.536312948)[&type=0]:0.518395086,((((((((10[&type=0]:0.2977362855)[&type=0]:0.5473791622)[&type=0]:0.5281869737)[&type=0]:0.4965747618)[&type=0]:0.536755725)[&type=0]:0.4893282464,((((8[&type=2]:0.1301792254)[&type=1]:1.243179893)[&type=0]:0.4946663287)[&type=0]:0.4916443901)[&type=0]:0.5362913174)[&type=0]:0.5393012838)[&type=0]:0.5350052332,(((((5[&type=2]:0.173317248)[&type=2]:1.486935976)[&type=1]:0.8881744641)[&type=0]:0.4923866848,((((7[&type=1]:0.4430713129)[&type=1]:1.124596537)[&type=0]:0.4915929058)[&type=0]:0.5159186473)[&type=0]:0.4656349701)[&type=0]:0.4771876381,((((4[&type=3]:0.6597367559)[&type=1]:0.9506927264)[&type=1]:1.078905666,(((6[&type=1]:0.3101182183)[&type=1]:0.7872153939)[&type=1]:1.103073858)[&type=0]:0.4889276784)[&type=0]:0.4354935528,(((((9[&type=3]:0.1416000553)[&type=1]:0.9612053662,((12[&type=0]:0.0160933408)[&type=0]:0.4995183673)[&type=0]:0.5871937134)[&type=0]:0.5595680455,(((11[&type=1]:0.1244051404)[&type=0]:0.4221686028)[&type=0]:0.5657392679)[&type=0]:0.550060456)[&type=0]:0.4531051483)[&type=0]:0.4907869725)[&type=0]:0.5185631138)[&type=0]:0.3931733095)[&type=0]:0.4522656605)[&type=0]:0.5629860586):0.4667462698;";
    static final int SEEDS = 300;

    RealParameter lifetime;
    Parameterization model;

    @BeforeEach
    public void setUp() {
        lifetime = new RealParameter("0.5 1 2 3");
        lifetime.setBounds(0.0, Double.POSITIVE_INFINITY); // parameters made from strings have no upper bound set
        model = new Parameterization();
        model.initByName("nTypes", 4,
                "lifetime", lifetime,
                "shape", new IntegerParameter("100 50 20 5"),
                "death", new RealParameter("0.05 0.1 0.1 0.1"),
                "symTransitions", new RealParameter("0.4 0.1 0 0 0 0.2 0 0 0 0 1 0 0 0 0 1"),
                "asymTransitions", new RealParameter("0 0.5 0 0 0 0 0.5 0.3 0 0 0 0 0 0 0 0"),
                "sampling", new RealParameter("0.1 0.1 0.1 0.1"),
                "originTime", 5.0);
    }

    private AnnotatedTreeParser tree() {
        AnnotatedTreeParser tree = new AnnotatedTreeParser();
        tree.initByName("newick", NEWICK);
        return tree;
    }

    // the type at the node and the type after every type change, per node
    private List<List<Integer>> typeHistories(AnnotatedTreeParser tree) {
        List<List<Integer>> histories = new ArrayList<>();
        for (Node n : tree.getNodesAsArray()) {
            List<Integer> types = new ArrayList<>();
            for (EventNode event : ((AnnotatedNode) n).getEvents()) {
                if (types.isEmpty() || types.get(types.size() - 1) != event.getType()) types.add(event.getType());
            }
            histories.add(types);
        }
        return histories;
    }

    @Test
    public void testStartingTreeIsValid() {
        assertDoesNotThrow(() -> tree().validate(model));
    }

    // type changes stay where they are, the density is consistent also with several types per branch
    @Test
    public void testEventSampler() {
        for (boolean drawEventCount : new boolean[]{true, false}) {
            Randomizer.setSeed(1);
            AnnotatedTreeParser tree = tree();
            List<List<Integer>> before = typeHistories(tree);
            List<List<Double>> changeHeights = changeHeights(tree);

            EventSampler operator = new EventSampler();
            operator.initByName("tree", tree, "parameterization", model, "drawEventCount", drawEventCount, "proportionBranches", 1.0, "weight", 1.0);
            double logHR = operator.proposal();

            assertTrue(Double.isFinite(logHR));
            assertEquals(before, typeHistories(tree));
            assertEquals(changeHeights, changeHeights(tree));
            assertDoesNotThrow(() -> tree.validate(model));

            for (Node node : tree.getNodesAsArray()) {
                double drawn = operator.resampleEvents((AnnotatedNode) node, drawEventCount);
                assertEquals(drawn, operator.getBranchProbability((AnnotatedNode) node, drawEventCount), 1e-9, "node " + node.getNr());
            }
        }
    }

    // heights of the type changes per node
    private List<List<Double>> changeHeights(AnnotatedTreeParser tree) {
        List<List<Double>> heights = new ArrayList<>();
        for (Node n : tree.getNodesAsArray()) {
            List<EventNode> events = ((AnnotatedNode) n).getEvents();
            List<Double> h = new ArrayList<>();
            for (int e = 1; e < events.size(); e++) {
                if (events.get(e).getType() != events.get(e - 1).getType()) h.add(events.get(e).getHeight());
            }
            heights.add(h);
        }
        return heights;
    }

    @Test
    public void testUpDownEventSampler() {
        Randomizer.setSeed(1);
        AnnotatedTreeParser tree = tree();
        UpDownEventSampler operator = new UpDownEventSampler();
        operator.initByName("tree", tree, "parameterization", model, "up", lifetime, "scaleFactor", 0.8, "weight", 1.0);

        assertTrue(Double.isFinite(operator.proposal()));
        assertNotEquals(0.5, lifetime.getValue(0), 1e-12);
        assertDoesNotThrow(() -> tree.validate(model));
    }

    // elementWise scales the lifetime of one type j and redraws exactly the branches that carry type j (these are fixed
    // by the type changes, which the move keeps); the log Hastings ratio is log f plus the old minus new densities
    @Test
    public void testUpDownEventSamplerElementWise() {
        Randomizer.setSeed(1);
        AnnotatedTreeParser tree = tree();
        UpDownEventSampler operator = new UpDownEventSampler();
        operator.initByName("tree", tree, "parameterization", model, "up", lifetime, "elementWise", true,
                "scaleFactor", 0.3, "optimise", false, "weight", 1.0);
        Node[] nodes = tree.getNodesAsArray();
        boolean[] typeScaled = new boolean[lifetime.getDimension()];
        for (int proposal = 0; proposal < 100; proposal++) {
            Double[] oldLifetimes = lifetime.getValues();
            double[] oldDensities = new double[nodes.length];
            List<List<EventNode>> oldEvents = new ArrayList<>();
            for (int i = 0; i < nodes.length; i++) {
                oldDensities[i] = operator.getBranchProbability((AnnotatedNode) nodes[i], true);
                oldEvents.add(((AnnotatedNode) nodes[i]).getEvents());
            }

            double logHR = operator.proposal();
            if (logHR == Double.NEGATIVE_INFINITY) continue;

            int changed = -1;
            for (int j = 0; j < oldLifetimes.length; j++) {
                if (lifetime.getValue(j) != oldLifetimes[j].doubleValue()) {
                    assertEquals(-1, changed, "only one lifetime changes, proposal " + proposal);
                    changed = j;
                }
            }
            assertNotEquals(-1, changed, "proposal " + proposal);
            typeScaled[changed] = true;

            double expected = Math.log(lifetime.getValue(changed) / oldLifetimes[changed]);
            for (int i = 0; i < nodes.length; i++) {
                boolean carries = false;
                for (EventNode event : oldEvents.get(i)) {
                    carries |= event.getType() == changed;
                }
                boolean redrawn = oldEvents.get(i) != ((AnnotatedNode) nodes[i]).getEvents();
                assertEquals(carries, redrawn, "node " + nodes[i].getNr() + ", type " + changed + ", proposal " + proposal);
                if (redrawn) {
                    expected += oldDensities[i] - operator.getBranchProbability((AnnotatedNode) nodes[i], true);
                }
            }
            assertEquals(expected, logHR, 1e-9, "proposal " + proposal);
            assertDoesNotThrow(() -> tree.validate(model), "proposal " + proposal);
        }
        for (int j = 0; j < typeScaled.length; j++) {
            assertTrue(typeScaled[j], "type " + j + " was scaled");
        }
    }

    // ShapeEventScaler with c and then 1/c restores the events, for the segments of every type (many of which start
    // at a type change)
    @Test
    public void testShapeEventScalerInverse() {
        AnnotatedTreeParser tree = tree();
        ShapeEventScaler operator = AnnotatedTreeOperatorTest.shapeEventScaler(tree, model);
        for (int type = 0; type < 4; type++) {
            boolean moved = AnnotatedTreeOperatorTest.inverseRestoresEvents(operator, tree, type, 0.6);
            assertEquals(AnnotatedTreeOperatorTest.movableEvents(tree, model, type) > 0, moved, "type " + type);
        }
    }

    // ShapeEventScaler scales the shape of one type j and moves only the hidden events of type j: node heights, type
    // changes and the events of other types stay, and log HR = (1 - M/2) log f with M the hidden events of type j
    @Test
    public void testShapeEventScaler() {
        Randomizer.setSeed(1);
        AnnotatedTreeParser tree = tree();
        ShapeEventScaler operator = AnnotatedTreeOperatorTest.shapeEventScaler(tree, model);
        RealParameter shape = operator.shape;
        boolean[] typeScaled = new boolean[4];
        for (int proposal = 0; proposal < 100; proposal++) {
            Double[] oldShapes = shape.getValues();
            List<List<EventNode>> oldEvents = new ArrayList<>();
            for (Node node : tree.getNodesAsArray()) {
                List<EventNode> copy = new ArrayList<>();
                for (EventNode event : ((AnnotatedNode) node).getEvents()) copy.add(event.copy());
                oldEvents.add(copy);
            }

            double logHR = operator.proposal();
            if (logHR == Double.NEGATIVE_INFINITY) {
                for (int j = 0; j < 4; j++) shape.setValue(j, oldShapes[j]); // a failed proposal leaves the shape scaled
                continue;
            }

            int changed = -1;
            for (int j = 0; j < 4; j++) {
                if (shape.getValue(j) != oldShapes[j].doubleValue()) {
                    assertEquals(-1, changed, "only one shape changes, proposal " + proposal);
                    changed = j;
                }
            }
            assertNotEquals(-1, changed, "proposal " + proposal);
            typeScaled[changed] = true;

            int hidden = AnnotatedTreeOperatorTest.movableEvents(tree, model, changed);
            assertEquals((1 - hidden / 2.0) * Math.log(shape.getValue(changed) / oldShapes[changed]), logHR, 1e-9,
                    "proposal " + proposal);
            for (Node node : tree.getNodesAsArray()) {
                List<EventNode> before = oldEvents.get(node.getNr()), after = ((AnnotatedNode) node).getEvents();
                assertEquals(before.size(), after.size());
                for (int e = 0; e < before.size(); e++) {
                    assertEquals(before.get(e).getType(), after.get(e).getType());
                    boolean hiddenOfType = e > 0 && before.get(e).getType() == changed
                            && before.get(e - 1).getType() == changed
                            && !(e == 1 && node.isLeaf() && node.getHeight() == 0.0); // the age of a sampled cell stays
                    if (!hiddenOfType) {
                        assertEquals(before.get(e).getHeight(), after.get(e).getHeight(), 0.0,
                                "node " + node.getNr() + ", event " + e + ", proposal " + proposal);
                    }
                }
            }
            assertDoesNotThrow(() -> tree.validate(model), "proposal " + proposal);
        }
        for (int j = 0; j < 4; j++) {
            assertTrue(typeScaled[j], "type " + j + " was scaled");
        }
    }

    private void runProposals(Function<AnnotatedTreeParser, AnnotatedTreeOperator> create) {
        int finite = 0;
        for (int seed = 1; seed <= SEEDS; seed++) {
            Randomizer.setSeed(seed);
            AnnotatedTreeParser tree = tree();
            double logHR = create.apply(tree).proposal();
            if (logHR == Double.NEGATIVE_INFINITY) continue;
            finite++;
            assertFalse(Double.isNaN(logHR), "seed " + seed);
            assertDoesNotThrow(() -> tree.validate(model), "seed " + seed);
        }
        assertTrue(finite > SEEDS / 20, "enough valid proposals: " + finite);
    }

    @Test
    public void testWilsonBalding() {
        runProposals(tree -> {
            AnnotatedWilsonBalding operator = new AnnotatedWilsonBalding();
            operator.initByName("tree", tree, "parameterization", model, "weight", 1.0);
            return operator;
        });
    }

    @Test
    public void testNarrowExchange() {
        runProposals(tree -> exchange(tree, true));
    }

    @Test
    public void testWideExchange() {
        runProposals(tree -> exchange(tree, false));
    }

    @Test
    public void testSubtreeSlide() {
        runProposals(tree -> {
            AnnotatedSubtreeSlide operator = new AnnotatedSubtreeSlide();
            operator.initByName("tree", tree, "parameterization", model, "weight", 1.0, "size", 3);
            return operator;
        });
    }

    private AnnotatedExchange exchange(AnnotatedTreeParser tree, boolean narrow) {
        AnnotatedExchange operator = new AnnotatedExchange();
        operator.initByName("tree", tree, "parameterization", model, "isNarrow", narrow, "weight", 1.0);
        return operator;
    }
}
