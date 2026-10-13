package adb.operators;

import adb.distribution.Parameterization;
import adb.tree.AnnotatedNode;
import adb.tree.AnnotatedNode.EventNode;
import adb.tree.AnnotatedTreeParser;
import beast.base.evolution.tree.Node;
import beast.base.inference.parameter.IntegerParameter;
import beast.base.inference.parameter.RealParameter;
import beast.base.util.Randomizer;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Wilson-Balding, exchange and subtree slide on a two-type tree with type changes on several branches:
 * every finite proposal must leave a valid tree. An exchange keeps all type changes on their branches;
 * a slide can turn a type change into a division (and the other way round), so the count is not fixed there.
 */
public class AnnotatedTopologyOperatorsTest {

    static final double ORIGIN = 6.0;
    static final int SEEDS = 500;

    // no symmetric differentiation, so some moves create impossible divisions
    private Parameterization model() {
        Parameterization model = new Parameterization();
        model.initByName("nTypes", 2,
                "lifetime", new RealParameter("1 1"),
                "shape", new IntegerParameter("5 5"),
                "death", new RealParameter("0.1 0.1"),
                "symTransitions", new RealParameter("0.5 0 0 0.5"),
                "asymTransitions", new RealParameter("0 0.5 0.5 0"),
                "sampling", new RealParameter("0.5 0.5"),
                "originTime", ORIGIN);
        return model;
    }

    // caterpillar with a type change above the cherry a and above the node b, and hidden events on leaves 3 and 4
    private AnnotatedTreeParser tree() {
        AnnotatedTreeParser tree = new AnnotatedTreeParser();
        tree.initByName("newick", "((((1:1,2:1):2,3:3):1,4:4):1,5:5):1;");
        for (Node n : tree.getExternalNodes()) {
            AnnotatedNode node = (AnnotatedNode) n;
            switch (n.getID()) {
                case "1":
                    ((AnnotatedNode) n.getParent()).addEvent(new EventNode(1, 1.2), true);
                    ((AnnotatedNode) n.getParent().getParent()).addEvent(new EventNode(1, 3.5), true);
                    break;
                case "3":
                    node.addEvent(new EventNode(0, 1.5), true);
                    node.addEvent(new EventNode(0, 2.5), true);
                    break;
                case "4":
                    node.addEvent(new EventNode(0, 2.0), true);
                    node.addEvent(new EventNode(0, 3.0), true);
                    break;
            }
        }
        return tree;
    }

    private int typeChanges(AnnotatedTreeParser tree) {
        int count = 0;
        for (Node n : tree.getNodesAsArray()) {
            List<EventNode> events = ((AnnotatedNode) n).getEvents();
            for (int e = 1; e < events.size(); e++) {
                if (events.get(e).getType() != events.get(e - 1).getType()) count++;
            }
        }
        return count;
    }

    private void assertValid(AnnotatedTreeParser tree, int seed, boolean keepsTypeChanges) {
        Parameterization model = model();
        assertDoesNotThrow(() -> tree.validate(model), "seed " + seed);
        if (keepsTypeChanges) {
            assertEquals(2, typeChanges(tree), "type changes are kept, seed " + seed);
        }
    }

    private void runProposals(Function<AnnotatedTreeParser, AnnotatedTreeOperator> create, boolean keepsTypeChanges) {
        int finite = 0;
        for (int seed = 1; seed <= SEEDS; seed++) {
            Randomizer.setSeed(seed);
            AnnotatedTreeParser tree = tree();
            assertEquals(2, typeChanges(tree));
            double logHR = create.apply(tree).proposal();
            if (logHR == Double.NEGATIVE_INFINITY) continue;
            finite++;
            assertFalse(Double.isNaN(logHR), "seed " + seed);
            assertValid(tree, seed, keepsTypeChanges);
        }
        assertTrue(finite > SEEDS / 20, "enough valid proposals: " + finite);
    }

    private AnnotatedExchange exchange(AnnotatedTreeParser tree, boolean narrow) {
        AnnotatedExchange operator = new AnnotatedExchange();
        operator.initByName("tree", tree, "parameterization", model(), "isNarrow", narrow, "weight", 1.0);
        return operator;
    }

    @Test
    public void testScaleEventsAnchoredAtNode() {
        AnnotatedNode i = new AnnotatedNode();
        i.setHeight(2.0);
        List<EventNode> events = new ArrayList<>();
        events.add(new EventNode(0, 2.0));
        events.add(new EventNode(0, 2.5)); // hidden event, same type
        events.add(new EventNode(1, 3.0)); // type change
        events.add(new EventNode(1, 3.5)); // hidden event
        events.add(new EventNode(0, 3.8)); // type change
        i.setEvents(events);

        // branch [2, 4] shrinks to [2, 2.5]: all events stay above the node
        double logJacobian = AnnotatedTreeOperator.scaleEvents(i, 4.0, 2.5);

        double[] expected = {2.0, 2.125, 2.25, 2.375, 2.45};
        for (int e = 0; e < expected.length; e++) {
            assertEquals(expected[e], i.getEvent(e).getHeight(), 1e-12);
        }
        assertEquals(2 * Math.log(0.25), logJacobian, 1e-12, "two type changes, scale 0.25");

        // growing again: the reverse map has the inverse Jacobian
        assertEquals(-2 * Math.log(0.25), AnnotatedTreeOperator.scaleEvents(i, 2.5, 4.0), 1e-12);
        assertEquals(3.8, i.getEvent(4).getHeight(), 1e-12);
    }

    @Test
    public void testWilsonBalding() {
        runProposals(tree -> {
            AnnotatedWilsonBalding operator = new AnnotatedWilsonBalding();
            operator.initByName("tree", tree, "parameterization", model(), "weight", 1.0);
            return operator;
        }, false);
    }

    @Test
    public void testNarrowExchange() {
        runProposals(tree -> exchange(tree, true), true);
    }

    @Test
    public void testWideExchange() {
        runProposals(tree -> exchange(tree, false), true);
    }

    @Test
    public void testSubtreeSlide() {
        runProposals(tree -> {
            AnnotatedSubtreeSlide operator = new AnnotatedSubtreeSlide();
            operator.initByName("tree", tree, "parameterization", model(), "weight", 1.0, "size", 3);
            return operator;
        }, false);
    }
}
