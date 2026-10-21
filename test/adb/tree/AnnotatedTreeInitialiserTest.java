package adb.tree;

import adb.distribution.ADBTreeDistribution;
import adb.distribution.Parameterization;
import adb.tree.AnnotatedNode.EventNode;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeParser;
import beast.base.inference.State;
import beast.base.inference.StateNode;
import beast.base.inference.parameter.IntegerParameter;
import beast.base.inference.parameter.RealParameter;
import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class AnnotatedTreeInitialiserTest {

    static final double ORIGIN = 8.0;

    // chain of types 0 -> 1 -> 2 -> 3 (asymmetric divisions), each type also self-renews symmetrically
    private Parameterization chainModel(int originType) {
        Parameterization model = new Parameterization();
        model.initByName("nTypes", 4,
                "lifetime", new RealParameter("0.5 0.5 0.5 0.5"),
                "shape", new IntegerParameter("5 5 5 5"),
                "death", new RealParameter("0.1 0.1 0.1 0.1"),
                "symTransitions", new RealParameter("0.5 0 0 0  0 0.5 0 0  0 0 0.5 0  0 0 0 1"),
                "asymTransitions", new RealParameter("0 0.5 0 0  0 0 0.5 0  0 0 0 0.5  0 0 0 0"),
                "sampling", new RealParameter("0.5 0.5 0.5 0.5"),
                "originTime", ORIGIN,
                "originType", originType);
        return model;
    }

    private AnnotatedTree initialise(String newick, Parameterization model) {
        AnnotatedTree tree = new AnnotatedTreeInitialiser();
        tree.initByName("branchingTree", new TreeParser(newick, true), "parameterization", model);
        return tree;
    }

    // types along the branch above a node, from the node upwards, without repetitions
    private List<Integer> typeChanges(AnnotatedNode node) {
        List<Integer> types = new ArrayList<>();
        for (EventNode event : node.getEvents()) {
            if (types.isEmpty() || types.get(types.size() - 1) != event.getType()) {
                types.add(event.getType());
            }
        }
        return types;
    }

    // the starting tree must be a valid history under the model
    private void assertValid(AnnotatedTree tree, Parameterization model) {
        assertDoesNotThrow(() -> tree.validate(model));
    }

    // two type changes on the same branch (this used to give wrong event indices)
    @Test
    public void testSeveralTypeChangesOnOneBranch() {
        Parameterization model = chainModel(0);
        AnnotatedTree tree = initialise("(1[&type=0]:5.0,2[&type=3]:5.0):1.0;", model);
        assertValid(tree, model);
        AnnotatedNode tip = (AnnotatedNode) tree.getNode(1);
        assertEquals(List.of(3, 2, 1), typeChanges(tip));
        assertTrue(tip.getEventCount() > 3, "hidden events are drawn in every segment");

        ADBTreeDistribution distribution = new ADBTreeDistribution();
        distribution.initByName("tree", tree, "parameterization", model);
        assertFalse(Double.isInfinite(distribution.calculateTreeLogLikelihood(tree)));
    }

    // the root type differs from the origin type: the stem needs a type change
    @Test
    public void testStemStartsWithOriginType() {
        Parameterization model = chainModel(0);
        AnnotatedTree tree = initialise("(1[&type=1]:5.0,2[&type=2]:5.0):1.0;", model);

        assertValid(tree, model);
        AnnotatedNode root = (AnnotatedNode) tree.getRoot();
        assertEquals(1, root.getType());
        assertEquals(List.of(1, 0), typeChanges(root));
    }

    // types that cannot be connected used to make the initialisation loop forever
    @Test
    public void testIncompatibleTypesFail() {
        Parameterization model = new Parameterization();
        model.initByName("nTypes", 2,
                "lifetime", new RealParameter("1 1"),
                "shape", new IntegerParameter("1 1"),
                "death", new RealParameter("0.1 0.1"),
                "symTransitions", new RealParameter("1 0 0 1"),
                "asymTransitions", new RealParameter("0 0 0 0"),
                "sampling", new RealParameter("0.5 0.5"),
                "originTime", ORIGIN);

        assertTimeoutPreemptively(Duration.ofSeconds(10), () ->
                assertThrows(RuntimeException.class, () ->
                        initialise("(1[&type=0]:5.0,2[&type=1]:5.0):1.0;", model)));
    }

    private Parameterization singleTypeModel(RealParameter lifetime) {
        Parameterization model = new Parameterization();
        model.initByName("lifetime", lifetime, "shape", new RealParameter("10"), "death", new RealParameter("0.1"),
                "sampling", new RealParameter("0.5"), "originTime", ORIGIN);
        return model;
    }

    private State state(StateNode... nodes) {
        State state = new State();
        state.initByName("stateNode", List.of(nodes));
        state.initialise();
        return state;
    }

    // a parameterization that shares parameters with the State breaks recalculation and restore of tree moves
    @Test
    public void testParameterizationInStateFails() {
        RealParameter lifetime = new RealParameter("1");
        AnnotatedTreeInitialiser tree = (AnnotatedTreeInitialiser) initialise("(1:5.0,2:5.0):1.0;", singleTypeModel(lifetime));
        state(lifetime, tree);
        assertThrows(IllegalArgumentException.class, tree::initStateNodes);
    }

    // its own parameterization, with parameters outside the State, is fine
    @Test
    public void testOwnParameterizationPasses() {
        RealParameter lifetime = new RealParameter("1");
        AnnotatedTreeInitialiser tree = (AnnotatedTreeInitialiser) initialise("(1:5.0,2:5.0):1.0;",
                singleTypeModel(new RealParameter("1")));
        state(lifetime, tree);
        assertDoesNotThrow(tree::initStateNodes);
    }
}
