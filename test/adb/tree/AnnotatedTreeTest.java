package adb.tree;

import adb.distribution.Parameterization;
import beast.base.evolution.alignment.Taxon;
import beast.base.evolution.alignment.TaxonSet;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.parameter.IntegerParameter;
import beast.base.inference.parameter.RealParameter;
import beast.base.evolution.tree.TreeParser;
import org.junit.jupiter.api.Test;
import org.w3c.dom.Document;
import org.w3c.dom.Element;

import javax.xml.parsers.DocumentBuilderFactory;
import java.util.ArrayList;
import java.util.List;
import java.util.TreeMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

public class AnnotatedTreeTest {

    /** Exposes the protected state methods. */
    static class StatefulTree extends AnnotatedTreeParser {
        void doStore() { store(); }
        void doRestore() { restore(); }
    }

    private Parameterization model() {
        Parameterization model = new Parameterization();
        model.initByName("nTypes", 1,
                "lifetime", new RealParameter("1.0"),
                "shape", new IntegerParameter("5"),
                "death", new RealParameter("0.1"),
                "sampling", new RealParameter("0.5"),
                "originTime", 10.0);
        return model;
    }

    // a tree built from taxa only: every node needs its own event
    @Test
    public void testCaterpillarFromTaxa() {
        TaxonSet taxa = new TaxonSet();
        taxa.initByName("taxon", new Taxon("a"), "taxon", new Taxon("b"), "taxon", new Taxon("c"));
        AnnotatedTree tree = new AnnotatedTree();
        tree.initByName("taxonset", taxa);

        assertEquals(5, tree.getNodeCount());
        assertDoesNotThrow(() -> tree.validate(model()));
    }

    // store and restore bring back the events together with the topology
    @Test
    public void testStoreRestoreEvents() {
        StatefulTree tree = new StatefulTree();
        tree.initByName("newick", "((1:0.4):0.4,2:0.8):1;");
        AnnotatedNode leaf = (AnnotatedNode) tree.getNode(0);
        assertEquals(2, leaf.getEventCount());
        String before = tree.toString();

        tree.doStore();
        leaf.getEvents().get(1).setHeight(0.7);
        leaf.addEvent(new AnnotatedNode.EventNode(0, 0.75), true);
        assertEquals(3, leaf.getEventCount());

        tree.doRestore();
        AnnotatedNode restored = (AnnotatedNode) tree.getNode(0);
        assertEquals(2, restored.getEventCount());
        assertEquals(0.4, restored.getEvent(1).getHeight(), 1e-12);
        assertEquals(before, tree.toString());
    }

    // the checksum reacts to moving an event within its branch
    @Test
    public void testChecksumSeesSmallChanges() {
        AnnotatedTreeParser tree = new AnnotatedTreeParser();
        tree.initByName("newick", "((1:0.4):0.4,2:0.8):1;");
        int before = tree.getChecksum();
        ((AnnotatedNode) tree.getNode(0)).getEvents().get(1).setHeight(0.5);
        assertNotEquals(before, tree.getChecksum());
    }

    // ---- two types: type 0 divides symmetrically or into (0, 1), type 1 only symmetrically ----

    // lineage of leaf 1 is type 1 at the tip, type 0 above the event at 0.4
    static final String EVENT_NEWICK = "((1[&type=1]:0.4)[&type=0]:0.4,2[&type=0]:0.8)[&type=0]:1.0;";

    private Parameterization twoTypeModel() {
        Parameterization model = new Parameterization();
        model.initByName("nTypes", 2,
                "lifetime", new RealParameter("1 1"),
                "shape", new IntegerParameter("5 5"),
                "death", new RealParameter("0.1 0.1"),
                "symTransitions", new RealParameter("0.5 0 0 1"),
                "asymTransitions", new RealParameter("0 0.5 0 0"),
                "sampling", new RealParameter("0.5 0.5"),
                "originTime", 10.0);
        return model;
    }

    private AnnotatedTreeParser parse(String newick) {
        AnnotatedTreeParser tree = new AnnotatedTreeParser();
        tree.initByName("newick", newick);
        return tree;
    }

    // events per leaf label as "type@height", independent of the node numbering
    private Map<String, List<String>> histories(AnnotatedTree tree) {
        Map<String, List<String>> histories = new TreeMap<>();
        for (Node leaf : tree.getExternalNodes()) {
            List<String> events = new ArrayList<>();
            for (AnnotatedNode.EventNode e : ((AnnotatedNode) leaf).getEvents()) {
                events.add(e.getType() + "@" + String.format("%.9f", e.getHeight()));
            }
            histories.put(leaf.getID(), events);
        }
        return histories;
    }

    // Newick with single-child nodes: every single-child node becomes an event of the branch below it
    @Test
    public void testConvertEventTree() {
        AnnotatedTreeParser tree = parse(EVENT_NEWICK);

        assertEquals(3, tree.getNodeCount());
        AnnotatedNode leaf = (AnnotatedNode) tree.getExternalNodes().stream().filter(n -> n.getID().equals("1")).findFirst().get();
        assertEquals(2, leaf.getEventCount());
        assertEquals(1, leaf.getType());
        assertEquals(0, leaf.getInitialType());
        assertEquals(0.4, leaf.getEvent(1).getHeight(), 1e-12);
        assertEquals(0.8, tree.getRoot().getHeight(), 1e-12);
        assertEquals(0, ((AnnotatedNode) tree.getRoot()).getType());
        assertDoesNotThrow(() -> tree.validate(twoTypeModel()));
    }

    // a branching tree: each node gets one event, with the type from the meta data
    @Test
    public void testConvertBranchingTree() {
        Tree branching = new TreeParser("(1[&type=1]:1.0,2[&type=0]:1.0)[&type=0]:1.0;", true);
        AnnotatedTree tree = new AnnotatedTree();
        tree.convertBranchingTree(branching);

        assertEquals(3, tree.getNodeCount());
        for (Node n : tree.getNodesAsArray()) {
            assertEquals(1, ((AnnotatedNode) n).getEventCount());
        }
        assertEquals(0, ((AnnotatedNode) tree.getRoot()).getType());
        int leafTypes = 0;
        for (Node leaf : tree.getExternalNodes()) leafTypes += ((AnnotatedNode) leaf).getType();
        assertEquals(1, leafTypes);
        assertDoesNotThrow(() -> tree.validate(twoTypeModel()));
    }

    // AnnotatedTree -> tree with single-child nodes -> AnnotatedTree gives the same histories
    @Test
    public void testConvertAnnotatedTreeRoundTrip() {
        AnnotatedTreeParser tree = parse(EVENT_NEWICK);

        Tree flat = tree.convertAnnotatedTree();
        assertEquals(4, flat.getNodeCount(), "one extra node for the event");

        AnnotatedTree back = new AnnotatedTree();
        back.convertEventTree(flat);
        assertEquals(histories(tree), histories(back));
    }

    // the string representation is parsed back to the same histories
    @Test
    public void testStringRoundTrip() {
        AnnotatedTreeParser tree = parse(EVENT_NEWICK);
        AnnotatedTreeParser parsed = parse(tree.toString());
        assertEquals(histories(tree), histories(parsed));
        assertDoesNotThrow(() -> parsed.validate(twoTypeModel()));
    }

    // fromXML (state file) reads both forms, with and without single-child nodes; tips are numbered from 0 there
    @Test
    public void testFromXML() throws Exception {
        for (String newick : new String[]{"((0[&type=1]:0.4)[&type=0]:0.4,1[&type=0]:0.8)[&type=0]:1.0;",
                "(0[&type=1]:1.0,1[&type=0]:1.0)[&type=0]:1.0;"}) {
            Document doc = DocumentBuilderFactory.newInstance().newDocumentBuilder().newDocument();
            Element element = doc.createElement("tree");
            element.setTextContent(newick);

            AnnotatedTree tree = new AnnotatedTree();
            tree.fromXML(element);

            assertEquals(3, tree.getNodeCount());
            assertDoesNotThrow(() -> tree.validate(twoTypeModel()), newick);
        }
    }

    // store/restore and copies keep types as well as heights, and are independent of the original
    @Test
    public void testStoreRestoreAndCopyKeepTypes() {
        StatefulTree tree = new StatefulTree();
        tree.initByName("newick", EVENT_NEWICK);
        int before = tree.getChecksum();

        AnnotatedTree copy = tree.copy();
        copy.initArrays();
        assertEquals(before, copy.getChecksum());

        tree.doStore();
        AnnotatedNode leaf = (AnnotatedNode) tree.getNode(0);
        leaf.getEvent(0).setType(0);
        leaf.removeEvent(1);
        assertNotEquals(before, tree.getChecksum());
        assertEquals(before, copy.getChecksum(), "the copy is independent");

        tree.doRestore();
        assertEquals(before, tree.getChecksum());

        AnnotatedTree assigned = new AnnotatedTree();
        assigned.assignFrom(tree);
        assertEquals(before, assigned.getChecksum());
    }

    // validate names the violation
    @Test
    public void testValidateRejectsInvalidTrees() {
        Parameterization model = twoTypeModel();

        AnnotatedTreeParser unsorted = parse(EVENT_NEWICK);
        AnnotatedNode leaf = (AnnotatedNode) unsorted.getNode(0);
        leaf.getEvent(1).setHeight(0.0);
        assertThrows(IllegalStateException.class, () -> unsorted.validate(model));

        // a type change must go from type 0 (above) to type 1 (below), not the other way round
        AnnotatedTreeParser wrongDirection = parse("((1[&type=0]:0.4)[&type=1]:0.4,2[&type=0]:0.8)[&type=0]:1.0;");
        assertThrows(IllegalStateException.class, () -> wrongDirection.validate(model));

        // type 1 cannot divide into types 0 and 0
        AnnotatedTreeParser impossibleDivision = parse("(1[&type=0]:1.0,2[&type=0]:1.0)[&type=1]:1.0;");
        assertThrows(IllegalStateException.class, () -> impossibleDivision.validate(model));

        // the stem starts with the origin type 0, but the root is of type 1
        AnnotatedTreeParser wrongStem = parse("(1[&type=1]:1.0,2[&type=1]:1.0)[&type=1]:1.0;");
        assertThrows(IllegalStateException.class, () -> wrongStem.validate(model));

        // without origin there is no stem: an event above the root (from the single-child node on top) is not allowed
        Parameterization noOrigin = new Parameterization();
        noOrigin.initByName("nTypes", 2, "lifetime", new RealParameter("1 1"), "shape", new IntegerParameter("5 5"),
                "death", new RealParameter("0.1 0.1"), "symTransitions", new RealParameter("0.5 0 0 1"),
                "asymTransitions", new RealParameter("0 0.5 0 0"), "sampling", new RealParameter("0.5 0.5"));
        AnnotatedTreeParser stem = parse("(((1[&type=1]:0.4)[&type=0]:0.4,2[&type=0]:0.8)[&type=0]:1.0)[&type=0]:0.5;");
        assertThrows(IllegalStateException.class, () -> stem.validate(noOrigin));
        ((AnnotatedNode) stem.getRoot()).clearEvents();
        assertDoesNotThrow(() -> stem.validate(noOrigin));
    }
}
