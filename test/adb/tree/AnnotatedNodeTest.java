package adb.tree;

import adb.tree.AnnotatedNode.EventNode;
import beast.base.evolution.tree.Node;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class AnnotatedNodeTest {

    private AnnotatedNode nodeWithEvents(int... types) {
        AnnotatedNode node = new AnnotatedNode();
        node.setHeight(1.0);
        List<EventNode> events = new ArrayList<>();
        for (int i = 0; i < types.length; i++) {
            events.add(new EventNode(types[i], 1.0 + i));
        }
        node.setEvents(events);
        return node;
    }

    @Test
    public void testTypeOfPlainNodeFromMetaData() {
        Node node = new Node();
        assertEquals(0, AnnotatedNode.getTypeOf(node), "missing meta data defaults to type 0");

        node.setMetaData("type", 2);       // Integer, as written by BranchingTreeLogger
        assertEquals(2, AnnotatedNode.getTypeOf(node));

        node.setMetaData("type", 3.0);     // Double, as parsed from Newick
        assertEquals(3, AnnotatedNode.getTypeOf(node));
    }

    @Test
    public void testTypeOfAnnotatedNodeUsesEvents() {
        AnnotatedNode annotated = nodeWithEvents(1, 0);
        annotated.setMetaData("type", 5.0); // stale meta data must be ignored

        Node asNode = annotated; // statically typed as Node, as in the likelihood
        assertEquals(1, AnnotatedNode.getTypeOf(asNode));
        assertEquals(annotated.getType(), AnnotatedNode.getTypeOf(asNode));
    }

    @Test
    public void testTypeOfAnnotatedNodeWithoutEventsUsesMetaData() {
        AnnotatedNode node = new AnnotatedNode();
        node.setMetaData("type", 2);
        assertEquals(2, AnnotatedNode.getTypeOf(node));
    }

    @Test
    public void testFindEventAbove() {
        AnnotatedNode node = nodeWithEvents(0, 0, 0, 0); // heights 1, 2, 3, 4; event 0 is the node
        assertEquals(1, node.findEventAbove(0.5), "the node event itself is never returned");
        assertEquals(1, node.findEventAbove(1.5));
        assertEquals(2, node.findEventAbove(2.0), "strictly above");
        assertEquals(4, node.findEventAbove(4.0), "no event above: event count");
    }

    @Test
    public void testCopyIsDeep() {
        AnnotatedNode node = nodeWithEvents(1, 0);
        AnnotatedNode copy = node.copy();
        copy.getEvent(1).setHeight(9.0);
        copy.setType(2);
        assertEquals(2.0, node.getEvent(1).getHeight(), 0.0);
        assertEquals(1, node.getType());
    }

    @Test
    public void testClearEventsKeepsType() {
        AnnotatedNode node = nodeWithEvents(1, 0, 0);
        node.clearEvents();
        assertEquals(1, node.getEventCount());
        assertEquals(1, node.getType());
        assertEquals(node.getHeight(), node.getEvent(0).getHeight(), 0.0);
    }
}
