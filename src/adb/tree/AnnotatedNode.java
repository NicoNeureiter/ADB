package adb.tree;

import adb.util.Utils;
import beast.base.core.Description;
import beast.base.evolution.tree.Node;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;


/**
 * A node that carries the events along the branch above it (towards the parent).
 * The first event is the node itself, the others are hidden events in ascending order of height.
 * The type of an event is the type of the lineage from that event up to the next one (or up to the parent).
 * So getType() is the type at the node, and getInitialType() the type just below the parent.
 * The height of the first event is expected to equal the node height.
 */
@Description("An annotated node in a multi-type phylogenetic tree")
public class AnnotatedNode extends Node {

    // for sorting and comparing events by their timings
    private static final Comparator<EventNode> comparator = (n1, n2) ->
            Double.compare(n1.getHeight(), n2.getHeight());


    // marks all events on the branch above this node (backwards in time, so heights are ascending)
    // this node is first in the list
    protected List<EventNode> events = new ArrayList<>();


    // Internal class: A node marking a branching event along the lineage.
    public static class EventNode {

        protected int type;
        protected double height;

        public EventNode(int type, double height) {
            this.type = type;
            this.height = height;
        }

        public int getType() {
            return type;
        }

        public void setType(int type) {
            this.type = type;
        }

        public double getHeight() {
            return height;
        }

        public void setHeight(double height) {
            this.height = height;
        }

        public EventNode copy() {
            return new EventNode(type, height);
        }
    }


    public List<EventNode> getEvents() {
        return events;
    }

    public void setEvents(List<EventNode> events) {
        if (events.isEmpty()) {
            throw new IllegalStateException("AnnotatedNode cannot have an empty event list!");
        }

        startEditing();
        this.events = events;
    }

    // restore event list with the node itself being the only event
    public void clearEvents() {
        startEditing();
        int type = getTypeOf(this); // read before the events are removed
        events.clear();
        events.add(new EventNode(type, height));
    }

    public int getEventCount() {
        return events.size();
    }

    public EventNode getEvent(int idx) {
        return events.get(idx);
    }

    public void setEvent(EventNode event, int idx) {
        startEditing();
        events.set(idx, event);
    }

    public int getInitialType() { // past type
        return events.get(events.size() - 1).type;
    }

    public int getType() { // present type
        return events.get(0).type;
    }

    public void setType(int type) {
        startEditing();
        events.get(0).setType(type);
    }

    /**
     * Get the type of any node: the present type from the event list of an annotated node,
     * otherwise (plain nodes, or annotated nodes without events yet) the "type" meta data (default 0).
     * Always use this for variables declared as Node: calling getType() requires an AnnotatedNode.
     */
    public static int getTypeOf(Node node) {
        if (node instanceof AnnotatedNode && !((AnnotatedNode) node).events.isEmpty()) {
            return ((AnnotatedNode) node).getType();
        }
        Object type = node.getMetaData("type");
        return type == null ? 0 : ((Number) type).intValue();
    }

    public void addEvent(EventNode event, boolean append) {
        startEditing();
        if (append) { // append at end
            events.add(event);
        } else { // find the correct spot
            int idx = Collections.binarySearch(events, event, comparator);
            if (idx < 0) { idx = -(idx + 1); } // convert to insertion point
            events.add(idx, event);
        }
    }

    public void addEvent(EventNode event, int idx) { // add at given position, remaining entries will be shifted +1
        startEditing();
        events.add(idx, event);
    }

    public void addEvents(List<EventNode> events, boolean append) {
        startEditing();
        if (append) { // append at end
            this.events.addAll(events);
        } else { // find the correct spot
            for (EventNode event : events) {
                addEvent(event, false);
            }
        }
    }

    public void addEvents(List<EventNode> events, int ix) {
        startEditing();
        this.events.addAll(ix, events);
    }

    public void removeEvent(int idx) {
        startEditing();
        if (idx >= events.size())
            throw new IllegalArgumentException("Index to removeEvent() out of range.");
        events.remove(idx);
    }

    // binary search to find the index of the first hidden event above some height threshold
    public int findEventAbove(double height) {
        int ix = events.size();
        int min = 1;
        int max = events.size() - 1;
        while (min <= max) {
            int mid = (min + max) / 2;
            if (events.get(mid).getHeight() > height) {
                ix = mid;
                max = mid - 1; // try to find an even earlier valid index
            } else {
                min = mid + 1;
            }
        }
        return ix;
    }

    public void sortEvents() {
        startEditing();
        events.sort(comparator);
    }

    public EventNode getParentEvent(int idx) {
        int pidx = idx + 1;
        if (pidx < events.size()) {
            return events.get(pidx); // predecessor is a hidden node
        } else {
            AnnotatedNode parent = (AnnotatedNode)this.parent;
            if (parent == null) {
                return null; // this event is first in the whole tree
            } else {
                return parent.events.get(0); // predecessor is a branching node
            }
        }
    }

    public List<EventNode> getChildEvents(int idx) { // get next event(s)
        List<EventNode> children = new ArrayList<>();
        if (idx > 0) {
            // one child (hidden node)
            children.add(events.get(idx - 1));
        } else {
            if (this.isLeaf()) {
                // no children
                return null;
            } else {
                // two children
                AnnotatedNode left = (AnnotatedNode)getLeft();
                children.add(left.getEvent(left.getEventCount() - 1));

                AnnotatedNode right = (AnnotatedNode)getRight();
                children.add(right.getEvent(right.getEventCount() - 1));
            }
        }
        return children;
    }

    public double[] getWaitingTimes() { // if operating on trees without stem branch
        double[] times = new double[events.size()];
        for (int i = 0; i < times.length; i++) {
            if (this.getParentEvent(i) == null) { // this only be true for the 0th event of the root
                times[i] = 0;
            } else {
                times[i] = this.getParentEvent(i).getHeight() - events.get(i).getHeight();
            }
        }
        return times;
    }

    public double[] getWaitingTimes(double origin) { // if operating on trees with stem branch
        double[] times = new double[events.size()];
        for (int i = 0; i < times.length; i++) {
            if (getParentEvent(i) == null) { // relevant for the first event in the tree
                times[i] = origin - events.get(i).getHeight();
            } else {
                times[i] = getParentEvent(i).getHeight() - events.get(i).getHeight();
            }
        }
        return times;
    }

    // re-distribute events according to Dirichlet distribution
    public void distributeEvents(double alpha, double origin) {
        startEditing();

        double[] segmentLengths = this.getWaitingTimes(origin); // TODO: special option for stem branch?
        Utils.distributeDirichlet(segmentLengths, alpha);

        double height = getHeight();
        for (int i = 1; i < events.size(); i++) {
            height = height + segmentLengths[i - 1];
            events.get(i).setHeight(height);
        }
    }

    // re-distribute events according to Dirichlet distribution
    public static void distributeEvents(List<EventNode> events, double start, double end, double alpha) {
        int n = events.size();

        // collect waiting times
        double[] segmentLengths = new double[n + 1];
        segmentLengths[0] = events.get(0).getHeight() - start;
        for (int i = 1; i < n; i++) {
            segmentLengths[i] = events.get(i).getHeight() - events.get(i - 1).getHeight();
        }
        segmentLengths[n] = end - events.get(n - 1).getHeight();

        // re-distribute
        Utils.distributeDirichlet(segmentLengths, alpha);

        double height = start;
        for (int i = 0; i < n; i++) {
            height = height + segmentLengths[i];
            events.get(i).setHeight(height);
        }
    }

    // deep copy of events
    public void assignEventsFrom(AnnotatedNode node) {
        events.clear();
        for (EventNode event : node.events) {
            events.add(event.copy());
        }
    }

    /**
     * ************************ *
     * Methods ported from Node *
     * ************************ *
     */

    /**
     * Assign values from a tree in array representation *
     */
    @Override
    public void assignFrom(Node[] nodes, final Node node) {
        height = node.getHeight();
        labelNr = node.getNr();
        metaDataString = node.metaDataString;
        parent = null;
        ID = node.getID();
        assignEventsFrom((AnnotatedNode)node);

        if (node.getLeft() != null) {
            setLeft(nodes[node.getLeft().getNr()]);
            getLeft().assignFrom(nodes, node.getLeft());
            getLeft().setParent(this);
            if (node.getRight() != null) {
                setRight(nodes[node.getRight().getNr()]);
                getRight().assignFrom(nodes, node.getRight());
                getRight().setParent(this);
            }
        }
    }

    /**
     * (Deep) copy of node *
     */
    @Override
    public AnnotatedNode copy() {
        AnnotatedNode node = new AnnotatedNode();
        node.height = height;
        node.labelNr = labelNr;
        node.metaDataString = metaDataString;
        node.parent = null;
        node.ID = ID;
        node.events = new ArrayList<>();
        for (EventNode e : events) {
            node.events.add(e.copy());
        }
        if (getLeft() != null) {
            node.setLeft(getLeft().copy());
            node.getLeft().setParent(node);
            if (getRight() != null) {
                node.setRight(getRight().copy());
                node.getRight().setParent(node);
            }
        }
        return node;
    }

}
