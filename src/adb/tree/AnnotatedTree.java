package adb.tree;

import adb.tree.AnnotatedNode.EventNode;

import adb.distribution.Parameterization;
import beast.base.core.Description;
import beast.base.core.Log;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeParser;
import beast.base.inference.StateNode;
import beast.base.inference.StateNodeInitialiser;

import java.io.PrintStream;
import java.util.*;

import static adb.tree.AnnotatedNode.getTypeOf;


/**
 * A tree whose nodes are AnnotatedNodes: each node carries the events (type changes and hidden events of the same type)
 * along the branch above it, see AnnotatedNode. Node numbers index the node array (leaves first, root last),
 * and the tree must keep that numbering. Store, restore and the assign methods copy the events together with
 * the topology and heights. Trees are read and written as Newick with one single-child node per event,
 * and can be built from a plain branching tree (convertBranchingTree), where every branch only has its node as event.
 * Use validate to check a tree against a model.
 */
@Description("An annotated multi-type phylogenetic tree.")
public class AnnotatedTree extends Tree {

    public AnnotatedTree() { };

    public AnnotatedTree(Node root) {

        if (!(root instanceof AnnotatedNode))
            throw new IllegalArgumentException("Attempted to instantiate "
                    + "annotated tree with regular root node.");

        setRoot(root); // also updates nodeCount
        internalNodeCount = root.getInternalNodeCount();
        leafNodeCount = root.getLeafNodeCount();
        initArrays();
    }


    @Override
    public void initAndValidate() {

        if (m_initial.get() != null && !(this instanceof StateNodeInitialiser)) {

            if (!(m_initial.get() instanceof AnnotatedTree)) {
                throw new IllegalArgumentException("Attempted to initialise "
                        + "multi-type tree with regular tree object.");
            }

            AnnotatedTree other = (AnnotatedTree)m_initial.get();
            root = other.root.copy();
            nodeCount = other.nodeCount;
            internalNodeCount = other.internalNodeCount;
            leafNodeCount = other.leafNodeCount;
        }

        if (nodeCount < 0) { // if only taxa provided
            if (m_taxonset.get() != null) {
                // make a caterpillar
                List<String> sTaxa = m_taxonset.get().asStringList();
                Node left = newEventNode(0, 0, sTaxa.get(0));
                for (int i = 1; i < sTaxa.size(); i++) {
                    Node right = newEventNode(i, 0, sTaxa.get(i));
                    Node parent = newEventNode(sTaxa.size() + i - 1, i, null);
                    left.setParent(parent);
                    parent.setLeft(left);
                    right.setParent(parent);
                    parent.setRight(right);
                    left = parent;
                }
                root = left;
                leafNodeCount = sTaxa.size();
                nodeCount = leafNodeCount * 2 - 1;
                internalNodeCount = leafNodeCount - 1;

            } else {
                // make dummy tree with a single root node
                root = newEventNode(0, 0, null);
                root.setTree(this);
                nodeCount = 1;
                internalNodeCount = 0;
                leafNodeCount = 1;
            }
        }

        if (nodeCount >= 0) {
            initArrays();
        }

        processTraits(m_traitList.get());

        // ensure tree is compatible with traits
        if (hasDateTrait()) {
            adjustTreeNodeHeights(root);
        }

        // ensure all nodes have their taxon names set up
        String[] taxa = getTaxaNames();
        for (int i = 0; i < getNodeCount() && i < taxa.length; i++) {
            if(taxa[i] != null) {
                if (m_nodes[i] == null) {
                    Log.warning("WARNING: Expected a node for taxon " + taxa[i] + " but did not find one in the expected location in the m_nodes array");
                } else if(m_nodes[i].getID() == null) {
                    m_nodes[i].setID(taxa[i]);
                }
            }
        }

    }

    // a node of type 0 with its own event only, for basic tree construction
    private AnnotatedNode newEventNode(int nr, double height, String id) {
        AnnotatedNode node = new AnnotatedNode();
        node.setNr(nr);
        node.setHeight(height);
        node.setID(id);
        node.getEvents().add(new EventNode(0, height));
        return node;
    }


    @Override
    protected AnnotatedNode newNode() {
        return new AnnotatedNode();
    }


    /**
     * Converts a tree in AnnotatedTree: a tree with single-child nodes is read as an event tree (convertEventTree),
     * otherwise it must be strictly bifurcating (convertBranchingTree).
     */
    public void convertTree(Tree tree) {
        boolean containsEvents = false;
        for (int i = 0; i < tree.getNodeCount(); i++) {
            if (tree.getNode(i).getChildCount() == 1) {
                containsEvents = true;
                break;
            }
        }

        if (containsEvents) {
            convertEventTree(tree);
        } else {
            convertBranchingTree(tree);
        }
    }


    /**
     * Converts a strictly bifurcating tree in AnnotatedTree
     */
    // cf. https://github.com/tgvaughan/MultiTypeTree/blob/master/src/multitypetree/evolution/tree/MultiTypeTreeFromUntypedNewick.java
    protected void convertBranchingTree(Tree tree) {

        // create all nodes
        AnnotatedNode[] annotatedNodes = new AnnotatedNode[tree.getNodeCount()];
        for (int i = 0; i < annotatedNodes.length; i++) {
            Node node = tree.getNode(i);
            AnnotatedNode aNode = new AnnotatedNode();
            aNode.setNr(i);
            aNode.setHeight(node.getHeight());
            aNode.setID(node.getID());
            List<EventNode> events = new ArrayList<>();
            EventNode branchingEvent = new EventNode(getTypeOf(node), node.getHeight());
            events.add(branchingEvent);
            aNode.setEvents(events);
            annotatedNodes[i] = aNode;
        }

        // preserve branching structure
        for (int i = 0; i < annotatedNodes.length; i++) {
            AnnotatedNode aNode = annotatedNodes[i];
            Node node = tree.getNode(i);

            if (node.isRoot()) {
                aNode.setParent(null);
            } else {
                aNode.setParent(annotatedNodes[node.getParent().getNr()]);
            }

            while (aNode.getChildrenMutable().size() < node.getChildCount()) {
                aNode.getChildrenMutable().add(null);
            }

            for (int c = 0; c < node.getChildCount(); c++) {
                aNode.setChild(c, annotatedNodes[node.getChild(c).getNr()]);
            }
        }

        // construct AnnotatedTree
        AnnotatedNode aRoot = annotatedNodes[tree.getRoot().getNr()];
        assignFromWithoutID(new AnnotatedTree(aRoot));
        initArrays();
    }


    /**
     * Converts a tree with single-child nodes in AnnotatedTree
     */
    // cf. https://github.com/tgvaughan/MultiTypeTree/blob/master/src/multitypetree/evolution/tree/MultiTypeTree.java#L538
    // but using tips-to-root traversal
    protected void convertEventTree(Tree tree) {

        // map to keep track of the AnnotatedNodes (key: original Node Nr, value: new AnnotatedNode)
        Map<Integer, AnnotatedNode> nodeMap = new HashMap<>();

        // get nodes sorted by height (tips first)
        Node[] nodes = tree.getNodesAsArray();

        // initialise counter of nodes
        AnnotatedNode aRoot = null;
        int nextNr = 0;

        for (Node node : nodes) {
            // only create AnnotatedNode for tips or branching points
            if (node.getChildCount() == 1) continue;

            AnnotatedNode aNode = new AnnotatedNode();
            aNode.setHeight(node.getHeight());
            aNode.setID(node.getID());
            aNode.setNr(nextNr);

            List<EventNode> events = new ArrayList<>();
            events.add(new EventNode(getTypeOf(node), node.getHeight()));
            // CLIMB UP: look at the lineage above this node and collect all single-child nodes as events for THIS node
            Node parent = node.getParent();
            while (parent != null && parent.getChildCount() == 1) {
                events.add(new EventNode(getTypeOf(parent), parent.getHeight()));
                parent = parent.getParent();
            }
            aNode.setEvents(events);

            // BRANCHING: store this node so its parent can find it later
            if (node.getChildCount() == 2) {
                // find the children we already created and add them to this parent
                aNode.addChild(findBranchingChild(node.getLeft(), nodeMap));
                aNode.addChild(findBranchingChild(node.getRight(), nodeMap));
            }

            nodeMap.put(node.getNr(), aNode);
            aRoot = aNode;
            nextNr++;
        }

        // construct AnnotatedTree
        assignFromWithoutID(new AnnotatedTree(aRoot));
        initArrays();
    }


   // helper to skip the single-child event nodes and find the underlying AnnotatedNode child stored in the map
    private AnnotatedNode findBranchingChild(Node child, Map<Integer, AnnotatedNode> nodeMap) {
        Node current = child;
        while (current.getChildCount() == 1) {
            current = current.getLeft();
        }
        return nodeMap.get(current.getNr());
    }


    /**
     * Converts AnnotatedTree to Tree with single-child nodes compatible with Newick format
     */
    // adapted from https://github.com/tgvaughan/MultiTypeTree/blob/master/src/multitypetree/evolution/tree/MultiTypeTree.java#L444
    public Tree convertAnnotatedTree() {

        // create new tree to modify
        Tree tree = copy();
        tree.initArrays();

        int nextNodeNr = getNodeCount();
        Node eventNode;
        Node newRoot = tree.getRoot(); // might be updated

        for (Node node : getNodesAsArray()) {

            AnnotatedNode aNode = (AnnotatedNode)node;
            int nodeNr = node.getNr();

            Node startNode = tree.getNode(nodeNr);
            startNode.setMetaData("type", aNode.getType());
            startNode.metaDataString = String.format("%s=%d", "type", aNode.getType());

            Node endNode = startNode.getParent();

            Node branchNode = startNode;
            for (int i = 1; i < aNode.getEventCount(); i++) {

                // create and label new node
                eventNode = new AnnotatedNode();
                eventNode.setNr(nextNodeNr);
                nextNodeNr++;

                // connect to child and parent
                branchNode.setParent(eventNode);
                eventNode.addChild(branchNode);

                // ensure height and type are set
                eventNode.setHeight(aNode.getEvent(i).getHeight());
                eventNode.setMetaData("type", aNode.getEvent(i).getType());
                eventNode.metaDataString = String.format("%s=%d", "type", aNode.getEvent(i).getType());

                // update branchNode
                branchNode = eventNode;
            }

            // mark new root
            if (endNode == null) {
                newRoot = branchNode;
                continue;
            }

            // ensure final branchNode is connected to the original parent
            branchNode.setParent(endNode);
            if (endNode.getLeft() == startNode) {
                endNode.setLeft(branchNode);
            } else {
                endNode.setRight(branchNode);
            }
            endNode.setMetaData("type", ((AnnotatedNode)node.getParent()).getType());
            endNode.metaDataString = String.format("%s=%d", "type", ((AnnotatedNode)node.getParent()).getType());
        }

        // post-hoc, renumber internal nodes
        numberInternalNodes(newRoot, newRoot.getAllLeafNodes().size());

        return new Tree(newRoot);
    }


    // helper to assign sensible node numbers to each internal node (children before parents)
    private int numberInternalNodes(Node node, int nextNr) {
        if (node.isLeaf())
            return nextNr;

        for (Node child : node.getChildren())
            nextNr = numberInternalNodes(child, nextNr);

        node.setNr(nextNr);
        //node.setID(String.valueOf(nextNr));

        return nextNr + 1;
    }


    /**
     * Checks that the tree is a valid history under the model, and throws an IllegalStateException naming the first
     * violation otherwise. Every node needs its own event first, with the node height, followed by hidden events in
     * strictly ascending order below the parent (below the origin for the root). A type change must go from a
     * direct progenitor to a descendant type, the division at every internal node must be possible,
     * and with an origin the stem must start with the origin type.
     * Linear in the number of events, intended for tests and assertions rather than for every MCMC step.
     */
    public void validate(Parameterization model) {
        for (Node n : getNodesAsArray()) {
            AnnotatedNode node = (AnnotatedNode) n;
            List<EventNode> events = node.getEvents();
            String where = "node " + node.getNr() + ": ";
            if (events.isEmpty() || events.get(0).getHeight() != node.getHeight()) {
                throw new IllegalStateException(where + "the first event must be the node itself");
            }
            for (int e = 1; e < events.size(); e++) {
                if (events.get(e).getHeight() <= events.get(e - 1).getHeight()) {
                    throw new IllegalStateException(where + "events must be in ascending order");
                }
                int below = events.get(e - 1).getType(), above = events.get(e).getType();
                if (above != below && !model.isDirectProgenitor(above, below)) {
                    throw new IllegalStateException(where + "type " + above + " cannot change to type " + below);
                }
            }
            double top = node.isRoot() ? (model.getOriginTime() == null ? Double.POSITIVE_INFINITY : model.getOriginTime())
                    : node.getParent().getHeight();
            if (top <= node.getHeight() || events.get(events.size() - 1).getHeight() >= top) {
                throw new IllegalStateException(where + "events must lie below the parent");
            }
            if (!node.isLeaf()) {
                AnnotatedNode left = (AnnotatedNode) node.getLeft(), right = (AnnotatedNode) node.getRight();
                if (model.getDivisionFactor(node.getType(), left.getInitialType(), right.getInitialType()) <= 0.0) {
                    throw new IllegalStateException(where + "division of type " + node.getType() + " into types "
                            + left.getInitialType() + " and " + right.getInitialType() + " is not possible");
                }
            }
        }
        AnnotatedNode root = (AnnotatedNode) getRoot();
        if (model.getOriginTime() != null && root.getInitialType() != model.getOriginType()) {
            throw new IllegalStateException("the stem must start with the origin type " + model.getOriginType());
        }
    }


    /*
      ************************ *
      Methods ported from Tree *
      (adapted from MultiTypeTree)
      ************************ *
     */

    /**
     * Initialise tree-as-array representation + its stored variant *
     */
    @Override
    public void initArrays() {
        // initialise tree-as-array representation + its stored variant
        m_nodes = new AnnotatedNode[nodeCount];
        listNodes((AnnotatedNode)root, (AnnotatedNode[])m_nodes);
        m_storedNodes = new AnnotatedNode[nodeCount];
        Node copy = root.copy();
        listNodes((AnnotatedNode)copy, (AnnotatedNode[])m_storedNodes);
    }

    /**
     * Convert tree to array representation *
     */
    private void listNodes(AnnotatedNode node, AnnotatedNode[] nodes) {
        nodes[node.getNr()] = node;
        node.setTree(this);
        if (!node.isLeaf()) {
            listNodes((AnnotatedNode)node.getLeft(), nodes);
            if (node.getRight()!=null)
                listNodes((AnnotatedNode)node.getRight(), nodes);
        }
    }

    /**
     * Deep copy, returns a completely new tree *
     */
    @Override
    public AnnotatedTree copy() {
        AnnotatedTree tree = new AnnotatedTree();
        tree.ID = ID;
        tree.index = index;
        tree.root = root.copy();
        tree.nodeCount = nodeCount;
        tree.internalNodeCount = internalNodeCount;
        tree.leafNodeCount = leafNodeCount;
        return tree;
    }

    /**
     * Copy of all values from existing annotated tree *
     */
    @Override
    public void assignFrom(StateNode other) {
        AnnotatedTree tree = (AnnotatedTree) other;
        AnnotatedNode[] nodes = new AnnotatedNode[tree.getNodeCount()];
        for (int i = 0; i < tree.getNodeCount(); i++) {
            nodes[i] = new AnnotatedNode();
        }
        ID = tree.ID;
        root = nodes[tree.root.getNr()];
        root.assignFrom(nodes, tree.root);
        root.setParent(null);
        nodeCount = tree.nodeCount;
        internalNodeCount = tree.internalNodeCount;
        leafNodeCount = tree.leafNodeCount;
        initArrays();
    }

    /**
     * As assignFrom, but only copy tree structure *
     */
    @Override
    public void assignFromFragile(final StateNode other) {
        AnnotatedTree tree = (AnnotatedTree)other;
        if (m_nodes == null) {
            initArrays();
        }
        root = m_nodes[tree.root.getNr()];
        Node[] otherNodes = tree.m_nodes;
        int rootNr = root.getNr();
        assignFromFragileHelper(0, rootNr, otherNodes);
        root.setHeight(otherNodes[rootNr].getHeight());
        root.setParent(null);

        ((AnnotatedNode)root).assignEventsFrom((AnnotatedNode)otherNodes[rootNr]);

        if (otherNodes[rootNr].getLeft() != null) {
            root.setLeft(m_nodes[otherNodes[rootNr].getLeft().getNr()]);
        } else {
            root.setLeft(null);
        }
        if (otherNodes[rootNr].getRight() != null) {
            root.setRight(m_nodes[otherNodes[rootNr].getRight().getNr()]);
        } else {
            root.setRight(null);
        }
        assignFromFragileHelper(rootNr + 1, nodeCount, otherNodes);
    }

    /**
     * Helper to assignFromFragile *
     */
    private void assignFromFragileHelper(int start, int end, Node[] otherNodes) {
        for (int i = start; i < end; i++) {
            AnnotatedNode sink = (AnnotatedNode)m_nodes[i];
            AnnotatedNode src = (AnnotatedNode)otherNodes[i];
            sink.setHeight(src.getHeight());
            sink.setParent(m_nodes[src.getParent().getNr()]);
            sink.assignEventsFrom(src);
            if (src.getLeft() != null) {
                sink.setLeft(m_nodes[src.getLeft().getNr()]);
                if (src.getRight() != null) {
                    sink.setRight(m_nodes[src.getRight().getNr()]);
                } else {
                    sink.setRight(null);
                }
            }
        }
    }

    /**
     * Store current StateNode *
     */
    @Override
    protected void store() {
        int rootNr = root.getNr();
        storedRoot = m_storedNodes[rootNr];

        storeNodes(0, rootNr);

        storedRoot.setHeight(m_nodes[rootNr].getHeight());
        ((AnnotatedNode)storedRoot).assignEventsFrom((AnnotatedNode)m_nodes[rootNr]);
        storedRoot.setParent(null);
        if (root.getLeft() != null) {
            storedRoot.setLeft(m_storedNodes[root.getLeft().getNr()]);
        } else {
            storedRoot.setLeft(null);
        }
        if (root.getRight() != null) {
            storedRoot.setRight(m_storedNodes[root.getRight().getNr()]);
        } else {
            storedRoot.setRight(null);
        }

        storeNodes(rootNr + 1, nodeCount);
    }

    /**
     * Helper to store *
     */
    private void storeNodes(int start, int end) {
        for (int i = start; i < end; i++) {
            AnnotatedNode sink = (AnnotatedNode)m_storedNodes[i];
            AnnotatedNode src = (AnnotatedNode)m_nodes[i];
            sink.setHeight(src.getHeight());
            sink.setParent(m_storedNodes[src.getParent().getNr()]);
            if (src.getLeft() != null) {
                sink.setLeft(m_storedNodes[src.getLeft().getNr()]);
                if (src.getRight() != null)
                    sink.setRight(m_storedNodes[src.getRight().getNr()]);
                else
                    sink.setRight(null);
            }
            sink.assignEventsFrom(src);
        }
    }

    /**
     * Loggable interface *
     */
    @Override
    public void init(PrintStream printStream) {
        printStream.println("#NEXUS\n");
        printStream.println("Begin taxa;");
        printStream.println("\tDimensions ntax=" + getLeafNodeCount() + ";");
        printStream.println("\t\tTaxlabels");
        for (int i = 0; i < getLeafNodeCount(); i++)
            printStream.println("\t\t\t" + getNodesAsArray()[i].getID());
        printStream.println("\t\t\t;");
        printStream.println("End;");

        printStream.println("Begin trees;");
        printStream.println("\tTranslate");
        for (int i = 0; i < getLeafNodeCount(); i++) {
            printStream.print("\t\t\t" + (getNodesAsArray()[i].getNr() + 1)
                    + " " + getNodesAsArray()[i].getID());
            if (i < getLeafNodeCount()-1)
                printStream.print(",");
            printStream.print("\n");
        }
        printStream.print("\t\t\t;");
    }

    @Override
    public void log(long i, PrintStream printStream) {
        printStream.print("tree STATE_" + i + " = ");
        printStream.print(toString());
        printStream.print(";");
    }

    @Override
    public void close(PrintStream printStream) {
        printStream.println("End;");
    }

    /**
     * String representation for logging *
     */
    @Override
    public String toString() {
        // behaves differently if writing a state file
        StackTraceElement[] ste = Thread.currentThread().getStackTrace();
        if (ste[2].getMethodName().equals("toXML")) {
            // use toShortNewick to generate Newick string without taxon labels
            return convertAnnotatedTree().getRoot().toShortNewick(true);
        } else{
            return convertAnnotatedTree().getRoot().toSortedNewick(new int[1], true);
        }
    }

    // TODO: currently slightly different from MultiTypeTree version
    /**
     * Reconstruct tree from XML fragment in the form of a DOM node *
     */
    @Override
    public void fromXML(final org.w3c.dom.Node node) {
        Tree tree = new TreeParser();
        tree.initByName(
                "newick", node.getTextContent(),
                "adjustTipHeights", true,
                "IsLabelledNewick", false,
                "offset", 0);

        convertTree(tree);
    }


    /**
     * Debug check that two trees are the same: hash of the event heights, the event types and the number of
     * events per node, in node order.
     */
    @Override
    public int getChecksum() {
        List<Double> heights = new ArrayList<>();
        List<Integer> types = new ArrayList<>();
        for (Node node : getNodesAsArray()) {
            List<EventNode> events = ((AnnotatedNode) node).getEvents();
            types.add(events.size());
            for (EventNode event : events) {
                heights.add(event.getHeight());
                types.add(event.getType());
            }
        }
        return Objects.hash(heights, types);
    }

}
