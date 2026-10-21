package adb.tree;

import adb.tree.AnnotatedNode.EventNode;
import adb.distribution.Parameterization;
import beast.base.core.BEASTInterface;
import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.core.Log;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.StateNode;
import beast.base.inference.StateNodeInitialiser;
import beast.base.util.Randomizer;

import java.util.ArrayList;
import java.util.List;

import static adb.tree.AnnotatedNode.distributeEvents;

/**
 * Creates an annotated tree from a standard branching tree, e.g. a UPGMA tree, optionally with types at the tips.
 * First, node heights can be scaled. For multi-type trees, the types of the internal nodes are then
 * inferred from the tip types by parsimony. Where two daughter types cannot result from a division,
 * intermediate type changes are added along the branches, following the transition probabilities of the
 * parameterization (a type change from a progenitor to a descendant type needs a non-zero probability).
 * Likewise, type changes are added along the stem so that it starts with the origin type.
 * Finally, hidden events are placed along every branch, in regular segments according to the
 * lifetime distribution of the type, and then randomly shifted.
 * Initialisation fails if the tip types cannot be explained by the transitions of the parameterization.
 */
@Description("Class to initialize an AnnotatedTree from Tree")
public class AnnotatedTreeInitialiser extends AnnotatedTree implements StateNodeInitialiser {

    public Input<Tree> branchingTreeInput =
            new Input<>("branchingTree", "a standard BEAST2 branching tree", Input.Validate.REQUIRED);

    public Input<Parameterization> parameterizationInput =
            new Input<>("parameterization", "ADB parameterization", Input.Validate.REQUIRED);

    public Input<Double> scaleInput =
            new Input<>("scale", "factor used to multiply internal node heights", 1.0);
    // TODO: potentially use automatic scaling to adapt branch lengths to lifetimes


    Parameterization parameterization;

    @Override
    public void initAndValidate() {
        super.initAndValidate();

        Tree branchingTree = branchingTreeInput.get();
        double scale = scaleInput.get();
        parameterization = parameterizationInput.get();

        // scale tree
        if (scale != 1.0) {
            for (Node node : branchingTree.getNodesAsArray()) {
                node.setHeight(node.getHeight() * scale);
            }
        }

        // convert to AnnotatedTree
        convertBranchingTree(branchingTree);

        // parsimonous ancestral types
        if (parameterization.getNTypes() > 1) {
            for (Node node : getInternalNodes()) {
                addAncestralTypes((AnnotatedNode)node);
            }
            if (parameterization.getOriginTime() != null) {
                addStemTypes((AnnotatedNode)getRoot(), parameterization.getOriginTime());
            }
        }

        // draw events along branches
        /* IntStream.range(0, getNodeCount())
                .parallel()
                .forEach(i -> { */
        for (int i = 0; i < getNodeCount(); i++) {
            AnnotatedNode node = (AnnotatedNode)getNode(i);
            if (node.isRoot()) {
                if (parameterization.getOriginTime() != null) {
                    drawEvents(node, parameterization.getOriginTime());
                }
            } else {
                drawEvents(node, node.getParent().getHeight());
            }
            //});
        }
    }


    // Draw hidden events along the branch above a node, between each event (type change) and the next one
    private void drawEvents(AnnotatedNode node, double origin) {
        List<EventNode> typeEvents = node.getEvents(); // the node itself and the type changes
        List<EventNode> events = new ArrayList<>();
        for (int i = 0; i < typeEvents.size(); i++) {
            EventNode event = typeEvents.get(i);
            double end = (i + 1 < typeEvents.size()) ? typeEvents.get(i + 1).getHeight() : origin;
            events.add(event);
            events.addAll(drawHiddenEvents(event.getType(), event.getHeight(), end));
        }
        node.setEvents(events);
    }


    // Draw hidden events along a (single-type) branch segment according to lifetime distribution
    private List<EventNode> drawHiddenEvents(int type, double start, double end) {
        double lifetime = parameterization.getLifetime(type);
        double shape = parameterization.getShape(type);
        double length = end - start;
        int eventCount = (int)(length / lifetime);

        List<EventNode> events = new ArrayList<>();
        if (eventCount > 0) {
            double segmentLength = length / (eventCount + 1);
            double time = end - segmentLength;
            while (time > start) {
                events.add(0, new EventNode(type, time));
                time -= segmentLength;
            }
            distributeEvents(events, start, end, shape);
        }
        return events;
    }


    // Propagate types from tips to root, adding minimal intermediate transitions
    private void addAncestralTypes(AnnotatedNode node) {
        int nTypes = parameterization.getNTypes();

        int type;
        int maxSteps = 100 * nTypes; // stop if the transitions contain cycles that never resolve the types
        for (int step = 0; ; step++) {
            List<EventNode> children = node.getChildEvents(0);
            int i = children.get(0).getType();
            int j = children.get(1).getType();

            // determine node type
            if (i == j) { // symmetric division
                type = 0;
                for (int k = 0; k < nTypes; k++) {
                    if (parameterization.getSymTransition(k, i) > parameterization.getSymTransition(type, i)) {
                        type = k;
                    }
                }
                break;
            } else { // asymmetric division
                // resolve impossible transitions
                if (parameterization.getAsymTransition(i, j) == 0.0 && parameterization.getAsymTransition(j, i) == 0.0) {
                    if (step == maxSteps) {
                        throw new IllegalArgumentException("Could not resolve the types " + i + " and " + j + " in the starting tree.");
                    }

                    // select downstream branch to add intermediate types, preferably leading towards the other type
                    int child;
                    int ancestor;
                    if (parameterization.isProgenitor(i, j)) {
                        child = 1;
                        ancestor = i;
                    } else {
                        child = 0;
                        ancestor = parameterization.isProgenitor(j, i) ? j : -1;
                    }

                    // sample progenitor and height
                    int k = sampleProgenitor(children.get(child).getType(), ancestor);
                    double height = Randomizer.uniform(children.get(child).getHeight(), node.getHeight());
                    EventNode event = new EventNode(k, height);
                    ((AnnotatedNode)(node.getChild(child))).addEvent(event, true);

                } else {
                    type = (parameterization.getAsymTransition(i, j) >= parameterization.getAsymTransition(j, i)) ? i : j;
                    break;
                }
            }
        }
        node.setType(type);
    }


    // Sample a direct progenitor of a type (other than the type itself). If an ancestor type is given (not -1),
    // only progenitors that are this ancestor or descend from it are used, so that the path leads towards it.
    private int sampleProgenitor(int type, int ancestor) {
        List<Integer> candidates = new ArrayList<>();
        for (int k : parameterization.getDirectProgenitors(type)) {
            if (k != type && (ancestor < 0 || k == ancestor || parameterization.isProgenitor(ancestor, k))) {
                candidates.add(k);
            }
        }
        if (candidates.isEmpty()) {
            throw new IllegalArgumentException("No progenitor type found for type " + type +
                    ": the types in the starting tree are incompatible with the transition probabilities.");
        }
        return candidates.get(Randomizer.nextInt(candidates.size()));
    }


    // Add type changes along the stem such that it starts with the origin type
    private void addStemTypes(AnnotatedNode root, double origin) {
        int originType = parameterization.getOriginType();
        int rootType = root.getInitialType();
        if (originType != rootType && !parameterization.isProgenitor(originType, rootType)) {
            Log.warning("WARNING: Type at origin is incompatible with the starting tree.");
            return;
        }

        double height = root.getHeight();
        while (root.getInitialType() != originType) {
            int k = sampleProgenitor(root.getInitialType(), originType);
            height = Randomizer.uniform(height, origin);
            root.addEvent(new EventNode(k, height), true);
        }
    }

    
    /**
     * Check that the parameterization gives fixed start values. If it shares parameters with the State
     * (e.g. parameterization="@model"), the tree has an input that depends on the State,
     * and tree moves are then no longer recalculated or restored correctly.
     */
    @Override
    public void initStateNodes() {
        for (BEASTInterface input : parameterization.listActiveBEASTObjects()) {
            if (input instanceof StateNode && ((StateNode) input).getState() != null) {
                throw new IllegalArgumentException("The parameterization of " + getID() + " contains " + input.getID()
                        + ", which is in the State: give the initialiser its own parameterization with the start values");
            }
        }
    }

    @Override
    public void getInitialisedStateNodes(List<StateNode> stateNodeList) {
        stateNodeList.add(this);
    }
}
