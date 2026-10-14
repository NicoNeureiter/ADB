package adb.tree;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeParser;
import beast.base.inference.StateNode;
import beast.base.inference.StateNodeInitialiser;

import java.io.*;
import java.util.List;


/**
 * Reads the tree from a Newick string, or from one line of a file with several trees (fileName and treeIndex,
 * counting from 1, trees separated by semicolons). Types are read from the type meta data of every node
 * (for example 1[&type=0]:0.5); nodes without it have type 0. The type of an internal or single-child node is the
 * type of the lineage above it.
 *
 * If the Newick contains single-child nodes, each of them is an event on the branch below it, and the tree is
 * converted with convertEventTree. Otherwise, the tree is bifurcating, and convertBranchingTree gives every branch
 * only its node as event.
 *
 * By default (adjustTipHeights) all tips are moved to height 0, as there are no dates. Set it to false to keep
 * the tip heights as given. The scale input multiplies the internal node heights.
 */
@Description("Initialises an AnnotatedTree from a Newick tree with type meta data and optional single-child event nodes.")
public class AnnotatedTreeParser extends AnnotatedTree implements StateNodeInitialiser {

    public Input<String> newickInput =
            new Input<>("newick", "tree in newick format");

    public Input<String> fileNameInput =
            new Input<>("fileName", "name of file containing the tree in newick format",
                    Input.Validate.XOR, newickInput);

    public Input<Integer> treeIndexInput =
            new Input<>("treeIndex", "index of tree in tree file, counting from 1 (default 1).", 1);

    public Input<Boolean> adjustTipHeightsInput =
            new Input<>("adjustTipHeights", "adjust tip heights in tree? (default true)", true);

    public Input<Double> scaleInput =
            new Input<>("scale", "factor used to multiply internal node heights during parsing", 1.0);


    public AnnotatedTreeParser() { }


    @Override
    public void initAndValidate() {

        String newick = fileNameInput.get() != null
                ? readTreeFromFile(fileNameInput.get(), treeIndexInput.get())
                : newickInput.get();

        super.initAndValidate();

        // read tree with TreeParser, which only handles Node, then convert it to AnnotatedNodes
        Tree tree = new TreeParser();
        tree.initByName(
                "newick", newick,
                "adjustTipHeights", adjustTipHeightsInput.get(),
                "scale", scaleInput.get(),
                "IsLabelledNewick", true); // tip labels are taken from the Newick, unlike in AnnotatedTree.fromXML

        convertTree(tree);
    }


    // adapted from feast.fileio.TreeFromNewickFile
    static String readTreeFromFile(String fileName, int treeIndex) {
        StringBuilder newickBuilder = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new FileReader(fileName))) {
            String line;
            while ((line = reader.readLine()) != null)
                newickBuilder.append(line.trim());
        } catch (IOException e) {
            throw new RuntimeException("Could not read input file " + fileName + ".", e);
        }
        String content = newickBuilder.toString();

        String[] treeStrings = content.split(";");
        if (treeIndex < 1 || treeIndex > treeStrings.length)
            throw new IllegalArgumentException("Tree index " + treeIndex + " is not between 1 and the number of trees in "
                    + fileName + " (" + treeStrings.length + ").");
        return treeStrings[treeIndex - 1];
    }


    @Override
    public void initStateNodes() { }

    @Override
    public void getInitialisedStateNodes(List<StateNode> stateNodeList) {
        stateNodeList.add(this);
    }

}
