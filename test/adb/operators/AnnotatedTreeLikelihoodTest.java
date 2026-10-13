package adb.operators;

import adb.distribution.Parameterization;
import adb.tree.AnnotatedTreeParser;
import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.likelihood.TreeLikelihood;
import beast.base.evolution.sitemodel.SiteModel;
import beast.base.evolution.substitutionmodel.JukesCantor;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.parameter.IntegerParameter;
import beast.base.inference.parameter.RealParameter;
import beast.base.util.Randomizer;
import org.junit.jupiter.api.Test;

import java.util.function.Function;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Operators on an annotated tree combined with a sequence likelihood: the nodes an operator marks as dirty must be
 * enough for the incremental tree likelihood to equal a full recalculation. Moves that change only hidden events
 * must not mark anything, so they never force the sequence likelihood to be recomputed.
 */
public class AnnotatedTreeLikelihoodTest {

    static final String NEWICK = "(((((((5:0.704651623):0.9815616586):0.9724532301):1.076834454):1.249319317,((((2:0.7965609946):1.052640805):0.9526174057):1.110375181,(((3:0.7945840806):1.02083498):1.106252391):0.9905229342):1.072625896):0.904531321,(((((4:0.7883689605):1.024284023):0.9996663794,((1:0.9184522562):0.9222398679):0.9716272388):0.9727550889):1.034266685):1.070010466):1.036790172):0.9334925396;";
    static final int SEEDS = 100;

    private Parameterization model() {
        Parameterization model = new Parameterization();
        model.initByName("nTypes", 1,
                "lifetime", new RealParameter("0.5"),
                "shape", new IntegerParameter("5"),
                "death", new RealParameter("0.1"),
                "sampling", new RealParameter("0.5"),
                "originTime", 12.0);
        return model;
    }

    // exposes the first step of an MCMC update: forget what is dirty, except for the dirty nodes of the tree
    static class IncrementalTreeLikelihood extends TreeLikelihood {
        double calculateIncremental() {
            assertTrue(requiresRecalculation());
            return calculateLogP();
        }
    }

    private IncrementalTreeLikelihood likelihood(Tree tree) {
        Alignment data = new Alignment();
        data.initByName("dataType", "nucleotide",
                "sequence", new Sequence("1", "ACGTACGTAAGGCT"),
                "sequence", new Sequence("2", "ACGTTCGTAAGGCA"),
                "sequence", new Sequence("3", "ACCTACGTCAGGCT"),
                "sequence", new Sequence("4", "TCGTACGAAAGGCT"),
                "sequence", new Sequence("5", "ACGTACGTAAGTCT"));
        SiteModel siteModel = new SiteModel();
        siteModel.initByName("substModel", new JukesCantor());
        IncrementalTreeLikelihood likelihood = new IncrementalTreeLikelihood();
        likelihood.initByName("data", data, "tree", tree, "siteModel", siteModel);
        return likelihood;
    }

    private void assertIncrementalMatchesFull(Function<Tree, AnnotatedTreeOperator> create, boolean changesHeights) {
        int finite = 0;
        for (int seed = 1; seed <= SEEDS; seed++) {
            Randomizer.setSeed(seed);
            AnnotatedTreeParser tree = new AnnotatedTreeParser();
            tree.initByName("newick", NEWICK);
            IncrementalTreeLikelihood likelihood = likelihood(tree);
            double before = likelihood.calculateLogP();
            likelihood.store();
            tree.getRoot().makeAllDirty(Tree.IS_CLEAN);
            tree.setSomethingIsDirty(false);

            double logHR = create.apply(tree).proposal();
            if (logHR == Double.NEGATIVE_INFINITY) continue;
            finite++;

            double incremental = likelihood.calculateIncremental();
            double full = likelihood(tree).calculateLogP();
            assertEquals(full, incremental, 1e-9, "incremental equals full likelihood, seed " + seed);
            if (!changesHeights) {
                assertEquals(before, incremental, 1e-9, "hidden events do not change the sequence likelihood, seed " + seed);
            }
        }
        assertTrue(finite > SEEDS / 10, "enough valid proposals: " + finite);
    }

    @Test
    public void testEventSamplerMarksNothing() {
        // the nodes themselves stay clean
        Randomizer.setSeed(1);
        AnnotatedTreeParser tree = new AnnotatedTreeParser();
        tree.initByName("newick", NEWICK);
        tree.getRoot().makeAllDirty(Tree.IS_CLEAN);

        EventSampler operator = new EventSampler();
        operator.initByName("tree", tree, "parameterization", model(), "proportionBranches", 1.0, "weight", 1.0);
        operator.proposal();

        for (Node node : tree.getNodesAsArray()) {
            assertEquals(Tree.IS_CLEAN, node.isDirty());
        }
    }

    @Test
    public void testEventSampler() {
        assertIncrementalMatchesFull(tree -> {
            EventSampler operator = new EventSampler();
            operator.initByName("tree", tree, "parameterization", model(), "proportionBranches", 1.0, "weight", 1.0);
            return operator;
        }, false);
    }

    @Test
    public void testWilsonBalding() {
        assertIncrementalMatchesFull(tree -> {
            AnnotatedWilsonBalding operator = new AnnotatedWilsonBalding();
            operator.initByName("tree", tree, "parameterization", model(), "weight", 1.0);
            return operator;
        }, true);
    }

    @Test
    public void testNarrowExchange() {
        assertIncrementalMatchesFull(tree -> exchange(tree, true), true);
    }

    @Test
    public void testWideExchange() {
        assertIncrementalMatchesFull(tree -> exchange(tree, false), true);
    }

    @Test
    public void testSubtreeSlide() {
        assertIncrementalMatchesFull(tree -> {
            AnnotatedSubtreeSlide operator = new AnnotatedSubtreeSlide();
            operator.initByName("tree", tree, "parameterization", model(), "weight", 1.0, "size", 3);
            return operator;
        }, true);
    }

    @Test
    public void testScale() {
        assertIncrementalMatchesFull(tree -> {
            AnnotatedScale operator = new AnnotatedScale();
            operator.initByName("tree", tree, "parameterization", model(), "scaleFactor", 0.9, "weight", 1.0);
            return operator;
        }, true);
    }

    private AnnotatedExchange exchange(Tree tree, boolean narrow) {
        AnnotatedExchange operator = new AnnotatedExchange();
        operator.initByName("tree", tree, "parameterization", model(), "isNarrow", narrow, "weight", 1.0);
        return operator;
    }
}
