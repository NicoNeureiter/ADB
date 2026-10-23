package adb.operators;

import adb.distribution.ADBTreeDistribution;
import adb.distribution.Parameterization;
import adb.tree.AnnotatedNode;
import adb.tree.AnnotatedNode.EventNode;
import adb.tree.AnnotatedTreeInitialiser;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeParser;
import beast.base.inference.parameter.RealParameter;
import beast.base.util.Randomizer;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * The density of an annotated tree, integrated over the hidden events, is the density of the plain tree, for which
 * ADBTreeDistribution integrates the events out numerically. The integral is estimated branch by branch (stem
 * included) by importance sampling with resampleEvents. This ties together the annotated likelihood, the plain
 * likelihood and the normalisation of the event proposal, which the operator tests only check for consistency.
 */
public class EventIntegralTest {

    static final int DRAWS = 3000, N_STEPS = 4096;

    // log density of the plain tree minus that of the annotated tree integrated over its hidden events
    static double integralError(String newick, double origin, double lifetime, double shape, double death, double rho) {
        Parameterization model = new Parameterization();
        model.initByName("nTypes", 1, "lifetime", new RealParameter("" + lifetime), "shape", new RealParameter("" + shape),
                "death", new RealParameter("" + death), "sampling", new RealParameter("" + rho), "originTime", origin);

        TreeParser plain = new TreeParser(newick, false);
        ADBTreeDistribution plainDensity = new ADBTreeDistribution();
        plainDensity.initByName("tree", plain, "parameterization", model, "nSteps", N_STEPS);

        AnnotatedTreeInitialiser tree = new AnnotatedTreeInitialiser();
        tree.initByName("branchingTree", new TreeParser(newick, false), "parameterization", model);
        ADBTreeDistribution density = new ADBTreeDistribution();
        density.initByName("tree", tree, "parameterization", model, "nSteps", N_STEPS);
        EventSampler sampler = new EventSampler();
        sampler.initByName("tree", tree, "parameterization", model, "weight", 1.0);

        // the branch densities are factors of the tree density, so each branch is integrated with the others fixed
        double logP = density.calculateLogP(), integral = logP;
        for (Node n : tree.getNodesAsArray()) {
            AnnotatedNode node = (AnnotatedNode) n;
            List<EventNode> events = copy(node.getEvents());
            double[] logW = new double[DRAWS];
            for (int k = 0; k < DRAWS; k++) {
                double logq = sampler.resampleEvents(node, true);
                logW[k] = density.calculateLogP() - logP - logq;
                node.setEvents(copy(events));
            }
            double max = Double.NEGATIVE_INFINITY, sum = 0;
            for (double w : logW) max = Math.max(max, w);
            for (double w : logW) sum += Math.exp(w - max);
            integral += max + Math.log(sum / DRAWS);
        }
        return plainDensity.calculateLogP() - integral;
    }

    static List<EventNode> copy(List<EventNode> events) {
        List<EventNode> copy = new ArrayList<>();
        for (EventNode event : events) copy.add(event.copy());
        return copy;
    }

    // tolerance 0.03 for the grid error of the plain density and the Monte Carlo error
    @Test
    public void testCherry() {
        Randomizer.setSeed(1);
        assertEquals(0.0, integralError("(A:1.0,B:1.0);", 4.0, 1.0, 10.0, 0.3, 0.5), 0.03);
    }

    @Test
    public void testNearCriticalWithSparseSampling() {
        Randomizer.setSeed(2);
        assertEquals(0.0, integralError("((A:1.0,B:1.0):1.5,C:2.5);", 4.0, 0.5, 2.0, 0.45, 0.05), 0.03);
    }
}
