package adb.distribution;

import bdmmprime.distribution.BirthDeathMigrationDistribution;
import bdmmprime.parameterization.CanonicalParameterization;
import bdmmprime.parameterization.SkylineMatrixParameter;
import bdmmprime.parameterization.SkylineVectorParameter;
import bdmmprime.parameterization.TimedParameter;
import bdmmprime.parameterization.TypeSet;
import bdmmprime.trajectories.simulation.SimulatedTree;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeParser;
import beast.base.inference.parameter.RealParameter;

import java.util.Arrays;
import java.util.stream.Collectors;


/**
 * The tree density of BDMM-Prime for a model with shape 1, as a reference for ADBTreeDistribution.
 *
 * With shape 1 the lifetimes are exponential, and the process is a multi-type birth-death process: a lineage of type i
 * with mean lifetime l and death probability d has the death rate d / l, the birth rate (1 - d) sym(i,i) / l with a
 * daughter of the same type, and the rate (1 - d) asym(i,j) / l of a division into i and j (a birth of a j from an i,
 * in the terms of BDMM-Prime). Sampling happens only at the present, with probability rho of the type. BDMM-Prime
 * has no division into two lineages of another type j, so sym(i,j) has to be 0 for i different from j.
 *
 * BDMM-Prime conditions on the survival of the process from the origin and gives the density of a tree with labelled
 * tips, with the factor 2^(n-1) / n!, so it is directly comparable to ADBTreeDistribution.
 */
final class BDMMPrimeReference {

    private BDMMPrimeReference() {}


    /** Whether the model can be written as a BDMM-Prime model: sym(i,j) = 0 for i different from j, and shape 1. */
    static boolean isRepresentable(Parameterization model) {
        for (int i = 0; i < model.getNTypes(); i++) {
            if (model.getShape(i) != 1) return false;
            for (int j = 0; j < model.getNTypes(); j++) {
                if (i != j && model.getSymTransition(i, j) != 0.0) return false;
            }
        }
        return true;
    }


    /** The BDMM-Prime parameterization of the model (one time interval, rho sampling at the present). */
    static CanonicalParameterization parameterization(Parameterization model) {
        if (!isRepresentable(model)) {
            throw new IllegalArgumentException("The model has no BDMM-Prime counterpart.");
        }
        int n = model.getNTypes();

        double[] birth = new double[n];
        double[] death = new double[n];
        double[] rho = new double[n];
        double[] crossBirth = new double[n * (n - 1)];
        int k = 0;
        for (int i = 0; i < n; i++) {
            double rate = (1 - model.getDeath(i)) / model.getLifetime(i);
            birth[i] = rate * model.getSymTransition(i, i);
            death[i] = model.getDeath(i) / model.getLifetime(i);
            rho[i] = model.getSampling(i);
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    crossBirth[k++] = rate * model.getAsymTransition(i, j);
                }
            }
        }

        RealParameter processLength = new RealParameter(Double.toString(model.getOriginTime()));
        CanonicalParameterization parameterization = new CanonicalParameterization();
        parameterization.initByName(
                "typeSet", new TypeSet(n),
                "processLength", processLength,
                "birthRate", new SkylineVectorParameter(null, new RealParameter(join(birth)), n),
                "deathRate", new SkylineVectorParameter(null, new RealParameter(join(death)), n),
                "birthRateAmongDemes", new SkylineMatrixParameter(null, new RealParameter(crossBirth.length == 0 ? "0" : join(crossBirth)), n),
                "migrationRate", new SkylineMatrixParameter(null, new RealParameter("0"), n),
                "samplingRate", new SkylineVectorParameter(null, new RealParameter("0"), n),
                "rhoSampling", new TimedParameter(processLength, new RealParameter(join(rho)), n),
                "removalProb", new SkylineVectorParameter(null, new RealParameter("1"), n));
        return parameterization;
    }


    private static RealParameter startTypeProbs(Parameterization model) {
        double[] startProbs = new double[model.getNTypes()];
        startProbs[model.getOriginType()] = 1.0;
        return new RealParameter(join(startProbs));
    }


    /**
     * Log density of the tree (types of the tips as meta data "type"), conditioned on the survival of the process
     * from the origin of the model, which starts with the origin type.
     */
    static double logDensity(Parameterization model, Tree tree) {
        BirthDeathMigrationDistribution density = new BirthDeathMigrationDistribution();
        density.initByName(
                "parameterization", parameterization(model),
                "startTypePriorProbs", startTypeProbs(model),
                "tree", tree,
                "typeLabel", "type",
                "parallelize", false);
        return density.calculateLogP();
    }


    /**
     * A tree simulated by BDMM-Prime from the model: the process runs from the origin (with the origin type) to the
     * present, all lineages are sampled with the probability rho of their type at the present, and the draw is
     * repeated until at least one lineage is sampled. This is the conditioning of the density. The random numbers
     * come from the BEAST Randomizer.
     *
     * The simulated tree has single-child nodes where the sampled lineage changed its type at a division whose other
     * daughter has no sampled descendants. They are removed, and the types of the tips are kept, so the result is the
     * tree of the tips, with the heights of the divisions that have sampled descendants on both sides. This is also what
     * the density is for. The tips are labelled 1, ..., n with the type as meta data.
     */
    static Tree simulateTree(Parameterization model) {
        RealParameter finalSampleOffset = new RealParameter("0.0");
        SimulatedTree simulated = new SimulatedTree();
        simulated.initByName(
                "parameterization", parameterization(model),
                "finalSampleOffset", finalSampleOffset,
                "startTypePriorProbs", startTypeProbs(model),
                "minSamples", 1,
                "typeLabel", "type");

        int[] tipCount = {0};
        String newick = toNewick(simulated.getRoot(), simulated.getRoot().getHeight(), tipCount) + ";";
        return new TreeParser(newick, true);
    }


    // single-child nodes are skipped, their branch is merged with the one below
    private static String toNewick(Node node, double parentHeight, int[] tipCount) {
        if (node.getChildCount() == 1) {
            return toNewick(node.getChild(0), parentHeight, tipCount);
        }
        String length = ":" + (parentHeight - node.getHeight());
        if (node.isLeaf()) {
            int type = Integer.parseInt(node.getMetaData("type").toString().replace("\"", ""));
            return (++tipCount[0]) + "[&type=" + type + "]" + length;
        }
        return "(" + toNewick(node.getChild(0), node.getHeight(), tipCount) + "," +
                toNewick(node.getChild(1), node.getHeight(), tipCount) + ")" + length;
    }


    private static String join(double[] x) {
        return Arrays.stream(x).mapToObj(Double::toString).collect(Collectors.joining(" "));
    }
}
