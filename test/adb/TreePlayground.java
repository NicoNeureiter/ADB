package adb;

import adb.distribution.ADBTreeDistribution;
import adb.distribution.Parameterization;
import adb.operators.*;
import adb.tree.AnnotatedTree;
import adb.tree.AnnotatedTreeInitialiser;
import adb.tree.AnnotatedTreeParser;
import beast.base.inference.parameter.IntegerParameter;
import beast.base.inference.parameter.RealParameter;
import beast.base.util.Randomizer;

import java.io.File;

/**
 * Scratch pad: read a tree, validate it against the model, apply moves and print the result.
 */
public class TreePlayground {

    // ---- edit me ----
    static String NEWICK = "((((1[&type=3]:0.5050647894)[&type=3]:2.629032343,(((3[&type=3]:0.02658926371)[&type=1]:1.167256789)[&type=1]:0.9582704094)[&type=1]:0.9819806705)[&type=1]:0.8807615117,((((2[&type=1]:0.3540064826)[&type=1]:0.9311262134)[&type=1]:1.069878825)[&type=1]:1.123534175)[&type=0]:0.536312948)[&type=0]:0.518395086,((((((((10[&type=0]:0.2977362855)[&type=0]:0.5473791622)[&type=0]:0.5281869737)[&type=0]:0.4965747618)[&type=0]:0.536755725)[&type=0]:0.4893282464,((((8[&type=2]:0.1301792254)[&type=1]:1.243179893)[&type=0]:0.4946663287)[&type=0]:0.4916443901)[&type=0]:0.5362913174)[&type=0]:0.5393012838)[&type=0]:0.5350052332,(((((5[&type=2]:0.173317248)[&type=2]:1.486935976)[&type=1]:0.8881744641)[&type=0]:0.4923866848,((((7[&type=1]:0.4430713129)[&type=1]:1.124596537)[&type=0]:0.4915929058)[&type=0]:0.5159186473)[&type=0]:0.4656349701)[&type=0]:0.4771876381,((((4[&type=3]:0.6597367559)[&type=1]:0.9506927264)[&type=1]:1.078905666,(((6[&type=1]:0.3101182183)[&type=1]:0.7872153939)[&type=1]:1.103073858)[&type=0]:0.4889276784)[&type=0]:0.4354935528,(((((9[&type=3]:0.1416000553)[&type=1]:0.9612053662,((12[&type=0]:0.0160933408)[&type=0]:0.4995183673)[&type=0]:0.5871937134)[&type=0]:0.5595680455,(((11[&type=1]:0.1244051404)[&type=0]:0.4221686028)[&type=0]:0.5657392679)[&type=0]:0.550060456)[&type=0]:0.4531051483)[&type=0]:0.4907869725)[&type=0]:0.5185631138)[&type=0]:0.3931733095)[&type=0]:0.4522656605)[&type=0]:0.5629860586):0.4667462698;";  // or null
    static String FILE = null;           // used if NEWICK is null
    static int TREE_INDEX = 1;           // index in the file, counting from 1
    static boolean INITIALISE = false;   // true: input is tip-typed or untyped, draw ancestral types and events
    static boolean LIKELIHOOD = true;   // true: also print the ADB log-likelihood before and after each move (N_MOVES = 0: only compute it)
    static String MOVE = "exchange";     // eventSampler | wilsonBalding | exchange | subtreeSlide | scale | jointEventSampler
    static int N_MOVES = 1;
    static long SEED = 1;                // first seed to try
    static int MAX_TRIES = 100;          // seeds to try per move if rejected


    static RealParameter lifetime = new RealParameter("0.5 1 2 3");

    static Parameterization model() {
        Parameterization model = new Parameterization();
        model.initByName("nTypes", 4,
                "lifetime", lifetime,
                "shape", new IntegerParameter("100 50 20 5"),
                "death", new RealParameter("0.05 0.1 0.1 0.1"),
                "symTransitions", new RealParameter("0.4 0.1 0 0 0 0.2 0 0 0 0 1 0 0 0 0 1"),
                "asymTransitions", new RealParameter("0 0.5 0 0 0 0 0.5 0.3 0 0 0 0 0 0 0 0"),
                "sampling", new RealParameter("0.1 0.1 0.1 0.1"),
                "originTime", 5.0);
        return model;
    }

    static AnnotatedTreeOperator operator(String move, AnnotatedTree tree, Parameterization model) {
        AnnotatedTreeOperator op;
        switch (move) {
            case "eventSampler":
                op = new EventSampler();
                op.initByName("tree", tree, "parameterization", model, "weight", 1.0,
                        "drawEventCount", true, "proportionBranches", 1.0);
                break;
            case "wilsonBalding":
                op = new AnnotatedWilsonBalding();
                op.initByName("tree", tree, "parameterization", model, "weight", 1.0);
                break;
            case "exchange":
                op = new AnnotatedExchange();
                op.initByName("tree", tree, "parameterization", model, "weight", 1.0, "isNarrow", false);
                break;
            case "subtreeSlide":
                op = new AnnotatedSubtreeSlide();
                op.initByName("tree", tree, "parameterization", model, "weight", 1.0, "size", 3);
                break;
            case "scale":
                op = new AnnotatedScale();
                op.initByName("tree", tree, "parameterization", model, "weight", 1.0, "scaleFactor", 0.75);
                break;
            case "jointEventSampler":
                op = new JointEventSampler();
                op.initByName("tree", tree, "parameterization", model, "weight", 1.0,
                        "up", lifetime, "proportionBranches", 1.0);
                break;
            default:
                throw new IllegalArgumentException("unknown move " + move);
        }
        return op;
    }

    static double logLikelihood(AnnotatedTree tree, Parameterization model) {
        ADBTreeDistribution distribution = new ADBTreeDistribution();
        distribution.initByName("tree", tree, "parameterization", model);
        return distribution.calculateTreeLogLikelihood(tree);
    }

    static void validate(AnnotatedTree tree, Parameterization model) {
        try {
            tree.validate(model);
            System.out.println("valid");
        } catch (IllegalStateException e) {
            System.out.println("INVALID: " + e.getMessage());
        }
    }

    public static void main(String[] args) {
        if (args.length > 0) {
            if (new File(args[0]).isFile()) { FILE = args[0]; NEWICK = null; }
            else NEWICK = args[0];
        }
        Parameterization model = model();

        AnnotatedTree tree = new AnnotatedTreeParser();
        if (NEWICK != null) tree.initByName("newick", NEWICK);
        else tree.initByName("fileName", FILE, "treeIndex", TREE_INDEX);

        if (INITIALISE) {
            AnnotatedTree initialised = new AnnotatedTreeInitialiser();
            initialised.initByName("branchingTree", tree, "parameterization", model);
            tree = initialised;
        }
        System.out.println(tree);
        validate(tree, model);

        if (LIKELIHOOD) System.out.println("log-likelihood = " + logLikelihood(tree, model));

        AnnotatedTreeOperator op = operator(MOVE, tree, model);
        long seed = SEED;
        for (int i = 0; i < N_MOVES; i++) {
            // on rejection (logHR = -Infinity) retry with the next seed
            AnnotatedTree original = tree.copy();
            double logHR = Double.NEGATIVE_INFINITY;
            for (int t = 0; t < MAX_TRIES && logHR == Double.NEGATIVE_INFINITY; t++, seed++) {
                Randomizer.setSeed(seed);
                logHR = op.proposal();
                if (logHR == Double.NEGATIVE_INFINITY) tree.assignFrom(original); // restore before retrying
            }
            if (logHR == Double.NEGATIVE_INFINITY) {
                System.out.println("\n" + MOVE + " rejected for all " + MAX_TRIES + " seeds");
                break;
            }
            System.out.println("\n" + MOVE + " logHR = " + logHR + " (seed " + (seed - 1) + ")");
            System.out.println(tree);
            validate(tree, model);
            if (LIKELIHOOD) System.out.println("log-likelihood = " + logLikelihood(tree, model));
        }
        if (MOVE.equals("jointEventSampler")) System.out.println("lifetime = " + lifetime);
    }
}
