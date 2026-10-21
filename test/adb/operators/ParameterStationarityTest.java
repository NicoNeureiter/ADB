package adb.operators;

import adb.distribution.Parameterization;
import adb.tree.AnnotatedTree;
import beast.base.inference.Operator;
import beast.base.inference.parameter.RealParameter;
import beast.base.util.Randomizer;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.function.Function;

import static adb.operators.StationarityTest.*;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Stationarity test of operators that change parameters, alone or together with the tree and its hidden events.
 * The Hastings ratio is a property of the proposal alone, so any target with exact draws can test it. Here the target
 * is the ADB density of the tree under the fixed parameters of the simulation, times independent log-normal priors on
 * two moving parameters: the lifetime of the model that the operator uses (for example to redraw the events), and a
 * parameter that only has its prior. Exact draws are the trees of test_data (see ExactTreeSamples) together with
 * parameters drawn from the priors. A few Metropolis-Hastings steps must leave this distribution unchanged, so the
 * paired differences of the tree statistics and of the parameters must have mean 0 (as in StationarityTest).
 * Operators that keep the topology and the node heights leave most tree statistics unchanged; the log target density
 * and the squared log distances of the parameters from the prior centres see a wrong Hastings ratio, which moves the
 * chain towards less likely states. That the operator's model has a different lifetime than the target lowers the
 * acceptance but does not change the test. A control with a wrong Hastings ratio must fail. Tagged slow.
 */
@Tag("slow")
public class ParameterStationarityTest {

    /** More steps than in StationarityTest, since the acceptance is lower (about 10% for UpDownEventSampler). */
    static final int STEPS = 40;

    /** Standard deviation of the log-normal priors, centred on the simulation lifetime and on 1. */
    static final double PRIOR_SD = 0.2;

    /**
     * What an operator under test can work on: the tree of the samples, the model of the operator (the target model
     * with its own lifetime parameter), and the two moving parameters, each with one entry per type.
     */
    record MovingParameters(ExactTreeSamples samples, Parameterization model, RealParameter lifetime, RealParameter other) { }

    /** Runs STEPS MH steps from every tree of the configuration, with the parameters drawn from the priors. */
    static Result stationarity(String config, String operatorName, Function<MovingParameters, Operator> create) {
        Randomizer.setSeed(12345);
        ExactTreeSamples samples = new ExactTreeSamples(config);
        Parameterization target = samples.model;
        int nTypes = target.getNTypes();

        RealParameter lifetime = new RealParameter(Collections.nCopies(nTypes, 1.0).toArray(new Double[0]));
        lifetime.setBounds(0.0, Double.POSITIVE_INFINITY);
        RealParameter other = new RealParameter(Collections.nCopies(nTypes, 1.0).toArray(new Double[0]));
        other.setBounds(0.0, Double.POSITIVE_INFINITY);
        Parameterization model = new Parameterization();
        model.initByName("nTypes", nTypes, "lifetime", lifetime,
                "shape", target.shapeParameterInput.get(), "death", target.deathParameterInput.get(),
                "symTransitions", target.symTransitionsParameterInput.get(),
                "asymTransitions", target.asymTransitionsParameterInput.get(),
                "sampling", target.samplingParameterInput.get(), "originTime", target.getOriginTime());
        Operator operator = create.apply(new MovingParameters(samples, model, lifetime, other));

        RealParameter[] parameters = {lifetime, other};
        double[] centres = {target.getLifetime(0), 1.0};
        List<String> names = new ArrayList<>(statisticNames(nTypes));
        for (String parameter : new String[]{"lifetime", "other"}) {
            for (int j = 0; j < nTypes; j++) {
                names.add(parameter + " " + j);
                names.add("log^2 " + parameter + " " + j);
            }
        }
        names.add("logTarget");
        int nStats = names.size(), nTrees = samples.size();
        double[] sum = new double[nStats], sumSquares = new double[nStats];

        int proposals = 0, accepted = 0;
        for (int k = 0; k < nTrees; k++) {
            samples.load(k);
            for (int p = 0; p < parameters.length; p++) {
                for (int j = 0; j < nTypes; j++) {
                    parameters[p].setValue(j, centres[p] * Math.exp(PRIOR_SD * Randomizer.nextGaussian()));
                }
            }
            double logP = samples.logP() + logPrior(parameters, centres);
            double[] before = statistics(samples.tree, parameters, centres, logP);
            for (int step = 0; step < STEPS; step++) {
                AnnotatedTree backup = new AnnotatedTree();
                backup.assignFrom(samples.tree);
                Double[][] oldValues = {lifetime.getValues(), other.getValues()};

                proposals++;
                double logHR = operator.proposal();
                double newLogP = logHR == Double.NEGATIVE_INFINITY ? Double.NEGATIVE_INFINITY
                        : samples.density.calculateLogP() + logPrior(parameters, centres);
                if (Math.log(Randomizer.nextDouble()) < newLogP - logP + logHR) {
                    accepted++;
                    logP = newLogP;
                } else {
                    samples.tree.assignFrom(backup);
                    for (int p = 0; p < parameters.length; p++) {
                        for (int j = 0; j < nTypes; j++) parameters[p].setValue(j, oldValues[p][j]);
                    }
                }
            }
            double[] after = statistics(samples.tree, parameters, centres, logP);
            for (int s = 0; s < nStats; s++) {
                double diff = after[s] - before[s];
                sum[s] += diff;
                sumSquares[s] += diff * diff;
            }
        }
        System.out.println();
        return report(config, operatorName, names, sum, sumSquares, nTrees, STEPS, (double) accepted / proposals);
    }

    // log density of the independent log-normal priors, up to a constant
    private static double logPrior(RealParameter[] parameters, double[] centres) {
        double logP = 0;
        for (int p = 0; p < parameters.length; p++) {
            for (int j = 0; j < parameters[p].getDimension(); j++) {
                double z = Math.log(parameters[p].getValue(j) / centres[p]) / PRIOR_SD;
                logP += -0.5 * z * z - Math.log(parameters[p].getValue(j));
            }
        }
        return logP;
    }

    // the tree statistics of StationarityTest, every parameter value with its squared log distance from the centre,
    // and the log target density
    private static double[] statistics(AnnotatedTree tree, RealParameter[] parameters, double[] centres, double logTarget) {
        List<Double> values = new ArrayList<>();
        int nTypes = parameters[0].getDimension();
        for (double value : StationarityTest.statistics(tree, nTypes)) values.add(value);
        for (int p = 0; p < parameters.length; p++) {
            for (int j = 0; j < nTypes; j++) {
                double value = parameters[p].getValue(j);
                values.add(value);
                values.add(Math.pow(Math.log(value / centres[p]), 2));
            }
        }
        values.add(logTarget);
        return values.stream().mapToDouble(Double::doubleValue).toArray();
    }


    static Operator upDownEventSampler(MovingParameters m, boolean elementWise) {
        UpDownEventSampler operator = new UpDownEventSampler();
        operator.initByName("tree", m.samples().tree, "parameterization", m.model(), "up", m.lifetime(),
                "down", m.other(), "elementWise", elementWise, "scaleFactor", 0.2, "proportionBranches", 0.5,
                "optimise", false, "weight", 1.0);
        return operator;
    }


    // the tree and the lifetime go up together (time-scale ridge), the other parameter inversely
    static Operator scale(MovingParameters m, boolean drawEvents) {
        AnnotatedScale operator = new AnnotatedScale();
        operator.initByName("tree", m.samples().tree, "parameterization", m.model(), "parameter", m.lifetime(),
                "parameterInverse", m.other(), "drawEvents", drawEvents, "scaleFactor", 0.1, "optimise", false,
                "weight", 1.0);
        return operator;
    }


    // the other parameter is the shape of the operator's model, which it scales; the target density keeps the shape
    // of the simulation
    static ShapeEventScaler shapeEventScaler(MovingParameters m) {
        ShapeEventScaler operator = new ShapeEventScaler();
        operator.initByName("tree", m.samples().tree, "parameterization", AnnotatedTreeOperatorTest.withShape(m.model(), m.other()),
                "scaleFactor", 0.1, "optimise", false, "weight", 1.0);
        return operator;
    }

    // control: ShapeEventScaler without the Jacobian M log c of the event map, i.e. only the log f of the scaling
    static Operator shapeEventScalerWithoutJacobian(MovingParameters m) {
        ShapeEventScaler operator = shapeEventScaler(m);
        RealParameter shape = m.other();
        return new Operator() {
            @Override
            public void initAndValidate() { }

            @Override
            public double proposal() {
                Double[] oldShape = shape.getValues();
                double logHR = operator.proposal();
                if (logHR == Double.NEGATIVE_INFINITY) return logHR;
                for (int j = 0; j < shape.getDimension(); j++) {
                    if (shape.getValue(j).doubleValue() != oldShape[j]) {
                        double logc = -0.5 * Math.log(shape.getValue(j) / oldShape[j]);
                        return logHR - AnnotatedTreeOperatorTest.movableEvents(m.samples().tree, m.model(), j) * logc;
                    }
                }
                return logHR;
            }
        };
    }


    @Test
    public void testControlBiasedShapeEventScalerFails() {
        Result result = stationarity(ONE, "ShapeEventScaler with Hastings ratio + log 2",
                m -> biased(shapeEventScaler(m)));
        assertTrue(drifts(result), "a Hastings ratio that is off by log 2 must be detected");
    }

    @Test
    public void testControlShapeEventScalerWithoutJacobianFails() {
        Result result = stationarity(ONE, "ShapeEventScaler without the Jacobian of the events",
                ParameterStationarityTest::shapeEventScalerWithoutJacobian);
        assertTrue(drifts(result), "a missing Jacobian must be detected");
    }

    @Test
    public void testShapeEventScalerOneType() {
        assertStationary(stationarity(ONE, "ShapeEventScaler", ParameterStationarityTest::shapeEventScaler));
    }

    @Test
    public void testShapeEventScalerTwoTypes() {
        assertStationary(stationarity(TWO, "ShapeEventScaler", ParameterStationarityTest::shapeEventScaler));
    }


    @Test
    public void testControlBiasedUpDownEventSamplerFails() {
        Result result = stationarity(ONE, "UpDownEventSampler with Hastings ratio + log 2",
                m -> biased(upDownEventSampler(m, false)));
        assertTrue(drifts(result), "a Hastings ratio that is off by log 2 must be detected");
    }

    @Test
    public void testUpDownEventSamplerOneType() {
        assertStationary(stationarity(ONE, "UpDownEventSampler", m -> upDownEventSampler(m, false)));
    }

    @Test
    public void testUpDownEventSamplerTwoTypes() {
        assertStationary(stationarity(TWO, "UpDownEventSampler", m -> upDownEventSampler(m, false)));
    }

    @Test
    public void testUpDownEventSamplerElementWiseTwoTypes() {
        assertStationary(stationarity(TWO, "UpDownEventSampler, elementWise", m -> upDownEventSampler(m, true)));
    }

    @Test
    public void testScaleOneType() {
        assertStationary(stationarity(ONE, "scale with lifetime", m -> scale(m, false)));
    }

    @Test
    public void testScaleTwoTypes() {
        assertStationary(stationarity(TWO, "scale with lifetime", m -> scale(m, false)));
    }

    @Test
    public void testScaleDrawEventsOneType() {
        assertStationary(stationarity(ONE, "scale with lifetime, drawEvents", m -> scale(m, true)));
    }

    @Test
    public void testScaleDrawEventsTwoTypes() {
        assertStationary(stationarity(TWO, "scale with lifetime, drawEvents", m -> scale(m, true)));
    }
}
