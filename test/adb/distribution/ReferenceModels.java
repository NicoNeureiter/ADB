package adb.distribution;

import adb.util.Utils;
import beast.base.inference.parameter.IntegerParameter;
import beast.base.inference.parameter.RealParameter;

import java.util.Arrays;
import java.util.stream.Collectors;


/**
 * Models and independent reference solutions for the tests of the likelihood.
 *
 * The reference for P0 uses that a gamma lifetime with an integer shape k is a sum of k exponential stages, each with
 * rate k/lifetime. Then P0 is the solution of a system of ordinary differential equations on the states (type, stage),
 * which is integrated here with the classical Runge-Kutta method. This shares no code with the Picard iteration with
 * FFT in P0System.
 */
final class ReferenceModels {

    private ReferenceModels() {}


    /** Parameters of a model, a matrix is given row by row. */
    static Parameterization model(double[] lifetime, int[] shape, double[] death, double[] sampling,
                                  double[][] sym, double[][] asym, double originTime) {
        int n = lifetime.length;
        return model(n, vector(lifetime), shape(shape), vector(death), vector(sampling), matrix(sym), matrix(asym), originTime);
    }

    /** Single type with a symmetric division only. */
    static Parameterization singleType(double lifetime, int shape, double death, double sampling, double originTime) {
        return model(new double[]{lifetime}, new int[]{shape}, new double[]{death}, new double[]{sampling},
                new double[][]{{1.0}}, new double[][]{{0.0}}, originTime);
    }

    /**
     * Two types. Type 0 divides symmetrically into 0 and 0 with probability 0.2 and asymmetrically into 0 and 1 with
     * probability 0.8, type 1 divides symmetrically with probability 0.6 and asymmetrically into 1 and 0 with 0.4.
     */
    static Parameterization twoTypes(int shape0, int shape1, double originTime) {
        return model(new double[]{2, 5}, new int[]{shape0, shape1}, new double[]{0.1, 0.2}, new double[]{0.5, 0.5},
                new double[][]{{0.2, 0}, {0, 0.6}}, new double[][]{{0, 0.8}, {0.4, 0}}, originTime);
    }

    /** Three types, in which type 0 gives rise to 1 and 2, type 1 gives rise to 2 and type 2 only divides symmetrically. */
    static Parameterization threeTypes(int shape0, int shape1, int shape2, double originTime) {
        return model(new double[]{1, 2, 3}, new int[]{shape0, shape1, shape2}, new double[]{0.05, 0.1, 0.1},
                new double[]{0.3, 0.3, 0.3},
                new double[][]{{0.5, 0, 0}, {0, 0.6, 0}, {0, 0, 1}},
                new double[][]{{0, 0.3, 0.2}, {0, 0, 0.4}, {0, 0, 0}}, originTime);
    }

    private static Parameterization model(int n, String lifetime, String shape, String death, String sampling,
                                          String sym, String asym, double originTime) {
        Parameterization model = new Parameterization();
        model.initByName("nTypes", n,
                "lifetime", new RealParameter(lifetime),
                "shape", new IntegerParameter(shape),
                "death", new RealParameter(death),
                "symTransitions", new RealParameter(sym),
                "asymTransitions", new RealParameter(asym),
                "sampling", new RealParameter(sampling),
                "originTime", originTime,
                "originType", 0);
        return model;
    }

    private static String vector(double[] x) {
        return Arrays.stream(x).mapToObj(Double::toString).collect(Collectors.joining(" "));
    }

    private static String shape(int[] x) {
        return Arrays.stream(x).mapToObj(Integer::toString).collect(Collectors.joining(" "));
    }

    private static String matrix(double[][] x) {
        return Arrays.stream(x).map(ReferenceModels::vector).collect(Collectors.joining(" "));
    }


    /** The time points of a grid with nSteps steps up to the origin time. */
    static double[] timeArray(Parameterization model, int nSteps) {
        return Utils.linSpace(0, model.getOriginTime(), nSteps);
    }

    /** P0 of the model on the time grid, from P0System with the given settings. */
    static double[][] solveP0(Parameterization model, int nSteps, int maxIterations, double tolerance) {
        double[] timeArray = timeArray(model, nSteps);
        double timeStep = model.getOriginTime() / nSteps;
        LifetimeDistributions lifetimeDistributions = new LifetimeDistributions(
                model.getLifetimeParameter(), model.getShapeParameter(), timeArray);
        P0System system = new P0System(model, lifetimeDistributions, maxIterations, tolerance, timeArray, timeStep);
        return system.getP0();
    }

    /** P0 of the model on the time grid, from P0System with default settings. */
    static double[][] solveP0(Parameterization model, int nSteps) {
        return solveP0(model, nSteps, 100, 1e-12);
    }


    /**
     * P0 at the time points of the grid with nSteps steps up to the origin time, from the differential equations of
     * the stages. Q(i,m) is the probability of no sampled descendants of a lineage of type i in stage m (m = 0 is the
     * first stage, and P0 of type i is Q(i,0)). With the stage rate r = shape/lifetime:
     * Q(i,m)' = r (Q(i,m+1) - Q(i,m)), and for the last stage Q(i,last)' = r (G_i - Q(i,last)), where
     * G_i = d_i + (1 - d_i) sum_j (sym(i,j) P0_j^2 + asym(i,j) P0_i P0_j) is the probability of no sampled descendants
     * after the end of the lifetime. At time 0, Q(i,m) = 1 - rho_i.
     */
    static double[][] referenceP0(Parameterization model, int nSteps) {
        int n = model.getNTypes();
        int[] offset = new int[n + 1];
        for (int i = 0; i < n; i++) {
            offset[i + 1] = offset[i] + (int) model.getShape(i);
        }
        double[] y = new double[offset[n]];
        for (int i = 0; i < n; i++) {
            Arrays.fill(y, offset[i], offset[i + 1], 1 - model.getSampling(i));
        }

        double dx = model.getOriginTime() / nSteps;
        int substeps = 8;
        double h = dx / substeps;
        double[][] P0 = new double[n][nSteps];
        for (int w = 0; w < nSteps; w++) {
            for (int s = 0; s < substeps; s++) {
                y = rungeKutta(model, offset, y, h);
            }
            for (int i = 0; i < n; i++) {
                P0[i][w] = y[offset[i]];
            }
        }
        return P0;
    }

    private static double[] rungeKutta(Parameterization model, int[] offset, double[] y, double h) {
        double[] k1 = derivative(model, offset, y);
        double[] k2 = derivative(model, offset, add(y, k1, h / 2));
        double[] k3 = derivative(model, offset, add(y, k2, h / 2));
        double[] k4 = derivative(model, offset, add(y, k3, h));
        double[] next = new double[y.length];
        for (int i = 0; i < y.length; i++) {
            next[i] = y[i] + h / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
        }
        return next;
    }

    private static double[] add(double[] y, double[] k, double factor) {
        double[] sum = new double[y.length];
        for (int i = 0; i < y.length; i++) {
            sum[i] = y[i] + factor * k[i];
        }
        return sum;
    }

    private static double[] derivative(Parameterization model, int[] offset, double[] y) {
        int n = model.getNTypes();
        double[] dy = new double[y.length];
        for (int i = 0; i < n; i++) {
            double sum = 0;
            for (int j = 0; j < n; j++) {
                sum += model.getSymTransition(i, j) * y[offset[j]] * y[offset[j]] +
                        model.getAsymTransition(i, j) * y[offset[i]] * y[offset[j]];
            }
            double afterLifetime = model.getDeath(i) + (1 - model.getDeath(i)) * sum;

            double rate = model.getShape(i) / model.getLifetime(i);
            int last = offset[i + 1] - 1;
            for (int m = offset[i]; m < last; m++) {
                dy[m] = rate * (y[m + 1] - y[m]);
            }
            dy[last] = rate * (afterLifetime - y[last]);
        }
        return dy;
    }


    /**
     * Closed form of P0 for one type with shape 1 (a birth-death process with sampling at the present), for a death
     * probability that is not 0.5. With the rate m = 1/lifetime, the equation P0' = m (d + (1 - d) P0^2 - P0) has the
     * roots 1 and r = d / (1 - d), and the solution (P0 - 1) / (P0 - r) = C exp(m (1 - 2d) t).
     */
    static double[] closedFormP0(Parameterization model, int nSteps) {
        double d = model.getDeath(0);
        double r = d / (1 - d);
        double rate = 1 / model.getLifetime(0);
        double start = 1 - model.getSampling(0);
        double c = (start - 1) / (start - r);

        double[] timeArray = timeArray(model, nSteps);
        double[] P0 = new double[nSteps];
        for (int w = 0; w < nSteps; w++) {
            double z = c * Math.exp(rate * (1 - 2 * d) * timeArray[w]);
            P0[w] = (1 - r * z) / (1 - z);
        }
        return P0;
    }

    static double maxAbsDifference(double[] a, double[] b) {
        double max = 0;
        for (int i = 0; i < a.length; i++) {
            max = Math.max(max, Math.abs(a[i] - b[i]));
        }
        return max;
    }

    static double maxAbsDifference(double[][] a, double[][] b) {
        double max = 0;
        for (int i = 0; i < a.length; i++) {
            max = Math.max(max, maxAbsDifference(a[i], b[i]));
        }
        return max;
    }
}
