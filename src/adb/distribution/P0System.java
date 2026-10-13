package adb.distribution;

import adb.distribution.LifetimeDistributions.*;
import adb.util.Utils;

import java.util.stream.IntStream;


/**
 * Solves the system of integral equations for P0_i(t), the probability that a lineage of type i
 * alive at time t before the present has no sampled descendants.
 * The equations are solved by Picard iteration on a uniform time grid, using FFT convolution
 * with the lifetime densities.
 * The solution is cached and only recomputed when the parameterization changes.
 */
public class P0System {

    Parameterization parameterization;
    int nTypes;
    LifetimeDistributions lifetimeDistributions;

    boolean dirty;

    int maxIt;
    double tol;
    int nSteps;
    double[] timeArray;
    double timeStep;

    // stores P0 for each type
    double[][] P0;
    double[][] storedP0;


    public P0System(Parameterization parameterization, LifetimeDistributions lifetimeDistributions,
                    int maxIt, double tol, double[] timeArray, double timeStep) {
        this.parameterization = parameterization;
        this.nTypes = parameterization.getNTypes();
        this.lifetimeDistributions = lifetimeDistributions;

        this.maxIt = maxIt;
        this.tol = tol;
        this.timeArray = timeArray;
        this.timeStep = timeStep;
        this.nSteps = timeArray.length;

        P0 = new double[nTypes][nSteps];
        storedP0 = new double[nTypes][nSteps];
        dirty = true;
        solveP0();
    }


    private void solveP0() {

        if (!dirty) return;

        LifetimeDistribution[] distributions = lifetimeDistributions.getLifetimeDistributions();
        double[] d = new double[nTypes];
        double[] rho = new double[nTypes];
        for (int i = 0; i < nTypes; i++) {
            d[i] = parameterization.getDeath(i);
            rho[i] = parameterization.getSampling(i);
        }

        // notation: it = iteration, w = integration variable (time), i,j,k = types
        // initialize matrix
        double[][] X0 = new double[nTypes][nSteps];
        for (int i = 0; i < nTypes; i++) {
            double[] cdf = distributions[i].getCDF();
            for (int w = 0; w < nSteps; w++) {
                X0[i][w] = (1 - rho[i]) * (1 - cdf[w]) + d[i] * cdf[w];
            }
        }

        // set up iteration
        double err = 1;
        int it = 0;
        double[][] X = X0;

        // iterate
        while (err > tol && it < maxIt) {
            double[][] Xn = new double[nTypes][nSteps]; // next iteration

            if (nTypes == 1) {
                int i = 0; // single type
                // get vectors for convolution
                double[] y = new double[nSteps];
                for (int w = 0; w < nSteps; w++) { // multiply elementwise on times
                    y[w] = X[i][w] * X[i][w];
                }
                // partially convolve
                double[] I = Utils.convolveFFT(distributions[i].getTransformedPDF(), y, nSteps, timeStep);
                // sum
                for (int w = 0; w < nSteps; w++) {
                    Xn[i][w] = X0[i][w] + (1 - d[i]) * I[w];
                }

                // add regularization (enforce monotonicity, cf. Theorem 2 in Jones, 2011)
                if (d[i] < 0.5 && (1-rho[i]) > d[i]/(1-d[i])) {
                    Utils.enforceMonotonicity(Xn[i], -1);
                } else {
                    Utils.enforceMonotonicity(Xn[i], 1);
                }
            }

            else {
                final double[][] Xc = X; // current X, read-only
                IntStream.range(0, nTypes).parallel().forEach(i -> { // parallelize expensive computation
                    // get vectors for convolution
                    double[] y = new double[nSteps];
                    for (int w = 0; w < nSteps; w++) { // multiply elementwise on times
                        for (int j = 0; j < nTypes; j++) { // sum over all types k
                            y[w] += parameterization.getSymTransition(i,j) * Xc[j][w] * Xc[j][w] +
                                    parameterization.getAsymTransition(i,j) * Xc[i][w] * Xc[j][w];
                        }
                    }
                    // partially convolve
                    double[] I = Utils.convolveFFT(distributions[i].getTransformedPDF(), y, nSteps, timeStep);
                    // sum
                    for (int w = 0; w < nSteps; w++) {
                        Xn[i][w] = X0[i][w] + (1 - d[i]) * I[w];
                        // keep bounds
                        if (Xn[i][w] < 0) {
                            Xn[i][w] = 0;
                        }
                        if (Xn[i][w] > 1) {
                            Xn[i][w] = 1;
                        }
                    }
                });
            }

            // compute error
            err = Utils.getError(X, Xn);

            // update
            X = Xn;
            it++;
        }

        if (it == maxIt) {
            System.err.printf("calculateP0 Warning: max iterations reached with error: %.2f%n", err);
        }

        // set final
        P0 = X;
        dirty = false;
    }


    public double[][] getP0() {
        solveP0();
        return P0;
    }


    protected void isDirty() {
        if (parameterization.isDirtyCalculation()) {
            dirty = true;
        }
    }


    protected void store() {
        for (int i = 0; i < nTypes; i++) {
            System.arraycopy(P0[i], 0, storedP0[i], 0, nSteps);
        }
        dirty = false;
    }


    protected void restore() {
        double[][] tmp;
        tmp = P0;
        P0 = storedP0;
        storedP0 = tmp;

        dirty = false;
    }


    protected void accept() {
        dirty = false;
    }
}
