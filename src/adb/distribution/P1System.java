package adb.distribution;

import adb.distribution.LifetimeDistributions.LifetimeDistribution;
import adb.util.Utils;


/**
 * Solves the system of integral equations for P1_ij(t), the probability that a lineage of type i
 * alive at time t before the present has exactly one sampled descendant, of type j.
 * Uses the solution of P0System and the same Picard iteration and FFT convolution on a uniform
 * time grid. The solution is cached and only recomputed when the parameterization changes.
 * Only used when computing phylodynamic likelihood on unannotated tree.
 */
public class P1System {

    Parameterization parameterization;
    int nTypes;
    LifetimeDistributions lifetimeDistributions;
    P0System P0System;

    boolean dirty;

    int maxIt;
    double tol;
    int nSteps;
    double[] timeArray;
    double timeStep;

    // stores P1 for each type
    double[][][] P1;
    double[][][] storedP1;


    public P1System(Parameterization parameterization, LifetimeDistributions lifetimeDistributions, P0System P0System,
                    int maxIt, double tol, double[] timeArray, double timeStep) {

        this.parameterization = parameterization;
        this.nTypes = parameterization.getNTypes();
        this.lifetimeDistributions = lifetimeDistributions;
        this.P0System = P0System;

        this.maxIt = maxIt;
        this.tol = tol;
        this.timeArray = timeArray;
        this.timeStep = timeStep;
        this.nSteps = timeArray.length;

        P1 = new double[nTypes][nTypes][nSteps];
        storedP1 = new double[nTypes][nTypes][nSteps];
        dirty = true;
        solveP1();
    }


    private void solveP1() {

        if (!dirty) return;

        LifetimeDistribution[] distributions = lifetimeDistributions.getLifetimeDistributions();
        double[] d = new double[nTypes];
        double[] rho = new double[nTypes];
        for (int i = 0; i < nTypes; i++) {
            d[i] = parameterization.getDeath(i);
            rho[i] = parameterization.getSampling(i);
        }

        double[][] P0 = P0System.getP0();

        // notation: it = iteration, w = integration variable (time), i,j,k = types
        // initialize matrix
        double[][][] X0 = new double[nTypes][nTypes][nSteps];
        for (int i = 0; i < nTypes; i++) {
            double[] cdf =  distributions[i].getCDF();
            for (int w = 0; w < nSteps; w++) {
                X0[i][i][w] = rho[i] * (1 - cdf[w]);
            }
        }

        // set up iteration
        double err = 1;
        int it = 0;
        double[][][] X = X0;

        // iterate
        while (err > tol && it < maxIt) {
            double[][][] Xn = new double[nTypes][nTypes][nSteps]; // next iteration

            for (int i = 0; i < nTypes; i++) {
                for (int j = 0; j < nTypes; j++) {
                    // get vectors for convolution
                    double[] y = new double[nSteps];
                    for (int w = 0; w < nSteps; w++) { // multiply elementwise on times
                        for (int k = 0; k < nTypes; k++) { // sum over all types k
                            y[w] += parameterization.getSymTransition(i, k) * P0[k][w] * X[k][j][w] +
                                    0.5 * parameterization.getAsymTransition(i, k) * (P0[i][w] * X[k][j][w] + P0[k][w] * X[i][j][w]);
                        }
                    }

                    // partially convolve
                    double[] I = Utils.convolveFFT(distributions[i].getTransformedPDF(), y, nSteps, timeStep);

                    // sum
                    for (int w = 0; w < nSteps; w++) {
                        Xn[i][j][w] = X0[i][j][w] + 2 * (1 - d[i]) * I[w];
                    }
                }
            }

            // compute error
            err = Utils.getError(X, Xn);

            // update
            X = Xn;
            it++;
        }

        if (it == maxIt) {
            System.err.printf("calculateP1 Warning: max iterations reached with error: %.2f%n", err);
        }

        // set final
        P1 = X;
        dirty = false;
    }


    public double[][][] getP1() {
        solveP1();
        return P1;
    }


    protected void isDirty() {
        if (parameterization.isDirtyCalculation()) {
            dirty = true;
        }
    }


    protected void store() {
        for (int i = 0; i < nTypes; i++) {
            for (int j = 0; j < nTypes; j++) {
                System.arraycopy(P1[i][j], 0, storedP1[i][j], 0, nSteps);
            }
        }
        dirty = false;
    }


    protected void restore() {
        double[][][] tmp;
        tmp = P1;
        P1 = storedP1;
        storedP1 = tmp;

        dirty = false;
    }

    protected void accept() {
        dirty = false;
    }
}
