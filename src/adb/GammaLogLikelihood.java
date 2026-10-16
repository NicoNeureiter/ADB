package adb;

import adb.util.Utils;
import beast.base.core.Log;
import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.distribution.GammaDistribution;
import org.apache.commons.math3.analysis.UnivariateFunction;

import java.util.Arrays;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.IntStream;

import static adb.util.Utils.TRANSFORM_FORWARD;


/*
Class for solving the equations for P0, P1 and B
and for calculating the probability of a tree based on its branching times given the ADB model parameters
 */
@Deprecated
public class GammaLogLikelihood {


    // Nested class to hold densities
    public static class Densities {
        public Complex[] pdfFFT;
        public double[] cdf;
        public double[] seq;
        public double dx;
        public Densities(Complex[] pdfFFT, double[] cdf, double[] seq, double dx) {
            this.pdfFFT = pdfFFT;
            this.cdf = cdf;
            this.seq = seq;
            this.dx = dx;
        }
    }

    // Main function for calculating the likelihood of the tree
    /* Inputs
    a: scale (theta), b: shape (k), d: death probability, rho: sampling probability, origin: age of the tree
    intS: start times of internal branches, intE: end times of internal branches, extE: end times of external branches (backwards in time)
    densities: pre-calculated (and FFT-transformed) pdf and cdf, together with underlying time array (0,origin) and step
    P0: pre-calculated array with extinction probabilities
    P1: pre-calculated array with probabilities of single sampled descendants
    Options for calculating branch probabilities -
    maxIt: maximum number of iterations, tol: error tolerance, m: step size (when solving integral equations with FFT),
    approx: approximate B?
     */
    public static double calcLogLikelihood(double a, Number b, double d, double rho,
                                           Densities densities, double[] P0, double[] P1,
                                           double[] intS, double[] intE, double[] extE,
                                           int maxIt, double tol, int m, boolean approx) {

        // check if extinction is certain
        if (P0[P0.length - 1] == 1.0) {
            return Double.NEGATIVE_INFINITY;
        }

        // extend P0 and P1 to calculate probabilities of tiny branches
        double[] seq = densities.seq;
        double[] extSeq = new double[seq.length + 1];
        double[] extP0 = new double[P0.length + 1];
        double[] extP1 = new double[P1.length + 1];
        extSeq[0] = 0;
        extP0[0] = 1 - rho;
        extP1[0] = rho;
        System.arraycopy(seq, 0, extSeq, 1, seq.length);
        System.arraycopy(P0, 0, extP0, 1, P0.length);
        System.arraycopy(P1, 0, extP1, 1, P1.length);

        // interpolate P1
        UnivariateFunction function = Utils.interpolator.interpolate(extSeq, extP1);

        // sum probabilities over external branches
        double logP1 = 0;
        for (int i = 0; i < extE.length; i++) {
            logP1 += Math.log(function.value(extE[i])); // if probability at some tip < 0 due to rounding errors, this will be -Infinity
        }
        if (logP1 == Double.NEGATIVE_INFINITY) {
            return Double.NEGATIVE_INFINITY;
        }

        // calculate probabilities of internal branches
        double[] B;
        if (approx) {
            B = approxB(a, b.intValue(), d, intS, intE, extSeq, extP0, tol);
        } else {
            B = calcB(a, b, d, intS, intE, extSeq, extP0, maxIt, tol, m);
        }

        // sum over all internal branches
        double logB = 0;
        for (int i = 0; i < B.length; i++) {
            logB += Math.log(B[i]); // if density for some branch < 0 due to rounding errors, this will be -Infinity
        }
        if (logB == Double.NEGATIVE_INFINITY) {
            return Double.NEGATIVE_INFINITY;
        }

        // sum (with conditioning on survival)
        return -Math.log(1 - P0[P0.length - 1]) + logP1 + logB;
    }


    // Function for calculating the transformed density functions
    public static Densities getDensities(double a, Number b, double origin, int m) {

        // initialize distribution
        GammaDistribution gammaDist = new GammaDistribution(b.doubleValue(), a);

        // generate linearly spaced values between 0 and origin
        double[] tSeq = new double[m];
        final double dx = origin / m;
        for (int i = 0; i < m; i++) {
            tSeq[i] = dx * (i + 1);
        }
        assert tSeq[m - 1] == origin;

        // calculate CDF and FFT from PDF from 0 to origin
        double[] pdf = new double[m];
        double[] cdf = new double[m];
        for (int i = 0; i < m; i++) {
            pdf[i] = Math.exp(gammaDist.logDensity(tSeq[i])); // gammaDist.density(tSeq[i]) - use log to prevent underflow
            cdf[i] = gammaDist.cumulativeProbability(tSeq[i]);
        }
        Complex[] pdfFFT = Utils.fft.transform(Utils.padZeros(pdf), TRANSFORM_FORWARD);

        return new Densities(pdfFFT, cdf, tSeq, dx);
    }


    // Function for calculating the extinction probability
    public static double[] calcP0(Complex[] pdfFFT, double[] cdf, double d, double rho, double dx, int maxIt, double tol) {

        // get length
        int n = cdf.length;

        // initialize
        double[] X0 = new double[n];
        for (int i = 0; i < n; i++) {
            X0[i] = (1 - rho) * (1 - cdf[i]) + d * cdf[i];
        }

        // set up iteration
        double err = 1;
        int it = 0;
        double[] X = X0;

        // iterate
        while (err > tol && it < maxIt) {

            // square
            double[] y = new double[n];
            for (int i = 0; i < n; i++) {
                y[i] = X[i] * X[i];
            }

            // partially convolve
            double[] I = Utils.convolveFFT(pdfFFT, y, n, dx);

            // sum
            double[] Xi = new double[n];
            for (int i = 0; i < n; i++) {
                Xi[i] = X0[i] + (1 - d) * I[i];
            }

            // add regularization: force values to be non-increasing or non-decreasing
            if ((1-rho) > d/(1-d)) {
                Utils.forceOrder(Xi, "non-increasing");
            } else if ((1-rho) < d/(1-d)) {
                Utils.forceOrder(Xi, "non-decreasing");
            }

            // compute error
            err = Utils.getError(Xi, X);

            // update
            X = Xi;
            it++;
        }

        // for asserting convergence
        if (it == maxIt) {
            // print error in debug mode
            Log.debug.println("calcP0: max iterations reached with error:  " + err);
        }

        return X;
    }


    // Function for calculating the probability of a single descendant
    public static double[] calcP1(Complex[] pdfFFT, double[] cdf, double[] P0, double d, double rho, double dx, int maxIt, double tol) {

        // get length
        int n = cdf.length;

        // initialize
        double[] X0 = new double[n];
        for (int i = 0; i < n; i++) {
            X0[i] = rho * (1 - cdf[i]);
        }

        // set up iteration
        double err = 1;
        int it = 0;
        double[] X = X0;

        // iterate
        while (err > tol && it < maxIt) {

            // multiply
            double[] y = new double[n];
            for (int i = 0; i < n; i++) {
                y[i] = P0[i] * X[i];
            }

            // partially convolve
            double[] I = Utils.convolveFFT(pdfFFT, y, n, dx);

            // sum
            double[] Xi = new double[n];
            for (int i = 0; i < n; i++) {
                Xi[i] = X0[i] + 2 * (1 - d) * I[i];
            }

            // compute error
            err = Utils.getError(X, Xi);

            // update
            X = Xi;
            it++;
        }

        if (it == maxIt) {
            Log.debug.println("calcP1: max iterations reached with error:  " + err);
        }

        return X;
    }


    // Function for calculating branch probabilities
    public static double[] calcB(double a, Number b, double d,
                                 double[] s, double[] e, double[] t0, double[] P0,
                                 int maxIt, double tol, int m) {

        // initialize distribution
        GammaDistribution gammaDist = new GammaDistribution(b.doubleValue(), a);

        // interpolate P0
        UnivariateFunction function = Utils.interpolator.interpolate(t0, P0);

        // get number of branches
        assert s.length == e.length; // for each branch, a start and end time must be given
        int n = s.length;

        double[] B = new double[n];
        // for each branch:
        // use Stream for parallelization
        IntStream.range(0, n)
                .parallel()
                .forEach(x -> {
                    double sx = s[x]; // start of the branch
                    double ex = e[x]; // end of the branch

                    // generate linearly spaced values between start and end
                    double[] tSeq = new double[m];
                    double[] age_seq = new double[m];
                    double dx = (ex - sx) / m;
                    for (int i = 0; i < m; i++) {
                        tSeq[i] = sx + dx * (i + 1);
                        age_seq[i] = tSeq[i] - sx;
                    }
                    assert age_seq[m - 1] == ex - sx;

                    // calculate the PDF of the gamma distribution and compute P0
                    double[] pdf = new double[m];
                    double[] P = new double[m];
                    for (int i = 0; i < m; i++) {
                        pdf[i] = Math.exp(gammaDist.logDensity(age_seq[i]));
                        P[i] = function.value(tSeq[i]);
                    }
                    Complex[] Ft = Utils.fft.transform(Utils.padZeros(pdf), TRANSFORM_FORWARD); // perform FFT

                    // initialize
                    double[] X0 = new double[m];
                    for (int i = 0; i < m; i++) {
                        X0[i] = (1 - d) * pdf[i];
                    }

                    // set up iteration
                    double err = 1;
                    int it = 0;
                    double[] X = X0;

                    // iterate
                    while (err > tol && it < maxIt) {

                        // multiply
                        double[] y = new double[m];
                        for (int i = 0; i < m; i++) {
                            y[i] = P[i] * X[i];
                        }

                        // partially convolve
                        double[] I = Utils.convolveFFT(Ft, y, m, dx);

                        // sum
                        double[] Xi = new double[m];
                        for (int i = 0; i < m; i++) {
                            Xi[i] = X0[i] + 2 * (1 - d) * I[i];
                        }

                        // compute error
                        err = Utils.getError(X, Xi);

                        // update
                        X = Xi;
                        it++;
                    }

                    if (it == maxIt) {
                        Log.debug.println("calcB: max iterations reached with error:  " + err);
                    }

                    // take last element
                    B[x] = X[m - 1];
                });

        return B;
    }


    // Function for approximating branch probabilities
    public static double[] approxB(double a, int b, double d,
                                   double[] s, double[] e, double[] t0, double[] P0,
                                   double tol) {

        // use cashing to avoid creating a new GammaDistribution object with shape i*b for i=1,2,... in each iteration
        // create a map to store the distributions for each i (once needed) and re-use them
        // use a thread-safe and dynamic map
        ConcurrentHashMap<Integer, GammaDistribution> gammaCache = new ConcurrentHashMap<>();

        // get number of branches
        assert s.length == e.length; // for each branch, a start and end time must be given
        int n = s.length;

        double[] B = new double[n];
        // for each branch:
        // use Stream for parallelization
        IntStream.range(0, n)
                .parallel()
                .forEach(x -> {
                    double sx = s[x]; // start of the branch
                    double ex = e[x]; // end of the branch

                    // get average P0 over branch
                    // use binary search to find closest indices to the branch lengths in t0 (t0 is sorted per definition!)
                    double[] P0Slice = Arrays.copyOfRange(P0, Utils.findClosestIndex(t0, sx), Utils.findClosestIndex(t0, ex) + 1);
                    double P0M = Utils.getMean(P0Slice);

                    // initialize the approximation
                    int k = (int) ((ex - sx) / (a * b)); // around this k, the term b_k will be maximal
                    double branchProb = 0;

                    int i = 0; // increase (start with 0 and treat k as minimum number of terms)
                    double term = 1;
                    while (term > tol || i <= k) { // only stops when both term <= tol and i > k!
                        GammaDistribution gammaDist;
                        // check if GammaDistribution with the given shape is already in the cache
                        if (gammaCache.containsKey(i + 1)) {
                            gammaDist = gammaCache.get(i + 1);
                        } else { // otherwise, add it
                            gammaDist = new GammaDistribution((i + 1) * b, a);
                            gammaCache.put(i + 1, gammaDist);
                        }
                        term = Math.pow(2, i) * Math.pow(1 - d, i + 1) * Math.pow(P0M, i) * Math.exp(gammaDist.logDensity(ex - sx)); // calculate b_i
                        branchProb += term;
                        i++;
                    }

                    B[x] = branchProb;
                });

        return B;
    }



    // Simpler function for calculating the log likelihood of a birth-death tree with sampling (Stadler, JTB 2010, DOI 10.1016/j.jtbi.2010.09.010)
    public static double calcBDLogLikelihood(double a, double d, double rho, double origin,
                                             double[] intS, double[] extE) {

        // get birth and death rate
        double lambda = (1 - d) / a;
        double mu = d / a;

        // get all bifurcation times
        double[] t = new double[intS.length + 1];
        System.arraycopy(intS, 0, t, 0, intS.length);
        t[t.length - 1] = origin;

        // get branch probabilities (no sampling through time)
        double c1 = Math.abs(lambda - mu);
        double c2 = -(lambda - mu - 2 * lambda * rho) / c1;
        double[] B = new double[t.length];
        IntStream.range(0, t.length)
                .parallel()
                .forEach(i -> {
                    B[i] = 2 * (1 - Math.pow(c2, 2)) +
                            Math.exp(-c1 * t[i]) * Math.pow(1 - c2, 2) +
                            Math.exp(c1 * t[i]) * Math.pow(1 + c2, 2);
                });

        // make log and sum
        double logB = 0;
        for (int i = 0; i < B.length; i++) {
            logB += Math.log(B[i]);
        }

        // condition on survival
        double p0 = 1 - (rho * (lambda - mu)) / (rho * lambda + (lambda * (1 - rho) - mu) * Math.exp(-(lambda - mu) * origin));
        if (p0 == 1.0) {
            return Double.NEGATIVE_INFINITY; // avoid division by 0 due to rounding errors
        }

        // get log likelihood
        int n = extE.length; // number of tips
        double logL = -Math.log(1 - p0) + (n - 1) * Math.log(lambda) + n * Math.log(4 * rho) - logB;

        return logL;
    }
}