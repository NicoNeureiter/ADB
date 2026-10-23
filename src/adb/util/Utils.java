package adb.util;

import beast.base.util.Randomizer;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.special.Gamma;
import org.apache.commons.math3.analysis.interpolation.LinearInterpolator;
import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.transform.DftNormalization;
import org.apache.commons.math3.transform.FastFourierTransformer;
import org.apache.commons.math3.transform.TransformType;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.List;

/**
 * Static helpers for the likelihood and the operators: FFT convolution on the uniform time grid, array
 * utilities, and sampling of event counts and positions. A grid of n steps on (0, T] has spacing T/n and
 * points T/n, 2T/n, ..., T. The FFT length is 2n, so n must be a power of two.
 */
public class Utils {

    /** FFT with one transformer per thread (safe for parallelization). */
    public static ThreadLocal<FastFourierTransformer> FFT =
            ThreadLocal.withInitial(() -> new FastFourierTransformer(DftNormalization.STANDARD));

    /** FFT shared by all threads. */
    public static FastFourierTransformer fft = new FastFourierTransformer(DftNormalization.STANDARD);

    /** Linear interpolator. */
    public static LinearInterpolator interpolator = new LinearInterpolator();

    public static TransformType TRANSFORM_FORWARD = TransformType.FORWARD;
    public static TransformType TRANSFORM_INVERSE = TransformType.INVERSE;


    /** True if x is a positive power of 2. */
    // https://stackoverflow.com/questions/600293/how-to-check-if-a-number-is-a-power-of-2
    public static boolean isPowerOfTwo(int x) {
        return (x > 0) && ((x & (x - 1)) == 0);
    }


    /** Copy of x padded with zeros to twice its length, as needed for a linear convolution by FFT. */
    public static double[] padZeros(double[] x) {
        int n = x.length;
        double[] xp = new double[n * 2];
        System.arraycopy(x, 0, xp, 0, n);
        return xp;
    }


    /** The n equally spaced points after start up to and including end. */
    public static double[] linSpace(double start, double end, int n) {
        double dx = (end - start) / n;
        double[] seq = new double[n];
        for (int i = 0; i < n; i++) {
            seq[i] = start + dx * (i + 1);
        }
        return seq;
    }


    /**
     * Partial convolution using FFT:
     * First n values of the convolution of two functions on the grid, times the grid spacing eps.
     * The first function is given as its FFT fx, padded to length 2n. The second is y, with n values.
     */
    public static double[] convolveFFT(Complex[] fx, double[] y, int n, double eps) {

        FastFourierTransformer fft = FFT.get();

        // perform FFT on padded y
        Complex[] fy = fft.transform(padZeros(y), TRANSFORM_FORWARD);

        // element-wise multiplication of fx and fy (convolution in Fourier space)
        Complex[] fz = new Complex[fx.length];
        for (int i = 0; i < fx.length; i++) {
            fz[i] = fx[i].multiply(fy[i]);
        }

        // perform inverse FFT to get the result back in time domain
        Complex[] z = fft.transform(fz, TRANSFORM_INVERSE);

        // extract the real part and scale it by eps
        double[] z_real = new double[n];
        for (int i = 0; i < n; i++) {
            z_real[i] = z[i].getReal() * eps;
        }

        return z_real;
    }


    /** Arithmetic mean of an array. */
    public static double getMean(double[] array) {
        double sum = 0;
        for (double num : array) {
            sum += num;
        }
        return sum / array.length;
    }

    /** Superseded by enforceMonotonicity, still used by GammaLogLikelihood. */
    @Deprecated
    public static void forceOrder(double[] array, String order) {
        if (order.equals("non-increasing")) {
            for (int i = 1; i < array.length; i++) {
                if (array[i] > array[i - 1]) {
                    array[i] = array[i - 1];  // adjust to maintain non-increasing order
                }
            }
        } else if (order.equals("non-decreasing")) {
            for (int i = 1; i < array.length; i++) {
                if (array[i] < array[i - 1]) {
                    array[i] = array[i - 1];  // adjust to maintain non-decreasing order
                }
            }
        } else {
            throw new IllegalArgumentException("Invalid order: use non-increasing or non-decreasing");
        }
    }

    /** Makes the array monotone in place by clamping each value to its predecessor. Order is -1 for non-increasing, 1 for non-decreasing. */
    public static void enforceMonotonicity(double[] array, int order) {
        if (order == -1) {
            for (int i = 1; i < array.length; i++) {
                if (array[i] > array[i - 1]) {
                    array[i] = array[i - 1]; // adjust to maintain non-increasing order
                }
            }
        } else if (order == 1) {
            for (int i = 1; i < array.length; i++) {
                if (array[i] < array[i - 1]) {
                    array[i] = array[i - 1]; // adjust to maintain non-decreasing order
                }
            }
        } else {
            throw new IllegalArgumentException("Invalid order: use -1 for non-increasing or 1 for non-decreasing");
        }
    }


    /** Index of the entry closest to value in an ascending array, by binary search. Ties go to the lower index. */
    // https://stackoverflow.com/questions/30245166/find-the-nearest-closest-value-in-a-sorted-list
    // complexity log(n) instead of n in a loop
    public static int findClosestIndex(double[] array, double value) {
        // if value is at boundaries
        if (value <= array[0]) {
            return 0;
        }
        if (value >= array[array.length - 1]) {
            return array.length - 1;
        }

        // do binary search
        int index = Arrays.binarySearch(array, value);

        if (index >= 0) { // exact match found
            return index;
        } else { // no exact match: binarySearch returns (-(insertion point) - 1)
            int insertionPoint = -(index + 1);

            // return the index of the closest value
            if ((value - array[insertionPoint - 1]) <= (array[insertionPoint] - value)) { // value is closer to the previous value
                return insertionPoint - 1;
            } else { // value is closer to the next value
                return insertionPoint;
            }
        }
    }


    /** Euclidean distance between two arrays of the same shape (element-wise). */
    public static double getError(double[] x, double[] y) {
        int n = x.length;
        double sum = 0;
        for (int i = 0; i < n; i++) {
            double diff = x[i] - y[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }

    public static double getError(double[][] X, double[][] Y) {
        int n = X.length; // number of rows
        int m = X[0].length;  // number of columns
        double sum = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                double diff = X[i][j] - Y[i][j];
                sum += diff * diff;
            }
        }
        return Math.sqrt(sum);
    }

    public static double getError(double[][][] X, double[][][] Y){
        int n = X.length; // number of arrays
        int m = X[0].length;  // number of rows
        int o = X[0][0].length;  // number of columns
        double sum = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                for (int k = 0; k < o; k++) {
                    double diff = X[i][j][k] - Y[i][j][k];
                    sum += diff * diff;
                }
            }
        }
        return Math.sqrt(sum);
    }

    /** Redistributes the sum of x over its entries in place, with proportions from a symmetric Dirichlet(alpha). */
    // https://stats.stackexchange.com/questions/69210/drawing-from-dirichlet-distribution
    public static void distributeDirichlet(double[] x, double alpha){
        int n = x.length;
        double sumX = 0;
        for (int i = 0; i < n; i++) {
            sumX += x[i];
        }

        // draw n independent random samples from Gamma distribution
        double[] y = new double[n];
        double sumY = 0;
        for (int i = 0; i < n; i++) {
            y[i] = Randomizer.nextGamma(alpha, 1);
            sumY += y[i];
        }

        // divide by the sum to obtain a sample from the Dirichlet distribution
        // and multiply the original value in array with the sampled factor
        double[] p = new double[n];
        for (int i = 0; i < n; i++) {
            p[i] = y[i] / sumY;
            x[i] = p[i] * sumX;
        }
    }


    /** As distributeDirichlet, and returns the log density of the new x: the Dirichlet density of the proportions, scaled by the sum. */
    public static double distributeDirichletProbability(double[] x, double alpha){
        int n = x.length;
        double sumX = 0;
        for (int i = 0; i < n; i++) {
            sumX += x[i];
        }

        // draw n independent random samples from Gamma distribution
        double[] y = new double[n];
        double sumY = 0;
        for (int i = 0; i < n; i++) {
            y[i] = Randomizer.nextGamma(alpha, 1);
            sumY += y[i];
        }

        // divide by the sum to obtain a sample from the Dirichlet distribution
        // and multiply the original value in array with the sampled factor
        // sum logs for probability
        double logP = 0;
        double[] p = new double[n];
        for (int i = 0; i < n; i++) {
            p[i] = y[i] / sumY;
            logP += Math.log(p[i]);
            x[i] = p[i] * sumX;
        }

        // probability under Dirichlet distribution
        double logDirichlet = Gamma.logGamma(n * alpha) - n * Gamma.logGamma(alpha) + (alpha - 1) * logP;
        // probability accounting for transformation
        return  -(n - 1) * Math.log(sumX) + logDirichlet;
    }


    /** Draws a normal value rounded to the nearest integer, with everything below 1.5 put on 1. */
    public static int sampleDiscretizedNormal(double mean, double sd) {
        double continuousSample = mean + Randomizer.nextGaussian() * sd;
        // continuity correction
        int discreteSample = (int) Math.round(continuousSample);
        // truncate at 1
        return Math.max(1, discreteSample);
    }


    /** Log probability of k under sampleDiscretizedNormal. */
    public static double logProbabilityDiscretizedNormal(int k, double mean, double sd) {
        NormalDistribution normal = new NormalDistribution(null, mean, sd);
        double prob;
        if (k == 1) {
            // fold the zero bin into the first admissible event count
            prob = normal.cumulativeProbability(1.5);
        } else {
            // continuity correction bin: [k - 0.5, k + 0.5]
            prob = normal.cumulativeProbability(k + 0.5) - normal.cumulativeProbability(k - 0.5);
        }
        return Math.log(prob);
    }


    /**
     * Approximate P0 at time t before the present: P0 of a linear birth-death process with sampling probability rho,
     * the extinction probability q = d / (1 - d) and the Malthusian growth rate r = (shape / lifetime)
     * ((2 (1 - d))^(1 / shape) - 1) of the age-dependent process, so 1 - rho at t = 0 and q for large t if r > 0.
     */
    public static double approximateP0(double t, double lifetime, double shape, double death, double rho) {
        double r = shape / lifetime * (Math.pow(2 * (1 - death), 1 / shape) - 1);
        // birth rate with lambda - mu = r and mu / lambda = q, and its limit (1 - d) / lifetime at d = 1/2
        double lambda = death == 0.5 ? 0.5 / lifetime : (1 - death) * r / (1 - 2 * death);
        double growth = r == 0 ? t : -Math.expm1(-r * t) / r;
        return 1 - rho / (Math.exp(-r * t) + rho * lambda * growth);
    }

    /**
     * Event count n >= 1 of a renewal segment with gamma(alpha) waiting times: n - 1 = m hidden events
     * have weight Y^m / Gamma(alpha (m + 1)), normalised by the Mittag-Leffler function E_{alpha,alpha}(Y).
     * Drawn by inverting the CDF of the weights, truncated to the window around the mode where they exceed 1e-16 of
     * the maximum.
     */
    public static int sampleMittagLefflerCount(double logY, double alpha) {
        MittagLefflerWeights w = mittagLefflerLogWeights(logY, alpha);
        double logZ = logSumExp(w.logW);
        double u = Randomizer.nextDouble();
        double cdf = 0;
        for (int i = 0; i < w.logW.length; i++) {
            cdf += Math.exp(w.logW[i] - logZ);
            if (u < cdf) return w.first + i + 1;
        }
        return w.first + w.logW.length;
    }

    /** Log probability of n under sampleMittagLefflerCount. */
    public static double logProbabilityMittagLefflerCount(int n, double logY, double alpha) {
        return mittagLefflerLogWeight(n - 1, logY, alpha) - logSumExp(mittagLefflerLogWeights(logY, alpha).logW);
    }

    // log weights logW[i] of m = first + i: log-concave in m, so taken outwards from the approximate mode
    // m + 1 = (Y^(1/alpha) + 1/2) / alpha until 1e-16 below the running maximum on both sides
    private static MittagLefflerWeights mittagLefflerLogWeights(double logY, double alpha) {
        int mode = (int) Math.max(0, Math.round((Math.exp(logY / alpha) + 0.5) / alpha - 1));
        double cutoff = Math.log(1e-16);
        double max = mittagLefflerLogWeight(mode, logY, alpha);
        int lo = mode, hi = mode;
        for (double w; lo > 0 && (w = mittagLefflerLogWeight(lo - 1, logY, alpha)) - max > cutoff; lo--) {
            max = Math.max(max, w);
        }
        for (double w; (w = mittagLefflerLogWeight(hi + 1, logY, alpha)) - max > cutoff; hi++) {
            max = Math.max(max, w);
        }
        double[] logW = new double[hi - lo + 1];
        for (int m = lo; m <= hi; m++) logW[m - lo] = mittagLefflerLogWeight(m, logY, alpha);
        return new MittagLefflerWeights(lo, logW);
    }

    private record MittagLefflerWeights(int first, double[] logW) {}

    private static double mittagLefflerLogWeight(int m, double logY, double alpha) {
        return (m == 0 ? 0 : m * logY) - Gamma.logGamma(alpha * (m + 1));
    }

    private static double logSumExp(double[] x) {
        double max = Arrays.stream(x).max().orElse(Double.NEGATIVE_INFINITY);
        double sum = 0;
        for (double v : x) sum += Math.exp(v - max);
        return max + Math.log(sum);
    }


    /** For debugging: writes arrays of equal length as columns of a csv file, every thin-th row. */
    public static void saveArrays(List<double[]> arrays, int thin, String header, String fileName) {

        int length = arrays.get(0).length;

        try (PrintWriter writer = new PrintWriter(new FileWriter(fileName))) {
            writer.println(header);

            for (int i = 0; i < length; i += thin) {
                for (int j = 0; j < arrays.size(); j++) {
                    writer.printf("%.6f", arrays.get(j)[i]);

                    // print a comma after every column except the very last one
                    if (j < arrays.size() - 1) {
                        writer.print(",");
                    }
                }
                writer.println();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
