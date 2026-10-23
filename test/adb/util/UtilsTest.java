package adb.util;

import org.apache.commons.math3.distribution.PoissonDistribution;
import org.apache.commons.math3.special.Gamma;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/** Mittag-Leffler event counts. */
public class UtilsTest {

    // for shape 1 the weights Y^m / m! make the number of hidden events m = n - 1 Poisson(Y)
    @Test
    public void testMittagLefflerCountIsPoissonForShapeOne() {
        for (double y : new double[]{0.01, 0.5, 3.0, 40.0}) {
            PoissonDistribution poisson = new PoissonDistribution(y);
            for (int m = 0; m < 60; m++) {
                assertEquals(poisson.logProbability(m), Utils.logProbabilityMittagLefflerCount(m + 1, Math.log(y), 1.0),
                        1e-9, "Y = " + y + ", m = " + m);
            }
        }
    }

    // the probabilities on the window around the mode match those normalised by the sum over all counts,
    // from a near-certain single event to about 10^4 events
    @Test
    public void testMittagLefflerCountMatchesFullSum() {
        for (double alpha : new double[]{1.0, 2.5, 10.0}) {
            for (double events : new double[]{0.01, 0.5, 3.0, 40.0, 1000.0, 10000.0}) {
                double logY = alpha * Math.log(alpha * events);
                int nMax = (int) (3 * events + 100);
                double[] logW = new double[nMax];
                double max = Double.NEGATIVE_INFINITY;
                for (int n = 1; n <= nMax; n++) {
                    logW[n - 1] = (n - 1) * logY - Gamma.logGamma(alpha * n);
                    max = Math.max(max, logW[n - 1]);
                }
                double sum = 0;
                for (double w : logW) sum += Math.exp(w - max);
                double logZ = max + Math.log(sum);
                for (int n = 1; n <= nMax; n++) {
                    double p = Math.exp(logW[n - 1] - logZ);
                    if (p < 1e-12) continue;
                    assertEquals(p, Math.exp(Utils.logProbabilityMittagLefflerCount(n, logY, alpha)), 1e-12 * p,
                            "alpha = " + alpha + ", events = " + events + ", n = " + n);
                }
            }
        }
    }
}
