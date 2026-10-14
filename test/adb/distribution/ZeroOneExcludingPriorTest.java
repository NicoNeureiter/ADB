package adb.distribution;

import beast.base.inference.distribution.Beta;
import beast.base.inference.distribution.Prior;
import beast.base.inference.parameter.RealParameter;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class ZeroOneExcludingPriorTest {

    static final Beta BETA = new Beta();
    static {
        BETA.initByName("alpha", new RealParameter("2.0"), "beta", new RealParameter("3.0"));
    }

    static RealParameter parameter(String values) {
        RealParameter p = new RealParameter(values);
        p.setLower(0.0);
        p.setUpper(1.0);
        return p;
    }

    static Prior prior(Prior prior, RealParameter x) {
        prior.initByName("x", x, "distr", BETA);
        return prior;
    }

    @Test
    public void testSameAsPriorWithoutZerosAndOnes() {
        RealParameter x = parameter("0.2 0.5 0.9");
        assertEquals(prior(new Prior(), x).calculateLogP(), prior(new ZeroOneExcludingPrior(), x).calculateLogP(), 1e-12);
    }

    @Test
    public void testZerosAndOnesAreSkipped() {
        double expected = BETA.logDensity(0.3) + BETA.logDensity(0.6);
        assertEquals(expected, prior(new ZeroOneExcludingPrior(), parameter("0 0.3 1 0.6 0")).calculateLogP(), 1e-12);
        // the beta prior itself is not finite at 0 and 1
        assertEquals(Double.NEGATIVE_INFINITY, prior(new Prior(), parameter("0 0.3 1 0.6 0")).calculateLogP());
    }

    @Test
    public void testAllZerosAndOnesGiveZero() {
        assertEquals(0.0, prior(new ZeroOneExcludingPrior(), parameter("0 1 0 1")).calculateLogP());
    }

    @Test
    public void testOutsideBoundsIsMinusInfinity() {
        RealParameter x = parameter("0.3 0.6 0");
        Prior prior = prior(new ZeroOneExcludingPrior(), x);
        x.setLower(0.4); // 0.3 is now below the bound
        assertEquals(Double.NEGATIVE_INFINITY, prior.calculateLogP());

        // a value of 1 outside the bounds is not skipped
        RealParameter y = parameter("0.3 1");
        y.setUpper(0.9);
        assertEquals(Double.NEGATIVE_INFINITY, prior(new ZeroOneExcludingPrior(), y).calculateLogP());
    }
}
