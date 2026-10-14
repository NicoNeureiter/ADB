package adb.operators;

import beast.base.inference.parameter.RealParameter;
import beast.base.util.Randomizer;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class MatrixDeltaExchangeOperatorTest {

    // three types: sym and asym rows sum to 1 together; type 2 only divides into {2, 2}
    static final String SYM = "0.5 0 0  0 0.6 0  0 0 1";
    static final String ASYM = "0 0.3 0.2  0 0 0.4  0 0 0";

    static MatrixDeltaExchangeOperator create(double delta, RealParameter... parameters) {
        MatrixDeltaExchangeOperator operator = new MatrixDeltaExchangeOperator();
        operator.initByName("parameter", java.util.List.of(parameters), "delta", delta, "weight", 1.0);
        return operator;
    }

    static RealParameter matrix(String values) {
        RealParameter p = new RealParameter(values);
        p.setLower(0.0);
        p.setUpper(1.0);
        return p;
    }

    static double rowSum(int row, RealParameter... parameters) {
        double sum = 0;
        for (RealParameter p : parameters) {
            for (int j = 0; j < 3; j++) sum += p.getValue(row * 3 + j);
        }
        return sum;
    }

    @Test
    public void testRowSumsZerosAndOnesAreKept() {
        Randomizer.setSeed(1);
        RealParameter sym = matrix(SYM), asym = matrix(ASYM);
        MatrixDeltaExchangeOperator operator = create(0.3, sym, asym);

        int moves = 0;
        for (int step = 0; step < 5000; step++) {
            if (operator.proposal() != Double.NEGATIVE_INFINITY) moves++;
            for (int row = 0; row < 3; row++) {
                assertEquals(1.0, rowSum(row, sym, asym), 1e-12);
            }
            for (int k = 0; k < 9; k++) {
                for (RealParameter p : new RealParameter[]{sym, asym}) {
                    assertTrue(p.getValue(k) >= 0.0 && p.getValue(k) <= 1.0);
                }
            }
            for (int k : new int[]{1, 2, 3, 5, 6, 7}) assertEquals(0.0, sym.getValue(k));
            for (int k : new int[]{0, 3, 4, 6, 7, 8}) assertEquals(0.0, asym.getValue(k));
            assertEquals(1.0, sym.getValue(8));
        }
        assertTrue(moves > 1000, "most proposals should move, got " + moves);
        assertNotEquals(0.5, sym.getValue(0), 1e-6);
        assertNotEquals(0.3, asym.getValue(1), 1e-6);
        assertNotEquals(0.4, asym.getValue(5), 1e-6);
    }

    @Test
    public void testSingleMatrix() {
        Randomizer.setSeed(2);
        RealParameter p = matrix("0.2 0.3 0.5  1 0 0  0 0.5 0.5");
        MatrixDeltaExchangeOperator operator = create(0.2, p);
        for (int step = 0; step < 2000; step++) {
            operator.proposal();
            assertEquals(1.0, rowSum(0, p), 1e-12);
            assertEquals(1.0, rowSum(2, p), 1e-12);
            assertEquals(1.0, p.getValue(3));
        }
        assertNotEquals(0.2, p.getValue(0), 1e-6);
        assertNotEquals(0.5, p.getValue(7), 1e-6);
    }

    @Test
    public void testLowerBoundIsRespected() {
        Randomizer.setSeed(4);
        RealParameter p = new RealParameter("0.4 0.6 0  0 1 0  0 0 1");
        p.setLower(0.3);
        p.setUpper(1.0);
        MatrixDeltaExchangeOperator operator = create(0.5, p);
        for (int step = 0; step < 2000; step++) {
            operator.proposal();
            assertTrue(p.getValue(0) >= 0.3 && p.getValue(1) >= 0.3);
        }
    }

    /** With a flat target, the symmetric proposal must leave the uniform distribution on the simplex invariant. */
    @Test
    public void testStationaryDistributionIsUniform() {
        Randomizer.setSeed(5);
        RealParameter sym = matrix("0.2 0 0  0 1 0  0 0 1"), asym = matrix("0 0.3 0.5  0 0 0  0 0 0");
        MatrixDeltaExchangeOperator operator = create(0.5, sym, asym);

        int steps = 400000;
        double[] sum = new double[3], sumSquares = new double[3];
        for (int step = 0; step < steps; step++) {
            operator.proposal(); // a rejected proposal changes nothing, an accepted one is always accepted (flat target)
            double[] x = {sym.getValue(0), asym.getValue(1), asym.getValue(2)};
            for (int k = 0; k < 3; k++) {
                sum[k] += x[k];
                sumSquares[k] += x[k] * x[k];
            }
        }
        // Dirichlet(1, 1, 1): mean 1/3, E[x^2] = 1/6
        for (int k = 0; k < 3; k++) {
            assertEquals(1.0 / 3, sum[k] / steps, 0.01);
            assertEquals(1.0 / 6, sumSquares[k] / steps, 0.01);
        }
    }

    @Test
    public void testInvalidInputs() {
        assertThrows(RuntimeException.class, () -> create(0.1));
        assertThrows(RuntimeException.class, () -> create(0.1, matrix("0.5 0.5 0.5 0.5 0.5")));
        assertThrows(RuntimeException.class, () -> create(0.1, matrix("0.5 0.5 0 0"), matrix("0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5")));
        assertThrows(RuntimeException.class, () -> create(0.1, matrix("1 0 0 1"))); // nothing can move
        assertThrows(RuntimeException.class, () -> create(0.1, matrix("0.5 0 0 1"))); // one entry per row only
    }
}
