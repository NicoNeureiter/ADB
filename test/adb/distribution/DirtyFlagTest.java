package adb.distribution;

import adb.util.Utils;
import beast.base.inference.parameter.IntegerParameter;
import beast.base.inference.parameter.RealParameter;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Checks that P0, P1 and the lifetime distributions are computed once per parameter change
 * (not on every access) and that store/restore still bring back the old state.
 */
public class DirtyFlagTest {

    RealParameter lifetime;
    IntegerParameter shape;
    Parameterization parameterization;
    LifetimeDistributions lifetimeDistributions;
    P0System p0System;
    P1System p1System;

    @BeforeEach
    public void setUp() {
        lifetime = new RealParameter("1 2");
        shape = new IntegerParameter("1 2");
        double origin = 5;

        parameterization = new Parameterization();
        parameterization.initByName("nTypes", 2,
                "lifetime", lifetime,
                "shape", shape,
                "death", new RealParameter("0.2 0.1"),
                "symTransitions", new RealParameter("0.5 0.5 0 1"),
                "asymTransitions", new RealParameter("0 0 0 0"),
                "sampling", new RealParameter("0.5 0.5"),
                "originTime", origin,
                "originType", 0);

        int nSteps = 1 << 10; // bit shifting operator, x << n is equivalent to x * 2^n
        double[] timeArray = Utils.linSpace(0, origin, nSteps);
        double timeStep = origin / nSteps;

        lifetimeDistributions = new LifetimeDistributions(lifetime, shape, timeArray);
        p0System = new P0System(parameterization, lifetimeDistributions, 100, 1e-12, timeArray, timeStep);
        p1System = new P1System(parameterization, lifetimeDistributions, p0System, 100, 1e-12, timeArray, timeStep);
    }

    @Test
    public void testP0SolvedOncePerChange() {
        p0System.dirty = true;
        double[][] first = p0System.getP0();
        assertFalse(p0System.dirty, "dirty flag must be cleared after solving");
        assertSame(first, p0System.getP0(), "P0 must not be solved again until the next change");
    }

    @Test
    public void testP1SolvedOncePerChange() {
        p1System.dirty = true;
        double[][][] first = p1System.getP1();
        assertFalse(p1System.dirty, "dirty flag must be cleared after solving");
        assertSame(first, p1System.getP1(), "P1 must not be solved again until the next change");
    }

    @Test
    public void testLifetimeDistributionsUpdatedOncePerChange() {
        lifetimeDistributions.store();
        lifetime.setValue(0, 1.5);
        lifetimeDistributions.findDirty();

        LifetimeDistributions.LifetimeDistribution first = lifetimeDistributions.getLifetimeDistributions()[0];
        assertEquals(1.5, first.lifetime, 0.0);
        assertSame(first, lifetimeDistributions.getLifetimeDistributions()[0],
                "distribution must not be recomputed until the next change");

        // nothing changed: findDirty must not trigger a recomputation
        lifetimeDistributions.findDirty();
        assertSame(first, lifetimeDistributions.getLifetimeDistributions()[0]);
    }

    @Test
    public void testLifetimeDistributionsRestore() {
        double[] oldCdf = lifetimeDistributions.getLifetimeDistributions()[0].getCDF().clone();

        lifetimeDistributions.store();
        lifetime.setValue(0, 1.5);
        lifetimeDistributions.findDirty();
        double[] newCdf = lifetimeDistributions.getLifetimeDistributions()[0].getCDF().clone();
        assertFalse(java.util.Arrays.equals(oldCdf, newCdf), "CDF should change with the lifetime");

        // reject: the old distribution must be back, even though it was already recomputed once
        lifetimeDistributions.restore();
        LifetimeDistributions.LifetimeDistribution restored = lifetimeDistributions.getLifetimeDistributions()[0];
        assertEquals(1.0, restored.lifetime, 0.0);
        assertArrayEquals(oldCdf, restored.getCDF(), 0.0);
    }

    @Test
    public void testP0RestoreAfterRecalculation() {
        double[][] oldP0 = p0System.getP0();
        double[] old = oldP0[0].clone();

        p0System.store();
        lifetime.setValue(0, 1.5);
        lifetimeDistributions.findDirty();
        p0System.dirty = true;
        double[] changed = p0System.getP0()[0].clone();
        assertFalse(java.util.Arrays.equals(old, changed), "P0 should change with the lifetime");

        p0System.restore();
        assertArrayEquals(old, p0System.getP0()[0], 0.0);
    }
}
