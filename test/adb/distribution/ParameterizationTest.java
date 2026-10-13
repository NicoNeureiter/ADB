package adb.distribution;

import beast.base.inference.parameter.RealParameter;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class ParameterizationTest {

    private Parameterization twoTypeModel(RealParameter sym, RealParameter asym) {
        Parameterization model = new Parameterization();
        model.initByName("nTypes", 2,
                "lifetime", new RealParameter("2 5"),
                "shape", new RealParameter("1 1"),
                "death", new RealParameter("0.1 0.2"),
                "symTransitions", sym,
                "asymTransitions", asym,
                "sampling", new RealParameter("0.5 0.5"),
                "originTime", 20.0,
                "originType", 0);
        return model;
    }

    // test model parameters and settings
    @Test
    public void testParametrization() throws Exception {
        Parameterization model = twoTypeModel(new RealParameter("0.2 0 0 0.6"), new RealParameter("0 0.8 0.4 0"));

        System.out.println(model);
        assertTrue(model.isDirectProgenitor(0, 1));
        assertTrue(model.isDirectProgenitor(1, 0));
        assertTrue(model.isDirectProgenitor(0, 0));
    }

    // transitions that become zero must disappear from the graph, and the dirty flag must be cleared
    @Test
    public void testTransitionsUpdate() {
        RealParameter asym = new RealParameter("0 0.8 0 0");
        Parameterization model = twoTypeModel(new RealParameter("0.2 0 0 1"), asym);
        assertTrue(model.isDirectProgenitor(0, 1));
        assertTrue(model.isProgenitor(0, 1));
        assertFalse(model.isProgenitor(1, 0));

        model.store();
        asym.setValue(1, 0.0);
        model.dirtyTransitions = true;

        assertFalse(model.isDirectProgenitor(0, 1), "removed transition must not remain in the graph");
        assertFalse(model.isProgenitor(0, 1));
        assertFalse(model.dirtyTransitions, "dirty flag must be cleared after the update");

        // reject: the old graph is back
        model.restore();
        assertTrue(model.isDirectProgenitor(0, 1));
        assertTrue(model.isProgenitor(0, 1));
    }

    @Test
    public void testDivisionFactor() {
        Parameterization model = twoTypeModel(new RealParameter("0.2 0 0 0.6"), new RealParameter("0 0.8 0.4 0"));
        assertEquals(0.2, model.getDivisionFactor(0, 0, 0), 1e-12, "symmetric self-renewal");
        assertEquals(0.0, model.getDivisionFactor(0, 1, 1), 1e-12, "symmetric differentiation not possible");
        assertEquals(0.4, model.getDivisionFactor(0, 0, 1), 1e-12, "asymmetric: 0.5 * asym(0,1)");
        assertEquals(0.4, model.getDivisionFactor(0, 1, 0), 1e-12, "order of the daughters does not matter");
        assertEquals(0.2, model.getDivisionFactor(1, 1, 0), 1e-12, "asymmetric: 0.5 * asym(1,0)");
        assertEquals(0.6, model.getDivisionFactor(1, 1, 1), 1e-12);
    }

    // a type that only divides asymmetrically: the diagonal must not be mistaken for an impossible division
    @Test
    public void testDivisionFactorAsymmetricOnlyType() {
        Parameterization model = twoTypeModel(new RealParameter("0 0 0 1"), new RealParameter("0 1 0 0"));
        assertEquals(0.5, model.getDivisionFactor(0, 0, 1), 1e-12);
        assertEquals(0.5, model.getDivisionFactor(0, 1, 0), 1e-12);
        assertEquals(0.0, model.getDivisionFactor(0, 0, 0), 1e-12);
        assertEquals(0.0, model.getDivisionFactor(0, 1, 1), 1e-12);
        assertEquals(0.0, model.getDivisionFactor(1, 0, 1), 1e-12, "no asymmetric division from type 1");
        assertFalse(model.isDirectProgenitor(0, 0), "isDirectProgenitor(k,k) is false here, so it cannot be used for the division check");
    }
}
