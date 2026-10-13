package adb.distribution;

import adb.util.Utils;
import beast.base.inference.parameter.Parameter;
import beast.base.inference.parameter.RealParameter;
import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.distribution.GammaDistribution;

import java.util.BitSet;

import static adb.util.Utils.TRANSFORM_FORWARD;


/**
 * Holds the gamma lifetime distribution of each type (mean lifetime and shape), evaluated on the
 * uniform time grid: the CDF and the FFT of the PDF, ready for the convolutions in P0System and P1System.
 * Only the types whose lifetime or shape changed are recomputed (see findDirty).
 */
public class LifetimeDistributions {

    RealParameter lifetimeParameter;
    Parameter shapeParameter;
    int nTypes;
    double[] timeArray;
    int nSteps;

    // for identifying dirty indices
    boolean dirty;
    BitSet dirtyIndices = new BitSet();

    // output
    LifetimeDistribution[] lifetimeDistributions;
    LifetimeDistribution[] storedLifetimeDistributions;


    // Internal class
    public static class LifetimeDistribution {
        double lifetime;
        double shape;
        Complex[] pdfFFT;
        double[] cdf;

        public LifetimeDistribution(double lifetime, double shape, Complex[] pdfFFT, double[] cdf) {
            this.lifetime = lifetime;
            this.shape = shape;
            this.pdfFFT = pdfFFT;
            this.cdf = cdf;
        }

        public void copyFrom(LifetimeDistribution other) {
            // deep copy of all parameters
            this.lifetime = other.lifetime;
            this.shape = other.shape;
            for (int i = 0; i < other.pdfFFT.length; i++) {
                this.pdfFFT[i] = other.pdfFFT[i];
            }
            System.arraycopy(other.cdf, 0, this.cdf, 0, other.cdf.length);
        }

        public double[] getCDF() {
            return cdf;
        }

        public Complex[] getTransformedPDF() {
            return pdfFFT;
        }
    }


    public LifetimeDistributions(RealParameter lifetimeParameter, Parameter shapeParameter, double[] timeArray) {
        this.lifetimeParameter = lifetimeParameter;
        this.shapeParameter = shapeParameter;
        nTypes = lifetimeParameter.getDimension();
        this.timeArray = timeArray;
        nSteps = timeArray.length;

        lifetimeDistributions = new LifetimeDistribution[nTypes];
        storedLifetimeDistributions = new LifetimeDistribution[nTypes];

        // get all initial distributions
        dirty = true;
        dirtyIndices.set(0, nTypes);
        update();

        // copy to stored (avoid null)
        for (int i = 0; i < nTypes; i++) {
            storedLifetimeDistributions[i] = new LifetimeDistribution(
                    lifetimeDistributions[i].lifetime,
                    lifetimeDistributions[i].shape,
                    lifetimeDistributions[i].pdfFFT,
                    lifetimeDistributions[i].cdf);
        }

        dirtyIndices.clear();
        dirty = false;
    }


    private void update() {

        if (!dirty) { return; }

        // only update what actually changed!
        for (int i = dirtyIndices.nextSetBit(0); i >= 0; i = dirtyIndices.nextSetBit(i+1)) {
            updateLifetimeDistribution(i);
        }

        // dirtyIndices is kept until store/accept/restore, since restore() needs it
        dirty = false;
    }


    private void updateLifetimeDistribution(int x) {
        double lifetime = lifetimeParameter.getArrayValue(x);
        double shape = shapeParameter.getArrayValue(x);

        // initialize distribution
        double scale = lifetime / shape;
        GammaDistribution gammaDist = new GammaDistribution(shape, scale);

        // calculate PDF and CDF on given array
        double[] pdf = new double[nSteps];
        double[] cdf = new double[nSteps];
        for (int i = 0; i < nSteps; i++) {
            pdf[i] = Math.exp(gammaDist.logDensity(timeArray[i])); // use log to prevent underflow
            cdf[i] = gammaDist.cumulativeProbability(timeArray[i]);
        }

        // transform PDF
        Complex[] pdfFFT = Utils.fft.transform(Utils.padZeros(pdf), TRANSFORM_FORWARD);

        // store in list
        lifetimeDistributions[x] = new LifetimeDistribution(lifetime, shape, pdfFFT, cdf);
    }


    public LifetimeDistribution[] getLifetimeDistributions() {
        update();
        return lifetimeDistributions;
    }


    protected void findDirty() {
        for (int i = 0; i < nTypes; i++) {
            // mark this index for recalculation
            if (lifetimeParameter.getArrayValue(i) != lifetimeDistributions[i].lifetime ||
                    shapeParameter.getArrayValue(i) != lifetimeDistributions[i].shape) {
                dirtyIndices.set(i);
                dirty = true;
            }
        }
    }


    protected void store() {
        for (int i = 0; i < nTypes; i++) {
            storedLifetimeDistributions[i].copyFrom(lifetimeDistributions[i]);
        }
        dirtyIndices.clear();
        dirty = false;
    }


    protected void restore() {
        // only iterate over bits that are set to true
        for (int i = dirtyIndices.nextSetBit(0); i >= 0; i = dirtyIndices.nextSetBit(i+1)) {
            lifetimeDistributions[i].copyFrom(storedLifetimeDistributions[i]);
        }
        dirtyIndices.clear();
        dirty = false;
    }


    protected void accept() {
        dirtyIndices.clear();
        dirty = false;
    }
}
