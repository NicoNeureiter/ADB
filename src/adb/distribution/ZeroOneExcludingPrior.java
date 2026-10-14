package adb.distribution;

import beast.base.core.Description;
import beast.base.core.Function;
import beast.base.inference.distribution.Prior;
import beast.base.inference.parameter.IntegerParameter;
import beast.base.inference.parameter.RealParameter;

/**
 * Meant for the entries of a transition probability matrix, where impossible transitions are fixed at 0 and
 * transitions that always happen are fixed at 1. Such entries are not parameters of the model, so they add
 * nothing to the prior, and a density like the beta density would be zero or infinite there. All other elements are
 * handled as in Prior, one by one. A value outside the bounds of the parameter still gives a log density of minus
 * infinity, also if it is 0 or 1. Since the density is evaluated per element, it does not work with a multivariate
 * distribution such as the Dirichlet.
 */
@Description("Like Prior, but excludes any zero and one elements of the input parameter from the calculation.")
public class ZeroOneExcludingPrior extends Prior {

    @Override
    public double calculateLogP() {
        Function x = m_x.get();

        double lower = Double.NEGATIVE_INFINITY;
        double upper = Double.POSITIVE_INFINITY;
        if (x instanceof RealParameter) {
            lower = ((RealParameter) x).getLower();
            upper = ((RealParameter) x).getUpper();
        } else if (x instanceof IntegerParameter) {
            lower = ((IntegerParameter) x).getLower();
            upper = ((IntegerParameter) x).getUpper();
        }

        logP = 0.0;
        for (double value : x.getDoubleValues()) {
            if (value < lower || value > upper) {
                logP = Double.NEGATIVE_INFINITY;
                return logP;
            }
            if (value == 0.0 || value == 1.0) continue;

            logP += dist.logDensity(value);
        }

        if (logP == Double.POSITIVE_INFINITY) {
            logP = Double.NEGATIVE_INFINITY;
        }
        return logP;
    }

}
