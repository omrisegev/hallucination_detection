import itertools
import unittest
import numpy as np
from spectral_utils.aligned_context_predictors import bocpd_mean, past_mean, noreset_mean


def enumerated_prediction(x, hazard):
    # Enumerate all partitions of past data and possible reset BEFORE next x.
    result = [0.]
    for t in range(1, len(x)):
        masses = []; forecasts = []
        for cuts in itertools.product((0, 1), repeat=t-1):
            mass = np.prod([hazard if b else 1-hazard for b in cuts])
            mean = 0.; var = 1.
            for j in range(t):
                if j and cuts[j-1]: mean = 0.; var = 1.
                mass *= np.exp(-.5*(x[j]-mean)**2/(1+var))/np.sqrt(2*np.pi*(1+var))
                mean += var/(1+var)*(x[j]-mean)
                var /= 1+var
            masses.append(mass); forecasts.append((1-hazard)*mean)
        result.append(np.dot(masses, forecasts)/np.sum(masses))
    return result


class AlignedPredictorTests(unittest.TestCase):
    def test_exact_partition_forecast(self):
        x = np.array([[.2, -.5], [.7, .3], [3., 2.], [-1., .1], [.4, 1.]])
        for hazard in (1/32, .2, .8):
            p = bocpd_mean(x, hazard)
            for k in range(2):
                np.testing.assert_allclose(p[:, k], enumerated_prediction(x[:, k], hazard), atol=1e-13)

    def test_noreset_closed_form(self):
        x = np.arange(12.).reshape(6, 2)
        expected = np.array([x[:t].sum(0)/(t+1) for t in range(len(x))])
        np.testing.assert_allclose(noreset_mean(x), expected)
        np.testing.assert_allclose(bocpd_mean(x, 0), expected)

    def test_no_current_or_future(self):
        x = np.random.default_rng(41).normal(size=(40, 5)); y = x.copy(); y[17:] += 30
        for model in (bocpd_mean, past_mean, noreset_mean):
            np.testing.assert_array_equal(model(x)[:18], model(y)[:18])

    def test_mean_scalar_and_edge_cases(self):
        x = np.arange(105.).reshape(21, 5)
        expected = np.array([x[max(0,t-16):t].mean(0) if t else np.zeros(5) for t in range(len(x))])
        np.testing.assert_allclose(past_mean(x), expected)
        for model in (bocpd_mean, past_mean, noreset_mean):
            np.testing.assert_array_equal(model(np.zeros((1,5))), np.zeros((1,5)))
            np.testing.assert_array_equal(model(np.zeros((30,5))), np.zeros((30,5)))
            with self.assertRaises(ValueError): model(np.array([[np.nan]]))


if __name__ == '__main__': unittest.main()
