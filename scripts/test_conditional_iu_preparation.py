"""Unit checks only: no benchmark data, fitted models, or scientific results."""
import os
for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[name] = '1'
from pathlib import Path
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from spectral_utils.conditional_iu_covariance import borrowed_iu_weights
from spectral_utils.available_memory import available_gib, cgroup_remaining
from spectral_utils.upcr import upcr_fit_covariance
from spectral_utils.laplacian_upcr import IU_FIT_DEFAULTS


def run():
    rng = np.random.default_rng(13)
    x = rng.normal(size=(50, 6)); c = x.T @ x / len(x)
    prior = .6 * c + .4 * np.eye(6)
    baseline = upcr_fit_covariance(c, **IU_FIT_DEFAULTS).w
    def forbidden(_):
        raise AssertionError('alpha zero must not fit or inspect the prior')
    zero, meta = borrowed_iu_weights(baseline, None, None, 0., covariance_solver=forbidden)
    np.testing.assert_array_equal(zero, baseline)
    np.testing.assert_array_equal(x @ zero, x @ baseline)
    assert not meta['fitted']
    for alpha in (.25, 1.):
        actual, _ = borrowed_iu_weights(baseline, c, prior, alpha)
        expected = upcr_fit_covariance((1-alpha)*c + alpha*prior, **IU_FIT_DEFAULTS).w
        np.testing.assert_array_equal(actual, expected)
    calls = []
    def solve_only(blended):
        calls.append(blended.copy())
        return np.linalg.solve(blended, np.ones(6)), dict(solver='test_solve_only')
    borrowed_iu_weights(baseline, c, prior, .25, covariance_solver=solve_only)
    np.testing.assert_allclose(calls[0], .75*c + .25*prior)
    for invalid in (-.1, 1.1, np.nan):
        try: borrowed_iu_weights(baseline, c, prior, invalid)
        except ValueError: pass
        else: raise AssertionError('invalid alpha accepted')
    try: borrowed_iu_weights(baseline, c, -np.eye(6), .5)
    except ValueError: pass
    else: raise AssertionError('indefinite prior accepted')
    try: borrowed_iu_weights(baseline, c, prior, .5, covariance_solver=forbidden)
    except AssertionError: pass
    else: raise AssertionError('solver failure silently replaced')
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root/'memory.max').write_text('1000')
        (root/'memory.current').write_text('300')
        assert cgroup_remaining(root) == 700
        (root/'memory.max').write_text('max')
        assert cgroup_remaining(root) is None
    assert available_gib() >= 0.
    return dict(status='PASS', alpha_zero_exact=True, canonical_IU_replay=True,
                explicit_solver_failures=True, memory_guard=True, benchmark_run=False)


if __name__ == '__main__':
    print(run())
