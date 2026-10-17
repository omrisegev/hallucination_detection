"""Plan section 17.2 tests for S3, run BEFORE real-data scoring."""
import numpy as np
from spectral_utils import ssl_s3 as S3
from spectral_utils import ssl_s1 as S1
rng = np.random.default_rng(0)


def test_contributions_sum_to_base_exactly():
    for task, teacher in [('pb', S1.teacher_pb), ('prm', S1.teacher_prm)]:
        Z = S1.answer_z(rng.normal(size=(7, 11))); h = S3.contributions(Z, task)
        assert np.allclose(h.sum(1), teacher(Z), atol=1e-14) and h.shape == (7, 11)
    Z1 = S1.answer_z(rng.normal(size=(1, 11))); assert np.allclose(S3.contributions(Z1, 'pb').sum(1), 1.)


def test_residualizer_removes_the_total_and_flags_inactive_columns():
    H = rng.normal(size=(500, 4)); H[:, 2] = 3 * H.sum(1) + 0.5        # column 2 = affine in the total (raw residual 0)
    b = H.sum(1); w = rng.uniform(.5, 1.5, 500); rz = S3.fit_residualizer(H, b, w)
    R, U = S3.residualize(H, b, rz); wn = w / w.sum()
    assert np.allclose(wn @ (R * (b - wn @ b)[:, None]), 0, atol=1e-9)      # residuals weighted-orthogonal to b
    assert np.allclose(wn @ R, 0, atol=1e-9)
    # column 2 is 3*b + .5 -> after residualization it is ~0 -> inactive? No: b contains column 2 itself, so not exactly affine.
    H2 = rng.normal(size=(300, 3)); b2 = rng.normal(size=300); H2[:, 1] = 2 * b2 - 1; rz2 = S3.fit_residualizer(H2, b2, np.ones(300))
    assert not rz2['active'][1] and rz2['active'][0] and rz2['active'][2]
    _, U2 = S3.residualize(H2, b2, rz2); assert np.all(U2[:, 1] == 0)


def test_neutral_direction_rule_and_orientation():
    vals = np.array([.2, .9, 1.1, 3.]); Q, _ = np.linalg.qr(rng.normal(size=(4, 4))); cov = Q @ np.diag(vals) @ Q.T; active = np.ones(4, bool)
    d = S3.neutral_direction(cov, active)
    assert d['status'] == 'OK' and abs(d['eigenvalue'] - .9) < 1e-9 and d['v'].sum() > 0 and abs(np.linalg.norm(d['v']) - 1) < 1e-12   # tie 0.9 vs 1.1 -> lower
    assert np.allclose(cov @ d['v'], .9 * d['v'], atol=1e-9)
    tie = S3.neutral_direction(np.diag([1., 1., 5.]), np.ones(3, bool)); assert tie['status'] == 'UNIDENTIFIED' and 'eigengap' in tie['reason']
    sub = S3.neutral_direction(np.diag([1., 7., 1.]), np.array([True, True, False])); assert sub['v'][2] == 0 and abs(sub['eigenvalue'] - 1) < 1e-12
    none = S3.neutral_direction(np.eye(2), np.zeros(2, bool)); assert none['status'] == 'UNIDENTIFIED'
    r0 = S3.random_direction(active, 0); r1 = S3.random_direction(active, 0); r2 = S3.random_direction(active, 1)
    assert np.array_equal(r0['v'], r1['v']) and not np.array_equal(r0['v'], r2['v']) and r0['v'].sum() > 0


def test_correction_edge_cases():
    base = rng.normal(size=6); aux = rng.normal(size=6)
    assert np.array_equal(S3.corrected(base, aux, dose=0.), base) and np.array_equal(S3.corrected(base, np.zeros(6)), base)
    cov, ncell = S3.cell_averaged_cov(rng.normal(size=(100, 3)), np.ones(100), np.array(['a'] * 50 + ['b'] * 50)); assert cov.shape == (3, 3) and ncell == 2 and np.allclose(cov, cov.T)
