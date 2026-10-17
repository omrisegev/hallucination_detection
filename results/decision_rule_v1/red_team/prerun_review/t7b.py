import sys; sys.path.insert(0, '.')
import numpy as np
from harness import drr
core, em = drr.A._cvf()
# latent class 1 = CORRECT (low mark rates), class 0 = ERROR (high mark rates); prior = P(class 1) = 0.8
e = np.column_stack([np.linspace(.6, .8, 11), np.linspace(.1, .3, 11)])   # col0 = P(mark|class0=error), col1 = P(mark|class1=correct)
built = core.orient(em.EMModel('ds', 0.8, e, diagnostics={'converged': True}), 11)
orig = em.fit_em
em.fit_em = lambda *a, **k: built
try:
    rng = np.random.default_rng(0); M = rng.random((5000, 11)) < .4; votes = np.where(M, 1., -1.)
    model, est = drr.ds_fit(votes, 1)
finally:
    em.fit_em = orig
print('orientation', model.orientation, 'prevalence(error)', est['prevalence'], 'psi[0]', est['psi'][0], 'eta[0]', est['eta'][0])
p = drr.ds_posterior(M.astype(float), est); pm = model.predict(votes)
print('closed form vs model.predict max abs', np.abs(p - pm).max())
# direct Bayes reference: P(class0 | marks)
l0 = np.log(0.2) + M @ np.log(e[:, 0]) + (~M) @ np.log(1 - e[:, 0]); l1 = np.log(0.8) + M @ np.log(e[:, 1]) + (~M) @ np.log(1 - e[:, 1])
ref = 1 / (1 + np.exp(l1 - l0)); print('vs direct Bayes P(error class)', np.abs(p - ref).max())
assert model.orientation == -1 and abs(est['prevalence'] - 0.2) < 1e-12 and np.abs(p - pm).max() < 1e-10 and np.abs(p - ref).max() < 1e-10
print('PASS T7c orientation -1 branch maps psi/eta/prevalence to the error class')
