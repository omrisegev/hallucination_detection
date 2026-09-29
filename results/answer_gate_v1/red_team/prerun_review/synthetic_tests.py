"""Synthetic tests for answer_gate_run.py pieces. No project labels or metrics are read."""
import sys
import numpy as np
from scipy.stats import rankdata
ROOT = r'C:/Users/omris/TAU/hallucination_detection/.worktrees/decision-rule-v1'
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + '/scripts/experiments')
import answer_gate_run as agr  # noqa: E402
from spectral_utils import fusion_utils  # noqa: E402
from spectral_utils.streaming_utils import anchor_orient  # noqa: E402
from spectral_utils.upcr import upcr_fit, upcr_pipeline_faithful  # noqa: E402

rng = np.random.default_rng(0); results = {}


def zfit(X):
    return (X - X.mean(0)) / X.std(0)


# ---------------- T1: lsml_linear reproduces lsml_continuous (K>=2 natural, K=1 forced, single-view groups)
n = 800; y = rng.normal(size=n)
lat = [y + rng.normal(scale=.8, size=n) for _ in range(3)]
cols = []
for g, L in enumerate(lat):
    for _ in range([4, 3, 1][g]):          # third group is a single view
        cols.append(L + rng.normal(scale=.5, size=n))
X = zfit(np.column_stack(cols))
W, fused, meta = agr.lsml_linear(X)
results['T1a_K>=2_reconstruct_maxabs'] = float(np.max(np.abs(X @ W - fused)))
results['T1a_K'] = int(meta['K']); results['T1a_group_sizes'] = [len(ix) for ix, _ in meta['group_weights']]
# K=1 forced through the groups seam (monkeypatch the module-level name lsml_linear calls)
orig = agr.lsml_continuous
agr.lsml_continuous = lambda *v, **kw: orig(*v, groups=np.zeros(len(v), int), **kw)
W1, f1, m1 = agr.lsml_linear(X); agr.lsml_continuous = orig
results['T1b_K=1_reconstruct_maxabs'] = float(np.max(np.abs(X @ W1 - f1))); results['T1b_K'] = int(m1['K'])
# forced partition with singleton groups (K=3, two singletons)
agr.lsml_continuous = lambda *v, **kw: orig(*v, groups=np.r_[np.zeros(len(v) - 2, int), 1, 2], **kw)
W3, f3, m3 = agr.lsml_linear(X); agr.lsml_continuous = orig
results['T1c_singletons_reconstruct_maxabs'] = float(np.max(np.abs(X @ W3 - f3))); results['T1c_K'] = int(m3['K'])
# non-triviality: a wrong reconstruction (ignoring cross weights) must violate the gate
Wbad = np.zeros(X.shape[1])
for ix, w in meta['group_weights']: Wbad[ix] = w
results['T1d_wrong_reconstruction_maxabs(should be >>1e-8)'] = float(np.max(np.abs(X @ Wbad - fused)))
# sign convention used in D2: (Z*sig) @ W == Z @ (W*sig)
sig = np.where(rng.random(X.shape[1]) < .5, -1., 1.); Ws, fs, _ = agr.lsml_linear(X * sig)
results['T1e_signed_views_maxabs'] = float(np.max(np.abs(X @ (Ws * sig) - fs)))

# ---------------- T2: anchor_orient direction
epr = rng.normal(size=500); s = -2 * epr + rng.normal(size=500)
o, fl = anchor_orient(s, epr); results['T2_flipped_when_anticorrelated'] = bool(fl); results['T2_corr_after'] = float(np.corrcoef(o, epr)[0, 1])
o2, fl2 = anchor_orient(-s, epr); results['T2_not_flipped_when_correlated'] = (not fl2)

# ---------------- T3: U-PCR derived path == upcr_pipeline_faithful(orient='derived')
m, nn = 12, 600; yy = rng.normal(size=nn)
Fr = np.column_stack([(1 if j % 3 else -1) * (yy * rng.uniform(.3, 1)) + rng.normal(size=nn) for j in range(m)])
names = [f'f{j}' for j in range(m)]
score_pipe, res_pipe = upcr_pipeline_faithful({nm: Fr[:, j] for j, nm in enumerate(names)}, names, orient='derived', **agr.FIT)
Z = zfit(Fr); Ff = Z.T
probe = upcr_fit(Ff, **agr.FIT); pol = np.sign(probe.rho_hat_full); pol[pol == 0] = 1.0
res = upcr_fit(Ff * pol[:, None], **agr.FIT); w = res.w * pol
results['T3_upcr_derived_vs_pipeline_maxabs'] = float(np.max(np.abs(Z @ w - score_pipe)))
results['T3_res_has_keep'] = hasattr(res, 'keep') and res.keep.dtype == bool

# ---------------- T4: length-matched AUROC (code formula from answer_gate_run.py:278-280)
def lm_auc(pos, neg, ns_pos, ns_neg):
    bins = np.quantile(ns_neg, [0, .2, .4, .6, .8, 1]); eb = np.clip(np.searchsorted(bins, ns_pos, 'right') - 1, 0, 4); mb = np.clip(np.searchsorted(bins, ns_neg, 'right') - 1, 0, 4)
    wts = np.array([(mb == b).mean() / max((eb == b).mean(), 1e-12) for b in eb])
    cmp_ = (pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])
    return float((wts[:, None] * cmp_).sum() / (wts.sum() * len(neg)))
def auc(p, q):
    s_ = np.r_[p, q]; r_ = rankdata(s_); return float((r_[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(q)))
# score depends only on length; positives are longer on average -> raw AUROC > .5, length-matched ~ .5
lm, raw = [], []
for rep in range(40):
    ns_p = rng.integers(3, 30, 3000); ns_p = np.where(rng.random(3000) < .6, rng.integers(15, 30, 3000), ns_p)
    ns_n = rng.integers(3, 30, 300)
    sp = ns_p + rng.normal(scale=3, size=3000); sn = ns_n + rng.normal(scale=3, size=300)
    lm.append(lm_auc(sp, sn, ns_p, ns_n)); raw.append(auc(sp, sn))
results['T4_raw_auc_mean(expect >.5)'] = float(np.mean(raw)); results['T4_length_matched_auc_mean(expect ~.5)'] = float(np.mean(lm))
# equal-weight check: identical length distributions -> weighted == unweighted
ns_p = rng.integers(3, 30, 2000); ns_n = rng.integers(3, 30, 400); sp = rng.normal(size=2000) + .3; sn = rng.normal(size=400)
results['T4_same_length_dist_lm_minus_raw'] = lm_auc(sp, sn, ns_p, ns_n) - auc(sp, sn)
# integer-tie bins (duplicate quantile edges) must not crash
ns_n2 = np.r_[np.full(100, 5), np.arange(6, 16)]; results['T4_dup_edges_ok'] = np.isfinite(lm_auc(sp[:300], sn[:110], rng.integers(3, 30, 300), ns_n2))

# ---------------- T5: PB F1 and prediction conventions
a = np.zeros((1, 2, 4)); a[0, 0] = [0, 10, 0, 10]; a[0, 1] = [5, 10, 10, 10]
tp = lambda A: (lambda ae, ac: (np.where((ae == 0) & (ac == 0), 0.0, agr.ratio(2 * ae * ac, ae + ac)), ae, ac))(agr.ratio(A[..., 0], A[..., 1]), agr.ratio(A[..., 2], A[..., 3]))
f, ae, ac = tp(a.sum(0)); results['T5_pbf1_both_zero->0'] = float(f[0]); results['T5_pbf1_half_full'] = float(f[1])  # expect 2*.5*1/1.5

# ---------------- T6: prm_parts vs hand-computed PRMScore
c = np.array([50., 10., 20., 5.]); pp = agr.prm_parts(c)
f1 = 2 * 50 / (2 * 50 + 10 + 5); f1e = 2 * 20 / (2 * 20 + 5 + 10)
results['T6_prmscore_diff'] = float(pp['prmscore'] - (f1 + f1e) / 2)

for k, v in results.items(): print(f'{k:55s} {v}')
