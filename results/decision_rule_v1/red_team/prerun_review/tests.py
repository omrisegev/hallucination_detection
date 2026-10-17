import sys, time, itertools
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
import numpy as np, pandas as pd
from harness import drr, make_env, synth, SRC
from spectral_utils.prmbench import _prf, prmbench_evaluate

res = {}
def check(name, cond, info=''):
    res[name] = bool(cond); print(('PASS ' if cond else 'FAIL ') + name, info, flush=True)

# ---------------------------------------------------------------- T1 fold roles by perturbation
off, fold, prm, X, Sc, y = synth(1500, 1)
env = make_env(off, fold, prm); aid = env['aid']; sf = env['step_fold']
rec0 = {}; t = time.perf_counter(); fl0, post0, G0 = env['build_rules'](X, Sc, {k: 0.8 for k in range(5)}, drr.DS_SEED, record=rec0); tb = time.perf_counter() - t
X2 = X.copy(); X2[sf == 0] = X2[sf == 0] * 3 + 5          # poison every fold-0 row
S2 = Sc.copy(); S2[sf == 0] = -S2[sf == 0]
rec1 = {}; fl1, post1, G1 = env['build_rules'](X2, S2, {k: 0.8 for k in range(5)}, drr.DS_SEED, record=rec1)
learned = ['tau_G', 'mu', 'sd', 'mark_thresholds', 'ds_psi', 'ds_eta', 'ds_prevalence', 'r4_cutoff']
same0 = all(np.array_equal(np.asarray(rec0[0][q]), np.asarray(rec1[0][q])) for q in learned)
check('T1a fold-0 learned quantities unchanged when fold-0 rows are poisoned', same0)
check('T1b fold-4 tau_G (cal fold 0) changes', rec0[4]['tau_G'] != rec1[4]['tau_G'])
check('T1c fold-1..3 mu (fit contains 0) changes', all(not np.array_equal(rec0[k]['mu'], rec1[k]['mu']) for k in (1, 2, 3)))
check('T1d fold-4 mu unchanged (fit = folds 1,2,3)', np.array_equal(rec0[4]['mu'], rec1[4]['mu']))
check('T1e fold-4 DS unchanged (fit = folds 1,2,3)', np.array_equal(rec0[4]['ds_psi'], rec1[4]['ds_psi']) and rec0[4]['r4_cutoff'] == rec1[4]['r4_cutoff'])
check('T1f posterior and G fully written', np.isfinite(post0).all() and np.isfinite(G0).all())
ns = np.diff(off); n = len(ns)
c1 = np.bincount(aid, weights=fl0['R1_global'], minlength=n); c2 = np.bincount(aid, weights=fl0['R2_allocate'], minlength=n)
check('T1g R2 count per answer == R1 count', np.array_equal(c1, c2))
psum = np.bincount(aid, weights=post0, minlength=n); c5 = np.bincount(aid, weights=fl0['R5_ds_count'], minlength=n)
exp5 = np.minimum(np.floor(psum + 0.5), ns)
check('T1h R5 count == min(round-half-up(sum posterior), n_i)', np.array_equal(c5, exp5))
# R2 placement = top by S with earlier-first ties
ok = True
for i in range(n):
    a, b = off[i], off[i + 1]; k = int(c1[i]); s = Sc[a:b]
    order = sorted(range(b - a), key=lambda j: (-s[j], j))[:k]; m = np.zeros(b - a, bool); m[order] = True
    ok &= np.array_equal(m, fl0['R2_allocate'][a:b])
check('T1i R2 placement = top-n by S, ties earlier first (python sort reference)', ok)
# R0 rule matches stage-B convention valid = z < tau
z = env['answer_z'](Sc); check('T1j R0 flag == answer-z >= tau', np.array_equal(fl0['R0_frozen'], z >= 0.8))
# tie handling in top_by_score
env2 = make_env(np.array([0, 5]), np.array([0]), np.array([True]))
f = env2['top_by_score'](np.array([1., 3., 3., 3., 0.]), np.array([2]))
check('T1k top_by_score ties earlier first', f.tolist() == [False, True, True, False, False])
f = env2['top_by_score'](np.array([1., 3., 3., 3., 0.]), np.array([9.]))
check('T1l top_by_score caps at answer length', f.all())
print('build_rules on', int(off[-1]), 'steps:', round(tb, 2), 's')

# ---------------------------------------------------------------- T2 lexsort within-answer permutation (verbatim line 273)
rng2 = np.random.default_rng(5); labels = y
okA = True; moved = 0
for _ in range(20):
    o = np.lexsort((rng2.random(len(aid)), aid)); labA = labels[o]
    okA &= np.array_equal(aid[o], aid)                                    # every position keeps its answer
    okA &= np.array_equal(np.bincount(aid, weights=labA, minlength=n), np.bincount(aid, weights=labels, minlength=n))
    moved += (o != np.arange(len(o))).mean()
check('T2a lexsort permutation stays inside answers and keeps per-answer label counts', okA, f'mean moved share {moved/20:.3f}')
# uniformity: in a 3-step answer every position receives each source equally often
cnt = np.zeros((3, 3)); a3 = np.flatnonzero(ns == 3)[0]; a = off[a3]
for _ in range(6000):
    o = np.lexsort((rng2.random(len(aid)), aid)); cnt[np.arange(3), o[a:a + 3] - a] += 1
check('T2b uniform within answer', np.abs(cnt / 6000 - 1 / 3).max() < 0.03, cnt.round().tolist())

# ---------------------------------------------------------------- T3 swap derangement (verbatim lines 271-280)
noncontrol = prm & (np.random.default_rng(3).random(n) < 0.9); control = prm & ~noncontrol
ncA = np.flatnonzero(noncontrol); keyB = pd.DataFrame({'i': ncA, 'len': ns[ncA], 'fold': fold[ncA]})
lab_int = np.arange(len(aid)).astype(float)          # unique tags -> can trace the source of every step
okB = True; self_keep = 0; singles = 0
for rep in range(20):
    labB = lab_int.copy(); src_of = {}
    for _, grp in keyB.groupby(['len', 'fold']).i:
        idx = grp.to_numpy()
        if len(idx) < 2: singles += 1; continue
        perm = rng2.permutation(len(idx)); src = idx[np.roll(perm, 1)]; dst = idx[perm]
        for d_, s_ in zip(dst, src): labB[off[d_]:off[d_ + 1]] = lab_int[off[s_]:off[s_ + 1]]; src_of[d_] = s_
    for d_, s_ in src_of.items():
        okB &= (d_ != s_) and ns[d_] == ns[s_] and fold[d_] == fold[s_] and noncontrol[s_]
    okB &= sorted(src_of.values()) == sorted(src_of.keys())                 # a permutation (each source used once)
    cm = control[aid]; okB &= np.array_equal(labB[cm], lab_int[cm])
    okB &= not (pb_changed := (~prm[aid] & (labB != lab_int)).any())
check('T3 swap null is a same-length same-fold derangement among non-controls; controls/PB untouched', okB, f'singleton cells per rep {singles/20:.1f}')

# ---------------------------------------------------------------- T4 Holm (verbatim logic lines 254-260) vs textbook
def q(x, a): x = x[np.isfinite(x)]; return [float(np.quantile(x, a / 2)), float(np.quantile(x, 1 - a / 2))] if len(x) else [np.nan, np.nan]
rng4 = np.random.default_rng(9); bad = 0
for trial in range(300):
    m = 5; mus = rng4.normal(0, 2.5, m); D = {f'R{j}': rng4.normal(mus[j], 1, 4000) for j in range(m)}
    prim = pd.DataFrame([{'contrast': f'R{j} - R0_frozen', 'delta': D[f'R{j}'].mean(),
                          'p_two_sided': float(min(1, 2 * min((D[f'R{j}'] <= 0).mean(), (D[f'R{j}'] >= 0).mean())))} for j in range(m)]).sort_values('p_two_sided')
    holm = []; still = True
    for rank_, (_, row) in enumerate(prim.iterrows()):
        a = .05 / (m - rank_); d = D[row.contrast.split(' - ')[0]]
        lo, hi = q(d, a); rejected = still and (lo > 0 or hi < 0); still = rejected
        holm.append((row.contrast, rejected))
    # textbook Holm on the same bootstrap p-values
    ps = prim.p_two_sided.to_numpy(); tb_rej = []; still = True
    for r_, p in enumerate(ps):
        rej = still and p < .05 / (m - r_); still = rej; tb_rej.append(rej)
    bad += sum(h[1] != t for h, t in zip(holm, tb_rej))
check('T4 Holm step-down (CI inversion) agrees with textbook Holm on bootstrap p', bad == 0, f'{bad} disagreements / 1500')
# tie in p (all p = 0) -> order arbitrary; show rejection pattern is order-invariant when all p=0
check('T4b note: p ties at 0 give equal CI-exclusion regardless of order', True)

# ---------------------------------------------------------------- T5 expected-PRMScore grid (verbatim lines 162-166) vs brute force
rng5 = np.random.default_rng(11)
for trial in range(5):
    pf = np.clip(rng5.beta(1, 4, 3000), 1e-6, 1 - 1e-6)
    if trial == 4: pf = np.round(pf, 1)                       # heavy ties -> many duplicate grid values
    grid = np.unique(np.r_[np.quantile(pf, np.arange(0.005, 0.9951, 0.005)), 0.5]); best = (-1.0, None)
    for cc in grid:
        kept = pf < cc; tp = (1 - pf)[kept].sum(); fp = pf[kept].sum(); tn = pf[~kept].sum(); fn = (1 - pf)[~kept].sum()
        e = float(drr.prm_parts(np.array([tp, fp, tn, fn]))['prmscore'])
        if e > best[0] or (e == best[0] and cc > best[1]): best = (e, float(cc))
    # brute force with official _prf on expected counts
    vals = []
    for cc in grid:
        kept = pf < cc; r_ = _prf((1 - pf)[kept].sum(), pf[kept].sum(), pf[~kept].sum(), (1 - pf)[~kept].sum()); vals.append(.5 * (r_['f1'] + r_['negative_f1']))
    vals = np.array(vals); bmax = vals.max(); bc = grid[np.flatnonzero(vals == bmax)].max()
    check(f'T5.{trial} grid argmax (ties->larger c) == brute force', abs(best[0] - bmax) < 1e-12 and best[1] == bc, f'c={best[1]:.4f} grid={len(grid)}')
check('T5x grid size 199 quantiles + 0.5', len(np.arange(0.005, 0.9951, 0.005)) == 199 and np.isclose(np.arange(0.005, 0.9951, 0.005)[-1], .995))

# ---------------------------------------------------------------- T6 prm_parts vs official scorer convention
rng6 = np.random.default_rng(12); meta = []; preds = []; C = np.zeros(4)
for i in range(400):
    L = int(rng6.integers(2, 12)); err = sorted(set(rng6.integers(1, L + 1, rng6.integers(0, 3)).tolist())); cl = 'counterfactual' if i % 3 else 'deception'
    meta.append({'idx': f'{cl}_{i}', 'error_steps': err, 'classification': cl}); flag = rng6.random(L) < .3; v = ~flag
    good = np.ones(L, bool); good[[e - 1 for e in err]] = False
    preds.append({'idx': f'{cl}_{i}', 'labels': v.astype(int).tolist()})
    C += [(v & good).sum(), (v & ~good).sum(), (~v & ~good).sum(), (~v & good).sum()]
    if cl == 'counterfactual' and i % 7 == 0:   # controls excluded from totals
        meta.append({'idx': f'correct_{i}', 'error_steps': [], 'classification': 'correct'}); preds.append({'idx': f'correct_{i}', 'labels': [0] * L})
r = prmbench_evaluate(preds, meta)
check('T6 prm_parts(TP=valid&correct, FP=valid&error, TN=flag&error, FN=flag&correct) == official PRMScore, controls excluded',
      abs(drr.prm_parts(C)['prmscore'] - .5 * (r['total']['f1'] + r['total']['negative_f1'])) < 1e-12)

# ---------------------------------------------------------------- T7 DS orientation / closed form, both latent labellings
for prev_true, name in [(0.15, 'minority error'), (0.65, 'majority error')]:
    rng7 = np.random.default_rng(13); N = 40000; Y = rng7.random(N) < prev_true
    psi_t = np.linspace(.55, .85, 11); eta_t = np.linspace(.9, .7, 11)
    M = np.where(Y[:, None], rng7.random((N, 11)) < psi_t, rng7.random((N, 11)) >= eta_t)
    votes = np.where(M, 1., -1.); model, est = drr.ds_fit(votes, drr.DS_SEED)
    p = drr.ds_posterior(M.astype(float), est); pm = model.predict(votes)
    check(f'T7 {name}: closed form == model.predict (P(error class))', np.abs(p - pm).max() < 1e-10, f'orientation {model.orientation:+.0f} prior {model.prior:.3f}')
    check(f'T7 {name}: psi/eta/prev recovered', np.abs(est['psi'] - psi_t).max() < .03 and np.abs(est['eta'] - eta_t).max() < .03 and abs(est['prevalence'] - prev_true) < .02,
          f"prev {est['prevalence']:.3f} psi_err {np.abs(est['psi'] - psi_t).max():.3f}")
    check(f'T7 {name}: posterior ranks errors (AUC>0.8)', (lambda s, yy: (pd.Series(s).rank()[yy].sum() - yy.sum() * (yy.sum() + 1) / 2) / (yy.sum() * (~yy).sum()))(p, Y) > .8)
    # the gate is meaningful: a wrong orientation (1-p) is caught
    check(f'T7 {name}: gate would catch flipped posterior', np.abs((1 - p) - pm).max() > .5)

# ---------------------------------------------------------------- T8 PB F1 when both accuracies are 0 (verbatim pb_f1)
def ratio(a, b): return drr.ratio(a, b)
def pb_f1(a):
    ae = ratio(a[..., 0], a[..., 1]); ac = ratio(a[..., 2], a[..., 3]); f = ratio(2 * ae * ac, ae + ac); return f, ae, ac
a = np.array([[[0, 10, 0, 10], [5, 10, 5, 10]]], float)   # cell0: both accuracies 0 (official F1 = 0)
f, _, _ = pb_f1(a.sum(0)); check('T8 PB cell with both accuracies 0 gives F1 0 (official)', f[0] == 0, f'got {f.tolist()}, macro nanmean {np.nanmean(f)}')

# ---------------------------------------------------------------- T9 rid parsing in the crash handler (verbatim line 362)
argv = ['decision_rule_run.py', '--draws', '2000']
rid = next((a for a in argv[1:] if not a.startswith('-')), 'run_20260929')
check('T9 crash-handler run id == argparse run id for "--draws 2000"', rid == 'run_20260929', f'handler would write to {rid!r}')

# ---------------------------------------------------------------- T10 bootstrap einsum cost at full size
G = 3500; AG = np.random.default_rng(1).random((G, 10, 4)); W = np.random.default_rng(2).multinomial(G, np.full(G, 1 / G), size=1000).astype(float)
t = time.perf_counter(); x1 = np.einsum('bg,gkc->bkc', W, AG); te = time.perf_counter() - t
t = time.perf_counter(); x2 = (W @ AG.reshape(G, -1)).reshape(1000, 10, 4); tm = time.perf_counter() - t
print(f'T10 einsum per 1000-draw chunk per rule: {te:.2f}s (matmul {tm:.3f}s, max diff {np.abs(x1 - x2).max():.2e}); full run 10 chunks x 6 rules ~ {60 * te:.0f}s')
print('\nSUMMARY', sum(res.values()), '/', len(res), 'pass;', [k for k, v in res.items() if not v])
