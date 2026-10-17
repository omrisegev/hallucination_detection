"""Follow-ups: per-class C2/C5 tables, position/length bins with group counts, PRM tie sensitivity,
structural null for control false alarms, alternative handling of the 16 inert answers."""
import json, pickle, sys
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd
from scipy.stats import rankdata

MAIN = Path(r'C:/Users/omris/TAU/hallucination_detection')
W = MAIN / '.worktrees/ssl-pseudolabel-residual-v1'
R = MAIN / '.worktrees/readout-quickest-detection-v1/results/step_evidence_v1'
ST = W / 'results/expectation_realization_v1'
sys.path.insert(0, str(MAIN / '.worktrees/depth-feature-fusion-v1'))
from spectral_utils.prmbench import prmbench_evaluate

ans = pd.read_csv(R / 'OOF_ANSWERS.csv', encoding='utf-8-sig')
Z = np.load(R / 'OOF_STEP_SCORES.npz'); off = Z['offsets']; lab = Z['labels'].astype(bool)
n = len(ans); ns = np.diff(off); S = int(off[-1])
prm = ~ans.cell.str.startswith('pb_').to_numpy(); P = np.flatnonzero(prm); ids = ans.id.to_numpy()
fold = ans.fold.to_numpy(); g_all = ans.source_group.to_numpy()
freeze = json.loads((R / 'INPUT_FREEZE.json').read_text(encoding='utf-8-sig'))
meta = {m['idx']: m for m in pickle.load(open(freeze['prm_metadata']['path'], 'rb')).values()}
cls = np.array([meta[ids[i]]['classification'] if prm[i] else '' for i in range(n)])
control = prm & (cls == 'correct'); nc = prm & ~control
has_err = np.array([prm[i] and lab[off[i]:off[i+1]].any() for i in range(n)])
has_ok = np.array([prm[i] and (~lab[off[i]:off[i+1]]).any() for i in range(n)])
ms = nc & (cls == 'multi_solutions'); inert = nc & ~has_err & ~ms
prm_step = np.repeat(prm, ns); step_fold = np.repeat(fold, ns)
FB = np.load(ST / 'run_20260927_stage_b/STEP_SCORES.npz'); FB2 = np.load(ST / 'run_20260927_stage_b2/STEP_SCORES.npz')
TH = json.loads((ST / 'run_20260927_stage_b_thr/THRESHOLDS.json').read_text('utf8')); TH2 = json.loads((ST / 'run_20260927_stage_b2_thr/THRESHOLDS.json').read_text('utf8'))
def zt(x): return (x - x.mean()) / max(x.std(), 1e-8)
def answer_z(s):
    z = np.full(S, np.nan)
    for i in P: z[off[i]:off[i+1]] = zt(s[off[i]:off[i+1]])
    return z
def q80(z): return {k: float(np.quantile(z[prm_step & (step_fold == (k+1) % 5)], .8)) for k in range(5)}
def valid_from(z, tau):
    v = np.zeros(S, bool)
    for k in range(5):
        sel = prm_step & (step_fold == k); v[sel] = z[sel] < tau[k]
    return v
score, valid = {}, {}
for m in ['B13_equal', 'S_equal', 'G1_sml', 'B13_lsml', 'S_lsml', 'fam421', 'ct7', 'step_index']:
    score[m] = FB[m]; valid[m] = valid_from(answer_z(FB[m]), {int(k): v for k, v in TH[m].items()})
score['B_sml__merge'] = FB2['B_sml__merge']; valid['B_sml__merge'] = valid_from(answer_z(FB2['B_sml__merge']), {int(k): v for k, v in TH2['B_sml__merge'].items()})
ch = np.load(ST / 'run_20260927_stage_b_thr/CHANNELS.npz'); names = [str(x) for x in ch['names']]
rd = ch['values'][:, names.index('realized_drv')].astype(float); score['realized_drv'] = rd
zr = answer_z(rd); valid['realized_drv'] = valid_from(zr, q80(zr))
reward = np.full(S, np.nan)
for i in P: reward[off[i]:off[i+1]] = np.asarray(meta[ids[i]]['rewards'], float)
risk = 1 - reward; score['PRM'] = risk
zp = answer_z(risk); valid['PRM_z_q80'] = valid_from(zp, q80(zp)); valid['PRM_raw_q80'] = valid_from(risk, q80(risk)); valid['PRM_native'] = prm_step & (reward >= .5)

def counts(v, mask, labels=lab):
    c = np.zeros(4)
    for i in np.flatnonzero(mask):
        vv = v[off[i]:off[i+1]]; gg = ~labels[off[i]:off[i+1]]
        c += [(vv & gg).sum(), (vv & ~gg).sum(), (~vv & ~gg).sum(), (~vv & gg).sum()]
    return c
def ps(c):
    tp, fp, tn, fn = c
    p = tp/(tp+fp); r = tp/(tp+fn)
    if tn + fp == 0: return np.nan
    p2 = tn/(tn+fn); r2 = tn/(tn+fp)
    return 0.5*(2*p*r/(p+r) + (2*p2*r2/(p2+r2) if p2 + r2 else 0))
def wauc(y, s):
    n1 = int(y.sum()); n0 = len(y) - n1
    return float((rankdata(s)[y].sum() - n1*(n1+1)/2)/(n1*n0))

# within-AUC sanity for realized_drv and PRM
E = np.flatnonzero(nc & has_err & has_ok)
print('within-AUC realized_drv', np.mean([wauc(lab[off[i]:off[i+1]], rd[off[i]:off[i+1]]) for i in E]), 'PRM', np.mean([wauc(lab[off[i]:off[i+1]], risk[off[i]:off[i+1]]) for i in E]), 'N', len(E))

# C2 / C5 per class, all 8 error classes, with group counts
CL8 = ['redundency', 'circular', 'counterfactual', 'step_contradiction', 'domain_inconsistency', 'confidence', 'missing_condition', 'deception']
rows = []
for c in CL8 + ['multi_solutions']:
    mk = nc & (cls == c)
    r = {'class': c, 'answers': int(mk.sum()), 'groups': len(set(g_all[mk])), 'err_steps': int(sum(lab[off[i]:off[i+1]].sum() for i in np.flatnonzero(mk)))}
    for m in ['S_equal', 'B13_equal', 'realized_drv', 'PRM_z_q80', 'PRM_raw_q80', 'PRM_native']:
        r[m] = round(ps(counts(valid[m], mk)), 4)
    r['S-B13'] = round(r['S_equal'] - r['B13_equal'], 4) if np.isfinite(r['S_equal']) else np.nan
    r['S-PRMz'] = round(r['S_equal'] - r['PRM_z_q80'], 4) if np.isfinite(r['S_equal']) else np.nan
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))

# per-fold sign of per-class S_equal - B13_equal (C5) and S_equal - PRM_z (C2)
pf = []
for c in CL8:
    d5 = []; d2 = []
    for k in range(5):
        mk = nc & (cls == c) & (fold == k)
        d5.append(round(ps(counts(valid['S_equal'], mk)) - ps(counts(valid['B13_equal'], mk)), 4))
        d2.append(round(ps(counts(valid['S_equal'], mk)) - ps(counts(valid['PRM_z_q80'], mk)), 4))
    pf.append({'class': c, 'S-B13 by fold': d5, 'S-PRMz by fold': d2})
print(pd.DataFrame(pf).to_string(index=False))

# position bins (C5 secondary) with groups and class composition
fe = {i: int(np.flatnonzero(lab[off[i]:off[i+1]])[0]) for i in np.flatnonzero(nc & has_err)}
def pb(i):
    f = fe[i]; rel = f/(ns[i]-1) if ns[i] > 1 else 0.0
    return 'first_step' if f == 0 else 'early' if rel <= 1/3 else 'middle' if rel <= 2/3 else 'late'
posl = np.array(['' for _ in range(n)], dtype=object)
for i in fe: posl[i] = pb(i)
for b in ['first_step', 'early', 'middle', 'late']:
    mk = posl == b
    top = Counter(cls[mk]).most_common(3)
    d = ps(counts(valid['S_equal'], mk)) - ps(counts(valid['B13_equal'], mk))
    hitS = np.mean([lab[off[i] + int(np.argmax(score['S_equal'][off[i]:off[i+1]]))] for i in np.flatnonzero(mk)])
    hitB = np.mean([lab[off[i] + int(np.argmax(score['B13_equal'][off[i]:off[i+1]]))] for i in np.flatnonzero(mk)])
    print(b, 'answers', int(mk.sum()), 'groups', len(set(g_all[mk])), 'eligible', int((mk & has_ok).sum()), 'top classes', top,
          'S-B13 prmscore', round(d, 4), 'hit S', round(hitS, 4), 'hit B13', round(hitB, 4), 'min fold', min(Counter(fold[mk]).values()))
# length quartiles
q = np.quantile(ns[nc], [.25, .5, .75]); print('length edges', q)
ll = np.array([f'q{int(np.searchsorted(q, ns[i], side="left")) + 1}' if nc[i] else '' for i in range(n)])
print('length bins', dict(Counter(ll[nc])), {b: len(set(g_all[ll == b])) for b in ['q1', 'q2', 'q3', 'q4']})

# PRM tie sensitivity on argmax hit (C6)
Eall = np.flatnonzero(nc & has_err)
def hit_rule(s, rule, rng=None):
    h = []
    for i in Eall:
        v = s[off[i]:off[i+1]]; t = np.flatnonzero(v >= v.max() - 8*np.finfo(float).eps)
        j = t[0] if rule == 'first' else t[-1] if rule == 'last' else rng.choice(t)
        h.append(lab[off[i] + j])
    return np.array(h)
hf = hit_rule(risk, 'first'); hl = hit_rule(risk, 'last'); hr = hit_rule(risk, 'rand', np.random.default_rng(0))
print('PRM hit earliest/latest/random tie-break', hf.mean(), hl.mean(), hr.mean(), 'answers changed earliest->latest', int((hf != hl).sum()))
hS = hit_rule(score['S_equal'], 'first'); hR = hit_rule(rd, 'first')
for nm, hp in (('earliest', hf), ('latest', hl)):
    print('S_equal|PRM', nm, {'both': int((hS & hp).sum()), 'only_S': int((hS & ~hp).sum()), 'only_PRM': int((~hS & hp).sum()), 'neither': int((~hS & ~hp).sum())})
    print('realized_drv|PRM', nm, {'both': int((hR & hp).sum()), 'only_rd': int((hR & ~hp).sum()), 'only_PRM': int((~hR & hp).sum()), 'neither': int((~hR & ~hp).sum())})
rw = reward[prm_step]; print('reward unique values', len(np.unique(rw)), 'of', len(rw), 'decimals sample', rw[:8])
# tied-argmax PRM answers by class
tiedc = Counter(cls[i] for i in Eall if (lambda v: (v >= v.max() - 8*np.finfo(float).eps).sum() > 1)(risk[off[i]:off[i+1]]))
print('PRM tied-argmax answers by class', dict(tiedc))

# structural null for control false alarms: random per-step scores under the same answer-z q80 rule
rng = np.random.default_rng(0); sh = []
for rep_ in range(5):
    rn = rng.standard_normal(S); zz = answer_z(rn); vv = valid_from(zz, q80(zz))
    fl = np.array([(~vv[off[i]:off[i+1]]).sum() for i in np.flatnonzero(control)])
    sh.append(((fl > 0).mean(), fl.sum() / ns[control].sum()))
print('random-score controls: share with >=1 flag, step flag rate', sh)
# lower bound: controls whose answer-z max is below the threshold is possible only for some n
print('control step count distribution', dict(sorted(Counter(ns[control]).items())[:10]), 'min', ns[control].min())

# task 4: alternative handling of the 16 inert answers: clip out-of-range indices to the last step (annotation-off-by-one reading)
lab2 = lab.copy()
for i in np.flatnonzero(inert): lab2[off[i+1] - 1] = True
alt = []
for c in ['confidence', 'counterfactual', 'deception', 'missing_condition', 'total']:
    mk = nc & ((cls == c) if c != 'total' else True)
    for a, b in [('S_equal', 'B13_equal'), ('S_equal', 'realized_drv'), ('S_equal', 'PRM_z_q80'), ('S_equal', 'PRM_raw_q80'), ('S_equal', 'PRM_native'), ('realized_drv', 'PRM_z_q80')]:
        off_ = ps(counts(valid[a], mk)) - ps(counts(valid[b], mk)); clip = ps(counts(valid[a], mk, lab2)) - ps(counts(valid[b], mk, lab2))
        alt.append({'stratum': c, 'contrast': f'{a}-{b}', 'official': round(off_, 5), 'clip_to_last_step': round(clip, 5), 'sign_flip': bool(np.sign(off_) != np.sign(clip))})
print(pd.DataFrame(alt).to_string(index=False))
# how many of the 16 inert answers: last-step flagged by each arm
for m in ['S_equal', 'B13_equal', 'realized_drv', 'PRM_z_q80', 'PRM_native']:
    print(m, 'inert answers with last step flagged', int(sum(not valid[m][off[i+1]-1] for i in np.flatnonzero(inert))), '/ 16')
# confidence-class ordering detail
for c in ['confidence']:
    for excl in (False, True):
        mk = nc & (cls == c) & (~inert if excl else True)
        print(c, 'excluding inert' if excl else 'official', sorted([(round(ps(counts(valid[m], mk)), 5), m) for m in valid], reverse=True)[:8])
