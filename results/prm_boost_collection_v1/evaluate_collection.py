"""PRM boosting collection v1, stage 2 (local; external gold never leaves this machine).

Reads the label-free stage-1 score exports (docs/experiments/PRM_BOOSTING_COLLECTION_PLAN_2026-10-21.md) and
computes, for any arm and any count/threshold rule:
  * the official metric: PRMBench PRMScore via spectral_utils.prmbench.prmbench_evaluate (+ per-classification);
    Socratic PRMScore / Hard2Verify step Balanced F1 via spectral_utils.family_external_metrics.summary;
  * within-answer AUC and pooled AUROC;
  * FPR/FNR/TPR/TNR/Precision, Pos/Neg accuracy (positive = VALID step, as in the official evaluators);
  * PRMBench categories and Simplicity/Soundness/Sensitivity;
  * paired bootstrap CIs: by source group on PRMBench, by answer on the external sets (2,000 draws).

Rules (labels are never used to score or to pick a rule; ORACLE and K2 are references only):
  native : the PRM's own reward <= 0.5 decision (PRMBench native_error_flag; external native_valid)
  Q80    : answer-z score above the pooled 80th percentile. PRMBench uses '>' (evaluate_amended.flags_for);
           Socratic/Hard2Verify use '>=' (eval_socratic_ablation5.az_q80). Each benchmark keeps its own code.
  GAP    : results/error_count_label_free_v1/banks_robustness.py:gap_counts on the answer-z scores, then top-k
  K2     : flag the top 2 steps of every answer (reference)
  ORACLE : flag the true number of error steps of every answer (reference, uses labels)

Data root: env COLLECTION_DATA (default ../../../hd_collection_data relative to the repo), layout
  prmbench/ABL5_SCORES_<tag>.npz, ABL5_CHANNELS_<tag>.npz, ABL5_RESULTS_<tag>.csv, ABL5_DIAG_<tag>.json
  prmbench_signabs/...                         (stage 1.2, |v| SML weights, no anchor flip)
  socratic/SCORES_<prm>.npz (+ _CHANNELS.npz, .json), socratic_signabs/, h2v/, h2v_signabs/
  frozen_socratic/, frozen_h2v/                (the frozen scores_* files: LSML_CONT on the external sets)

    python evaluate_collection.py parity          # every parity anchor of plan section 2 -> PARITY_STAGE2.csv
    python evaluate_collection.py status          # plan 2.1 tables -> tables/
"""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_k, '4')
import csv, json, sys
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from spectral_utils.prmbench import prmbench_evaluate  # noqa: E402
from spectral_utils.family_external_metrics import summary as ext_summary, components as ext_components  # noqa: E402
from spectral_utils.external_generalization.evaluation import confusion as ext_confusion  # noqa: E402

DATA = Path(os.environ.get('COLLECTION_DATA', ROOT.parent / 'hd_collection_data'))
TABLES = Path(os.environ.get('COLLECTION_TABLES', HERE / 'tables'))
META_JSON = ROOT / 'results/advisor_completion_2026-10-11/inputs/prmbench_meta.json'
GOLD = {'socratic': ROOT / 'results/remote_claims_replay_2026-10-15/inputs/socratic_gold.json',
        'hard2verify': ROOT / 'results/remote_claims_replay_2026-10-15/inputs/hard2verify_gold.json'}
PRMS = ['qwen_prm_7b', 'qwen_prm_72b', 'pure_prm', 'rlhflow_prm', 'math_shepherd']
EXT_NAME = {'qwen_prm_7b': 'qwen_prm'}                    # external files use 'qwen_prm' for the 7B
PRM_LABEL = {'qwen_prm_7b': 'Qwen2.5-Math-PRM-7B', 'qwen_prm_72b': 'Qwen2.5-Math-PRM-72B', 'pure_prm': 'PURE-PRM-7B',
             'rlhflow_prm': 'RLHFlow-PRM-Mistral-8B', 'math_shepherd': 'Math-Shepherd-Mistral-7B'}
BENCHES = ['prmbench', 'socratic', 'hard2verify']
DRAWS, SEED_PB, SEED_EXT = 2000, 20261013, 20261018
RULES = ['native', 'Q80', 'GAP', 'K2', 'ORACLE']

# Access labels (plan "Rules"): every table row carries one.
ACCESS = {'PRM': 'trained by its authors (PRM)', 'LSML_CONT': 'label-selected channels',
          'S5': 'label-selected channels', 'HALF': 'fixed control', 'LR': 'supervised',
          'fusion': 'label-free fusion', 'control': 'label-free control (no telemetry or no PRM)'}

CATS = [('NR', 'redundency'), ('NCL', 'circular'), ('ES', 'counterfactual'), ('SC', 'step_contradiction'),
        ('DC', 'domain_inconsistency'), ('CI', 'confidence'), ('PS', 'missing_condition'), ('DR', 'deception'),
        ('MS', 'multi_solutions')]
SSS = {'Simplicity': ['NR', 'NCL'], 'Soundness': ['ES', 'SC', 'DC', 'CI'], 'Sensitivity': ['PS', 'DR', 'MS']}


def access_of(arm: str) -> str:
    a = arm.split('_')[0]
    if arm.startswith('PRM_') or arm in ('PRM', 'PRMPROB', 'PRM_native', 'PRM_Q80', 'PRM_Q80_PROB'):
        return ACCESS['PRM']
    if arm == 'LSML_CONT' or arm.endswith('_S5'):
        return ACCESS['LSML_CONT']
    if a in ('HALF', 'HALFP'):
        return ACCESS['HALF']
    if a in ('POS', 'LEN', 'PRMPOS', 'PRMPOSLEN') or arm.startswith(('POS_ONLY', 'LEN_ONLY')):
        return ACCESS['control']
    if 'PERM' in a or 'RES' in a:
        return 'label-free control (' + ('permuted telemetry' if 'PERM' in a else 'position/length residualised') + ')'
    return ACCESS['fusion']


# ----------------------------------------------------------------------------------------------- helpers
def answer_z(s, off):
    """Per-answer loop, exactly as EV.az / common_ec.answer_z. Do NOT vectorize: GAP picks argmax over gaps that are
    exactly tied in theory for quantized rewards (PRM 1-reward), so a 1e-16 change flips k* (PRMBench Qwen-7B PRM-prob
    GAP 68.01 -> 68.00 with a reduceat version, checked 2026-10-21)."""
    out = np.zeros(len(s))
    for a, b in zip(off[:-1], off[1:]):
        v = s[a:b]; out[a:b] = (v - v.mean()) / max(v.std(), 1e-8)
    return out


def gap_counts(z, off):
    """Copied verbatim from results/error_count_label_free_v1/banks_robustness.py:gap_counts."""
    k = []
    for a, b in zip(off[:-1], off[1:]):
        m = b - a
        if m == 1: k.append(1); continue
        zz = np.sort(z[a:b])[::-1]; km = min(int(np.ceil(m / 2)), m - 1)
        k.append(int(np.argmax(zz[:km] - zz[1:km + 1]) + 1))
    return np.array(k)


def topk_flags(s, off, k):
    """common_ec.topk_flags: the k_a highest steps of answer a (stable order, ties by position)."""
    k = np.broadcast_to(np.asarray(k, int), (len(off) - 1,))
    f = np.zeros(off[-1], bool)
    for j, (a, b) in enumerate(zip(off[:-1], off[1:])):
        kk = int(min(max(k[j], 0), b - a))
        if kk: f[a + np.argsort(-s[a:b], kind='stable')[:kk]] = True
    return f


def auc(y, s):
    y = np.asarray(y, bool); n1 = int(y.sum()); n0 = len(y) - n1
    if not n1 or not n0: return np.nan
    return float((rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def per_answer_auc(err, s, off, keep=None):
    v = np.full(len(off) - 1, np.nan)
    for j, (a, b) in enumerate(zip(off[:-1], off[1:])):
        m = slice(a, b) if keep is None else np.flatnonzero(keep[a:b]) + a
        v[j] = auc(err[m], s[m])
    return v


# ----------------------------------------------------------------------------------------------- loaders
class Cell:
    """One (benchmark, PRM) cell on its SCORED steps (PRMBench: all steps; external: non-empty steps).
    err: error label per scored step; keep: steps that enter the metric; off: answer offsets over scored steps."""
    bench: str; prm: str; ids: np.ndarray; off: np.ndarray; err: np.ndarray; keep: np.ndarray
    arms: dict; native_flag: np.ndarray; groups: np.ndarray; extra: dict

    def z(self, arm):
        cache = self.__dict__.setdefault('_z', {})
        if arm not in cache: cache[arm] = answer_z(self.arms[arm], self.off)
        return cache[arm]

    def flags(self, arm, rule):
        if rule == 'native':
            return self.native_flag
        cache = self.__dict__.setdefault('_f', {})
        if (arm, rule) not in cache: cache[(arm, rule)] = self._flags(arm, rule)
        return cache[(arm, rule)]

    def _flags(self, arm, rule):
        z = self.z(arm)
        if rule == 'Q80':
            return (z > np.quantile(z, .8)) if self.bench == 'prmbench' else (z >= np.quantile(z, .8))
        if rule == 'GAP': return topk_flags(z, self.off, gap_counts(z, self.off))
        if rule == 'K2': return topk_flags(z, self.off, 2)
        if rule == 'ORACLE': return topk_flags(z, self.off, self.true_count)
        raise ValueError(rule)

    @property
    def true_count(self):
        return np.add.reduceat((self.err & self.keep).astype(int), self.off[:-1])


@lru_cache(None)
def prmbench_meta():
    return {r['idx']: r for r in json.loads(META_JSON.read_text())['rows']}


def load_prmbench(prm, variant='prmbench'):
    S = dict(np.load(DATA / variant / f'ABL5_SCORES_{prm}.npz'))
    meta = prmbench_meta(); c = Cell(); c.bench, c.prm = 'prmbench', prm
    c.ids = S['row_ids'].astype(str); c.off = S['offsets'].astype(np.int64)
    c.err = np.zeros(c.off[-1], bool)
    for j, i in enumerate(c.ids):                               # labels from raw 1-based error_steps (in range)
        a, b = c.off[j], c.off[j + 1]; assert b - a == meta[i]['n_steps'], i
        c.err[a:b] = np.isin(np.arange(1, b - a + 1), meta[i]['error_steps'])
    c.cls = np.array([meta[i]['classification'] for i in c.ids]); c.cat = np.array([meta[i]['category'] for i in c.ids])
    c.keep = np.repeat(c.cls != 'correct', np.diff(c.off))      # official pool: the 758 control rows excluded
    c.arms = {k[len('score__'):]: v.astype(float) for k, v in S.items() if k.startswith('score__')}
    c.native_flag = S['native_error_flag'].astype(bool); c.groups = S['source_group'].astype(str)
    c.extra = {k: S[k] for k in ('reward', 'position', 'log_step_tokens')}
    c.meta_rows = [meta[i] for i in c.ids]
    return c


@lru_cache(None)
def gold(bench):
    return {g['uid']: g for g in json.loads(GOLD[bench].read_text(encoding='utf8'))}


def load_external(bench, prm, variant=None, uids_keep=None):
    d = variant or {'socratic': 'socratic', 'hard2verify': 'h2v'}[bench]
    S = dict(np.load(DATA / d / f'SCORES_{EXT_NAME.get(prm, prm)}.npz'))
    G = gold(bench); c = Cell(); c.bench, c.prm = bench, prm
    uids = S['uids'].astype(str); lens = S['nonempty_lengths'].astype(int); ne = S['nonempty'].astype(bool)
    starts = np.r_[0, np.cumsum(lens)]
    ans = np.arange(len(uids)) if uids_keep is None else np.flatnonzero(np.isin(uids, list(uids_keep)))
    ne_idx = np.cumsum(ne) - 1                                   # position of each full step among non-empty ones
    full_rows = np.concatenate([np.arange(starts[j], starts[j + 1]) for j in ans])
    ne_rows = full_rows[ne[full_rows]]
    c.ids = uids[ans]
    c.off = np.r_[0, np.cumsum([ne[starts[j]:starts[j + 1]].sum() for j in ans])].astype(np.int64)
    corr = np.concatenate([np.asarray(G[u]['correct'], bool) for u in uids])
    inc = np.concatenate([np.asarray(G[u]['include'], bool) for u in uids])
    assert len(corr) == len(ne), (bench, prm)
    c.err = ~corr[ne_rows]; c.keep = inc[ne_rows]
    c.arms = {k[len('score__'):]: v[ne_idx[ne_rows]].astype(float) for k, v in S.items() if k.startswith('score__')}
    c.native_flag = (S['native_valid'][ne_rows] == 0)
    c.groups = c.ids.copy()                                      # bootstrap by answer
    # empty steps are predicted incorrect and still enter the official confusion (eval_socratic_ablation5)
    nv = S['native_valid'].astype(int)
    c.empty = {'correct': corr[full_rows][~ne[full_rows]], 'include': inc[full_rows][~ne[full_rows]],
               'native_valid': nv[full_rows][~ne[full_rows]].astype(bool),
               'answer': np.repeat(np.arange(len(ans)), [(~ne[starts[j]:starts[j + 1]]).sum() for j in ans])}
    c.cat = np.array([G[u].get('category', '') for u in c.ids])
    c.extra = {}
    return c


def load_frozen_lsml(c):
    """LSML_CONT on the external sets exists only in the frozen scores_<bench> files (label-free)."""
    d = {'socratic': 'frozen_socratic', 'hard2verify': 'frozen_h2v'}[c.bench]
    p = DATA / d / f'SCORES_{EXT_NAME.get(c.prm, c.prm)}.npz'
    if not p.exists(): return None
    F = dict(np.load(p))
    if 'score__LSML_CONT' not in F: return None
    uids = F['uids'].astype(str); lens = F['nonempty_lengths'].astype(int); ne = F['nonempty'].astype(bool)
    pos = {u: j for j, u in enumerate(uids)}; starts = np.r_[0, np.cumsum(lens)]; nst = np.r_[0, np.cumsum(ne)]
    out = []
    for u in c.ids:
        j = pos[str(u)]; a, b = starts[j], starts[j + 1]
        out.append(F['score__LSML_CONT'][nst[a]:nst[b]])
    v = np.concatenate(out)
    return v if len(v) == c.off[-1] else None


def load(bench, prm, **kw):
    return load_prmbench(prm, **kw) if bench == 'prmbench' else load_external(bench, prm, **kw)


# ----------------------------------------------------------------------------------------------- metrics
def counts_by_answer(c, f, native=False):
    """[tp, fp, tn, fn] per answer, POSITIVE = valid step (official orientation); only steps in keep.
    External: empty steps are added as 'predicted incorrect' (official Socratic/Hard2Verify scoring)."""
    y_valid = ~c.err; p_valid = ~f; m = c.keep
    cols = np.column_stack([y_valid & p_valid & m, ~y_valid & p_valid & m, ~y_valid & ~p_valid & m,
                            y_valid & ~p_valid & m]).astype(np.int64)
    C = np.add.reduceat(cols, c.off[:-1], axis=0)
    C[np.diff(c.off) == 0] = 0
    if c.bench != 'prmbench' and len(c.empty['correct']):
        e = c.empty; pv = e['native_valid'] if native else np.zeros(len(e['correct']), bool)   # else: predicted incorrect
        yv, inc = e['correct'], e['include']
        ec = np.column_stack([yv & pv & inc, ~yv & pv & inc, ~yv & ~pv & inc, yv & ~pv & inc]).astype(np.int64)
        np.add.at(C, e['answer'], ec)
    return C


def official_from_counts(C, bench):
    tp, fp, tn, fn = np.moveaxis(np.asarray(C, float), -1, 0)
    if bench in ('prmbench', 'socratic'):
        f1p = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=(2 * tp + fp + fn) > 0)
        f1n = np.divide(2 * tn, 2 * tn + fp + fn, out=np.zeros_like(tn), where=(2 * tn + fp + fn) > 0)
        return 50 * (f1p + f1n) if bench == 'prmbench' else .5 * (f1p + f1n)
    tpr = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    tnr = np.divide(tn, tn + fp, out=np.zeros_like(tn), where=(tn + fp) > 0)
    return np.divide(2 * tpr * tnr, tpr + tnr, out=np.zeros_like(tpr), where=(tpr + tnr) > 0)


def official(c, f, native=False):
    """Official metric through the official code paths (scale: PRMBench 0-100; Socratic/H2V 0-1)."""
    if c.bench == 'prmbench':
        preds = [{'idx': i, 'labels': (~f[a:b]).astype(int).tolist()} for i, a, b in zip(c.ids, c.off[:-1], c.off[1:])]
        o = prmbench_evaluate(preds, c.meta_rows); t = o['total']; bc = o['by_classification']
        cls = {k: 50 * (bc['f1'][k] + bc['negative_f1'][k]) for k in bc['f1'] if k != 'correct'}
        return 50 * (t['f1'] + t['negative_f1']), cls
    return ext_summary(counts_by_answer(c, f, native), c.bench)['metric'], {}


def rates(C):
    tp, fp, tn, fn = np.asarray(C, float).sum(0) if np.ndim(C) == 2 else np.asarray(C, float)
    q = lambda a, b: 100 * a / b if b else float('nan')
    return dict(TPR=q(tp, tp + fn), TNR=q(tn, tn + fp), FPR=q(fp, fp + tn), FNR=q(fn, tp + fn),
                Precision=q(tp, tp + fp), PosAcc=q(tp, tp + fn), NegAcc=q(tn, tn + fp),
                TP=int(tp), FP=int(fp), TN=int(tn), FN=int(fn))


def category_scores(c, f):
    """PRISM Table 7 semantics: official per-class PRMScore; multi_solutions (no error steps) = valid-step F1."""
    assert c.bench == 'prmbench'
    step_cls = np.repeat(c.cls, np.diff(c.off)); out = {}
    for short, cl in CATS:
        m = step_cls == cl
        tp = np.sum(~c.err & ~f & m); fp = np.sum(c.err & ~f & m); tn = np.sum(c.err & f & m); fn = np.sum(~c.err & f & m)
        f1p = 2 * tp / (2 * tp + fp + fn) if tp else 0.; f1n = 2 * tn / (2 * tn + fp + fn) if tn else 0.
        out[short] = 100 * f1p if cl == 'multi_solutions' else 50 * (f1p + f1n)
    for g, mem in SSS.items():
        out[g] = float(np.mean([out[k] for k in mem]))
    return out


def within(c, arm):
    cache = c.__dict__.setdefault('_w', {})
    if arm not in cache: cache[arm] = _within(c, arm)
    return cache[arm]


def _within(c, arm):
    s = c.arms[arm]
    keep = c.keep if c.bench != 'prmbench' else None             # PRMBench: every step of every answer (ablation5)
    return per_answer_auc(c.err, s, c.off, keep)


def pooled_auroc(c, arm):
    return auc(c.err[c.keep], c.arms[arm][c.keep])


def measure(c, arm, rule, fast=False):
    """fast=True: PRMBench PRMScore from the pooled confusion (identical to prmbench_evaluate's total on the
    non-control pool; asserted in collection_status for every cell), no per-classification columns."""
    f = c.flags(arm, rule)
    C = counts_by_answer(c, f, rule == 'native')
    if fast and c.bench == 'prmbench':
        met, cls = float(official_from_counts(C.sum(0), 'prmbench')), {}
    else:
        met, cls = official(c, f, rule == 'native')
    wa = within(c, arm)
    k = np.add.reduceat(f.astype(int), c.off[:-1])
    row = dict(benchmark=c.bench, prm=c.prm, arm=arm, rule=rule, access=access_of(arm),
               official=float(met), within_auc=float(np.nanmean(wa)), n_auc=int(np.isfinite(wa).sum()),
               pooled_auroc=pooled_auroc(c, arm),
               flag_rate=float(f[c.keep].mean()), mean_flags_per_answer=float(k.mean()), n_answers=len(c.ids),
               **{k2: v for k2, v in rates(C).items()})
    for k2, v in cls.items(): row[f'official__{k2}'] = v
    return row, C, wa


# ----------------------------------------------------------------------------------------------- bootstrap
def boot_weights(c, seed=None):
    u, inv = np.unique(c.groups, return_inverse=True)
    rng = np.random.default_rng(seed if seed is not None else (SEED_PB if c.bench == 'prmbench' else SEED_EXT))
    W = np.vstack([np.ones(len(u)), rng.multinomial(len(u), np.full(len(u), 1 / len(u)), size=DRAWS)])
    return W, inv, len(u)


def boot_stats(c, C, wa, W, inv, ng):
    G = np.zeros((ng, 6)); fin = np.isfinite(wa)
    np.add.at(G, inv, np.column_stack([np.where(fin, wa, 0), fin, C]))
    t = W @ G
    return official_from_counts(t[:, 2:], c.bench), t[:, 0] / t[:, 1]


def contrast(c, a, b, W=None):
    """a, b = (arm, rule). Paired bootstrap of official metric and within-AUC: delta, 95% CI."""
    if W is None: W = boot_weights(c)
    Wm, inv, ng = W
    ra, Ca, wa = measure(c, *a, fast=True); rb, Cb, wb = measure(c, *b, fast=True)
    oa, aa = boot_stats(c, Ca, wa, Wm, inv, ng); ob, ab = boot_stats(c, Cb, wb, Wm, inv, ng)
    out = []
    for mt, d, pt in (('official', oa - ob, ra['official'] - rb['official']), ('within_auc', aa - ab, ra['within_auc'] - rb['within_auc'])):
        out.append(dict(benchmark=c.bench, prm=c.prm, contrast=f'{a[0]}[{a[1]}] - {b[0]}[{b[1]}]', metric=mt,
                        delta=float(pt), delta_bootstrap_point=float(d[0]),
                        ci_low=float(np.quantile(d[1:], .025)), ci_high=float(np.quantile(d[1:], .975)),
                        n_groups=int(ng), draws=DRAWS))
    return out


# ----------------------------------------------------------------------------------------------- weight share
def prm_weight_share(c, arm, bank, score=None):
    """Least-squares weights of the fused score on its input views [z(PRM logit), bank channels]; share=|w0|/sum|w|.
    Same definition as results/sml_weight_estimator_v1/socratic_sign/prm_placement.py. Label-free."""
    ch = channels(c, bank)
    if ch is None: return None
    zp = answer_z(c.arms['PRM_Q80' if c.bench == 'prmbench' else 'PRM'], c.off)
    X = np.column_stack([zp, ch, np.ones(len(zp))]); s = c.arms[f'{arm}_{bank}'] if score is None else score
    wf, *_ = np.linalg.lstsq(X, s, rcond=None); w = wf[:-1]
    r2 = 1 - np.sum((s - X @ wf) ** 2) / np.sum((s - s.mean()) ** 2)
    return dict(prm_weight_share=float(abs(w[0]) / np.abs(w).sum()), prm_weight_sign=int(np.sign(w[0])),
                min_channel_weight=float(w[1:].min()), n_negative_channel_weights=int((w[1:] < 0).sum()), fit_r2=float(r2))


def channels(c, bank, variant=None):
    if c.bench == 'prmbench':
        p = DATA / (variant or 'prmbench') / f'ABL5_CHANNELS_{c.prm}.npz'
        if not p.exists(): return None
        Z = np.load(p); assert np.array_equal(Z['offsets'], c.off)
        return Z[f'channels__{bank}'].astype(float)
    d = variant or {'socratic': 'socratic', 'hard2verify': 'h2v'}[c.bench]
    p = DATA / d / f'SCORES_{EXT_NAME.get(c.prm, c.prm)}_CHANNELS.npz'
    if not p.exists(): return None
    Z = np.load(p); uids = Z['uids'].astype(str); off = Z['offsets'].astype(np.int64)
    pos = {u: j for j, u in enumerate(uids)}
    return np.concatenate([Z[f'channels__{bank}'][off[pos[u]]:off[pos[u] + 1]] for u in c.ids]).astype(float)


# ----------------------------------------------------------------------------------------------- io
def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, keys); w.writeheader(); w.writerows(rows)


def read_csv(path):
    with open(path, newline='', encoding='utf-8-sig') as fh:
        return list(csv.DictReader(fh))


def ext_arm(c, arm):
    """Name mapping between the PRMBench and the external arm names."""
    if c.bench == 'prmbench': return arm
    return {'PRM_native': 'PRM', 'PRM_Q80': 'PRM', 'PRM_Q80_PROB': 'PRMPROB'}.get(arm, arm)


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'parity'
    if cmd == 'parity':
        from collection_parity import main as _m; _m()
    elif cmd == 'status':
        from collection_status import main as _m; _m()
    else:
        raise SystemExit(f'unknown command {cmd}')
