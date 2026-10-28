"""ProcessBench metrics for PRM boosting (local only; reads gold labels).

Gold: `label` = 0-based first erroneous step, -1 = correct solution (ProcessBench contract, never the PRMBench one).
Scores are RISK-oriented (higher = more likely erroneous), as written by score_external.py / score_chain_pb.py.

1. Official ProcessBench metric (Qwen/ProcessBench code/run_eval.py; spectral_utils.processbench.first_error_f1):
   predicted first error = first step flagged erroneous, -1 if none. error_acc = exact-match rate on erroneous
   answers, correct_acc = rate of -1 on correct answers, F1 = harmonic mean, per subset; "average" = mean of the
   four subset F1s (ProcessBench's Avg. column).
   PRISM (arXiv 2606.09078, App. 10.2 / Table 6) reports ProcessBench TPR / TNR / F1. Its Baseline row equals our
   2026-08-10 Qwen2.5-Math-PRM-7B run to 0.1 (GSM8K 71.0 / 96.4 / 81.8, MATH 67.0 / 91.1 / 77.2), so PRISM's
   TPR = error_acc, TNR = correct_acc and F1 = official F1. They are exported under those names too.
2. Step-level confusion on the labelled domain (our definition, reported beside PRISM's names, not as theirs):
   steps of correct answers (all negative) plus steps 0..label of erroneous answers (step `label` positive).
   Steps after the first error are unlabelled and excluded. TPR, TNR and F1 of the erroneous class.
3. Within-answer AUC: in each erroneous answer with label >= 1, AUC of the first-error step against the steps
   before it (same domain as 2). Answers with label 0 or -1 have one class only and are undefined.
4. Pooled step AUROC (CROP, arXiv 2605.30085, App. Table 6 "Step AUROC"). CROP does not state its ProcessBench step
   target, so three declared targets are reported; `prefix` is primary:
     prefix      domain of 2, positive = first-error step
     first_only  all steps, positive = first-error step only (later steps negative)
     cumulative  all steps, positive = every step at or after the first error
   CROP reports means over 10 trace-level 60/20/20 test splits; ours is the full population (no split is needed:
   nothing here is fitted on labels). Check on job 177776's PRM-7B rewards: first_only gives 0.883, inside CROP's
   Direct PRM 0.8801 +/- 0.0059, while prefix gives 0.954 and cumulative 0.825; first_only is therefore the target
   to quote beside CROP (their stated 6.6% positive rate does not match it, 8.6%: open question).
Empty-token steps (score NaN) are excluded from the AUCs; decision rules treat them as the caller says.
5. Pre-registration (../PREREGISTRATION_HE.md, approved 2026-10-19): the NC first-error rule (section 3), the
   question-clustered, subset-stratified paired bootstrap and the E1 / E2 / verdict criterion (section 4).
"""
import numpy as np
from scipy.stats import rankdata

SUBSETS = ('gsm8k', 'math', 'olympiadbench', 'omnimath')
NO_ERROR = -1
TARGETS = ('prefix', 'first_only', 'cumulative')


def first_flag(flags):
    """Index of the first True, or -1."""
    f = np.flatnonzero(np.asarray(flags, bool))
    return int(f[0]) if len(f) else NO_ERROR


def auc(y, s):
    """Mann-Whitney AUC of score s for positives y (ties = 1/2); None when a class is empty."""
    y, s = np.asarray(y, bool), np.asarray(s, float)
    pos, neg = int(y.sum()), int((~y).sum())
    if not pos or not neg:
        return None
    return float((rankdata(s)[y].sum() - pos * (pos + 1) / 2) / (pos * neg))


def official(labels, preds, subsets):
    """Per-subset error_acc / correct_acc / F1 (percent) and the average of the subset F1s."""
    labels, preds, subsets = np.asarray(labels), np.asarray(preds), np.asarray(subsets)
    out = {}
    for s in SUBSETS:
        m = subsets == s
        if not m.any():
            continue
        e, c = m & (labels != NO_ERROR), m & (labels == NO_ERROR)
        ea = 100 * float(np.mean(preds[e] == labels[e])) if e.any() else None
        ca = 100 * float(np.mean(preds[c] == NO_ERROR)) if c.any() else None
        if ea is None or ca is None:
            f1 = None
        else:
            f1 = 2 * ea * ca / (ea + ca) if ea + ca > 0 else 0.0
        out[s] = {'error_acc': ea, 'correct_acc': ca, 'f1': f1, 'n_error': int(e.sum()), 'n_correct': int(c.sum())}
    f1s = [v['f1'] for v in out.values()]
    out['average'] = {'f1': float(np.mean(f1s)) if f1s and None not in f1s else None}
    return out


def step_targets(label, n):
    """(y, include) per target for one answer with n steps."""
    k = np.arange(n)
    if label == NO_ERROR:
        z = np.zeros(n, bool)
        return {'prefix': (z, np.ones(n, bool)), 'first_only': (z, np.ones(n, bool)),
                'cumulative': (z, np.ones(n, bool))}
    return {'prefix': (k == label, k <= label), 'first_only': (k == label, np.ones(n, bool)),
            'cumulative': (k >= label, np.ones(n, bool))}


def step_arrays(labels, lens):
    """Concatenated (y, include) over answers for every target."""
    per = [step_targets(int(l), int(n)) for l, n in zip(labels, lens)]
    return {t: (np.concatenate([p[t][0] for p in per]), np.concatenate([p[t][1] for p in per])) for t in TARGETS}


def pooled_auroc(labels, lens, risk, target='prefix'):
    y, inc = step_arrays(labels, lens)[target]
    keep = inc & np.isfinite(risk)
    return auc(y[keep], np.asarray(risk)[keep])


def within_auc_per_answer(labels, lens, risk):
    """Per-answer AUC (NaN where undefined) on the prefix domain."""
    starts = np.r_[0, np.cumsum(lens)]; out = np.full(len(labels), np.nan)
    for i, (l, n) in enumerate(zip(labels, lens)):
        if l < 1:
            continue
        s = np.asarray(risk[starts[i]:starts[i] + l + 1], float); y = np.arange(l + 1) == l
        ok = np.isfinite(s)
        v = auc(y[ok], s[ok])
        out[i] = np.nan if v is None else v
    return out


def step_confusion(labels, lens, flags):
    """TPR / TNR / F1(error class) on the prefix domain; flags = predicted erroneous, all steps."""
    y, inc = step_arrays(labels, lens)['prefix']
    p = np.asarray(flags, bool)[inc]; y = y[inc]
    tp, fn = int((p & y).sum()), int((~p & y).sum()); tn, fp = int((~p & ~y).sum()), int((p & ~y).sum())
    tpr = tp / (tp + fn) if tp + fn else None; tnr = tn / (tn + fp) if tn + fp else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return {'tpr': tpr, 'tnr': tnr, 'f1_error': f1, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}


# ---------------------------------------------------------------- pre-registration (PREREGISTRATION_HE.md)
PRIMARY = ('CHAIN_N16_NC', 'PRM_native')   # registered contrast: arm with rule NC minus the PRM's native rule
E2_MARGIN = -1.0                           # E2: CI low of the average-F1 difference must be above -1.0 point
STRONG = ('qwen_prm', 'qwen_prm_72b')      # at least one of these must succeed
N_PRMS, N_NEED = 5, 3
REGISTERED_PRMS = ('qwen_prm', 'pure_prm', 'rlhflow_prm', 'math_shepherd', 'qwen_prm_72b')


def nc_flags(lens, nonempty, native_flags, score_ne):
    """NC rule (native count, fused order), section 3. All arrays cover every step of every answer in order, except
    score_ne, which covers the non-empty steps only (the SCORES layout). Per answer:
      1 gate: no step flagged by the PRM's native rule (reward < 0.5) -> nothing flagged, prediction -1;
      2 k = number of NON-EMPTY steps the native rule flagged;
      3 flag the k non-empty steps with the highest score; ties -> the earlier step first;
      4 the prediction is the earliest flagged step (first_flag of the returned flags);
      5 k = 0 although the gate fired (only empty steps flagged) -> the native flags, i.e. the PRM's own first error.
    Returns (flags over all steps, {'n_gate': answers past the gate, 'n_k0': answers that took case 5})."""
    lens = np.asarray(lens, int); ne = np.asarray(nonempty, bool); nat = np.asarray(native_flags, bool)
    full = np.full(len(ne), np.nan); full[ne] = np.asarray(score_ne, float)
    if not np.isfinite(full[ne]).all():
        raise ValueError('nc_flags: non-finite score on a non-empty step')
    flags = np.zeros(len(ne), bool); starts = np.r_[0, np.cumsum(lens)]; n_gate = n_k0 = 0
    for a, b in zip(starts[:-1], starts[1:]):
        if not nat[a:b].any():
            continue
        n_gate += 1
        idx = np.flatnonzero(ne[a:b]); k = int(nat[a:b][idx].sum())
        if k == 0:
            flags[a:b] = nat[a:b]; n_k0 += 1
            continue
        order = np.lexsort((idx, -full[a:b][idx]))   # score descending, then step index ascending
        flags[a + idx[order[:k]]] = True
    return flags, {'n_gate': n_gate, 'n_k0': n_k0}


def norm_question(q):
    """Cluster key: the question text with every whitespace run collapsed to one space."""
    return ' '.join(str(q).split())


def clustered_counts(subsets, clusters, rng, draws):
    """Paired bootstrap multiplicities, shape (draws, n): stratified by subset (SUBSETS order) and clustered by question.
    In each subset its C clusters are drawn C times with replacement; every answer of a drawn cluster enters with the
    cluster's multiplicity. Clusters are numbered by first appearance, so with singleton clusters this is the
    answer-level stratified bootstrap of the same rng stream. A cluster spanning two subsets is an error."""
    subsets, clusters = np.asarray(subsets), np.asarray(clusters)
    seen = {}
    for c, s in zip(clusters, subsets):
        if seen.setdefault(c, s) != s:
            raise ValueError(f'cluster {c!r} spans subsets {seen[c]} and {s}')
    counts = np.zeros((draws, len(subsets)))
    for s in SUBSETS:
        m = np.flatnonzero(subsets == s)
        if not len(m):
            continue
        _, first, inv = np.unique(clusters[m], return_index=True, return_inverse=True)
        rank = np.empty(len(first), int); rank[np.argsort(first, kind='stable')] = np.arange(len(first))
        cid, C = rank[inv.ravel()], len(first)
        pick = rng.choice(C, size=(draws, C))
        mult = np.zeros((draws, C)); np.add.at(mult, (np.repeat(np.arange(draws), C), pick.ravel()), 1)
        counts[:, m] = mult[:, cid]
    return counts


def prm_success(boot_rows, model, contrast=PRIMARY, population='scored'):
    """E1 and E2 for one PRM from the bootstrap rows (dicts with model, population, contrast, metric, delta, ci_low,
    ci_high). E1: within-AUC difference with CI low > 0. E2: average official F1 (unscored answers counted as misses)
    difference with CI low > E2_MARGIN. Missing rows give None, and the PRM does not succeed."""
    name = f'{contrast[0]} - {contrast[1]}'
    get = {r['metric']: r for r in boot_rows
           if r['model'] == model and r['population'] == population and r['contrast'] == name}
    e1r, e2r = get.get('within_auc'), get.get('avg_f1_failures_wrong')
    def ok(r, bound):
        return None if r is None or not np.isfinite(r['ci_low']) else bool(r['ci_low'] > bound)
    e1, e2 = ok(e1r, 0.0), ok(e2r, E2_MARGIN)
    pick = lambda r: None if r is None else {k: float(r[k]) for k in ('delta', 'ci_low', 'ci_high')}
    return {'E1': e1, 'E2': e2, 'success': bool(e1 and e2), 'within_auc': pick(e1r), 'avg_f1_failures_wrong': pick(e2r)}


def verdict(per_prm):
    """Overall verdict, section 4: success in at least N_NEED of the N_PRMS PRMs, and at least one of them in STRONG.
    per_prm: {model: prm_success(...)}. With fewer than N_PRMS registered PRMs evaluated the verdict is INCOMPLETE."""
    wins = sorted(m for m, r in per_prm.items() if r['success'])
    missing = [m for m in REGISTERED_PRMS if m not in per_prm]
    passed = len(wins) >= N_NEED and any(m in STRONG for m in wins)
    return {'verdict': 'INCOMPLETE' if missing else ('SUCCESS' if passed else 'FAIL'), 'n_success': len(wins),
            'successes': wins, 'strong_success': [m for m in wins if m in STRONG], 'missing_prms': missing,
            'rule': f'>= {N_NEED} of {N_PRMS} PRMs with E1 and E2, at least one of {list(STRONG)}'}
