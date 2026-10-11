"""Official-format PRMScore for label-free step scores on PRMBench (advisor page, 2026-10-15 meeting).

Turns frozen step scores into binary per-step labels with a label-free flag rule, then scores them
with the repo's port of PRMBench's own evaluator (spectral_utils/prmbench.py::prmbench_evaluate,
a port of mr_eval prmtest_classified). No label enters any rule or threshold.

Flag rules (all thresholds are 80th percentiles over every scored step; no labels):
  R0      flag a step if its answer-z fused score S >= q80(S). About 20% of every answer.
  R1      flag a step if its global-scale score G >= q80(G). G = equal mean of the DS survivors,
          each standardized once over all steps (decision_rule_v1 R1_global).
  R2      count from R1 (n_i = R1 flags in answer i), placement by the S ranking (decision_rule_v1).
  R1F     as R1, but on the method's own global-scale fused score G__<method> (learned weights applied
          to globally standardized channels). Only when the input has it. New; never run before.
  TOP1/2  exactly 1 or 2 flags per answer by S. Fixed-count baselines; TOP2 matches PRMBench's
          ~2 planted errors per answer (answer_gate_v1), so it is a construction-prior control.

PRMScore = 0.5 * (F1 on correct steps + F1 on error steps), the official definition. The pooled
"Overall" excludes the synthetic `correct` control rows, as upstream. Category scores are the mean
of member-class PRMScores; multi_solutions has no error steps (negative F1 undefined/0), so it is
reported but left out of the Sensitivity mean, and this is stated in the output.
redundency/circular go through the validity fallback (we have no redundancy head): a declared
adaptation, recorded in the output.

Input NPZ contract (one bank):
  offsets      int  (A+1,)  step offsets per answer, in the same order as `idx`
  idx          str  (A,)    PRMBench idx (e.g. "circular_prm_test_p1_29", "correct_...")
  S__<method>  float (N,)   answer-z fused step score, higher = more likely an error
  G            float (N,)   global-scale equal-mean score for R1/R2 (optional; R1/R2 skipped without it)
  G__<method>  float (N,)   global-scale fused score for R1F (optional)
Metadata: --meta JSON list of {idx, classification, error_steps} (from spectral_utils.prmbench.load_prmbench),
or omit it and the script calls load_prmbench() (needs Hugging Face access).

Usage: python official_prmscore_v1.py --repo <hallucination_detection> --scores bank_N16.npz --bank N16 \
          --out out_dir [--meta meta.json] [--prm-manifest dataset_cache/four_localization/prmbench_qwen25math7b_full/manifest.json]
       python official_prmscore_v1.py --repo <repo> --smoke
"""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np

CLASSES = ['redundency', 'circular', 'counterfactual', 'step_contradiction', 'domain_inconsistency', 'confidence',
           'missing_condition', 'deception', 'multi_solutions']
CATS = {'simplicity': ['redundency', 'circular'],
        'soundness': ['counterfactual', 'step_contradiction', 'domain_inconsistency', 'confidence'],
        'sensitivity': ['missing_condition', 'deception']}  # multi_solutions reported, not averaged (no error steps)


def flags_threshold(x, q=0.8):
    return x >= np.quantile(x, q)


def flags_count(S, off, counts):
    out = np.zeros(len(S), bool)
    for i, (a, b) in enumerate(zip(off[:-1], off[1:])):
        k = int(min(counts[i], b - a))
        if k > 0:
            order = np.lexsort((np.arange(b - a), -S[a:b]))  # highest score first, ties: earlier step first
            out[a + order[:k]] = True
    return out


def rules(Z, off, method):
    S = Z['S__' + method]; R = {'R0': flags_threshold(S)}
    n = np.diff(off)
    R['TOP1'] = flags_count(S, off, np.ones(len(n), int))
    R['TOP2'] = flags_count(S, off, np.full(len(n), 2))
    if 'G' in Z:
        r1 = flags_threshold(Z['G']); R['R1'] = r1
        R['R2'] = flags_count(S, off, np.add.reduceat(r1.astype(int), off[:-1]) * (n > 0))
    if 'G__' + method in Z:
        R['R1F'] = flags_threshold(Z['G__' + method])
    return R


def prmscore(f1, nf1):
    return (f1 + nf1) / 2 if f1 >= 0 and nf1 >= 0 else float('nan')


def score(prmbench, flags, off, idx, meta):
    # labels[i] == 1 means VALID, so a flag (error) is label 0
    preds = [{'idx': str(u), 'labels': (~flags[a:b]).astype(int).tolist()} for u, a, b in zip(idx, off[:-1], off[1:])]
    res = prmbench.prmbench_evaluate(preds, meta)
    t, bc = res['total'], res['by_classification']
    row = {'overall': prmscore(t['f1'], t['negative_f1']), 'f1_correct': t['f1'], 'f1_error': t['negative_f1']}
    for c in CLASSES:
        if c in bc['f1']:
            row[c] = prmscore(bc['f1'][c], bc['negative_f1'][c]) if c != 'multi_solutions' else bc['f1'][c]
    for k, mem in CATS.items():
        row[k] = float(np.nanmean([row[c] for c in mem if c in row]))
    row['used_redundancy_head'] = res['used_redundancy_head']; row['n_scored'] = res['n_predictions_scored']
    return row


def describe(flags, off, idx):
    per = np.add.reduceat(flags.astype(int), off[:-1])
    ctrl = np.array([str(u).startswith('correct_') for u in idx])
    return {'step_flag_rate': float(flags.mean()), 'answers_with_flag': float((per > 0).mean()),
            'controls_with_flag': float((per[ctrl] > 0).mean()) if ctrl.any() else float('nan')}


def prm_reference(manifest):
    r = json.load(open(manifest))['results']; t, bc = r['total'], r['by_classification']
    row = {'overall': prmscore(t['f1'], t['negative_f1']), 'f1_correct': t['f1'], 'f1_error': t['negative_f1']}
    for c in CLASSES:
        row[c] = prmscore(bc['f1'][c], bc['negative_f1'][c]) if c != 'multi_solutions' else bc['f1'][c]
    for k, mem in CATS.items():
        row[k] = float(np.nanmean([row[c] for c in mem]))
    return row


def run(prmbench, Z, meta, bank, out, prm_manifest=None):
    off = Z['offsets'].astype(np.int64); idx = Z['idx'].astype(str)
    assert off[0] == 0 and off[-1] == len(Z[[k for k in Z if k.startswith('S__')][0]]), 'offsets do not match scores'
    known = {m['idx'] for m in meta}; missing = [u for u in idx if u not in known]
    assert not missing, f'{len(missing)} idx not in PRMBench meta, e.g. {missing[:3]}'
    nst = {m['idx']: m.get('n_steps') for m in meta}
    bad = [u for u, n in zip(idx, np.diff(off)) if nst[u] is not None and nst[u] != n]
    assert not bad, f'{len(bad)} answers whose step count differs from the meta, e.g. {bad[:3]}'
    rows = []
    for method in sorted(k[3:] for k in Z if k.startswith('S__')):
        for rule, fl in rules(Z, off, method).items():
            rows.append({'bank': bank, 'method': method, 'rule': rule, **score(prmbench, fl, off, idx, meta), **describe(fl, off, idx)})
    if prm_manifest:
        rows.append({'bank': '-', 'method': 'Qwen2.5-Math-PRM-7B (supervised)', 'rule': 'native 0.5', **prm_reference(prm_manifest)})
    out.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / f'OFFICIAL_PRMSCORE_{bank}.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)
    json.dump({'bank': bank, 'rows': rows, 'note': __doc__.split('Input NPZ')[0].strip()}, open(out / f'OFFICIAL_PRMSCORE_{bank}.json', 'w'), indent=1, default=float)
    return rows


def smoke(prmbench):
    """Synthetic check: perfect scores give PRMScore 1 under an oracle count; rules respect the label-free contract."""
    rng = np.random.default_rng(0); meta, sizes, err = [], [], []
    for j in range(60):
        c = CLASSES[j % 8]; n = int(rng.integers(4, 9)); e = sorted(rng.choice(np.arange(1, n + 1), 2, replace=False).tolist())
        meta.append({'idx': f'{c}_s{j}', 'classification': c, 'error_steps': e}); sizes.append(n); err.append(e)
    for j in range(10):
        meta.append({'idx': f'correct_s{j}', 'classification': 'correct', 'error_steps': []}); sizes.append(5); err.append([])
    off = np.r_[0, np.cumsum(sizes)]; lab = np.zeros(off[-1], bool)
    for i, e in enumerate(err):
        for s in e: lab[off[i] + s - 1] = True
    S = lab + 0.01 * rng.random(len(lab))
    Z = {'offsets': off, 'idx': np.array([m['idx'] for m in meta]), 'S__perfect': S, 'G': S.copy(), 'G__perfect': S.copy()}
    rows = {r['rule']: r for r in run(prmbench, Z, meta, 'SMOKE', Path('/tmp/official_prmscore_smoke'))}
    assert abs(rows['TOP2']['overall'] - 1) < 1e-12, rows['TOP2']['overall']           # 2 errors per answer, perfect ranking
    assert rows['R1']['controls_with_flag'] == 0 and rows['R0']['controls_with_flag'] == 0  # controls carry no error signal
    assert abs(rows['R2']['step_flag_rate'] - rows['R1']['step_flag_rate']) < 1e-12        # R2 keeps R1's counts
    assert rows['R1F']['overall'] == rows['R1']['overall']                                  # same score, same flags
    assert all(0 <= v['overall'] <= 1 for v in rows.values())
    print('smoke ok', {k: round(v['overall'], 4) for k, v in rows.items()})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True); ap.add_argument('--scores'); ap.add_argument('--bank', default='N16')
    ap.add_argument('--out', default='official_prmscore_v1'); ap.add_argument('--meta'); ap.add_argument('--prm-manifest')
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    # load the evaluator module directly: the spectral_utils package import pulls in torch
    import importlib.util
    spec = importlib.util.spec_from_file_location('prmbench', Path(a.repo) / 'spectral_utils' / 'prmbench.py')
    prmbench = importlib.util.module_from_spec(spec); spec.loader.exec_module(prmbench)
    if a.smoke:
        smoke(prmbench); sys.exit()
    Z = dict(np.load(a.scores, allow_pickle=False))
    meta = json.load(open(a.meta)) if a.meta else prmbench.load_prmbench()[0]
    if isinstance(meta, dict):  # thinkfix_rescore_v1 export wraps the rows: {"source": ..., "rows": [...]}
        meta = meta['rows']
    run(prmbench, Z, meta, a.bank, Path(a.out), a.prm_manifest)
