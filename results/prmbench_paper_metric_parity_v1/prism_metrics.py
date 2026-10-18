"""PRISM (arXiv 2606.09078) metrics for our arms.

PRISM Table 2 (PRMBench, Qwen-PRM-7B baseline): FPR 69.34, FNR 4.27, Precision 89.40, TPR 95.36, TNR 30.66,
PRMScore 65.50. Table 1 calls TPR 'Pos-Acc' and TNR 'Neg-Acc'. The POSITIVE class is a CORRECT (valid) step, as
in the official PRMBench evaluator (labels==1 means valid). So:
  TPR = valid steps kept valid / valid steps        FNR = 1 - TPR
  TNR = error steps flagged / error steps           FPR = error steps passed as valid / error steps = 1 - TNR
  Precision = valid-predicted steps that are valid
Pooled over every step of the 6,211 non-control PRMBench answers (the synthetic 'correct' control rows are
excluded, as in the official PRMScore). Table 7 categories use the official per-classification PRMScore;
multi_solutions has no error steps, so its score is the valid-step F1 (matches the published 99.7 scale).
ProcessBench Table 6 (TPR = accuracy on erroneous samples, TNR = accuracy on correct samples, F1 = harmonic mean)
is computed for PRM_native only; the fused arms have no backbone telemetry on ProcessBench (GAP).
"""
import csv, json, pickle
import numpy as np
from common import (HERE, ROOT, ARMS, load_population, get_scores, flags_for, prmscore_counts, prmscore,
                    answer_z, within_auc)

CATS = [('NR', 'redundency'), ('NCL', 'circular'), ('ES', 'counterfactual'), ('SC', 'step_contradiction'),
        ('DC', 'domain_inconsistency'), ('CI', 'confidence'), ('PS', 'missing_condition'), ('DR', 'deception'),
        ('MS', 'multi_solutions')]
GROUPS = {'Simplicity': ['NR', 'NCL'], 'Soundness': ['ES', 'SC', 'DC', 'CI'], 'Sensitivity': ['PS', 'DR', 'MS']}
PRISM_T2 = {'Baseline (Qwen-PRM-7B)': dict(FPR=69.34, FNR=4.27, Precision=89.40, TPR=95.36, TNR=30.66, PRMScore=65.50),
            '+ Pairwise Loss (12k)': dict(FPR=61.13, FNR=7.10, Precision=88.96, TPR=92.41, TNR=38.86, PRMScore=67.00),
            '+ Pairwise Loss (12k) + CL': dict(FPR=60.58, FNR=7.20, Precision=90.4, TPR=92.79, TNR=39.42, PRMScore=67.30),
            '+ Pointwise Loss + Data Aug. (132k)': dict(FPR=67.00, FNR=6.00, Precision=89.60, TPR=94.00, TNR=33.50, PRMScore=65.60),
            'PRISM': dict(FPR=47.13, FNR=12.69, Precision=91.93, TPR=87.30, TNR=52.86, PRMScore=68.00)}
PRISM_T7 = {'Baseline': [65.5, 49.1, 55.0, 52.1, 71.7, 67.4, 66.3, 78.5, 71.0, 57.7, 69.1, 99.7, 75.5],
            '132k Curriculum-4 (best FPR)': [68.0, 53.5, 66.0, 59.7, 73.8, 67.5, 67.1, 75.6, 71.0, 62.4, 70.1, 98.7, 77.1],
            '132k Curriculum-3': [68.0, 52.3, 63.7, 58.0, 73.9, 68.5, 67.4, 77.4, 71.8, 61.1, 70.6, 99.3, 77.0]}
T7_COLS = ['Overall', 'NR', 'NCL', 'Simplicity', 'ES', 'SC', 'DC', 'CI', 'Soundness', 'PS', 'DR', 'MS', 'Sensitivity']
PRISM_T6 = {'gsm8k': (71.0, 96.4, 81.8), 'math': (67.0, 91.1, 77.2), 'omnimath': (54.8, 66.1, 83.4),
            'olympiadbench': (54.6, 85.5, 66.7)}


def rates(y, f, m):
    tp, fp, tn, fn = prmscore_counts(y, f, m)
    q = lambda a, b: 100 * a / b if b else float('nan')
    return dict(TPR=q(tp, tp + fn), TNR=q(tn, tn + fp), FPR=q(fp, fp + tn), FNR=q(fn, tp + fn),
                Precision=q(tp, tp + fp), PRMScore=prmscore(tp, fp, tn, fn), TP=tp, FP=fp, TN=tn, FN=fn)


def category_scores(y, f, off, cls):
    step_cls = np.repeat(cls, np.diff(off)); out = {}
    for short, c in CATS:
        m = step_cls == c
        tp, fp, tn, fn = prmscore_counts(y, f, m)
        out[short] = 100 * 2 * tp / (2 * tp + fp + fn) if c == 'multi_solutions' else prmscore(tp, fp, tn, fn)
    for g, mem in GROUPS.items():
        out[g] = float(np.mean([out[k] for k in mem]))
    return out


def threshold_curve(arm, s, y, off, m):
    """FPR/FNR (PRISM orientation) as the decision threshold moves; fused arms move the answer-z cut."""
    v = s if arm == 'PRM_native' else answer_z(s, off)
    ts = np.quantile(v[m], np.linspace(0.005, 0.995, 199))
    pts = []
    for t in ts:
        f = v >= t
        r = rates(y, f, m); pts.append((r['FPR'], r['FNR']))
    return pts


def processbench_native():
    out = {}
    for sub in ['gsm8k', 'math', 'olympiadbench', 'omnimath']:
        d = pickle.load(open(ROOT / f'dataset_cache/four_localization/pb_prm_qwen25math7b_full/pb_prm_{sub}.pkl', 'rb'))
        ok_err, ok_cor = [], []
        for r in d.values():
            rw = r['rewards']; rw = json.loads(rw) if isinstance(rw, str) else rw
            pred = next((i for i, v in enumerate(rw) if v < .5), -1); lab = int(r['label'])
            (ok_cor if lab == -1 else ok_err).append(pred == lab)
        e, c = 100 * np.mean(ok_err), 100 * np.mean(ok_cor)
        out[sub] = dict(TPR=e, TNR=c, F1=2 * e * c / (e + c), n_err=len(ok_err), n_cor=len(ok_cor))
    return out


def main():
    P = load_population(); S = get_scores(P)
    y, off, cls = P['y'], P['off'], P['cls']
    m = np.repeat(cls != 'correct', np.diff(off))
    res, cats, curves = {}, {}, {}
    for a in ARMS:
        f = flags_for(a, S[a], off)
        res[a] = rates(y, f, m); cats[a] = category_scores(y, f, off, cls)
        # PRISM's FNR (4.27) and Precision (89.40) are only reproduced when the synthetic 'correct' control rows
        # are pooled in; its TPR/TNR/PRMScore match the official non-control pool. Report both pools.
        res[a + '__with_control_rows'] = rates(y, f, np.ones_like(m))
        res[a]['flag_rate_nonctrl'] = float(f[m].mean())
        curves[a] = threshold_curve(a, S[a], y, off, m)
    # Operating point matched to PRISM's TPR (87.30): pick the global threshold whose TPR is closest.
    matched = {}
    for a in ARMS:
        v = S[a] if a == 'PRM_native' else answer_z(S[a], off)
        ts = np.quantile(v[m], np.linspace(0.60, 0.95, 701))
        best = min((abs(rates(y, v >= t, m)['TPR'] - 87.30), t) for t in ts)[1]
        matched[a] = rates(y, v >= best, m); matched[a]['threshold'] = float(best)
    pb = processbench_native()
    json.dump({'rates': res, 'categories': cats, 'curves': curves, 'matched_tpr_87_30': matched, 'processbench_native': pb,
               'n_answers_nonctrl': int((cls != 'correct').sum()), 'n_steps_nonctrl': int(m.sum())},
              open(HERE / 'PRISM_RESULTS.json', 'w'), indent=1)
    for a in list(res):
        print(a, {k: round(v, 2) for k, v in res[a].items() if k in ('FPR', 'FNR', 'Precision', 'TPR', 'TNR', 'PRMScore')})
    print('PB', pb)


if __name__ == '__main__':
    main()
