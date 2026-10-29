"""evaluate_sla.py — Evaluate ProcessBench Step-level Localization Accuracy (SLA) under the GAP decision rule.

Compares TELEM16 (TELE_N16) and controls against published results from "Mind the Gap"
(Chen et al., ICML 2026, Table 3).

Under the Mind the Gap protocol:
- SLA is computed ONLY on answers containing an error (y != -1); clean answers are not scored.
- Exact match: predicted step == gold first erroneous step (tolerance = 0).
- Subsets: GSM8K (207 error answers), MATH (594), OlympiadBench (661), Omni-MATH (759). Total = 2,221.

Under our GAP rule:
- Within each answer of length m, scores are sorted descending.
- The maximum gap between consecutive sorted scores for the top km = min(ceil(m/2), m-1) steps defines k.
- The top-k highest scoring steps are flagged.
- The predicted first error is the chronologically first flagged step: min { i : step i is flagged }.
"""
import os, sys, json, csv
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

# Discover scores dir and gold file
SCORES_CANDIDATES = [
    os.environ.get('PB_SCORES_DIR'),
    str(ROOT / 'results/prm_boost_processbench_v1/scores'),
    r'C:\Users\omris\TAU\wt_pb_integ\results\prm_boost_processbench_v1\scores',
]
SCORES_DIR = next((p for p in SCORES_CANDIDATES if p and Path(p).exists()), None)

GOLD_CANDIDATES = [
    os.environ.get('PB_GOLD_PATH'),
    str(ROOT / 'results/prm_boost_processbench_v1/local/processbench_gold.json'),
    r'C:\Users\omris\TAU\wt_pb_integ\results\prm_boost_processbench_v1\local\processbench_gold.json',
]
GOLD_PATH = next((p for p in GOLD_CANDIDATES if p and Path(p).exists()), None)

SUBSETS = ('gsm8k', 'math', 'olympiadbench', 'omnimath')

# Published numbers from Chen et al., ICML 2026, Table 3 (Shannon Drop)
MIND_THE_GAP = {
    'Qwen3-4B (Shannon Drop)': {
        'gsm8k': 43.42, 'math': 32.03, 'olympiadbench': 43.06, 'omnimath': 38.04,
        'macro': 39.14, 'overall': 38.43
    },
    'Qwen3-8B (Shannon Drop)': {
        'gsm8k': 46.11, 'math': 32.90, 'olympiadbench': 41.52, 'omnimath': 37.04,
        'macro': 39.39, 'overall': 38.11
    }
}

def compute_gap_first_flag(scores, lengths):
    """Compute predicted step under GAP rule:
    1. Chronological first flagged step: min { i in flagged }
    2. Argmax (top-1) highest scoring step
    """
    starts = np.r_[0, np.cumsum(lengths)]
    preds_chrono = []
    preds_argmax = []
    
    for a, b in zip(starts[:-1], starts[1:]):
        s = scores[a:b]
        m = len(s)
        if m == 0:
            preds_chrono.append(-1)
            preds_argmax.append(-1)
            continue
        if m == 1:
            preds_chrono.append(0)
            preds_argmax.append(0)
            continue
            
        order = np.argsort(-s, kind='stable')
        srt = s[order]
        km = min(int(np.ceil(m / 2)), m - 1)
        gaps = srt[:-1] - srt[1:]
        k = np.argmax(gaps[:km]) + 1
        
        flagged = order[:k]
        preds_chrono.append(int(np.min(flagged)))
        preds_argmax.append(int(order[0]))
        
    return np.array(preds_chrono), np.array(preds_argmax)

def main():
    if not SCORES_DIR or not GOLD_PATH:
        print(f"Error: Could not locate scores dir ({SCORES_DIR}) or gold path ({GOLD_PATH})")
        sys.exit(1)
        
    gold_list = json.load(open(GOLD_PATH))
    gold = {x['uid']: x for x in gold_list}
    
    models = ['qwen_prm', 'qwen_prm_72b', 'pure_prm', 'rlhflow_prm', 'math_shepherd']
    arms_to_eval = [
        ('TELE_N16', 'score__TELE_N16'),
        ('MEAN_N16', 'score__MEAN_N16'),
        ('CHAIN_N16', 'score__CHAIN_N16'),
        ('TELE_N13', 'score__TELE_N13'),
        ('HALF_N16', 'score__HALF_N16'),
        ('PRM', 'score__PRM'),
    ]
    
    results = []
    
    # Add Mind the Gap baseline rows
    for comp_name, comp_vals in MIND_THE_GAP.items():
        results.append({
            'model': 'Mind the Gap paper',
            'arm': comp_name,
            'rule': 'Shannon Drop (ICML 2026 Table 3)',
            'macro_sla': comp_vals['macro'],
            'overall_sla': comp_vals['overall'],
            'gsm8k_sla': comp_vals['gsm8k'],
            'math_sla': comp_vals['math'],
            'olympiadbench_sla': comp_vals['olympiadbench'],
            'omnimath_sla': comp_vals['omnimath'],
        })
        
    print("=" * 110)
    print("ProcessBench Step-level Localization Accuracy (SLA) under GAP Rule vs Mind the Gap Table 3")
    print("=" * 110)
    print(f"{'Model':<16} | {'Arm':<12} | {'Rule':<14} | {'Macro SLA':>9} | {'Overall':>7} | {'GSM8K':>6} | {'MATH':>6} | {'Olympiad':>8} | {'OmniMATH':>8}")
    print("-" * 110)
    
    for r in results:
        print(f"{r['model']:<16} | {r['arm']:<12} | {r['rule']:<14} | {r['macro_sla']:>8.2f}% | {r['overall_sla']:>6.2f}% | {r['gsm8k_sla']:>5.2f}% | {r['math_sla']:>5.2f}% | {r['olympiadbench_sla']:>7.2f}% | {r['omnimath_sla']:>7.2f}%")
    print("-" * 110)

    for m in models:
        f_path = Path(SCORES_DIR) / f'SCORES_CHAIN_{m}.npz'
        if not f_path.exists():
            continue
        d = np.load(f_path)
        uids = [str(u) for u in d['uids']]
        lens = d['nonempty_lengths'].astype(int)
        
        G = [gold[u] for u in uids]
        labels = np.array([g['label'] for g in G])
        subs = np.array([g['subset'] for g in G])
        err_mask = (labels != -1)
        
        for arm_name, key in arms_to_eval:
            if key not in d:
                continue
            sc = d[key]
            p_chrono, p_argmax = compute_gap_first_flag(sc, lens)
            
            # Chronological first flag
            sub_sla_c = {}
            for s in SUBSETS:
                sm = err_mask & (subs == s)
                sub_sla_c[s] = 100.0 * np.mean(p_chrono[sm] == labels[sm])
            macro_c = np.mean(list(sub_sla_c.values()))
            overall_c = 100.0 * np.mean(p_chrono[err_mask] == labels[err_mask])
            
            row_c = {
                'model': m,
                'arm': arm_name,
                'rule': 'GAP_first_flag',
                'macro_sla': round(macro_c, 2),
                'overall_sla': round(overall_c, 2),
                'gsm8k_sla': round(sub_sla_c['gsm8k'], 2),
                'math_sla': round(sub_sla_c['math'], 2),
                'olympiadbench_sla': round(sub_sla_c['olympiadbench'], 2),
                'omnimath_sla': round(sub_sla_c['omnimath'], 2),
            }
            results.append(row_c)
            print(f"{m:<16} | {arm_name:<12} | {'GAP_first':<14} | {macro_c:>8.2f}% | {overall_c:>6.2f}% | {sub_sla_c['gsm8k']:>5.2f}% | {sub_sla_c['math']:>5.2f}% | {sub_sla_c['olympiadbench']:>7.2f}% | {sub_sla_c['omnimath']:>7.2f}%")
            
            # Argmax (top-1) control
            if arm_name in ('TELE_N16', 'PRM'):
                sub_sla_a = {}
                for s in SUBSETS:
                    sm = err_mask & (subs == s)
                    sub_sla_a[s] = 100.0 * np.mean(p_argmax[sm] == labels[sm])
                macro_a = np.mean(list(sub_sla_a.values()))
                overall_a = 100.0 * np.mean(p_argmax[err_mask] == labels[err_mask])
                
                row_a = {
                    'model': m,
                    'arm': arm_name,
                    'rule': 'Argmax_top1',
                    'macro_sla': round(macro_a, 2),
                    'overall_sla': round(overall_a, 2),
                    'gsm8k_sla': round(sub_sla_a['gsm8k'], 2),
                    'math_sla': round(sub_sla_a['math'], 2),
                    'olympiadbench_sla': round(sub_sla_a['olympiadbench'], 2),
                    'omnimath_sla': round(sub_sla_a['omnimath'], 2),
                }
                results.append(row_a)
                print(f"{m:<16} | {arm_name:<12} | {'Argmax_top1':<14} | {macro_a:>8.2f}% | {overall_a:>6.2f}% | {sub_sla_a['gsm8k']:>5.2f}% | {sub_sla_a['math']:>5.2f}% | {sub_sla_a['olympiadbench']:>7.2f}% | {sub_sla_a['omnimath']:>7.2f}%")
        print("-" * 110)
        
    out_csv = HERE / 'RESULTS.csv'
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'model', 'arm', 'rule', 'macro_sla', 'overall_sla',
            'gsm8k_sla', 'math_sla', 'olympiadbench_sla', 'omnimath_sla'
        ])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved results to {out_csv}")

if __name__ == '__main__':
    main()
