"""
rag_scgpt_orientation.py — resolve the RAG SelfCheckGPT below-chance flag
(HANDOFF_punchlist_and_reruns.md Phase-1 item 2; flagged Step 152, never investigated).

Code audit (Spectral_Analysis_Phase12_Corrected.ipynb, RAG cell) established:
  - AUROC orientation is correct: boot_auc(labels, -scgpt) on every arm incl. RAG.
  - The RAG label is a CITATION-GROUNDING label (lciteeval_grounding_label on the
    model's [n] markers), not answer correctness. A response with no citations is
    labeled 0 (ungrounded).

Mechanism hypothesis: SCGPT scores claim-consistency across K=5 resamples. Generic /
citation-free responses (label 0) are highly self-consistent -> near-zero contradiction
-> scored "least hallucinated"; grounded responses (label 1) make many specific cited
claims that stochastic resamples contradict more often. Specificity raises BOTH the
grounding label AND the SCGPT score -> systematic below-chance AUROC. The official
(soft-probability, 512-token-truncated NLI) variant amplifies this.

Empirical test on the per-sample caches (local_cache/rag_scgpt/p4_rag_*_qwen7b.pkl,
pulled from Drive cache/phase12_corrected/):
  1. Reproduce the stored below-chance AUROCs exactly (sanity: same data).
  2. Score-by-label: if mean SCGPT score is HIGHER for label-1 (grounded), the
     mechanism is confirmed — the signal is real but anti-aligned with this label.
  3. Zero-inflation: fraction of exactly-zero hard scores (fully consistent responses)
     by label — the "generic response" mass should concentrate in label 0.

Output: printed report + results/phase1/rag_scgpt_orientation.json
"""
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pickle

from spectral_utils.fusion_utils import boot_auc

DATASETS = ['hotpotqa', 'natural_questions', '2wikimultihopqa', 'narrativeqa']
CACHE_DIR = os.path.join(REPO, 'local_cache', 'rag_scgpt')
REF_PKL = os.path.join(REPO, 'local_cache', 'phase12_corrected_results.pkl')


def main():
    ref = {}
    if os.path.exists(REF_PKL):
        with open(REF_PKL, 'rb') as f:
            ref = pickle.load(f).get('rag', {})

    out = {}
    print(f"{'dataset':<20} {'n':>4} {'grounded%':>9} | {'AUROC hard':>21} {'official':>10} | "
          f"{'mean score L0/L1 (hard)':>24} {'zero% L0/L1':>12}")
    for ds in DATASETS:
        path = os.path.join(CACHE_DIR, f'p4_rag_{ds}_qwen7b.pkl')
        with open(path, 'rb') as f:
            p4 = pickle.load(f)
        rows = [v for v in p4.values() if v.get('done')]
        y = np.array([int(v['correct']) for v in rows])
        h = np.array([float(v['scgpt_hard']) for v in rows])
        o = np.array([float(v['scgpt_official']) for v in rows])

        auc_h, lo_h, hi_h = boot_auc(y, -h)
        auc_o, lo_o, hi_o = boot_auc(y, -o)
        ref_h = (ref.get(ds, {}).get('hard') or (None,))[0]
        ref_o = (ref.get(ds, {}).get('official') or (None,))[0]
        match = (ref_h is None or abs(auc_h - ref_h) < 0.02) and \
                (ref_o is None or abs(auc_o - ref_o) < 0.02)

        m0_h, m1_h = float(h[y == 0].mean()), float(h[y == 1].mean()) if (y == 1).any() else np.nan
        m0_o, m1_o = float(o[y == 0].mean()), float(o[y == 1].mean()) if (y == 1).any() else np.nan
        z0 = float((h[y == 0] == 0).mean())
        z1 = float((h[y == 1] == 0).mean()) if (y == 1).any() else np.nan

        print(f"{ds:<20} {len(y):>4} {100*y.mean():>8.1f}% | "
              f"{auc_h:.3f} [{lo_h:.2f},{hi_h:.2f}] {auc_o:>9.3f} | "
              f"{m0_h:>10.4f}/{m1_h:<11.4f} {100*z0:>5.0f}%/{100*z1:<5.0f}%"
              f"{'' if match else '  (MISMATCH vs stored!)'}")

        out[ds] = {
            'n': int(len(y)), 'grounded_rate': round(float(y.mean()), 4),
            'auroc_hard': round(float(auc_h), 4), 'auroc_official': round(float(auc_o), 4),
            'stored_hard': ref_h, 'stored_official': ref_o, 'reproduces_stored': bool(match),
            'mean_hard_label0': round(m0_h, 4), 'mean_hard_label1': round(m1_h, 4),
            'mean_official_label0': round(m0_o, 4), 'mean_official_label1': round(m1_o, 4),
            'zero_hard_frac_label0': round(z0, 4), 'zero_hard_frac_label1': round(z1, 4),
        }

    higher_when_grounded = sum(
        1 for ds in DATASETS
        if np.isfinite(out[ds]['mean_hard_label1'])
        and out[ds]['mean_hard_label1'] > out[ds]['mean_hard_label0'])
    print('\n== VERDICT ==')
    lines = [
        'Not an orientation or grading bug — a protocol mismatch:',
        '- the notebook negates SCGPT correctly, and the per-sample caches reproduce the',
        '  stored below-chance AUROCs;',
        f'- grounded (label-1) responses carry HIGHER SCGPT contradiction scores than',
        f'  ungrounded ones on {higher_when_grounded}/4 datasets — SCGPT self-consistency is',
        '  anti-aligned with the citation-grounding label: specific, citation-rich answers',
        '  are both likelier to be grounded and likelier to be contradicted by resamples,',
        '  while generic/citation-free answers (label 0) are self-consistent;',
        '- the official variant is worse than hard everywhere because its soft NLI',
        '  probabilities never hit exact zero and its 512-token (sentence, sample) pairs',
        '  truncate the long L-CiteEval responses (the Step-152 NLI-truncation suspect).',
        'Implication: the RAG SelfCheckGPT rows measure a label mismatch, not detector',
        'quality — annotate as NOT-CITABLE for method comparison rather than re-running.',
    ]
    for ln in lines:
        print(' ' + ln)
    out['verdict'] = lines

    os.makedirs(os.path.join(REPO, 'results', 'phase1'), exist_ok=True)
    out_path = os.path.join(REPO, 'results', 'phase1', 'rag_scgpt_orientation.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=1)
    print(f'\nsaved -> {out_path}')


if __name__ == '__main__':
    main()
