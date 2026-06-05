# Handoff — Next Session

## Your first task: build the notebook and tell me when done

Build `Spectral_Analysis_Consolidated_Results_LSML_v2.ipynb` — a CPU-only Colab notebook that re-runs L-SML with pre-oriented classifiers on all cached features from phases 1–11. Then rebuild the HTML comparison table.

---

## Context (read this before starting)

### What L-SML is
L-SML (Latent Spectral Meta-Learner) is from Jaffé, Fetaya, Nadler (2016), a continuation of Parisi, Nadler, Kluger (PNAS 2014). Algorithm 2 from the 2016 paper: detect dependent classifier groups → fuse within each group via SML → fuse across groups via SML. No labels anywhere.

### Why we are re-running
The existing Step 107 notebook (`Spectral_Analysis_Consolidated_Results_LSML.ipynb`) ran `sml_unsupervised` which resolves feature sign via Paper 2 assumption (iii) — no pre-orientation. We now want to pre-orient each feature using `FEATURE_SIGNS` (offline consensus from Step 110) before binarizing. This gives the algorithm correctly-oriented +/-1 inputs rather than relying on assumption (iii) to guess. Same algorithm, better inputs. Still fully unsupervised at test time.

### The correct pipeline
```python
from spectral_utils import binarize_classifiers, FEAT_NAMES
from spectral_utils.fusion_utils import lsml_fuse, boot_auc

FEATURE_SIGNS = {
    'epr': -1, 'trace_length': 1, 'spectral_entropy': -1,
    'low_band_power': -1, 'high_band_power': -1, 'hl_ratio': -1,
    'dominant_freq': -1, 'spectral_centroid': -1,
    'stft_max_high_power': -1, 'stft_spectral_entropy': -1,
    'rpdi': -1, 'sw_var_peak': -1,
    'pe_mean': -1, 'hurst_exponent': 1,
    'cusum_max': -1, 'cusum_shift_idx': 1,
}
# Convention: +1 = higher value means more likely correct; -1 = higher value means hallucination

# For each (domain, model, dataset) cell:
binary = binarize_classifiers(feats_dict, FEATURE_SIGNS)
scores, meta = lsml_fuse(*binary.values())
auc, lo, hi = boot_auc(labels, scores)
# If auc < 0.5, flip: auc, lo, hi = boot_auc(labels, -scores)
```

### Where the cached features live (Google Drive)
All at `hallucination_detection/consolidated_results/`:
- `math500_res.pkl` — MATH-500, multiple models
- `gsm8k_res.pkl` — GSM8K / Llama-3.1-8B
- `gpqa_res.pkl` — GPQA Diamond, multiple models
- `rag_feats_all.pkl` — RAG L-CiteEval, multiple models x 4 datasets
- `qa_res.pkl` — Factual QA Phase 9

Each pkl contains a dict: `{key: (feats_dict, labels)}` where `feats_dict = {feat_name: np.array}` and `labels = np.array` of 0/1.

### Verify master branch before building
The user said: "We merged all to master so you can use its version — verify me."
Before building: confirm `spectral_utils/fusion_utils.py` on master has `binarize_classifiers` and `lsml_fuse`. Run `git log --oneline -5` and check.

---

## Notebook spec

**Name**: `Spectral_Analysis_Consolidated_Results_LSML_v2.ipynb`

**Cell structure**:
1. Clone master + pip install + imports (`binarize_classifiers`, `lsml_fuse`, `boot_auc`, `FEAT_NAMES`)
2. Mount Drive + define paths + define `FEATURE_SIGNS` + define `run_lsml_v2(feats_dict, labels, key)` helper
3. MATH-500 — load `math500_res.pkl`, run per key, save `lsml_v2_math500_res.pkl`
4. GSM8K — load `gsm8k_res.pkl`, run per key, save `lsml_v2_gsm8k_res.pkl`
5. GPQA — load `gpqa_res.pkl`, run per key, save `lsml_v2_gpqa_res.pkl`
6. RAG — load `rag_feats_all.pkl`, run per key, save `lsml_v2_rag_res.pkl`
7. Factual QA — load `qa_res.pkl`, run per key, save `lsml_v2_qa_res.pkl`
8. Summary table — combine all, print per-domain sorted table, save `lsml_v2_results_all.pkl` + `lsml_v2_summary.csv`

**The `run_lsml_v2` helper**:
```python
def run_lsml_v2(feats_dict, labels, key):
    labels = np.asarray(labels, dtype=int)
    if len(set(labels.tolist())) < 2:
        return None
    feat_names_valid = [f for f in FEAT_NAMES if f in feats_dict]
    stack = np.column_stack([np.array(feats_dict[f], dtype=float) for f in feat_names_valid])
    valid = ~np.any(np.isnan(stack), axis=1)
    if valid.sum() < 20:
        return None
    fd_valid = {f: np.array(feats_dict[f], dtype=float)[valid] for f in feat_names_valid}
    lbl_valid = labels[valid]
    binary = binarize_classifiers(fd_valid, FEATURE_SIGNS)
    scores, meta = lsml_fuse(*[binary[f] for f in feat_names_valid])
    auc, lo, hi = boot_auc(lbl_valid, scores)
    if auc < 0.5:
        auc, lo, hi = boot_auc(lbl_valid, -scores)
    return {'auc': auc, 'lo': lo, 'hi': hi, 'K': meta['K'],
            'n': int(valid.sum()), 'n_pos': int(lbl_valid.sum()), 'key': key}
```

**Save pattern for every domain cell**:
```python
FORCE = False
OUT_PATH = f'{OUT_DIR}/lsml_v2_math500_res.pkl'
if not FORCE and os.path.exists(OUT_PATH):
    with open(OUT_PATH, 'rb') as f: MATH500_V2 = pickle.load(f)
    print(f'Loaded {len(MATH500_V2)} keys from cache')
else:
    MATH500_V2 = {}
    for key, (fd, lbl) in MATH500_FEATS.items():
        MATH500_V2[key] = run_lsml_v2(fd, lbl, key)
        with open(OUT_PATH, 'wb') as f: pickle.dump(MATH500_V2, f)
    print(f'Done: {len(MATH500_V2)} keys')
```

---

## After the notebook runs — rebuild the HTML

Rebuild `Phase12_Comparison_Results.html` with:
- **Our method rows**: v2 L-SML numbers only (from `lsml_v2_summary.csv`)
- **Competitor rows**: from PROGRESS.md "Available competitor numbers" table
- **Rules**: same-model, same-dataset, same-task ONLY. No cross-model comparisons.
- **Color coding**: blue = ours, green = competitor we computed ourselves, yellow = number from a paper
- **Structure**: one section per domain, one table per model within that domain
- **L-SML explanation box**: cite Jaffé-Fetaya-Nadler (2016) as the source of L-SML. Mention it is a continuation of Parisi-Nadler-Kluger (2014). Do NOT say we combine two methods — L-SML (2016) already contains SML (2014) as a subroutine.
- **No Step 100 supervised numbers anywhere**

---

## Other running experiments

- **Phase 13** (`Spectral_Analysis_MathComp_Phase13.ipynb`): L-SML vs EDIS on GSM8K+MATH500+AMC23+AIME24, Qwen2.5-Math-1.5B. Check Drive cache at `cache/mathcomp_phase13/`.
- **Phase 14** (`Spectral_Analysis_Phase14_GPQA_Comparison.ipynb`): L-SML vs VC/SC on GPQA Diamond, DeepSeek-R1-0528-Qwen3-8B. Check Drive cache.

When either finishes, add their results to the HTML as new sections.

---

## Start of session checklist
1. Run `/session-start`
2. `git log --oneline -5` — confirm master is up to date
3. Verify `binarize_classifiers` and `lsml_fuse` are in `spectral_utils/fusion_utils.py` on master
4. Build the notebook per the spec above
5. Tell the user when done and show the summary table
