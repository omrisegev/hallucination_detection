# Notebook Plan: `Spectral_Analysis_GSM8K_Normalization_Ablation.ipynb`

**Purpose:** Two independent experiments in one notebook:
- **Part A** — GSM8K normalization ablation + Nadler-vs-average ablation. No new inference; loads from Phase 7 cache.
- **Part B** — GPQA Diamond with Qwen2.5-72B-Instruct. New inference needed; full pipeline from scratch.

**Drive base dir:** `/content/drive/MyDrive/epr_spectral_gsm8k_vs_lapei/`

---

## Part A — GSM8K Normalization & Fusion Ablation

### Cell A0 — Title (Markdown)

```
# Spectral Analysis — GSM8K Normalization Ablation + GPQA Diamond (72B)

**Part A:** Re-run Phase 7 feature fusion with z-score normalization fix.
Compare: unnormalized Nadler (Phase 7 baseline 76.0%) vs normalized Nadler vs simple average.

**Part B:** GPQA Diamond inference with Qwen2.5-72B-Instruct (4-bit).
Prior GPQA (7B models) was ~65% AUC — weak because model accuracy was only ~30%.
Qwen2.5-72B reaches ~65% accuracy on GPQA → expect meaningful spectral signal.
```

---

### Cell A1 — Install & Imports

**Actions:**
1. `!pip install git+https://github.com/omrisegev/hallucination_detection.git -q`
2. Standard imports: `numpy`, `matplotlib`, `scipy.stats`, `itertools`, `os`, `pickle`, `warnings`
3. sklearn: `roc_auc_score`
4. spectral_utils imports:
   - `from spectral_utils import load_cache, save_cache, extract_all_features, sw_var_peak_with_window, FEAT_NAMES, zscore, boot_auc, nadler_fuse, simple_average_fusion, best_nadler_on`
5. Set matplotlib style: `plt.style.use('seaborn-v0_8-whitegrid')`, figsize defaults, color palette

**Expected output:** `spectral_utils 0.1.0 loaded`

---

### Cell A2 — Mount Drive + Config

**Actions:**
1. `from google.colab import drive; drive.mount('/content/drive')`
2. Define paths:
   ```python
   RUN_DIR  = '/content/drive/MyDrive/epr_spectral_gsm8k_vs_lapei/Llama-3.1-8B-Instruct__gsm8k_T1.0/'
   PLOT_DIR = '/content/drive/MyDrive/epr_spectral_gsm8k_vs_lapei/plots/'
   os.makedirs(PLOT_DIR, exist_ok=True)
   ```
3. Reference constants (LapEigvals paper numbers):
   ```python
   LAPEI_SUPERVISED   = 0.872  # Table 1, supervised
   LAPEI_UNSUPERVISED = 0.720  # AttentionScore, unsupervised
   PHASE7_UNNORM      = 0.760  # Our Phase 7 unnormalized result
   ```

**Expected output:** `Drive mounted. Plot dir ready.`

---

### Cell A3 — Load Phase 7 Data

**Actions:**
1. Load `phase7_results.pkl` → `results`
2. Load `inference_cache.pkl` → `cache`
3. Extract:
   - `labels = np.array(results['raw_labels'])`
   - `feat_arrays = {k: np.array(v) for k, v in results['feat_arrays'].items()}`
   - `feat_names = results['feat_names']`
   - `ablation_data = results['ablation']`  (list of {w, auc, lo, hi})
   - `raw_ents = [cache[i]['all_entropies'] for i in sorted(cache) if cache[i].get('done')]`
4. Print summary table:
   - n_samples, n_correct, accuracy
   - avg trace length
   - Phase 7 reported AUC (76.0%)
   - Best subset from Phase 7

**Expected output:**
```
Loaded 1319 samples | 1043 correct (79.1%) | avg trace 193.6 tok
Phase 7 unnormalized Nadler: 76.0% [72.5, 79.3]
Best subset: trace_length + low_band_power + stft_spectral_entropy + sw_var_peak
```

---

### Cell A4 — Individual Feature AUCs

**Actions:**
1. For each feature in `FEAT_NAMES`: call `boot_auc(labels, feat_arrays[fn])` and `boot_auc(labels, -feat_arrays[fn])`, keep best.
2. Store `feat_aucs = {fn: (auc, lo, hi, sign)}` sorted by AUC descending.
3. Print ranked table: feature | AUC | 95% CI | sign.

**Note:** Individual AUC is rank-invariant → identical whether normalized or not. This table is the baseline for the feature comparison plot.

**Expected output:**
```
sw_var_peak          73.9%  [70.5, 77.5]   sign=+1
trace_length         71.5%  [67.8, 75.0]   sign=-1
epr                  70.7%  [66.9, 74.6]   sign=-1
...
stft_max_high_power  52.7%  [48.9, 56.7]   sign=-1
```

---

### Cell A5 — Unnormalized Nadler (Reproduce Phase 7 Baseline)

**Purpose:** Confirm we can reproduce 76.0% from Phase 7 with the raw features already in `feat_arrays`. This validates that the loaded data is correct before comparing.

**Actions:**
1. Define `best_nadler_unnorm(feats_dict, feat_names_, labels_, max_size=4)` — an inline function that:
   - Orients by sign (same as spectral_utils)
   - Does NOT z-score
   - Applies ρ-filter (spearmanr, threshold 0.75)
   - Exhaustive subset search, `nadler_fuse`
   - Returns `(best_auc, best_lo, best_hi, best_subset, best_weights)`
2. Run it. Print result.
3. Assert result is within ±0.5pp of 76.0%.

**Expected output:**
```
Unnormalized Nadler: 76.0% [72.5, 79.3]
Best subset: trace_length + low_band_power + stft_spectral_entropy + sw_var_peak
✓ Matches Phase 7 reported result.
```

---

### Cell A6 — Normalized Nadler (spectral_utils with zscore)

**Purpose:** The key experiment. Run `best_nadler_on` from spectral_utils, which applies zscore inside.

**Actions:**
1. Call:
   ```python
   auc_norm, lo_norm, hi_norm, subset_norm = best_nadler_on(
       feat_arrays, feat_names, labels,
       max_size=4, label='GSM8K-normalized', compare_mean=True
   )
   ```
2. `compare_mean=True` → spectral_utils prints the Nadler Lift over simple average for the best subset.
3. Print summary:
   - Unnormalized: 76.0%
   - Normalized: X%
   - Delta: +/- Y pp
   - Best subset (may differ from unnormalized)
   - Nadler lift over simple average

**Expected output (values will be filled in after run):**
```
[GSM8K-normalized] 12 features, 12 informative, max_size=4 → 781 raw combos
  size=2: ...
  size=4: ...
Normalized Nadler: XX.X% [XX.X, XX.X]
Best subset: ...

Nadler Lift over simple average (subset: ...):
  Nadler : XX.X%  [XX.X, XX.X]
  Mean   : XX.X%  [XX.X, XX.X]
  Lift   : +X.X pp
```

---

### Cell A7 — All-Subset Comparison Scan

**Purpose:** For every valid subset (size 2–4, ρ-filtered), compute both Nadler AUC and simple-average AUC. This gives data for the scatter plot.

**Actions:**
1. Orient and z-score all features once (reuse sign from Cell A4):
   ```python
   sign_map = {fn: feat_aucs[fn][3] for fn in feat_names}
   oriented_z = {fn: zscore(feat_arrays[fn] * sign_map[fn]) for fn in feat_names}
   ```
2. Precompute Spearman ρ on z-scored features.
3. Identify informative features (individual AUC > 0.50).
4. Loop over all valid subsets (size 2–4, no pair with |ρ| ≥ 0.75):
   - `fused_n, w_n = nadler_fuse(*[oriented_z[fn] for fn in s])`
   - `fused_a, _   = simple_average_fusion(*[oriented_z[fn] for fn in s])`
   - `auc_n = roc_auc_score(labels, fused_n)`  (point estimate, no bootstrap — faster)
   - `auc_a = roc_auc_score(labels, fused_a)`
   - Store: `{'subset': s, 'size': len(s), 'nadler_auc': auc_n, 'avg_auc': auc_a, 'lift': auc_n - auc_a, 'weights': w_n}`
5. Sort by Nadler AUC descending. Print top 10.
6. Print aggregate stats: median lift, mean lift, % subsets where Nadler > Average.

**Expected output:**
```
Scanned ~540 valid subsets.
Top 10 by Nadler AUC:
  1. trace_length+low_band_power+stft_spectral_entropy+sw_var_peak  Nadler=XX.X%  Avg=XX.X%  Lift=+X.Xpp
  ...

Aggregate: median lift=+X.Xpp, mean lift=+X.Xpp, Nadler > Avg in XX% of subsets.
```

---

### Cell A8 — Results Summary Table (Pre-Plots)

Print a clean final comparison table before entering the plot section:

```
METHOD                              AUROC      SUPERVISION    ACCESS
--------------------------------------------------------------------
LapEigvals supervised (Table 1)     87.2%      Labeled (80%)  White-box
Normalized Nadler (this notebook)   XX.X%      None           Gray-box
Unnormalized Nadler (Phase 7)       76.0%      None           Gray-box
Simple average (best norm. subset)  XX.X%      None           Gray-box
LapEigvals unsupervised             72.0%      None           White-box
Semantic Entropy (Llama-3-8B)       70.0%      None           Black-box
EPR mean (our 'epr' feature)        70.7%      None           Gray-box
```

---

### Cell A9 — Plot 1: Method Comparison Bar Chart

**Type:** Horizontal bar chart  
**x-axis:** AUROC (%)  
**y-axis:** Method name  
**Data:** All rows from the summary table above  
**Colors:**
- Our methods: blue shades
- LapEigvals: orange (supervised), light orange (unsupervised)
- Semantic Entropy / EPR: grey
**Annotations:** Δ vs LapEigvals supervised shown on each bar  
**Reference line:** Vertical dashed line at LapEigvals supervised (87.2%)  
**Save:** `PLOT_DIR + 'A1_method_comparison.png'`

---

### Cell A10 — Plot 2: Individual Feature AUCs

**Type:** Horizontal bar chart, sorted by AUC  
**Data:** `feat_aucs` dict from Cell A4  
**Colors:** Green gradient (higher AUC = darker)  
**Annotations:** AUC value on each bar  
**Reference line:** Vertical dashed at 50% (chance) and at Semantic Entropy 70%  
**Note:** Individual AUCs are normalization-invariant (rank-based); this shows the raw discriminative power of each feature before fusion.  
**Save:** `PLOT_DIR + 'A2_feature_aucs.png'`

---

### Cell A11 — Plot 3: Nadler vs Average Scatter (All Valid Subsets)

**Type:** Scatter plot  
**x-axis:** Simple average AUC (%)  
**y-axis:** Nadler AUC (%)  
**Data:** All `~540` valid subsets from Cell A7  
**Color:** Subset size (size=2: blue, size=3: orange, size=4: green)  
**Diagonal line:** y=x dashed black (= no lift; points above = Nadler wins)  
**Highlight:** Best Nadler subset with a star marker + annotation  
**Highlight:** Best Average subset with a diamond marker  
**Legend:** Subset sizes + diagonal explanation  
**Title:** f"Nadler vs Simple Average — {n_above}/{total} subsets Nadler wins"  
**Save:** `PLOT_DIR + 'A3_nadler_vs_average_scatter.png'`

---

### Cell A12 — Plot 4: Fusion Weights Comparison

**Type:** Side-by-side grouped bar charts (2 panels)  
**Panel 1 (left):** Normalized Nadler weights for `subset_norm` features  
**Panel 2 (right):** Unnormalized Nadler weights for `subset_unnorm` features  
**x-axis:** Feature names (abbreviated)  
**y-axis:** Weight (0 to 1, sums to 1)  
**Colors:** Same feature → same color across panels  
**Title:** "Nadler Fusion Weights: Normalized vs Unnormalized"  
**Annotation:** Note if best subsets differ between normalized and unnormalized runs  
**Key insight to convey:** Unnormalized weights may be dominated by `trace_length` (~300 scale); normalized weights should reflect statistical complementarity.  
**Save:** `PLOT_DIR + 'A4_fusion_weights.png'`

---

### Cell A13 — Plot 5: H(n) Trajectory Examples

**Type:** Line plots, 2 panels  
**Data:** `raw_ents` from Cell A3  

**Selection strategy:**
- Pick 4 correct samples: longest traces (avg trace ~193 tok; pick traces > 200 tok)
- Pick 4 incorrect samples: same length constraint, random

**Panel 1:** Correct responses — 4 individual traces (thin, low alpha) + mean trace (thick green)  
**Panel 2:** Incorrect responses — 4 individual traces (thin, low alpha) + mean trace (thick red)  
**x-axis:** Token position  
**y-axis:** Entropy H(n)  
**Annotations:**
- Mark the sliding window position of `sw_var_peak` for the mean trace (highlight the window with max variance)
- Show sw_var_peak value for correct vs incorrect mean

**Save:** `PLOT_DIR + 'A5_trajectory_examples.png'`

---

### Cell A14 — Plot 6: Feature KDE (Correct vs Incorrect)

**Type:** 2×2 grid of KDE plots  
**Features:** Top 4 by individual AUC: `sw_var_peak`, `trace_length`, `epr`, `low_band_power`  
**Per subplot:**
- KDE of feature values for correct (green, filled) and incorrect (red, filled, lower alpha)
- Vertical dashed lines at medians
- AUC annotation in corner
**x-axis:** Feature value  
**y-axis:** Density  
**Save:** `PLOT_DIR + 'A6_feature_kde.png'`

---

### Cell A15 — Plot 7: Window Ablation

**Type:** Line plot with error bars  
**Data:** `ablation_data` from `phase7_results.pkl` (w ∈ {3, 5, 7, 9, 16})  
**x-axis:** Window size w  
**y-axis:** AUC (%)  
**Error bars:** 95% CI from bootstrap  
**Highlight:** Best window (w=16) with star marker  
**Reference line:** Horizontal dashed at mean EPR AUC (70.7%)  
**Annotation:** "Larger windows suit long GSM8K traces (avg 193 tokens)"  
**Save:** `PLOT_DIR + 'A7_window_ablation.png'`

---

### Cell A16 — Save Ablation Results

**Actions:**
1. Save a summary pkl: `PLOT_DIR + 'gsm8k_ablation_results.pkl'` containing:
   - `unnorm_auc, unnorm_lo, unnorm_hi, unnorm_subset`
   - `norm_auc, norm_lo, norm_hi, norm_subset`
   - `norm_delta_pp` (= (norm_auc - unnorm_auc) * 100)
   - `nadler_lift_pp` (= Nadler AUC - simple avg AUC on best norm subset)
   - `subset_records` (all-subset scan data)
   - `feat_aucs`
2. Print final summary:
   ```
   ══════════════════════════════════════════════
   GSM8K Normalization Ablation — Final Results
   ══════════════════════════════════════════════
   Unnormalized Nadler (Phase 7):  76.0% [72.5, 79.3]
   Normalized Nadler (this run):   XX.X% [XX.X, XX.X]
   Normalization lift:             +X.X pp

   Nadler lift over simple avg:    +X.X pp  (on normalized best subset)

   vs LapEigvals supervised:       -XX.X pp
   vs LapEigvals unsupervised:     +X.X pp
   ══════════════════════════════════════════════
   ```

---

## Part B — GPQA Diamond with Qwen2.5-72B-Instruct

### Cell B0 — Part B Title (Markdown)

```markdown
## Part B: GPQA Diamond — Qwen2.5-72B-Instruct (4-bit)

**Motivation:**
Prior GPQA experiments (Phase 4/5) used 7B models with ~30% accuracy on GPQA Diamond.
At 30% accuracy (barely above 25% random), the model is at its knowledge ceiling —
it can't reason incorrectly in a meaningful way. Hallucination detection requires
a model that CAN reason but sometimes fails.

Qwen2.5-72B-Instruct achieves ~65% accuracy on GPQA Diamond.
At 65:35 correct:incorrect, the spectral entropy signal should be far more discriminative.

**Access:** Fully open (no HF gating). Loaded in 4-bit quantization on Colab A100.
**Cache:** Resumable. Will skip samples already in cache.
```

---

### Cell B1 — Install Inference Dependencies

**Actions:**
1. `!pip install bitsandbytes transformers>=4.40 accelerate datasets -q`
2. HuggingFace login via `userdata.get('HF_TOKEN')`
3. Print GPU info: `!nvidia-smi | head -3`

---

### Cell B2 — GPQA Config

**Actions:**
```python
GPQA_CFG = {
    'model_id':    'Qwen/Qwen2.5-72B-Instruct',
    'dataset':     'gpqa_diamond',
    'temp':        1.0,
    'max_new':     1024,       # GPQA needs longer responses than GSM8K
    'quantize':    True,       # 4-bit required for 72B on A100
}
GPQA_CFG['model_short'] = 'Qwen2.5-72B-Instruct'
GPQA_CFG['run_key']     = f"Qwen2.5-72B-Instruct__gpqa_T1.0"

GPQA_BASE_DIR = '/content/drive/MyDrive/epr_spectral_gpqa_72b/'
os.makedirs(GPQA_BASE_DIR, exist_ok=True)

GPQA_RUN_DIR     = os.path.join(GPQA_BASE_DIR, GPQA_CFG['run_key'])
GPQA_CACHE_PATH  = os.path.join(GPQA_RUN_DIR,  'inference_cache.pkl')
GPQA_RESULTS_PATH = os.path.join(GPQA_RUN_DIR, 'gpqa72b_results.pkl')
os.makedirs(GPQA_RUN_DIR, exist_ok=True)
```

---

### Cell B3 — Load GPQA Dataset

**Actions:**
1. `from spectral_utils.data_loaders import load_gpqa, gpqa_prompt_and_answer, is_correct_gpqa`
2. `gpqa_data = load_gpqa()` → 198 problems
3. Print: n_problems, example question + choices (first item)
4. Note: `gpqa_prompt_and_answer(row, idx)` shuffles choices deterministically per idx so the correct answer position is randomized.

**Expected output:** `Loaded 198 GPQA Diamond problems.`

---

### Cell B4 — Load Model (Qwen2.5-72B, 4-bit)

**Actions:**
1. Check if results already saved → skip model loading if so.
2. Check cache: how many samples already done?
3. Only load model if there are remaining samples:
   ```python
   from spectral_utils import load_model, free_memory
   model, tokenizer = load_model(GPQA_CFG['model_id'], quantize_4bit=True)
   ```
4. Print GPU memory usage after load.

**Note in cell:** 72B in 4-bit ≈ 36GB. Requires Colab A100 (40GB). If OOM, fall back to Qwen2.5-32B-Instruct (also open, ~65% GPQA, ~16GB in 4-bit).

---

### Cell B5 — Inference Loop (Resumable)

**Actions:**
```python
if not os.path.exists(GPQA_RESULTS_PATH):
    from spectral_utils import generate_full
    from tqdm import tqdm

    gpqa_cache = load_cache(GPQA_CACHE_PATH)
    remaining  = [i for i in range(len(gpqa_data)) if not gpqa_cache.get(i, {}).get('done')]
    print(f'Cache: {len(gpqa_data)-len(remaining)}/{len(gpqa_data)} done. Remaining: {len(remaining)}')

    for i in tqdm(remaining, desc='GPQA-72B'):
        row    = gpqa_data[i]
        prompt, correct_letter = gpqa_prompt_and_answer(row, i)
        full_text, all_ents    = generate_full(model, tokenizer, prompt,
                                               temperature=GPQA_CFG['temp'],
                                               max_new_tokens=GPQA_CFG['max_new'])
        correct    = is_correct_gpqa(full_text, correct_letter)
        has_answer = extract_gpqa_answer(full_text) != ''
        gpqa_cache[i] = {
            'done': True, 'full_text': full_text,
            'all_entropies': all_ents, 'correct': correct,
            'has_answer': has_answer, 'correct_letter': correct_letter,
        }
        if i % 10 == 0:
            save_cache(gpqa_cache, GPQA_CACHE_PATH)
        free_memory()

    save_cache(gpqa_cache, GPQA_CACHE_PATH)
    del model, tokenizer; free_memory()
else:
    print('Results already exist — skipping inference.')
    gpqa_cache = load_cache(GPQA_CACHE_PATH)
```

**Print after loop:**
- n_done / n_total
- Accuracy (compare to expected ~65%)
- Format rate (answered with a letter A/B/C/D)

---

### Cell B6 — GPQA Feature Extraction

**Actions:**
1. Extract usable samples (done=True, len(ents) >= 8).
2. For each: `extract_all_features(ents)` → 12 features.
3. Build:
   - `gpqa_labels = np.array([int(v['correct']) for v in usable])`
   - `gpqa_feat_arrays = {fn: np.array(...) for fn in FEAT_NAMES}`
   - `gpqa_raw_ents = [v['all_entropies'] for v in usable]`
4. Print:
   - n_samples, n_correct, accuracy
   - avg trace length
   - n_samples with traces < 32 (STFT floor)
5. Compare accuracy to prior GPQA (Phase 4: ~30%) and expected (~65%).

---

### Cell B7 — GPQA Window Ablation

**Actions:**
1. WINDOW_SIZES = [3, 5, 7, 9, 16]
2. For each w: `sw_var_peak_with_window(ents, w)` for all samples.
3. `boot_auc(gpqa_labels, sw_vals)` for each w.
4. Print table: w | AUC | CI | eligible.
5. `gpqa_best_w = argmax AUC`
6. Override `gpqa_feat_arrays['sw_var_peak']` with best-window version.

**Note:** For GPQA with longer responses (1024 tokens max), window may behave differently than GSM8K.

---

### Cell B8 — GPQA Individual Feature AUCs

**Actions:**
1. Same as Cell A4 but for GPQA data.
2. Print ranked table.
3. Compare to Phase 4/5 GPQA results (all were near chance: 51–65%).
4. Key question: does 72B model produce stronger individual feature signals than 7B?

**Reference values from Phase 4/5 GPQA:**
- Best Phase 5 (Mistral-7B, T=1.0): 65.4% (dominant_freq + stft_max_high_power)
- Best Phase 4 (Mistral-7B, T=1.5): 65.0%
- Most individual features were 51–60% for 7B models on GPQA

---

### Cell B9 — GPQA Normalized Nadler Fusion

**Actions:**
1. `best_nadler_on(gpqa_feat_arrays, FEAT_NAMES, gpqa_labels, max_size=4, label='GPQA-72B', compare_mean=True)`
2. Print result with full context:
   ```
   GPQA Diamond — Qwen2.5-72B-Instruct, T=1.0
   ─────────────────────────────────────────────
   Our normalized Nadler:   XX.X% [XX.X, XX.X]
   Prior best (Mistral-7B): 65.4% [57.3, 73.4]
   Delta vs prior:          +X.X pp
   ─────────────────────────────────────────────
   ```

---

### Cell B10 — GPQA Decision Gates

**Gates:**
| Gate | Condition | Pass | Fail |
|------|-----------|------|------|
| G0 | len(labels) ≥ 150 | Sufficient samples | Need more inference |
| G1 | accuracy ∈ [50%, 80%] | Model in sweet spot | Too easy or too hard |
| G2 | best individual AUC > 57% | Spectral structure exists | No signal above chance |
| G3 | fusion AUC > 65.4% | Beat prior GPQA best | 72B didn't help |
| G4 | fusion AUC > 72% | Strong result: spectral works on science MCQ | Math-specific only |
| G5 | CI lower > 60% | Statistically reliable | Too few samples / noisy |
| G6 | Nadler lift > 0 pp | Fusion justified over average | Nadler adds nothing here |

**Interpretation print block:**
```python
if g4_passed:
    print("→ Spectral features generalize to graduate-level science MCQ when the model is competent.")
elif g3_passed:
    print("→ Spectral features transfer to science MCQ with 72B model. Not as strong as math.")
elif g2_passed:
    print("→ Marginal signal. 72B helps vs 7B but domain gap (science MCQ vs math) remains.")
else:
    print("→ No spectral signal even with 72B. Science MCQ may be inherently different from math.")
```

---

### Cell B11 — GPQA Visualizations

**Plot B1: GPQA Feature AUCs (vs prior 7B results)**

- Grouped bar chart
- Two groups per feature: 72B (blue) and prior best 7B (grey)
- Only show features where 72B > 7B to illustrate improvement
- Reference line at 50% and prior best fusion (65.4%)
- Save: `PLOT_DIR + 'B1_gpqa_feature_aucs_comparison.png'`

**Plot B2: GPQA vs MATH-500 AUC Landscape**

- Scatter plot or bar chart comparing our best AUC across all datasets/models:
  - MATH-500/Qwen2.5-7B/T=1.5: 96.6%
  - MATH-500/Qwen2.5-7B/T=1.0: 90.0%
  - MATH-500/Qwen2.5-1.5B/T=1.5: 88.3%
  - GSM8K/Llama-3.1-8B: 76.0% (unnorm) / XX.X% (norm)
  - GPQA/Qwen2.5-72B: XX.X%
  - GPQA/Mistral-7B: 65.4%
  - HotpotQA/Mistral-7B: 59.5%
- Color by dataset type (math=green, science=orange, factual=red)
- Adds one more point to the dataset-difficulty gradient story
- Save: `PLOT_DIR + 'B2_full_results_landscape.png'`

**Plot B3 (optional): GPQA H(n) Trajectories**

- Same as Cell A13 but for GPQA samples
- Use traces > 100 tokens for clean visualization
- Side-by-side: correct vs incorrect (4 each)
- Save: `PLOT_DIR + 'B3_gpqa_trajectories.png'`

---

### Cell B12 — Save GPQA Results

**Actions:**
1. Save `gpqa72b_results.pkl` to `GPQA_RUN_DIR` with:
   - `model`, `model_id`, `dataset`, `temp`
   - `n_samples`, `accuracy`, `avg_trace`
   - `feat_names`, `raw_labels`, `feat_arrays`
   - `auc_map`, `sign_map`
   - `ablation` (window ablation results)
   - `best_window`
   - `fusion_auc`, `fusion_lo`, `fusion_hi`, `best_subset`
   - `nadler_lift_pp`
   - `prior_gpqa_7b_auc = 0.654`
   - `delta_vs_prior`

2. Print final summary for Part B.

---

### Cell B13 — Full Notebook Summary (Markdown)

Final markdown cell summarizing both parts:

```markdown
## Results Summary

### Part A — GSM8K Normalization Ablation

| Method | AUROC | Notes |
|--------|-------|-------|
| LapEigvals supervised | 87.2% | White-box, labeled |
| Normalized Nadler (this run) | XX.X% | Gray-box, unsupervised |
| Unnormalized Nadler (Phase 7) | 76.0% | Gray-box, no zscore |
| Simple average (norm. subset) | XX.X% | Gray-box, no covariance |
| LapEigvals unsupervised | 72.0% | White-box, no labels |
| Mean EPR (our 'epr' feature) | 70.7% | Single feature |
| Semantic Entropy (literature) | 70.0% | Multi-sample, black-box |

**Normalization lift:** +X.X pp  
**Nadler lift over simple average:** +X.X pp  

### Part B — GPQA Diamond (Qwen2.5-72B)

| Method | AUROC | Notes |
|--------|-------|-------|
| Normalized Nadler (72B) | XX.X% | This run |
| Prior best (Mistral-7B) | 65.4% | Phase 5 |
| Random chance (4-way MCQ) | 50.0% | Baseline |

**Delta vs prior:** +X.X pp  
**Key finding:** [spectral features generalize / don't generalize] to graduate-level science MCQ  
  when the model is sufficiently competent (~65% accuracy).
```

---

## Implementation Notes

### Drive directories
- **Part A inputs:** `/content/drive/MyDrive/epr_spectral_gsm8k_vs_lapei/Llama-3.1-8B-Instruct__gsm8k_T1.0/`
- **Part A plots + results:** `/content/drive/MyDrive/epr_spectral_gsm8k_vs_lapei/plots/`
- **Part B cache + results:** `/content/drive/MyDrive/epr_spectral_gpqa_72b/Qwen2.5-72B-Instruct__gpqa_T1.0/`

### Colab runtime requirements
- **Part A:** Any GPU (CPU could work, but GPU faster for bootstrap). ~10 min.
- **Part B inference:** A100 required (72B 4-bit ≈ 36GB). Estimated inference time: ~3–4 hours for 198 samples.
- **Part B analysis:** Any GPU after inference completes.

### Package imports from spectral_utils
```python
from spectral_utils import (
    load_cache, save_cache,
    load_model, generate_full, free_memory,
    extract_all_features, sw_var_peak_with_window, FEAT_NAMES,
    zscore, boot_auc, nadler_fuse, simple_average_fusion, best_nadler_on,
)
from spectral_utils.data_loaders import (
    load_gpqa, gpqa_prompt_and_answer, extract_gpqa_answer, is_correct_gpqa,
)
```

### Fallback: if Qwen2.5-72B OOMs
Replace in Cell B2:
```python
'model_id': 'Qwen/Qwen2.5-32B-Instruct'   # ~16GB 4-bit, ~60% GPQA acc
```
No other code changes needed — same pipeline.

---

## Cell Count Summary

| Part | Cells | Type |
|------|-------|------|
| A0 | 1 | Markdown title |
| A1–A16 | 16 | Code |
| B0 | 1 | Markdown title |
| B1–B13 | 13 | Code + 1 Markdown |
| **Total** | **31** | |
