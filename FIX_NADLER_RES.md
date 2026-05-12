# Fix: NADLER_RES / LEN_RES / PCA_RES disappearing after kernel restart

## What's wrong

Cell 14's metadata has `"background_save": true` and `"execution_count": null`.
The cell printed all 12 results, but the kernel disconnected before formally
finishing, so `NADLER_RES` was lost from memory. Cells 16, 17, 18 then error
with `NameError: name 'NADLER_RES' is not defined`.

Same risk applies to Cell 15 (`LEN_RES`) and Cell 16 (`PCA_RES`) — long
computations that vanish on disconnect.

## Fix: persist each result dict to disk at the end of the cell that produces it

Replace the source of **three cells** in `Spectral_Analysis_Phase10_Main_RAG.ipynb`.
Each new cell:
1. If the variable is already in memory → skip (no-op).
2. Else if the `.pkl` exists on Drive → load it.
3. Else compute, then save to Drive.

To force a fresh recompute, either delete the `.pkl` from Drive or flip the
`FORCE_RECOMPUTE_*` flag at the top of the cell.

---

## Cell 14 — replace the entire source with this

```python
# Cell 14 — Best Nadler subset + weights per cell
#
# Persist to disk so long-running results survive kernel restarts / Colab
# background_save disconnects. On a fresh kernel after disconnect, just re-run
# this cell — it'll reload from disk instead of recomputing (~minutes saved).
# To force recomputation: delete RES_DIR/nadler_res.pkl, or set FORCE=True below.
import itertools

NADLER_PATH = os.path.join(RES_DIR, 'nadler_res.pkl')
FORCE_RECOMPUTE_NADLER = False

if not FORCE_RECOMPUTE_NADLER and 'NADLER_RES' in globals() and NADLER_RES:
    print(f'NADLER_RES already in memory ({len(NADLER_RES)} cells); skipping.')
elif not FORCE_RECOMPUTE_NADLER and os.path.exists(NADLER_PATH):
    with open(NADLER_PATH, 'rb') as _f:
        NADLER_RES = pickle.load(_f)
    print(f'NADLER_RES loaded from {NADLER_PATH} ({len(NADLER_RES)} cells).')
else:
    NADLER_RES = {}  # (model, dataset) -> {auc, lo, hi, subset, weights, sign_map}
    for (mk, ds), c in ALL_CELLS.items():
        if not c['features']:
            continue
        keys = list(c['features'][0].keys())
        feat_dict = {k: np.array([f[k] for f in c['features']]) for k in keys}
        labels_arr = np.array(c['labels'])

        # Compute sign orientation map (same logic best_nadler_on uses internally,
        # repeated here so we can persist the signs alongside the weights).
        sign_map = {}
        for k in keys:
            ap, *_ = boot_auc(labels_arr,  feat_dict[k])
            an, *_ = boot_auc(labels_arr, -feat_dict[k])
            sign_map[k] = +1 if (not np.isnan(ap) and ap >= an) else -1

        auc, lo, hi, subset, weights = best_nadler_on(
            feat_dict, keys, labels_arr,
            max_size=4, label=f'{mk}/{ds}', compare_mean=False,
        )
        NADLER_RES[(mk, ds)] = {
            'auc': auc, 'lo': lo, 'hi': hi,
            'subset': list(subset) if subset else [],
            'weights': list(weights) if weights is not None else [],
            'sign_map': sign_map,
        }

    with open(NADLER_PATH, 'wb') as _f:
        pickle.dump(NADLER_RES, _f)
    print(f'NADLER_RES saved to {NADLER_PATH}')

print('=== Best Nadler per cell ===')
for (mk, ds), r in NADLER_RES.items():
    sub_str = ' + '.join(r['subset']) if r['subset'] else '(none)'
    print(f'  [{mk:<10s}/{ds:<22s}] AUC={100*r["auc"]:.1f}% [{100*r["lo"]:.1f},{100*r["hi"]:.1f}]  subset: {sub_str}')
```

---

## Cell 15 — replace the entire source with this

```python
# Cell 15 — Length-controlled analysis per cell
#
# Headline number for the thesis defensibility:
#   spectral-only Nadler AUC vs trace_length-alone AUC.
# Persisted to disk like NADLER_RES — survives kernel restarts.

LEN_PATH = os.path.join(RES_DIR, 'len_res.pkl')
FORCE_RECOMPUTE_LEN = False

if not FORCE_RECOMPUTE_LEN and 'LEN_RES' in globals() and LEN_RES:
    print(f'LEN_RES already in memory ({len(LEN_RES)} cells); skipping.')
elif not FORCE_RECOMPUTE_LEN and os.path.exists(LEN_PATH):
    with open(LEN_PATH, 'rb') as _f:
        LEN_RES = pickle.load(_f)
    print(f'LEN_RES loaded from {LEN_PATH} ({len(LEN_RES)} cells).')
else:
    LEN_RES = {}
    for (mk, ds), c in ALL_CELLS.items():
        if not c['features']:
            continue
        keys = list(c['features'][0].keys())
        feat_dict = {k: np.array([f[k] for f in c['features']]) for k in keys}
        labels_arr = np.array(c['labels'])

        # 1) trace_length alone
        tl = feat_dict['trace_length']
        a_tl, lo_tl, hi_tl = boot_auc(labels_arr, tl)
        if not np.isnan(a_tl) and a_tl < 0.5:
            a_tl, lo_tl, hi_tl = boot_auc(labels_arr, -tl)

        # 2) Spectral-only Nadler (excludes trace_length)
        spectral_keys = [k for k in keys if k != 'trace_length']
        a_sp, lo_sp, hi_sp, sub_sp, w_sp = best_nadler_on(
            feat_dict, spectral_keys, labels_arr,
            max_size=4, label=f'{mk}/{ds}-spectralonly', compare_mean=False,
        )

        LEN_RES[(mk, ds)] = {
            'trace_length_auc':   a_tl,
            'trace_length_ci':    (lo_tl, hi_tl),
            'spectral_only_auc':  a_sp,
            'spectral_only_ci':   (lo_sp, hi_sp),
            'spectral_only_subset':  list(sub_sp) if sub_sp else [],
            'spectral_only_weights': list(w_sp) if w_sp is not None else [],
            'lift_over_length_pp':   100 * (a_sp - a_tl),
        }
    with open(LEN_PATH, 'wb') as _f:
        pickle.dump(LEN_RES, _f)
    print(f'LEN_RES saved to {LEN_PATH}')

print('=== Length-controlled summary ===')
print(f'{"cell":<35s}  {"len-only":>10s}  {"spec-only":>10s}  {"Δ (pp)":>10s}')
for (mk, ds), r in LEN_RES.items():
    print(f'  {mk+"/"+ds:<33s}  {100*r["trace_length_auc"]:>9.1f}%  '
          f'{100*r["spectral_only_auc"]:>9.1f}%  {r["lift_over_length_pp"]:>+9.1f}')
```

---

## Cell 16 — replace the entire source with this

```python
# Cell 16 — PCA diagnostic per cell (PC1 AUC vs Nadler AUC)
# Persisted to disk like NADLER_RES — survives kernel restarts.
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

PCA_PATH = os.path.join(RES_DIR, 'pca_res.pkl')
FORCE_RECOMPUTE_PCA = False

if not FORCE_RECOMPUTE_PCA and 'PCA_RES' in globals() and PCA_RES:
    print(f'PCA_RES already in memory ({len(PCA_RES)} cells); skipping.')
elif not FORCE_RECOMPUTE_PCA and os.path.exists(PCA_PATH):
    with open(PCA_PATH, 'rb') as _f:
        PCA_RES = pickle.load(_f)
    print(f'PCA_RES loaded from {PCA_PATH} ({len(PCA_RES)} cells).')
else:
    PCA_RES = {}
    for (mk, ds), c in ALL_CELLS.items():
        if not c['features']:
            continue
        keys = list(c['features'][0].keys())
        X = np.array([[f[k] for k in keys] for f in c['features']])
        Xs = StandardScaler().fit_transform(X)
        pca = PCA(n_components=min(5, X.shape[1])).fit(Xs)
        pc1 = pca.transform(Xs)[:, 0]
        labels_arr = np.array(c['labels'])
        a_pc1 = roc_auc_score(labels_arr, pc1)
        if a_pc1 < 0.5:
            a_pc1 = roc_auc_score(labels_arr, -pc1)
        loadings = pd.Series(np.abs(pca.components_[0]), index=keys).sort_values(ascending=False)
        PCA_RES[(mk, ds)] = {
            'pc1_auc':         a_pc1,
            'pc1_var_ratio':   float(pca.explained_variance_ratio_[0]),
            'top3_loadings':   loadings.head(3).to_dict(),
            'nadler_lift_over_pc1_pp': 100 * (NADLER_RES[(mk, ds)]['auc'] - a_pc1),
        }
    with open(PCA_PATH, 'wb') as _f:
        pickle.dump(PCA_RES, _f)
    print(f'PCA_RES saved to {PCA_PATH}')

print('=== PCA vs Nadler ===')
print(f'{"cell":<35s}  {"PC1 AUC":>9s}  {"Nadler":>9s}  {"lift":>7s}  PC1 top3')
for (mk, ds), r in PCA_RES.items():
    top3 = ', '.join(f'{k}({v:.2f})' for k, v in r['top3_loadings'].items())
    print(f'  {mk+"/"+ds:<33s}  {100*r["pc1_auc"]:>8.1f}%  '
          f'{100*NADLER_RES[(mk,ds)]["auc"]:>8.1f}%  {r["nadler_lift_over_pc1_pp"]:>+6.1f}  {top3}')
```

---

## How to recover the current session (no recompute needed)

Since Cell 14 already ran and printed results, the NADLER_RES dict was alive
in some prior kernel that's now gone. You have two paths:

**Path A — quickest** (if your current kernel still has `ALL_CELLS` from Cell 11):
1. Apply the Cell 14 source above.
2. Run Cell 14 → it will recompute and save to `nadler_res.pkl`.
3. Apply Cells 15 and 16 source above.
4. Run them in order.
5. Run Cells 17, 18, 20, 21, 24 — they'll find `NADLER_RES` in memory.

Recomputing Cell 14 takes ~3 min total (12 cells × ~15s each). Cell 15 is similar.

**Path B — if the kernel died entirely**: same as Path A but you have to re-run
Cell 11 (feature extraction) first — that's the only thing that depends on the
raw inference checkpoints, which ARE on Drive, so it's fast.

After running once with the new code, the `.pkl` files exist on Drive and
every future session will load them in milliseconds.

---

## Why this didn't need a package change

The `best_nadler_on` package function already returns the 5-tuple correctly
(commit `b3c45a4` on master). The bug is purely the notebook losing state on
kernel disconnect. Persisting to Drive is the standard pattern in this
notebook — Cell 6's `run_inference_for_cell` already does it for raw outputs,
and Cell 11 already does it for features. This just extends the same pattern
to the analysis results.
