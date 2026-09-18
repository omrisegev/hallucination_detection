# HANDOFF — White-box per-layer views for the localization population

Claude, 2026-09-18. Step 421. Read this before touching the layer-view data.

## 1. What exists now, and why it matters

A per-layer logit-lens field over **the entire localization evaluation population**:
13,769 answers / 145,597 steps / 6,968,779 tokens, teacher-forced, one forward pass per
provided chain. Nine cells: ProcessBench {gsm8k, math, olympiadbench, omnimath} ×
{Qwen3-4B, Qwen3-8B}, plus PRMBench × Qwen3-8B.

This is the **first measurement channel in the project that is not a function of the
output distribution of one greedy pass.** Every one of the 50 streams in
`digitfree_broad50`, every CT7 view, and every channel in the token-level bank is some
transform of the same top-k distribution at a single position. These quantities measure
how that distribution is *built* across 36 layers, which is a different process, not a
smoothing or an innovation of the same series.

Step 414's conclusion was that more than three conditionally independent sources
"cannot be built by adding transforms of the output distribution of one greedy pass. It
requires a measurement channel we do not currently extract — internal layer states,
multiple samples of the same answer, or a second model." **This is the first of those
three.** Whether it delivers is unmeasured.

## 2. Where the data is

On the cluster, under `$SHARED = /shared/cycle2_tau_averbuch_prj/omrisegev1`:

```
$SHARED/results/pb_layer_views_qwen3_4b/{gsm8k,math,olympiadbench,omnimath}/
$SHARED/results/pb_layer_views_qwen3_8b/{gsm8k,math,olympiadbench,omnimath}/
$SHARED/results/prmbench_layer_views_qwen3_8b/
```

Each directory holds `MANIFEST.json` and `rows/<row_id>.npz`, one file per answer,
written with atomic rename so a partial run is safe to resume. 5.5 GB total
(1.7 + 1.7 + 2.1). Backed up to `gdrive:hallucination_detection/cluster_results/` under
the same three directory names.

`row_id` joins to `results/localization_full_benchmark_v3/evaluation/JOINED.json`
(`records[i]["row_id"]`, `records[i]["cell"]`), which is how you align to `offsets`,
`target`, `labels` and to every published score on this population.

**Cycle-2 is very likely being retired.** Our account and `$HOME` are now `cycle3_*`
while all of this sits under `cycle2_*`. The tree is still writable and carries no
retirement notice, but plan a move. The Drive copy is the durable one.

## 3. Exact schema of one `rows/<row_id>.npz`

Axis order is fixed by `cluster/layer_lens.py`:
`MODULES = ("attn", "mlp", "resid")`, layers `0..L-1` in model order, tokens in
generation order. `L = 36` and `V = 151,936` for both Qwen3-4B and Qwen3-8B.

| array | shape | dtype | meaning |
|---|---|---|---|
| `lens_H` | `[3, 36, T]` | f16 | Shannon entropy of the lens distribution, **full vocabulary** |
| `lens_logp_tgt` | `[3, 36, T]` | f16 | lens log-prob of the token actually in the chain (depth-resolved spilled energy) |
| `lens_logp_top1` | `[3, 36, T]` | f16 | max lens log-prob (commitment) |
| `lens_kl_final` | `[3, 36, T]` | f16 | `KL(lens_l ‖ lens_final)`, the DoLa contrast direction |
| `resid_norm` | `[36, T]` | f16 | `‖x_l,t‖` |
| `cov_eigs` | `[36, 32]` | **f32** | top-32 eigenvalues of the centred token covariance of `x_l` |
| `hid_proj` | `[36, 256]` | f16 | token-mean of `x_l` under a fixed seeded Gaussian projection |
| `final_lens_H` | `[T]` | f32 | final-layer residual lens entropy, **top-15 renormalised** (the gate input) |
| `final_lens_H_fullvocab` | `[T]` | f32 | the same, full vocabulary |
| `step_starts`, `step_ends` | `[S]` | i64 | official step spans, identical to the telemetry |
| `gen_token_ids` | `[T]` | i32 | provided-chain token ids |
| `gate_flag` | scalar str | — | empty when the row passed the gate |

`MANIFEST.json` carries task, subset, model, modules, quantities, `n_layers`,
`hidden_size`, `proj_dim`, `cov_eigs_r`, batch caps, `tol_median`, telemetry path,
Slurm job id, gate counts, status and wall seconds.

## 4. Four things that will bite you

**`lens_H` is full-vocabulary and is NOT comparable to the cached `token_entropies`.**
The cache is top-15 renormalised (`backfill_views.ENTROPY_TOPK = 15`). These are
different statistics. This is exactly the defect that would have aborted the run: the
gate compared them and would have failed by construction at `--tol-median 2e-2`. Use
`final_lens_H` when you need the comparable quantity, `final_lens_H_fullvocab` /
`lens_H` when you want the full-vocab one as a view in its own right. Both are saved
deliberately.

**`cov_eigs` is float32 on purpose.** Residual Gram eigenvalues routinely exceed the
float16 maximum of 65,504 on a trained model;
`results/whitebox_layer_fusion_v2/DATA_INVENTORY.md` records 47,008 non-finite entries
on the Qwen3-8B cell from exactly that overflow. Do not "tidy" it back to f16.

**`hidden_states[L]` is never read, and must not be.** HuggingFace applies the final
norm before appending the last entry, so `hidden_states[L] == Norm_final(x_L)` while
entries `0..L-1` are the raw pre-norm streams. The residual stream is reconstructed
from the taps via `x_l = x_{l-1} + a_l + m_l`. Reading it directly double-norms and
corrupts the KL reference that every other layer is measured against.
`scripts/smoke_layer_lens.py` locks this down; `verify_residual_reconstruction` is a
one-shot per-cell architecture guard.

**`HID_PROJ_SEED = 20260811` must never change.** The projection is shared across every
candidate and cell so that corpus-level subspace methods see a consistent basis. A
changed seed silently invalidates cross-cell comparisons.

## 5. Provenance and integrity

Every row was gate-checked against the existing telemetry and **zero rows failed**, in
all nine cells (3,400 + 3,400 + 6,969 = 13,769 checked, 0 failed). That is a strong
statement: the token axis of this field is identical by construction to the token axis
of every number the project has published on this population, because the driver reuses
`build_items` from both teacher-forced scoring drivers rather than rebuilding the
prompts.

Runtime ~18.4 GPU-hours of the ~5,070 remaining on the account. 851 B/token measured
against 944 B/token predicted from the 14 pre-existing sidecars, i.e. the byte model
held to about 10%.

## 6. What to measure first — and what NOT to do first

**Do the conditional participation ratio along depth before any fusion.** 36 layers × 3
taps × 4 quantities is 432 numbers per token, and the project's own history says what
to expect: prefix innovations sat at .87–.96 with the family they were differenced
from, and adding them moved the effective count by at most 0.15. Adjacent layers are
almost certainly near-duplicates in the same way. The question worth answering is how
many effective dimensions survive along depth, and whether TriLens's claim that the
three taps are *independently* informative survives a label-free measurement.

Run it at **step level with within-label-class centring**, because that is the quantity
the three-source threshold was defined on. A marginal token-level participation ratio
is not comparable to the 1.80 / 2.46 / 2.83 numbers in the ledger — see
`docs/HANDOFF_TOKEN_PROBABILITIES.md` §4 for how that mismatch already produced a
misleading 9.69.

**Do not** build a 432-column bank and fuse it. **Do not** read the eventual number
against 3 without first running a noise floor (shuffle tokens within each answer,
independently per channel, and recompute). And **do not** promote anything from this
data to a candidate without the gate-free per-subset SLA reported beside macro-F1.

## 7. Reproducing or extending

```bash
ssh aircc 'export SLURM_CONF_SERVER=controller-primary; \
  S=/shared/cycle2_tau_averbuch_prj/omrisegev1; cd $S/code && \
  sbatch -p power-gpu --qos=owner_940 --time=24:00:00 \
    cluster/submit_localization_layer_views.sbatch \
    --task processbench --subset gsm8k --model Qwen/Qwen3-8B \
    --telemetry $S/results/pb_qwen3_8b/processbench_gsm8k.pkl \
    --out $S/results/pb_layer_views_qwen3_8b/gsm8k'
```

Local CPU smoke first, always: `python cluster/run_localization_layer_views.py --smoke`.
Sync with `bash cluster/sync_code.sh`. Gate order is smoke → N=30 pilot → full.

**The Slurm trap**: `ssh aircc '<cmd>'` is a non-login shell and never sources
`/etc/profile.d`, so `SLURM_CONF_SERVER` is unset and every `squeue`/`sbatch` dies with
`fatal: Could not establish a configuration source`. It reads exactly like an outage and
is not one. `sdata` keeps working, which makes it more confusing. Documented in
`cluster/README.md`.

## 8. What is still not available

Attention was **never captured for any population in this project**. The reducer
(`attn_laplacian_capture`) exists and is unit-tested, the scorer
(`scripts/score_lapeigvals.py`) exists and carries a starvation guard, and the one cell
whose name promises it has `capture: {}` in its manifest. Building an attention channel
means a new extraction, not an analysis. Same for multi-sample self-consistency on this
population: it is teacher-forced K=1 by construction.
