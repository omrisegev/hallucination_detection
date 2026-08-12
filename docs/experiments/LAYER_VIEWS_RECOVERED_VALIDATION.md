# Recovered validation evidence — layer-view cells (Step 244)

**Raised by**: Codex, 2026-08-13. **Resolved**: 2026-08-13, job 186485.

Four completed layer-view cells had **correct, complete sidecars but destroyed validation
reports**. This document recovers the evidence, proves the recovery is faithful, proves the
recovery was non-destructive, and records the fix.

---

## 1. What happened

Job **184777** — the `--dependency=afterany` resume link of **184776** — ran after 184776 had
already finished every cell. It overwrote four reports with `n_traces=0`,
`gate_b_pass=false`, `n_tokens=0`.

**Mechanism** (confirmed in the code and reproduced on a CPU fixture): `gate.add()` lived
inside `for ci, c in todo:`, where `todo` is the list of candidates still *missing* a field.
On a resume where every field exists, `todo` is empty for every problem, so the gate
accumulated nothing; `gate_b_verdict` then returned `False` for *"no comparable
token_entropies traces"*, and that verdict was written over the real one.

The corrupted reports were **not obviously empty** — `n_candidates` survived (it is seeded
from the sidecar key count) while `n_tokens` went to 0, so they looked populated. The 184777
log shows the fingerprint plainly:

```
[gate] noise_gsm8k_phi3mini T=1.0: median|dH|=nan frac_close=nan GATE-B FAIL
[layers] noise_gsm8k_phi3mini T=1.0 DONE: 1319 candidates, 0 tokens, 315.6 MB
```

**The sidecars were never at risk.** Only the separate report files were affected.

---

## 2. Recovered validation

Recovered two independent ways: (a) the surviving **184776** Slurm log, and (b) a
**validation-only replay** (job 186485) over the same fixed first-50 candidates, same
prompts, same tokenizer, same warpers, same weights.

| | ars_gsm8k_r1distill8b | noise_gsm8k_phi3mini | lapeigvals_gsm8k_phi35 | noise_gsm8k_mistral7b |
|---|---|---|---|---|
| model | DeepSeek-R1-Distill-Llama-8B | Phi-3-mini-4k-instruct | Phi-3.5-mini-instruct | Mistral-7B-Instruct-v0.3 |
| revision | `6a6f4aa4197940add57724a7707d069478df56b1` | `f39ac1d28e925b323eae81227eaba4464caced4e` | `2fe192450127e6a83f7441aef6e3ca586c338b77` | `c170c708c41dac9275d15a8fff4eca08d52bab71` |
| temp / dtype / attn | 0.0 / bfloat16 / eager | 1.0 / bfloat16 / eager | 1.0 / bfloat16 / eager | 1.0 / bfloat16 / eager |
| **n_layers** | 32 | 32 | 32 | 32 |
| **gate n_traces** | 50 | 50 | 50 | 50 |
| **gate n_tokens** | 22,893 | 14,726 | 16,725 | 17,809 |
| **n_len_mismatch** | **0** | **0** | **0** | **0** |
| **median \|dH\|** | 6.696e-05 | 1.975e-04 | 1.363e-05 | 2.910e-04 |
| p99 \|dH\| | 4.625e-02 | 8.605e-02 | 1.226e-01 | 6.028e-02 |
| max \|dH\| | 1.830e-01 | 2.095e-01 | 3.097e-01 | 2.124e-01 |
| **frac_close** | 0.99284 | 0.94357 | 0.93901 | 0.98085 |
| **first_tok_median** | 1.167e-02 | 2.214e-02 | 3.721e-02 | 1.526e-03 |
| median_r | 0.999609 | 0.998936 | 0.997363 | 0.999602 |
| **GATE-B** | **PASS** | **PASS** | **PASS** | **PASS** |
| **residual_identity_max_abs** | **0.0** | **0.0** | **0.0** | **0.0** |
| **lens_max_abs** | **0.0** | **0.0** | **0.0** | **0.0** |
| logit_scale | 37.25 | 59.50 | 79.00 | 32.50 |
| extracted candidates | 500 | 1,319 | 1,319 | 1,319 |
| extracted tokens | 224,251 | 351,264 | 405,797 | 440,131 |
| sidecar | 195.4 MB | 315.6 MB | 361.0 MB | 389.6 MB |

`n_len_mismatch = 0` on every cell means every gated trace aligned exactly — no
off-by-one in the prompt reconstruction. `residual_identity_max_abs` and `lens_max_abs` at
**exactly 0.0** mean the tapped MHSA/FFN writes reconstruct the residual stream bit-for-bit
and the lens reproduces the model's own head bit-for-bit, on the live model and dtype.

**Hook ordering and the final-KL identity** are covered structurally rather than by a stored
scalar: the residual identity `x_l = x_{l-1} + a_l + m_l` holding at 0.0 *is* the proof that
the hooks are on the right submodules in the right order (any misordering breaks it), and
`scripts/smoke_layer_lens.py` additionally asserts `KL(resid_L ‖ resid_L) ≈ 0` and that
`hidden_states[L] ≠ x_L`. `hidden_size` is carried unchanged in each sidecar's `_meta`.

### Why the recovery is faithful, not an independent re-estimate

The replay reproduces the original log **to every printed digit**:

| Cell | 184776 log (original) | 186485 replay | |
|---|---|---|---|
| ars_gsm8k_r1distill8b | median 6.70e-05, frac_close 0.993 | 6.70e-05, 0.993 | identical |
| noise_gsm8k_phi3mini | 1.97e-04, 0.944 | 1.97e-04, 0.944 | identical |
| lapeigvals_gsm8k_phi35 | 1.36e-05, 0.939 | 1.36e-05, 0.939 | identical |
| noise_gsm8k_mistral7b | 2.91e-04, 0.981 | 2.91e-04, 0.981 | identical |

Same candidates, same prompts, same warpers, same weights — so these are the *original*
numbers, not a fresh measurement that happens to agree.

---

## 3. Proof the recovery was non-destructive

SHA-256 of every sidecar, taken **before** job 186485 and again **after**:

| Cell | SHA-256 (identical before and after) |
|---|---|
| ars_gsm8k_r1distill8b | `0014c54b6fd9883b1970a58b4cc2f364cf4882ad57fbec005f3c7366cb78fbcc` |
| noise_gsm8k_phi3mini | `3792f3a1e10a18491091e43398abbbdd0850439c35a842f9bc54bda0dd1ed7f0` |
| lapeigvals_gsm8k_phi35 | `0b4107403f07d08306a5e0f858628cfb8b86a7beca88aaab26c26e793f94d755` |
| noise_gsm8k_mistral7b | `1de6787d236238b02e8c0afd9e7ee87cfa0a185736076f9147057bc9822f200a` |

Source raw caches (untouched throughout; this driver never writes them):

| Cell | raw cache SHA-256 |
|---|---|
| ars_gsm8k_r1distill8b / `raw_gsm8k_T0.0.pkl` | `ae33ac6139828c1a69fb8887b950cc956323707d537d10c20bebf1108d8f2dc8` |
| noise_gsm8k_phi3mini / `raw_gsm8k_T1.0.pkl` | `10ccb627b29021b9f20757b500285404215097c858cfcea0b2e20c89f8fae5d5` |
| lapeigvals_gsm8k_phi35 / `raw_gsm8k_T1.0.pkl` | `0cb4b31f13fb59f34f786db1d931f1d99daee71a0cfa10402e523a859477dde8` |
| noise_gsm8k_mistral7b / `raw_gsm8k_T1.0.pkl` | `9c80391981959c8196f0816db0caca818e4828a5cfbcd31bafb4e9271af93ccf` |

The corrupted `layer_views_report_<cell>.json` files were **also left in place**, deliberately
— the recovery is written to a separate `RECOVERED_VALIDATION.json`, so the evidence of the
failure survives next to the evidence of the fix.

**Paths.** Cluster: `/shared/cycle2_tau_averbuch_prj/omrisegev1/results/repgrid/<cell>/`.
Drive: `gdrive:hallucination_detection/cluster_results/layer_views/<cell>/`, with the raw
hash listing at `.../layer_views/_provenance/layer_views_sha256.txt`.

**Command** (recorded in each `RECOVERED_VALIDATION.json` as `command`/`argv`):

```
python cluster/run_layer_views.py \
  --cells ars_gsm8k_r1distill8b,noise_gsm8k_phi3mini,lapeigvals_gsm8k_phi35,noise_gsm8k_mistral7b \
  --validate-only --gate-n 50 --report-name RECOVERED_VALIDATION.json
```

**Code commit**: `b7f24f5` (branch `whitebox/per-layer-views`). The `git_sha` field inside the
JSON is empty because `cluster/sync_code.sh` ships the tree without `.git` — a known
limitation of `_git_sha()`; the commit is recorded here instead.

---

## 4. The fix

Three changes, defence in depth (commit `38d3a37`):

1. **The gate no longer keys off the has-work list.** A candidate earns a forward pass if it
   needs its field **or** the gate is still filling; only the *field write* is skipped when the
   field exists. A resume now genuinely re-validates instead of measuring nothing.
2. **Validation evidence is stored in the sidecar's own `_meta`**, so it cannot be orphaned
   from the data it validates, and is never blanked — a run that gates nothing carries the
   prior forward.
3. **A zero-trace run reports `no_op_resume` with the prior verdict, not a gate FAILURE.**
   Measuring nothing is not evidence of failure.

A fourth bug surfaced while testing the fix: **`--validate-only` was not actually
non-destructive.** It skipped the final `flush()` but the periodic `--checkpoint-every` flush
inside the loop still fired, so a "validation replay" would have rewritten the sidecar it was
meant to leave alone. `flush()` is now a no-op under `--validate-only` — one guard covering
both paths. Without this, the replay above would have invalidated its own SHA-256 proof.

`scripts/smoke_layer_views_resume.py` reproduces the original failure on a CPU fixture
(extract, then resume with nothing to do) and asserts the resume preserves the verdict,
preserves the sidecar byte-for-byte, and that `--validate-only` does not touch the file. It is
what caught the checkpoint-flush bug.

---

## 5. Standing lesson

Validation evidence that lives in a *different file* from the data it validates can be
orphaned or overwritten while the data stays perfectly good. This is the same class as the
Step-193 staleness carriers. Two rules now hold for this arm:

- Evidence travels **inside** the artefact (`_meta.validation`).
- A run that measured nothing must never be allowed to **overwrite** a run that measured
  something.
