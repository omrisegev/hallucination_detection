# Standalone-model telemetry for the five PRMs on PRMBench (plan, 2026-10-22)

**Requested by Omri, 2026-10-22.** Every PRM-boosting result so far computes telemetry from the PRM's own
fine-tuned backbone (`cluster/run_prmbench_prm_telemetry.py --mode prm`). That telemetry has seen the PRM's
supervised training. Omri wants the same telemetry from each PRM's base model **as released, standalone**. Then one
PRMBench table per PRM can show every relevant result: mean / L-SML, Q80 / GAP, with and without the PRM output.

**Scope.** PRMBench only, the 6,969 answers of the existing runs. The standalone telemetry is reported **alone**:
plain mean and L-SML. It is **not** fused with the PRM score (Omri: "I have no intention of fusing them with the PRM
output"). Goal: an advisor update email on 2026-10-23. The other machine runs everything on AIRCC.

**Why it matters.** Qwen-7B is the only PRM where both sources exist (`results/prm_boosting_v1/README.md`, jobs
282802/282803). There, N16 telemetry gives within-AUC 0.7885 / PRMScore 65.93 from Qwen2.5-Math-7B-Instruct and
0.8131 / 67.20 from the PRM backbone, against 0.8012 for the PRM itself. The current documents' finding that
"telemetry alone ranks steps better than the PRM" was measured on backbone telemetry. Without the standalone runs we
cannot say whether that is an independent signal or the PRM's training seen through its hidden states.

## Models (base checkpoints from the model cards, checked 2026-10-22)

| PRM | standalone model | prompt style | notes |
|---|---|---|---|
| Qwen2.5-Math-PRM-7B | `Qwen/Qwen2.5-Math-7B-Instruct` | `qwen_cot_chat` | re-run with this code for uniform provenance; the 2026-10-12 run (job 282803) is the cross-check |
| Qwen2.5-Math-PRM-72B | `Qwen/Qwen2.5-Math-72B-Instruct` | `qwen_cot_chat` | about 145 GB; `--max-batch-tokens 8192` |
| PURE-PRM-7B | `Qwen/Qwen2.5-Math-7B` (base, not Instruct) | `qwen_cot_chat` | the base tokenizer must carry a chat template; the driver fails loudly if it does not |
| RLHFlow Llama3.1-8B-PRM-Mistral-Data | `meta-llama/Llama-3.1-8B-Instruct` | `chat` | gated: needs an HF token with access |
| Math-Shepherd-Mistral-7B | `mistralai/Mistral-7B-v0.1` | `plain` | no chat template: BOS + question + blank line |

Prompt styles (`--prompt-style`, `--mode instruct`):
- **`qwen_cot_chat`:** the historical code path, byte-identical on a tiny-model smoke test. Qwen chat template with
  the Qwen2.5-Math CoT system prompt; user = question; the steps joined by a blank line follow the generation prompt.
- **`chat`:** the same, without a system message.
- **`plain`:** for a base model without a template.

The text of the steps and the step token spans use the same `build_chain` / `step_token_spans` as every other cell.
Each run takes its rows (ids, question, steps, order) from the PRM's own backbone pkl (`--meta-pkl`), so the
answers and step counts match the backbone run exactly.

## Pipeline

1. **GPU, AIRCC** (`results/prm_standalone_telemetry_v1/aircc/submit_gpu.sh`): one teacher-forced pass per model,
   step tokens only, top-50 log-probs, the same output schema as before.
2. **CPU, AIRCC** (`aircc/downstream.sbatch`):
   - P1 parity first: rebuild the frozen pure_prm backbone bank with the exact command and compare it exactly.
   - Then, per PRM: the bank (`results/prm_boosting_v1/build_bank.py`, digit ids from the standalone tokenizer) and
     `ablation5.py` with the standalone bank and the PRM's own pkl. Export tag `<prm>__standalone`.
3. **Local, the other machine** (`results/prm_standalone_telemetry_v1/standalone_table.py`): reads the backbone
   and standalone exports from `COLLECTION_DATA`. It checks that both have the same answers, offsets, rewards and
   groups, and that every backbone row reproduces `prm_boost_collection_v1/tables/STATUS_MAIN.csv` exactly. Output
   per PRM:
   - rows: the PRM (native, same rule); backbone telemetry (mean, L-SML); standalone telemetry (mean, L-SML);
     PRM + backbone telemetry (mean, L-SML);
   - columns: within-AUC; PRMScore under Q80, GAP, flag-2 and the true count;
   - paired bootstrap contrasts (GAP, 2,000 draws by source group): standalone vs backbone; L-SML vs mean on the
     standalone telemetry; standalone L-SML vs the PRM; backbone L-SML vs the PRM.

## Budget and gates
- **Timing:** 12 rows per model, 1 h wall each (CLAUDE.md 2026-09-24 rule). Reference rates: the backbone pass was
  0.5-0.8 s/row for the 7B-class PRMs and 4.1 s/row for the 72B (`prm_boost_processbench_v1` timing, unbatched).
  The standalone pass is batched.
- **Rough full-run estimate before timing:** 0.3-0.8 GPU-h per 7B-class model and 2-5 GPU-h for the 72B, so about
  4-8 GPU-h in total.
- **Gate (Omri, 2026-10-22: "it should just run them, I don't care how long it will take"):** no budget limit. The
  timing jobs stay as a correctness check only. If they are clean (COMPLETE, 0 failures, 0 context overflow), submit
  the full runs at once. Stop and report only on a correctness failure.
- **Disk:** about 190 GB of new checkpoints (72B: about 145 GB) plus about 1 GB of raw telemetry per model.
- No sbatch without a `/preflight` PASS in the same session.

## HANDOFF for the AIRCC machine (in order)
1. **Prefetch, login node:**
   - `df -h $S` and the quota: about 200 GB must be free.
   - `export HF_TOKEN=...` (Llama access), then `bash results/prm_standalone_telemetry_v1/aircc/prefetch.sh`.
   - It prints one snapshot per model. Each must be exactly one directory.
   - Record `tokenizer_config.json` → `chat_template` presence for `Qwen/Qwen2.5-Math-7B`.
2. **Code:**
   - `bash cluster/sync_code.sh`, then freeze it: `cp -a $S/code $S/code_standalone_v1` (CODE_DIR).
   - `mkdir -p $W/code` and copy in `results/prm_boosting_v1/build_bank.py`, `spectral_utils/digit_token_ids.py`,
     `results/prm_boost_collection_v1/aircc/parity_npz.py` and `results/prm_standalone_telemetry_v1/aircc/common.sh`.
   - Check the paths in `common.sh`: `PRM_PKL` for all five, `$O/bank_pure_prm/FEATURES.npz`, the mask1 base
     `$R/thinkfix_rescore_v1/source_bank_thinkfix_mask1/FEATURES.npz` and `$S/code_thinkfix_rescore_v1`.
3. **`/preflight`:**
   - Tree, disk and TMPDIR on both machines.
   - The wheelhouse's transformers version, which must load Llama-3.1.
   - A CPU tokenizer smoke for the five models, which must print `TOKENIZER_SMOKE PASS`:
     `python3 aircc/tokenizer_smoke.py $CODE_DIR <prm>=<snapshot dir>=<style>=<PRM_PKL> ...`, one argument per PRM.
     It needs only the tokenizers. It passed locally on Llama-3.1-8B-Instruct, Mistral-7B-v0.1 and a Qwen2.5
     tokenizer; the Qwen-Math tokenizers are blocked by the Mac's network filter. Llama's template adds its default
     system header ("Today Date: 26 Jul 2024"). That is the model as released, so it is kept.
   - `pytest tests/test_standalone_prompt.py`.
4. **Timing:** `bash aircc/submit_gpu.sh timing`. Report s/row, max input tokens and the status from each
   `timing_<prm>/manifest.json`. If all five are clean, go straight to step 5: there is no budget stop (Omri,
   2026-10-22). Raise a wall in `common.sh` (`SA_WALL`) if the measured rate needs it.
5. **Full:** `bash aircc/submit_gpu.sh full`. A job that exits 85 (wall) is resumed by re-running the same command
   for that PRM. The driver refuses a resume with a different model, rows or prompt style.
6. **Downstream:** `sbatch aircc/downstream.sbatch` (from `$W/code`). It must print `P1 PASS` and
   `SA_DOWNSTREAM_DONE`.
7. **Upload:** `nohup bash aircc/upload_drive.sh > $W/upload.log 2>&1 &`.
8. **Local tables:**
   - Copy `ABL5_SCORES_<prm>__standalone.npz` into `$COLLECTION_DATA/prmbench_standalone/`.
   - Run `python results/prm_standalone_telemetry_v1/standalone_table.py`. Every parity line must PASS.
   - Commit `tables/` and the manifests; do not commit the raw pkls.
   - Optional cross-check: compare the Qwen-7B standalone run with the 2026-10-12 one (job 282803) on the first
     40 rows. Same model and prompt, so the telemetry should agree to bf16 precision.

## What is not claimed
The standalone runs test where the telemetry signal comes from. They are not a new fusion method. All PRMBench data
remain development data. Changing to standalone telemetry does not alter any frozen result.
