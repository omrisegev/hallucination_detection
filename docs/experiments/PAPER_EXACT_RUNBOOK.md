# Paper-exact acquisition — runbook

Execution ladder for `HANDOFF_paper_exact_cluster_acquisition_2026-08-16.md`, as implemented.
Every command below is copy-pasteable. Nothing here reports a result.

**Branch:** `paper-exact/acquisition-v1`
**Contract:** `spectral_utils/paper_exact/` (`paper_exact_acquisition_v1`)
**Shared root:** `SH=/shared/cycle2_tau_averbuch_prj/omrisegev1`

## The rule the ladder encodes

Promotion depends **only** on schema, hashes, causality, parser coverage, determinism,
checkpoint/resume and resource safety — never on whether a method wins. Published values are
regression targets. If a stage's number disagrees with its paper, diagnose provenance; do not
tune toward the target.

## 0. Local gates — run before every sync

```bash
python scripts/test_paper_exact.py            # 86 checks: contract, causality, metrics
python scripts/smoke_paper_exact_drivers.py   # 32 checks: stub-model decode + manifest dry-runs
python scripts/recover_refrain_vocabulary.py  # re-derives V_BASE from the PDF's underlines
```

Both must pass. The dry-run checks exist because three of the first four cluster jobs died at
the manifest gate on faults that were seconds of CPU work to detect locally.

## 1. Sync

```bash
bash cluster/sync_code.sh
# sync_code.sh excludes *.pdf; the seven pinned papers must be copied once so the cluster can
# hash them itself:
#   scp "papers/<each of the 7>" aircc:$SH/code/papers/
```

On the cluster, regenerate the live sbatch (gitignored — it carries the token):

```bash
cd $SH/code
tok=$(grep -m1 '^export HF_TOKEN=' cluster/submit_inference.sbatch | cut -d= -f2-)
sed "s|^export HF_TOKEN=.*|export HF_TOKEN=$tok|" \
    cluster/submit_paper_exact.sbatch.template > cluster/submit_paper_exact.sbatch
```

Always launch through `submit_paper_exact.sbatch`, **not** `cpu_job.sbatch` — the latter does
not export `HF_TOKEN`, which is what made the first `meta-llama` prefetch return 401 despite the
token having access.

## 2. P0 + prefetch (no GPU work of consequence)

```bash
SUB="sbatch -p power-gpu --qos=owner_880"
$SUB -J pe_prefetch cluster/submit_paper_exact.sbatch scripts/paper_exact_prefetch.py \
     --stages L1,S1,S2,M1,C1
$SUB -J pe_p0 cluster/submit_paper_exact.sbatch scripts/paper_exact_p0_audit.py \
     --out $SH/results/paper_exact/p0 --clone-dir $SH/src --check-cluster
```

Run prefetch **first** — P0's `models_prefetched` check reads the hub cache, so running it
before the download reports a stale miss. Outputs `P0_REPORT.json`, `MODEL_REVISIONS.json`
(cite these revisions in every manifest) and `BLOCKED_ASSETS.json` for W1.

## 3. Pilots — protocol checks, never table rows

```bash
$SUB -J pe_l1_pilot --time=08:00:00 cluster/submit_paper_exact.sbatch \
     run_paper_exact_uprm_judge.py --mode pilot --out $SH/results/paper_exact/l1_uprm_judge_pilot
$SUB -J pe_s1_pilot --time=08:00:00 cluster/submit_paper_exact.sbatch \
     run_paper_exact_refrain.py --mode pilot --n-samples 30 \
     --out $SH/results/paper_exact/s1_refrain_pilot
$SUB -J pe_m1_pilot --time=08:00:00 cluster/submit_paper_exact.sbatch \
     run_paper_exact_deepconf.py --mode pilot --k 32 --n-questions 30 \
     --out $SH/results/paper_exact/m1_deepconf_pilot
$SUB -J pe_s2_pilot --time=08:00:00 cluster/submit_paper_exact.sbatch \
     run_paper_exact_leash.py --model meta-llama/Llama-3.1-8B-Instruct --dataset gsm8k \
     --mode pilot --sweep --out $SH/results/paper_exact/s2_leash_pilot
```

**What each pilot must produce before its full run is allowed**

| Pilot | Gate to read | Promotes when |
|---|---|---|
| L1 | `GATE_L1-uprm-judge-pilot.json`, `SUMMARY.json` | manifest passes; tokenization-failure count near zero; per-subset F1 in the neighbourhood of 49.8/42.8/29.4/26.6 |
| S1 | `GATE_S1-refrain-pilot.json`, `SUMMARY.json`, `BANDIT_STATE.json` | both arms complete; `stop_reason` shows real `policy` stops; parser coverage high; bandit state written |
| M1 | `GATE_M-deepconf-pilot.json`, `THROUGHPUT.json` | **the equality audit passes** (§4 below); `tokens_per_s` measured — this is what makes M2 schedulable |
| S2 | `GATE_S2-leash-pilot.json` | sweep completes; one central choice frozen **before** the full run |

A pilot is an implementation check. REFRAIN's 30-question pilot gives at most 25 adaptive
rounds after five cold-start pulls, so it cannot be the paper-specified MATH-500 attempt.

## 4. M1's equality audit — the gate that licenses the DeepConf name

```bash
python scripts/paper_exact_deepconf_offline.py \
    --run $SH/results/paper_exact/m1_deepconf_pilot \
    --out $SH/results/paper_exact/m1_offline
```

Read `GATE_M-deepconf-offline.json`. Until `deepconf_equality_audit` and `logits_stage_is_raw`
both pass, every DeepConf-derived number in this project is a **named proxy**, not DeepConf.
Do not submit M2 before this passes: 1.8B tokens acquired under an unvalidated confidence
definition is 1.8B tokens of unusable pool.

## 5. Full runs

```bash
# L1 — all 3,400 ProcessBench rows, the uPRM paper's own backbone
$SUB -J pe_l1_full --time=24:00:00 cluster/submit_paper_exact.sbatch \
     run_paper_exact_uprm_judge.py --mode full --out $SH/results/paper_exact/l1_uprm_judge_full

# S1 — REFRAIN, Qwen3-8B x MATH-500, both arms.
# The refrain arm CANNOT be sharded: SW-UCB state crosses questions and a shuffle is a
# different algorithm. The driver refuses --n-shards>1 for it. Vanilla may be sharded.
$SUB -J pe_s1_full --time=24:00:00 cluster/submit_paper_exact.sbatch \
     run_paper_exact_refrain.py --mode full --arms vanilla,refrain \
     --out $SH/results/paper_exact/s1_refrain_full

# S2 — one job per (model, dataset) cell; NO --sweep (the driver refuses it in full mode)
for m in meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct \
         microsoft/Phi-3-mini-128k-instruct mistralai/Mistral-7B-v0.1; do
  for d in gsm8k aqua; do
    $SUB -J pe_s2_${d} --time=24:00:00 cluster/submit_paper_exact.sbatch \
         run_paper_exact_leash.py --model $m --dataset $d --mode full \
         --out $SH/results/paper_exact/s2_leash_$(basename $m)_$d
  done
done

# M2 — full DeepConf pool, 30 x 4,096. Sharded by (question, trace) so all 8 workers finish
# together; sharding by question would idle 6 of them at the tail.
for s in 0 1 2 3 4 5 6 7; do
  $SUB -J pe_m2_$s --time=24:00:00 cluster/submit_paper_exact.sbatch \
       run_paper_exact_deepconf.py --mode full --k 4096 \
       --shard $s --n-shards 8 --out $SH/results/paper_exact/m2_deepconf_full
done
```

Full runs require a **clean tree** (`verify_manifest(require_clean_tree=True)`), so commit
before launching. Resume is always safe: same `--out`, resume by stable trace key, and the
manifest refuses any pinned-field drift.

Exit code 85 means "checkpointed, incomplete" — resubmit the identical command.

## 5b. Sizing M2 from measured throughput — do not guess this

The plan's 75–150 GPU-hour estimate for M2 was wrong by two orders of magnitude, and the
first M1 pilot burned an 8-hour slot discovering it. Size the run from `THROUGHPUT.json`,
never from an estimate.

Measured on Qwen3-8B / AIME24 / B200, one GPU:

| batch | tok/s | note |
|---:|---:|---|
| 1 | 47 | first pilot; 8B at batch 1 re-reads 16 GB of weights per token |
| 6 | 145.6 | first batched probe, before the live-tensor fix |
| 24 | *measure it* | production settings (`--audit-every 64`) |

Mean trace length measured at **9,655 tokens** (the paper implies ~15.1k), so the pool is

```
30 questions x 4,096 traces x 9,655 tokens = 1.186e9 tokens
```

Shard count follows directly:

```
gpu_hours   = 1.186e9 / tok_s_per_gpu / 3600
n_shards    = ceil(gpu_hours / target_wall_hours)     # target_wall_hours <= 24 per job
```

At 145.6 tok/s that is 2,262 GPU-hours — 16 shards would take 141 h of wall time, which is
too long. Do not submit M2 until the batch-24 probe shows a rate that puts the run inside a
few days at a shard count the cluster can absorb politely. If it does not, the options in
order of preference are: raise the batch size until memory or tok/s stops improving; raise
the shard count; and only then reconsider the pool size with Omri, since a reduced pool
cannot claim the paper's table.

KV-cache ceiling, for choosing the batch: Qwen3-8B is 36 layers x 8 KV heads x 128 dim, i.e.
**~147 KB per token per trace**. At the 32k cap that is ~4.8 GB per trace, so a 183 GB B200
holds ~34 worst-case traces alongside the 16 GB of weights. Batch 24 leaves headroom; the
driver halves the batch on OOM and logs it, per handoff §6 (reduce batch size and nothing
else).

## 6. Offline analysis (CPU, no GPU)

```bash
python scripts/paper_exact_deepconf_offline.py --run $SH/results/paper_exact/m2_deepconf_full \
       --out $SH/results/paper_exact/m2_offline
python scripts/paper_exact_l0_table.py --inventory --roots $SH/results   # wire sources first
python scripts/paper_exact_l0_table.py --roots $SH/results --out $SH/results/paper_exact/l0
python scripts/paper_exact_status.py --root $SH/results/paper_exact --squeue \
       --out results/paper_exact/STATUS.md
```

## 7. Stages deliberately not scheduled

| Stage | Why |
|---|---|
| **W1** Streaming Hallucination Detection | anonymous code endpoint unreachable at audit. `BLOCKED_ASSETS.json` is the deliverable; no substitute corpus, labeller or prompt may fill the row. |
| **L3** full trained uPRM | LoRA + an RL estimator described only in an appendix, no official code or checkpoint, ~44 H200-hours, and its honest ceiling is still `paper-specified`. L1 (their own control, their backbone) plus L2's released PRM/critic ceilings already bracket us above and below on the same 3,400 rows. Omri's call, 2026-08-17. |

## 8. Environment facts that cost a job to learn

- Compute nodes reach the **HuggingFace hub but not PyPI**. In-job `pip install` fails with a
  DNS error. The drivers depend only on the NGC image plus the warm shared cache; REFRAIN's
  redundancy scorer is `paper_exact.refrain.MiniLMEncoder` on plain `transformers` for exactly
  this reason.
- `sinfo` is not on PATH inside the Pyxis container. P0 reports Slurm visibility, does not gate
  on it.
- `cpu_job.sbatch` does not export `HF_TOKEN`; gated repos 401 under it even when the token has
  access.
- Python's string `hash()` is salted per process. Per-trace seeds use SHA-256, and
  `PYTHONHASHSEED=0` is pinned in the sbatch.

## 9. Reporting discipline

Per the handoff's closing rule: do **not** write conclusions into `HISTORY.md`, `PROGRESS.md`
or `Research_Directions.md` until the full result identities and report hashes are frozen, and
do not commit raw cluster data. `STATUS.md` and the per-stage `SUMMARY.json` files are progress
artifacts, not findings.

Never rank rows from different lanes in one table: localization, prefix detection,
single-trace stopping and multi-trace adaptive compute answer different questions on different
populations. Mind-the-Gap's native SLA belongs in its own panel — it is computed on erroneous
traces only.
