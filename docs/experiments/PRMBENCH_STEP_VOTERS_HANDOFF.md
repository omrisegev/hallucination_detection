# HANDOFF — PRMBench step voters on AIRCC (for the agent on the cluster machine)

Written 2026-10-16 by Claude (Mac session) for Omri. **Execute this on the machine that has the
`aircc` ssh alias and TAU VPN.** Every command is exact. Stop and ask Omri at each 🛑.

## 0. Goal and context
Omri's idea: use several process scorers (PRMs, LLM judges, a critic) as **label-free voters** on the
same reasoning steps, and with them:
1. estimate how many erroneous steps each answer has (MoM, `spectral_utils/error_count.py::fit_mom`);
2. fuse them with continuous L-SML (`spectral_utils/fusion_utils.py::lsml_continuous`).

**ProcessBench pilot already done (negative):** `results/prm_voters_processbench_pilot_2026-10-16/README.md`.
- Qwen2.5-Math-PRM-7B alone beats both fusions: answer AUC 0.944 vs L-SML 0.913.
- PRMScore under Q80 / MoM-count is −5.9 / −7.6pp for L-SML.
- MoM underestimates the share of erroneous answers: 0.52 vs 0.65 truth.

ProcessBench has only first-error labels, so the real test needs **PRMBench**, which labels every
step. Only PRM-7B has been run there so far. This handoff adds the other voters.

| Voter | Model | Track | Driver |
|---|---|---|---|
| j8b | Qwen/Qwen3-8B (uPRM LLM-as-judge control) | A | `cluster/run_prmbench_step_voters.py --kind uprm_judge` (this branch) |
| j14b | Qwen/Qwen2.5-14B-Instruct (same control) | A | same |
| c72b | Qwen/Qwen2.5-72B-Instruct (ProcessBench critic prompt) | A | `--kind critic` |
| prm72b | Qwen/Qwen2.5-Math-PRM-72B | B | `cluster/run_prmbench_prm_telemetry.py --adapter qwen_prm_72b` (Codex branch) |
| rlhflow | RLHFlow/Llama3.1-8B-PRM-Mistral-Data | B | `--adapter rlhflow_prm` |
| pure | jinachris/PURE-PRM-7B | B | `--adapter pure_prm` (needs the Qwen2.5-Math-7B base head cached) |
| mshep | peiyi9979/math-shepherd-mistral-7b-prm | B | `--adapter math_shepherd` |

**Do NOT run Skywork or ReasonEval.** Both are blocked:
- Skywork: n+1 rewards for n steps; the protocol awaits Omri's decision.
- ReasonEval: its head is unvalidated.

## 1. Code — two trees, two ISOLATED cluster dirs
`SH=/shared/cycle2_tau_averbuch_prj/omrisegev1`.

**Never run `cluster/sync_code.sh`.** It overwrites the shared `$SH/code` with your whole tree, and
other sessions run jobs from there.

```bash
git fetch origin
git worktree add ../hd-prm-voters   65bcf88310d7ee98a2b4b314bd0602e7db064b9c       # Track A: integrated voters + error_count dependencies
git worktree add ../hd-prm-gen      fba8e8d9672313551bed341cb9b9ef6a55e4db07       # Track B, Codex package, use UNCHANGED
```
Sync each tree to its own dir with a narrow tar (run from inside each worktree):
```bash
tar czf - --exclude=.git --exclude=results --exclude=cache --exclude=dataset_cache --exclude=local_cache \
  --exclude=.worktrees --exclude='*.pkl' --exclude='*.pdf' --exclude='*.ipynb' --exclude=__pycache__ . \
 | ssh aircc "mkdir -p $SH/code_prm_voters_v1 && tar xzf - -C $SH/code_prm_voters_v1"      # in ../hd-prm-voters
# same command with code_prm_boost_gen_v1                                                   # in ../hd-prm-gen
```
Each needs `cluster/requirements.txt`, which is included in the tar.

## 2. Local CPU gates (before any GPU)
```bash
cd ../hd-prm-voters
PRM_SMOKE_TOKENIZER=<any local Qwen2.5 tokenizer dir> python scripts/smoke_prmbench_step_voters.py   # must print ALL SMOKE CHECKS PASSED
cd ../hd-prm-gen
python -m unittest discover -s tests -p 'test_prm*.py' -v        # Codex's own suite; see its VALIDATION.json for env vars
```
🛑 If either fails, stop.

## 3. Cluster preflight (read-only; use the `cluster-ops` sub-agent / `/aircc-status`, never raw ssh loops)
1. `squeue -u $USER`: note what is running. Do not touch other jobs.
2. `sacct -X -S now-14days`: confirm Account/partition/QoS. The last PRMBench job used `-A cycle3_tau_averbuch_prj -p power-gpu --qos=owner_940`.
3. HF cache (`$SH/hf_cache/hub`) must hold:
   - the 7 models above;
   - `Qwen/Qwen2.5-Math-7B` (PURE head);
   - the PRMBench dataset (`ls $SH/hf_cache/datasets | grep -i prm`).

   All sbatch files run **offline**. Prefetch anything missing with the repo's `cluster/prefetch.sbatch` pattern (login node / CPU job), never inside a GPU job.
4. Locate the existing PRM-7B telemetry pkl that Track B's T1 compares against: `find $SH -name prmbench_telemetry.pkl -size +500M 2>/dev/null` (expected `.../prm_boost_v1/full_prm/prmbench_telemetry.pkl`). Call it `$T7`.
5. `df -h $SH`. The critic saves top-50 logprobs; budget about 20 GB.

## 4. Pilots (N=30)
Set
`ACC="-A cycle3_tau_averbuch_prj -p power-gpu --qos=owner_940"` (or what step 3.2 confirmed) and
`OUT=$SH/results/prmbench_voters_v1`.

```bash
cd $SH/code_prm_voters_v1      # sbatch cds to CODE_DIR itself; this cd is just for the relative sbatch path
A="sbatch $ACC cluster/submit_prmbench_step_voters.sbatch"
$A --kind uprm_judge --model Qwen/Qwen3-8B             --n-samples 30 --out $OUT/uprm_qwen3_8b_pilot
$A --kind uprm_judge --model Qwen/Qwen2.5-14B-Instruct --n-samples 30 --out $OUT/uprm_qwen25_14b_pilot
$A --kind critic     --model Qwen/Qwen2.5-72B-Instruct --n-samples 30 --out $OUT/critic_qwen72b_pilot

cd $SH/code_prm_boost_gen_v1
B="sbatch $ACC --export=ALL,CODE_DIR=$SH/code_prm_boost_gen_v1 cluster/submit_prmbench_prm_telemetry.sbatch"
$B --mode prm --adapter qwen_prm --meta-pkl $T7 --n-samples 40 --compare-to $T7 --out $OUT/prm7b_t1     # T1 parity
```
**Gate: T1 must pass before any new PRM runs** (exact rewards / ids / spans; telemetry within 1e-5).
🛑 If T1 fails, stop and report it; do not run the new adapters. When it passes:
```bash
for pair in rlhflow:rlhflow_prm pure:pure_prm mathshepherd:math_shepherd; do
  name=${pair%%:*}; adapter=${pair#*:}
  $B --mode prm --adapter "$adapter" --meta-pkl "$T7" --n-samples 30 --out "$OUT/${name}_pilot"
done
sbatch $ACC --time=08:00:00 --export=ALL,CODE_DIR=$SH/code_prm_boost_gen_v1 cluster/submit_prmbench_prm_telemetry.sbatch \
  --mode prm --adapter qwen_prm_72b --meta-pkl $T7 --n-samples 30 --out $OUT/qwen_prm_72b_pilot
```
If the 72B models OOM on 1 GPU, resubmit that job with `--gpus=2`.

Pilot pass criteria:
- **Judges** (`manifest_uprm_judge.json` → `stats`): `n_failed` = 0. A failure is a marker-tokenization break, and these pilots are the first real Qwen3/Qwen2.5-Instruct tokenizer check on PRMBench text. Also no record with `recovery_error`.
- **Critic:** `n_truncated` ≤ 2/30 and `n_unscored` ≤ 2/30.
- **PRMs:** the job exits 0 (exit 2 = counted failures; read its manifest); no `context_overflow` rows; `n_step_text_mismatch` = 0.
- Anything else: 🛑 report and stop.

## 5. Full runs
Each long job's **resume chain must be linear**: `--dependency=afterany:<previous link>`. Exit 85 is not auto-requeued.
sbatch options must come BEFORE the script path; everything after it goes to the driver.
`chain N SBATCH_OPTS -- SCRIPT ARGS...` submits N linearly dependent links:
```bash
chain() { local n=$1; shift; local opts=(); while [ "$1" != "--" ]; do opts+=("$1"); shift; done; shift
  local prev="" dep=(); for _ in $(seq "$n"); do
    dep=(); [ -n "$prev" ] && dep=(--dependency=afterany:$prev)
    prev=$(sbatch --parsable $ACC "${opts[@]}" "${dep[@]}" "$@"); echo "  link $prev"; done; }

cd $SH/code_prm_voters_v1
V=cluster/submit_prmbench_step_voters.sbatch
chain 2 -- $V --kind uprm_judge --model Qwen/Qwen3-8B             --out $OUT/uprm_qwen3_8b_full
chain 2 -- $V --kind uprm_judge --model Qwen/Qwen2.5-14B-Instruct --out $OUT/uprm_qwen25_14b_full
# critic: 6 parallel shards, each its own linear chain of 3 links (~100+ GPU-h total; the PB critic needed 6 walls for 3,400 rows)
for i in 0 1 2 3 4 5; do
  chain 3 -- $V --kind critic --model Qwen/Qwen2.5-72B-Instruct --n-shards 6 --shard $i --out $OUT/critic_qwen72b_full
done

cd $SH/code_prm_boost_gen_v1
P=cluster/submit_prmbench_prm_telemetry.sbatch
G=--export=ALL,CODE_DIR=$SH/code_prm_boost_gen_v1
for pair in rlhflow:rlhflow_prm pure:pure_prm mathshepherd:math_shepherd; do
  name=${pair%%:*}; adapter=${pair#*:}
  chain 2 $G -- $P --mode prm --adapter "$adapter" --meta-pkl "$T7" --out "$OUT/${name}_full"
done
chain 3 $G --time=08:00:00 -- $P --mode prm --adapter qwen_prm_72b --meta-pkl $T7 --out $OUT/qwen_prm_72b_full
```
Record every job id printed.
The extra chain links are harmless when the first link finishes: a resumed driver sees every row done and exits 0.

**Do not resync either code dir while these chains are alive.** A requeue resumes from whatever is on disk.

## 6. Results → Drive and local
On the cluster (rclone at `$SH/bin/rclone`, remote `gdrive:`), detached:
```bash
nohup $SH/bin/rclone copy $OUT gdrive:hallucination_detection/cluster_results/prmbench_voters_v1 \
  --transfers 4 --checkers 8 --drive-chunk-size 64M --stats-one-line --stats 60s -v > $SH/logs/upload_prmbench_voters.log 2>&1 &
```
Pull the pkls to the analysis machine under `cache/prmbench_voters_v1/` (gitignored).

## 7. Analysis (CPU, run in the background — the official evaluator is slow-ish)
```bash
cd ../hd-prm-voters
python scripts/run_prm_voters_prmbench.py \
  --prm7b dataset_cache/four_localization/prmbench_qwen25math7b_full/prmbench_prm.pkl \
  --judge j8b=cache/prmbench_voters_v1/uprm_qwen3_8b_full/prmbench_uprm_judge.pkl \
  --judge j14b=cache/prmbench_voters_v1/uprm_qwen25_14b_full/prmbench_uprm_judge.pkl \
  --critic c72b='cache/prmbench_voters_v1/critic_qwen72b_full/prmbench_critic_shard*of6.pkl' \
  --prm prm72b=cache/prmbench_voters_v1/qwen_prm_72b_full/prmbench_telemetry.pkl \
  --prm rlhflow=cache/prmbench_voters_v1/rlhflow_full/prmbench_telemetry.pkl \
  --prm pure=cache/prmbench_voters_v1/pure_full/prmbench_telemetry.pkl \
  --prm mshep=cache/prmbench_voters_v1/mathshepherd_full/prmbench_telemetry.pkl \
  --out results/prm_voters_prmbench_v1
```
On real PRM-7B input the script reproduces PRM-7B's official PRMScore, **0.6546**. Check this first.

The script reports, on the 6,211 noncontrol answers:
- mean erroneous-step count per answer for each voter and MoM, vs truth (MAE, bias, exact rate);
- official PRMScore for each single voter, L-SML (all voters) and the MoM posterior under NATIVE / Q80 / MOM_COUNT;
- step AUC;
- paired source-group bootstrap of L-SML minus the best single voter.

Also run it once with **only the four PRMs** (drop `--judge` / `--critic`; keep PRM-7B plus three `--prm`). This answers whether comparably strong PRMs fuse better than the mixed pool did on ProcessBench. That is the main question.

## 8. Report back
1. Write `results/prm_voters_prmbench_v1/README.md` with:
   - the two runs (all voters; PRMs only);
   - the count table and PRMScore table;
   - coverage / drops, and each pilot's stats.
2. Append a `HISTORY.md` step tagged `[prm-voters]` (union-merge rules in CLAUDE.md) plus a short `PROGRESS.md` entry.
3. Commit on `claude/prm-voters-processbench-pilot` and push.

Never commit pkls or a live HF token.
