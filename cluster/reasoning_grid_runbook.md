# Reasoning replication-grid runbook (Step 166)

Turnkey cluster steps for the 7 staged reasoning cells. Each runs **inference only** (K=1,
default capture = `token_entropies` + `top_k_logprobs`); we score **our** L-SML offline and
compare to the competitor's **published** AUROC. No competitor detector is reproduced.

`$SHARED = /shared/cycle2_tau_averbuch_prj/omrisegev1`. Partition/QoS: `power-gpu` / `owner_880`
(from `cluster/aircc.env`). Gated repos (Llama, Mistral) need the live `submit_inference.sbatch`
with a real `HF_TOKEN` — already set up (Step 156); Phi-3.5 / Qwen are not gated.

## The cells

| Preset id | Model | Dataset | N (full) | Published Y (fair, unsup) | Gated |
|---|---|---|---|---|---|
| `lapeigvals_gsm8k_llama3b`      | Llama-3.2-3B-Instruct        | gsm8k   | 1319 | AttentionScore 0.717 (probe 0.870) | yes |
| `lapeigvals_gsm8k_phi35`        | Phi-3.5-mini-instruct        | gsm8k   | 1319 | AttentionScore 0.666 (probe 0.885) | no  |
| `lapeigvals_gsm8k_nemo`         | Mistral-Nemo-Instruct-2407   | gsm8k   | 1319 | AttentionScore 0.630 (probe 0.890) | yes |
| `lapeigvals_gsm8k_mistral24b`   | Mistral-Small-24B-Instruct-2501 | gsm8k | 1319 | AttentionScore 0.576 (probe 0.925) | yes |
| `ars_gsm8k_qwen3_8b`            | Qwen3-8B                     | gsm8k   | 500  | ARS 90.37 (supervised)             | no  |
| `ars_math500_qwen3_8b`         | Qwen3-8B                     | math500 | 500  | ARS 78.66 (supervised)             | no  |
| `internalstates_gsm8k_qwen25_7b` | Qwen2.5-7B-Instruct        | gsm8k   | 500  | Internal-States 79.15 (supervised) | no  |

`ars_gsm8k_r1distill8b` (vs ARS 74.72) is already staged from Step 165; MATH-500/R1-Distill we
already have offline (GOOD_5 0.844).

## Gate ladder (per cell — CLAUDE.md: smoke → N=30 pilot → full N)

Smoke already PASSES for all (Step 166). Then, **with TAU VPN up**:

```bash
# 0. connectivity (a hang = VPN down)
ssh -o ConnectTimeout=5 aircc 'echo ok'

# 1. sync working tree (push-independent, do NOT git push)
bash cluster/sync_code.sh

# 2. N=30 PILOT (CLI --n-samples overrides the preset's full N)
ID=lapeigvals_gsm8k_phi35   # repeat per preset id
ssh aircc "cd $SHARED/code && sbatch -p power-gpu --qos=owner_880 \
    cluster/submit_inference.sbatch --preset $ID --n-samples 30 \
    --out $SHARED/results/repgrid/${ID}_pilot"

# 3. check the pilot: acc in [0.20,0.85] and trace NOT pinned at max_new
#    (use /aircc-status <jobid> or the cluster-ops agent — never raw ssh loops)
```

Pilot watch-outs: **Mistral-Small-24B / Qwen3-8B** are strong → may ceiling on GSM8K (acc >0.85 →
cell REJECTED by the band gate, expected). **Llama-3.2-3B / Phi-3.5** are the healthy mid-accuracy
cells. Reasoning models (Qwen3) keep thinking mode ON → verify traces aren't truncated at `max_new`.

```bash
# 4. FULL N (drop --n-samples; preset carries N=1319 / 500)
ssh aircc "cd $SHARED/code && sbatch -p power-gpu --qos=owner_880 \
    cluster/submit_inference.sbatch --preset $ID \
    --out $SHARED/results/repgrid/${ID}"
```

## Offline scoring + report (local CPU, after `/aircc-fetch`)

```bash
# fetch lands raw pkl + manifest under cache/repgrid/<id>/ (gitignored)
python scripts/inspect_cell.py cache/repgrid/${ID}          # schema/label-split check
python scripts/score_repgrid.py --cells ${ID}               # our L-SML GOOD_5 X vs published Y
python scripts/score_edis.py --pkl cache/repgrid/${ID}/raw_*.pkl --cell ${ID}   # optional EDIS
```

Then append the new rows (our X + CI, supervision, source, note) to
`results/reasoning_benchmark.csv` and regenerate:

```bash
python scripts/advisor_report.py     # reasoning-first report; guardrail scan must stay clean
```

For LapEigvals cells, confirm the report compares our **unsupervised** X to the **AttentionScore**
(unsup) Y — the probe number is the supervised ceiling, not the fair head-to-head.
