# Raw-channel readouts and cumulative-vote fusion, binary versus soft — v1

Status: PIPELINE COMPLETE AND SMOKE-TESTED ON PILOT CELLS; FULL-CELL RUN PENDING (needs the
telemetry that lives on AIRCC / Drive, not in git). Started 2026-09-20 (Step 424).

## Omri's request (2026-09-20)

1. Do not fuse the locators of existing algorithms (Step 423 did). Use the eleven raw token
   channels of the Claude feature bank and find, for each channel, the readout that suits it, so
   that some channels may localize **late** errors better.
2. Run the cumulative-vote fusion in its **soft** version as well as the binary one, and compare
   them in one table.
3. Verify the Mind-the-Gap numbers (done: `results/cumulative_vote_fusion_v1/MIND_THE_GAP_VERIFICATION.md`).

## The eleven channels

`spectral_utils/claude_feature_bank_v1.py` (copied from `codex/claude-feature-bank-token-lsml-v1`,
unchanged): q15_H1, q15_VE1, chosen_surprisal, logprob_margin, true_tail50, energy_level,
energy_innovation, top15_turnover, top50_js, dominant_freq16, bocpd_p0. All manually oriented to
"larger = more risk", all causal, no digit channel. Input: the project's rich-save telemetry row
(`token_entropies`, `token_spilled_energies`, `token_logsumexp`, `top_k_logprobs` with K ≥ 50,
`step_token_spans`, `label`).

## Readouts (token series → per-official-step profile)

Every channel is first standardized **within the answer** (median / IQR). Then:

| readout | definition | targets |
|---|---|---|
| top5 | mean of the 5 largest tokens in the step (the frozen incumbent readout) | level |
| top10 | mean of the 10 largest | level |
| max | largest token | extreme |
| mean | step mean | level |
| log_top5 | negative Laplacian-of-Gaussian (σ = 2.5 tokens) of the series, then top5 | delta / spike |
| cusum_top5 | \|CUSUM\| of the standardized series, then top5 | level shift |
| onset80 | prefix profile: top5(s) up to the **first** step reaching 80% of the answer's max top5, −∞ after; its argmax is that first step | onset, anti-late |

A (channel, readout) pair is a localizer: ŝ = argmax of the profile (earliest step wins ties).
Two rosters are fused: `fixed_top5` (every channel read by top5, label-free) and
`train_selected` (per channel, the readout with the best SLA **on the training folds**; this is a
label-selected development choice and is labelled as such everywhere).

## Fusion, out-of-fold on the same (question, n) instances

- **Binary**: v_j(n) = +1 if ŝ_j ≤ n. Median of positions, SML weighted median, L-SML mode,
  Dawid-Skene mode (`scripts/experiments/cumulative_vote_fusion_v1.py`).
- **Soft**: F_j(n) = Σ_{s≤n} softmax(z(r_j(s))/τ). Instances carry 2F_j(n) − 1; fits are
  `sml_fuse_signed` on the continuous columns and `lsml_continuous` with group discovery. Fused
  CDF = Σ_j w_j F_j (weights clipped at 0, normalized), pool-adjacent-violators, then the mode of
  the pmf (0/1 loss) or its median. As τ → 0 the soft equal-weight median reproduces the binary
  median (numerical identity check in every fold; 0.90–1.00 on the pilots, the residue being
  exact ties).

Protocol: Mind-the-Gap SLA on erroneous answers only, no gate; five question folds by a stable
hash of the unit id; labels used only for evaluation, the paired bootstrap, and the declared
`train_selected` readout choice inside training folds.

## Pilot feasibility runs (NOT evidence — 30 answers per cell; CLAUDE.md 2026-09-07)

Cells reachable from this session through the Drive connector (10 MB cap): Llama-3.1-8B pilots
gsm8k + math (60 answers, 30 erroneous) and Qwen3-4B pilot gsm8k (30 answers). Outputs in
`results/raw_channel_readout_fusion_v1/pilot_*`.

What the pilots show about **the machinery**, which is all they can show:

- The readout grid runs on all 77 (channel, readout) pairs; `onset80` cuts the late fraction from
  0.37 (top5, mean over channels) to 0.17 at the same mean SLA, and is the training-fold choice
  for q15_H1, q15_VE1 and energy_level in 5/5, 5/5 and 4/5 folds. This is the hypothesised
  anti-late readout; the full cells decide whether it holds.
- Binary and soft fusion agree on the pilots within their (enormous) intervals; the soft **median**
  readout is markedly worse than the soft **mode** on short answers because the diffuse softmax
  mass shifts the CDF's 0.5 crossing later. Use the mode for the soft path under 0/1 loss.
- τ ∈ {0.25, 0.5, 1, 2} moves the soft results by at most a few answers on 30; no τ is selected.

## Running the full cells (the actual experiment)

On the machine that holds the caches (`dataset_cache/repgrid/pb_{llama31_8b,qwen3_4b,qwen3_8b}/
processbench_{gsm8k,math,olympiadbench,omnimath}.pkl`, LFS on Drive / AIRCC):

```
python scripts/experiments/raw_channel_readout_fusion_v1.py \
  --cell gsm8k=dataset_cache/repgrid/pb_llama31_8b/processbench_gsm8k.pkl \
  --cell math=dataset_cache/repgrid/pb_llama31_8b/processbench_math.pkl \
  --cell olympiadbench=dataset_cache/repgrid/pb_llama31_8b/processbench_olympiadbench.pkl \
  --cell omnimath=dataset_cache/repgrid/pb_llama31_8b/processbench_omnimath.pkl \
  --out results/raw_channel_readout_fusion_v1/full_llama31_8b_tau1
```

and the same for `pb_qwen3_4b` / `pb_qwen3_8b`. CPU only; the pilot (60 answers) takes ~40 s,
so a full scorer (3,400 answers, up to 900 tokens each, BOCPD per answer) is minutes, not
hours. No GPU, no inference, no label enters a fit.

Deliverables the full run produces automatically: `REPORT.md` with the per-subset readout grid
(SLA and late fraction for every channel × readout), the binary-versus-soft fusion table with
paired intervals against the max-entropy top5 localizer, and per-fold fits (weights, ψ/η,
L-SML groups, readout choices, τ→0 identity).

Pre-registered reading of the full result, written before any full cell was scored:
- Primary comparison: `fixed_top5:bin:*` versus `fixed_top5:soft:*_mode` (label-free rosters).
  `train_selected` rows are development ceilings.
- The long-chain question from Step 423: does `onset80` (or `log_top5`) on any channel reduce the
  late fraction on OlympiadBench / Omni-MATH without losing exact accuracy, and does a fusion that
  contains such localizers beat the top5-only fusion there? Report the late fraction next to SLA.
- No promotion from a single scorer; PRMBench and the gate are outside this study.

## Boundaries

- Pilot numbers are feasibility only. No comparative claim is made from them.
- Dataset-level `label` in the caches is the first-error step (0-based, −1 clean); clean answers
  are carried but not scored (SLA protocol).
- The Drive connector cannot fetch files above 10 MB, which excludes every full cell and the
  Omni-MATH pilots; the four full cells per scorer must be run where the caches are.

## Addendum 2026-09-21 — clarifications and the PRMBench extension

The discussion that followed the pilot run (eigenvector-versus-EM, what the matrix is, how the
weights are used at inference, and how the same fusion transfers to PRMBench's every-step
ranking with a per-step rather than cumulative encoding) is recorded in
`docs/research_notes/cumulative_vote_fusion_clarifications_2026-09-21.md`. The PRMBench mode is
designed there and not yet implemented.
