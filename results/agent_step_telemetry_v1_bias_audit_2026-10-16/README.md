# Agent step telemetry v1 — bias audit (Qwen3-8B label-free NLL/entropy), 2026-10-16

Run: `python -I audit_{tprm,apb,aft}.py <scratchpad>/out <this dir>` (inputs: compact `*_feats.json` + `inputs_v2_20261013`).
Outputs: `tprm_audit.{md,json}`, `apb_audit.{md,json}`, `aft_audit.{md,json}`. Orientation fixed a priori (higher = error). All residualisation is label-free (per-subset OLS on log token length / position).

## ToolPRMBench — pair construction is subset-specific; the pooled numbers mix opposite artefacts
- GTA: chosen = gold (always a JSON tool call, 96% with templated prose); rejected = model output, 58% prose-only. Same function name in 100%. NLL 0.339 is a human-gold-vs-model-text effect. Only 4 same-format pairs.
- ToolSandbox: chosen is shorter in 84% (median 36 vs 50 tokens): chosen = canonical milestone call (`timestamp_to_datetime_info` 43/130), rejected = more args with invented literals. Signal survives control: length-residualised entropy 0.823, NLL 0.769; length-matched entropy 0.750.
- bfcl: same format, no length gap → entropy 0.486, NLL 0.441 (chance/anti). tooltalk: entropy anti 0.314, persists length-matched (0.292); summed NLL 0.767.
- Pooled after length control: matched pairs (n=162) NLL 0.519 / entropy 0.494; residualised entropy 0.602 (driven by ToolSandbox).
- Fair: report per subset only; use same-format pairs (bfcl, tooltalk, ToolSandbox; drop GTA or report it as a format-provenance probe); report length-residualised and length-matched pairwise accuracy next to raw.

## AgentProcessBench — position and label propagation dominate
- 76% of -1 steps come after the first error (tau2 87%, gaia 76%). Absolute position alone gives AUROC 0.714 (gaia 0.811, tau2 0.823). In gaia, entropy falls with position (ρ=-0.33), which explains the 0.445 inversion.
- Length: AUROC 0.567 overall, bfcl 0.672. Entropy-length ρ 0.41–0.67.
- First error vs the clean steps before it, same trajectory: entropy 0.659 overall (bfcl 0.655, gaia 0.716, hotpotqa 0.686, tau2 0.602). Length-residualised: gaia 0.767, hotpotqa 0.586, bfcl 0.455, tau2 0.467. So the gaia flip is a protocol artefact; bfcl/tau2 signal is length.
- Label 0 (428 steps) has little effect (0.580 vs 0.582).
- Fair: first-error localisation within each trajectory (first -1 against preceding +1 steps), per subset, reporting the length-residualised score and the length-only baseline.

## AFTraj-2K — injected errors are detected through injection artefacts
- All 728 injected runs have exactly one safe twin. The text is byte-identical up to the mistake step (math 99.7%, coding 100%, agentic re-exec 45%), so prefix telemetry is identical (paired win before the mistake 0.50).
- Paired win of trajectory mean entropy: coding 1.00, math 0.985, agentic 0.63. In math the mistake step itself wins only 0.77, but the later, scripted steps win 0.997. 80% of math-injected runs carry non-numeric call ids (`call_verify_shortcut`); safe runs have none (0%).
- Coding: the injected step (358 vs 83 chars, e.g. "finalize without handing off") ends the run within 1 step in 73% of cases.
- Diagnosed (natural) errors: entropy AUROC 0.45–0.47. Argmax at the mistake step 0.078 vs random 0.052 (math) and 0.177 vs 0.146 (agentic).
- Fair: headline = diagnosed subset (online prefix up to k*, mistake-step localisation). Injected pairs only as a paired twin test restricted to the mistake step (0.82 overall). The post-mistake suffix should not count.
