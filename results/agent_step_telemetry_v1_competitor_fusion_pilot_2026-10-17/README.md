# Competitor + telemetry fusion pilot (CPU, label-free), 2026-10-17

Script: `fusion_pilot.py`. Raw output: `FUSION_PILOT.json`. The protocols and helpers are imported unchanged from
`../agent_step_telemetry_v1_n16_2026-10-16/fair_protocols/fair.py`.

**Fusion.** Equal weights, fixed a priori, no labels.
- **TPRM:** each view's pair preference is divided by its per-subset RMS, and the two are summed. RMS does not depend on sign, so the scaling needs no label.
- **APB:** per-subset z of the judge error score + z of our error score.
- **Abstentions:** a competitor with no output scores 0, which counts as a tie (0.5). Competitor accuracy over all pairs reproduces the published numbers up to these abstentions.

**Two kinds of fusion.**
- **TPRM is cross-backbone.** The competitors' own backbones were not run; our telemetry is from Qwen3-8B.
- **APB is same-backbone.** Both the Qwen3-8B judge and the Qwen3-8B telemetry come from the same model.

## TPRM: pairwise accuracy, competitor alone → fused with our length-residualized score
Same-format pairs, except GTA (all pairs, format probe). Gains are paired-bootstrap 95% CIs.

| subset (n) | competitor | alone | + mean_ent (resid) | + IU-PCR cellz (resid) |
|---|---|---|---|---|
| bfcl (111) | Skywork | .532 | .550 (+.018 [-.05,.08]) | .468 (-.063) |
| | Nemotron | .396 | .505 (**+.108** [.05,.17]) | .459 (+.063) |
| | Coder-Next | .405 | .459 (+.054 [.01,.10]) | .486 (**+.081** [.04,.13]) |
| tooltalk (86) | Skywork | .465 | .430 (-.035) | .360 (-.105) |
| | Nemotron | .762 | .605 (**-.157**) | .686 (-.076) |
| | Coder-Next | .855 | .767 (**-.087**) | .779 (-.076) |
| ToolSandbox (113) | Skywork | .788 | .841 (+.053 [.00,.11]) | .805 (+.018) |
| | Nemotron | .429 | .531 (+.102 [.05,.16]) | .619 (**+.190** [.11,.27]) |
| | Coder-Next | .827 | .867 (+.040 [.01,.07]) | .841 (+.013) |
| GTA probe (118) | Skywork | .542 | .712 (+.169) | .483 (-.059) |
| | Nemotron | .445 | .593 (+.148) | .483 (+.038) |
| | Coder-Next | .801 | .831 (+.030) | .712 (-.089) |

## APB: first error vs prior clean steps, n=468 error trajectories; judge = Qwen3-8B LLM judge (-1/0/+1)
| our component | judge alone | ours alone | fused | gain vs judge | step AUROC judge / fused |
|---|---|---|---|---|---|
| IU-PCR ansz, raw | .592 | .709 | .752 | +.161 [.129,.194] | .581 / .666 |
| IU-PCR ansz, length-resid | .592 | .623 | .692 | +.101 [.067,.132] | .581 / .641 |
| **IU-PCR ansz, length+position-resid** | .592 | .613 | **.681** | **+.089 [.055,.121]** | .581 / .637 |
| L-SML ansz, length+position-resid | .592 | .615 | .683 | +.092 [.062,.123] | .581 / .635 |
| mean_ent, length+position-resid | .592 | .528 | .613 | +.021 [-.011,.054] | .581 / .624 |

**Reading.** The judge's 3-level label leaves many ties inside a trajectory. Fusion mostly breaks those ties, and does so better than chance. With the length+position-residualized component, the gain cannot come from length or step position. On the fused score, per subset, gaia_dev is .69, hotpotqa .70 and tau2 .71.

**Caveats.**
- Many cells were compared and the CIs are not corrected for multiplicity.
- On TPRM, `cellz` is the non-canonical normalization.
- A judge+length or judge+position control was not run. Position alone wins 1.0 in the first-error units by construction, so only residualized components are fair.
