# Agent step telemetry v1 — label-free NLL/entropy baselines (Qwen3-8B)

## ToolPRMBench — pairwise accuracy (chosen action less "bad"), 445 pairs; chance = 0.500
| feature | all | GTA | ToolSandbox | bfcl | tooltalk |
|---|---|---|---|---|---|
| mean_nll | 0.436 [0.391,0.483] | 0.339 (n=118) | 0.423 (n=130) | 0.441 (n=111) | 0.581 (n=86) |
| max_nll | 0.422 [0.382,0.467] | 0.432 (n=118) | 0.500 (n=130) | 0.428 (n=111) | 0.285 (n=86) |
| mean_ent | 0.584 [0.539,0.631] | 0.703 (n=118) | 0.738 (n=130) | 0.486 (n=111) | 0.314 (n=86) |
| len | 0.625 [0.582,0.667] | 0.487 (n=118) | 0.838 (n=130) | 0.514 (n=111) | 0.634 (n=86) |

## AgentProcessBench — step AUROC, error(-1) vs correct(+1); 2547 err / 5316 ok steps (label-0 excluded)
| feature | all | bfcl | gaia_dev | hotpotqa | tau2 |
|---|---|---|---|---|---|
| mean_nll | 0.491 [0.478,0.504] | 0.649 | 0.346 | 0.532 | 0.436 |
| max_nll | 0.487 [0.474,0.500] | 0.644 | 0.302 | 0.501 | 0.442 |
| mean_ent | 0.582 [0.569,0.595] | 0.675 | 0.445 | 0.680 | 0.525 |
| len | 0.567 [0.551,0.581] | 0.672 | 0.453 | 0.618 | 0.508 |

APB first-error localization (argmax step == first -1 step), 601 trajectories; random = 0.216
- mean_nll: 0.245
- max_nll: 0.280
- mean_ent: 0.316
- len: 0.301

APB trajectory AUROC (final_label -1 vs +1, max over steps), 503 fail / 459 ok
- mean_nll: 0.475
- max_nll: 0.450
- mean_ent: 0.596
- len: 0.534

## AFTraj-2K — trajectory AUROC unsafe vs safe; 1114 unsafe / 1162 safe
| feature | agg | all | agentic | coding | math |
|---|---|---|---|---|---|
| mean_nll | mean | 0.639 [0.616,0.664] | 0.447 | 0.772 | 0.784 |
| mean_nll | max | 0.443 [0.420,0.467] | 0.404 | 0.398 | 0.570 |
| max_nll | mean | 0.654 [0.631,0.676] | 0.429 | 0.851 | 0.761 |
| max_nll | max | 0.474 [0.451,0.497] | 0.441 | 0.551 | 0.422 |
| mean_ent | mean | 0.692 [0.670,0.713] | 0.511 | 0.951 | 0.829 |
| mean_ent | max | 0.666 [0.643,0.687] | 0.431 | 0.886 | 0.756 |
| len | mean | 0.523 [0.500,0.548] | 0.503 | 0.623 | 0.537 |
| len | max | 0.440 [0.415,0.463] | 0.471 | 0.492 | 0.429 |

AFT mistake-step localization (argmax step == mistake_step), 1114 unsafe runs; random = 0.128
- mean_nll: 0.243
- max_nll: 0.179
- mean_ent: 0.195
- len: 0.185
## AFT breakdown by unsafe_source (mean_ent, mean over steps; safe runs of same domain as negatives)
| domain | source | n | AUROC mean_ent | AUROC n_steps |
|---|---|---|---|---|
| agentic | diagnosed | 322 | 0.472 | 0.460 |
| agentic | injected | 148 | 0.598 | 0.260 |
| coding | injected | 247 | 0.951 | 0.305 |
| math | diagnosed | 64 | 0.448 | 0.656 |
| math | injected | 333 | 0.902 | 0.200 |

Reading: the high AFT AUROC comes only from synthetically *injected* errors; on naturally *diagnosed* errors it is at/below chance.

Inputs: Drive `cluster_results/agent_step_telemetry_v1/*_qwen3_8b_813d11b_full/records` (Qwen3-8B, teacher-forced, complete=True),
labels `local_cache/agent_step_telemetry_v1/inputs_v2_20261013/evaluator_only/`. Orientation fixed a priori (higher = more likely error); no fitting.
