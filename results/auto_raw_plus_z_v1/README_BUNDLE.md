# CAT34 bundle (auto_raw_plus_z_v1): inputs, raw features, labels and code

Drive: `gdrive:hallucination_detection/cluster_results/auto_raw_plus_z_v1/bundle/`. Git: branch
`claude/auto-raw-plus-z-2026-11-03`, `results/auto_raw_plus_z_v1/`; the results are in commit f667f6679.

The bundle holds everything needed to use the CAT34 inputs or rerun the experiment on another computer,
without AIRCC and without the 1 GB telemetry pickles.

## Layout

| path | what |
|---|---|
| `inputs/INPUTS_<prm>.npz` | **Use this.** One file per PRM (qwen_prm_7b, qwen_prm_72b, pure_prm, rlhflow_prm). It has the exact 34-column L-SML input matrix `X34` with `names34`, the untransformed `raw17` with `raw17_names`, step labels and metadata (keys below). |
| `raw_features/<prm>/FEATURES.npz` + `MANIFEST.json` | The original raw PRM-backbone bank the run read, unchanged. It has 56 channels for ProcessBench and PRMBench cells; CAT34 uses cell `prmbench_qwen3_8b` and the 16 N16 channels. Qwen-7B comes from AIRCC `results/prm_boost_v1/bank_prm`, the others from `results/prm_boost_gen_v1/bank_<prm>`. |
| `labels/OOF_STEP_SCORES.npz`, `labels/OOF_ANSWERS.csv` | Step labels (`labels`, aligned to the FEATURES.npz offsets) and the answer-to-`source_group` table the run used. |
| `code/` | `auto_raw_plus_z.py` (the run), `export_bundle.py` (wrote `inputs/`), `reproduce_from_bundle.py`, `make_tables.py`, the two sbatch files. `code/spectral_utils/` holds vendored copies of `fusion_utils.py` (`lsml_continuous`) and `prmbench.py` (`prmbench_evaluate`); they need only numpy, scipy and scikit-learn. |
| `FEATURES.md` | The 34 columns: source key, what each measures, sign, transform. |
| `README.md`, `HANDOFF.md`, `TABLES.md`, `RESULTS.csv`, `BOOTSTRAP.csv`, `DIAG.csv` | Results and documentation of the run. |
| `SHA256SUMS.txt` | sha256 of every file in the bundle. Check with `sha256sum -c SHA256SUMS.txt` from the bundle root. |

## INPUTS_<prm>.npz keys

The file has one row per step, 94,203 steps in 6,969 answers. Steps of answer j are `offsets[j]:offsets[j+1]`.

- `row_ids`: PRMBench answer id; `offsets`; `step_index`: 0-based within the answer; `answer_of_step`.
- `step_error`: bool, the step label used for the AUCs.
- `classification`: PRMBench category; `correct` = the unmodified control answer.
- `error_steps_flat` / `error_steps_offsets`: the official 1-based error steps used by PRMScore.
- `source_group`: groups answers that share a source question; used for the source-group bootstrap.
- `reward`: the PRM's step validity probability; `prm_native_valid`: the PRM's own step labels.
- `raw17`, `raw17_names`: untransformed values. Column 0 is `log((1-r)/r)`; columns 1-16 are the
  FEATURES.npz values of N16, already risk-oriented.
- `digit_active`, `digit_names`: where the three digit channels are defined.
- `X34`, `names34`: columns 0-16 `z:<channel>` (answer-z, = current AUTO_N16 inputs); columns 17-33
  `raw:<channel>` (global rank -> normal scores). See FEATURES.md.

## Rerun from the bundle (any computer, CPU, about 1 min per PRM)

```
python code/reproduce_from_bundle.py inputs/INPUTS_qwen_prm_7b.npz code RESULTS.csv
```

This fits Z17 (`X34[:, :17]`) and CAT34 (`X34`) with `lsml_continuous` defaults and applies the Q80 rule. It
prints within-AUC and PRMScore and checks them against `RESULTS.csv`. On Windows (numpy 2.x) it printed
MATCH for all 8 numbers (Z17 and CAT34 for the 4 PRMs). The second argument can also be the repo root at
commit f667f6679 instead of `code`.

To use the features in another fusion, load `X34` (or `raw17` and apply your own transform). Keep
`offsets` for anything per answer.

## Rerun from raw (AIRCC)

`code/run.sbatch` (the run, job 304515) and `code/run_bundle.sbatch` (rerun plus this bundle, job 304575).
They read the AIRCC paths listed in HANDOFF.md. Job 304575 reproduced job 304515's
RESULTS/BOOTSTRAP/DIAG files byte for byte, and Z17 matched the stored AUTO_N16 on all 4 PRMs.

The telemetry pickles that the PRM rewards and metadata came from are not in the bundle; they are 1.0-1.1 GB
each. Their content that the run uses (rewards, PRM labels, category, error steps) is in `INPUTS_<prm>.npz`.
They are on AIRCC at `results/prm_boost_v1/full_prm/` and `results/prm_boost_gen_v1/full_<prm>/`.

## Socratic-PRMBench (`bundle/socratic/`, added 2026-11-03)

The same layout: `inputs/INPUTS_<prm>.npz`; `raw_features/<prm>/` (FEATURES.npz + REWARDS.npz + MANIFEST.json,
bank_v2); `labels/socratic.json` (gold: per answer `uid`, `correct`, `include`); and `code/`
(`auto_raw_plus_z_socratic.py`, `make_tables.py`). It also has the README, TABLES and result CSVs, and its own
`SHA256SUMS.txt`. The INPUTS keys are those of the PRMBench files plus `step_included`, `nonempty`,
`nonempty_lengths`, `position` and `score__<arm>`. The PRMBench `error_steps` / `classification` keys do not
apply here.

Rerun on any computer with the repo at this branch:
`python results/auto_raw_plus_z_v1/socratic/auto_raw_plus_z_socratic.py <repo> <tag> raw_features/<prm>
labels/socratic.json results/sml_weight_estimator_v1/socratic_sign/RESULTS_socratic_placement.csv <out>`.
Here `<tag>` is one of qwen_prm_7b, qwen_prm_72b, pure_prm, rlhflow_prm. Point it at a folder holding that PRM's
FEATURES.npz and REWARDS.npz. Z17 must equal the stored AUTO_N16; the script stops if it does not.
