# prm_new_features_v1: CPU feasibility of new per-step PRM features (2026-10-26)

Note: `docs/research_notes/PRM_MODEL_NEW_FEATURES_2026-10-26.md`.
- **Scope:** label-free, development data (Socratic-PRMBench), CPU only, no new inference.
- **Population:** every row: 2,995 answers and 26,052 non-empty steps. This is not a subset.
- **What it measures:** redundancy only, not quality. It reports correlation and R² against the PRM channel and N16.

Command (venv Python 3.11, numpy 2.4.6):

```
python results/prm_new_features_v1/feasibility.py <data_dir> results/prm_new_features_v1 > results/prm_new_features_v1/feasibility.log
```

Inputs: read-only copies from `gdrive:hallucination_detection/cluster_results/prm_boost_external_v1/`. They are not
committed. Hashes (sha256):

| file | sha256 |
|---|---|
| full_qwen_prm_socratic/external_telemetry.pkl | 1cc61f65866404767326f198250381fbd400b120ceb4106a7a2e698f7dd8b474 (= bank MANIFEST `raw_sha256`) |
| full_qwen_prm_socratic/bank_v2/FEATURES.npz | 6d58e6d8fb1b951393e4b9a031ffdb899bbd2251a1dadff10922b5aca895c548 (= MANIFEST `features_sha256`) |
| full_qwen_prm_socratic/bank_v2/REWARDS.npz | a50c6dd316ef1b1be6946ed6308af7e3c89ba0ad3d0d9b2ec456560ccb21698f |
| full_qwen_prm_72b_socratic/bank_v2/FEATURES.npz | 740624a7f6836ce88f89704b4449feaa139ae3bbf15f899abfb6d6a785e576f7 |
| full_qwen_prm_72b_socratic/bank_v2/REWARDS.npz | 7d978048330bee5f867465078b71658efd3c9c98143c0ca7705114626a52be4e |
| full_pure_prm_socratic/bank_v2/FEATURES.npz | acb909eb9957bc66d23612430b800e5dec337b24701adb97b5ae9cf7e44ce656 |
| full_pure_prm_socratic/bank_v2/REWARDS.npz | bb5674b26c35da67e06b88a288e2e7ceddfd18f7bf2cd9400dd0c26991b729aa |
| full_rlhflow_prm_socratic/bank_v2/FEATURES.npz | 7e6aeabd2d8f0b713d85c8fe4f67f236d82064d6c22f6b528476659e1881da7d |
| full_rlhflow_prm_socratic/bank_v2/REWARDS.npz | 20017488268559ece83122f6633d458b413dde64414514be9d4442789f0bba7a |

Outputs:

| file | contents |
|---|---|
| `FEASIBILITY.csv` | one row per candidate: valid rate, r with the PRM, max \|r\| with N16, R² on N16, R² on everything, r with position |
| `N16_REFERENCE.csv` | the same R² for each existing N16 channel against the rest |
| `FEASIBILITY_META.json` | population and the `H1_first_token` identity check |
| `feasibility.log` | stdout of the run |

The optional third argument (a backbone `tokenizer.json`) adds the two reflective-token channels. It has not been run
with a real vocabulary: the tokenizer is not on this Mac or on Drive. A synthetic-vocabulary smoke test of that path
passed; its output was discarded.
