# HANDOFF: cross-model answer gate check (2026-11-14)

Status: DONE. Verdict NO-GO (see README.md). Nothing else is running.

Reproduce on another computer:
1. Stage 1 needs hd_collection_data and a checkout with results/c4u_telemetry_only_v1 (run.py, bracha_panel.py):
   `COLLECTION_DATA=<hd_collection_data> HD_ROOT=<checkout> OUT_DIR=<dir> python extract.py`
   The outputs of the PC run are already in steps/ (you do not need to run stage 1 again).
2. Stage 2 needs only steps/: `python analyze.py` (about 10 min, numpy/scipy/sklearn), then `python figures.py`.

Decisions in this run:
- Step score = L-SML of 16 global rank-normal channels without per-answer z (no PRM). Not the stored TELE_N16 (that one has answer-z).
- "Clearly below .26" was made concrete as < .20 before the run.
- DS votes: each model votes at its own q-quantile. q is not identified without labels, so 3 values are shown.

Open (needs Omri): the cross-model direction fails. The review's fallbacks were: (a) synthetic-error PU learning (GPU),
(b) an explicit budget gate (for example q0.3), (c) open gate (no-harm).
