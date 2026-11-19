Checks of the external review's claims (2026-11-19). Dev only, CPU, no inference; gold labels only as checks.
Code base: commit 804cd4d (results/ssl_one_experiment_v1/*.py), Python 3.13.3, HD_ROOT=analysis_v1/code.
python verify_review.py <code dir> <W> <processbench_gold.json> VERIFY_REVIEW.json
python verify_onset_global.py <code dir> <W> VERIFY_ONSET_GLOBAL.json
Discussion: ../REVIEW_RESPONSE_2026-11-19.md
