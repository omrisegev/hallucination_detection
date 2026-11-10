# ProcessBench first-error rule (change-point posterior) - 2026-11-10

Thread "Design first-error rule for ProcessBench". Branch claude/pb-first-error-rule-2026-11-10 (from
claude/c4u-vs-competitors-2026-11-07). Math: METHOD.md. Numbers: TABLES.md, RESULTS.csv, BOOTSTRAP.csv. Charts: charts/.

| file | what |
|---|---|
| export_pb_cells.py | ran ONCE on the PC (needs hd_collection_data): exports cells/PB_<prm>.npz (frozen C4U rank/posterior/flags, R1, G16, arms, standalone N16) |
| cells/ | the 4 exports (force-added; results/**/*.npz is git-ignored) + EXPORT_MANIFEST.json (sha256) |
| cp_rule.py | the change-point chain: histograms, EM, posterior, MAP |
| pbeval.py | ProcessBench F1 / SLA from predicted first errors (gold: results/prm_boost_processbench_v1/local/processbench_gold.json) |
| lab.py, run.py | all arms x 4 PRMs -> RESULTS.csv, PREDS.npz |
| analyze.py | paired bootstrap, TABLES-ready BOOTSTRAP.csv, charts |

Reproduce (cloud or PC, numpy + scipy + matplotlib, ~1 min): `python run.py && python analyze.py`.
