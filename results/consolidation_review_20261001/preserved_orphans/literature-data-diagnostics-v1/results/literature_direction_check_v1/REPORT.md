# Literature direction check v1

This run independently checks saved full-data diagnostics; it does not fit a new model or modify benchmark scores.

## Sequential/networked direction

The statistic is lag-1 residual step correlation minus a within-answer permutation control. It tests whether ordered steps contain structure beyond the current one-unit RBM; it is not an exact last-token/first-token test.

| bank | all mean excess | 95% CI | exact | early | late | gate-miss | clean-correct | false-alarm |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 6 | 0.15710 | [0.15242, 0.16172] | 940/0.20309 | 2395/0.19554 | 244/0.30684 | 863/0.16219 | 1181/0.10112 | 1177/0.10592 |
| 12 | 0.13656 | [0.13230, 0.14104] | 778/0.17193 | 2664/0.17890 | 137/0.23355 | 863/0.13610 | 1181/0.07704 | 1177/0.08440 |

## What the paper-linked checks support

- Conditional reliability: the matched early/late interaction is retained from the saved audit; it justifies one carefully frozen conditional-fusion test, not a deployable router.
- Dependent classifiers/graphs: family specialization is real, but the graph-coupled router and held-out c-STG gate did not pass; no graph expansion is justified by these results.
- RBM/deep energy: residual correlations exceed one-hidden synthetic draws, but the existing task diagnostics do not show that extra capacity would improve localization; keep exact one-unit controls before deeper/CD variants.
- Sequential/networked: ordered residual structure is present. The category comparison below tests whether it is stronger in useful or failed localizations; a positive structure alone is not enough for a new model.

Full machine-readable details: `LITERATURE_DIRECTION_CHECK.json` and `SERIAL_CASES.csv`.
