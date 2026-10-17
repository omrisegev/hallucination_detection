# Gate / locator attribution of the frozen broad50 run

The gate is identical across methods. Any PB difference between these methods is caused by locator decisions on gate-open answers. PRMB within-AUC is ungated.

| Method | Error hit | Gate only | Locator only | Both | Clean false alarm | Raw error exact % |
|---|---:|---:|---:|---:|---:|---:|
| entropy_H1 | 1102 | 299 | 2519 | 522 | 935 | 31.54 |
| equal50 | 1178 | 300 | 2443 | 521 | 935 | 33.27 |
| continuous | 1173 | 298 | 2448 | 523 | 935 | 33.12 |
| joint | 1193 | 293 | 2428 | 528 | 935 | 33.45 |
| joint_balanced | 1191 | 293 | 2430 | 528 | 935 | 33.41 |
| original4 | 1155 | 318 | 2466 | 503 | 935 | 33.16 |
| innovation5 | 1268 | 321 | 2353 | 500 | 935 | 35.77 |

Gate only: the locator peak is right, but gate is closed. Locator only: gate open, wrong peak. Both: gate closed and wrong peak.

The JSON contains label-oracle diagnostics and every cell. These ceilings are not achievable methods. PRMB ranking regressions cannot be explained by this PB gate.
