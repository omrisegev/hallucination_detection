# Full-population position-profile controls

Frozen post-review diagnostic; development-only.

| Method | PB % | Within AUC | PRMScore |
|---|---:|---:|---:|
| profile_only | 37.1600 | 0.753876 | 0.640343 |
| constant_profile | 37.4553 | 0.753436 | 0.636365 |
| mean_detrended | 39.6992 | 0.758086 | 0.637325 |
| location_scale_detrended | 38.5039 | 0.756870 | 0.636673 |
| mean__H0lim_VE0_VE075_VE1 | 37.4749 | 0.753436 | 0.634412 |
| append_innovation__H0lim | 39.8314 | 0.760293 | 0.638830 |

Historical earlier-VE0/VE075-peak PB only: 39.3857%. This is a step-choice rule, not a PRMB score curve.

See METRICS.json for all paired contrasts and primary98.75% intervals. Independent scalar PB and pairwise within-AUC audit: PASS.
Fits use no labels; source-excluded cell/length profiles borrow other answers. First-token missing history remains zero.
All references, Top10 order and gate are unchanged. Profile controls do not establish complete removal of every position interaction.
Original innovation5 remains visible; no new candidate is promoted by this diagnostic.
