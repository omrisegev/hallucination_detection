# Step 396 (Claude): digit disagreement as answer-gate evidence

Protocol frozen before results: `docs/experiments/DIGIT_GATE_EVIDENCE_20260916.md`. Script `claude_digit_gate_eval.py`;
full numbers `DIGIT_GATE_EVAL.json`. Development data, all 13,769 answers (gate decisions concern the 6,800 PB answers).
Locators frozen; every detector uses the same transductive within-cell midrank >= .33 rule, so the opened fraction
per cell is matched. Labels enter only the evaluation. Frozen-gate replays of innovation5 (39.8314%) and digit025
(41.3300%) reproduce exactly.

## Results

| locator @ gate detector | PB all-8 % | clean acc % | error exact % | error hits (+/− vs current gate) | clean successes (+/−) | correct peaks suppressed |
|---|---:|---:|---:|---:|---:|---:|
| innovation5 @ tail15 (current) | 39.8314 | 60.35 | 28.55 | 1268 | 1423 | 321 |
| innovation5 @ digit_rate | 37.2362 | 57.85 | 26.43 | 1174 (+218/−312) | 1364 (+471/−530) | 415 |
| innovation5 @ digit_count | 36.1160 | 52.42 | 26.86 | 1193 (+226/−301) | 1236 (+427/−614) | 396 |
| innovation5 @ digit_presence (confound control) | 29.7984 | 36.64 | 23.71 | 1053 | 864 | 536 |
| **innovation5 @ equal_rank(tail15, digit_rate)** | **41.0671** | **63.19** | **28.75** | 1277 (+144/−135) | 1490 (+286/−219) | 312 |
| innovation5 @ equal_rank(tail15, digit_count) | 41.0269 | 63.32 | 28.52 | 1267 (+138/−139) | 1493 (+267/−197) | 322 |
| digit025 @ tail15 (current) | 41.3300 | 60.35 | 30.14 | 1339 | 1423 | 357 |
| **digit025 @ equal_rank(tail15, digit_rate)** | **43.2546** | **63.19** | **31.02** | 1378 (+177/−138) | 1490 (+286/−219) | 318 |
| digit025 @ equal_rank(tail15, digit_count) | 43.1073 | 63.32 | 30.64 | 1361 (+163/−141) | 1493 (+267/−197) | 335 |

Paired source-group bootstrap, 10,000 draws:

| contrast | PB delta (pp) | CI | level |
|---|---:|---|---|
| innovation5: equal_rank(tail15, digit_rate) − tail15 (primary) | +1.236 | [−0.008, +2.478] | 97.5% |
| innovation5: digit_rate − tail15 (primary) | −2.595 | [−4.751, −0.530] | 97.5% |
| innovation5: equal_rank(tail15, digit_count) − tail15 | +1.196 | [+0.106, +2.295] | 95% |
| digit025: equal_rank(tail15, digit_rate) − tail15 | +1.925 | [+0.788, +3.065] | 95% |
| digit025: equal_rank(tail15, digit_count) − tail15 | +1.777 | [+0.636, +2.899] | 95% |

Detector AUC (erroneous vs clean, mean over the 8 PB cells): tail15 .800, digit_rate .729, digit_count .729,
digit_presence .568, equal_rank(tail15, digit_rate) .832, equal_rank(tail15, digit_count) .831.

## Reading

- Digit disagreement alone is a weaker gate than tail mass (AUC .73 vs .80) but complementary: their equal rank
  combination raises detector AUC to .83 and improves **both** clean accuracy (60.4 → 63.2) and error exact accuracy
  at the same opened fraction. The number of digits alone (presence) is near chance (.57) and destroys the gate, so
  the effect is the disagreement, not digit count or answer length.
- End-to-end on development data, the same new view used twice (locator correction at .25 and gate evidence) moves
  innovation5 from 39.83% PB / .7603 within to 43.25% PB / .7760 within. The primary contrast on the frozen innovation5
  locator touches zero at 97.5% ([−0.008, +2.478]); the secondary-locator contrasts exclude zero at 95%.
- Correct peaks suppressed by the gate fall only slightly (321 → 312): the new gate mostly changes *which* answers
  open, trading 135 lost error hits for 144 gained and 219 lost clean successes for 286 gained.

## Caveats

Development population, fully exposed; the digit view was selected on this population (Step 393 screening), and the
gate combination is the first and only combination tried here (no q sweep, no learned gate, no IU: two views only).
Untouched confirmation (MR-GSM8K) and a matched comparison against the earlier gate-feature selection contract
(`results/gate_feature_readout_selection_v1`, q=.3 token-mean rule) remain to be done. Math-specific view.
