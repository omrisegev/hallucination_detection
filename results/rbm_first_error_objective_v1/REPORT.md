# First-error objective: complete

Branch codex/rbm-first-error-objective-v1; base f9d984266.
All6,800 ProcessBench answers,40 group-disjoint fits and unchanged token bank12
RBM correction/Logit/Top10/gate. Only the PB training objective changed:
step BCE versus categorical first-error loss over all steps of erroneous answers.
Clean answers had no first-error target and contributed zero location loss.

| PB objective | PB macro | Q4 | Q8 | exact hits | early | late |
|---|---:|---:|---:|---:|---:|---:|
| Step BCE (previous) | 37.2042% | 38.1921% | 36.2163% | 1507 | 1110 | 1825 |
| First-error listwise | 35.6451% | 35.9679% | 35.3222% | 1453 | 922 | 2067 |

Primary first-error minus BCE: -1.5591pp;97.5%CI [-3.1739, 0.0887];
within-AUC difference is exactly zero because PRMB rows are copied from the previous arm.
First-error loses in all eight PB cells. It gains 481 exact gated successes and loses 556;
412 of the losses are late choices. The result is not a universal rejection of first-error
training: it rejects this full-answer softmax correction under this frozen Top10 contract.

Checks PASS: direct first-error gradient/loss tests; zero-update replay inherited from
the matched study;40 saved models and6,800 test answers replayed; PB metrics and PRMB
identity replayed;10000 bootstrap draws reviewed. No nonconverged fits.
PRMB rows are inherited unchanged and are not new transfer evidence.

The verifier was amended after the first run to assert the intended PB-versus-BCE
comparison while requiring exact PRMB identity. The amendment and final hash are
recorded in MANIFEST_AMENDMENT.json; source inputs and saved models were unchanged.

Decision: keep step BCE as the supervised correction diagnostic reference. Do not open
another first-error variant, graph, position term or capacity sweep from this result.
Development evidence only; no untouched confirmation.
