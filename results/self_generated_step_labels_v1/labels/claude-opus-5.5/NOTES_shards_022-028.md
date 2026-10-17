Judge claude-opus-5.5, shards 022-028. Start 2026-09-29 00:44, end 2026-09-29 00:56 (local clock).

Suspected reference errors:
- J0698: problem says 8:00 AM to 11:00 PM (15 h -> 1800); reference uses 11 AM (360). Candidate labelled -1 with reference_suspect.
- J0671: reference computes third-race loss as 1.5*10 instead of 1.5*11 (answer should be 3.5, not 3).

Ambiguous items / interpretation calls:
- J0711: candidate answers -3600 ("area decreases by 3600") vs reference 3600; treated as matching.
- J0722: "how many magazines" read by reference as issues (66); candidate counted subscriptions (5). Low confidence.
- J0594 (seat for the host), J0608 (simple vs compound interest), J0619 (base of 25% fee), J0673 (5- vs 7-day week), J0703 (bonus on old vs new salary), J0720 (sequential vs parallel tasks): wording ambiguous; labelled against the reference reading at medium confidence.
- J0615: odd "total 18 hours" wording; reference sums per-animal times.

Borderline first-step choices:
- J0628, J0704, J0610: correct final answer but a false statement or invalid inference in a step.
- J0572 (unproven linear ansatz, all computations right) labelled -1.
- J0564, J0575, J0655, J0645, J0657, J0725: where an error could be placed across two steps, the earlier step was chosen.
- J0614: small arithmetic slips plus a decimal final answer (1.73) instead of the exact sqrt(3).
