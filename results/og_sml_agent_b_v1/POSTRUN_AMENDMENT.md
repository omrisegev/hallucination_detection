# Post-run provenance amendment

Date: 2026-09-04

This amendment records a non-ideal rewrite of `T0_EXECUTION_REGISTRY.json` and a
disclosure-only rerun.  It exists so the current registry is not misrepresented
as immutable preregistration evidence.

## Timeline

1. Runner SHA `cd049d167ef245e55e830b1ddaee3f371629f176c3594c516cd0dc7f09baa64e`
   failed before reading the ledger because the repository root was absent from
   `sys.path`.  It produced no T0 result.
2. The import path was corrected.  Runner SHA
   `aa705684e97f32adf35ff7f1f02b072aeffa87d8e1e57938c6d668dbc4a491fd`
   produced the first successful T0 result under the already frozen prediction
   and stop rule.  The execution-registry SHA after that run was
   `8887750f1d857e9409ac33ddf19b7b731229bf4c46b7880d0ed6989aaac5d554`.
3. Independent review noted that `primary_gate_pass` is not a pure optimizer
   stability signal: in this ledger it equals the regularization/weight-sensitivity
   verdict in all 18 lanes, while multistart itself passes 15/18.  The runner was
   changed only to add that gate inventory and narrow the explanatory wording.
   No graph definition, input, prediction, comparison, gate, or numerical
   calculation changed.
4. The disclosure-only runner SHA
   `73b1d34dfc9ea7a2394936c1d08ec3498d3b7a24ca78a48cdf673e1ab8ce267f`
   reproduced the same cross-tab and `J` values.  The current execution-registry
   SHA is `020352d1e5fccd164d3e152b4b18b75b7f48adc845f0ff9f3789db7e2ba994b9`.

## First successful versus disclosure-only artifacts

The first successful result had these hashes:

- `T0_REPORT.json`: `a3f4954cfeb7717f650e176e38fa5c4c90bc55384fb2df8aa3d655ee63bce317`
- `T0_LANES.csv`: `4efc7809e14ac629e80770c031f17e7cb490023e3960caef8f73b762e8140b77`
- `T0_REPORT.md`: `2d0bff6cd1a6d4109bedee1bb99b0cf565dbe6ceeb140318839021ad6a250bd5`
- `T0_MANIFEST.json`: `a6748fe2ef06bdf49324230272ec1fbbbdb76d85d882fea4ffc41a2b1f5ba203`
- first renderer source: `e3c00a28bf42ad5906c351f823b8f45770eaf36fc3b4e315802511691960adbc`

The current disclosure-only result has these hashes:

- `T0_REPORT.json`: `2bbce5db2c58134bb2e44235e139dfbb5045b0bfaac7575acfc49fd1157e1b84`
- `T0_LANES.csv`: `4efc7809e14ac629e80770c031f17e7cb490023e3960caef8f73b762e8140b77`
- `T0_REPORT.md`: `8bf7d215ec8698eb8636717cddd85b3e4ecd32abd8f02029d9dd34eeccb3f4dd`
- `T0_MANIFEST.json`: `58d51e7ddec1bec7102ee0e625b224ff4e8c10fd61a36c2304912ca24e8014cc`
- current renderer source: `68ee6d8ef63d666b5838c18208554452e01821a3614c7e79c817d311f3a6a315`

The lane CSV is byte-identical across both successful runs.  Both runs report:

- prior pass and admissible: 0;
- prior pass and inadmissible: 3;
- prior fail and admissible: 6;
- prior fail and inadmissible: 9;
- minimum prior-pass `J_selection`: 0;
- maximum prior-fail `J_selection`: `0.07705855880674317`;
- terminal status: `T0_FALSIFIED_STOP_BEFORE_STEPS_0_6`.

## Claim boundary

The literal T0 prediction and Remark 6's proposed retrospective explanation are
falsified for the structures C-v2 actually fitted.  The graph theorems themselves
are not falsified.  C-v2 fitted one INTERNAL hard partition in every lane;
provenance was only a separately fitted reference, so the proposal's premise of
an overlapping INTERNAL+provenance primary fit was factually inapplicable.

