# Graph Geometry Selection Research V1 — final mechanism audit

**Status: `PASS_WITH_CAVEATS`; bounded finding: `GEOMETRY_SEARCH_SELECTION_OPTIMISM`.**

This independent post-transfer audit was read-only with respect to frozen development, post-report, plot, and transfer artifacts. It created only this report and `MECHANISM_FINAL.json`.

## Exact anchor and factorial decomposition

The two required anchors reproduce exactly:

- fixed residual union-k7, one-SE/tail guard, canonical trust: **+0.25147679442711046pp**, bootstrap CI **[+0.027372046023322976, +0.4584870048518398]pp**, 6/8 positive families;
- fixed residual union-k7, max-mean, canonical trust: **+0.449629196668661pp**, CI **[-0.0018748594028006169, +0.8955066042413282]pp**, 6/8 positive families.

The exact point decomposition is:

`0.25147679442711046 + 0.19815240224155053 = 0.44962919666866100pp`, where the second term is max-mean minus one-SE at fixed union-k7/canonical trust. The legacy searched V1 value decomposes as

`0.25147679442711046 + 0.19815240224155053 + 0.001976638455165327 + 0 + 0 = 0.4516058351238263pp`,

where the remaining terms are matched geometry capacity, trust-grid, and legacy-five-lambda versus common-eight-lambda effects. Thus essentially all of the apparent +0.251-to-approximately-+0.450 jump is selector aggressiveness; graph search contributes only +0.001976638455165327pp in that matched contrast.

All 12 common-eight-lambda factorial cells are:

| Capacity / selector | canonical | V1 | expanded |
|---|---:|---:|---:|
| fixed / one-SE | +0.2514767944 | +0.1598000249 | +0.1598000249 |
| fixed / max-mean | +0.4496291967 | +0.4496291967 | +0.4496291967 |
| searched / one-SE | +0.2671109758 | +0.1875589038 | +0.1875589038 |
| searched / max-mean | +0.4516058351 | +0.4516058351 | +0.4516058351 |

The one-SE V1-minus-canonical effect is negative for fixed (-0.0916767695pp, CI [-0.1743617905, -0.0199384756]) and searched (-0.0795520720pp, CI [-0.1822056777, -0.0035777532]); max-mean trust effects are exactly zero. V1 and expanded point selections coincide. The lambda grid is common across the factorial: `[0.03, 0.1, 0.3, 1, 3, 10, 30, 100]`.

## Policy-matched oracle, regret, and selectors

| Policy | Method | mean delta (pp) | policy-matched regret (pp) | oracle agreement |
|---|---|---:|---:|---:|
| fixed strength | canonical union-k7 | +0.2514767944 | 0.2005444990 | 2/8 |
| fixed strength | intrinsic label-free | +0.2198206244 | 0.2322006691 | 1/8 |
| fixed strength | held-family oracle | +0.4520212935 | 0 | 8/8 |
| one-SE | fixed union-k7 | +0.2514767944 | 0.2856233466 | 1/8 |
| one-SE | supervised donor-label selector | +0.2236220957 | 0.3134780453 | 3/8 |
| one-SE | held-family oracle | +0.5371001410 | 0 | 8/8 |
| max-mean | fixed union-k7 | +0.4496291967 | 0.2732102560 | 1/8 |
| max-mean | supervised donor-label selector | +0.4366823332 | 0.2861571194 | 3/8 |
| max-mean | held-family oracle | +0.7228394527 | 0 | 8/8 |

The full-tuple held-label ceiling is +1.0408366486pp and is not a policy-matched geometry oracle. Relative to fixed union-k7, the intrinsic selector changes the fixed-strength mean by **-0.0316561700pp**; the supervised selector changes one-SE by **-0.0278546988pp** and max-mean by **-0.0129468634pp**. Donor-label geometry ranks correlate with held ranks (mean Spearman 0.678571), but the selected policies do not improve transfer. Intrinsic ranks do not identify held performance (mean Spearman 0.020833).

## Matched DiD selection optimism

The audited estimand is `(searched_inner - fixed_inner) - (searched_outer - fixed_outer)`.

| Candidate bank | Policy | DiD (pp) | positive families | exact one-sided sign-flip p |
|---|---|---:|---:|---:|
| Phase A four geometries | one-SE | -0.0081036513 | 3/8 | 0.68750000 |
| Phase A four geometries | max-mean | +0.0034948789 | 2/8 | 0.37500000 |
| Phase B selector bank | one-SE | +0.2513550868 | 6/8 | 0.04296875 |
| Phase B selector bank | max-mean | +0.1636122088 | 5/8 | 0.26562500 |

Phase A shows neither useful geometry gain nor optimism. The expanded Phase B bank shows clear one-SE selection optimism; max-mean has a positive but noisy point estimate. This supports the bounded optimism finding, not a deployment gain from geometry search.

## Full versus cross actuator

`full` and `cross` are disjoint frozen arms. No selector was allowed to choose actuator, cross has no lambda, and transfer cross entries explicitly record `actuator_was_selected=false`.

At canonical trust, full-minus-cross is +0.0063538006pp for residual union-k7 under one-SE (CI [+0.0018476871, +0.0117347395], p=0.0078125) and +0.0003276770pp for adaptive-k7 (CI [-0.0037252931, +0.0038328723], p=0.4609375). Under max-mean these become +0.1302363945pp (CI crosses zero, p=0.08203125) and +0.0612993015pp (CI crosses zero, p=0.0859375). Across the other geometries, conservative one-SE increments range from -0.0233013168pp (contribution) to +0.0498350880pp (shrinkage-Mahalanobis); aggressive max-mean increments are generally larger and less stable. These are actuator contrasts, not actuator-selection gains.

For all-source `cbar`, union-k7 and adaptive-k7 have norms 0.2216044915 and 0.1898630519; at lambda 0.03, `cos(d_full,-cbar)` is 0.9999110182 and 0.9999028272. Leave-one-source cosine means are 0.9903798945 and 0.9861924082, while real-to-node-permutation separation ratios are 22.6195 and 16.8276. The conservative correction therefore behaves almost exactly like the pooled cross-gradient direction, and that direction is stable and structurally non-permutation-like.

The new fixed-strength node outcome controls remain nuanced. For union-k7, full real-minus-permutation-mean is +0.2588076417pp with family CI [+0.0661979072, +0.4312540660], but the 20-permutation rank p is 0.142857; adaptive full is +0.2397736999pp, CI [+0.0616340944, +0.4247616598], rank p=0.047619. Full and cross controls are nearly identical for both. The inherited union-k7 control used a different deterministic permutation protocol and has p=0.047619; it must not be pooled with the new geometry-matched null. The inherited attribution audit remains `FAIL` because union-k7 did not beat the contribution/DUFS attribution gate; its controls apply only to that exact canonical arm.

## Supported conclusions

- **Useful geometry gain:** not supported. Matched Phase A search effects are +0.0156341814pp under one-SE/canonical and +0.0019766385pp under max-mean, with intervals crossing zero.
- **Selector gain:** max-mean explains a +0.1981524022pp point increase over one-SE, but its paired family CI crosses zero. Neither the label-free nor supervised geometry selector beats fixed union-k7 under its matched policy.
- **Held-label geometry headroom:** supported descriptively: policy-matched oracles reach +0.4520pp fixed-strength, +0.5371pp one-SE, and +0.7228pp max-mean.
- **Headroom identified without labels:** not supported. The intrinsic selector has 1/8 oracle agreement, 0.020833 mean rank correlation, and 0.232201pp matched regret.
- **Selection optimism:** supported for the expanded one-SE selector bank; max-mean evidence is directionally positive but inconclusive.

External transfer is retrospective stress testing, not confirmation. The intrinsic label-free method beats canonical on 4/5 opened panels but loses on HLE and was worse in the matched development policy; supervised transfer is also mixed. This does not overturn the development conclusion.

## Exclusions, plots, and tests

No SU-rho or SU covariance-cleaning arm is present: `RUN_DEFINITION.json` has `su_covariance_or_rho_arms=[]`, `PROVENANCE_AUDIT.json` has `su_arms_present=false`, and the selector roster excludes SU. DUFS is control-only.

Plot05 was corrected during the final audit. Its current three panels use fully policy-matched fixed-strength, one-SE, and max-mean comparators and oracles; the +1.040837pp full-tuple ceiling is explicitly separate. Current Plot05 is therefore not a blocker.

All relevant tests passed: 14/14 core graph-geometry tests, 4/4 post-report audit tests, and the transfer test. These include disjoint actuator keys, no lambda for cross, policy separation, matched-DiD semantics, static no-SU enforcement, sanitized manifest chaining, and frozen transfer boundaries.

Self-hash canonicalization: SHA-256 over the exact UTF-8 bytes preceding the `self_sha256` footer line.

self_sha256: 501e6f6244ad8bb4bae885b38f2b45b7a5c98ad767dc0de3fb901a802d6460a2
