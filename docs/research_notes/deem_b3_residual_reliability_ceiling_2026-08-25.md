# B3 residual reliability and ceiling audit (2026-08-25)

## Decision

`NO_TRANSFERABLE_RELIABILITY_SIGNAL_AT_REQUIRED_SCALE`.

This is a retrospective/C-tier diagnostic, not confirmation.  The natural
24-cell labels had already been opened elsewhere.  Within this audit, the
official eight screen cells were nevertheless kept as the only labels used to
select the primary gate; the rule was serialized before loading the other 16
sidecars.

## Question and data split

The audit asks whether sample-level, target-free diagnostics can identify when
an IU or IU-PGRD residual should modify frozen five-seed B3.  The diagnostics
are within-cell B3 rank, IU/B3 disagreement, residual sign and magnitude, B3
seed instability, IU-family residual novelty, and B3 roughness on the frozen
true-LOO residual graph.

The screen has one cell from each of eight dataset families.  The remaining 16
cells belong only to GSM8K, Math500, and TriviaQA, so the held result tests
same-dataset-family/model transfer rather than transfer to eight new families.

## Screen-selected primary

The simplest winning rule in the core gate menu was:

```text
score = B3_z + 0.75 * PGRD_correction_z * I(B3_rank <= 0.50)
```

It gained only +0.000526 equal-family AUROC on the eight-cell screen.  Frozen
application to the other 16 cells changed equal-family AUROC by -0.000067 and
equal-family AUPRC by -0.000434.  Cell-macro AUROC changed by +0.000330.  The
rule won on 8 cells and lost on 8; family mean deltas were +0.001043 for GSM8K,
-0.000370 for Math500, and -0.000875 for TriviaQA.  It therefore misses the
requested +0.0025 equal-family AUROC threshold by a wide margin.

## Expanded diagnostics and donor-direction stability

An expanded screen-only menu that included every requested diagnostic selected
the full IU-minus-B3 residual on the lowest 15% graph-novelty rows.  It gained
+0.000823 on screen but lost -0.001493 equal-family AUROC on the 16-cell panel.
This sensitivity was added after the primary held labels had opened, so it is
post-hoc descriptive evidence only.

A separate target-free stability check recomputed the PGRD direction after
leaving out each donor dataset family.  Its bounded reliability weight was

```text
abs(mean_j sign(correction_j)) /
    (1 + sd_j(correction_j) / mean_j(abs(correction_j)))
```

Multiplying the frozen PGRD correction by this weight yielded +0.000574
equal-family AUROC and +0.000411 equal-family AUPRC on the remaining cells
(cell-macro changes +0.001696 and +0.002292, respectively).  The reliability
weight's AUROC for the pointwise event "the correction moves toward the true
class" was only 0.522.  The donor directions are often sign-consistent, but
that consistency does not reliably identify useful corrections.

## Supervised ceiling

There is residual headroom if target labels choose the step separately in each
cell:

- best scalar IU-minus-B3 step: +0.003661 equal-family AUROC;
- best scalar PGRD step: +0.004268;
- best two-dimensional IU/PGRD step: +0.007610.

These are in-sample, per-target supervised grid-search ceilings.  They are not
methods and are intentionally optimistic.  Their contrast with the failed
transferred gates locates the bottleneck: the residual directions contain
ranking information, but neither B3 confidence/novelty nor donor-direction
consensus identifies the target-specific sign and step reliably.

## Reproduction

Run:

```bash
/Users/osegev/Desktop/hallucination_detection/.venv/bin/python \
  scripts/diagnose_deem_b3_residual_reliability_ceiling.py
```

Machine-readable outputs are under
`local_cache/deem_b3_moe_v1/residual_reliability_ceiling_v1/`:
`FROZEN_GATE.json`, `RESULTS.json`, and `PER_CELL.csv`.
