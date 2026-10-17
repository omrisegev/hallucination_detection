# Independent final review receipt

Requested by the user and launched after the analysis/report were completed.
Reviewer: separate agent `/root/v10_final_independent_review`.
Scope: new `advisor_v10_followup_20261011` scripts, outputs, figures and report;
read-only review, independent numerical checks, no inference or method fitting.

**Result: no major arithmetic or interpretation error found in this package.**
This result does not clear the separate unresolved research-code findings.

Two minor findings were accepted and fixed:

1. The position extension called its 45 development contrasts “registered.”
   The script/result now say “fixed-for-run development contrasts.” The extension
   was rerun; checks passed. No numerical algorithm or result changed.
2. The initial top1 reproduction command pointed to temporary research sources.
   It now points to the preserved `sources/dev_run_patched.py`.

Independent checks actually performed:

- Native PRM error flags use reward < 0.5; all 40 exact midpoint ties stay valid.
- All five frozen pooled scores and exhaustive tie-preserving scalar optima
  reproduce through separate ascending unique-score aggregation.
- Perfect-ranking ceilings reproduce from closed-form confusion counts.
- L-SML, IU-PCR and PRM AUC/top1/position means reproduce through sorted-list
  and bisect calculations independent of the main pair/rank implementation.
- Two representative 5,000 whole-source-group intervals reproduce exactly:
  L-SML minus PRM for top1 and pooled macro-F1.
- LR ordering matches three frozen method arrays; four-way disagreement counts
  reproduce for both continuous methods.
- Full Sensitivity reproduces as 0.693382 / 0.695425 / 0.755078.
- All 43 original manifest entries matched before the accepted edits; the parent
  regenerated and checked the final manifest after this receipt.
- Both exported figures were visually inspected.

Limits: the reviewer did not certify raw inference, upstream feature extraction,
historical LR cache authenticity, or untouched generalization. It independently
reran two representative bootstrap intervals and reviewed the shared computation;
it did not independently rerun all 45 position contrasts.
