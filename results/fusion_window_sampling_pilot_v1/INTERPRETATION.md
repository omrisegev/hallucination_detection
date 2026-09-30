## What we learned and how to continue

Keep full-grid IU as the working feature-fusion reference. This is a continuity
choice, not a claim that it beats every comparator. No tested sampling rule
establishes a consistent two-task advantage. Equal fusion remains essential:
under transposed DUFS it gives the same PB F1 as IU and a higher PRMB point
estimate; the learned-fusion advantage remains unresolved.

For IU, transposed DUFS changes PRMB by +0.01402 with CI [-0.04608,+0.07772]
and PB by -4.02 points with CI [-12.12,0]. Its shuffled control has the
largest observed IU PB score, 21.88%, versus full 17.71%, but the improvement
interval is [-8.51,+17.73] points. This is not evidence that the learned
window placement works. Do not select the shuffled control as a winner based
on this already inspected, very small development set.

IU top-risk sampling raises pooled PRMB from 0.62261 to 0.64217 and PB from
17.71% to 19.05%. Both paired intervals include zero. With the parent gate
fixed, PB stays at 17.71%; its observed improvement comes from changed
clean/error gating, not a higher aggregate score for choosing the error step.
Its mean within-answer PRMB AUROC falls from 0.67567 to 0.66647. Pooled rank
improvement therefore does not by itself prove improved within-answer
localization; fitting-subset normalization also changes cross-answer scores.

For Joint graph fusion, every reduced selector gives PB 4.17% versus 8.33%
for full fitting. Transposed DUFS lowers coverage from 43/58 to 35/58, with
only four valid PRMB answers. Uniform sampling raises coverage to 46/58 but
reduces PRMB AUROC by 0.02976 on the seven common valid answers, with CI
[-0.05126,+0.00454]. More admissible fits and better localization are different
outcomes. A near-zero or degenerate interval on a tiny selected subgroup
does not establish equivalence or population reliability.

The graph selectors are also sensitive to within-window block perturbation:
mean selected-set Jaccard is 0.646 for transposed DUFS and 0.474 for window
diffusion. DUFS seed agreement is higher, 0.859, so optimization-seed agreement
alone misses measurement sensitivity. No short-error preservation conclusion
is possible because the eligible PB group has no <=32-token first-error steps.

The next bounded direction is to keep every fitting row and study feature
reliability and label-free Joint regularization/stability. Retain lambda-zero,
meaningful graph, permuted graph, IU and equal references. Any reliability
penalty must affect the fusion fit or weight solve; scaling columns only to
undo it by z-scoring is not an intervention. Local block sensitivity is an
observable diagnostic, not verified measurement-noise variance or a LOCA
burst model. Separate changes in fusion ranking from the clean/error gate.

Sparse end-to-end scoring, task-aware sampling, the full comparator registry,
untouched confirmation on both benchmarks and the frozen-candidate 24-cell
transfer remain open. This stage develops and audits our fusion method; it
does not replace it with a standalone geometric detector or finish the wider
research objective.
