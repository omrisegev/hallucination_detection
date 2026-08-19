# Layer-organic NRM result

Status: **PRELIMINARY / VALIDATION BLOCKED**.

The requested grouping was implemented exactly: one residual transformer
layer is one group, with entropy, target NLL, and top-1 surprisal retained as
three separate internal features. The result does not support adopting this
variant.

## Main numbers

Across the ten eligible 32-layer cells:

| Method | Macro AUROC | Macro AUPRC |
|---|---:|---:|
| Final-layer NLL | 0.7140 | 0.6007 |
| Compressed one-expert/layer IU-PCR | 0.6155 | 0.5129 |
| Atomic layer-triad IU-PCR | 0.5763 | 0.4969 |
| Organic NRM LODO | 0.5776 | 0.4987 |
| Organic NRM LOMO | 0.5758 | 0.4962 |
| Organic NRM LOCO | 0.5753 | 0.4961 |

The all-32-layer LODO correction over matched atomic IU-PCR is small and
positive: AUROC **+0.00136** [0.00079, 0.00194], AUPRC **+0.00185**
[0.00091, 0.00284], with AUROC W/T/L 6/1/3. It does not generalize across the
other source definitions: LOMO AUROC is -0.00049 [-0.00103, 0.00007] and LOCO
is -0.00091 [-0.00147, -0.00033].

The two cleaner controls are negative:

- same Llama-3.1-8B model across six datasets: -0.00075 AUROC
  [-0.00147, -0.00002], 2/1/3 W/T/L;
- same GSM8K dataset across five 32-layer models: -0.00154 AUROC
  [-0.00238, -0.00066], 0/2/3.

The KL-to-final sensitivity is also negative relative to its matched atomic
IU baseline: -0.00098 AUROC [-0.00153, -0.00041].

## Interpretation

The user's grouping premise is structurally sound, but the current NRM rule
does not exploit it robustly. The only positive transfer definition is the
heterogeneous ten-cell LODO aggregate; it reverses under the exact same-model
and same-dataset controls. The effect therefore looks cohort-dependent, not a
stable layer-level nuisance correction.

Keeping the three metrics atomic also weakens the IU base relative to first
averaging them into one expert per layer: -0.03925 macro AUROC
[-0.05973, -0.01799]. This comparison is highly heterogeneous (5 wins / 5
losses) and includes a severe TriviaQA failure (0.221 atomic vs 0.828
compressed). It says that the local measurements are genuinely related, but
merely declaring them a group does not regularize their *within-layer* IU
weights; the previous equal-mean compression did.

## Decision

Do not adopt layer-organic NRM and do not replace the frozen white-box result.
If this idea is revisited, the next distinct hypothesis should be a
within-layer fusion that first produces one regularized layer expert and only
then applies cross-layer NRM. That would be a new method, not a re-labeling of
this result, and should be frozen before another evaluation.

Canonical report: `REPORT.html`.
