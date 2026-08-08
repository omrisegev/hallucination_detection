# Advisor brief: from GL-LIU v1 to unified DUFS-LIU

## One-sentence result

We tested whether the same DUFS-LIU construction should be used for global
error detection and token localization. It improves ProcessBench F1 from
31.36% to 31.72%, while expanding the local feature pool from
5 to 28 curves reduces it to 29.03%.

## What we built

GL-LIU has two heads over one LLM generation:

1. a global head decides whether the complete reasoning trace contains an error;
2. a local head ranks tokens to find the first erroneous step.

Both heads use the same two-component Laplacian IU-PCR equation. GL-LIU v1 used
a DUFS-gated sample graph globally and a temporal-chain graph locally. The new
unified candidate uses a DUFS-gated sample graph in both heads.

## What the controlled experiment says

- Global DUFS-LIU again beats global IU-PCR in all eight cells, by about 0.22
  AUROC percentage points.
- Local DUFS-LIU with the five frozen curves is slightly better than temporal
  LIU outside the development cells.
- The unified system is +0.37 ProcessBench-F1 points over
  GL-LIU v1 and +6.02 points over Mind the Gap.
- The unified improvement is mixed: five cell wins and three losses.
- The 28-curve locator loses 2.70 points against the five-curve
  locator and loses in seven of eight cells.

## Interpretation

The DUFS graph is useful as a small regularizer, but DUFS does not know what an
error is. With many token curves it learns a stable geometry of token
confidence and distribution shape that does not match first-error position.
The five native dynamics contain less information but a better target-aligned
inductive bias.

## Proposed discussion decision

Freeze two local candidates for external confirmation:

- primary simplicity candidate: local DUFS-LIU, five views;
- robustness control: temporal LIU, the same five views.

Do not optimize another token feature pool on the current ProcessBench labels.
The next decision should come from a new dataset/model family and additional
published localization baselines.

## Exact claim we can make now

On the existing ProcessBench outputs and shared repeated-calibration protocol,
using DUFS-LIU in both heads gives the best internal macro result, 31.72%
F1. The gain over frozen GL-LIU v1 is small and not uniform, so it is a leading
candidate rather than a confirmed replacement.
