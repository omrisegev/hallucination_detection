# Tail15/50 screening: normalization error found while answering user's question

This is a code/formula audit, not a quality experiment or a replacement result.
Claude's digit-disagreement replay remains valid: it uses token IDs, not this
tail calculation. Do not change the frozen digit results or original Claude files.

In results/claude_real_checks_v1/claude_new_view_ceiling.py, lines64-66,
the tail signals use exp(lp - token_logsumexp), although lp is ALREADY a
full-vocabulary-normalized log probability.
Producer cluster/backfill_views.py candidate_quantities calls log_softmax;
cluster/run_teacher_forced.py saves top_k_logprobs_raw as top_k_logprobs.
Here “raw” distinguishes logits before generation warpers; it does not mean
unnormalized logits. token_logsumexp is logsumexp of the original raw logits.

Correct missing probability mass outside retained K:

    clip(1 - sum(exp(saved_logprobs[:, :K])), 0, 1)

Subtracting logsumexp again yields approximately1-exp(-Z)*head_mass. Taking
its log afterwards does not repair this double normalization.

Implementation check on the first row returned by the GSM8K/Qwen3-4B source map
(71 tokens, source processbench_gsm8k.pkl; NOT a quality subset):

| Quantity | min | median | max |
| --- | ---: | ---: | ---: |
| raw-logit logsumexp | 25.58487 | 33.25 | 46.5 |
| correct tail15 before float-rounding clip | -1.16e-7 | 4.3453e-7 | .00899260 |
| Claude screening tail15 | .99999999999233 | .9999999999999963 | 1 |

Small negative correct-tail values reflect stored float32 rounding and are
handled by the existing residual_tail_mass tolerance5e-7 and clipping.
The independent digit study already checked source hashes and sorted logprobs;
it did not independently validate Claude's six other screening streams.

Consequences:

- The screening tail15/tail50 4.10% rank1 result on707 open common misses is not
  a valid evaluation of true missing probability mass. Its correlations are
  also not the correlations of true tail mass.
- The reported12-stream participation ratio includes these malformed tail
  coordinates; it should not be interpreted as a validated augmented bank.
- Our deployed frozen tail15 gate calls residual_tail_mass and does NOT perform
  the extra subtraction. No evidence here invalidates that gate or its replay.
- A corrected full-population tail screening remains to be performed before
  ranking these signals against digit disagreement. No new result is asserted.

Information distinction: the renormalized q15 shape does not determine the
total probability mass retained in the head. Missing mass therefore restores
information discarded by that normalization. It is nevertheless determined by
the absolute top15 probabilities and is not statistically independent of them.
It measures how much probability lies outside the head, not how that tail is
distributed or the full-vocabulary entropy.

Claude screened seven views: provided-token surprisal, provided-token rank,
mass above the provided token, top1/provided-token logprob gap, tail15, tail50,
and digit disagreement. Only digit disagreement was taken through the reported
full fusion experiment. Operator/equality disagreement and a digit-based second
gate were proposals, not completed experiments in the files inspected.
