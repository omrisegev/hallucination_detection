# Step-level error labeling: judge instructions (packet v1)

You are an expert mathematics grader acting as an independent judge. Each item is a math
problem, a reference solution with its final answer, and a candidate solution split into
steps indexed from 0. For every item you decide which step of the candidate solution is the
earliest one that contains an error, or that no step contains an error. Your labels will be
compared with another judge who works independently, and used as ground truth for evaluating
error-localization methods. Accuracy matters more than speed.

All paths below are relative to the experiment directory given in your launch prompt.

## 1. What you may read and use

- Read only: this file, `packet/PACKET_MANIFEST.json`, `packet/FORMAT_EXAMPLE_ITEM.json`,
  `packet/FORMAT_EXAMPLE_LABELS.jsonl`, `packet/check_labels.py`, the shards in
  `packet/shards/`, and your own output directory `labels/<JUDGE_ID>/`.
- Do NOT open, list or search anything else in the repository or on the machine: no code,
  results, datasets, caches, git history, the `private/` directory, or any other judge's
  directory under `labels/`. Do not use the web. The items are anonymized on purpose; do not
  try to find out where they come from or which model wrote them.
- You may use Python or the shell only as a calculator or scratchpad (check arithmetic,
  simplify an expression, test a numeric claim) and to run `packet/check_labels.py`. Never
  write code that assigns labels (keyword rules, answer-matching heuristics, batch scoring).
  Every label must come from your own reading of that item.

## 2. Per-item procedure

1. Read the problem, then the reference solution and the reference final answer.
2. Read the candidate steps in order. For each step ask: given the problem and the steps
   before it, does this step state something mathematically false, make an invalid
   inference, or misuse the problem's conditions?
3. The first such step is `first_error_step`. Keep reading to the end to collect every step
   that introduces a new error (`error_steps`) and to check the final answer.
4. If no step contains an error, `first_error_step` is -1 and `error_steps` is empty.

Items are independent. The same problem can appear several times with different candidate
solutions; judge each one on its own. Do not assume any proportion of correct or incorrect
solutions.

## 3. What counts as an error

A step is erroneous if it contains at least one of the following (use the type of the first
error in `first_error_type`):

| type | meaning |
|---|---|
| `calculation` | an arithmetic or algebraic mistake (wrong value, sign, simplification, expansion) |
| `misread_problem` | misunderstanding or misusing the problem statement: wrong quantity, a dropped or invented condition, answering a different question |
| `concept` | a wrong formula, definition or theorem, or a method that does not apply |
| `logic` | an invalid inference: a conclusion that does not follow, circular reasoning, an unjustified restriction of cases, a necessary condition treated as sufficient |
| `unjustified_claim` | a false statement presented as established, which the solution relies on |
| `incomplete` | an omission that makes the step's conclusion invalid (a missed case or solution, an ignored domain restriction) |
| `final_answer` | the reasoning is right but the stated final answer is wrong or misreported (miscopied value, wrong requested form or unit) |
| `other` | anything else that makes the step wrong; explain it in the rationale |

Not errors:

- Style, verbosity, redundancy, headings, restating the problem, harmless notation.
- A valid method that differs from the reference solution.
- A step that correctly carries forward a value already made wrong by an earlier step. It is
  not a new error and does not go into `error_steps`.
- Intermediate rounding or approximation that changes no stated result and not the final answer.
- An exploratory attempt that the solution itself evaluates correctly (for example "Try x = 2:
  ... is not 0, so not a root", when that arithmetic is right).

Special cases:

- Self-corrected errors: if a step states something wrong and a later step notices and fixes
  it, the wrong step still counts. Set `self_corrected` to true.
- Final answer: decide equivalence mathematically (1/2 and 0.5, equivalent algebraic forms,
  formatting differences). If the candidate's final answer is not equivalent to the reference,
  at least one step is erroneous, so `first_error_step` cannot be -1. If you are confident the
  reference itself is wrong, set `reference_suspect` to true and judge the candidate on the
  mathematics.
- A correct final answer does not make the steps correct: check every step anyway.
- Step boundaries are fixed. Judge each step as given even when the split is awkward. If one
  error spans two steps, report the earlier one.

## 4. Output format

For each assigned shard `packet/shards/shard_XXX.jsonl`, write `labels/<JUDGE_ID>/shard_XXX.jsonl`
with exactly one JSON object per input line, in the same order, and nothing else in the file.
See `packet/FORMAT_EXAMPLE_ITEM.json` and `packet/FORMAT_EXAMPLE_LABELS.jsonl` for a made-up
example (not from the packet).

```json
{"item_id": "J0000", "judge_id": "<JUDGE_ID>", "n_steps": 4,
 "candidate_final_answer": "25/64", "final_answer_matches_reference": false,
 "first_error_step": 2, "error_steps": [2], "first_error_type": "misread_problem",
 "self_corrected": false, "reference_suspect": false, "confidence": "high",
 "rationale": "Step 2 uses 5/8 for the second draw, ignoring 'without replacement'; it should be 4/7, giving 5/14."}
```

| field | rule |
|---|---|
| `item_id` | copied from the input line |
| `judge_id` | your judge id from the launch prompt |
| `n_steps` | number of steps in the item (a consistency check) |
| `candidate_final_answer` | the final answer the candidate states, as a short string; `null` if it states none |
| `final_answer_matches_reference` | true or false, by mathematical equivalence |
| `first_error_step` | index of the earliest erroneous step, or -1 |
| `error_steps` | sorted list of every step that introduces a new error; empty exactly when `first_error_step` is -1; otherwise its smallest element equals `first_error_step` |
| `first_error_type` | one of the types in section 3, or `none` exactly when `first_error_step` is -1 |
| `self_corrected` | true if a later step fixes the first error; false otherwise (and when -1) |
| `reference_suspect` | true only if you are confident the reference final answer is wrong |
| `confidence` | `high`, `medium` or `low`: your confidence in `first_error_step` |
| `rationale` | at most 50 words. Name the concrete mistake: quote the wrong expression or value and give the correct one. For -1, say briefly why the solution holds. |

## 5. Working through your shards

- Process your assigned shards in order. Finish every item of a shard, then write the shard's
  output file in one write.
- Resume rule: if `labels/<JUDGE_ID>/shard_XXX.jsonl` already exists and
  `python packet/check_labels.py labels/<JUDGE_ID> --shards XXX-XXX` passes, skip that shard.
- When all your shards are written, run
  `python packet/check_labels.py labels/<JUDGE_ID> --shards A-B` for your range and fix
  anything it reports.
- Then write `labels/<JUDGE_ID>/NOTES_shards_A-B.md` (at most 15 lines): start and end time,
  and systematic difficulties only (ambiguous items, awkward step splits, suspected reference
  errors), each with its item_id. Do not report label statistics.
