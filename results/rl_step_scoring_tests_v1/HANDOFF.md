# RL step-scoring tests (2026-11-14)

Thread "Explore step scoring in RL post-training". Omri asked to run the two offline tests that the literature
scan proposed. No GPU. No new inference.

## Test 2: first-error localization on wrong answers (Step-DPO use case)

Step-DPO (arXiv 2406.18629, Sec. 3.2) keeps only answers with a wrong final answer. Then it finds the first
wrong step "manually or using GPT-4". We test if our label-free step score can do this step.

| item | value |
|---|---|
| step score | stored TELE16 standalone (`arm__TELE_N16` = L-SML on the 16 stored answer-z channels, R0) from one teacher-forced pass of the base model. No PRM. No labels. |
| base models (cells) | qwen_prm_7b = Qwen2.5-Math-7B-Instruct; qwen_prm_72b = Qwen2.5-Math-72B-Instruct; pure_prm = Qwen2.5-Math-7B (base); rlhflow_prm = Llama-3.1-8B-Instruct |
| rule | GAP (k = largest gap in the top ceil(m/2) of answer-z scores, flag top-k), prediction = earliest flagged step. GAP always flags >= 1 step. Second readout: argmax step. |
| population | answers with >= 1 error: ProcessBench 2,221 of 3,400; PRMBench 6,035 of 6,969 |
| metric | exact first-error accuracy (%). PB: macro over 4 subsets (= ProcessBench "error" accuracy = SLA). PRMBench: over all erroneous answers. |
| references | PRM of the same cell (first step with error prob > .5, else argmax); position only + GAP; always step 0; PB only: LLM critics from ProcessBench Tables 5 and 7 (paper numbers, same 3,400 answers, not rerun) |
| CI | 95%, 2,000 bootstrap draws over answers (PB stratified by subset). PRMBench draws answers, not source groups, so its CI is too narrow. |

Results: TABLES.md. Data: PB cells from results/pb_first_error_rule_v1/cells/; PRMBench cells/PRMB_<prm>.npz
(exported on the PC by export_prmbench_sa.py, loader E.load('prmbench', '<prm>__standalone')).

Reproduce (numpy only): `python -I test2_pb.py` and `python -I test2_prmbench.py`.

## Test 1: step score vs Monte Carlo rollout values

NOT RUN. No benchmark in this project has MC rollout values. Options (A: new rollouts on 300 PB wrong answers;
B: Math-Shepherd MC labels + one telemetry pass) wait for Omri's decision on GPU cost.
