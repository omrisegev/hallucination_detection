"""
inscope_cells.py — the single definition of the 25 in-scope evaluation cells.

Scope was fixed by Omri on 2026-07-20 (Step 191): the thesis targets **reasoning (math)
and single-answer factual QA**. RAG (multi-hop retrieval, lciteeval) and GPQA (science
MCQ) are OUT — the Step-191 honest-ceiling check found GPQA uniformly at chance
(every feature 0.51-0.55) and RAG signal confined to one sub-dataset and bottlenecked by
feature sign rather than pool size.

Before Step 193 this roster was copy-pasted into three scripts with no shared constant
(inscope_report.py, selector_compare_inscope.py, inscope_orientation_audit.py). Import it
from here instead, so a scope change is a one-line edit rather than a three-way diff.
"""

QA_CELLS = [
    'epr_triviaqa_mistral24b', 'inside_coqa_llama7b', 'losnet_hotpotqa_mistral7b',
    'sciq_llama8b', 'se_nq_open_llama8b', 'se_squad_v2_llama8b',
    'seiclr_triviaqa_opt30b', 'semenergy_triviaqa_qwen3_8b',
    'spilled_triviaqa_llama8b', 'truthfulqa_llama8b',
]

MATH_CELLS = [
    'ars_gsm8k_r1distill8b', 'internalstates_gsm8k_qwen25_7b',
    'lapeigvals_gsm8k_llama3b', 'lapeigvals_gsm8k_llama8b',
    'lapeigvals_gsm8k_mistral24b', 'lapeigvals_gsm8k_nemo',
    'lapeigvals_gsm8k_phi35', 'noise_gsm8k_mistral7b', 'noise_gsm8k_phi3mini',
    'math500_dsmath7b', 'math500_qwenmath7b', 'math500_r1distill8b',
    'math500_r1distill8b_mn4096', 'trace_gsm8k_llama8b_k10',
    'trace_math500_qwenmath15b_k10',
]

INSCOPE = QA_CELLS + MATH_CELLS

GROUP = {c: 'QA' for c in QA_CELLS}
GROUP.update({c: 'math' for c in MATH_CELLS})

# The 6 cluster-era cells that arrived after the h16 artifacts were built. They have no
# published competitor anywhere, and were missing from GroupFS@16 / the LR oracle /
# pos_rate until the Step-193 backfill.
CLUSTER_CELLS = [
    'math500_dsmath7b', 'math500_qwenmath7b', 'math500_r1distill8b',
    'math500_r1distill8b_mn4096', 'trace_gsm8k_llama8b_k10',
    'trace_math500_qwenmath15b_k10',
]

assert len(QA_CELLS) == 10, len(QA_CELLS)
assert len(MATH_CELLS) == 15, len(MATH_CELLS)
assert len(INSCOPE) == 25 == len(set(INSCOPE))
assert set(CLUSTER_CELLS) <= set(MATH_CELLS)
