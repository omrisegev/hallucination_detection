"""
inscope_cells.py — the single definition of the in-scope evaluation cells.

Scope was fixed by Omri on 2026-07-20 (Step 191): the thesis targets **reasoning (math)
and single-answer factual QA**. RAG (multi-hop retrieval, lciteeval) and GPQA (science
MCQ) are OUT — the Step-191 honest-ceiling check found GPQA uniformly at chance
(every feature 0.51-0.55) and RAG signal confined to one sub-dataset and bottlenecked by
feature sign rather than pool size.

Before Step 193 this roster was copy-pasted into three scripts with no shared constant
(inscope_report.py, selector_compare_inscope.py, inscope_orientation_audit.py). Import it
from here instead, so a scope change is a one-line edit rather than a three-way diff.

STEP 216 — THE ROSTER IS 24, NOT 25. `inside_coqa_llama7b` is REJECTED for a data
defect, not for a result. It is recorded in `REJECTED_CELLS` rather than deleted, so the
reason travels with the roster and the cell stays loadable for the write-up. See the
dict below and `spectral_utils.answer_span` for the measurement behind it.
"""

# ── rejected cells ────────────────────────────────────────────────────────────
# A cell lands here only for a defect in the GENERATED DATA that no offline
# processing can repair — never because it scores badly. The distinction matters:
# a hard cell is a result and stays in every macro; a broken cell is measuring
# something other than what the method claims to measure.
#
# Mirrored in `spectral_utils.answer_span.UNREPAIRABLE_CELLS`, which is what the
# feature-extraction side reads. Keep the two in sync — `assert` at the bottom.
REJECTED_CELLS = {
    'inside_coqa_llama7b': (
        "Mistral chat template applied to a LLaMA-7B *base* checkpoint (the preset "
        "never set raw_prompt=True). Measured by scripts/answer_span_audit.py over "
        "the raw pkl: 73.4% of generations pinned at max_new=256, median answer 8 "
        "tokens = 3.5% of the trace, and 45.1% of first lines are [/INST] echoes, "
        "fabricated 'Question:' turns, raw passage text, or empty. Cropping cannot "
        "fix it — the 45% broken rows are a degenerate sub-population whose label is "
        "near-deterministic given the prompt broke (pos_rate 0.002 on broken rows vs "
        "0.239 on usable ones), so a crop would trade one artifact for another. "
        "Needs re-generation with raw_prompt=True. Step 216."
    ),
}

# ── the roster, before rejection ──────────────────────────────────────────────
# Kept so a report can state what was removed and quote its pre-rejection numbers.
QA_CELLS_ALL = [
    'epr_triviaqa_mistral24b', 'inside_coqa_llama7b', 'losnet_hotpotqa_mistral7b',
    'sciq_llama8b', 'se_nq_open_llama8b', 'se_squad_v2_llama8b',
    'seiclr_triviaqa_opt30b', 'semenergy_triviaqa_qwen3_8b',
    'spilled_triviaqa_llama8b', 'truthfulqa_llama8b',
]

MATH_CELLS_ALL = [
    'ars_gsm8k_r1distill8b', 'internalstates_gsm8k_qwen25_7b',
    'lapeigvals_gsm8k_llama3b', 'lapeigvals_gsm8k_llama8b',
    'lapeigvals_gsm8k_mistral24b', 'lapeigvals_gsm8k_nemo',
    'lapeigvals_gsm8k_phi35', 'noise_gsm8k_mistral7b', 'noise_gsm8k_phi3mini',
    'math500_dsmath7b', 'math500_qwenmath7b', 'math500_r1distill8b',
    'math500_r1distill8b_mn4096', 'trace_gsm8k_llama8b_k10',
    'trace_math500_qwenmath15b_k10',
]

INSCOPE_ALL = QA_CELLS_ALL + MATH_CELLS_ALL

# ── the roster everything scores on ───────────────────────────────────────────
QA_CELLS = [c for c in QA_CELLS_ALL if c not in REJECTED_CELLS]
MATH_CELLS = [c for c in MATH_CELLS_ALL if c not in REJECTED_CELLS]
INSCOPE = QA_CELLS + MATH_CELLS

GROUP = {c: 'QA' for c in QA_CELLS}
GROUP.update({c: 'math' for c in MATH_CELLS})

# Cells whose FEATURES are computed over the answer span rather than the full run-on
# generation (Step 216). Not a scope change — the cell stays in every macro; what
# changed is that its features and its label are now read off the same span.
CROPPED_CELLS = ['seiclr_triviaqa_opt30b']

# The 6 cluster-era cells that arrived after the h16 artifacts were built. They have no
# published competitor anywhere, and were missing from GroupFS@16 / the LR oracle /
# pos_rate until the Step-193 backfill.
CLUSTER_CELLS = [
    'math500_dsmath7b', 'math500_qwenmath7b', 'math500_r1distill8b',
    'math500_r1distill8b_mn4096', 'trace_gsm8k_llama8b_k10',
    'trace_math500_qwenmath15b_k10',
]

assert len(QA_CELLS) == 9, len(QA_CELLS)       # was 10 before Step 216
assert len(MATH_CELLS) == 15, len(MATH_CELLS)
assert len(INSCOPE) == 24 == len(set(INSCOPE))  # was 25 before Step 216
assert set(CLUSTER_CELLS) <= set(MATH_CELLS)
assert set(CROPPED_CELLS) <= set(INSCOPE)
assert not (set(REJECTED_CELLS) & set(INSCOPE))
assert set(REJECTED_CELLS) <= set(INSCOPE_ALL)

# The two rejection registries must agree, or the feature side and the scoring side
# disagree about what the roster is — exactly the three-way-diff failure this module
# was created to prevent.
try:
    from spectral_utils.answer_span import UNREPAIRABLE_CELLS as _UNREPAIRABLE
except ImportError:                                   # pragma: no cover
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from spectral_utils.answer_span import UNREPAIRABLE_CELLS as _UNREPAIRABLE
assert set(_UNREPAIRABLE) == set(REJECTED_CELLS), (
    f"rejection registries disagree: answer_span {sorted(_UNREPAIRABLE)} vs "
    f"inscope_cells {sorted(REJECTED_CELLS)}")
