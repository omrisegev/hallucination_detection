"""
presets_ragtruth.py — validation-only generation presets for the RAGTruth Evidence-Contrast
campaign (`cluster/run_conditional_rescore.py`).

New file per the Step-186 worktree convention; `cluster/presets.py` picks these up through a
three-line `try/except` at its bottom, so merge-back never conflicts on the shared registry.

WHY THIS FILE EXISTS AT ALL
----------------------------------------------------------------------------------------
The conditional-rescore campaign scores Qwen2.5-1.5B-Instruct against FIXED RAGTruth
responses via teacher-forcing (`run_conditional_rescore.py --validate` is that driver's
Gate B). Gate B needs a GENERATED cell of the SAME model with `gen_token_ids` +
`token_entropies` already saved, to prove the driver's forward-pass reconstruction agrees
with what generation actually produced (see `run_teacher_forced.py`'s docstring for why this
is not optional — a wrong prompt/template/decoding-config combination is invisible in the
teacher-forced output and fatal downstream). No existing preset uses this checkpoint.

THE QWEN2.5 REPETITION-PENALTY TRAP
----------------------------------------------------------------------------------------
Qwen2.5-Instruct's `generation_config.json` ships `repetition_penalty=1.05` as a MODEL
DEFAULT — a real logits processor, applied even under greedy decoding, that
`run_inference.py`/`model_utils.generate_full` inherit whenever a preset's own
`repetition_penalty` is left `None` (HF's `generate()` then falls back to the model config).
The teacher-forced campaign passes `warpers=None` and no `rep_penalty` to
`candidate_quantities` — a deliberate choice, because RAGTruth's six original generator
models were NOT ours and there is no warp of OURS to mirror for the actual scoring passes.
That choice is only valid if the SAME penalty-free code path is what this Gate-B cell
generates with, so **`repetition_penalty=1.0` is set explicitly below to disable the
processor** (1.0 is a no-op multiplier), rather than leaving it `None` and inheriting
Qwen2.5's 1.05. If Gate B is ever re-run against a cell that used the model default instead,
it will FAIL for exactly this reason — that is the gate doing its job, not a bug.
"""

_COMMON = dict(
    paper="RAGTruth Evidence-Contrast U-PCR/DUFS-LIU campaign — Gate-B validation only",
    k=1,
    temps=[0.0],                    # greedy
    repetition_penalty=1.0,         # SEE MODULE DOCSTRING — disables Qwen2.5's 1.05 default
    gen_top_p=None,
    gen_top_k=None,
    logprob_top_k=50,
    capture={"logsumexp": True},
    acc_band=(0.0, 1.0),            # not a scientific cell; never gated on accuracy
    min_minority=0,
    head_to_head=None,
)


def register_all(presets: dict, preset_fn) -> dict:
    """Merge the RAGTruth Gate-B preset(s) into `cluster.presets.PRESETS` (in place).

    `preset_fn` is `presets._preset`, passed in rather than imported so there is no
    circular import while `presets.py` is still executing. Refuses to shadow an existing
    preset id (the Step-193 staleness failure mode).
    """
    new = {}

    new["gateb_gsm8k_qwen25_15b"] = preset_fn(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        dataset="gsm8k", split="test", n_samples=100,
        max_new=1024,
        notes=(
            "VALIDATION-ONLY cell for cluster/run_conditional_rescore.py's Gate B. Never a "
            "data cell for any reported result. Greedy, repetition_penalty=1.0 forced (see "
            "module docstring — Qwen2.5-Instruct's generation_config default is 1.05 and "
            "would silently invalidate the gate). N=100 is ample for the gate's per-token "
            "median/correlation statistics; this is not an accuracy measurement."
        ),
        **_COMMON,
    )

    for key, value in new.items():
        if key in presets:
            raise KeyError(f"presets_ragtruth.register_all: refusing to shadow existing "
                           f"preset id {key!r}")
    presets.update(new)
    return presets
