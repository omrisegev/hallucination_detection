"""
presets_localization.py — replication presets for "Mind the Gap: Catching Hallucinations via
Evidence Drop on the Reasoning Manifold" (Chen, Chen, Yue, Li; ICML 2026, PMLR 306).

New file per the Step-186 worktree convention; `cluster/presets.py` picks these up through a
three-line `try/except` at its bottom, so merge-back never conflicts on the shared registry.

Digest with the full protocol + the five verified traps in the paper:
    papers/digests/mind-the-gap-catching-hallucinations-via-evidence-drop.md

WHAT THE PAPER FIXES, AND WHAT IT LEAVES OPEN
---------------------------------------------
Fixed and matched here: greedy **τ = 0**, nucleus **top-p 0.95**, decoding **top-K 20**,
backbones **Qwen3-4B / Qwen3-8B**, datasets **GSM8K** and **MATH**.

Left completely unspecified (grep the extract for `template`, `max_new`, `repetition`, `seed` —
zero relevant hits): the prompt template, the **thinking mode**, the max generation length, and
the seed. Those are ours to choose, and one of them decides whether the reproduction is
comparable at all.

THE OPERATING POINT IS THE WHOLE BALLGAME
-----------------------------------------
Selective accuracy (their Eq. 13) and AURC are both monotone in the base error rate. "Shannon
Drop 88.26%" measured at 59-66% base accuracy and the same statistic measured at 90% accuracy
are simply different quantities. So matching their accuracy is a **precondition** for the
comparison, not a nice-to-have.

Their reported accuracies — GSM8K Qwen3-8B **91.07 ± 0.26**, Qwen3-4B **87.63 ± 0.31**, MATH
Qwen3-8B **~66** (App. E.2 Table 6; Table 1's 59.24 is the 4B value copied into the 8B row) —
are all consistent with Qwen3 **non-thinking** and inconsistent with thinking-on. Our own
thinking-on runs of nearly this setup land at **94.2% GSM8K / 90.0% MATH-500**
(`cache/repgrid/ars_{gsm8k,math500}_qwen3_8b_reject`), i.e. 3pp and ~25pp high.

Hence: `/no_think` is the PRIMARY arm, and `evdrop_gsm8k_qwen3_8b_think` is the CONTROL that
proves the mode rather than assuming it. Whichever arm lands nearer the paper is the
reproduction; the other is reported as a sensitivity. This is pre-registered — see the plan's
verification gate 3.

Three defects the non-thinking choice fixes at once:
  1. the operating point (above);
  2. **the calibration set**. Their rule uses only the *incorrect* half of D_cal. At N=500 and
     94.2% accuracy that is 29 negatives total -> ~14 in D_cal, and the alpha=0.05 quantile of 14
     points is just the minimum order statistic. Non-thinking (~9% GSM8K / ~35% MATH error) at
     full N gives ~120 / ~500 negatives;
  3. **the truncation confound**. In the thinking-on caches, 15 of 29 GSM8K negatives and 23 of
     50 MATH negatives are pinned at the cap — half the negative class is truncation, not
     hallucination. Short non-thinking answers at 1024/2048 should make this ~zero; it is
     measured and reported either way (`frac_pinned` within the negative class).

NOTE ON `gen_top_p` / `gen_top_k`: both are **inert at tau = 0**. `model_utils.generate_full`
sets `sampling = temp > 1e-4` and passes `top_k=None, top_p=None` under greedy, so the saved
`top_k_logprobs` are raw log-softmax and the paper's Eq. 10 (renormalized top-20 entropy) is
exactly reconstructible offline. They are set anyway for provenance, exactly as the paper states
them — and the paper's own decoding top-K = 20 is inert under greedy for the same reason.
"""

# The paper's own accuracy figures, used as the pre-registered operating-point target.
# GSM8K comes from Table 1. MATH comes from **Appendix E.2 Table 6**, NOT Table 1: Table 1
# reports 59.24 +/- 0.21 for BOTH 4B and 8B (identical across two different models, while GSM8K
# correctly differs), and Table 6 gives the real per-model values. See digest caveat 5.
PAPER_ACCURACY = {
    ("gsm8k", "Qwen/Qwen3-4B"): 87.63,
    ("gsm8k", "Qwen/Qwen3-8B"): 91.07,
    ("math",  "Qwen/Qwen3-4B"): 58.12,   # mean of Table 6's 57.92 / 58.56 / 57.88
    ("math",  "Qwen/Qwen3-8B"): 66.16,   # mean of Table 6's 66.12 / 66.40 / 65.96
}

# Evidence Drop's own method defaults (Sec. 3.3 + Appendix E.1), recorded here so the offline
# scorer and the preset cannot drift apart. NOTE: Appendix E.1's Table 5 panels (a) and (b) are
# byte-identical, so only one of the M / EMA-span ablations was actually run — do not describe
# these as ablation-confirmed. M = 10 also beats M = 5 (90.75 vs 88.26) for Shannon Drop.
EVDROP_DEFAULTS = {"M": 5, "ema_span": 5, "top_k": 20}

# The accuracy-band gate exists to protect AUROC, which needs both classes populated. These
# cells are NOT scored by AUROC — selective accuracy and AURC are *designed* for the
# high-accuracy regime, and a ~91%-accurate model is the paper's intended operating point.
# The default band (0.20, 0.85) with min_minority 30 would REJECT every cell here, which is
# exactly what happened to `ars_*_qwen3_8b_reject` and is why those cells were misread as
# failures. Widened deliberately, with the reason attached.
EVDROP_ACC_BAND = (0.05, 0.98)
EVDROP_MIN_MINORITY = 10

_COMMON = dict(
    paper="Mind the Gap / Evidence Drop (ICML 2026, PMLR 306)",
    k=1,
    temps=[0.0],            # greedy, tau = 0
    gen_top_p=0.95,         # paper's nucleus threshold (inert at tau = 0)
    gen_top_k=20,           # paper's decoding top-K  (inert at tau = 0)
    logprob_top_k=50,       # top-20 is a strict subset -> Eq. 10 reconstructible offline
    capture={"logsumexp": True},   # also yields `top_k_logprobs_raw` (pre-warper) + Z_n
    acc_band=EVDROP_ACC_BAND,
    min_minority=EVDROP_MIN_MINORITY,
    head_to_head="SAME-MODEL",
)

_NO_THINK = " /no_think"


def _published(dataset, model):
    """Reproduction targets carried on the cell, so the manifest records what it is aiming at."""
    return {
        "method": "Evidence Drop (Shannon Drop)",
        "metric": "selective accuracy @ alpha, and AURC x1000",
        "pretrained_accuracy": PAPER_ACCURACY[(dataset, model)],
        "accuracy_source": ("Table 1" if dataset == "gsm8k"
                            else "Appendix E.2 Table 6 (NOT Table 1 — see digest caveat 5)"),
        "evdrop_defaults": EVDROP_DEFAULTS,
        "model_note": (
            "Selective accuracy and AURC are monotone in base error rate, so the "
            "`pretrained_accuracy` above is a PRECONDITION for comparability, not a "
            "secondary statistic. Verification gate 3."
        ),
    }


def register_all(presets: dict, preset_fn) -> dict:
    """Merge the Evidence Drop presets into `cluster.presets.PRESETS` (in place).

    `preset_fn` is `presets._preset`, passed in rather than imported so there is no circular
    import while `presets.py` is still executing.

    Refuses to shadow an existing preset id: silently redefining one would change what an
    already-scored cell means without changing its name (the Step-193 staleness failure mode).
    """
    new = {}

    # ── Primary arm: non-thinking, the paper's operating point ────────────────
    for model, tag in (("Qwen/Qwen3-4B", "qwen3_4b"), ("Qwen/Qwen3-8B", "qwen3_8b")):
        new[f"evdrop_gsm8k_{tag}"] = preset_fn(
            model=model,
            dataset="gsm8k", split="test", n_samples=1319,   # the full GSM8K test split
            max_new=1024,
            prompt_suffix=_NO_THINK,
            published=_published("gsm8k", model),
            notes=(
                "Evidence Drop answer-level cell, NON-THINKING (primary arm). Target accuracy "
                f"{PAPER_ACCURACY[('gsm8k', model)]} (paper Table 1). max_new=1024 is ample for a "
                "non-thinking GSM8K answer; if `frac_pinned` within the negative class exceeds "
                "5%, raise it and re-run rather than reporting a truncation-contaminated "
                "negative class. Full test split so the calibration half has enough incorrect "
                "samples for the alpha-quantile to be an actual quantile."
            ),
            **_COMMON,
        )
        new[f"evdrop_math_{tag}"] = preset_fn(
            model=model,
            dataset="math", split="test", n_samples=1500,
            max_new=2048,
            prompt_suffix=_NO_THINK,
            published=_published("math", model),
            notes=(
                "Evidence Drop answer-level cell, NON-THINKING (primary arm), on the FULL MATH "
                "test split (Hendrycks et al., 2021) — not MATH-500. The paper says 'MATH' and "
                "the strings MATH-500/MATH500/500 appear nowhere in it; the two have different "
                "difficulty mixes, and selective accuracy / AURC are monotone in base error. "
                f"Target accuracy {PAPER_ACCURACY[('math', model)]} (App. E.2 Table 6). Rows are "
                "seeded-shuffled across all 7 subjects before truncation "
                "(spectral_utils.localization_data.MATH_SAMPLE_SEED), so N=1500 is "
                "subject-representative and reproducible."
            ),
            **_COMMON,
        )

    # ── Control arm: thinking-on, GSM8K only ──────────────────────────────────
    # Cheap, and it is what turns "the paper probably used non-thinking" from an assumption
    # into a measurement. max_new mirrors `ars_gsm8k_qwen3_8b` (8192), whose notes record 13%
    # of traces pinned at 4096 — do not lower it.
    _think = dict(_COMMON)
    new["evdrop_gsm8k_qwen3_8b_think"] = preset_fn(
        model="Qwen/Qwen3-8B",
        dataset="gsm8k", split="test", n_samples=500,
        max_new=8192,
        published=_published("gsm8k", "Qwen/Qwen3-8B"),
        notes=(
            "CONTROL arm for the thinking-mode ambiguity — Qwen3 thinking ON (no /no_think "
            "suffix). The paper never states a thinking mode; this arm plus the primary "
            "non-thinking arm decide it by accuracy instead of by assumption. Expected to land "
            "near our existing `ars_gsm8k_qwen3_8b_reject` cache (94.2%), i.e. ~3pp ABOVE the "
            "paper's 91.07 — which is the evidence that non-thinking is the right arm. If this "
            "arm is instead the closer one, switch the primary and record it."
        ),
        **_think,
    )

    clash = sorted(set(new) & set(presets))
    if clash:
        raise KeyError(
            f"presets_localization would shadow existing preset id(s): {clash}. "
            "Rename instead of overriding — an override changes what an already-scored cell "
            "means without changing its id."
        )
    presets.update(new)
    return presets
