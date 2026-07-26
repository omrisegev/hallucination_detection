"""
glossary.py — decodes every nickname this project uses for a subset, selector
family, variant, pool mode, or role tag, into a one-line meaning + HISTORY
pointer. Single source of truth; `scripts/build_glossary.py` renders it to
the root `GLOSSARY.md`.

Design (Omri, 2026-07-22/23): "we should include in the files generated a
mapping that maps between the nickname you gave the method and what it
actually means." Kept terse — one line + a HISTORY.md step pointer for the
full story — so it never duplicates (and drifts from) HISTORY.md's narrative.

Coverage is checked, not just declared: `build_glossary.py` diffs this
module's entries against every variant string that actually appears in
results/selector_bench/*.csv and fails loudly on any gap, so a new selector
variant can't go undocumented silently.
"""

import re

from .subset_sweep import CANONICAL_POOL, H16

# ---------------------------------------------------------------------------
# Fixed reference subsets (hand-curated, NOT feature-selection output).
# Member features pulled live from reference_macros.MACROS so the feature
# list can never drift from what is actually benched.
# ---------------------------------------------------------------------------

REFERENCE_SUBSET_NOTES = {
    "ref.GOOD_5": ("The original 5-feature hand-picked subset; the "
                  "long-standing compatibility baseline.", "Step 100s"),
    "ref.GOOD_6": ("GOOD_5 + varentropy; +1.1pp macro over GOOD_5 on the "
                  "19-cell grid, was the headline fixed subset until "
                  "LOCO_5.", "Step 182/184"),
    "ref.LOCO_5": ("A NEW 5-view subset found by exhaustive enumeration over "
                  "the 30-view pool, validated leave-one-cell-out (not "
                  "chosen in-sample like GOOD_5/6) — the current best fixed "
                  "subset, beats GOOD_6 by +0.73pp.", "Step 195"),
    "ref.STABLE_H9": ("A 9-feature subset from the original 16 H(n) views, "
                      "stable across models/temperatures/domains in early "
                      "phases.", "Phase 4-5 era"),
    "ref.top_macro_5": ("Top 5 features by individual informativeness across "
                        "the repgrid grid.", "Step 155 era"),
    "ref.consensus_4": ("4 features that were consensus picks across "
                        "multiple early subset searches.", "Step 155 era"),
    "ref.ALL_H16": ("All 16 original H(n) spectral features, no selection — "
                    "the un-pruned floor/ceiling reference.", "Phase 4"),
}

# ---------------------------------------------------------------------------
# Selector families (spectral_utils/selectors/*.py) — full record per family:
# paper it's from (or "no single paper" for house heuristics), what it
# actually relies on mechanically, and where/how well it did for us — even
# though the performance numbers duplicate HISTORY.md, Omri asked for this
# file to be able to answer "what does each algorithm mean" on its own,
# without needing to cross-reference HISTORY.md's narrative (2026-07-23).
#
# build_glossary.py asserts every spectral_utils.selectors.registered() name
# has an entry here, so a new family can't ship undocumented.
# ---------------------------------------------------------------------------

FAMILY_NOTES = {
    "a7_iter_consensus": {
        "paper": "None — Extension H in-house design (Step 199 pivot), in the "
                 "spirit of Jaffe/Nadler consensus estimation but with the "
                 "target rebuilt each iteration.",
        "relies_on": "Its own consensus target rebuilt per iteration; no `epr` "
                     "anchor and no `GOOD_6` seeds, which is the whole point "
                     "of the prior-free arm.",
        "performance": "It is not. Fixed and re-measured in Step 202: with the "
                       "`epr` anchor 0.7378 (-2.16pp vs GOOD_6, W/L 8/17, "
                       "p=0.0105); genuinely prior-free 0.6524 (-10.70pp, W/L "
                       "6/19, p=0.0010). Step 200's reported 0.6840 "
                       "'prior-free' actually used the anchor, and the truly "
                       "prior-free arm was 0.5103 (chance) before the fixes. "
                       "Part of Extension H, closed as bounded.",
        "history": "Steps 200-202",
    },
    "reference_macros": {
        "paper": "None — not a selector.",
        "relies_on": "Emits the fixed subsets (GOOD_5/6, LOCO_5, ...) "
                     "through the exact same select-then-fuse-then-score "
                     "path every learned selector uses, so they land in the "
                     "same leaderboard with identical metrics.",
        "performance": "This is the comparison target, not a candidate — "
                       "see the Reference subsets section above for numbers.",
        "history": "Step 186",
    },
    "a1_residual": {
        "paper": "Jaffe, Nadler, Kluger — \"Estimating the Accuracies of "
                 "Multiple Classifiers Without Labels\" (arXiv:1407.7644, "
                 "2014). Reuses the paper's own Eq-14 structural residual — "
                 "how well a K-group rank-one covariance model fits a "
                 "subset's correlation structure — but as a SELECTION "
                 "criterion, which is not what the paper proposes it for "
                 "(the paper uses it internally at fusion time; we search "
                 "over subsets to minimize it).",
        "relies_on": "For each candidate subset: build the covariance, "
                     "spectral-cluster it into K groups, measure the "
                     "residual of that fit. Also includes a router between "
                     "L-SML and U-PCR fusion based on which has the lower "
                     "residual, and swapping the K-selection rule for the "
                     "Ahn-Horenstein / Kritchman-Nadler rank tests.",
        "performance": "Every variant scored BELOW GOOD_5 on the 25 "
                       "in-scope cells (best: a1.router@good5 0.7494 vs "
                       "GOOD_5 0.7519). Verdict: the residual objective is "
                       "not admissible as a selection criterion — it is "
                       "nearly orthogonal to separability (weakly "
                       "ANTI-correlated with AUROC, Spearman ~ -0.11 to "
                       "-0.17).",
        "history": "Step 186",
    },
    "a2_groupfs": {
        "paper": "GroupFS: Lifshitz, Lindenbaum, Mishne, Meir, Benisty — "
                 "\"Unsupervised Feature Selection Through Group "
                 "Discovery\" (AAAI 2026, arXiv:2511.09166); no official "
                 "code, clean-room torch reimplementation. Its selection "
                 "signal is replaced by DUFS: Lindenbaum, Shaham, Svirsky, "
                 "Peterfreund, Kluger — \"Differentiable Unsupervised "
                 "Feature Selection based on a Gated Laplacian\" (NeurIPS "
                 "2021, arXiv:2007.04728) — see a2.dufs below.",
        "relies_on": "GroupFS: Gumbel-softmax soft group assignment + "
                     "per-group stochastic gates, trained against sample-"
                     "graph smoothness + feature-graph smoothness + group "
                     "sparsity (3-term loss). DUFS: per-feature stochastic "
                     "gates trained against Laplacian-score sample "
                     "smoothness alone. Both use the self-tuning kernel "
                     "graph + t=2 random-walk diffusion.",
        "performance": "a2.dufs 0.7502, a2.select (GroupFS's own readout) "
                       "0.7481 — both just under GOOD_5 (0.7519). The "
                       "2026-07-23 fidelity audit found BOTH are faithful "
                       "to their papers' equations, but GroupFS's own "
                       "group-gate readout saturates open under our CPU "
                       "budget, so selection was replaced with the DUFS "
                       "per-feature signal (documented deviation) — meaning "
                       "a2.select vs a2.dufs is really \"one gate mechanism, "
                       "two readouts,\" not two different papers head to "
                       "head. See results/advisor_inscope/"
                       "fs_paper_fidelity.md for the full audit.",
        "history": "Step 186, fidelity-audited Step 196",
    },
    "a3_concrete_ae": {
        "paper": "Balin, Abid, Zou — \"Concrete Autoencoders: "
                 "Differentiable Feature Selection and Reconstruction\" "
                 "(ICML 2019, arXiv:1901.09346); reimplemented torch-CPU "
                 "with a linear decoder (paper's canonical repo is Keras).",
        "relies_on": "A concrete (Gumbel-softmax-relaxed) selector layer "
                     "picks k of p features to minimize LINEAR "
                     "RECONSTRUCTION error of the full feature set — "
                     "entirely label-free, no correctness signal at all.",
        "performance": "Best: a3.cae_k3 0.7388, below GOOD_5. Known risk "
                       "(Step 151): reconstruction-good does not imply "
                       "label-relevant — on z-scored unit-variance columns, "
                       "a pure-noise feature is reconstructable only by "
                       "selecting itself, so all its reconstruction value "
                       "comes from predicting OTHER features, whether or "
                       "not that tracks detection AUROC.",
        "history": "Step 186",
    },
    "a4_antigravity": {
        "paper": "No single paper — classical building blocks: greedy "
                 "Column Subset Selection (CSSP, linear-reconstruction "
                 "greedy forward selection) and anchor-correlation ranking "
                 "(a house heuristic, not from a specific paper).",
        "relies_on": "a4.recon: greedily add the feature that most reduces "
                     "linear reconstruction error of the whole pool. "
                     "a4.anchor: rank by |Spearman| correlation with the "
                     "cell's anchor view (epr).",
        "performance": "a4.anchor-family was Step 187's best learned "
                       "selector on H16 (0.6593) but Step 189 found it "
                       "statistically indistinguishable from bare epr "
                       "(25W/26L) — it just picks epr's most-correlated "
                       "clones, high relevance / zero diversity. This "
                       "finding motivated a5_mrmr (adds a redundancy term).",
        "history": "Step 186, salvage diagnosis Step 187/189",
    },
    "a5_mrmr": {
        "paper": "Peng, Long, Ding — \"Feature Selection Based on Mutual "
                 "Information: Criteria of Max-Dependency, Max-Relevance, "
                 "and Min-Redundancy\" (mRMR; IEEE TPAMI 27(8), 2005). Uses "
                 "|Spearman| in place of mutual information (label-free "
                 "relevance proxy against the anchor, not true MI).",
        "relies_on": "Greedy: score(j|S) = relevance(j) - alpha * "
                     "redundancy(j|S), relevance = |Spearman| with the "
                     "anchor, redundancy = mean |Spearman| with already-"
                     "picked features (reuses the cell's cached rho matrix, "
                     "no recompute). alpha=0 reproduces a4.anchor exactly.",
        "performance": "Best: a5.mrmr_a0.7_adapt 0.7379, still below "
                       "GOOD_5 — adding the redundancy term diversifies the "
                       "picks (verified: alpha=0 clones near-identical "
                       "features, alpha=0.7 spreads across the pool) but "
                       "does not translate into a competitive AUROC.",
        "history": "Step 189",
    },
    "a6_pseudolabel_gates": {
        "paper": "No single paper — Omri's idea (2026-07-22), built on top "
                 "of a2_groupfs's DUFS gate machinery (Lindenbaum et al. "
                 "2021, see a2_groupfs) plus the project's own prior "
                 "pseudo-label mechanism (`fusion_utils."
                 "best_nadler_pseudo_label`, Step ~100, which built a "
                 "pseudo-label by MAJORITY-VOTE BINARIZATION for an "
                 "exhaustive search — a6 instead uses CONTINUOUS L-SML "
                 "fusion and spends the pseudo-label on the gate objective).",
        "relies_on": "Fuse 4 seed views (epr, low_band_power, "
                     "spectral_entropy, cusum_max — a label-free "
                     "priority-list rule, held out of the selectable pool "
                     "to avoid circularity) via continuous L-SML into a "
                     "pseudo-label; supervise DUFS's per-feature gates with "
                     "a CENTERED agreement-with-pseudo-label reward on top "
                     "of the unsupervised Laplacian-smoothness term.",
        "performance": "a6.pl_dufs 0.7524 — the ONLY learned selector to "
                       "nominally edge GOOD_5 (+0.05pp, not significant), "
                       "and ADOPTED AS THE SELECTOR OF RECORD despite both "
                       "pre-registered gates (mechanism rho>=0.30, "
                       "performance >=+1.0pp) FAILING. Seed-choice "
                       "experiments (2026-07-23, WS4): swapping seeds for "
                       "LOCO_5's views or diverse-centrality picks moves "
                       "it by only -0.02 to -0.07pp — the seed set is not "
                       "the lever. A full-pool (no-held-out-seeds) variant "
                       "(a6.fp_dufs) scores about the same (0.7508).",
        "history": "Step 194, seed/full-pool experiments Step 196",
    },
    "classical_fs": {
        "paper": "Three methods, three papers, one family: Laplacian "
                 "Score (He, Cai, Niyogi, NIPS 2005); SPEC/phi2 (Zhao & "
                 "Liu, ICML 2007); MCFS (Cai, Zhang, He, KDD 2010).",
        "relies_on": "All three build one sample-similarity graph per cell "
                     "then rank features against it: Laplacian Score and "
                     "SPEC by a normalized-Laplacian Rayleigh quotient "
                     "(lower=better, feature varies ALONG the manifold not "
                     "across it); MCFS by L1-regressing top-K spectral "
                     "embeddings onto the features (higher=better, feature "
                     "reconstructs cluster structure).",
        "performance": "The textbook unsupervised-FS baselines this "
                       "project's learned selectors are meant to beat; all "
                       "score well below GOOD_5.",
        "history": "Step 186",
    },
    "simple_stats": {
        "paper": "No paper — deliberately naive floor, motivated by the "
                 "Rajabinasab et al. 2026 guardrail finding that many "
                 "unsupervised FS methods lose to random selection.",
        "relies_on": "random (uniform draw, THE floor), mad (median-"
                     "absolute-deviation rank), kurtosis (non-Gaussianity "
                     "rank), decorrelation (greedy min mutual |Spearman|, "
                     "no relevance term).",
        "performance": "The sanity floor every real selector must clear. "
                       "Variance-based selection is deliberately excluded: "
                       "UnlabeledCell.V is z-scored per column (unit "
                       "variance), so raw variance carries no signal here.",
        "history": "Step 186",
    },
}

# ---------------------------------------------------------------------------
# Individual variant strings — the specific "aN.foo" names that appear in
# results/selector_bench/*.csv rows. One line each; the family entry above
# gives the mechanism, this gives what's specific about THIS variant.
# ---------------------------------------------------------------------------

VARIANT_NOTES = {
    "a6.pruned_dufs": (
        "a6 gates plus a hyperparameter-pruning pass (target-size cap "
        "K_max=15, `logprob_margin` anchor). Audited numbers: 0.7537 macro / "
        "0.7141 QA / 0.7801 math, mean size 17.0 -- below `a6.pl_dufs`.",
        "Step 197"),
    "a1.router@good5": ("Routes GOOD_5's fusion to whichever of L-SML/U-PCR "
                        "has the better structural residual on this cell.",
                        "Step 186"),
    "a1.router@loco5": ("Same router, but on LOCO_5 instead of GOOD_5 — "
                        "beats router@good5 but still loses to LOCO_5 "
                        "un-routed.", "Step 195/196 (WS6)"),
    "a1.router@minres": ("Router on the greedy min-residual subset.",
                        "Step 186"),
    "a1.good5+K_ah": ("GOOD_5 with K (L-SML group count) chosen by the "
                      "Ahn-Horenstein rank test instead of residual "
                      "minimization.", "Step 186"),
    "a1.good5+K_kn": ("GOOD_5 with K chosen by the Kritchman-Nadler rank "
                      "test.", "Step 186"),
    "a1.minres+K_ah": ("Greedy min-residual subset with the Ahn-Horenstein "
                       "K rule.", "Step 186"),
    "a1.relres_greedy": ("Greedy search minimizing the RELATIVE residual "
                         "(raw residual / off-diagonal correlation energy) "
                         "— the structure-seeking form.", "Step 186"),
    "a1.upcrres_greedy": ("Greedy search minimizing the U-PCR k=1 projection "
                         "residual.", "Step 186"),

    "a2.select": ("GroupFS's own selection readout (feature-granular, via "
                  "the group discovery mechanism).", "Step 186"),
    "a2.select+groups": ("a2.select's chosen features, fused WITH the "
                        "discovered group assignment (clustering swap) "
                        "instead of re-discovering groups at fusion time.",
                        "Step 186"),
    "a2.groups@good5": ("GOOD_5's fixed features, fused with GroupFS's "
                        "discovered group assignment — isolates the "
                        "clustering's own value.", "Step 186"),
    "a2.dufs": ("DUFS (Gated-Laplacian) per-feature gates — the 2021 "
               "predecessor GroupFS's own selection signal was replaced "
               "with (see a2_groupfs family note).", "Step 186, audited "
               "Step 195"),
    "a2.dufs_pf": ("DUFS's own Eq. (7) PARAMETER-FREE loss — no lambda to "
                  "tune, more faithful to the paper than a2.dufs's label-"
                  "free lambda-stability search.", "Step 196 (WS8)"),

    "a6.pl_dufs": ("THE headline a6 output: seed views + DUFS gates "
                  "supervised by the pseudo-label. Adopted as selector of "
                  "record despite both pre-registered gates FAILING their "
                  "bar.", "Step 194"),
    "a6.pl_dufs_noseed": ("a6.pl_dufs's gated picks WITHOUT the always-on "
                          "seed views — isolates the gates' own "
                          "contribution.", "Step 194"),
    "a6.pl_rank": ("Size-matched ablation: top-|corr|-with-pseudo-label "
                  "ranking, no gate training. If this ties pl_dufs, the "
                  "gate machinery adds nothing.", "Step 194"),
    "a6.dufs": ("Unsupervised control: a6's DUFS gates with NO pseudo-label "
               "term (lambda3=0) — should reproduce a2.dufs closely.",
               "Step 194"),
    "a6.fp_dufs": ("Omri's two-stage idea (4b): pseudo-label from ALL 30 "
                  "views (not 4 seeds) supervises gates over ALL 30 views. "
                  "No circularity guard possible — nothing is held out.",
                  "Step 196 (WS4 Arm A)"),
    "a6.fp_rank": ("Size-matched ranking ablation of a6.fp_dufs.",
                  "Step 196 (WS4 Arm A)"),
    "a6.pl_dufs@loco5": ("a6.pl_dufs with the LOCO_5 views as the seed set "
                        "instead of the default 4 — tests Omri's 4a (was "
                        "the seed choice too quick?). Answer: no change "
                        "(+0.01pp).", "Step 196 (WS4 Arm B)"),
    "a6.pl_dufs@central4": ("a6.pl_dufs with seeds chosen by diverse-"
                           "centrality (most consensus-correlated, capped "
                           "for redundancy) instead of the fixed priority "
                           "list.", "Step 196 (WS4 Arm B)"),

    "a4.good5+K_intrinsic_ah": ("GOOD_5 features with K set by an "
                               "intrinsic-dimension Ahn-Horenstein rank "
                               "test.", "Step 186"),
    "a4.good5+K_intrinsic_kn": ("GOOD_5 features with K set by an "
                               "intrinsic-dimension Kritchman-Nadler rank "
                               "test.", "Step 186"),
    "a4.intrinsic_k_ah": ("a4.recon's features with the intrinsic-dimension "
                         "Ahn-Horenstein K rule.", "Step 186"),
}

# ---------------------------------------------------------------------------
# Size/parameter-sweep families: many families emit the SAME method at
# several fixed sizes (or an adaptive size), which would otherwise be ~70
# near-duplicate VARIANT_NOTES lines — exactly the kind of nickname clutter
# Omri flagged. Documented once as a base + a shared suffix legend instead.
# ---------------------------------------------------------------------------

SUFFIX_NOTES = {
    "_s{N}": "Best subset of exactly size N found by this method.",
    "_adapt": "Size chosen adaptively by the method's own rule, not fixed.",
    "_k{N}": "Bottleneck / latent dimension N (Concrete Autoencoder).",
}

_SIZE_SUFFIX_RE = re.compile(r"_(s\d+|adapt)$")
_K_SUFFIX_RE = re.compile(r"_k\d+$")

BASE_VARIANT_NOTES = {
    "a1.minres_exh": ("Exhaustive search minimizing the raw Eq-14 residual.",
                     "Step 186"),
    "a1.relres_exh": ("Exhaustive search minimizing the RELATIVE residual "
                      "(raw / off-diagonal correlation energy).", "Step 186"),
    "a3.cae": ("Concrete Autoencoder feature pick at a given bottleneck "
              "size.", "Step 186"),
    "a4.anchor": ("Anchor-affinity ranking.", "Step 186"),
    "a4.recon": ("Greedy CSSP linear-reconstruction selection.", "Step 186"),
    "a5.mrmr_a0.0": ("mRMR, alpha=0 (pure relevance, no redundancy penalty) "
                    "— reproduces a4.anchor.", "Step 189"),
    "a5.mrmr_a0.3": ("mRMR, alpha=0.3 (light redundancy penalty).",
                    "Step 189"),
    "a5.mrmr_a0.5": ("mRMR, alpha=0.5.", "Step 189"),
    "a5.mrmr_a0.7": ("mRMR, alpha=0.7 (strong redundancy penalty).",
                    "Step 189"),
    "lapscore": ("Laplacian Score (He-Cai-Niyogi 2005) — classic "
                "unsupervised FS.", "Step 186"),
    "spec": ("SPEC (Zhao-Liu 2007) spectral feature selection.", "Step 186"),
    "mcfs": ("MCFS (Cai-Zhang-He 2010) multi-cluster feature selection.",
            "Step 186"),
    "decorr": ("Greedy decorrelation floor: pick the least mutually-"
              "correlated features.", "Step 186"),
    "kurtosis": ("Kurtosis-ranked floor.", "Step 186"),
    "mad": ("Median-absolute-deviation-ranked floor.", "Step 186"),
    "random": ("Random floor — the sanity check every real selector must "
              "beat.", "Step 186"),
}


def resolve(variant):
    """(note, step, suffix_label_or_None) for ANY variant string, exact or
    base+suffix. Returns None if truly undocumented."""
    if variant in VARIANT_NOTES:
        note, step = VARIANT_NOTES[variant]
        return note, step, None
    if variant in REFERENCE_SUBSET_NOTES:
        note, step = REFERENCE_SUBSET_NOTES[variant]
        return note, step, None
    m = _K_SUFFIX_RE.search(variant)
    if m:
        base = variant[:m.start()]
        if base in BASE_VARIANT_NOTES:
            note, step = BASE_VARIANT_NOTES[base]
            return note, step, "_k{N}"
    m = _SIZE_SUFFIX_RE.search(variant)
    if m:
        base = variant[:m.start()]
        if base in BASE_VARIANT_NOTES:
            note, step = BASE_VARIANT_NOTES[base]
            suffix = "_adapt" if m.group(1) == "adapt" else "_s{N}"
            return note, step, suffix
    if variant in BASE_VARIANT_NOTES:            # bare base, no suffix
        note, step = BASE_VARIANT_NOTES[variant]
        return note, step, None
    return None

# ---------------------------------------------------------------------------
# Role tags (added to the scoreboard, Step 196 / WS7) — what KIND of row a
# variant is, independent of how well it scores.
# ---------------------------------------------------------------------------

OUR_ALGORITHM = "a6.pl_dufs"


def role_of(variant):
    """The scoreboard's role tag for a variant string. Single source of
    truth — scripts/run_eval_pipeline.py imports this rather than
    re-implementing it, so the tag can never drift between the two."""
    if variant.startswith("ref."):
        return "reference_macro (hand-curated, not FS output)"
    if variant == OUR_ALGORITHM:
        return "OUR ALGORITHM (selector of record)"
    if variant.startswith("a1.router"):
        return "router (fixed subset, learned fusion choice only)"
    return "fs_selector_candidate"


ROLE_NOTES = {
    "reference_macro (hand-curated, not FS output)":
        "A human picked these features by looking at prior sweeps. Not "
        "the output of any of our label-free selection algorithms.",
    "OUR ALGORITHM (selector of record)":
        "The one selector variant actually adopted as our method's output "
        "-- what we would deploy. Currently a6.pl_dufs. CAVEAT (Step 203): "
        "adopted by default, not by merit -- both its pre-registered gates "
        "failed (mechanism 0.207 vs 0.30 bar; performance +0.22pp vs "
        "+1.0pp bar), and it is label-free at RUNTIME but seeded from "
        "GOOD_6, which was chosen using answer keys. Treat 0.7524 as the "
        "number to beat, not a strong result.",
    "fs_selector_candidate":
        "Output of one of our label-free selection algorithms (a1-a6, "
        "classical_fs), benched but not adopted as the selector of record.",
    "router (fixed subset, learned fusion choice only)":
        "Features are fixed (from a reference subset); only the FUSION "
        "method (L-SML vs U-PCR) is chosen per-cell.",
}

# ---------------------------------------------------------------------------
# Individual FEATURES (the 30 views a selector actually chooses among on the
# 25 in-scope cells) — what each computes, where it's from, and where it's
# empirically strongest. The "where best" domain split is NOT hand-typed:
# build_glossary.py merges it live from
# results/selector_bench/inscope_feature_orientation_summary.csv (QA_mean vs
# math_mean oriented AUROC), so it tracks the data, not a stale guess.
# ---------------------------------------------------------------------------

FEATURE_NOTES = {
    # --- the 16 original H(n) spectral/time-domain features (Phase 4) -----
    "epr": ("Mean token entropy H(n) over the trace — Entropy Production "
           "Rate. The project's founding signal; see "
           "\"Learned Hallucination Detection in Black-Box LLMs using "
           "Token-level Entropy Production Rate\" (project origin paper). "
           "Also the default anchor for label-free global-sign resolution "
           "(anchor_orient).", "Phase 1 / Step 148 (anchor role)"),
    "trace_length": ("Number of generated tokens.", "Phase 4"),
    "spectral_entropy": ("Shannon entropy of the normalized power spectral "
                        "density of the mean-centered entropy trace "
                        "(frequency-domain flatness).", "Phase 4"),
    "low_band_power": ("Fraction of spectral power in the low-frequency "
                      "band (0, 0.10].", "Phase 4"),
    "high_band_power": ("Fraction of spectral power in the high-frequency "
                       "band [0.40, 0.50].", "Phase 4"),
    "hl_ratio": ("high_band_power / low_band_power.", "Phase 4"),
    "dominant_freq": ("Frequency of the largest non-DC power-spectrum "
                     "component.", "Phase 4"),
    "spectral_centroid": ("Power-weighted mean frequency of the entropy "
                         "trace's spectrum.", "Phase 4"),
    "stft_max_high_power": ("Max, over sliding time windows, of the local "
                           "(STFT) fraction of power in the high-frequency "
                           "band — captures a LOCALIZED burst the global "
                           "spectrum can miss.", "Phase 4"),
    "stft_spectral_entropy": ("Mean, over sliding time windows, of the "
                             "local STFT frame's spectral entropy.",
                             "Phase 4"),
    "rpdi": ("Ratio of the trace's tail-mean entropy to its overall mean — "
            "a regime-shift/tail-drift proxy.", "Phase 4"),
    "sw_var_peak": ("Max variance in a sliding window over the entropy "
                   "trace — local burstiness.", "Phase 4"),
    "pe_mean": ("Mean sliding-window Permutation Entropy (ordinal-pattern "
               "complexity, Bandt-Pompe).", "Phase 4"),
    "hurst_exponent": ("Self-similarity exponent of the entropy trace via "
                      "Rescaled-Range (R/S) analysis.", "Phase 4"),
    "cusum_max": ("Max absolute CUSUM (cumulative sum of mean-centered "
                 "residuals) — magnitude of the largest regime shift.",
                 "Phase 4"),
    "cusum_shift_idx": ("Normalized position (0-1) of the CUSUM peak — "
                       "WHERE in the trace the shift happens, "
                       "complementing cusum_max's HOW BIG.", "Phase 4"),

    # --- spilled-energy variants, from Delta E(n) = -log p(sampled token) --
    "epr_spilled": ("Mean spilled energy Delta E(n). Decouples from epr "
                   "when the model is globally uncertain but samples a "
                   "safe token (high H, low Delta E), or confident but "
                   "generates a rare token (low H, high Delta E).",
                   "Step ~140s (Z_n / spilled-energy era)"),
    "sw_var_peak_spilled": ("sw_var_peak computed on Delta E(n) instead of "
                           "H(n).", "Step ~140s"),
    "cusum_max_spilled": ("cusum_max computed on Delta E(n).", "Step ~140s"),
    "min_spilled": ("Minimum Delta E(n) value across the trace — the "
                   "single most confident token.", "Step ~140s"),

    # --- REPGRID_VIEWS: energy (Z_n) + top-K logprob features (Step 182) ---
    "epr_energy": ("Mean of the raw full-vocab log-partition series Z_n "
                  "(token_logsumexp) — mirrors epr_spilled but computed on "
                  "the true partition function rather than the spilled-"
                  "energy proxy.", "Step 182 (Z_n backfill)"),
    "min_energy": ("Minimum Z_n across the trace.", "Step 182"),
    "sw_var_peak_energy": ("sw_var_peak computed on Z_n.", "Step 182"),
    "cusum_max_energy": ("cusum_max computed on Z_n.", "Step 182"),
    "mean_top1_logprob": ("Mean log-probability of the actually-sampled "
                         "(top-1) token, from the saved top-K logprobs.",
                         "Step 182"),
    "logprob_margin": ("Mean margin between the top-1 and top-2 token "
                      "log-probabilities — a per-token confidence gap.",
                      "Step 182"),
    "mean_logprob_entropy": ("Mean entropy of the renormalized top-K token "
                            "distribution (a K-truncated approximation of "
                            "the full-vocab entropy).", "Step 182"),
    "varentropy": ("Mean per-token VARIANCE of surprisal (-log p) over the "
                  "top-K support — dispersion of information content, "
                  "distinct from the mean surprisal itself. Kadavath et "
                  "al. 2022 (\"Language Models (Mostly) Know What They "
                  "Know\"). The GOOD_5 -> GOOD_6 addition.",
                  "Step 182/184"),
    "renyi_entropy_2": ("Mean per-token order-2 (collision) Renyi entropy, "
                       "-log(sum p^2), on the renormalized top-K "
                       "distribution.", "Step 182"),
    "topk_tail_mass": ("Mean per-token probability mass OUTSIDE the top-5 "
                      "tokens — a concentration proxy (near 0 = peaked "
                      "distribution, larger = flat/uncertain). A LOCO_5 "
                      "member.", "Step 182"),
}

# CANONICAL_POOL also defines 15 "Stage-0 derived" views (temporal-model +
# anomaly-scorer) that are NOT part of the 30-feature repgrid pool above —
# Step 191 found they don't apply to the 25 in-scope QA+math cells at all
# (they're RAG/GPQA-era derived views). Listed for completeness since they
# are still in CANONICAL_POOL / the "c46" name.
OUT_OF_SCOPE_DERIVED_VIEWS_NOTE = (
    "10 temporal-model views (bocpd_*, hmm_*, ar_*, kalman_*) + 5 anomaly-"
    "scorer views (mahalanobis, gmm_nll, kde_nll, iforest, ae, prae) exist "
    "in CANONICAL_POOL but are Stage-0 derived views built by "
    "build_derived_views.py for the (now out-of-scope) RAG/GPQA cells — "
    "they never populate on the 25 in-scope QA+math cells. This is the "
    "'30 live views, not 46' discrepancy Step 191 found."
)

# ---------------------------------------------------------------------------
# Pool modes — how many / which features a selector was allowed to choose
# from. "c46" is itself a known misnomer, worth stating plainly.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Method / study terminology (Step 203, the trimming study under
# results/pruning_study/). These are words that appear in the write-ups and
# mean something specific here — kept in the generator so they survive a
# GLOSSARY.md rebuild.
# ---------------------------------------------------------------------------

METHOD_TERM_NOTES = {
    "misfit (a.k.a. fit score, residual)":
        "The Eq.(14) L-SML residual `_residual_lsml(R, c)` — how far the "
        "observed covariance sits from what a one-shared-cause model predicts "
        "under grouping `c`. LOWER = better fit. Every trimming rule in this "
        "project steers by minimising it; Step 203 found that direction is "
        "inverted (see 'sign inversion').",
    "sign inversion":
        "Step 203's headline: misfit correlates +0.223 with AUROC (24/25 "
        "cells, within-size) and repairing the worst-fitting group is -2.22pp "
        "vs a random-group control (p=0.032). Worse-fitting subsets score "
        "HIGHER, because near-duplicate strong views (epr / epr_spilled / "
        "epr_energy / mean_top1_logprob) break the rank-one model precisely by "
        "being strong AND duplicated. Poor fit marks where the signal is, not "
        "where the junk is.",
    "localizer":
        "Using the PER-GROUP misfit to choose WHICH group to repair, as "
        "opposed to using total misfit to rank whole subsets. Different "
        "quantity, different job: the localizer discriminates strongly but "
        "points the wrong way. Do not conflate the two when citing whether "
        "'the residual is informative'.",
    "degenerate residual":
        "`scripts/test_iterative_lsml_pruning.py::compute_lsml_residual` = "
        "||Cov*v1 - lambda1*v1||, which is ZERO BY CONSTRUCTION (measured "
        "2e-15) since v1 is Cov's own eigenvector. Every number that file "
        "produced is void, including the 0.7004 once recorded against Omri's "
        "trimming idea. Do not reuse that file.",
    "near-tie":
        "A removal step whose runner-up candidate improves the fit within 10% "
        "of the best candidate. ~11 of 18 steps per cell — so the tie-breaker, "
        "not the criterion, makes most of the decisions.",
    "graph scope (tie-breaker)":
        "Which measurements build the ANSWER-by-ANSWER graph used by "
        "`classical_fs._laplacian_score`. Restricting attention to a cluster "
        "does NOT take a subgraph — it rebuilds a different graph. "
        "Within-cluster comparison is legitimate; cross-cluster is not (the "
        "score is a Rayleigh quotient normalised inside its own graph). Swept "
        "as all-30 / surviving / group-only / group-minus-candidate / "
        "anchor-only.",
    "automatic picker vs fixed subset":
        "An AUTOMATIC PICKER chooses views itself, per cell, at runtime "
        "(a6.pl_dufs, 0.7524 — the bar a new label-free method must clear). A "
        "FIXED SUBSET was chosen once USING labels and then reused (GOOD_6 "
        "0.7594; LOCO_5 0.7705 on 24 cells). Comparing a label-free selector "
        "against GOOD_6 is comparing against an anchor, not a fair target.",
    "stale sweep cache":
        "results/subset_sweep/repgrid__*.npz — ~1.03M scored subsets, of which "
        "only 5/19 cells still reproduce (disagreements to 0.374 AUROC) "
        "because cells were re-graded after the sweep. The npz files look "
        "healthy; only re-scoring detects it. Audit before any reuse: "
        "results/pruning_study/03_size_and_criterion/cache_staleness_audit.csv.",
    "compute_score_matrix=False":
        "Flag on `lsml_continuous` that skips the O(m^4) Eq.(15) score matrix "
        "on the `groups=`-given path, where nothing reads it. 103x faster at "
        "m=30, output bit-identical. Default True preserves the old meta dict. "
        "Use it in any subset sweep. (Step 203 also vectorised "
        "`_score_matrix_lsml` 34x and the `_residual_lsml` / "
        "`_estimate_von_voff` inner loops.)",
}

POOL_MODE_NOTES = {
    "h16": (f"The original {len(H16)} H(n) spectral features "
           "(FEAT_NAMES[:16]) — every cell has these, oldest pool.",
           "Phase 4"),
    "c46": (f"CANONICAL_POOL in code has {len(CANONICAL_POOL)} candidate "
           "views (spectral + temporal + anomaly-scorer + repgrid "
           "energy/logprob) — hence the name. BUT on repgrid cells "
           "(the 25 in-scope QA+math cells), the anomaly-scorer views "
           "don't apply, so the LIVE per-cell pool is usually 27-30, not "
           "46. 'c46' names the pool DEFINITION, not the per-cell count — "
           "check a cell's actual size, don't assume 46.",
           "Step 191 (found the 30-not-46 discrepancy)"),
}
