"""Can any PUBLISHED feature-selection condition replace U-PCR's keep rule?

WHY THIS EXISTS
    U-PCR's deployed keep rule (upcr.py:287-293) is
        keep view i  <=>  rho_hat_i >= min_frac * Var(y)  AND
                          rho_hat_i >= max_j(rho_hat_j) / exclude_frac
    i.e. PER-FEATURE, MARGINAL, THRESHOLDED on each view's own correlation with the
    latent. That is exactly the shape Steps 221-222 closed: the TRUE correlation with
    correctness buys +0.08pp (p=0.62) and no label-free per-feature ranker clears the
    matched floor.

    Steps 223's two arms then closed two more shapes. l0-CCA (Lindenbaum, Salhov,
    AVERBUCH, Kluger) -- trained joint gates on a cross-channel total-correlation loss
    -- came in at -0.12pp / -0.47pp. Five set-level covariance functionals (composite
    reliability) came in at +0.08pp best, Holm 0.72. The label-handed oracle running the
    SAME search clears the floor by +1.88pp, so the search works and the objective is
    what fails.

    THE GAP THIS SCRIPT FILLS. Eight published conditions are already implemented in
    this repository (spectral_utils/selectors/) and NOT ONE has ever been evaluated as a
    keep rule in this channel. They were benchmarked in the Step-186 selector bench
    against a different baseline (GOOD_5/GOOD_6 macro) on a different harness, which
    does not answer "does this rule beat random selection at the same size, inside
    U-PCR". No new algorithm code is written here: every condition is called through the
    existing `@register` contract, unmodified.

THE ONE CONDITION BEING SWITCHED
    Polarity from sign(rho_hat), the U-PCR fit, the weights, the anchor and the scoring
    are all UNCHANGED. Only the source of the column subset changes. Applied by passing
    a column subset to fit_cols(..., exclusion=False). No deployed file is edited.

    TWO ARENAS, both reported, because they answer different questions:
      full   the condition chooses from the WHOLE pool (m ~ 28). This is the true
             "swap the rule" comparison -- U-PCR's rule also chooses from the whole
             pool -- and it is the PRIMARY.
      keep   the condition chooses from half A's deployed keep set (~21 views), i.e.
             U-PCR's rule runs first and the new rule prunes it. This is the arena in
             which exp12's room (+2.25pp) and exp13's floor (-0.84pp) are defined, so it
             is the COMPARABILITY arm. It is two conditions stacked, not one, and must
             never be quoted as "replacing" the rule.

WHY THE FLOOR IS PER-CONDITION AND PER-SIZE
    These selectors emit their own native sizes (3-18 observed), not k. A size-s subset
    cannot be compared to a floor drawn at size 11.75. So EVERY variant is compared to
    N_FLOOR random subsets of ITS OWN size drawn from ITS OWN arena's population. exp13's
    fixed-k floor and exp12's good set are still carried as reference rows.

    DELIBERATELY NOT DONE: forcing every condition to size k. The selectors expose
    nested prefixes only up to size 6 (SIZES = (4,5,6)), so an at-k readout would mean
    reimplementing each paper's ranking beyond position 6. That is exactly the kind of
    reimplementation that would stop measuring the published condition. The size-matched
    floor is the correct control and needs no such surgery.

WHAT IS PRE-REGISTERED, WRITTEN BEFORE THE RUN

    FIDELITY -- WHAT THESE ARMS ACTUALLY ARE (2026-08-04 review, before the run)
    The honest scope of this experiment is "PUBLISHED SCORING FUNCTIONS, OUR SIZE RULE",
    NOT "published conditions". Only ONE of the candidates (a2.dufs_pf) carries its
    paper's own parameter-free size rule; Laplacian Score, SPEC, MCFS, mRMR and CAE all
    define a RANKING only, and the number kept is a user parameter in every one of those
    papers. The size rules at classical_fs.py:76-85, a5_mrmr.py:86-99 and
    a3_concrete_ae.py:195-204 are OURS. Any null must be stated at that scope.

    Three further consequences, all measured rather than assumed:
      * SIX of the candidates are hard-capped at size <= 8 by module constants, while the
        incumbent keep set is ~11.75 views. No "adapt" arm can propose a set as large as
        the rule it replaces. The size-matched floor keeps the contrast valid; the
        vs-DEPLOYED contrast is where the cap shows up.
      * The DUFS signed-gate trap is REAL and was avoided: every readout uses mu > 0, not
        |mu|. On one probed cell the two rules disagree on 4 of 13 kept views, so an |mu|
        readout would have been a different selector wearing the same name.
      * No label leakage anywhere in the family: cache=None, UnlabeledCell has no labels,
        and no hand-picked feature list reaches any designated primary.

    THE PRIMARY FAMILY -- FIVE arms, one designated variant each:

      a2.dufs_pf          DUFS Eq.(7), the paper's PARAMETER-FREE loss (Lindenbaum et
                          al., NeurIPS 2021). VERDICT: FAITHFUL -- Eq. (7) exact
                          including the 1/m, signed mu>0 readout exact, and the only arm
                          here whose SIZE rule is the paper's own. Condition: keep the
                          views whose signed gates stay open under an objective that
                          preserves the sample graph's local neighbourhood structure.
      a3.cae              Concrete Autoencoder (Balin, Abid, Zou, ICML 2019). VERDICT:
                          FAITHFUL WITH DOCUMENTED DEVIATIONS -- concrete layer, the
                          T0=10 -> 0.01 anneal and the argmax readout are exact, but the
                          shipped subset is post-processed by an exhaustive swap search
                          on the eval objective (deviation 5), so it is best described as
                          CAE-INITIALISED BEST-SUBSET RECONSTRUCTION.
      lapscore_adapt      Laplacian Score (He, Cai, Niyogi, NIPS 2005). VERDICT: score
                          exact term-for-term; size rule ours.
      spec_adapt          SPEC phi_2 (Zhao & Liu, ICML 2007). VERDICT: phi_2 exact;
                          size rule ours.
      a1.relres_greedy    Eq-14 / tetrad structural residual (Jaffe, Nadler, Kluger),
                          via detect_dependent_groups. Condition: keep the subset whose
                          covariance best FITS the assumed structure. CAVEAT, registered
                          in advance: a1's greedy optimises size 3 over 200 sampled seed
                          triples but every larger size over a single greedy step, so its
                          chosen size collapses to 3 in most rows. That is a property of
                          OUR search, not of the paper, and the arm is read as "the Eq-14
                          objective at size ~3", not as a size-selection method.

    DEMOTED FROM THE FAMILY by the same review -- reported as sensitivity, and NEVER
    written up as "paper X's condition failed" (full reasons in the DEMOTED dict below):
      a2.select            GroupFS's published keep rule is replaced by DUFS gates.
      mcfs_adapt           undocumented re-weighting of the Lasso coefficients.
      a5.mrmr_a0.5_adapt   seeded on the hand-picked `epr` anchor; size is a constant 8.
      a1.upcrres_greedy    our U-PCR residual, not Jaffe/Nadler/Kluger Eq. 14.

    MULTIPLICITY: flat Holm-Bonferroni over the FIVE, computed SEPARATELY within each
    arena (the arenas answer different questions and are not one family). Per-arm CIs are
    reported beside the adjusted p-values and never replaced by them.

    EVERYTHING ELSE IS SENSITIVITY, outside the family, reported with CIs and no Holm:
      * the fixed-size siblings (_s4/_s5/_s6, cae_k3..k8, the other mrmr alphas)
      * a4.* (our own Step-187 construction, not a published condition)
      * a6.* (Step-194 pseudo-label DUFS -- the selector of record, but it consumes an
        L-SML pseudo-label, so it is a REFERENCE, not a label-free paper condition)
      * a7.iter_consensus (Extension H, ours)
      * ref.* (GOOD_5 / GOOD_6 / LOCO_5 / STABLE_H9 ... -- HAND-PICKED PRIORS. Reference
        and compatibility rows ONLY. Never the contribution, never a "method".)
      * random_s4/5/6 (simple_stats) -- the explicit null control. It must land at its
        own floor; if it does not, the floor construction is wrong.
      * anything whose name contains good5 / loco5 / central4 -- prior-carrying, so
        structurally ineligible for the primary family whatever it scores.

    DECISION RULE, fixed in advance:
      PRIMARY    TWO contrasts, both reported for every arm, neither allowed to stand in
                 for the other:
                   A  vs N_FLOOR random subsets of the SAME size from the SAME arena.
                      Asks: is the rule doing anything beyond keeping that many views?
                   B  vs the DEPLOYED U-PCR rule (full pool, its own exclusion active).
                      Asks: should we swap the rule? THIS IS THE QUESTION.
                 Paired over the 24 test sets. Positioned against the room (+2.25pp) and
                 exp13's fixed-k floor (-0.84pp), both re-derived on these rows.

                 WHY BOTH, and why B cannot be dropped: contrast A is a LOW BAR at small
                 sizes. A 1-cell pilot (2026-08-04, before the registered run) showed
                 hand-picked 5-view subsets clearing their own size-5 floor by +9 to
                 +11pp — because a random 5-view subset is very weak — while the deployed
                 21-view keep set is far stronger than any random 5. An arm can therefore
                 win contrast A decisively and still be a downgrade. Recorded here as a
                 design fix motivated by that structural fact, not by the pilot's values,
                 which were 1 cell and are not quotable.
      SECONDARY  overlap with exp12's good set against exp13's COMPOSITION-MATCHED null
                 at the variant's own size. Per exp14, an arm that clears only the
                 overlap test has reproduced the old outcome, not beaten it.
      POWER GATE exp13's replayed floor and deployed AUROC must reproduce to 1e-9, and
                 the room and floor must re-derive at +2.25 / -0.84pp. The label-handed
                 oracle (+1.88pp, exp15) already established that the channel is
                 reachable at fixed k, so a null here is interpretable.

    A NULL RESULT IS A RESULT. If no published condition clears its own matched floor,
    the closure extends from "per-feature rankers" and "set-level covariance functionals"
    to "the published unsupervised feature-selection literature as applied to this
    channel" -- across all four of the clusters that literature falls into. That is
    written here so it cannot be renegotiated afterwards.

REPRODUCTION
    Splits are exp12's, replayed through exp13's machinery exactly as exp14 and exp15 do:
    same per-cell crc32 seed, same replayed random consumption, `nrng` draws in the same
    order (including exp13's two post-floor cond_overlap_null calls). ALL new randomness
    is on SEPARATE generators (`frng` for the size-matched floors, `orng` for the overlap
    nulls, `srng` handed to the selectors), so exp13/14/15's arms are invariant to what is
    added here by construction rather than by ordering discipline. The run asserts, per
    split and to 1e-9, that the deployed AUROC and the pruning floor match exp13's
    splits.csv before any new number is trusted.

    Cells are independent, so the work is parallelised OVER CELLS. Every generator is
    seeded from the cell key alone, so worker assignment cannot change any number.

Run:  python scripts/upcr_study/exp16_paper_conditions.py                 (~2 h, 8 procs)
      python scripts/upcr_study/exp16_paper_conditions.py --cells 2 --workers 2
      python scripts/upcr_study/exp16_paper_conditions.py --skip a6_pseudolabel_gates
Out:  results/upcr_study/16_paper_conditions/
"""
import argparse
import csv
import os
import sys
import time
import zlib
from multiprocessing import Pool

import numpy as np
from scipy.stats import spearmanr, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as S                                                    # noqa: E402

from exp10_channel_ceilings import (                                  # noqa: E402
    FIT, N_SPLITS, derive_cell, derived_arm_gate, fit_cols, _zsc, _all_fields,
)
from exp13_incumbent_anchored_ranking import (                        # noqa: E402
    N_RANDOM, N_OVERLAP_DRAWS, TOL_REPRO,
    prune_incumbent, cond_overlap_null, load_exp12_splits,
)
from spectral_utils.subset_sweep import ALL_SIGNS, CANONICAL_POOL     # noqa: E402
from spectral_utils.upcr import upcr_fit                              # noqa: E402
from spectral_utils.selector_bench import UnlabeledCell               # noqa: E402
import spectral_utils.selectors as SEL                                # noqa: E402

EXP13_DIR = "13_incumbent_anchored_ranking"
OUT_DIR = "16_paper_conditions"

ARENAS = ("full", "keep")
N_FLOOR = N_RANDOM               # 25, matched to exp12/exp13's floor
ROOM_PP = 2.25
FLOOR_PP = -0.84
MIN_COLS = 3                     # fit_cols returns NaN below this (Eq. 21 needs >= 3)

# The primary family, set by the 2026-08-04 fidelity review (see FIDELITY below). FIVE
# arms, not eight: three candidates were demoted because the thing they run is not the
# published condition. Named BEFORE the run.
PRIMARY = (
    "a2.dufs_pf",           # DUFS Eq.(7), parameter-free      (Lindenbaum, NeurIPS'21)
    "a3.cae",               # Concrete Autoencoder + swap-refine (Balin, Abid, Zou'19)
    "lapscore_adapt",       # Laplacian Score                  (He, Cai, Niyogi NIPS'05)
    "spec_adapt",           # SPEC phi_2                       (Zhao & Liu ICML'07)
    "a1.relres_greedy",     # Eq-14 structural residual        (Jaffe, Nadler, Kluger)
)
# Demoted from the family by the fidelity review, with the reason. Reported as
# sensitivity, never as "paper X's condition failed".
# ROUND 2, pre-registered 2026-08-05 BEFORE the run that scores it. Both arms were built
# after the Round-1 pre-registration, so their status differs and must be reported as such:
#   a8 (LS-CAE)  was already scored in the Round-1 sweep as SENSITIVITY. Its numbers have
#                been seen, so it CANNOT be pre-registered now — it is exploratory and
#                needs confirmation on a run that was registered before seeing it.
#   a9 (DPP)     has NOT been run. It is pre-registered here, in advance, as a family of
#                one, with the same decision rule as Round 1.
ROUND2_PREREGISTERED = ("dpp",)
ROUND2_EXPLORATORY = ("lscae", "lscae.recon_only")

DEMOTED = {
    "a2.select": "GroupFS's published keep rule (rank groups by their own group gate "
                 "means) is replaced by DUFS per-feature gates read at group "
                 "granularity — a2_groupfs.py:512-515, deviation 8. The grouping is "
                 "GroupFS; the SELECTION is DUFS, and selection is what this experiment "
                 "measures.",
    "mcfs_adapt": "carries an undocumented max(0, 1 - lambda_c) re-weighting of the "
                  "Lasso coefficients (classical_fs.py:178-179) that appears nowhere in "
                  "Cai et al. and changes the ranking; the module docstring states the "
                  "plain paper formula instead.",
    "a5.mrmr_a0.5_adapt": "relevance is |rho(feature, cell.anchor)| and the anchor "
                          "resolves to `epr` = GOOD_5[0], so the greedy seeds on a "
                          "HAND-PICKED feature (57/57 rows in the committed bench). "
                          "Also its 'adaptive' size is a constant 8 — the break "
                          "condition at a5_mrmr.py:86-99 is unreachable.",
    "a1.upcrres_greedy": "runs OUR U-PCR rank-1 projection residual "
                         "(fusion_utils.py:1652), not Jaffe/Nadler/Kluger Eq. 14. The "
                         "Eq-14 arm is a1.relres_greedy, which is now the designated "
                         "one.",
}
# Structurally ineligible for the primary family whatever they score: these carry a
# hand-picked feature list. NOTE this is a NAME check only — it cannot see a prior that
# enters through the data, which is exactly how a5.mrmr_* acquires `epr` (see DEMOTED).
# Any new primary must be checked by reading it, not by trusting this gate.
PRIOR_TOKENS = ("good5", "good_5", "good6", "loco5", "loco_5", "central4",
                "stable_h9", "ref.", "all_h16", "entropy_6", "consensus_4",
                "top_macro", "anchor")

_CELLS = _PREV12 = _PREV13 = None    # per-worker caches, filled by `_init_worker`


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _assert_prior_free():
    """A designated primary must not carry a hand-picked feature list.

    This is a NAME check and it is not sufficient on its own — a5.mrmr_* passed it while
    seeding its greedy on `epr` through `cell.anchor`, which is why that arm is in
    DEMOTED. It is kept as a cheap tripwire against a renamed variant, and every primary
    is additionally verified by reading it (see FIDELITY in the module docstring)."""
    for v in PRIMARY:
        low = v.lower()
        bad = [t for t in PRIOR_TOKENS if t in low]
        if bad:
            raise RuntimeError(
                f"PRIMARY variant {v!r} carries a prior token {bad} — hand-picked "
                "subsets are reference rows, never the contribution (CLAUDE.md).")
        if v in DEMOTED:
            raise RuntimeError(
                f"PRIMARY variant {v!r} is in DEMOTED: {DEMOTED[v]}")


_assert_prior_free()


def _check(ck, rep, name, got, want):
    if not np.isfinite(got) and not np.isfinite(want):
        return
    # A finiteness MISMATCH must raise. `abs(nan - 0.77) > tol` is False, so without this
    # branch a split that silently went NaN here while exp13 recorded a real number would
    # pass the reproduction gate.
    if np.isfinite(got) != np.isfinite(want):
        raise RuntimeError(
            f"{ck} rep {rep}: {name} = {got} vs exp13's {want} — one is NaN and the "
            "other is not; the reproduction gate did not fire but should have")
    if abs(got - want) > TOL_REPRO:
        raise RuntimeError(
            f"{ck} rep {rep}: {name} = {got:.12f} != exp13's {want:.12f} — a stream "
            "this script must not touch has moved")


def _substream(*parts):
    """A generator seeded from its REQUEST, not from a running stream.

    The floors and overlap nulls are drawn lazily on cache miss, in selector-iteration
    order. On one shared generator that makes every floor depend on which OTHER families
    ran, so `--skip` would silently change the surviving arms' numbers. Seeding each
    request from (cell, rep, arena, size, ...) makes them invariant to iteration order,
    to --skip and to --cells."""
    key = "|".join(str(p) for p in parts)
    return np.random.default_rng(zlib.crc32(key.encode()) % (2 ** 32))


def load_exp13_splits():
    path = os.path.join(S.outdir(EXP13_DIR), "splits.csv")
    with open(path, newline="") as f:
        return {(r["cell"], int(r["rep"])): r for r in csv.DictReader(f)}


def build_ucell(ck, V_half, anchor_half, anchor_name, pool, cols):
    """An UnlabeledCell over `cols` of half A. Carries NO labels and no positive rate —
    that is the whole point of the UnlabeledCell contract, and it is what makes a
    selector's output label-free by construction rather than by inspection."""
    cols = sorted(int(c) for c in cols)
    Vs = np.asarray(V_half[:, cols], dtype=float)
    names = [pool[c] for c in cols]
    if Vs.shape[1] > 2:                       # matches subset_sweep.prepare_cell:401
        rho = spearmanr(Vs)[0]
    else:
        rho = np.corrcoef(Vs.T)
    rho = np.abs(np.nan_to_num(np.atleast_2d(rho), nan=0.0))
    return UnlabeledCell(
        domain=ck.split("_")[0], cell_key=ck, pool=names,
        pool_bits=np.array([CANONICAL_POOL.index(f) for f in names], dtype=np.uint8),
        # anchor_name must be the cell's RESOLVED anchor feature, not a literal: a6
        # branches on `cell.anchor_name in cell.pool` to force-include the anchor in its
        # seed set, and a fabricated name would silently disable that branch and run a
        # different selector under the published name. `.copy()` because the array is
        # shared across every family in the split.
        V=Vs, anchor=np.asarray(anchor_half, dtype=float).copy(),
        anchor_name=anchor_name, rho=rho), cols


def size_matched_floor(cb, ck, rep, arena, population, size, cache):
    """Mean held-out AUROC of N_FLOOR random subsets of `size` drawn from `population`.

    Cached per (arena, size) WITHIN a split — the floor depends on neither the condition
    nor its identity, only on how many views it kept and from where. Keyed by the arena
    NAME, not by `id(population)`: ids are reused after garbage collection, and a stale
    hit would silently hand one arena's floor to the other."""
    pop = np.asarray(sorted(int(c) for c in population), dtype=int)
    size = int(min(size, len(pop)))       # clamp BEFORE keying: a request above |pop|
    key = (arena, size)                   # and one at |pop| are the same draw
    if key in cache:
        return cache[key]
    if size < MIN_COLS:
        cache[key] = (float("nan"), 0)
        return cache[key]
    g = _substream("floor", ck, rep, arena, size)
    vals = []
    for _ in range(N_FLOOR):
        pick = g.choice(pop, size, replace=False)
        vals.append(fit_cols(cb, pick, exclusion=False)[0])
    out = (float(np.nanmean(vals)), int(np.sum(~np.isfinite(vals))))
    cache[key] = out
    return out


# ---------------------------------------------------------------------------
# per-cell driver
# ---------------------------------------------------------------------------

def run_cell(ck, cell, dcell, prev12, prev13, args):
    V0 = cell["V"]
    y = dcell["labels"]
    n, m = V0.shape
    hand = np.array([ALL_SIGNS.get(f, +1) for f in cell["pool"]], dtype=float)
    pool = list(cell["pool"])
    anchor_name = getattr(cell.get("unlabeled"), "anchor_name", None) or pool[0]

    rng = np.random.default_rng(zlib.crc32(ck.encode()) % (2 ** 32))
    nrng = np.random.default_rng(0xC0FFEE + (zlib.crc32(ck.encode()) % 10 ** 6))
    # All NEW randomness is on per-request substreams (`_substream`), never on `rng` or
    # `nrng`, so exp13/14/15's arms are invariant to anything added here by construction
    # rather than by ordering discipline.
    names = [nm for nm in sorted(SEL.registered()) if nm not in args.skip]
    if args.only:
        names = [nm for nm in names if nm in set(args.only)]
    long_rows, ref_rows = [], []

    for rep in range(N_SPLITS):
        # --- exp12's split, reproduced exactly (identical to exp13/14/15) ----------
        idx = rng.permutation(n)
        a_idx, b_idx = idx[: n // 2], idx[n // 2:]
        if len(np.unique(y[a_idx])) < 2 or len(np.unique(y[b_idx])) < 2:
            continue
        raw_a = np.column_stack([_zsc(V0[a_idx, j]) for j in range(m)]) * hand
        raw_b = np.column_stack([_zsc(V0[b_idx, j]) for j in range(m)]) * hand
        try:
            pol = np.sign(upcr_fit(raw_a.T, **FIT).rho_hat_full)
        except Exception:
            continue
        pol[pol == 0] = 1.0
        ca = {"V": raw_a * pol, "anchor": _zsc(dcell["anchor"][a_idx]), "labels": y[a_idx]}
        cb = {"V": raw_b * pol, "anchor": _zsc(dcell["anchor"][b_idx]), "labels": y[b_idx]}

        _, res_a = fit_cols(ca, range(m))
        if res_a is None:
            continue
        start_a = [int(j) for j in np.where(res_a.keep)[0]]

        ref12, ref13 = prev12.get((ck, rep)), prev13.get((ck, rep))
        if ref12 is None or ref13 is None:
            raise RuntimeError(f"{ck} rep {rep}: no exp12/exp13 row — splits diverged")
        k = int(ref12["k"])
        cols_greedy = [pool.index(f) for f in ref12["greedy_cols"].split("|")]

        auc_dep = fit_cols(cb, range(m))[0]
        _check(ck, rep, "auroc_deployed", auc_dep, float(ref13["auroc_deployed"]))

        ref = {"cell": ck, "rep": rep, "m": m, "k": k,
               "n_keep_deployed_halfA": len(start_a),
               "auroc_deployed": auc_dep,
               "auroc_greedy": float(ref12["auroc_greedy"]),
               "auroc_keepset": fit_cols(cb, start_a, exclusion=False)[0]}

        # --- the two arenas ---------------------------------------------------------
        populations = {"full": list(range(m)), "keep": list(start_a)}
        floor_cache, null_cache, keep_set = {}, {}, set(start_a)
        for arena in ARENAS:
            if arena not in args.arenas:
                continue
            popcols = populations[arena]
            for fam in names:
                # Rebuilt per family: `bench_selector` only ever hands one selector one
                # cell, so reuse across families is untested and a selector that mutated
                # `cell.V` in place would silently corrupt every family after it.
                uc, cols_map = build_ucell(ck, ca["V"], ca["anchor"], anchor_name,
                                           pool, popcols)
                fn = SEL.get_selector(fam)
                t0 = time.time()
                # Seeded from (cell, rep, arena, family) so a selector's seed does not
                # depend on which OTHER families ran — a --skip or --cells pilot must
                # reproduce the full run's numbers exactly, not merely resemble them.
                seed = zlib.crc32(f"{ck}|{rep}|{arena}|{fam}".encode()) % (2 ** 31)
                try:
                    sels = fn(uc, np.random.default_rng([int(seed), 0]), cache=None)
                except Exception as exc:
                    long_rows.append({
                        "cell": ck, "rep": rep, "arena": arena, "family": fam,
                        "variant": f"{fam}.RAISED", "size": 0, "auroc": np.nan,
                        "floor": np.nan, "overlap": np.nan, "overlap_null": np.nan,
                        "fallback": 1, "seconds": time.time() - t0,
                        "error": f"{type(exc).__name__}: {exc}"})
                    continue
                secs = time.time() - t0
                for s in sels:
                    local = [int(c) for c in s["cols"]]
                    gcols = sorted(cols_map[c] for c in local)
                    size = len(gcols)
                    auc = (fit_cols(cb, gcols, exclusion=False)[0]
                           if size >= MIN_COLS else float("nan"))
                    fl, _ = size_matched_floor(cb, ck, rep, arena, popcols, size,
                                               floor_cache)
                    ov = (len(set(gcols) & set(cols_greedy)) / size
                          if size else float("nan"))
                    # The composition-matched null depends ONLY on (size, how many of
                    # the picked columns are inside the keep set) — not on which ones.
                    # Caching on that pair turns ~190 x 2000 draws per split into ~20,
                    # which is the difference between this script running and not.
                    n_in = sum(1 for j in gcols if j in keep_set)
                    ckey = (size, n_in)
                    if size < 1:
                        nul = float("nan")
                    elif ckey in null_cache:
                        nul = null_cache[ckey]
                    else:
                        nul = cond_overlap_null(
                            cols_greedy, gcols, start_a, m, size,
                            _substream("null", ck, rep, size, n_in))
                        null_cache[ckey] = nul
                    long_rows.append({
                        "cell": ck, "rep": rep, "arena": arena, "family": fam,
                        "variant": s["variant"], "size": size, "auroc": auc,
                        "floor": fl, "overlap": ov, "overlap_null": nul,
                        "fallback": int(bool(s.get("fallback", False))),
                        "seconds": secs / max(len(sels), 1), "error": ""})

        # --- exp13's floor, replayed on `nrng` in exp13's own order -----------------
        rnd = []
        for _ in range(N_RANDOM):
            order = [int(j) for j in nrng.permutation(m)]
            cols_r = prune_incumbent(start_a, order, k, m)
            rnd.append(fit_cols(cb, cols_r, exclusion=False)[0])
        ref["auroc_random_prune"] = float(np.nanmean(rnd))
        _check(ck, rep, "auroc_random_prune", ref["auroc_random_prune"],
               float(ref13["auroc_random_prune"]))

        # exp13 draws two conditional nulls from `nrng` AFTER the floor; the stream must
        # advance identically or the NEXT split's floor diverges. Replaying them also
        # re-checks both values, which is a free extra gate (exp14/exp15 do the same).
        for nm in ("truecorr", "rho"):
            rebuilt = [pool.index(f) for f in ref12[f"{nm}_cols_ownk"].split("|")]
            got = cond_overlap_null(cols_greedy, rebuilt, start_a, m, k, nrng)
            _check(ck, rep, f"overlap_{nm}_conditional_null", got,
                   float(ref13[f"overlap_{nm}_conditional_null"]))

        # A free consistency check on the new floor machinery: at arena=keep and size=k
        # the size-matched draw and exp13's prune_incumbent draw the SAME distribution
        # (a uniform k-subset of start_a), so the two floors must agree in expectation.
        if "keep" in args.arenas:
            fk, _ = size_matched_floor(cb, ck, rep, "keep", populations["keep"], k,
                                       floor_cache)
            ref["floor_keep_at_k"] = fk
            # On the 2/120 splits where k > |keep set| the size-matched draw clamps to
            # the whole keep set while exp13's prune_incumbent legitimately reaches
            # outside it, so the two are not the same draw and the check below is not
            # meaningful there. Flagged rather than silently averaged in.
            ref["k_exceeds_keepset"] = int(k > len(start_a))

        # --- exp12's random consumption, replayed so the next split lines up --------
        for _ in range(2 * N_OVERLAP_DRAWS + N_RANDOM):
            rng.choice(m, k, replace=False)

        ref_rows.append(ref)

    return long_rows, ref_rows


# ---------------------------------------------------------------------------
# multiprocessing plumbing (spawn-safe: every generator is seeded from the cell key,
# so which worker takes which cell cannot change a number)
# ---------------------------------------------------------------------------

def _init_worker():
    global _CELLS, _PREV12, _PREV13
    _CELLS = S.load()
    _PREV12, _PREV13 = load_exp12_splits(), load_exp13_splits()


def _work(payload):
    ck, args = payload
    cell = _CELLS[ck]
    t0 = time.time()
    long_rows, ref_rows = run_cell(ck, cell, derive_cell(cell),
                                   _PREV12, _PREV13, args)
    return ck, long_rows, ref_rows, time.time() - t0


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def collapse_long(long_rows, refs):
    """Per (arena, variant, cell) mean over the cell's splits — the unit of the paired
    test is the CELL, matching exp12-15.

    Each row also carries that cell's DEPLOYED numbers, because the size-matched floor
    alone is a misleading bar. A rule that keeps 5 views beats a random 5-view subset
    easily (the pilot showed +9 to +11pp for hand-picked 5-view sets) and can still lose
    to the deployed 21-view keep set. Both contrasts are reported; the one that answers
    "should we swap the rule" is the comparison against `deployed`."""
    ref_by_cell = {r["cell"]: r for r in refs}
    out = {}
    for r in long_rows:
        out.setdefault((r["arena"], r["variant"]), {}).setdefault(r["cell"], []).append(r)
    agg = {}
    for key, bycell in out.items():
        rows = []
        for ck, rs in bycell.items():
            # FALLBACK SPLITS ARE EXCLUDED from every performance contrast, and the rate
            # is carried instead. Every torch-backed selector returns the WHOLE POOL with
            # fallback=True on an exception. Scored naively that is a size-|pop| "keep
            # set" whose size-matched floor draws the whole population every time, so
            # floor == auroc exactly and the split contributes delta 0 — shrinking a real
            # effect toward the floor in proportion to how often the method crashed, and
            # reporting a broken run as "at the floor". Registered rule, applied before
            # any number is read: score only the splits where the rule actually selected,
            # and report `fallback` beside every arm so a high rate is visible, not
            # absorbed.
            ok = [x for x in rs if not int(x["fallback"])]

            def mean(f, src):
                v = [float(x[f]) for x in src if np.isfinite(float(x[f]))]
                return float(np.mean(v)) if v else float("nan")

            # Shared finite mask so `auroc` and `floor` are averaged over the SAME splits
            # — otherwise the "paired" difference stops being a mean of paired
            # differences (a upcr_fit failure yields a finite floor beside a NaN auroc).
            pairs = [x for x in ok
                     if np.isfinite(float(x["auroc"])) and np.isfinite(float(x["floor"]))]
            ref = ref_by_cell.get(ck, {})
            rows.append({"cell": ck, "family": rs[0]["family"],
                         "auroc": mean("auroc", pairs), "floor": mean("floor", pairs),
                         "size": mean("size", ok), "overlap": mean("overlap", ok),
                         "overlap_null": mean("overlap_null", ok),
                         "fallback": float(np.mean([int(x["fallback"]) for x in rs])),
                         "n_scored": len(pairs), "n_splits": len(rs),
                         "deployed": ref.get("auroc_deployed", float("nan")),
                         "keepset": ref.get("auroc_keepset", float("nan")),
                         "good": ref.get("auroc_greedy", float("nan"))})
        agg[key] = rows
    return agg


def arm_stats(rows):
    """Both contrasts for one arm, plus the diagnostics that keep them honest."""
    st = paired(rows, "floor", "auroc")
    st["vs_deployed"] = paired(rows, "deployed", "auroc")
    st["vs_keepset"] = paired(rows, "keepset", "auroc")
    st["vs_good"] = paired(rows, "good", "auroc")
    st["mean_size"] = float(np.nanmean([r["size"] for r in rows]))
    st["fallback_rate"] = float(np.nanmean([r["fallback"] for r in rows]))
    st["n_cells_scored"] = int(sum(1 for r in rows if np.isfinite(r["auroc"])))
    st["overlap_excess"] = paired(rows, "overlap_null", "overlap",
                                  scale=1.0)["mean_delta"]
    return st


def paired(rows, base, arm, scale=100.0):
    d = np.array([r[arm] - r[base] for r in rows], dtype=float) * scale
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return {"mean_delta": float("nan"), "n": 0, "ci95": [np.nan, np.nan],
                "wins": 0, "losses": 0, "p": float("nan")}
    # Vectorised: this is called ~800 times over the two arenas, and the Python-loop form
    # added a 10-20 minute single-threaded tail after two hours of compute.
    g = np.random.default_rng(0)
    bs = d[g.integers(0, len(d), size=(20000, len(d)))].mean(axis=1)
    return {"mean_delta": float(d.mean()),
            "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "n": int(len(d)),
            "p": float(wilcoxon(d).pvalue) if len(d) > 5 and np.any(d != 0)
                 else float("nan")}


def holm(pvals, family_size):
    """Holm-Bonferroni over a family of PRE-REGISTERED size.

    `family_size` is passed explicitly and is NOT `len(pvals)`. A designated primary can
    go missing three ways — --skip, a selector that never emits its variant, or a NaN p
    from too few surviving cells — and every one of them would otherwise shrink the
    multiplier and make the survivors look MORE significant than the pre-registration
    allows. A missing or non-finite arm is carried at p = 1.0, which keeps the family at
    its registered size and is conservative in the correct direction."""
    filled = {k: (v if np.isfinite(v) else 1.0) for k, v in pvals.items()}
    items = sorted(filled.items(), key=lambda kv: kv[1])
    n, out, run = int(family_size), {}, 0.0
    if len(items) > n:
        raise RuntimeError(f"holm: {len(items)} tests in a family registered at {n}")
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, p * (n - i)))
        out[k] = run
    return out


def fmt(label, st, unit="pp"):
    return (f"  {label:34s} {st['mean_delta']:+7.2f}{unit}  "
            f"[{st['ci95'][0]:+.2f}, {st['ci95'][1]:+.2f}]  "
            f"{st['wins']:2d}W/{st['losses']:2d}L  p={st['p']:.4f}  n={st['n']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, default=None, help="pilot on the first N cells")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--skip", nargs="*", default=[], help="selector families to skip")
    ap.add_argument("--only", nargs="*", default=None,
                    help="run ONLY these selector families. Legitimate because every "
                         "floor and null is drawn from a substream seeded by "
                         "(cell, rep, arena, size), so a single-family run produces "
                         "numbers identical to the same family inside a full sweep.")
    ap.add_argument("--out-suffix", default="",
                    help="append to the output dir name, so a partial run does not "
                         "clobber a completed sweep")
    ap.add_argument("--arenas", nargs="*", default=list(ARENAS), choices=list(ARENAS))
    ap.add_argument("--resume", action="store_true",
                    help="reload finished cells from splits_*.csv and skip them")
    args = ap.parse_args()
    args.skip = set(args.skip)

    out = S.outdir(OUT_DIR + args.out_suffix)
    # Load the reference CSVs HERE, before the Pool: an exception inside a Pool
    # initializer makes multiprocessing respawn the worker forever instead of failing,
    # so a missing exp12/exp13 splits.csv would hang rather than raise.
    load_exp12_splits(), load_exp13_splits()
    cells = S.load()
    S.validity_check(cells)
    derived_arm_gate({k: derive_cell(c) for k, c in cells.items()})

    keys = list(cells)[: args.cells] if args.cells else list(cells)
    print(f"\n{len(keys)} cells x {N_SPLITS} splits, arenas={args.arenas}, "
          f"{args.workers} workers, skip={sorted(args.skip) or 'none'}\n", flush=True)

    # Results are checkpointed AFTER EVERY CELL. This run is hours long and every number
    # in it lives in memory until the end otherwise — an interruption at 90% costs
    # everything, which is exactly the failure the project's incremental-save rule
    # exists to prevent. `--resume` reloads finished cells and skips them.
    long_path = os.path.join(out, "splits_long.csv")
    ref_path = os.path.join(out, "splits_ref.csv")
    long_rows, ref_rows = [], []
    if args.resume and os.path.exists(long_path) and os.path.exists(ref_path):
        long_rows = list(csv.DictReader(open(long_path, newline="")))
        ref_rows = list(csv.DictReader(open(ref_path, newline="")))
        done = {r["cell"] for r in ref_rows}
        keys = [k for k in keys if k not in done]
        print(f"  resuming: {len(done)} cells already done, {len(keys)} to go")
    else:
        for stale in (long_path, ref_path, os.path.join(out, "summary.json")):
            if os.path.exists(stale):
                os.remove(stale)
                print(f"  removed stale {os.path.basename(stale)}")

    t0 = time.time()

    def checkpoint():
        S.save_csv(long_path, long_rows, _all_fields(long_rows))
        S.save_csv(ref_path, ref_rows, _all_fields(ref_rows))

    payload = [(ck, args) for ck in keys]
    if keys and args.workers > 1:
        with Pool(args.workers, initializer=_init_worker) as pool:
            for i, (ck, lr, rr, dt) in enumerate(pool.imap_unordered(_work, payload), 1):
                long_rows += lr
                ref_rows += rr
                checkpoint()
                print(f"[{i:2d}/{len(keys)}] {S.plain_cell(ck)[:30]:30s} "
                      f"{len(lr):4d} rows  [{dt:.0f}s cell, "
                      f"{time.time()-t0:.0f}s total, checkpointed]", flush=True)
    elif keys:
        _init_worker()
        for i, p in enumerate(payload, 1):
            ck, lr, rr, dt = _work(p)
            long_rows += lr
            ref_rows += rr
            checkpoint()
            print(f"[{i:2d}/{len(keys)}] {S.plain_cell(ck)[:30]:30s} {len(lr):4d} rows "
                  f"[{dt:.0f}s, checkpointed]", flush=True)
    checkpoint()

    # Rows round-tripped through CSV come back as strings; the aggregation does float()
    # on every numeric field, so coerce once here rather than sprinkling casts.
    for r in long_rows + ref_rows:
        for k_, v_ in list(r.items()):
            if k_ in ("cell", "arena", "family", "variant", "error"):
                continue
            try:
                r[k_] = float(v_)
            except (TypeError, ValueError):
                pass

    # ---- reference bars, re-derived on these rows, not quoted ----------------------
    refs = []
    for ck in dict.fromkeys(r["cell"] for r in ref_rows):
        rs = [r for r in ref_rows if r["cell"] == ck]
        row = {"cell": ck}
        for key in rs[0]:
            if key in ("cell", "rep"):
                continue
            v = [float(r[key]) for r in rs if np.isfinite(float(r[key]))]
            row[key] = float(np.mean(v)) if v else float("nan")
        refs.append(row)

    if not refs:
        raise SystemExit("no split survived — nothing to summarise")
    summary = {"n_cells": len(refs), "n_splits": len(ref_rows),
               "arenas": list(args.arenas), "skipped": sorted(args.skip),
               "is_partial_run": bool(args.cells or args.skip
                                      or set(args.arenas) != set(ARENAS)),
               "primary_family": list(PRIMARY), "demoted": DEMOTED,
               "PROVENANCE": ("Pre-registered in this file's docstring before the run; "
                              "the primary family was cut from eight to five by the "
                              "2026-08-04 fidelity review, also before the run. Splits "
                              "are exp12's; exp13's arms asserted per split to "
                              f"{TOL_REPRO}.")}

    print("\n" + "=" * 92)
    print("THE BAR — re-derived on these rows, not quoted")
    summary["room"] = paired(refs, "auroc_random_prune", "auroc_greedy")
    print(fmt("the good set (the room)", summary["room"]))
    summary["floor_vs_deployed"] = paired(refs, "auroc_deployed", "auroc_random_prune")
    print(fmt("exp13 fixed-k floor vs deployed", summary["floor_vs_deployed"]))
    print(f"  expected: room {ROOM_PP:+.2f}pp, floor {FLOOR_PP:+.2f}pp")
    if "floor_keep_at_k" in refs[0]:
        summary["floor_machinery_check"] = paired(refs, "auroc_random_prune",
                                                  "floor_keep_at_k")
        print(fmt("NEW size-matched floor@k vs exp13's", summary["floor_machinery_check"])
              + "   <- must be ~0")

    agg = collapse_long(long_rows, refs)
    summary["arms"] = {}
    for arena in args.arenas:
        vs = sorted(v for (a, v) in agg if a == arena)
        prim, pv_floor, pv_dep = {}, {}, {}
        print(f"\n{'=' * 92}\nARENA '{arena}'")
        print("  contrast A = vs a random subset of ITS OWN size from the SAME "
              "population (is the rule doing anything?)")
        print("  contrast B = vs the DEPLOYED U-PCR rule on the full pool (should we "
              "swap the rule? <- the question)")
        print(f"\n  PRIMARY FAMILY — {len(PRIMARY)} published scoring functions "
              f"(our size rule on 4 of 5; see FIDELITY in the docstring):")
        absent = [v for v in PRIMARY if (arena, v) not in agg]
        summary.setdefault("absent_primaries", {})[arena] = absent
        for v in PRIMARY:
            if (arena, v) not in agg:
                print(f"  {v:34s} ABSENT — carried at p=1.0, family stays "
                      f"{len(PRIMARY)}")
                continue
            st = arm_stats(agg[(arena, v)])
            prim[v] = st
            pv_floor[v] = st["p"]
            pv_dep[v] = st["vs_deployed"]["p"]
            print(fmt(v, st) + f"  size={st['mean_size']:.1f}")
            print(fmt("      ^ vs DEPLOYED", st["vs_deployed"]))
        # Family size is the REGISTERED len(PRIMARY), never len(pv_*): an arm that went
        # missing (--skip, a variant never emitted, a NaN p) must not shrink the
        # multiplier and make the survivors look more significant than registered.
        hp_floor = holm(pv_floor, len(PRIMARY))
        hp_dep = holm(pv_dep, len(PRIMARY))
        print(f"\n  Holm-adjusted within this arena "
              f"(flat over the registered {len(PRIMARY)}):")
        for v in sorted(hp_floor, key=lambda x: hp_floor[x]):
            print(f"    {v:34s} vs floor {hp_floor[v]:.4f}   "
                  f"vs deployed {hp_dep.get(v, float('nan')):.4f}")

        sens = {v: arm_stats(agg[(arena, v)]) for v in vs if v not in PRIMARY}
        print("\n  SENSITIVITY / REFERENCE (not in the family, no Holm) — "
              "top 12 by vs-DEPLOYED:")
        for v, st in sorted(sens.items(),
                            key=lambda kv: -kv[1]["vs_deployed"]["mean_delta"])[:12]:
            tag = " [PRIOR-CARRYING]" if any(t in v.lower() for t in PRIOR_TOKENS) else ""
            print(fmt(v, st["vs_deployed"]) + f"  size={st['mean_size']:.1f}{tag}")
        print(f"    ... {len(sens)} sensitivity variants in total, all in summary.json")
        summary["arms"][arena] = {"primary": prim,
                                  "primary_holm_adjusted_p_vs_floor": hp_floor,
                                  "primary_holm_adjusted_p_vs_deployed": hp_dep,
                                  "sensitivity": sens}

    S.save_json(os.path.join(out, "summary.json"), summary)
    print(f"\nWrote -> {out}   [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
