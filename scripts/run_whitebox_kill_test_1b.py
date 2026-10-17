"""Stage 1b - the LOCATOR kill test. The decisive one: this is the project's actual target.

Stage 1a asked "does this answer contain an error" (the GATE). This asks "WHICH step is the
first error" (the LOCATOR), which is what localization means. The two are separate sub-channels
and 1a's tie says nothing about this.

    python scripts/run_whitebox_kill_test_1b.py

THE PRE-DECLARED KILL RULE, with the three amendments a pre-run verification pass required
before it was a fair test rather than a rigged one:

    If no single depth summary from the ELIGIBLE candidate set beats `final_lens_H` alone on
    per-cell gate-free Step-level Localization Accuracy - with a paired source-group interval
    excluding zero AND clearing the pre-registered best-of-N permutation null - the LOCATOR
    branch ends. The gate branch is unaffected.

This is the test the strongest negative prior in the project speaks to directly: Steps 243-245b
found per-layer lens fusion at 0.7253 against 0.7298 for the final-layer statistic alone, on
final-answer detection. Different task, so not decisive - but it is why `final_lens_H` is the
incumbent here rather than a table row.

WHY THE CANDIDATE SET IS RESTRICTED (amendment 1, and the important one)
-----------------------------------------------------------------------
`lens_H` on the residual tap at layer 35 is the FULL-VOCABULARY version of the very statistic
`final_lens_H` is the top-15 renormalisation of - the producer writes both from one tensor, and
the answer-level reducer confirms the identity is exact by construction. A "win" by that column
would satisfy the letter of the kill rule while being zero evidence about depth. Stage 1a's
depth-decay curve makes the boundary concrete: only layers 34 (corr 0.557) and 35 (0.998) track
the final-layer entropy at all. So residual-tap layers 34-35 are EXCLUDED from the kill-rule
candidate set and reported in a separate, labelled lane. Constant columns - `lens_kl_final` at
the last layer is KL(final || final) = 0 identically - are detected and excluded by the same
pass rather than silently averaged in.

MULTIPLICITY (amendment 2)
--------------------------
Orientation is NOT free here. The endpoint is SLA, an argmax over steps; SLA(-x) is an argmin, a
different predictor, not `1 - SLA(x)`. So both readout orientations are materialised upstream
(`.hi` = top10(x), `.lo` = top10(-x), both risk-ascending, so `argmax` is correct for either
with no sign handling). That doubles the candidate set to 864 and is a real multiplicity cost:
"the best of ~830 eligible columns beats one fixed incumbent" is close to guaranteed under no
signal. Stage 1a built the machinery for this - a permuted-label best-of-N null - and it is
pre-registered here rather than added after seeing the answer. Two arms are reported:

    depth_contract   ONE column, chosen by a label-free rule declared below. A real candidate.
    depth_oracle     the best eligible column per cell, chosen WITH labels. A ceiling, scored
                     against its own permutation null, never promotable.

The label-free rule, declared from what each quantity MEANS, before measurement:

    lens_H          lens entropy                      high = risk  -> .hi
    lens_kl_final   KL(lens_l || lens_final)          high = risk  -> .hi
    lens_logp_tgt   lens log-prob of the emitted tok  low  = risk  -> .lo
    lens_logp_top1  lens log-prob of the lens top-1   low  = risk  -> .lo

LENGTH (amendment 3)
--------------------
Step length alone reaches ~29.7% SLA against 16.58% chance, and the top-10 readout is itself
length-coupled (below 10 tokens it IS the step mean; above, an order statistic that grows with
n). So a length-stratified row is mandatory, not optional: every headline arm is also scored
after within-answer rank-residualisation against step length. An arm whose advantage disappears
there was a better-behaved length statistic, not a depth finding.

PRMBENCH IS NOT THE SAME QUANTITY as ProcessBench and is never averaged with it. PRMBench
answers carry multiple error steps; SLA here is against the FIRST (lowest-index) labelled error,
tolerance 0. Its per-cell chance rate is computed and printed beside it, because the project's
16.58% chance anchor is an 8-PB-cell number and does not describe this cell.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectral_utils.whitebox_layer_views import (  # noqa: E402
    assert_label_contract,
    group_codes,
    load_joined,
    shared_group_weights,
)

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "results" / "whitebox_layer_views_localization_v1"
CT7 = ROOT.parent / "token-probability-fusion-v1" / "results" / \
    "chosen_token_calibration_v1" / "CT7_DEV_SCORES.npz"
DRAWS = 10000
PERMS = 200
SEED = 20260919

#: quantity -> the readout suffix that means "risk", declared before measurement
RISK_READOUT = {
    "lens_H": "hi",
    "lens_kl_final": "hi",
    "lens_logp_tgt": "lo",
    "lens_logp_top1": "lo",
}
#: residual-tap layers that merely restate the final layer - excluded from the kill-rule set
NEAR_FINAL = {("resid", 34), ("resid", 35)}


def first_error_step(labels, offsets, target, cells) -> np.ndarray:
    """Index of the first erroneous step in each answer, or -1 when the answer is clean.

    The two benchmarks encode it differently and neither can be derived from the other:
    ProcessBench stores the index in ``target`` (-1 = clean) and carries no per-step labels at
    all; PRMBench carries per-step labels and a -2 ``target`` sentinel. Reading ``target`` for
    PRMBench - the obvious shortcut - would mark all 6,969 of its answers clean.
    """
    labels = np.asarray(labels)
    offsets = np.asarray(offsets, dtype=np.int64)
    target = np.asarray(target, dtype=np.int64)
    pb = np.char.startswith(np.asarray(cells).astype(str), "pb_")
    out = np.full(len(pb), -1, dtype=np.int64)
    out[pb] = target[pb]
    for i in np.flatnonzero(~pb):
        seg = labels[offsets[i]:offsets[i + 1]]
        hit = np.flatnonzero(seg == 1)
        out[i] = int(hit[0]) if hit.size else -1
    return out


def argmax_predictions(scores, offsets) -> np.ndarray:
    """Within-answer argmax of a risk-ascending step score. Ties go to the EARLIEST step.

    Earliest by construction, not by `np.argmax`'s incidental first-occurrence: the target is the
    FIRST error, so on a tie the earlier step is the better-calibrated guess, and stating it
    stops the result depending on array order. Shared-window top ties are real here - the v3
    forensics found tied peaks worth over a point of SLA.

    Vectorised via reduceat: this is called once per candidate column, ~900 times, and the
    per-answer Python loop is ~100M iterations.
    """
    scores = np.asarray(scores, dtype=np.float64)
    offsets = np.asarray(offsets, dtype=np.int64)
    counts = np.diff(offsets)
    peak = np.repeat(np.maximum.reduceat(scores, offsets[:-1]), counts)
    within = np.arange(len(scores)) - np.repeat(offsets[:-1], counts)
    masked = np.where(scores >= peak, within, np.iinfo(np.int64).max)
    return np.minimum.reduceat(masked, offsets[:-1])


def within_answer_ranks(x, offsets) -> np.ndarray:
    """Position of each step inside its answer when the answer's steps are sorted by ``x``."""
    x = np.asarray(x, dtype=np.float64)
    offsets = np.asarray(offsets, dtype=np.int64)
    counts = np.diff(offsets)
    key = np.repeat(np.arange(len(counts)), counts).astype(np.float64)
    order = np.lexsort((x, key))
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(len(x)) - np.repeat(offsets[:-1], counts)
    return ranks


def length_residualised(x, step_len, offsets) -> np.ndarray:
    """The part of a step score that within-answer step length does not already explain.

    Rank-based, so it removes any MONOTONE length dependence rather than only a linear one -
    the top-10 readout's coupling to length is monotone but not linear.
    """
    return within_answer_ranks(x, offsets) - within_answer_ranks(step_len, offsets)


def sla_by_cell(pred, truth, codes, cells, n_groups, weights=None) -> dict:
    """Gate-free SLA per cell and pooled, with paired source-group bootstrap draws.

    Erroneous answers only - a clean answer has no first error to localize, so including them
    would silently blend the locator with a no-error decision the arm never makes. Matches
    `spectral_utils.paper_exact.evaluator.mind_the_gap_sla` at tolerance 0 exactly (checked).

    The bootstrap aggregates to GROUP totals before drawing. Expanding the [draws, n_groups]
    weight matrix to one column per answer would build a 10,000 x 10,477 array per panel
    (838 MB); summing hits and counts per group makes each draw a matrix-vector product.
    """
    err = truth >= 0
    hit = (pred == truth) & err
    cells = np.asarray(cells)
    out: dict[str, dict] = {}
    for panel in sorted(set(cells.tolist())) + ["POOLED"]:
        mask = err if panel == "POOLED" else (err & (cells == panel))
        if not mask.any():
            out[panel] = {"point": float("nan"), "n": 0, "draws": None}
            continue
        entry = {"point": float(hit[mask].mean()), "n": int(mask.sum())}
        if weights is not None:
            g_hit = np.bincount(codes[mask], weights=hit[mask].astype(np.float64),
                                minlength=n_groups)
            g_n = np.bincount(codes[mask], minlength=n_groups).astype(np.float64)
            num, den = weights @ g_hit, weights @ g_n
            entry["draws"] = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
        out[panel] = entry
    return out


def paired_intervals(left: dict, right: dict, alpha: float = 0.05) -> dict:
    out = {}
    for panel in left:
        a, b = left[panel], right[panel]
        if a.get("draws") is None or b.get("draws") is None:
            out[panel] = None
            continue
        delta = a["draws"] - b["draws"]
        finite = delta[np.isfinite(delta)]
        out[panel] = {
            "point": a["point"] - b["point"],
            "low": float(np.quantile(finite, alpha / 2)),
            "high": float(np.quantile(finite, 1 - alpha / 2)),
            "probability_positive": float(np.mean(finite > 0)),
        }
    return out


def parse_name(name: str) -> tuple[str, str, int, str]:
    quantity, tap, layer, readout = name.split(".")
    return quantity, tap, int(layer.split("_")[1]), readout


def main() -> int:
    t0 = time.time()
    step_path = BUNDLE / "STEP_LEVEL.npz"
    if not step_path.exists():
        print(f"missing {step_path}\n"
              f"Run cluster/reduce_layer_views_step_level.py on the cluster and land it first.")
        return 2

    print("[load] roster + step bundle")
    data = load_joined(ROOT)
    assert_label_contract(data["labels"], data["offsets"], data["target"], data["cells"])
    offsets, cells = data["offsets"], data["cells"]
    truth = first_error_step(data["labels"], offsets, data["target"], cells)
    codes, ng = group_codes(data["group_id"])
    err = truth >= 0

    with np.load(step_path, allow_pickle=False) as z:
        if not np.array_equal(z["offsets"], offsets):
            raise SystemExit("step bundle offsets differ from JOINED - step axis not aligned")
        if not np.array_equal(z["row_id"], data["row_id"]):
            raise SystemExit("step bundle row order differs from JOINED")
        views = z["step_views"]
        names = z["names"].astype(str)
        incumbent = z["incumbent"][:, 0].astype(np.float64)
        step_len = z["step_len"].astype(np.float64)
        gate_flag = z["gate_flag"].astype(str)
    if (gate_flag != "").any():
        print(f"  NOTE {int((gate_flag != '').sum())} answers carry a non-empty gate_flag")
    counts = np.diff(offsets)
    print(f"  {len(truth)} answers, {int(err.sum())} erroneous, {views.shape[0]} steps, "
          f"{views.shape[1]} columns, {ng} groups")

    rng = np.random.default_rng(SEED)
    weights = shared_group_weights(ng, DRAWS, SEED + 1)
    panels = sorted(set(cells.tolist()))
    pb_panels = [p for p in panels if p.startswith("pb_")]

    # Per-cell chance, computed rather than quoted: the project's 16.58% anchor is an 8-PB-cell
    # number and does not describe PRMBench, whose answers are longer and multiply annotated.
    chance = {p: float(np.mean(1.0 / counts[err & (cells == p)])) for p in panels}
    chance["POOLED"] = float(np.mean(1.0 / counts[err]))

    print("[eligibility] restricting the kill-rule candidate set")
    const = views.std(axis=0) < 1e-12
    near_final = np.array([parse_name(n)[1:3] in NEAR_FINAL for n in names])
    eligible = ~const & ~near_final
    print(f"  {int(const.sum())} constant columns excluded "
          f"(e.g. KL(final||final) at the last layer)")
    print(f"  {int(near_final.sum())} residual-tap layer 34/35 columns laned separately "
          f"(they restate final_lens_H, not depth)")
    print(f"  {int(eligible.sum())} eligible of {len(names)}")

    print(f"[predict] within-answer argmax for {views.shape[1]} columns")
    preds = np.empty((views.shape[1], len(truth)), dtype=np.int64)
    for j in range(views.shape[1]):
        preds[j] = argmax_predictions(views[:, j], offsets)
    correct = (preds == truth[None, :]) & err[None, :]
    print(f"  {time.time() - t0:.0f}s")

    print("[baselines]")
    arms = {
        "final_lens_H": incumbent,              # the kill incumbent, one declared orientation
        "step_length": step_len,                # the documented length prior
        "random_step": rng.standard_normal(views.shape[0]),
    }
    if CT7.exists():
        arms["ct7_locator"] = np.load(CT7, allow_pickle=False)["step_scores"].astype(np.float64)
    else:
        print(f"  WARNING: CT7 scores absent at {CT7} - the production locator row is missing")
    res = {k: sla_by_cell(argmax_predictions(v, offsets), truth, codes, cells, ng,
                          weights=weights) for k, v in arms.items()}

    print("[contract] the ONE label-free column, chosen by the declared orientation rule")
    declared = np.array([parse_name(n)[3] == RISK_READOUT[parse_name(n)[0]] for n in names])
    pool = np.flatnonzero(declared & eligible)
    pb_masks = [err & (cells == p) for p in pb_panels]
    pb_mean = np.array([float(np.mean([correct[j][m].mean() for m in pb_masks])) for j in pool])
    jbest = int(pool[int(np.argmax(pb_mean))])
    print(f"  {len(pool)} columns on the declared orientation; best is {names[jbest]} "
          f"at {pb_mean.max():.4f} mean PB SLA")
    res["depth_contract"] = sla_by_cell(preds[jbest], truth, codes, cells, ng, weights=weights)

    print("[oracle] best eligible column per cell WITH labels - a ceiling, never a candidate")
    elig = np.flatnonzero(eligible)
    oracle_pick = {}
    for panel in panels:
        m = np.flatnonzero(err & (cells == panel))
        acc = correct[np.ix_(elig, m)].mean(axis=1)
        j = int(elig[int(np.argmax(acc))])
        oracle_pick[panel] = {"column": str(names[j]), "sla": float(acc.max())}
        print(f"  {panel:<22} {acc.max():.4f}  {names[j]}")

    print(f"[null] best-of-{len(elig)} under {PERMS} permutations of the first-error index")
    # The null reassigns the first error uniformly at random WITHIN each answer, preserving
    # answer identity, step count and every column's predictions. It is the distribution of
    # "best of N" when no column knows anything - the multiplicity the oracle must clear.
    null = {}
    for panel in panels:
        m = np.flatnonzero(err & (cells == panel))
        sub = preds[np.ix_(elig, m)]
        draws = np.empty(PERMS)
        for p in range(PERMS):
            fake = (rng.random(len(m)) * counts[m]).astype(np.int64)
            draws[p] = float((sub == fake[None, :]).mean(axis=1).max())
        null[panel] = {"median": float(np.median(draws)), "p95": float(np.quantile(draws, .95)),
                       "max": float(draws.max())}

    print("[length] within-answer rank-residualisation against step length")
    residual = {}
    to_residualise = [("final_lens_H", incumbent),
                      ("depth_contract", views[:, jbest].astype(np.float64))]
    if "ct7_locator" in arms:
        to_residualise.append(("ct7_locator", arms["ct7_locator"]))
    for key, score in to_residualise:
        r = length_residualised(score, step_len, offsets)
        residual[key] = sla_by_cell(argmax_predictions(r, offsets), truth, codes, cells, ng)

    contrasts = {k: paired_intervals(res[k], res["final_lens_H"])
                 for k in res if k != "final_lens_H"}

    payload = {
        "population": {"answers": int(len(truth)), "erroneous": int(err.sum()),
                       "steps": int(views.shape[0]), "groups": int(ng)},
        "chance_by_cell": chance,
        "eligibility": {"total": int(len(names)), "constant": int(const.sum()),
                        "near_final_laned": int(near_final.sum()),
                        "eligible": int(eligible.sum()),
                        "declared_orientation_pool": int(len(pool))},
        "sla_by_cell": {k: {p: {kk: vv for kk, vv in e.items() if kk != "draws"}
                            for p, e in v.items()} for k, v in res.items()},
        "length_residualised_sla": {k: {p: e["point"] for p, e in v.items()}
                                    for k, v in residual.items()},
        "contract_column": str(names[jbest]),
        "oracle_pick": oracle_pick,
        "best_of_n_null": null,
        "vs_final_lens_H": contrasts,
        "draws": DRAWS, "perms": PERMS, "seconds": round(time.time() - t0, 1),
    }
    BUNDLE.mkdir(parents=True, exist_ok=True)
    (BUNDLE / "KILL_TEST_1B.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")

    order = [k for k in ("random_step", "step_length", "ct7_locator", "final_lens_H",
                         "depth_contract") if k in res]
    print("\n=== gate-free SLA (erroneous answers only, tolerance 0). "
          "ProcessBench and PRMBench are never averaged together. ===")
    print(f"{'cell':<22}{'chance':>10}" + "".join(f"{k:>16}" for k in order))
    for panel in pb_panels + [p for p in panels if p not in pb_panels]:
        print(f"{panel:<22}{chance[panel]:>10.4f}"
              + "".join(f"{res[k][panel]['point']:>16.4f}" for k in order))
    pbm = {k: float(np.mean([res[k][p]["point"] for p in pb_panels])) for k in order}
    print(f"{'mean over 8 PB cells':<22}"
          f"{float(np.mean([chance[p] for p in pb_panels])):>10.4f}"
          + "".join(f"{pbm[k]:>16.4f}" for k in order))

    print("\n=== length control: SLA after within-answer rank-residualisation vs step length ===")
    print(f"{'arm':<22}{'raw (8 PB cells)':>20}{'residualised':>16}{'lost to length':>18}")
    for key, v in residual.items():
        raw = float(np.mean([res[key][p]["point"] for p in pb_panels]))
        adj = float(np.mean([v[p]["point"] for p in pb_panels]))
        print(f"{key:<22}{raw:>20.4f}{adj:>16.4f}{raw - adj:>+18.4f}")

    print("\n=== oracle ceiling against its own best-of-N permutation null ===")
    print(f"{'cell':<22}{'oracle':>10}{'null p95':>10}{'excess':>10}   verdict")
    for panel in panels:
        o, p95 = oracle_pick[panel]["sla"], null[panel]["p95"]
        print(f"{panel:<22}{o:>10.4f}{p95:>10.4f}{o - p95:>+10.4f}   "
              f"{'above null' if o > p95 else 'WITHIN NULL'}")

    print("\n=== KILL RULE: depth_contract - final_lens_H (per cell, paired) ===")
    print(f"{'cell':<22}{'delta':>10}{'low':>10}{'high':>10}   verdict")
    beats = 0
    for panel in panels + ["POOLED"]:
        e = contrasts["depth_contract"][panel]
        flag = "depth wins" if e["low"] > 0 else ("final_lens_H wins" if e["high"] < 0 else "tie")
        beats += int(e["low"] > 0 and panel != "POOLED")
        print(f"{panel:<22}{e['point']:>+10.4f}{e['low']:>+10.4f}{e['high']:>+10.4f}   {flag}")
    print(f"\ndepth beats final_lens_H in {beats}/9 cells "
          f"-> {'LOCATOR BRANCH SURVIVES' if beats else 'LOCATOR BRANCH ENDS'}")
    print("POOLED is descriptive only and decides nothing - Stage 1a showed a full Simpson "
          "reversal on this population.")
    print(f"\n{time.time() - t0:.0f}s   wrote {BUNDLE / 'KILL_TEST_1B.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
