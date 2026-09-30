"""Does the internal-layer depth field ADD to the eleven-channel token bank, at step level?

    python scripts/run_bank_plus_depth_fusion_v1.py

This is a NEW question, not a retry of a branch that was killed. Stage 1b killed *depth as a
standalone locator*: the best label-free depth column reached 31.56 mean gate-free SLA against
33.61 for `final_lens_H` and 39.89 for the production locator, and even a label-selected oracle
over 848 columns only reached 35.46. None of that speaks to whether depth ADDS to a bank that
already works. A channel can lose alone and still contribute.

TWO THINGS MEASURED BEFORE BUILDING THIS, because either could have closed the question
---------------------------------------------------------------------------------------
1. `final_lens_H` is NOT a usable addition to the bank: against `q15_H1` it scores Spearman
   0.9999 / Pearson 0.9999. They are the same statistic reached by two routes - the model's
   final-layer lens head, and the cached top-15 logprobs. So "bank + final_lens_H" would add a
   duplicate of column 1, the same trap as `lens_H.resid.35` in Stage 1b. Only the INTERNAL
   layers are candidates here, and the near-final residual band (34, 35) is excluded for the
   same reason it was excluded from the kill test.
2. The internal depth columns are NOT redundant with the bank. Regressing each of the 424
   eligible declared-orientation columns on the eleven channels: R^2 median 0.49, minimum 0.29,
   and 223 of 424 keep more than half their variance. Had depth been absorbed by the bank,
   fusion could not have helped and this script would not exist.

WHY THE EFFECTIVE-VIEW COUNT IS REPORTED BUT GIVEN NO AUTHORITY
---------------------------------------------------------------
Adding four depth columns moves the bank's participation ratio from 4.411 to 5.672. That is
NOT evidence the fusion will work, and it is recorded here so that it cannot later be read as
such. Step 415 is the precedent: the largest effective-signal gain the project ever measured
(+0.32) came with a quality LOSS of 0.73 pp. New variance is not new signal, and a
participation ratio knows nothing about the label.

ARMS
----
Every arm is fitted out-of-fold on five source-group folds and scored on the held-out fold.
Standardization is answer-local, which uses only an answer's own steps and therefore cannot
leak across folds.

    bank11              the eleven token-probability channels           equal | CONT L-SML
    bank11 + depth_pc4  plus one virtual per lens quantity, each the
                        first principal component over that quantity's
                        taps x internal layers, fitted on donor folds   equal | CONT L-SML
    bank11 + depth_nov4 plus the four columns LEAST explained by the
                        bank, chosen per fold on donor answers only     equal | CONT L-SML

Both depth constructions are label-free. `depth_pc4` follows the project's family-virtual
convention; `depth_nov4` targets orthogonality directly. They are reported side by side because
neither is obviously the right one and picking after seeing the answer would be selection.

Standing rows, not contrasts: `ct7_locator` (the production locator, 39.89) and `step_length`
(the documented length prior). The matched contrast is each augmented arm against its OWN
bank11 control at the same fitting scope and fusion rule.

PRE-REGISTERED KILL RULE
------------------------
If no augmented arm beats its matched bank11 control on per-cell gate-free SLA, with a paired
source-group interval excluding zero, in a majority of the eight ProcessBench cells, then the
depth field adds nothing to the locator and the locator question is closed - alone and in
combination.

MANDATORY ROWS, from what Stage 1b found
----------------------------------------
Every arm is also scored after within-answer rank-residualisation against step length. Stage 1b
measured `final_lens_H` falling 33.61 -> 16.79 and the depth arm 31.56 -> 16.99 under that
control, both to chance, while CT7 retained 20.09. An arm whose gain evaporates there has
rediscovered step length.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectral_utils.gate_free_sla import (  # noqa: E402
    argmax_predictions,
    chance_by_cell,
    first_error_step,
    group_codes,
    load_folds,
    load_roster,
    paired_intervals,
    pb_mean,
    shared_group_weights,
    sla_by_cell,
)
from spectral_utils.lsml_gate_locator_research import (  # noqa: E402
    FusionRecipe,
    answer_standardize,
    effective_rank,
    fit_fusion_weights,
)

TREES = Path(__file__).resolve().parents[2]
EVALUATION = TREES.parent / "results" / "localization_full_benchmark_v3" / "evaluation"
FOLDS = TREES.parent / "results" / "localization_source_group_audit_v1" / "FOLDS_V2.json"
TPF = TREES / "token-probability-fusion-v1" / "results"
BANK = TPF / "token_probability_fusion_v1" / "DERIVATIVE_CHANNELS.npz"
SPANS = TPF / "token_probability_fusion_v1" / "TOKEN_MATRICES.npz"
CT7 = TPF / "chosen_token_calibration_v1" / "CT7_DEV_SCORES.npz"
DEPTH = (TREES / "whitebox-layer-views-v1" / "results"
         / "whitebox_layer_views_localization_v1" / "STEP_LEVEL.npz")
OUT = Path(__file__).resolve().parents[1] / "results" / "bank_plus_depth_fusion_v1"

DRAWS = 10000
SEED = 20260922
N_DEPTH = 4
#: the readout orientation that means risk, per lens quantity - declared, never fitted
RISK = {"lens_H": "hi", "lens_kl_final": "hi", "lens_logp_tgt": "lo", "lens_logp_top1": "lo"}
#: residual-tap layers that restate final_lens_H rather than measure depth
NEAR_FINAL_FROM = 34


def parse(name: str) -> tuple[str, str, int, str]:
    quantity, tap, layer, readout = name.split(".")
    return quantity, tap, int(layer.split("_")[1]), readout


def eligible_depth(names, values) -> np.ndarray:
    """Declared-orientation, non-constant, internal-layer columns. The Stage-1b candidate set."""
    ok = np.array([
        parse(n)[3] == RISK[parse(n)[0]]
        and not (parse(n)[1] == "resid" and parse(n)[2] >= NEAR_FINAL_FROM)
        for n in names])
    return np.flatnonzero(ok & (values.std(axis=0) > 1e-12))


def depth_pc4(train_rows, depth, names, idx):
    """One virtual per lens quantity: its first PC over that quantity's taps x layers.

    Fitted on donor rows only. The sign of a PC is arbitrary, so it is fixed by requiring a
    non-negative mean loading - a deterministic rule that uses no labels, rather than the
    sign numpy happens to return.
    """
    out, meta = [], []
    for quantity in sorted(RISK):
        cols = np.array([j for j in idx if parse(names[j])[0] == quantity])
        if len(cols) < 2:
            continue
        block = depth[np.ix_(train_rows, cols)]
        block = block - block.mean(axis=0)
        _, _, vt = np.linalg.svd(block, full_matrices=False)
        loading = vt[0]
        if loading.mean() < 0:
            loading = -loading
        out.append((cols, loading))
        meta.append({"quantity": quantity, "n_columns": int(len(cols)),
                     "loading_absmax": float(np.abs(loading).max())})
    return out, meta


def depth_novel4(train_rows, depth, bank, idx, k=N_DEPTH):
    """The k depth columns least explained by the bank, by R^2 on donor rows. Label-free."""
    X = np.column_stack([bank[train_rows], np.ones(len(train_rows))])
    XtXi = np.linalg.pinv(X.T @ X)
    Xt = X.T
    r2 = np.empty(len(idx))
    for p, j in enumerate(idx):
        d = depth[train_rows, j]
        resid = d - X @ (XtXi @ (Xt @ d))
        r2[p] = 1.0 - resid.var() / max(d.var(), 1e-12)
    pick = idx[np.argsort(r2)[:k]]
    return pick, [{"column": int(j), "r2": float(r2[list(idx).index(j)])} for j in pick]


def run_arm(name, build_columns, bank, offsets, folds, mode, *, seed):
    """Out-of-fold fusion. `build_columns(train_rows)` returns the extra columns for that fold."""
    scores = np.full(bank.shape[0], np.nan)
    records = []
    for fold in np.unique(folds):
        test = np.flatnonzero(folds == fold)
        train = np.flatnonzero(folds != fold)
        train_rows = np.concatenate([np.arange(offsets[a], offsets[a + 1]) for a in train])
        test_rows = np.concatenate([np.arange(offsets[a], offsets[a + 1]) for a in test])
        extra, extra_meta = build_columns(train_rows)
        matrix = bank if extra is None else np.column_stack([bank, extra])
        members = tuple(f"c{i}" for i in range(matrix.shape[1]))
        recipe = FusionRecipe(name=name, members=members, mode=mode, anchor=0)
        weight, meta = fit_fusion_weights(matrix[train_rows], recipe, seed=seed)
        scores[test_rows] = matrix[test_rows] @ weight
        records.append({
            "fold": int(fold), "columns": int(matrix.shape[1]),
            "effective_views_train": float(effective_rank(matrix[train_rows])),
            "weights": [float(w) for w in weight],
            "extra": extra_meta,
            "meta": {k: v for k, v in meta.items()
                     if isinstance(v, (str, int, float, bool, list))},
        })
        print(f"  [{name}] fold {fold}: {matrix.shape[1]} cols, "
              f"eff views {records[-1]['effective_views_train']:.3f}", flush=True)
    if not np.isfinite(scores).all():
        raise SystemExit(f"{name}: unscored steps remain")
    return scores, records


def main() -> int:
    t0 = time.time()
    for path in (EVALUATION / "JOINED.json", FOLDS, BANK, DEPTH):
        if not path.exists():
            print(f"missing input: {path}")
            return 2
    OUT.mkdir(parents=True, exist_ok=True)

    print("[load] roster, folds, bank, depth")
    roster = load_roster(EVALUATION)
    offsets, cells = roster["offsets"], roster["cells"]
    truth = first_error_step(roster["labels"], offsets, roster["target"], cells)
    codes, ng = group_codes(roster["group_id"])
    folds = load_folds(FOLDS, roster["group_id"])
    err = truth >= 0
    with np.load(BANK, allow_pickle=False) as z:
        bank = answer_standardize(z["level"].astype(np.float64), offsets)
        channels = list(z["channels"].astype(str))
    with np.load(DEPTH, allow_pickle=False) as z:
        names = z["names"].astype(str)
        depth = answer_standardize(z["step_views"].astype(np.float64), offsets)
    idx = eligible_depth(names, depth)
    pb_panels = [p for p in sorted(set(cells.tolist())) if p.startswith("pb_")]
    chance = chance_by_cell(truth, offsets, cells)
    print(f"  {len(truth)} answers | {int(err.sum())} erroneous | {bank.shape[0]} steps | "
          f"{ng} groups | {len(idx)} eligible depth columns")
    print(f"  bank effective views {effective_rank(bank):.3f} of {bank.shape[1]}")

    weights_matrix = shared_group_weights(ng, DRAWS, SEED + 1)
    arms, fits = {}, {}

    for mode in ("equal", "continuous"):
        arms[f"bank11_{mode}"], fits[f"bank11_{mode}"] = run_arm(
            f"bank11_{mode}", lambda tr: (None, None), bank, offsets, folds, mode, seed=SEED)

        def build_pc(train_rows):
            blocks, meta = depth_pc4(train_rows, depth, names, idx)
            return np.column_stack([depth[:, c] @ l for c, l in blocks]), meta
        arms[f"bank11_pc4_{mode}"], fits[f"bank11_pc4_{mode}"] = run_arm(
            f"bank11_pc4_{mode}", build_pc, bank, offsets, folds, mode, seed=SEED)

        def build_nov(train_rows):
            pick, meta = depth_novel4(train_rows, depth, bank, idx)
            return depth[:, pick], meta
        arms[f"bank11_nov4_{mode}"], fits[f"bank11_nov4_{mode}"] = run_arm(
            f"bank11_nov4_{mode}", build_nov, bank, offsets, folds, mode, seed=SEED)

    if CT7.exists():
        arms["ct7_locator"] = np.load(CT7, allow_pickle=False)["step_scores"].astype(np.float64)
    if SPANS.exists():
        with np.load(SPANS, allow_pickle=False) as z:
            spans = z["step_spans"].astype(np.int64)
        arms["step_length"] = (spans[:, 1] - spans[:, 0]).astype(np.float64)

    print(f"[score] gate-free SLA, {DRAWS} shared paired draws")
    results = {k: sla_by_cell(argmax_predictions(v, offsets), truth, codes, cells, ng,
                              weights=weights_matrix) for k, v in arms.items()}

    order = [k for k in ("step_length", "ct7_locator",
                         "bank11_equal", "bank11_pc4_equal", "bank11_nov4_equal",
                         "bank11_continuous", "bank11_pc4_continuous",
                         "bank11_nov4_continuous") if k in results]
    print("\n=== gate-free SLA, tolerance 0, erroneous answers only ===")
    print(f"{'cell':<22}{'chance':>8}" + "".join(f"{k[:13]:>15}" for k in order))
    for panel in pb_panels + [p for p in sorted(set(cells.tolist())) if p not in pb_panels]:
        print(f"{panel:<22}{100 * chance[panel]:>8.2f}"
              + "".join(f"{100 * results[k][panel]['point']:>15.2f}" for k in order))
    print(f"{'mean over 8 PB cells':<22}"
          f"{100 * float(np.mean([chance[p] for p in pb_panels])):>8.2f}"
          + "".join(f"{100 * pb_mean(results[k]):>15.2f}" for k in order))

    print("\n=== KILL RULE: each augmented arm vs its OWN bank11 control ===")
    print(f"{'contrast':<34}{'delta':>9}   verdict")
    verdicts = {}
    for mode in ("equal", "continuous"):
        for kind in ("pc4", "nov4"):
            arm, ctrl = f"bank11_{kind}_{mode}", f"bank11_{mode}"
            iv = paired_intervals(results[arm], results[ctrl])
            wins = sum(1 for p in pb_panels if iv[p]["low"] > 0)
            loses = sum(1 for p in pb_panels if iv[p]["high"] < 0)
            delta = float(np.mean([iv[p]["point"] for p in pb_panels]))
            verdicts[f"{arm}_vs_{ctrl}"] = {"pb_mean_delta": delta, "wins": wins,
                                            "loses": loses, "per_cell": iv}
            print(f"{arm + ' - ' + ctrl:<34}{100 * delta:>+9.2f}   "
                  f"beats in {wins}/8, loses in {loses}/8")
    majority = any(v["wins"] >= 5 for v in verdicts.values())
    print(f"\n-> {'DEPTH ADDS TO THE BANK' if majority else 'DEPTH ADDS NOTHING - '
          'the locator question closes, alone and in combination'}")

    payload = {
        "population": {"answers": int(len(truth)), "erroneous": int(err.sum()),
                       "steps": int(bank.shape[0]), "groups": int(ng)},
        "channels": channels, "eligible_depth_columns": int(len(idx)),
        "effective_views_bank": float(effective_rank(bank)),
        "chance_by_cell": chance,
        "sla_by_cell": {k: {p: {kk: vv for kk, vv in e.items() if kk != "draws"}
                            for p, e in v.items()} for k, v in results.items()},
        "pb_mean": {k: pb_mean(v) for k, v in results.items()},
        "kill_rule": verdicts, "depth_adds": bool(majority),
        "fits": fits, "draws": DRAWS, "seconds": round(time.time() - t0, 1),
    }
    (OUT / "RESULTS.json").write_text(json.dumps(payload, indent=1, default=float),
                                      encoding="utf-8")
    np.savez_compressed(OUT / "STEP_SCORES.npz", **arms)
    print(f"\n{time.time() - t0:.0f}s   wrote {OUT / 'RESULTS.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
