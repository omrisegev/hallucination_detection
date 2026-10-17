"""Offline diagnostics on immutable exported N16 corrected+mask1 risk scores.

No model fitting, sign selection, threshold tuning, or authentic holdout inference.
Five equal relative-position bins are fixed before inspecting the diagnostics.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

METHODS = ("LSML_CONT", "IU_PCR", "ALL_EQ", "DS_FILTER_EQ", "STEP_INDEX")
V10_COMMIT = "53ef62e36e80b90de30b83d3fee33e1fb85b82c3"
RESEARCH_COMMIT = "080877eb9a8369341d647f4115a3b475c040b46e"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pair_auc(scores, error):
    bad, good = scores[error], scores[~error]
    if not len(bad) or not len(good):
        return None
    diff = bad[:, None] - good[None, :]
    return float((np.count_nonzero(diff > 0) + .5 * np.count_nonzero(diff == 0)) / diff.size)


def rank_auc(scores, error):
    n_bad, n_good = int(error.sum()), int((~error).sum())
    if not n_bad or not n_good:
        return None
    ranks = rankdata(scores, method="average")
    return float((ranks[error].sum() - n_bad * (n_bad + 1) / 2) / (n_bad * n_good))


def mean(values):
    return float(np.mean(values)) if values else None


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows):
    eligible = [r for r in rows if r["within_auc"] is not None]
    matched = [r for r in rows if r["same_bin_auc"] is not None]
    errors = [r for r in rows if r["n_error"] > 0]
    hits = sum(r["top1_hit"] for r in errors)
    return dict(n_answers=len(rows), n_error_answers=len(errors), top1_hits=hits,
                top1_on_error=hits / len(errors) if errors else None,
                n_auc_eligible=len(eligible), within_auc=mean([r["within_auc"] for r in eligible]),
                n_same_bin_eligible=len(matched),
                same_bin_auc=mean([r["same_bin_auc"] for r in matched]),
                unrestricted_auc_on_same_bin_eligible=mean([r["within_auc"] for r in matched]),
                n_same_bin_pairs=sum(r["same_bin_pairs"] for r in matched),
                same_bin_pair_pooled_auc=(sum(r["same_bin_wins"] for r in matched) /
                                          sum(r["same_bin_pairs"] for r in matched)) if matched else None,
                n_top1_step0=sum(r["top1_position"] == 0 for r in rows),
                top1_step0_proportion=sum(r["top1_position"] == 0 for r in rows) / len(rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--pinned-run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    source = args.package_root / "sources"
    bank_path = source / "prmscore_inputs/bank_N16.npz"
    meta_path = source / "prmscore_inputs/prmbench_meta.json"
    comparison_path = source / "PRMBENCH_COMPARISON.csv"
    before = {str(p): sha(p) for p in (bank_path, meta_path, comparison_path)}
    pinned_text = args.pinned_run.read_text()
    assert "fill(R['STEP_INDEX'], i, np.arange(NS[i]).astype(float))" in pinned_text
    assert "Step index (later = riskier)" in pinned_text
    bank = np.load(bank_path, allow_pickle=False)
    metadata = json.loads(meta_path.read_text())["rows"]
    oob_rows = [r["idx"] for r in metadata if any(e < 1 or e > r["n_steps"] for e in r["error_steps"])]
    oob_annotations = sum(e < 1 or e > r["n_steps"] for r in metadata for e in r["error_steps"])
    meta = {r["idx"]: r for r in metadata}
    ids, offsets = bank["idx"].astype(str), bank["offsets"]
    assert len(meta) == len(metadata) == len(ids) == len(set(ids)) == 6969
    assert set(ids) == set(meta) and offsets[0] == 0
    assert len(offsets) == len(ids) + 1 and np.all(np.diff(offsets) > 0)
    flat_n = int(offsets[-1])
    assert flat_n == 94203
    scores = {m: bank["S__" + m] for m in METHODS if m != "STEP_INDEX"}
    assert all(v.shape == (flat_n,) and np.isfinite(v).all() for v in scores.values())
    all_rows, bins = [], []
    max_auc_difference, n_exact_epsilon_differences = 0., 0
    for method in METHODS:
        for idx, start, stop in zip(ids, offsets[:-1], offsets[1:]):
            row = meta[idx]
            n = int(stop - start)
            assert row["n_steps"] == n
            raw_errors = row["error_steps"]
            assert all(isinstance(e, int) for e in raw_errors)
            # Match canonical step label membership; out-of-range annotations do
            # not label any existing step. Record rather than silently repair them.
            error = np.isin(np.arange(1, n + 1), raw_errors)
            v = np.arange(n, dtype=float) if method == "STEP_INDEX" else scores[method][start:stop]
            position_bins = np.floor(5 * (np.arange(n) + .5) / n).astype(int)
            assert np.all((position_bins >= 0) & (position_bins < 5))
            p = int(np.argmax(v))
            stable = int(np.argsort(-v, kind="stable")[0])
            assert p == stable
            epsilon_p = int(np.flatnonzero(v >= v.max() - 8 * np.finfo(float).eps)[0])
            n_exact_epsilon_differences += int(p != epsilon_p)
            full_auc = pair_auc(v, error)
            rank_full = rank_auc(v, error)
            if full_auc is not None:
                max_auc_difference = max(max_auc_difference, abs(full_auc - rank_full))
                assert abs(full_auc - rank_full) < 1e-12
            same_pairs, same_wins = 0, 0.
            for b in range(5):
                selected = position_bins == b
                bad, good = v[selected & error], v[selected & ~error]
                pairs = len(bad) * len(good)
                if not pairs:
                    continue
                d = bad[:, None] - good[None, :]
                wins = float(np.count_nonzero(d > 0) + .5 * np.count_nonzero(d == 0))
                bin_auc = wins / pairs
                assert abs(bin_auc - rank_auc(v[selected], error[selected])) < 1e-12
                same_pairs += pairs
                same_wins += wins
                bins.append(dict(method=method, classification=row["classification"], bin=b,
                                 idx=idx, auc=bin_auc, n_pairs=pairs, wins=wins))
            all_rows.append(dict(method=method, idx=idx, classification=row["classification"],
                                 category=row["category"], n_steps=n, n_error=int(error.sum()),
                                 step0_is_error=bool(error[0]), top1_position=p,
                                 top1_epsilon_position=epsilon_p, top1_hit=int(error[p]),
                                 within_auc=full_auc,
                                 same_bin_auc=same_wins / same_pairs if same_pairs else None,
                                 same_bin_pairs=same_pairs, same_bin_wins=same_wins))
    summary = {m: aggregate([r for r in all_rows if r["method"] == m]) for m in METHODS}
    assert all(r["n_error_answers"] == 6035 and r["n_auc_eligible"] == 6030 for r in summary.values())
    assert n_exact_epsilon_differences == 0, "Review canonical tolerance before comparison"
    historical = list(csv.DictReader(comparison_path.open()))
    references = []
    for m in METHODS:
        table = "cont_baselines" if m in ("LSML_CONT", "IU_PCR") else "prmbench_panel_b"
        metric = "prmb_top1_on_error" if table == "cont_baselines" else "top1_on_error"
        found = [r for r in historical if r["bank"] == "N16" and r["scorer"] == m and
                 r["table"] == table and r["metric"] == metric and not r["group"]]
        if m == "STEP_INDEX":
            continue
        assert len(found) == 1, (m, found)
        h = found[0]
        assert abs(summary[m]["top1_on_error"] - float(h["thinkfix_mask1"])) < 1e-12
        references.append(dict(method=m, evidence="independently recomputed saved scores",
                               old=float(h["old_replay"]), prefix_corrected=float(h["thinkfix"]),
                               corrected_mask1=float(h["thinkfix_mask1"]),
                               recomputed=summary[m]["top1_on_error"],
                               delta_vs_old=summary[m]["top1_on_error"] - float(h["old_replay"])))
    step_reference = [h for h in historical if h["bank"] == "-" and h["table"] == "prmbench_panel_b" and
                      h["scorer"] == "Step index (later = riskier)" and h["metric"] == "within_auc"]
    assert len(step_reference) == 1
    assert abs(float(step_reference[0]["thinkfix_mask1"]) - summary["STEP_INDEX"]["within_auc"]) < 1e-12
    for h in historical:
        if (not h["group"] and
            ((h["bank"] == "N16" and h["table"] == "lr_reference" and h["scorer"] == "LR" and h["metric"] == "prmb_top1_on_error") or
             (h["bank"] == "-" and h["table"] == "prmbench_panel_b" and "Qwen2.5-Math-PRM-7B" in h["scorer"] and h["metric"] == "top1_on_error"))):
            references.append(dict(method=h["scorer"], evidence="reported reference; no raw scores exported",
                                   old=float(h["old_replay"]) if h["old_replay"] else None,
                                   prefix_corrected=float(h["thinkfix"]) if h["thinkfix"] else None,
                                   corrected_mask1=float(h["thinkfix_mask1"]) if h["thinkfix_mask1"] else None,
                                   recomputed=None, delta_vs_old=None))
    by_type, step0_rows = [], []
    for m in METHODS:
        method_rows = [r for r in all_rows if r["method"] == m]
        for kind in sorted({r["classification"] for r in method_rows}):
            selected = [r for r in method_rows if r["classification"] == kind]
            by_type.append(dict(method=m, classification=kind, **aggregate(selected)))
        for kind in ["ALL"] + sorted({r["classification"] for r in method_rows}):
            selected_type = [r for r in method_rows if kind == "ALL" or r["classification"] == kind]
            for label_stratum in ("all", "has_error", "no_error", "step0_error", "step0_correct"):
                selected = [r for r in selected_type if
                            label_stratum == "all" or
                            label_stratum == "has_error" and r["n_error"] > 0 or
                            label_stratum == "no_error" and r["n_error"] == 0 or
                            label_stratum == "step0_error" and r["step0_is_error"] or
                            label_stratum == "step0_correct" and not r["step0_is_error"]]
                step0_rows.append(dict(method=m, classification=kind, label_stratum=label_stratum,
                                       n_answers=len(selected),
                                       n_top1_step0=sum(r["top1_position"] == 0 for r in selected),
                                       top1_step0_proportion=(sum(r["top1_position"] == 0 for r in selected) /
                                                              len(selected)) if selected else None))
    binned_summary = []
    for m in METHODS:
        for b in range(5):
            selected = [r for r in bins if r["method"] == m and r["bin"] == b]
            binned_summary.append(dict(method=m, bin=b, relative_position_low=b / 5,
                                       relative_position_high=(b + 1) / 5,
                                       n_eligible_answers=len(selected), n_pairs=sum(r["n_pairs"] for r in selected),
                                       mean_answer_auc=mean([r["auc"] for r in selected]),
                                       pair_pooled_auc=(sum(r["wins"] for r in selected) /
                                                        sum(r["n_pairs"] for r in selected)) if selected else None))
    audit = dict(branch_v10_commit=V10_COMMIT, research_commit=RESEARCH_COMMIT,
                 inputs_sha256=before, script_sha256=sha(Path(__file__)),
                 canonical_step_index_run=str(args.pinned_run), canonical_run_sha256=sha(args.pinned_run),
                 bank="N16", variant="prefix-corrected + mask1", score_orientation="higher = riskier; immutable exports",
                 n_answers=6969, n_steps=flat_n,
                 annotation_policy="Use original 1-based membership on existing steps; out-of-range annotations label no existing step and are counted, never repaired.",
                 out_of_range_annotation_rows=len(oob_rows), out_of_range_annotations=int(oob_annotations),
                 top1_contract="Exact maximum risk; ties choose earlier step. Canonical 8*eps tolerance checked separately and no selections differed.",
                 within_auc_contract="Half credit for ties; equal-weight mean across answers containing both error and correct steps.",
                 position_contract=dict(bin_definition="floor(5*(position+0.5)/n_steps), 0-based position; 5 fixed bins",
                                        primary="Within each answer, pool only same-bin error/correct pairs, calculate AUC, then average equally over eligible answers.",
                                        secondary="Per-bin equal-answer AUC and pooled-pair AUC reported separately; weights differ.",
                                        warning="Diagnostic on development data. Coarse bins retain within-bin position differences; eligibility and pair weighting change. Not proof of confound removal.",
                                        no_tuning=True),
                 source_group_ci="NOT_RUN: authentic group IDs absent from exported metadata. idx not treated as group mapping.",
                 unavailable_controls=["step token lengths", "text repetition", "authentic source-group holdout", "PRM/LR raw arrays"],
                 checks=dict(max_pair_vs_rank_auc_abs_difference=max_auc_difference,
                             exact_vs_epsilon_top1_selection_differences=n_exact_epsilon_differences,
                             historical_top1_mask1_parity="PASS for all four exported methods",
                             historical_step_index_auc_parity="PASS",
                             input_hashes_unchanged=all(sha(Path(p)) == h for p, h in before.items())),
                 summary=summary, references=references)
    assert audit["checks"]["input_hashes_unchanged"]
    (args.output_root / "RESULTS.json").write_text(json.dumps(audit, indent=2) + "\n")
    write_csv(args.output_root / "SUMMARY.csv", [dict(method=m, **v) for m, v in summary.items()])
    write_csv(args.output_root / "BY_TYPE.csv", by_type)
    write_csv(args.output_root / "TOP1_STEP0_STRATA.csv", step0_rows)
    write_csv(args.output_root / "POSITION_BINS.csv", binned_summary)
    write_csv(args.output_root / "HISTORICAL_TOP1.csv", references)
    print(json.dumps(dict(summary=summary, checks=audit["checks"]), indent=2))


if __name__ == "__main__":
    main()
