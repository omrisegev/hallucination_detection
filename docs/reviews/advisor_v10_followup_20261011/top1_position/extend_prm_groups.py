"""Saved PRM + authentic source-group extension; no model fitting or selection."""
import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

METHODS = ("LSML_CONT", "IU_PCR", "ALL_EQ", "DS_FILTER_EQ", "STEP_INDEX", "PRM")
CONTRASTS = (("LSML_CONT", "PRM"), ("IU_PCR", "PRM"),
             ("LSML_CONT", "ALL_EQ"), ("IU_PCR", "ALL_EQ"),
             ("LSML_CONT", "DS_FILTER_EQ"))
COHORTS = ("ALL", "redundency", "circular")
METRICS = ("within_auc", "top1_on_error", "same_bin_auc")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def auc(v, e):
    bad, good = v[e], v[~e]
    if not len(bad) or not len(good):
        return np.nan
    delta = bad[:, None] - good[None, :]
    value = float((np.count_nonzero(delta > 0) + .5 * np.count_nonzero(delta == 0)) / delta.size)
    ranks = rankdata(v)
    other = (ranks[e].sum() - len(bad) * (len(bad) + 1) / 2) / (len(bad) * len(good))
    assert abs(value - other) < 1e-12
    return value


def csv_out(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package-root", type=Path, required=True)
    ap.add_argument("--extra-inputs", type=Path, default=Path(__file__).resolve().parent.parent / "sources")
    ap.add_argument("--output-root", type=Path, required=True)
    a = ap.parse_args()
    bp = a.package_root / "sources/prmscore_inputs/bank_N16.npz"
    mp = a.package_root / "sources/prmscore_inputs/prmbench_meta.json"
    cp = a.package_root / "sources/PRMBENCH_COMPARISON.csv"
    rp = a.extra_inputs / "prmbench_prm.pkl"
    gp = a.extra_inputs / "CANONICAL_GROUPS.json"
    original_hashes = {str(p): sha(p) for p in (bp, mp, cp, rp, gp)}
    bank = np.load(bp, allow_pickle=False)
    ids = list(bank["idx"].astype(str))
    offsets = bank["offsets"]
    metadata = {r["idx"]: r for r in json.loads(mp.read_text())["rows"]}
    raw = pickle.load(rp.open("rb"))
    prm_rows = {r["idx"]: r for r in raw.values()}
    group_document = json.loads(gp.read_text())
    group_map = {r["row_id"]: r["canonical_group_id"] for r in group_document["rows"]}
    assert len(prm_rows) == len(raw) == len(ids) == len(set(ids)) == 6969
    assert len(group_map) == len(group_document["rows"])
    assert set(ids) == set(prm_rows) == set(metadata)
    assert set(ids) <= set(group_map), "All exported rows require authentic group mapping"
    names = sorted({group_map[idx] for idx in ids})
    assert len(names) == 707
    group_number = {name: j for j, name in enumerate(names)}
    g = np.array([group_number[group_map[idx]] for idx in ids])
    kind = np.array([metadata[idx]["classification"] for idx in ids])
    values = {m: {k: np.full(len(ids), np.nan) for k in METRICS + ("full_auc_matched",)} for m in METHODS}
    native_midpoint_ties = 0
    for i, idx in enumerate(ids):
        md, r = metadata[idx], prm_rows[idx]
        start, stop = offsets[i:i + 2]
        n = int(stop - start)
        assert r["n_steps"] == md["n_steps"] == n
        assert r["classification"] == md["classification"] and r["category"] == md["category"]
        assert r["error_steps"] == md["error_steps"]
        rewards = np.array(r["rewards"], dtype=float)
        assert rewards.shape == (n,) and np.isfinite(rewards).all()
        assert np.all((rewards >= 0) & (rewards <= 1))
        assert np.array_equal(np.array(r["labels"]), (rewards >= .5).astype(int))
        native_midpoint_ties += int(np.count_nonzero(rewards == .5))
        e = np.isin(np.arange(1, n + 1), md["error_steps"])
        bins = np.floor(5 * (np.arange(n) + .5) / n).astype(int)
        for m in METHODS:
            v = (1 - rewards if m == "PRM" else np.arange(n, dtype=float) if m == "STEP_INDEX"
                 else bank["S__" + m][start:stop])
            top = int(np.argmax(v))
            assert top == int(np.argsort(-v, kind="stable")[0])
            if e.any():
                values[m]["top1_on_error"][i] = int(e[top])
            full = auc(v, e)
            values[m]["within_auc"][i] = full
            wins, pairs = 0., 0
            for b in range(5):
                chosen = bins == b
                sube, subv = e[chosen], v[chosen]
                n_pairs = int(sube.sum() * (~sube).sum())
                if n_pairs:
                    wins += auc(subv, sube) * n_pairs
                    pairs += n_pairs
            if pairs:
                values[m]["same_bin_auc"][i] = wins / pairs
                values[m]["full_auc_matched"][i] = full
    assert native_midpoint_ties == 40
    hist = list(csv.DictReader(cp.open()))
    for metric in ("within_auc", "top1_on_error"):
        ref = [r for r in hist if r["table"] == "prmbench_panel_b" and r["bank"] == "-" and
               r["scorer"] == "Qwen2.5-Math-PRM-7B (supervised)" and r["metric"] == metric]
        assert len(ref) == 1 and abs(float(ref[0]["thinkfix_mask1"]) - np.nanmean(values["PRM"][metric])) < 1e-12
    prior = json.loads((a.output_root / "RESULTS.json").read_text())
    for m in METHODS[:-1]:
        for metric in METRICS:
            target = prior["summary"][m][metric]
            assert abs(float(np.nanmean(values[m][metric])) - target) < 1e-12
    rng = np.random.default_rng(20261011)
    draws = rng.integers(0, len(names), size=(5000, len(names)))
    weights = np.zeros((5000, len(names)), dtype=np.int16)
    for i, draw in enumerate(draws):
        weights[i] = np.bincount(draw, minlength=len(names))
    assert np.all(weights.sum(axis=1) == len(names))
    group_draw_sha = hashlib.sha256(draws.astype("<i8").tobytes()).hexdigest()
    del draws
    aggregates, contrasts = [], []
    for cohort in COHORTS:
        cohort_mask = np.ones(len(ids), dtype=bool) if cohort == "ALL" else kind == cohort
        for metric in METRICS:
            for m in METHODS:
                eligible = cohort_mask & np.isfinite(values[m][metric])
                aggregates.append(dict(cohort=cohort, metric=metric, method=m,
                                       n_answers=int(eligible.sum()), n_groups=int(len(set(g[eligible]))),
                                       value=float(np.mean(values[m][metric][eligible]))))
            for lhs, rhs in CONTRASTS:
                eligible = cohort_mask & np.isfinite(values[lhs][metric]) & np.isfinite(values[rhs][metric])
                assert np.array_equal(eligible, cohort_mask & np.isfinite(values[lhs][metric]))
                assert np.array_equal(eligible, cohort_mask & np.isfinite(values[rhs][metric]))
                difference = values[lhs][metric][eligible] - values[rhs][metric][eligible]
                sums = np.bincount(g[eligible], weights=difference, minlength=len(names))
                counts = np.bincount(g[eligible], minlength=len(names))
                numerators = np.einsum("ij,j->i", weights, sums, optimize=False)
                denominators = np.einsum("ij,j->i", weights, counts, optimize=False)
                assert np.all(denominators > 0)
                boot = numerators / denominators
                lo, hi = map(float, np.quantile(boot, [.025, .975]))
                contrasts.append(dict(cohort=cohort, metric=metric, lhs=lhs, rhs=rhs,
                                      n_answers=int(eligible.sum()), n_groups=int(len(set(g[eligible]))),
                                      difference=float(difference.mean()), ci95_low=lo, ci95_high=hi,
                                      n_draws=5000, bootstrap_se=float(boot.std(ddof=1))))
                # One draw also checked by explicit expansion of whole-group multiplicities.
                expanded = np.repeat(difference, weights[0, g[eligible]])
                assert abs(float(expanded.mean()) - boot[0]) < 1e-12
    figure_data = []
    for m in METHODS:
        same = np.isfinite(values[m]["same_bin_auc"])
        assert same.sum() == 4691
        figure_data.append(dict(method=m, n_answers=int(same.sum()),
                                unrestricted_same_answers=float(np.mean(values[m]["within_auc"][same])),
                                same_bin_auc=float(np.mean(values[m]["same_bin_auc"][same]))))
    a.output_root.mkdir(parents=True, exist_ok=True)
    csv_out(a.output_root / "PRM_GROUP_AGGREGATES.csv", aggregates)
    csv_out(a.output_root / "SOURCE_GROUP_CONTRASTS.csv", contrasts)
    csv_out(a.output_root / "SAME_ELIGIBLE_FIGURE_DATA.csv", figure_data)
    checks = dict(input_hashes_unchanged=all(sha(Path(p)) == h for p, h in original_hashes.items()),
                  prm_historical_auc_top1_parity="PASS", prior_stage_parity="PASS",
                  pair_vs_rank_auc="PASS", authentic_all_row_group_alignment="PASS", n_groups=707,
                  explicit_group_expansion_first_draw="PASS for all 45 contrasts", rewards_half_ties=40,
                  native_labels_match_rewards_ge_half="PASS")
    assert checks["input_hashes_unchanged"]
    report = dict(status="COMPLETED", scope="Saved-score development diagnostics; no method refit, threshold selection or held-out test",
                  supersedes="Initial RESULTS.json asset limitations concerning PRM arrays and authentic group availability; original stage remains historical",
                  script_sha256=sha(Path(__file__)), input_sha256=original_hashes,
                  grouping_version=group_document.get("grouping_version"), grouping_rule=group_document.get("rule"),
                  n_all_group_map_rows=len(group_map), n_aligned_answers=len(ids), n_aligned_groups=707,
                  bootstrap=dict(seed=20261011, n_draws=5000, resampling="Each draw samples 707 authentic groups with replacement from the aligned population; all answer rows inherit whole-group multiplicity; metric retains equal-answer weighting",
                                 group_order="lexicographic canonical_group_id", sampled_group_indices_sha256=group_draw_sha,
                                 intervals="Paired percentile 95%; descriptive; no multiplicity correction across 45 fixed-for-run development contrasts",
                                 frozen_contrasts=CONTRASTS, cohorts=COHORTS, metrics=METRICS),
                  position_contract=prior["position_contract"], checks=checks, aggregates=aggregates, contrasts=contrasts,
                  figure_data=figure_data, remaining_limits=["Development selection; no holdout", "Coarse relative-position bins leave residual position variation", "Token lengths and repetition features absent", "LR raw scores absent"])
    (a.output_root / "PRM_GROUP_EXTENSION.json").write_text(json.dumps(report, indent=2) + "\n")
    # Scientific figure is a standalone export, with the same population on both ends.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = ("STEP_INDEX", "ALL_EQ", "DS_FILTER_EQ", "LSML_CONT", "IU_PCR", "PRM")
    data = {r["method"]: r for r in figure_data}
    labels = dict(STEP_INDEX="Position only", ALL_EQ="Plain average", DS_FILTER_EQ="DS-filter average",
                  LSML_CONT="Continuous L-SML", IU_PCR="IU-PCR", PRM="Qwen PRM-7B")
    fig, ax = plt.subplots(figsize=(9, 5.3))
    for i, m in enumerate(order):
        r = data[m]
        ax.plot([r["same_bin_auc"], r["unrestricted_same_answers"]], [i, i], color="#b8c4cc", lw=3)
        ax.scatter(r["unrestricted_same_answers"], i, color="#bf6235", s=55, zorder=3)
        ax.scatter(r["same_bin_auc"], i, color="#246c9c", s=55, zorder=3)
        ax.annotate(f'{r["same_bin_auc"]:.3f}', (r["same_bin_auc"], i), xytext=(0, 10),
                    textcoords="offset points", ha="center", fontsize=9, color="#246c9c")
        ax.annotate(f'{r["unrestricted_same_answers"]:.3f}', (r["unrestricted_same_answers"], i),
                    xytext=(0, -17), textcoords="offset points", ha="center", fontsize=9, color="#bf6235")
    ax.set_yticks(range(len(order)), [labels[m] for m in order])
    ax.set_xlim(.49, .84)
    ax.invert_yaxis()
    ax.set_xlabel("Within-answer AUC · equal weight per eligible answer")
    ax.set_title("Does ranking signal remain within coarse position bins?", loc="left", fontsize=15)
    ax.grid(axis="x", alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.scatter([], [], color="#bf6235", label="Unrestricted pairs, same 4,691 answers")
    ax.scatter([], [], color="#246c9c", label="Same-bin pairs, fixed five position bins")
    ax.legend(loc="center left", bbox_to_anchor=(.01, .55), frameon=False, fontsize=8)
    fig.text(.02, .015, "4,691 / 6,030 AUC-eligible answers retain same-bin pairs. Development diagnostic; residual position differences remain.", fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, 1))
    fig.savefig(a.output_root / "SAME_ELIGIBLE_POSITION.svg", bbox_inches="tight")
    fig.savefig(a.output_root / "SAME_ELIGIBLE_POSITION.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(dict(checks=checks, prm=[r for r in aggregates if r["method"] == "PRM"],
                          overall_contrasts=[r for r in contrasts if r["cohort"] == "ALL"]), indent=2))


if __name__ == "__main__":
    main()
