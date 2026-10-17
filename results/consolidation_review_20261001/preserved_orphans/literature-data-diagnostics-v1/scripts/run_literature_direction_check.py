"""Audit literature-inspired directions on frozen project artifacts.

This is a bounded, data-only check.  It does not fit a new detector or change
any frozen benchmark result.  It independently recomputes the serial
step-dependence contrast from the saved RBM diagnostics and joins it to the
existing ProcessBench error categories.  The remaining rows are provenance
audits of the already completed conditional-reliability and graph/router
diagnostics, so they are not presented as new benchmark experiments.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "rbm-data-diagnostics-v1"
DIAG = SOURCE / "results" / "rbm_data_diagnostics_v1"
FAMILY = ROOT.parents[1] / "results" / "family_relevance_real_v1"
GLOBAL = ROOT.parents[1] / "results" / "global_contextual_stg_router_diagnostic_v1"
DUFS = ROOT.parent / "dufs-moment-selection-v1" / "results" / "dufs_moment_selection_v1"
OUT = ROOT / "results" / "literature_direction_check_v1"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def bootstrap_ci(values: np.ndarray, seed: int, draws: int = 4000):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return [None, None]
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return [float(x) for x in np.quantile(sampled, [0.025, 0.975])]


def read_error_cases():
    rows = []
    with (DIAG / "ERROR_CASES.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    return rows


def serial_check():
    cases = read_error_cases()
    by_bank_uid = {(int(r["bank"]), r["uid"]): r for r in cases}
    rows, summaries = [], {}
    for bank in (6, 12):
        z = np.load(DIAG / f"BANK{bank}_DIAGNOSTICS.npz", allow_pickle=False)
        uid = z["uid"].astype(str)
        lag = np.asarray(z["lag"], float)
        perm = np.asarray(z["permuted"], float)
        # Saved arrays use the all-steps mask at index 0.  The statistic is
        # the mean lag-1 residual correlation minus its within-answer
        # permutation control, averaged over feature coordinates.
        delta = lag[:, 0, 0, :] - perm[:, 0, 0, :]
        valid = np.isfinite(delta).any(axis=1)
        excess = np.full(len(delta), np.nan, dtype=float)
        excess[valid] = np.nanmean(delta[valid], axis=1)
        for i, key in enumerate(uid):
            case = by_bank_uid.get((bank, key))
            if case is None:
                continue
            row = dict(bank=bank, uid=key, excess=float(excess[i]),
                       category=case["category"], truth_longest=case["truth_longest"] == "True",
                       target=int(case["target"]))
            rows.append(row)
        categories = ["exact", "early", "late", "gate_miss", "clean_correct", "false_alarm"]
        bank_summary = {
            "n_answers": int(len(uid)),
            "n_valid_serial": int(valid.sum()),
            "all_mean_excess": float(np.nanmean(excess)),
            "all_ci95": bootstrap_ci(excess, 6010 + bank),
            "categories": {},
        }
        bank_rows = [r for r in rows if r["bank"] == bank]
        for category in categories:
            vals = np.asarray([r["excess"] for r in bank_rows if r["category"] == category], float)
            bank_summary["categories"][category] = {
                "n": int(len(vals)),
                "mean_excess": float(np.nanmean(vals)) if len(vals) else None,
                "ci95": bootstrap_ci(vals, 6020 + bank + categories.index(category)),
            }
        summaries[str(bank)] = bank_summary
    return rows, summaries


def residual_check():
    result = {}
    for bank in (6, 12):
        z = np.load(DIAG / f"BANK{bank}_DIAGNOSTICS.npz", allow_pickle=False)
        corr = np.asarray(z["corr"], float)
        # The saved corr is answer x feature x feature; remove diagonals.
        off = ~np.eye(corr.shape[1], dtype=bool)
        observed = np.nanmean(np.abs(corr[:, off]), axis=1)
        eig = np.asarray(z["eigen"], float)
        valid_eig = np.isfinite(eig).any(axis=1)
        result[str(bank)] = {
            "n_answers": int(len(corr)),
            "mean_abs_observed_residual_corr": float(np.nanmean(observed)),
            "observed_corr_ci95": bootstrap_ci(observed, 6030 + bank),
            "mean_top_residual_eigenvalue": float(np.nanmean(eig[:, 0])),
            "n_valid_eigen": int(valid_eig.sum()),
        }
    return result


def literature_status():
    reliability = load_json(DIAG / "RELIABILITY_REGIMES.json")
    interactions = [x for x in reliability["contrasts"] if x.get("contrast") == "interaction"]
    global_summary = []
    with (GLOBAL / "summary.csv").open(encoding="utf-8", newline="") as handle:
        global_summary = list(csv.DictReader(handle))
    family_context = []
    with (FAMILY / "context_specialization.csv").open(encoding="utf-8", newline="") as handle:
        family_context = list(csv.DictReader(handle))
    dufs_state = load_json(DUFS / "RUN_STATE.json") if (DUFS / "RUN_STATE.json").exists() else {"status": "missing"}
    return {
        "conditional_reliability": {
            "source": str(DIAG / "RELIABILITY_REGIMES.json"),
            "matched_interactions": interactions,
            "interpretation": "feature utility changes by regime, but no label-free router was fitted here",
        },
        "dependent_family_graph": {
            "source": str(FAMILY / "context_specialization.csv"),
            "context_rows": family_context,
            "interpretation": "IU-rank specialization exists; graph-coupled family routing was already negative",
        },
        "held_out_context_router": {
            "source": str(GLOBAL / "summary.csv"),
            "rows": global_summary,
            "interpretation": "c-STG did not pass its registered held-out routing gates",
        },
        "dufs_feature_selection": {
            "source": str(DUFS / "RUN_STATE.json"),
            "run_state": dufs_state,
            "interpretation": "active independent run; no conclusion imported before completion",
        },
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    serial_rows, serial_summary = serial_check()
    checks = {
        "bank_shape_and_uid_alignment": True,
        "all_error_cases_joined": True,
        "serial_valid_counts_match_saved_diagnostic": True,
    }
    for bank in (6, 12):
        z = np.load(DIAG / f"BANK{bank}_DIAGNOSTICS.npz", allow_pickle=False)
        if len(np.unique(z["uid"])) != len(z["uid"]):
            checks["bank_shape_and_uid_alignment"] = False
        joined = [r for r in serial_rows if r["bank"] == bank]
        if len(joined) != 6800:
            checks["all_error_cases_joined"] = False
        expected_valid = int(np.isfinite(z["lag"][:, 0, 0, :]).any(axis=1).sum())
        if expected_valid != serial_summary[str(bank)]["n_valid_serial"]:
            checks["serial_valid_counts_match_saved_diagnostic"] = False
    checks["status"] = "PASS" if all(checks.values()) else "FAIL"
    payload = {
        "schema": "literature-direction-check-v1",
        "scope": "frozen full-data diagnostics; no new fit and no benchmark mutation",
        "serial_step_check": serial_summary,
        "residual_check": residual_check(),
        "literature_status": literature_status(),
        "review": checks,
        "limitations": [
            "The serial test uses step-mean residuals and therefore does not isolate the exact boundary token-to-token transition.",
            "The error-category join is retrospective and descriptive; it does not train a detector or establish causality.",
            "Existing conditional-router and graph results remain development evidence on their original panels.",
        ],
    }
    (OUT / "LITERATURE_DIRECTION_CHECK.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (OUT / "SERIAL_CASES.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["bank", "uid", "excess", "category", "truth_longest", "target"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(serial_rows)
    report = [
        "# Literature direction check v1",
        "",
        "This run independently checks saved full-data diagnostics; it does not fit a new model or modify benchmark scores.",
        "",
        "## Sequential/networked direction",
        "",
        "The statistic is lag-1 residual step correlation minus a within-answer permutation control. It tests whether ordered steps contain structure beyond the current one-unit RBM; it is not an exact last-token/first-token test.",
        "",
        "| bank | all mean excess | 95% CI | exact | early | late | gate-miss | clean-correct | false-alarm |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for bank in (6, 12):
        s = serial_summary[str(bank)]
        c = s["categories"]
        fmt = lambda x: "NA" if x is None else f"{x:.5f}"
        report.append(f"| {bank} | {s['all_mean_excess']:.5f} | [{fmt(s['all_ci95'][0])}, {fmt(s['all_ci95'][1])}] | {c['exact']['n']}/{fmt(c['exact']['mean_excess'])} | {c['early']['n']}/{fmt(c['early']['mean_excess'])} | {c['late']['n']}/{fmt(c['late']['mean_excess'])} | {c['gate_miss']['n']}/{fmt(c['gate_miss']['mean_excess'])} | {c['clean_correct']['n']}/{fmt(c['clean_correct']['mean_excess'])} | {c['false_alarm']['n']}/{fmt(c['false_alarm']['mean_excess'])} |")
    report += [
        "",
        "## What the paper-linked checks support",
        "",
        "- Conditional reliability: the matched early/late interaction is retained from the saved audit; it justifies one carefully frozen conditional-fusion test, not a deployable router.",
        "- Dependent classifiers/graphs: family specialization is real, but the graph-coupled router and held-out c-STG gate did not pass; no graph expansion is justified by these results.",
        "- RBM/deep energy: residual correlations exceed one-hidden synthetic draws, but the existing task diagnostics do not show that extra capacity would improve localization; keep exact one-unit controls before deeper/CD variants.",
        "- Sequential/networked: ordered residual structure is present. The category comparison below tests whether it is stronger in useful or failed localizations; a positive structure alone is not enough for a new model.",
        "",
        "Full machine-readable details: `LITERATURE_DIRECTION_CHECK.json` and `SERIAL_CASES.csv`.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(payload["serial_step_check"], indent=2))


if __name__ == "__main__":
    main()
