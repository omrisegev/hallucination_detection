"""Verify frozen representation-pilot outputs and render its factual report."""
from pathlib import Path
import ast
import collections
import hashlib
import json
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/answer_localization_representation_pilot_v1"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fmt(value):
    return "undefined" if value is None else f"{value:.5f}"


def main():
    prepared, frozen, evaluation = [load(OUT / name) for name in ("PREPARED.json", "SCORES_FROZEN.json", "EVALUATION.json")]
    for filename, expected in frozen["files"].items():
        assert sha(filename) == expected, filename
    for filename, expected in prepared["source_hashes"].items():
        assert sha(filename) == expected, filename
    # Check raw-column identity without importing the producer or experiment.
    def literal_assignment(path, name):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        node = next(n for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
        return ast.literal_eval(node.value)
    schema_source = ROOT.parent / "hd_jlsml_v2_wt/spectral_utils/token_feature_views.py"
    actual_schema = literal_assignment(ROOT / "spectral_utils/answer_localization_v2.py", "STREAM_NAMES")
    assert actual_schema == ("trace_length_series",) + literal_assignment(schema_source, "BROAD_TOKEN_VIEWS")
    # Rejoin labels directly by ID; the metric checks below alone would reuse
    # the evaluator's joined targets and could not detect an alignment error.
    release = load(OUT / "RELEASE.json")
    assert sha(OUT / "RELEASE.json") == prepared["release_manifest_sha256"]
    label_checks = 0
    for cell in sorted({r["cell"] for r in evaluation["rows"]}):
        source = release["cells"][cell]
        assert sha(source["label_path"]) == source["label_opaque_sha256"]
        with np.load(source["label_path"], allow_pickle=False) as labels:
            for row in (r for r in evaluation["rows"] if r["cell"] == cell):
                matches = np.flatnonzero(labels["row_ids"] == row["row_id"])
                assert len(matches) == 1
                index = int(matches[0])
                if cell.startswith("prm"):
                    offsets = labels["step_flag_offsets"]
                    target = labels["step_error_flags"][offsets[index]:offsets[index+1]]
                    assert len(target) == row["steps"]
                else:
                    target = labels["first_error"][index]
                np.testing.assert_array_equal(target, row["target"])
                label_checks += 1
    # Inspect every actual projection and span mapping, using frozen fit values
    # and parameters rather than calling the original scoring functions.
    projection_checks, max_projection_error = 0, 0.
    groups, coverage = collections.defaultdict(collections.Counter), collections.Counter()
    readout_failures, fit_seconds = collections.Counter(), []
    sys.path.insert(0, str(ROOT / "local_cache/short_cycle01_code"))
    from spectral_utils.window_localization import WINDOW_FEATURE_NAMES
    primitives = ("entropy_series", "spilled_series", "energy_series", "top1_logprob_series",
                  "logprob_margin_series", "topk_entropy_series", "topk_varentropy_series",
                  "topk_renyi2_series", "topk_tail_mass_series")
    moment_names = [f"{s}__{op}" for s in primitives for op in ("level", "sd", "slope")]
    for row in evaluation["rows"]:
        meta = load(OUT / "scores" / f"{row['uid']}.json")
        fit_seconds.append(meta["elapsed_seconds"])
        with np.load(OUT / "scores" / f"{row['uid']}.npz", allow_pickle=False) as arrays:
            assert len(arrays["step_starts"]) == row["steps"]
            for arm, valid in row["valid"].items():
                if valid:
                    coverage[arm] += 1
                detail = meta["report"]["methods"].get(arm, {})
                if valid and not detail.get("readout_valid"):
                    readout_failures[arm] += 1
                if arm == "entropy_mean_w8" or not arm.endswith("__equal"):
                    continue
                rep = arm.removesuffix("__equal")
                rep_meta = meta["report"]["representations"].get(rep, {})
                shared = rep_meta.get("shared", {})
                grouping = shared.get("grouping", {})
                if grouping.get("status") == "SELECTED":
                    groups[rep][grouping["K"]] += 1
                if rep == "legacy_fixed32" or f"{rep}__features" not in arrays.files or not shared:
                    continue
                schema = WINDOW_FEATURE_NAMES if rep.startswith("global") else moment_names
                cols = [list(schema).index(s) for s in shared["active_features"]]
                values = arrays[f"{rep}__features"][:, cols]
                fit = arrays[f"{rep}__fit_indices"]
                z = (values - np.asarray(shared["mean"])) / np.asarray(shared["sd"])
                z -= z[fit].mean(axis=0)
                z *= np.asarray(shared["feature_signs"])
                a, b = arrays[f"{rep}__starts"], arrays[f"{rep}__ends"]
                for method in ("equal", "iu", "joint_lambda0", "joint_graph010", "joint_graph_permuted"):
                    key = f"{rep}__{method}"
                    fitted = meta["report"]["methods"].get(key, {})
                    if "standardized_weights" not in fitted:
                        continue
                    expected = -z @ np.asarray(fitted["standardized_weights"])
                    actual = arrays[f"{key}__window"]
                    error = float(np.max(np.abs(expected-actual)))
                    max_projection_error = max(max_projection_error, error)
                    assert error < 1e-9
                    assert abs(float(np.std(actual[fit]))-1.) < 1e-8
                    tokens, counts = np.zeros(row["tokens"]), np.zeros(row["tokens"], dtype=int)
                    for lo, hi, score in zip(a, b, actual):
                        tokens[lo:hi] += score
                        counts[lo:hi] += 1
                    assert (counts > 0).all()
                    tokens /= counts
                    mapped = [float(np.max(tokens[lo:hi])) for lo, hi in zip(arrays["step_starts"], arrays["step_ends"])]
                    np.testing.assert_allclose(mapped, arrays[f"{key}__step"], atol=1e-9, rtol=0.)
                    projection_checks += 1
    # Independent recomputation of both endpoints, including failed-fit penalties.
    for arm, metrics in evaluation["metrics"].items():
        prm = [r for r in evaluation["rows"] if r["cell"].startswith("prm") and r["valid"].get(arm)]
        if prm:
            y, s = np.concatenate([r["target"] for r in prm]), np.concatenate([r["scores"][arm] for r in prm])
            if len(set(y)) == 2:
                assert abs(float(roc_auc_score(y, s))-metrics["prm"]["auroc"]) < 1e-12
        cell_f1 = []
        for cell, expected in metrics["pb"]["cells"].items():
            rows = [r for r in evaluation["rows"] if r["cell"] == cell]
            hits = {"clean": [], "error": []}
            for r in rows:
                assert -1 <= r["target"] < r["steps"]
                hit = r["decision_valid"].get(arm, False) and r["predictions"].get(arm) == r["target"]
                hits["clean" if r["target"] == -1 else "error"].append(int(hit))
            clean, error = np.mean(hits["clean"]), np.mean(hits["error"])
            value = 2*clean*error/(clean+error) if clean+error else 0.
            assert abs(value-expected["f1"]) < 1e-12
            cell_f1.append(value)
        assert abs(float(np.mean(cell_f1))-metrics["pb"]["macro_f1"]) < 1e-12
    audit = {"projection_checks": projection_checks, "max_projection_error": max_projection_error,
             "group_K_counts": dict(groups), "coverage": dict(coverage), "readout_failures": dict(readout_failures),
             "seconds_sum": sum(fit_seconds), "seconds_median": float(np.median(fit_seconds)),
             "seconds_max": max(fit_seconds), "evaluation_sha256": sha(OUT / "EVALUATION.json"),
             "review_script_sha256": sha(__file__), "scores_and_code_hashes_verified": True,
             "direct_label_id_rejoins": label_checks, "raw_column_schema": "PASS; 29 columns",
             "raw_schema_source_sha256": sha(schema_source),
             "endpoint_recomputation": "PASS; all 19 arms", "score_span_mapping": "PASS"}
    (OUT / "REVIEW.json").write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# Answer-localization representation pilot v1", "", "2026-09-07. Retrospective development stress test; no winner promoted.", "",
             "The short-window representation improves fit coverage, but the fixed mixture/first-crossing readout fails on ProcessBench. A graph benefit is not established. The next experiment should isolate the chronological readout while retaining these frozen feature scores and the Joint representation hypotheses.", "",
             "## Frozen scope", "", "58 answers: 12 PRMBench, 46 ProcessBench across four Qwen3-8B subsets. Selection used source-group hashes in three length bins, without labels. This deliberately stresses short and long traces; its aggregate is not a representative full-benchmark estimate. Every selected answer has a distinct registered source group within its cell. All data were already exposed in Claude v2. Qwen3-4B is registered for later replication.", "",
             "Release: `localization-cached-v1-20260907`. Protocol: `docs/experiments/ANSWER_LOCALIZATION_REPRESENTATION_PILOT_V1.md`. Four representations separate historical borrowed signs, within-answer signs, the original 30 measurements, a 27-coordinate primitive-moment bank, and widths 32/8. The answer-local sign convention retains a declared negative-entropy anchor; it is not anchor-free. Equal, canonical IU and Joint lambda-zero are included, with meaningful/permuted graphs on the local-sign lanes.", "",
             "## Main findings", "",
             "- IU/equal coverage increases from 38/58 at width 32 to 58/58 with moments at width 8. Joint coverage is 32/58 in the legacy recipe and 43/58 in moments-8. The new moment bank at width 32 has only 27/58 valid Joint fits; a new feature bank alone did not fix every partition.",
             "- Moments-8 selects K=4 in 14 answers and K=3 in 29. The global-30 local-sign bank also selects K=4 and K=6 in some answers. This refutes a universal 'always three groups' reading of the earlier two-fold diagnosis, but is not an accuracy result.",
             "- On the seven common valid PRMB answers for moments-8 Joint/IU: meaningful graph AUROC 0.66255 versus IU 0.64964, difference +0.01291, CI [-0.03016,+0.04864]. Graph versus lambda zero and permutation is unresolved. These very small common subsets cannot establish a method advantage.",
             "- The BIC-selected mixture followed by the first threshold-crossing step gives PB macro-F1 zero for all 18 fusion arms; the entropy control gives 0.08333. This is a failure of these full pilot pipelines. It is not evidence that all underlying risk rankings are useless.",
             "- For moments-8 IU, 26/46 PB predictions are step 0, 19 are no-error and one is step 1; none hits the true first error. In a post-hoc error-only diagnostic, argmax risk hits 7/25 erroneous answers. That is not a replacement result or a validated alternative readout; it shows why the choice of locator must be isolated.", "",
             "## Descriptive table: differing availability is explicit", "",
             "Do not rank this entire table as a common-population leaderboard. PRMB AUROC uses the available valid answers listed; the paired table below uses shared IDs. PB uses all 46 selected answers and counts unavailable predictions as misses in either class.", "",
             "| Arm | Valid fits / 58 | PRMB answers | PRMB AUROC | PB macro-F1 |", "|---|---:|---:|---:|---:|"]
    for arm, metrics in evaluation["metrics"].items():
        lines.append(f"| `{arm}` | {coverage[arm]} | {metrics['prm']['answers']} | {fmt(metrics['prm']['auroc'])} | {fmt(metrics['pb']['macro_f1'])} |")
    lines += ["", "## Paired PRMB contrasts", "", "Intervals are exploratory, unadjusted source-group bootstraps, 1,000 draws. Common-cohort counts are small. The PB intervals [0,0] for two failed zero-hit pipelines are degenerate resampling results, not proof of population equivalence.", "",
              "| Left minus right | Common PRMB answers | Observed AUROC difference | 95% interval |", "|---|---:|---:|---:|"]
    for name, pair in evaluation["paired"].items():
        a, b = pair["left_prm"]["auroc"], pair["right_prm"]["auroc"]
        ci = pair["uncertainty"]["prm_common_valid_ci95"]
        lines.append(f"| `{name}` | {pair['left_prm']['answers']} | {fmt(a-b) if a is not None and b is not None else 'undefined'} | {str([round(x,5) for x in ci]) if ci else 'undefined'} |")
    lines += ["", "## Review and historical continuity", "",
              f"Five scientific-contract tests passed before freezing the run. Independent projection and span replay passed for {projection_checks} maps, with maximum projection error {max_projection_error:.3g}. Both benchmark endpoints reproduce for all 19 arms. Frozen code and all 116 score/metadata hashes match. Direct source-label rejoins by ID match all {label_checks} evaluated answers, and the 29 raw column names/order match the upstream schema. `REVIEW.json` records these checks; the audit does not prove that mixture states are correctness states.", "",
              "Scoring finished in about 206 seconds using three CPU workers, with per-answer checkpoints. Labels were decoded only in the later evaluation phase. The original pilots, Claude's worktree and prior frozen benchmark releases were preserved.", "",
              "The historical 30-long-answer pilot's IU 0.70070 and common-24 graph 0.69188 are on different IDs from this stress test; the new cohort does not retroactively invalidate them or provide a direct delta. The historical recipe is recomputed on the new selected IDs as `legacy_fixed32`. Claude's pooled/calibrated PB numbers use another no-error protocol and belong in a separate lane. The 24-cell final-answer benchmark is a different target and remains a later frozen-candidate transfer experiment.", "",
              "## Next bounded work", "",
              "Freeze a chronological readout comparison over the saved window scores: retain this failed first-crossing baseline, include a simple peak locator and a fixed no-error control, then compare HMM/BOCPD and an explicit IMM/switching-filter adaptation. Keep feature-fusion weights fixed so the contribution is measurable. Do not label a Gaussian HMM as an IMM reproduction. Preserve KalmanNet, LOCA, Diverging Flows and graph token/window sampling in their separate registered follow-ups. Joint feature/hyperparameter work remains active, with no higher-K forcing or label-guided parameter selection.", "",
              "A publication winner still requires full matched benchmark coverage, a declared selection procedure and genuinely untouched confirmation on both tasks. This pilot has not achieved that objective.", ""]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
