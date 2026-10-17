"""algorithm_external_v1 evaluator: seal every cell's predictions (no label access), then evaluate the 23 arms.
Minimal-diff adaptation of scripts/evaluate_family_external_v3.py (reviewed): same confusion, official-metric components,
source-question bootstrap, official evaluator replay, overlap / development-disjoint panel and seal logic.  Changes: the run
layout (PREDICTIONS_UNSEALED.json written by scripts/score_algorithm_external_v1.py), the protocol's contrasts (primary family
24, Bonferroni; secondary at 95%), and a within-answer AUC paired bootstrap per contrast.
Usage: python scripts/evaluate_algorithm_external_v1.py --run results/algorithm_external_v1/run_20260929 [--seal-only]"""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
import ast
import json
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import recall_score
ROOT = Path(__file__).resolve().parents[1]; MAIN = Path(r"C:\Users\omris\TAU\hallucination_detection")
sys.path.insert(0, str(ROOT))
from spectral_utils.external_generalization.artifacts import atomic_json, file_hash
from spectral_utils.external_generalization.contracts import Answer, digest, overlap_manifest, question_key
from spectral_utils.external_generalization.evaluation import confusion, within_auc
from spectral_utils.family_external_metrics import summary, bootstrap, contrast
from spectral_utils.label_sanity import check_labels, feasibility_tag
CELLS = {"hard2verify_qwen3_8b": "hard2verify", "socratic_qwen3_8b": "socratic", "socratic_qwq32b": "socratic"}
BANKS = ["B16", "B23", "B35", "B54"]
PRIMARY = [pair for bk in BANKS for pair in ((f"F_{bk}_BASE", "ct7"), (f"F_{bk}_GRP", f"F_{bk}_BASE"))]
SECONDARY = [pair for bk in BANKS for pair in ((f"F_{bk}_BASE", "frozen_equal"), (f"F_{bk}_BASE", "frozen_lsml"), (f"R_{bk}_BASE", f"F_{bk}_BASE"),
                                                (f"R_{bk}_GRP", f"R_{bk}_BASE"), (f"F_{bk}_SW", f"F_{bk}_BASE"))]
DRAWS, SEED, AUC_DRAWS, AUC_SEED = 100000, 20260924, 20000, 20260930


def load(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, np.generic):
        return safe_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def pinned_functions(path, names, namespace):
    nodes = [n for n in ast.parse(path.read_text(encoding="utf8")).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError("official functions missing")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def official_replay(rows, gold, result, benchmark, arms, official):
    """Replay the full official evaluator and all categories (unchanged from the V3 evaluator)."""
    out = {}
    for arm in arms:
        if benchmark == "hard2verify":
            y = [int(v) for g in gold for v, keep in zip(g["correct"], g["include"]) if keep]
            p = [int(v) for g in gold for v, keep in zip(rows[g["uid"]]["predictions"][arm], g["include"]) if keep]
            value = official["hard"]["calculate_metrics"](p, y)["balanced_f1_score"]
            if value != round(100 * result["arms"][arm]["metric"], 2):
                raise ValueError("Hard2 official mismatch")
            out[arm] = {"primary_percent_rounded": value, "pass": True}
            continue
        if not all(all(g["include"]) for g in gold):
            raise ValueError("Socratic official replay requires original inclusion")
        meta, predictions = [], []
        for g in gold:
            errors = [i+1 for i, c in enumerate(g["correct"]) if not c] + g.get("out_of_range_error_indices", [])
            meta.append(dict(idx=g["uid"], classification=g["category"], error_steps=errors))
            predictions.append(dict(idx=g["uid"], scores={"step_level_validity_labels": rows[g["uid"]]["predictions"][arm]}))
        o = official["soc"]["evaluate_function"](predictions, meta)
        mapping = {"precision": "precision_correct", "recall": "recall_correct", "f1": "f1_correct",
                   "negative_precision": "precision_error", "negative_recall": "recall_error", "negative_f1": "f1_error"}
        max_error = 0.0
        for official_name, our_name in mapping.items():
            max_error = max(max_error, abs(o["total_hallucination_results"][official_name] - result["arms"][arm][our_name]))
            for category, value in o["hallucination_type_results"][official_name].items():
                max_error = max(max_error, abs(value - result["categories"][category]["arms"][arm][our_name]))
        if max_error > 1e-12:
            raise ValueError("Socratic official component/category mismatch")
        out[arm] = dict(pass_=True, max_component_category_error=max_error, categories=len(result["categories"]))
    return out


def auc_contrast(aucs, left, right, groups):
    """Paired source-question bootstrap of the within-answer AUC difference (answers where both are defined)."""
    a, b = np.asarray(aucs[left], float), np.asarray(aucs[right], float); ok = np.isfinite(a) & np.isfinite(b)
    dx = (a - b)[ok]; g = np.asarray(groups)[ok]
    unique, index = np.unique(g, return_inverse=True); s = np.bincount(index, weights=dx, minlength=len(unique)); c = np.bincount(index, minlength=len(unique)).astype(float)
    rng = np.random.default_rng(AUC_SEED); out = np.empty(AUC_DRAWS)
    for start in range(0, AUC_DRAWS, 500):
        W = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)), size=min(500, AUC_DRAWS - start)).astype(float)
        out[start:start + len(W)] = (W @ s) / (W @ c)
    return dict(left=left, right=right, delta=float(dx.mean()), ci95=np.quantile(out, [.025, .975]).tolist(), answers=int(ok.sum()), groups=len(unique), draws=AUC_DRAWS, seed=AUC_SEED)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, default=MAIN/"scratch/external_generalization_private/inputs")
    parser.add_argument("--seal-only", action="store_true")
    args = parser.parse_args()
    out = args.run; gen = out.parent
    protocol, bundle, scoring = gen/"PROTOCOL.json", gen/"BUNDLE.json", out/"SCORING_STATUS.json"
    if load(protocol)["status"] != "FROZEN" or load(bundle)["protocol_sha256"] != file_hash(protocol):
        raise ValueError("protocol not frozen or changed")
    status = load(scoring)
    if status["bundle_sha256"] != file_hash(bundle) or status.get("external_labels_opened") is not False:
        raise ValueError("scoring used another bundle or touched labels")
    if status["code"]["score_script_sha256"] != file_hash(ROOT/"scripts/score_algorithm_external_v1.py") or load(bundle)["code"]["fit_script_sha256"] != file_hash(ROOT/"scripts/fit_algorithm_external_v1.py"):
        raise ValueError("fit or scoring script changed since it produced the bundle / predictions")
    arms = status["cells"][next(iter(CELLS))]["arms"]
    if any(status["cells"][c]["arms"] != arms for c in CELLS):
        raise ValueError("arm registry differs between cells")
    family = len(PRIMARY) * len(CELLS)
    if family != 24 or any(a not in arms for pair in PRIMARY + SECONDARY for a in pair):
        raise ValueError("contrast registry mismatch")
    answers = {b: [Answer(**{**r, "steps": tuple(r["steps"])}) for r in load(args.inputs/b/"answers.json")] for b in set(CELLS.values())}
    all_rows, seals = {}, {}
    for cell, bench in CELLS.items():
        rows = load(out/cell/"PREDICTIONS_UNSEALED.json")
        if set(rows) != {r.uid for r in answers[bench]}:
            raise ValueError("incomplete population: "+cell)
        for answer in answers[bench]:
            row = rows[answer.uid]
            if set(row["scores"]) != set(arms) or set(row["predictions"]) != set(arms):
                raise ValueError("missing or extra arm")
            for arm in arms:
                if len(row["scores"][arm]) != len(answer.steps) or len(row["predictions"][arm]) != len(answer.steps):
                    raise ValueError("step alignment")
                if not set(row["predictions"][arm]) <= {0, 1}:
                    raise ValueError("nonbinary decision")
                ne = np.asarray(row["nonempty"], bool); sc = np.asarray([np.nan if v is None else v for v in row["scores"][arm]], float)
                if not np.isfinite(sc[ne]).all():
                    raise ValueError("nonfinite score on a non-empty step (checked before sealing)")
        feat = ROOT/"results/external_banks_v4"/cell
        seal = dict(prediction_sha256=digest(rows), protocol_sha256=file_hash(protocol), bundle_sha256=file_hash(bundle),
                    scoring_status_sha256=file_hash(scoring), features_sha256=file_hash(feat/"FEATURES.npz"), features_manifest_sha256=file_hash(feat/"MANIFEST.json"),
                    score_script_sha256=file_hash(ROOT/"scripts/score_algorithm_external_v1.py"), fit_script_sha256=file_hash(ROOT/"scripts/fit_algorithm_external_v1.py"),
                    answers=len(rows), arms=arms)
        if (out/cell/"SEAL.json").exists() and load(out/cell/"SEAL.json") != seal:
            raise ValueError("sealed predictions changed")
        atomic_json(out/cell/"SEAL.json", seal)
        all_rows[cell], seals[cell] = rows, seal
    atomic_json(out/"ALL_CELLS_SEALED.json", dict(seals=seals, protocol_sha256=file_hash(protocol)))
    if args.seal_only:
        print("sealed", {c: s["prediction_sha256"][:12] for c, s in seals.items()})
        return
    # No external annotations are read above this boundary.
    gold = {b: {g["uid"]: g for g in load(args.inputs/"evaluator_only"/(b+".json"))} for b in set(CELLS.values())}
    overlap = overlap_manifest(answers)
    devkeys = set()
    for name in ("QUESTION_METADATA.json", "PB_QUESTION_METADATA.json"):
        devkeys.update(r["question_whitespace_sha256"] for r in load(MAIN/"results/localization_source_group_audit_v1"/name)["rows"])
    excluded = {overlap["groups"][r.uid] for values in answers.values() for r in values
                if any(question_key(t) in devkeys for t in (r.question, r.original_question))}
    overlap["development_overlapping_groups"] = sorted(excluded)
    atomic_json(out/"OVERLAP.json", overlap)
    source = MAIN/"scratch/external_generalization_private/sources"
    hp, sp = source/"hard2verify/utils.py", source/"prmeval_classified_task.py"
    previous = load(MAIN/"results/lsml_external_generalization_v1/evaluation/OFFICIAL_METRIC_REPLAY.json")
    for path in (hp, sp):
        if previous["sources"][str(path.relative_to(MAIN))]["sha256"] != file_hash(path):
            raise ValueError("official evaluator source changed")
    official = {"hard": pinned_functions(hp, {"calculate_metrics"}, {"recall_score": recall_score}),
                "soc": pinned_functions(sp, {"evaluate_function", "eval_on_hallucination_step"}, {})}
    all_metrics, comparisons, auc_comparisons, disjoint_comparisons, official_result = {}, [], [], [], {}
    for cell, bench in CELLS.items():
        rows = all_rows[cell]; uids = [r.uid for r in answers[bench]]
        if set(gold[bench]) != set(uids):
            raise ValueError("gold population mismatch")
        groups = [overlap["groups"][uid] for uid in uids]
        disjoint = np.array([g not in excluded for g in groups])
        sanity = check_labels([v for uid in uids for v, keep in zip(gold[bench][uid]["correct"], gold[bench][uid]["include"]) if keep])
        if not sanity.ok:
            raise ValueError(sanity.summary())
        counts = {arm: [] for arm in arms}; aucs = {arm: [] for arm in arms}
        categories, lengths, positions = {}, {}, {}
        for uid in uids:
            g, r = gold[bench][uid], rows[uid]
            include = np.asarray(g["include"], bool); valid = include & np.asarray(r["nonempty"], bool); n = len(include)
            length = "1-4" if n <= 4 else "5-8" if n <= 8 else "9-16" if n <= 16 else "17+"
            cat = g["category"]
            for table, key in ((categories, cat), (lengths, length)):
                table.setdefault(key, {arm: [] for arm in arms})
            for arm in arms:
                c = confusion(g["correct"], r["predictions"][arm], include)
                counts[arm].append(c); categories[cat][arm].append(c); lengths[length][arm].append(c)
                for q in range(4):
                    key = f"Q{q+1}"; positions.setdefault(key, {a: [] for a in arms})
                    mask = include & (np.minimum(3, (np.arange(n)*4//max(n, 1))) == q)
                    positions[key][arm].append(confusion(g["correct"], r["predictions"][arm], mask))
                risk = np.asarray([np.nan if v is None else v for v in r["scores"][arm]])
                if not np.isfinite(risk[valid]).all():
                    raise ValueError("nonfinite score on valid step")
                value = within_auc(g["correct"], risk, valid)
                aucs[arm].append(np.nan if value is None else value)
        def panels(table):
            return {key: dict(answers=len(next(iter(value.values()))), arms={arm: summary(c, bench) for arm, c in value.items()}) for key, value in table.items()}
        result = dict(benchmark=bench, answers=len(uids), n_checked=len(uids), n_total=len(answers[bench]), flag=sanity.flag_string(),
                      coverage_flag=feasibility_tag(len(uids), len(answers[bench])), groups=len(set(groups)), disjoint_answers=int(disjoint.sum()),
                      steps=int(np.asarray(counts[arms[0]]).sum()), categories=panels(categories), length_bins=panels(lengths),
                      relative_position=panels(positions), arms={})
        names, samples, sentinels = bootstrap(counts, groups, bench, DRAWS, SEED)
        result["bootstrap_sentinel_draws"] = sentinels
        for arm in arms:
            a = summary(counts[arm], bench); au = np.asarray(aucs[arm], float)
            a.update(within_auc=float(np.nanmean(au)) if np.isfinite(au).any() else None, within_auc_answers=int(np.isfinite(au).sum()),
                     disjoint=summary(np.asarray(counts[arm])[disjoint], bench),
                     ci95_descriptive=np.quantile(samples[:, names.index(arm)][np.isfinite(samples[:, names.index(arm)])], [.025, .975]).tolist())
            result["arms"][arm] = a
        for panel, pairs, fam in (("primary", PRIMARY, family), ("secondary", SECONDARY, 1)):
            for left, right in pairs:
                comparisons.append(dict(cell=cell, panel=panel, **contrast(counts, names, samples, left, right, groups, bench, fam, SEED)))
                auc_comparisons.append(dict(cell=cell, panel=panel, **auc_contrast(aucs, left, right, groups)))
        dc = {k: np.asarray(v)[disjoint] for k, v in counts.items()}; dg = np.asarray(groups)[disjoint].tolist()
        dn, ds = (names, samples) if disjoint.all() else bootstrap(dc, dg, bench, DRAWS, SEED)[:2]
        for left, right in PRIMARY:
            disjoint_comparisons.append(dict(cell=cell, panel="development_disjoint_sensitivity", **contrast(dc, dn, ds, left, right, dg, bench, family, SEED)))
        official_result[cell] = official_replay(rows, [gold[bench][u] for u in uids], result, bench, arms, official)
        np.savez_compressed(out/cell/"COUNTS.npz", **{k: np.asarray(v) for k, v in counts.items()}, uids=np.array(uids), groups=np.array(groups))
        np.savez_compressed(out/cell/"ANSWER_AUC.npz", **{k: np.asarray(v, float) for k, v in aucs.items()})
        atomic_json(out/cell/"METRICS.json", safe_json(result)); all_metrics[cell] = result
        print(cell, "completed", len(uids), "answers", flush=True)
    atomic_json(out/"METRICS.json", safe_json(all_metrics))
    atomic_json(out/"CONTRASTS.json", safe_json(comparisons)); atomic_json(out/"WITHIN_AUC_CONTRASTS.json", safe_json(auc_comparisons))
    atomic_json(out/"DISJOINT_CONTRASTS.json", safe_json(disjoint_comparisons))
    atomic_json(out/"OFFICIAL_METRIC_REPLAY.json", dict(pass_=True, cells=official_result, sources={str(p.relative_to(MAIN)): file_hash(p) for p in (hp, sp)}))
    atomic_json(out/"EVALUATION_PROVENANCE.json", dict(primary_family=family, draws=DRAWS, seed=SEED, auc_draws=AUC_DRAWS,
                labels={b: file_hash(args.inputs/"evaluator_only"/(b+".json")) for b in gold}, evaluator_sha256=file_hash(Path(__file__)),
                all_cells_sealed_sha256=file_hash(out/"ALL_CELLS_SEALED.json")))


if __name__ == "__main__":
    main()
