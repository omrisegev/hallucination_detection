"""Seal every cell, then evaluate the locked ten-arm exploratory follow-up."""
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
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spectral_utils.external_generalization.artifacts import atomic_json, file_hash
from spectral_utils.external_generalization.contracts import Answer, digest, overlap_manifest, question_key
from spectral_utils.external_generalization.evaluation import confusion, within_auc
from spectral_utils.family_external_metrics import summary, bootstrap, contrast
from spectral_utils.label_sanity import check_labels, feasibility_tag
CELLS = {"hard2verify_qwen3_8b": "hard2verify", "socratic_qwen3_8b": "socratic",
         "socratic_qwq32b": "socratic"}


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
    """Replay full evaluator and all categories, including its degenerate sentinels."""
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
        out[arm] = dict(pass_=True, max_component_category_error=max_error,
                       categories=len(result["categories"]), official_components=o["total_hallucination_results"])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT/"results/family_tail_external_v1")
    parser.add_argument("--inputs", type=Path, default=ROOT/"scratch/external_generalization_private/inputs")
    parser.add_argument("--seal-only", action="store_true")
    args = parser.parse_args()
    out = args.root
    plan = load(out/"EXECUTION_PLAN.json")
    lockpath = ROOT/"results/family_tail_transfer_v1/TRANSFER_LOCK_V1.json"
    lock = load(lockpath)
    if file_hash(lockpath) != plan["lock_sha256"]:
        raise ValueError("source lock changed")
    arms = list(lock["rows"])
    contrasts = lock["external_primary_contrasts"]
    if arms != plan["methods"] or len(contrasts)*len(CELLS) != plan["primary_family"]:
        raise ValueError("arm/contrast registry mismatch")
    freeze = load(out/"IMPLEMENTATION_FREEZE.json")
    for path, expected in freeze["files"].items():
        if file_hash(ROOT/path) != expected:
            raise ValueError("frozen code changed: "+path)
    for path in ("scripts/evaluate_family_external.py", "spectral_utils/family_external_metrics.py"):
        if path not in freeze["files"]:
            raise ValueError("analysis must be frozen before evaluation")
    execution = load(out/"CPU_EXECUTION.json")
    if execution["identity"]["implementation_freeze"] != file_hash(out/"IMPLEMENTATION_FREEZE.json"):
        raise ValueError("run used different implementation freeze")
    expected_identity = digest(execution["identity"])
    answers = {b: [Answer(**{**r, "steps": tuple(r["steps"])}) for r in load(args.inputs/b/"answers.json")]
               for b in set(CELLS.values())}
    all_rows, seals = {}, {}
    for cell, bench in CELLS.items():
        rows = {}
        for path in sorted((out/cell).glob("shard_*/*.record.json")):
            if (path.parent/"WRITER.lock").exists():
                raise ValueError("active writer")
            record = load(path)
            if record["run_identity"] != expected_identity or load(path.parent/"RUN.json")["identity"] != expected_identity:
                raise ValueError("mixed run identity")
            if record["uid"] in rows:
                raise ValueError("duplicate answer")
            rows[record["uid"]] = record["payload"]
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
        seal = dict(prediction_sha256=digest(rows), lock_sha256=file_hash(lockpath), answers=len(rows), arms=arms,
                    implementation_freeze_sha256=file_hash(out/"IMPLEMENTATION_FREEZE.json"))
        if (out/cell/"SEAL.json").exists() and load(out/cell/"SEAL.json") != seal:
            raise ValueError("sealed predictions changed")
        atomic_json(out/cell/"SEAL.json", seal)
        atomic_json(out/cell/"PREDICTIONS.json", rows)
        all_rows[cell], seals[cell] = rows, seal
    atomic_json(out/"ALL_CELLS_SEALED.json", dict(seals=seals, lock_sha256=file_hash(lockpath)))
    if args.seal_only:
        return
    # No external annotations are read above this boundary.
    gold = {b: {g["uid"]: g for g in load(args.inputs/"evaluator_only"/(b+".json"))} for b in set(CELLS.values())}
    overlap = overlap_manifest(answers)
    devkeys = set()
    for name in ("QUESTION_METADATA.json", "PB_QUESTION_METADATA.json"):
        devkeys.update(r["question_whitespace_sha256"] for r in load(ROOT/"results/localization_source_group_audit_v1"/name)["rows"])
    excluded = {overlap["groups"][r.uid] for values in answers.values() for r in values
                if any(question_key(t) in devkeys for t in (r.question, r.original_question))}
    overlap["development_overlapping_groups"] = sorted(excluded)
    overlap["caveat"] = "Exact normalized question hashes and connected source-ID groups; not semantic or pretraining contamination exclusion."
    atomic_json(out/"OVERLAP.json", overlap)
    source = ROOT/"scratch/external_generalization_private/sources"
    hp, sp = source/"hard2verify/utils.py", source/"prmeval_classified_task.py"
    previous = load(ROOT/"results/lsml_external_generalization_v1/evaluation/OFFICIAL_METRIC_REPLAY.json")
    for path in (hp, sp):
        if previous["sources"][str(path.relative_to(ROOT))]["sha256"] != file_hash(path):
            raise ValueError("official evaluator source changed")
    official = {"hard": pinned_functions(hp, {"calculate_metrics"}, {"recall_score": recall_score}),
                "soc": pinned_functions(sp, {"evaluate_function", "eval_on_hallucination_step"}, {})}
    all_metrics, comparisons, disjoint_comparisons, official_result = {}, [], [], {}
    for cell, bench in CELLS.items():
        rows = all_rows[cell]
        uids = [r.uid for r in answers[bench]]
        if set(gold[bench]) != set(uids):
            raise ValueError("gold population mismatch")
        groups = [overlap["groups"][uid] for uid in uids]
        disjoint = np.array([g not in excluded for g in groups])
        sanity = check_labels([v for uid in uids for v, keep in zip(gold[bench][uid]["correct"], gold[bench][uid]["include"]) if keep])
        if not sanity.ok:
            raise ValueError(sanity.summary())
        counts = {arm: [] for arm in arms}
        aucs = {arm: [] for arm in arms}
        categories, lengths, positions = {}, {}, {}
        flawless = {arm: 0 for arm in arms}
        flawless_n = 0
        for uid in uids:
            g, r = gold[bench][uid], rows[uid]
            include = np.asarray(g["include"], bool)
            valid = include & np.asarray(r["nonempty"], bool)
            n = len(include)
            length = "1-4" if n <= 4 else "5-8" if n <= 8 else "9-16" if n <= 16 else "17+"
            cat = g["category"]
            for table, key in ((categories, cat), (lengths, length)):
                table.setdefault(key, {arm: [] for arm in arms})
            is_flawless = bool(np.asarray(g["correct"], bool)[include].all())
            flawless_n += is_flawless
            for arm in arms:
                c = confusion(g["correct"], r["predictions"][arm], include)
                counts[arm].append(c)
                categories[cat][arm].append(c)
                lengths[length][arm].append(c)
                for q in range(4):
                    key = f"Q{q+1}"
                    positions.setdefault(key, {a: [] for a in arms})
                    mask = include & (np.minimum(3, (np.arange(n)*4//max(n, 1))) == q)
                    positions[key][arm].append(confusion(g["correct"], r["predictions"][arm], mask))
                risk = np.asarray([np.nan if v is None else v for v in r["scores"][arm]])
                if not np.isfinite(risk[valid]).all():
                    raise ValueError("nonfinite score on valid step")
                value = within_auc(g["correct"], risk, valid)
                if value is not None:
                    aucs[arm].append(value)
                flawless[arm] += is_flawless and bool((np.asarray(r["predictions"][arm])[include] == 0).any())
        def panels(table):
            return {key: dict(answers=len(next(iter(value.values()))),
                              arms={arm: summary(c, bench) for arm, c in value.items()}) for key, value in table.items()}
        result = dict(benchmark=bench, answers=len(uids), n_checked=len(uids), n_total=len(answers[bench]),
                      flag=sanity.flag_string(), coverage_flag=feasibility_tag(len(uids),len(answers[bench])),
                      groups=len(set(groups)), disjoint_answers=int(disjoint.sum()), disjoint_groups=len(set(np.asarray(groups)[disjoint])),
                      steps=int(np.asarray(counts[arms[0]]).sum()), tokens=sum(r["tokens"] for r in rows.values()),
                      empty_steps=sum(np.count_nonzero(~np.asarray(r["nonempty"],bool)) for r in rows.values()),
                      process_cpu_seconds_sum=sum(r["process_cpu_seconds"] for r in rows.values()),
                      elapsed_record_seconds_sum=sum(r["cpu_seconds"] for r in rows.values()),
                      categories=panels(categories), length_bins=panels(lengths), relative_position=panels(positions),
                      flawless_answers=flawless_n, arms={})
        names, samples, sentinels = bootstrap(counts, groups, bench, plan["draws"], plan["seed"])
        result["bootstrap_sentinel_draws"] = sentinels
        for arm in arms:
            a = summary(counts[arm], bench)
            a.update(within_auc=float(np.mean(aucs[arm])) if aucs[arm] else None, within_auc_answers=len(aucs[arm]),
                     disjoint=summary(np.asarray(counts[arm])[disjoint],bench),
                     flawless_false_flag_answers=int(flawless[arm]), flawless_false_flag_rate=flawless[arm]/flawless_n if flawless_n else None,
                     ci95_descriptive=np.quantile(samples[:, names.index(arm)][np.isfinite(samples[:, names.index(arm)])],[.025,.975]).tolist())
            result["arms"][arm] = a
        result["decision_changes"] = {}
        for left, right in contrasts:
            changes = [np.asarray(rows[u]["predictions"][left]) != np.asarray(rows[u]["predictions"][right]) for u in uids]
            result["decision_changes"][left+"__vs__"+right] = dict(answers=sum(bool(v.any()) for v in changes), steps=sum(int(v.sum()) for v in changes))
            comparisons.append(dict(cell=cell, **contrast(counts,names,samples,left,right,groups,bench,plan["primary_family"],plan["seed"])))
        dc = {k: np.asarray(v)[disjoint] for k,v in counts.items()}
        dg = np.asarray(groups)[disjoint].tolist()
        if disjoint.all():
            dn, ds = names, samples
        else:
            dn, ds, _ = bootstrap(dc,dg,bench,plan["draws"],plan["seed"])
        for left,right in contrasts:
            disjoint_comparisons.append(dict(cell=cell,panel="observed_disjoint_sensitivity",
                                            **contrast(dc,dn,ds,left,right,dg,bench,plan["primary_family"],plan["seed"])))
        official_result[cell] = official_replay(rows,[gold[bench][u] for u in uids],result,bench,arms,official)
        np.savez_compressed(out/cell/"COUNTS.npz",**{k:np.asarray(v) for k,v in counts.items()},uids=np.array(uids),groups=np.array(groups))
        np.savez_compressed(out/cell/"BOOTSTRAP.npz",arms=np.array(names),metrics=samples)
        atomic_json(out/cell/"METRICS.json",safe_json(result))
        all_metrics[cell] = result
        print(cell, "completed", len(uids), "answers", flush=True)
    atomic_json(out/"METRICS.json",safe_json(all_metrics))
    atomic_json(out/"CONTRASTS.json",safe_json(comparisons))
    atomic_json(out/"DISJOINT_CONTRASTS.json",safe_json(disjoint_comparisons))
    atomic_json(out/"OFFICIAL_METRIC_REPLAY.json",dict(pass_=True,cells=official_result,
                sources={str(p.relative_to(ROOT)):file_hash(p) for p in (hp,sp)}))
    atomic_json(out/"EVALUATION_PROVENANCE.json",dict(complete=len(comparisons)==plan["primary_family"],
                access=plan["access"],draws=plan["draws"],seed=plan["seed"],family=plan["primary_family"],
                labels={b:file_hash(args.inputs/"evaluator_only"/(b+".json")) for b in gold},
                evaluator_sha256=file_hash(__file__),helper_sha256=file_hash(ROOT/"spectral_utils/family_external_metrics.py"),
                lock_sha256=file_hash(lockpath),implementation_freeze_sha256=file_hash(out/"IMPLEMENTATION_FREEZE.json"),
                all_cells_sealed_sha256=file_hash(out/"ALL_CELLS_SEALED.json")))


if __name__ == "__main__":
    main()
