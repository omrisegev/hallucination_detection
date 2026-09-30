"""PRMScore adapter: official PRMBench evaluator on our frozen step risks (all 6,969 rows).

Uses the existing port of `prmtest_classified/task.py::evaluate_function`
(`spectral_utils.prmbench.prmbench_evaluate`) with metadata from the cached PRM run
(`dataset_cache/four_localization/prmbench_qwen25math7b_full/prmbench_prm.pkl`, 6,969 rows,
ids identical to the benchmark's PRMBench records). The evaluator wants, per row, a binary
label per step (1 = the scorer asserts the step is VALID). Our methods output a step RISK, so
the adapter needs one threshold: risk >= tau -> step invalid.

tau rules (all reported):
  nested_labels : per outer source-group fold, tau maximizing PRMScore on the other folds.
  quantile_q    : label-free, tau = q-quantile of pooled step risks over the training folds.
  within_top1   : label-free, answer-local: only the argmax step is flagged (first-error-style).
PRMScore = 0.5 * F1 + 0.5 * negative-F1 (PRMBench's default weights); the evaluator's per-
classification tables are kept. The supervised PRM's own rewards (< 0.5 -> invalid) give the
published-style reference row on the same evaluator. Development evidence only.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spectral_utils.prmbench import prmbench_evaluate  # noqa: E402

BENCH = ROOT / "results" / "localization_full_benchmark_v3"
SHRINK = ROOT / "results" / "fusion_shrinkage_iu_v1"
FOLDS = ROOT / "results" / "localization_source_group_audit_v1" / "FOLDS_V2.json"
PRM = ROOT / "dataset_cache" / "four_localization" / "prmbench_qwen25math7b_full" / "prmbench_prm.pkl"
OUT = ROOT / "results" / "prmscore_adapter_v1"
METHODS = ["dual__iu", "dual__equal", "entropy_parent", "context__iu", "full__joint__alw"]
QUANTILES = (0.7, 0.8, 0.9)


def prmscore(res):
    t = res["total"]
    return 0.5 * t["f1"] + 0.5 * t["negative_f1"]


def evaluate(pred_labels, meta):
    preds = [{"idx": idx, "labels": [int(x) for x in labs]} for idx, labs in pred_labels.items()]
    res = prmbench_evaluate(preds, meta)
    return dict(prmscore=prmscore(res), f1=res["total"]["f1"], negative_f1=res["total"]["negative_f1"],
                per_class={c: dict(f1=v.get("f1"), negative_f1=v.get("negative_f1")) for c, v in res["per_classification"].items()}
                if "per_classification" in res else None, raw_keys=list(res.keys()))


def main():
    OUT.mkdir(exist_ok=True)
    j = json.loads((BENCH / "evaluation" / "JOINED.json").read_text(encoding="utf-8"))
    z = np.load(BENCH / "evaluation" / "JOINED.npz")
    recs, arms = j["records"], j["arms"]
    offsets = z["offsets"]; SC = z["scores"]; VA = z["valid"]
    prm_rows = [i for i, r in enumerate(recs) if r["cell"].startswith("prm")]
    meta_all = list(pickle.load(open(PRM, "rb")).values())
    meta_by_idx = {m["idx"]: m for m in meta_all}
    folds = json.loads(FOLDS.read_text(encoding="utf-8"))
    outer = np.array([folds["outer"].get(recs[i]["group_id"], -1) for i in prm_rows])
    assert (outer >= 0).all()
    # step risks per method
    risks = {}
    for m in METHODS:
        risks[m] = {}
        if m in arms:
            c = arms.index(m)
            for i in prm_rows:
                if VA[i, c]:
                    risks[m][recs[i]["row_id"]] = SC[offsets[i]:offsets[i + 1], c]
        else:
            for i in prm_rows:
                p = SHRINK / "scores" / f"{recs[i]['uid']}.npz"
                if p.exists():
                    with np.load(p) as a:
                        if m + "__risk" in a.files:
                            risks[m][recs[i]["row_id"]] = a[m + "__risk"]
    ids = [recs[i]["row_id"] for i in prm_rows]
    for n, idx in enumerate(ids[:200]):
        assert int(meta_by_idx[idx]["n_steps"]) == recs[prm_rows[n]]["steps"], idx

    def labels_from(risk, tau):
        r = np.asarray(risk, float)
        return (~(r >= tau)).astype(int)            # 1 = valid

    def run_rule(m, rule, q=None):
        pred = {}
        taus = {}
        for k in sorted(set(outer.tolist())):
            train = [ids[n] for n in range(len(ids)) if outer[n] != k and ids[n] in risks[m]]
            test = [ids[n] for n in range(len(ids)) if outer[n] == k and ids[n] in risks[m]]
            pooled = np.concatenate([risks[m][t] for t in train])
            if rule == "quantile":
                tau = float(np.quantile(pooled, q))
            else:
                grid = np.quantile(pooled, np.linspace(0.5, 0.99, 50))
                best, tau = -1, None
                meta_train = [meta_by_idx[t] for t in train]
                for g in grid:
                    s = evaluate({t: labels_from(risks[m][t], g) for t in train}, meta_train)["prmscore"]
                    if s > best:
                        best, tau = s, float(g)
            taus[int(k)] = tau
            for t in test:
                pred[t] = labels_from(risks[m][t], tau)
        return pred, taus

    results = {}
    # supervised PRM reference on the same evaluator
    prm_pred = {m["idx"]: [int(float(x) >= 0.5) for x in (m["rewards"] if isinstance(m["rewards"], list) else json.loads(m["rewards"]))]
                for m in meta_all}
    results["qwen25_math_prm_7b_rewards>=0.5"] = evaluate(prm_pred, meta_all)
    print("PRM reference:", {k: round(v, 4) for k, v in results["qwen25_math_prm_7b_rewards>=0.5"].items() if isinstance(v, float)})
    for m in METHODS:
        if not risks[m]:
            print(m, "no scores"); continue
        meta_m = [meta_by_idx[t] for t in ids if t in risks[m]]
        # answer-local top-1 rule
        pred = {t: (np.arange(len(risks[m][t])) != int(np.argmax(risks[m][t]))).astype(int) for t in risks[m]}
        results[f"{m}|within_top1"] = evaluate(pred, meta_m)
        for q in QUANTILES:
            pred, taus = run_rule(m, "quantile", q); results[f"{m}|quantile_{q}"] = dict(**evaluate(pred, meta_m), taus=taus)
        pred, taus = run_rule(m, "labels"); results[f"{m}|nested_labels"] = dict(**evaluate(pred, meta_m), taus=taus)
        for k, v in results.items():
            if k.startswith(m + "|"):
                print(f"{k:34s} PRMScore {v['prmscore']:.4f}  F1 {v['f1']:.4f}  negF1 {v['negative_f1']:.4f}  rows {len(meta_m)}")
    json.dump(results, open(OUT / "METRICS.json", "w"), indent=1, default=float)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
