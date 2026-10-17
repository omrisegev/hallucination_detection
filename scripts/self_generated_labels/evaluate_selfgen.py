"""Evaluate the sealed frozen-method scores on self-generated answers against judge consensus.

Declared in docs/experiments/SELF_GENERATED_STEP_LABELS_V1.md ("Frozen-method evaluation") before
this script was run. Labels: consensus first-error step = both judges give the same
`first_error_step` (-1 = no error); disagreements are excluded. Metrics, identical code for the
self-generated set and the teacher-forced ProcessBench reference:

  * ProcessBench F1 at the frozen thresholds: predicted first error = first nonempty step whose
    decision is "incorrect" (risk >= threshold), -1 if none; harmonic mean of exact-match accuracy
    on erroneous answers and no-flag accuracy on clean answers.
  * within-answer AUC: in erroneous answers with a preceding step, the first-error step against
    the steps before it (risk scores); mean over answers.
  * argmax accuracy: fraction of erroneous answers whose highest-risk step is the first error
    (threshold-free).
Reference: the same arms' cross-fitted source scores (VALIDATION.npz, per-fold calibration) on
ProcessBench GSM8K and MATH for Qwen3-8B and Qwen3-4B telemetry: text written by other models.
Bootstrap: 2000 draws over question groups, 95% percentile intervals, seed 20260929.
"""
import json
import os
from collections import defaultdict

import numpy as np

MAIN = os.environ.get("HD_MAIN_CHECKOUT", r"C:\Users\omris\TAU\hallucination_detection")
ROOT = "results/self_generated_step_labels_v1"
JUDGES = ("claude-opus-5.5", "gpt-6-sol")
ARMS = ("frozen_lsml", "frozen_equal", "frozen_partition_equal",
        "local_lsml", "local_equal", "local_partition_equal", "ct7")
PB_CELLS = {"pb_gsm8k_q8": ("gsm8k", "Qwen3-8B"), "pb_gsm8k_q4": ("gsm8k", "Qwen3-4B"),
            "pb_math_q8": ("math", "Qwen3-8B"), "pb_math_q4": ("math", "Qwen3-4B")}
DRAWS, SEED = 2000, 20260929


def answer_stats(risk, pred_correct, nonempty, label):
    """Per-answer contributions: (is_error, exact_hit, clean_hit, within_auc or nan, argmax_hit)."""
    risk = np.asarray(risk, float)
    ok = np.asarray(nonempty, bool)
    flagged = [i for i in range(len(risk)) if ok[i] and not pred_correct[i]]
    pred_first = flagged[0] if flagged else -1
    if label == -1:
        return 0, 0, int(pred_first == -1), np.nan, 0
    exact = int(pred_first == label)
    before = [i for i in range(label) if ok[i]]
    if before and ok[label]:
        wins = sum((risk[label] > risk[i]) + 0.5 * (risk[label] == risk[i]) for i in before)
        within = wins / len(before)
    else:
        within = np.nan
    valid = np.where(ok, risk, -np.inf)
    return 1, exact, 0, within, int(int(np.argmax(valid)) == label)


def summarize(rows):
    rows = np.asarray(rows, float)  # columns: is_error, exact, clean_hit, within, argmax
    err = rows[:, 0] == 1
    acc_e = rows[err, 1].mean() if err.any() else np.nan
    acc_c = rows[~err, 2].mean() if (~err).any() else np.nan
    f1 = 2 * acc_e * acc_c / (acc_e + acc_c) if acc_e + acc_c > 0 else 0.0
    w = rows[err, 3]
    return {"pb_f1": 100 * f1, "acc_error_exact": acc_e, "acc_clean": acc_c,
            "within_auc": np.nanmean(w) if np.isfinite(w).any() else np.nan,
            "argmax_acc": rows[err, 4].mean() if err.any() else np.nan,
            "n_error": int(err.sum()), "n_clean": int((~err).sum()),
            "n_within": int(np.isfinite(w).sum())}


def with_ci(stats_by_group):
    groups = list(stats_by_group)
    point = summarize([r for g in groups for r in stats_by_group[g]])
    rng = np.random.default_rng(SEED)
    boots = defaultdict(list)
    for _ in range(DRAWS):
        pick = rng.integers(len(groups), size=len(groups))
        s = summarize([r for i in pick for r in stats_by_group[groups[i]]])
        for k in ("pb_f1", "within_auc", "argmax_acc"):
            boots[k].append(s[k])
    for k, v in boots.items():
        v = np.asarray(v)
        v = v[np.isfinite(v)]
        point[k + "_ci95"] = [float(np.quantile(v, .025)), float(np.quantile(v, .975))] if len(v) else None
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in point.items()}


def consensus_labels():
    lab = {}
    for j in JUDGES:
        d = f"{ROOT}/labels/{j}"
        for name in sorted(os.listdir(d)):
            if name.startswith("shard_"):
                for line in open(f"{d}/{name}", encoding="utf-8"):
                    x = json.loads(line)
                    lab.setdefault(x["item_id"], {})[j] = x["first_error_step"]
    return {i: v[JUDGES[0]] for i, v in lab.items() if v[JUDGES[0]] == v[JUDGES[1]]}


def selfgen_panel(cons, key):
    scores = [json.loads(l) for l in open(f"{ROOT}/scoring/SCORES.jsonl", encoding="utf-8")]
    out = {}
    for arm in ARMS:
        by = defaultdict(lambda: defaultdict(list))
        for s in scores:
            i = s["item_id"]
            if i not in cons:
                continue
            k = key[i]
            model = "Qwen3-8B" if k["cell"].endswith("8b") else "Qwen3-4B"
            st = answer_stats(s["scores"][arm], s["predictions"][arm], s["nonempty"], cons[i])
            grp = f"{k['dataset']}::{k['src_idx']}"
            for sub in ("all", k["dataset"], model, f"{k['dataset']}/{model}"):
                by[sub][grp].append(st)
        out[arm] = {sub: with_ci(g) for sub, g in by.items()}
    return out


def pb_panel():
    joined = json.load(open(os.path.join(MAIN, "results/localization_full_benchmark_v3/evaluation/JOINED.json")))
    target = np.load(os.path.join(MAIN, "results/localization_full_benchmark_v3/evaluation/JOINED.npz"))["target"]
    v = np.load(os.path.join(MAIN, "results/lsml_external_generalization_v1/evaluation/source/VALIDATION.npz"))
    off = v["offsets"]
    out = {}
    for arm in ARMS:
        sc, pr = v[arm + "_score"], v[arm + "_pred"]
        by = defaultdict(lambda: defaultdict(list))
        for r in joined["records"]:
            if r["cell"] not in PB_CELLS:
                continue
            a = r["row"]
            risk, pred = sc[off[a]:off[a + 1]], pr[off[a]:off[a + 1]]
            nonempty = np.isfinite(risk)
            st = answer_stats(np.nan_to_num(risk, nan=-np.inf), pred, nonempty, int(target[a]))
            ds, model = PB_CELLS[r["cell"]]
            for sub in ("all", ds, model, f"{ds}/{model}"):
                by[sub][r["group_id"]].append(st)
        out[arm] = {sub: with_ci(g) for sub, g in by.items()}
    return out


def main():
    key = {k["item_id"]: k for k in map(json.loads, open(f"{ROOT}/private/ITEM_KEY.jsonl", encoding="utf-8"))}
    cons = {i: l for i, l in consensus_labels().items() if key[i]["source"] == "own"}
    res = {"labels": {"consensus_own_answers": len(cons),
                      "consensus_error": sum(l != -1 for l in cons.values()),
                      "consensus_clean": sum(l == -1 for l in cons.values())},
           "self_generated_on_policy": selfgen_panel(cons, key),
           "processbench_teacher_forced_reference": pb_panel(),
           "bootstrap": {"draws": DRAWS, "seed": SEED, "unit": "question group"}}
    os.makedirs(f"{ROOT}/analysis", exist_ok=True)
    json.dump(res, open(f"{ROOT}/analysis/FROZEN_METHODS_V1.json", "w"), indent=1)
    print(json.dumps(res["labels"]))
    print(f"{'arm':24s} {'self-gen F1':>12s} {'PB-TF F1':>9s} {'self-gen within':>16s} {'PB-TF within':>13s} {'self argmax':>12s} {'PB argmax':>10s}")
    for arm in ARMS:
        a, b = res["self_generated_on_policy"][arm]["all"], res["processbench_teacher_forced_reference"][arm]["all"]
        print(f"{arm:24s} {a['pb_f1']:12.2f} {b['pb_f1']:9.2f} {a['within_auc']:16.4f} {b['within_auc']:13.4f} "
              f"{a['argmax_acc']:12.4f} {b['argmax_acc']:10.4f}")


if __name__ == "__main__":
    main()
