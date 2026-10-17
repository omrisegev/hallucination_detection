"""Analysis steps 1-3 of docs/experiments/SELF_GENERATED_STEP_LABELS_V1.md (declared before labels).

1. Validity: each judge (and their exact-agreement consensus) against ProcessBench human labels
   on the 240 hidden validation items; ProcessBench F1 = harmonic mean of error-item and
   no-error-item accuracy, per subset.
2. Agreement on own answers: exact first-error agreement, both -1, Cohen's kappa on
   error/no-error, step distance when both flag an error.
3. Final-answer calls against the inference-time grader.
Writes results/self_generated_step_labels_v1/analysis/AGREEMENT_V1.json. No detection method
is scored here.
"""
import json
import os
from collections import Counter

ROOT = "results/self_generated_step_labels_v1"
JUDGES = ("claude-opus-5.5", "gpt-6-sol")


def load_labels(judge):
    out = {}
    d = f"{ROOT}/labels/{judge}"
    for name in sorted(os.listdir(d)):
        if name.startswith("shard_") and name.endswith(".jsonl"):
            for line in open(f"{d}/{name}", encoding="utf-8"):
                if line.strip():
                    lab = json.loads(line)
                    out[lab["item_id"]] = lab
    return out


def f1(a, b):
    return 0.0 if a + b == 0 else 2 * a * b / (a + b)


def kappa(x, y):
    n = len(x)
    po = sum(a == b for a, b in zip(x, y)) / n
    px, py = sum(x) / n, sum(y) / n
    pe = px * py + (1 - px) * (1 - py)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def main():
    key = {k["item_id"]: k for k in map(json.loads, open(f"{ROOT}/private/ITEM_KEY.jsonl", encoding="utf-8"))}
    L = {j: load_labels(j) for j in JUDGES}
    for j in JUDGES:
        assert set(L[j]) == set(key), j
    fe = {j: {i: L[j][i]["first_error_step"] for i in key} for j in JUDGES}
    cons = {i: fe[JUDGES[0]][i] for i in key if fe[JUDGES[0]][i] == fe[JUDGES[1]][i]}
    res = {"n_items": len(key)}

    # 1. validity on ProcessBench
    val = {}
    for subset in ("gsm8k", "math", "all"):
        ids = [i for i, k in key.items() if k["source"] == "processbench" and subset in ("all", k["subset"])]
        err = [i for i in ids if key[i]["pb_label"] != -1]
        ok = [i for i in ids if key[i]["pb_label"] == -1]
        row = {"n_error": len(err), "n_no_error": len(ok)}
        for name, pred in list(fe.items()) + [("consensus", cons)]:
            e_ids = [i for i in err if i in pred]
            o_ids = [i for i in ok if i in pred]
            acc_e = sum(pred[i] == key[i]["pb_label"] for i in e_ids) / len(e_ids)
            acc_o = sum(pred[i] == -1 for i in o_ids) / len(o_ids)
            row[name] = {"acc_error_exact": round(acc_e, 4), "acc_no_error": round(acc_o, 4),
                         "pb_f1": round(100 * f1(acc_e, acc_o), 2),
                         "coverage": len(e_ids) + len(o_ids),
                         "error_flagged_but_wrong_step": sum(pred[i] not in (-1, key[i]["pb_label"]) for i in e_ids),
                         "error_missed": sum(pred[i] == -1 for i in e_ids)}
        val[subset] = row
    res["validity_vs_processbench_humans"] = val

    # 2. agreement on own answers
    agr = {}
    for group in ("all", "auto_wrong", "auto_correct", "gsm8k", "math"):
        ids = [i for i, k in key.items() if k["source"] == "own" and (
            group == "all" or (group == "auto_wrong" and not k["auto_label_correct"])
            or (group == "auto_correct" and k["auto_label_correct"]) or group == k.get("dataset"))]
        a = [fe[JUDGES[0]][i] for i in ids]
        b = [fe[JUDGES[1]][i] for i in ids]
        both_err = [(x, y) for x, y in zip(a, b) if x != -1 and y != -1]
        dist = Counter(min(abs(x - y), 3) for x, y in both_err)
        agr[group] = {
            "n": len(ids),
            "exact_first_error_agreement": round(sum(x == y for x, y in zip(a, b)) / len(ids), 4),
            "both_no_error": sum(x == -1 and y == -1 for x, y in zip(a, b)),
            "both_error": len(both_err),
            "only_" + JUDGES[0] + "_error": sum(x != -1 and y == -1 for x, y in zip(a, b)),
            "only_" + JUDGES[1] + "_error": sum(x == -1 and y != -1 for x, y in zip(a, b)),
            "kappa_error_vs_no_error": round(kappa([x != -1 for x in a], [y != -1 for y in b]), 4),
            "both_error_step_distance": {("3+" if k == 3 else str(k)): v for k, v in sorted(dist.items())},
            "consensus_items": sum(x == y for x, y in zip(a, b)),
        }
    res["agreement_own_answers"] = agr

    # 3. final-answer calls vs inference-time grader
    fa = {}
    own = [i for i, k in key.items() if k["source"] == "own"]
    for j in JUDGES:
        c = Counter((key[i]["auto_label_correct"], L[j][i]["final_answer_matches_reference"]) for i in own)
        fa[j] = {"grader_correct_judge_correct": c[(True, True)], "grader_correct_judge_wrong": c[(True, False)],
                 "grader_wrong_judge_correct": c[(False, True)], "grader_wrong_judge_wrong": c[(False, False)],
                 "reference_suspect": sum(L[j][i]["reference_suspect"] for i in key)}
    both_disagree = [i for i in own if all(L[j][i]["final_answer_matches_reference"] != key[i]["auto_label_correct"] for j in JUDGES)]
    fa["both_judges_disagree_with_grader"] = both_disagree
    res["final_answer_vs_grader"] = fa

    os.makedirs(f"{ROOT}/analysis", exist_ok=True)
    with open(f"{ROOT}/analysis/AGREEMENT_V1.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
