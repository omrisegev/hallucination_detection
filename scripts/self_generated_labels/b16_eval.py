"""Does B16 (per model per dataset, BASE and GRP) behave the same on the model's own answers as on
teacher-forced ProcessBench? Declared in docs/experiments/SELF_GENERATED_STEP_LABELS_V1.md
("B16 per-dataset behaviour check") before this script was run.

Pairs: own GSM8K/MATH answers of Qwen3-4B/8B (own telemetry, judge-consensus labels) against
ProcessBench pb_gsm8k_q4/q8, pb_math_q4/q8 (Step 462 per-dataset scores, human labels).
Metrics, identical code on both sides, erroneous answers only:
  * first-error hit: earliest argmax step == first error (per_dataset_fit_run.py:274,291);
  * within-answer AUC: first-error step against the steps before it (answers with >= 1 earlier step).
Equivalence (declared): |own - ProcessBench| < 0.03 within-AUC and the 95% interval of the
difference contains 0. Independent bootstrap over question groups, 2000 draws, seed 20260929.
"""
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

MAIN = Path(os.environ.get("HD_MAIN_CHECKOUT", r"C:\Users\omris\TAU\hallucination_detection"))
SSL = MAIN / ".worktrees/ssl-pseudolabel-residual-v1"
ROOT = Path(__file__).resolve().parents[2] / "results/self_generated_step_labels_v1"
JUDGES = ("claude-opus-5.5", "gpt-6-sol")
PAIRS = {"gsm8k/Qwen3-4B": ("evdrop_gsm8k_qwen3_4b", "pb_gsm8k_q4"), "gsm8k/Qwen3-8B": ("evdrop_gsm8k_qwen3_8b", "pb_gsm8k_q8"),
         "math/Qwen3-4B": ("evdrop_math_qwen3_4b", "pb_math_q4"), "math/Qwen3-8B": ("evdrop_math_qwen3_8b", "pb_math_q8")}
ARMS = ("B16__BASE", "B16__GRP")
DRAWS, SEED, MARGIN = 2000, 20260929, 0.03


def earliest_argmax(v):
    return int(np.flatnonzero(v >= v.max() - 8 * np.finfo(float).eps)[0])


def answer_row(s, f):
    """(hit, within-AUC or nan) for one erroneous answer with first error f."""
    hit = float(earliest_argmax(s) == f)
    if f == 0:
        return hit, np.nan
    before = s[:f]
    return hit, float(((s[f] > before) + 0.5 * (s[f] == before)).mean())


def summarize(rows):
    r = np.asarray(rows, float)
    return r[:, 0].mean(), np.nanmean(r[:, 1]) if np.isfinite(r[:, 1]).any() else np.nan


def boot(groups, rng):
    g = list(groups)
    pick = rng.integers(len(g), size=len(g))
    return summarize([x for i in pick for x in groups[g[i]]])


def consensus():
    lab = defaultdict(dict)
    for j in JUDGES:
        for name in sorted(os.listdir(ROOT / "labels" / j)):
            if name.startswith("shard_"):
                for line in open(ROOT / "labels" / j / name, encoding="utf-8"):
                    x = json.loads(line)
                    lab[x["item_id"]][j] = x["first_error_step"]
    return {i: v[JUDGES[0]] for i, v in lab.items() if v[JUDGES[0]] == v[JUDGES[1]]}


def main():
    cons = consensus()
    fit = json.load(open(ROOT / "b16/FIT.json"))
    own_sc = np.load(ROOT / "b16/STEP_SCORES.npz")
    ooff = own_sc["offsets"]
    ans = pd.read_csv(MAIN / ".worktrees/readout-quickest-detection-v1/results/step_evidence_v1/OOF_ANSWERS.csv", encoding="utf-8-sig")
    pb_sc = np.load(SSL / "results/per_dataset_fit_v1/run_20260929/STEP_SCORES.npz")
    poff = pb_sc["offsets"]
    rng = np.random.default_rng(SEED)
    res = {"equivalence_margin_within_auc": MARGIN, "draws": DRAWS, "seed": SEED, "cells": {}}
    for arm in ARMS:
        pooled = {"own": {}, "pb": {}}
        for tag, (cell, pbcell) in PAIRS.items():
            own = defaultdict(list)
            for i, m in enumerate(fit["answers"]):
                if m["cell"] != cell or m["item_id"] not in cons or cons[m["item_id"]] < 0:
                    continue
                own[f"{cell}::{m['src_idx']}"].append(answer_row(own_sc[arm][ooff[i]:ooff[i + 1]], cons[m["item_id"]]))
            pb = defaultdict(list)
            for i in np.flatnonzero((ans.cell == pbcell).to_numpy()):
                t = int(ans.target.iloc[i])
                if t >= 0:
                    pb[ans.source_group.iloc[i]].append(answer_row(pb_sc[arm][poff[i]:poff[i + 1]], t))
            pooled["own"].update(own); pooled["pb"].update({f"{pbcell}::{k}": v for k, v in pb.items()})
            res["cells"].setdefault(tag, {})[arm] = compare(own, pb, rng)
        res.setdefault("pooled", {})[arm] = compare(pooled["own"], pooled["pb"], rng)
    json.dump(res, open(ROOT / "analysis/B16_BEHAVIOUR_V1.json", "w"), indent=1)
    for arm in ARMS:
        print(arm)
        for tag in list(PAIRS) + ["pooled"]:
            c = res["pooled"][arm] if tag == "pooled" else res["cells"][tag][arm]
            print(f"  {tag:16s} within own {c['own_within']:.3f} pb {c['pb_within']:.3f} d {c['d_within']:+.3f} {np.round(c['d_within_ci95'], 3)} "
                  f"| hit own {c['own_hit']:.3f} pb {c['pb_hit']:.3f} d {c['d_hit']:+.3f} {np.round(c['d_hit_ci95'], 3)} "
                  f"| n {c['own_n']}/{c['pb_n']} | equivalent: {c['equivalent_within']}")


def compare(own, pb, rng):
    oh, ow = summarize([x for v in own.values() for x in v])
    ph, pw = summarize([x for v in pb.values() for x in v])
    dh, dw = [], []
    for _ in range(DRAWS):
        a, b = boot(own, rng), boot(pb, rng)
        dh.append(a[0] - b[0]); dw.append(a[1] - b[1])
    ci = lambda d: [float(np.nanquantile(d, .025)), float(np.nanquantile(d, .975))]
    cw = ci(dw)
    return {"own_hit": oh, "pb_hit": ph, "d_hit": oh - ph, "d_hit_ci95": ci(dh),
            "own_within": ow, "pb_within": pw, "d_within": ow - pw, "d_within_ci95": cw,
            "own_n": sum(map(len, own.values())), "pb_n": sum(map(len, pb.values())),
            "equivalent_within": bool(abs(ow - pw) < MARGIN and cw[0] <= 0 <= cw[1])}


if __name__ == "__main__":
    main()
