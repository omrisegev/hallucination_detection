"""own_error_types_v1: per-error-type localization of the label-free B16 score on Qwen3's own answers.

Usage:  python analyze.py gate     -> reproduction gate only (no category labels read)
        python analyze.py          -> merge labels, agreement, per-category metrics, figure

Metric definitions mirror scripts/self_generated_labels/b16_eval.py (worktree self-generated-step-labels-v1):
hit = earliest argmax == first error; source within-AUC = first-error step vs the steps before it (ties .5,
step-0 errors NaN). Secondary within-AUC = first-error step vs all other steps. Bootstrap over source groups
(cell::src_idx) inside each category, 2000 draws, seed 20261007. See PROTOCOL.json.
"""
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MAIN = Path(r"C:\Users\omris\TAU\hallucination_detection")
SRC = MAIN / ".worktrees/self-generated-step-labels-v1/results/self_generated_step_labels_v1"
PRMB_TYPES = MAIN / "results/monday_nested_banks_v2/PANEL_B_PRMBENCH_TYPES.csv"
DRAWS, SEED = 2000, 20261007
PRM = ["redundency", "circular", "counterfactual", "step_contradiction", "domain_inconsistency", "confidence",
       "missing_condition", "deception"]
NAT = ["calculation_error", "misread_problem", "wrong_method", "other"]
CATS = PRM + NAT
ARMS = ["B16__BASE", "B16__GRP", "STEP_INDEX", "STEP_LENGTH"]
N16_METHODS = ["HEM_MOM_cc", "MEAN_MOM_cc", "MEAN_MOM_ga", "TCI_ONLY_ga", "ALL_EQ"]


def earliest_argmax(v):
    return int(np.flatnonzero(v >= v.max() - 8 * np.finfo(float).eps)[0])


def rows_for(s, f):
    """(hit, within_src, within_all) for one answer with first error f (source definitions)."""
    s = np.asarray(s, float)
    hit = float(earliest_argmax(s) == f)
    src = np.nan if f == 0 else float(((s[f] > s[:f]) + 0.5 * (s[f] == s[:f])).mean())
    o = np.delete(s, f)
    alls = float(((s[f] > o) + 0.5 * (s[f] == o)).mean()) if len(o) else np.nan
    return hit, src, alls


def load_answers():
    final = {}
    for line in open(SRC / "FINAL_LABELS_V1.jsonl", encoding="utf-8"):
        x = json.loads(line)
        if x["first_error_step"] is not None and x["first_error_step"] >= 0:
            final[x["item_id"]] = x["first_error_step"]
    fit = json.load(open(SRC / "b16/FIT.json"))
    sc = np.load(SRC / "b16/STEP_SCORES.npz")
    off = sc["offsets"]
    recs = []
    for i, m in enumerate(fit["answers"]):
        if m["item_id"] not in final:
            continue
        f = final[m["item_id"]]
        a, b = off[i], off[i + 1]
        assert b - a == m["n_steps"] == len(m["step_token_spans"])
        r = {"item_id": m["item_id"], "cell": m["cell"], "group": f"{m['cell']}::{m['src_idx']}", "f": f,
             "n_steps": m["n_steps"], "dataset": "gsm8k" if "gsm8k" in m["cell"] else "math"}
        series = {"B16__BASE": sc["B16__BASE"][a:b], "B16__GRP": sc["B16__GRP"][a:b],
                  "STEP_INDEX": np.arange(b - a, dtype=float),
                  "STEP_LENGTH": np.array([e - s for s, e in m["step_token_spans"]], float)}
        for arm, s in series.items():
            r[f"{arm}__hit"], r[f"{arm}__wsrc"], r[f"{arm}__wall"] = rows_for(s, f)
        recs.append(r)
    df = pd.DataFrame(recs)
    assert len(df) == 417 and df.item_id.is_unique and set(df.item_id) == set(final)
    return df


def gate(df):
    out = {}
    exp = {"B16__BASE": (0.7566244482541014, 0.2637889688249401), "B16__GRP": (0.7386453060889738, 0.26139088729016785)}
    for arm, (w, h) in exp.items():
        gw, gh, nw = df[f"{arm}__wsrc"].mean(), df[f"{arm}__hit"].mean(), int(df[f"{arm}__wsrc"].notna().sum())
        ok = abs(gw - w) < 1e-9 and abs(gh - h) < 1e-9 and nw == 388 and len(df) == 417
        out[arm] = {"within_src": gw, "expected_within": w, "n_within": nw, "hit": gh, "expected_hit": h, "n_hit": len(df), "pass": ok}
    out["pass"] = all(v["pass"] for v in out.values() if isinstance(v, dict))
    return out


def boot_ci(sub, cols, rng):
    """Group bootstrap of column means (nanmean) inside one subset. Returns {col: (lo, hi)}."""
    groups, inv = np.unique(sub.group.to_numpy(), return_inverse=True)
    G = len(groups)
    W = rng.multinomial(G, np.full(G, 1 / G), size=DRAWS).astype(float)  # draws x groups
    res = {}
    for c in cols:
        v = sub[c].to_numpy(float)
        ok = np.isfinite(v)
        num = np.zeros(G); den = np.zeros(G)
        np.add.at(num, inv[ok], v[ok]); np.add.at(den, inv[ok], 1.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            m = (W @ num) / (W @ den)
        res[c] = (float(np.nanquantile(m, .025)), float(np.nanquantile(m, .975))) if ok.sum() >= 2 else (np.nan, np.nan)
    return res


def merge_pass(tag):
    rows = []
    for p in sorted(glob.glob(str(HERE / tag / "out_*.jsonl"))):
        rows += [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    with open(HERE / f"labels_{tag}.jsonl", "w", encoding="utf-8") as f:
        for r in sorted(rows, key=lambda r: r["item_id"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return pd.DataFrame(rows)


def kappa(a, b):
    a, b = list(a), list(b)
    cats = sorted(set(a) | set(b))
    n = len(a)
    po = np.mean([x == y for x, y in zip(a, b)])
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[c] * cb[c] for c in cats) / n ** 2
    return float((po - pe) / (1 - pe)) if pe < 1 else np.nan, float(po)


def collapse(c):
    return "prmbench_type" if c in PRM else c if c in ("calculation_error", "misread_problem") else "wrong_method_or_other"


def agreement(p1, p2):
    m = p1.merge(p2, on="item_id", suffixes=("_1", "_2"))
    out = {"n": len(m)}
    for name, f in (("primary_12way", lambda r, s: r[f"category_{s}"]),
                    ("nearest_prmbench_8way", lambda r, s: r[f"nearest_prmbench_{s}"]),
                    ("collapsed_4way", lambda r, s: collapse(r[f"category_{s}"]))):
        a = [f(r, 1) for _, r in m.iterrows()]; b = [f(r, 2) for _, r in m.iterrows()]
        k, po = kappa(a, b)
        conf = pd.crosstab(pd.Series(a, name="pass1"), pd.Series(b, name="pass2"))
        out[name] = {"cohen_kappa": k, "raw_agreement": po, "confusion_pass1_rows_pass2_cols": conf.to_dict(orient="index")}
    hc = m[(m.confidence_1 == "high") & (m.confidence_2 == "high")]
    if len(hc):
        k, po = kappa(hc.category_1, hc.category_2)
        out["primary_12way_both_high_confidence"] = {"n": len(hc), "cohen_kappa": k, "raw_agreement": po}
    out["disagreements"] = m.loc[m.category_1 != m.category_2, ["item_id", "category_1", "category_2", "justification_1", "justification_2"]].to_dict(orient="records")
    return out


def judge_crosstab(p1):
    """Descriptive: our primary category vs the round-1 Claude judge's own first_error_type (its 8-type scheme)."""
    t = {}
    for p in sorted((SRC / "labels/claude-opus-5.5").glob("shard_*.jsonl")):
        for l in open(p, encoding="utf-8"):
            x = json.loads(l); t[x["item_id"]] = (x["first_error_step"], x["first_error_type"])
    fin = {json.loads(l)["item_id"]: json.loads(l)["first_error_step"] for l in open(SRC / "FINAL_LABELS_V1.jsonl", encoding="utf-8")}
    rows = [(r.category, t[r.item_id][1]) for r in p1.itertuples() if r.item_id in t and t[r.item_id][0] == fin[r.item_id]]
    ct = pd.crosstab(pd.Series([a for a, _ in rows], name="ours"), pd.Series([b for _, b in rows], name="judge_type"))
    return {"n_same_step": len(rows), "table": ct.to_dict(orient="index")}


def per_category(df, lab):
    d = df.merge(lab[["item_id", "category", "nearest_prmbench", "prmbench_fit"]], on="item_id", how="left")
    assert d.category.notna().all()
    d["rel_pos"] = d.f / (d.n_steps - 1).clip(lower=1)
    prm = pd.read_csv(PRMB_TYPES)
    cols = [f"{a}__{m}" for a in ARMS for m in ("wsrc", "wall", "hit")]
    out = []
    strata = [(c, d.category == c) for c in CATS]
    strata += [("ALL_PRMBENCH_TYPES", d.category.isin(PRM)), ("ALL_NATURAL_BUCKETS", d.category.isin(NAT)),
               ("ALL", np.ones(len(d), bool))]
    strata += [(f"nearest:{c}|fit=good", (d.nearest_prmbench == c) & (d.prmbench_fit == "good")) for c in PRM]
    for name, mask in strata:
        sub = d[np.asarray(mask)]
        row = {"category": name, "n_answers": len(sub), "n_within_src": int(sub["B16__BASE__wsrc"].notna().sum()),
               "n_gsm8k": int((sub.dataset == "gsm8k").sum()), "n_math": int((sub.dataset == "math").sum()),
               "median_n_steps": float(sub.n_steps.median()) if len(sub) else np.nan,
               "mean_first_error_step": float(sub.f.mean()) if len(sub) else np.nan,
               "mean_relative_position": float(sub.rel_pos.mean()) if len(sub) else np.nan,
               "share_error_at_step0": float((sub.f == 0).mean()) if len(sub) else np.nan,
               "share_error_at_last_step": float((sub.f == sub.n_steps - 1).mean()) if len(sub) else np.nan}
        if len(sub):
            ci = boot_ci(sub, cols, np.random.default_rng(SEED))
            for c in cols:
                row[c] = float(sub[c].mean()) if sub[c].notna().any() else np.nan
                row[c + "_lo"], row[c + "_hi"] = ci[c]
        key = name.split(":")[1].split("|")[0] if name.startswith("nearest:") else name
        if key in PRM:
            g = prm[prm.group == key]
            row["prmb_N16_within_mean5"] = float(g[(g.bank == "N16") & g.scorer.isin(N16_METHODS)].within_auc.mean())
            row["prmb_N16_top1_mean5"] = float(g[(g.bank == "N16") & g.scorer.isin(N16_METHODS)].top1_on_error.mean())
            for lbl, sc in (("prmb_supervised_prm", "Qwen2.5-Math-PRM-7B (supervised)"), ("prmb_step_index", "Step index (later = riskier)"),
                            ("prmb_step_length", "Step token length")):
                rr = g[g.scorer == sc]
                row[lbl + "_within"] = float(rr.within_auc.iloc[0]); row[lbl + "_top1"] = float(rr.top1_on_error.iloc[0])
        out.append(row)
    return pd.DataFrame(out), d


def figure(pc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = pc[pc.category.isin(CATS)].copy()
    show = pd.concat([show, pc[pc.category == "ALL"]])
    x = np.arange(len(show))
    fig, axes = plt.subplots(2, 1, figsize=(max(9, 1.0 * len(show) + 2), 9.2), facecolor="#fcfcfb", sharex=True)
    names = [c.replace("_", "\n") if c != "ALL" else "ALL own\nerrors" for c in show.category]
    has = show["prmb_N16_within_mean5"].notna().to_numpy()
    for ax, key, title in ((axes[0], "wsrc", "(a) Source definition: first-error step vs the steps BEFORE it (step-0 errors excluded; step index = 1.0 by construction, not drawn)"),
                           (axes[1], "wall", "(b) First-error step vs ALL other steps (closer to PRMBench's error-vs-non-error AUC)")):
        ax.set_facecolor("#fcfcfb")
        y = show[f"B16__BASE__{key}"].to_numpy(); lo = show[f"B16__BASE__{key}_lo"].to_numpy(); hi = show[f"B16__BASE__{key}_hi"].to_numpy()
        ax.errorbar(x - 0.12, y, yerr=[y - lo, hi - y], fmt="o", color="#2a78d6", ms=7, capsize=3, lw=1.4, label="Own answers: B16 BASE (95% group-bootstrap CI)")
        if key == "wall":
            ax.scatter(x - 0.12, show["STEP_INDEX__wall"], marker="D", s=46, facecolors="none", edgecolors="#8a8a8a", lw=1.4, zorder=3, label="Own answers: step-index baseline (later = riskier)")
        if key == "wall":  # PRMBench's within-AUC is error steps vs non-error steps over ALL steps -> panel (b) only
            ax.scatter(x[has] + 0.16, show["prmb_N16_within_mean5"][has], marker="o", s=52, facecolors="none", edgecolors="#2a78d6", lw=1.6, zorder=3, label="PRMBench planted errors: N16, mean of 5 fusions")
            ax.scatter(x[has] + 0.16, show["prmb_supervised_prm_within"][has], marker="s", s=40, color="#eb6834", zorder=3, label="PRMBench: Qwen2.5-Math-PRM-7B (supervised)")
        ax.axhline(0.5, color="#bbbbbb", lw=0.8, ls="--")
        ax.set_ylabel("Within-answer AUC"); ax.set_ylim(0.2, 1.03)
        ax.set_title(title, fontsize=9.5, loc="left")
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8.5, loc="lower center", ncol=2, frameon=False)
    axes[1].set_xticks(x); axes[1].set_xticklabels([f"{n}\nN={k} ({w} in a)" for n, k, w in zip(names, show.n_answers, show.n_within_src)], fontsize=8)
    fig.suptitle("Natural first errors by type: label-free B16 localization on Qwen3's own answers", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures/per_category_within_auc.png", dpi=160, facecolor="#fcfcfb")
    fig.savefig(HERE / "figures/per_category_within_auc.svg", facecolor="#fcfcfb")


def main():
    df = load_answers()
    g = gate(df)
    json.dump(g, open(HERE / "GATE.json", "w"), indent=1)
    print(json.dumps(g, indent=1))
    if not g["pass"]:
        sys.exit("reproduction gate FAILED - stop")
    if len(sys.argv) > 1 and sys.argv[1] == "gate":
        return
    p1, p2 = merge_pass("pass1"), merge_pass("pass2")
    assert len(p1) == 417 and p1.item_id.is_unique, len(p1)
    ids2 = json.load(open(HERE / "inputs/PASS2_IDS.json"))["item_ids"]
    assert sorted(p2.item_id) == sorted(ids2), len(p2)
    ag = agreement(p1, p2)
    ag["judge_type_crosstab_pass1"] = judge_crosstab(p1)
    ag["pass1_distribution"] = Counter(p1.category).most_common()
    ag["pass1_nearest_prmbench_by_fit"] = pd.crosstab(p1.nearest_prmbench, p1.prmbench_fit).to_dict(orient="index")
    ag["pass1_confidence"] = Counter(p1.confidence).most_common()
    ag["pass1_disagree_with_step"] = int(p1.disagree_with_step.sum())
    json.dump(ag, open(HERE / "AGREEMENT.json", "w"), indent=1, default=float)
    pc, d = per_category(df, p1)
    pc.to_csv(HERE / "PER_CATEGORY.csv", index=False)
    d.drop(columns=[]).to_csv(HERE / "PER_ANSWER.csv", index=False)
    figure(pc)
    # dataset-stratified view (categories are unevenly spread over GSM8K and MATH)
    bd = []
    for ds in ("gsm8k", "math"):
        sub = d[d.dataset == ds]
        for c in CATS + ["ALL"]:
            s = sub if c == "ALL" else sub[sub.category == c]
            if len(s) == 0:
                continue
            ci = boot_ci(s, ["B16__BASE__wsrc", "B16__BASE__hit"], np.random.default_rng(SEED))
            bd.append({"dataset": ds, "category": c, "n_answers": len(s), "n_within_src": int(s.B16__BASE__wsrc.notna().sum()),
                       "B16__BASE__wsrc": s.B16__BASE__wsrc.mean(), "wsrc_lo": ci["B16__BASE__wsrc"][0], "wsrc_hi": ci["B16__BASE__wsrc"][1],
                       "B16__BASE__hit": s.B16__BASE__hit.mean(), "hit_lo": ci["B16__BASE__hit"][0], "hit_hi": ci["B16__BASE__hit"][1],
                       "B16__BASE__wall": s.B16__BASE__wall.mean(), "STEP_LENGTH__wsrc": s.STEP_LENGTH__wsrc.mean(),
                       "mean_relative_position": s.rel_pos.mean()})
    pd.DataFrame(bd).to_csv(HERE / "PER_CATEGORY_BY_DATASET.csv", index=False)
    pd.set_option("display.width", 250)
    print(pc[["category", "n_answers", "n_within_src", "B16__BASE__wsrc", "B16__BASE__wall", "B16__BASE__hit", "STEP_INDEX__wall",
              "STEP_LENGTH__wall", "prmb_N16_within_mean5", "prmb_supervised_prm_within"]].round(3).to_string())
    print({k: (v["cohen_kappa"], v["raw_agreement"]) for k, v in ag.items() if isinstance(v, dict) and "cohen_kappa" in v})


if __name__ == "__main__":
    main()
