"""
Step 148 streaming pilot — report: merge per-cell results, evaluate gates,
print the competitor-context table, and save plots.

Reads the per-cell checkpoints written by scripts/streaming_pilot.py
(results/sp_*.pkl), merges them, then:

    - per-cell prefix-AUROC table (L-SML vs baselines at every budget)
    - G1: AUROC@50%-of-trace >= 95% of full-trace AUROC (>=2 clean cells)
    - G2: fused L-SML >= best DeepConf window + 2pp at >=2 budgets, >=2 cells
    - G3: context vs supervised streaming-probe paper (arXiv:2601.02170)
    - E3/E4 online-monitor summary
    - plots to results/figs/

Usage:
    python scripts/streaming_pilot_report.py
"""
import glob
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RES_GLOB = "results/sp_*.pkl"
FIG_DIR = "results/figs"

CLEAN_CELLS = ["gsm8k/Llama-3.1-8B", "math500/Qwen2.5-Math-1.5B_T1.0"]

# arXiv:2601.02170 — SUPERVISED hidden-state probes, step-annotation labels.
# Prefix-level detection AUC. Protocol differs from ours (unsupervised,
# logprob-only, final-answer labels): context, not head-to-head.
COMPETITOR = {
    "LLaMA-3.1-8B (theirs, supervised)": 72.69,
    "Qwen2.5-7B (theirs, supervised)": 81.05,
    "DeepSeek-R1-8B (theirs, supervised)": 92.18,
}
OUR_VS_THEIRS = {
    "gsm8k/Llama-3.1-8B": "LLaMA-3.1-8B (theirs, supervised)",
    "math500/Qwen2.5-Math-1.5B_T1.0": "Qwen2.5-7B (theirs, supervised)",
    "gpqa/DeepSeek-R1-7B_TRUNC": "DeepSeek-R1-8B (theirs, supervised)",
    "gpqa/DeepSeek-R1-8B_K2_TRUNC": "DeepSeek-R1-8B (theirs, supervised)",
}

MAIN_TAGS = ["lsml16", "lsml5", "epr", "max_ent", "tail_conf",
             "deepconf_w32", "deepconf_w64", "deepconf_w128"]


def load_merged():
    res = {}
    for path in sorted(glob.glob(RES_GLOB)):
        if "smoke" in path:
            continue
        with open(path, "rb") as f:
            part = pickle.load(f)
        res.update(part)
    return res


def best_deepconf(auc_dict):
    return max(auc_dict[f"deepconf_w{w}"][0] for w in (32, 64, 128))


def abs_keys_sorted(e12):
    ks = [k for k in e12["abs"] if k != "full"]
    return sorted(ks) + (["full"] if "full" in e12["abs"] else [])


def print_cell_tables(res):
    for cell, cr in res.items():
        e12 = cr.get("e1_e2", {})
        if not e12.get("abs"):
            continue
        print(f"\n=== {cell} ===")
        hdr = f"{'budget':>8s}" + "".join(f"{t:>14s}" for t in MAIN_TAGS)
        print(hdr)
        for kind, keys in (("abs", abs_keys_sorted(e12)),
                           ("frac", sorted(e12.get("frac", {})))):
            for k in keys:
                entry = e12[kind][k]
                row = f"{kind[0]}={k!s:>6s}"
                for t in MAIN_TAGS:
                    a = entry["auc"].get(t, (np.nan,))[0]
                    row += f"{a:>14.3f}"
                print(row)


def gate_g1(res):
    print("\n--- G1: early detectability (AUROC@50% >= 95% of full-trace) ---")
    passing = []
    for cell, cr in res.items():
        e12 = cr.get("e1_e2", {})
        if 0.5 not in e12.get("frac", {}) or 1.0 not in e12.get("frac", {}):
            continue
        for tag in ("lsml16", "lsml5", "epr", "deepconf_w64"):
            half = e12["frac"][0.5]["auc"][tag][0]
            full = e12["frac"][1.0]["auc"][tag][0]
            ok = full > 0.5 and half >= 0.95 * full
            mark = "PASS" if ok else "fail"
            flag = "" if cell in CLEAN_CELLS else "  [TRUNC cell]"
            print(f"  {cell:38s} {tag:12s} half={half:.3f} full={full:.3f} -> {mark}{flag}")
            if ok and tag == "lsml16" and cell in CLEAN_CELLS:
                passing.append(cell)
    verdict = "PASS" if len(passing) >= 2 else "FAIL"
    print(f"G1 verdict (lsml16, clean cells): {len(passing)}/2 needed -> {verdict}")
    return verdict


def gate_g2(res):
    print("\n--- G2 (primary): L-SML >= best DeepConf + 2pp at >=2 budgets on >=2 cells ---")
    cells_passing = []
    for cell, cr in res.items():
        e12 = cr.get("e1_e2", {})
        if not e12.get("abs"):
            continue
        wins = []
        for k in abs_keys_sorted(e12):
            entry = e12["abs"][k]
            l16 = entry["auc"]["lsml16"][0]
            dc = best_deepconf(entry["auc"])
            if l16 >= dc + 0.02:
                wins.append((k, l16, dc))
        flag = "" if cell in CLEAN_CELLS else "  [TRUNC cell]"
        win_s = ", ".join(f"{k}: {a:.3f} vs {d:.3f}" for k, a, d in wins) or "none"
        print(f"  {cell:38s} budgets won: {win_s}{flag}")
        if len(wins) >= 2 and cell in CLEAN_CELLS:
            cells_passing.append(cell)
    verdict = "PASS" if len(cells_passing) >= 2 else "FAIL"
    print(f"G2 verdict (clean cells with >=2 winning budgets): "
          f"{len(cells_passing)}/2 needed -> {verdict}")
    return verdict


def gate_g3(res):
    print("\n--- G3: context vs supervised streaming probes (arXiv:2601.02170) ---")
    print("Their protocol: SUPERVISED hidden-state probes, step-level annotation")
    print("labels. Ours: UNSUPERVISED, output logprobs only, final-answer labels.")
    print("Same task family, not the same benchmark — context only.\n")
    print(f"{'cell (ours)':40s}{'ours lsml16':>12s}{'ours best':>12s}"
          f"{'theirs':>9s}  their setup")
    for cell, cr in res.items():
        e12 = cr.get("e1_e2", {})
        if "full" not in e12.get("abs", {}):
            continue
        entry = e12["abs"]["full"]
        l16 = entry["auc"]["lsml16"][0] * 100
        best_tag = max(MAIN_TAGS, key=lambda t: entry["auc"][t][0])
        best = entry["auc"][best_tag][0] * 100
        comp = OUR_VS_THEIRS.get(cell)
        theirs = COMPETITOR.get(comp, np.nan)
        print(f"{cell:40s}{l16:>11.1f}%{best:>10.1f}% ({best_tag})"
              f"{theirs:>8.1f}  {comp or '-'}")


def monitor_summary(res):
    print("\n--- E3/E4: online monitor (flag mid-generation) ---")
    for cell, cr in res.items():
        if "e4" not in cr:
            continue
        best = max(cr["e4"], key=lambda k: cr["e4"][k]["fa10"]["detection_rate"])
        f5, f10 = cr["e4"][best]["fa5"], cr["e4"][best]["fa10"]
        flag = "" if cell in CLEAN_CELLS else "  [TRUNC cell]"
        print(f"  {cell:38s} best={best:14s} "
              f"det@FA5={f5['detection_rate']:.2f} (saves {f5['frac_tokens_saved']:.0%}) "
              f"det@FA10={f10['detection_rate']:.2f} (saves {f10['frac_tokens_saved']:.0%})"
              f"{flag}")


def make_plots(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG_DIR, exist_ok=True)
    cells = [c for c in res if res[c].get("e1_e2", {}).get("abs")]
    plot_tags = [("lsml16", "tab:blue", "-"), ("lsml5", "tab:cyan", "-"),
                 ("deepconf_w64", "tab:red", "--"), ("epr", "tab:orange", "--"),
                 ("max_ent", "tab:gray", ":")]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for ax, cell in zip(axes.flat, cells):
        e12 = res[cell]["e1_e2"]
        keys = abs_keys_sorted(e12)
        xs = [k if k != "full" else 2048 for k in keys]
        for tag, color, ls in plot_tags:
            ys = [e12["abs"][k]["auc"][tag][0] for k in keys]
            lo = [e12["abs"][k]["auc"][tag][1] for k in keys]
            hi = [e12["abs"][k]["auc"][tag][2] for k in keys]
            ax.plot(xs, ys, color=color, ls=ls, marker="o", ms=3, label=tag)
            if tag == "lsml16":
                ax.fill_between(xs, lo, hi, color=color, alpha=0.15)
        ax.axhline(0.5, color="k", lw=0.5)
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(k) for k in keys], rotation=45)
        ax.set_title(cell, fontsize=10)
        ax.set_xlabel("token budget")
        ax.set_ylabel("AUROC")
    axes.flat[0].legend(fontsize=8)
    for ax in axes.flat[len(cells):]:
        ax.axis("off")
    fig.suptitle("Prefix-AUROC vs token budget (unsupervised, final-answer labels)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "prefix_auroc_abs.png"), dpi=150)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for ax, cell in zip(axes.flat, cells):
        e12 = res[cell]["e1_e2"]
        fr = sorted(e12["frac"])
        for tag, color, ls in plot_tags:
            ys = [e12["frac"][f]["auc"][tag][0] for f in fr]
            ax.plot(fr, ys, color=color, ls=ls, marker="o", ms=3, label=tag)
        ax.axhline(0.5, color="k", lw=0.5)
        ax.set_title(cell, fontsize=10)
        ax.set_xlabel("fraction of trace (oracle length)")
        ax.set_ylabel("AUROC")
    axes.flat[0].legend(fontsize=8)
    for ax in axes.flat[len(cells):]:
        ax.axis("off")
    fig.suptitle("Prefix-AUROC vs trace fraction")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "prefix_auroc_frac.png"), dpi=150)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for ax, cell in zip(axes.flat, cells):
        e3 = res[cell].get("e3", {})
        for rk, curve in e3.items():
            fa = [c["false_alarm_rate"] for c in curve]
            det = [c["detection_rate"] for c in curve]
            order = np.argsort(fa)
            ax.plot(np.array(fa)[order], np.array(det)[order],
                    marker=".", ms=3, label=rk)
        ax.plot([0, 1], [0, 1], "k:", lw=0.5)
        ax.set_title(cell, fontsize=10)
        ax.set_xlabel("false-alarm rate (correct traces flagged)")
        ax.set_ylabel("detection rate (wrong traces flagged)")
    axes.flat[0].legend(fontsize=7)
    for ax in axes.flat[len(cells):]:
        ax.axis("off")
    fig.suptitle("Online monitor: flag-while-generating tradeoff")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "online_monitor.png"), dpi=150)
    print(f"\nplots saved to {FIG_DIR}/")


def main():
    res = load_merged()
    if not res:
        print("no results found under", RES_GLOB)
        return 1
    print(f"cells loaded: {list(res)}")
    print_cell_tables(res)
    g1 = gate_g1(res)
    g2 = gate_g2(res)
    gate_g3(res)
    monitor_summary(res)
    make_plots(res)
    print(f"\nGates: G1={g1}  G2={g2}  "
          f"(pass both -> propose streaming direction; G2 fail -> spectral "
          f"suite adds nothing over windowed mean in streaming regime)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
