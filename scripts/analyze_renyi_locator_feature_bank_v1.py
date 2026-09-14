#!/usr/bin/env python3
"""Write the reviewed Experiment-3 findings and its Pareto plot."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/renyi_locator_feature_bank_v1"
REPLAY = ROOT / "results/renyi_locator_integrated_replay_v1"
TAIL = ROOT / "results/tail15_localization_q_v1/METRICS.json"


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf8")
    temporary.replace(path)


def method(q: int, h1: int, hinf: int, solver: str) -> str:
    return f"ve1q{q}__h1{h1}__hinf{hinf}__{solver}"


def factor_effect(metrics, factor: str) -> dict:
    solvers = ("raw_step_equal", "scale_step_equal", "answer_z_local_iu")
    if factor == "H1":
        pairs = [(method(q, 1, hinf, solver), method(q, 0, hinf, solver)) for q in (15, 50) for hinf in (0, 1) for solver in solvers]
    elif factor == "Hinf":
        pairs = [(method(q, h1, 1, solver), method(q, h1, 0, solver)) for q in (15, 50) for h1 in (0, 1) for solver in solvers]
    elif factor == "q50":
        pairs = [(method(50, h1, hinf, solver), method(15, h1, hinf, solver)) for h1 in (0, 1) for hinf in (0, 1) for solver in solvers]
    else:
        raise ValueError(factor)
    result = {"pairs": len(pairs)}
    for key in ("pb_all8", "prm_within", "prmscore_q08"):
        values = np.asarray([metrics[first][key] - metrics[second][key] for first, second in pairs])
        result[key] = {
            "mean_delta": float(values.mean()),
            "wins": int(np.sum(values > 0)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }
    return result


def label(name: str) -> str:
    bank, solver = name.rsplit("__", 1)
    parts = bank.split("__")
    q = parts[0].replace("ve1", "VE1-")
    features = []
    if parts[1] == "h11":
        features.append("H1")
    if parts[2] == "hinf1":
        features.append("Hinf")
    return q + ("+" + "+".join(features) if features else "") + "/" + {"raw_step_equal": "raw", "scale_step_equal": "scale", "answer_z_local_iu": "IU"}[solver]


def plot(metrics, selected, baseline) -> None:
    colors = {"raw_step_equal": "#1f77b4", "scale_step_equal": "#ff7f0e", "answer_z_local_iu": "#2ca02c"}
    names = list(metrics)
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=150)
    for solver, color in colors.items():
        group = [name for name in names if name.endswith(solver)]
        for q, marker in ((15, "o"), (50, "s")):
            subset = [name for name in group if name.startswith(f"ve1q{q}")]
            ax.scatter([metrics[name]["prm_within"] for name in subset], [100 * metrics[name]["pb_all8"] for name in subset], c=color, marker=marker, s=55, alpha=.8, label=f"{solver.replace('_step_equal','').replace('answer_z_local_iu','local IU')}, VE1 q{q}")
    best_pb = max(names, key=lambda name: metrics[name]["pb_all8"])
    best_prm = max(names, key=lambda name: metrics[name]["prm_within"])
    highlights = [(baseline, "current", "*", "black"), (selected, "min-regret", "X", "#d62728"), (best_pb, "best PB", "D", "#9467bd"), (best_prm, "best PRM", "^", "#8c564b")]
    used = set()
    for name, text, marker, color in highlights:
        if name in used:
            continue
        used.add(name)
        x, y = metrics[name]["prm_within"], 100 * metrics[name]["pb_all8"]
        ax.scatter([x], [y], marker=marker, s=180, c=color, edgecolors="white", linewidths=1.1, zorder=5)
        ax.annotate(text, (x, y), xytext=(6, 7), textcoords="offset points", fontsize=9)
    ax.set_xlabel("PRMB within-answer AUROC")
    ax.set_ylabel("ProcessBench all-8 macro-F1 (%)")
    ax.set_title("Experiment 3: no feature-bank arm dominates the current locator")
    ax.grid(alpha=.25)
    ax.legend(fontsize=7.5, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "PARETO.png")
    plt.close(fig)


def main() -> None:
    experiment = json.loads((OUT / "METRICS.json").read_text())
    replay = json.loads((REPLAY / "METRICS.json").read_text())
    tail = json.loads(TAIL.read_text())
    metrics = experiment["metrics"]
    selected = experiment["selection"]["selected"]
    baseline = experiment["selection"]["baseline"]
    effects = {factor: factor_effect(metrics, factor) for factor in ("H1", "Hinf", "q50")}
    findings = {
        "schema": "renyi-locator-feature-bank-findings-v1",
        "status": "PASS",
        "factor_effects": effects,
        "best_pb": max(metrics, key=lambda name: metrics[name]["pb_all8"]),
        "best_prm_within": max(metrics, key=lambda name: metrics[name]["prm_within"]),
        "minimum_regret": selected,
        "integrated_replay_promotion": replay["promotion"],
        "final_recommendation": replay["integrated_recommendation"],
        "questions": {
            "H1": "No: mean PB and PRMB effects are negative and only one of 12 matched comparisons improves each primary metric.",
            "Hinf": "No uniform gain: isolated best-PB and best-PRMB arms trade away the other benchmark.",
            "VE1_q50": "No promotion: small mean PRMB/PRMScore gains trade against a larger mean PB loss.",
            "fusion": "Raw equal is the balanced choice; scale-only favors PB, while local IU is worse on the joint frontier.",
            "integration": "The independent replay rejects the minimum-regret candidate and retains the frozen q15/raw locator plus tail15 q=.33 gate.",
        },
    }
    atomic_json(OUT / "FINDINGS.json", findings)
    plot(metrics, selected, baseline)

    ranking = experiment["selection"]["ranking"]
    regret = {row["method"]: row["regret"] for row in ranking}
    lines = [
        "# Renyi locator feature-bank experiment v1 — results",
        "",
        "Status: **COMPLETE / REVIEW PASS**",
        "Population: 13,769 answers, 145,597 steps. Development comparison; not external confirmation.",
        "",
        "## Questions and answers",
        "",
        "1. **Does native H1 help the locator?** No. Across 12 matched pairs it changes PB by "
        f"{100*effects['H1']['pb_all8']['mean_delta']:+.3f} percentage points on average and PRMB-within by {effects['H1']['prm_within']['mean_delta']:+.6f}; only "
        f"{effects['H1']['pb_all8']['wins']}/12 PB and {effects['H1']['prm_within']['wins']}/12 PRMB comparisons improve.",
        "2. **Does q15 Hinf help?** Not uniformly. It creates the single best PB and PRMB arms in different solvers, but its mean effects are "
        f"{100*effects['Hinf']['pb_all8']['mean_delta']:+.3f} PB points and {effects['Hinf']['prm_within']['mean_delta']:+.6f} PRMB. Neither isolated winner is a joint improvement.",
        "3. **Should VE1 q15 be replaced by VE1 q50?** No for the uniform locator. Its mean effect is "
        f"{100*effects['q50']['pb_all8']['mean_delta']:+.3f} PB points and {effects['q50']['prm_within']['mean_delta']:+.6f} PRMB. PRMScore improves in all 12 matched pairs, but that does not offset the PB loss under the frozen joint objective.",
        "4. **Does a deployable fusion capture the complementary errors?** No tested fusion does. The label-using union localizes "
        f"{experiment['union_diagnostic']['additional_union_hits']} additional error answers beyond the baseline, but no label-free arm improves both primary benchmarks. This is evidence for future conditional selection, not a deployable result.",
        "5. **Does the full candidate help with the frozen gate?** No. The independent replay gives "
        f"PB delta {100*replay['deltas_candidate_minus_reference']['pb_all8']:+.3f} percentage points and PRMB-within delta {replay['deltas_candidate_minus_reference']['prm_within']:+.6f}; the latter exceeds the frozen .002 loss margin.",
        "",
        "## Headline comparison",
        "",
        "| Method | PB all-8 | PB raw exact | PRMB within | PRMB fold | pooled OOF | PRMScore | Decision |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    headline = [
        (baseline, "retain"),
        ("ve1q15__h10__hinf1__scale_step_equal", "best PB only"),
        ("ve1q15__h10__hinf1__raw_step_equal", "best PRMB only"),
        (selected, "min-regret; rejected by replay"),
    ]
    for name, decision in headline:
        value = metrics[name]
        lines.append(f"| `{label(name)}` | {100*value['pb_all8']:.4f}% | {100*value['pb_raw_exact']:.4f}% | {value['prm_within']:.6f} | {value['prm_fold_auc']:.6f} | {value['prm_pooled_oof_descriptive']:.6f} | {value['prmscore_q08']:.6f} | {decision} |")
    lines.extend(["", "## All 24 arms", "", "| Feature bank / solver | PB all-8 | PRMB within | PRMScore | regret |", "|---|---:|---:|---:|---:|"])
    for name in metrics:
        value = metrics[name]
        lines.append(f"| `{label(name)}` | {100*value['pb_all8']:.4f}% | {value['prm_within']:.6f} | {value['prmscore_q08']:.6f} | {regret[name]:.3f} |")
    start = tail["localization"]["start_original_static_entropy_q03"]["macros"]["all"]
    current = tail["localization"]["q15_tail15_top10_selected_q"]["macros"]["all"]
    starting_prm = tail["prmbench"]["starting_locator"]
    current_prm = tail["prmbench"]["current_locator"]
    lines.extend([
        "",
        "## Final integrated recommendation versus the original start",
        "",
        "The recommendation is unchanged: q15 H0lim/VE0/VE0.75/VE1, natural-unit per-view Top10 equal locator, plus tail15 whole-answer Top10 gate at q=.33.",
        "",
        "| Metric | Original start | Recommended integration | Delta |",
        "|---|---:|---:|---:|",
        f"| PB all-8 | {100*start:.4f}% | {100*current:.4f}% | {100*(current-start):+.4f} pp |",
        f"| PRMB within | {starting_prm['prm_within']:.6f} | {current_prm['prm_within']:.6f} | {current_prm['prm_within']-starting_prm['prm_within']:+.6f} |",
        f"| PRMB fold AUROC | {starting_prm['prm_fold_auc']:.6f} | {current_prm['prm_fold_auc']:.6f} | {current_prm['prm_fold_auc']-starting_prm['prm_fold_auc']:+.6f} |",
        f"| PRMB pooled OOF | {starting_prm['prm_pooled_oof_descriptive']:.6f} | {current_prm['prm_pooled_oof_descriptive']:.6f} | {current_prm['prm_pooled_oof_descriptive']-starting_prm['prm_pooled_oof_descriptive']:+.6f} |",
        f"| PRMScore | {starting_prm['prmscore_q08']:.6f} | {current_prm['prmscore_q08']:.6f} | {current_prm['prmscore_q08']-starting_prm['prmscore_q08']:+.6f} |",
        "",
        "The PB comparison includes both accepted changes (locator and gate). The PRMB comparison isolates the locator because PRMB has no answer-level gate. The slight PRMScore decrease prevents claiming universal improvement; the gains are PB and AUROC gains.",
        "",
        "## Replay evidence",
        "",
        f"- Both independently recomputed score streams are bitwise identical to their frozen archives (`max_abs=0`).",
        f"- Candidate PB delta 95% group-bootstrap CI: [{replay['bootstrap']['pb_all8']['interval'][0]:+.6f}, {replay['bootstrap']['pb_all8']['interval'][1]:+.6f}].",
        f"- Candidate PRMB-within delta 95% group-bootstrap CI: [{replay['bootstrap']['prm_within']['interval'][0]:+.6f}, {replay['bootstrap']['prm_within']['interval'][1]:+.6f}].",
        "- Joint L-SML was deliberately deferred; the baseline four-view bank is structurally inadmissible and the expensive extension is only warranted if a cheaper arm first succeeds.",
        "",
        "![Experiment 3 Pareto plot](PARETO.png)",
        "",
    ])
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf8")
    replay_lines = [
        "# Renyi locator integrated replay v1 — results",
        "",
        "Status: **COMPLETE / REVIEW PASS**",
        "",
        "The independently reconstructed minimum-regret candidate is not promoted. Both its score stream and the current reference are bitwise identical to the Experiment-3 archives (`max_abs=0`) over 13,769 answers and 145,597 steps.",
        "",
        "| Method | PB all-8 | PB raw exact | PRMB within | pooled OOF | PRMScore |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in (baseline, selected):
        value = replay["metrics"][name]
        replay_lines.append(f"| `{label(name)}` | {100*value['pb_all8']:.4f}% | {100*value['pb_raw_exact']:.4f}% | {value['prm_within']:.6f} | {value['prm_pooled_oof_descriptive']:.6f} | {value['prmscore_q08']:.6f} |")
    replay_lines.extend([
        "",
        f"Candidate minus current: PB {100*replay['deltas_candidate_minus_reference']['pb_all8']:+.4f} pp (95% CI [{100*replay['bootstrap']['pb_all8']['interval'][0]:+.4f}, {100*replay['bootstrap']['pb_all8']['interval'][1]:+.4f}]); PRMB-within {replay['deltas_candidate_minus_reference']['prm_within']:+.6f} (95% CI [{replay['bootstrap']['prm_within']['interval'][0]:+.6f}, {replay['bootstrap']['prm_within']['interval'][1]:+.6f}]).",
        "",
        "The PRMB loss exceeds the frozen .002 noninferiority margin. Final integrated recommendation: retain q15 H0lim/VE0/VE0.75/VE1 in natural units, Top10 per view, equal step mean, and tail15 whole-answer Top10 gate q=.33.",
        "",
        "This is a development replay; external/new-model confirmation is still required.",
        "",
    ])
    (REPLAY / "REPORT.md").write_text("\n".join(replay_lines), encoding="utf8")
    print(json.dumps(findings, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
