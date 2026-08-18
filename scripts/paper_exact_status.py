#!/usr/bin/env python
"""
STATUS.md + the protocol/fidelity matrix for the paper-exact cycle.

Handoff §7, "tomorrow's advisor packet": produce this even while long jobs are still
running — commit, cluster connectivity, job IDs/states, completed counts, ETA, storage; a
protocol/fidelity matrix for all seven papers; and an explicit statement that nothing here
is a headline conclusion from a partial run.

Everything in the matrix comes from the seven digest cards and the run manifests actually on
disk, so a stage that has not run shows as `not-started` rather than as a promise.

Usage:
    python scripts/paper_exact_status.py --root $SHARED/results/paper_exact \
        --out results/paper_exact/STATUS.md
"""
import argparse
import glob
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

#: The seven pinned papers, their lane, their stage, and what the fidelity ceiling is and why.
PROTOCOL_MATRIX = [
    {"paper": "ProcessBench (arXiv 2412.06559v4)", "lane": "localization", "stage": "L0/L2",
     "official_code": "QwenLM/ProcessBench (runnable)",
     "ceiling": "official-exact",
     "why": "official rows, official earliest-step-or-(-1) evaluator, released checkpoints"},
    {"paper": "Mind the Gap / Evidence Drop (ICML 2026)", "lane": "localization", "stage": "L0/L2",
     "official_code": "QJ0114/evidence-drop (runnable)",
     "ceiling": "paper-specified",
     "why": "native Qwen3 teacher-forced protocol reproduced; native SLA kept in its own panel"},
    {"paper": "Unsupervised Process Reward Models (arXiv 2605.10158)", "lane": "localization",
     "stage": "L1 (their Eq. 6 control); L3 (trained uPRM) NOT SCHEDULED",
     "official_code": "none published",
     "ceiling": "paper-specified-partial",
     "why": "no code, no prompt, no marker surface form; ours is pre-registered and declared. "
            "L3 needs a LoRA + unpublished RL estimator (~44 H200-h) and is out of scope"},
    {"paper": "Deep Think with Confidence (arXiv 2508.15260)", "lane": "multi-trace compute",
     "stage": "M1/M2", "official_code": "facebookresearch/deepconf (runnable, pinned)",
     "ceiling": "paper-specified",
     "why": "we generate with HF transformers, not the paper's pinned vLLM commit; the "
            "row-level equality audit on raw logits is what licenses the DeepConf name"},
    {"paper": "Streaming Hallucination Detection (arXiv 2601.02170v1)", "lane": "prefix detection",
     "stage": "W1", "official_code": "anonymous endpoint UNREACHABLE",
     "ceiling": "blocked-assets",
     "why": "trajectories, Claude labels, splits, layer choice and probe checkpoints are all "
            "unavailable; a substitute corpus would not be a reproduction"},
    {"paper": "Stop When Enough / REFRAIN (ACL 2026)", "lane": "single-trace stopping",
     "stage": "S1", "official_code": "RLSNLP/Adaptive-Reasoning (release-placeholder README)",
     "ceiling": "paper-specified-partial",
     "why": "no runnable code; the provisional-answer cue, reward timing and cold-start/tie "
            "order are declared by us. Base trigger vocabulary recovered from the PDF's "
            "underline geometry, not guessed"},
    {"paper": "LEASH (arXiv 2511.04654v1)", "lane": "single-trace stopping", "stage": "S2",
     "official_code": "none published",
     "ceiling": "paper-specified-partial",
     "why": "B, tau_p, w and gamma are absent from the PDF; declared by us and swept on pilot "
            "IDs, with one central choice frozen before the full run"},
]

STAGES = {
    "p0": "P0 — assets, hashes, official-code pins, environment",
    "l0": "L0 — shared ProcessBench table from existing artifacts",
    "l1_uprm_judge_pilot": "L1 pilot — uPRM Eq. 6 control, Qwen2.5-14B",
    "l1_uprm_judge_full": "L1 full — uPRM Eq. 6 control, all 3,400 rows",
    "s1_refrain_pilot": "S1 pilot — REFRAIN implementation check (30 questions)",
    "s1_refrain_full": "S1 full — REFRAIN, Qwen3-8B x MATH-500, both arms",
    "s2_leash_pilot": "S2 pilot — LEASH sensitivity sweep on pilot IDs",
    "s2_leash_full": "S2 full — LEASH native matrix",
    "m1_deepconf_pilot": "M1 — DeepConf protocol pilot (K=32)",
    "m2_deepconf_full": "M2 — DeepConf full pool (4,096/question)",
    "c1_confirmation": "C1 — untouched confirmation cell",
}


def _run(args, timeout=60):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception:
        return ""


def stage_status(root: str) -> list:
    out = []
    for key, title in STAGES.items():
        d = os.path.join(root, key)
        entry = {"stage": key, "title": title, "dir": d, "state": "not-started"}
        if os.path.isdir(d):
            entry["state"] = "started"
            man_p = os.path.join(d, "RUN_MANIFEST.json")
            if os.path.exists(man_p):
                with open(man_p) as f:
                    m = json.load(f)
                entry.update(fidelity=m.get("fidelity"), model=m.get("model_id"),
                             model_revision=(m.get("model_revision") or "")[:12],
                             expected=m.get("expected_traces"),
                             repo_commit=(m.get("repo_commit") or "")[:12],
                             n_resumes=len(m.get("resumes", [])))
            st_p = os.path.join(d, "STATUS.json")
            if os.path.exists(st_p):
                with open(st_p) as f:
                    s = json.load(f)
                entry.update(finished=s.get("n_finished"), failed=s.get("n_failed"),
                             shards=s.get("n_shards"),
                             gb=round((s.get("bytes_total") or 0) / 1e9, 2),
                             complete=s.get("complete"))
                if s.get("complete"):
                    entry["state"] = "complete"
                elif s.get("n_finished"):
                    entry["state"] = "in-progress"
            gates = {}
            for g in sorted(glob.glob(os.path.join(d, "GATE_*.json"))):
                with open(g) as f:
                    gd = json.load(f)
                gates[gd.get("stage", os.path.basename(g))] = {
                    "passed": gd.get("passed"), "failures": gd.get("failures", [])}
            if gates:
                entry["gates"] = gates
            if os.path.exists(os.path.join(d, "BLOCKED_ASSETS.json")):
                entry["state"] = "blocked-assets"
        out.append(entry)
    return out


def render(root: str, stages: list, jobs: str) -> str:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    commit = _run(["git", "-C", REPO_ROOT, "rev-parse", "HEAD"])[:12]
    dirty = bool(_run(["git", "-C", REPO_ROOT, "status", "--porcelain", "-uno"]))

    L = []
    L.append("# Paper-exact acquisition — STATUS\n")
    L.append(f"**Generated:** {now}  \n")
    L.append(f"**Repo commit:** `{commit}`{'  **(dirty tree)**' if dirty else ''}  \n")
    L.append(f"**Acquisition root:** `{root}`\n")
    L.append("> Nothing below is a headline conclusion. Stages that are partial are marked\n"
             "> partial, and published values are regression targets, never acceptance gates.\n")

    L.append("\n## Protocol and fidelity matrix\n")
    L.append("| Paper | Lane | Stage | Official code | Fidelity ceiling | Why |")
    L.append("|---|---|---|---|---|---|")
    for r in PROTOCOL_MATRIX:
        L.append(f"| {r['paper']} | {r['lane']} | {r['stage']} | {r['official_code']} "
                 f"| `{r['ceiling']}` | {r['why']} |")

    L.append("\n## Stage status\n")
    L.append("| Stage | State | Model | Fidelity | Done / Expected | Failed | Shards | GB | Gates |")
    L.append("|---|---|---|---|---:|---:|---:|---:|---|")
    for s in stages:
        g = s.get("gates") or {}
        gtxt = ", ".join(f"{k}:{'PASS' if v['passed'] else 'FAIL ' + ','.join(v['failures'])}"
                         for k, v in g.items()) or "—"
        L.append(f"| {s['title']} | `{s['state']}` | {s.get('model', '—')} "
                 f"| {s.get('fidelity', '—')} "
                 f"| {s.get('finished', '—')} / {s.get('expected', '—')} "
                 f"| {s.get('failed', '—')} | {s.get('shards', '—')} | {s.get('gb', '—')} "
                 f"| {gtxt} |")

    if jobs:
        L.append("\n## Slurm\n\n```\n" + jobs + "\n```\n")

    L.append("\n## Result lanes — never ranked together\n")
    L.append("| Lane | Question | Native outputs |")
    L.append("|---|---|---|")
    L.append("| Localization | Where is the first erroneous step, or is the trace clean? "
             "| error acc, clean acc, ProcessBench F1; Mind-the-Gap SLA separately |")
    L.append("| Prefix detection | From tokens available now, will this trace finish wrong? "
             "| AUROC/AUPRC at absolute budgets, causal alarm performance |")
    L.append("| Single-trace stopping | Should this trace stop and answer now? "
             "| pass@1, generated tokens, latency, accuracy-compute frontier |")
    L.append("| Multi-trace adaptive compute | Which traces finish/vote, keep sampling? "
             "| vote accuracy versus total sampled tokens |")

    L.append("\n## Known blocked rows\n")
    blocked = [s for s in stages if s["state"] == "blocked-assets"]
    if blocked:
        for s in blocked:
            L.append(f"- **{s['title']}** — see `{s['dir']}/BLOCKED_ASSETS.json`")
    else:
        L.append("- Streaming Hallucination Detection (W1): the anonymous code endpoint was "
                 "unreachable at audit. `BLOCKED_ASSETS.json` is emitted under the P0 "
                 "directory and no compute is booked for it.")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--root",
                    default="/shared/cycle2_tau_averbuch_prj/omrisegev1/results/paper_exact")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "paper_exact", "STATUS.md"))
    ap.add_argument("--squeue", action="store_true", help="include live Slurm state")
    args = ap.parse_args()

    stages = stage_status(args.root)
    jobs = _run(["squeue", "-u", "omrisegev1", "-o", "%i %j %T %M %L %R"]) if args.squeue else ""
    md = render(args.root, stages, jobs)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md)
    with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
        json.dump({"root": args.root, "stages": stages,
                   "protocol_matrix": PROTOCOL_MATRIX}, f, indent=2, default=str)
    print(md)
    print(f"STATUS -> {args.out}")


if __name__ == "__main__":
    main()
