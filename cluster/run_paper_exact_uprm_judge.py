#!/usr/bin/env python
"""
L1 — the uPRM paper's own cheap "LLM-as-a-Judge" control, Qwen2.5-14B, full ProcessBench.

Handoff §L1. This implements Eq. 6 of "Unsupervised Process Reward Models" (Gadetsky et al.,
arXiv:2605.10158) — the paper's *independent* per-trajectory scoring baseline — on the
**paper's own backbone**, Qwen2.5-14B-Instruct, over all 3,400 official ProcessBench rows.

It is NOT uPRM. uPRM needs a LoRA-tuned model trained with a bespoke RL objective (~44
H200 GPU-hours) and is L3, conditional and separate. Never rename this row uPRM.

Why it is the first localization GPU priority (handoff §L1): it is the only *fair
same-backbone inference-only control* available. Comparing our label-free single-pass score
against a trained PRM or a 72B critic compares access tiers, not methods; this row is in our
own tier and on the paper's backbone.

Paper targets for this control (regression, not gates):
    GSM8K 49.8 | MATH 42.8 | OlympiadBench 29.4 | Omni-MATH 26.6  (ProcessBench F1)

The marker surface form is ours, not theirs
-------------------------------------------
The paper publishes no code and no exact prompt/marker rendering, so
`spectral_utils/uprm_baseline.py` pre-registers "+"/"-" with a short convention-explaining
system message, and derives each marker's real token id from the actual following text
(the BPE-merge bug its docstring documents). Report this as *our reproduction of their
baseline*, never as their number. Fidelity: `paper-specified-partial`.

Usage:
    python cluster/run_paper_exact_uprm_judge.py --mode pilot \
        --out $SHARED/results/paper_exact/l1_uprm_judge_pilot
    python cluster/run_paper_exact_uprm_judge.py --mode full \
        --out $SHARED/results/paper_exact/l1_uprm_judge_full
"""
import argparse
import json
import os
import signal
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    import transformers.modeling_utils as _mu
    _mu.check_torch_load_is_safe = lambda *a, **k: None
except Exception:
    pass

from spectral_utils import load_model, free_memory
from spectral_utils.paper_exact import evaluator as EV
from spectral_utils.paper_exact.gates import Gate
from spectral_utils.paper_exact.manifest import build_manifest, write_manifest, verify_manifest
from spectral_utils.paper_exact.shards import ShardWriter, read_shards
from spectral_utils.processbench import SUBSETS, NO_ERROR, load_processbench
from spectral_utils.uprm_baseline import (SYSTEM_PROMPT, localize_first_error, score_candidates)

EXIT_INCOMPLETE = 85
STOP = {"flag": False}
PAPER_PDF = os.path.join(REPO_ROOT, "papers", "Unsupervised Process Reward Models.pdf")
PAPER_TARGETS = {"gsm8k": 49.8, "math": 42.8, "olympiadbench": 29.4, "omnimath": 26.6}


def _on_sigterm(signum, frame):
    STOP["flag"] = True
    print("[l1] SIGTERM — will checkpoint after the current row", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct",
                    help="the uPRM paper's own scoring backbone")
    ap.add_argument("--model-revision", default="main")
    ap.add_argument("--mode", default="pilot", choices=["smoke", "pilot", "full"])
    ap.add_argument("--subsets", default=",".join(SUBSETS))
    ap.add_argument("--n-samples", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--attn-impl", default="sdpa")
    ap.add_argument("--dry-run", action="store_true",
                    help="build and verify the manifest from synthetic rows, then exit")
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, _on_sigterm)
    n = args.n_samples if args.n_samples is not None else (
        3 if args.mode == "smoke" else 30 if args.mode == "pilot" else None)
    subsets = [s.strip() for s in args.subsets.split(",") if s.strip()]

    rows = []
    if args.dry_run:
        rows = [{"id": f"dryrun-{i}", "problem": "p", "steps": ["a", "b"], "label": -1,
                 "subset": subsets[0]} for i in range(3)]
    else:
        for subset in subsets:
            for r in load_processbench(subset, n):
                rows.append({**r, "subset": subset})
    print(f"[l1] mode={args.mode} model={args.model} subsets={subsets} rows={len(rows)}",
          flush=True)

    os.makedirs(args.out, exist_ok=True)
    if args.dry_run:
        tok_only = type("T", (), {"chat_template": "DRY-RUN-CHAT-TEMPLATE"})()
    else:
        from transformers import AutoTokenizer
        tok_only = AutoTokenizer.from_pretrained(args.model)
    man = build_manifest(
        run_id=os.path.basename(args.out.rstrip("/")),
        paper_title="Unsupervised Process Reward Models — Eq. 6 LLM-as-a-Judge control",
        paper_pdf_path=PAPER_PDF,
        fidelity="paper-specified-partial",
        dataset_source="Qwen/ProcessBench", dataset_revision="official",
        dataset_example_ids=[f"{r['subset']}:{r.get('id', i)}" for i, r in enumerate(rows)],
        model_id=args.model, model_revision=args.model_revision,
        prompt_text=SYSTEM_PROMPT, chat_template=tok_only.chat_template or "",
        decoding={"mode": "teacher-forced marker probabilities", "no_generation": True},
        seed_policy={"deterministic": True, "why": "one forward pass, no sampling"},
        max_new_tokens=0,
        stop_behavior={"none": "scoring pass, nothing is generated"},
        signal_definitions={
            "marker_logprobs": "next-token log-probabilities of the '+'/'-' marker tokens "
                               "at each step boundary, renormalised over {+,-}",
            "S_j": "Eq. 6 cumulative score for candidate first-error step j",
        },
        logits_stage="raw",
        official_code_url="", official_code_commit="none-published",
        container_image=os.environ.get("SLURM_CONTAINER_IMAGE",
                                       "nvcr.io/nvidia/pytorch:25.01-py3"),
        evaluator_revision=EV.EVALUATOR_REVISION,
        declared_deviations=[
            {"field": "marker_surface_form", "paper_says": "markers '+'/'-' (Eq. 4), no code",
             "we_do": "literal ' +' / ' -' with a convention-explaining system message, "
                      "marker token ids derived per following-context",
             "why": "no released prompt; a raw solution gives the model no reason to read "
                    "those characters as evaluative"},
            {"field": "forward_passes", "paper_says": "one marked sequence per candidate j",
             "we_do": "one all-'+' pass, S(j) as a cumulative sum",
             "why": "steps before j are '+' for every j, so the boundary distributions are "
                    "identical across candidates — algebraically equivalent, T-times cheaper"},
        ],
        repo_root=REPO_ROOT,
        extra={"mode": args.mode, "subsets": subsets,
               "paper_targets_f1": PAPER_TARGETS,
               "not_uprm": "this is the paper's own Eq. 6 control, not the trained uPRM"},
    )
    man["expected_traces"] = len(rows)
    problems = verify_manifest(man, require_clean_tree=(args.mode == "full"))
    write_manifest(man, args.out)

    gate = Gate(f"L1-uprm-judge-{args.mode}", args.out)
    gate.check("manifest_complete", not problems, f"{len(problems)} problems", problems)
    gate.check("backbone_is_paper_backbone", "14B" in args.model,
               f"{args.model} — the control is only fair on Qwen2.5-14B-Instruct")
    gate.finish(raise_on_fail=True)
    if args.dry_run:
        print(f"[l1] DRY RUN OK — manifest builds and verifies "
              f"(fidelity={man['fidelity']}, {len(man['declared_deviations'])} deviations)",
              flush=True)
        return

    mdl, tok = load_model(args.model, attn_impl=args.attn_impl)
    mdl.eval()

    expected = [f"{r['subset']}:{r.get('id', i)}" for i, r in enumerate(rows)]
    writer = ShardWriter(args.out, expected_keys=expected)
    done, incomplete = writer.done_keys(), False

    for i, row in enumerate(rows):
        key = expected[i]
        if key in done:
            continue
        if STOP["flag"]:
            incomplete = True
            break
        t0 = time.time()
        try:
            scores = score_candidates(mdl, tok, row["problem"], row["steps"])
            pred, failed = localize_first_error(scores), None
        except ValueError as e:
            # A tokenization-assumption break on one row must not lose the shard. It is
            # recorded as an unparsed prediction, which processbench_f1 counts as wrong —
            # never dropped, which would let the metric rise by refusing hard rows.
            scores, pred, failed = {}, None, str(e)
        writer.add({
            "trace_key": key, "question_id": key, "subset": row["subset"],
            "prompt_text": SYSTEM_PROMPT, "prompt_token_ids": [],
            "gen_token_ids": [], "full_text": "",
            "problem": row["problem"], "steps": row["steps"],
            "generator": row.get("generator"),
            "label": int(row["label"]), "scores": scores, "prediction": pred,
            "failed": failed, "wall_s": round(time.time() - t0, 2),
        })
        if (i + 1) % 25 == 0:
            print(f"[l1] {i + 1}/{len(rows)} pred={pred} label={row['label']}", flush=True)

    writer.close()
    free_memory()
    if incomplete:
        print("[l1] PREEMPTED — resubmit with the same --out to resume", flush=True)
        sys.exit(EXIT_INCOMPLETE)

    summarize(args.out, subsets)
    print(f"\nL1 COMPLETE -> {args.out}", flush=True)


def summarize(run_dir: str, subsets):
    by_subset = {s: {"preds": [], "labels": []} for s in subsets}
    n_failed = 0
    for rec in read_shards(run_dir, verify=False):
        b = by_subset.setdefault(rec["subset"], {"preds": [], "labels": []})
        b["preds"].append(rec["prediction"])
        b["labels"].append(rec["label"])
        n_failed += int(rec.get("failed") is not None)
    per = {s: EV.processbench_f1(v["preds"], v["labels"]) for s, v in by_subset.items()
           if v["labels"]}
    out = {"evaluator_revision": EV.EVALUATOR_REVISION, "per_subset": per,
           "macro_f1": EV.macro_f1(per), "n_tokenization_failures": n_failed,
           "paper_targets_f1": PAPER_TARGETS,
           "note": "paper targets are regression references, not acceptance gates"}
    for s, st in per.items():
        tgt = PAPER_TARGETS.get(s)
        print(f"[l1] {s}: F1={100 * st['f1']:.1f} (err {100 * st['error_acc']:.1f} / "
              f"cor {100 * st['correct_acc']:.1f}) n={st['n_total']}"
              + (f"  [paper {tgt}]" if tgt else ""), flush=True)
    print(f"[l1] macro F1 = {100 * out['macro_f1']:.1f}", flush=True)
    path = os.path.join(run_dir, "SUMMARY.json")
    with open(path + ".tmp", "w") as f:
        json.dump(out, f, indent=2, default=float)
    os.replace(path + ".tmp", path)
    return out


if __name__ == "__main__":
    main()
