#!/usr/bin/env python
"""
S2 — LEASH native single-trace stopping (AIRCC cluster driver).

"Logit-Entropy Adaptive Stopping Heuristic" (Quamar & Areeb, arXiv:2511.04654v1) on its
native cells: GSM8K (300-example test subset) and AQuA-RAT test, across the four published
models. Rationale decoding T=0.7 / top-p 0.95 with EOS disabled; final answer greedy.

Every row is `paper-specified-partial` and says so. Four constants the algorithm cannot run
without — the logit clip band `B`, the saturation threshold `tau_p`, the warm-up `w`, and
the entropy-drop gate `gamma` — are absent from the paper. They are declared in
`spectral_utils/paper_exact/leash.py`, swept on pilot IDs with `--sweep`, and one central
choice is frozen before the full run. **The best grid point is never the reproduction**
(handoff §S2); `--sweep` therefore refuses to run in full mode.

Published regression targets (LEASH / CoT accuracy, token reduction):
    Llama-3.1-8B  GSM8K 62.32 / 74.33, -30.97%   AQuA 54.68 / 63.20, -28.60%
    Mistral-7B    GSM8K 38.67 / 47.20, -35.12%   AQuA 19.25 / 26.38, -34.20%
    Phi-3-Mini    GSM8K 69.87 / 82.67, -41.50%   AQuA 50.24 / 61.67, -28.30%
    Qwen2.5-7B    GSM8K 54.85 / 65.33, -33.45%   AQuA 68.15 / 77.35, -28.15%

Its operating point buys ~30% tokens at ~10 accuracy points, so it enters our tables as a
declared sensitivity baseline on the full accuracy-versus-token frontier, never as a
matched-accuracy competitor.

Usage:
    python cluster/run_paper_exact_leash.py --model meta-llama/Llama-3.1-8B-Instruct \
        --dataset gsm8k --mode pilot --sweep --out $SHARED/results/paper_exact/s2_leash_pilot
    python cluster/run_paper_exact_leash.py --model meta-llama/Llama-3.1-8B-Instruct \
        --dataset gsm8k --mode full --out $SHARED/results/paper_exact/s2_leash_llama_gsm8k
"""
import argparse
import hashlib
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

import numpy as np
import torch

from spectral_utils import load_model, free_memory
from spectral_utils.paper_exact import evaluator as EV
from spectral_utils.paper_exact import leash as LS
from spectral_utils.paper_exact.gates import Gate
from spectral_utils.paper_exact.manifest import build_manifest, write_manifest, verify_manifest
from spectral_utils.paper_exact.shards import ShardWriter, read_shards
from spectral_utils.paper_exact.telemetry import DecodeConfig, stream_generate

EXIT_INCOMPLETE = 85
STOP = {"flag": False}
PAPER_PDF = os.path.join(
    REPO_ROOT, "papers",
    "LEASH Logit-Entropy Adaptive Stopping Heuristic for Efficient Chain-of-Thought Reasoning (arXiv 2511.04654v1).pdf")

#: DECLARED: the paper says only "All methods have prompts according to the task they need
#: to perform" and gives no template, and no GSM8K sampling seed.
RATIONALE_PROMPT = ("{question}\n\nThink step by step and explain your reasoning.")
ANSWER_PROMPT = "\n\nTherefore, the final answer is"
GSM8K_SUBSET_SEED = 42
GSM8K_SUBSET_N = 300


def _on_sigterm(signum, frame):
    STOP["flag"] = True
    print("[s2] SIGTERM — will checkpoint after the current question", flush=True)


def load_rows(dataset: str, n=None):
    from datasets import load_dataset
    if dataset == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        rng = np.random.default_rng(GSM8K_SUBSET_SEED)
        pick = sorted(rng.choice(len(ds), size=min(GSM8K_SUBSET_N, len(ds)), replace=False))
        rows = [{"question_id": f"gsm8k:{i}", "index": int(i), "problem": ds[int(i)]["question"],
                 "answer": ds[int(i)]["answer"].split("####")[-1].strip()} for i in pick]
    elif dataset == "aqua":
        ds = load_dataset("deepmind/aqua_rat", "raw", split="test")
        rows = [{"question_id": f"aqua:{i}", "index": i,
                 "problem": r["question"] + "\nOptions: " + " ".join(r["options"]),
                 "answer": str(r["correct"])} for i, r in enumerate(ds)]
    else:
        raise ValueError(f"unknown dataset {dataset!r}")
    return rows[:n] if n else rows


def run_question(mdl, tok, row, cfg: LS.LeashConfig, arm: str, generator, tokenizer_eos):
    """One question under `arm` in {'leash', 'cot', 'nocot'}."""
    t0 = time.time()
    q = RATIONALE_PROMPT.format(question=row["problem"])
    chat = tok.apply_chat_template([{"role": "user", "content": q}],
                                   tokenize=False, add_generation_prompt=True)
    ids = torch.tensor(tok(chat, add_special_tokens=False).input_ids)

    if arm == "nocot":
        # No-CoT: the direct-answer control, no rationale at all.
        acfg = DecodeConfig(**LS.ANSWER_DECODING, max_new_tokens=48, logprob_top_k=0,
                            eos_token_ids=tokenizer_eos, keep_top_k_arrays=False)
        direct = tok.apply_chat_template(
            [{"role": "user", "content": row["problem"] +
              "\n\nGive only the final numeric answer."}],
            tokenize=False, add_generation_prompt=True)
        out = stream_generate(mdl, tok, torch.tensor(
            tok(direct, add_special_tokens=False).input_ids), acfg, generator=generator)
        rationale, stopper = {"n_tokens": 0, "channels": {}, "stop_reason": "n/a",
                              "full_text": "", "gen_token_ids": []}, None
        answer = out
    else:
        stopper = LS.LeashStopper(cfg) if arm == "leash" else None
        # EOS is disabled during the rationale (Algorithm 1 line 2) for BOTH arms: the
        # comparison is between stopping rules, so the CoT control must face the same
        # generation contract and stop only at M.
        rcfg = DecodeConfig(**LS.RATIONALE_DECODING, max_new_tokens=cfg.M,
                            logprob_top_k=50, eos_token_ids=(), keep_top_k_arrays=True)
        fired = {"at": None}

        def check(_text, ch):
            """`ch` is the live TokenChannels; LEASH reads only the newest step's
            raw-logit signals, which is what makes it causal by construction."""
            if stopper.push(ch.raw_entropy[-1], ch.raw_margin[-1], ch.raw_pmax[-1]):
                fired["at"] = len(ch.raw_entropy)
                return True
            return False

        rationale = stream_generate(mdl, tok, ids, rcfg,
                                    stop_check=check if arm == "leash" else None,
                                    generator=generator)
        # Second stage: greedy short answer conditioned on the rationale (Algorithm 1 line 11).
        acfg = DecodeConfig(**LS.ANSWER_DECODING, max_new_tokens=48, logprob_top_k=0,
                            eos_token_ids=tokenizer_eos, keep_top_k_arrays=False)
        a_ids = torch.tensor(tok(chat + rationale["full_text"] + ANSWER_PROMPT,
                                 add_special_tokens=False).input_ids)
        answer = stream_generate(mdl, tok, a_ids, acfg, generator=generator)

    graded = EV.grade_math(answer["full_text"], row["answer"])
    return {
        "trace_key": f"{arm}:{cfg.setting_label}:{row['question_id']}",
        "question_id": row["question_id"], "arm": arm, "setting_label": cfg.setting_label,
        "prompt_text": q, "prompt_token_ids": ids.tolist(),
        "gen_token_ids": rationale["gen_token_ids"], "full_text": rationale["full_text"],
        "channels": rationale["channels"],
        "answer_text": answer["full_text"], "answer_token_ids": answer["gen_token_ids"],
        "n_reasoning_tokens": rationale["n_tokens"], "n_closure_tokens": answer["n_tokens"],
        "n_total_tokens": rationale["n_tokens"] + answer["n_tokens"],
        "stop_reason": rationale["stop_reason"],
        "stopped_early": rationale["stop_reason"] == "policy",
        "closure_generated": True,
        "leash": stopper.diagnostics() if stopper is not None else None,
        "gold_answer": row["answer"], "correct": graded["correct"],
        "pred_answer": graded["pred_answer"], "parse_status": graded["parse_status"],
        "wall_s": round(time.time() - t0, 2),
        **({"raw_top_k_logprobs": rationale["raw_top_k_logprobs"]}
           if "raw_top_k_logprobs" in rationale else {}),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--model-revision", default="main")
    ap.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "aqua"])
    ap.add_argument("--arms", default="leash,cot,nocot")
    ap.add_argument("--mode", default="pilot", choices=["smoke", "pilot", "full"])
    ap.add_argument("--n-samples", type=int, default=None)
    ap.add_argument("--sweep", action="store_true",
                    help="pilot-only: run the pre-registered sensitivity grid")
    ap.add_argument("--out", required=True)
    ap.add_argument("--attn-impl", default="sdpa")
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, _on_sigterm)
    if args.sweep and args.mode == "full":
        sys.exit("--sweep is pilot-only. Handoff §S2: freeze one central choice before the "
                 "full evaluation; the best post-hoc grid point is never the reproduction.")

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    n = args.n_samples if args.n_samples is not None else (
        3 if args.mode == "smoke" else 30 if args.mode == "pilot" else None)
    rows = load_rows(args.dataset, n)
    configs = LS.grid_points() if args.sweep else [LS.LeashConfig()]
    # The grid is swept for the leash arm only — cot/nocot do not read the constants, and
    # running them 81 times would multiply the control's cost for identical traces.
    print(f"[s2] model={args.model} dataset={args.dataset} arms={arms} rows={len(rows)} "
          f"configs={len(configs)}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    from transformers import AutoTokenizer
    tok_only = AutoTokenizer.from_pretrained(args.model)
    man = build_manifest(
        run_id=os.path.basename(args.out.rstrip("/")),
        paper_title="LEASH: Logit-Entropy Adaptive Stopping Heuristic",
        paper_pdf_path=PAPER_PDF, fidelity="paper-specified-partial",
        dataset_source={"gsm8k": "openai/gsm8k", "aqua": "deepmind/aqua_rat"}[args.dataset],
        dataset_revision="test",
        dataset_example_ids=[r["question_id"] for r in rows],
        model_id=args.model, model_revision=args.model_revision,
        prompt_text=RATIONALE_PROMPT, chat_template=tok_only.chat_template or "",
        decoding={"rationale": LS.RATIONALE_DECODING, "answer": LS.ANSWER_DECODING,
                  "eos_disabled_during_rationale": True},
        seed_policy={"seed": 42, "gsm8k_subset_seed": GSM8K_SUBSET_SEED},
        max_new_tokens=LS.M_MAX,
        stop_behavior={"rationale_cap": LS.M_MAX, "policy": "LEASH Alg. 1 (leash arm only)",
                       "second_stage": ANSWER_PROMPT},
        signal_definitions={"H": "Eq. 1 full-vocabulary entropy on clipped fp32 logits",
                            "M": "Eq. 2 top-two log-probability margin",
                            "pmax": "Eq. 3 saturation indicator input"},
        logits_stage="raw", official_code_url="", official_code_commit="none-published",
        container_image=os.environ.get("SLURM_CONTAINER_IMAGE",
                                       "nvcr.io/nvidia/pytorch:25.01-py3"),
        evaluator_revision=EV.EVALUATOR_REVISION,
        declared_deviations=[
            {"field": "B / tau_p / w / gamma",
             "paper_says": "claims concrete settings are reported; they are not in the PDF",
             "we_do": f"declared {LS.CENTRAL_CHOICE}, swept on pilot IDs",
             "why": "the algorithm cannot run without them"},
            {"field": "prompts", "paper_says": "'prompts according to the task'",
             "we_do": f"{RATIONALE_PROMPT!r} / {ANSWER_PROMPT!r}",
             "why": "no template published"},
            {"field": "gsm8k_subset", "paper_says": "random 300 with a fixed undisclosed seed",
             "we_do": f"numpy default_rng({GSM8K_SUBSET_SEED}), 300 of the test split",
             "why": "seed not disclosed"},
        ],
        repo_root=REPO_ROOT,
        extra={"mode": args.mode, "arms": arms, "sweep": args.sweep,
               "leash_config": LS.LeashConfig().as_manifest(),
               "sensitivity_grid": {k: list(v) for k, v in LS.SENSITIVITY_GRID.items()}},
    )
    man["expected_traces"] = len(rows) * (len(configs) if "leash" in arms else 0) + \
        len(rows) * len([a for a in arms if a != "leash"])
    problems = verify_manifest(man)
    write_manifest(man, args.out)

    gate = Gate(f"S2-leash-{args.mode}", args.out)
    gate.check("manifest_complete", not problems, f"{len(problems)} problems", problems)
    gate.check("sweep_is_pilot_only", not (args.sweep and args.mode == "full"),
               "the frozen central choice is what the full run uses")
    gate.check("fidelity_is_partial", man["fidelity"] == "paper-specified-partial",
               "four constants are declared by us, not by the paper")
    gate.finish(raise_on_fail=True)

    mdl, tok = load_model(args.model, attn_impl=args.attn_impl)
    mdl.eval()
    cfg_eos = getattr(mdl.generation_config, "eos_token_id", None) or tok.eos_token_id
    eos = tuple(int(e) for e in (cfg_eos if isinstance(cfg_eos, (list, tuple)) else [cfg_eos])
                if e is not None)

    expected = []
    for arm in arms:
        for c in (configs if arm == "leash" else [LS.LeashConfig()]):
            expected += [f"{arm}:{c.setting_label}:{r['question_id']}" for r in rows]
    writer = ShardWriter(args.out, expected_keys=expected)
    done, incomplete = writer.done_keys(), False
    generator = torch.Generator(device=mdl.device)

    for arm in arms:
        for c in (configs if arm == "leash" else [LS.LeashConfig()]):
            for row in rows:
                key = f"{arm}:{c.setting_label}:{row['question_id']}"
                if key in done:
                    continue
                if STOP["flag"]:
                    incomplete = True
                    break
                generator.manual_seed(
                    42 + int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % (2 ** 31))
                try:
                    rec = run_question(mdl, tok, row, c, arm, generator, eos)
                except Exception as e:  # noqa: BLE001
                    writer.add_failure(key, row["question_id"], repr(e))
                    print(f"[s2] FAILED {key}: {e!r}", flush=True)
                    continue
                writer.add(rec)
                print(f"[s2] {key} tok={rec['n_total_tokens']} stop={rec['stop_reason']} "
                      f"correct={rec['correct']} {rec['wall_s']}s", flush=True)
            if incomplete:
                break
        if incomplete:
            break

    writer.close()
    free_memory()
    if incomplete:
        print("[s2] PREEMPTED — resubmit with the same --out to resume", flush=True)
        sys.exit(EXIT_INCOMPLETE)

    summarize(args.out)
    print(f"\nS2 COMPLETE -> {args.out}", flush=True)


def summarize(run_dir: str):
    cells = {}
    for rec in read_shards(run_dir, verify=False):
        cells.setdefault((rec["arm"], rec["setting_label"]), []).append(rec)
    out = {"evaluator_revision": EV.EVALUATOR_REVISION, "cells": {}}
    for (arm, label), recs in sorted(cells.items()):
        ta = EV.token_accounting(recs)
        out["cells"][f"{arm}|{label}"] = {
            "n": len(recs), "pass_at_1": EV.pass_at_1([r["correct"] for r in recs]),
            "n_stopped_early": sum(1 for r in recs if r["stopped_early"]), **ta}
        print(f"[s2] {arm}|{label}: acc={out['cells'][f'{arm}|{label}']['pass_at_1']:.4f} "
              f"tokens={ta['total_tokens']}", flush=True)
    base = out["cells"].get("cot|central")
    if base:
        for k, v in out["cells"].items():
            if k.startswith("leash"):
                v["token_reduction_vs_cot"] = (
                    1.0 - v["total_tokens"] / base["total_tokens"]) if base["total_tokens"] else None
                v["accuracy_delta_vs_cot"] = v["pass_at_1"] - base["pass_at_1"]
    path = os.path.join(run_dir, "SUMMARY.json")
    with open(path + ".tmp", "w") as f:
        json.dump(out, f, indent=2, default=float)
    os.replace(path + ".tmp", path)
    return out


if __name__ == "__main__":
    main()
