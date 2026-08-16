#!/usr/bin/env python
"""
S1 — REFRAIN native single-trace stopping reproduction (AIRCC cluster driver).

Reproduces "Stop When Enough" (ACL 2026, 2026.acl-long.1256) on its primary cell:
Qwen3-8B thinking mode x MATH-500, prompt P0, T=0.6 / top-p 0.95 / top-k 20, 16,384-token
cap, seed 42. Runs BOTH arms — `vanilla` and `refrain` — through the identical prompt,
sampler, tokenizer, EOS set and telemetry, so the accuracy-versus-token frontier is a
controlled comparison and not a comparison of two pipelines.

Published regression targets (NOT acceptance gates, handoff §1):
    vanilla  91.40% pass@1 / 2.64M tokens
    REFRAIN  91.20% pass@1 / 1.61M tokens

Fidelity: `paper-specified-partial`. The official repository is a release-placeholder
README, and three constants are declared by us (provisional-answer cue, reward timing,
cold-start/tie order) — see `spectral_utils/paper_exact/refrain.py`.

REFRAIN cannot be sharded
-------------------------
Its SW-UCB reward buffers persist across questions, so the refrain arm must walk the frozen
MATH-500 order on ONE worker, in order. The bandit state is checkpointed with the shards and
restored on requeue; without that a preempted job would silently restart the bandit cold
mid-dataset and produce a trace file that looks perfectly normal. The vanilla arm has no
such coupling and may be sharded with --shard/--n-shards.

Also captures the four frozen project channels on every token of both arms, so our own
causal method can later be evaluated on exactly these traces (handoff §S1). Our forced-closure
policy is a SEPARATE adapted-common-protocol experiment and is not run here.

Usage:
    # implementation pilot (30 ordered questions) — not a reproduction
    python cluster/run_paper_exact_refrain.py --arms vanilla,refrain --n-samples 30 \
        --out $SHARED/results/paper_exact/s1_refrain_pilot --mode pilot

    # full paper-specified cell
    python cluster/run_paper_exact_refrain.py --arms vanilla,refrain \
        --out $SHARED/results/paper_exact/s1_refrain_full --mode full
"""
import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:  # NGC torch 2.6.0a0 trips transformers' version guard; see cluster/run_inference.py
    import transformers.modeling_utils as _mu
    _mu.check_torch_load_is_safe = lambda *a, **k: None
except Exception:
    pass

import numpy as np
import torch

from spectral_utils import load_model, free_memory
from spectral_utils.paper_exact import evaluator as EV
from spectral_utils.paper_exact import refrain as RF
from spectral_utils.paper_exact.gates import Gate
from spectral_utils.paper_exact.manifest import build_manifest, write_manifest, verify_manifest
from spectral_utils.paper_exact.shards import ShardWriter
from spectral_utils.paper_exact.telemetry import DecodeConfig, stream_generate, score_continuation

EXIT_INCOMPLETE = 85
STOP = {"flag": False}
PAPER_PDF = os.path.join(
    REPO_ROOT, "papers",
    "Stop When Enough Adaptive Early-Stopping for Chain-of-Thought Reasoning (ACL 2026).pdf")
BANDIT_STATE = "BANDIT_STATE.json"


def _on_sigterm(signum, frame):
    STOP["flag"] = True
    print("[s1] SIGTERM — will checkpoint after the current question", flush=True)


# ── data ────────────────────────────────────────────────────────────────────────

def load_math500(n=None):
    """MATH-500 in its native dataset order. Order is pinned in the manifest because the
    bandit couples questions; a shuffle is a different algorithm, not a different seed."""
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    rows = []
    for i, r in enumerate(ds):
        rows.append({"question_id": str(r.get("unique_id", i)), "index": i,
                     "problem": r["problem"], "answer": str(r["answer"])})
    return rows[:n] if n else rows


# ── generation ──────────────────────────────────────────────────────────────────

def build_prompt_ids(tok, question: str):
    """P0 through Qwen3's chat template in official thinking mode."""
    text = RF.format_p0(question)
    chat = tok.apply_chat_template([{"role": "user", "content": text}],
                                   tokenize=False, add_generation_prompt=True,
                                   enable_thinking=True)
    return text, chat, torch.tensor(tok(chat, add_special_tokens=False).input_ids)


def eos_ids(tok, mdl):
    cfg = getattr(mdl.generation_config, "eos_token_id", None) or tok.eos_token_id
    ids = list(cfg) if isinstance(cfg, (list, tuple)) else [cfg]
    for name in ("<|im_end|>", "<|endoftext|>"):
        tid = tok.convert_tokens_to_ids(name)
        if tid is not None and tid >= 0 and tid not in ids:
            ids.append(tid)
    return tuple(int(i) for i in ids if i is not None)


def forced_closure(mdl, tok, chat_prompt: str, reasoning_text: str, cfg: DecodeConfig,
                   max_closure_tokens: int = 256, generator=None) -> dict:
    """REFRAIN §3.3: halt reasoning, then elicit the answer with `Final Answer: \\boxed{`.

    Closes the Qwen3 thinking block first — leaving `<think>` open would put the closure
    inside the reasoning channel, where the model continues reasoning instead of answering,
    and the "stop" would save no tokens at all.
    """
    closure_text = reasoning_text
    if "</think>" not in closure_text:
        closure_text += "\n</think>\n\n"
    closure_text += RF.CLOSURE_PROMPT
    ids = torch.tensor(tok(chat_prompt + closure_text, add_special_tokens=False).input_ids)
    ccfg = DecodeConfig(**{**cfg.__dict__, "max_new_tokens": int(max_closure_tokens)})
    out = stream_generate(mdl, tok, ids, ccfg, generator=generator)
    out["closure_prefix"] = RF.CLOSURE_PROMPT
    out["answer_text"] = RF.CLOSURE_PROMPT + out["full_text"]
    return out


def run_question(mdl, tok, sbert, row, arm: str, tau, cfg: DecodeConfig,
                 rcfg: RF.RefrainConfig, generator, max_closure_tokens: int) -> dict:
    """One question, one arm. Returns a complete acquisition record."""
    t0 = time.time()
    prompt_text, chat, ids = build_prompt_ids(tok, row["problem"])

    stopper = None
    if arm == "refrain":
        stopper = RF.StepStopper(sbert, tau, rcfg)
    gen = stream_generate(mdl, tok, ids, cfg,
                          stop_check=stopper if arm == "refrain" else None,
                          generator=generator)

    rec = {
        "trace_key": f"{arm}:{row['question_id']}",
        "question_id": row["question_id"],
        "arm": arm,
        "dataset_index": row["index"],
        "prompt_text": prompt_text,
        "chat_prompt": chat,
        "prompt_token_ids": ids.tolist(),
        "gen_token_ids": gen["gen_token_ids"],
        "full_text": gen["full_text"],
        "raw_text": gen["raw_text"],
        "channels": gen["channels"],
        "n_reasoning_tokens": gen["n_tokens"],
        "stop_reason": gen["stop_reason"],
        "stopped_early": gen["stop_reason"] == "policy",
        "tau": tau,
        "gold_answer": row["answer"],
        "wall_s": None,
    }
    for k in ("raw_top_k_logprobs", "sampled_top_k_logprobs"):
        if k in gen:
            rec[k] = gen[k]
    if stopper is not None:
        rec["stopper"] = stopper.diagnostics()

    # ── forced closure ──
    # REFRAIN forces closure after a policy stop. A trace that ended on EOS or the length
    # cap already contains its own answer; forcing another one would change the vanilla arm
    # into a different method and inflate its token count.
    closure_tokens, answer_source = 0, "natural"
    scored_text = gen["full_text"]
    if rec["stopped_early"]:
        clo = forced_closure(mdl, tok, chat, gen["raw_text"], cfg,
                             max_closure_tokens, generator)
        rec["closure"] = {"text": clo["answer_text"], "n_tokens": clo["n_tokens"],
                          "stop_reason": clo["stop_reason"], "channels": clo["channels"]}
        closure_tokens = clo["n_tokens"]
        answer_source = "forced_closure"
        scored_text = clo["answer_text"]
    rec["n_closure_tokens"] = closure_tokens
    rec["closure_generated"] = bool(closure_tokens) or not rec["stopped_early"]
    rec["answer_source"] = answer_source
    rec["n_total_tokens"] = rec["n_reasoning_tokens"] + closure_tokens

    graded = EV.grade_math(scored_text, row["answer"])
    rec.update({"correct": graded["correct"], "pred_answer": graded["pred_answer"],
                "parse_status": graded["parse_status"]})

    # ── Eq. 6 answer-only likelihood, for the bandit reward ──
    ans_text, ans_ids = RF.extract_boxed_answer_ids(tok, scored_text)
    if ans_ids:
        # rsplit, not split: extract_boxed takes the LAST \boxed{...} (models box
        # intermediate results too), so the context must end at that same one or the
        # likelihood would be conditioned on a prefix the answer never followed.
        ctx = tok(chat + scored_text.rsplit("\\boxed{", 1)[0] + "\\boxed{",
                  add_special_tokens=False).input_ids
        sc = score_continuation(mdl, tok, ctx, ans_ids)
    else:
        # No boxed region means Eq. 6 is undefined. Score 0 is the honest encoding: the
        # policy produced no well-formed answer, which the reward should penalise rather
        # than skip — skipping would hide unparseable arms from the bandit entirely.
        sc = {"score": 0.0, "mean_logprob": float("nan"), "n": 0}
    rec["answer_score"] = sc
    rec["boxed_answer_text"] = ans_text
    rec["wall_s"] = round(time.time() - t0, 2)
    return rec


# ── driver ──────────────────────────────────────────────────────────────────────

def save_bandit(run_dir, bandit, reward_state):
    path = os.path.join(run_dir, BANDIT_STATE)
    with open(path + ".tmp", "w") as f:
        json.dump({"bandit": bandit.state(), "reward": reward_state.state()}, f, indent=2)
    os.replace(path + ".tmp", path)


def load_bandit(run_dir, bandit, reward_state):
    path = os.path.join(run_dir, BANDIT_STATE)
    if not os.path.exists(path):
        return False
    with open(path) as f:
        st = json.load(f)
    bandit.load_state(st["bandit"])
    reward_state.load_state(st["reward"])
    print(f"[s1] restored bandit at round {bandit.k}, L_bar={reward_state.mean}", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--model-revision", default="main")
    ap.add_argument("--arms", default="vanilla,refrain")
    ap.add_argument("--mode", default="pilot", choices=["smoke", "pilot", "full"])
    ap.add_argument("--n-samples", type=int, default=None)
    ap.add_argument("--max-new", type=int, default=RF.MAX_NEW_TOKENS)
    ap.add_argument("--max-closure-tokens", type=int, default=256)
    ap.add_argument("--vocabulary", default="base", choices=sorted(RF.VOCABULARIES))
    ap.add_argument("--fixed-tau", type=float, default=None,
                    help="ablation: disable the bandit and pin tau")
    ap.add_argument("--logprob-top-k", type=int, default=50)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--attn-impl", default="sdpa")
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, _on_sigterm)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    if args.n_shards > 1 and "refrain" in arms:
        sys.exit("refrain's SW-UCB state crosses questions and cannot be sharded; "
                 "run --arms refrain with --n-shards 1")

    n = args.n_samples if args.n_samples is not None else (
        5 if args.mode == "smoke" else 30 if args.mode == "pilot" else None)
    rows = load_math500(n)
    if args.n_shards > 1:
        rows = [r for i, r in enumerate(rows) if i % args.n_shards == args.shard]
    print(f"[s1] mode={args.mode} arms={arms} n={len(rows)} out={args.out}", flush=True)

    rcfg = RF.RefrainConfig(vocabulary=args.vocabulary, max_new_tokens=args.max_new)
    os.makedirs(args.out, exist_ok=True)

    # ── manifest before anything is generated ──
    from transformers import AutoTokenizer
    tok_only = AutoTokenizer.from_pretrained(args.model)
    man = build_manifest(
        run_id=os.path.basename(args.out.rstrip("/")),
        paper_title="Stop When Enough: Adaptive Early-Stopping for Chain-of-Thought Reasoning",
        paper_pdf_path=PAPER_PDF,
        fidelity="paper-specified-partial",
        dataset_source="HuggingFaceH4/MATH-500", dataset_revision="test",
        dataset_example_ids=[r["question_id"] for r in rows],
        model_id=args.model, model_revision=args.model_revision,
        prompt_text=RF.PROMPT_P0, chat_template=tok_only.chat_template or "",
        decoding={**RF.QWEN3_DECODING, "logprob_top_k": args.logprob_top_k,
                  "thinking_mode": True},
        seed_policy={"seed": RF.SEED, "torch_generator": "cuda", "per_question_reseed": False},
        max_new_tokens=args.max_new,
        stop_behavior={"eos": "generation_config + <|im_end|>",
                       "cap": args.max_new,
                       "policy": "REFRAIN Alg. 1 (refrain arm only)",
                       "forced_closure": RF.CLOSURE_PROMPT,
                       "max_closure_tokens": args.max_closure_tokens},
        signal_definitions={
            "raw_entropy": "full-vocabulary Shannon entropy from raw logits",
            "raw_logsumexp": "log-partition over the full raw-logit vocabulary",
            "spilled_energy": "-log p(sampled token), raw",
            "deepconf_conf": "DeepConf paper_eq2 over raw top-20 logprobs",
            "answer_score": "REFRAIN Eq. 6 length-normalised geometric-mean likelihood",
        },
        logits_stage="both",
        official_code_url="https://github.com/RLSNLP/Adaptive-Reasoning",
        official_code_commit="release-placeholder-README (no runnable code at audit)",
        container_image=os.environ.get("SLURM_CONTAINER_IMAGE",
                                       "nvcr.io/nvidia/pytorch:25.01-py3"),
        evaluator_revision=EV.EVALUATOR_REVISION,
        declared_deviations=[
            {"field": "provisional_answer_cue_c",
             "paper_says": "e.g. 'answer is/should be'",
             "we_do": list(RF.PROVISIONAL_ANSWER_CUES),
             "why": "the paper gives an example, not the set"},
            {"field": "reward_timing",
             "paper_says": "R = Score - lambda*L/L_bar with L_bar a running mean",
             "we_do": "L_bar over previous rounds, updated after the reward",
             "why": "the paper does not say whether the current sample is included"},
            {"field": "cold_start_and_tie_order",
             "paper_says": "'arbitrary such t'; ties undefined",
             "we_do": "ascending tau for both",
             "why": "determinism and reproducibility"},
            {"field": "official_code", "paper_says": "code will be released",
             "we_do": "PDF-based implementation",
             "why": "repository was a release-placeholder README when audited"},
        ],
        repo_root=REPO_ROOT,
        extra={"refrain_config": rcfg.as_manifest(), "mode": args.mode,
               "arms": arms, "shard": args.shard, "n_shards": args.n_shards,
               "fixed_tau": args.fixed_tau},
    )
    man["expected_traces"] = len(rows) * len(arms)
    problems = [p for p in verify_manifest(man) if not p.startswith("repo_dirty")]
    write_manifest(man, args.out)

    gate = Gate(f"S1-refrain-{args.mode}", args.out)
    gate.check("manifest_complete", not problems, f"{len(problems)} problems", problems)
    gate.check("vocabulary_is_base_for_headline",
               args.vocabulary == "base" or args.mode != "full",
               f"vocabulary={args.vocabulary}; only 'base' reproduces the headline row")
    gate.finish(raise_on_fail=True)

    # ── model ──
    mdl, tok = load_model(args.model, attn_impl=args.attn_impl)
    mdl.eval()
    sbert = RF.load_sbert(rcfg.sbert_model, device=str(mdl.device)) if "refrain" in arms else None
    cfg = DecodeConfig(**RF.QWEN3_DECODING, max_new_tokens=args.max_new, seed=RF.SEED,
                       logprob_top_k=args.logprob_top_k, eos_token_ids=eos_ids(tok, mdl))

    expected = [f"{a}:{r['question_id']}" for a in arms for r in rows]
    writer = ShardWriter(args.out, expected_keys=expected)
    done = writer.done_keys()

    bandit = RF.SWUCB(rcfg.tau_grid, rcfg.window, rcfg.ucb_c)
    reward_state = RF.RewardState()
    load_bandit(args.out, bandit, reward_state)

    generator = torch.Generator(device=mdl.device)
    generator.manual_seed(RF.SEED)

    incomplete = False
    for arm in arms:
        for row in rows:
            key = f"{arm}:{row['question_id']}"
            if key in done:
                continue
            if STOP["flag"]:
                incomplete = True
                break
            if arm == "refrain":
                if args.fixed_tau is not None:
                    tau, diag = args.fixed_tau, {"reason": "fixed"}
                else:
                    tau, diag = bandit.select()
            else:
                tau, diag = None, None
            try:
                rec = run_question(mdl, tok, sbert, row, arm, tau, cfg, rcfg,
                                   generator, args.max_closure_tokens)
            except Exception as e:  # noqa: BLE001 — one bad row must not lose the shard
                writer.add_failure(key, row["question_id"], repr(e))
                print(f"[s1] FAILED {key}: {e!r}", flush=True)
                continue
            if arm == "refrain":
                rw = reward_state.reward(rec["answer_score"]["score"], rec["n_total_tokens"])
                if args.fixed_tau is None:
                    bandit.update(tau, rw["reward"])
                reward_state.observe(rec["n_total_tokens"])
                rec["bandit"] = {"selection": diag, "reward": rw}
                save_bandit(args.out, bandit, reward_state)
            writer.add(rec)
            print(f"[s1] {key} tau={tau} tok={rec['n_total_tokens']} "
                  f"stop={rec['stop_reason']} correct={rec['correct']} "
                  f"{rec['wall_s']}s", flush=True)
        if incomplete:
            break

    writer.close()
    if "refrain" in arms:
        save_bandit(args.out, bandit, reward_state)
    free_memory()

    if incomplete:
        print("[s1] PREEMPTED — resubmit with the same --out to resume", flush=True)
        sys.exit(EXIT_INCOMPLETE)

    summarize(args.out, arms)
    print(f"\nS1 COMPLETE -> {args.out}", flush=True)


def summarize(run_dir: str, arms):
    """Per-arm pass@1 and token accounting, written next to the acquisition.

    A convenience summary only. The reported table is built offline by
    scripts/paper_exact_report.py from the shards, so this file is never the source of a
    published number.
    """
    from spectral_utils.paper_exact.shards import read_shards
    per_arm = {a: [] for a in arms}
    for rec in read_shards(run_dir, verify=False):
        per_arm.setdefault(rec["arm"], []).append(rec)
    out = {"written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "evaluator_revision": EV.EVALUATOR_REVISION, "arms": {}}
    for arm, recs in per_arm.items():
        if not recs:
            continue
        ta = EV.token_accounting(recs)
        out["arms"][arm] = {
            "n": len(recs),
            "pass_at_1": EV.pass_at_1([r["correct"] for r in recs]),
            "parser_coverage": EV.parser_coverage([r["parse_status"] for r in recs]),
            "stop_reasons": {k: sum(1 for r in recs if r["stop_reason"] == k)
                             for k in ("eos", "length", "policy")},
            **ta,
        }
        print(f"[s1] {arm}: pass@1={out['arms'][arm]['pass_at_1']:.4f} "
              f"tokens={ta['total_tokens']} ({ta['mean_tokens_per_trace']:.0f}/trace)",
              flush=True)
    path = os.path.join(run_dir, "SUMMARY.json")
    with open(path + ".tmp", "w") as f:
        json.dump(out, f, indent=2, default=float)
    os.replace(path + ".tmp", path)
    return out


if __name__ == "__main__":
    main()
