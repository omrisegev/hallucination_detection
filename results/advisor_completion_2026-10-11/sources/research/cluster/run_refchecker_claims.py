#!/usr/bin/env python
"""
RefChecker claim-level panel (Benchmark 2b of
docs/experiments/FOUR_LOCALIZATION_BENCHMARKS_CLUSTER_HANDOFF.md) — the published open checker
AND our own teacher-forced telemetry, computed in ONE pass over the SAME claim rows.

Both arms in one driver on purpose: the apples-to-apples rule requires identical example ids,
identical claim text, and identical gold. Running them as two jobs would make that an assertion
to be checked later; running them in one loop makes it true by construction.

ARM 1 — the competitor: RefChecker's own `NLIChecker`, reproduced
--------------------------------------------------------------------------------------
Faithful to `refchecker/checker/nli_checker.py` (fetched 2026-08-11):
  * model `ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli` (their default);
  * `tokenizer(references, claims)` as a PAIR, `max_length=512`, truncation on;
  * `argmax(softmax(logits))` into `["Entailment", "Neutral", "Contradiction"]`;
  * multi-passage merge from `checker_base.merge_multi_psg_ret`: Entailment if ANY passage
    entails, else Contradiction if ANY contradicts, else Neutral.
This is the strongest FULLY OPEN checker in the official codebase. The paper's strongest
configuration additionally uses proprietary extractors/checkers (GPT-4 / Claude 2); those
numbers stay quoted as published context and are never reproduced here.

ARM 2 — ours: evidence-contrast telemetry over the same fixed claim
--------------------------------------------------------------------------------------
Each claim's deterministic textual rendering (`spectral_utils.refchecker.triplet_text`) is
teacher-forced under two conditions — with the reference in the prompt (`full`) and without it
(`noctx`) — saving the same per-token quantities every other arm in this project consumes. This
is the RAGTruth evidence-contrast design transplanted to a claim unit, and it is an
**ADAPTATION (fidelity level 3)**, not a reproduction of anything: a separately scored claim is
not an unchanged span of the original response, and the report must say so.

WHAT THIS PANEL DOES NOT DO
--------------------------------------------------------------------------------------
It does not run claim EXTRACTION. The benchmark's human labels are attached to triplets
extracted by Claude 2, so a different extractor produces claims the gold does not cover. Fixing
the claim set to the shipped, human-labelled triplets is what makes the two arms comparable at
all — and it means this panel measures the CHECKING stage only. Recorded in the manifest.

Usage:
    python cluster/run_refchecker_claims.py \\
        --data-dir data/refchecker_protocol \\
        --out $SHARED/results/refchecker_knowhalbench_open_full
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

try:
    import transformers.modeling_utils as _mu
    _mu.check_torch_load_is_safe = lambda *a, **k: None
except Exception:
    pass

from backfill_views import forward_batch, candidate_quantities
from spectral_utils import load_model, load_cache, save_cache_atomic, free_memory
from spectral_utils.model_utils import fmt_prompt
from spectral_utils.refchecker import (
    CONTEXT_SETTINGS, LABELS, load_refchecker, three_way_metrics,
)

EXIT_INCOMPLETE = 85
STOP = {"flag": False}

NLI_MODEL = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
NLI_MAX_LENGTH = 512


def _on_sigterm(signum, frame):
    STOP["flag"] = True
    print("[refchecker] SIGTERM received — will checkpoint after the current batch", flush=True)


def merge_multi_psg_ret(labels):
    """`refchecker/checker/checker_base.py::merge_multi_psg_ret`, verbatim."""
    if "Entailment" in labels:
        return "Entailment"
    if "Contradiction" in labels:
        return "Contradiction"
    return "Neutral"


@torch.no_grad()
def nli_check(model, tokenizer, claims, batch_size=16, device="cuda"):
    """One (reference, claim) pair per passage; returns a flat list of per-pair labels."""
    pairs = []
    for i, claim in enumerate(claims):
        for passage in (claim["context"] or [""]):
            pairs.append((i, passage, claim["claim_text"]))

    labels = []
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start:start + batch_size]
        inputs = tokenizer([p for _, p, _ in chunk], [c for _, _, c in chunk],
                           max_length=NLI_MAX_LENGTH, truncation=True,
                           return_tensors="pt", padding=True, return_token_type_ids=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        preds = model(**inputs).logits.softmax(dim=-1).argmax(dim=-1).cpu()
        labels.extend(LABELS[int(p)] for p in preds)

    per_claim = {}
    for (i, _, _), label in zip(pairs, labels):
        per_claim.setdefault(i, []).append(label)
    return [merge_multi_psg_ret(per_claim.get(i, ["Neutral"])) for i in range(len(claims))]


def claim_prompt(claim, with_context: bool) -> str:
    """The two evidence conditions. Only the reference block differs; every other character is
    identical between conditions, mirroring the prompt-surgery contract used on RAGTruth."""
    question = claim["question"].strip()
    if with_context:
        reference = "\n\n".join(p.strip() for p in (claim["context"] or []) if p.strip())
        return (f"Question: {question}\n\nReference:\n{reference}\n\n"
                f"State one fact from the reference.\nFact:")
    return f"Question: {question}\n\nState one fact from the reference.\nFact:"


def score_telemetry(mdl, tok, claims, cache, out_path, cfg):
    """Teacher-force each claim under both conditions. Cache key `f"{claim_key}::{condition}"`,
    so resume is exact at (claim, condition) granularity."""
    todo = [c for c in claims if f"{claim_key(c)}::noctx" not in cache]
    print(f"[refchecker] telemetry: {len(claims) - len(todo)}/{len(claims)} claims done",
          flush=True)

    for i, claim in enumerate(todo):
        if STOP["flag"]:
            save_cache_atomic(cache, out_path)
            print(f"PREEMPTED — checkpoint saved with {len(cache)} items", flush=True)
            return False
        gen_ids = tok(claim["claim_text"], add_special_tokens=False).input_ids
        if not gen_ids:
            continue
        items = []
        for condition, with_context in (("full", True), ("noctx", False)):
            prompt = fmt_prompt(tok, claim_prompt(claim, with_context))
            items.append({"prompt_ids": tok(prompt).input_ids, "gen_ids": gen_ids,
                          "condition": condition})
        slices = forward_batch(mdl, items)
        for it, logits in zip(items, slices):
            q = candidate_quantities(logits, gen_ids, warpers=None,
                                     raw_top_k=cfg.logprob_top_k, post_top_k=cfg.logprob_top_k)
            cache[f"{claim_key(claim)}::{it['condition']}"] = {
                "example_id": claim["example_id"], "setting": claim["setting"],
                "generator": claim["generator"], "claim_index": claim["claim_index"],
                "triplet": claim["triplet"], "claim_text": claim["claim_text"],
                "human_label": claim["human_label"],
                "label_unsupported": claim["label_unsupported"],
                "condition": it["condition"],
                "gen_token_ids": gen_ids,
                "token_entropies": q["token_entropies_recomputed"],
                "token_spilled_energies": q["token_spilled_energies"],
                "token_logsumexp": q["token_logsumexp"],
                "top_k_logprobs": q["top_k_logprobs_raw"],
            }
        del slices
        if (i + 1) % 200 == 0 or i + 1 == len(todo):
            print(f"[refchecker] telemetry {i + 1}/{len(todo)} claims", flush=True)
        if (i + 1) % cfg.checkpoint_every == 0:
            save_cache_atomic(cache, out_path)

    save_cache_atomic(cache, out_path)
    return True


def claim_key(claim) -> str:
    return f"{claim['setting']}|{claim['generator']}|{claim['example_id']}|{claim['claim_index']}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data-dir", default=os.path.join(REPO_ROOT, "data", "refchecker_protocol"))
    ap.add_argument("--scorer-model", default="Qwen/Qwen3-8B")
    ap.add_argument("--nli-model", default=NLI_MODEL)
    ap.add_argument("--nli-batch-size", type=int, default=16)
    ap.add_argument("--settings", default=",".join(CONTEXT_SETTINGS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--logprob-top-k", type=int, default=50)
    ap.add_argument("--checkpoint-every", type=int, default=200)
    ap.add_argument("--skip-telemetry", action="store_true")
    cfg = ap.parse_args()

    signal.signal(signal.SIGTERM, _on_sigterm)
    out_dir = cfg.out or "results_refchecker"
    os.makedirs(out_dir, exist_ok=True)

    settings = [s.strip() for s in cfg.settings.split(",") if s.strip()]
    claims, diag = load_refchecker(data_dir=cfg.data_dir, settings=settings)
    print(f"[refchecker] loaded {len(claims)} claims: {json.dumps(diag)}", flush=True)
    if not claims:
        print("[refchecker] BLOCKED — no claims loaded; the corpus is missing or unusable. "
              "Run cluster/prepare_refchecker_data.py first.", flush=True)
        sys.exit(2)

    t0 = time.time()

    # ── ARM 1: the open NLI checker ──────────────────────────────────────────────
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    print(f"[refchecker] NLI checker: {cfg.nli_model}", flush=True)
    nli_tok = AutoTokenizer.from_pretrained(cfg.nli_model)
    nli_mdl = AutoModelForSequenceClassification.from_pretrained(cfg.nli_model).to("cuda").eval()
    preds = nli_check(nli_mdl, nli_tok, claims, batch_size=cfg.nli_batch_size)
    del nli_mdl
    free_memory()

    gold = [c["human_label"] for c in claims]
    checker_results = {"overall": three_way_metrics(gold, preds)}
    for setting in settings:
        idx = [i for i, c in enumerate(claims) if c["setting"] == setting]
        if idx:
            checker_results[setting] = three_way_metrics([gold[i] for i in idx],
                                                         [preds[i] for i in idx])

    checker_path = os.path.join(out_dir, "nli_checker_predictions.json")
    with open(checker_path + ".tmp", "w") as f:
        json.dump([{ "claim_key": claim_key(c), "example_id": c["example_id"],
                     "setting": c["setting"], "generator": c["generator"],
                     "claim_index": c["claim_index"], "claim_text": c["claim_text"],
                     "human_label": c["human_label"], "predicted_label": p}
                   for c, p in zip(claims, preds)], f, indent=2)
    os.replace(checker_path + ".tmp", checker_path)
    print("\n=== NLI CHECKER (3-way) ===")
    print(json.dumps({k: {"accuracy": v["accuracy"], "macro_f1": v["macro_f1"], "n": v["n"]}
                      for k, v in checker_results.items()}, indent=2))

    # ── ARM 2: our teacher-forced telemetry over the identical claims ────────────
    telemetry_complete = True
    if not cfg.skip_telemetry:
        mdl, tok = load_model(cfg.scorer_model, quantize_4bit=False)
        out_path = os.path.join(out_dir, "refchecker_claim_telemetry.pkl")
        cache = load_cache(out_path)
        telemetry_complete = score_telemetry(mdl, tok, claims, cache, out_path, cfg)

    manifest = {
        "driver": "run_refchecker_claims.py",
        "paper": "Knowledge-Centric Hallucination Detection / RefChecker "
                 "(Hu et al., EMNLP 2024, aclanthology.org/2024.emnlp-main.395/)",
        "benchmark_data": "github.com/amazon-science/RefChecker benchmark/, assembled by "
                          "cluster/prepare_refchecker_data.py",
        "claim_set": "FIXED to the shipped, human-labelled claude2_response_kg triplets. Claim "
                     "EXTRACTION is out of scope: the human labels are attached to Claude-2 "
                     "extracted triplets, so a different extractor yields claims this gold does "
                     "not cover. This panel measures the CHECKING stage only.",
        "arms": {
            "competitor": {
                "method": "RefChecker NLIChecker (strongest fully open checker in the official "
                          "codebase)",
                "model": cfg.nli_model,
                "protocol": "tokenizer(references, claims) pair, max_length=512, truncation, "
                            "argmax softmax into [Entailment, Neutral, Contradiction]; "
                            "multi-passage merge = Entailment if any, else Contradiction if any, "
                            "else Neutral (checker_base.merge_multi_psg_ret)",
                "fidelity_level": 2,
                "results": checker_results,
            },
            "ours": {
                "method": "evidence-contrast teacher-forced telemetry over the identical fixed "
                          "claim (full vs noctx)",
                "model": cfg.scorer_model,
                "fidelity_level": 3,
                "adaptation_note": "a separately scored claim is NOT an unchanged span of the "
                                   "original response; scored only under the binary "
                                   "supported/unsupported collapse, never in the paper's "
                                   "three-way column",
                "complete": telemetry_complete,
            },
        },
        "proprietary_configurations": "the paper's strongest extractor/checker pairs use GPT-4 / "
                                      "Claude 2 and are quoted as published context only "
                                      "(fidelity level 4), never reproduced here",
        "corpus_diagnostics": diag,
        "elapsed_sec": time.time() - t0,
        "job_id": os.environ.get("SLURM_JOB_ID", ""),
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path = os.path.join(out_dir, "manifest.json")
    with open(path + ".tmp", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    os.replace(path + ".tmp", path)

    if not telemetry_complete:
        print("[refchecker] INCOMPLETE — resubmit with the same --out to resume", flush=True)
        sys.exit(EXIT_INCOMPLETE)
    print(f"\nCOMPLETE -> {out_dir}", flush=True)
    free_memory()


if __name__ == "__main__":
    main()
