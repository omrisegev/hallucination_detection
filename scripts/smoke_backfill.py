#!/usr/bin/env python
"""
smoke_backfill.py — CPU known-answer gate for cluster/backfill_views.py.

MUST pass before any backfill job is submitted (same rule as smoke_preset.py for
presets). No GPU, one tiny-model download (hf-internal-testing/tiny-random-
LlamaForCausalLM, ~few MB, float32 => generation-time and teacher-forced values
agree to ~1e-5, so the equality assertions are tight).

What it proves:
  1. round-trip  — keys captured at GENERATION time (via generate_full with
     capture_logsumexp=True) are reproduced by the teacher-forced backfill to
     atol=1e-4: token_logsumexp, top_k_logprobs_raw, token_spilled_energies,
     top_k_logprobs (incl. the temperature+top-k warp chain).
  2. gates       — Gate B (recomputed entropies vs saved) and Gate A (Z_n) pass on
     clean data; --validate-only writes nothing (byte-identical pkl).
  3. append-only — pre-existing keys byte-unchanged; append_key raises on overwrite;
     a re-run is a no-op (0 candidates written).
  4. schemas     — both the repgrid {idx:{candidates}} schema (template recipe) and
     the flat Colab schema (stored_text recipe) round-trip.
  5. resolver    — resolve_spec on two real repgrid presets (manifest + preset merge,
     raw_prompt from preset).

Usage:  python scripts/smoke_backfill.py
"""
import copy
import json
import os
import pickle
import shutil
import sys
import tempfile
from types import SimpleNamespace

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "cluster"))

import numpy as np
import torch

from spectral_utils import generate_full
from spectral_utils.model_utils import fmt_prompt
import backfill_specs
from backfill_specs import resolve_spec
from backfill_views import (APPEND_KEYS, append_key, process_pkl, _present,
                            build_warpers, forward_batch)

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"
TEMP = 0.7
TOPK_GEN = 50
LP_TOPK = 8

QUESTIONS = [
    "What is 2 + 2?",
    "Name a color of the sky.",
    "How many legs does a cat have?",
    "What comes after Tuesday?",
]

PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [ok]   {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")


def default_args(**over):
    d = dict(validate_only=False, limit=None, batch=2, checkpoint_every=2,
             gate_n=10, tol_median=2e-2, tol_first=5e-2, min_frac_close=0.90)
    d.update(over)
    return SimpleNamespace(**d)


def load_tiny():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TINY)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(TINY, dtype=torch.float32)
    mdl.eval()
    return mdl, tok


def make_candidates(mdl, tok, question, k=2):
    outs = []
    for _ in range(k):
        r = generate_full(mdl, tok, question, temperature=TEMP, max_new_tokens=24,
                          logprob_top_k=LP_TOPK, gen_top_k=TOPK_GEN,
                          capture_logsumexp=True)
        r["label"] = True
        outs.append(r)
    return outs


def strip(cand, keys):
    for k in keys:
        cand.pop(k, None)


def cand_matches(cand, ref, atol=1e-4):
    """Appended keys match the generation-time reference; untouched keys identical."""
    ok = True
    ok &= np.allclose(cand["token_logsumexp"], ref["token_logsumexp"], atol=atol)
    ok &= np.allclose(cand["token_spilled_energies"], ref["token_spilled_energies"],
                      atol=atol)
    for key in ("top_k_logprobs", "top_k_logprobs_raw"):
        a, b = cand[key], ref[key]
        ok &= bool(np.array_equal(a["ids"], b["ids"]))
        ok &= bool(np.allclose(a["logprobs"], b["logprobs"], atol=atol))
    # never-touched keys must be EXACTLY the originals
    ok &= cand["token_entropies"] == ref["token_entropies"]
    ok &= cand["full_text"] == ref["full_text"]
    ok &= cand["label"] == ref["label"]
    ok &= cand["gen_token_ids"] == ref["gen_token_ids"]
    return ok


def main():
    root = tempfile.mkdtemp(prefix="smoke_backfill_")
    print(f"[smoke] fixture root: {root}")
    mdl, tok = load_tiny()
    print(f"[smoke] tiny model loaded ({TINY}, float32 CPU)")

    # ── fixture 1: repgrid schema, template recipe ────────────────────────────
    cache = {}
    for i, q in enumerate(QUESTIONS):
        cache[i] = {"question": q, "gold_row": {"question": q},
                    "candidates": make_candidates(mdl, tok, q, k=2)}
    pristine = copy.deepcopy(cache)
    # strip target keys from half the candidates; problem 0 cand 0 stays complete
    # (exercises the skip path AND keeps a Gate-A Z_n reference in the gate set)
    strip(cache[0]["candidates"][1], APPEND_KEYS)
    strip(cache[1]["candidates"][0], ("token_logsumexp", "top_k_logprobs_raw"))
    strip(cache[1]["candidates"][1], ("token_spilled_energies", "top_k_logprobs"))
    strip(cache[2]["candidates"][0], APPEND_KEYS)
    strip(cache[2]["candidates"][1], APPEND_KEYS)
    strip(cache[3]["candidates"][0], ("token_logsumexp", "top_k_logprobs_raw"))

    cell_dir = os.path.join(root, "smoke_repgrid")
    os.makedirs(cell_dir)
    pkl_path = os.path.join(cell_dir, f"raw_smoke_T{TEMP}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(cache, f)

    backfill_specs.BACKFILL_SPECS["smoke_repgrid"] = {
        "origin": "colab", "data_dir": "smoke_repgrid", "pkl_glob": "raw_*.pkl",
        "schema": "repgrid", "model": TINY, "logprob_top_k": LP_TOPK,
        "warp": {"temperature": TEMP, "top_k": TOPK_GEN, "top_p": None},
        "prompt_recipe": {"kind": "template", "template": "{question}",
                          "raw_prompt": False},
    }
    spec = resolve_spec("smoke_repgrid", root)

    # ── Test A: validate-only writes nothing, gates pass ──────────────────────
    print("\n[smoke] Test A — validate-only")
    bytes_before = open(pkl_path, "rb").read()
    completed, rep = process_pkl(mdl, tok, spec, TEMP, pkl_path,
                                 default_args(validate_only=True))
    check("A1 validate-only completes", completed)
    check("A2 Gate B passes on clean fixture", rep and rep.get("gate_b_pass"),
          f"reasons={rep.get('gate_b_reasons') if rep else None}")
    h = rep["gate"]["H"]
    check("A3 recomputed H matches saved (median<1e-4)",
          h.get("median_abs", 1) < 1e-4, f"median={h.get('median_abs')}")
    z = rep["gate"]["Z"]
    check("A4 Gate A Z_n matches where stored (median<1e-4)",
          z.get("median_abs", 1) < 1e-4, f"median={z.get('median_abs')}")
    check("A5 validate-only wrote nothing",
          open(pkl_path, "rb").read() == bytes_before)

    # ── Test B: full run appends correct values ───────────────────────────────
    print("\n[smoke] Test B — full backfill round-trip")
    completed, rep = process_pkl(mdl, tok, spec, TEMP, pkl_path, default_args())
    check("B1 full run completes", completed)
    check("B2 wrote the 6 stripped candidates",
          rep.get("n_candidates_written") == 6,
          f"wrote={rep.get('n_candidates_written')}")
    with open(pkl_path, "rb") as f:
        after = pickle.load(f)
    all_ok = all(
        cand_matches(after[i]["candidates"][j], pristine[i]["candidates"][j])
        for i in after for j in range(2))
    check("B3 every candidate matches generation-time values (atol=1e-4)", all_ok)
    untouched = after[0]["candidates"][0]
    ref = pristine[0]["candidates"][0]
    check("B4 untouched candidate byte-identical",
          pickle.dumps(untouched) == pickle.dumps(ref))

    # ── Test C: re-run is a no-op ─────────────────────────────────────────────
    print("\n[smoke] Test C — resume semantics")
    completed, rep = process_pkl(mdl, tok, spec, TEMP, pkl_path, default_args())
    check("C1 re-run writes 0 candidates", rep.get("n_candidates_written") == 0,
          f"wrote={rep.get('n_candidates_written')}")

    # ── Test D: overwrite raises ──────────────────────────────────────────────
    print("\n[smoke] Test D — append-only invariant")
    try:
        append_key(after[0]["candidates"][0], "token_logsumexp", [1.0])
        check("D1 append_key raises on overwrite", False)
    except RuntimeError:
        check("D1 append_key raises on overwrite", True)

    # ── Test E: flat schema + stored_text recipe ──────────────────────────────
    print("\n[smoke] Test E — flat schema + stored_text recipe")
    flat = {}
    for i, q in enumerate(QUESTIONS[:2]):
        c = make_candidates(mdl, tok, q, k=1)[0]
        c["question"] = q
        c["prompt"] = fmt_prompt(tok, q)  # the exact templated prompt string
        flat[i] = c
    flat_pristine = copy.deepcopy(flat)
    strip(flat[0], APPEND_KEYS)
    strip(flat[1], ("token_logsumexp", "top_k_logprobs_raw"))
    flat_dir = os.path.join(root, "smoke_flat")
    os.makedirs(flat_dir)
    flat_pkl = os.path.join(flat_dir, "inference_cache.pkl")
    with open(flat_pkl, "wb") as f:
        pickle.dump(flat, f)
    backfill_specs.BACKFILL_SPECS["smoke_flat"] = {
        "origin": "colab", "data_dir": "smoke_flat", "pkl_glob": "*.pkl",
        "schema": "flat", "model": TINY, "logprob_top_k": LP_TOPK,
        "warp": {"temperature": TEMP, "top_k": TOPK_GEN, "top_p": None},
        "prompt_recipe": {"kind": "stored_text", "key": "prompt",
                          "add_special_tokens": True},
    }
    fspec = resolve_spec("smoke_flat", root)
    completed, rep = process_pkl(mdl, tok, fspec, TEMP, flat_pkl, default_args())
    check("E1 flat run completes + Gate B passes",
          completed and rep.get("gate_b_pass"),
          f"reasons={rep.get('gate_b_reasons') if rep else None}")
    with open(flat_pkl, "rb") as f:
        flat_after = pickle.load(f)
    check("E2 flat candidates match generation-time values",
          all(cand_matches(flat_after[i], flat_pristine[i]) for i in flat_after))

    # ── Test G: tier-2r — alias keys + full_text roundtrip, no gen ids ────────
    # Mimics an old Colab cache (text/ents/correct, nothing else). Reference
    # values are computed via the LOOP-BASED model_utils formulas on per-step
    # warped scores, so the vectorized backfill path is cross-checked against an
    # independent implementation, not against itself.
    print("\n[smoke] Test G — tier-2r roundtrip (aliases, retokenized gen ids)")
    from spectral_utils.model_utils import (fmt_prompt as _fmt,
                                            token_entropies_and_spilled,
                                            extract_top_k_logprobs,
                                            token_logsumexp_from_scores)
    warpers = build_warpers(TEMP, TOPK_GEN, None)
    rt_cache, rt_ref = {}, {}
    sentences = ["The quick brown fox jumps over the lazy dog.",
                 "Paris is the capital of France and a large city."]
    for i, (q, sent) in enumerate(zip(QUESTIONS[:2], sentences)):
        gen_ids = tok(sent, add_special_tokens=False).input_ids
        prompt_ids = tok(_fmt(tok, f"Q: {q}")).input_ids
        raw = forward_batch(mdl, [{"prompt_ids": prompt_ids, "gen_ids": gen_ids}])[0]
        # generation-time-equivalent references from per-step warped scores
        dummy = torch.zeros((1, 0), dtype=torch.long)
        scores = []
        for j in range(len(gen_ids)):
            s = raw[j].unsqueeze(0).clone()
            if warpers is not None:
                for proc in warpers:
                    s = proc(dummy, s)
            scores.append(s)
        ents, spilled = token_entropies_and_spilled(
            scores, torch.tensor(gen_ids), K=15)
        rt_ref[i] = {
            "gen_ids": gen_ids,
            "token_logsumexp": token_logsumexp_from_scores(
                [raw[j].unsqueeze(0) for j in range(len(gen_ids))]),
            "token_spilled_energies": spilled,
            "top_k_logprobs": extract_top_k_logprobs(scores, LP_TOPK),
        }
        # old-cache shape: alias keys only, trailing extra entropy = the EOS step
        rt_cache[i] = {"question": q, "text": sent, "ents": ents + [0.123],
                       "correct": bool(i % 2)}
    rt_dir = os.path.join(root, "smoke_roundtrip")
    os.makedirs(rt_dir)
    rt_pkl = os.path.join(rt_dir, "inference_cache.pkl")
    with open(rt_pkl, "wb") as f:
        pickle.dump(rt_cache, f)
    backfill_specs.BACKFILL_SPECS["smoke_roundtrip"] = {
        "origin": "colab", "data_dir": "smoke_roundtrip",
        "pkl_glob": "*.pkl", "schema": "flat", "model": TINY,
        "logprob_top_k": LP_TOPK, "allow_roundtrip": True,
        "warp": {"temperature": TEMP, "top_k": TOPK_GEN, "top_p": None},
        "prompt_recipe": {"kind": "template_question", "template": "Q: {question}"},
    }
    rspec = resolve_spec("smoke_roundtrip", root)
    completed, rep = process_pkl(mdl, tok, rspec, TEMP, rt_pkl, default_args())
    check("G1 roundtrip run completes + Gate B passes",
          completed and rep.get("gate_b_pass"),
          f"reasons={rep.get('gate_b_reasons') if rep else None}")
    with open(rt_pkl, "rb") as f:
        rt_after = pickle.load(f)
    g_ok = True
    for i, ref in rt_ref.items():
        c = rt_after[i]
        g_ok &= c.get("gen_token_ids") == ref["gen_ids"]
        g_ok &= np.allclose(c["token_logsumexp"], ref["token_logsumexp"], atol=1e-4)
        g_ok &= np.allclose(c["token_spilled_energies"],
                            ref["token_spilled_energies"], atol=1e-4)
        g_ok &= bool(np.array_equal(c["top_k_logprobs"]["ids"],
                                    ref["top_k_logprobs"]["ids"]))
        g_ok &= len(c["token_logsumexp"]) == len(ref["gen_ids"])  # trimmed vs ents+1
        g_ok &= c["ents"][-1] == 0.123  # original alias trace untouched
    check("G2 roundtrip ids + appended keys match loop-based references", g_ok)

    # ── Test F: repgrid resolver on real presets ──────────────────────────────
    print("\n[smoke] Test F — resolve_spec on real repgrid manifests")
    fake_root = os.path.join(root, "fakeshared")
    for cid, pklname in [("lapeigvals_gsm8k_llama8b", "raw_gsm8k_T1.0.pkl"),
                         ("seiclr_triviaqa_opt30b", "raw_trivia_qa_rougel_T1.0.pkl")]:
        src_man = os.path.join(REPO, "cache", "repgrid", cid, "manifest.json")
        if not os.path.exists(src_man):
            print(f"  [skip] {cid}: no local manifest")
            continue
        d = os.path.join(fake_root, "results", "repgrid", cid)
        os.makedirs(d, exist_ok=True)
        shutil.copy(src_man, os.path.join(d, "manifest.json"))
        man = json.load(open(src_man))
        # stub pkls named after the real dataset/temps so the resolver finds them
        for t in man.get("temps", [1.0]):
            with open(os.path.join(d, f"raw_{man['dataset']}_T{t}.pkl"), "wb") as f:
                pickle.dump({}, f)
        s = resolve_spec(cid, fake_root)
        check(f"F  {cid}: model resolved", s.model == man["model"],
              f"{s.model} != {man['model']}")
        if cid == "seiclr_triviaqa_opt30b":
            check("F  seiclr raw_prompt=True from preset (manifest lacks it)",
                  s.prompt_recipe["raw_prompt"] is True)
        else:
            check(f"F  {cid}: chat-template prompt",
                  s.prompt_recipe["raw_prompt"] is False)

    shutil.rmtree(root, ignore_errors=True)
    print(f"\n[smoke] {PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
