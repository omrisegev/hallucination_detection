#!/usr/bin/env python
"""CPU-only regression test for the Step-244 no-op-resume corruption.

THE BUG: `gate.add()` lived inside the `for ci, c in todo:` loop, where `todo` is the
list of candidates still MISSING their field. On a resume where every field already
exists, `todo` is empty for every problem, so the gate accumulated zero traces,
`gate_b_verdict` returned False for "no comparable token_entropies traces", and that
verdict was written over the real one. Observed live on job 184777 (the afterany resume
link of 184776): four completed cells ended up with n_traces=0, gate_b_pass=false,
n_tokens=0, while their sidecars were complete and correct.

This test drives the real driver twice against a fixture — a first pass that extracts,
then a second pass with nothing to do — and asserts the second pass does not destroy the
first pass's verdict. It uses a tiny random Llama, so it needs no network and no GPU.

Usage:  python scripts/smoke_layer_views_resume.py
"""
import json
import os
import shutil
import sys
import tempfile
import types

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "cluster"))

import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers

import run_layer_views as RLV
from spectral_utils import save_cache_atomic, load_cache

VOCAB = 256
N_PROBLEMS = 6


def tiny_model():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=VOCAB, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=3, num_attention_heads=4,
                      num_key_value_heads=4, max_position_embeddings=256)
    return LlamaForCausalLM(cfg).eval().float()


def tiny_tokenizer():
    vocab = {str(i): i for i in range(VOCAB)}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="0"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="0", pad_token="0")


def build_fixture(root, mdl, tok):
    """A repgrid-schema cache whose token_entropies are the model's OWN top-15
    entropies, so a correctly reconstructed prompt passes Gate B."""
    data_dir = os.path.join(root, "results", "repgrid", "fixture_cell")
    os.makedirs(data_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    cache = {}
    for i in range(N_PROBLEMS):
        prompt_ids = [int(x) for x in rng.integers(1, VOCAB, 5)]
        gen_ids = [int(x) for x in rng.integers(1, VOCAB, 7)]
        seq = torch.tensor([prompt_ids + gen_ids])
        with torch.no_grad():
            logits = mdl(input_ids=seq, use_cache=False).logits
        raw = logits[0, len(prompt_ids) - 1: len(prompt_ids) - 1 + len(gen_ids)]
        lp = raw.float().log_softmax(-1)
        top = lp.topk(15, dim=-1).values
        p = top.exp()
        p = p / p.sum(-1, keepdim=True)
        ents = (-(p * torch.log(p + 1e-12)).sum(-1)).tolist()
        cache[i] = {"question": " ".join(str(x) for x in prompt_ids),
                    "gold_row": {}, "candidates": [{
                        "gen_token_ids": gen_ids, "token_entropies": ents,
                        "label": int(i % 2), "full_text": ""}]}
    save_cache_atomic(cache, os.path.join(data_dir, "raw_fixture_T1.0.pkl"))
    return data_dir


def make_spec(data_dir):
    return types.SimpleNamespace(
        cell_id="fixture_cell", model="fixture", dtype="float32",
        data_dir=data_dir, schema="repgrid",
        pkls=[(1.0, os.path.join(data_dir, "raw_fixture_T1.0.pkl"))],
        prompt_recipe={"kind": "template", "template": "{question}",
                       "raw_prompt": True},
        warp_base={"top_k": None, "top_p": None, "rep_penalty": None},
        raw_top_k=1, logprob_top_k=1, allow_roundtrip=False,
        repetition_penalty=None, no_repeat_ngram_size=None)


def args(**over):
    base = dict(validate_only=False, limit=None, max_gen_tokens=0,
                checkpoint_every=2, gate_n=3, tol_median=2e-2, tol_first=5e-2,
                min_frac_close=0.90, proj_dim=16, cov_eigs_r=4, arch_tol=5e-2,
                report_name=None)
    base.update(over)
    return types.SimpleNamespace(**base)


def main():
    fails = []

    def check(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            fails.append(name)

    root = tempfile.mkdtemp(prefix="layerviews_resume_")
    try:
        mdl, tok = tiny_model(), tiny_tokenizer()
        data_dir = build_fixture(root, mdl, tok)
        spec = make_spec(data_dir)
        pkl = spec.pkls[0][1]

        print("pass 1 — fresh extraction")
        done, r1 = RLV.process_pkl(mdl, tok, spec, 1.0, pkl, args())
        check("pass 1 completed", done)
        check("pass 1 gated real traces", r1["gate"].get("n_traces", 0) > 0,
              f"n_traces={r1['gate'].get('n_traces')}")
        check("pass 1 GATE-B passed", r1["gate_b_pass"] is True)
        check("pass 1 wrote fields", r1["n_candidates"] == N_PROBLEMS,
              f"n_candidates={r1['n_candidates']}")
        check("pass 1 arch guard ran", r1["arch_check"].get("done") is True)
        side1 = load_cache(os.path.join(data_dir, "layer_views_T1.0.pkl"))
        check("pass 1 stored validation inside the sidecar",
              bool(side1.get("_meta", {}).get("validation", {}).get("gate")))

        print("pass 2 — the no-op resume that caused Step 244")
        done, r2 = RLV.process_pkl(mdl, tok, spec, 1.0, pkl, args())
        check("pass 2 completed", done)
        # The PRIMARY fix is that the gate no longer keys off the has-work list, so a
        # resume genuinely RE-VALIDATES instead of measuring nothing. The
        # carry-forward path (no_op_resume) is the fallback for when not a single
        # candidate can be gated. Either is acceptable; reporting False is not.
        check("pass 2 re-gated instead of measuring nothing (primary fix)",
              r2["gate"].get("n_traces", 0) > 0,
              f"n_traces={r2['gate'].get('n_traces')}, "
              f"no_op_resume={r2.get('no_op_resume')}")
        check("pass 2 did NOT report gate_b_pass=false",
              r2["gate_b_pass"] is not False,
              f"gate_b_pass={r2['gate_b_pass']!r}")
        check("pass 2 verdict is a real PASS", r2["gate_b_pass"] is True)
        check("pass 2 preserved the real gate metrics",
              r2["gate"].get("n_traces", 0) > 0,
              f"n_traces={r2['gate'].get('n_traces')}")
        check("pass 2 preserved arch-check metrics",
              r2["arch_check"].get("residual_identity_max_abs") is not None)
        check("pass 2 rewrote no fields", r2["n_tokens"] == 0)

        side2 = load_cache(os.path.join(data_dir, "layer_views_T1.0.pkl"))
        v1 = side1["_meta"]["validation"]["gate"]
        v2 = side2["_meta"]["validation"]["gate"]
        check("sidecar validation survived the resume unchanged",
              v1.get("n_traces") == v2.get("n_traces")
              and v1.get("median_abs") == v2.get("median_abs"),
              f"{v1.get('n_traces')} -> {v2.get('n_traces')}")
        check("sidecar fields intact after resume",
              len([k for k in side2 if k != "_meta"]) == N_PROBLEMS)

        print("pass 3 — validate-only replay must not touch the sidecar")
        before = os.path.getmtime(os.path.join(data_dir, "layer_views_T1.0.pkl"))
        done, r3 = RLV.process_pkl(mdl, tok, spec, 1.0, pkl,
                                   args(validate_only=True))
        after = os.path.getmtime(os.path.join(data_dir, "layer_views_T1.0.pkl"))
        check("validate-only re-gated real traces",
              r3["gate"].get("n_traces", 0) > 0,
              f"n_traces={r3['gate'].get('n_traces')}")
        check("validate-only reproduced the PASS", r3["gate_b_pass"] is True)
        check("validate-only left the sidecar untouched", before == after)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print()
    if fails:
        print(f"RESUME SMOKE FAILED: {len(fails)} check(s) — {fails}")
        return 1
    print("RESUME SMOKE PASSED — no-op resume no longer destroys validation evidence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
