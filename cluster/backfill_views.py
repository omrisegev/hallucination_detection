#!/usr/bin/env python
"""
Teacher-forced view backfill for existing inference pkls (full-coverage plan,
HANDOFF_full_coverage.md).

For each candidate that already has `gen_token_ids` (and its labels/traces — which are
NEVER touched), one teacher-forced forward pass over prompt+generation recovers every
probability-derived quantity the original run did not capture:

  raw logits            -> token_logsumexp (Z_n), top_k_logprobs_raw        [always if missing]
  re-warped logits      -> token_spilled_energies, top_k_logprobs           [only if missing]
  (warp = the cell's temperature -> top-k -> top-p chain, exactly as HF generate()
   applied it to out.scores at generation time; at T=0 post-warp == raw)

Append-only is a hard invariant: writes go through append_key(), which raises if the
key already exists on the candidate. Labels, full_text, token_entropies, token_offsets
and every published number are untouched.

Validation gates (run before anything is written):
  Gate A  (informational) — cells that already carry token_logsumexp: recompute and
          compare. Measures the bf16 decode-vs-forward kernel noise floor.
  Gate B  (BLOCKING)      — every cell: recompute the post-warp top-15 token_entropies
          (same formula as token_entropies_and_spilled) and compare to the SAVED trace.
          A wrong prompt / chat template / warp produces O(0.1-1 nat) systematic
          divergence, cleanly separable from kernel noise. Fail => the cell is skipped,
          nothing written.

Preemption-safe exactly like run_inference.py: SIGTERM sets a flag, the loop
checkpoints via save_cache_atomic and exits EXIT_INCOMPLETE (85) so a chained
--dependency=afterany job resumes; key presence is the resume marker.

Usage:
  python cluster/backfill_views.py --list
  python cluster/backfill_views.py --cells spilled_triviaqa_llama8b --validate-only
  python cluster/backfill_views.py --cells lapeigvals_gsm8k_llama8b,sciq_llama8b
  python cluster/backfill_views.py --cells noise_gsm8k_mistral7b --dry-run
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from backfill_specs import BACKFILL_SPECS, list_backfill_cells, resolve_spec
from spectral_utils import load_model, load_cache, save_cache_atomic, free_memory
from spectral_utils.model_utils import fmt_prompt

# NGC pytorch:25.01 ships torch 2.6.0a0; transformers parses "2.6.0a0" as < 2.6 and
# blocks torch.load of .bin checkpoints (deepseek-math-7b-instruct has no safetensors
# — killed probe job 123730 at model load). Same neutralization as run_inference.py:
# we only load trusted well-known repos, and upgrading torch in the image is forbidden.
try:
    import transformers.modeling_utils as _mu
    _mu.check_torch_load_is_safe = lambda *a, **k: None
except Exception:
    pass

EXIT_INCOMPLETE = 85
STOP = {"flag": False}

# Keys this driver may append. Everything else in the candidate is read-only.
RAW_KEYS = ("token_logsumexp", "top_k_logprobs_raw")
POST_KEYS = ("token_spilled_energies", "top_k_logprobs")
APPEND_KEYS = RAW_KEYS + POST_KEYS

ENTROPY_TOPK = 15  # token_entropies_and_spilled's K — Gate B must match it exactly


def _on_sigterm(signum, frame):
    STOP["flag"] = True
    print("[backfill] SIGTERM received — will checkpoint after current batch", flush=True)


def _stub_pcre():
    """gptqmodel's logger does `import pcre`; stub it with stdlib re (the reliable
    path per CLAUDE.md — pypcre needs libpcre2-dev, `pcre` on PyPI is unrelated)."""
    import re as _re
    import types as _types
    if "pcre" in sys.modules:
        return
    _pcre = _types.ModuleType("pcre")
    for _fn in ("compile", "match", "search", "findall", "sub", "split", "fullmatch"):
        setattr(_pcre, _fn, getattr(_re, _fn))
    _pcre.error = _re.error
    _pcre.Pattern = _re.Pattern
    _pcre.Match = _re.Match
    _flags = ("IGNORECASE", "MULTILINE", "DOTALL", "VERBOSE", "UNICODE", "ASCII")
    for _flag in _flags:
        setattr(_pcre, _flag, getattr(_re, _flag))
    _pcre.Flag = _types.SimpleNamespace(**{f: getattr(_re, f) for f in _flags})
    sys.modules["pcre"] = _pcre


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""  # cluster code dir is synced without .git


def _present(v):
    if v is None:
        return False
    if isinstance(v, (list, tuple, dict, str)):
        return len(v) > 0
    return True


def append_key(cand, key, val):
    """Append-only write. Raises if the candidate already carries the key."""
    if _present(cand.get(key)):
        raise RuntimeError(f"append_key: refusing to overwrite existing key {key!r}")
    cand[key] = val


# Colab-era caches predate the standardized schema — read through aliases, always
# WRITE canonical names (schema_dump.json 2026-07-18: phase4/5/6/gpqa_72b use
# all_entropies/correct; phase9 uses text/ents/correct).
ALIASES = {
    "token_entropies":        ("token_entropies", "all_entropies", "ents"),
    "full_text":              ("full_text", "text"),
    "label":                  ("label", "correct"),
    "gen_token_ids":          ("gen_token_ids",),
    "token_spilled_energies": ("token_spilled_energies",),
    "token_logsumexp":        ("token_logsumexp",),
    "top_k_logprobs":         ("top_k_logprobs",),
}


def get_aliased(cand, key):
    for a in ALIASES.get(key, (key,)):
        v = cand.get(a)
        if _present(v):
            return v
    return None


# Roundtrip gen-ids (tier-2r cells: full_text saved, gen_token_ids not). full_text
# was decoded with skip_special_tokens + strip, so the trailing EOS entropy has no
# text token: expect len(trace) - len(retok ids) in [0..3] (EOS + strip effects);
# -1 tolerated for cap-hit traces with a stripped trailing partial token.
ROUNDTRIP_DELTA_RANGE = (-1, 3)


def candidate_gen_ids(tok, cand, allow_roundtrip):
    """Return (gen_ids, source, delta) or (None, reason, None) when unusable."""
    ids = get_aliased(cand, "gen_token_ids")
    if ids is not None:
        return list(ids), "stored", 0
    if not allow_roundtrip:
        return None, "no gen_token_ids (roundtrip disabled)", None
    text = get_aliased(cand, "full_text")
    if not text:
        return None, "no gen_token_ids and no full_text", None
    ids = tok(text, add_special_tokens=False).input_ids
    trace = get_aliased(cand, "token_entropies") or []
    delta = len(trace) - len(ids)
    lo, hi = ROUNDTRIP_DELTA_RANGE
    if not (lo <= delta <= hi):
        return None, f"roundtrip length delta {delta} outside [{lo},{hi}]", None
    return list(ids), "retokenized", delta


# ── schema iteration ──────────────────────────────────────────────────────────

def iter_problems(cache, schema):
    """Yield (idx, gold_row, question, candidates) uniformly over all schemas.
    Candidates are the ORIGINAL mutable dicts — writes propagate to the pkl."""
    if schema == "repgrid":
        for idx in sorted(cache):
            entry = cache[idx]
            yield idx, entry.get("gold_row"), entry.get("question", ""), entry["candidates"]
    elif schema == "flat":  # {idx: cand} — the entry IS the single candidate
        for idx in sorted(cache):
            entry = cache[idx]
            yield idx, entry, entry.get("question", ""), [entry]
    elif schema == "list":  # [cand, ...]; question under item.question (phase9)
        for idx, entry in enumerate(cache):
            gold_row = entry.get("item") or entry
            yield idx, gold_row, gold_row.get("question", ""), [entry]
    elif schema == "phase10":  # [{idx, row, output}] — cand is entry["output"]
        for entry in cache:
            row = entry["row"]
            yield entry["idx"], row, row.get("question", ""), [entry["output"]]
    else:
        raise ValueError(f"unknown schema {schema!r}")


# ── prompt reconstruction ─────────────────────────────────────────────────────

_DATASET_CACHE = {}


def _dataset_rows(loader):
    """Lazy per-process dataset cache for dataset_by_idx recipes."""
    if loader not in _DATASET_CACHE:
        from spectral_utils import data_loaders as dl
        if loader == "math500_300":
            _DATASET_CACHE[loader] = dl.load_math500(300)
        elif loader == "gpqa_diamond":
            _DATASET_CACHE[loader] = dl.load_gpqa()
        else:
            raise ValueError(f"unknown dataset_by_idx loader {loader!r}")
    return _DATASET_CACHE[loader]


def build_prompt_ids(tok, recipe, gold_row, question, cand, idx=None):
    """Rebuild the exact prompt token ids the original generation consumed.

    Correctness is enforced by Gate B, not assumed here — a recipe is allowed to be a
    best guess. Returns list[int] or raises ValueError for irreconstructible cells.
    """
    kind = recipe["kind"]

    if kind == "template_question":
        # exact template string from the originating notebook, on the saved question
        if not question:
            raise ValueError("template_question: no saved question")
        msg = recipe["template"].format(question=question)
        prompt = msg if recipe.get("raw_prompt") else fmt_prompt(tok, msg)
        return tok(prompt).input_ids

    if kind == "dataset_by_idx":
        # phase4/5-era caches saved no question — idx is the dataset row index
        # (verified deterministic loaders); the saved gold cross-checks alignment.
        from spectral_utils import data_loaders as dl
        row = _dataset_rows(recipe["loader"])[idx]
        if recipe["loader"] == "math500_300":
            msg = dl.math_prompt(row)
            saved_gold = cand.get("gold")
            if saved_gold and saved_gold != row.get("solution", ""):
                raise ValueError(f"gold mismatch at idx {idx} (dataset alignment)")
        else:  # gpqa_diamond — deterministic per-idx option shuffle
            msg, letter = dl.gpqa_prompt_and_answer(row, idx)
            saved = cand.get("gold") or cand.get("correct_letter")
            if saved and saved != letter:
                raise ValueError(f"correct-letter mismatch at idx {idx} "
                                 f"(saved {saved!r} vs recomputed {letter!r})")
        prompt = fmt_prompt(tok, msg)
        return tok(prompt).input_ids

    if kind == "lciteeval":
        # phase10 RAG: full normalized row (question + docs) is saved in the pkl
        from spectral_utils.data_loaders import lciteeval_prompt
        msg = lciteeval_prompt(gold_row)
        prompt = fmt_prompt(tok, msg)
        return tok(prompt).input_ids

    if kind == "package_prompt":
        # prompt builder lives in spectral_utils.data_loaders, takes the question str
        from spectral_utils import data_loaders as dl
        if not question:
            raise ValueError("package_prompt: no saved question")
        msg = getattr(dl, recipe["fn"])(question)
        prompt = fmt_prompt(tok, msg)
        return tok(prompt).input_ids
    if kind == "stored_ids":
        ids = cand.get(recipe["key"]) or (gold_row or {}).get(recipe["key"])
        if not ids:
            raise ValueError(f"stored_ids key {recipe['key']!r} missing")
        return list(ids)

    if kind == "stored_text":
        text = cand.get(recipe["key"]) or (gold_row or {}).get(recipe["key"])
        if not text:
            raise ValueError(f"stored_text key {recipe['key']!r} missing")
        return tok(text, add_special_tokens=recipe.get("add_special_tokens", True)).input_ids

    if kind == "dataset_fn":
        from run_inference import DATASETS  # lazy: pulls in data_loaders
        _, prompt_fn, _ = DATASETS[recipe["dataset"]]
        msg = prompt_fn(gold_row)
        if recipe.get("prompt_suffix"):
            msg = f"{msg}{recipe['prompt_suffix']}"
        prompt = msg if recipe.get("raw_prompt") else fmt_prompt(tok, msg)
        return tok(prompt).input_ids

    if kind == "template":
        msg = recipe["template"].format(question=question)
        prompt = msg if recipe.get("raw_prompt") else fmt_prompt(tok, msg)
        return tok(prompt).input_ids

    raise ValueError(f"cell is {kind} — tier-3, cannot backfill")


# ── warp (must mirror HF generate()'s warper chain over out.scores) ──────────

def build_warpers(temperature, top_k, top_p):
    """The exact post-processing chain generate() applied to produce out.scores.
    Returns None for greedy (T<=1e-4): no warpers ran, scores == raw logits."""
    if temperature is None or temperature <= 1e-4:
        return None
    from transformers import LogitsProcessorList
    from transformers.generation.logits_process import (
        TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper)
    lst = LogitsProcessorList()
    if temperature != 1.0:
        lst.append(TemperatureLogitsWarper(temperature))
    if top_k:  # generate() applies top_k whenever it is set and non-zero
        lst.append(TopKLogitsWarper(top_k=int(top_k)))
    if top_p is not None and top_p < 1.0:
        lst.append(TopPLogitsWarper(top_p=float(top_p)))
    return lst


# ── per-candidate quantity computation ───────────────────────────────────────

def _rep_penalize_chunk(blk, mask, gen_tgt, penalty):
    """HF RepetitionPenaltyLogitsProcessor semantics, teacher-forced: at each
    position the penalized set is every token id already in the prefix
    (prompt + generation so far). `mask` [V] bool evolves across chunks."""
    out = blk.clone()
    for j in range(out.shape[0]):
        row = out[j]
        vals = row[mask]
        row[mask] = torch.where(vals > 0, vals / penalty, vals * penalty)
        mask[gen_tgt[j]] = True
    return out


def candidate_quantities(raw_logits, gen_ids, warpers, raw_top_k, post_top_k,
                         chunk=1024, rep_penalty=None, prompt_ids=None):
    """Compute all derivable quantities from one candidate's raw logits [T, V].

    Chunked over positions so the float32 buffers stay ~chunk x V. Formulas mirror
    model_utils exactly:
      token_logsumexp        = logsumexp(raw)                       (raw, full vocab)
      top_k_logprobs_raw     = topk(log_softmax(raw))
      token_spilled_energies = -log_softmax(warped)[gen_id]
      top_k_logprobs         = topk(log_softmax(warped))
      H_recomputed (gate)    = top-15 renormalized entropy of log_softmax(warped)
    rep_penalty (a PROCESSOR — applied before the temperature/top-k/top-p warpers,
    exactly as generate() orders them) is needed for runs where the model's own
    generation_config default applied (e.g. Qwen2.5-Instruct's 1.05).
    """
    T, V = raw_logits.shape
    dev = raw_logits.device
    zs, spilled, ents = [], [], []
    ids_raw, lps_raw, ids_post, lps_post = [], [], [], []
    dummy = torch.zeros((1, 0), dtype=torch.long, device=dev)
    rep_mask = None
    if rep_penalty:
        rep_mask = torch.zeros(V, dtype=torch.bool, device=dev)
        if prompt_ids is not None:
            rep_mask[torch.as_tensor(prompt_ids, device=dev)] = True
    for s in range(0, T, chunk):
        blk = raw_logits[s:s + chunk].float()
        tgt = gen_ids[s:s + chunk]
        n = blk.shape[0]

        zs.append(torch.logsumexp(blk, dim=-1))
        lp_raw = blk.log_softmax(dim=-1)
        tk = lp_raw.topk(min(raw_top_k, V), dim=-1)
        ids_raw.append(tk.indices.to(torch.int32).cpu())
        lps_raw.append(tk.values.to(torch.float32).cpu())

        if rep_penalty:
            blk = _rep_penalize_chunk(blk, rep_mask, tgt, rep_penalty)
        if warpers is not None:
            w = blk.clone()
            for proc in warpers:
                w = proc(dummy, w)
            lp_post = w.log_softmax(dim=-1)
        elif rep_penalty:
            lp_post = blk.log_softmax(dim=-1)
        else:
            lp_post = lp_raw
        spilled.append(-lp_post[torch.arange(n, device=dev), tgt])
        tkp = lp_post.topk(min(post_top_k, V), dim=-1)
        ids_post.append(tkp.indices.to(torch.int32).cpu())
        lps_post.append(tkp.values.to(torch.float32).cpu())

        top15 = lp_post.topk(min(ENTROPY_TOPK, V), dim=-1).values
        p = top15.exp()
        p = p / (p.sum(dim=-1, keepdim=True) + 1e-12)
        ents.append(-(p * torch.log(p + 1e-12)).sum(dim=-1))

    return {
        "token_logsumexp": torch.cat(zs).cpu().tolist(),
        "top_k_logprobs_raw": {"ids": torch.cat(ids_raw).numpy(),
                               "logprobs": torch.cat(lps_raw).numpy()},
        "token_spilled_energies": torch.cat(spilled).cpu().tolist(),
        "top_k_logprobs": {"ids": torch.cat(ids_post).numpy(),
                           "logprobs": torch.cat(lps_post).numpy()},
        "token_entropies_recomputed": torch.cat(ents).cpu().tolist(),
    }


# ── teacher-forced forward over a batch of candidates ────────────────────────

def forward_batch(mdl, items):
    """items: list of dicts with 'prompt_ids' and 'gen_ids' (lists of int).
    Returns per item the raw logits [T_gen, V] slice aligned to gen tokens
    (logits at position prompt_len-1+j predict gen token j). Right padding is
    exact for causal LMs — pads sit after each sequence's real tokens."""
    dev = mdl.device
    seqs = [it["prompt_ids"] + it["gen_ids"] for it in items]
    maxlen = max(len(s) for s in seqs)
    pad_id = 0
    input_ids = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
    for i, s in enumerate(seqs):
        input_ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        attn[i, :len(s)] = 1
    with torch.no_grad():
        out = mdl(input_ids=input_ids.to(dev), attention_mask=attn.to(dev),
                  use_cache=False)
    slices = []
    for i, it in enumerate(items):
        plen, tg = len(it["prompt_ids"]), len(it["gen_ids"])
        slices.append(out.logits[i, plen - 1: plen - 1 + tg])
    return slices


# ── gate statistics ───────────────────────────────────────────────────────────

class GateStats:
    """Accumulates saved-vs-recomputed diffs for one comparison key."""

    def __init__(self, name):
        self.name = name
        self.abs_diffs = []          # all per-token |Δ|
        self.first_tok = []          # first-token |Δ|
        self.trace_r = []            # per-trace Pearson r
        self.n_traces = 0
        self.n_len_mismatch = 0

    def add(self, saved, recomputed):
        self.n_traces += 1
        if len(saved) != len(recomputed):
            self.n_len_mismatch += 1
            return
        a = np.asarray(saved, dtype=np.float64)
        b = np.asarray(recomputed, dtype=np.float64)
        d = np.abs(a - b)
        self.abs_diffs.append(d)
        if len(d):
            self.first_tok.append(d[0])
        if len(a) >= 3 and a.std() > 1e-12 and b.std() > 1e-12:
            self.trace_r.append(float(np.corrcoef(a, b)[0, 1]))

    CLOSE_AT = 5e-2  # |Δ| below this counts as "close" for frac_close

    def summary(self):
        if not self.abs_diffs:
            return {"name": self.name, "n_traces": self.n_traces,
                    "n_len_mismatch": self.n_len_mismatch, "empty": True}
        d = np.concatenate(self.abs_diffs)
        r = np.asarray(self.trace_r) if self.trace_r else np.asarray([np.nan])
        ft = np.asarray(self.first_tok) if self.first_tok else np.asarray([np.nan])
        return {
            "name": self.name,
            "n_traces": self.n_traces,
            "n_len_mismatch": self.n_len_mismatch,
            "n_tokens": int(d.size),
            "median_abs": float(np.median(d)),
            "p99_abs": float(np.percentile(d, 99)),
            "max_abs": float(d.max()),
            "frac_close": float(np.mean(d <= self.CLOSE_AT)),
            "first_tok_median": float(np.nanmedian(ft)),
            "first_tok_p99": float(np.nanpercentile(ft, 99)),
            "median_r": float(np.nanmedian(r)),
            "frac_r_ge_0999": float(np.mean(r >= 0.999)) if self.trace_r else None,
        }


def gate_b_verdict(h_summary, tol_median, tol_first, min_frac_close):
    """Blocking prompt/warp-reconstruction gate on the recomputed entropies.

    Calibrated on the Gate-A validate-only run (job 123504, B200 bf16, 7 cells /
    5 model families): with a CORRECT prompt+warp, median|dH| lands at 1e-5..3e-3
    and ~1% of tokens still jump by ~0.1 nat — bf16 kernel noise (incremental
    decode vs full-sequence forward) flipping tokens at the top-k / top-15
    boundaries. A WRONG prompt shifts essentially every token by 0.1-1+ nat.
    The discriminating statistics are therefore the MEDIANS (whole-trace and
    first-token) plus the fraction of close tokens — never the p99 tail, and
    never per-trace Pearson r (one flipped token tanks r on a 15-token QA trace;
    both remain in the report as informational)."""
    if h_summary.get("empty"):
        return False, ["no comparable token_entropies traces"]
    reasons = []
    if h_summary["n_len_mismatch"] > 0:
        reasons.append(f"{h_summary['n_len_mismatch']}/{h_summary['n_traces']} "
                       f"trace-length mismatches (alignment error)")
    if h_summary["median_abs"] > tol_median:
        reasons.append(f"median|dH| {h_summary['median_abs']:.2e} > {tol_median:.0e}")
    if h_summary["frac_close"] < min_frac_close:
        reasons.append(f"only {h_summary['frac_close']:.0%} of tokens within "
                       f"{GateStats.CLOSE_AT:g} (need {min_frac_close:.0%})")
    ftm = h_summary.get("first_tok_median")
    if ftm is not None and not np.isnan(ftm) and ftm > tol_first:
        reasons.append(f"first-token median|dH| {ftm:.2e} > {tol_first:.0e} "
                       f"(prompt-mismatch fingerprint)")
    return not reasons, reasons


# ── per-cell processing ───────────────────────────────────────────────────────

def pending_keys(cand):
    """Which appendable keys this candidate is missing."""
    return [k for k in APPEND_KEYS if not _present(cand.get(k))]


def warp_variants(spec_warp, gen_cfg):
    """Warp hypotheses for --probe-warp: the spec's warp, plus the model's own
    generation_config defaults (top_p / top_k / repetition_penalty) that a run
    would have inherited if the generating code did not pass the kwarg (Colab
    notebooks) or if the transformers version did not treat an explicit None as
    an override. Returns [(name, {top_k, top_p, rep_penalty}), ...]."""
    base = {"top_k": spec_warp.get("top_k"), "top_p": spec_warp.get("top_p"),
            "rep_penalty": spec_warp.get("rep_penalty")}
    gp = getattr(gen_cfg, "top_p", None)
    gk = getattr(gen_cfg, "top_k", None)
    gr = getattr(gen_cfg, "repetition_penalty", None)
    gp = gp if (gp and gp < 1.0 and gp != base["top_p"]) else None
    gk = gk if (gk and gk != base["top_k"]) else None
    gr = gr if (gr and abs(gr - 1.0) > 1e-9) else None
    variants = [("spec", dict(base))]
    if gp:
        variants.append(("+cfg_top_p", {**base, "top_p": gp}))
    if gr:
        variants.append(("+cfg_rep", {**base, "rep_penalty": gr}))
    if gp and gr:
        variants.append(("+cfg_top_p+rep", {**base, "top_p": gp, "rep_penalty": gr}))
    if gk:
        variants.append(("+cfg_all", {"top_k": gk,
                                      "top_p": gp or base["top_p"],
                                      "rep_penalty": gr or base["rep_penalty"]}))
    return variants


def probe_warp_cell(mdl, tok, spec, temp, problems, args, pkl_path):
    """One forward pass per gate problem; every warp variant is evaluated on the
    same raw logits against the SAVED entropy traces. Writes nothing."""
    allow_rt = bool(getattr(spec, "allow_roundtrip", False))
    variants = warp_variants(spec.warp_base, mdl.generation_config)
    vstats = {name: GateStats(name) for name, _ in variants}
    vwarp = {name: w for name, w in variants}
    for idx, gold_row, question, cands in problems[:args.gate_n]:
        if STOP["flag"]:
            return {"pkl": os.path.basename(pkl_path), "error": "preempted mid-probe"}
        try:
            prompt_ids = build_prompt_ids(tok, spec.prompt_recipe, gold_row, question,
                                          cands[0], idx=idx)
        except ValueError as e:
            return {"pkl": os.path.basename(pkl_path),
                    "error": f"prompt reconstruction failed: {e}"}
        for c in cands:
            ids, source, delta = candidate_gen_ids(tok, c, allow_rt)
            if ids is None:
                continue
            raw = forward_batch(mdl, [{"prompt_ids": prompt_ids, "gen_ids": ids}])[0]
            gen = torch.tensor(ids, dtype=torch.long, device=raw.device)
            saved_h = (get_aliased(c, "token_entropies") or [])[:len(ids)]
            for name, w in variants:
                warpers = build_warpers(temp, w["top_k"], w["top_p"])
                q = candidate_quantities(raw, gen, warpers, 1, 1,
                                         rep_penalty=w.get("rep_penalty"),
                                         prompt_ids=prompt_ids)
                vstats[name].add(saved_h, q["token_entropies_recomputed"])
            del raw
    out = {}
    for name, s in vstats.items():
        summ = s.summary()
        ok, reasons = gate_b_verdict(summ, args.tol_median, args.tol_first,
                                     args.min_frac_close)
        out[name] = {"warp": vwarp[name], "summary": summ,
                     "pass": ok, "reasons": reasons}
        print(f"[probe] {spec.cell_id} T={temp} {name:16s} "
              f"warp={vwarp[name]} median|dH|={summ.get('median_abs', float('nan')):.2e} "
              f"frac_close={summ.get('frac_close', float('nan')):.3f} "
              f"{'PASS' if ok else 'FAIL'}", flush=True)
    return {"pkl": os.path.basename(pkl_path), "temp": temp, "probe": out,
            "validate_only": True}


def process_pkl(mdl, tok, spec, temp, pkl_path, args):
    """Gate + (unless validate-only / gate-fail) backfill one raw pkl.
    Returns (completed, report_dict). completed=False => preempted."""
    warpers = build_warpers(temp, spec.warp_base["top_k"], spec.warp_base["top_p"])
    cache = load_cache(pkl_path)
    if not cache:
        return True, {"pkl": os.path.basename(pkl_path), "error": "empty cache"}

    if spec.repetition_penalty or spec.no_repeat_ngram_size:
        # These processors depend on the generated prefix; the saved post-warp traces
        # would need them re-applied token-by-token. No analysis cell uses them.
        return True, {"pkl": os.path.basename(pkl_path),
                      "error": "repetition_penalty/no_repeat_ngram_size set — "
                               "post-warp backfill unsupported for this cell"}

    problems = list(iter_problems(cache, spec.schema))
    if args.limit:
        problems = problems[:args.limit]

    if getattr(args, "probe_warp", False):
        return True, probe_warp_cell(mdl, tok, spec, temp, problems, args, pkl_path)

    # ---- Phase 1: gate on the first gate_n problems --------------------------
    rep_pen = spec.warp_base.get("rep_penalty")
    allow_rt = bool(getattr(spec, "allow_roundtrip", False))
    stats = {"H": GateStats("token_entropies"),
             "dE": GateStats("token_spilled_energies"),
             "Z": GateStats("token_logsumexp")}
    t0 = time.time()
    gate_tokens = 0
    n_rt_skipped = 0
    gate_problems = problems[:args.gate_n]
    for idx, gold_row, question, cands in gate_problems:
        if STOP["flag"]:
            return False, None
        try:
            prompt_ids = build_prompt_ids(tok, spec.prompt_recipe, gold_row, question,
                                          cands[0], idx=idx)
        except ValueError as e:
            return True, {"pkl": os.path.basename(pkl_path),
                          "error": f"prompt reconstruction failed: {e}"}
        items = []
        for c in cands:
            ids, source, delta = candidate_gen_ids(tok, c, allow_rt)
            if ids is None:
                n_rt_skipped += 1
                continue
            items.append({"prompt_ids": prompt_ids, "gen_ids": ids, "cand": c})
        for bs in range(0, len(items), args.batch):
            batch = items[bs:bs + args.batch]
            slices = forward_batch(mdl, batch)
            for it, raw in zip(batch, slices):
                gen = torch.tensor(it["gen_ids"], dtype=torch.long, device=raw.device)
                q = candidate_quantities(raw, gen, warpers, spec.raw_top_k,
                                         spec.logprob_top_k, rep_penalty=rep_pen,
                                         prompt_ids=it["prompt_ids"])
                c = it["cand"]
                n = len(it["gen_ids"])
                # roundtrip traces run 0-3 tokens longer than the retok ids (EOS/strip)
                # — compare over the aligned prefix; misalignment still fails medians.
                saved_h = (get_aliased(c, "token_entropies") or [])[:n]
                stats["H"].add(saved_h, q["token_entropies_recomputed"])
                saved_de = get_aliased(c, "token_spilled_energies")
                if saved_de is not None:
                    stats["dE"].add(saved_de[:n], q["token_spilled_energies"])
                saved_z = get_aliased(c, "token_logsumexp")
                if saved_z is not None:
                    stats["Z"].add(saved_z[:n], q["token_logsumexp"])
                gate_tokens += n
            del slices
    gate = {k: s.summary() for k, s in stats.items()}
    if n_rt_skipped:
        gate["n_unusable_candidates"] = n_rt_skipped
    ok_b, reasons_b = gate_b_verdict(gate["H"], args.tol_median, args.tol_first,
                                     args.min_frac_close)
    print(f"[gate] {spec.cell_id} T={temp}: "
          f"H median|d|={gate['H'].get('median_abs', float('nan')):.2e} "
          f"p99={gate['H'].get('p99_abs', float('nan')):.2e} "
          f"GATE-B {'PASS' if ok_b else 'FAIL'}"
          + (f" ({'; '.join(reasons_b)})" if reasons_b else ""), flush=True)
    if gate["Z"]["n_traces"] == 0:
        gate["Z"]["note"] = "no stored token_logsumexp — Gate A not applicable"
    else:
        print(f"[gate] {spec.cell_id} T={temp}: GATE-A Z_n "
              f"median|d|={gate['Z'].get('median_abs', float('nan')):.2e} "
              f"max={gate['Z'].get('max_abs', float('nan')):.2e} "
              f"median_r={gate['Z'].get('median_r', float('nan')):.6f}", flush=True)

    report = {"pkl": os.path.basename(pkl_path), "temp": temp,
              "gate": gate, "gate_b_pass": ok_b, "gate_b_reasons": reasons_b,
              "gate_n_problems": len(gate_problems), "gate_tokens": gate_tokens,
              "validate_only": bool(args.validate_only)}

    if args.validate_only:
        report["gate_seconds"] = round(time.time() - t0, 1)
        return True, report

    if not ok_b:
        print(f"[backfill] {spec.cell_id} T={temp}: GATE-B FAIL — cell skipped, "
              f"nothing written", flush=True)
        return True, report

    # ---- Phase 2: append missing keys (resumable, append-only) ---------------
    n_written = 0
    n_skipped = 0
    write_tokens = 0
    since_ckpt = 0
    for idx, gold_row, question, cands in problems:
        todo = [c for c in cands
                if (_present(get_aliased(c, "gen_token_ids")) or allow_rt)
                and pending_keys(c)]
        if not todo:
            n_skipped += len(cands)
            continue
        if STOP["flag"]:
            save_cache_atomic(cache, pkl_path)
            print(f"[backfill] PREEMPTED — checkpoint saved at T={temp} problem={idx}",
                  flush=True)
            return False, None
        try:
            prompt_ids = build_prompt_ids(tok, spec.prompt_recipe, gold_row, question,
                                          todo[0], idx=idx)
        except ValueError as e:
            report.setdefault("prompt_errors", []).append({"idx": idx, "error": str(e)})
            continue
        items = []
        for c in todo:
            ids, source, delta = candidate_gen_ids(tok, c, allow_rt)
            if ids is None:
                report.setdefault("candidate_skips", []).append(
                    {"idx": idx, "reason": source})
                continue
            items.append({"prompt_ids": prompt_ids, "gen_ids": ids, "cand": c,
                          "ids_source": source})
        for bs in range(0, len(items), args.batch):
            batch = items[bs:bs + args.batch]
            slices = forward_batch(mdl, batch)
            for it, raw in zip(batch, slices):
                gen = torch.tensor(it["gen_ids"], dtype=torch.long, device=raw.device)
                q = candidate_quantities(raw, gen, warpers, spec.raw_top_k,
                                         spec.logprob_top_k, rep_penalty=rep_pen,
                                         prompt_ids=it["prompt_ids"])
                c = it["cand"]
                for key in pending_keys(c):
                    append_key(c, key, q[key])
                # roundtrip cells become tier-2 permanently: persist the validated ids
                if it["ids_source"] == "retokenized" and \
                        not _present(c.get("gen_token_ids")):
                    append_key(c, "gen_token_ids", it["gen_ids"])
                n_written += 1
                write_tokens += len(it["gen_ids"])
            del slices
        since_ckpt += 1
        if since_ckpt >= args.checkpoint_every:
            save_cache_atomic(cache, pkl_path)
            since_ckpt = 0
            print(f"[backfill] {spec.cell_id} T={temp}: checkpoint at problem {idx} "
                  f"({n_written} candidates backfilled)", flush=True)

    save_cache_atomic(cache, pkl_path)
    report.update({"n_candidates_written": n_written,
                   "n_candidates_already_complete": n_skipped,
                   "write_tokens": write_tokens,
                   "seconds": round(time.time() - t0, 1)})
    print(f"[backfill] {spec.cell_id} T={temp} DONE: {n_written} candidates backfilled, "
          f"{n_skipped} already complete, {write_tokens} tokens, "
          f"{time.time() - t0:.0f}s", flush=True)
    return True, report


def write_cell_report(spec, pkl_reports, args):
    # Per-cell filename: several colab cells share one data_dir (phase9 qa, rag),
    # and a shared "backfill_report.json" gets overwritten by the last cell — which
    # lost the cot cells' candidate_skips in job 123884.
    path = os.path.join(spec.data_dir, f"backfill_report_{spec.cell_id}.json")
    doc = {
        "cell_id": spec.cell_id,
        "model": spec.model,
        "keys": list(APPEND_KEYS),
        "raw_top_k": spec.raw_top_k,
        "validate_only": bool(args.validate_only),
        "tolerances": {"tol_median": args.tol_median, "tol_first": args.tol_first,
                       "min_frac_close": args.min_frac_close},
        "git_sha": _git_sha(),
        "job_id": os.environ.get("SLURM_JOB_ID", ""),
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pkls": pkl_reports,
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=2, default=str)
    os.replace(tmp, path)
    print(f"[backfill] report -> {path}", flush=True)


def update_manifest(spec, pkl_reports):
    man_path = os.path.join(spec.data_dir, "manifest.json")
    if not os.path.exists(man_path):
        return
    man = json.load(open(man_path))
    man.setdefault("backfill", []).append({
        "keys_added": list(APPEND_KEYS),
        "raw_top_k": spec.raw_top_k,
        "job_id": os.environ.get("SLURM_JOB_ID", ""),
        "git_sha": _git_sha(),
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gate_b": {r["pkl"]: r.get("gate_b_pass") for r in pkl_reports if "gate" in r},
    })
    tmp = man_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(man, f, indent=2, default=str)
    os.replace(tmp, man_path)


# ── dry run ───────────────────────────────────────────────────────────────────

def dry_run(spec, args):
    total_cand, todo_cand, todo_tokens = 0, 0, 0
    allow_rt = bool(getattr(spec, "allow_roundtrip", False))
    for temp, pkl_path in spec.pkls:
        cache = load_cache(pkl_path)
        for idx, gold_row, question, cands in iter_problems(cache, spec.schema):
            for c in cands:
                total_cand += 1
                has_ids = _present(get_aliased(c, "gen_token_ids"))
                trace = (get_aliased(c, "gen_token_ids")
                         or get_aliased(c, "token_entropies") or [])
                if (has_ids or (allow_rt and get_aliased(c, "full_text"))) \
                        and pending_keys(c):
                    todo_cand += 1
                    # +25% rough prompt overhead (prompt tokens also pass the forward)
                    todo_tokens += int(len(trace) * 1.25)
    print(f"  {spec.cell_id:34s} model={spec.model}")
    print(f"    candidates: {todo_cand}/{total_cand} need backfill, "
          f"~{todo_tokens/1e6:.2f}M forward tokens")
    return todo_tokens


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Teacher-forced view backfill",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--cells", default=None,
                    help="comma-separated cell ids (see --list)")
    ap.add_argument("--data-root",
                    default="/shared/cycle2_tau_averbuch_prj/omrisegev1",
                    help="root that spec data_dirs are relative to")
    ap.add_argument("--validate-only", action="store_true",
                    help="run the gates on --gate-n problems, write NOTHING")
    ap.add_argument("--limit", type=int, default=None, help="max problems per pkl")
    ap.add_argument("--batch", type=int, default=4, help="candidates per forward pass")
    ap.add_argument("--checkpoint-every", type=int, default=25,
                    help="save cache every N problems with new keys")
    ap.add_argument("--gate-n", type=int, default=50,
                    help="problems used for the pre-write gate")
    ap.add_argument("--tol-median", type=float, default=2e-2,
                    help="Gate B: max median |dH| (nats); correct prompts measured "
                         "1e-5..3e-3 on B200 bf16 (job 123504)")
    ap.add_argument("--tol-first", type=float, default=5e-2,
                    help="Gate B: max MEDIAN first-token |dH| across traces")
    ap.add_argument("--min-frac-close", type=float, default=0.90,
                    help="Gate B: min fraction of tokens with |dH| <= 0.05")
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"],
                    help="attention implementation for the teacher-forced forward")
    ap.add_argument("--probe-warp", action="store_true",
                    help="validate-only warp probe: evaluate the spec warp AND the "
                         "model generation_config default variants (top_p/top_k/"
                         "repetition_penalty) against the saved entropy traces")
    ap.add_argument("--dry-run", action="store_true",
                    help="no model load: count candidates/tokens needing backfill")
    ap.add_argument("--list", action="store_true", help="list known cells and exit")
    args = ap.parse_args()

    if args.list:
        for cid in list_backfill_cells():
            print(f"  {cid:36s} origin={BACKFILL_SPECS[cid]['origin']}")
        return

    if not args.cells:
        raise SystemExit("pass --cells id1,id2,... (or --list)")
    cell_ids = [c.strip() for c in args.cells.split(",") if c.strip()]
    specs = [resolve_spec(cid, args.data_root) for cid in cell_ids]

    if args.dry_run:
        total = sum(dry_run(s, args) for s in specs)
        print(f"\n  TOTAL ~{total/1e6:.2f}M forward tokens")
        return

    if any("awq" in s.model.lower() or "gptq" in s.model.lower() for s in specs):
        _stub_pcre()

    signal.signal(signal.SIGTERM, _on_sigterm)
    if torch.cuda.is_available():
        print(f"[backfill] GPU: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("[backfill] WARNING: no CUDA — running on CPU", flush=True)

    # group by (model, dtype) so each model loads once per dtype — phase4/5-era
    # cells teacher-force in float16 to match their generation dtype
    specs.sort(key=lambda s: (s.model, s.dtype))
    current_load = None
    mdl = tok = None
    any_gate_fail = False
    for spec in specs:
        if (spec.model, spec.dtype) != current_load:
            if mdl is not None:
                del mdl, tok
                free_memory()
            print(f"\n[backfill] loading model {spec.model} "
                  f"(attn={args.attn}, dtype={spec.dtype})", flush=True)
            mdl, tok = load_model(spec.model, attn_impl=args.attn, dtype=spec.dtype)
            current_load = (spec.model, spec.dtype)
        print(f"\n=== {spec.cell_id} ({len(spec.pkls)} pkl(s)) ===", flush=True)
        pkl_reports = []
        for temp, pkl_path in spec.pkls:
            completed, rep = process_pkl(mdl, tok, spec, temp, pkl_path, args)
            if not completed:
                write_cell_report(spec, pkl_reports, args)
                print("[backfill] INCOMPLETE — resubmit with the same args to resume",
                      flush=True)
                sys.exit(EXIT_INCOMPLETE)
            pkl_reports.append(rep)
            if rep and not rep.get("gate_b_pass", True) and not args.validate_only:
                any_gate_fail = True
        write_cell_report(spec, pkl_reports, args)
        if not args.validate_only and all(r.get("gate_b_pass") for r in pkl_reports
                                          if "gate" in r):
            update_manifest(spec, pkl_reports)

    print("\n[backfill] ALL CELLS PROCESSED"
          + (" (some cells FAILED Gate B — see reports)" if any_gate_fail else ""),
          flush=True)


if __name__ == "__main__":
    main()
