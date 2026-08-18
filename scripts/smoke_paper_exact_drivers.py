#!/usr/bin/env python
"""
CPU smoke gate for the paper-exact cluster drivers.

Runs the real decode loop, the real policies, the real manifest/shard/gate machinery and the
real evaluator against a **stub language model** — a tiny deterministic module with a
real tokenizer's interface. No GPU, no weights, seconds to run.

Its job is to catch the class of bug that otherwise only appears after a Slurm allocation
has been granted and a model has been loaded: a wrong keyword, a channel that is a list
where a float was expected, a stop hook that never fires, a closure that re-prefills the
wrong text, a shard record missing a required key. Four of the six bugs in this project's
Step-163 pilot were of exactly this kind, which is why `cluster/presets.py` already gates
new presets on an offline smoke.

Usage:
    python scripts/smoke_paper_exact_drivers.py
"""
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from spectral_utils.paper_exact import evaluator as EV        # noqa: E402
from spectral_utils.paper_exact import leash as LS            # noqa: E402
from spectral_utils.paper_exact import refrain as RF          # noqa: E402
from spectral_utils.paper_exact.shards import ShardWriter, verify_shards  # noqa: E402
from spectral_utils.paper_exact.telemetry import (            # noqa: E402
    DecodeConfig, IncrementalDetokenizer, score_continuation, stream_generate)

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))
    return bool(cond)


# ── stubs ───────────────────────────────────────────────────────────────────────

class StubTokenizer:
    """Character-level tokenizer over a small alphabet, with a chat template."""
    ALPHABET = list(" \nabcdefghijklmnopqrstuvwxyz0123456789.,:{}\\+-*/=?!'()[]")

    def __init__(self):
        self.itos = ["<pad>", "<eos>"] + self.ALPHABET
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.eos_token_id = 1
        self.chat_template = "STUB"

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        ids = [self.stoi.get(c, self.stoi[" "]) for c in text]
        return type("Enc", (), {"input_ids": ids, "offset_mapping": None})()

    def encode(self, text, add_special_tokens=False):
        return self(text).input_ids

    def decode(self, ids, skip_special_tokens=True):
        out = []
        for i in ids:
            i = int(i)
            if i < 2:
                if not skip_special_tokens and i == 1:
                    out.append("")
                continue
            out.append(self.itos[i])
        return "".join(out)

    def convert_tokens_to_ids(self, t):
        return -1

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
        return "user: " + msgs[-1]["content"] + "\nassistant: "


class StubModel(torch.nn.Module):
    """Emits a fixed script, then EOS. Logits are a one-hot-ish ramp so entropy, margin and
    pmax are all well defined and vary over the trace."""

    def __init__(self, tok, script: str):
        super().__init__()
        self.tok = tok
        self.script = [tok.stoi.get(c, tok.stoi[" "]) for c in script] + [tok.eos_token_id]
        self.V = len(tok.itos)
        self.device = torch.device("cpu")
        self.generation_config = type("GC", (), {"eos_token_id": tok.eos_token_id})()
        self._n_prompt = None

    def forward(self, input_ids=None, past_key_values=None, use_cache=True, **kw):
        if past_key_values is None:
            self._n_prompt = int(input_ids.shape[1])
            emitted = 0
        else:
            emitted = int(past_key_values)
        idx = min(emitted, len(self.script) - 1)
        target = self.script[idx]
        logits = torch.full((1, int(input_ids.shape[1]), self.V), -4.0)
        # a mild ramp so H, margin and pmax move as the trace proceeds
        logits[0, -1, :] += torch.linspace(0, 1.5, self.V) * (0.3 + 0.02 * idx)
        logits[0, -1, target] = 8.0
        return type("Out", (), {"logits": logits, "past_key_values": emitted + 1})()

    def eval(self):
        return self


def _cfg(tok, **kw):
    base = dict(temperature=0.0, top_p=1.0, top_k=0, max_new_tokens=400,
                logprob_top_k=8, conf_topk=4, eos_token_ids=(tok.eos_token_id,),
                keep_top_k_arrays=True)
    base.update(kw)
    return DecodeConfig(**base)


# ── tests ───────────────────────────────────────────────────────────────────────

def test_detokenizer():
    print("\n[incremental detokenizer]")
    tok = StubTokenizer()
    text = ("step one is fine.\n\nso the answer is 12\n\nwait, so the answer is 12\n\n" * 6)
    ids = tok.encode(text)
    d = IncrementalDetokenizer(tok, window=8)
    for i in ids:
        d.append(i)
    check("splice equals a full decode", d.text == tok.decode(ids, skip_special_tokens=False),
          f"{len(ids)} tokens, window=8")


def test_stream_and_channels():
    print("\n[stream_generate]")
    tok = StubTokenizer()
    script = "the answer is 12.\n\nso the final answer is \\boxed{12}"
    mdl = StubModel(tok, script)
    ids = torch.tensor(tok.encode("q: "))
    out = stream_generate(mdl, tok, ids, _cfg(tok))
    check("decodes the scripted text", out["full_text"] == script, out["full_text"][:40])
    check("stops on EOS", out["stop_reason"] == "eos")
    ch = out["channels"]
    n = out["n_tokens"]
    check("every channel has one entry per token",
          all(len(v) == n for v in ch.values()), f"n={n}")
    check("entropy is finite and positive", np.all(np.isfinite(ch["raw_entropy"]))
          and np.all(np.asarray(ch["raw_entropy"]) > 0))
    check("spilled energy equals -log p(sampled)",
          np.allclose(ch["spilled_energy"], -np.asarray(ch["raw_logprob_sampled"])))
    check("pmax in (0, 1]", np.all(np.asarray(ch["raw_pmax"]) > 0)
          and np.all(np.asarray(ch["raw_pmax"]) <= 1))
    check("margin is non-negative", np.all(np.asarray(ch["raw_margin"]) >= -1e-9))
    check("deepconf confidence is positive", np.all(np.asarray(ch["deepconf_conf"]) > 0))
    check("raw top-k arrays retained with the right shape",
          out["raw_top_k_logprobs"]["logprobs"].shape == (n, 8))
    check("no KV cache leaked into the record", "past_key_values" not in out)

    lean = stream_generate(mdl, tok, ids, _cfg(tok, keep_top_k_arrays=False))
    check("keep_top_k_arrays=False drops the big arrays", "raw_top_k_logprobs" not in lean)

    graded = EV.grade_math(out["full_text"], "12")
    check("stub trace grades correct via the boxed parse",
          graded["correct"] and graded["parse_status"] == "boxed", str(graded))


def test_refrain_stop_and_closure():
    print("\n[refrain policy on the stub]")
    tok = StubTokenizer()
    # a trace that proposes an answer, then reflects redundantly -> Alg. 1 must fire
    script = ("first i compute the product.\n\n"
              "so the answer is 12\n\n"
              "wait, so the answer is 12\n\n"
              "and here is a lot more reasoning that should never be generated.\n\n")
    mdl = StubModel(tok, script)

    class Enc:
        _v = {}

        def encode(self, xs, normalize_embeddings=True):
            out = []
            for x in xs:
                ws = [w.strip(",.") for w in str(x).lower().split()]
                for w in ws:
                    self._v.setdefault(w, len(self._v))
                v = np.zeros(128)
                for w in ws:
                    v[self._v[w] % 128] += 1
                nn = np.linalg.norm(v)
                out.append(v / nn if nn else v)
            return np.array(out)

    rcfg = RF.RefrainConfig()
    stopper = RF.StepStopper(Enc(), tau=0.70, cfg=rcfg)
    ids = torch.tensor(tok.encode("q: "))
    out = stream_generate(mdl, tok, ids, _cfg(tok), stop_check=stopper)
    check("REFRAIN stops before the trace finishes", out["stop_reason"] == "policy",
          f"stop_reason={out['stop_reason']}, {out['n_tokens']} tokens")
    fired = stopper.diagnostics()["fired"]
    check("it fires on the redundant reflective step",
          fired is not None and "wait" in fired["step_text"],
          fired["step_text"] if fired else "none")

    full = stream_generate(mdl, StubTokenizer(), ids, _cfg(tok))
    check("stopping actually saves tokens", out["n_tokens"] < full["n_tokens"],
          f"{out['n_tokens']} vs {full['n_tokens']}")

    # forced closure re-prefills the reasoning + the closure prefix
    from cluster.run_paper_exact_refrain import forced_closure
    clo = forced_closure(mdl, tok, "user: q\nassistant: ", out["raw_text"], _cfg(tok),
                         max_closure_tokens=32)
    check("forced closure emits an answer prefixed by the boxed opener",
          clo["answer_text"].startswith(RF.CLOSURE_PROMPT), clo["answer_text"][:30])
    check("closure token count is recorded", clo["n_tokens"] > 0)

    sc = score_continuation(mdl, tok, tok.encode("user: q\nassistant: "), tok.encode("12"))
    check("Eq. 6 score is a probability in (0, 1]",
          0.0 < sc["score"] <= 1.0 and sc["n"] == 2, str(sc))


def test_leash_on_stub():
    print("\n[leash policy on the stub]")
    tok = StubTokenizer()
    mdl = StubModel(tok, "reasoning " * 60)
    cfg = LS.LeashConfig(m=8, w=0, k=4, L=3, gamma=0.0, tau_p=0.999)
    stopper = LS.LeashStopper(cfg)
    fired = {"at": None}

    def chk(_t, ch):
        if stopper.push(ch.raw_entropy[-1], ch.raw_margin[-1], ch.raw_pmax[-1]):
            fired["at"] = len(ch.raw_entropy)
            return True
        return False

    out = stream_generate(mdl, tok, torch.tensor(tok.encode("q: ")),
                          _cfg(tok, max_new_tokens=cfg.M, eos_token_ids=()),
                          stop_check=chk)
    check("LEASH reads live channels without error", out["n_tokens"] > 0,
          f"stop_reason={out['stop_reason']}, fired_at={fired['at']}")
    check("if it fires, it is at or after t_min",
          fired["at"] is None or fired["at"] >= cfg.t_min, f"{fired['at']} vs {cfg.t_min}")
    check("EOS disabled means it never stops on EOS", out["stop_reason"] != "eos")


def test_batch_equivalence():
    """Batched decoding must reproduce single-trace decoding exactly, under greedy sampling.

    This is the load-bearing check for the whole DeepConf pool: batching is a throughput
    optimisation, and the moment it changes a channel value by more than float noise it has
    silently become a different experiment. Greedy removes sampling as a confound, so any
    disagreement is a bug in the batched path.

    Also covers left-padding (mixed prompt lengths) and finished-row compaction, which are
    the two places a batched decoder usually goes wrong.
    """
    print("\n[batch vs single equivalence]")
    from spectral_utils.paper_exact.telemetry import batch_generate
    tok = StubTokenizer()
    scripts = ["the answer is \\boxed{7}",
               "a much longer chain of reasoning here, then \\boxed{12}",
               "short \\boxed{3}"]
    prompts = ["q: ", "question number two: ", "q3: "]

    singles = []
    for sc, pr in zip(scripts, prompts):
        mdl = StubModel(tok, sc)
        singles.append(stream_generate(mdl, tok, torch.tensor(tok.encode(pr)),
                                       _cfg(tok, temperature=0.0)))

    class StubCache:
        """Minimal stand-in for a transformers Cache: tracks how many tokens have been emitted
        and which original rows the current batch holds, and exposes the same
        `batch_select_indices` method the real caches use, so `_compact_cache` exercises its
        documented path rather than a fallback."""

        def __init__(self, emitted, rows):
            self.emitted, self.rows = emitted, list(rows)

        def batch_select_indices(self, keep):
            self.rows = [self.rows[int(i)] for i in keep.tolist()]
            return self

    class MultiStub(StubModel):
        """One stub emitting a different script per batch row, so the batched path is really
        decoding three distinct sequences rather than three copies of one."""

        def __init__(self, tok, scripts):
            super().__init__(tok, scripts[0])
            self.scripts = [[tok.stoi.get(c, tok.stoi[" "]) for c in s] + [tok.eos_token_id]
                            for s in scripts]

        def forward(self, input_ids=None, attention_mask=None, past_key_values=None,
                    use_cache=True, **kw):
            if past_key_values is None:
                cache = StubCache(0, range(len(self.scripts)))
            else:
                cache = past_key_values
            B, L = input_ids.shape
            assert B == len(cache.rows), f"batch {B} vs cache rows {len(cache.rows)}"
            logits = torch.full((B, L, self.V), -4.0)
            for r, orig in enumerate(cache.rows):
                sc = self.scripts[orig]
                target = sc[min(cache.emitted, len(sc) - 1)]
                logits[r, -1, :] += torch.linspace(0, 1.5, self.V) * (0.3 + 0.02 * cache.emitted)
                logits[r, -1, target] = 8.0
            cache.emitted += 1
            return type("Out", (), {"logits": logits, "past_key_values": cache})()

    batched = batch_generate(MultiStub(tok, scripts), tok,
                             [tok.encode(p) for p in prompts],
                             _cfg(tok, temperature=0.0), pad_token_id=0,
                             compact_finished=False)

    check("batch returns one record per prompt", len(batched) == 3, f"{len(batched)}")
    for i, (s, b) in enumerate(zip(singles, batched)):
        check(f"row {i}: identical token ids", s["gen_token_ids"] == b["gen_token_ids"],
              f"{len(s['gen_token_ids'])} vs {len(b['gen_token_ids'])}")
        check(f"row {i}: identical stop reason", s["stop_reason"] == b["stop_reason"])
        for chan in ("raw_entropy", "raw_logsumexp", "raw_pmax", "raw_margin",
                     "spilled_energy", "deepconf_conf"):
            ok = np.allclose(s["channels"][chan], b["channels"][chan], atol=1e-5, rtol=1e-5)
            check(f"row {i}: {chan} matches", ok,
                  "" if ok else f"max diff "
                  f"{np.max(np.abs(np.asarray(s['channels'][chan]) - np.asarray(b['channels'][chan]))):.2e}")

    compacted = batch_generate(MultiStub(tok, scripts), tok,
                               [tok.encode(p) for p in prompts],
                               _cfg(tok, temperature=0.0), pad_token_id=0,
                               compact_finished=True)
    check("compaction does not change any trace",
          all(c["gen_token_ids"] == b["gen_token_ids"] for c, b in zip(compacted, batched))
          and all(np.allclose(c["channels"]["raw_entropy"], b["channels"]["raw_entropy"],
                              atol=1e-5) for c, b in zip(compacted, batched)),
          "rows that finish early are dropped from the batch, not from the record")


def test_shard_records():
    print("\n[shard records from a driver-shaped payload]")
    tok = StubTokenizer()
    mdl = StubModel(tok, "the answer is \\boxed{7}")
    out = stream_generate(mdl, tok, torch.tensor(tok.encode("q: ")), _cfg(tok))
    tmp = tempfile.mkdtemp()
    try:
        w = ShardWriter(tmp, expected_keys=["vanilla:q1"], max_traces=1)
        w.add({"trace_key": "vanilla:q1", "question_id": "q1", "prompt_text": "q",
               "prompt_token_ids": [1, 2], "gen_token_ids": out["gen_token_ids"],
               "full_text": out["full_text"], "channels": out["channels"],
               "raw_top_k_logprobs": out["raw_top_k_logprobs"]})
        w.close()
        rep = verify_shards(tmp)
        check("a real telemetry record shards and verifies", rep["ok"] and rep["n_traces"] == 1,
              f"{rep['bytes_total']} bytes")
        try:
            ShardWriter(tmp, expected_keys=["x"]).add({"question_id": "q"})
            missing_ok = False
        except KeyError:
            missing_ok = True
        check("a record missing required keys is refused", missing_ok)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_driver_imports():
    print("\n[driver importability]")
    import importlib
    for mod in ("cluster.run_paper_exact_refrain", "cluster.run_paper_exact_deepconf",
                "cluster.run_paper_exact_leash", "cluster.run_paper_exact_uprm_judge"):
        try:
            importlib.import_module(mod)
            check(f"import {mod.split('.')[-1]}", True)
        except Exception as e:  # noqa: BLE001
            check(f"import {mod.split('.')[-1]}", False, repr(e))


def test_driver_manifests():
    """Every driver must build and verify its own manifest, locally, before submission.

    This exists because it did not: three of the first four cluster jobs died at the manifest
    gate after a Slurm allocation had already been granted — one on a self-contradictory
    fidelity label (`paper-specified` alongside declared deviations), two on a required field
    whose empty value was legitimate. All three were seconds of CPU work to detect and cost a
    GPU round trip instead. `--dry-run` builds the real manifest from synthetic rows, with no
    dataset and no model, so that never happens again.
    """
    print("\n[driver manifest dry-runs]")
    import subprocess
    tmp = tempfile.mkdtemp()
    try:
        for driver, tag in (("run_paper_exact_refrain.py", "s1"),
                            ("run_paper_exact_deepconf.py", "m1"),
                            ("run_paper_exact_leash.py", "s2"),
                            ("run_paper_exact_uprm_judge.py", "l1")):
            p = subprocess.run(
                [sys.executable, os.path.join(REPO_ROOT, "cluster", driver),
                 "--dry-run", "--out", os.path.join(tmp, tag)],
                capture_output=True, text=True, timeout=300)
            ok = p.returncode == 0 and "DRY RUN OK" in p.stdout
            detail = ""
            if not ok:
                bad = [l for l in (p.stdout + p.stderr).splitlines()
                       if "FAIL" in l or "Error" in l or "error" in l]
                detail = (bad[-1] if bad else (p.stderr.strip().splitlines() or [""])[-1])[:140]
            check(f"manifest dry-run {tag} ({driver})", ok, detail)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    print("CPU smoke gate for the paper-exact drivers (stub model, no GPU)")
    test_driver_imports()
    test_driver_manifests()
    test_detokenizer()
    test_stream_and_channels()
    test_refrain_stop_and_closure()
    test_leash_on_stub()
    test_batch_equivalence()
    test_shard_records()

    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{'=' * 70}\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    if n_fail:
        for name, ok, detail in RESULTS:
            if not ok:
                print(f"  - {name}  {detail}")
        sys.exit(1)
    print("DRIVER SMOKE GATE PASS")


if __name__ == "__main__":
    main()
