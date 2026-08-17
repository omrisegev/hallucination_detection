#!/usr/bin/env python
"""
P1 — CPU-only regression and causal-integrity suite for `paper_exact_acquisition_v1`.

Handoff §P1 and §6. This runs before any GPU job and is the promotion gate for
smoke -> pilot -> full. It tests the properties that make a number *valid*, never
whether a number is *good*:

  - manifest schema, immutability and pinned-field drift detection
  - shard atomicity, resume-by-key, duplicate rejection, orphan quarantine, hash verify
  - suffix invariance: identical prefixes with arbitrary different suffixes must produce
    identical prefix scores
  - tokenwise and chunked causal replay agree
  - the primary carries no final response length
  - alarm thresholds are calibrated on max_t score(t) over the whole horizon
  - AUROC95 = 0.5 + 0.95*(AUROC_full - 0.5), not 0.95*AUROC_full
  - ProcessBench F1 is the harmonic mean of the two accuracies, and SLA is on the
    erroneous subset only
  - DeepConf percentile direction, group-window semantics, and refusal to call
    post-warper telemetry exact
  - REFRAIN: recovered base vocabulary, cross-question bandit state, cold-start order,
    reward cold-start, resume fidelity
  - LEASH: the three stopping conditions, and that unpinned constants stay declared

Usage:
    python scripts/test_paper_exact.py            # all
    python scripts/test_paper_exact.py --only refrain
"""
import argparse
import json
import os
import shutil
import sys
import tempfile

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from spectral_utils.paper_exact import evaluator as E          # noqa: E402
from spectral_utils.paper_exact import deepconf as DC          # noqa: E402
from spectral_utils.paper_exact import refrain as RF           # noqa: E402
from spectral_utils.paper_exact import leash as LS             # noqa: E402
from spectral_utils.paper_exact import manifest as MF          # noqa: E402
from spectral_utils.paper_exact.shards import ShardWriter, verify_shards, read_shards  # noqa: E402

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))
    return bool(cond)


# ── manifest ────────────────────────────────────────────────────────────────────

def _demo_manifest(run_id="t1", model="Qwen/Qwen3-8B"):
    return MF.build_manifest(
        run_id=run_id, paper_title="T", paper_pdf_path="/nonexistent.pdf",
        fidelity="paper-specified", dataset_source="ds", dataset_revision="r1",
        dataset_example_ids=["a", "b", "c"], model_id=model, model_revision="main",
        prompt_text="P {question}", chat_template="TPL",
        decoding={"temperature": 0.6}, seed_policy={"seed": 42}, max_new_tokens=128,
        stop_behavior={"eos": True}, signal_definitions={"H": "full-vocab"},
        logits_stage="raw", hidden_state_layers=[], official_code_url="u",
        official_code_commit="c", container_image="img", evaluator_revision=E.EVALUATOR_REVISION,
        declared_deviations=[], repo_root=REPO_ROOT)


def test_manifest():
    print("\n[manifest]")
    m = _demo_manifest()
    check("order hash matches ids", m["dataset_order_sha256"] == MF.sha256_order(["a", "b", "c"]))
    check("order hash is order-sensitive",
          MF.sha256_order(["a", "b", "c"]) != MF.sha256_order(["b", "a", "c"]))
    check("prompt hash matches", m["prompt_sha256"] == MF.sha256_text("P {question}"))
    bad = dict(m); bad.pop("model_id")
    check("verify catches missing field", any("model_id" in p for p in MF.verify_manifest(bad)))
    bad2 = dict(m); bad2["dataset_example_ids"] = ["a", "b"]
    check("verify catches order-hash drift",
          any("dataset_order_sha256" in p for p in MF.verify_manifest(bad2)))
    try:
        MF.build_manifest(**{**{k: v for k, v in []}, }) if False else None
        MF.build_manifest(
            run_id="x", paper_title="T", paper_pdf_path="/n.pdf", fidelity="paper-specified",
            dataset_source="d", dataset_revision="r", dataset_example_ids=["a"],
            model_id="m", model_revision="v", prompt_text="p", chat_template="t",
            decoding={}, seed_policy={}, max_new_tokens=1, stop_behavior={},
            signal_definitions={}, logits_stage="raw",
            declared_deviations=[{"field": "cue", "paper_says": "e.g.", "we_do": "x", "why": "y"}],
            repo_root=REPO_ROOT)
        ok = False
    except ValueError:
        ok = True
    check("paper-specified + deviations is rejected", ok,
          "a run that fills in omitted constants is paper-specified-partial")

    tmp = tempfile.mkdtemp()
    try:
        MF.write_manifest(m, tmp)
        MF.write_manifest(_demo_manifest(), tmp)          # same pins -> resume
        again = MF.load_manifest(tmp)
        check("resume appends rather than overwrites", len(again.get("resumes", [])) == 1)
        try:
            MF.write_manifest(_demo_manifest(model="other/model"), tmp)
            drift_ok = False
        except ValueError:
            drift_ok = True
        check("pinned-field drift is refused", drift_ok)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── shards ──────────────────────────────────────────────────────────────────────

def _rec(i):
    return {"trace_key": f"k{i}", "question_id": f"q{i // 2}", "prompt_text": "p",
            "prompt_token_ids": [1, 2], "gen_token_ids": [3, 4], "full_text": "t"}


def test_shards():
    print("\n[shards]")
    tmp = tempfile.mkdtemp()
    try:
        keys = [f"k{i}" for i in range(10)]
        w = ShardWriter(tmp, expected_keys=keys, max_traces=4)
        for i in range(6):
            w.add(_rec(i))
        w.close()
        check("committed all buffered traces", len(w.done_keys()) == 6, f"{len(w.done_keys())}")

        w2 = ShardWriter(tmp, expected_keys=keys, max_traces=4)
        check("resume skips finished keys", w2.pending() == [f"k{i}" for i in range(6, 10)])
        try:
            w2.add(_rec(0)); dup_ok = False
        except KeyError:
            dup_ok = True
        check("duplicate trace_key rejected", dup_ok)
        for i in range(6, 10):
            w2.add(_rec(i))
        w2.close()

        rep = verify_shards(tmp)
        check("integrity clean", rep["ok"] and rep["n_traces"] == 10, json.dumps(rep["problems"]))
        check("unique keys == traces", rep["n_unique_keys"] == 10)
        check("read_shards round-trips", len(list(read_shards(tmp))) == 10)

        st = json.load(open(os.path.join(tmp, "STATUS.json")))
        check("STATUS reports complete", st["complete"] and st["n_finished"] == 10)

        # orphan: a shard file that never made it into INDEX.jsonl
        orphan = os.path.join(tmp, "shards", "shard_09999.pkl")
        with open(orphan, "wb") as f:
            f.write(b"truncated-garbage")
        ShardWriter(tmp, expected_keys=keys)
        check("unindexed shard quarantined, not read",
              not os.path.exists(orphan)
              and os.path.exists(os.path.join(tmp, "quarantine", "shard_09999.pkl")))

        # corruption must be caught, not silently consumed
        idx = [json.loads(l) for l in open(os.path.join(tmp, "INDEX.jsonl"))]
        victim = os.path.join(tmp, idx[0]["path"])
        with open(victim, "ab") as f:
            f.write(b"x")
        try:
            list(read_shards(tmp, verify=True)); caught = False
        except ValueError:
            caught = True
        check("corrupted shard raises on read", caught)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── causality ───────────────────────────────────────────────────────────────────

def _prefix_score(channels, t):
    """A stand-in causal score: it may only see row[:t]. Deliberately uses a statistic
    (mean of a prefix-centred cumulative sum) that WOULD leak if built on the full trace."""
    from spectral_utils.paper_exact.telemetry import causal_prefix_channels
    ch = causal_prefix_channels(channels, t)
    h = np.asarray(ch["raw_entropy"], dtype=float)
    if h.size == 0:
        return 0.0
    cs = np.cumsum(h - h.mean())
    return float(np.max(np.abs(cs)) / h.size)


def test_parallel_parts():
    """A sharded acquisition must reassemble from per-worker directories.

    This exists because the M2 submission was one command away from running 32 workers into
    one output directory. That would have collided three ways — duplicate shard numbers
    overwriting each other's files, a STATUS.json describing whichever worker wrote last, and
    orphan-quarantine moving a shard another worker was still writing — and none of it would
    have raised. The structural answer is one directory per worker; these checks pin it.
    """
    print("\n[parallel part directories]")
    from spectral_utils.paper_exact.shards import iter_run_dirs
    tmp = tempfile.mkdtemp()
    try:
        n_workers, per = 4, 5
        for w in range(n_workers):
            d = os.path.join(tmp, f"part_{w:02d}")
            keys = [f"k{w}_{i}" for i in range(per)]
            wr = ShardWriter(d, expected_keys=keys, max_traces=2)
            for i in range(per):
                wr.add({"trace_key": f"k{w}_{i}", "question_id": f"q{w}",
                        "prompt_text": "p", "prompt_token_ids": [1],
                        "gen_token_ids": [2], "full_text": "t"})
            wr.close()

        check("iter_run_dirs finds every worker",
              len(iter_run_dirs(tmp)) == n_workers, f"{len(iter_run_dirs(tmp))}")
        check("iter_run_dirs is a no-op on a single-worker dir",
              iter_run_dirs(os.path.join(tmp, "part_00")) == [os.path.join(tmp, "part_00")])
        recs = list(read_shards(tmp))
        check("read_shards reassembles all workers",
              len(recs) == n_workers * per, f"{len(recs)}")
        rep = verify_shards(tmp)
        check("verify_shards spans the workers",
              rep["ok"] and rep["n_workers"] == n_workers
              and rep["n_traces"] == n_workers * per, json.dumps(rep["problems"]))

        # A sharding bug that handed the same unit to two workers must surface, not silently
        # double-weight that trace in the pool.
        d = os.path.join(tmp, "part_99")
        wr = ShardWriter(d, expected_keys=["k0_0"], max_traces=1)
        wr.add({"trace_key": "k0_0", "question_id": "q0", "prompt_text": "p",
                "prompt_token_ids": [1], "gen_token_ids": [2], "full_text": "t"})
        wr.close()
        rep2 = verify_shards(tmp)
        check("cross-worker duplicate keys are detected",
              not rep2["ok"] and any("duplicate" in p for p in rep2["problems"]),
              str(rep2["problems"][:1]))

        # Separate directories means separate counters — no clobbering.
        st0 = json.load(open(os.path.join(tmp, "part_00", "STATUS.json")))
        st1 = json.load(open(os.path.join(tmp, "part_01", "STATUS.json")))
        check("each worker keeps its own STATUS.json",
              st0["n_finished"] == per and st1["n_finished"] == per)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_causality():
    print("\n[causality]")
    rng = np.random.default_rng(0)
    base = list(rng.normal(2.0, 0.5, 200))
    a = {"raw_entropy": base + list(rng.normal(5.0, 0.5, 100))}
    b = {"raw_entropy": base + list(rng.normal(-3.0, 2.0, 300))}
    same = all(abs(_prefix_score(a, t) - _prefix_score(b, t)) < 1e-12
               for t in (16, 32, 64, 128, 200))
    check("suffix invariance: identical prefixes -> identical scores", same,
          "arbitrary different suffixes appended after t=200")

    # tokenwise vs chunked replay
    tokenwise = [_prefix_score(a, t) for t in range(1, 201)]
    chunked = []
    for start in range(0, 200, 32):
        for t in range(start + 1, min(start + 33, 201)):
            chunked.append(_prefix_score(a, t))
    check("tokenwise == chunked replay",
          np.allclose(tokenwise, chunked, atol=0, rtol=0))

    # no future length
    short = {"raw_entropy": base[:100]}
    long = {"raw_entropy": base[:100] + list(rng.normal(0, 1, 500))}
    check("score at t is independent of eventual trace length",
          abs(_prefix_score(short, 64) - _prefix_score(long, 64)) < 1e-12)


def test_alarm_calibration():
    print("\n[alarm calibration]")
    rng = np.random.default_rng(1)
    # 200 correct traces, each monitored at 40 budgets
    paths = [rng.normal(0, 1, 40) for _ in range(200)]
    cal = E.calibrate_ever_alarm_threshold(paths, target_fpr=0.05)
    ever = np.mean([np.max(p) >= cal["threshold"] for p in paths])
    check("ever-alarm FPR hits the target", abs(ever - 0.05) <= 0.02, f"realized={ever:.3f}")
    # the naive fixed-time threshold is what this replaces
    naive = np.quantile(np.concatenate(paths), 0.95)
    ever_naive = np.mean([np.max(p) >= naive for p in paths])
    check("fixed-time 5% threshold really does over-alarm", ever_naive > 0.25,
          f"repeated-monitor FPR would be {ever_naive:.2f}, not 0.05")


def test_metrics():
    print("\n[metrics]")
    check("AUROC95 is 95% of above-chance signal",
          abs(E.auroc95_target(0.61) - 0.6045) < 1e-9,
          f"got {E.auroc95_target(0.61):.4f}; 0.95*0.61={0.95 * 0.61:.4f} would be wrong")
    y = [0, 0, 1, 1]
    check("AUROC perfect separation", abs(E.auroc(y, [0.1, 0.2, 0.8, 0.9]) - 1.0) < 1e-12)
    check("AUROC ties -> 0.5", abs(E.auroc(y, [1, 1, 1, 1]) - 0.5) < 1e-12 or
          np.isnan(E.auroc(y, [1, 1, 1, 1])))
    check("AUROC inverted -> 0", abs(E.auroc(y, [0.9, 0.8, 0.2, 0.1])) < 1e-12)
    ap = E.auprc([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    check("AUPRC perfect ranking -> 1", abs(ap - 1.0) < 1e-12, f"{ap:.4f}")

    # ProcessBench: harmonic mean of the two accuracies, NOT precision/recall F1
    preds = [1, 2, -1, -1, 3, -1]
    labs = [1, 5, -1, -1, 3, 2]
    st = E.processbench_f1(preds, labs)
    # erroneous rows: idx 0,1,4,5 -> hits at 0 and 4 -> 0.5 ; clean rows: idx 2,3 -> 1.0
    check("PB error accuracy", abs(st["error_acc"] - 0.5) < 1e-12, f"{st['error_acc']}")
    check("PB correct accuracy", abs(st["correct_acc"] - 1.0) < 1e-12)
    check("PB F1 is harmonic mean", abs(st["f1"] - (2 * .5 * 1.0 / 1.5)) < 1e-12, f"{st['f1']:.4f}")
    st2 = E.processbench_f1([None, 2, -1, -1, 3, -1], labs)
    check("unparsed prediction counts as wrong, not abstention",
          st2["error_acc"] < st["error_acc"] and st2["n_unparsed"] == 1)

    sla = E.mind_the_gap_sla(preds, labs)
    check("SLA is computed on erroneous traces only", sla["n"] == 4 and abs(sla["sla"] - 0.5) < 1e-12)

    p = E.parse_math_answer("so the result is \\boxed{\\frac{1}{2}}")
    check("boxed parse is balanced-brace", p["status"] == "boxed", f"{p}")
    p2 = E.parse_math_answer("the answer is 42")
    check("unboxed parse is flagged as fallback", p2["status"] == "fallback_number")
    check("parser coverage excludes fallbacks",
          abs(E.parser_coverage(["boxed", "boxed", "fallback_number", "none"]) - 0.5) < 1e-12)

    # Multiple-choice grading. The regression these lock down: AQuA-RAT's gold answer is an
    # option letter, and grading it with the math parser compares "A" to a parsed number, so
    # every answer scores wrong. It is silent and total — three AQuA cells came back 0.0%,
    # 0.0% and 0.4% where guessing among five options pays 20%.
    check("math grader cannot score a letter answer",
          not E.grade_math("the answer is (C)", "C")["correct"])
    check("choice grader scores the same answer correctly",
          E.grade_choice("the answer is (C)", "C")["correct"])
    for text, gold, want, why in [
        ("Therefore, the final answer is (C)", "C", True, "parenthesised option"),
        ("Answer: B", "B", True, "colon form"),
        ("So the correct option is E", "E", True, "prose form"),
        ("\\boxed{A}", "A", True, "boxed letter"),
        ("Therefore, the final answer is (C)", "D", False, "wrong option is wrong"),
        ("42", "C", False, "a number is not an option"),
        ("", "A", False, "empty generation"),
    ]:
        g = E.grade_choice(text, gold)
        check(f"choice grader: {why}", g["correct"] == want, f"{text!r} gold={gold} -> {g}")
    # The option letters also appear where the question lists its choices, so a parse that
    # took the FIRST match would read the menu instead of the answer.
    g = E.grade_choice("Options: (A) 5 (B) 7 (C) 9. The answer is (C)", "C")
    check("choice grader takes the last match, not the option list", g["correct"], f"{g}")

    ta = E.token_accounting([{"n_reasoning_tokens": 100, "n_closure_tokens": 10,
                              "stopped_early": True, "closure_generated": True}])
    check("token accounting sums reasoning + closure", ta["total_tokens"] == 110)
    tb = E.token_accounting([{"n_reasoning_tokens": 100, "n_closure_tokens": 0,
                              "stopped_early": True, "closure_generated": False}])
    check("truncation without closure is not realized savings", not tb["realized_savings_valid"])


def test_bootstrap():
    print("\n[grouped bootstrap]")
    rng = np.random.default_rng(2)
    # 50 questions x 10 correlated traces: trace-level CIs would be ~sqrt(10) too narrow
    groups = {}
    for q in range(50):
        mu = rng.normal(0, 1)
        groups[q] = list(rng.normal(mu, 0.05, 10))
    g = E.grouped_bootstrap(groups, lambda vals: float(np.mean(np.concatenate([np.array(v) for v in vals]))))
    flat = np.concatenate([np.array(v) for v in groups.values()])
    naive_se = flat.std(ddof=1) / np.sqrt(len(flat))
    grouped_se = (g["hi"] - g["lo"]) / (2 * 1.96)
    check("grouped CI is wider than the trace-level illusion", grouped_se > 1.5 * naive_se,
          f"grouped_se={grouped_se:.4f} vs trace-level {naive_se:.4f}")
    d = E.paired_grouped_bootstrap(groups, {k: [x + 0.5 for x in v] for k, v in groups.items()},
                                   lambda vals: float(np.mean(np.concatenate(
                                       [np.array(v) for v in vals]))))
    check("paired delta recovers a constant shift",
          abs(d["delta"] + 0.5) < 1e-9 and d["excludes_zero"], f"{d['delta']:.4f}")


def test_deepconf():
    print("\n[deepconf]")
    lp = np.array([-0.1, -1.0, -2.0, -3.0])
    check("Eq.2 averages -log p over top-k",
          abs(DC.conf_paper_eq2(lp, 4) - np.mean([0.1, 1.0, 2.0, 3.0])) < 1e-12)
    check("Appendix G.4 drops the sampled token at index 0",
          abs(DC.conf_appendix_g4(lp, 3) - np.mean([1.0, 2.0, 3.0])) < 1e-12)
    check("the two variants genuinely differ",
          abs(DC.conf_paper_eq2(lp, 4) - DC.conf_appendix_g4(lp, 3)) > 0.4,
          "so the pinned variant must be recorded before generation")

    conf = np.concatenate([np.full(100, 5.0), np.full(100, 1.0), np.full(100, 5.0)])
    g = DC.group_confidences(conf, window=100)
    check("group windows overlap with stride 1", len(g) == len(conf) - 100 + 1)
    check("lowest group finds the dip", abs(DC.lowest_group_conf(conf, 100) - 1.0) < 1e-9)
    check("short trace yields one whole-trace group",
          len(DC.group_confidences(np.ones(10), window=2048)) == 1)
    check("tail uses the final tokens", abs(DC.tail_conf(conf, 50) - 5.0) < 1e-9)

    warm = np.arange(16, dtype=float)  # 0..15
    thr_low = DC.online_threshold(warm, eta=10)
    thr_high = DC.online_threshold(warm, eta=90)
    check("eta=10 keeps only the most confident tenth", thr_low > thr_high,
          f"low={thr_low:.2f} high={thr_high:.2f}")
    check("threshold is the (100-eta)th percentile",
          abs(thr_low - np.percentile(warm, 90)) < 1e-12)
    check("no termination before a full window",
          not DC.online_should_terminate(np.zeros(100), threshold=1e9, window=2048))
    check("termination when the current window drops below s",
          DC.online_should_terminate(np.zeros(2048), threshold=1.0, window=2048))

    res = DC.filter_and_vote(["a", "a", "b"], [1.0, 1.0, 10.0], eta=None, weighted=True)
    check("confidence weighting can outvote a raw majority", res["answer"] == "b",
          f"votes={res['votes']}")
    res2 = DC.filter_and_vote(["a", "a", "b"], [1.0, 1.0, 10.0], eta=None, weighted=False)
    check("unweighted majority is the plain vote", res2["answer"] == "a")
    check("consensus beta gate", DC.consensus_reached({"a": 96.0, "b": 4.0}, 0.95)
          and not DC.consensus_reached({"a": 90.0, "b": 10.0}, 0.95))

    aud = DC.equality_audit([1.0, 2.0], [1.0, 2.0], logits_stage="post-warper")
    check("post-warper telemetry can never pass the equality audit", not aud["passed"],
          aud["reason"][:60])
    aud2 = DC.equality_audit([1.0, 2.0], [1.0, 2.0 + 1e-9], logits_stage="raw")
    check("raw equality within tolerance passes", aud2["passed"])
    aud3 = DC.equality_audit([1.0, 2.0], [1.0, 2.1], logits_stage="raw")
    check("raw inequality fails", not aud3["passed"], f"max_diff={aud3['max_abs_diff']:.3f}")

    try:
        DC.trace_token_confidence(np.zeros((3, 20)), variant="appendix_g4_exclude_sampled",
                                  sampled_first=False)
        layout_ok = False
    except ValueError:
        layout_ok = True
    check("G.4 variant refuses a descending (non-vLLM) layout", layout_ok)


def test_refrain():
    print("\n[refrain]")
    check("base vocabulary has 19 phrases", len(RF.V_BASE) == 19, f"{len(RF.V_BASE)}")
    check("base excludes the Section 5.2 expansions",
          "let me double check" not in RF.V_BASE and "wait a moment" not in RF.V_BASE)
    check("base excludes the new category",
          "the plan is to" not in RF.V_BASE and "the plan is to" in RF.V_NEW_CATEGORY)
    check("in-cat expansion is a strict superset",
          set(RF.V_BASE) < set(RF.V_INCAT_EXPANSION))

    check("segmentation splits on blank lines",
          RF.segment_steps("a\n\nb\n\n\nc") == ["a", "b", "c"])
    check("reflective detection is substring, case-insensitive",
          RF.is_reflective("Wait, that cannot be right"))
    check("non-reflective step is not flagged", not RF.is_reflective("2 + 2 = 4"))
    check("curly apostrophes fold",
          RF.is_reflective("I'm not certain here", RF.V_INCAT_EXPANSION))
    check("provisional cue must be in a STRICTLY prior step",
          RF.has_provisional_answer(["so the answer is 5"])
          and not RF.has_provisional_answer([]))

    b = RF.SWUCB(arms=(0.60, 0.65, 0.70), window=100, c=1.0)
    picks = []
    for _ in range(3):
        a, d = b.select()
        picks.append((a, d["reason"]))
        b.update(a, 0.5)
    check("cold start plays every arm once, ascending tau",
          picks == [(0.60, "cold_start"), (0.65, "cold_start"), (0.70, "cold_start")], f"{picks}")
    b.update(0.70, 10.0)
    a, d = b.select()
    check("after cold start UCB takes over", d["reason"] == "ucb")

    b2 = RF.SWUCB(arms=(0.60, 0.65, 0.70))
    b2.load_state(b.state())
    check("bandit state survives preemption",
          b2.k == b.k and b2.state()["means"] == b.state()["means"],
          "a requeued job must not restart the bandit cold mid-dataset")

    rs = RF.RewardState()
    r1 = rs.reward(0.9, 1000)
    check("first sample uses the 1e-4*L cold start",
          r1["mode"] == "cold_start" and abs(r1["reward"] - (0.9 - 0.1)) < 1e-12, f"{r1}")
    rs.observe(1000)
    r2 = rs.reward(0.9, 2000)
    check("later samples use lambda*L/Lbar",
          r2["mode"] == "running_mean" and abs(r2["reward"] - (0.9 - 0.2 * 2.0)) < 1e-12, f"{r2}")

    class _Enc:
        """Deterministic bag-of-words stub encoder standing in for all-MiniLM-L6-v2.

        Real SBERT is not needed to test the *control flow* of Algorithm 1, only a
        similarity that is high for near-duplicate steps and low for unrelated ones.
        """
        _vocab = {}

        def encode(self, xs, normalize_embeddings=True):
            out = []
            for x in xs:
                words = [w.strip(",.;:") for w in str(x).lower().split()]
                for w in words:
                    self._vocab.setdefault(w, len(self._vocab))
                v = np.zeros(256)
                for w in words:
                    v[self._vocab[w] % 256] += 1.0
                n = np.linalg.norm(v)
                out.append(v / n if n else v)
            return np.array(out)

    cfg = RF.RefrainConfig()
    enc = _Enc()
    redundant = "so the answer is 12\n\nwait, so the answer is 12\n\n"
    phi_redundant = float(np.dot(*enc.encode(["so the answer is 12",
                                              "wait, so the answer is 12"])))
    check("stub encoder scores the near-duplicate step above tau", phi_redundant >= 0.70,
          f"phi={phi_redundant:.3f}")

    st = RF.StepStopper(enc, tau=0.70, cfg=cfg)
    check("no stop without a prior provisional answer",
          not st("step one text\n\nwait, let me check that\n\n"))
    st2 = RF.StepStopper(enc, tau=0.70, cfg=cfg)
    check("no stop without a reflection trigger",
          not st2("so the answer is 12\n\nso the answer is 12\n\n"))
    st3 = RF.StepStopper(enc, tau=0.70, cfg=cfg)
    check("stop fires on prior-answer + reflection + redundancy", st3(redundant),
          f"{st3.diagnostics()['fired']}")
    st5 = RF.StepStopper(enc, tau=0.99, cfg=cfg)
    check("a redundancy below tau does not stop", not st5(redundant),
          f"phi={phi_redundant:.3f} < tau=0.99")
    st4 = RF.StepStopper(enc, tau=0.70, cfg=cfg)
    check("an incomplete final step is not judged",
          not st4("so the answer is 12\n\nwait, so the answer is 12"),
          "the trailing step has no blank-line terminator yet")


def test_leash():
    print("\n[leash]")
    cfg = LS.LeashConfig()
    check("t_min = max(m + w, k + L)", cfg.t_min == max(64 + 16, 8 + 5), f"{cfg.t_min}")
    man = cfg.as_manifest()
    check("unpinned constants stay in their own block",
          set(man["declared_by_us"]) == {"B", "tau_p", "w", "gamma"})
    check("fidelity is paper-specified-partial", man["fidelity"] == "paper-specified-partial")
    check("sensitivity grid enumerates 81 points", len(LS.grid_points()) == 81)

    sig = LS.step_signals(np.array([10.0, 9.0, 0.0, -5.0]), B=30.0)
    check("margin is top1 - top2 logprob", abs(sig["M"] - 1.0) < 1e-9, f"{sig['M']:.4f}")
    check("clipping handles non-finite logits",
          np.isfinite(LS.step_signals(np.array([np.nan, np.inf, 1.0]), B=5.0)["H"]))

    # never fires before t_min, even with a perfect plateau
    s = LS.LeashStopper(LS.LeashConfig(m=8, w=0, k=4, L=3, gamma=0.0))
    fired_at = None
    for t in range(1, 40):
        H = 2.0 if t <= 4 else 1.0      # a drop, then a flat plateau
        if s.push(H, 5.0, 0.5):
            fired_at = t
            break
    check("fires only at or after t_min", fired_at is not None and fired_at >= 8, f"{fired_at}")

    s2 = LS.LeashStopper(LS.LeashConfig(m=8, w=0, k=4, L=3, gamma=5.0))
    fired2 = any(s2.push(2.0 if t <= 4 else 1.0, 5.0, 0.5) for t in range(1, 60))
    check("the entropy-drop gate blocks a stop when the drop is too small", not fired2)

    s3 = LS.LeashStopper(LS.LeashConfig(m=8, w=0, k=4, L=3, gamma=0.0, tau_p=0.5))
    fired3 = any(s3.push(2.0 if t <= 4 else 1.0, 5.0, 0.99) for t in range(1, 60))
    check("saturated steps are excluded from the vote and cannot trigger a stop", not fired3,
          "every step has pmax=0.99 >= tau_p")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    tests = {
        "manifest": test_manifest, "shards": test_shards,
        "parts": test_parallel_parts, "causality": test_causality,
        "alarm": test_alarm_calibration, "metrics": test_metrics, "bootstrap": test_bootstrap,
        "deepconf": test_deepconf, "refrain": test_refrain, "leash": test_leash,
    }
    print(f"P1 regression suite — evaluator {E.EVALUATOR_REVISION}")
    for name, fn in tests.items():
        if args.only and args.only != name:
            continue
        fn()

    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{'=' * 70}\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    if n_fail:
        print("FAILED:")
        for name, ok, detail in RESULTS:
            if not ok:
                print(f"  - {name}  {detail}")
        sys.exit(1)
    print("P1 GATE PASS")


if __name__ == "__main__":
    main()
