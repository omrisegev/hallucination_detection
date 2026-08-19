#!/usr/bin/env python
"""CPU-only known-answer tests for ``run_layer_views_reference.py``.

No Hugging Face download, GPU, raw cache, or sidecar is required.  A tiny local
causal LM with Llama-shaped module boundaries exercises the real hook collector
and full-vocabulary logit-lens implementation.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
CLUSTER = os.path.join(REPO, "cluster")
if CLUSTER not in sys.path:
    sys.path.insert(0, CLUSTER)

import run_layer_views_reference as ref


FAILURES: list[str] = []


def check(condition, message):
    print(f"  [{'ok  ' if condition else 'FAIL'}] {message}")
    if not condition:
        FAILURES.append(message)


def fixture_model():
    torch.manual_seed(20260812)
    model = ref._FixtureCausalLM(vocab=13, hidden=8, layers=3)
    model.eval()
    return model


def test_hooks_and_alignment():
    print("== hooks and causal-token alignment ==")
    model = fixture_model()
    prompt, generated = [1, 2, 3, 4], [5, 6, 7, 8]
    capture = ref.capture_forward(model, prompt, generated)
    diag = capture.hook_diagnostics
    check(diag["hook_order_pass"], "hook order is layer-major attn -> mlp -> resid")
    check(diag["hook_keys_pass"], "exactly three hooks fire for every layer")
    check(diag["hook_shapes_pass"], "all captured states are [generated_tokens, hidden]")
    check(diag["n_hook_events"] == 9, "three-layer fixture emits exactly nine hook events")
    check(capture.final_logits.shape == (4, 13), "ordinary logits align one-for-one to targets")

    try:
        ref.capture_forward(model, [], generated)
        check(False, "empty prompt is rejected")
    except ValueError:
        check(True, "empty prompt is rejected")
    try:
        ref.capture_forward(model, prompt, [])
        check(False, "empty generated trace is rejected")
    except ValueError:
        check(True, "empty generated trace is rejected")


def test_lens_and_final_identity():
    print("\n== logit lens and final-residual identity ==")
    model = fixture_model()
    prompt, generated = [1, 2, 3, 4], [5, 6, 7, 8]
    capture = ref.capture_forward(model, prompt, generated)
    lens, diag = ref.compute_lens_quantities(model, capture, generated, chunk_tokens=2)
    expected = (3, 3, 4)
    for key in ref.QUANTITIES:
        check(lens[key].shape == expected, f"{key} has [module,layer,token] shape")
        check(np.isfinite(lens[key]).all(), f"{key} is finite")
    check(diag["final_residual_logit_pass"], "final residual lens exactly reproduces model logits")
    check(diag["final_residual_kl_identity_pass"], "KL(final residual || final) is exactly zero")
    check(np.array_equal(lens["lens_kl_final"][2, -1], np.zeros(4, dtype=np.float32)),
          "stored final residual KL vector is an exact zero identity")

    # Chunking changes memory use, never the definition.
    lens_one, _ = ref.compute_lens_quantities(model, capture, generated, chunk_tokens=1)
    for key in ref.QUANTITIES:
        check(np.allclose(lens[key], lens_one[key], atol=1e-6),
              f"{key} is invariant to token chunk size")


def test_float16_agreement_and_corruption():
    print("\n== sidecar float16 floor and corruption detection ==")
    model = fixture_model()
    generated = [5, 6, 7, 8]
    capture = ref.capture_forward(model, [1, 2, 3, 4], generated)
    lens, _ = ref.compute_lens_quantities(model, capture, generated, chunk_tokens=2)
    for key in ref.QUANTITIES:
        side = lens[key].astype(np.float16)
        result = ref._agreement(lens[key], side, key)
        check(result["pass"], f"{key} passes after realistic float16 sidecar quantization")
    corrupt = lens["lens_H"].astype(np.float16).astype(np.float32) + 0.2
    result = ref._agreement(lens["lens_H"], corrupt, "corrupt")
    check(not result["pass"], "systematic 0.2-nat corruption fails the registered threshold")
    wrong_shape = ref._agreement(lens["lens_H"], corrupt[:, :, :-1], "wrong_shape")
    check(not wrong_shape["pass"] and not wrong_shape["shape_pass"],
          "token-alignment shape mismatch fails closed")


def test_fixture_report_and_cli():
    print("\n== self-contained fixture report and CLI ==")
    report = ref.fixture_report()
    check(report["status"] == "PASS", "in-process deterministic fixture passes")
    check(report["architecture_fidelity_pass"], "fixture sets explicit fidelity verdict")
    check(report["corrected_gate_b"]["pass"], "fixture exercises corrected entropy Gate B")
    check(all(row["pass"] for row in report["repeatability"].values()),
          "fixture repeatability stays within the measured floor")

    with tempfile.TemporaryDirectory(prefix="layer_ref_test_") as tmp:
        out = os.path.join(tmp, "fixture.json")
        env = dict(os.environ)
        proc = subprocess.run(
            [sys.executable, os.path.join(CLUSTER, "run_layer_views_reference.py"),
             "--dry-run-fixture", "--out", out],
            cwd=REPO,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        check(proc.returncode == 0, f"fixture CLI exits zero ({proc.stdout.strip()})")
        check(os.path.exists(out), "fixture CLI writes JSON atomically")
        payload = json.load(open(out, encoding="utf-8"))
        check(payload["status"] == "PASS", "fixture CLI JSON contains PASS verdict")
        check("cells" not in payload, "fixture report cannot be mistaken for a live two-cell pilot")


def main():
    test_hooks_and_alignment()
    test_lens_and_final_identity()
    test_float16_agreement_and_corruption()
    test_fixture_report_and_cli()
    print(f"\n{'ALL PASS' if not FAILURES else f'{len(FAILURES)} FAILURE(S)'}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
