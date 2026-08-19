#!/usr/bin/env python
"""Independent architecture/fidelity pilot for ``layer-lens-v1`` sidecars.

The sidecars produced by AIRCC job 183956 contain plausible tensors, but their
report says that no token was checked and that the architecture check never ran.
This driver is deliberately independent of that missing capture implementation.
It reconstructs the original prompt, teacher-forces the saved generated token
IDs through Llama-3.1-8B, and hooks every decoder layer in the only order used by
the sidecar contract::

    self_attn output -> mlp output -> decoder-layer residual output

Every hooked vector is passed through the model's final RMSNorm and LM head.  The
resulting full-vocabulary logit-lens quantities are compared with the saved
float16 sidecar.  The ordinary model logits also pass through the already-
validated Gate-B path from :mod:`cluster.backfill_views`, including the original
temperature/top-k/top-p warp, so nested repgrid candidates are actually checked.

The pilot is read-only with respect to raw caches and sidecars.  It writes an
atomic JSON report and a resumable ``.state.pkl`` next to it.  A SIGTERM saves the
state and exits 85, matching the project's chained Slurm-resume convention.

Live usage on AIRCC (20 candidates *per* cell)::

    python cluster/run_layer_views_reference.py \
      --cells gsm8k,nq_open --n-candidates 20 \
      --out /shared/cycle2_tau_averbuch_prj/omrisegev1/results/\
whitebox_layer_reference_pilot/report.json

CPU-only known-answer fixture (no model download and no source data)::

    python cluster/run_layer_views_reference.py --dry-run-fixture --out /tmp/report.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import signal
import subprocess
import sys
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
CLUSTER_DIR = os.path.dirname(os.path.abspath(__file__))
if CLUSTER_DIR not in sys.path:
    sys.path.insert(0, CLUSTER_DIR)

import numpy as np
import torch

from backfill_specs import resolve_spec
from backfill_views import (
    build_prompt_ids,
    build_warpers,
    candidate_gen_ids,
    candidate_quantities,
    iter_problems,
)
from spectral_utils import free_memory, load_cache, load_model


MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
EXPECTED_LAYERS = 32
EXPECTED_HIDDEN = 4096
MODULES = ("attn", "mlp", "resid")
QUANTITIES = ("lens_H", "lens_logp_tgt", "lens_logp_top1", "lens_kl_final")
QUANTITY_KEYS = {
    "lens_H": "entropy",
    "lens_logp_tgt": "target_logp",
    "lens_logp_top1": "top1_logp",
    "lens_kl_final": "kl_to_final",
}

GATE_MEDIAN_MAX = 2e-2
GATE_FIRST_MAX = 5e-2
GATE_MIN_FRAC_CLOSE = 0.90
CLOSE_AT = 5e-2
FINAL_LOGIT_MAX = 1e-4
FINAL_KL_MAX = 1e-6
EXIT_INCOMPLETE = 85
STOP = {"flag": False}


@dataclass(frozen=True)
class PilotCell:
    alias: str
    cell_id: str
    dataset: str
    temperature: float
    raw_name: str
    sidecar_name: str


CELLS = {
    "gsm8k": PilotCell(
        alias="gsm8k",
        cell_id="lapeigvals_gsm8k_llama8b",
        dataset="gsm8k",
        temperature=1.0,
        raw_name="raw_gsm8k_T1.0.pkl",
        sidecar_name="layer_views_T1.0.pkl",
    ),
    "nq_open": PilotCell(
        alias="nq_open",
        cell_id="se_nq_open_llama8b",
        dataset="nq_open",
        temperature=0.5,
        raw_name="raw_nq_open_T0.5.pkl",
        sidecar_name="layer_views_T0.5.pkl",
    ),
}
CELL_ALIASES = {
    **{name: spec for name, spec in CELLS.items()},
    **{spec.cell_id: spec for spec in CELLS.values()},
    "gsm8k_t1": CELLS["gsm8k"],
    "nq": CELLS["nq_open"],
    "nq_open_t0.5": CELLS["nq_open"],
}


def _on_sigterm(signum, frame):
    del signum, frame
    STOP["flag"] = True
    print("[layer-reference] SIGTERM received; saving after current candidate", flush=True)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return ""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_json(obj: Mapping[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")
    os.replace(tmp, path)


def _atomic_pickle(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _sha256(path: str, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _tensor_from_output(output: Any, where: str) -> torch.Tensor:
    """Return the hidden tensor from a HF module output without architecture guesses."""
    value = output[0] if isinstance(output, (tuple, list)) else output
    if not torch.is_tensor(value) or value.ndim != 3:
        shape = getattr(value, "shape", None)
        raise RuntimeError(f"{where}: expected [batch,seq,hidden] tensor, got {type(value)} {shape}")
    return value


def llama_parts(model: torch.nn.Module) -> tuple[Any, Sequence[Any], Any, Any]:
    """Resolve the decoder, layers, final norm and LM head of a Llama causal LM.

    This intentionally fails closed instead of searching arbitrary submodules: a
    surprising wrapper would invalidate the hook-order claim and needs inspection.
    """
    decoder = getattr(model, "model", None)
    layers = getattr(decoder, "layers", None)
    norm = getattr(decoder, "norm", None)
    head = getattr(model, "lm_head", None)
    if decoder is None or layers is None or norm is None or head is None:
        raise RuntimeError("expected a causal LM exposing model.layers, model.norm, and lm_head")
    for idx, layer in enumerate(layers):
        if not hasattr(layer, "self_attn") or not hasattr(layer, "mlp"):
            raise RuntimeError(f"decoder layer {idx} lacks self_attn or mlp")
    return decoder, layers, norm, head


class LayerHookCapture(AbstractContextManager):
    """Capture aligned generated-token prediction states from all decoder hooks."""

    def __init__(self, layers: Sequence[Any], start: int, stop: int):
        if start < 0 or stop <= start:
            raise ValueError(f"invalid prediction slice [{start}:{stop}]")
        self.layers = layers
        self.start = int(start)
        self.stop = int(stop)
        self.events: list[tuple[int, str]] = []
        self.states: dict[tuple[str, int], torch.Tensor] = {}
        self._handles: list[Any] = []

    def _hook(self, layer_idx: int, module_name: str):
        def capture(module, inputs, output):
            del module, inputs
            hidden = _tensor_from_output(output, f"layer {layer_idx} {module_name}")
            if hidden.shape[0] != 1 or hidden.shape[1] < self.stop:
                raise RuntimeError(
                    f"layer {layer_idx} {module_name}: shape {tuple(hidden.shape)} cannot "
                    f"serve prediction slice [{self.start}:{self.stop}]"
                )
            key = (module_name, layer_idx)
            if key in self.states:
                raise RuntimeError(f"hook fired twice for {key}")
            # Keep the original compute dtype/device for an independent LM-head pass.
            self.states[key] = hidden[0, self.start:self.stop].detach().clone()
            self.events.append((layer_idx, module_name))

        return capture

    def __enter__(self):
        for idx, layer in enumerate(self.layers):
            self._handles.append(layer.self_attn.register_forward_hook(self._hook(idx, "attn")))
            self._handles.append(layer.mlp.register_forward_hook(self._hook(idx, "mlp")))
            self._handles.append(layer.register_forward_hook(self._hook(idx, "resid")))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        return False

    def expected_events(self) -> list[tuple[int, str]]:
        return [(idx, module) for idx in range(len(self.layers)) for module in MODULES]

    def validate(self, hidden_size: int, n_tokens: int) -> dict[str, Any]:
        expected = self.expected_events()
        expected_keys = {(module, idx) for idx in range(len(self.layers)) for module in MODULES}
        shapes = {f"{module}:{idx}": list(t.shape) for (module, idx), t in self.states.items()}
        keys_ok = set(self.states) == expected_keys
        shapes_ok = keys_ok and all(
            tuple(t.shape) == (n_tokens, hidden_size) for t in self.states.values()
        )
        return {
            "hook_order_pass": self.events == expected,
            "hook_keys_pass": keys_ok,
            "hook_shapes_pass": shapes_ok,
            "n_hook_events": len(self.events),
            "expected_hook_events": len(expected),
            "shapes": shapes,
        }


@dataclass
class CapturedForward:
    states: dict[tuple[str, int], torch.Tensor]
    final_logits: torch.Tensor
    hook_diagnostics: dict[str, Any]
    n_layers: int
    hidden_size: int
    n_tokens: int


def capture_forward(
    model: torch.nn.Module,
    prompt_ids: Sequence[int],
    gen_ids: Sequence[int],
) -> CapturedForward:
    """One source-independent teacher-forced forward pass with all 96 hooks."""
    if not prompt_ids:
        raise ValueError("prompt_ids cannot be empty (the first generated token needs a predecessor)")
    if not gen_ids:
        raise ValueError("gen_ids cannot be empty")
    _, layers, _, _ = llama_parts(model)
    device = next(model.parameters()).device
    sequence = list(prompt_ids) + list(gen_ids)
    input_ids = torch.as_tensor(sequence, dtype=torch.long, device=device).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    start = len(prompt_ids) - 1
    stop = start + len(gen_ids)
    hidden_size = int(getattr(model.config, "hidden_size", 0))
    if hidden_size <= 0:
        raise RuntimeError("model.config.hidden_size is missing")

    with LayerHookCapture(layers, start, stop) as hooks:
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = output.logits[0, start:stop].detach().clone()
    if logits.shape[0] != len(gen_ids):
        raise RuntimeError(f"ordinary logits have {logits.shape[0]} rows, need {len(gen_ids)}")
    diag = hooks.validate(hidden_size=hidden_size, n_tokens=len(gen_ids))
    return CapturedForward(
        states=hooks.states,
        final_logits=logits,
        hook_diagnostics=diag,
        n_layers=len(layers),
        hidden_size=hidden_size,
        n_tokens=len(gen_ids),
    )


def _lens_block(logits: torch.Tensor, targets: torch.Tensor, final_logp: torch.Tensor):
    logp = torch.log_softmax(logits.float(), dim=-1)
    probs = logp.exp()
    rows = torch.arange(logp.shape[0], device=logp.device)
    return (
        -(probs * logp).sum(dim=-1),
        logp[rows, targets],
        logp.max(dim=-1).values,
        (probs * (logp - final_logp)).sum(dim=-1),
    )


def compute_lens_quantities(
    model: torch.nn.Module,
    captured: CapturedForward,
    gen_ids: Sequence[int],
    chunk_tokens: int = 16,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Apply final RMSNorm+LM head to every captured state, bounded by token chunks."""
    if chunk_tokens < 1:
        raise ValueError("chunk_tokens must be positive")
    _, layers, norm, head = llama_parts(model)
    target = torch.as_tensor(gen_ids, dtype=torch.long, device=captured.final_logits.device)
    shape = (len(MODULES), len(layers), len(gen_ids))
    out = {key: np.empty(shape, dtype=np.float32) for key in QUANTITIES}
    max_final_logit_abs = 0.0

    # Calculate final-reference log probabilities once.  For Llama-3.1-8B and
    # 168 GSM8K tokens this is ~86 MB float32, comfortably bounded on a B200.
    final_logp = torch.log_softmax(captured.final_logits.float(), dim=-1)
    with torch.no_grad():
        for module_idx, module_name in enumerate(MODULES):
            for layer_idx in range(len(layers)):
                hidden = captured.states[(module_name, layer_idx)]
                parts = {key: [] for key in QUANTITIES}
                for start in range(0, len(gen_ids), chunk_tokens):
                    stop = min(start + chunk_tokens, len(gen_ids))
                    logits = head(norm(hidden[start:stop]))
                    if module_name == "resid" and layer_idx == len(layers) - 1:
                        delta = (logits.float() - captured.final_logits[start:stop].float()).abs()
                        if delta.numel():
                            max_final_logit_abs = max(max_final_logit_abs, float(delta.max().item()))
                    vals = _lens_block(logits, target[start:stop], final_logp[start:stop])
                    for key, value in zip(QUANTITIES, vals):
                        parts[key].append(value.detach().cpu().numpy().astype(np.float32))
                for key in QUANTITIES:
                    out[key][module_idx, layer_idx] = np.concatenate(parts[key])

    final_kl = out["lens_kl_final"][MODULES.index("resid"), -1]
    diag = {
        "final_residual_logit_max_abs": max_final_logit_abs,
        "final_residual_logit_pass": max_final_logit_abs <= FINAL_LOGIT_MAX,
        "final_residual_kl_max_abs": float(np.max(np.abs(final_kl))),
        "final_residual_kl_identity_pass": bool(np.max(np.abs(final_kl)) <= FINAL_KL_MAX),
    }
    return out, diag


def residual_norms(captured: CapturedForward) -> np.ndarray:
    rows = []
    for layer_idx in range(captured.n_layers):
        norm = torch.linalg.vector_norm(captured.states[("resid", layer_idx)].float(), dim=-1)
        rows.append(norm.detach().cpu().numpy().astype(np.float32))
    return np.stack(rows)


def _abs_summary(abs_values: np.ndarray, close_at: float = CLOSE_AT) -> dict[str, Any]:
    values = np.asarray(abs_values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"empty": True, "n": 0, "pass": False}
    median = float(np.median(values))
    frac_close = float(np.mean(values <= close_at))
    return {
        "empty": False,
        "n": int(values.size),
        "median_abs": median,
        "p99_abs": float(np.percentile(values, 99)),
        "max_abs": float(np.max(values)),
        "frac_within_0.05": frac_close,
        "pass": bool(median <= GATE_MEDIAN_MAX and frac_close >= GATE_MIN_FRAC_CLOSE),
    }


def _agreement(actual: np.ndarray, expected: np.ndarray, name: str) -> dict[str, Any]:
    a = np.asarray(actual, dtype=np.float32)
    b = np.asarray(expected, dtype=np.float32)
    if a.shape != b.shape:
        return {
            "name": name,
            "shape_actual": list(a.shape),
            "shape_expected": list(b.shape),
            "shape_pass": False,
            "finite_pass": False,
            "pass": False,
            "abs_diff": np.asarray([], dtype=np.float32),
        }
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    diff = np.abs(a - b).astype(np.float32)
    summary = _abs_summary(diff)
    summary.update({
        "name": name,
        "shape_actual": list(a.shape),
        "shape_expected": list(b.shape),
        "shape_pass": True,
        "finite_pass": finite,
        "pass": bool(finite and summary["pass"]),
        "abs_diff": diff,
    })
    return summary


def _relative_agreement(actual: np.ndarray, expected: np.ndarray, name: str) -> dict[str, Any]:
    a = np.asarray(actual, dtype=np.float32)
    b = np.asarray(expected, dtype=np.float32)
    if a.shape != b.shape:
        return {"name": name, "shape_pass": False, "pass": False,
                "rel_diff": np.asarray([], dtype=np.float32)}
    rel = np.abs(a - b) / np.maximum(1.0, np.abs(b))
    values = rel[np.isfinite(rel)]
    median = float(np.median(values)) if values.size else float("inf")
    frac = float(np.mean(values <= 0.05)) if values.size else 0.0
    return {
        "name": name,
        "shape_pass": True,
        "finite_pass": bool(np.isfinite(a).all() and np.isfinite(b).all()),
        "n": int(values.size),
        "median_relative_abs": median,
        "p99_relative_abs": float(np.percentile(values, 99)) if values.size else None,
        "max_relative_abs": float(np.max(values)) if values.size else None,
        "frac_within_0.05_relative": frac,
        "pass": bool(values.size and median <= 0.02 and frac >= 0.90),
        "rel_diff": rel.astype(np.float32),
    }


def _stored_raw_reference(candidate: Mapping[str, Any], gen_ids: Sequence[int]):
    topk = candidate.get("top_k_logprobs_raw")
    if not isinstance(topk, Mapping) or "ids" not in topk or "logprobs" not in topk:
        return None
    ids = np.asarray(topk["ids"])
    logp = np.asarray(topk["logprobs"], dtype=np.float32)
    if ids.ndim != 2 or logp.shape != ids.shape or ids.shape[0] != len(gen_ids):
        return None
    target = np.full(len(gen_ids), np.nan, dtype=np.float32)
    for row, token_id in enumerate(gen_ids):
        hit = np.flatnonzero(ids[row] == int(token_id))
        if hit.size:
            target[row] = logp[row, hit[0]]
    return {"top1_ids": ids[:, 0], "top1_logp": logp[:, 0], "target_logp": target}


def validate_candidate(
    model: torch.nn.Module,
    prompt_ids: Sequence[int],
    gen_ids: Sequence[int],
    candidate: Mapping[str, Any],
    sidecar_row: Mapping[str, Any],
    warpers: Any,
    chunk_tokens: int,
    repeat: bool,
) -> dict[str, Any]:
    """Run every source-independent check for one candidate."""
    n_tokens = len(gen_ids)
    side_n = int(sidecar_row.get("n_gen_tokens", -1))
    side_shapes = {
        key: list(np.asarray(sidecar_row.get(key)).shape) if key in sidecar_row else None
        for key in QUANTITIES
    }
    expected_shape = [len(MODULES), EXPECTED_LAYERS, n_tokens]
    alignment = {
        "n_gen_tokens": n_tokens,
        "n_saved_entropies": len(candidate.get("token_entropies") or []),
        "n_sidecar_tokens": side_n,
        "sidecar_shapes": side_shapes,
    }
    alignment["pass"] = bool(
        n_tokens > 0
        and alignment["n_saved_entropies"] == n_tokens
        and side_n == n_tokens
        and all(shape == expected_shape for shape in side_shapes.values())
    )

    captured = capture_forward(model, prompt_ids, gen_ids)
    lens, final_diag = compute_lens_quantities(model, captured, gen_ids, chunk_tokens)
    lens_agreement = {
        key: _agreement(lens[key], np.asarray(sidecar_row[key]), key) for key in QUANTITIES
    }
    norm_agreement = _relative_agreement(
        residual_norms(captured), np.asarray(sidecar_row.get("resid_norm")), "resid_norm"
    )

    # Corrected nested-candidate Gate B: ordinary aligned logits -> exact original
    # generation warp -> top-15 renormalized entropy -> saved per-token trace.
    target = torch.as_tensor(gen_ids, dtype=torch.long, device=captured.final_logits.device)
    q = candidate_quantities(
        captured.final_logits,
        target,
        warpers=warpers,
        raw_top_k=50,
        post_top_k=50,
    )
    saved_h = np.asarray(candidate.get("token_entropies") or [], dtype=np.float32)
    recomputed_h = np.asarray(q["token_entropies_recomputed"], dtype=np.float32)
    gate_b = _agreement(recomputed_h, saved_h, "token_entropies")
    first_abs = (
        float(abs(recomputed_h[0] - saved_h[0]))
        if recomputed_h.size and saved_h.size and recomputed_h.shape == saved_h.shape
        else None
    )
    gate_b["first_token_abs"] = first_abs
    # Numerical Gate-B thresholds are defined over the pilot population, not
    # candidate-by-candidate.  A single short trace may contain a legitimate
    # bf16 top-K boundary swap; only shape/finiteness are mechanical row gates.
    gate_b["candidate_numeric_pass"] = gate_b["pass"]
    gate_b["pass"] = bool(
        gate_b.get("shape_pass") and gate_b.get("finite_pass") and first_abs is not None
    )

    # Independent raw-logit alignment against values saved months before job 183956.
    raw_ref = _stored_raw_reference(candidate, gen_ids)
    raw_alignment: dict[str, Any] = {"available": raw_ref is not None, "pass": False}
    final_lp = torch.log_softmax(captured.final_logits.float(), dim=-1).detach().cpu().numpy()
    if raw_ref is not None:
        actual_top1 = final_lp.max(axis=1)
        actual_top1_ids = final_lp.argmax(axis=1)
        top1 = _agreement(actual_top1, raw_ref["top1_logp"], "saved_raw_top1_logp")
        mask = np.isfinite(raw_ref["target_logp"])
        rows = np.arange(n_tokens)
        actual_target = final_lp[rows, np.asarray(gen_ids, dtype=int)]
        target_agree = _agreement(
            actual_target[mask], raw_ref["target_logp"][mask], "saved_raw_target_logp"
        )
        id_rate = float(np.mean(actual_top1_ids == raw_ref["top1_ids"]))
        raw_alignment.update({
            "top1_logp": _strip_arrays(top1),
            "target_logp": _strip_arrays(target_agree),
            "target_coverage": float(np.mean(mask)),
            "top1_id_agreement": id_rate,
            "pass": bool(top1["pass"] and target_agree["pass"] and id_rate >= 0.90),
            "top1_abs_diff": top1["abs_diff"],
            "target_abs_diff": target_agree["abs_diff"],
        })

    float16_floor = {
        key: np.abs(lens[key] - lens[key].astype(np.float16).astype(np.float32)).astype(np.float32)
        for key in QUANTITIES
    }
    repeat_diffs: dict[str, np.ndarray] = {}
    repeat_diag: dict[str, Any] = {"performed": False, "pass": True}
    if repeat:
        again = capture_forward(model, prompt_ids, gen_ids)
        lens_again, again_final = compute_lens_quantities(model, again, gen_ids, chunk_tokens)
        repeat_summaries = {}
        for key in QUANTITIES:
            diff = np.abs(lens[key] - lens_again[key]).astype(np.float32)
            repeat_diffs[key] = diff
            repeat_summaries[key] = _strip_arrays(_abs_summary(diff))
        repeat_diag = {
            "performed": True,
            "quantities": repeat_summaries,
            "final_residual": again_final,
            "pass": bool(
                all(summary["pass"] for summary in repeat_summaries.values())
                and again_final["final_residual_logit_pass"]
                and again_final["final_residual_kl_identity_pass"]
            ),
        }
        del again, lens_again

    side_final_kl = np.asarray(sidecar_row["lens_kl_final"], dtype=np.float32)[2, -1]
    side_final_kl_max = float(np.max(np.abs(side_final_kl)))
    geometry_shapes = {
        key: list(np.asarray(sidecar_row.get(key)).shape)
        for key in ("cov_eigs", "hid_proj") if key in sidecar_row
    }
    hook_pass = all(
        captured.hook_diagnostics[key]
        for key in ("hook_order_pass", "hook_keys_pass", "hook_shapes_pass")
    )
    record = {
        "alignment": alignment,
        "architecture": {
            "n_layers": captured.n_layers,
            "hidden_size": captured.hidden_size,
            **captured.hook_diagnostics,
            **final_diag,
            "saved_final_residual_kl_max_abs": side_final_kl_max,
            "saved_final_residual_kl_identity_pass": side_final_kl_max == 0.0,
        },
        "lens_agreement": {key: _strip_arrays(value) for key, value in lens_agreement.items()},
        "resid_norm_agreement": _strip_arrays(norm_agreement),
        "gate_b": _strip_arrays(gate_b),
        "raw_logit_alignment": _strip_arrays(raw_alignment),
        "repeatability": repeat_diag,
        "geometry": {
            "shapes": geometry_shapes,
            "semantics_verified": False,
            "disposition": "omitted from performance analysis until the missing generator "
                           "defines projection/covariance semantics",
        },
        # Compact numeric evidence retained only in the resumable state.  The JSON
        # report contains summaries, not token-level arrays.
        "_lens_abs_diff": {key: value["abs_diff"] for key, value in lens_agreement.items()},
        "_float16_abs_diff": float16_floor,
        "_repeat_abs_diff": repeat_diffs,
        "_gate_abs_diff": gate_b.get("abs_diff", np.asarray([], dtype=np.float32)),
        "_norm_rel_diff": norm_agreement.get("rel_diff", np.asarray([], dtype=np.float32)),
        "_raw_top1_abs_diff": raw_alignment.get("top1_abs_diff", np.asarray([], dtype=np.float32)),
        "_raw_target_abs_diff": raw_alignment.get("target_abs_diff", np.asarray([], dtype=np.float32)),
        "_raw_top1_id_match": (
            actual_top1_ids == raw_ref["top1_ids"] if raw_ref is not None
            else np.asarray([], dtype=bool)
        ),
    }
    lens_mechanical = all(
        value.get("shape_pass") and value.get("finite_pass")
        for value in lens_agreement.values()
    )
    norm_mechanical = bool(
        norm_agreement.get("shape_pass") and norm_agreement.get("finite_pass")
    )
    raw_mechanical = bool(
        raw_ref is not None
        and raw_alignment.get("top1_logp", {}).get("shape_pass")
        and raw_alignment.get("top1_logp", {}).get("finite_pass")
        and raw_alignment.get("target_logp", {}).get("shape_pass")
        and raw_alignment.get("target_logp", {}).get("finite_pass")
    )
    record["pass"] = bool(
        alignment["pass"]
        and captured.n_layers == EXPECTED_LAYERS
        and captured.hidden_size == EXPECTED_HIDDEN
        and hook_pass
        and final_diag["final_residual_logit_pass"]
        and final_diag["final_residual_kl_identity_pass"]
        and side_final_kl_max == 0.0
        and lens_mechanical
        and norm_mechanical
        and gate_b["pass"]
        and raw_mechanical
        and repeat_diag["pass"]
    )
    del captured, lens
    return record


def _strip_arrays(value: Any) -> Any:
    """Recursively remove private/large ndarray evidence from report-facing values."""
    if isinstance(value, Mapping):
        return {
            key: _strip_arrays(item)
            for key, item in value.items()
            if not key.startswith("_") and not isinstance(item, np.ndarray)
        }
    if isinstance(value, list):
        return [_strip_arrays(item) for item in value]
    return value


def _concat_record_arrays(records: Iterable[Mapping[str, Any]], field: str, key: str | None = None):
    arrays = []
    for record in records:
        value = record.get(field, {})
        if key is not None:
            value = value.get(key, np.asarray([], dtype=np.float32))
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size:
            arrays.append(arr)
    return np.concatenate(arrays) if arrays else np.asarray([], dtype=np.float32)


def _cell_report(
    spec: PilotCell,
    state_cell: Mapping[str, Any],
    requested: int,
    repeat_n: int,
):
    records = list(state_cell.get("records", {}).values())
    lens = {
        key: _strip_arrays(_abs_summary(_concat_record_arrays(records, "_lens_abs_diff", key)))
        for key in QUANTITIES
    }
    floors = {
        key: _strip_arrays(_abs_summary(_concat_record_arrays(records, "_float16_abs_diff", key)))
        for key in QUANTITIES
    }
    repeats = {}
    for key in QUANTITIES:
        values = _concat_record_arrays(records, "_repeat_abs_diff", key)
        repeats[key] = _strip_arrays(_abs_summary(values)) if values.size else {"performed": False}
    repeat_pass = bool(
        repeat_n == 0
        or (
            sum(bool(r.get("repeatability", {}).get("performed")) for r in records)
            == min(repeat_n, requested)
            and all(value.get("pass", False) for value in repeats.values())
        )
    )
    gate_values = _concat_record_arrays(records, "_gate_abs_diff")
    gate = _strip_arrays(_abs_summary(gate_values))
    first = [r.get("gate_b", {}).get("first_token_abs") for r in records]
    first = [float(v) for v in first if v is not None and np.isfinite(v)]
    first_median = float(np.median(first)) if first else None
    gate["first_token_median_abs"] = first_median
    gate["pass"] = bool(
        gate.get("pass", False)
        and first_median is not None
        and first_median <= GATE_FIRST_MAX
        and len(records) == requested
    )
    top1 = _strip_arrays(_abs_summary(_concat_record_arrays(records, "_raw_top1_abs_diff")))
    target = _strip_arrays(_abs_summary(_concat_record_arrays(records, "_raw_target_abs_diff")))
    id_matches = _concat_record_arrays(records, "_raw_top1_id_match")
    top1_id_rate = float(np.mean(id_matches)) if id_matches.size else None
    raw_pass = bool(
        top1.get("pass", False)
        and target.get("pass", False)
        and top1_id_rate is not None
        and top1_id_rate >= 0.90
    )
    norm_values = _concat_record_arrays(records, "_norm_rel_diff")
    if norm_values.size:
        norm_summary = {
            "n": int(norm_values.size),
            "median_relative_abs": float(np.median(norm_values)),
            "p99_relative_abs": float(np.percentile(norm_values, 99)),
            "max_relative_abs": float(np.max(norm_values)),
            "frac_within_0.05_relative": float(np.mean(norm_values <= 0.05)),
        }
        norm_summary["pass"] = bool(
            norm_summary["median_relative_abs"] <= 0.02
            and norm_summary["frac_within_0.05_relative"] >= 0.90
        )
    else:
        norm_summary = {"empty": True, "pass": False}
    all_record_pass = bool(len(records) == requested and all(r.get("pass") for r in records))
    return {
        "cell": asdict(spec),
        "raw_path": state_cell.get("raw_path"),
        "sidecar_path": state_cell.get("sidecar_path"),
        "source_sha256": state_cell.get("source_sha256"),
        "sidecar_sha256": state_cell.get("sidecar_sha256"),
        "raw_size": state_cell.get("raw_size"),
        "sidecar_size": state_cell.get("sidecar_size"),
        "sidecar_meta": state_cell.get("sidecar_meta"),
        "n_requested": requested,
        "n_processed": len(records),
        "complete": len(records) == requested,
        "corrected_gate_b": gate,
        "lens_source_agreement": lens,
        "float16_quantization_floor": floors,
        "repeatability_floor": repeats,
        "residual_norm_source_agreement": norm_summary,
        "saved_raw_logit_agreement": {
            "top1_logp": top1,
            "target_logp": target,
            "top1_id_agreement": top1_id_rate,
            "pass": raw_pass,
        },
        "candidate_checks": {
            key: _strip_arrays(record) for key, record in state_cell.get("records", {}).items()
        },
        "geometry_semantics_verified": False,
        "geometry_disposition": "omit geometry performance; capture implementation is missing",
        "pass": bool(
            all_record_pass
            and gate["pass"]
            and all(v["pass"] for v in lens.values())
            and norm_summary["pass"]
            and raw_pass
            and repeat_pass
        ),
    }


def build_report(state: Mapping[str, Any], complete: bool) -> dict[str, Any]:
    requested = int(state["config"]["n_candidates"])
    repeat_n = int(state["config"]["repeat_n"])
    cells = {
        alias: _cell_report(
            CELLS[alias], state["cells"].get(alias, {}), requested, repeat_n
        )
        for alias in state["config"]["cells"]
    }
    all_complete = bool(complete and cells and all(cell["complete"] for cell in cells.values()))
    passed = bool(all_complete and all(cell["pass"] for cell in cells.values()))
    return {
        "schema_version": "layer-reference-pilot-v1",
        "driver": "cluster/run_layer_views_reference.py",
        "git_sha": _git_sha(),
        "job_id": os.environ.get("SLURM_JOB_ID", ""),
        "written_utc": _utcnow(),
        "status": "PASS" if passed else "INCOMPLETE" if not all_complete else "FAIL",
        "architecture_fidelity_pass": passed,
        "claim_boundary": (
            "This validates the Llama hook/logit-lens contract for two cells only. "
            "Projection and covariance geometry remain intentionally unvalidated."
        ),
        "quantity_contracts": {
            "corrected_gate_b": (
                "ordinary final logits -> original temperature/top-k/top-p warp -> "
                "renormalized top-15 entropy, compared with raw token_entropies"
            ),
            "sidecar_lens": (
                "unwarped full-vocabulary log-softmax of final-normed hooked states, "
                "compared only with layer-lens sidecar quantities"
            ),
            "non_equivalence": (
                "raw sampling-warped token_entropies and sidecar lens_H are different "
                "quantities and are never compared to each other"
            ),
        },
        "expected_architecture": {
            "model": MODEL_ID,
            "n_layers": EXPECTED_LAYERS,
            "hidden_size": EXPECTED_HIDDEN,
            "module_order": list(MODULES),
        },
        "thresholds": {
            "median_abs_max": GATE_MEDIAN_MAX,
            "first_token_median_abs_max": GATE_FIRST_MAX,
            "min_fraction_within_0.05": GATE_MIN_FRAC_CLOSE,
            "final_residual_logit_max_abs": FINAL_LOGIT_MAX,
            "final_residual_kl_max_abs": FINAL_KL_MAX,
        },
        "config": state["config"],
        "cells": cells,
    }


def _flatten_candidates(cache: Any, schema: str, limit: int):
    selected = []
    for problem_idx, gold_row, question, candidates in iter_problems(cache, schema):
        for candidate_idx, candidate in enumerate(candidates):
            selected.append((problem_idx, candidate_idx, gold_row, question, candidate))
            if len(selected) >= limit:
                return selected
    return selected


def _resolve_cell_paths(spec: Any, pilot: PilotCell, sidecar_root: str | None):
    raw_path = dict(spec.pkls).get(pilot.temperature)
    if raw_path is None:
        raise FileNotFoundError(f"{pilot.cell_id}: no raw pkl for T={pilot.temperature}")
    if os.path.basename(raw_path) != pilot.raw_name:
        raise RuntimeError(f"{pilot.cell_id}: resolved {raw_path}, expected {pilot.raw_name}")
    side_dir = (
        os.path.join(sidecar_root, pilot.cell_id) if sidecar_root else spec.data_dir
    )
    sidecar_path = os.path.join(side_dir, pilot.sidecar_name)
    if not os.path.exists(sidecar_path):
        raise FileNotFoundError(f"missing sidecar: {sidecar_path}")
    return raw_path, sidecar_path


def _config_signature(config: Mapping[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _empty_state(config: Mapping[str, Any]) -> dict[str, Any]:
    return {"signature": _config_signature(config), "config": dict(config), "cells": {}}


def run_live(cfg: argparse.Namespace) -> dict[str, Any]:
    selected_specs = []
    for name in [part.strip() for part in cfg.cells.split(",") if part.strip()]:
        if name not in CELL_ALIASES:
            raise SystemExit(f"unknown --cells value {name!r}; choose gsm8k,nq_open")
        pilot = CELL_ALIASES[name]
        if pilot not in selected_specs:
            selected_specs.append(pilot)
    if not selected_specs:
        raise SystemExit("--cells selected no cells")

    config = {
        "cells": [spec.alias for spec in selected_specs],
        "n_candidates": int(cfg.n_candidates),
        "repeat_n": int(cfg.repeat_n),
        "chunk_tokens": int(cfg.chunk_tokens),
        "model": cfg.model,
        "data_root": os.path.abspath(cfg.data_root),
        "sidecar_root": os.path.abspath(cfg.sidecar_root) if cfg.sidecar_root else None,
    }
    state_path = cfg.state or f"{cfg.out}.state.pkl"
    if os.path.exists(state_path):
        with open(state_path, "rb") as handle:
            state = pickle.load(handle)
        if state.get("signature") != _config_signature(config):
            raise SystemExit(
                f"resume state {state_path} belongs to a different configuration; "
                "choose another --out/--state"
            )
        print(f"[layer-reference] resuming {state_path}", flush=True)
    else:
        state = _empty_state(config)

    # Resolve and hash sources before allocating GPU memory.
    resolved = []
    for pilot in selected_specs:
        backfill_spec = resolve_spec(pilot.cell_id, cfg.data_root)
        raw_path, sidecar_path = _resolve_cell_paths(backfill_spec, pilot, cfg.sidecar_root)
        raw_stat, side_stat = os.stat(raw_path), os.stat(sidecar_path)
        cell_state = state["cells"].setdefault(pilot.alias, {"records": {}})
        raw_hash, side_hash = _sha256(raw_path), _sha256(sidecar_path)
        if cell_state.get("records"):
            if cell_state.get("source_sha256") != raw_hash:
                raise SystemExit(f"{raw_path}: source changed since the resume checkpoint")
            if cell_state.get("sidecar_sha256") != side_hash:
                raise SystemExit(f"{sidecar_path}: sidecar changed since the resume checkpoint")
        cell_state.update(
            raw_path=raw_path,
            sidecar_path=sidecar_path,
            source_sha256=raw_hash,
            sidecar_sha256=side_hash,
            raw_size=raw_stat.st_size,
            sidecar_size=side_stat.st_size,
        )
        resolved.append((pilot, backfill_spec, raw_path, sidecar_path))
    _atomic_pickle(state, state_path)
    _atomic_json(build_report(state, complete=False), cfg.out)

    print(f"[layer-reference] loading {cfg.model} (attn={cfg.attn}, bfloat16)", flush=True)
    model, tokenizer = load_model(cfg.model, attn_impl=cfg.attn, dtype="bfloat16")
    model.eval()
    _, layers, _, _ = llama_parts(model)
    hidden = int(model.config.hidden_size)
    if len(layers) != EXPECTED_LAYERS or hidden != EXPECTED_HIDDEN:
        raise SystemExit(
            f"architecture mismatch: got {len(layers)} layers/{hidden} hidden, "
            f"need {EXPECTED_LAYERS}/{EXPECTED_HIDDEN}"
        )

    for pilot, backfill_spec, raw_path, sidecar_path in resolved:
        print(f"\n=== {pilot.cell_id} T={pilot.temperature} ===", flush=True)
        cache = load_cache(raw_path)
        sidecar = load_cache(sidecar_path)
        meta = sidecar.get("_meta") if isinstance(sidecar, Mapping) else None
        if not isinstance(meta, Mapping):
            raise RuntimeError(f"{sidecar_path}: missing _meta")
        required_meta = {
            "version": "layer-lens-v1",
            "model": MODEL_ID,
            "n_layers": EXPECTED_LAYERS,
            "hidden_size": EXPECTED_HIDDEN,
            "modules": list(MODULES),
            "quantities": list(QUANTITIES),
        }
        mismatches = {
            key: {"got": meta.get(key), "expected": expected}
            for key, expected in required_meta.items() if meta.get(key) != expected
        }
        if mismatches:
            raise RuntimeError(f"{sidecar_path}: metadata mismatch: {mismatches}")
        state_cell = state["cells"][pilot.alias]
        state_cell["sidecar_meta"] = dict(meta)
        chosen = _flatten_candidates(cache, backfill_spec.schema, cfg.n_candidates)
        if len(chosen) != cfg.n_candidates:
            raise RuntimeError(
                f"{pilot.cell_id}: requested {cfg.n_candidates}, found {len(chosen)} candidates"
            )
        warpers = build_warpers(
            pilot.temperature,
            backfill_spec.warp_base.get("top_k"),
            backfill_spec.warp_base.get("top_p"),
        )

        for ordinal, (problem_idx, candidate_idx, gold_row, question, candidate) in enumerate(chosen):
            row_id = f"{problem_idx}:{candidate_idx}"
            if row_id in state_cell["records"]:
                print(f"[resume] {pilot.alias} {row_id}", flush=True)
                continue
            if row_id not in sidecar:
                raise RuntimeError(f"{sidecar_path}: missing row {row_id}")
            ids, source, delta = candidate_gen_ids(tokenizer, candidate, allow_roundtrip=False)
            if ids is None or source != "stored" or delta != 0:
                raise RuntimeError(f"{pilot.cell_id} {row_id}: unusable token IDs ({source})")
            prompt_ids = build_prompt_ids(
                tokenizer,
                backfill_spec.prompt_recipe,
                gold_row,
                question,
                candidate,
                idx=problem_idx,
            )
            print(
                f"[candidate] {pilot.alias} {ordinal + 1}/{cfg.n_candidates} {row_id} "
                f"prompt={len(prompt_ids)} gen={len(ids)}",
                flush=True,
            )
            record = validate_candidate(
                model=model,
                prompt_ids=prompt_ids,
                gen_ids=ids,
                candidate=candidate,
                sidecar_row=sidecar[row_id],
                warpers=warpers,
                chunk_tokens=cfg.chunk_tokens,
                repeat=ordinal < cfg.repeat_n,
            )
            record["row_id"] = row_id
            record["problem_group"] = str(problem_idx)
            record["pass"] = bool(record["pass"])
            state_cell["records"][row_id] = record
            _atomic_pickle(state, state_path)
            _atomic_json(build_report(state, complete=False), cfg.out)
            print(f"[candidate] {row_id}: {'PASS' if record['pass'] else 'FAIL'}", flush=True)
            if STOP["flag"]:
                print(f"[layer-reference] checkpointed to {state_path}", flush=True)
                free_memory()
                raise SystemExit(EXIT_INCOMPLETE)

        del cache, sidecar

    report = build_report(state, complete=True)
    _atomic_pickle(state, state_path)
    _atomic_json(report, cfg.out)
    free_memory()
    print(f"\n[layer-reference] {report['status']} -> {cfg.out}", flush=True)
    return report


# ---- deterministic CPU-only fixture -----------------------------------------

class _FixtureAttention(torch.nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.proj = torch.nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        return (torch.tanh(self.proj(x)), None)


class _FixtureMLP(torch.nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.proj = torch.nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        return torch.sin(self.proj(x))


class _FixtureLayer(torch.nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.self_attn = _FixtureAttention(hidden)
        self.mlp = _FixtureMLP(hidden)

    def forward(self, x):
        x = x + self.self_attn(x)[0]
        x = x + self.mlp(x)
        return (x,)


class _FixtureDecoder(torch.nn.Module):
    def __init__(self, vocab: int, hidden: int, layers: int):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab, hidden)
        self.layers = torch.nn.ModuleList([_FixtureLayer(hidden) for _ in range(layers)])
        self.norm = torch.nn.LayerNorm(hidden)

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)[0]
        return self.norm(x)


class _FixtureCausalLM(torch.nn.Module):
    def __init__(self, vocab: int = 13, hidden: int = 8, layers: int = 3):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden, num_hidden_layers=layers)
        self.model = _FixtureDecoder(vocab, hidden, layers)
        self.lm_head = torch.nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        del attention_mask, use_cache
        return SimpleNamespace(logits=self.lm_head(self.model(input_ids)))


def fixture_report() -> dict[str, Any]:
    """Exercise hooks, token alignment, lens algebra and float16 comparison on CPU."""
    torch.manual_seed(20260812)
    model = _FixtureCausalLM()
    model.eval()
    prompt_ids, gen_ids = [1, 2, 3, 4], [5, 6, 7, 8]
    captured = capture_forward(model, prompt_ids, gen_ids)
    lens, final = compute_lens_quantities(model, captured, gen_ids, chunk_tokens=2)
    captured_again = capture_forward(model, prompt_ids, gen_ids)
    lens_again, final_again = compute_lens_quantities(
        model, captured_again, gen_ids, chunk_tokens=3
    )
    side = {key: value.astype(np.float16) for key, value in lens.items()}
    side["resid_norm"] = residual_norms(captured).astype(np.float16)
    side["n_gen_tokens"] = len(gen_ids)
    agreements = {key: _agreement(lens[key], side[key], key) for key in QUANTITIES}
    repeatability = {
        key: _abs_summary(np.abs(lens[key] - lens_again[key])) for key in QUANTITIES
    }
    norm = _relative_agreement(residual_norms(captured), side["resid_norm"], "resid_norm")
    target = torch.as_tensor(gen_ids, dtype=torch.long)
    gate_values = candidate_quantities(
        captured.final_logits,
        target,
        warpers=None,
        raw_top_k=13,
        post_top_k=13,
    )["token_entropies_recomputed"]
    # Treat these independently calculated values as the generation-time trace.
    # This exercises the exact Gate-B comparison used live without transformers.
    gate_b = _agreement(
        np.asarray(gate_values, dtype=np.float32),
        np.asarray(gate_values, dtype=np.float16),
        "token_entropies",
    )
    gate_b["first_token_abs"] = float(gate_b["abs_diff"][0])
    gate_b["pass"] = bool(gate_b["pass"] and gate_b["first_token_abs"] <= GATE_FIRST_MAX)
    hook_pass = all(
        captured.hook_diagnostics[key]
        for key in ("hook_order_pass", "hook_keys_pass", "hook_shapes_pass")
    )
    passed = bool(
        hook_pass
        and captured.n_layers == 3
        and captured.hidden_size == 8
        and final["final_residual_logit_pass"]
        and final["final_residual_kl_identity_pass"]
        and final_again["final_residual_logit_pass"]
        and final_again["final_residual_kl_identity_pass"]
        and all(value["pass"] for value in agreements.values())
        and all(value["pass"] for value in repeatability.values())
        and norm["pass"]
        and gate_b["pass"]
    )
    return {
        "schema_version": "layer-reference-fixture-v1",
        "driver": "cluster/run_layer_views_reference.py",
        "written_utc": _utcnow(),
        "status": "PASS" if passed else "FAIL",
        "architecture_fidelity_pass": passed,
        "fixture": {"layers": 3, "hidden_size": 8, "vocab": 13, "n_tokens": 4},
        "hook_diagnostics": captured.hook_diagnostics,
        "final_residual": final,
        "corrected_gate_b": _strip_arrays(gate_b),
        "lens_float16_agreement": {
            key: _strip_arrays(value) for key, value in agreements.items()
        },
        "repeatability": {
            key: _strip_arrays(value) for key, value in repeatability.items()
        },
        "resid_norm_float16_agreement": _strip_arrays(norm),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Source-independent Llama layer-view architecture/fidelity pilot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cells", default="gsm8k,nq_open",
                        help="comma-separated pilot cells: gsm8k,nq_open")
    parser.add_argument("--n-candidates", type=int, default=20, help="candidate cap per cell")
    parser.add_argument("--repeat-n", type=int, default=2,
                        help="first N candidates per cell rerun for self-consistency")
    parser.add_argument("--chunk-tokens", type=int, default=64,
                        help="tokens per full-vocabulary logit-lens projection")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--attn", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument(
        "--data-root",
        default="/shared/cycle2_tau_averbuch_prj/omrisegev1",
        help="root containing results/repgrid/<cell>",
    )
    parser.add_argument(
        "--sidecar-root",
        default=None,
        help="optional root containing <cell>/layer_views_*.pkl; default is raw cell dir",
    )
    parser.add_argument(
        "--out",
        default="/shared/cycle2_tau_averbuch_prj/omrisegev1/results/"
                "whitebox_layer_reference_pilot/report.json",
    )
    parser.add_argument("--state", default=None, help="resume-state pkl; default <out>.state.pkl")
    parser.add_argument("--dry-run-fixture", action="store_true",
                        help="CPU-only deterministic fixture; reads no model/data")
    args = parser.parse_args(argv)
    if args.n_candidates < 1 or args.repeat_n < 0 or args.chunk_tokens < 1:
        parser.error("--n-candidates/--chunk-tokens must be positive and --repeat-n nonnegative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    cfg = parse_args(argv)
    if cfg.dry_run_fixture:
        report = fixture_report()
        _atomic_json(report, cfg.out)
        print(f"[layer-reference] fixture {report['status']} -> {cfg.out}")
        return 0 if report["architecture_fidelity_pass"] else 1
    if cfg.model != MODEL_ID:
        raise SystemExit(f"this fidelity pilot is frozen to {MODEL_ID}; got {cfg.model}")
    signal.signal(signal.SIGTERM, _on_sigterm)
    report = run_live(cfg)
    return 0 if report["architecture_fidelity_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
