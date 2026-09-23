"""Timing selection, measured cost projection and an explicit full-run gate."""
import numpy as np


def smoke_indices(lengths, uids, limit=12):
    if not 1 <= limit <= 12 or len(lengths) != len(uids) or not len(uids):
        raise ValueError("invalid smoke selection")
    order = sorted(range(len(uids)), key=lambda i: (lengths[i], uids[i]))
    # Even quantiles cover all length quartiles and always include the maximum.
    positions = np.linspace(0, len(order)-1, min(limit, len(order))).round().astype(int)
    return [order[i] for i in positions]


def cost_estimate(full_lengths, measurements, *, gpus=1, retry_factor=1.2):
    if len(measurements) < 2 or gpus < 1 or retry_factor < 1:
        raise ValueError("insufficient measured timing")
    x = np.asarray([r["context_tokens"] for r in measurements], float)
    y = np.asarray([r["seconds"] for r in measurements], float)
    if (x <= 0).any() or (y <= 0).any():
        raise ValueError("nonpositive timings")
    # Conservative envelope includes quadratic attention scaling, not only tokens/sec.
    lengths = np.asarray(full_lengths, float)
    nearest = np.abs(lengths[:, None] - x).argmin(1)
    ratios = lengths / x[nearest]
    seconds = y[nearest] * np.maximum(ratios, ratios**2)
    return {"measured_examples": len(x), "full_examples": len(lengths),
            "estimated_gpu_hours": float(seconds.sum()*gpus*retry_factor/3600),
            "estimated_wall_hours": float(seconds.sum()*retry_factor/3600),
            "retry_factor": retry_factor, "max_context_measured": int(x.max()),
            "max_context_full": int(lengths.max()),
            "peak_gpu_bytes": max(r.get("peak_gpu_bytes", 0) for r in measurements),
            "storage_bytes": int(sum(r.get("bytes", 0) for r in measurements)/x.sum()*lengths.sum()),
            "cpu_hours": float(sum(r.get("cpu_seconds", 0) for r in measurements)/x.sum()*lengths.sum()/3600),
            "generation_cost_included": all(r.get("generation_measured", False) for r in measurements)}


def authorize(mode, *, preflight, protocol_hash, estimate_hash=None, decision=None):
    if preflight.get("verdict") != "PASS" or preflight.get("protocol_hash") != protocol_hash:
        raise ValueError("submission requires same-protocol session preflight PASS")
    if not preflight.get("session_id"):
        raise ValueError("preflight session missing")
    if mode == "smoke":
        return
    if mode != "full" or not decision or decision.get("approved_by") != "user":
        raise ValueError("full inference requires the user's recorded budget decision")
    if decision.get("protocol_hash") != protocol_hash or decision.get("estimate_hash") != estimate_hash:
        raise ValueError("budget decision does not cover these locked estimates")
    if decision.get("max_gpu_hours", 0) <= 0:
        raise ValueError("full-run GPU budget missing")
