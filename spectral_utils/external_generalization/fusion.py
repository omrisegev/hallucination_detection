"""Bank11 transfer: no labels accepted, no implicit equal-weight fallback."""
import time
import numpy as np
from scipy.stats import spearmanr
from ._bank11.fusion_utils import lsml_continuous
from ._bank11.claude_feature_bank_v1 import build_token_feature_matrix, FEATURE_NAMES

EPS = 1e-12
ARMS = ("frozen_lsml", "frozen_equal", "frozen_partition_equal",
        "local_lsml", "local_equal", "local_partition_equal")


def standardize(x):
    x = np.asarray(x, dtype=np.float64)
    if not x.size or not np.isfinite(x).all():
        raise ValueError("empty/nonfinite observations")
    sd = x.std(axis=0)
    return np.divide(x - x.mean(axis=0), sd, out=np.zeros_like(x), where=sd > EPS)


def top10(x, spans):
    x = np.asarray(x, dtype=np.float64)
    output = []
    for a, b in spans:
        if not 0 <= a < b <= len(x):
            raise ValueError("invalid step span")
        k = min(10, b - a)
        output.append(np.partition(x[a:b], b - a - k, axis=0)[-k:].mean(axis=0))
    return np.asarray(output)


def answer_z(x):
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("nonfinite step scores")
    return (x - x.mean()) / x.std() if x.std() > 1e-8 else np.zeros_like(x)


def fit_weights(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < 3 or len(x) < 3 * x.shape[1] or not np.isfinite(x).all():
        raise ValueError("insufficient or invalid fitting observations")
    if x[:, 0].std() <= EPS:
        raise ValueError("fixed entropy orientation anchor is inactive")
    _, meta = lsml_continuous(*x.T, compute_score_matrix=False, small_m_guard=True)
    w = np.zeros(x.shape[1])
    for cross, (idx, within) in zip(meta["cross_weights"], meta["group_weights"]):
        w[np.asarray(idx, int)] = np.asarray(within) * cross
    rho = float(spearmanr(x @ w, x[:, 0]).statistic)
    if not np.isfinite(w).all() or abs(w).sum() <= EPS or not np.isfinite(rho):
        raise ValueError("invalid weights or undefined anchor orientation")
    if rho < 0:
        w *= -1
    w /= abs(w).sum()
    return {"weights": w.tolist(), "groups": np.asarray(meta["c"], int).tolist(),
            "anchor_spearman": abs(rho), "anchor_flipped": rho < 0,
            "residual": float(meta["residual"]),
            "small_m_guarded": [list(v) for v in meta["small_m_guarded"]]}


def partition_equal(groups):
    g = np.asarray(groups)
    return np.array([1 / (len(np.unique(g)) * np.count_nonzero(g == k)) for k in g])


def frozen_scores(token_matrix, spans, fit):
    step = standardize(top10(token_matrix, spans))
    return {"frozen_lsml": answer_z(step @ np.asarray(fit["weights"])),
            "frozen_equal": answer_z(step.mean(axis=1)),
            "frozen_partition_equal": answer_z(step @ partition_equal(fit["groups"]))}


def local_scores(token_matrix, spans):
    t0 = time.perf_counter()
    raw = np.asarray(token_matrix, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 11 or not np.isfinite(raw).all():
        raise ValueError("invalid bank11 telemetry; not an estimator failure")
    z = standardize(raw)
    sampled = z[::8]
    active = np.flatnonzero(sampled.std(axis=0) > EPS)
    diagnostics = {"active_channels": active.tolist(), "fit_rows": len(sampled), "native": False}
    fallback = answer_z(top10(raw[:, 2], spans))
    try:
        if 0 not in active:
            raise ValueError("inactive entropy anchor")
        fit = fit_weights(sampled[:, active])
        curves = {"local_lsml": z[:, active] @ np.asarray(fit["weights"]),
                  "local_equal": z[:, active].mean(axis=1),
                  "local_partition_equal": z[:, active] @ partition_equal(fit["groups"])}
        output = {name: answer_z(top10(curve, spans)) for name, curve in curves.items()}
        diagnostics.update(fit, native=True, fallback=None)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        # Identical routing on all three arms isolates fusion, including failures.
        output = {name: fallback.copy() for name in ARMS if name.startswith("local_")}
        diagnostics.update(fallback="chosen_surprisal", reason=str(exc))
    diagnostics["cpu_seconds"] = time.perf_counter() - t0
    return output, diagnostics


def validate_telemetry(row):
    ids = np.asarray(row["gen_token_ids"])
    lp = np.asarray(row["top_k_logprobs"]["logprobs"])
    top_ids = np.asarray(row["top_k_logprobs"]["ids"])
    if lp.shape != (len(ids), 50) or top_ids.shape != lp.shape or not np.isfinite(lp).all():
        raise ValueError("unaligned/nonfinite top50")
    if (lp > 1e-5).any() or (np.diff(lp, axis=1) > 1e-5).any() or (np.exp(lp).sum(1) > 1.001).any():
        raise ValueError("top50 is not sorted raw log probabilities")
    for name in ("token_entropies", "token_spilled_energies", "token_logsumexp"):
        x = np.asarray(row[name])
        if x.shape != ids.shape or not np.isfinite(x).all():
            raise ValueError("invalid telemetry series: " + name)
    hits = ids[:, None] == top_ids
    selected = hits.any(axis=1)
    chosen = -np.asarray(row["token_spilled_energies"])
    if not np.allclose(chosen[selected], lp[selected][hits[selected]], atol=2e-4, rtol=2e-4):
        raise ValueError("actual-token logprob disagrees with top50")
    if (chosen[~selected] > lp[~selected, -1] + 2e-4).any():
        raise ValueError("out-of-top50 token probability exceeds cutoff")
    return build_token_feature_matrix(row)


def decisions(scores, thresholds):
    return {arm: (np.asarray(s) < thresholds[arm]).tolist() for arm, s in scores.items()}
