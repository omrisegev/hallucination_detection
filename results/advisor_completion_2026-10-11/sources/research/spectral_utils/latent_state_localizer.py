"""Label-free latent-state localization over an IU-PCR token-risk curve.

This module is deliberately small and task-agnostic.  It receives one scalar
risk sequence per trace; it never accepts correctness labels, text, or
benchmark step boundaries.  The intended ProcessBench adapter first fuses the
five frozen token views with ordinary two-component IU-PCR, then fits one of
two Gaussian hidden-state models to the resulting scalar sequences:

``reversible``
    A two-state HMM whose high-risk state may be entered and left repeatedly.
    This is the primary model because earlier project diagnostics found that
    uncertainty bursts need not persist after a reasoning error.

``absorbing``
    A one-change-point HMM.  State 0 may transition to state 1, and state 1 is
    absorbing.  This is the registered falsification control for the stronger
    claim that the observable risk regime remains changed after the first
    error.

Both models have one-dimensional Gaussian emissions with a variance shared by
the states.  Shared variance prevents a broad-variance component from being
silently called the "high-risk" state.  Three deterministic jittered starts
are fitted by EM; the valid candidate with greatest *unlabeled* likelihood is
selected.  State identity is fixed without labels by requiring state 1 to have
the larger mean IU-PCR risk.

The localization curve is the smoothed posterior probability of *entering*
the high-risk state: ``P(S_0=1 | r)`` at the first token and
``P(S_{t-1}=0,S_t=1 | r)`` afterwards.  This differs from state occupancy:
occupancy creates a plateau after a persistent transition, whereas entry
probability directly estimates the onset requested by first-error benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np


EPS = 1e-12
DEFAULT_SEEDS = (11, 23, 37)
DEFAULT_MIN_SEED_LOCATOR_AGREEMENT = 0.80


@dataclass(frozen=True)
class SharedVarianceHMM:
    """One fitted two-state scalar Gaussian HMM."""

    kind: str
    means: np.ndarray
    variance: float
    transition: np.ndarray
    start: np.ndarray
    risk_mean: float
    risk_scale: float
    log_likelihood: float
    occupancy: np.ndarray
    separation: float
    seed: int
    n_iter_used: int
    valid: bool
    guard_failures: tuple[str, ...] = ()


@dataclass(frozen=True)
class LatentStateFit:
    """Selected model plus all deterministic-start diagnostics."""

    kind: str
    selected: SharedVarianceHMM | None
    candidates: tuple[SharedVarianceHMM, ...]
    fallback: bool
    fallback_reason: str | None
    diagnostics: dict = field(default_factory=dict)


def _validate_kind(kind: str) -> str:
    kind = str(kind).lower()
    if kind not in {"reversible", "absorbing"}:
        raise ValueError("kind must be 'reversible' or 'absorbing'")
    return kind


def _validate_sequences(sequences: Iterable[Sequence[float]]) -> list[np.ndarray]:
    output = []
    for index, values in enumerate(sequences):
        array = np.asarray(values, dtype=float)
        if array.ndim != 1:
            raise ValueError(f"sequence {index} is not one-dimensional")
        if len(array) < 2:
            raise ValueError(f"sequence {index} has fewer than two tokens")
        if not np.isfinite(array).all():
            raise ValueError(f"sequence {index} contains non-finite values")
        output.append(array)
    if not output:
        raise ValueError("at least one risk sequence is required")
    return output


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    matrix = np.maximum(matrix, EPS)
    return matrix / matrix.sum(axis=1, keepdims=True)


def _log_emissions(values: np.ndarray, means: np.ndarray, variance: float) -> np.ndarray:
    variance = max(float(variance), EPS)
    diff = values[:, None] - np.asarray(means, dtype=float)[None, :]
    return -0.5 * (np.log(2.0 * np.pi * variance) + diff * diff / variance)


def forward_backward(
    values: Sequence[float],
    means: Sequence[float],
    variance: float,
    transition: np.ndarray,
    start: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return smoothed state and transition posteriors in log space.

    ``gamma`` has shape ``(T, 2)`` and ``xi`` has shape ``(T-1, 2, 2)``.
    ``xi[t,a,b]`` is ``P(S_t=a,S_{t+1}=b | values)``.
    """

    values = np.asarray(values, dtype=float)
    means = np.asarray(means, dtype=float)
    transition = np.asarray(transition, dtype=float)
    start = np.asarray(start, dtype=float)
    if values.ndim != 1 or len(values) < 1 or not np.isfinite(values).all():
        raise ValueError("values must be a finite one-dimensional non-empty array")
    if means.shape != (2,) or transition.shape != (2, 2) or start.shape != (2,):
        raise ValueError("forward_backward expects exactly two states")
    if np.any(transition < 0) or np.any(start < 0):
        raise ValueError("probabilities must be nonnegative")
    if not np.allclose(transition.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("transition rows must sum to one")
    if not np.isclose(start.sum(), 1.0, atol=1e-10):
        raise ValueError("start probabilities must sum to one")

    # Work in log space.  The absorbing control can assign exactly zero
    # probability to paths that start in state 1 or return to state 0.  A
    # probability-domain backward recursion may then repeatedly divide by a
    # near-zero forward scale when the only reachable state has an extremely
    # unlikely emission.  That overflow was observed on a real ProcessBench
    # cell.  Two states make the log-domain recursion cheap, and ``logaddexp``
    # handles the exact structural zeros without an arbitrary probability
    # floor.
    log_b = _log_emissions(values, means, variance)
    with np.errstate(divide="ignore"):
        log_transition = np.where(transition > 0.0, np.log(transition), -np.inf)
        log_start = np.where(start > 0.0, np.log(start), -np.inf)
    length = len(values)
    log_alpha = np.empty((length, 2), dtype=float)
    log_beta = np.empty((length, 2), dtype=float)
    log_alpha[0] = log_start + log_b[0]
    for token in range(1, length):
        log_alpha[token, 0] = log_b[token, 0] + np.logaddexp(
            log_alpha[token - 1, 0] + log_transition[0, 0],
            log_alpha[token - 1, 1] + log_transition[1, 0],
        )
        log_alpha[token, 1] = log_b[token, 1] + np.logaddexp(
            log_alpha[token - 1, 0] + log_transition[0, 1],
            log_alpha[token - 1, 1] + log_transition[1, 1],
        )
    log_likelihood = float(np.logaddexp(log_alpha[-1, 0], log_alpha[-1, 1]))
    if not np.isfinite(log_likelihood):
        raise FloatingPointError("non-finite HMM sequence log likelihood")

    log_beta[-1] = 0.0
    for token in range(length - 2, -1, -1):
        log_beta[token, 0] = np.logaddexp(
            log_transition[0, 0] + log_b[token + 1, 0] + log_beta[token + 1, 0],
            log_transition[0, 1] + log_b[token + 1, 1] + log_beta[token + 1, 1],
        )
        log_beta[token, 1] = np.logaddexp(
            log_transition[1, 0] + log_b[token + 1, 0] + log_beta[token + 1, 0],
            log_transition[1, 1] + log_b[token + 1, 1] + log_beta[token + 1, 1],
        )

    log_gamma = log_alpha + log_beta - log_likelihood
    gamma_shift = np.max(log_gamma, axis=1, keepdims=True)
    gamma = np.exp(log_gamma - gamma_shift)
    gamma /= gamma.sum(axis=1, keepdims=True)

    xi = np.empty((max(length - 1, 0), 2, 2), dtype=float)
    for token in range(length - 1):
        log_xi = (
            log_alpha[token][:, None]
            + log_transition
            + log_b[token + 1][None, :]
            + log_beta[token + 1][None, :]
            - log_likelihood
        )
        xi_shift = float(np.max(log_xi))
        xi[token] = np.exp(log_xi - xi_shift)
        xi[token] /= xi[token].sum()
    return gamma, xi, log_likelihood


def _initial_parameters(kind: str, pooled: np.ndarray, seed: int, median_length: float):
    rng = np.random.default_rng(int(seed))
    lower, upper = np.quantile(pooled, (0.25, 0.75))
    spread = max(float(upper - lower), 0.25)
    means = np.array([lower, upper], dtype=float)
    means += rng.normal(0.0, 0.025 * spread, size=2)
    means.sort()
    variance = max(float(np.var(pooled)), 1e-3)
    if kind == "reversible":
        # High-risk visits may be brief; only the low-risk state receives a
        # strong sticky initialization.  EM remains free to change both rows.
        transition = np.array([[0.985, 0.015], [0.25, 0.75]], dtype=float)
        transition += rng.normal(0.0, 0.002, size=(2, 2))
        transition = _normalize_rows(transition)
        start = np.array([0.95, 0.05], dtype=float)
    else:
        hazard = float(np.clip(1.0 / max(median_length, 2.0), 1e-4, 0.20))
        hazard *= float(np.exp(rng.normal(0.0, 0.08)))
        hazard = float(np.clip(hazard, 1e-5, 0.25))
        transition = np.array([[1.0 - hazard, hazard], [0.0, 1.0]], dtype=float)
        # Registered one-change-point control: every trace starts before the
        # change. Keep pi fixed instead of learning a first-token shortcut.
        start = np.array([1.0, 0.0], dtype=float)
    return means, variance, transition, start


def _order_reversible(means, transition, start, occupancy=None):
    order = np.argsort(np.asarray(means, dtype=float))
    means = np.asarray(means, dtype=float)[order]
    transition = np.asarray(transition, dtype=float)[np.ix_(order, order)]
    start = np.asarray(start, dtype=float)[order]
    if occupancy is None:
        return means, transition, start
    return means, transition, start, np.asarray(occupancy, dtype=float)[order]


def _fit_one(
    normalized: list[np.ndarray],
    *,
    kind: str,
    seed: int,
    risk_mean: float,
    risk_scale: float,
    max_iter: int,
    tolerance: float,
    variance_floor: float,
    min_occupancy: float,
    min_separation: float,
    probability_floor: float,
) -> SharedVarianceHMM:
    pooled = np.concatenate(normalized)
    means, variance, transition, start = _initial_parameters(
        kind, pooled, seed, float(np.median([len(values) for values in normalized]))
    )
    previous = -np.inf
    log_likelihood = -np.inf
    occupancy = np.full(2, 0.5)
    n_iter_used = 0

    for iteration in range(int(max_iter)):
        n_iter_used = iteration + 1
        gamma0 = np.zeros(2, dtype=float)
        gamma_sum = np.zeros(2, dtype=float)
        xi_sum = np.zeros((2, 2), dtype=float)
        mean_num = np.zeros(2, dtype=float)
        posteriors = []
        log_likelihood = 0.0
        for values in normalized:
            gamma, xi, sequence_ll = forward_backward(
                values, means, variance, transition, start
            )
            posteriors.append((values, gamma))
            gamma0 += gamma[0]
            gamma_sum += gamma.sum(axis=0)
            if len(xi):
                xi_sum += xi.sum(axis=0)
            mean_num += gamma.T @ values
            log_likelihood += sequence_ll

        means = mean_num / np.maximum(gamma_sum, EPS)
        variance_num = 0.0
        for values, gamma in posteriors:
            variance_num += float(np.sum(gamma * (values[:, None] - means[None, :]) ** 2))
        variance = max(float(variance_num / max(float(gamma_sum.sum()), EPS)), variance_floor)

        if kind == "reversible":
            start = np.maximum(gamma0, probability_floor)
            start /= start.sum()
            transition = xi_sum + probability_floor
            transition = _normalize_rows(transition)
            means, transition, start = _order_reversible(means, transition, start)
        else:
            start = np.array([1.0, 0.0], dtype=float)
            row0 = xi_sum[0] + probability_floor
            row0 /= row0.sum()
            transition = np.array([[row0[0], row0[1]], [0.0, 1.0]], dtype=float)

        if np.isfinite(previous) and abs(log_likelihood - previous) <= (
            tolerance * max(abs(previous), 1.0)
        ):
            break
        previous = log_likelihood

    # Recompute sufficient diagnostics under the final parameter update.
    gamma_sum = np.zeros(2, dtype=float)
    log_likelihood = 0.0
    for values in normalized:
        gamma, _, sequence_ll = forward_backward(values, means, variance, transition, start)
        gamma_sum += gamma.sum(axis=0)
        log_likelihood += sequence_ll
    occupancy = gamma_sum / max(float(gamma_sum.sum()), EPS)
    if kind == "reversible":
        means, transition, start, occupancy = _order_reversible(
            means, transition, start, occupancy
        )

    separation = float((means[1] - means[0]) / np.sqrt(max(variance, EPS)))
    failures = []
    arrays = (means, transition, start, occupancy)
    if not np.isfinite(log_likelihood) or not np.isfinite(variance) or not all(
        np.isfinite(array).all() for array in arrays
    ):
        failures.append("nonfinite_parameters")
    if means[1] <= means[0]:
        failures.append("state_order_reversed")
    if float(np.min(occupancy)) < float(min_occupancy):
        failures.append("state_collapse")
    if separation < float(min_separation):
        failures.append("weak_state_separation")
    if kind == "reversible":
        free = transition.ravel()
    else:
        free = transition[0]
    if np.any(free <= probability_floor) or np.any(free >= 1.0 - probability_floor):
        failures.append("transition_boundary")

    return SharedVarianceHMM(
        kind=kind,
        means=np.asarray(means, dtype=float),
        variance=float(variance),
        transition=np.asarray(transition, dtype=float),
        start=np.asarray(start, dtype=float),
        risk_mean=float(risk_mean),
        risk_scale=float(risk_scale),
        log_likelihood=float(log_likelihood),
        occupancy=np.asarray(occupancy, dtype=float),
        separation=separation,
        seed=int(seed),
        n_iter_used=int(n_iter_used),
        valid=not failures,
        guard_failures=tuple(failures),
    )


def posterior_entry_curve(model: SharedVarianceHMM, values: Sequence[float]) -> np.ndarray:
    """Return the high-risk-state entry posterior on the original token grid."""

    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("values must be a finite one-dimensional non-empty array")
    normalized = (values - model.risk_mean) / model.risk_scale
    gamma, xi, _ = forward_backward(
        normalized,
        model.means,
        model.variance,
        model.transition,
        model.start,
    )
    entry = np.empty(len(values), dtype=float)
    entry[0] = gamma[0, 1]
    if len(values) > 1:
        entry[1:] = xi[:, 0, 1]
    return np.clip(entry, 0.0, 1.0)


def posterior_high_risk_curve(model: SharedVarianceHMM, values: Sequence[float]) -> np.ndarray:
    """Return state-1 occupancy for diagnostics, not onset localization."""

    values = np.asarray(values, dtype=float)
    normalized = (values - model.risk_mean) / model.risk_scale
    gamma, _, _ = forward_backward(
        normalized,
        model.means,
        model.variance,
        model.transition,
        model.start,
    )
    return gamma[:, 1]


def _candidate_agreement(candidates, risk_sequences):
    valid = [candidate for candidate in candidates if candidate.valid]
    if len(valid) < 2:
        return {
            "n_valid_candidates": len(valid),
            "mean_pair_exact_argmax_agreement": float("nan"),
            "mean_pair_normalized_argmax_displacement": float("nan"),
        }
    locators = []
    for candidate in valid:
        locators.append(np.asarray([
            int(np.argmax(posterior_entry_curve(candidate, values)))
            for values in risk_sequences
        ], dtype=int))
    exact, displacement = [], []
    lengths = np.asarray([len(values) for values in risk_sequences], dtype=float)
    for left in range(len(locators)):
        for right in range(left + 1, len(locators)):
            exact.append(float(np.mean(locators[left] == locators[right])))
            displacement.append(float(np.mean(
                np.abs(locators[left] - locators[right]) / np.maximum(lengths, 1.0)
            )))
    return {
        "n_valid_candidates": len(valid),
        "mean_pair_exact_argmax_agreement": float(np.mean(exact)),
        "mean_pair_normalized_argmax_displacement": float(np.mean(displacement)),
    }


def _fit_one_or_invalid(
    normalized: list[np.ndarray],
    *,
    kind: str,
    seed: int,
    risk_mean: float,
    risk_scale: float,
    **fit_kwargs,
) -> SharedVarianceHMM:
    """Turn a seed-local numerical failure into a guard-invalid candidate.

    Structural validation errors are intentionally not caught.  A single EM
    start may encounter a numerically impossible path without invalidating the
    other deterministic starts or the exact IU-PCR fallback policy.
    """

    try:
        return _fit_one(
            normalized,
            kind=kind,
            seed=seed,
            risk_mean=risk_mean,
            risk_scale=risk_scale,
            **fit_kwargs,
        )
    except FloatingPointError:
        if kind == "reversible":
            transition = np.full((2, 2), 0.5, dtype=float)
            start = np.full(2, 0.5, dtype=float)
        else:
            transition = np.array([[0.5, 0.5], [0.0, 1.0]], dtype=float)
            start = np.array([1.0, 0.0], dtype=float)
        return SharedVarianceHMM(
            kind=kind,
            means=np.zeros(2, dtype=float),
            variance=1.0,
            transition=transition,
            start=start,
            risk_mean=float(risk_mean),
            risk_scale=float(risk_scale),
            log_likelihood=-float(np.finfo(float).max),
            occupancy=np.full(2, 0.5, dtype=float),
            separation=0.0,
            seed=int(seed),
            n_iter_used=0,
            valid=False,
            guard_failures=("numerical_failure",),
        )


def fit_upcr_initialized_hmm(
    risk_sequences: Iterable[Sequence[float]],
    *,
    kind: str = "reversible",
    seeds: Sequence[int] = DEFAULT_SEEDS,
    max_iter: int = 80,
    tolerance: float = 1e-5,
    variance_floor: float = 1e-3,
    min_occupancy: float = 0.02,
    min_separation: float = 0.25,
    probability_floor: float = 1e-6,
    min_seed_locator_agreement: float = DEFAULT_MIN_SEED_LOCATOR_AGREEMENT,
) -> LatentStateFit:
    """Fit deterministic-start HMM candidates without labels.

    The caller supplies the IU-PCR risk curves used for both initialization and
    emission fitting.  If every start fails a label-free guard, ``selected`` is
    ``None`` and the caller must fall back to the original IU-PCR argmax.
    """

    kind = _validate_kind(kind)
    sequences = _validate_sequences(risk_sequences)
    seeds = tuple(int(seed) for seed in seeds)
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("exactly three distinct deterministic seeds are required")
    if not 0.0 <= float(min_seed_locator_agreement) <= 1.0:
        raise ValueError("min_seed_locator_agreement must be in [0, 1]")
    pooled = np.concatenate(sequences)
    risk_mean = float(pooled.mean())
    risk_scale = float(pooled.std())
    if not np.isfinite(risk_scale) or risk_scale < 1e-8:
        return LatentStateFit(
            kind=kind,
            selected=None,
            candidates=(),
            fallback=True,
            fallback_reason="constant_iu_risk",
            diagnostics={"risk_mean": risk_mean, "risk_scale": risk_scale},
        )
    normalized = [(values - risk_mean) / risk_scale for values in sequences]
    candidates = tuple(
        _fit_one_or_invalid(
            normalized,
            kind=kind,
            seed=seed,
            risk_mean=risk_mean,
            risk_scale=risk_scale,
            max_iter=max_iter,
            tolerance=tolerance,
            variance_floor=variance_floor,
            min_occupancy=min_occupancy,
            min_separation=min_separation,
            probability_floor=probability_floor,
        )
        for seed in seeds
    )
    agreement = _candidate_agreement(candidates, sequences)
    valid = [candidate for candidate in candidates if candidate.valid]
    mean_agreement = agreement["mean_pair_exact_argmax_agreement"]
    agreement_guard_passed = (
        len(valid) >= 2
        and np.isfinite(mean_agreement)
        and mean_agreement >= float(min_seed_locator_agreement)
    )
    selected = (
        max(valid, key=lambda candidate: candidate.log_likelihood)
        if valid and agreement_guard_passed
        else None
    )
    if not valid:
        fallback_reason = "all_hmm_starts_failed_parameter_guards"
    elif len(valid) < 2:
        fallback_reason = "fewer_than_two_guard_valid_starts"
    elif not agreement_guard_passed:
        fallback_reason = "unstable_seed_locators"
    else:
        fallback_reason = None
    diagnostics = {
        "risk_mean": risk_mean,
        "risk_scale": risk_scale,
        "n_fit_sequences": len(sequences),
        "n_fit_tokens": int(sum(len(values) for values in sequences)),
        "selection_rule": "maximum unlabeled log likelihood among guard-valid starts",
        "candidate_log_likelihoods": [candidate.log_likelihood for candidate in candidates],
        "candidate_valid": [candidate.valid for candidate in candidates],
        "candidate_guard_failures": [list(candidate.guard_failures) for candidate in candidates],
        "minimum_seed_locator_agreement": float(min_seed_locator_agreement),
        "seed_locator_agreement_guard_passed": bool(agreement_guard_passed),
        **agreement,
    }
    return LatentStateFit(
        kind=kind,
        selected=selected,
        candidates=candidates,
        fallback=selected is None,
        fallback_reason=fallback_reason,
        diagnostics=diagnostics,
    )


def apply_latent_state_fit(
    fit: LatentStateFit,
    risk_sequences: Iterable[Sequence[float]],
) -> tuple[list[np.ndarray], np.ndarray, dict]:
    """Apply a fitted HMM or fall back exactly to IU-PCR argmax.

    On fallback the returned curves are the input IU-PCR risk curves.  This is
    intentional: the benchmark adapter can hash and evaluate one output shape
    regardless of whether the latent-state guards passed.
    """

    sequences = [np.asarray(values, dtype=float) for values in risk_sequences]
    if any(values.ndim != 1 or not len(values) or not np.isfinite(values).all()
           for values in sequences):
        raise ValueError("application sequences must be finite non-empty vectors")
    if fit.selected is None:
        curves = [values.copy() for values in sequences]
        mean_entry_mass = float("nan")
        mean_peak_entry_probability = float("nan")
        mean_normalized_entry_entropy = float("nan")
        fraction_without_entry_above_0p10 = float("nan")
        output_curve_kind = "iu_risk_fallback"
    else:
        curves = [posterior_entry_curve(fit.selected, values) for values in sequences]
        mean_entry_mass = float(np.mean([np.mean(values) for values in curves]))
        peaks = np.asarray([np.max(values) for values in curves], dtype=float)
        mean_peak_entry_probability = float(np.mean(peaks))
        entropies = []
        for values in curves:
            probability = values / max(float(np.sum(values)), EPS)
            entropy = -float(np.sum(
                probability * np.log(np.maximum(probability, EPS))
            ))
            entropies.append(entropy / max(float(np.log(len(values))), 1.0))
        mean_normalized_entry_entropy = float(np.mean(entropies))
        fraction_without_entry_above_0p10 = float(np.mean(peaks < 0.10))
        output_curve_kind = "posterior_state_entry_probability"
    locator = np.asarray([int(np.argmax(values)) for values in curves], dtype=int)
    diagnostics = {
        "used_fallback": fit.selected is None,
        "fallback_reason": fit.fallback_reason,
        "selected_seed": None if fit.selected is None else fit.selected.seed,
        "output_curve_kind": output_curve_kind,
        "mean_entry_mass": mean_entry_mass,
        "mean_peak_entry_probability": mean_peak_entry_probability,
        "mean_normalized_entry_entropy": mean_normalized_entry_entropy,
        "fraction_without_entry_above_0p10": fraction_without_entry_above_0p10,
        "mean_normalized_entry_position": float(np.mean([
            int(np.argmax(values)) / max(len(values) - 1, 1) for values in curves
        ])),
    }
    return curves, locator, diagnostics


def model_to_dict(model: SharedVarianceHMM | None) -> dict | None:
    """JSON-safe representation for experiment diagnostics."""

    if model is None:
        return None
    return {
        "kind": model.kind,
        "means": model.means.tolist(),
        "variance": model.variance,
        "transition": model.transition.tolist(),
        "start": model.start.tolist(),
        "risk_mean": model.risk_mean,
        "risk_scale": model.risk_scale,
        "log_likelihood": model.log_likelihood,
        "occupancy": model.occupancy.tolist(),
        "separation": model.separation,
        "seed": model.seed,
        "n_iter_used": model.n_iter_used,
        "valid": model.valid,
        "guard_failures": list(model.guard_failures),
    }


__all__ = [
    "DEFAULT_SEEDS",
    "LatentStateFit",
    "SharedVarianceHMM",
    "apply_latent_state_fit",
    "fit_upcr_initialized_hmm",
    "forward_backward",
    "model_to_dict",
    "posterior_entry_curve",
    "posterior_high_risk_curve",
]
