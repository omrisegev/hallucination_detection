#!/usr/bin/env python3
"""Mechanical checks for the cross-scale localization input layer."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_utils.ciw_cross_scale_localization import fit_cross_scale_token_head
from spectral_utils.reconstruction_benchmark.localization_fit import _fit_token_iu


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[str, ...], np.ndarray]:
    rng = np.random.default_rng(20260827)
    n_rows, tokens_per_row, n_features = 50, 30, 29
    latent = rng.normal(size=(n_rows, n_features))
    response_risk = -latent.mean(axis=1) + 0.1 * rng.normal(size=n_rows)
    parts = []
    starts, ends = [], []
    for row in range(n_rows):
        base = row * tokens_per_row
        trend = np.linspace(-0.6, 0.6, tokens_per_row)[:, None]
        parts.append(latent[row] + 0.25 * trend + 0.35 * rng.normal(size=(tokens_per_row, n_features)))
        for step in range(3):
            starts.append(base + 10 * step)
            ends.append(base + 10 * (step + 1))
    return (
        np.vstack(parts),
        np.arange(0, (n_rows + 1) * tokens_per_row, tokens_per_row, dtype=np.int64),
        np.asarray(starts, dtype=np.int64),
        np.asarray(ends, dtype=np.int64),
        tuple(f"row_{index:02d}" for index in range(n_rows)),
        response_risk,
    )


def main() -> None:
    values, offsets, starts, ends, rows, response = fixture()
    baseline_token, _ = _fit_token_iu(SimpleNamespace(token_confidence=values))
    baseline_step = np.asarray([
        np.max(baseline_token[lo:hi]) for lo, hi in zip(starts, ends)
    ])
    alias = fit_cross_scale_token_head(
        values, offsets, starts, ends, rows, response, max_gate=0.0
    )
    if np.max(np.abs(alias.step_risk - baseline_step)) > 1e-12:
        raise AssertionError("zero gate is not a numerical token-IU alias")

    first = fit_cross_scale_token_head(values, offsets, starts, ends, rows, response)
    second = fit_cross_scale_token_head(values, offsets, starts, ends, rows, response)
    if not np.array_equal(first.step_risk, second.step_risk):
        raise AssertionError("cross-scale IU fit is not deterministic")
    if not (np.all(first.gate >= 0.0) and np.all(first.gate <= 0.5)):
        raise AssertionError("cross-scale gate left its frozen bounds")
    if first.step_risk.shape != starts.shape or not np.isfinite(first.step_risk).all():
        raise AssertionError("cross-scale IU output is malformed")

    sparse = fit_cross_scale_token_head(
        values, offsets, starts, ends, rows, response, fusion="su"
    )
    if sparse.diagnostics["fusion"] != "su_pcr_reproduction":
        raise AssertionError("SU-PCR arm did not use the registered sparse solver")
    if sparse.step_risk.shape != starts.shape or not np.isfinite(sparse.step_risk).all():
        raise AssertionError("cross-scale SU output is malformed")
    print("PASS: CIW cross-scale localization mechanics")


if __name__ == "__main__":
    main()
