"""Decimal80 oracle for the compensated bridge, using synthetic inputs only."""
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
from spectral_utils import error_count_inputs as ec


def main():
    rng = np.random.default_rng(669)
    n = 300
    raw = rng.normal(size=(n, 3))*np.array([1e-10, 1e-7, 1.]) + np.array([1e-3, 1e8, 1e-8])
    active = np.ones(raw.shape, bool)
    donor_mu = np.array([1e8, -1e8, .739])
    donor_r = np.array([1e-10, 1.31e-8, .263])
    pair = ec._absolute_coordinates(raw, active, donor_mu, donor_r)
    worst = 0.
    with localcontext() as context:
        context.prec = 80
        for i in range(n):
            for j in range(3):
                exact = (Decimal.from_float(raw[i, j])-Decimal.from_float(donor_mu[j]))/Decimal.from_float(donor_r[j])
                actual = Decimal.from_float(pair[0][i, j])+Decimal.from_float(pair[1][i, j])
                worst = max(worst, float(abs(actual-exact)/max(Decimal(1), abs(exact))))
    local_mu, local_sd = raw.mean(0), raw.std(0)
    restored = ec._pair_multiply(pair, (donor_r, np.zeros_like(donor_r)))
    numerator = ec._pair_add(restored, ec._two_sum(donor_mu, -local_mu))
    recovered = ec._pair_divide(numerator, (local_sd, np.zeros_like(local_sd)))
    canonical = (raw-local_mu)/local_sd
    inverse_error = float(abs(recovered[0]+recovered[1]-canonical).max())
    naive_inverse = (donor_r*((raw-donor_mu)/donor_r)+donor_mu-local_mu)/local_sd
    naive_error = float(abs(naive_inverse-canonical).max())
    assert worst < 1e-30 and inverse_error < 1e-9 and naive_error > 1.
    groups = [{"columns": [0, 1], "weights": [.71, -.63]}, {"columns": [2], "weights": [.89]}]
    observed = ec._pair_group_scores(pair, groups)
    group_error = 0.
    with localcontext() as context:
        context.prec = 80
        for i in range(n):
            for k, group in enumerate(groups):
                exact = sum(Decimal.from_float(weight)*(Decimal.from_float(raw[i, j])-Decimal.from_float(donor_mu[j]))/Decimal.from_float(donor_r[j])
                            for j, weight in zip(group["columns"], group["weights"]))
                group_error = max(group_error, abs(observed[i, k]-float(exact)))
    assert group_error == 0.
    evidence = {"status": "PASS", "scope": "independent synthetic Decimal80 oracle only; no study labels or study-data fitting",
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "coordinate_max_relative_error": worst, "inverse_V_max_absolute_error": inverse_error,
                "ordinary_float64_inverse_max_absolute_error": naive_error,
                "group_scores_vs_single_rounded_Decimal80_max_absolute_error": group_error,
                "unchanged_absolute_tolerance": 1e-9,
                "definition": "active*(raw-donor_mu)/donor_r with original within-group weights; compensation changes arithmetic precision only",
                "limits": "Finite normal representable arithmetic in tested domain; no generic near-overflow/underflow guarantee",
                "helper_sha256": hashlib.sha256((ROOT/"spectral_utils/error_count_inputs.py").read_bytes()).hexdigest()}
    Path(__file__).with_name("INDEPENDENT_COMPENSATED_BRIDGE_AUDIT.json").write_text(json.dumps(evidence, indent=2, allow_nan=False)+"\n")
    print(json.dumps(evidence, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
