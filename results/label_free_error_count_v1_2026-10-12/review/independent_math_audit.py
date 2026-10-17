"""Independent synthetic arithmetic audit; no study-data fitting or label access."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.stats import rankdata
from spectral_utils import error_count as ec
from spectral_utils.error_count_evaluation import SourceBootstrap


def independent_partial(x, y, z):
    x, y, z = (rankdata(a, method="average") for a in (x, y, z))
    design = np.column_stack((np.ones(len(z)), z))
    rx = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    ry = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    if np.linalg.norm(rx) < 1e-10 or np.linalg.norm(ry) < 1e-10:
        return np.nan
    return float(rx @ ry / np.sqrt((rx @ rx) * (ry @ ry)))


def main():
    evidence = {"status": "PASS", "scope": "synthetic enumeration and arithmetic only; no real study data accessed",
                "reviewer": "independent_review", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    psi, eta, pi = np.array([.83, .74, .91]), np.array([.92, .82, .88]), .17
    votes = np.array(list(itertools.product((-1., 1.), repeat=3)))
    mass = pi*np.where(votes > 0, psi, 1-psi).prod(1) + (1-pi)*np.where(votes > 0, 1-eta, eta).prod(1)
    means = mass @ votes
    raw_pairs = mass @ (votes[:, [0, 0, 1]] * votes[:, [1, 2, 2]])
    raw_third = mass @ votes.prod(1)
    triple = ec._triple_batch(means, raw_pairs, raw_third)
    errors = {"prevalence": abs(float(triple["prevalence"][0])-pi),
              "sensitivity": float(abs(triple["psi_raw"][0]-psi).max()),
              "specificity": float(abs(triple["eta_raw"][0]-eta).max())}
    assert triple["valid"][0] and max(errors.values()) < 1e-12
    evidence["closed_triple_exact_latent_distribution_max_abs_errors"] = errors

    model = dict(status="PASS", n_vote_columns=3, columns=[0, 1, 2], psi=psi, eta=eta, prevalence=pi)
    observed = votes[[1, 7, 0, 4, 2]]
    offsets = np.array([0, 2, 5])
    log0, log1 = ec._class_loglik(observed, model)
    objective = ec._zero_objective(log0, log1, offsets)
    max_ll, max_gradient, max_pmf = 0., 0., 0.
    for rho, theta in [(0., .3), (.4, .27), (.91, .63)]:
        nll, gradient = objective((rho, theta))
        independent_ll, expected_pmfs = [], []
        for a, b in zip(offsets[:-1], offsets[1:]):
            joint = np.zeros(b-a+1)
            clean = rho*np.exp(log0[a:b].sum())
            for labels in itertools.product((0, 1), repeat=b-a):
                mask = np.asarray(labels, bool)
                likelihood = np.exp(np.where(mask, log1[a:b], log0[a:b]).sum())
                prior = np.where(mask, theta, 1-theta).prod()
                joint[sum(labels)] += (1-rho)*likelihood*prior
            joint[0] += clean
            independent_ll.append(np.log(joint.sum()))
            expected_pmfs.append(joint / joint.sum())
        max_ll = max(max_ll, abs(nll+np.mean(independent_ll)))
        mixture = dict(model, rho=rho, theta=theta)
        output = ec.predict_zero_mixture(observed, offsets, mixture)
        for i, expected in enumerate(expected_pmfs):
            left, right = output["pmf_offsets"][i:i+2]
            max_pmf = max(max_pmf, float(abs(np.asarray(output["pmf"])[left:right]-expected).max()))
        if rho > 0:
            center, h = np.array([rho, theta]), 1e-6
            numerical = np.array([(objective(center+np.eye(2)[j]*h)[0]-objective(center-np.eye(2)[j]*h)[0])/(2*h)
                                  for j in range(2)])
            max_gradient = max(max_gradient, float(abs(numerical-gradient).max()))
    assert max_ll < 1e-12 and max_gradient < 1e-7 and max_pmf < 1e-12
    evidence["zero_mixture_vs_independent_latent_enumeration"] = dict(loglik_max_abs=max_ll, gradient_max_abs=max_gradient,
                                                                      posterior_count_pmf_max_abs=max_pmf)

    probabilities = np.array([0., 1., .17, .38, .71, .92])
    independent_pmf = np.zeros(len(probabilities)+1)
    for labels in itertools.product((0, 1), repeat=len(probabilities)):
        independent_pmf[sum(labels)] += np.where(np.asarray(labels, bool), probabilities, 1-probabilities).prod()
    output = ec.count_distribution(probabilities, [0, len(probabilities)])
    pmf_error = float(abs(np.asarray(output["pmf"])-independent_pmf).max())
    assert pmf_error < 1e-12
    for k in range(len(probabilities)+1):
        assert np.abs(np.arange(len(probabilities)+1)-output["median_lower"][0]) @ independent_pmf <= np.abs(np.arange(len(probabilities)+1)-k) @ independent_pmf + 1e-12
    tie = ec.count_distribution([.5], [0, 1])
    assert tie["mean_half_up"] == [1] and tie["median_lower"] == [0]
    evidence["poisson_binomial_vs_independent_enumeration"] = dict(pmf_max_abs=pmf_error, lower_median_absolute_loss_minimizer=True,
                                                                 half_up_and_lower_median_tie_distinct=True)

    rng = np.random.default_rng(847)
    groups = np.repeat(np.arange(5), 4)
    multiplicity = np.array([[1, 1, 1, 1, 1], [2, 0, 1, 0, 2], [0, 1, 2, 1, 1]])
    estimate, truth, length = rng.integers(0, 6, 20), rng.integers(0, 6, 20), rng.integers(3, 20, 20)
    bootstrap = SourceBootstrap(groups, weights=multiplicity)
    observed = bootstrap.correlation(estimate, truth, length, np.ones(20, bool))
    expected = []
    for draw in multiplicity:
        repeated = draw[bootstrap.inverse]
        expected.append(independent_partial(np.repeat(estimate, repeated), np.repeat(truth, repeated), np.repeat(length, repeated)))
    partial_error = float(abs(observed-expected).max())
    assert partial_error < 1e-12
    evidence["whole_source_partial_spearman_vs_explicit_repeat_and_OLS_max_abs"] = partial_error
    paths = ["spectral_utils/error_count.py", "spectral_utils/error_count_inputs.py", "spectral_utils/error_count_evaluation.py",
             "scripts/run_label_free_error_count_v1.py"]
    evidence["reviewed_code_sha256"] = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
    destination = Path(__file__).with_name("INDEPENDENT_MATH_AUDIT.json")
    destination.write_text(json.dumps(evidence, indent=2, allow_nan=False)+"\n")
    print(json.dumps(evidence, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
