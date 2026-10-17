"""Step 417 confirmation: remove the step-index profile of the chosen-token z-test.

Diagnosis (no labels): the answer-standardized pooled z-test is +2.08 SD at step 0 and -0.06 to -0.41
at every later step, so the view peaks at step 0 in 96% of ProcessBench error answers. A linear
position trend cannot remove a one-step spike.

One variant, label-free: subtract the mean value at each absolute step index (0..7, 8+), fitted on the
four training folds only, then re-standardize within the answer. Same Step 413 contract and bootstrap.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.run_digitfree20_ladder_v1 as L  # noqa: E402
import scripts.analyze_chosen_token_step_tests_v2 as S  # noqa: E402
from scripts.run_lsml_gate_locator_research_v1 import score_locator  # noqa: E402
from spectral_utils.chosen_token_calibration import SUFFICIENT, step_z_readouts  # noqa: E402
from spectral_utils.lsml_gate_locator_research import FusionRecipe, fit_fusion_weights  # noqa: E402

OUT = ROOT / "results/chosen_token_calibration_v1"
SEED = 417000
BINS = 9  # absolute step index 0..7 and 8+


def main():
    data = L.load_data(); offsets = data["offsets"]; cells = data["cells"].astype(str)
    target = np.asarray(data["target"]); gate = np.asarray(data["gate"], bool)
    n_steps = np.diff(offsets); folds = np.repeat(data["folds"], n_steps)
    step_bin = np.minimum(np.concatenate([np.arange(k) for k in n_steps]), BINS - 1)
    suff = S.load_folder("extracted_sufficient", offsets, len(SUFFICIENT))
    z = S.astd(step_z_readouts(suff)[:, 2], offsets)

    resid = np.empty_like(z); profiles = []
    for outer in range(5):
        tr = folds != outer; te = ~tr
        prof = np.array([z[tr & (step_bin == b)].mean() for b in range(BINS)])
        profiles.append(prof.tolist())
        resid[te] = z[te] - prof[step_bin[te]]
    zprof = S.astd(resid, offsets)

    x20 = L.build_matrix(data)
    six = x20[:, [list(L.ROSTER20).index(n) for n in ("H0lim", "ve0", "ve0.75", "ve1", "H0lim_prefix_innovation", "bocpd_residual")]]
    mats = {"six": six, "six+z_prof": np.column_stack([six, zprof]),
            "B20": x20, "B20+z_prof": np.column_stack([x20, zprof])}
    anchors = {"six": 0, "six+z_prof": 0, "B20": L.ANCHOR, "B20+z_prof": L.ANCHOR}
    oof = {}
    for bank, M in mats.items():
        oof[f"{bank}_equal"] = M.mean(axis=1)
        if bank.endswith("z_prof"):
            s = np.empty(len(M))
            for outer in range(5):
                te = folds == outer
                w, _ = fit_fusion_weights(M[~te], FusionRecipe(bank, tuple(f"s{j}" for j in range(M.shape[1])),
                                                               "continuous", anchor=anchors[bank]), seed=SEED)
                s[te] = M[te] @ w
            oof[f"{bank}_lsml"] = s
    oof["z_prof_alone"] = zprof
    results = {a: score_locator(s, gate, data) for a, s in oof.items()}
    metrics = {a: {"pb": r["pb"], "within": r["within"]} for a, r in results.items()}

    pb = np.char.startswith(cells, "pb_"); err = pb & (target >= 0)
    peaks_alone = np.asarray(results["z_prof_alone"]["peaks"])
    peak_six = np.asarray(results["six_equal"]["peaks"]); peak_new = np.asarray(results["six+z_prof_equal"]["peaks"])
    m = err & gate
    diag = {"view_alone_peak_at_step0_pb_errors": float(np.mean(peaks_alone[err] == 0)),
            "profile_by_step_index_fold0": profiles[0],
            "gate_open_errors": int(m.sum()),
            "exact_six": int((peak_six[m] == target[m]).sum()), "exact_six_plus_view": int((peak_new[m] == target[m]).sum()),
            "moved_earlier": int((peak_new[m] < peak_six[m]).sum()), "moved_later": int((peak_new[m] > peak_six[m]).sum())}

    arms, pbd, wd = L.bootstrap(data, results, 10000, SEED + 1)
    ix = {a: i for i, a in enumerate(arms)}
    contrasts = {}
    for a, b in (("six+z_prof_equal", "six_equal"), ("B20+z_prof_equal", "B20_equal"),
                 ("six+z_prof_lsml", "six+z_prof_equal"), ("B20+z_prof_lsml", "B20+z_prof_equal")):
        contrasts[f"{a} - {b}"] = {"pb": L.interval(pbd[:, ix[a]] - pbd[:, ix[b]], metrics[a]["pb"] - metrics[b]["pb"]),
                                   "within": L.interval(wd[:, ix[a]] - wd[:, ix[b]], metrics[a]["within"] - metrics[b]["within"])}
    report = {"diagnostics": diag, "metrics": metrics, "contrasts": contrasts}
    (OUT / "STEP_PROFILE_V1.json").write_text(json.dumps(L.clean(report), indent=1, sort_keys=True) + "\n")
    print(json.dumps(L.clean(diag), indent=1))
    for a in sorted(metrics, key=lambda k: -metrics[k]["pb"]):
        print(f"{a:22s} PB {100 * metrics[a]['pb']:.2f}  within {metrics[a]['within']:.4f}")
    for k, v in contrasts.items():
        p, w = v["pb"], v["within"]
        flag = lambda i: "*" if i["low"] > 0 or i["high"] < 0 else " "  # noqa: E731
        print(f"{k:40s} PB {100*p['point']:+.2f} [{100*p['low']:+.2f},{100*p['high']:+.2f}]{flag(p)} "
              f"within {w['point']:+.4f} [{w['low']:+.4f},{w['high']:+.4f}]{flag(w)}")


if __name__ == "__main__":
    main()
