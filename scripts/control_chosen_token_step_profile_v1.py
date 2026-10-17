"""Step 417 control: is the gain the chosen-token evidence, or only the step-index prior?

The profile-removed view is z - profile[step index]. Control arms add ONLY the prior part,
-profile[step index] (fitted on training folds, answer-standardized), to the same streams.
A second control keeps z but neutralizes step 0 only (set to the answer mean of the other steps
before standardization), which removes the spike without imposing the full profile.
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

OUT = ROOT / "results/chosen_token_calibration_v1"
SEED = 417100
BINS = 9


def main():
    data = L.load_data(); offsets = data["offsets"]; gate = np.asarray(data["gate"], bool)
    n_steps = np.diff(offsets); folds = np.repeat(data["folds"], n_steps)
    step_idx = np.concatenate([np.arange(k) for k in n_steps]); step_bin = np.minimum(step_idx, BINS - 1)
    z = S.astd(step_z_readouts(S.load_folder("extracted_sufficient", offsets, len(SUFFICIENT)))[:, 2], offsets)
    prior = np.empty_like(z); resid = np.empty_like(z)
    for outer in range(5):
        tr = folds != outer; te = ~tr
        prof = np.array([z[tr & (step_bin == b)].mean() for b in range(BINS)])
        prior[te] = -prof[step_bin[te]]; resid[te] = z[te] - prof[step_bin[te]]
    zprof = S.astd(resid, offsets); prior_only = S.astd(prior, offsets)
    z_nostep0 = z.copy()
    for a, b in zip(offsets[:-1], offsets[1:]):
        if b - a > 1:
            z_nostep0[a] = z[a + 1:b].mean()
    z_nostep0 = S.astd(z_nostep0, offsets)

    x20 = L.build_matrix(data)
    six = x20[:, [list(L.ROSTER20).index(n) for n in ("H0lim", "ve0", "ve0.75", "ve1", "H0lim_prefix_innovation", "bocpd_residual")]]
    oof = {"six": six.mean(1),
           "six+view_profile_removed": np.column_stack([six, zprof]).mean(1),
           "six+prior_only": np.column_stack([six, prior_only]).mean(1),
           "six+view_step0_neutral": np.column_stack([six, z_nostep0]).mean(1),
           "B20": x20.mean(1),
           "B20+view_profile_removed": np.column_stack([x20, zprof]).mean(1),
           "B20+prior_only": np.column_stack([x20, prior_only]).mean(1),
           "prior_only_alone": prior_only, "view_step0_neutral_alone": z_nostep0}
    results = {a: score_locator(s, gate, data) for a, s in oof.items()}
    metrics = {a: {"pb": r["pb"], "within": r["within"]} for a, r in results.items()}
    arms, pbd, wd = L.bootstrap(data, results, 10000, SEED)
    ix = {a: i for i, a in enumerate(arms)}
    contrasts = {}
    for a, b in (("six+view_profile_removed", "six+prior_only"), ("six+prior_only", "six"),
                 ("six+view_step0_neutral", "six"), ("six+view_profile_removed", "six+view_step0_neutral"),
                 ("B20+view_profile_removed", "B20+prior_only"), ("B20+prior_only", "B20")):
        contrasts[f"{a} - {b}"] = {"pb": L.interval(pbd[:, ix[a]] - pbd[:, ix[b]], metrics[a]["pb"] - metrics[b]["pb"]),
                                   "within": L.interval(wd[:, ix[a]] - wd[:, ix[b]], metrics[a]["within"] - metrics[b]["within"])}
    (OUT / "STEP_PROFILE_CONTROL_V1.json").write_text(json.dumps(L.clean({"metrics": metrics, "contrasts": contrasts}), indent=1, sort_keys=True) + "\n")
    for a in sorted(metrics, key=lambda k: -metrics[k]["pb"]):
        print(f"{a:28s} PB {100 * metrics[a]['pb']:.2f}  within {metrics[a]['within']:.4f}")
    for k, v in contrasts.items():
        p, w = v["pb"], v["within"]
        flag = lambda i: "*" if i["low"] > 0 or i["high"] < 0 else " "  # noqa: E731
        print(f"{k:52s} PB {100*p['point']:+.2f} [{100*p['low']:+.2f},{100*p['high']:+.2f}]{flag(p)} "
              f"within {w['point']:+.4f} [{w['low']:+.4f},{w['high']:+.4f}]{flag(w)}")


if __name__ == "__main__":
    main()
