"""Step 417 diagnostic: why does the entropy-free view help PRMB ranking but not the PB peak?

Compares the equal-weight six-stream leader with the same six plus the pooled chosen-token z-test
(with and without label-free position removal). Equal weights need no fitting, so the scores are
exact replays of the Step 416 arms. Labels are used for diagnosis only.

The frozen gate does not depend on the locator, so clean answers are scored identically by every
arm; every ProcessBench difference comes from error answers whose gate is open.
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


def main():
    data = L.load_data(); offsets = data["offsets"]; cells = data["cells"].astype(str)
    target = np.asarray(data["target"]); gate = np.asarray(data["gate"], bool)
    n_steps = np.diff(offsets); folds = np.repeat(data["folds"], n_steps)
    suff = S.load_folder("extracted_sufficient", offsets, len(SUFFICIENT))
    z = S.astd(step_z_readouts(suff)[:, 2], offsets)
    rel_pos = np.concatenate([np.arange(n) / max(n - 1, 1) for n in n_steps])
    pos_std = S.astd(rel_pos, offsets)
    resid = np.empty_like(z)
    for outer in range(5):
        tr = folds != outer; te = ~tr
        resid[te] = z[te] - float(np.dot(z[tr], pos_std[tr]) / np.dot(pos_std[tr], pos_std[tr])) * pos_std[te]
    zpos = S.astd(resid, offsets)
    x20 = L.build_matrix(data)
    six = x20[:, [list(L.ROSTER20).index(n) for n in ("H0lim", "ve0", "ve0.75", "ve1", "H0lim_prefix_innovation", "bocpd_residual")]]
    arms = {"six": six.mean(axis=1),
            "six+z_pooled": np.column_stack([six, z]).mean(axis=1),
            "six+z_pooled_pos": np.column_stack([six, zpos]).mean(axis=1),
            "z_pooled_pos_alone": zpos}
    res = {a: score_locator(s, gate, data) for a, s in arms.items()}
    print({a: round(100 * r["pb"], 2) for a, r in res.items()}, "(replay of Step 416 PB)")

    pb = np.char.startswith(cells, "pb_")
    err = pb & (target >= 0); clean = pb & (target < 0)
    for a in arms:  # clean-answer CORRECTNESS is identical across arms (the predicted index may differ)
        ok = res[a]["prediction"][clean] == target[clean]
        assert np.array_equal(ok, res["six"]["prediction"][clean] == target[clean])
    peaks = {a: np.asarray(r["peaks"]) for a, r in res.items()}
    rel_peak = {a: np.array([p / max(n - 1, 1) for p, n in zip(peaks[a], n_steps)]) for a in arms}
    report = {}
    for gate_state, gmask in (("gate_open", gate), ("gate_closed", ~gate)):
        m = err & gmask
        block = {"error_answers": int(m.sum())}
        for a in arms:
            p = peaks[a][m]; t = target[m]
            block[a] = {"exact": int((p == t).sum()), "early": int((p < t).sum()), "late": int((p > t).sum()),
                        "mean_signed_offset_steps": float((p - t).mean()),
                        "mean_relative_peak_position": float(rel_peak[a][m].mean())}
        report[gate_state] = block
    m = err & gate
    t = target[m]; base = peaks["six"][m]
    for a in ("six+z_pooled", "six+z_pooled_pos"):
        p = peaks[a][m]
        moved = p != base
        gained = (p == t) & (base != t); lost = (p != t) & (base == t)
        report[f"transitions_{a}"] = {
            "moved": int(moved.sum()), "moved_earlier": int((p < base).sum()), "moved_later": int((p > base).sum()),
            "gained": int(gained.sum()), "lost": int(lost.sum()),
            "lost_moved_earlier": int((lost & (p < base)).sum()), "lost_moved_later": int((lost & (p > base)).sum()),
            "gained_moved_earlier": int((gained & (p < base)).sum()), "gained_moved_later": int((gained & (p > base)).sum()),
            "true_error_relative_position_of_lost": float(np.mean(t[lost] / np.maximum(n_steps[m][lost] - 1, 1))) if lost.any() else None,
            "true_error_relative_position_of_gained": float(np.mean(t[gained] / np.maximum(n_steps[m][gained] - 1, 1))) if gained.any() else None,
            "true_error_relative_position_all": float(np.mean(t / np.maximum(n_steps[m] - 1, 1))),
        }
    # where does the view alone put its peak, relative to the truth, on the same open-gate errors
    report["per_cell_gate_open_exact"] = {}
    for c in sorted(set(cells[pb])):
        mc = err & gate & (cells == c)
        report["per_cell_gate_open_exact"][c] = {a: int((peaks[a][mc] == target[mc]).sum()) for a in ("six", "six+z_pooled", "six+z_pooled_pos")} | {"n": int(mc.sum())}
    (OUT / "PB_SPLIT_DIAGNOSTIC.json").write_text(json.dumps(L.clean(report), indent=1, sort_keys=True) + "\n")
    print(json.dumps(L.clean(report), indent=1))


if __name__ == "__main__":
    main()
