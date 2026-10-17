"""Freeze candidate CT7: replay its development scores exactly and write an immutable manifest.

Refuses to overwrite an existing manifest whose recipe or code hashes differ.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.run_digitfree20_ladder_v1 as L  # noqa: E402
import scripts.analyze_chosen_token_step_tests_v2 as S  # noqa: E402
from scripts.run_lsml_gate_locator_research_v1 import score_locator  # noqa: E402
from spectral_utils import frozen_locator_ct7 as C  # noqa: E402
from spectral_utils.chosen_token_calibration import SUFFICIENT  # noqa: E402

OUT = ROOT / "results/chosen_token_calibration_v1"
TEMPORAL = ROOT.parents[1] / ".worktrees/temporal-research-20260915"
EXPECTED = {"pb": 0.4119, "within": 0.7724}  # Step 417 six + step-0-neutral view, equal weight (rounded)


def sha_file(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    data = L.load_data(); offsets = data["offsets"]; gate = np.asarray(data["gate"], bool)
    with np.load(ROOT / "results/joint_feature_selection_bocpd_v1/INPUTS.npz") as z:
        bocpd = z["bocpd"]
    suff = S.load_folder("extracted_sufficient", offsets, len(SUFFICIENT))
    scores = C.candidate_step_scores(data["x"], bocpd, suff, offsets)
    r = score_locator(scores, gate, data)
    assert abs(r["pb"] - EXPECTED["pb"]) < 6e-5 and abs(r["within"] - EXPECTED["within"]) < 6e-5, (r["pb"], r["within"])
    score_path = OUT / "CT7_DEV_SCORES.npz"
    if score_path.exists():  # never rewrite frozen scores; require an exact replay instead
        with np.load(score_path) as z:
            if not np.array_equal(z["step_scores"], scores):
                raise ValueError("frozen development scores do not replay exactly")
    else:
        np.savez_compressed(score_path, step_scores=scores, gate=gate)

    code = [ROOT / "spectral_utils/frozen_locator_ct7.py", ROOT / "spectral_utils/chosen_token_calibration.py",
            ROOT / "spectral_utils/digitfree_broad50.py", ROOT / "spectral_utils/renyi_alpha_sweep.py",
            ROOT / "scripts/run_digitfree_broad50_v1.py", ROOT / "scripts/run_chosen_token_calibration_steps_v2.py",
            ROOT / "scripts/run_joint_feature_selection_bocpd_v1.py", ROOT / "scripts/run_lsml_gate_locator_research_v1.py",
            TEMPORAL / "spectral_utils/aligned_context_predictors.py", TEMPORAL / "scripts/run_aligned_context_predictors.py"]
    inputs = [ROOT / "results/joint_feature_selection_bocpd_v1/INPUTS.npz",
              TEMPORAL / "results/aligned_context_predictors_v1/SCORES_FROZEN.npz",
              TEMPORAL / "results/temporal_research_baseline_v1/SCORES_FROZEN.npz",
              ROOT / "results/fusion_independence_atlas_v1/dependence/EVALUATION.npz",
              ROOT / "results/fusion_independence_atlas_v1/baseline_replay/SCORES_FROZEN.npz"]
    manifest = {
        "candidate_id": C.CANDIDATE_ID,
        "frozen": "2026-09-17",
        "status": "FROZEN_DEVELOPMENT_CANDIDATE_NOT_CONFIRMED",
        "recipe": {
            "views": list(C.SIX) + ["bocpd_residual", "chosen_token_z_despiked"],
            "fusion": "equal weight 1/7, no fitted parameter, no sign anchor",
            "step_readout": "Top10 step mean for the five bank streams; pooled z-test for the chosen-token view",
            "chosen_token_z": "sum(-log q(x) - H(q)) / sqrt(sum VE(q) + n*0.01) on renormalized top-50 q; tokens outside top-50 censored below the smallest listed mass; answer-standardized; step 0 set to the mean of the answer's other steps; re-standardized",
            "bocpd_residual": "answer-standardized (historical_bocpd - innovation5) from aligned_context_predictors_v1 SCORES_FROZEN.npz: mean of five signed residuals against reset-before-observation Gaussian BOCPD, hazard 1/32, observation variance 1, prior mean 0 / variance 1 (bocpd_mean in codex/temporal-research-20260915 @ ff3b02852)",
            "locator": "argmax step within the answer",
            "no_error_decision": "frozen non-digit tail15 whole-answer Top10-mean prominence, within-cell midrank >= .33 (not part of the locator)",
            "access": "gray-box: one teacher-forced forward pass, top-50 log-probabilities and provided-token surprisal; offline whole-answer processing; no labels; no other answers",
        },
        "development_evidence": {
            "population": "13,769 answers (8 ProcessBench cells + PRMBench qwen3-8b), v3 labels, v2 source groups",
            "pb_macro_f1": r["pb"], "prmb_within_auc": r["within"], "prmb_within_n": r["within_n"],
            "comparators_same_gate": {"six_streams_equal": [0.4027, 0.7589],
                                      "frozen_bocpd_corrected_innovation5": [0.4037, 0.7632],
                                      "equal20": [0.3919, 0.7558]},
            "paired_contrasts_10000_source_group_draws": {
                "candidate_profile_variant_minus_six": {"pb_pp": [0.97, 0.06, 1.88], "within": [0.0146, 0.0128, 0.0163]},
                "step0_neutral_candidate_minus_six": {"pb_pp": [0.92, -0.00, 1.86], "within": [0.0135, 0.0118, 0.0153]},
                "token_view_over_position_prior_only": {"pb_pp": [1.59, 0.67, 2.56], "within": [0.0061, 0.0043, 0.0078]}},
            "caveats": [
                "Fifth readout of the same statistic evaluated on these answers on 2026-09-17 (Top10, pooled z, position-removed, profile-removed, step-0-neutral).",
                "The step-0 neutralization was motivated by a label-using peak diagnostic.",
                "The five entropy streams and the BOCPD residual were selected on this population in earlier work.",
                "The gate was historically developed on these data.",
                "The step-0-neutral variant (frozen) has no fitted parameter; its PB interval touches zero; the profile-removal variant's does not.",
                "Continuous L-SML and IU-PCR on these views are not better than equal weight; this is not a fusion-weighting result.",
                "Effective conditional independent views in this pool: 1.80 of 7 (participation ratio).",
            ],
        },
        "confirmation_requirements": [
            "Score the recipe unchanged on answers never used in development; no refit, no re-selection, no readout change.",
            "Primary endpoints: ProcessBench-style macro-F1 with the unchanged gate rule and PRMBench-style within-answer AUROC, reported separately.",
            "Mandatory comparators on the same answers: the six streams at equal weight, BOCPD-corrected innovation5, equal20, and the position-prior-only control.",
            "Paired source-group bootstrap; a claim requires both endpoints to favour the candidate over the six streams.",
            "A new scorer on previously seen ProcessBench answers is not untouched confirmation (CLAUDE.md).",
        ],
        "code_sha256": {p.as_posix().split("hallucination_detection/")[-1]: sha_file(p) for p in code},
        "input_sha256": {p.as_posix().split("hallucination_detection/")[-1]: sha_file(p) for p in inputs},
        "dev_scores_sha256": sha_file(OUT / "CT7_DEV_SCORES.npz"),
        "dev_scores_steps": int(len(scores)),
    }
    path = OUT / "FROZEN_CANDIDATE_CT7.json"
    canonical = json.loads(json.dumps(manifest, sort_keys=True))
    if path.exists():
        old = json.loads(path.read_text())
        for key in ("candidate_id", "recipe", "code_sha256", "input_sha256"):
            if old[key] != canonical[key]:
                raise ValueError(f"frozen candidate drift in {key}; issue a new candidate id instead")
        print("manifest already frozen and consistent"); return
    path.write_text(json.dumps(canonical, indent=1, sort_keys=True) + "\n")
    print(f"FROZEN {C.CANDIDATE_ID}: PB {100 * r['pb']:.2f} within {r['within']:.4f}, {time.strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()
