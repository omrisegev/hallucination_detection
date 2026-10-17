#!/usr/bin/env python
"""The participation-ratio noise floor — computed, not inferred.

Section 6 of the Stage B pre-registration, and Omri's explicit requirement: the floor
is a separate computation, not an interpretation of the real number. So this re-runs
the ENTIRE Stage B pipeline — shuffle, standardize, fit L-SML on donor folds, score,
build the views — on tokens permuted within each answer independently per channel, and
measures the same two statistics on the result.

What the shuffle destroys and what it keeps: permuting within an answer keeps each
answer's marginal distribution per channel exactly, and destroys token order and the
cross-channel alignment at a given token. So any structure that survives is a property
of the marginals, and any structure that vanishes was carried by order or by
channel-to-channel coupling.

Both statistics are measured, because amendment 1 established they are different
things:

  conditional PR   (sum lambda)^2 / sum(lambda^2) of the within-label-centred view
                   correlation. Comparable to the project's 1.80 / 2.46 / 2.83.
  weight IPR       1 / sum((|w|/sum|w|)^2) over the fitted weights, ceiling 11.
                   Comparable to the 9.69. Sign-blind.

Pre-registered reading, recorded before this ran: if the shuffled null ALSO collapses,
the contrast says nothing about dimensionality and must not be reported as if it did.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ROSTER = ROOT / "results" / "localization_full_benchmark_v3" / "evaluation"
RES = ROOT / "results" / "token_probability_fusion_v1"
OUT = RES / "PR_NOISE_FLOOR.json"
SEED = 20260918

_b = importlib.util.spec_from_file_location("stage_b", Path(__file__).with_name("stage_b_2x2_v1.py"))
SB = importlib.util.module_from_spec(_b)
_b.loader.exec_module(SB)

from spectral_utils.derivative_step_channel_v1 import level_step_readout  # noqa: E402


def main() -> None:
    records = json.load(open(ROSTER / "JOINED.json", encoding="utf8"))["records"]
    with np.load(ROSTER / "JOINED.npz", allow_pickle=False) as z:
        offsets = np.asarray(z["offsets"], int)
        labels = np.asarray(z["labels"], int)
        target = np.asarray(z["target"], int)
    cells = np.asarray([r["cell"] for r in records], str)
    outer = json.load(open(ROSTER / "FOLDS_V2.json", encoding="utf8"))["outer"]
    folds = np.asarray([outer[r["group_id"]] for r in records], int)

    with np.load(RES / "TOKEN_MATRICES.npz", allow_pickle=False) as z:
        tokens = z["tokens"].astype(float)
        tok_off = np.asarray(z["token_offsets"], int)
        step_spans = np.asarray(z["step_spans"], int)

    n = len(records)
    rng = np.random.default_rng(SEED)
    shuffled_tok, shuffled_lvl, spans = [], [], []
    for i in range(n):
        m = tokens[tok_off[i]:tok_off[i + 1]].copy()
        for c in range(m.shape[1]):            # independently per channel
            rng.shuffle(m[:, c])
        s = step_spans[offsets[i]:offsets[i + 1]]
        shuffled_tok.append(m)
        spans.append(s)
        shuffled_lvl.append(level_step_readout(m, s))

    design = {
        "C1_pooled_before": (shuffled_tok, spans),
        "C2_answerlocal_before": ([SB.answer_local(m) for m in shuffled_tok], spans),
        "C3_pooled_after": (shuffled_lvl, None),
        "C4_answerlocal_after": ([SB.answer_local(m) for m in shuffled_lvl], None),
    }

    prm = np.repeat(np.char.startswith(cells, "prmbench_"), np.diff(offsets))
    valid = prm & (labels >= 0)
    y = labels[valid] == 1

    report = {"schema": "token-probability-fusion-v1-pr-noise-floor",
              "shuffle": "tokens permuted within each answer, independently per channel",
              "seed": SEED, "null": {}}
    for name, (mats, sp) in design.items():
        print(f"[null] {name}", flush=True)
        res = SB.run_cell(mats, sp, folds, offsets)
        report["null"][name] = {
            "conditional_participation_ratio":
                SB.conditional_participation_ratio(res["views"], valid, y),
            "weight_ipr_mean": float(np.mean([f["weight_ipr"] for f in res["fits"]])),
            # Reported so the null is legible as a whole: a shuffled pipeline should
            # localize near chance, and if it does not, the shuffle did not work.
            "mean_sla_l_sml": float(np.mean([
                v["sla"] for v in SB.sla_gate_free(
                    SB.peaks_of(res["l_sml"], offsets), target, cells).values()])),
        }

    OUT.write_text(json.dumps(report, indent=1), encoding="utf8")

    real = json.load(open(RES / "STAGE_B_2X2.json", encoding="utf8"))["cells"]
    print()
    print("=" * 84)
    print("PARTICIPATION RATIO vs ITS SEPARATELY COMPUTED NOISE FLOOR")
    print("=" * 84)
    print(f"{'cell':26s} {'cond PR':>9s} {'null':>8s} {'w-IPR':>8s} {'null':>8s} {'null SLA':>9s}")
    for name in design:
        r, nu = real[name], report["null"][name]
        print(f"{name:26s} {r['conditional_participation_ratio']:9.2f} "
              f"{nu['conditional_participation_ratio']:8.2f} "
              f"{r['weight_ipr_mean']:8.2f} {nu['weight_ipr_mean']:8.2f} "
              f"{100*nu['mean_sla_l_sml']:9.2f}")
    print()
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
