#!/usr/bin/env python
"""Item 3 (2026-09-23): continuous L-SML on CT7's seven streams at TOKEN level, fitted BEFORE the
Top10 readout, versus equal weight on the same streams and versus CT7 (Top10 per view, then equal).

Protocol: docs/experiments/CT7_TOKEN_LSML_V1.md (written before any number).
Input: CT7_TOKEN_MATRICES.npz from scripts/diagnostics/extract_ct7_token_streams_v1.py.
Reuses the Stage B machinery (`spectral_utils/claude_feature_bank_v1`: donor standardizer, L-SML
weights with the positive-sum gauge, token fusion) with two declared additions: invalid tokens
never enter a fit, and CT7's step-0 rule is applied to the chosen-token column at token level.

    python -B scripts/experiments/ct7_token_lsml_v1.py --config configs/ct7_token_lsml_v1.json
    python -B scripts/experiments/ct7_token_lsml_v1.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ct7_levers_common as L  # noqa: E402

L.ensure_spectral_package()
from spectral_utils.claude_feature_bank_v1 import fit_l_sml_weights, fit_token_standardizer, fuse_token_matrix  # noqa: E402
from spectral_utils.ct7_token_streams import (  # noqa: E402
    STREAMS, answer_streams, despike_step0, masked_answer_local, masked_step_top10, step0_token_mask, synthetic_row,
)
from spectral_utils.digitfree_broad50 import masked_answer_standardize  # noqa: E402

SCHEMA = "ct7-token-lsml-v1"
TOKEN_CAP = 60_000
CHOSEN = 6
SIX = list(range(6))
ALL = list(range(7))


def weight_ipr(w):
    w = np.abs(np.asarray(w, float)); return float(1.0 / np.sum((w / w.sum()) ** 2))


def conditional_pr(views, y):
    m = np.array(views, float, copy=True)
    for cls in (True, False):
        sel = y == cls
        if sel.any():
            m[sel] -= m[sel].mean(0)
    keep = m.std(0) > 1e-12
    if keep.sum() < 2:
        return float(keep.sum())
    lam = np.maximum(np.linalg.eigvalsh(np.corrcoef(m[:, keep].T)), 0.0)
    return float(lam.sum() ** 2 / (lam ** 2).sum())


class TokenData:
    def __init__(self, npz, d):
        z = np.load(npz)
        self.tokens = z["tokens"].astype(float); self.valid = np.asarray(z["valid"], bool)
        self.toff = np.asarray(z["token_offsets"], int); self.spans = np.asarray(z["step_spans"], int)
        self.step0 = np.asarray(z["step0_token_mask"], bool)
        assert list(z["channels"]) == list(STREAMS) and self.tokens.shape == (self.toff[-1], 7)
        assert self.spans.shape == (int(d.off[-1]), 2) and np.array_equal(np.diff(self.toff), [r["tokens"] for r in d.records])
        self.d = d
        self.mats = [self.tokens[a:b] for a, b in zip(self.toff[:-1], self.toff[1:])]
        self.vals = [self.valid[a:b] for a, b in zip(self.toff[:-1], self.toff[1:])]
        self.sp = [self.spans[a:b] for a, b in zip(d.off[:-1], d.off[1:])]
        for i in (0, d.n // 2, d.n - 1):
            s = self.sp[i]; assert s[0, 0] == 0 and s[-1, 1] == len(self.mats[i])


def prepared_matrices(td: TokenData, *, despike: bool, cols):
    """Per answer: (matrix restricted to cols, validity, all-valid row mask), with the token-level
    step-0 rule applied to the chosen-token column when requested."""
    out = []
    for x, v, sp in zip(td.mats, td.vals, td.sp):
        xx = despike_step0(x, v, sp, [CHOSEN]) if despike else x
        vv = v[:, cols]; out.append((xx[:, cols], vv, vv.all(1)))
    return out


def run_arm(td: TokenData, prep, *, axis: str, fusion: str, exclude_step0_fit: bool = False):
    """Fit per outer fold on donors (valid rows only), score held-out answers, masked Top10, answer-z."""
    d = td.d; total = int(d.off[-1]); step = np.full(total, np.nan); fits = []
    mats = [masked_answer_local(x, v) if axis == "answer" else x for x, v, _ in prep]
    for fold in range(5):
        train = np.flatnonzero(d.fold != fold); test = np.flatnonzero(d.fold == fold)
        def fit_rows(i):
            m = prep[i][2]
            if exclude_step0_fit:
                m = m & ~td.step0[td.toff[i]:td.toff[i + 1]]
            return mats[i][m]
        std = fit_token_standardizer((fit_rows(i) for i in train), cap=TOKEN_CAP)
        if fusion == "lsml":
            w, meta = fit_l_sml_weights((fit_rows(i) for i in train), std)
            fits.append({"fold": fold, "weights": w.tolist(), "weight_ipr": weight_ipr(w), "K": int(meta["K"]),
                         "groups": np.asarray(meta["c"]).tolist(), "negative_weights": [STREAMS[j] for j in np.flatnonzero(w < 0) if j < len(STREAMS)],
                         "small_m_flags": meta.get("small_m_flags"), "train_answers": int(len(train))})
        else:
            w = None; fits.append({"fold": fold, "weights": None, "train_answers": int(len(train))})
        for i in test:
            fl, fe = fuse_token_matrix(mats[i], std, weights=w)
            fused = fl if fusion == "lsml" else fe
            a, b = d.off[i], d.off[i + 1]
            step[a:b] = masked_step_top10(fused, prep[i][2], td.sp[i])
    assert np.isfinite(step[np.repeat(True, total)]).sum() > 0
    return L.answer_z(step, d.off), fits


def ct7_style_arm(td: TokenData, prep):
    """Top10 per stream over valid tokens, answer-standardized, equal mean (CT7's architecture on
    these token streams; the seventh view differs from CT7's pooled z-test by construction)."""
    d = td.d; total = int(d.off[-1]); views = np.full((total, 7), np.nan)
    for i, (x, v, _) in enumerate(prep):
        a, b = d.off[i], d.off[i + 1]
        for j in range(7):
            views[a:b, j] = masked_step_top10(x[:, j], v[:, j], td.sp[i])
    z = masked_answer_standardize(views, np.isfinite(views), d.off)
    return z.mean(1), z


def build_methods(td: TokenData):
    d = td.d; methods, fits, extras = {}, {}, {}
    if "ct7" in d.references:
        methods["ct7"] = L.method_from_scores(d, d.references["ct7"])
    prep7 = prepared_matrices(td, despike=True, cols=ALL)
    prep7_raw = prepared_matrices(td, despike=False, cols=ALL)
    prep6 = prepared_matrices(td, despike=True, cols=SIX)
    s, z7 = ct7_style_arm(td, prep7); methods["ct7_top10_equal7"] = L.method_from_scores(d, L.answer_z(s, d.off))
    s, _ = ct7_style_arm(td, prep7_raw); methods["ct7_top10_equal7_nodespike"] = L.method_from_scores(d, L.answer_z(s, d.off))
    arms = [("T_E1", prep7, "pooled", "equal", False), ("T_C1", prep7, "pooled", "lsml", False),
            ("T_E2", prep7, "answer", "equal", False), ("T_C2", prep7, "answer", "lsml", False),
            ("T_E1_six", prep6, "pooled", "equal", False), ("T_C1_six", prep6, "pooled", "lsml", False),
            ("T_E2_six", prep6, "answer", "equal", False), ("T_C2_six", prep6, "answer", "lsml", False),
            ("T_C2_nostep0fit", prep7, "answer", "lsml", True)]
    for name, prep, axis, fusion, ex in arms:
        started = time.perf_counter()
        s, f = run_arm(td, prep, axis=axis, fusion=fusion, exclude_step0_fit=ex)
        methods[name] = L.method_from_scores(d, s); fits[name] = {"fits": f, "seconds": time.perf_counter() - started}
    # conditional participation ratio of the seven Top10 step views on labelled PRMB steps
    prm_steps = np.repeat(d.prm, np.diff(d.off)); ok = prm_steps & np.isfinite(z7).all(1)
    extras["conditional_participation_ratio_top10_views"] = conditional_pr(z7[ok], d.labels[ok] == 1)
    return methods, fits, extras


def planned(names):
    pairs = []
    def add(a, b, why):
        if a in names and b in names and (a, b, why) not in pairs:
            pairs.append((a, b, why))
    add("T_C2", "T_E2", "PRIMARY_lsml_minus_equal_answer_local"); add("T_C1", "T_E1", "lsml_minus_equal_pooled")
    add("T_C2_six", "T_E2_six", "lsml_minus_equal_answer_local_six"); add("T_C1_six", "T_E1_six", "lsml_minus_equal_pooled_six")
    for a in ("T_C1", "T_C2", "T_E1", "T_E2", "ct7_top10_equal7"):
        add(a, "ct7", "arm_minus_ct7")
    add("T_E2", "ct7_top10_equal7", "before_minus_after_readout_equal"); add("T_C2", "ct7_top10_equal7", "before_lsml_minus_after_equal")
    add("T_C1", "T_C1_six", "seventh_stream_pooled"); add("T_C2", "T_C2_six", "seventh_stream_answer_local")
    add("T_C2", "T_C1", "answer_local_minus_pooled"); add("T_C2_nostep0fit", "T_C2", "step0_excluded_from_fit")
    add("ct7_top10_equal7", "ct7_top10_equal7_nodespike", "despike_minus_raw")
    return pairs


def synthetic_token_data(tmp: Path):
    d, _ = L.synthetic_dataset(tmp, n_answers=220)
    rng = np.random.default_rng(3)
    mats, vals, spans, step0 = [], [], [], []
    for i, r in enumerate(d.records):
        row = synthetic_row(rng, int(r["tokens"]), int(r["steps"]))
        p = row["top_k_logprobs"]
        x, v = answer_streams(p["logprobs"], p["ids"], row["gen_token_ids"], row["token_spilled_energies"])
        sp = row["step_token_spans"]
        # plant a bump at the labelled step so the dry run has something to localize
        t = d.target[i]
        if t >= 0:
            a, b = sp[t]; x[a:b] += 1.0
        mats.append(x); vals.append(v); spans.append(sp); step0.append(step0_token_mask(sp, len(x)))
    toff = np.concatenate([[0], np.cumsum([len(m) for m in mats])])
    np.savez(tmp / "CT7_TOKEN_MATRICES.npz", tokens=np.vstack(mats).astype(np.float32), valid=np.vstack(vals),
             token_offsets=toff, step_spans=np.vstack(spans).astype(np.int32), step0_token_mask=np.concatenate(step0),
             channels=np.asarray(STREAMS, dtype=str))
    return d


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config"); p.add_argument("--dry-run", action="store_true"); p.add_argument("--draws", type=int)
    args = p.parse_args(); started = time.perf_counter()
    if args.dry_run:
        tmp = Path(tempfile.mkdtemp(prefix="ct7_token_lsml_dry_")); d = synthetic_token_data(tmp)
        td = TokenData(tmp / "CT7_TOKEN_MATRICES.npz", d); out = d.out
    else:
        d = L.light_dataset(args.config); paths = d.c["paths"]; out = d.out
        L.run_freeze(out, [Path(__file__), L.ROOT / "scripts/experiments/ct7_levers_common.py",
                           L.ROOT / "spectral_utils/ct7_token_streams.py", L.ROOT / "spectral_utils/claude_feature_bank_v1.py",
                           Path(args.config).resolve()],
                     [Path(paths[k]) for k in ("roster", "joined", "folds", "ct7", "ct7_tokens") if k in paths],
                     {"schema": SCHEMA, "candidate_id": d.c["candidate_id"], "development_only": True})
        td = TokenData(paths["ct7_tokens"], d)
        L.replay_ct7(d)
    methods, fits, extras = build_methods(td)
    contrasts = planned(list(methods))
    result = L.evaluate_methods(d, methods, contrasts_extra=contrasts, strata_contrasts=contrasts, prmscore=not args.dry_run, draws=args.draws)
    rows = L.summary_rows(d, methods, result); L.write_summary_csv(out / "SUMMARY.csv", rows)
    record = {"schema": SCHEMA, "development_only": True, "note": L.DEVELOPMENT_NOTE, "streams": list(STREAMS),
              "access": {"pooled_donor_standardizer_and_weights": ["T_E1", "T_C1", "T_E1_six", "T_C1_six"],
                         "answer_local_standardization_then_pooled_donor_weights": ["T_E2", "T_C2", "T_E2_six", "T_C2_six", "T_C2_nostep0fit"],
                         "fit_free": ["ct7", "ct7_top10_equal7", "ct7_top10_equal7_nodespike"]},
              "fits": fits, "extras": extras, "pb": result["pb"], "prm": result["prm"], "strata": result["strata"],
              "timing": result["timing"], "total_seconds": time.perf_counter() - started, "dry_run": bool(args.dry_run)}
    L.dump(out / "RESULTS.json", record)
    print(f"{'method':28s} {'SLA':>7s} {'F1':>7s} {'within':>7s} {'early':>6s} {'late':>6s}")
    for r in rows:
        print(f"{r['method']:28s} {100*r['sla_macro8']:7.2f} {100*r['f1_common_gate']:7.2f} {r['within_auc']:7.4f} {r['early']:6.3f} {r['late']:6.3f}")
    for c in result["uncertainty"]["contrasts"]:
        if c["endpoint"] == "pb_sla" and c["contrast"].startswith(("PRIMARY", "lsml_minus_equal", "arm_minus_ct7")):
            print(f"{c['a']:>18s} - {c['b']:<18s} SLA {c['delta']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]")
    for name, f in fits.items():
        if f["fits"][0].get("weights") is not None:
            print(name, "K per fold", [x["K"] for x in f["fits"]], "IPR", [round(x["weight_ipr"], 2) for x in f["fits"]],
                  "negative", [x["negative_weights"] for x in f["fits"]])
    print("cond. PR of the seven Top10 views:", round(extras["conditional_participation_ratio_top10_views"], 3))
    print("written:", out, f"({time.perf_counter() - started:.1f}s)")


if __name__ == "__main__":
    main()
