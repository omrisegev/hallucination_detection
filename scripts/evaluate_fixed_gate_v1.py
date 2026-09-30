"""Fixed-threshold gate on raw answer-level telemetry, applied to frozen answer-only peaks.

Consumes only frozen outputs of results/localization_full_benchmark_v3 (19 anchor
arms, all 13,769 records) and the raw token telemetry memmaps in its inputs/.
Writes results/fusion_fixed_gate_v1/. No fusion weight, peak or PRMB ranking
is changed; PRMB metrics are therefore identical to the frozen evaluation and
are not recomputed here. See spectral_utils/fixed_gate_readout.py.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from spectral_utils.fixed_gate_readout import (  # noqa: E402
    DETECTORS, STREAM_NAMES, calibrate_threshold, error_modes, gate_predictions, macro_all, nested_gate,
    oracle_gate, paired_group_bootstrap, self_test, summarize,
)
from spectral_utils.historical_fusion_evaluation import pb_metrics  # noqa: E402

BENCH = os.path.join(ROOT, "results", "localization_full_benchmark_v3")
FOLDS = os.path.join(ROOT, "results", "localization_source_group_audit_v1", "FOLDS_V2.json")
OUT = os.path.join(ROOT, "results", "fusion_fixed_gate_v1")
ARMS = ("dual__iu", "dual__equal", "dual__graph_perm", "context__iu", "moment__iu", "entropy_parent")
QUANTILES = (0.2, 0.3, 0.4, 0.5)
FAMILIES = {"pb_gsm8k": "easy", "pb_math": "easy", "pb_olympiadbench": "hard", "pb_omnimath": "hard"}


def load_benchmark():
    j = json.load(open(os.path.join(BENCH, "evaluation", "JOINED.json")))
    z = np.load(os.path.join(BENCH, "evaluation", "JOINED.npz"))
    return j, {k: z[k] for k in z.files}


def answer_detectors(records):
    """Raw answer-level telemetry summaries per record (PB cells only; NaN elsewhere)."""
    by_row = {(r["cell"], r["row_id"]): i for i, r in enumerate(records)}
    n = len(records)
    out = {name: np.full(n, np.nan) for name in DETECTORS}
    cells = sorted(set(r["cell"] for r in records if r["cell"].startswith("pb_")))
    for cell in cells:
        d = os.path.join(BENCH, "inputs", cell)
        raw = np.load(os.path.join(d, "raw.npy"), mmap_mode="r")
        offsets = np.load(os.path.join(d, "token_offsets.npy"))
        row_ids = np.load(os.path.join(d, "row_ids.npy"), allow_pickle=True)
        assert raw.shape[1] == len(STREAM_NAMES)
        for k, rid in enumerate(row_ids):
            i = by_row[(cell, str(rid))]
            block = np.asarray(raw[offsets[k]:offsets[k + 1]])
            for name, (stream, how) in DETECTORS.items():
                out[name][i] = summarize(block[:, STREAM_NAMES.index(stream)], how)
    return out


def main():
    assert self_test()
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    j, z = load_benchmark()
    records, arms = j["records"], j["arms"]
    cells = np.array([r["cell"] for r in records])
    groups = np.array([r["group_id"] for r in records])
    pb = np.array([c.startswith("pb_") for c in cells])
    folds = json.load(open(FOLDS))
    outer = np.array([folds["outer"].get(g, -1) for g in groups])
    assert (outer[pb] >= 0).all(), "every PB group must have an outer fold"
    target = z["target"]
    detectors = answer_detectors(records)
    print(f"detectors computed in {time.time() - t0:.1f}s")

    # Answer-level separability of each detector (context only, label-using diagnostic).
    from sklearn.metrics import roc_auc_score
    separability = {}
    for name, d in detectors.items():
        separability[name] = {}
        for cell in sorted(set(cells[pb])):
            m = pb & (cells == cell) & np.isfinite(d)
            separability[name][cell] = float(roc_auc_score(target[m] >= 0, d[m]))
        separability[name]["mean"] = float(np.mean([v for k, v in separability[name].items() if k != "mean"]))

    results = {"scope": "frozen answer-only peaks from localization_full_benchmark_v3; gate replaced only",
               "population": "all 6,800 ProcessBench model-answer rows (8 cells); PRMB rows unchanged",
               "readout": "gate opens -> frozen argmax step; closed -> no error (-1); invalid rows fail",
               "detector_separability_auc": separability, "arms": {}}

    sub = lambda a: a[pb]  # noqa: E731
    T, C, G, O = sub(target), sub(cells), sub(groups), sub(outer)
    for arm in ARMS:
        m = arms.index(arm)
        peak = sub(z["peaks"][:, m])
        valid = sub(z["valid"][:, m])
        saved_pred = sub(z["predictions"][:, m])
        saved_valid = sub(z["decision"][:, m])
        rows = {}

        def add(name, pred, v, extra=None):
            met = pb_metrics(T, pred, v, C)
            rows[name] = dict(macros=met["macros"],
                              cells={c: dict(f1=x["f1"], clean_acc=x["clean_accuracy"], err_acc=x["error_exact_accuracy"])
                                     for c, x in met["cells"].items()},
                              modes=error_modes(T, pred, peak, v, C), **(extra or {}))
            return pred, v

        base_pred, base_valid = add("gmm_bic_saved", saved_pred, saved_valid, dict(access="answer-only", labels="none"))
        add("oracle_gate", *oracle_gate(T, peak, valid), dict(access="label oracle", labels="test labels (upper bound)"))
        add("always_error", np.where(valid, peak, -1), valid, dict(access="none", labels="none"))
        add("always_clean", np.full(len(T), -1), valid, dict(access="none", labels="none"))

        for dname in DETECTORS:
            d = sub(detectors[dname])
            # (a) nested label-calibrated constant, one per outer fold
            pred, v, thr = nested_gate(d, peak, valid, T, C, O, "labels")
            add(f"{dname}|nested_labels", pred, v, dict(access="raw answer telemetry + one constant",
                                                       labels="training folds only", thresholds=thr))
            # (b) label-free quantile constants
            for q in QUANTILES:
                pred, v, thr = nested_gate(d, peak, valid, T, C, O, "quantile", q=q)
                add(f"{dname}|quantile_{q}", pred, v, dict(access="raw answer telemetry + one constant",
                                                          labels="none (quantile of unlabeled training rows)",
                                                          thresholds=thr))
            # (c) cross-family transfer of a label-chosen constant
            fam = np.array([FAMILIES[c.rsplit("_", 1)[0]] for c in C])
            for src, dst in (("easy", "hard"), ("hard", "easy")):
                tr, te = fam == src, fam == dst
                t = calibrate_threshold(d[tr], peak[tr], valid[tr], T[tr], C[tr])["threshold"]
                pred = np.full(len(T), -1); v = np.zeros(len(T), bool)
                pred[te], v[te] = gate_predictions(d[te], t, peak[te], valid[te])
                met = pb_metrics(T[te], pred[te], v[te], C[te])
                rows[f"{dname}|transfer_{src}_to_{dst}"] = dict(
                    macros=met["macros"], threshold=float(t), evaluated_cells=sorted(set(C[te])),
                    cells={c: dict(f1=x["f1"]) for c, x in met["cells"].items()},
                    access="raw answer telemetry + one constant", labels=f"{src} family only")
            # (d) in-sample sensitivity curve (single global constant, labels on all rows; reference only)
            cal = calibrate_threshold(d, peak, valid, T, C)
            rows[f"{dname}|insample_curve"] = dict(threshold=cal["threshold"], macro_all=cal["training_macro"],
                                                  grid=cal["grid"], curve=cal["curve"], labels="all rows (in-sample reference)")

        # paired bootstrap of the two headline gates against the saved GMM gate
        boots = {}
        for name in (f"entropy_w8max|nested_labels", f"entropy_w8max|quantile_0.3", f"entropy_mean|nested_labels",
                     f"topk_tail_mass_mean|nested_labels"):
            pred, v, _ = (nested_gate(sub(detectors[name.split("|")[0]]), peak, valid, T, C, O, "labels")
                          if name.endswith("nested_labels") else
                          nested_gate(sub(detectors[name.split("|")[0]]), peak, valid, T, C, O, "quantile", q=0.3))
            boots[name + " minus gmm_bic_saved"] = paired_group_bootstrap(T, C, G, v, pred, base_valid, base_pred)
        results["arms"][arm] = dict(rows=rows, paired_bootstrap_macro_all=boots)
        print(f"{arm}: gmm {rows['gmm_bic_saved']['macros']['all']:.4f}  "
              f"entropy_w8max nested {rows['entropy_w8max|nested_labels']['macros']['all']:.4f}  "
              f"q0.3 {rows['entropy_w8max|quantile_0.3']['macros']['all']:.4f}  "
              f"oracle {rows['oracle_gate']['macros']['all']:.4f}   ({time.time() - t0:.0f}s)")

    results["seconds"] = time.time() - t0
    json.dump(results, open(os.path.join(OUT, "METRICS.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(OUT, "DETECTORS.npz"), **{k: v for k, v in detectors.items()})
    print("wrote", OUT)


if __name__ == "__main__":
    main()
