"""Hash-checked, resumable full-population baseline replay and bank diagnostics."""
from __future__ import annotations
import argparse
import gc
import io
import json
from pathlib import Path
import signal
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
from scripts import run_direct_probability_temporal as evaluator
from scripts import run_renyi_position_temporal_fusion as common
from spectral_utils.temporal_research_features import BASELINE, FEATURES, SUBSETS, score_features
from spectral_utils.renyi_locator_feature_bank import BANKS, feature_matrix, score_bank
from spectral_utils.direct_probability_fusion_v2 import residual_tail_mass
from spectral_utils.math_gate_selection import percentile_by_cell
from spectral_utils.pb_prediction_bundle import prediction_bundle

STOP = False


def stop_handler(*_):
    global STOP
    STOP = True


def read_json(path):
    return json.loads(path.read_text(encoding="utf8"))


def audit_inputs(source, out):
    manifest_path = ROOT / "results/renyi_position_temporal_fusion_v1/MANIFEST.json"
    manifest = read_json(manifest_path)
    rows = []
    for origin, expected in manifest["hashes"].items():
        if "/dataset_cache/" in origin:
            path = source / "dataset_cache" / origin.split("/dataset_cache/", 1)[1]
        elif "/results/" in origin and any(k in origin for k in ("JOINED.", "FOLDS_V2", "fusion_fixed_gate_v1/")):
            path = source / "results" / origin.split("/results/", 1)[1]
        else:
            continue
        actual = common.sha256_file(path) if path.is_file() else None
        rows.append(dict(origin=origin, local=str(path), expected=expected, actual=actual,
                         size=path.stat().st_size if path.is_file() else None, match=actual == expected))
        print("[hash]", path.name, actual == expected, flush=True)
    report = dict(status="PASS" if rows and all(r["match"] for r in rows) else "FAILED",
                  source_manifest_sha256=common.sha256_file(manifest_path), files=rows)
    common.atomic_json(out / "INPUT_AUDIT.json", report)
    if report["status"] != "PASS":
        raise ValueError("source/benchmark fingerprint mismatch")
    return report


def load_contract(source):
    evaluator.old.configure_source_root(source)
    records = read_json(evaluator.old.BENCH / "evaluation/JOINED.json")["records"]
    with np.load(evaluator.old.BENCH / "evaluation/JOINED.npz", allow_pickle=False) as f:
        joined = {k: f[k] for k in f.files}
    if len(records) != 13769 or joined["offsets"][-1] != 145597 or len({r["uid"] for r in records}) != 13769:
        raise ValueError("full frozen roster required")
    np.testing.assert_array_equal(joined["offsets"], np.r_[0, np.cumsum([r["steps"] for r in records])])
    return records, joined


def score_raw(con, records, joined, out, smoke=False):
    done = {r[0] for r in con.execute("SELECT idx FROM answers")}
    selected = set(common.smoke_selection(records)) if smoke else set(range(len(records)))
    started = time.perf_counter()
    for cell, path, kind, dataset in evaluator.source_specs():
        indices = [i for i, r in enumerate(records) if r["cell"] == cell and i in selected and i not in done]
        if not indices:
            continue
        print("[load]", cell, len(indices), flush=True)
        rows = evaluator.old._source_row_map(evaluator.old.load_pickle(path), kind=kind, dataset=dataset)
        for i in indices:
            record, row = records[i], rows[records[i]["row_id"]]
            lp = np.asarray(evaluator.old._topk_payload(row)["logprobs"], dtype=np.float64)
            entropy = np.asarray(row["token_entropies"], dtype=np.float64)
            spans = np.asarray(row["step_token_spans"], dtype=np.int64)
            if spans.shape != (record["steps"], 2) or len(lp) != record["tokens"]:
                raise ValueError("raw token/span identity mismatch: " + record["uid"])
            scores, peaks, info, matrix = score_features(lp, entropy, spans)
            # Recompute cheap historical factorial controls in this same raw pass.
            features = feature_matrix(lp, entropy)
            for bank in BANKS:
                for solver in ("raw_step_equal", "scale_step_equal"):
                    s, _ = score_bank(features, spans, bank, solver)
                    scores[bank.name + "__" + solver] = s
            tail = residual_tail_mass(lp[:, :15])
            k = min(10, len(tail))
            gate = float(np.partition(tail, len(tail)-k)[-k:].mean())
            # Independent entropy formula; tolerate only original float32 rounding.
            q = np.exp(lp[:, :15] - np.max(lp[:, :15], axis=1, keepdims=True))
            q /= q.sum(axis=1, keepdims=True)
            entropy_delta = float(np.max(np.abs(-np.sum(q*np.log(q), axis=1) - entropy)))
            if entropy_delta > 2e-6:
                raise ValueError("top15 entropy identity mismatch")
            info.update(uid=record["uid"], gate=gate, readout_peaks=peaks, entropy_max_error=entropy_delta)
            blob = common.packed(**{"steps__" + name: s for name, s in scores.items()},
                                 features=matrix, logprobs15=lp[:, :15].astype(np.float32),
                                 spans=spans, entropy=entropy)
            con.execute("INSERT INTO answers VALUES (?,?,?)", (i, blob, common.dumps(info)))
            done.add(i)
            if len(done) % 50 == 0 or STOP:
                con.commit()
                common.atomic_json(out / "RUN_STATE.json", dict(status="SCORING", completed=len(done), expected=len(selected),
                                   seconds=time.perf_counter()-started, last_cell=cell, smoke=smoke))
                print("[score]", len(done), "/", len(selected), flush=True)
            if STOP:
                return False
        con.commit()
        del rows
        gc.collect()
    return len(done) == len(selected)


def evaluate(con, records, joined, out, draws):
    scores, infos = {}, []
    gate = np.full(len(records), np.nan)
    for expected, (i, blob, info_json) in enumerate(con.execute("SELECT idx,payload,info FROM answers ORDER BY idx")):
        if i != expected:
            raise ValueError("checkpoint roster gap")
        info = json.loads(info_json)
        if info["uid"] != records[i]["uid"]:
            raise ValueError("checkpoint UID drift")
        infos.append(info); gate[i] = info["gate"]
        with np.load(io.BytesIO(blob), allow_pickle=False) as row:
            for key in row.files:
                if key.startswith("steps__"):
                    name = key[7:]
                    if name not in scores:
                        scores[name] = np.full(int(joined["offsets"][-1]), np.nan)
                    scores[name][joined["offsets"][i]:joined["offsets"][i+1]] = row[key]
        if (i+1) % 1000 == 0:
            print("[assemble]",i+1,len(records),flush=True)
    if len(infos) != len(records) or any(not np.isfinite(s).all() for s in scores.values()):
        raise ValueError("evaluation requires complete score coverage")
    cells = np.array([r["cell"] for r in records]); pb = np.char.startswith(cells, "pb_")
    percentile = np.full(len(records), np.nan)
    percentile[pb] = percentile_by_cell(gate[pb], cells[pb])
    opened = percentile >= .33
    archive = out / "SCORES_FROZEN.npz"
    np.savez_compressed(archive, **{"steps__"+n: s for n, s in scores.items()}, gate_raw=gate, gate_percentile=percentile)
    common.atomic_json(out / "SCORE_FREEZE.json", dict(sha256=common.sha256_file(archive), methods=list(scores),
                       answers=len(records), steps=int(joined["offsets"][-1]), gate_q=.33))
    metrics, per = evaluator.evaluate_arrays(records, joined, scores, fold_auc=True, pb_gate_open=opened)
    for name in infos[0]["readout_peaks"]:
        peaks = np.array([r["readout_peaks"][name] for r in infos])
        value, pred, valid = prediction_bundle(joined["target"], cells, peaks, np.ones(len(records), bool), opened)
        metrics["readout__"+name] = value
        per["readout__"+name] = dict(prediction=pred, decision_valid=valid, peak=peaks, valid=valid,
                                    within=np.full(len(records), np.nan))
    reference = metrics[BASELINE]
    # Historical results are checked before publishing any new feature comparison.
    historical = read_json(ROOT / "results/renyi_locator_integrated_replay_v1/METRICS.json")["metrics"]
    historical_name = "ve1q15__h10__hinf0__raw_step_equal"
    checks = {}
    for key in ("pb_all8", "prm_within", "prmscore_q08"):
        checks[key] = dict(actual=reference[key], expected=historical[historical_name][key],
                           difference=reference[key]-historical[historical_name][key])
    replay_pass = all(abs(v["difference"]) < 2e-8 for v in checks.values())
    common.atomic_json(out / "BASELINE_REPLAY.json", dict(status="PASS" if replay_pass else "MISMATCH", checks=checks,
                       original_score_archive_available=False, independent_raw_recompute=True))
    common.atomic_json(out / "METRICS.json", dict(metrics=metrics, status="EVALUATED", development_only=True,
                       baseline_replay_pass=replay_pass, gate="transductive within-cell midrank >= .33"))
    if not replay_pass:
        raise ValueError("baseline changed; investigate before promotion or model experiments")
    # Regenerate the matched historical factorial PB bundles without touching originals.
    corrections = {}
    old = read_json(ROOT / "results/renyi_locator_feature_bank_v1/METRICS.json")["metrics"]
    for name in old:
        if name in metrics:
            corrected = dict(old[name]); corrected.update({k:v for k,v in metrics[name].items() if k.startswith("pb_")})
            corrections[name] = dict(original=old[name], corrected=corrected,
                                     changed_fields=[k for k in corrected if corrected[k] != old[name].get(k)])
    common.atomic_json(out / "HISTORICAL_PB_CORRECTIONS.json", dict(corrected=len(corrections), expected=47,
                       status="PARTIAL" if len(corrections) < 47 else "COMPLETE", records=corrections,
                       pending="Other methods require saved scores or matched model replay; originals preserved."))
    primary = [("mean__VE0", BASELINE), ("mean__VE075", BASELINE)]
    pairs = primary + [(n, BASELINE) for n in SUBSETS if n != BASELINE and (n, BASELINE) not in primary]
    pairs += [(n, BASELINE) for n in scores if n.startswith("append_innovation__")]
    contrasts = evaluator.paired_bootstrap(records, joined, per, draws=draws, pairs=pairs,
                                          primary_pairs=set(primary), primary_ci=.975)
    for (a,b) in pairs:
        contrasts[a+"_minus_"+b]["pb_delta"] = metrics[a]["pb_all8"] - metrics[b]["pb_all8"]
    common.atomic_json(out / "CONTRASTS.json", contrasts)
    diagnostic = {}
    singles = ["mean__"+f for f in FEATURES]
    error = pb & (joined["target"] >= 0)
    hits = np.column_stack([per[n]["decision_valid"] & (per[n]["prediction"] == joined["target"]) & error for n in singles])
    diagnostic["expert_peak_selection_oracle_hits"] = int(hits.any(axis=1).sum())
    diagnostic["shared_hits"] = (hits.astype(int).T @ hits.astype(int)).tolist()
    diagnostic["unique_hits"] = {n: int((hits[:,j] & (hits.sum(axis=1)==1)).sum()) for j,n in enumerate(singles)}
    intersections = np.sum([r["top10_intersection"] for r in infos], axis=0)
    unions = np.sum([r["top10_union"] for r in infos], axis=0)
    diagnostic["token_top10_micro_jaccard"] = (intersections/unions).tolist()
    diagnostic["entropy_max_error"] = max(r["entropy_max_error"] for r in infos)
    diagnostic["feature_names"] = list(FEATURES)
    common.atomic_json(out / "DIAGNOSTICS.json", diagnostic)
    names = list(SUBSETS)
    frontier = [n for n in names if not any(metrics[m]["pb_all8"] >= metrics[n]["pb_all8"] and
                    metrics[m]["prm_within"] >= metrics[n]["prm_within"] and
                    (metrics[m]["pb_all8"] > metrics[n]["pb_all8"] or metrics[m]["prm_within"] > metrics[n]["prm_within"])
                    for m in names)]
    pb_best = min(names, key=lambda n:(-metrics[n]["pb_all8"], len(SUBSETS[n]), -metrics[n]["prmscore_q08"]))
    within_best = min(names, key=lambda n:(-metrics[n]["prm_within"], len(SUBSETS[n]), -metrics[n]["prmscore_q08"]))
    common.atomic_json(out / "PARETO.json", dict(frontier=frontier, provisional_banks=list(dict.fromkeys([BASELINE,pb_best,within_best])),
                       note="Bank freeze remains pending DUFS31 and complete feature diagnostics."))
    lines = ["# Baseline replay and feature-bank development results", "", "Independent raw-data baseline replay: PASS.", "",
             "All 13,769 answers; 145,597 steps. Frozen tail15 Top10 gate q=.33.", "",
             "| Method | PB % | Within AUC | PRMScore |", "|---|---:|---:|---:|"]
    for n in names + ["entropy15", "append_innovation__H0lim", "append_innovation__VE075"]:
        m = metrics[n]
        lines.append(f"| {n} | {100*m['pb_all8']:.4f} | {m['prm_within']:.6f} | {m['prmscore_q08']:.6f} |")
    lines.extend(["", "These are development results. See CONTRASTS.json for paired group intervals.",
                  "Historical PB repair is partial; missing learned-method archives are explicitly pending."])
    (out / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf8")
    np.savez_compressed(out / "PREDICTIONS.npz", **{k+"__"+n:v for n,p in per.items() for k,v in p.items()})
    common.atomic_json(out / "RUN_STATE.json", dict(status="BASELINE_AND_SUBSETS_COMPLETE", completed=len(records),
                       full_program_complete=False, next="Complete 47 repairs, matched RBM/IU and feature diagnostics; then model tracks."))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--out", type=Path, default=ROOT / "results/temporal_research_baseline_v1")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--evaluate-only", action="store_true")
    p.add_argument("--draws", type=int, default=10000)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, stop_handler)
    audit = audit_inputs(args.source_root, args.out)
    records, joined = load_contract(args.source_root)
    manifest = dict(schema="temporal-research-baseline-v1", inputs=audit, smoke=args.smoke,
                    extraction_sha256=common.sha256_file(ROOT/"spectral_utils/temporal_research_features.py"),
                    program_sha256=common.sha256_file(ROOT/"docs/experiments/TEMPORAL_RESEARCH_PROGRAM_20260915.md"))
    existing = args.out / "MANIFEST.json"
    if existing.exists() and read_json(existing) != manifest:
        raise ValueError("immutable checkpoint manifest mismatch")
    common.atomic_json(existing, manifest)
    con = sqlite3.connect(args.out / "CHECKPOINT.sqlite")
    con.execute("CREATE TABLE IF NOT EXISTS answers (idx INTEGER PRIMARY KEY, payload BLOB NOT NULL, info TEXT NOT NULL)")
    try:
        with threadpool_limits(limits=1):
            complete = True if args.evaluate_only else score_raw(con, records, joined, args.out, args.smoke)
            if args.smoke:
                common.atomic_json(args.out/"RUN_STATE.json", dict(status="SMOKE_COMPLETE" if complete else "STOPPED", quality_claim=False))
            elif complete:
                evaluate(con, records, joined, args.out, args.draws)
    except BaseException as error:
        common.atomic_json(args.out/"RUN_STATE.json", dict(status="FAILED", error=f"{type(error).__name__}: {error}"))
        raise
    finally:
        con.close()


if __name__ == "__main__":
    main()
