"""Stage 0 readout controls + Stage 1 numeral-provenance rollback on the frozen
localization benchmark. Protocol: docs/experiments/READOUT_LENGTH_CONTROL_AND_PROVENANCE_V1.md.

Evaluation is imported unchanged from scripts/run_direct_probability_temporal.py
(evaluate_arrays, paired_bootstrap) so every number is comparable with Steps 334-341.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_direct_probability_fusion_v2 as old  # noqa: E402
from scripts.run_direct_probability_temporal import evaluate_arrays, paired_bootstrap, atomic_json  # noqa: E402
from spectral_utils import provenance_readout as pr  # noqa: E402
from spectral_utils.answer_localization_v2 import STREAM_NAMES  # noqa: E402

OUT = ROOT / "results" / "readout_length_control_and_provenance_v1"
STREAMS = {"entropy": STREAM_NAMES.index("entropy_series"),
           "varentropy": STREAM_NAMES.index("topk_varentropy_series")}
REPRO = {"entropy_top10": (0.354444, 0.7301113611124386, 0.625425539209566),
         "varentropy_top10": (0.356755, 0.742464548435578, 0.6327768738968429)}
SEED_RANDOM, SEED_SHUFFLE = 2026091101, 2026091102
PB_FILES = {"gsm8k": "processbench_gsm8k.pkl", "math": "processbench_math.pkl",
            "olympiadbench": "processbench_olympiadbench.pkl", "omnimath": "processbench_omnimath.pkl"}
TOKENIZERS = {"q4": "Qwen/Qwen3-4B", "q8": "Qwen/Qwen3-8B"}
STAGE0 = ("length", "random_step", "position_first")
STAGE1 = ("rise_vs_history", "first_near_max", "provenance_reassign", "provenance_duplicate", "provenance_shuffled")
BOOT_DRAWS = 10000


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def tokenizer_for(cell: str):
    key = "q8" if cell == "prmbench_qwen3_8b" else cell.rsplit("_", 1)[1]
    return TOKENIZERS[key]


class Decoder:
    def __init__(self):
        from transformers import AutoTokenizer
        self.tok = {n: AutoTokenizer.from_pretrained(n, local_files_only=True) for n in set(TOKENIZERS.values())}
        self.cache = {n: {} for n in self.tok}

    def texts(self, name: str, ids) -> list[str]:
        c, t = self.cache[name], self.tok[name]
        out = []
        for i in ids:
            i = int(i)
            if i not in c:
                c[i] = t.decode([i])
            out.append(c[i])
        return out

    def decode(self, name: str, ids) -> str:
        return self.tok[name].decode([int(i) for i in ids])

    def info(self):
        return {n: dict(vocab=len(t), name_or_path=str(t.name_or_path)) for n, t in self.tok.items()}


def load_sources(source: Path, records):
    """Frozen telemetry per cell + cached token ids / step text, aligned to records."""
    bench = source / "results" / "localization_full_benchmark_v3" / "inputs"
    tele, index = {}, {}
    for cell in sorted({r["cell"] for r in records}):
        d = bench / cell
        tele[cell] = dict(raw=np.load(d / "raw.npy", mmap_mode="r"), row_ids=np.load(d / "row_ids.npy"),
                          starts=np.load(d / "step_starts.npy"), ends=np.load(d / "step_ends.npy"),
                          sro=np.load(d / "step_row_offsets.npy"))
        index[cell] = {rid: i for i, rid in enumerate(tele[cell]["row_ids"].tolist())}
    cache = {}
    for model, sub in (("pb_qwen3_4b", "q4"), ("pb_qwen3_8b", "q8")):
        for subset, fname in PB_FILES.items():
            rows = old.load_pickle(source / "dataset_cache" / "repgrid" / model / fname)
            cache[f"pb_{subset}_{sub}"] = {f"{subset}::{row['id']}": row for row in rows.values()}
    prm = old.load_pickle(source / "dataset_cache" / "four_localization" / "prmbench_qwen3_8b_telemetry_full" / "prmbench_telemetry.pkl")
    cache["prmbench_qwen3_8b"] = {row["idx"]: row for row in prm.values()}
    return tele, index, cache


def answer_view(rec, tele, index, cache):
    cell = rec["cell"]; t = tele[cell]; i = index[cell][rec["row_id"]]
    a, b = int(t["sro"][i]), int(t["sro"][i + 1])
    starts, ends = t["starts"][a:b], t["ends"][a:b]
    o = int(starts[0]); T = int(ends[-1]) - o
    row = cache[cell][rec["row_id"]]
    if len(row["gen_token_ids"]) != T or len(row["steps"]) != b - a or rec["tokens"] != T or rec["steps"] != b - a:
        raise ValueError(f"alignment mismatch {rec['uid']}: cache {len(row['gen_token_ids'])}/{len(row['steps'])} vs telemetry {T}/{b-a}")
    spans = [(int(u) - o, int(w) - o) for u, w in zip(starts, ends)]
    if [tuple(s) for s in row["step_token_spans"]] != spans:
        raise ValueError(f"step span mismatch {rec['uid']}")
    raw = np.asarray(t["raw"][o:o + T, :], float)
    problem = row.get("problem", row.get("question", ""))
    return dict(raw=raw, starts=np.array([s for s, _ in spans]), ends=np.array([e for _, e in spans]),
                ids=row["gen_token_ids"], steps=row["steps"], problem=problem, T=T, K=b - a)


def score_answer(v, dec: Decoder, tok_name: str, rng_random, rng_shuffle, roundtrip_stats):
    starts, ends, K, T = v["starts"], v["ends"], v["K"], v["T"]
    texts = dec.texts(tok_name, v["ids"])
    for k, (u, w) in enumerate(zip(starts, ends)):
        got, want = pr.normalize_ws(dec.decode(tok_name, v["ids"][u:w])), pr.normalize_ws(v["steps"][k])
        roundtrip_stats["steps"] += 1
        if got != want:
            roundtrip_stats["mismatch"] += 1
            if len(roundtrip_stats["examples"]) < 5:
                roundtrip_stats["examples"].append(dict(got=got[:120], want=want[:120]))
    given = pr.given_literals(v["problem"])
    nums = pr.build_numerals(texts, starts, ends, given)
    memberships = {m: pr.reassign(nums, starts, ends, T, m, rng=rng_shuffle if m == "shuffled" else None)
                   for m in ("reassign", "duplicate", "shuffled")}
    scores = {"length": pr.length_scores(starts, ends), "random_step": pr.random_scores(K, rng_random),
              "position_first": pr.position_first_scores(K)}
    for st, col in STREAMS.items():
        tok = v["raw"][:, col]
        base = pr.step_scores_top_k(tok, starts, ends)
        scores[f"{st}_top10"] = base
        scores[f"{st}_rise_vs_history"] = pr.rise_vs_history(base)
        scores[f"{st}_first_near_max"] = pr.first_near_max(base)
        for m in ("reassign", "duplicate", "shuffled"):
            scores[f"{st}_provenance_{m}"] = pr.step_scores_from_pairs(tok, memberships[m].pairs, K)
    r = memberships["reassign"]
    stats = dict(numerals=r.numerals, given=r.given_numerals, inherited=r.inherited_numerals, moved_tokens=r.moved_tokens,
                 answers_with_move=int(r.moved_tokens > 0))
    return scores, stats


def stratified_panel(records, joined, per, scores, offsets):
    """Raw exact / within-one on erroneous PB answers, split by whether the true
    step is the longest; early/exact/late counts and the clipped diff histogram."""
    cells = np.array([r["cell"] for r in records]); target = joined["target"]; pb = np.char.startswith(cells, "pb_")
    longest = np.full(len(records), False)
    for i in range(len(records)):
        if pb[i] and target[i] >= 0:
            longest[i] = int(np.argmax(scores["length"][offsets[i]:offsets[i + 1]])) == target[i]
    err = pb & (target >= 0)
    chance = float(np.mean([1.0 / (offsets[i + 1] - offsets[i]) for i in np.flatnonzero(err)]))
    out = {"n_error_answers": int(err.sum()), "n_true_is_longest": int((err & longest).sum()), "chance_exact": chance, "arms": {}}
    for name, p in per.items():
        d = p["peak"] - target; valid = p["valid"]
        row = {}
        for label, mask in (("all", err & valid), ("true_is_longest", err & valid & longest), ("true_not_longest", err & valid & ~longest)):
            n = int(mask.sum())
            row[label] = dict(n=n, exact=float(np.mean(d[mask] == 0)) if n else None, within_one=float(np.mean(np.abs(d[mask]) <= 1)) if n else None,
                              gated_exact=float(np.mean((p["prediction"] == target)[mask])) if n else None)
        m = err & valid
        row["early"] = int(np.sum(d[m] < 0)); row["exact"] = int(np.sum(d[m] == 0)); row["late"] = int(np.sum(d[m] > 0))
        row["late_over_early"] = row["late"] / max(row["early"], 1)
        hist = np.clip(d[m], -4, 4); row["diff_histogram"] = {str(k): int(np.sum(hist == k)) for k in range(-4, 5)}
        row["per_cell"] = {}
        for cell in sorted(set(cells[pb])):
            mc = m & (cells == cell); dc = d[mc]
            row["per_cell"][cell] = dict(n=int(mc.sum()), exact=float(np.mean(dc == 0)) if mc.any() else None,
                                         within_one=float(np.mean(np.abs(dc) <= 1)) if mc.any() else None,
                                         late=int(np.sum(dc > 0)), early=int(np.sum(dc < 0)),
                                         true_is_longest_exact=float(np.mean(d[mc & longest] == 0)) if (mc & longest).any() else None,
                                         true_not_longest_exact=float(np.mean(d[mc & ~longest] == 0)) if (mc & ~longest).any() else None)
        out["arms"][name] = row
    return out


def smoke_ids(records):
    picks = []
    for cell in sorted({r["cell"] for r in records}):
        idx = [i for i, r in enumerate(records) if r["cell"] == cell]
        toks = np.array([records[i]["tokens"] for i in idx]); order = np.argsort(toks)
        for q in (0, 0.5, 0.95):
            picks.append(idx[order[min(len(order) - 1, int(q * (len(order) - 1)))]])
    return sorted(set(picks))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-root", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=BOOT_DRAWS)
    args = ap.parse_args(); source = args.source_root.resolve()
    if args.bootstrap != BOOT_DRAWS:
        raise ValueError("protocol fixes bootstrap draws at 10000")
    old.configure_source_root(source); OUT.mkdir(parents=True, exist_ok=True)
    records = json.loads((old.BENCH / "evaluation/JOINED.json").read_text(encoding="utf8"))["records"]
    joined = np.load(old.BENCH / "evaluation/JOINED.npz", allow_pickle=False); offsets = joined["offsets"]
    if len(records) != 13769:
        raise ValueError("full roster mismatch")
    t0 = time.time(); tele, index, cache = load_sources(source, records); dec = Decoder()
    print(f"[load] sources + tokenizers {time.time()-t0:.0f}s", flush=True)
    rng_random, rng_shuffle = np.random.default_rng(SEED_RANDOM), np.random.default_rng(SEED_SHUFFLE)
    todo = smoke_ids(records) if args.smoke else range(len(records))
    names = list(STAGE0) + [f"{st}_{a}" for st in STREAMS for a in ("top10",) + STAGE1]
    flat = {n: np.full(int(offsets[-1]), np.nan) for n in names}
    roundtrip = dict(steps=0, mismatch=0, examples=[]); prov = {}; per_answer = []
    t0 = time.time()
    for n_done, i in enumerate(todo, 1):
        rec = records[i]; v = answer_view(rec, tele, index, cache)
        scores, stats = score_answer(v, dec, tokenizer_for(rec["cell"]), rng_random, rng_shuffle, roundtrip)
        sl = slice(int(offsets[i]), int(offsets[i + 1]))
        for n in names:
            flat[n][sl] = scores[n]
        c = prov.setdefault(rec["cell"], dict(answers=0, numerals=0, given=0, inherited=0, moved_tokens=0, answers_with_move=0))
        c["answers"] += 1
        for k in ("numerals", "given", "inherited", "moved_tokens", "answers_with_move"):
            c[k] += stats[k]
        if args.smoke:
            per_answer.append(dict(uid=rec["uid"], tokens=v["T"], steps=v["K"], **stats))
        if n_done % 1000 == 0:
            print(f"[score] {n_done}/{len(todo)} {time.time()-t0:.0f}s", flush=True)
    print(f"[score] done {len(todo)} answers in {time.time()-t0:.0f}s; step round-trip mismatches {roundtrip['mismatch']}/{roundtrip['steps']}", flush=True)
    manifest = dict(protocol="docs/experiments/READOUT_LENGTH_CONTROL_AND_PROVENANCE_V1.md",
                    code={p: sha(ROOT / p) for p in ("spectral_utils/provenance_readout.py", "scripts/run_readout_provenance_v1.py")},
                    inputs={str(p.relative_to(source)): sha(p) for p in [old.BENCH / "evaluation/JOINED.json", old.BENCH / "evaluation/JOINED.npz", old.FOLDS,
                            old.FIXED_GATE / "DETECTORS.npz", old.FIXED_GATE / "METRICS.json"] + [old.BENCH / "inputs" / c / "raw.npy" for c in sorted(tele)]},
                    tokenizers=dec.info(), seeds=dict(random_step=SEED_RANDOM, shuffled=SEED_SHUFFLE), roundtrip=roundtrip, provenance=prov)
    if args.smoke:
        atomic_json(OUT / "SMOKE.json", dict(status="FEASIBILITY_ONLY", n_answers=len(todo), rows=per_answer, manifest=manifest))
        print("[smoke] feasibility only; no benchmark metrics from a subset", flush=True)
        return
    if roundtrip["mismatch"]:
        atomic_json(OUT / "RUN_STATE.json", dict(status="FAILED", error="tokenizer round-trip mismatch", roundtrip=roundtrip))
        raise SystemExit("tokenizer round-trip mismatch; see RUN_STATE.json")
    atomic_json(OUT / "MANIFEST.json", manifest)
    metrics, per = evaluate_arrays(records, joined, flat)
    for name, (pb, within, prmscore) in REPRO.items():
        got = (metrics[name]["pb_all8"], metrics[name]["prm_within"], metrics[name]["prmscore_q08"])
        np.testing.assert_allclose(got, (pb, within, prmscore), atol=1e-6, rtol=0)
    print("[evaluate] reference reproduction verified", flush=True)
    pairs, primary = [], set()
    for st in STREAMS:
        ref, rea, shf = f"{st}_top10", f"{st}_provenance_reassign", f"{st}_provenance_shuffled"
        primary |= {(rea, ref), (rea, shf)}
        pairs += [(rea, ref), (rea, shf), (f"{st}_provenance_duplicate", ref), (shf, ref),
                  (f"{st}_rise_vs_history", ref), (f"{st}_first_near_max", ref)]
    pairs += [("length", "entropy_top10"), ("random_step", "entropy_top10"), ("position_first", "entropy_top10")]
    contrasts = paired_bootstrap(records, joined, per, draws=args.bootstrap, pairs=pairs, primary_pairs=primary)
    for key, c in contrasts.items():
        a, b = key.split("_minus_"); c["pb_delta"] = metrics[a]["pb_all8"] - metrics[b]["pb_all8"]
        c["prm_within_delta"] = (metrics[a]["prm_within"] or 0) - (metrics[b]["prm_within"] or 0)
    panel = stratified_panel(records, joined, per, flat, offsets)
    payload = dict(schema="readout-length-control-and-provenance-v1", n_answers=len(records), n_steps=int(offsets[-1]),
                   metrics=metrics, contrasts=contrasts, stratified=panel, provenance=prov, roundtrip=roundtrip,
                   scope="Full cached development. Fixed label-free rules on each answer's own tokens; frozen gate/readout/labels/folds; not untouched confirmation.")
    atomic_json(OUT / "METRICS.json", payload)
    np.savez_compressed(OUT / "SCORES.npz", **{"steps__" + m: s for m, s in flat.items()},
                        **{"prediction__" + m: p["prediction"] for m, p in per.items()},
                        **{"peak__" + m: p["peak"] for m, p in per.items()}, **{"valid__" + m: p["valid"] for m, p in per.items()})
    lines = ["method,PB_all8,PB_Q4,PB_Q8,PRMB_within,PRMB_pooled,PRMScore,raw_exact,exact_true_longest,exact_true_not_longest,within_one,early,late"]
    for m, x in metrics.items():
        s = panel["arms"][m]
        lines.append(",".join(str(v) for v in [m, x["pb_all8"], x["pb_q4"], x["pb_q8"], x["prm_within"], x["prm_pooled"], x["prmscore_q08"],
                                                s["all"]["exact"], s["true_is_longest"]["exact"], s["true_not_longest"]["exact"], s["all"]["within_one"], s["early"], s["late"]]))
    (OUT / "SUMMARY.csv").write_text("\n".join(lines) + "\n", encoding="utf8")
    atomic_json(OUT / "RUN_STATE.json", dict(status="COMPLETE", completed=len(records), expected=len(records)))
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        state_path = OUT / ("SMOKE_STATE.json" if "--smoke" in sys.argv else "RUN_STATE.json")
        state = json.loads(state_path.read_text(encoding="utf8")) if state_path.exists() else {}
        state.update(status="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED", error=f"{type(error).__name__}: {error}")
        atomic_json(state_path, state)
        raise
