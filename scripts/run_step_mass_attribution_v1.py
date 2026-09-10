"""Step-mass attribution along numeral dependencies on the frozen localization
benchmark. Protocol: docs/experiments/STEP_MASS_ATTRIBUTION_V1.md.

Loaders, tokenizer decoding, stratified panel and evaluation are reused from
scripts/run_readout_provenance_v1.py (Step 354) and scripts/run_direct_probability_temporal.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_direct_probability_fusion_v2 as old  # noqa: E402
from scripts.run_direct_probability_temporal import evaluate_arrays, paired_bootstrap, atomic_json  # noqa: E402
from scripts.run_readout_provenance_v1 import (Decoder, load_sources, answer_view, tokenizer_for,  # noqa: E402
                                               stratified_panel, smoke_ids, sha, STREAMS, REPRO, SEED_RANDOM)
from spectral_utils import provenance_readout as pr  # noqa: E402

OUT = ROOT / "results" / "step_mass_attribution_v1"
SEED_SHUFFLE = 2026091103
ALPHAS = {"a05": 0.5, "a10": 1.0}
ARMS = ("top10", "zscore_only", "attr_a05", "attr_a10", "attr_shuffled_a05", "attr_uniform_a05")
BOOT_DRAWS = 10000


def score_answer(v, dec, tok_name, rng_random, rng_shuffle, roundtrip):
    starts, ends, K, T = v["starts"], v["ends"], v["K"], v["T"]
    texts = dec.texts(tok_name, v["ids"])
    for k, (u, w) in enumerate(zip(starts, ends)):
        got, want = pr.normalize_ws(dec.decode(tok_name, v["ids"][u:w])), pr.normalize_ws(v["steps"][k])
        roundtrip["steps"] += 1
        if got != want:
            key = "mismatch" if any(ch.isdigit() for ch in got + want) else "nonnumeric_mismatch"
            roundtrip[key] = roundtrip.get(key, 0) + 1
    nums = pr.build_numerals(texts, starts, ends, pr.given_literals(v["problem"]))
    counts = pr.dependency_counts(nums)
    parents = {k for k, _ in counts}
    scores = {"length": pr.length_scores(starts, ends), "random_step": pr.random_scores(K, rng_random)}
    shuffle_seed = int(rng_shuffle.integers(2**31))  # same random parent targets for both streams
    for st, col in STREAMS.items():
        s = pr.step_scores_top_k(v["raw"][:, col], starts, ends); z = pr.zscore_steps(s)
        scores[f"{st}_top10"] = s
        scores[f"{st}_zscore_only"] = z
        for tag, a in ALPHAS.items():
            scores[f"{st}_attr_{tag}"] = pr.attribute_step_mass(z, counts, a)
        scores[f"{st}_attr_shuffled_a05"] = pr.attribute_step_mass(z, counts, 0.5, "shuffled", np.random.default_rng(shuffle_seed))
        scores[f"{st}_attr_uniform_a05"] = pr.attribute_step_mass(z, counts, 0.5, "uniform")
    dep = dict(steps=K, steps_with_parents=len(parents), edges=len(counts), inherited=int(sum(counts.values())),
               parents_per_dependent=(len(counts) / len(parents)) if parents else 0.0,
               parent_sets={int(k): sorted({j for (kk, j) in counts if kk == k}) for k in parents})
    return scores, dep


def dependency_panel(records, joined, per, deps):
    """Structure stats + a label-using mechanism ceiling: is the true step a parent
    (or ancestor through one hop) of the frozen argmax step? Diagnostic only."""
    cells = np.array([r["cell"] for r in records]); target = joined["target"]; pb = np.char.startswith(cells, "pb_")
    out = {"structure": {}, "ceiling": {}}
    for cell in sorted(set(cells)):
        idx = np.flatnonzero(cells == cell); d = [deps[i] for i in idx]
        out["structure"][cell] = dict(answers=len(idx), answers_with_edges=int(sum(x["edges"] > 0 for x in d)),
                                      steps=int(sum(x["steps"] for x in d)), steps_with_parents=int(sum(x["steps_with_parents"] for x in d)),
                                      edges=int(sum(x["edges"] for x in d)), inherited=int(sum(x["inherited"] for x in d)))
    for st in STREAMS:
        peak = per[f"{st}_top10"]["peak"]; err = pb & (target >= 0) & per[f"{st}_top10"]["valid"]
        late = err & (peak > target); n_late = int(late.sum())
        parent_hit = 0; any_parent = 0
        for i in np.flatnonzero(late):
            ps = deps[i]["parent_sets"].get(int(peak[i]), [])
            any_parent += bool(ps); parent_hit += int(target[i]) in ps
        out["ceiling"][st] = dict(late_misses=n_late, late_with_any_parent=any_parent, late_true_step_is_parent=parent_hit,
                                  frac_true_is_parent_of_late_peak=(parent_hit / n_late) if n_late else None)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-root", type=Path, required=True); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=BOOT_DRAWS)
    args = ap.parse_args(); source = args.source_root.resolve()
    if args.bootstrap != BOOT_DRAWS:
        raise ValueError("protocol fixes bootstrap draws at 10000")
    old.configure_source_root(source); OUT.mkdir(parents=True, exist_ok=True)
    records = json.loads((old.BENCH / "evaluation/JOINED.json").read_text(encoding="utf8"))["records"]
    joined = np.load(old.BENCH / "evaluation/JOINED.npz", allow_pickle=False); offsets = joined["offsets"]
    t0 = time.time(); tele, index, cache = load_sources(source, records); dec = Decoder()
    print(f"[load] {time.time()-t0:.0f}s", flush=True)
    rng_random, rng_shuffle = np.random.default_rng(SEED_RANDOM), np.random.default_rng(SEED_SHUFFLE)
    todo = smoke_ids(records) if args.smoke else range(len(records))
    names = ["length", "random_step"] + [f"{st}_{a}" for st in STREAMS for a in ARMS]
    flat = {n: np.full(int(offsets[-1]), np.nan) for n in names}
    roundtrip = dict(steps=0, mismatch=0, nonnumeric_mismatch=0); deps = {}
    t0 = time.time()
    for n_done, i in enumerate(todo, 1):
        rec = records[i]; v = answer_view(rec, tele, index, cache)
        scores, dep = score_answer(v, dec, tokenizer_for(rec["cell"]), rng_random, rng_shuffle, roundtrip)
        sl = slice(int(offsets[i]), int(offsets[i + 1]))
        for n in names:
            flat[n][sl] = scores[n]
        deps[i] = dep
        if n_done % 2000 == 0:
            print(f"[score] {n_done}/{len(todo)} {time.time()-t0:.0f}s", flush=True)
    print(f"[score] done {len(todo)} in {time.time()-t0:.0f}s; digit mismatches {roundtrip['mismatch']} (non-numeric {roundtrip['nonnumeric_mismatch']})", flush=True)
    manifest = dict(protocol="docs/experiments/STEP_MASS_ATTRIBUTION_V1.md",
                    code={p: sha(ROOT / p) for p in ("spectral_utils/provenance_readout.py", "scripts/run_step_mass_attribution_v1.py", "scripts/run_readout_provenance_v1.py")},
                    inputs={str(p.relative_to(source)): sha(p) for p in [old.BENCH / "evaluation/JOINED.json", old.BENCH / "evaluation/JOINED.npz", old.FOLDS,
                            old.FIXED_GATE / "DETECTORS.npz", old.FIXED_GATE / "METRICS.json"]},
                    tokenizers=dec.info(), seeds=dict(random_step=SEED_RANDOM, shuffled=SEED_SHUFFLE), alphas=ALPHAS, roundtrip=roundtrip)
    if args.smoke:
        rows = [dict(uid=records[i]["uid"], **{k: v for k, v in deps[i].items() if k != "parent_sets"}) for i in todo]
        atomic_json(OUT / "SMOKE.json", dict(status="FEASIBILITY_ONLY", n_answers=len(todo), rows=rows, manifest=manifest))
        print("[smoke] feasibility only", flush=True); return
    if roundtrip["mismatch"]:
        raise SystemExit("digit round-trip mismatch")
    atomic_json(OUT / "MANIFEST.json", manifest)
    metrics, per = evaluate_arrays(records, joined, flat)
    for name, (pb, within, prmscore) in REPRO.items():
        np.testing.assert_allclose((metrics[name]["pb_all8"], metrics[name]["prm_within"], metrics[name]["prmscore_q08"]), (pb, within, prmscore), atol=1e-6, rtol=0)
    for st in STREAMS:  # z-scoring alone must not change any decision
        assert np.array_equal(per[f"{st}_zscore_only"]["peak"], per[f"{st}_top10"]["peak"])
    print("[evaluate] references reproduce; zscore identity holds", flush=True)
    pairs, primary = [], set()
    for st in STREAMS:
        ref, a05, shf, uni, a10 = (f"{st}_{x}" for x in ("top10", "attr_a05", "attr_shuffled_a05", "attr_uniform_a05", "attr_a10"))
        primary |= {(a05, ref), (a05, shf)}
        pairs += [(a05, ref), (a05, shf), (a10, ref), (uni, ref), (a05, uni), (shf, ref)]
    contrasts = paired_bootstrap(records, joined, per, draws=args.bootstrap, pairs=pairs, primary_pairs=primary)
    for key, c in contrasts.items():
        a, b = key.split("_minus_"); c["pb_delta"] = metrics[a]["pb_all8"] - metrics[b]["pb_all8"]
        c["prm_within_delta"] = (metrics[a]["prm_within"] or 0) - (metrics[b]["prm_within"] or 0)
    panel = stratified_panel(records, joined, per, flat, offsets); dpanel = dependency_panel(records, joined, per, deps)
    payload = dict(schema="step-mass-attribution-v1", n_answers=len(records), n_steps=int(offsets[-1]), metrics=metrics, contrasts=contrasts,
                   stratified=panel, dependencies=dpanel, roundtrip=roundtrip,
                   scope="Full cached development. Fixed label-free attribution on each answer's own numeral dependencies; frozen gate/readout/labels/folds; not untouched confirmation. The dependency ceiling panel uses labels and is diagnostic only.")
    atomic_json(OUT / "METRICS.json", payload)
    np.savez_compressed(OUT / "SCORES.npz", **{"steps__" + m: s for m, s in flat.items()}, **{"prediction__" + m: p["prediction"] for m, p in per.items()},
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
        p = OUT / ("SMOKE_STATE.json" if "--smoke" in sys.argv else "RUN_STATE.json")
        st = json.loads(p.read_text(encoding="utf8")) if p.exists() else {}
        st.update(status="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED", error=f"{type(error).__name__}: {error}")
        atomic_json(p, st); raise
