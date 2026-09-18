"""Reduce the per-token layer-lens field to STEP level - the input Stage 1b (the LOCATOR) needs.

This is the other half of the white-box channel. `reduce_layer_views_answer_level.py` produced
the answer-constant geometry, which can only ever feed the GATE. The lens field is per-token and
is the only part of this capture that can say WHERE the error is, which is the project's actual
target.

    ssh aircc 'export SLURM_CONF_SERVER=controller-primary; \
      S=<the shared project root>; cd $S/code && \
      sbatch -p power-gpu --qos=<owner qos> cluster/cpu_job.sbatch \
        cluster/reduce_layer_views_step_level.py \
          --roots $S/results --joined $S/code/results/localization_full_benchmark_v3/evaluation \
          --out $S/results/layer_views_step_level_v1'

`--roots` has no default on purpose. The project account has moved between shared filesystems
(cycle2 -> cycle3) and the sibling script's docstring still names the old one; a wrong default
here burns a wall-clock job to produce "N rows missing". Pass it, and the script prints what it
resolved before reading anything.

Local dry run on synthetic rows, no cluster and no real data:
    python cluster/reduce_layer_views_step_level.py --smoke

WHAT IS EMITTED
---------------
`step_views` is [145597, 864] float32. The 432 channels are the full field - 3 taps (attn, mlp,
resid) x 4 lens quantities (lens_H, lens_logp_tgt, lens_logp_top1, lens_kl_final) x 36 layers -
each reduced to its step with the project's adopted readout, the mean of the largest min(10, n)
token values in the step (`chosen_token_calibration.step_top_readout`, k=10).

Each channel is stored TWICE, and the naming is the point:

    <quantity>.<tap>.layer_NN.hi   = top10( x)    "a HIGH value of this quantity means risk"
    <quantity>.<tap>.layer_NN.lo   = top10(-x)    "a LOW  value of this quantity means risk"

Both columns are therefore RISK-ASCENDING, and `argmax` over steps is the correct readout for
either one with no sign handling anywhere downstream. An earlier draft stored the second column
as `bot10 = -top10(-x)`, which is the same information but NOT risk-ascending: `argmax(bot10)`
finds the step with the highest low-tail, a meaningless statistic, while the intended predictor
is `argmin(bot10)`. That trap is removed here by construction rather than by a comment.

Why store both at all: at scoring time orientation is NOT free for this task. The endpoint is
SLA, an argmax over steps, and SLA(-x) is an argmin - a genuinely different predictor, not
`1 - SLA(x)`. (The AUROC identity that makes orientation free for a gate does not transfer.) So
the readout orientation has to be materialised. It follows that keeping both halves DOUBLES the
label-selectable candidate set, which is a real multiplicity cost the scorer must pay for with a
pre-registered permutation null - it is not a free symmetry.

`incumbent` is the SAME readout applied to `final_lens_H`, the top-15 renormalised final-layer
entropy, in ONE declared orientation (high entropy = risk). It is the kill incumbent for Stage
1b and is reduced from the same token array and the same spans, so no part of the comparison
can differ by preprocessing. It is deliberately not stored in both orientations: a kill bar that
can be chosen with labels is not a bar.

`step_len` is carried because step LENGTH alone reaches ~29.7% step-level localization accuracy
against 16.58% chance - a documented prior every step readout inherits. Note that the top-10
readout is itself length-coupled (for n < 10 it is the step mean; for n >= 10 it is an order
statistic that grows with n), so a length-stratified comparison is required downstream, not
optional.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

from reduce_layer_views_answer_level import (
    CELL_SOURCES,
    EXPECTED_COUNTS,
    N_ANSWERS,
    row_filename,
    stratified_pilot,
)

MODULES = ("attn", "mlp", "resid")
RESID = MODULES.index("resid")
QUANTITIES = ("lens_H", "lens_logp_tgt", "lens_logp_top1", "lens_kl_final")
N_LAYERS = 36
N_STEPS_TOTAL = 145597
TOP_K = 10


def step_top_readout(x: np.ndarray, starts: np.ndarray, ends: np.ndarray, k: int = TOP_K):
    """Mean of the largest min(k, n) token values per step, for every column at once.

    Vendored deliberately rather than imported: `spectral_utils.chosen_token_calibration` is not
    on the cluster container's sys.path, and importing across worktrees is the version-drift
    class the plan bars. `smoke()` checks it against an independent `np.sort` formulation - not
    against a retyped copy of the same `np.partition` expression, which would duplicate rather
    than detect an off-by-one in the partition index.
    """
    out = np.empty((len(starts), x.shape[1]), dtype=np.float64)
    for i, (a, b) in enumerate(zip(starts, ends)):
        seg = x[a:b]
        kk = min(k, b - a)
        out[i] = np.partition(seg, b - a - kk, axis=0)[-kk:].mean(axis=0)
    return out


def channel_names() -> list[str]:
    """Iteration order is tap-major, then quantity, then layer - matching `reduce_row`."""
    return [f"{quantity}.{module}.layer_{layer:02d}"
            for module in MODULES for quantity in QUANTITIES for layer in range(N_LAYERS)]


def check_spans(path, starts, ends, n_tokens, expected_steps, expected_tokens):
    """The real span contract, and the token-axis identity.

    `spectral_utils.processbench.build_chain` documents that step spans are "disjoint and ordered
    but need not be contiguous (the separator sits between)", and `assert_alignment` states that
    separator tokens legitimately fall outside every span and that coverage is REPORTED, not
    asserted. An earlier draft of this file required `starts[1:] == ends[:-1]` and
    `ends[-1] == n_tokens`; HISTORY records a real row with 1,112 steps and 16 separator tokens
    outside the spans, so that assert would have aborted the job and emitted nothing.

    The token-axis check is the one that actually defends against Step 313's failure class. A
    step COUNT check cannot: both sides are derived from `len(row["steps"])` by construction, so
    they agree even under a one-step shift, a re-tokenization drift or a wrong chat template.
    `JOINED.json` independently records each answer's token count, and comparing against it ties
    this capture to the frozen benchmark on the token axis.
    """
    if len(starts) != expected_steps:
        raise SystemExit(f"{path}: {len(starts)} spans, JOINED says {expected_steps} steps")
    if n_tokens != expected_tokens:
        raise SystemExit(f"{path}: {n_tokens} tokens, JOINED says {expected_tokens} "
                         f"- the token axis does not match the frozen benchmark")
    if (starts < 0).any() or (ends > n_tokens).any():
        raise SystemExit(f"{path}: span outside [0, {n_tokens})")
    if not (starts < ends).all():
        raise SystemExit(f"{path}: empty or inverted span (an unmapped step reaches here as "
                         f"a degenerate range)")
    if not (starts[1:] >= ends[:-1]).all():
        raise SystemExit(f"{path}: spans overlap or are out of order")
    return float((ends - starts).sum()) / max(n_tokens, 1)


def reduce_row(path: str, expected_steps: int, expected_tokens: int) -> dict:
    """One answer -> [n_steps, 864] risk-ascending readouts, the incumbent, and diagnostics."""
    with np.load(path, allow_pickle=False) as z:
        starts = np.asarray(z["step_starts"], dtype=np.int64)
        ends = np.asarray(z["step_ends"], dtype=np.int64)
        final_top15 = z["final_lens_H"].astype(np.float64)
        final_full = z["final_lens_H_fullvocab"].astype(np.float64)
        gate_flag = str(z["gate_flag"])
        n_tokens = int(final_top15.shape[0])
        # Each compressed [3, L, T] tensor is decompressed ONCE. Reading z[quantity] inside the
        # tap loop decompresses every tensor three times - ~21 GB over the population instead
        # of ~7 GB, on a single-wall job with no resume.
        raw = {quantity: z[quantity] for quantity in QUANTITIES}

    for quantity, arr in raw.items():
        if arr.shape[0] != len(MODULES) or arr.shape[1] != N_LAYERS:
            raise SystemExit(f"{path}: {quantity} has shape {arr.shape}, "
                             f"expected ({len(MODULES)}, {N_LAYERS}, T)")

    coverage = check_spans(path, starts, ends, n_tokens, expected_steps, expected_tokens)

    # Module-axis identity + wrong-module control, the same pair the answer-level sibling runs.
    # This script reads ALL THREE taps, so an attn/resid swap would silently invert every
    # "which tap carries the signal" conclusion while every other check still passed.
    resid_last = raw["lens_H"][RESID, -1].astype(np.float64)
    identity = float(np.max(np.abs(resid_last - final_full)))
    control = float(np.max(np.abs(raw["lens_H"][0, -1].astype(np.float64) - final_full)))

    tokens = np.concatenate(
        [raw[quantity][tap].astype(np.float64).T          # [T, L]
         for tap in range(len(MODULES)) for quantity in QUANTITIES], axis=1)

    hi = step_top_readout(tokens, starts, ends)
    lo = step_top_readout(-tokens, starts, ends)
    return {
        "step_views": np.concatenate([hi, lo], axis=1).astype(np.float32),
        "incumbent": step_top_readout(final_top15[:, None], starts, ends).astype(np.float32),
        "step_len": (ends - starts).astype(np.int32),
        "n_tokens": np.int32(n_tokens),
        "coverage": np.float32(coverage),
        "identity_check": np.float32(identity),
        "axis_control": np.float32(control),
        "gate_flag": gate_flag,
    }


def run(roots: str, joined_dir: str, out_dir: str, limit: int | None = None) -> int:
    print(f"[reduce-step] roots={roots}", flush=True)
    if not os.path.isdir(roots):
        raise SystemExit(f"--roots does not exist: {roots}")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(joined_dir, "JOINED.json"), encoding="utf-8") as handle:
        records = json.load(handle)["records"]
    if len(records) != N_ANSWERS:
        raise SystemExit(f"roster has {len(records)} records, expected {N_ANSWERS}")
    arrays = np.load(os.path.join(joined_dir, "JOINED.npz"), allow_pickle=False)
    offsets = np.asarray(arrays["offsets"], dtype=np.int64)
    if int(offsets[-1]) != N_STEPS_TOTAL:
        raise SystemExit(f"JOINED has {int(offsets[-1])} steps, expected {N_STEPS_TOTAL}")
    steps_per_answer = np.diff(offsets)

    rows = [(r["row_id"], r["cell"]) for r in records]
    expected = {(r["row_id"], r["cell"]): (int(steps_per_answer[i]), int(r["tokens"]))
                for i, r in enumerate(records)}
    if limit:
        rows = stratified_pilot(rows, limit)
        print(f"[reduce-step] PILOT: {len(rows)} rows over "
              f"{len(set(c for _, c in rows))} cells", flush=True)
    started = time.time()

    acc: dict[str, list] = {k: [] for k in ("step_views", "incumbent", "step_len", "n_tokens",
                                            "coverage", "identity_check", "axis_control")}
    row_ids, cells, flags, counts, missing = [], [], [], [], []
    for i, (row_id, cell) in enumerate(rows):
        directory, subset = CELL_SOURCES[cell]
        parts = [roots, directory] + ([subset] if subset else []) + \
                ["rows", row_filename(row_id, cell)]
        path = os.path.join(*parts)
        if not os.path.exists(path):
            missing.append(f"{cell}/{row_id}")
            continue
        n_steps, n_tokens = expected[(row_id, cell)]
        summary = reduce_row(path, n_steps, n_tokens)
        for key in acc:
            acc[key].append(summary[key])
        row_ids.append(row_id)
        cells.append(cell)
        flags.append(summary["gate_flag"])
        counts.append(len(summary["step_len"]))
        if (i + 1) % 500 == 0:
            print(f"[reduce-step] {i + 1}/{len(rows)}  {time.time() - started:.0f}s", flush=True)

    if missing:
        raise SystemExit(f"{len(missing)} rows missing, first 5: {missing[:5]}")

    step_views = np.concatenate(acc["step_views"], axis=0)
    out_offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    per_cell = {cell: int(sum(1 for c in cells if c == cell)) for cell in EXPECTED_COUNTS}
    if limit:
        if len(set(cells)) != len(EXPECTED_COUNTS):
            raise SystemExit(f"pilot touched {len(set(cells))} cells, expected all "
                             f"{len(EXPECTED_COUNTS)}")
    else:
        if len(row_ids) != N_ANSWERS:
            raise SystemExit(f"{len(row_ids)} rows reduced, expected {N_ANSWERS}")
        if per_cell != EXPECTED_COUNTS:
            raise SystemExit(f"per-cell counts drifted: {per_cell}")
        if not np.array_equal(out_offsets, offsets):
            raise SystemExit("emitted offsets differ from JOINED offsets")
    if not np.isfinite(step_views).all():
        raise SystemExit(f"{int((~np.isfinite(step_views)).sum())} non-finite readouts")

    identity = np.asarray(acc["identity_check"])
    control = np.asarray(acc["axis_control"])
    if not (float(identity.max()) <= 1e-6 and float(control.min()) > 1e-3):
        raise SystemExit(
            f"module-axis assertion inconclusive: identity_max={float(identity.max()):.3e} "
            f"(want ~0), control_min={float(control.min()):.3e} (want >0)")

    names = channel_names()
    out_npz = os.path.join(out_dir, "STEP_LEVEL_PILOT.npz" if limit else "STEP_LEVEL.npz")
    tmp = out_npz + ".tmp.npz"
    np.savez(tmp,
             step_views=step_views,
             incumbent=np.concatenate(acc["incumbent"], axis=0),
             step_len=np.concatenate(acc["step_len"], axis=0),
             n_tokens=np.asarray(acc["n_tokens"]),
             coverage=np.asarray(acc["coverage"]),
             identity_check=identity, axis_control=control,
             offsets=out_offsets,
             row_id=np.array(row_ids), cell=np.array(cells), gate_flag=np.array(flags),
             names=np.array([f"{n}.hi" for n in names] + [f"{n}.lo" for n in names]),
             incumbent_names=np.array(["final_lens_H.hi"]))
    os.replace(tmp, out_npz)

    digest = hashlib.sha256()
    with open(out_npz, "rb") as handle:                 # streamed: the file is ~0.5 GB
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    cov = np.asarray(acc["coverage"])
    manifest = {
        "n_answers": len(row_ids),
        "n_steps": int(out_offsets[-1]),
        "per_cell": per_cell,
        "n_columns": int(step_views.shape[1]),
        "readout": f"step_top_readout k={TOP_K}; .hi = top10(x), .lo = top10(-x), "
                   f"both risk-ascending",
        "token_coverage_min": float(cov.min()),
        "token_coverage_median": float(np.median(cov)),
        "identity_check_max": float(identity.max()),
        "axis_control_min": float(control.min()),
        "gate_flag_nonempty": int(sum(1 for f in flags if f)),
        "bytes": os.path.getsize(out_npz),
        "sha256": digest.hexdigest(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "git_sha": os.environ.get("GIT_SHA"),
        "seconds": round(time.time() - started, 1),
        "status": "PILOT" if limit else "COMPLETE",
    }
    name = "MANIFEST_STEP_PILOT.json" if limit else "MANIFEST_STEP.json"
    with open(os.path.join(out_dir, name), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=1)
    print(json.dumps(manifest, indent=1), flush=True)
    return 0


def _write_row(base, row_id, cell, starts, ends, n_tokens, rng, layers=N_LAYERS):
    directory, subset = CELL_SOURCES[cell]
    parts = [base, directory] + ([subset] if subset else []) + ["rows"]
    target_dir = os.path.join(*parts)
    os.makedirs(target_dir, exist_ok=True)
    payload = {q: rng.normal(6.0, 2.0, size=(3, layers, n_tokens)).astype(np.float16)
               for q in QUANTITIES}
    np.savez_compressed(
        os.path.join(target_dir, row_filename(row_id, cell)),
        step_starts=np.asarray(starts), step_ends=np.asarray(ends),
        final_lens_H=rng.normal(5.0, 1.0, size=n_tokens).astype(np.float32),
        final_lens_H_fullvocab=payload["lens_H"][RESID, -1].astype(np.float32),
        gate_flag=np.asarray(""), **payload)
    return os.path.join(target_dir, row_filename(row_id, cell))


def smoke() -> int:
    """Adversarial fixtures: separator gaps, a degenerate span, and an independent readout ref."""
    import shutil
    import tempfile

    base = tempfile.mkdtemp(prefix="reduce_step_smoke_")
    ok = True
    checks = []
    try:
        rng = np.random.default_rng(0)

        # NON-CONTIGUOUS spans with one-token separator gaps, and two leading/trailing tokens
        # outside every span - what the real tokenizer produces. The previous fixture used
        # cumsum, i.e. contiguous by construction, so it guaranteed the one property real data
        # violates and could never have caught the assert that would have aborted the job.
        lens = [6, 4, 11, 3]
        starts, ends, pos = [], [], 2
        for n in lens:
            starts.append(pos); ends.append(pos + n); pos += n + 1
        n_tokens = pos + 2
        path = _write_row(base, "gsm8k::gsm8k-0", "pb_gsm8k_q4", starts, ends, n_tokens, rng)
        summary = reduce_row(path, len(lens), n_tokens)
        checks += [
            ("non-contiguous spans with separator gaps are accepted", True),
            ("coverage reported below 1.0, not asserted",
             0.0 < float(summary["coverage"]) < 1.0),
            ("864 columns", summary["step_views"].shape == (len(lens), 864)),
            ("incumbent is one declared orientation", summary["incumbent"].shape == (len(lens), 1)),
            ("n_tokens carried", int(summary["n_tokens"]) == n_tokens),
            ("finite", bool(np.isfinite(summary["step_views"]).all())),
        ]

        # .hi and .lo must BOTH be risk-ascending, i.e. .lo is top10(-x), not -top10(-x).
        hi, lo = summary["step_views"][:, :432], summary["step_views"][:, 432:]
        checks.append((".lo is top10(-x) and risk-ascending (not bot10)",
                       bool((lo >= -hi - 1e-4).all()) and float(np.abs(lo + hi).max()) > 1e-3))

        # readout against an INDEPENDENT formulation, not a retyped np.partition
        x = rng.normal(size=(40, 5))
        sp = np.array([[0, 7], [9, 23], [25, 40]])
        ref = np.array([np.sort(x[a:b], axis=0)[-min(TOP_K, b - a):].mean(axis=0)
                        for a, b in sp])
        mine = step_top_readout(x, sp[:, 0], sp[:, 1])
        checks.append(("readout == independent np.sort reference",
                       float(np.abs(mine - ref).max()) < 1e-12))

        # every guard must FIRE
        for label, args in [
            ("wrong step count", (len(lens) + 1, n_tokens)),
            ("token count disagreeing with JOINED", (len(lens), n_tokens + 1)),
        ]:
            try:
                reduce_row(path, *args)
            except SystemExit:
                checks.append((f"guard fires on {label}", True))
            else:
                checks.append((f"guard fires on {label}", False))

        bad = _write_row(base, "gsm8k::gsm8k-1", "pb_gsm8k_q4",
                         [0, 5, 4], [5, 9, 12], 14, rng)     # out of order / overlapping
        try:
            reduce_row(bad, 3, 14)
        except SystemExit:
            checks.append(("guard fires on overlapping/out-of-order spans", True))
        else:
            checks.append(("guard fires on overlapping/out-of-order spans", False))

        degenerate = _write_row(base, "gsm8k::gsm8k-2", "pb_gsm8k_q4",
                                [0, 6, 6], [5, 6, 11], 12, rng)   # an unmapped step
        try:
            reduce_row(degenerate, 3, 12)
        except SystemExit:
            checks.append(("guard fires on a degenerate (unmapped) span", True))
        else:
            checks.append(("guard fires on a degenerate (unmapped) span", False))

        for label, passed in checks:
            print(f"  {'PASS' if passed else 'FAIL'}  {label}")
            ok &= bool(passed)
        print("SMOKE PASSED" if ok else "SMOKE FAILED")
        return 0 if ok else 1
    finally:
        shutil.rmtree(base, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", default=None,
                        help="the capture root; no default, see the module docstring")
    parser.add_argument("--joined",
                        default="results/localization_full_benchmark_v3/evaluation")
    parser.add_argument("--out", default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="stratified pilot over all nine cells; skips the full-count asserts")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        return smoke()
    if not args.out or not args.roots:
        parser.error("--roots and --out are required unless --smoke")
    return run(args.roots, args.joined, args.out, args.limit)


if __name__ == "__main__":
    sys.exit(main())
