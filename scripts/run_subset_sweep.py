"""
run_subset_sweep.py — exhaustive L-SML subset sweep over the local caches.

Enumerates ALL feature subsets of sizes --min-size..--max-size per cell,
fuses each with continuous L-SML (label-free anchor orientation), and stores
per-subset AUROC, clustering (K + group assignment), effective per-feature
weights and within-subset |Spearman| stats. Chunked + resumable; safe to kill.

Typical usage:
    python scripts/build_derived_views.py                  # Stage 0, once
    python scripts/run_subset_sweep.py --self-check        # invariants
    python scripts/run_subset_sweep.py --calibrate-only    # ETA table
    python scripts/run_subset_sweep.py --workers 8 --yes   # full run
    python scripts/subset_sweep_report.py                  # report

Outputs under results/subset_sweep/:
    <domain>__<cell>.npz            merged per-subset records
    <domain>__<cell>.manifest.json  pool, params, reference results, top-20
    augmentation.pkl                extra-view augmentation records
    sweep_summary.csv               one row per finished cell
"""

import argparse
import csv
import glob
import json
import os
import pickle
import sys
import time

# Children inherit the environment at spawn and import numpy fresh, so this
# pins worker BLAS to one thread (the eigen-problems here are <= 21x21;
# oversubscription would only add contention). Must run before pool creation.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np

# Windows consoles default to cp1252 — reconfigure so log lines with math
# glyphs (or unicode in cell keys) never crash the run.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from spectral_utils.fusion_utils import (
    lsml_continuous, paired_boot_delta_auc, zscore,
)
from spectral_utils.streaming_utils import anchor_orient
from spectral_utils.subset_sweep import (
    ALL_SIGNS, CANONICAL_POOL, GOOD_5, H16, MAX_K,
    augment_cell, calibrate, canonical_to_names, cell_paths,
    count_masks, enumerate_masks, eval_subset, fuse_subset, iter_cells,
    mask_to_cols, names_to_local_mask, pack_assignment, prepare_cell,
    reference_results, run_cell_sweep, top_masks, unpack_assignment,
)
from sklearn.metrics import roc_auc_score


def self_check(ctx, fd, method):
    """Invariant checks on one prepared cell. Raises AssertionError on failure."""
    rng = np.random.default_rng(0)
    p = len(ctx.pool)

    assert count_masks(16, 3, 16) == 65399
    masks = enumerate_masks(p, 3, p)
    assert len(masks) == count_masks(p, 3, p)

    c = rng.integers(0, 4, size=12)
    packed, _ = pack_assignment(c)
    relabeled = unpack_assignment(packed, 12)
    # same partition modulo label names
    for i in range(12):
        for j in range(12):
            assert (c[i] == c[j]) == (relabeled[i] == relabeled[j])

    # linear composition: V @ w_eff_raw == fused (pre-flip), 8 random subsets
    from spectral_utils.subset_sweep import compose_effective_weights
    for _ in range(8):
        m = int(rng.integers(3, min(p, 9)))
        cols = np.sort(rng.choice(p, size=m, replace=False))
        fused, meta = lsml_continuous(*[ctx.V[:, j] for j in cols], method=method)
        w_raw = compose_effective_weights(meta, m)
        recon = ctx.V[:, cols] @ w_raw
        assert np.allclose(recon, fused, atol=1e-9), \
            f"linear composition broken for cols={cols}"

    # GOOD_5 sweep record == direct pipeline call on identical inputs
    if ctx.n_imputed == 0 and all(f in ctx.pool for f in GOOD_5):
        names = sorted(GOOD_5, key=ctx.pool.index)  # eval order = ascending cols
        cols = mask_to_cols(names_to_local_mask(names, ctx.pool))
        rec = eval_subset(ctx.V, ctx.labels, ctx.anchor, ctx.rho, cols,
                          ctx.pool_bits, method)
        views = [zscore(np.asarray(fd[f], dtype=float) * ALL_SIGNS[f]) for f in names]
        fused, meta = lsml_continuous(*views, method=method)
        oriented, _ = anchor_orient(fused, ctx.anchor)
        direct = roc_auc_score(ctx.labels, oriented)
        assert abs(float(rec['auroc']) - direct) < 1e-6, \
            f"GOOD_5 mismatch: sweep {float(rec['auroc']):.6f} vs direct {direct:.6f}"
        assert int(rec['K']) == int(meta['K'])
    print(f"  [self-check] {ctx.domain}/{ctx.cell_key}: all invariants PASS "
          f"(p={p}, n={len(ctx.labels)})")


def top20_with_deltas(ctx, results, method, n_boot=1000):
    """Top-20 subsets with boot CIs and paired delta vs GOOD_5."""
    good5_local = names_to_local_mask([f for f in GOOD_5 if f in ctx.pool], ctx.pool)
    good5_scores = None
    if good5_local is not None and len(mask_to_cols(good5_local)) >= 3:
        good5_scores, _, _ = fuse_subset(ctx.V, ctx.anchor,
                                         mask_to_cols(good5_local), method)
    from spectral_utils.fusion_utils import boot_auc
    out = []
    for rank, row in enumerate(top_masks(results, 20), 1):
        canon = int(results['mask'][row])
        names = [f for f in canonical_to_names(canon) if f in ctx.pool]
        cols = mask_to_cols(names_to_local_mask(names, ctx.pool))
        oriented, flipped, meta = fuse_subset(ctx.V, ctx.anchor, cols, method)
        auc, lo, hi = boot_auc(ctx.labels, oriented, n=n_boot)
        entry = {
            'rank': rank, 'mask': canon, 'feats': names,
            'size': len(names), 'auroc': float(auc),
            'ci': [float(lo), float(hi)], 'K': int(meta['K']),
            'flipped': bool(flipped),
        }
        if good5_scores is not None:
            d, dlo, dhi = paired_boot_delta_auc(ctx.labels, oriented,
                                                good5_scores, n=n_boot)
            entry['delta_vs_good5'] = [float(d), float(dlo), float(dhi)]
        out.append(entry)
    return out


def write_summary(out_dir):
    rows = []
    for mpath in sorted(glob.glob(os.path.join(out_dir, '*.manifest.json'))):
        with open(mpath) as f:
            m = json.load(f)
        if 'sweep_seconds' not in m:
            continue
        ref = m.get('reference', {})
        t20 = m.get('top20', [])
        rows.append({
            'domain': m['domain'], 'cell_key': m['cell_key'],
            'n': m['n'], 'pos_rate': round(m['n_pos'] / m['n'], 3),
            'pool_size': len(m['pool']), 'n_subsets': m['n_subsets'],
            'best_auroc': t20[0]['auroc'] if t20 else None,
            'best_feats': '|'.join(t20[0]['feats']) if t20 else None,
            'best_K': t20[0]['K'] if t20 else None,
            'good5_auroc': (ref.get('GOOD_5') or {}).get('auroc'),
            'good5_rank': m.get('good5_rank'),
            'good5_pctile': m.get('good5_pctile'),
            'all16_auroc': (ref.get('ALL_H16') or {}).get('auroc'),
            'epr_auroc': (ref.get('_anchor_single') or {}).get('auroc'),
            'avg_auroc': (ref.get('_avg_pool') or {}).get('auroc'),
            'sweep_seconds': m['sweep_seconds'],
        })
    path = os.path.join(out_dir, 'sweep_summary.csv')
    if rows:
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return path, len(rows)


def good5_rank(ctx, results):
    ref_local = names_to_local_mask([f for f in GOOD_5 if f in ctx.pool], ctx.pool)
    if ref_local is None:
        return None, None
    from spectral_utils.subset_sweep import mask_to_canonical
    canon = int(mask_to_canonical(ref_local, ctx.pool_bits))
    auc = results['auroc'].astype(float)
    finite = np.isfinite(auc)
    idx = np.nonzero(results['mask'] == np.uint64(canon))[0]
    if len(idx) == 0:
        return None, None
    a = auc[idx[0]]
    rank = int((auc[finite] > a).sum()) + 1
    pctile = round(100.0 * (1 - rank / max(finite.sum(), 1)), 2)
    return rank, pctile


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', default=os.path.join(REPO_DIR, 'local_cache'))
    ap.add_argument('--out-dir', default=os.path.join(REPO_DIR, 'results', 'subset_sweep'))
    ap.add_argument('--domains', default=None,
                    help='comma list: math500,gsm8k,gpqa,rag,qa,trace')
    ap.add_argument('--cells', default=None, help='substring filter on cell keys')
    ap.add_argument('--min-size', type=int, default=3)
    ap.add_argument('--max-size', type=int, default=None)
    ap.add_argument('--method', choices=['residual', 'eigengap'], default='residual')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--chunk-size', type=int, default=2000)
    ap.add_argument('--no-resume', dest='resume', action='store_false')
    ap.add_argument('--include-derived', default=None,
                    help='comma list of extra views to fold into the exhaustive '
                         'enumeration (default: extras only via augmentation)')
    ap.add_argument('--with-trace-cells', action='store_true',
                    help='also sweep the self-contained raw-trace cells')
    ap.add_argument('--no-augment', dest='augment', action='store_false',
                    help='skip the extra-view augmentation stage')
    ap.add_argument('--limit', type=int, default=None,
                    help='smoke test: evaluate only the first N subsets per cell')
    ap.add_argument('--calibrate-only', action='store_true')
    ap.add_argument('--self-check', action='store_true',
                    help='run invariant checks on each selected cell, then exit')
    ap.add_argument('--yes', action='store_true',
                    help='skip the post-calibration confirmation')
    args = ap.parse_args()

    domains = args.domains.split(',') if args.domains else None
    cells = args.cells.split(',') if args.cells else None
    include = [x for x in (args.include_derived or '').split(',') if x]
    for x in include:
        if x not in CANONICAL_POOL:
            ap.error(f"--include-derived: unknown view {x!r}")
    pool_req = H16 + include

    os.makedirs(args.out_dir, exist_ok=True)
    dv = os.path.join(args.data_dir, 'derived_views.pkl')
    tc = os.path.join(args.data_dir, 'trace_cells.pkl') if args.with_trace_cells else None
    aug_path = os.path.join(args.out_dir, 'augmentation.pkl')

    todo = []
    for domain, cell_key, fd, labels in iter_cells(
            args.data_dir, domains=domains, cells=cells,
            derived_views_pkl=dv, trace_cells_pkl=tc):
        ctx = prepare_cell(domain, cell_key, fd, labels, feature_pool=pool_req,
                           min_size=args.min_size)
        if ctx is None:
            print(f"[skip] {domain}/{cell_key}: <{args.min_size} usable features")
            continue
        todo.append((ctx, fd))
    if not todo:
        print("No cells to process.")
        return
    print(f"{len(todo)} cells selected; enumeration pool = "
          f"{len(todo[0][0].pool)} features (requested {len(pool_req)})")

    if args.self_check:
        for ctx, fd in todo:
            self_check(ctx, fd, args.method)
        return

    # calibration gate
    total_s = 0.0
    print(f"\n=== Calibration ({args.method}) ===")
    for ctx, _ in todo:
        cal = calibrate(ctx, method=args.method, min_size=args.min_size,
                        max_size=args.max_size)
        n_sub = count_masks(len(ctx.pool), args.min_size,
                            min(args.max_size or len(ctx.pool), len(ctx.pool)))
        if args.limit:
            cal['projected_s'] *= min(1.0, args.limit / max(n_sub, 1))
        total_s += cal['projected_s']
        print(f"  {ctx.domain}/{ctx.cell_key}: n={len(ctx.labels)} p={len(ctx.pool)} "
              f"subsets={min(n_sub, args.limit or n_sub)} "
              f"per-size ms={cal['per_size_ms']} projected={cal['projected_s']:.0f}s")
    print(f"TOTAL projected: {total_s:.0f}s single-core "
          f"≈ {total_s / max(args.workers, 1) / 3600:.1f} h at {args.workers} workers\n")
    if args.calibrate_only:
        return
    if not args.yes:
        resp = input("Proceed with the sweep? [y/N] ").strip().lower()
        if resp != 'y':
            print("Aborted.")
            return

    for i, (ctx, fd) in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] {ctx.domain}/{ctx.cell_key}")
        results, manifest = run_cell_sweep(
            ctx, args.out_dir, method=args.method, min_size=args.min_size,
            max_size=args.max_size, chunk_size=args.chunk_size,
            workers=args.workers, limit=args.limit, resume=args.resume)

        paths = cell_paths(args.out_dir, ctx.domain, ctx.cell_key)
        if 'top20' not in manifest:
            manifest['reference'] = reference_results(ctx, results, method=args.method)
            manifest['top20'] = top20_with_deltas(ctx, results, args.method)
            rank, pctile = good5_rank(ctx, results)
            manifest['good5_rank'], manifest['good5_pctile'] = rank, pctile
            with open(paths['manifest'], 'w') as f:
                json.dump(manifest, f, indent=1)
            t20 = manifest['top20']
            if t20:
                print(f"  best: {t20[0]['auroc']:.3f} {t20[0]['feats']} "
                      f"(K={t20[0]['K']}); GOOD_5 rank {rank} "
                      f"(pctile {pctile})")

        if args.augment:
            aug = {}
            if os.path.exists(aug_path):
                with open(aug_path, 'rb') as f:
                    aug = pickle.load(f)
            key = (ctx.domain, ctx.cell_key)
            if key not in aug:
                t0 = time.time()
                recs = augment_cell(ctx, fd, results, method=args.method)
                aug[key] = recs
                tmp = aug_path + '.tmp'
                with open(tmp, 'wb') as f:
                    pickle.dump(aug, f)
                os.replace(tmp, aug_path)
                views = sorted({r['view'] for r in recs})
                print(f"  augmentation: {len(recs)} records over views {views} "
                      f"({time.time() - t0:.0f}s)")

        path, n = write_summary(args.out_dir)
        print(f"  summary: {n} cells -> {path}")

    print("\nSweep complete.")


if __name__ == '__main__':
    main()
