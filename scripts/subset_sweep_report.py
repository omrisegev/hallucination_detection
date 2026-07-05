"""
subset_sweep_report.py — aggregate the subset-sweep artifacts into
results/Subset_Sweep_Report.html + CSVs under results/subset_sweep/.

Sections (all on by default):
  1  Overview (cells, pools, runtime, NaN counts) + table1 CONT cross-check
  2  Per-cell top subsets + pinned GOOD_5 / STABLE_H9 / ALL_H16 references
  3  AUROC-vs-size landscape (percentile bands; per-domain charts, per-cell CSV)
  4  Feature marginal value (top-1% appearance enrichment + marginal lift)
  5  LOCO honest selection + global consensus subset (the citable numbers)
  6  K-census (K vs size, structural K=2 share, co-clustering heatmap)
  7  rho>=0.75 validation (binned AUROC vs rho_max, top-1% violation share)
  8  Weight stability across cells (GOOD_5 + consensus subset)
  9  Method grid on sampled subsets (L-SML vs flat SML vs average vs U-PCR)
 10  Extra-view augmentation (the "can we fuse the pivot signals" answer)
 11  Competitor comparison (recorded numbers, caveats rendered verbatim)
 12  Honesty appendix

Usage:
    python scripts/subset_sweep_report.py [--skip-method-grid] [--with-se-lite]
"""

import argparse
import csv
import glob
import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from sklearn.metrics import roc_auc_score

from spectral_utils.fusion_utils import sml_fuse_signed, upcr_fuse
from spectral_utils.streaming_utils import anchor_orient
from spectral_utils.subset_sweep import (
    CANONICAL_POOL, GOOD_5, H16, RECORD_FIELDS, RHO_FILTER,
    canonical_to_names, iter_cells, load_cell_results, mask_to_cols,
    names_to_local_mask, prepare_cell, unpack_assignment,
)

CHART_JS = 'https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js'

# Recorded competitor numbers — Phase 12 Corrected (Step 152, PROGRESS.md).
# CAVEAT (must stay attached wherever these are shown): 4 open issues before
# these are citable — MATH-500 L-SML sign flip (0.230 = 0.770 flipped, fusion
# invalid), RAG SelfCheckGPT below chance, SE drops vs old Phase 12
# (NLI-truncation suspect). Old Phase-12 numbers are intentionally NOT listed.
PHASE12_CORRECTED = {
    ('gsm8k', 'Llama-8B_T1.0'): {
        'SCGPT-official K=5': 0.701, 'D-SE K=10': 0.614, 'LW-SE K=10': 0.613,
        'SC K=10': 0.608, 'SCGPT-hard K=5': 0.601,
    },
    ('math500', 'Qwen-Math-7B_T1.0'): {
        'SC K=10': 0.863, 'D-SE K=10': 0.630, 'LW-SE K=10': 0.625,
        'SCGPT-hard K=5': 0.549, 'SCGPT-official K=5': 0.593,
    },
    ('gpqa', 'Qwen-7B_T1.0'): {
        'SCGPT-official K=5': 0.512, 'D-SE K=10': 0.504, 'LW-SE K=10': 0.501,
        'SC K=10': 0.504, 'VC': 0.428,
    },
    ('rag', 'Qwen-7B/hotpotqa'): {'SCGPT-hard K=5': 0.317, 'SCGPT-official K=5': 0.243},
    ('rag', 'Qwen-7B/natural-questions'): {'SCGPT-hard K=5': 0.393, 'SCGPT-official K=5': 0.322},
    ('rag', 'Qwen-7B/2wikimultihopqa'): {'SCGPT-hard K=5': 0.354, 'SCGPT-official K=5': 0.306},
    ('rag', 'Qwen-7B/narrativeqa'): {'SCGPT-hard K=5': 0.477, 'SCGPT-official K=5': 0.442},
}
PHASE12_CAVEAT = ("Phase 12 Corrected (Step 152) numbers - pending reconciliation: "
                  "MATH-500 L-SML sign flip unresolved, RAG SelfCheckGPT below "
                  "chance, SE drops vs old Phase 12 (NLI truncation suspect). "
                  "Shown for orientation only, not citable yet.")
LAPEIGVALS = {('gsm8k', 'Llama-8B_T1.0'): 0.720}  # literature, unsupervised (Step 22)

SE_LITE_FILES = {
    ('gsm8k', 'Llama-8B_T1.0'): 'raw_traces/p1_gsm8k_llama8b_k10.pkl',
    ('math500', 'Qwen-Math-7B_T1.0'): 'raw_traces/p4_math500_qwen7b_k10.pkl',
}


def popcount(arr):
    a = np.asarray(arr, dtype=np.uint64)
    try:
        return np.bitwise_count(a)
    except AttributeError:
        return np.array([int(x).bit_count() for x in a], dtype=np.int64)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_all(sweep_dir):
    cells = []
    for mpath in sorted(glob.glob(os.path.join(sweep_dir, '*.manifest.json'))):
        with open(mpath) as f:
            m = json.load(f)
        npz = mpath.replace('.manifest.json', '.npz')
        if 'sweep_seconds' not in m or not os.path.exists(npz):
            continue
        cells.append({'domain': m['domain'], 'cell_key': m['cell_key'],
                      'manifest': m, 'results': load_cell_results(npz)})
    return cells


def cached_cells(cells):
    return [c for c in cells if not c['manifest'].get('trace_derived')]


def finite_auc(res):
    a = res['auroc'].astype(float)
    return a, np.isfinite(a)


def _union_cols(rows):
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    return cols


def write_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=_union_cols(rows), restval='')
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Section computations
# ---------------------------------------------------------------------------

def sec_overview(cells, sweep_dir):
    rows = []
    for c in cells:
        m, res = c['manifest'], c['results']
        a, fin = finite_auc(res)
        rows.append({
            'domain': m['domain'], 'cell': m['cell_key'], 'n': m['n'],
            'pos_rate': round(m['n_pos'] / m['n'], 3),
            'pool': len(m['pool']), 'dropped': ';'.join(m['dropped']) or '-',
            'subsets': m['n_subsets'], 'nan_auroc': int((~fin).sum()),
            'K1_fallback': int((res['K'] == 1).sum()),
            'anchor': m['anchor_name'], 'method': m['method'],
            'seconds': m['sweep_seconds'],
        })
    write_csv(os.path.join(sweep_dir, 'overview.csv'), rows)
    return rows


def table1_crosscheck(cells):
    """Sweep GOOD_5 AUROC vs the method_comparison table1 CONT column."""
    path = os.path.join(REPO_DIR, 'results', 'method_comparison_table1.csv')
    if not os.path.exists(path):
        return []
    t1 = {}
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            v = (row.get('CONT') or '').strip().rstrip('%')
            if v:
                t1[(row['domain'], row['cell_key'])] = float(v) / 100.0
    out = []
    for c in cached_cells(cells):
        key = (c['domain'], c['cell_key'])
        ref = (c['manifest'].get('reference') or {}).get('GOOD_5') or {}
        sweep = ref.get('auroc')
        if key not in t1 or sweep is None:
            continue
        cont = t1[key]
        direct = abs(sweep - cont)
        flippd = abs((1 - sweep) - cont)
        out.append({'domain': key[0], 'cell': key[1],
                    'sweep_good5': round(sweep, 4), 'table1_CONT': round(cont, 4),
                    'match': 'direct' if direct <= flippd else '1-x',
                    'abs_diff': round(min(direct, flippd), 4)})
    return out


def sec_landscape(cells, sweep_dir):
    pct = [5, 25, 50, 75, 95]
    rows = []
    per_domain = defaultdict(lambda: defaultdict(list))  # domain -> size -> [medians...]
    for c in cells:
        a, fin = finite_auc(c['results'])
        sizes = c['results']['size']
        for s in np.unique(sizes):
            sel = fin & (sizes == s)
            if not sel.any():
                continue
            vals = a[sel]
            qs = np.percentile(vals, pct)
            row = {'domain': c['domain'], 'cell': c['cell_key'], 'size': int(s),
                   'n_subsets': int(sel.sum()), 'max': round(float(vals.max()), 4)}
            row.update({f'p{p}': round(float(q), 4) for p, q in zip(pct, qs)})
            rows.append(row)
            if not c['manifest'].get('trace_derived'):
                per_domain[c['domain']][int(s)].append(
                    {f'p{p}': float(q) for p, q in zip(pct, qs)} | {'max': float(vals.max())})
    write_csv(os.path.join(sweep_dir, 'landscape.csv'), rows)

    charts = {}
    for dom, by_size in per_domain.items():
        sizes = sorted(by_size)
        charts[dom] = {
            'sizes': sizes,
            'p25': [round(float(np.mean([r['p25'] for r in by_size[s]])), 4) for s in sizes],
            'p50': [round(float(np.mean([r['p50'] for r in by_size[s]])), 4) for s in sizes],
            'p75': [round(float(np.mean([r['p75'] for r in by_size[s]])), 4) for s in sizes],
            'max': [round(float(np.mean([r['max'] for r in by_size[s]])), 4) for s in sizes],
        }
    return rows, charts


def sec_feature_value(cells, sweep_dir):
    """Top-1% appearance enrichment + size-controlled marginal lift per feature."""
    per_cell = []
    for c in cells:
        res = c['results']
        a, fin = finite_auc(res)
        masks = res['mask']
        if fin.sum() < 100:
            continue
        thr = np.percentile(a[fin], 99)
        top = fin & (a >= thr)
        bot = fin & (a <= np.percentile(a[fin], 50))
        for f in c['manifest']['pool']:
            b = CANONICAL_POOL.index(f)
            has = ((masks >> np.uint64(b)) & np.uint64(1)).astype(bool)
            lift = []
            for s in np.unique(res['size']):
                sel = fin & (res['size'] == s)
                w, wo = sel & has, sel & ~has
                if w.any() and wo.any():
                    lift.append(a[w].mean() - a[wo].mean())
            per_cell.append({
                'domain': c['domain'], 'cell': c['cell_key'], 'feature': f,
                'rate_top1pct': round(float(has[top].mean()), 4) if top.any() else None,
                'rate_bottom50': round(float(has[bot].mean()), 4) if bot.any() else None,
                'rate_all': round(float(has[fin].mean()), 4),
                'marginal_lift': round(float(np.mean(lift)), 4) if lift else None,
                'single_auroc': round((c['manifest'].get('reference', {})
                                       .get('_singles', {}).get(f, float('nan'))), 4),
            })
    write_csv(os.path.join(sweep_dir, 'feature_value.csv'), per_cell)

    agg = defaultdict(lambda: {'top': [], 'lift': [], 'single': []})
    for r in per_cell:
        if r['rate_top1pct'] is not None:
            agg[r['feature']]['top'].append(r['rate_top1pct'])
        if r['marginal_lift'] is not None:
            agg[r['feature']]['lift'].append(r['marginal_lift'])
        if r['single_auroc'] is not None and np.isfinite(r['single_auroc']):
            agg[r['feature']]['single'].append(r['single_auroc'])
    table = [{'feature': f,
              'mean_rate_top1pct': round(float(np.mean(v['top'])), 3) if v['top'] else None,
              'mean_marginal_lift': round(float(np.mean(v['lift'])), 4) if v['lift'] else None,
              'mean_single_auroc': round(float(np.mean(v['single'])), 3) if v['single'] else None,
              'n_cells': len(v['top'])}
             for f, v in agg.items()]
    table.sort(key=lambda r: -(r['mean_rate_top1pct'] or 0))
    return table


def _mask_matrix(cells):
    """(universe_masks, M) — M[i, j] = AUROC of mask i on cell j (NaN if absent)."""
    cc = cached_cells(cells)
    counts = defaultdict(int)
    for c in cc:
        for m in np.unique(c['results']['mask']):
            counts[int(m)] += 1
    need = int(np.ceil(0.9 * len(cc)))
    universe = np.array(sorted(m for m, k in counts.items() if k >= need),
                        dtype=np.uint64)
    M = np.full((len(universe), len(cc)), np.nan)
    for j, c in enumerate(cc):
        masks, a = c['results']['mask'], c['results']['auroc'].astype(float)
        order = np.argsort(masks)
        pos = np.searchsorted(masks[order], universe)
        ok = (pos < len(masks)) & (masks[order][np.minimum(pos, len(masks) - 1)] == universe)
        M[ok, j] = a[order][pos[ok]]
    return universe, M, cc


def sec_loco(cells, sweep_dir):
    universe, M, cc = _mask_matrix(cells)
    n_cells = len(cc)
    enough = np.isfinite(M).sum(axis=1) >= int(np.ceil(0.9 * n_cells))
    rows, loco_aucs = [], []
    consensus_idx = None
    if enough.any():
        mean_all = np.nanmean(M[enough], axis=1)
        consensus_idx = np.flatnonzero(enough)[int(np.nanargmax(mean_all))]
    for j, c in enumerate(cc):
        others = np.delete(np.arange(n_cells), j)
        valid = enough & np.isfinite(M[:, j])
        if not valid.any() or not len(others):
            continue
        with np.errstate(invalid='ignore'):
            mean_others = np.nanmean(M[np.ix_(valid, others)], axis=1)
        if np.all(np.isnan(mean_others)):
            continue
        best_local = np.flatnonzero(valid)[int(np.nanargmax(mean_others))]
        loco_auc = float(M[best_local, j])
        oracle = float(np.nanmax(M[np.isfinite(M[:, j]), j]))
        ref = c['manifest'].get('reference', {})
        good5 = (ref.get('GOOD_5') or {}).get('auroc')
        all16 = (ref.get('ALL_H16') or {}).get('auroc')
        rows.append({
            'domain': c['domain'], 'cell': c['cell_key'],
            'loco_auroc': round(loco_auc, 4),
            'loco_subset': '|'.join(canonical_to_names(universe[best_local])),
            'good5': round(good5, 4) if good5 is not None else None,
            'all16': round(all16, 4) if all16 is not None else None,
            'consensus_auroc': (round(float(M[consensus_idx, j]), 4)
                                if consensus_idx is not None and np.isfinite(M[consensus_idx, j])
                                else None),
            'oracle_best_LABEL_PEEKING_CEILING': round(oracle, 4),
        })
        loco_aucs.append(loco_auc)
    write_csv(os.path.join(sweep_dir, 'loco.csv'), rows)
    consensus = (canonical_to_names(universe[consensus_idx])
                 if consensus_idx is not None else [])
    good5_vals = [r['good5'] for r in rows if r['good5'] is not None]
    oracle_vals = [r['oracle_best_LABEL_PEEKING_CEILING'] for r in rows]
    summary = {
        'n_candidate_masks': int(enough.sum()),
        'consensus_subset': consensus,
        'consensus_macro': (round(float(np.nanmean(M[consensus_idx])), 4)
                            if consensus_idx is not None else None),
        'loco_macro': round(float(np.mean(loco_aucs)), 4) if loco_aucs else None,
        'good5_macro': round(float(np.mean(good5_vals)), 4) if good5_vals else None,
        'oracle_macro_CEILING': (round(float(np.mean(oracle_vals)), 4)
                                 if oracle_vals else None),
    }
    return rows, summary


def sec_k_census(cells, sweep_dir):
    by_size = defaultdict(lambda: defaultdict(int))
    k2_structural, k2_total, k1_total, total = 0, 0, 0, 0
    for c in cells:
        res = c['results']
        for s in np.unique(res['size']):
            sel = res['size'] == s
            for k, cnt in zip(*np.unique(res['K'][sel], return_counts=True)):
                by_size[int(s)][int(k)] += int(cnt)
        k2 = res['K'] == 2
        k2_total += int(k2.sum())
        total += len(res['K'])
        k1_total += int((res['K'] == 1).sum())
        cw = np.abs(res['cross_w'][k2][:, :2].astype(float))
        if len(cw):
            k2_structural += int((np.abs(cw - 0.5) < 0.02).all(axis=1).sum())

    # co-clustering frequency over the 16 base features (cached cells)
    pair_same = np.zeros((len(H16), len(H16)))
    pair_n = np.zeros_like(pair_same)
    for c in cached_cells(cells):
        res, pool = c['results'], c['manifest']['pool']
        masks, assign = res['mask'], res['assign']
        for ia, fa in enumerate(H16):
            if fa not in pool:
                continue
            ba = CANONICAL_POOL.index(fa)
            for ib in range(ia + 1, len(H16)):
                fb = H16[ib]
                if fb not in pool:
                    continue
                bb = CANONICAL_POOL.index(fb)
                both = (((masks >> np.uint64(ba)) & np.uint64(1)) &
                        ((masks >> np.uint64(bb)) & np.uint64(1))).astype(bool)
                if not both.any():
                    continue
                sub_masks = masks[both]
                idx_a = popcount(sub_masks & np.uint64((1 << ba) - 1))
                idx_b = popcount(sub_masks & np.uint64((1 << bb) - 1))
                ga = (assign[both] >> (np.uint64(3) * idx_a.astype(np.uint64))) & np.uint64(7)
                gb = (assign[both] >> (np.uint64(3) * idx_b.astype(np.uint64))) & np.uint64(7)
                pair_same[ia, ib] += float((ga == gb).mean())
                pair_n[ia, ib] += 1
    with np.errstate(invalid='ignore'):
        cocluster = np.where(pair_n > 0, pair_same / np.maximum(pair_n, 1), np.nan)

    rows = [{'size': s, **{f'K{k}': v for k, v in sorted(ks.items())}}
            for s, ks in sorted(by_size.items())]
    write_csv(os.path.join(sweep_dir, 'k_census.csv'), rows)
    return {
        'by_size': {s: dict(ks) for s, ks in sorted(by_size.items())},
        'k1_rate': round(k1_total / max(total, 1), 4),
        'k2_rate': round(k2_total / max(total, 1), 4),
        'k2_structural_rate': round(k2_structural / max(k2_total, 1), 4),
        'cocluster': cocluster,
    }


def sec_rho_check(cells, sweep_dir):
    bins = [0.0, 0.25, 0.5, RHO_FILTER, 1.001]
    labels = ['<0.25', '0.25-0.5', f'0.5-{RHO_FILTER}', f'>={RHO_FILTER}']
    rows = []
    viol_top, viol_all = [], []
    for c in cells:
        res = c['results']
        a, fin = finite_auc(res)
        rho_max = res['rho_max'].astype(float)
        hi = res['rho_hi'].astype(int) > 0
        thr = np.percentile(a[fin], 99) if fin.sum() > 100 else np.inf
        top = fin & (a >= thr)
        if top.any():
            viol_top.append(float(hi[top].mean()))
            viol_all.append(float(hi[fin].mean()))
        which = np.digitize(rho_max, bins) - 1
        for b, lab in enumerate(labels):
            sel = fin & (which == b)
            if sel.sum() < 10:
                continue
            rows.append({'domain': c['domain'], 'cell': c['cell_key'],
                         'rho_bin': lab, 'n': int(sel.sum()),
                         'mean_auroc': round(float(a[sel].mean()), 4),
                         'p95_auroc': round(float(np.percentile(a[sel], 95)), 4)})
    write_csv(os.path.join(sweep_dir, 'rho_check.csv'), rows)

    agg = defaultdict(lambda: {'mean': [], 'p95': []})
    for r in rows:
        agg[r['rho_bin']]['mean'].append(r['mean_auroc'])
        agg[r['rho_bin']]['p95'].append(r['p95_auroc'])
    table = [{'rho_bin': lab,
              'mean_auroc': round(float(np.mean(agg[lab]['mean'])), 4),
              'p95_auroc': round(float(np.mean(agg[lab]['p95'])), 4),
              'n_cells': len(agg[lab]['mean'])}
             for lab in labels if agg[lab]['mean']]
    summary = {
        'share_with_violating_pair_top1pct': round(float(np.mean(viol_top)), 4) if viol_top else None,
        'share_with_violating_pair_all': round(float(np.mean(viol_all)), 4) if viol_all else None,
    }
    return table, summary


def sec_weight_stability(cells, sweep_dir, consensus):
    from sklearn.metrics import adjusted_rand_score
    out = []
    for name, feats in (('GOOD_5', GOOD_5), ('consensus', consensus)):
        if not feats or len(feats) < 3:
            continue
        canon = 0
        for f in feats:
            canon |= 1 << CANONICAL_POOL.index(f)
        weights, assigns = [], []
        for c in cached_cells(cells):
            res, pool = c['results'], c['manifest']['pool']
            if any(f not in pool for f in feats):
                continue
            idx = np.nonzero(res['mask'] == np.uint64(canon))[0]
            if not len(idx):
                continue
            row = idx[0]
            cols = [pool.index(f) for f in sorted(feats, key=pool.index)]
            weights.append(res['eff_w'][row].astype(float)[cols])
            assigns.append(unpack_assignment(res['assign'][row], len(feats)))
        if len(weights) < 3:
            continue
        W = np.vstack(weights)
        names_sorted = sorted(feats, key=lambda f: CANONICAL_POOL.index(f))
        aris = [adjusted_rand_score(assigns[i], assigns[j])
                for i in range(len(assigns)) for j in range(i + 1, len(assigns))]
        for k, f in enumerate(names_sorted):
            col = W[:, k]
            modal_sign = 1 if (col > 0).sum() >= (col < 0).sum() else -1
            out.append({
                'subset': name, 'feature': f, 'n_cells': len(col),
                'mean_abs_w': round(float(np.abs(col).mean()), 4),
                'cv_abs_w': round(float(np.abs(col).std() / max(np.abs(col).mean(), 1e-9)), 3),
                'sign_agreement': round(float((np.sign(col) == modal_sign).mean()), 3),
                'mean_pairwise_ARI': round(float(np.mean(aris)), 3),
            })
    write_csv(os.path.join(sweep_dir, 'weight_stability.csv'), out)
    return out


def sec_method_grid(cells, data_dir, sweep_dir, method, n_random=200, seed=42):
    """L-SML (from sweep) vs flat SML / simple average / U-PCR on sampled subsets."""
    ctxs = {}
    for domain, cell_key, fd, labels in iter_cells(
            data_dir, derived_views_pkl=os.path.join(data_dir, 'derived_views.pkl')):
        ctxs[(domain, cell_key)] = (fd, labels)
    rng = np.random.default_rng(seed)
    rows = []
    for c in cached_cells(cells):
        key = (c['domain'], c['cell_key'])
        if key not in ctxs:
            continue
        fd, labels = ctxs[key]
        ctx = prepare_cell(*key, fd, labels)
        if ctx is None or ctx.pool != c['manifest']['pool']:
            continue
        res = c['results']
        a, fin = finite_auc(res)
        pick = set(np.flatnonzero(fin)[np.argsort(a[fin])[::-1][:20]].tolist())
        g5 = names_to_local_mask([f for f in GOOD_5 if f in ctx.pool], ctx.pool)
        if g5 is not None:
            from spectral_utils.subset_sweep import mask_to_canonical
            canon = int(mask_to_canonical(g5, ctx.pool_bits))
            hit = np.nonzero(res['mask'] == np.uint64(canon))[0]
            pick.update(hit.tolist())
        pick.update(rng.choice(np.flatnonzero(fin), size=min(n_random, int(fin.sum())),
                               replace=False).tolist())
        for row in sorted(pick):
            names = [f for f in canonical_to_names(int(res['mask'][row])) if f in ctx.pool]
            cols = mask_to_cols(names_to_local_mask(names, ctx.pool))
            V = ctx.V[:, cols]
            flat, _ = sml_fuse_signed(*[V[:, i] for i in range(V.shape[1])])
            flat, _ = anchor_orient(flat, ctx.anchor)
            avg = V.mean(axis=1)
            try:
                w, _, _ = upcr_fuse(V.T)
                up = w @ V.T
                up, _ = anchor_orient(up, ctx.anchor)
                up_auc = (float(roc_auc_score(labels, up))
                          if np.std(up) > 1e-12 else np.nan)
            except Exception:
                up_auc = np.nan
            rows.append({
                'domain': c['domain'], 'cell': c['cell_key'],
                'size': int(res['size'][row]),
                'lsml': float(a[row]),
                'flat': (float(roc_auc_score(labels, flat))
                         if np.std(flat) > 1e-12 else np.nan),
                'avg': (float(roc_auc_score(labels, avg))
                        if np.std(avg) > 1e-12 else np.nan),
                'upcr': up_auc,
            })
    write_csv(os.path.join(sweep_dir, 'method_grid.csv'),
              [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
               for r in rows])
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for m in ('flat', 'avg', 'upcr'):
            if np.isfinite(r['lsml']) and np.isfinite(r[m]):
                agg[r['domain']][m].append(r['lsml'] - r[m])
    table = []
    for dom, deltas in sorted(agg.items()):
        entry = {'domain': dom, 'n': len(deltas['flat'])}
        for m in ('flat', 'avg', 'upcr'):
            d = np.array(deltas[m])
            entry[f'lsml_minus_{m}_pp'] = round(float(d.mean()) * 100, 2) if len(d) else None
            entry[f'winrate_vs_{m}'] = round(float((d > 0).mean()), 3) if len(d) else None
        table.append(entry)
    return table


def sec_augmentation(sweep_dir):
    path = os.path.join(sweep_dir, 'augmentation.pkl')
    if not os.path.exists(path):
        return [], []
    with open(path, 'rb') as f:
        aug = pickle.load(f)
    flat = [r for recs in aug.values() for r in recs]
    rows = [{k: (round(v, 4) if isinstance(v, float) else
                 ('|'.join(v) if isinstance(v, list) and k == 'base_feats' else v))
             for k, v in r.items() if k != 'delta_ci'} |
            {'delta_ci_lo': round(r['delta_ci'][0], 4) if r.get('delta_ci') else None,
             'delta_ci_hi': round(r['delta_ci'][1], 4) if r.get('delta_ci') else None}
            for r in flat]
    write_csv(os.path.join(sweep_dir, 'augmentation.csv'), rows)

    agg = defaultdict(lambda: {'d': [], 'sig+': 0, 'sig-': 0, 'cells': set(),
                               'vauc': [], 'rho': []})
    for r in flat:
        if r.get('delta') is None:
            continue
        v = agg[r['view']]
        v['d'].append(r['delta'])
        v['cells'].add((r['domain'], r['cell_key']))
        v['vauc'].append(r['view_auroc'])
        if r.get('rho_view_anchor') is not None:
            v['rho'].append(r['rho_view_anchor'])
        if r.get('delta_ci'):
            if r['delta_ci'][0] > 0:
                v['sig+'] += 1
            elif r['delta_ci'][1] < 0:
                v['sig-'] += 1
    table = [{'view': view,
              'n_cells': len(v['cells']), 'n_bases': len(v['d']),
              'mean_delta_pp': round(float(np.mean(v['d'])) * 100, 2),
              'sig_pos': v['sig+'], 'sig_neg': v['sig-'],
              'mean_view_auroc': round(float(np.mean(v['vauc'])), 3),
              'mean_rho_vs_anchor': round(float(np.mean(v['rho'])), 3) if v['rho'] else None}
             for view, v in sorted(agg.items())]
    table.sort(key=lambda r: -r['mean_delta_pp'])
    return table, rows


def se_lite_rows(data_dir):
    """D-SE-lite: entropy of the normalized-answer distribution over K samples.
    EXPERIMENTAL — K=10 caches carry their own question subsets/labels."""
    out = []
    for (domain, cell_key), rel in SE_LITE_FILES.items():
        path = os.path.join(data_dir, rel)
        if not os.path.exists(path):
            continue
        with open(path, 'rb') as f:
            cache = pickle.load(f)
        if isinstance(cache, dict):
            cache = [cache[k] for k in sorted(cache)]
        ses, labels = [], []
        for s in cache:
            if not isinstance(s, dict):
                continue
            answers = s.get('answers')
            corr = s.get('correct')
            if not answers or corr is None:
                continue
            norm = [str(x).strip().lower() for x in answers]
            _, counts = np.unique(norm, return_counts=True)
            p = counts / counts.sum()
            ses.append(-float(np.sum(p * np.log(p + 1e-12))))
            labels.append(int(bool(corr[0] if isinstance(corr, (list, tuple)) else corr)))
        if len(set(labels)) < 2:
            continue
        auc = roc_auc_score(labels, -np.asarray(ses))  # higher SE -> wrong
        out.append({'domain': domain, 'cell': cell_key,
                    'method': 'D-SE-lite (local recompute, experimental)',
                    'auroc': round(float(auc), 4), 'n': len(labels)})
    return out


def sec_competitors(cells, loco_rows, sweep_dir, se_rows=()):
    lr = {}
    lr_path = os.path.join(REPO_DIR, 'results', 'logistic_oracle.pkl')
    if os.path.exists(lr_path):
        with open(lr_path, 'rb') as f:
            for r in pickle.load(f):
                dom, _, key = r['cell'].partition('/')
                lr[(dom, key)] = r
    loco = {(r['domain'], r['cell']): r for r in loco_rows}
    rows = []
    for c in cached_cells(cells):
        key = (c['domain'], c['cell_key'])
        if key not in PHASE12_CORRECTED and key not in LAPEIGVALS:
            continue
        ref = c['manifest'].get('reference', {})
        lrow = loco.get(key, {})
        row = {
            'domain': key[0], 'cell': key[1],
            'LSML_LOCO': lrow.get('loco_auroc'),
            'LSML_GOOD5': (ref.get('GOOD_5') or {}).get('auroc'),
            'epr_single': (ref.get('_anchor_single') or {}).get('auroc'),
            'avg_pool': (ref.get('_avg_pool') or {}).get('auroc'),
            'LR16_oracle_supervised': (round(float(lr[key][f'lr_16']), 4)
                                       if key in lr and lr[key].get('lr_16') is not None
                                       else None),
            'LapEigvals_lit': LAPEIGVALS.get(key),
        }
        for name, auc in PHASE12_CORRECTED.get(key, {}).items():
            row[name] = auc
        rows.append(row)
    for r in se_rows:
        rows.append({'domain': r['domain'], 'cell': r['cell'],
                     r['method']: r['auroc']})
    write_csv(os.path.join(sweep_dir, 'competitors.csv'), rows)
    return rows


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

CSS = """
body{font-family:Segoe UI,Arial,sans-serif;margin:24px auto;max-width:1240px;color:#1a1a2e}
h1{border-bottom:3px solid #16213e}h2{color:#16213e;border-bottom:1px solid #ccc;margin-top:40px}
table{border-collapse:collapse;margin:12px 0;font-size:13px}
th,td{border:1px solid #ccc;padding:4px 8px;text-align:right}
th{background:#16213e;color:#fff;position:sticky;top:0}
td:first-child,td:nth-child(2),th:first-child,th:nth-child(2){text-align:left}
.caveat{background:#fff3cd;border:1px solid #ffc107;padding:10px;border-radius:4px;margin:10px 0}
.note{color:#555;font-size:13px}.mono{font-family:Consolas,monospace;font-size:12px}
.chartbox{width:560px;height:340px;display:inline-block;margin:8px}
.heat td{width:30px;height:22px;font-size:10px;text-align:center;padding:1px}
"""


def esc(x):
    return (str(x).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))


def html_table(rows, cols=None, fmt=None):
    if not rows:
        return '<p class="note">no data</p>'
    cols = cols or _union_cols(rows)
    fmt = fmt or {}
    h = ['<table><tr>' + ''.join(f'<th>{esc(c)}</th>' for c in cols) + '</tr>']
    for r in rows:
        tds = []
        for c in cols:
            v = r.get(c)
            if v is None:
                v = '-'
            elif c in fmt:
                v = fmt[c](v)
            tds.append(f'<td>{esc(v)}</td>')
        h.append('<tr>' + ''.join(tds) + '</tr>')
    h.append('</table>')
    return '\n'.join(h)


def heatmap_html(matrix, labels):
    h = ['<table class="heat"><tr><th></th>' +
         ''.join(f'<th title="{esc(l)}">{esc(l[:7])}</th>' for l in labels) + '</tr>']
    for i, l in enumerate(labels):
        tds = [f'<th style="text-align:left">{esc(l[:16])}</th>']
        for j in range(len(labels)):
            v = matrix[i, j] if j > i else (matrix[j, i] if i > j else np.nan)
            if np.isfinite(v):
                c = int(255 * (1 - v))
                tds.append(f'<td style="background:rgb(255,{c},{c})" '
                           f'title="{labels[i]}|{labels[j]}={v:.2f}">{v:.2f}</td>')
            else:
                tds.append('<td>-</td>')
        h.append('<tr>' + ''.join(tds) + '</tr>')
    h.append('</table>')
    return '\n'.join(h)


def landscape_charts_html(charts):
    boxes, js = [], []
    for i, (dom, d) in enumerate(sorted(charts.items())):
        boxes.append(f'<div class="chartbox"><canvas id="land{i}"></canvas></div>')
        js.append(f"""
new Chart(document.getElementById('land{i}'), {{type:'line',
 data:{{labels:{d['sizes']},datasets:[
  {{label:'max',data:{d['max']},borderColor:'#e63946',pointRadius:2}},
  {{label:'p75',data:{d['p75']},borderColor:'#457b9d',fill:'+1',backgroundColor:'rgba(69,123,157,.18)',pointRadius:0}},
  {{label:'median',data:{d['p50']},borderColor:'#1d3557',pointRadius:2}},
  {{label:'p25',data:{d['p25']},borderColor:'#457b9d',pointRadius:0}}]}},
 options:{{plugins:{{title:{{display:true,text:'{esc(dom)} — AUROC vs subset size (mean over cells)'}}}},
  scales:{{y:{{min:0.4,max:1.0}},x:{{title:{{display:true,text:'subset size'}}}}}},animation:false}}}});""")
    return '\n'.join(boxes), '\n'.join(js)


def build_html(out_path, S):
    top20_rows = []
    for c in S['cells']:
        m = c['manifest']
        for t in (m.get('top20') or [])[:5]:
            top20_rows.append({
                'domain': m['domain'], 'cell': m['cell_key'], 'rank': t['rank'],
                'subset': '|'.join(t['feats']), 'size': t['size'],
                'auroc': round(t['auroc'], 4),
                'ci': f"[{t['ci'][0]:.3f},{t['ci'][1]:.3f}]", 'K': t['K'],
                'd_vs_good5_pp': (round(t['delta_vs_good5'][0] * 100, 1)
                                  if t.get('delta_vs_good5') else None),
            })
        ref = m.get('reference', {})
        for name in ('GOOD_5', 'STABLE_H9', 'ALL_H16'):
            r = ref.get(name) or {}
            if r.get('auroc') is None:
                continue
            top20_rows.append({
                'domain': m['domain'], 'cell': m['cell_key'],
                'rank': name + (f" (rank {m.get('good5_rank')}, pctile {m.get('good5_pctile')})"
                                if name == 'GOOD_5' else ''),
                'subset': '|'.join(r['available']), 'size': len(r['available']),
                'auroc': round(r['auroc'], 4),
                'ci': f"[{r['ci'][0]:.3f},{r['ci'][1]:.3f}]", 'K': r.get('K'),
                'd_vs_good5_pp': None,
            })
    land_boxes, land_js = landscape_charts_html(S['charts'])

    parts = [f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>L-SML Subset Sweep Report</title><style>{CSS}</style>
<script src="{CHART_JS}"></script></head><body>
<h1>Exhaustive L-SML Feature-Subset Sweep</h1>
<p class="note">Continuous L-SML on every feature subset (sizes 3..pool) per cell.
Global sign resolved label-free (anchor = oriented epr). AUROC is raw — never
max(auc, 1-auc). In-cell "best" over the whole enumeration is a
<b>selection ceiling</b>; the honest numbers are the LOCO / consensus rows (section 5).
Full detail in the CSVs next to this file under results/subset_sweep/.</p>

<h2>1 — Overview</h2>
{html_table(S['overview'])}
<h3>Cross-check vs method_comparison_table1 CONT (GOOD_5)</h3>
<p class="note">table1 used best-orientation AUROC; the sweep is raw anchor-oriented —
"1-x" match means the anchor chose the opposite global sign on that cell.</p>
{html_table(S['crosscheck'])}

<h2>2 — Top subsets per cell (top-5 shown; top-20 in top20 CSV / manifests)</h2>
{html_table(top20_rows)}

<h2>3 — AUROC-vs-size landscape</h2>
{land_boxes}
<script>{land_js}</script>

<h2>4 — Feature marginal value (Bracha Q1)</h2>
<p class="note">rate_top1pct = how often the feature appears in the top-1% subsets of a
cell (symmetric enumeration makes rates directly comparable); marginal_lift =
mean AUROC difference with-vs-without the feature at fixed subset size.</p>
{html_table(S['feature_value'])}

<h2>5 — Honest selection (LOCO + consensus)</h2>
<p class="note">LOCO: subset chosen by macro AUROC over the OTHER cells, evaluated on the
held-out cell — no test-cell information used in selection. Consensus subset:
<b>{esc('|'.join(S['loco_summary'].get('consensus_subset', [])) or '-')}</b>.
The oracle column is the label-peeking ceiling (best of the full enumeration on
the same cell) — never cite it as method performance.</p>
{html_table([S['loco_summary']])}
{html_table(S['loco'])}

<h2>6 — K-census</h2>
<p class="note">K=2 cross-weights are structurally 0.50/0.50 (share below); adaptive
weighting only exists for K&ge;3. K=1 = clustering-failure fallback.</p>
{html_table([{'k1_rate': S['k_census']['k1_rate'], 'k2_rate': S['k_census']['k2_rate'],
              'k2_structural_0.5_share': S['k_census']['k2_structural_rate']}])}
{html_table([{'size': s, **{f'K{k}': v for k, v in ks.items()}}
             for s, ks in S['k_census']['by_size'].items()])}
<h3>Co-clustering frequency (P(same group) when both features in a subset)</h3>
{heatmap_html(S['k_census']['cocluster'], H16)}

<h2>7 — rho &ge; {RHO_FILTER} validation</h2>
{html_table(S['rho_table'])}
{html_table([S['rho_summary']])}

<h2>8 — Weight stability across cells</h2>
{html_table(S['weight_stability'])}
"""]
    if S.get('method_grid'):
        parts.append(f"""
<h2>9 — Method grid on sampled subsets (GOOD_5 + top-20 + 200 random per cell)</h2>
{html_table(S['method_grid'])}""")
    parts.append(f"""
<h2>10 — Extra-view augmentation (pivot signals as fusable views)</h2>
<p class="note">Each extra view v is added to reference + top-20 subsets: delta =
AUROC(S&cup;{{v}}) - AUROC(S), paired bootstrap. sig_pos / sig_neg count bases whose
95% CI excludes 0. Anomaly views are transductive but label-free; temporal views
exist only on the raw-trace cells.</p>
{html_table(S['augmentation'])}

<h2>11 — Competitor comparison</h2>
<div class="caveat"><b>Caveat:</b> {esc(PHASE12_CAVEAT)}</div>
<p class="note">LSML_LOCO is the honest selection number (section 5). LR16 oracle is
SUPERVISED (balanced 5-fold CV; SUPERVISED_ORACLE_CORRECTION.md). LapEigvals is a
literature number. Empty = not available for that cell.</p>
{html_table(S['competitors'])}

<h2>12 — Honesty appendix</h2>
<ul class="note">
<li><b>Label-free:</b> feature signs (fixed offline), z-scoring, L-SML fusion,
anchor orientation, LOCO/consensus subset selection, anomaly/temporal views.</li>
<li><b>Label-dependent (reporting only):</b> AUROC evaluation, in-cell top-K tables,
the oracle ceiling column, feature marginal-value stats.</li>
<li><b>Multiple comparisons:</b> the in-cell maximum over {esc(S['n_subsets_typical'])}
subsets is an extreme order statistic; on small cells its inflation is large.
Only LOCO/consensus numbers are selection-honest.</li>
<li>No max(auc, 1-auc) anywhere; global-sign flips are recorded per subset
('flipped') and GOOD_5 anchor-vs-table1 disagreements are listed in section 1.</li>
</ul>
<p class="note">Generated by scripts/subset_sweep_report.py.</p>
</body></html>""")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep-dir', default=os.path.join(REPO_DIR, 'results', 'subset_sweep'))
    ap.add_argument('--data-dir', default=os.path.join(REPO_DIR, 'local_cache'))
    ap.add_argument('--out', default=os.path.join(REPO_DIR, 'results', 'Subset_Sweep_Report.html'))
    ap.add_argument('--method', default='residual')
    ap.add_argument('--skip-method-grid', action='store_true')
    ap.add_argument('--with-se-lite', action='store_true')
    args = ap.parse_args()

    cells = load_all(args.sweep_dir)
    if not cells:
        print(f"No finished cells in {args.sweep_dir}")
        return
    print(f"Loaded {len(cells)} finished cells "
          f"({len(cached_cells(cells))} cached, "
          f"{len(cells) - len(cached_cells(cells))} trace-derived)")

    S = {'cells': cells}
    S['overview'] = sec_overview(cells, args.sweep_dir)
    S['crosscheck'] = table1_crosscheck(cells)
    _, S['charts'] = sec_landscape(cells, args.sweep_dir)
    S['feature_value'] = sec_feature_value(cells, args.sweep_dir)
    S['loco'], S['loco_summary'] = sec_loco(cells, args.sweep_dir)
    S['k_census'] = sec_k_census(cells, args.sweep_dir)
    S['rho_table'], S['rho_summary'] = sec_rho_check(cells, args.sweep_dir)
    S['weight_stability'] = sec_weight_stability(
        cells, args.sweep_dir, S['loco_summary'].get('consensus_subset', []))
    S['method_grid'] = ([] if args.skip_method_grid else
                        sec_method_grid(cells, args.data_dir, args.sweep_dir, args.method))
    S['augmentation'], _ = sec_augmentation(args.sweep_dir)
    se_rows = se_lite_rows(args.data_dir) if args.with_se_lite else []
    S['competitors'] = sec_competitors(cells, S['loco'], args.sweep_dir, se_rows)
    S['n_subsets_typical'] = int(np.median([c['manifest']['n_subsets'] for c in cells]))

    build_html(args.out, S)
    print(f"Report -> {args.out}")
    print(f"CSVs   -> {args.sweep_dir}")
    if S['loco_summary']:
        print(f"LOCO macro {S['loco_summary'].get('loco_macro')} | "
              f"GOOD_5 macro {S['loco_summary'].get('good5_macro')} | "
              f"oracle ceiling {S['loco_summary'].get('oracle_macro_CEILING')} | "
              f"consensus {S['loco_summary'].get('consensus_subset')}")


if __name__ == '__main__':
    main()
