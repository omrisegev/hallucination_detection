"""
selector_bench — evaluation harness for label-free per-cell feature-subset
selectors (Step 186+, the Jul-2026 "automatic feature-subset selection"
action item; design memo: docs/research_notes/feature_subset_selection_landscape.md).

A SELECTOR is a label-free algorithm that, given one cell's unlabeled feature
matrix, picks a subset of fusion features (and optionally a group assignment,
a K override, or a fusion family). This module provides:

  * UnlabeledCell — the only view of a cell a selector ever receives. It has
    NO labels field and no positive-rate: label leakage into selection is
    structurally impossible, not merely discouraged.
  * npz lookup — the Step-153 exhaustive sweep stored every enumerated
    H16-pool subset's record (results/subset_sweep/<domain>__<cell>.npz), so
    any H16 selection is scored by lookup, with its EXACT percentile within
    the same-size enumeration (= the exact random-subset CDF position, the
    Rajabinasab-2026 random-floor guardrail for free).
  * eval_subset_flex — live scoring for everything the lookup can't cover:
    the 46-pool arm, groups overrides (clustering swap), K overrides
    (rank-test rules), and the U-PCR fusion family.
  * bench_selector — the per-cell loop, incremental CSV, resume-safe.
  * self_check — lookup-vs-live agreement + GOOD_5 reproduction against
    sweep_summary.csv + the no-label-leak assertion.
  * summarize_bench — the single source of the comparison metrics (Stage 3);
    every number in the research note comes from here, none hand-typed.

Honest-evaluation contract (inherited from subset_sweep.py):
  fixed offline feature signs; global sign resolved label-free by
  anchor_orient against the oriented epr anchor; AUROC raw, never
  max(auc, 1-auc); label-peeking ceilings are ceilings, not results.
"""

import csv
import json
import os
import time
import zlib
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import roc_auc_score

from .fusion_utils import lsml_continuous, upcr_fuse
from .streaming_utils import anchor_orient
from .subset_sweep import (
    CANONICAL_POOL, GOOD_5, H16, RECORD_FIELDS,
    CellContext, eval_subset, iter_cells, load_cell_results,
    mask_to_canonical, mask_to_cols, names_to_local_mask, prepare_cell,
    sanitize,
)

BENCH_FIELDS = [
    'selector', 'variant', 'pool_mode', 'domain', 'cell', 'n', 'p_pool',
    'chosen', 'size', 'auroc', 'eval_mode', 'pctile_within_size', 'rand_med',
    'K', 'residual', 'fusion', 'flipped', 'fallback', 'seconds', 'seed',
    'diag_json',
]

# npz columns a selector may see through the label-free cache: everything a
# deployment-time enumeration would compute itself. NEVER 'auroc' (label-
# derived) and never 'eff_w'/'cross_w' (fitted post-hoc; unneeded).
CACHE_FIELDS = ('mask', 'size', 'residual', 'K', 'rho_mean', 'rho_max', 'rho_hi')


# ---------------------------------------------------------------------------
# UnlabeledCell — what a selector is allowed to see
# ---------------------------------------------------------------------------

@dataclass
class UnlabeledCell:
    """Label-free view of one cell. Deliberately has NO labels attribute and
    no positive rate — selectors receive this and only this."""
    domain: str
    cell_key: str
    pool: list             # usable feature names, CANONICAL_POOL order
    pool_bits: np.ndarray  # canonical bit index per pool column (uint8)
    V: np.ndarray          # (n, p) sign-oriented z-scored views
    anchor: np.ndarray     # (n,) oriented z-scored anchor view (label-free)
    anchor_name: str
    rho: np.ndarray        # (p, p) |Spearman| among pool columns
    dropped: dict = field(default_factory=dict)
    n_imputed: int = 0

    @property
    def n(self):
        return self.V.shape[0]

    @property
    def p(self):
        return self.V.shape[1]

    @classmethod
    def from_context(cls, ctx: CellContext):
        return cls(domain=ctx.domain, cell_key=ctx.cell_key, pool=ctx.pool,
                   pool_bits=ctx.pool_bits, V=ctx.V, anchor=ctx.anchor,
                   anchor_name=ctx.anchor_name, rho=ctx.rho,
                   dropped=ctx.dropped, n_imputed=ctx.n_imputed)


# ---------------------------------------------------------------------------
# npz lookup table
# ---------------------------------------------------------------------------

def load_npz_table(npz_dir, domain, cell_key):
    """Load one cell's exhaustive-sweep records, sorted by canonical mask.
    Returns None if the npz is absent (untracked file, main checkout only)."""
    path = os.path.join(npz_dir, f"{sanitize(domain)}__{sanitize(cell_key)}.npz")
    if not os.path.exists(path):
        return None
    rec = load_cell_results(path)
    order = np.argsort(rec['mask'])
    table = {k: rec[k][order] for k in RECORD_FIELDS}
    table['_path'] = path
    by_size = {}
    for s in np.unique(table['size']):
        a = table['auroc'][(table['size'] == s)].astype(float)
        by_size[int(s)] = np.sort(a[np.isfinite(a)])
    table['_auroc_by_size'] = by_size
    return table


def lookup_row(table, canon_mask):
    """Exact-match lookup by canonical mask; None if not enumerated."""
    masks = table['mask']
    i = int(np.searchsorted(masks, np.uint64(canon_mask)))
    if i < len(masks) and masks[i] == np.uint64(canon_mask):
        return {k: table[k][i] for k in RECORD_FIELDS}
    return None


def pctile_within_size(table, size, auroc):
    """Exact percentile of `auroc` among ALL enumerated same-size subsets —
    the enumeration is the full random-subset population, so this is the
    exact random-floor CDF position (mean rank under ties)."""
    arr = table['_auroc_by_size'].get(int(size))
    if arr is None or len(arr) == 0 or not np.isfinite(auroc):
        return float('nan')
    lo = int(np.searchsorted(arr, auroc, side='left'))
    hi = int(np.searchsorted(arr, auroc, side='right'))
    return 100.0 * (0.5 * (lo + hi)) / len(arr)


def selector_cache_from_table(table, pool_bits):
    """The label-free slice of the npz a selector may consult (a cache of
    what a deployment-time enumeration would compute itself). No 'auroc'."""
    if table is None:
        return None
    cache = {k: table[k] for k in CACHE_FIELDS}
    cache['pool_bits'] = np.asarray(pool_bits, dtype=np.uint8)
    return cache


def canonical_mask_to_local_cols(canon_mask, pool_bits):
    """Canonical bitmask -> ascending local column indices, or None if the
    mask uses a canonical bit outside this cell's pool."""
    bit_to_col = {int(b): j for j, b in enumerate(pool_bits)}
    cols = []
    mask = int(canon_mask)
    bit = 0
    while mask:
        if mask & 1:
            if bit not in bit_to_col:
                return None
            cols.append(bit_to_col[bit])
        mask >>= 1
        bit += 1
    return np.asarray(cols, dtype=np.int64)


def cols_to_canonical(cols, pool_bits):
    local = 0
    for j in cols:
        local |= 1 << int(j)
    return mask_to_canonical(local, pool_bits)


# ---------------------------------------------------------------------------
# live evaluation
# ---------------------------------------------------------------------------

def eval_subset_flex(ctx, cols, fusion='lsml', groups=None, K_override=None,
                     method='residual'):
    """
    Live scoring of one subset with the selector-bench extensions:
      fusion='lsml' (default) | 'upcr'; groups= clustering-swap assignment;
      K_override= forced K (rank-test rules). Default lsml with no overrides
      matches subset_sweep.eval_subset numerics exactly (same fusion, same
      label-free anchor orientation, raw AUROC).
    """
    cols = np.asarray(sorted(int(j) for j in cols), dtype=np.int64)
    m = len(cols)
    if m < 3:
        raise ValueError(f"subset size {m} < 3")
    out = {'fusion': fusion, 'K': -1, 'residual': float('nan')}

    if fusion == 'upcr':
        w, rho_hat, g2 = upcr_fuse(ctx.V[:, cols].T)
        fused = ctx.V[:, cols] @ w
    elif fusion == 'lsml':
        kw = {}
        if groups is not None:
            g = np.asarray(groups, dtype=int)
            if len(g) != m:
                raise ValueError(f"groups length {len(g)} != subset size {m}")
            kw['groups'] = g
        elif K_override is not None:
            kw['K_range'] = [int(K_override)]
        fused, meta = lsml_continuous(*[ctx.V[:, j] for j in cols],
                                      method=method, **kw)
        out['K'] = int(meta['K'])
        out['residual'] = float(meta['residual'])
    else:
        raise ValueError(f"unknown fusion {fusion!r}")

    oriented, flipped = anchor_orient(fused, ctx.anchor)
    if np.std(oriented) < 1e-12:
        auroc = float('nan')
    else:
        auroc = float(roc_auc_score(ctx.labels, oriented))
    out.update(auroc=auroc, flipped=bool(flipped), size=m,
               mask=cols_to_canonical(cols, ctx.pool_bits))
    return out


def sample_random_live(ctx, size, R=32, rng=None, max_size_cap=None):
    """Size-matched random-subset floor for live-scored pools (c46 arm):
    R random subsets of `size`, default-lsml scored; returns finite AUROCs."""
    rng = rng or np.random.default_rng(0)
    p = ctx.V.shape[1]
    size = int(min(size, p))
    aurocs = []
    for _ in range(R):
        cols = rng.choice(p, size=size, replace=False)
        rec = eval_subset(ctx.V, ctx.labels, ctx.anchor, ctx.rho,
                          np.sort(cols), ctx.pool_bits)
        a = float(rec['auroc'])
        if np.isfinite(a):
            aurocs.append(a)
    return np.asarray(aurocs)


# ---------------------------------------------------------------------------
# the bench loop
# ---------------------------------------------------------------------------

def _default_paths(data_root):
    data_dir = os.path.join(data_root, 'local_cache')
    return dict(
        data_dir=data_dir,
        derived_views_pkl=os.path.join(data_dir, 'derived_views.pkl'),
        trace_cells_pkl=os.path.join(data_dir, 'trace_cells.pkl'),
    )


# Step-182/184 temperature-mislabel fix: these 4 math500_res.pkl keys say
# "_T1.0" but hold Phase-4 T=1.5 runs. The sweep artifacts (npz/manifests/
# CSVs) were renamed to T1.5 at Step 184; the cache pkl keys were not.
# Canonical mapping: scripts/method_comparison.py::MISLABELED_KEYS (line ~90)
# — keep the two in sync.
MISLABELED_MATH500 = {
    'Qwen2.5-Math-1.5B-Instruct_T1.0': 'Qwen2.5-Math-1.5B-Instruct_T1.5',
    'Qwen-Math-7B_T1.0': 'Qwen-Math-7B_T1.5',
    'deepseek-math-7b-instruct_T1.0': 'deepseek-math-7b-instruct_T1.5',
    'DeepSeek-R1-Distill-Llama-8B_T1.0': 'DeepSeek-R1-Distill-Llama-8B_T1.5',
}


def iter_prepared_cells(data_root, pool_mode='h16', domains=None, cells=None):
    """Yield (ctx, feature_pool_name) over the canonical cell set."""
    paths = _default_paths(data_root)
    pool = H16 if pool_mode == 'h16' else CANONICAL_POOL
    for domain, cell_key, fd, labels in iter_cells(
            paths['data_dir'], domains=domains, cells=cells,
            derived_views_pkl=paths['derived_views_pkl'],
            trace_cells_pkl=paths['trace_cells_pkl']):
        if domain == 'math500' and cell_key in MISLABELED_MATH500:
            cell_key = MISLABELED_MATH500[cell_key]
        ctx = prepare_cell(domain, cell_key, fd, labels, feature_pool=pool)
        if ctx is None:
            print(f"[bench] {domain}/{cell_key}: <3 usable features — skipped")
            continue
        yield ctx


def _cell_rng(seed, domain, cell_key):
    return np.random.default_rng(
        [int(seed), zlib.crc32(f"{domain}/{cell_key}".encode())])


def _existing_keys(out_csv):
    done = set()
    if os.path.exists(out_csv):
        with open(out_csv, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                done.add((row['variant'], row['domain'], row['cell']))
    return done


def bench_selector(selector_name, pool_mode, data_root, npz_dir, out_csv,
                   seed=0, domains=None, cells=None, rand_R=32):
    """
    Run one registered selector family over the cell set; append one CSV row
    per (variant, cell), incrementally (resume-safe: existing rows skipped).

    Selector contract: callable(cell: UnlabeledCell, rng, cache) -> list of
    dicts, each with keys: variant (str), cols (int array into cell.V
    columns), and optionally groups (len(cols) ints), K (int), fusion
    ('lsml'|'upcr'), fallback (bool), diag (JSON-able dict).
    `cache` is the label-free npz slice (selector_cache_from_table) or None;
    selectors must degrade gracefully without it.
    """
    from .selectors import get_selector
    selector = get_selector(selector_name)

    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    done = _existing_keys(out_csv)
    new_file = not os.path.exists(out_csv)
    n_rows = 0

    with open(out_csv, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=BENCH_FIELDS)
        if new_file:
            writer.writeheader()

        for ctx in iter_prepared_cells(data_root, pool_mode, domains, cells):
            table = (load_npz_table(npz_dir, ctx.domain, ctx.cell_key)
                     if pool_mode == 'h16' else None)
            cache = selector_cache_from_table(table, ctx.pool_bits)
            ucell = UnlabeledCell.from_context(ctx)
            rng = _cell_rng(seed, ctx.domain, ctx.cell_key)

            t0 = time.time()
            try:
                selections = selector(ucell, rng, cache=cache)
            except Exception as e:
                print(f"[bench] {ctx.domain}/{ctx.cell_key}: selector raised "
                      f"{type(e).__name__}: {e} — fallback full pool")
                selections = [{'variant': f'{selector_name}.ERROR',
                               'cols': np.arange(ctx.V.shape[1]),
                               'fallback': True,
                               'diag': {'error': f'{type(e).__name__}: {e}'}}]
            sel_seconds = time.time() - t0

            for sel in selections:
                key = (sel['variant'], ctx.domain, ctx.cell_key)
                if key in done:
                    continue
                row = _evaluate_selection(ctx, table, sel, pool_mode,
                                          rand_R=rand_R, rng=rng)
                row.update(selector=selector_name, pool_mode=pool_mode,
                           domain=ctx.domain, cell=ctx.cell_key,
                           n=ctx.V.shape[0], p_pool=ctx.V.shape[1],
                           seconds=round(sel_seconds, 3), seed=seed)
                writer.writerow(row)
                f.flush()
                done.add(key)
                n_rows += 1
            print(f"[bench] {ctx.domain}/{ctx.cell_key}: "
                  f"{len(selections)} variants in {sel_seconds:.1f}s")
    return n_rows


def _evaluate_selection(ctx, table, sel, pool_mode, rand_R=32, rng=None):
    cols = np.asarray(sorted(int(j) for j in sel['cols']), dtype=np.int64)
    fusion = sel.get('fusion', 'lsml')
    groups = sel.get('groups')
    K_over = sel.get('K')
    is_default = fusion == 'lsml' and groups is None and K_over is None

    row = {
        'variant': sel['variant'],
        'chosen': '|'.join(ctx.pool[j] for j in cols),
        'size': len(cols),
        'fusion': fusion,
        'fallback': bool(sel.get('fallback', False)),
        'pctile_within_size': float('nan'),
        'rand_med': float('nan'),
        'diag_json': json.dumps(sel.get('diag', {}), default=str),
    }

    rec = None
    if is_default and table is not None:
        rec = lookup_row(table, cols_to_canonical(cols, ctx.pool_bits))
    if rec is not None:
        row.update(auroc=float(rec['auroc']), eval_mode='lookup',
                   K=int(rec['K']), residual=float(rec['residual']),
                   flipped=bool(rec['flipped']))
    else:
        out = eval_subset_flex(ctx, cols, fusion=fusion, groups=groups,
                               K_override=K_over)
        row.update(auroc=out['auroc'], eval_mode='live', K=out['K'],
                   residual=out['residual'], flipped=out['flipped'])

    # exact percentile only where the enumeration is the comparable
    # population: default-lsml rows on an enumerated pool
    if is_default and table is not None and np.isfinite(row['auroc']):
        row['pctile_within_size'] = pctile_within_size(
            table, len(cols), row['auroc'])
    elif is_default and table is None and np.isfinite(row['auroc']):
        floor = sample_random_live(ctx, len(cols), R=rand_R, rng=rng)
        if len(floor):
            row['rand_med'] = float(np.median(floor))
    return row


# ---------------------------------------------------------------------------
# self-check
# ---------------------------------------------------------------------------

def self_check(data_root, npz_dir, sweep_summary_csv=None, n_pairs=20, seed=0):
    """
    Bench integrity gate (run before trusting any bench number):
      1. UnlabeledCell leaks no labels (structural assertion).
      2. lookup-vs-live: random enumerated subsets re-scored live must match
         the stored float32 AUROC to 1e-6 (and K/flip/residual).
      3. GOOD_5 lookup reproduces sweep_summary.csv's good5_auroc per cell.
    Returns a dict report; raises AssertionError on any failure.
    """
    rng = np.random.default_rng(seed)
    report = {'cells': 0, 'pairs': 0, 'good5_checked': 0, 'good5_max_abs': 0.0}

    good5_ref = {}
    if sweep_summary_csv and os.path.exists(sweep_summary_csv):
        with open(sweep_summary_csv, newline='', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                if r.get('good5_auroc'):
                    good5_ref[(r['domain'], r['cell_key'])] = float(r['good5_auroc'])

    for ctx in iter_prepared_cells(data_root, 'h16'):
        table = load_npz_table(npz_dir, ctx.domain, ctx.cell_key)
        if table is None:
            continue
        report['cells'] += 1
        ucell = UnlabeledCell.from_context(ctx)
        assert not hasattr(ucell, 'labels'), "UnlabeledCell leaks labels"
        assert not hasattr(ucell, 'pos_rate'), "UnlabeledCell leaks pos_rate"

        # a couple of random lookup-vs-live pairs per cell until n_pairs total
        if report['pairs'] < n_pairs:
            p = ctx.V.shape[1]
            for _ in range(2):
                size = int(rng.integers(3, min(6, p) + 1))
                cols = np.sort(rng.choice(p, size=size, replace=False))
                rec = lookup_row(table, cols_to_canonical(cols, ctx.pool_bits))
                assert rec is not None, (
                    f"{ctx.domain}/{ctx.cell_key}: enumerated subset missing "
                    f"from npz — pool misalignment")
                live = eval_subset_flex(ctx, cols)
                stored = float(rec['auroc'])
                if np.isfinite(stored) or np.isfinite(live['auroc']):
                    d = abs(np.float32(live['auroc']) - np.float32(stored))
                    assert d <= 1e-6, (
                        f"{ctx.domain}/{ctx.cell_key} cols={list(cols)}: "
                        f"lookup {stored:.8f} != live {live['auroc']:.8f}")
                    assert int(rec['K']) == live['K'], "K mismatch"
                    assert bool(rec['flipped']) == live['flipped'], "flip mismatch"
                report['pairs'] += 1

        key = (ctx.domain, ctx.cell_key)
        if key in good5_ref and all(fname in ctx.pool for fname in GOOD_5):
            local = names_to_local_mask(GOOD_5, ctx.pool)
            rec = lookup_row(table, mask_to_canonical(local, ctx.pool_bits))
            assert rec is not None, f"{key}: GOOD_5 missing from npz"
            d = abs(float(rec['auroc']) - good5_ref[key])
            report['good5_max_abs'] = max(report['good5_max_abs'], d)
            assert d <= 2e-4, (
                f"{key}: GOOD_5 lookup {float(rec['auroc']):.6f} != "
                f"sweep_summary {good5_ref[key]:.6f}")
            report['good5_checked'] += 1

    assert report['cells'] > 0, "no cells with npz found — check --npz-dir"
    return report


# ---------------------------------------------------------------------------
# summarize_bench — the single source of comparison metrics (Stage 3)
# ---------------------------------------------------------------------------

def summarize_bench(df, comparators):
    """
    df: pandas DataFrame of bench rows (BENCH_FIELDS).
    comparators: per-cell DataFrame with columns
        domain, cell, good5_auroc, oracle_auroc  (h16 basis, from
        sweep_summary.csv).
    Returns a leaderboard DataFrame: one row per (variant, pool_mode), with
    the pre-registered metrics + gate verdicts. Cells with NaN AUROC count as
    losses (never silently dropped).
    """
    import pandas as pd
    from scipy import stats as sps

    df = df.copy()
    df['auroc'] = pd.to_numeric(df['auroc'], errors='coerce')
    merged = df.merge(comparators, on=['domain', 'cell'], how='left')

    rows = []
    for (variant, pool_mode), g in merged.groupby(['variant', 'pool_mode']):
        n_cells = len(g)
        auc = g['auroc'].to_numpy(dtype=float)
        g5 = g['good5_auroc'].to_numpy(dtype=float)
        orc = g['oracle_auroc'].to_numpy(dtype=float)
        pct = pd.to_numeric(g['pctile_within_size'], errors='coerce').to_numpy(dtype=float)
        rg_mask = g['domain'].isin(['rag', 'gpqa']).to_numpy()
        rep_mask = (g['domain'] == 'repgrid').to_numpy()

        auc_f = np.where(np.isfinite(auc), auc, 0.0)   # NaN counts as loss
        both = np.isfinite(auc) & np.isfinite(g5)
        delta = auc_f - np.nan_to_num(g5, nan=0.0)
        d = delta[both]
        try:
            wil_p = float(sps.wilcoxon(d).pvalue) if len(d) >= 6 and np.any(d != 0) else float('nan')
        except ValueError:
            wil_p = float('nan')
        wins = int((d > 1e-12).sum())
        losses = int((d < -1e-12).sum())
        ties = int(len(d) - wins - losses)
        sign_p = (float(sps.binomtest(wins, wins + losses).pvalue)
                  if wins + losses > 0 else float('nan'))

        gap_mask = both & np.isfinite(orc) & (orc - np.nan_to_num(g5, nan=np.inf) >= 0.005)
        gap = np.clip((auc_f - g5)[gap_mask] / (orc - g5)[gap_mask], -1.0, 1.5)
        gap_rg_mask = gap_mask & rg_mask
        gap_rg = np.clip((auc_f - g5)[gap_rg_mask] / (orc - g5)[gap_rg_mask], -1.0, 1.5)

        pct_ok = np.isfinite(pct)
        mean_pct = float(np.mean(pct[pct_ok])) if pct_ok.any() else float('nan')
        frac_above_rand = (float(np.mean(pct[pct_ok] > 50.0))
                           if pct_ok.any() else float('nan'))

        macro = float(np.mean(auc_f))
        macro_rep = float(np.mean(auc_f[rep_mask])) if rep_mask.any() else float('nan')
        g5_macro = float(np.mean(g5[np.isfinite(g5)])) if np.isfinite(g5).any() else float('nan')

        g_floor = (np.isfinite(mean_pct) and mean_pct > 50.0
                   and np.isfinite(frac_above_rand) and frac_above_rand > 0.5)
        d_macro = macro - g5_macro if np.isfinite(g5_macro) else float('nan')
        g_macro = ('SUCCESS' if (np.isfinite(d_macro) and d_macro > 0.01
                                 and np.isfinite(wil_p) and wil_p < 0.05)
                   else 'PASS' if (np.isfinite(d_macro) and d_macro >= -0.005)
                   else 'FAIL')
        g_domain = (bool(len(gap_rg)) and float(np.mean(gap_rg)) >= 0.25)

        rows.append({
            'variant': variant, 'pool_mode': pool_mode, 'n_cells': n_cells,
            'macro_auroc': round(macro, 4),
            'macro_repgrid': round(macro_rep, 4) if np.isfinite(macro_rep) else np.nan,
            'delta_vs_good5': round(d_macro, 4) if np.isfinite(d_macro) else np.nan,
            'wilcoxon_p': round(wil_p, 5) if np.isfinite(wil_p) else np.nan,
            'wins': wins, 'ties': ties, 'losses': losses,
            'sign_p': round(sign_p, 5) if np.isfinite(sign_p) else np.nan,
            'mean_pctile': round(mean_pct, 2) if np.isfinite(mean_pct) else np.nan,
            'frac_cells_above_random': (round(frac_above_rand, 3)
                                        if np.isfinite(frac_above_rand) else np.nan),
            'gap_captured': round(float(np.mean(gap)), 3) if len(gap) else np.nan,
            'gap_captured_rag_gpqa': (round(float(np.mean(gap_rg)), 3)
                                      if len(gap_rg) else np.nan),
            'n_nan_auroc': int((~np.isfinite(auc)).sum()),
            'n_fallback': int(g['fallback'].astype(str).isin(['True', 'true', '1']).sum()),
            'mean_size': round(float(np.mean(g['size'].astype(float))), 2),
            'mean_seconds': round(float(np.mean(g['seconds'].astype(float))), 3),
            'G_floor': 'PASS' if g_floor else ('n/a' if not pct_ok.any() else 'FAIL'),
            'G_macro': g_macro,
            'G_domain': ('PASS' if g_domain else ('n/a' if not len(gap_rg) else 'FAIL')),
        })
    import pandas as pd
    return (pd.DataFrame(rows)
            .sort_values(['pool_mode', 'macro_auroc'], ascending=[True, False])
            .reset_index(drop=True))
