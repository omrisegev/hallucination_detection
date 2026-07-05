"""
subset_sweep — exhaustive L-SML feature-subset enumeration over cached cells.

For every feature subset of sizes min_size..max_size on a cell, runs continuous
L-SML fusion (lsml_continuous) on pre-oriented z-scored views and records:
AUROC of the anchor-oriented fused score, the detected clustering (K, group
assignment), per-feature effective weights, and within-subset |Spearman| stats.

Honest-evaluation contract (SUPERVISED_ORACLE_CORRECTION.md + Step 148):
  * feature orientation = fixed offline signs (FEATURE_SIGNS / EXTRA_VIEW_SIGNS);
  * the fused score's global sign is resolved LABEL-FREE by anchor_orient
    against the cell's oriented `epr` view (subset-independent anchor);
  * AUROC is raw — never max(auc, 1-auc);
  * the in-cell best-of-all-subsets is a selection ceiling, not a result.
    Honest selection (LOCO / consensus) lives in scripts/subset_sweep_report.py.

Effective-weight convention (continuous L-SML only):
  lsml_continuous is exactly linear in its input views:
      fused = sum_g cross_w[g] * (X[:, idx_g] @ w_g)
  so w_eff[i] = within-group weight * its group's cross weight reproduces the
  fused score exactly (checkable invariant). If anchor_orient flips the score,
  w_eff and cross_w are negated to describe the ORIENTED score. Stored weights
  are L1-normalized. This identity does NOT hold for binary L-SML (np.sign of
  virtual classifiers breaks linearity) — this module is continuous-only.

Storage: one dict-of-arrays npz per cell (schema in RECORD_FIELDS), written in
resumable chunks. Masks are uint64 bitmasks over CANONICAL_POOL (fixed order),
comparable across cells regardless of per-cell feature availability.
"""

import itertools
import json
import math
import os
import pickle
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from .feature_utils import FEAT_NAMES, extract_all_features, compute_spilled_energy_features
from .fusion_utils import zscore, boot_auc, lsml_continuous
from .streaming_utils import FEATURE_SIGNS, anchor_orient, iter_trace_records
from .temporal_models import (
    fit_gaussian_hmm, hmm_trace_scores, bocpd_gaussian,
    ar_innovation_scores, kalman_innovation_scores,
)

# ---------------------------------------------------------------------------
# Canonical feature registry
# ---------------------------------------------------------------------------

# Extra views beyond the 20 FEAT_NAMES. All docstring-verified as
# "higher = more suspect" -> sign -1 under the FEATURE_SIGNS convention.
TEMPORAL_VIEWS = [
    'bocpd_ecp', 'bocpd_ecp_spilled', 'bocpd_mean_p0',
    'hmm_occupancy', 'hmm_tail_occupancy', 'hmm_switch_rate',
    'ar_mse_innov', 'ar_innov_ratio', 'kalman_mse_innov', 'kalman_nis',
]
ANOMALY_VIEWS = ['mahalanobis', 'gmm_nll', 'kde_nll', 'iforest', 'ae', 'prae']
EXTRA_VIEWS = TEMPORAL_VIEWS + ANOMALY_VIEWS
EXTRA_VIEW_SIGNS = {v: -1 for v in EXTRA_VIEWS}

# Bit i of a canonical mask <-> CANONICAL_POOL[i]. Frozen order: never reorder
# or insert — only append. Resume validation compares this list via manifests.
CANONICAL_POOL = list(FEAT_NAMES) + EXTRA_VIEWS
assert len(CANONICAL_POOL) <= 64, "canonical masks are uint64"

ALL_SIGNS = {**FEATURE_SIGNS, **EXTRA_VIEW_SIGNS}

# Default exhaustive-enumeration pool: the 16 H(n) features present in every
# cached cell. Spilled/temporal/anomaly views enter via the augmentation stage
# (or --include-derived) — folding them all in multiplies the enumeration ~64x.
H16 = list(FEAT_NAMES[:16])

GOOD_5 = ['epr', 'low_band_power', 'sw_var_peak', 'cusum_max', 'spectral_entropy']
STABLE_H9 = [
    'epr', 'low_band_power', 'high_band_power', 'hl_ratio',
    'spectral_centroid', 'sw_var_peak', 'rpdi', 'pe_mean', 'cusum_max',
]
REFERENCE_SUBSETS = {'GOOD_5': GOOD_5, 'STABLE_H9': STABLE_H9, 'ALL_H16': H16}

# Anchor for label-free global-sign resolution (Step 148). First pool feature
# with non-degenerate variance wins; oriented epr is the documented choice.
ANCHOR_PRIORITY = ['epr', 'low_band_power', 'spectral_entropy', 'cusum_max']

MAX_K = 8            # detect_dependent_groups caps K_range at min(m, 9) - 1
MAX_SUBSET_SIZE = 21  # 3 bits/member * 21 members = 63 bits in a uint64
RHO_FILTER = 0.75    # the L-SML dependence-filter threshold (recorded, not enforced)

PKL_NAMES = {
    'math500': 'math500_res.pkl',
    'gsm8k':   'gsm8k_res.pkl',
    'gpqa':    'gpqa_res.pkl',
    'rag':     'rag_feats_all.pkl',
    'qa':      'qa_res.pkl',
}

RECORD_FIELDS = ('mask', 'size', 'auroc', 'flipped', 'K', 'residual',
                 'assign', 'cross_w', 'eff_w', 'rho_mean', 'rho_max', 'rho_hi')


# ---------------------------------------------------------------------------
# Cache loading (canonical home of the loader; method_comparison.py keeps its
# own copy untouched)
# ---------------------------------------------------------------------------

def load_cached_feats(pkl_path):
    """Load feature pkl; handles {'feats': ...} wrapper and bare dict."""
    if not os.path.exists(pkl_path):
        return None
    with open(pkl_path, 'rb') as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and 'feats' in obj:
        return obj['feats']
    return obj


def is_saturated(arr, threshold: float = 0.40) -> bool:
    """True if more than `threshold` of the values equal the median."""
    a = np.asarray(arr, dtype=float)
    return float(np.mean(a == np.median(a))) > threshold


def iter_cells(data_dir, domains=None, cells=None,
               derived_views_pkl=None, trace_cells_pkl=None):
    """
    Yield (domain, cell_key, feats_dict, labels) over the local caches.

    Skips single-class cells (AUROC undefined). If derived_views_pkl exists,
    its per-cell anomaly views are merged into feats_dict (sample-aligned by
    construction — computed from the same cached feature matrix). If
    trace_cells_pkl exists, its self-contained trace-derived cells are yielded
    under domain 'trace' (different sample sets — never joined onto cached
    cells, never averaged into the cached-cell macro).
    """
    derived = {}
    if derived_views_pkl and os.path.exists(derived_views_pkl):
        with open(derived_views_pkl, 'rb') as f:
            derived = pickle.load(f)

    for domain, fname in PKL_NAMES.items():
        if domains and domain not in domains:
            continue
        d = load_cached_feats(os.path.join(data_dir, fname))
        if d is None:
            print(f"[iter_cells] {domain}: {fname} missing — skipped")
            continue
        for cell_key, (fd, labels) in d.items():
            if cells and not any(c in cell_key for c in cells):
                continue
            labels = np.asarray(labels, dtype=int)
            if len(np.unique(labels)) < 2:
                print(f"[iter_cells] {domain}/{cell_key}: single-class labels — skipped")
                continue
            fd = dict(fd)
            for name, arr in derived.get((domain, cell_key), {}).items():
                if arr is not None and len(arr) == len(labels):
                    fd[name] = arr
            yield domain, cell_key, fd, labels

    if trace_cells_pkl and os.path.exists(trace_cells_pkl):
        if domains and 'trace' not in domains:
            return
        with open(trace_cells_pkl, 'rb') as f:
            tc = pickle.load(f)
        for cell_key, (fd, labels) in tc.items():
            if cells and not any(c in cell_key for c in cells):
                continue
            labels = np.asarray(labels, dtype=int)
            if len(np.unique(labels)) < 2:
                print(f"[iter_cells] trace/{cell_key}: single-class labels — skipped")
                continue
            yield 'trace', cell_key, fd, labels


# ---------------------------------------------------------------------------
# Derived views (Stage 0 computations)
# ---------------------------------------------------------------------------

def compute_anomaly_views(fd, labels, feat_list=None, seed=42):
    """
    Anomaly-scorer views over a cell's feature matrix (all label-free,
    transductive: fit on the unlabeled cell, same status as L-SML's covariance).

    Returns {view_name: np.ndarray(n) or None}; ae/prae are None below
    AE_MIN_SAMPLES or if torch is unavailable.
    """
    from .anomaly_utils import build_feature_matrix, TRACKA_METHODS, AE_MIN_SAMPLES
    n = len(labels)
    X, available = build_feature_matrix(fd, feat_list or H16, n)
    out = {}
    if X is None:
        return {name: None for name in ANOMALY_VIEWS}
    name_map = {'mahalanobis': 'maha', 'gmm_nll': 'gmm2', 'kde_nll': 'kde',
                'iforest': 'iforest', 'ae': 'ae', 'prae': 'prae'}
    for view, key in name_map.items():
        if key in ('ae', 'prae') and n < AE_MIN_SAMPLES:
            out[view] = None
            continue
        try:
            res = TRACKA_METHODS[key](X)
            scores = res[0] if isinstance(res, tuple) else res
            out[view] = None if scores is None else np.asarray(scores, dtype=float)
        except Exception as e:
            print(f"  [anomaly] {view} failed: {type(e).__name__}: {e}")
            out[view] = None
    return out


def compute_trace_views(cache_obj, hazard_lambda=100.0, min_len=8):
    """
    Build a self-contained cell from a raw-trace cache: re-extract the 16
    H-features from each trace (guaranteeing sample alignment with the
    temporal views) plus BOCPD / HMM / AR / Kalman views, spilled-energy
    features and BOCPD-on-ΔE(n) where the cache carries token_spilled.

    Returns (feats_dict, labels) or None if fewer than 10 usable traces.
    """
    kept = []  # (ents, spilled_or_None, label)
    for rec in iter_trace_records(cache_obj):
        if len(rec['ents']) < min_len:
            continue
        fd1 = extract_all_features(rec['ents'])
        if fd1 is None:
            continue
        kept.append((rec['ents'], rec['spilled'], rec['label'], fd1))
    if len(kept) < 10:
        return None

    rows = []
    for ents, spilled, label, fd1 in kept:
        b = bocpd_gaussian(ents, hazard_lambda=hazard_lambda)
        fd1['bocpd_ecp'] = b['ecp']
        fd1['bocpd_mean_p0'] = b['mean_p0']
        ar = ar_innovation_scores(ents)
        fd1['ar_mse_innov'] = ar['mse_innov']
        fd1['ar_innov_ratio'] = ar['innov_ratio']
        ka = kalman_innovation_scores(ents)
        fd1['kalman_mse_innov'] = ka['mse_innov']
        fd1['kalman_nis'] = ka['nis']
        if spilled is not None:
            fd1.update(compute_spilled_energy_features(spilled))
            fd1['bocpd_ecp_spilled'] = bocpd_gaussian(
                spilled, hazard_lambda=hazard_lambda)['ecp']
        rows.append(fd1)

    obs_list = [ents for ents, *_ in kept]
    params = fit_gaussian_hmm(obs_list, n_states=2)
    hmm = hmm_trace_scores(params, obs_list)
    for i, fd1 in enumerate(rows):
        fd1['hmm_occupancy'] = float(hmm['occupancy'][i])
        fd1['hmm_tail_occupancy'] = float(hmm['tail_occupancy'][i])
        fd1['hmm_switch_rate'] = float(hmm['switch_rate'][i])

    labels = np.array([lbl for _, _, lbl, _ in kept], dtype=int)
    keys = sorted({k for fd1 in rows for k in fd1})
    feats_dict = {}
    for k in keys:
        if all(k in fd1 for fd1 in rows):  # drop views absent for some traces
            feats_dict[k] = np.array([fd1[k] for fd1 in rows], dtype=float)
    return feats_dict, labels


# ---------------------------------------------------------------------------
# Per-cell preparation
# ---------------------------------------------------------------------------

@dataclass
class CellContext:
    domain: str
    cell_key: str
    pool: list            # usable feature names, CANONICAL_POOL order
    pool_bits: np.ndarray  # canonical bit index per pool column (uint8)
    V: np.ndarray          # (n, p) float64 sign-oriented z-scored views
    anchor: np.ndarray     # (n,) oriented z-scored anchor view (label-free)
    anchor_name: str
    labels: np.ndarray     # (n,) int — used ONLY for AUROC
    rho: np.ndarray        # (p, p) float64 |Spearman| among pool columns
    dropped: dict = field(default_factory=dict)  # name -> reason
    n_imputed: int = 0


def prepare_cell(domain, cell_key, fd, labels, feature_pool=None,
                 sat_threshold=0.40, min_size=3):
    """
    Build a CellContext: filter to usable features, orient + z-score exactly as
    lsml_continuous_pipeline does (fusion_utils.py:625-629), precompute the
    full |Spearman| matrix and the anchor view. Returns None if < min_size
    usable features or single-class labels.
    """
    labels = np.asarray(labels, dtype=int)
    if len(np.unique(labels)) < 2:
        return None
    requested = [f for f in (feature_pool or H16) if f in CANONICAL_POOL]

    pool, dropped, columns, n_imputed = [], {}, [], 0
    for f in requested:
        if f not in fd:
            dropped[f] = 'missing'
            continue
        arr = np.asarray(fd[f], dtype=float)
        if len(arr) != len(labels):
            dropped[f] = 'length-mismatch'
            continue
        bad = ~np.isfinite(arr)
        if bad.any():
            if bad.all():
                dropped[f] = 'all-nonfinite'
                continue
            arr = arr.copy()
            arr[bad] = np.median(arr[~bad])
            n_imputed += int(bad.sum())
        if float(arr.std()) < 1e-8:
            dropped[f] = 'constant'
            continue
        if is_saturated(arr, sat_threshold):
            dropped[f] = 'saturated'
            continue
        pool.append(f)
        columns.append(zscore(arr * ALL_SIGNS.get(f, +1)))
    if len(pool) < min_size:
        return None

    anchor, anchor_name = None, None
    for f in ANCHOR_PRIORITY:
        if f in fd:
            a = np.asarray(fd[f], dtype=float)
            if len(a) == len(labels) and np.isfinite(a).all() and a.std() > 1e-8:
                anchor = zscore(a * ALL_SIGNS.get(f, +1))
                anchor_name = f
                break
    if anchor is None:
        anchor, anchor_name = columns[0], pool[0]

    V = np.column_stack(columns)
    # spearmanr(V)[0] is the (p, p) matrix; [0] indexing works on both the old
    # tuple return and the new SignificanceResult
    rho = spearmanr(V)[0] if V.shape[1] > 2 else np.corrcoef(V.T)
    rho = np.abs(np.nan_to_num(np.atleast_2d(rho), nan=0.0))

    return CellContext(
        domain=domain, cell_key=cell_key, pool=pool,
        pool_bits=np.array([CANONICAL_POOL.index(f) for f in pool], dtype=np.uint8),
        V=V, anchor=anchor, anchor_name=anchor_name, labels=labels, rho=rho,
        dropped=dropped, n_imputed=n_imputed,
    )


# ---------------------------------------------------------------------------
# Mask utilities
# ---------------------------------------------------------------------------

def count_masks(p, min_size=3, max_size=None):
    max_size = min(max_size or p, p)
    return sum(math.comb(p, s) for s in range(min_size, max_size + 1))


def enumerate_masks(p, min_size=3, max_size=None):
    """
    All local masks (bit j = pool column j) size-major then lexicographic.
    THIS ORDER IS FROZEN — chunk indices and resume depend on it.
    """
    max_size = min(max_size or p, p)
    if max_size > MAX_SUBSET_SIZE:
        raise ValueError(
            f"max_size={max_size} exceeds MAX_SUBSET_SIZE={MAX_SUBSET_SIZE} "
            f"(uint64 assignment packing limit) — restrict the pool or max size")
    out = np.empty(count_masks(p, min_size, max_size), dtype=np.uint64)
    i = 0
    for s in range(min_size, max_size + 1):
        for comb in itertools.combinations(range(p), s):
            m = 0
            for j in comb:
                m |= 1 << j
            out[i] = m
            i += 1
    return out


def mask_to_cols(mask):
    """Local mask -> ascending np.ndarray of pool column indices."""
    mask = int(mask)
    return np.array([j for j in range(mask.bit_length()) if mask >> j & 1],
                    dtype=np.int64)


def mask_to_canonical(local_mask, pool_bits):
    c = 0
    for j in mask_to_cols(local_mask):
        c |= 1 << int(pool_bits[j])
    return np.uint64(c)


def canonical_to_names(mask):
    mask = int(mask)
    return [CANONICAL_POOL[i] for i in range(mask.bit_length()) if mask >> i & 1]


def names_to_local_mask(names, pool):
    """Local mask for the given feature names, or None if any is not in pool."""
    m = 0
    for f in names:
        if f not in pool:
            return None
        m |= 1 << pool.index(f)
    return np.uint64(m)


def pack_assignment(c):
    """
    Pack a group-assignment array as 3 bits per subset member (member order =
    ascending pool columns), after relabeling groups by first appearance so the
    encoding is invariant to SpectralClustering's arbitrary label permutation.
    Returns (packed uint64, relabel dict old_label -> new_label).
    """
    mapping, packed = {}, 0
    for i, g in enumerate(np.asarray(c, dtype=int)):
        if g not in mapping:
            mapping[g] = len(mapping)
        packed |= mapping[g] << (3 * i)
    return np.uint64(packed), mapping


def unpack_assignment(packed, m):
    packed = int(packed)
    return np.array([(packed >> (3 * i)) & 0b111 for i in range(m)], dtype=int)


# ---------------------------------------------------------------------------
# Per-subset evaluation
# ---------------------------------------------------------------------------

def fuse_subset(V, anchor, cols, method='residual'):
    """lsml_continuous on precomputed views + label-free orientation.

    Returns (oriented_scores, flipped, meta)."""
    fused, meta = lsml_continuous(*[V[:, j] for j in cols], method=method)
    oriented, flipped = anchor_orient(fused, anchor)
    return oriented, flipped, meta


def compose_effective_weights(meta, m):
    """
    Raw (un-normalized) per-feature effective weights of the fused score:
    w_eff[i] = within-group weight * its group's cross weight. Groups are
    iterated in np.unique(c) order, matching lsml_continuous's construction.
    Satisfies X @ w_eff == fused exactly (continuous mode, pre-flip).
    """
    w_eff = np.zeros(m)
    cross = np.asarray(meta['cross_weights'], dtype=float)
    for g_pos, (idx, w) in enumerate(meta['group_weights']):
        w_eff[idx] = np.asarray(w, dtype=float) * cross[g_pos]
    return w_eff


def eval_subset(V, labels, anchor, rho, cols, pool_bits, method='residual'):
    """Evaluate one subset; returns a dict matching RECORD_FIELDS."""
    m = len(cols)
    oriented, flipped, meta = fuse_subset(V, anchor, cols, method=method)

    if oriented.std() < 1e-12:
        auroc = np.nan
    else:
        auroc = roc_auc_score(labels, oriented)

    sgn = -1.0 if flipped else 1.0
    w_eff = compose_effective_weights(meta, m) * sgn
    l1 = float(np.sum(np.abs(w_eff)))
    if l1 > 0:
        w_eff = w_eff / l1

    packed, mapping = pack_assignment(meta['c'])
    cross = np.full(MAX_K, np.nan)
    uniq = np.unique(np.asarray(meta['c'], dtype=int))
    cw = np.asarray(meta['cross_weights'], dtype=float) * sgn
    for g_pos, g in enumerate(uniq):
        cross[mapping[g]] = cw[g_pos]

    sub_rho = rho[np.ix_(cols, cols)][np.triu_indices(m, k=1)]
    local_mask = 0
    for j in cols:
        local_mask |= 1 << int(j)

    eff_full = np.full(V.shape[1], np.nan)
    eff_full[cols] = w_eff
    return {
        'mask': mask_to_canonical(local_mask, pool_bits),
        'size': np.uint8(m),
        'auroc': np.float32(auroc),
        'flipped': np.uint8(flipped),
        'K': np.uint8(meta['K']),
        'residual': np.float32(meta['residual']),
        'assign': packed,
        'cross_w': cross.astype(np.float16),
        'eff_w': eff_full.astype(np.float16),
        'rho_mean': np.float16(sub_rho.mean() if m > 1 else 0.0),
        'rho_max': np.float16(sub_rho.max() if m > 1 else 0.0),
        'rho_hi': np.uint8(int((sub_rho >= RHO_FILTER).sum())),
    }


# ---------------------------------------------------------------------------
# Chunked parallel runner (Windows-spawn safe: top-level fns + initializer)
# ---------------------------------------------------------------------------

_W = {}  # per-worker state set by _worker_init


def _worker_init(V, labels, anchor, rho, pool_bits, method):
    _W.update(V=V, labels=labels, anchor=anchor, rho=rho,
              pool_bits=pool_bits, method=method)


def evaluate_chunk(args):
    """Worker entry: (chunk_idx, local masks uint64 array) -> (chunk_idx, arrays)."""
    chunk_idx, masks = args
    n, p = len(masks), _W['V'].shape[1]
    out = {
        'mask': np.empty(n, np.uint64), 'size': np.empty(n, np.uint8),
        'auroc': np.empty(n, np.float32), 'flipped': np.empty(n, np.uint8),
        'K': np.empty(n, np.uint8), 'residual': np.empty(n, np.float32),
        'assign': np.empty(n, np.uint64),
        'cross_w': np.empty((n, MAX_K), np.float16),
        'eff_w': np.empty((n, p), np.float16),
        'rho_mean': np.empty(n, np.float16), 'rho_max': np.empty(n, np.float16),
        'rho_hi': np.empty(n, np.uint8),
    }
    for i, mask in enumerate(masks):
        rec = eval_subset(_W['V'], _W['labels'], _W['anchor'], _W['rho'],
                          mask_to_cols(mask), _W['pool_bits'], _W['method'])
        for k in RECORD_FIELDS:
            out[k][i] = rec[k]
    return chunk_idx, out


def sanitize(name):
    return re.sub(r'[^A-Za-z0-9_.-]', '_', str(name))


def cell_paths(out_dir, domain, cell_key):
    base = os.path.join(out_dir, f"{sanitize(domain)}__{sanitize(cell_key)}")
    return {'final': base + '.npz', 'manifest': base + '.manifest.json',
            'chunk_dir': os.path.join(base + '_chunks')}


def write_chunk(chunk_dir, chunk_idx, arrays):
    path = os.path.join(chunk_dir, f'chunk_{chunk_idx:05d}.npz')
    tmp = path + '.tmp.npz'
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _valid_chunk(chunk_dir, chunk_idx, expected_rows):
    path = os.path.join(chunk_dir, f'chunk_{chunk_idx:05d}.npz')
    if not os.path.exists(path):
        return False
    try:
        with np.load(path) as z:
            return len(z['mask']) == expected_rows and all(k in z for k in RECORD_FIELDS)
    except Exception:
        return False


def merge_chunks(chunk_dir, n_chunks, final_path):
    parts = []
    for k in range(n_chunks):
        with np.load(os.path.join(chunk_dir, f'chunk_{k:05d}.npz')) as z:
            parts.append({key: z[key] for key in RECORD_FIELDS})
    merged = {key: np.concatenate([p[key] for p in parts]) for key in RECORD_FIELDS}
    tmp = final_path + '.tmp.npz'
    np.savez_compressed(tmp, **merged)
    os.replace(tmp, final_path)
    return merged


def load_cell_results(final_path):
    with np.load(final_path) as z:
        return {key: z[key] for key in RECORD_FIELDS}


def _versions():
    import scipy
    import sklearn
    rev = ''
    try:
        rev = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        pass
    return {'python': sys.version.split()[0], 'numpy': np.__version__,
            'scipy': scipy.__version__, 'sklearn': sklearn.__version__,
            'git_rev': rev}


def build_manifest(ctx, method, min_size, max_size, chunk_size, n_subsets):
    return {
        'domain': ctx.domain, 'cell_key': ctx.cell_key,
        'pool': ctx.pool, 'pool_bits': ctx.pool_bits.tolist(),
        'n': int(len(ctx.labels)), 'n_pos': int(ctx.labels.sum()),
        'anchor_name': ctx.anchor_name, 'dropped': ctx.dropped,
        'n_imputed': ctx.n_imputed,
        'method': method, 'min_size': min_size,
        'max_size': int(min(max_size or len(ctx.pool), len(ctx.pool))),
        'chunk_size': chunk_size, 'n_subsets': int(n_subsets),
        'n_chunks': int(math.ceil(n_subsets / chunk_size)),
        'trace_derived': ctx.domain == 'trace',
        'versions': _versions(), 'created': time.strftime('%Y-%m-%d %H:%M:%S'),
    }


_RESUME_KEYS = ('pool', 'method', 'min_size', 'max_size', 'chunk_size', 'n_subsets')


def run_cell_sweep(ctx, out_dir, method='residual', min_size=3, max_size=None,
                   chunk_size=2000, workers=1, limit=None, resume=True,
                   log=print):
    """
    Full enumeration for one cell with chunked checkpoints + resume.
    Returns (results dict-of-arrays, manifest dict).
    """
    os.makedirs(out_dir, exist_ok=True)
    paths = cell_paths(out_dir, ctx.domain, ctx.cell_key)
    p = len(ctx.pool)

    masks = enumerate_masks(p, min_size, max_size)
    if limit:
        masks = masks[:limit]
    manifest = build_manifest(ctx, method, min_size, max_size, chunk_size, len(masks))

    if resume and os.path.exists(paths['final']) and os.path.exists(paths['manifest']):
        with open(paths['manifest']) as f:
            old = json.load(f)
        if all(old.get(k) == manifest[k] for k in _RESUME_KEYS):
            log(f"  [{ctx.domain}/{ctx.cell_key}] final npz exists — skipping")
            return load_cell_results(paths['final']), old
        raise RuntimeError(
            f"{paths['final']} exists with different parameters "
            f"({ {k: old.get(k) for k in _RESUME_KEYS if old.get(k) != manifest[k]} }); "
            f"delete it or match the original flags")

    if os.path.exists(paths['manifest']) and resume:
        with open(paths['manifest']) as f:
            old = json.load(f)
        if not all(old.get(k) == manifest[k] for k in _RESUME_KEYS):
            raise RuntimeError(
                f"in-progress manifest {paths['manifest']} does not match current "
                f"parameters — delete the *_chunks dir + manifest or match flags")
        manifest = old
    else:
        with open(paths['manifest'], 'w') as f:
            json.dump(manifest, f, indent=1)

    os.makedirs(paths['chunk_dir'], exist_ok=True)
    n_chunks = manifest['n_chunks']
    chunk_rows = lambda k: len(masks[k * chunk_size:(k + 1) * chunk_size])
    pending = [k for k in range(n_chunks)
               if not (resume and _valid_chunk(paths['chunk_dir'], k, chunk_rows(k)))]
    log(f"  [{ctx.domain}/{ctx.cell_key}] p={p} subsets={len(masks)} "
        f"chunks={n_chunks} pending={len(pending)} method={method} workers={workers}")

    t0 = time.time()
    if pending:
        if workers <= 1:
            _worker_init(ctx.V, ctx.labels, ctx.anchor, ctx.rho, ctx.pool_bits, method)
            for done, k in enumerate(pending, 1):
                _, arrays = evaluate_chunk((k, masks[k * chunk_size:(k + 1) * chunk_size]))
                write_chunk(paths['chunk_dir'], k, arrays)
                if done % 10 == 0 or done == len(pending):
                    el = time.time() - t0
                    log(f"    chunk {done}/{len(pending)} done — "
                        f"{el:.0f}s elapsed, ETA {el / done * (len(pending) - done):.0f}s")
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            with ProcessPoolExecutor(
                    max_workers=workers, initializer=_worker_init,
                    initargs=(ctx.V, ctx.labels, ctx.anchor, ctx.rho,
                              ctx.pool_bits, method)) as ex:
                futs = {ex.submit(evaluate_chunk,
                                  (k, masks[k * chunk_size:(k + 1) * chunk_size])): k
                        for k in pending}
                for done, fut in enumerate(as_completed(futs), 1):
                    k, arrays = fut.result()
                    write_chunk(paths['chunk_dir'], k, arrays)
                    if done % 10 == 0 or done == len(pending):
                        el = time.time() - t0
                        log(f"    chunk {done}/{len(pending)} done — "
                            f"{el:.0f}s elapsed, ETA {el / done * (len(pending) - done):.0f}s")

    results = merge_chunks(paths['chunk_dir'], n_chunks, paths['final'])
    manifest['sweep_seconds'] = round(time.time() - t0, 1)
    with open(paths['manifest'], 'w') as f:
        json.dump(manifest, f, indent=1)
    for k in range(n_chunks):
        try:
            os.remove(os.path.join(paths['chunk_dir'], f'chunk_{k:05d}.npz'))
        except OSError:
            pass
    try:
        os.rmdir(paths['chunk_dir'])
    except OSError:
        pass
    log(f"  [{ctx.domain}/{ctx.cell_key}] done in {manifest['sweep_seconds']}s "
        f"-> {paths['final']}")
    return results, manifest


def calibrate(ctx, method='residual', min_size=3, max_size=None, n_probe=45, seed=0):
    """
    Time ~n_probe subsets stratified over sizes; project the full-cell runtime
    by linear interpolation of ms-per-subset across sizes.
    Returns {'per_size_ms': {size: ms}, 'projected_s': float}.
    """
    rng = np.random.default_rng(seed)
    p = len(ctx.pool)
    max_size = min(max_size or p, p)
    probe_sizes = sorted({min_size, (min_size + max_size) // 2, max_size})
    per_size = {}
    _worker_init(ctx.V, ctx.labels, ctx.anchor, ctx.rho, ctx.pool_bits, method)
    for s in probe_sizes:
        reps = max(3, n_probe // len(probe_sizes))
        t0 = time.perf_counter()
        for _ in range(reps):
            cols = np.sort(rng.choice(p, size=s, replace=False))
            eval_subset(ctx.V, ctx.labels, ctx.anchor, ctx.rho, cols,
                        ctx.pool_bits, method)
        per_size[s] = (time.perf_counter() - t0) / reps * 1000
    xs, ys = list(per_size), [per_size[s] for s in per_size]
    total = sum(math.comb(p, s) * float(np.interp(s, xs, ys))
                for s in range(min_size, max_size + 1)) / 1000
    return {'per_size_ms': {k: round(v, 1) for k, v in per_size.items()},
            'projected_s': round(total, 1)}


# ---------------------------------------------------------------------------
# Reference subsets, baselines, augmentation
# ---------------------------------------------------------------------------

def reference_results(ctx, results, method='residual', n_boot=1000):
    """
    Boot-CI records for the named reference subsets + label-free baselines
    (anchor single view, simple average of the pool). Reference AUROCs are
    cross-checked against the sweep row for the same mask.
    """
    out = {}
    mask_index = {int(m): i for i, m in enumerate(results['mask'])}
    for name, feats in REFERENCE_SUBSETS.items():
        local = names_to_local_mask([f for f in feats if f in ctx.pool], ctx.pool)
        avail = [f for f in feats if f in ctx.pool]
        if local is None or len(avail) < 3:
            local = names_to_local_mask(avail, ctx.pool)
        if local is None or len(avail) < 3:
            out[name] = {'available': avail, 'auroc': None}
            continue
        cols = mask_to_cols(local)
        oriented, flipped, meta = fuse_subset(ctx.V, ctx.anchor, cols, method)
        auc, lo, hi = boot_auc(ctx.labels, oriented, n=n_boot)
        canon = int(mask_to_canonical(local, ctx.pool_bits))
        row = mask_index.get(canon)
        out[name] = {
            'available': avail, 'auroc': float(auc), 'ci': [float(lo), float(hi)],
            'K': int(meta['K']), 'flipped': bool(flipped), 'mask': canon,
            'sweep_row': None if row is None else int(row),
            'sweep_auroc': None if row is None else float(results['auroc'][row]),
        }

    singles = {}
    for j, f in enumerate(ctx.pool):
        singles[f] = float(roc_auc_score(ctx.labels, ctx.V[:, j]))
    a_auc, a_lo, a_hi = boot_auc(ctx.labels, ctx.anchor, n=n_boot)
    avg = ctx.V.mean(axis=1)
    g_auc, g_lo, g_hi = boot_auc(ctx.labels, avg, n=n_boot)
    out['_singles'] = singles
    out['_anchor_single'] = {'name': ctx.anchor_name, 'auroc': float(a_auc),
                             'ci': [float(a_lo), float(a_hi)]}
    out['_avg_pool'] = {'auroc': float(g_auc), 'ci': [float(g_lo), float(g_hi)]}
    return out


def top_masks(results, k=20):
    """Row indices of the top-k finite-AUROC subsets, descending."""
    auc = results['auroc'].astype(float)
    order = np.argsort(np.where(np.isfinite(auc), auc, -np.inf))[::-1]
    return [int(i) for i in order[:k] if np.isfinite(auc[i])]


def augment_cell(ctx, fd, results, method='residual', n_top=20, n_boot=1000):
    """
    Augmentation stage: for every extra view available on this cell (any
    CANONICAL_POOL name in fd that is NOT in the enumeration pool), evaluate
    base ∪ {view} for base ∈ references + top-n_top sweep subsets, with a
    paired bootstrap delta AUROC vs the base subset.

    Returns list of record dicts (JSON/pickle-friendly).
    """
    from .fusion_utils import paired_boot_delta_auc
    extras = []
    for f in CANONICAL_POOL:
        if f in ctx.pool or f not in fd:
            continue
        arr = np.asarray(fd[f], dtype=float)
        if len(arr) != len(ctx.labels) or not np.isfinite(arr).all() or arr.std() < 1e-8:
            continue
        extras.append((f, zscore(arr * ALL_SIGNS.get(f, +1))))
    if not extras:
        return []

    bases = {}
    for name, feats in REFERENCE_SUBSETS.items():
        local = names_to_local_mask([f for f in feats if f in ctx.pool], ctx.pool)
        if local is not None and len(mask_to_cols(local)) >= 3:
            bases[name] = mask_to_cols(local)
    for rank, row in enumerate(top_masks(results, n_top), 1):
        canon = int(results['mask'][row])
        names = [f for f in canonical_to_names(canon) if f in ctx.pool]
        bases[f'top{rank:02d}'] = mask_to_cols(names_to_local_mask(names, ctx.pool))

    records = []
    p = ctx.V.shape[1]
    for view_name, view in extras:
        V_ext = np.column_stack([ctx.V, view])
        rho_col = np.nan_to_num(np.array(
            [abs(spearmanr(view, ctx.V[:, j])[0]) for j in range(p)]), nan=0.0)
        rho_anchor = abs(spearmanr(view, ctx.anchor)[0])
        view_auc = float(roc_auc_score(ctx.labels, view))

        for base_name, base_cols in bases.items():
            base_or, _, _ = fuse_subset(ctx.V, ctx.anchor, base_cols, method)
            aug_cols = np.append(base_cols, p)
            aug_or, aug_fl, aug_meta = fuse_subset(V_ext, ctx.anchor, aug_cols, method)
            base_auc = (float(roc_auc_score(ctx.labels, base_or))
                        if base_or.std() > 1e-12 else np.nan)
            aug_auc = (float(roc_auc_score(ctx.labels, aug_or))
                       if aug_or.std() > 1e-12 else np.nan)
            try:
                d, d_lo, d_hi = paired_boot_delta_auc(
                    ctx.labels, aug_or, base_or, n=n_boot)
            except Exception:
                d = d_lo = d_hi = np.nan
            records.append({
                'domain': ctx.domain, 'cell_key': ctx.cell_key,
                'view': view_name, 'view_auroc': view_auc,
                'rho_view_anchor': float(rho_anchor) if np.isfinite(rho_anchor) else None,
                'rho_view_max': float(rho_col.max()),
                'base': base_name,
                'base_feats': [ctx.pool[j] for j in base_cols],
                'base_auroc': base_auc, 'aug_auroc': aug_auc,
                'delta': None if not np.isfinite(aug_auc) else aug_auc - base_auc,
                'delta_ci': [float(d_lo), float(d_hi)] if np.isfinite(d) else None,
                'aug_K': int(aug_meta['K']), 'aug_flipped': bool(aug_fl),
            })
    return records
