"""Claude's real-data checks on the full frozen benchmark (13,769 answers), label-free fitting.

1. Replay Codex's local native/simplex LL heads from the saved per-answer covariances (identity check).
2. Clustered-pairs U-PCR moments: drop same-group pairs, groups {H0lim, innovation}, {VE0}, {VE0.75, VE1}.
3. Per-answer 3-group partition stability (hierarchical clustering on within-answer correlation).
4. Difficulty-gate statistic g2/var_y across answers (already known to sit at the ceiling).
Evaluation uses the frozen gate, Top10 and evaluator of the temporal benchmark; PB and within-AUC are
the endpoints. PRMScore here uses the evaluator default calibration, not the nested pair-fold protocol.
"""
import sys, json, time
from pathlib import Path
from itertools import combinations
ROOT = Path(r'C:\Users\omris\TAU\hallucination_detection\.worktrees\temporal-research-20260915'); sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from spectral_utils.context_training import FeatureBundle
from spectral_utils.residual_moment_fusion import fit_head, stream_top10
from spectral_utils.energy_context_stability import simplex_qp
from scripts import run_temporal_research_baseline as base
from spectral_utils.temporal_research_features import BASELINE

OUT = ROOT / 'results/claude_real_checks_v1'; OUT.mkdir(exist_ok=True)
SRC = Path(r'C:\Users\omris\TAU\hallucination_detection')
RM = ROOT / 'results/residual_moment_fusion_v1'
GROUPS = [[0, 4], [1], [2, 3]]   # feature order: H0lim, VE0, VE075, VE1, innovation
PAIRS = [(i, j) for a in range(3) for b in range(a + 1, 3) for i in GROUPS[a] for j in GROUPS[b]]

def iu_moments_pairs(C, var_y, pairs):
    """Same grid/projection rule as cca_iu_isolation.iu_moments, restricted pair design."""
    m = len(C); A = np.zeros((len(pairs), m))
    for r, (i, j) in enumerate(pairs): A[r, i] = A[r, j] = 1
    b = np.array([C[i, j] for i, j in pairs])
    rho0 = np.linalg.lstsq(A, b, rcond=None)[0]
    v = np.linalg.eigh(C)[1][:, -1]
    grid = np.linspace(0., float(var_y), 300)
    rho = rho0[None, :] + .5 * grid[:, None]
    proj = (rho @ v)[:, None] * v[None, :]
    err = np.linalg.norm(rho - proj, axis=1) / (np.linalg.norm(rho, axis=1) + 1e-12)
    k = int(err.argmin()); chosen = rho[k]
    fitted = A @ chosen - grid[k]
    return chosen, grid[k], float(np.linalg.norm(fitted - b) / (np.linalg.norm(b) + 1e-12))

def fit_head_pairs(C, sd, var_y, pairs):
    rho, g2, resid = iu_moments_pairs(C, var_y, pairs)
    values, vectors = np.linalg.eigh(C); u = vectors[:, -2:]; v = values[-2:]
    a = u @ ((u.T @ rho) / (v + 1e-12)); raw = a / sd; norm = np.abs(raw).sum()
    native = raw / norm if norm > 1e-12 else np.full(len(sd), 1 / len(sd))
    B = sd / sd.mean(); Q = C * B[:, None] * B[None, :]; r = B * rho
    simplex = .75 / len(sd) + .25 * simplex_qp(Q, r)[0]
    return dict(native=native, simplex=simplex, rho=rho, g2=float(g2), additive_residual=resid)

def main():
    t0 = time.perf_counter()
    bundle = FeatureBundle(ROOT / 'results/temporal_context_data_v1', 'innovation5'); meta = bundle.metadata
    records, joined = base.load_contract(SRC); total = int(joined['offsets'][-1])
    assert [r['uid'] for r in records] == [m['uid'] for m in meta]
    # per-answer saved local level moments (outer exclusions cover every answer once)
    C_L = np.empty((len(meta), 5, 5)); SD = np.empty((len(meta), 5)); VY = np.empty(len(meta)); seen = np.zeros(len(meta), bool)
    codex = {k: np.full(total, np.nan) for k in ('local__native__LL', 'local__simplex__LL')}
    for f in range(5):
        d = np.load(RM / f'exclude_{f}_DIAGNOSTICS.npz'); ids = d['ids']
        C_L[ids] = d['covariance'][:, 0]; SD[ids] = d['sd']; VY[ids] = d['var_y']; seen[ids] = True
        with np.load(RM / f'exclude_{f}.npz') as s:
            for k in codex:
                v = s[k]; ok = np.isfinite(v); codex[k][ok] = v[ok]
    assert seen.all() and all(np.isfinite(v).all() for v in codex.values())
    names = ['replay_native_LL', 'replay_simplex_LL', 'cluster3_native_LL', 'cluster3_simplex_LL']
    scores = {n: np.full(total, np.nan) for n in names}
    partitions = []; g2c = []; resid_all = []; resid_c3 = []; rho_cos = []
    max_replay_delta = 0.
    for i, m in enumerate(meta):
        n = bundle.length[i]; off = bundle.offset[i]
        L = np.asarray(bundle.features[off:off + n])[:, bundle.columns].astype(float)
        spans = np.asarray(bundle.spans[m['step_start']:m['step_stop']]) - off
        S = stream_top10(L, spans); sl = slice(m['step_start'], m['step_stop'])
        full = fit_head(C_L[i], SD[i], VY[i]); c3 = fit_head_pairs(C_L[i], SD[i], VY[i], PAIRS)
        scores['replay_native_LL'][sl] = S @ full['native']; scores['replay_simplex_LL'][sl] = S @ full['simplex']
        scores['cluster3_native_LL'][sl] = S @ c3['native']; scores['cluster3_simplex_LL'][sl] = S @ c3['simplex']
        max_replay_delta = max(max_replay_delta, float(np.max(np.abs(scores['replay_native_LL'][sl] - codex['local__native__LL'][sl]))),
                               float(np.max(np.abs(scores['replay_simplex_LL'][sl] - codex['local__simplex__LL'][sl]))))
        g2c.append(full['g2'] / VY[i]); resid_all.append(full['additive_residual']); resid_c3.append(c3['additive_residual'])
        rho_cos.append(float(full['rho'] @ c3['rho'] / np.linalg.norm(full['rho']) / np.linalg.norm(c3['rho'])))
        # label-free 3-group partition from the within-answer correlation (C_L is sd-normalised)
        R = C_L[i] / np.sqrt(np.outer(np.diag(C_L[i]), np.diag(C_L[i]))); D = np.clip(1 - R, 0, 2); np.fill_diagonal(D, 0)
        lab = fcluster(linkage(squareform(D, checks=False), 'average'), 3, 'maxclust')
        partitions.append('|'.join(''.join(str(j) for j in np.flatnonzero(lab == c)) for c in sorted(set(lab), key=lambda c: np.flatnonzero(lab == c)[0])))
        if (i + 1) % 2000 == 0: print('[fit]', i + 1, round(time.perf_counter() - t0, 1), flush=True)
    assert all(np.isfinite(v).all() for v in scores.values())
    print('max replay delta vs Codex arrays:', max_replay_delta, flush=True)
    with np.load(ROOT / 'results/temporal_research_baseline_v1/SCORES_FROZEN.npz') as f:
        gate = f['gate_percentile'] >= .33; ref = {'innovation5': f['steps__append_innovation__H0lim'], 'original4': f['steps__' + BASELINE]}
    metrics, per = base.evaluator.evaluate_arrays(records, joined, scores, fold_auc=True, pb_gate_open=gate)
    rm, rp = base.evaluator.evaluate_arrays(records, joined, ref, fold_auc=True, pb_gate_open=gate)
    metrics.update(rm); per.update(rp)
    primary = [('cluster3_native_LL', 'replay_native_LL'), ('cluster3_simplex_LL', 'replay_simplex_LL')]
    pairs = primary + [('cluster3_native_LL', 'innovation5'), ('cluster3_simplex_LL', 'innovation5'), ('replay_simplex_LL', 'innovation5')]
    contrasts = base.evaluator.paired_bootstrap(records, joined, per, draws=10000, pairs=pairs, primary_pairs=set(primary), primary_ci=.975)
    for a, b in pairs: contrasts[a + '_minus_' + b]['pb_delta'] = metrics[a]['pb_all8'] - metrics[b]['pb_all8']
    # partition stability by fold and cell
    parts = np.array(partitions); folds = np.array([m['fold'] for m in meta]); cells = np.array([m['cell'] for m in meta])
    vals, counts = np.unique(parts, return_counts=True); order = np.argsort(-counts)
    top = [(str(vals[k]), int(counts[k])) for k in order[:6]]
    modal = vals[order[0]]
    by_fold = {int(f): float(np.mean(parts[folds == f] == modal)) for f in range(5)}
    by_cell = {c: float(np.mean(parts[cells == c] == modal)) for c in sorted(set(cells))}
    summary = dict(answers=len(meta), max_replay_delta=max_replay_delta,
                   g2_over_var_y=dict(min=float(np.min(g2c)), median=float(np.median(g2c)), at_ceiling_fraction=float(np.mean(np.array(g2c) >= .995))),
                   additive_residual_all_pairs=dict(median=float(np.median(resid_all)), q90=float(np.quantile(resid_all, .9))),
                   additive_residual_cross_group=dict(median=float(np.median(resid_c3)), q90=float(np.quantile(resid_c3, .9))),
                   rho_cosine_allpairs_vs_cluster3=dict(median=float(np.median(rho_cos)), q10=float(np.quantile(rho_cos, .1))),
                   partition_top=top, modal_partition=str(modal), modal_share_by_fold=by_fold, modal_share_by_cell=by_cell,
                   feature_order=['H0lim', 'VE0', 'VE075', 'VE1', 'innovation'], seconds=time.perf_counter() - t0)
    table = {k: dict(pb=metrics[k]['pb_all8'], within=metrics[k]['prm_within'], prmscore_default_calibration=metrics[k]['prmscore_q08']) for k in list(scores) + list(ref)}
    (OUT / 'METRICS.json').write_text(json.dumps(dict(table=table, contrasts=contrasts, summary=summary), indent=1, default=float), encoding='utf8')
    np.savez_compressed(OUT / 'SCORES.npz', **{'steps__' + k: v for k, v in scores.items()})
    lines = ['# Claude real-data checks (development, label-free fitting)', '', '| method | PB % | within | PRMScore (default cal.) |', '|---|---:|---:|---:|']
    for k, v in table.items(): lines.append(f"| {k} | {100*v['pb']:.4f} | {v['within']:.6f} | {v['prmscore_default_calibration']:.6f} |")
    lines += ['', '| contrast | PB delta | PB CI | within CI | level |', '|---|---:|---|---|---|']
    for k, v in contrasts.items(): lines.append(f"| {k} | {100*v['pb_delta']:+.4f} | {[round(100*x,4) for x in v['pb_ci']]} | {[round(x,6) for x in v['prm_within_ci']] if v.get('prm_within_ci') else None} | {v['ci_level']} |")
    lines += ['', '```', json.dumps(summary, indent=1, default=float), '```']
    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    print('\n'.join(lines))

if __name__ == '__main__':
    main()
