"""Renyi-order sweep driver (Stage 3b): 20 single-view arms on the frozen localization evaluator.

Reuses the v2 harness (`run_renyi_view_fusion_v2`: loading, checkpointing, evaluation, bootstrap) with the
sweep module swapped in at import time (top level, so spawned workers see the same roster).  Output:
results/renyi_alpha_sweep_v1/.  Smoke first (--smoke), then --allow-full.

Selection is label-guided development evidence: the sweep answers "which alpha, on this development set,
carries the best within-answer ranking / PB", it does not confirm a candidate.
"""
import os
import sys
from pathlib import Path
import numpy as np

os.environ['RENYI_V2_ROSTER'] = 'all'          # base OUT without a pass suffix; overridden below
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_renyi_view_fusion_v2 as run
from spectral_utils import renyi_alpha_sweep as sweep

# ---- top-level overrides (also executed in spawned workers via __mp_main__) ----
run.METHODS = sweep.METHODS; run.ALL_METHODS = sweep.METHODS; run.FAST_METHODS = sweep.METHODS; run.JOINT_METHODS = ()
run.FUSED_IN_ROSTER = (); run.FUSED_METHODS = (); run.SINGLE_VIEWS = sweep.VIEW_NAMES
run.MAX_WEIGHTS = sweep.MAX_WEIGHTS; run.COLUMN_NAMES = sweep.COLUMN_NAMES; run.DIAG_NAMES = sweep.DIAG_NAMES
run.ALPHAS = sweep.ALPHA_GRID; run.GROUP_OF = {}; run.GROUP_NAMES = ()
run.NOT_APPLICABLE = {}
run.fit_all = sweep.fit_all; run.view_diagnostics = sweep.view_diagnostics
run.ROSTER = 'sweep'; run.BASE_OUT = ROOT / 'results/renyi_alpha_sweep_v1'; run.OUT = run.BASE_OUT
run.NAMES = {m: ('Escort varentropy / alpha = %g' % sweep.ALPHA_OF[m]) if m.startswith('view__ve') else
             'Single Renyi view / alpha = ' + ('0 (limit: mean log q)' if m == 'view__H0lim' else
             ('inf (min-entropy)' if m == 'view__Hinf' else '%g' % sweep.ALPHA_OF[m])) for m in sweep.METHODS}
run.NAMES.update(entropy='Token entropy reference (frozen)', direct_iu='Direct probability IU-PCR (17 inputs, frozen)',
                 ref__varentropy15='Raw varentropy15 reference (frozen Step339)',
                 ref__varentropy15_iu='Varentropy15 contribution IU (frozen Step339)')
run.PRIMARY = set()      # no pre-registered winner contrast: this is a curve, every pair at 95 %


def contrast_pairs():
    hs = [m for m in sweep.METHODS if not m.startswith('view__ve')]; ves = [m for m in sweep.METHODS if m.startswith('view__ve')]
    pairs = []
    pairs += [(m, 'view__H1') for m in hs if m != 'view__H1']
    pairs += [(m, 'view__a0.1') for m in hs if m != 'view__a0.1']
    pairs += [(hs[i], hs[i + 1]) for i in range(len(hs) - 1)]                      # adjacent orders
    pairs += [(m, 'ref__varentropy15') for m in hs] + [(m, 'ref__varentropy15_iu') for m in hs]
    pairs += [(m, 'view__ve1') for m in ves if m != 'view__ve1']                   # escort varentropy vs varentropy15
    pairs += [(m, 'view__a0.1') for m in ves] + [(m, 'view__H1') for m in ves]
    pairs += [(ves[i], ves[i + 1]) for i in range(len(ves) - 1)]
    pairs += [('view__ve%g' % a, sweep._alpha_name(a)) for a in sweep.VE_GRID if a in sweep.ALPHA_GRID]   # same alpha, VE vs H
    seen, out = set(), []
    for p in pairs:
        if p not in seen: seen.add(p); out.append(p)
    return out


run.contrast_pairs = contrast_pairs


def aggregate_diagnostics(infos, scope):
    views = [r['view_diagnostics'] for r in infos]; cols = list(sweep.DIAG_NAMES); n = len(views)
    Sp = np.array([[[np.nan if v is None else v for v in row] for row in d['spearman']] for d in views], float)
    std = np.array([[np.nan if v is None else v for v in d['std']] for d in views], float)
    near = np.array([d['near_constant'] for d in views], bool)
    anchor = np.array([[np.nan if v is None else v for v in d['anchor_correlation']] for d in views], float)
    with np.errstate(invalid='ignore'):
        lim = cols.index('H0lim'); a01 = cols.index('a0.1'); h1 = cols.index('H1')
        return dict(scope=scope, n_answers=n, columns=cols,
                    note='spearman_with_* = mean over answers of the within-answer Spearman correlation of each view with the '
                         'alpha->0 limit view, with alpha=0.1 and with H1; std/near_constant per view; anchor = own varentropy15.',
                    spearman_mean=run.clean(np.nanmean(Sp, axis=0)),
                    spearman_with_limit={c: float(np.nanmean(Sp[:, j, lim])) for j, c in enumerate(cols)},
                    spearman_with_a0_1={c: float(np.nanmean(Sp[:, j, a01])) for j, c in enumerate(cols)},
                    spearman_with_h1={c: float(np.nanmean(Sp[:, j, h1])) for j, c in enumerate(cols)},
                    fraction_spearman_above_0_99_with_limit={c: float(np.nanmean(Sp[:, j, lim] > .99)) for j, c in enumerate(cols)},
                    per_column={c: dict(std=run._nanstats(std[:, j]),
                                        fraction_near_constant=float(near[:, j].mean()) if j < sweep.MAX_WEIGHTS else None,
                                        anchor_correlation=run._nanstats(anchor[:, j]) if j < sweep.MAX_WEIGHTS else None)
                                for j, c in enumerate(cols)},
                    orientation_flip_fraction={m: float(np.mean([r['diagnostics'][m]['orientation_flipped'] for r in infos if m in r['diagnostics']]))
                                               for m in sweep.METHODS},
                    fused_arms={}, not_applicable={})


run.aggregate_diagnostics = aggregate_diagnostics

_manifest_for = run.manifest_for


def manifest_for(source, v2, temporal, varentropy):
    m = _manifest_for(source, v2, temporal, varentropy)
    m.update(schema='renyi-alpha-sweep-v1', roster='sweep', methods=list(sweep.METHODS),
             alphas=[('inf' if np.isinf(a) else a) for a in sweep.ALPHA_GRID], columns=list(sweep.COLUMN_NAMES), groups={},
             stage='STAGE 3b: label-guided alpha sweep on development data (authorized 2026-09-13)')
    for path in (Path(__file__), ROOT / 'spectral_utils/renyi_alpha_sweep.py'):
        m['hashes'][str(path)] = run.base.old.sha256_file(path)
    return m


run.manifest_for = manifest_for

_evaluate = run.evaluate


def evaluate(con, records, joined, temporal, varentropy):
    _evaluate(con, records, joined, temporal, varentropy)
    import json
    m = json.loads((run.OUT / 'METRICS.json').read_text(encoding='utf8'))['metrics']
    # Escort varentropy at alpha = 1 must be the frozen varentropy15 (Step 339) on every endpoint.
    for key in ('pb_all8', 'prm_within', 'prm_pooled', 'prmscore_q08'):
        np.testing.assert_allclose(m['view__ve1'][key], m['ref__varentropy15'][key], atol=1e-9, rtol=0, err_msg='ve1 vs varentropy15 ' + key)
    print('[sweep] view__ve1 reproduces ref__varentropy15 on all endpoints', flush=True)


run.evaluate = evaluate


if __name__ == '__main__':
    try:
        run.main()
    except BaseException as error:
        import json
        name = 'SMOKE_STATE.json' if '--smoke' in sys.argv else 'RUN_STATE.json'
        path = run.OUT / name; state = json.loads(path.read_text(encoding='utf8')) if path.exists() else {}
        state.update(status='INTERRUPTED' if isinstance(error, KeyboardInterrupt) else 'FAILED', error=f'{type(error).__name__}: {error}')
        run.atomic_json_retry(path, state)
        raise
