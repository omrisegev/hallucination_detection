"""Registered contrasts and grouped uncertainty for full sampling evaluation.

Targets enter only this evaluation layer. No fitting or recipe selection.
"""
import numpy as np

SELECTORS = ('full', 'uniform', 'risk_top', 'dufs_transposed', 'dufs_permuted',
             'window_diffusion', 'entropy_tails', 'entropy_quantiles')
CORES = ('equal', 'iu', 'joint0', 'graph010', 'graph_perm',
         'equal_graph010', 'equal_graph_perm')


def contrasts():
    result = []
    def arm(selector, core): return f'sample_{selector}__{core}'
    def add(left, right, scope='all'):
        item = dict(left=left, right=right, scope=scope)
        if item not in result: result.append(item)
    for selector in SELECTORS[1:]:
        for core in CORES: add(arm(selector, core), arm('full', core))
        for left, right in (('iu', 'equal'), ('graph010', 'joint0'),
                            ('graph010', 'graph_perm'), ('graph010', 'iu'),
                            ('graph010', 'equal_graph010')):
            add(arm(selector, left), arm(selector, right))
        for core in ('iu', 'graph010'):
            add(arm(selector, core), arm('full', core), 'eligible')
            add(arm(selector, core), 'dual__equal_graph_perm')
        add(arm(selector, 'graph010'), arm('full', 'graph010'), 'eligible_both_native')
    for core in ('iu', 'graph010'):
        for other in ('uniform', 'risk_top', 'dufs_permuted'):
            add(arm('dufs_transposed', core), arm(other, core))
        add(arm('window_diffusion', core), arm('uniform', core))
    for selector in SELECTORS[-2:]:
        for core in ('iu', 'graph010', 'equal_graph_perm'):
            for other in ('full', 'uniform', 'risk_top'):
                add(arm(selector, core), arm(other, core))
    for core in ('iu', 'graph010', 'equal_graph_perm'):
        add(arm('entropy_tails', core), arm('entropy_quantiles', core))
    return result


def fold_points(ref, records, arrays, column, folds, selected=None):
    mask = arrays['valid'][:, column].copy()
    if selected is not None: mask &= selected
    mask &= np.array([r['cell'].startswith('prm') for r in records])
    owner = np.repeat(np.arange(len(records)), np.diff(arrays['offsets']))
    values = []
    for fold in range(5):
        steps = (mask & (folds == fold))[owner]
        values.append(ref.rank_auc(arrays['labels'][steps], arrays['scores'][steps, column]))
    mean = float(np.mean(values)) if all(v is not None for v in values) else None
    return dict(fold_mean_auc=mean, fold_aucs=values)


def intervals(ref, records, arms, arrays, folds, eligible, native, progress=None):
    """Joint source bootstrap, with common-valid PRMB and all-failure PB.

    A single multinomial draw supplies all panels, folds and scorers. The
    resampling conditions on fitted scores; it does not refit or redo selection.
    """
    groups = sorted({r['group_id'] for r in records})
    lookup = {g: i for i, g in enumerate(groups)}
    gi = np.array([lookup[r['group_id']] for r in records]); ng = len(groups)
    weights = np.random.default_rng(ref.SEED).multinomial(ng, np.full(ng, 1/ng), size=ref.DRAWS)
    owner = np.repeat(np.arange(len(records)), np.diff(arrays['offsets']))
    cells = np.array([r['cell'] for r in records]); prm = np.char.startswith(cells, 'prm')
    pb_cells = sorted(set(cells[np.char.startswith(cells, 'pb_')]))
    cache = {}

    def finite_mean(values):
        values = np.asarray(values, float)
        return float(values.mean()) if len(values) and np.isfinite(values).all() else None

    def summary(values):
        good = np.asarray(values)[np.isfinite(values)]
        return dict(ci95=np.quantile(good, [.025, .975]).tolist() if len(good) else None,
                    valid_draws=len(good))

    def divide(a, b):
        return np.divide(a, b, out=np.full(np.shape(a), np.nan, float), where=b != 0)

    def auc_draws(j, rows):
        steps = rows[owner]
        if not steps.any() or len(np.unique(arrays['labels'][steps])) < 2:
            return np.full(ref.DRAWS, np.nan), None
        plan = ref.auc_plan(arrays['labels'][steps], arrays['scores'][steps, j], gi[owner[steps]])
        return np.array([ref.weighted_auc(plan, w) for w in weights]), ref.weighted_auc(plan, np.ones(ng, int))

    def distribution(j, scope, common):
        key = (j, np.packbits(scope).tobytes(), np.packbits(common).tobytes())
        if key in cache: return cache[key]
        selected = prm & scope & common
        pooled, pooled_point = auc_draws(j, selected)
        fold_results = [auc_draws(j, selected & (folds == f)) for f in range(5)]
        fold_draws = np.mean([v[0] for v in fold_results], axis=0)
        fold_point = finite_mean([np.nan if v[1] is None else v[1] for v in fold_results])
        mixed = selected & np.isfinite(arrays['within'][:, j])
        num = np.bincount(gi[mixed], weights=arrays['within'][mixed, j], minlength=ng)
        den = np.bincount(gi[mixed], minlength=ng)
        within = divide(weights @ num, weights @ den)
        draws = dict(prm_pooled=pooled, prm_fold=fold_draws, within=within)
        point = dict(prm_pooled=pooled_point, prm_fold=fold_point,
                     within=finite_mean(arrays['within'][mixed, j]))
        cell_draws = {}; cell_points = {}
        success = arrays['decision'][:, j] & (arrays['predictions'][:, j] == arrays['target'])
        for cell in pb_cells:
            mask = scope & (cells == cell)
            clean = mask & (arrays['target'] == -1); error = mask & (arrays['target'] >= 0)
            def count(rows): return weights @ np.bincount(gi[rows], minlength=ng)
            ca = divide(count(clean & success), count(clean))
            ea = divide(count(error & success), count(error))
            cell_draws[cell] = np.divide(2*ca*ea, ca+ea, out=np.zeros_like(ca), where=ca+ea != 0)
            if clean.any() and error.any():
                c, e = success[clean].mean(), success[error].mean()
                cell_points[cell] = float(2*c*e/(c+e)) if c+e else 0.
            else: cell_points[cell] = np.nan
        for panel in ('q4', 'q8', 'all'):
            keys = [c for c in pb_cells if panel == 'all' or c.endswith(panel)]
            draws['pb_'+panel] = np.mean([cell_draws[c] for c in keys], axis=0)
            point['pb_'+panel] = finite_mean([cell_points[c] for c in keys])
        result = dict(draws=draws, point=point, prm_answers=int(selected.sum()),
                      mixed_answers=int(mixed.sum()), scope_answers=int(scope.sum()))
        cache[key] = result
        return result

    output = dict(status='IN_PROGRESS', draws=ref.DRAWS, seed=ref.SEED, source_groups=ng,
                  unit='canonical source group shared across tasks, scorers and fixed folds',
                  conditional_on_saved_predictions=True, multiplicity_adjusted=False,
                  absolute={}, paired=[])
    all_rows = np.ones(len(records), bool)
    for j, arm in enumerate(arms):
        d = distribution(j, all_rows, arrays['valid'][:, j])
        output['absolute'][arm] = dict(point=d['point'], intervals={k:summary(v) for k,v in d['draws'].items()},
                                     prm_answers=d['prm_answers'], mixed_answers=d['mixed_answers'])
        if progress: progress('ABSOLUTE_INTERVALS', j+1, len(arms), output)
    pairs = contrasts()
    for i, item in enumerate(pairs):
        j, k = arms.index(item['left']), arms.index(item['right'])
        scope = all_rows.copy()
        if item['scope'].startswith('eligible'): scope &= eligible
        if item['scope'] == 'eligible_both_native': scope &= native[:, j] & native[:, k]
        common = arrays['valid'][:, j] & arrays['valid'][:, k]
        dl, dr = distribution(j, scope, common), distribution(k, scope, common)
        delta = {key: None if dl['point'][key] is None or dr['point'][key] is None
                 else dl['point'][key]-dr['point'][key] for key in dl['point']}
        output['paired'].append(dict(item, left_point=dl['point'], right_point=dr['point'], delta=delta,
            intervals={key:summary(dl['draws'][key]-dr['draws'][key]) for key in dl['draws']},
            prm_common_answers=dl['prm_answers'], scope_answers=dl['scope_answers']))
        if progress: progress('PAIRED_INTERVALS', i+1, len(pairs), output)
    output['status'] = 'COMPLETE'
    return output
