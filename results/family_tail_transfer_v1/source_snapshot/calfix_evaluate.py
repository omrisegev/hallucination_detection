"""Evaluator of the calibration-corrected stages: reads a ScoreBundle (SCORES.npz with eval__/cal__
arrays per method), applies thresholds calibrated ONLY on each model's own calibration-fold scores,
and reports PRMScore (primary), within-answer AUC and ProcessBench SLA with paired source-group
bootstrap intervals.  Labels enter only here.  Panels:
  P1  q80 of the pooled (PB+PRMB) calibration-fold scores           label-free, external contract
  P1b q80 of the calibration fold's PRMBench scores                  label-free sensitivity
  P2  quantile selected by calibration-fold PRMScore (cal labels)     label-using, separate panel
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calfix_common import Population, dump, roles_of  # noqa: E402

DEPTH = Path(r'C:\Users\omris\TAU\hallucination_detection\.worktrees\depth-feature-fusion-v1')
PANELS = ('P1', 'P1b', 'P2')
QGRID = np.linspace(0.50, 0.99, 50)


def prmscore(c):
    """c[..., 4] = (TP, FP, TN, FN), positive = VALID step; official 0.5*(F1 + negative F1)."""
    tp, fp, tn, fn = np.moveaxis(np.asarray(c, float), -1, 0)
    def div(a, b): return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)
    return .5 * (div(2 * tp, 2 * tp + fp + fn) + div(2 * tn, 2 * tn + fp + fn))


def counts(valid_pred, y_valid, mask):
    p, y = valid_pred[mask], y_valid[mask]
    return np.array([np.sum(p & y), np.sum(p & ~y), np.sum(~p & ~y), np.sum(~p & y)], float)


def within_auc(y, s):
    k1 = int(y.sum()); k0 = len(y) - k1
    return float((rankdata(s)[y].sum() - k1 * (k1 + 1) / 2) / (k1 * k0))


def pvalue(d):
    return float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0))))


def holm(p):
    p = np.asarray(p, float); order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj


def evaluate(bundle_dir: Path, out_dir: Path, primary: list, descriptive: list, *, pop: Population | None = None,
             raw_methods=('ct7_raw',), draws: int = 20_000, seed: int = 20260924, labels: dict | None = None, rename: dict | None = None) -> dict:
    """rename: declared aliases {name stored in the bundle: evaluated name}; scores are untouched."""
    t0 = time.perf_counter(); out_dir.mkdir(parents=True, exist_ok=True)
    pop = pop or Population(); Z = np.load(bundle_dir / 'SCORES.npz'); rename = rename or {}
    assert np.array_equal(Z['offsets'], pop.off) and np.array_equal(Z['answer_fold'], pop.fold)
    stored = sorted(k[len('eval__'):] for k in Z.files if k.startswith('eval__'))
    assert set(rename) <= set(stored) and not set(rename.values()) & set(stored)
    E = {rename.get(m, m): Z['eval__' + m] for m in stored}; C = {rename.get(m, m): Z['cal__' + m] for m in stored}; methods = sorted(E)
    off, ns = pop.off, pop.ns; yv = ~pop.labels                                     # True = valid step
    nc_steps = np.repeat(pop.noncontrol, ns); prm_steps = np.repeat(pop.prm, ns)
    checks = {'methods': len(methods), 'aliases': rename, 'answer_z_max_dev': {}}
    for m in methods:                                                              # matched final normalization
        if m in raw_methods:
            continue
        dev = 0.0
        for s in (E[m], C[m]):
            mu = np.add.reduceat(s, off[:-1]) / ns; sd = np.sqrt(np.maximum(np.add.reduceat(s * s, off[:-1]) / ns - mu ** 2, 0))
            dev = max(dev, float(np.abs(mu).max()), float(np.min(np.stack([np.abs(sd - 1), sd]), 0).max()))
        checks['answer_z_max_dev'][m] = dev
        assert dev < 1e-6, (m, dev)
    # ---------------------------------------------------------------- thresholds and decisions
    thresholds = {m: {p: [] for p in PANELS} for m in methods}; valid = {(m, p): np.zeros(pop.total, bool) for m in methods for p in PANELS}
    for m in methods:
        for k in range(5):
            _fit, cal, ev = roles_of(k)
            cal_rows = pop.rows(pop.fold == cal); cs = C[m][cal_rows]
            tau = {'P1': float(np.quantile(cs, .8)), 'P1b': float(np.quantile(C[m][pop.rows(pop.prm & (pop.fold == cal))], .8))}
            grid = np.quantile(cs, QGRID); crow = pop.rows(pop.noncontrol & (pop.fold == cal)); sc = C[m][crow]; yc = yv[crow]
            pv = sc[None, :] < grid[:, None]
            gc = np.stack([(pv & yc).sum(1), (pv & ~yc).sum(1), (~pv & ~yc).sum(1), (~pv & yc).sum(1)], -1)
            gs = prmscore(gc); best = int(np.argmax(gs)); tau['P2'] = float(grid[best])
            ev_rows = pop.rows(pop.fold == ev)
            for p in PANELS:
                valid[m, p][ev_rows] = E[m][ev_rows] < tau[p]
                thresholds[m][p].append({'outer_fold': k, 'cal_fold': cal, 'tau': tau[p], **({'quantile': float(QGRID[best]), 'cal_prmscore': float(gs[best])} if p == 'P2' else {})})
    # ---------------------------------------------------------------- official replay
    sys.path.insert(0, str(DEPTH)); from spectral_utils.prmbench import prmbench_evaluate  # noqa: E402
    prm_idx = np.flatnonzero(pop.prm); metas = [pop.meta[pop.ids[i]] for i in prm_idx]
    point, rows, replay = {}, [], 0.0
    for m in methods:
        rec = {'method': m}
        for p in PANELS:
            c = counts(valid[m, p], yv, nc_steps); ps = float(prmscore(c))
            off_res = prmbench_evaluate([{'idx': pop.ids[i], 'labels': valid[m, p][off[i]:off[i + 1]].astype(int).tolist()} for i in prm_idx], metas)['total']
            replay = max(replay, abs(ps - .5 * (off_res['f1'] + off_res['negative_f1'])))
            rec[f'prmscore_{p}'] = ps
            if p == 'P1':
                tp, fp, tn, fn = c
                rec.update({'correct_step_recall_P1': tp / (tp + fn), 'error_step_recall_P1': tn / (tn + fp), 'pred_error_frac_P1': (tn + fn) / c.sum(),
                            'fold_prmscore_P1': [float(prmscore(counts(valid[m, p], yv, nc_steps & (pop.step_fold == f)))) for f in range(5)]})
        point[m] = rec; rows.append(rec)
    checks['official_replay_max_abs_diff'] = replay
    assert replay < 1e-12, replay
    # ---------------------------------------------------------------- within-answer AUC and ProcessBench SLA
    elig = np.flatnonzero(pop.eligible); pb_err = np.flatnonzero(pop.pb & (pop.target >= 0)); cellnames = sorted(set(pop.cells[pb_err]))
    auc, hit = {}, {}
    for m in methods:
        s = E[m]
        auc[m] = np.array([within_auc(pop.labels[off[i]:off[i + 1]], s[off[i]:off[i + 1]]) for i in elig])
        hit[m] = np.array([int(np.flatnonzero(s[off[i]:off[i + 1]] >= s[off[i]:off[i + 1]].max() - 8 * np.finfo(float).eps)[0]) == pop.target[i] for i in pb_err], float)
        point[m]['within_auc'] = float(auc[m].mean())
        point[m]['pb_sla_macro8'] = float(np.mean([hit[m][pop.cells[pb_err] == c].mean() for c in cellnames]))
    # ---------------------------------------------------------------- paired source-group bootstrap
    prm_groups, ginv = np.unique(pop.groups[prm_idx], return_inverse=True); gmap = np.full(pop.n, -1); gmap[prm_idx] = ginv; G = len(prm_groups)
    gstep = np.repeat(gmap, ns)
    gconf = {}
    for m in methods:
        for p in PANELS:
            v = valid[m, p]; sel = nc_steps
            gconf[m, p] = np.stack([np.bincount(gstep[sel], weights=(v & yv)[sel], minlength=G), np.bincount(gstep[sel], weights=(v & ~yv)[sel], minlength=G),
                                    np.bincount(gstep[sel], weights=(~v & ~yv)[sel], minlength=G), np.bincount(gstep[sel], weights=(~v & yv)[sel], minlength=G)], 1)
    auc_sum = {m: np.bincount(gmap[elig], weights=auc[m], minlength=G) for m in methods}; auc_cnt = np.bincount(gmap[elig], minlength=G).astype(float)
    pb_groups, pinv = np.unique(pop.groups[pb_err], return_inverse=True); H = len(pb_groups); cidx = np.array([cellnames.index(c) for c in pop.cells[pb_err]])
    pb_cnt = np.zeros((H, len(cellnames))); np.add.at(pb_cnt, (pinv, cidx), 1.0)
    pb_hit = {}
    for m in methods:
        a = np.zeros((H, len(cellnames))); np.add.at(a, (pinv, cidx), hit[m]); pb_hit[m] = a
    rng = np.random.default_rng(seed); est = {}
    for key in [(m, e) for m in methods for e in (*PANELS, 'auc', 'pb')]:
        est[key] = np.empty(draws)
    pos = 0
    while pos < draws:
        nb = min(2000, draws - pos)
        W = rng.multinomial(G, np.full(G, 1 / G), size=nb).astype(float); V = rng.multinomial(H, np.full(H, 1 / H), size=nb).astype(float)
        den_auc = W @ auc_cnt; den_pb = V @ pb_cnt
        for m in methods:
            for p in PANELS:
                est[m, p][pos:pos + nb] = prmscore(W @ gconf[m, p])
            est[m, 'auc'][pos:pos + nb] = (W @ auc_sum[m]) / den_auc
            est[m, 'pb'][pos:pos + nb] = np.mean((V @ pb_hit[m]) / den_pb, axis=1)
        pos += nb
    for m in methods:                                                              # bootstrap point = macro/pooled point
        point[m]['prmscore_P1_ci95'] = [float(np.quantile(est[m, 'P1'], q)) for q in (.025, .975)]
    # ---------------------------------------------------------------- contrasts
    label = {'P1': 'prmscore_P1', 'P1b': 'prmscore_P1b', 'P2': 'prmscore_P2', 'auc': 'within_auc', 'pb': 'pb_sla_macro8'}
    crow = []
    for fam, pairs in (('primary', primary), ('descriptive', descriptive)):
        for a, b, why in pairs:
            if a not in point or b not in point:
                crow.append({'family': fam, 'a': a, 'b': b, 'why': why, 'endpoint': 'MISSING'}); continue
            for e, name in label.items():
                d = est[a, e] - est[b, e]; delta = point[a][name] - point[b][name]
                r = {'family': fam, 'a': a, 'b': b, 'why': why, 'endpoint': name, 'delta': delta,
                     'ci95_lo': float(np.quantile(d, .025)), 'ci95_hi': float(np.quantile(d, .975)), 'p_boot': pvalue(d)}
                if e == 'P1':
                    fa, fb = np.array(point[a]['fold_prmscore_P1']), np.array(point[b]['fold_prmscore_P1']); r['folds_a_gt_b'] = int((fa > fb).sum())
                crow.append(r)
    CT = pd.DataFrame(crow)
    assert (CT.endpoint != 'MISSING').all(), CT[CT.endpoint == 'MISSING'][['a', 'b']].values.tolist()
    prim = (CT.family == 'primary') & (CT.endpoint == 'prmscore_P1')
    CT['p_holm_primary'] = np.nan; CT.loc[prim, 'p_holm_primary'] = holm(CT.loc[prim, 'p_boot'].to_numpy())
    CT.to_csv(out_dir / 'CONTRASTS.csv', index=False)
    MT = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, list)} | {'fold_prmscore_P1': ' '.join(f'{x:.4f}' for x in r['fold_prmscore_P1']),
                        'prmscore_P1_ci95': '[{:.4f}, {:.4f}]'.format(*r['prmscore_P1_ci95'])} for r in point.values()])
    if labels:
        MT.insert(1, 'label', MT.method.map(labels).fillna(''))
    MT.to_csv(out_dir / 'METRICS.csv', index=False)
    np.savez_compressed(out_dir / 'BOOTSTRAP_PRIMARY.npz', **{f'{a}__minus__{b}__P1': est[a, 'P1'] - est[b, 'P1'] for a, b, _ in primary if a in point and b in point})
    checks.update({'N': {'answers_all': pop.n, 'prmb_answers': int(pop.prm.sum()), 'prmb_noncontrol_answers': int(pop.noncontrol.sum()),
                         'prmb_noncontrol_steps': int(nc_steps.sum()), 'prmb_control_answers': int((pop.prm & ~pop.noncontrol).sum()),
                         'prmb_source_groups': G, 'within_auc_eligible_answers': len(elig), 'pb_error_answers': len(pb_err), 'pb_source_groups_error_answers': H,
                         'pb_cells': cellnames, 'steps_all': pop.total, 'draws': draws},
                   'official_policy': 'control (classification correct) answers excluded from pooled totals; every PRMB step of non-control answers included; no empty steps in the development population (min steps per answer = %d)' % int(ns.min()),
                   'seconds': time.perf_counter() - t0})
    dump(out_dir / 'THRESHOLDS.json', thresholds); dump(out_dir / 'EVAL_CHECKS.json', checks)
    return {'point': point, 'contrasts': CT, 'checks': checks}
