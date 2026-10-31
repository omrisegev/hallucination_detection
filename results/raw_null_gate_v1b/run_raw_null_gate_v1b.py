"""Raw-null gate, bug-fix release v1b (spec: docs/experiments/RAW_NULL_GATE_V1B_BUGFIX_2026-10-31.md).

Same cells, loaders, labels and endpoints as v1 (results/raw_null_gate_v1/run_raw_null_gate.py, imported read-only as a
library); the gates come from spectral_utils/raw_null_gate_v1b.py. Two variants, written to results/raw_null_gate_v1b/<v>/:
    fix       F1, F2, F4, F5, F6 (the bug fixes)
    fix_mono  the same plus F3 (monotone P(err); a known-practice correction, reported separately)

    python run_raw_null_gate_v1b.py selftest
    python run_raw_null_gate_v1b.py check <variant>
    python run_raw_null_gate_v1b.py run <variant>
    python run_raw_null_gate_v1b.py audit            # label-free FIX_AUDIT.json over v1, fix and fix_mono
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_k, '4')
import csv, hashlib, importlib.util, json, sys, time, traceback
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from spectral_utils import raw_null_gate_v1b as G  # noqa: E402

_spec = importlib.util.spec_from_file_location('run_v1', ROOT / 'results/raw_null_gate_v1/run_raw_null_gate.py')
V1 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(V1)   # library only: no v1 command is run

ARMS, BENCHES, PRMS, GATES = V1.ARMS, V1.BENCHES, V1.PRMS, V1.GATES
EXPECTED_MISSING = V1.EXPECTED_MISSING
VARIANTS = {'fix': False, 'fix_mono': True}                                   # variant -> monotone
FROZEN = ['spectral_utils/raw_null_gate_v1b.py', 'spectral_utils/raw_null_gate.py', 'spectral_utils/fusion_utils.py',
          'spectral_utils/prm_anchored_lsml.py', 'tests/test_raw_null_gate_v1b.py',
          'results/raw_null_gate_v1/run_raw_null_gate.py', 'results/raw_null_gate_v1b/run_raw_null_gate_v1b.py',
          'results/raw_null_gate_v1b/verify_raw_null_gate_v1b.py']


def sha(p):
    return hashlib.sha256((ROOT / p).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def check_frozen():
    proto = json.loads((HERE / 'PROTOCOL.json').read_text())
    missing = [p for p in FROZEN if not (ROOT / p).exists()]
    if missing:
        raise SystemExit(f'STOP: frozen files are missing: {missing}')
    now = {p: sha(p) for p in FROZEN}
    bad = [p for p in FROZEN if proto['sha256'].get(p) != now[p]]
    if bad:
        raise SystemExit(f'STOP: frozen files changed since PROTOCOL.json: {bad}')
    return proto, now


def out_dir(variant):
    if variant not in VARIANTS:
        raise SystemExit(f'variant must be one of {list(VARIANTS)}')
    d = HERE / variant; (d / 'labelfree').mkdir(parents=True, exist_ok=True)
    return d


# ----------------------------------------------------------------------------------------------- per cell
def labelfree_cell(arm, off, X_az, Z_raw, prm_risk_az, zr_prm, gap_fn, monotone):
    """v1's label-free stage with the v1b gates."""
    V, z_fused, p = V1.fit_arm(arm, X_az, prm_risk_az)
    if np.isinf(Z_raw).any() or (zr_prm is not None and not np.isfinite(zr_prm).all()):
        raise ValueError('infinite raw values')
    finite = np.isfinite(Z_raw)
    Zr = np.where(finite, Z_raw, 0.)
    if arm == 'AUTO':
        Zr = np.column_stack([zr_prm, Zr]); finite = np.column_stack([np.ones(len(zr_prm), bool), finite])
    defined = G.group_defined(finite, p)
    raw_fused = G.pooled_z(G.fused_linear(Zr, p))
    raw_groups = G.group_scores(Zr, p)
    orient = G.group_orientation(p)
    f_gap = gap_fn(z_fused)
    ag = V1.anchor_group(p) if arm == 'AUTO' else None
    out = {'G1': G.gate_g1(raw_fused, z_fused, off, monotone=monotone),
           'G1b': G.gate_g1b(raw_fused, z_fused, off, f_gap),
           'G2': G.gate_g2(raw_groups, orient, off, defined=defined, monotone=monotone),
           'G3': G.gate_g3(raw_groups, orient, off, anchor_group=ag, defined=defined, monotone=monotone)}
    o = out
    info = dict(K=p['K'], groups=p['c'], parity_linear=p['parity'], anchor_group=ag, orientation=orient.tolist(),
                cross_weights=np.asarray(p['cw']).tolist(), sign=p['sign'], share_defined=defined.mean(0).tolist(),
                g1_fit=o['G1']['fit'], g2_pi0=o['G2'].get('pi0'), g2_pi0_groups=o['G2'].get('pi0_groups'),
                g2_groups_used=o['G2'].get('groups_used'), g2_groups=o['G2'].get('groups'),
                g3_prevalence=o['G3'].get('prevalence'), g3_psi=o['G3'].get('psi'), g3_eta=o['G3'].get('eta'),
                g3_vote_rate=o['G3'].get('vote_rate'), g3_sml_ranking=o['G3'].get('sml_ranking'),
                g3_groups_used=o['G3'].get('groups_used'), g3_anchor_used=o['G3'].get('anchor_used'),
                g3_constant_voters=o['G3'].get('constant_voters'), g3_groups=o['G3'].get('groups'),
                g1_reason=o['G1'].get('reason', ''), g2_reason=o['G2'].get('reason', ''),
                g3_reason=o['G3'].get('reason', ''))
    return dict(z_fused=z_fused, raw_fused=raw_fused, f_gap=f_gap, gates=out, info=info)


def arrays(lf):
    """The npz contract of the spec: per-step and per-answer probabilities (NaN when the gate is invalid)."""
    g = lf['gates']; n_s, n_a = len(lf['z_fused']), len(g['G1']['clean'])
    nan = lambda k: np.full(k, np.nan)
    get = lambda gate, key, k: np.asarray(g[gate][key], float) if g[gate].get('ok', True) and key in g[gate] else nan(k)
    return dict(z_fused=lf['z_fused'], raw_fused=lf['raw_fused'], f_gap=lf['f_gap'],
                **{f'clean_{x}': np.asarray(g[x]['clean'], bool) for x in GATES},
                p_err_step_G1=get('G1', 'p_err_step', n_s), p_err_step_G2=get('G2', 'p_err_step', n_s),
                p_err_worst_G1=get('G1', 'p_err_worst', n_a), p_err_worst_G2=get('G2', 'p_err_worst', n_a),
                p_answer_G1b=get('G1b', 'p_answer', n_a), p_answer_G3=get('G3', 'p_answer', n_a))


def ece(p, y, bins=10):
    """As v1, but ranks round(P(err), 9) (F6), so 1e-13 float differences cannot move a step across a bin edge."""
    m = np.isfinite(p); p, y = p[m], y[m]
    if not len(p):
        return float('nan')
    r = np.round(p, 9)
    idx = np.minimum((np.argsort(np.argsort(r, kind='stable'), kind='stable') * bins) // len(p), bins - 1)
    return float(sum(abs(p[idx == k].mean() - y[idx == k].mean()) * (idx == k).mean() for k in range(bins) if (idx == k).any()))


def score_cell(lf, off, has_err, ev, step_err, stepmask, metric_fn):
    """As v1's score_cell; `ok` comes from the gate (G1 now reports its own); ece per F6."""
    n = np.diff(off)
    rows = [dict(gate='GAP_ONLY', metric=metric_fn(lf['f_gap']), closed_clean=0., kept_error=1., share_called_error=1.)]
    he = has_err[ev]
    for name in GATES:
        g = lf['gates'][name]
        ok = bool(g.get('ok', True))
        cv = np.asarray(g['clean'], bool)[ev]
        r = dict(gate=name, ok=ok, reason=g.get('reason', ''),
                 closed_clean=float(cv[~he].mean()) if (~he).any() else float('nan'),
                 kept_error=float((~cv[he]).mean()) if he.any() else float('nan'),
                 share_called_error=float((~cv).mean()), true_share_error=float(he.mean()),
                 metric=metric_fn(lf['f_gap'] & ~np.repeat(np.asarray(g['clean'], bool), n)))
        if 'p_err_step' in g:
            r['ece_step'] = ece(np.asarray(g['p_err_step'])[stepmask], step_err[stepmask])
        rows.append(r)
    return rows


def verdict(rows):
    """v1's pass rule; PRMBench no-harm counts only valid gates (F5)."""
    out = {}
    for arm, mode in ARMS:
        for gate in GATES:
            rs = [r for r in rows if r['arm'] == arm and r['mode'] == mode and r['gate'] == gate]
            if not rs:
                out[f'{arm}|{mode}|{gate}'] = dict(status='no cells'); continue
            valid = lambda r: str(r.get('ok', True)) not in ('False', 'false', '0')
            good = lambda r: valid(r) and float(r['closed_clean']) >= .30 and float(r['kept_error']) >= .85
            per = {}
            for b in ('processbench', 'crv_arith', 'crv_gsm8k'):
                bb = [r for r in rs if r['bench'] == b]
                per[b] = dict(n=len(bb), n_good=sum(good(r) for r in bb))
            gap = {(r['bench'], r['prm']): float(r['metric']) for r in rows
                   if r['arm'] == arm and r['mode'] == mode and r['gate'] == 'GAP_ONLY'}
            pb = [r for r in rs if r['bench'] == 'prmbench']
            pb_ok = sum(valid(r) and float(r['metric']) >= gap[(r['bench'], r['prm'])] - .5 for r in pb)
            if any(v['n'] < 4 for v in per.values()):
                out[f'{arm}|{mode}|{gate}'] = dict(status='incomplete', per_benchmark=per); continue
            passes = all(v['n_good'] >= 3 for v in per.values()) and (not pb or pb_ok >= 3)
            out[f'{arm}|{mode}|{gate}'] = dict(status='complete', per_benchmark=per,
                                               prmbench_no_harm=f'{pb_ok}/{len(pb)}', passes=bool(passes))
    return out


# ----------------------------------------------------------------------------------------------- commands
def cmd_check(variant):
    proto, now = check_frozen()
    d = out_dir(variant); mono = VARIANTS[variant]
    VE, R1 = V1.env()
    out, fail = [], False
    for arm, mode in ARMS:
        for bench in BENCHES:
            for prm in PRMS:
                r = dict(arm=arm, mode=mode, bench=bench, prm=prm)
                try:
                    c, X_az, Z_raw, pz, zr = V1.load_cell(VE, R1, arm, mode, bench, prm)
                    d1, d2 = V1.fused_parity(VE, R1, arm, mode, bench, prm, X_az, pz)
                    lf = labelfree_cell(arm, c.off, X_az, Z_raw, pz, zr, lambda s: R1.gap_flags(c, s), mono)
                    r.update(status='ok', n_answers=len(c.off) - 1, parity_fit=d1, parity_shipped=d2, K=lf['info']['K'])
                    if d1 > 1e-6 or (d2 is not None and d2 > 1e-6):
                        r['status'] = 'FAIL parity'; fail = True
                except FileNotFoundError as ex:
                    if (arm, mode, bench, prm) in EXPECTED_MISSING:
                        r.update(status='missing', reason=str(ex))
                    else:
                        r.update(status='FAIL unexpected missing file', reason=str(ex)); fail = True
                except Exception as ex:                                      # noqa: BLE001
                    r.update(status='FAIL', reason=f'{type(ex).__name__}: {ex}', trace=traceback.format_exc()[-1500:])
                    fail = True
                out.append(r); print(r['status'], arm, mode, bench, prm, r.get('reason', '')[:120], flush=True)
    (d / 'CHECK.json').write_text(json.dumps(dict(protocol=proto['name'], variant=variant, sha256=now, cells=out),
                                             indent=1, default=float))
    print('CHECK', 'FAILED' if fail else 'PASSED', sum(r['status'] == 'ok' for r in out), 'ok,',
          sum(r['status'] == 'missing' for r in out), 'missing')
    if fail:
        raise SystemExit(1)


def cmd_run(variant):
    proto, now = check_frozen()
    d = out_dir(variant); mono = VARIANTS[variant]
    chk = json.loads((d / 'CHECK.json').read_text())
    if chk.get('sha256') != now:
        raise SystemExit('STOP: CHECK.json was made with different code; run `check` again first.')
    if any(r['status'].startswith('FAIL') for r in chk['cells']):
        raise SystemExit('STOP: CHECK.json has failures.')
    VE, R1 = V1.env()
    rows, fits = [], []
    for cell in chk['cells']:
        if cell['status'] != 'ok':
            continue
        arm, mode, bench, prm = cell['arm'], cell['mode'], cell['bench'], cell['prm']
        t0 = time.time()
        c, X_az, Z_raw, pz, zr = V1.load_cell(VE, R1, arm, mode, bench, prm)
        lf = labelfree_cell(arm, c.off, X_az, Z_raw, pz, zr, lambda s: R1.gap_flags(c, s), mono)
        tag = f'{arm}_{mode}_{bench}_{prm}'
        np.savez_compressed(d / 'labelfree' / f'{tag}.npz', **arrays(lf))
        fits.append(dict(arm=arm, mode=mode, bench=bench, prm=prm, **lf['info']))
        (d / 'FITS.json').write_text(json.dumps(fits, indent=1, default=float))
        # ---------------- labels from here on (the label-free outputs of this cell are already on disk)
        has_err, ev, step_err, stepmask, keep_gold = V1.labels_of(c, bench)
        metric_fn = lambda f: R1.metric(c, bench, f, keep_gold)
        for r in score_cell(lf, c.off, has_err, ev, step_err, stepmask, metric_fn):
            rows.append(dict(arm=arm, mode=mode, bench=bench, prm=prm, **r))
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(d / 'RESULTS.csv', 'w', newline='') as fh:
            w = csv.DictWriter(fh, keys); w.writeheader(); w.writerows(rows)
        print(time.strftime('%H:%M:%S'), tag, f'{time.time() - t0:.0f}s',
              {r['gate']: (round(r['closed_clean'], 3), round(r['kept_error'], 3)) for r in rows[-5:]}, flush=True)
    (d / 'VERDICT.json').write_text(json.dumps(verdict(rows), indent=1, default=float))
    print('RUN DONE')


def _nonmono_pairs(score, p, mu0):
    """Adjacent pairs (sorted by score, above mu0, distinct scores) where P(err) falls."""
    m = np.isfinite(p) & (score > mu0)
    o = np.argsort(score[m], kind='stable'); s, q = score[m][o], p[m][o]
    return int(np.sum((np.diff(q) < -1e-12) & (np.diff(s) > 0)))


def cmd_audit():
    """Label-free proof that the fixes act, from FITS.json and the labelfree arrays of v1, fix and fix_mono."""
    src = {'v1': ROOT / 'results/raw_null_gate_v1', 'fix': HERE / 'fix', 'fix_mono': HERE / 'fix_mono'}
    fits = {k: {(f['arm'], f['mode'], f['bench'], f['prm']): f for f in json.loads((v / 'FITS.json').read_text())}
            for k, v in src.items()}
    verd = {k: json.loads((v / 'VERDICT.json').read_text()) for k, v in src.items()}
    cells = sorted(fits['v1'])
    grp = lambda f, key: [g for g in (f.get(key) or [])]
    # F2 was dropped; reported only: fits above the pure-Gaussian 99th percentile of pi0_raw at the smallest n (1.08)
    unid = lambda d: bool(d.get('ok')) and ((d.get('pi0_raw') or 0) > 1.08)
    res = {}
    for k in src:
        F = fits[k]
        grp_fits = [g for c in cells for g in grp(F[c], 'g2_groups')]
        res[k] = dict(
            g1_fits_ok=sum(bool(F[c]['g1_fit'].get('ok')) for c in cells),
            g1_fits_ok_with_pi0_raw_gt_1_08=sum(unid(F[c]['g1_fit']) for c in cells),
            group_fits_ok=sum(bool(g.get('ok')) for g in grp_fits),
            group_fits_ok_with_pi0_raw_gt_1_08=sum(unid(g) for g in grp_fits),
            group_fits_total=len(grp_fits),
            g2_cells_valid=sum(F[c].get('g2_pi0') is not None for c in cells),
            g3_cells_valid=sum(F[c].get('g3_prevalence') is not None for c in cells),
            g3_cells_valid_with_constant_voter=sum(F[c].get('g3_vote_rate') is not None and
                                                   any(r in (0., 1.) for r in F[c]['g3_vote_rate']) for c in cells),
            g3_constant_voters_dropped=sum(len(F[c].get('g3_constant_voters') or []) for c in cells),
            prmbench_no_harm={e: v.get('prmbench_no_harm') for e, v in verd[k].items()},
            passes=sum(bool(v.get('passes')) for v in verd[k].values()))
    # F3: non-monotone pairs of G1 P(err) above mu0 (v1 npz lacks P(err): fix vs fix_mono only), and steps F3 changes
    mono = dict(g1_nonmono_pairs={}, g1_steps_changed=0, g2_steps_changed=0, g1_cells_checked=0)
    for v in ('fix', 'fix_mono'):
        mono['g1_nonmono_pairs'][v] = 0
    for c in cells:
        tag = '_'.join(c)
        A = {v: np.load(HERE / v / 'labelfree' / f'{tag}.npz') for v in ('fix', 'fix_mono')}
        f1 = fits['fix'][c]['g1_fit']
        if not f1.get('ok'):
            continue
        mono['g1_cells_checked'] += 1
        for v in ('fix', 'fix_mono'):
            mono['g1_nonmono_pairs'][v] += _nonmono_pairs(A[v]['raw_fused'], A[v]['p_err_step_G1'], f1['mu0'])
        for gk, key in (('g1', 'p_err_step_G1'), ('g2', 'p_err_step_G2')):
            a, b = A['fix'][key], A['fix_mono'][key]
            mono[f'{gk}_steps_changed'] += int(np.sum(np.isfinite(a) & (np.abs(a - b) > 1e-12)))
    res['F3'] = mono
    (HERE / 'FIX_AUDIT.json').write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'prmbench_no_harm'} for k, v in res.items()}, indent=1))
    print('AUDIT DONE')


def cmd_selftest():
    import subprocess
    rc = subprocess.call([sys.executable, '-m', 'pytest', '-q', str(ROOT / 'tests/test_raw_null_gate_v1b.py')])
    if rc:
        raise SystemExit('STOP: unit tests failed')
    sys.path.insert(0, str(ROOT / 'tests'))
    import test_raw_null_gate as T  # noqa: E402
    rng = np.random.default_rng(11)
    X, off, y, has = T.synthetic_cell(rng, d=(3.0, 2.8, 2.5, 2.3))
    X_az = T.answer_z(X, off)
    Z_raw = np.column_stack([G.pooled_z(X[:, j]) for j in range(X.shape[1])])
    gap_fn = lambda s: T.topk_flags(s, off, T.gap_counts(s, off))
    n = np.diff(off); ev = np.ones(len(n), bool); stepmask = np.ones(len(y), bool)
    for mono in (False, True):
        lf = labelfree_cell('TELE16', off, X_az, Z_raw, None, None, gap_fn, mono)
        rows = score_cell(lf, off, has, ev, y.astype(float), stepmask, lambda f: float((f == y).mean()))
        for r in rows:
            print('selftest mono' if mono else 'selftest', r['gate'],
                  {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != 'gate'})
    print('SELFTEST PASSED')


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'selftest'
    if cmd in ('check', 'run'):
        {'check': cmd_check, 'run': cmd_run}[cmd](sys.argv[2] if len(sys.argv) > 2 else '')
    else:
        {'selftest': cmd_selftest, 'audit': cmd_audit}[cmd]()
