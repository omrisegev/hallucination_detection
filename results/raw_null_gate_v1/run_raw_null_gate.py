"""Raw-score clean-answer gate (pre-registration docs/experiments/RAW_NULL_GATE_PREREG_2026-10-30.md).

RUNBOOK: docs/handoffs/RAW_NULL_GATE_RUNBOOK_2026-10-30.md. Run the commands in this order and stop at the first
failure; never edit this file, spectral_utils/raw_null_gate.py or PROTOCOL.json (the hashes are checked).

    python run_raw_null_gate.py selftest   # synthetic, no data: the whole per-cell pipeline on a fake cell
    python run_raw_null_gate.py check      # every cell: loads, raw parity, fused-score parity -> CHECK.json
    python run_raw_null_gate.py run        # every cell: label-free stage saved first, then scoring -> RESULTS.csv,
                                           #   VERDICT.json, FITS.json, labelfree/*.npz
Arms: AUTO (PRM + N16, backbone), TELE16 and SUB5, each on the backbone ('prm') and standalone ('sa') telemetry.
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_k, '4')
import csv, hashlib, json, sys, time, traceback
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from spectral_utils import raw_null_gate as G  # noqa: E402

ARMS = [('AUTO', 'prm'), ('TELE16', 'prm'), ('TELE16', 'sa'), ('SUB5', 'prm'), ('SUB5', 'sa')]
BENCHES = ['prmbench', 'processbench', 'crv_arith', 'crv_gsm8k']
PRMS = ['qwen_prm_7b', 'qwen_prm_72b', 'pure_prm', 'rlhflow_prm']
GATES = ['G1', 'G1b', 'G2', 'G3']
N16N = ['q15_H1', 'q15_VE1', 'chosen_surprisal', 'logprob_margin', 'true_tail50', 'energy_level', 'energy_innovation',
        'top15_turnover', 'top50_js', 'dominant_freq16', 'bocpd_p0', 'realized_z', 'realized_drv', 'digit_alternative',
        'digit_spread', 'digit_alternative_innovation']
SUB5 = ['q15_H1', 'chosen_surprisal', 'top15_turnover', 'dominant_freq16', 'bocpd_p0', 'digit_spread']
FROZEN = ['spectral_utils/raw_null_gate.py', 'spectral_utils/fusion_utils.py', 'spectral_utils/prm_anchored_lsml.py',
          'tests/test_raw_null_gate.py', 'results/raw_null_gate_v1/run_raw_null_gate.py',
          'results/raw_null_gate_v1/verify_raw_null_gate.py']
EXPECTED_MISSING = {(arm, 'prm', 'prmbench', prm) for arm in ('AUTO', 'TELE16', 'SUB5') for prm in PRMS}


def sha(p):
    """SHA-256 of a file with CRLF normalised to LF, so a Windows checkout hashes like the Mac one."""
    return hashlib.sha256((ROOT / p).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def hashes():
    missing = [p for p in FROZEN if not (ROOT / p).exists()]
    if missing:
        raise SystemExit(f'STOP: frozen files are missing: {missing}. Check out the branch again (git status).')
    return {p: sha(p) for p in FROZEN}


def check_frozen():
    """Refuse to run if any frozen file differs from PROTOCOL.json (written when the code was frozen)."""
    proto = json.loads((HERE / 'PROTOCOL.json').read_text())
    now = hashes()
    bad = [p for p in FROZEN if proto['sha256'].get(p) != now[p]]
    if bad:
        raise SystemExit(f'STOP: frozen files changed since PROTOCOL.json: {bad}. Do not edit them; restore with '
                         f'`git checkout -- <file>`; never edit PROTOCOL.json.')
    return proto, now


# ----------------------------------------------------------------------------------------------- per cell (pure)
def fit_arm(arm, X_az, prm_risk_az=None):
    """The arm's usual answer-z L-SML, as a linear map. AUTO: [z(PRM), N16]; TELE16/SUB5: channels, sign by mean."""
    if arm == 'AUTO':
        V = np.column_stack([prm_risk_az, X_az])
        return (V,) + G.linear_lsml(V)
    return (X_az,) + G.linear_lsml(X_az, anchor=X_az.mean(1))


def anchor_group(p):
    """AUTO: the group that contains the PRM (column 0)."""
    for g, (idx, _) in enumerate(p['gw']):
        if 0 in set(int(i) for i in idx):
            return g
    return None


def labelfree_cell(arm, off, X_az, Z_raw, prm_risk_az, zr_prm, gap_fn):
    """Everything that does not touch labels. Z_raw: raw channels cleaned by votes_e1b.clean (pooled z, pooled
    cubic in position; NaN where undefined). zr_prm: the PRM risk cleaned the same way (AUTO only).
    The fused readout uses NaN -> 0; the per-group gates G2/G3 use each group's defined steps only."""
    V, z_fused, p = fit_arm(arm, X_az, prm_risk_az)
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
    ag = anchor_group(p) if arm == 'AUTO' else None
    out = {'G1': G.gate_g1(raw_fused, z_fused, off),
           'G1b': G.gate_g1b(raw_fused, z_fused, off, f_gap),
           'G2': G.gate_g2(raw_groups, orient, off, defined=defined),
           'G3': G.gate_g3(raw_groups, orient, off, anchor_group=ag, defined=defined)}
    info = dict(K=p['K'], groups=p['c'], parity_linear=p['parity'], anchor_group=ag, orientation=orient.tolist(),
                cross_weights=np.asarray(p['cw']).tolist(), sign=p['sign'], share_defined=defined.mean(0).tolist(),
                g1_fit=out['G1']['fit'], g2_pi0=out['G2'].get('pi0'), g2_pi0_groups=out['G2'].get('pi0_groups'),
                g2_groups_used=out['G2'].get('groups_used'), g2_groups=out['G2'].get('groups'),
                g3_prevalence=out['G3'].get('prevalence'), g3_psi=out['G3'].get('psi'), g3_eta=out['G3'].get('eta'),
                g3_vote_rate=out['G3'].get('vote_rate'), g3_sml_ranking=out['G3'].get('sml_ranking'),
                g3_groups_used=out['G3'].get('groups_used'), g3_anchor_used=out['G3'].get('anchor_used'),
                g3_groups=out['G3'].get('groups'), g2_reason=out['G2'].get('reason', ''),
                g3_reason=out['G3'].get('reason', ''))
    return dict(z_fused=z_fused, raw_fused=raw_fused, f_gap=f_gap, gates=out, info=info)


def ece(p, y, bins=10):
    m = np.isfinite(p); p, y = p[m], y[m]
    if not len(p):
        return float('nan')
    idx = np.minimum((np.argsort(np.argsort(p, kind='stable'), kind='stable') * bins) // len(p), bins - 1)
    return float(sum(abs(p[idx == k].mean() - y[idx == k].mean()) * (idx == k).mean() for k in range(bins) if (idx == k).any()))


def score_cell(lf, off, has_err, ev, step_err, stepmask, metric_fn):
    """Labels enter only here. Returns one row per gate (and the GAP-only reference)."""
    n = np.diff(off)
    rows = [dict(gate='GAP_ONLY', metric=metric_fn(lf['f_gap']), closed_clean=0., kept_error=1., share_called_error=1.)]
    he = has_err[ev]
    for name in GATES:
        g = lf['gates'][name]
        ok = bool(g.get('ok', True)) and (name != 'G1' or g['fit']['ok'])
        clean = np.asarray(g['clean'], bool)
        cv = clean[ev]
        r = dict(gate=name, ok=ok, reason=g.get('reason', '' if ok else g.get('fit', {}).get('reason', '')),
                 closed_clean=float(cv[~he].mean()) if (~he).any() else float('nan'),
                 kept_error=float((~cv[he]).mean()) if he.any() else float('nan'),
                 share_called_error=float((~cv).mean()), true_share_error=float(he.mean()),
                 metric=metric_fn(lf['f_gap'] & ~np.repeat(clean, n)))
        if 'p_err_step' in g:
            r['ece_step'] = ece(np.asarray(g['p_err_step'])[stepmask], step_err[stepmask])
        rows.append(r)
    return rows


# ----------------------------------------------------------------------------------------------- data (other machine)
def env():
    """The other machine's loaders, imported read-only (branch claude/ds-gate-2026-10-29)."""
    R = ROOT / 'results'
    for d in ('ds_gate_e1b_fuse', 'ds_gate_v1b_fixed', 'bracha_lr_transfer_v1', 'prm_boost_math_rigor_v1'):
        sys.path.insert(0, str(R / d))
    import votes_e1b as VE  # noqa: E402
    import run_e1 as R1  # noqa: E402
    return VE, R1


def load_cell(VE, R1, arm, mode, bench, prm):
    L, E = VE.L, VE.E
    c = L.load(bench, prm, mode)
    X_az = E.channels(c, 'N16'); R1.check_names(c)
    Xraw, act = VE.raw_matrix(mode, bench, prm, c)              # raises FileNotFoundError if absent; asserts parity
    if arm == 'SUB5':
        idx = [N16N.index(nm) for nm in SUB5]; X_az, Xraw, act = X_az[:, idx], Xraw[:, idx], act[:, idx]
    off = c.off; n = np.diff(off); pool = np.repeat(n, n) > 1
    Z_raw = VE.clean(Xraw, act, off, pool)
    prm_risk_az = zr_prm = None
    if arm == 'AUTO':
        risk = np.asarray(c.arms['PRM_Q80' if bench == 'prmbench' else 'PRM'], float)
        prm_risk_az = E.answer_z(risk, off)
        zr_prm = np.nan_to_num(VE.clean(risk[:, None], np.ones((len(risk), 1), bool), off, pool)[:, 0], nan=0.)
    return c, X_az, Z_raw, prm_risk_az, zr_prm


def fused_parity(VE, R1, arm, mode, bench, prm, X_az, prm_risk_az):
    """The answer-z fused score must equal the shipped one: stage8 S8.fit / S8.auto, and run_e1.arm_inputs where
    that arm/mode exists there (TELE16/SUB5 standalone, AUTO backbone)."""
    S8 = VE.L.S8
    _, mine, _ = fit_arm(arm, X_az, prm_risk_az)
    ref = S8.auto(X_az, prm_risk_az)[0] if arm == 'AUTO' else S8.fit(X_az, anchor=X_az.mean(1))[0]
    d1 = float(np.max(np.abs(mine - np.asarray(ref, float))))
    d2 = None
    if (arm == 'AUTO' and mode == 'prm') or (arm in ('TELE16', 'SUB5') and mode == 'sa'):
        d2 = float(np.max(np.abs(mine - np.asarray(R1.arm_inputs(arm, bench, prm)[3], float))))
    return d1, d2


def labels_of(c, bench):
    """has_err per answer, the evaluated answers, step labels and mask: run_e1.run_cell's own definitions."""
    off = c.off; n = np.diff(off)
    keep_gold = c.keep_gold.astype(bool) if bench == 'processbench' else None
    keep_eval = keep_gold if bench == 'processbench' else c.keep
    has_err = np.array([c.err[a:b][keep_eval[a:b]].any() for a, b in zip(off[:-1], off[1:])])
    ev = (c.cls != 'correct') if bench == 'prmbench' else np.ones(len(n), bool)
    stepmask = keep_eval & np.repeat(ev, n)
    return has_err, ev, c.err.astype(float), stepmask, keep_gold


# ----------------------------------------------------------------------------------------------- commands
def write_rows(rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(HERE / 'RESULTS.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, keys); w.writeheader(); w.writerows(rows)


def cmd_check():
    """Every cell: load, raw parity (inside votes_e1b.raw_matrix), fused-score parity, and the whole LABEL-FREE stage
    (so a crash shows here, not in the middle of `run`). No label is read. Missing raw data must be exactly the 12
    expected PRMBench backbone cells; anything else missing is a FAIL (wrong paths look like missing files)."""
    proto, now = check_frozen()
    VE, R1 = env()
    out, fail = [], False
    for arm, mode in ARMS:
        for bench in BENCHES:
            for prm in PRMS:
                r = dict(arm=arm, mode=mode, bench=bench, prm=prm)
                try:
                    c, X_az, Z_raw, pz, zr = load_cell(VE, R1, arm, mode, bench, prm)
                    d1, d2 = fused_parity(VE, R1, arm, mode, bench, prm, X_az, pz)
                    lf = labelfree_cell(arm, c.off, X_az, Z_raw, pz, zr, lambda s: R1.gap_flags(c, s))
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
    (HERE / 'CHECK.json').write_text(json.dumps(dict(protocol=proto['name'], sha256=now, cells=out), indent=1,
                                                default=float))
    print('CHECK', 'FAILED' if fail else 'PASSED', sum(r['status'] == 'ok' for r in out), 'ok,',
          sum(r['status'] == 'missing' for r in out), 'missing')
    if fail:
        raise SystemExit(1)


def cmd_run():
    proto, now = check_frozen()
    chk = json.loads((HERE / 'CHECK.json').read_text())
    if chk.get('sha256') != now:
        raise SystemExit('STOP: CHECK.json was made with different code; run `check` again first.')
    if any(r['status'].startswith('FAIL') for r in chk['cells']):
        raise SystemExit('STOP: CHECK.json has failures; run `check` and fix the environment first.')
    VE, R1 = env()
    (HERE / 'labelfree').mkdir(exist_ok=True)
    rows, fits = [], []
    for cell in chk['cells']:
        if cell['status'] != 'ok':
            continue
        arm, mode, bench, prm = cell['arm'], cell['mode'], cell['bench'], cell['prm']
        t0 = time.time()
        c, X_az, Z_raw, pz, zr = load_cell(VE, R1, arm, mode, bench, prm)
        lf = labelfree_cell(arm, c.off, X_az, Z_raw, pz, zr, lambda s: R1.gap_flags(c, s))
        tag = f'{arm}_{mode}_{bench}_{prm}'
        np.savez_compressed(HERE / 'labelfree' / f'{tag}.npz', z_fused=lf['z_fused'], raw_fused=lf['raw_fused'],
                            f_gap=lf['f_gap'], **{f'clean_{g}': np.asarray(lf['gates'][g]['clean'], bool) for g in GATES})
        fits.append(dict(arm=arm, mode=mode, bench=bench, prm=prm, **lf['info']))
        (HERE / 'FITS.json').write_text(json.dumps(fits, indent=1, default=float))
        # ---------------- labels from here on (the label-free outputs of this cell are already on disk)
        has_err, ev, step_err, stepmask, keep_gold = labels_of(c, bench)
        metric_fn = lambda f: R1.metric(c, bench, f, keep_gold)
        for r in score_cell(lf, c.off, has_err, ev, step_err, stepmask, metric_fn):
            rows.append(dict(arm=arm, mode=mode, bench=bench, prm=prm, **r))
        write_rows(rows)                                                     # incremental: a crash keeps the rows
        print(time.strftime('%H:%M:%S'), tag, f'{time.time() - t0:.0f}s',
              {r['gate']: (round(r['closed_clean'], 3), round(r['kept_error'], 3)) for r in rows[-5:]}, flush=True)
    (HERE / 'VERDICT.json').write_text(json.dumps(verdict(rows), indent=1, default=float))
    print('RUN DONE')


def verdict(rows):
    """Pre-registered rule, per (arm, mode, gate): Section 5 of the pre-registration. 'incomplete' when a required
    benchmark has fewer than 4 PRM cells (never read as a failure of the method)."""
    out = {}
    for arm, mode in ARMS:
        for gate in GATES:
            rs = [r for r in rows if r['arm'] == arm and r['mode'] == mode and r['gate'] == gate]
            if not rs:
                out[f'{arm}|{mode}|{gate}'] = dict(status='no cells'); continue
            good = lambda r: bool(r.get('ok', True)) and r['closed_clean'] >= .30 and r['kept_error'] >= .85
            per = {}
            for b in ('processbench', 'crv_arith', 'crv_gsm8k'):
                bb = [r for r in rs if r['bench'] == b]
                per[b] = dict(n=len(bb), n_good=sum(good(r) for r in bb))
            gap = {(r['bench'], r['prm']): r['metric'] for r in rows
                   if r['arm'] == arm and r['mode'] == mode and r['gate'] == 'GAP_ONLY'}
            pb = [r for r in rs if r['bench'] == 'prmbench']
            pb_ok = sum(r['metric'] >= gap[(r['bench'], r['prm'])] - .5 for r in pb)
            if any(v['n'] < 4 for v in per.values()):
                out[f'{arm}|{mode}|{gate}'] = dict(status='incomplete', per_benchmark=per); continue
            passes = all(v['n_good'] >= 3 for v in per.values()) and (not pb or pb_ok >= 3)
            out[f'{arm}|{mode}|{gate}'] = dict(status='complete', per_benchmark=per,
                                               prmbench_no_harm=f'{pb_ok}/{len(pb)}', passes=bool(passes))
    return out


def cmd_selftest():
    """The per-cell pipeline on a synthetic fake cell (no data, no loaders), including a group with 30% undefined
    (NaN) steps as digit channels have; then the unit tests."""
    import subprocess
    sys.path.insert(0, str(ROOT / 'tests'))
    import test_raw_null_gate as T  # noqa: E402
    rng = np.random.default_rng(11)
    X, off, y, has = T.synthetic_cell(rng, d=(3.0, 2.8, 2.5, 2.3))
    X_az = T.answer_z(X, off)
    Z_raw = np.column_stack([G.pooled_z(X[:, j]) for j in range(X.shape[1])])
    Z_raw[rng.random(len(y)) < .3, -2:] = np.nan                           # last group (2 channels) undefined on 30%
    risk = 2.5 * y + rng.standard_normal(len(y))
    gap_fn = lambda s: T.topk_flags(s, off, T.gap_counts(s, off))
    n = np.diff(off); ev = np.ones(len(n), bool); stepmask = np.ones(len(y), bool)
    metric_fn = lambda f: float((f == y).mean())
    for arm in ('TELE16', 'AUTO'):
        lf = labelfree_cell(arm, off, X_az, Z_raw, T.answer_z(risk[:, None], off)[:, 0] if arm == 'AUTO' else None,
                            G.pooled_z(risk) if arm == 'AUTO' else None, gap_fn)
        assert min(lf['info']['share_defined']) < .99, 'the NaN group was not detected as partly undefined'
        rows = score_cell(lf, off, has, ev, y.astype(float), stepmask, metric_fn)
        for r in rows:
            print('selftest', arm, r['gate'], {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != 'gate'})
            if r['gate'] != 'GAP_ONLY':
                assert r['closed_clean'] > .5 and r['kept_error'] > .85, (arm, r)
    v = verdict([dict(arm='TELE16', mode='sa', bench=b, prm=p, gate=g, ok=True, closed_clean=.5, kept_error=.9, metric=1.)
                 for b in BENCHES for p in PRMS for g in ['GAP_ONLY'] + GATES])
    assert v['TELE16|sa|G1']['passes'] and v['AUTO|prm|G1']['status'] == 'no cells', v
    rc = subprocess.call([sys.executable, '-m', 'pytest', '-q', str(ROOT / 'tests/test_raw_null_gate.py')])
    if rc:
        raise SystemExit('STOP: unit tests failed')
    print('SELFTEST PASSED')


if __name__ == '__main__':
    {'selftest': cmd_selftest, 'check': cmd_check, 'run': cmd_run}[sys.argv[1] if len(sys.argv) > 1 else 'selftest']()
