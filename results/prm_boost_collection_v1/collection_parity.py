"""Stage-2 parity anchors (plan section 2). Every number must reproduce before any table is built.

    python evaluate_collection.py parity      ->  PARITY_STAGE2.csv  (one row per checked number; PASS/FAIL + diff)

Anchors:
  A1 Qwen-7B PRMBench, plan values: PRM native 65.49; CHAIN_N16 Q80 68.25 / GAP 70.40 / k=2 70.80 / ORACLE 75.40;
     PRM prob Q80 67.32 / GAP 68.01; PRM logit Q80 65.82 / GAP 66.37. Tolerance 0.005 (values given to 2 decimals).
  A2 all 5 PRMs PRMBench: every row of the committed ABL5_RESULTS (results/prm_boost_ablation5_v1/, Math-Shepherd
     from digitfix_v1/digitfix/): PRMScore and within-AUC, recomputed from the exported scores. Tolerance 1e-9.
  A3 Socratic: results/prm_boost_ablation5_v1/socratic/RESULTS_socratic.csv (4 PRMs) and digitfix_v1/socratic/
     RESULTS_socratic.csv (Math-Shepherd, bank_v3): official metric and within-AUC of every arm. Tolerance 1e-9.
     Plus results/error_count_label_free_v1/gap_test/RESULTS.csv: CHAIN_N16 Q80 and GAP, Qwen-7B and 72B
     (GAP - Q80 = +1.37 / +1.01). Tolerance 1e-6.
  A4 Hard2Verify, 177-answer common set: refs/hard2verify_common_177_RESULTS.csv (copied from branch
     claude/prm-boost-external-2026-10-18, results/prm_boost_external_v1/eval_hard2verify/common_177/). The 4 PRMs
     with unchanged banks: every arm present on both sides; Math-Shepherd: PRM-only arms (its bank was rebuilt with
     the digit fix, so its fused arms are expected to change and are reported, not gated). Tolerance 1e-9.
"""
import numpy as np
import evaluate_collection as E

REPO_ABL = E.ROOT / 'results/prm_boost_ablation5_v1'
TOL_EXACT = 1e-9


def chk(rows, anchor, cell, quantity, ours, ref, tol, source, gated=True):
    o, r = float(ours), float(ref)
    if np.isnan(o) and np.isnan(r): ok, d = True, 0.
    else: d = o - r; ok = bool(np.isfinite(d) and abs(d) <= tol)
    rows.append(dict(anchor=anchor, cell=cell, quantity=quantity, ours=float(ours), reference=float(ref), diff=d,
                     tol=tol, status=('PASS' if ok else 'FAIL') if gated else ('INFO_same' if ok else 'INFO_changed'),
                     source=source))


def a1(rows):
    c = E.load_prmbench('qwen_prm_7b'); src = 'plan section 2 (values to 2 decimals)'
    exp = [('PRM_native', 'native', 65.49), ('CHAIN_N16', 'Q80', 68.25), ('CHAIN_N16', 'GAP', 70.40),
           ('CHAIN_N16', 'K2', 70.80), ('CHAIN_N16', 'ORACLE', 75.40), ('PRM_Q80_PROB', 'Q80', 67.32),
           ('PRM_Q80_PROB', 'GAP', 68.01), ('PRM_Q80', 'Q80', 65.82), ('PRM_Q80', 'GAP', 66.37)]
    for arm, rule, v in exp:
        r, _, _ = E.measure(c, arm, rule)
        chk(rows, 'A1', 'prmbench/qwen_prm_7b', f'{arm}[{rule}] PRMScore', round(r['official'], 2), v, .005, src)


def a2(rows):
    for prm in E.PRMS:
        ref_p = (REPO_ABL / 'digitfix_v1/digitfix' if prm == 'math_shepherd' else REPO_ABL) / f'ABL5_RESULTS_{prm}.csv'
        exp_p = E.DATA / 'prmbench' / f'ABL5_RESULTS_{prm}.csv'
        same = ref_p.read_bytes().replace(b'\r\n', b'\n') == exp_p.read_bytes().replace(b'\r\n', b'\n')
        chk(rows, 'A2', f'prmbench/{prm}', 'exported ABL5_RESULTS == committed file (bytes)', float(same), 1., 0,
            str(ref_p.relative_to(E.ROOT)))
        c = E.load_prmbench(prm)
        for r in E.read_csv(ref_p):
            ours, _, _ = E.measure(c, r['arm'], r['rule'])
            for q, k in (('PRMScore', 'prmscore'), ('within-AUC', 'within_auc')):
                chk(rows, 'A2', f'prmbench/{prm}', f"{r['arm']}[{r['rule']}] {q}",
                    ours['official' if k == 'prmscore' else k], r[k], TOL_EXACT, str(ref_p.relative_to(E.ROOT)))


def ext_rule(arm_csv):
    if arm_csv == 'PRM_native': return 'PRM', 'native'
    assert arm_csv.endswith('_Q80'), arm_csv
    a = arm_csv[:-4]
    return {'PRM': 'PRM', 'LSML_N16_PRMG': 'PRMG_N16', 'LSML_N13_PRMG': 'PRMG_N13', 'LSML_N11_PRMG': 'PRMG_N11'}.get(a, a), 'Q80'


def ext_check(rows, anchor, bench, prm, c, ref_rows, src, gated_fn):
    lsml = E.load_frozen_lsml(c)
    if lsml is not None: c.arms['LSML_CONT'] = lsml
    for r in ref_rows:
        arm, rule = ext_rule(r['arm'])
        if arm not in c.arms:
            rows.append(dict(anchor=anchor, cell=f'{bench}/{prm}', quantity=f"{r['arm']} (not in collection)",
                             status='SKIPPED', source=src)); continue
        ours, _, _ = E.measure(c, arm, rule)
        for q, k in (('official', 'official_metric'), ('within-AUC', 'within_auc')):
            chk(rows, anchor, f'{bench}/{prm}', f"{arm}[{rule}] {q}", ours['official' if q == 'official' else 'within_auc'],
                r[k], TOL_EXACT, src, gated=gated_fn(arm))


def a3(rows):
    for prm in E.PRMS:
        p = (REPO_ABL / 'digitfix_v1/socratic' if prm == 'math_shepherd' else REPO_ABL / 'socratic') / 'RESULTS_socratic.csv'
        ref = [r for r in E.read_csv(p) if r['model'] == E.EXT_NAME.get(prm, prm)]
        ext_check(rows, 'A3', 'socratic', prm, E.load_external('socratic', prm), ref, str(p.relative_to(E.ROOT)), lambda a: True)
    p = E.ROOT / 'results/error_count_label_free_v1/gap_test/RESULTS.csv'
    G = {(r['arm'], r['rule']): float(r['metric']) for r in E.read_csv(p) if r['bench'] == 'socratic'}
    for prm in ('qwen_prm_7b', 'qwen_prm_72b'):
        c = E.load_external('socratic', prm); key = E.EXT_NAME.get(prm, prm)
        v = {}
        for rule in ('Q80', 'GAP'):
            v[rule] = 100 * E.measure(c, 'CHAIN_N16', rule)[0]['official']
            chk(rows, 'A3', f'socratic/{prm}', f'CHAIN_N16[{rule}] PRMScore x100', v[rule], G[(f'{key}__CHAIN_N16', rule)], 1e-6,
                str(p.relative_to(E.ROOT)))
        chk(rows, 'A3', f'socratic/{prm}', 'CHAIN_N16 GAP - Q80 x100', v['GAP'] - v['Q80'],
            G[(f'{key}__CHAIN_N16', 'GAP')] - G[(f'{key}__CHAIN_N16', 'Q80')], 1e-6, str(p.relative_to(E.ROOT)))


def h2v_common_uids():
    sets = [set(np.load(E.DATA / 'h2v' / f'SCORES_{E.EXT_NAME.get(p, p)}.npz')['uids'].astype(str)) for p in E.PRMS]
    return set.intersection(*sets)


def a4(rows):
    p = E.HERE / 'refs/hard2verify_common_177_RESULTS.csv'
    common = h2v_common_uids()
    rows.append(dict(anchor='A4', cell='hard2verify', quantity='n common answers', ours=len(common), reference=177,
                     diff=len(common) - 177, tol=0, status='PASS' if len(common) == 177 else 'FAIL', source=str(p.relative_to(E.ROOT))))
    for prm in E.PRMS:
        ref = [r for r in E.read_csv(p) if r['model'] == E.EXT_NAME.get(prm, prm)]
        c = E.load_external('hard2verify', prm, uids_keep=common)
        gated = (lambda a: True) if prm != 'math_shepherd' else (lambda a: a in ('PRM', 'PRMPROB'))
        ext_check(rows, 'A4', 'hard2verify', prm, c, ref, str(p.relative_to(E.ROOT)), gated)


def main():
    rows = []
    for f in (a1, a2, a3, a4):
        try:
            f(rows)
        except FileNotFoundError as e:
            rows.append(dict(anchor=f.__name__.upper(), status='MISSING_INPUT', source=str(e)))
    E.write_csv(E.HERE / 'PARITY_STAGE2.csv', rows)
    from collections import Counter
    by = {}
    for r in rows: by.setdefault(r['anchor'], Counter())[r['status']] += 1
    for a, cnt in by.items(): print(a, dict(cnt))
    for r in rows:
        if r['status'] in ('FAIL', 'MISSING_INPUT'): print('  ', r)
    worst = max((abs(r['diff']) for r in rows if r.get('status') == 'PASS'), default=0)
    print('max |diff| among PASS rows:', worst)


if __name__ == '__main__':
    main()
