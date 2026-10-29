"""Out-of-sample generalization in PRMBench's PRMScore form (0.5*F1(correct) + 0.5*F1(error), pooled steps) on every
benchmark, GAP rule for every arm: the PRM (same rule), AUTO (PRM + backbone telemetry) and TELE16 (standalone telemetry).
Socratic, CRV and DeltaBench already report this form (parity: equals their main metric); Hard2Verify (official: balanced
F1) and ProcessBench (official: first-error F1; here the prefix domain, steps up to the first error) are recomputed in it.

    python prmscore_form.py   # -> PRMSCORE_FORM.csv, PARITY.json
"""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'bracha_lr_transfer_v1'))
import lr_transfer as L  # noqa: E402  (cell loaders, as in Bracha's experiment)
A, E, S8 = L.A, L.E, L.S8
REF = {(r['benchmark'], r['telemetry'], r['prm'], r['arm'], r['metric']): float(r['value'])
       for r in E.read_csv(HERE.parent / 'bracha_lr_transfer_v1/RESULTS.csv')}
rows, par = [], []
form = lambda c, f: 100 * float(E.official_from_counts(E.counts_by_answer(c, f).sum(0), 'socratic'))
for bench in ['socratic', 'hard2verify', 'processbench', 'crv_arith', 'crv_gsm8k', 'deltabench']:
    for prm in L.PRMS:
        c = L.load(bench, prm, 'prm')
        cs = A._load('hard2verify', prm) if bench == 'hard2verify' else c
        cs.arms['AUTO_N16'], _ = S8.auto(E.channels(cs, 'N16'), L.zprm(cs, bench))
        if cs is not c:
            pos = {u: j for j, u in enumerate(cs.ids)}
            c.arms['AUTO_N16'] = np.concatenate([cs.arms['AUTO_N16'][cs.off[pos[u]]:cs.off[pos[u] + 1]] for u in c.ids])
        if bench == 'processbench': c.keep = c.keep_gold
        prm_arm = A.prm_scale_arm(bench, prm)
        for label, arm in (('PRM', prm_arm), ('AUTO', 'AUTO_N16')):
            rows.append(dict(benchmark=bench, prm=prm, arm=label, prmscore_form=form(c, c.flags(arm, 'GAP'))))
        cb = L.load(bench, prm, 'sa')
        if bench == 'processbench': cb.keep = cb.keep_gold
        rows.append(dict(benchmark=bench, prm=prm, arm='TELE16', prmscore_form=form(cb, cb.flags('TELE_N16', 'GAP'))))
        if bench in ('socratic', 'crv_arith', 'crv_gsm8k', 'deltabench'):
            for lab, tel, arm in (('PRM', 'backbone', 'PRM'), ('AUTO', 'backbone', 'AUTO_N16'), ('TELE16', 'standalone', 'TELE_N16')):
                got = next(r['prmscore_form'] for r in rows[-3:] if r['arm'] == lab)
                ref = REF[(bench, tel, prm, arm, 'macro_f1_GAP')]
                par.append(dict(check=f'{bench} {prm} {lab} = main metric', got=got, ref=ref, ok=abs(got - ref) < 1e-6))
        print(bench, prm, [round(r['prmscore_form'], 2) for r in rows[-3:]], flush=True)
        del c, cs, cb
E.write_csv(HERE / 'PRMSCORE_FORM.csv', rows)
json.dump(dict(n=len(par), failed=[p['check'] for p in par if not p['ok']], checks=par), open(HERE / 'PARITY.json', 'w'), indent=1)
print('parity failed', [p['check'] for p in par if not p['ok']])
