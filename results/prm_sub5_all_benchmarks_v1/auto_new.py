"""AUTO_N16 (free L-SML grouping over [z(PRM), N16]: the PRM is one of the channels) on the backbone cells that
prm_boost_math_rigor_v1/stage8 did not cover (ProcessBench, CRV-Arith, CRV-GSM8K, PRM800K, DeltaBench), beside
CHAIN_N16 (PRM forced into its own view) and PRMG_N16 (PRM forced into its own group, one stage).

auto() and chain() are imported unchanged from stage8_auto_forced_k.py; cells and metrics from sub5_all.py.
Parity: chain() must reproduce the stored CHAIN_N16 to 1e-5 in every cell, and AUTO on PRMBench Qwen-7B must
reproduce stage8's 69.865 PRMScore GAP.

    python auto_new.py      # -> AUTO_NEW.csv, AUTO_NEW_PARITY.json
"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sub5_all as A  # noqa: E402  (redirects E.load / E.channels to every cell)
sys.path.insert(0, str(A.ROOT / 'results/prm_boost_math_rigor_v1'))
import stage8_auto_forced_k as S8  # noqa: E402

E = A.E
BENCHES = ['processbench', 'crv_arith', 'crv_gsm8k', 'prm800k', 'deltabench']


def cell(bench, prm):
    key = f'cab|prm|{bench}'
    A.REG[key] = dict(mode='prm', bench=bench, common=A.common_uids(bench))
    c = A.E.load(key, prm); c.n_gold = len(c.ids)
    V = E.channels(c, 'N16'); zp = E.answer_z(c.arms['PRM'], c.off)    # CHAIN_N16's PRM view (logit risk)
    s_ch, _ = S8.chain(c, V, zp)
    d_ch = float(np.max(np.abs(s_ch - c.arms['CHAIN_N16'])))
    s_au, info = S8.auto(V, zp)
    c.arms['AUTO_N16'] = s_au
    c.arms['CHAIN_REFIT'] = s_ch      # CHAIN refitted on exactly the answers AUTO sees (differs from the stored arm only
                                      # on DeltaBench, whose stored CHAIN was fitted on every answer the PRM scored)
    return c, d_ch, info


def main():
    rows, par = [], []
    # stage8 parity: PRMBench Qwen-7B AUTO
    c = E.load('prmbench', 'qwen_prm_7b'); V = E.channels(c, 'N16'); zp = E.answer_z(c.arms['PRM_Q80'], c.off)
    s, _ = S8.auto(V, zp); c.arms['AUTO_N16'] = s
    got = next(r['PRMScore_GAP'] for r in A.CMP.prism_rows(c, ['AUTO_N16']))
    ref = next(r for r in json.loads((A.ROOT / 'results/prm_boost_math_rigor_v1/stage8/AUTO_FORCEDK_RESULTS.json').read_text())
               if r['bench'] == 'prmbench' and r['prm'] == 'qwen_prm_7b')['A']['auto']['gap_prmscore']
    par.append(dict(check='AUTO PRMBench qwen_prm_7b vs stage8', got=got, ref=ref, ok=abs(got - ref) < 1e-6)); print(par[-1], flush=True)
    del c
    for bench in BENCHES:
        for prm in A.PRMS:
            c, d_ch, info = cell(bench, prm)
            par.append(dict(check=f'CHAIN refit {bench}/{prm}', max_abs_diff=d_ch, ok=d_ch < 1e-5 or bench == 'deltabench',
                            note='DeltaBench: stored CHAIN fitted on all scored answers, refit on the 447-answer common set' if bench == 'deltabench' else ''))
            arms = [(a, a, 'GAP') for a in ('PRM', 'CHAIN_N16', 'CHAIN_REFIT', 'PRMG_N16', 'AUTO_N16')]
            R = A.rows_pb(c, arms) if bench == 'processbench' else A.rows_cab(c, 'prm', bench, prm, arms)
            for r in R:
                rows.append(dict(benchmark=bench, prm=prm, prm_alone_in_auto=info['prm_alone'], auto_K=info['K'], **r))
            print(bench, prm, 'chain parity', d_ch, 'auto', info, flush=True)
            del c
    E.write_csv(HERE / 'AUTO_NEW.csv', rows)
    json.dump(par, open(HERE / 'AUTO_NEW_PARITY.json', 'w'), indent=1, default=float)
    print('parity failures', [p for p in par if not p['ok']])


if __name__ == '__main__':
    main()
