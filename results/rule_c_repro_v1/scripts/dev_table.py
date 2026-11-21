"""Dev-side table for rule C (PC reproduction). LABEL-USING evaluation only; rules come from the rebuilt RULES_<bench>.npz.
Arms: C, no gate, pi-hat sC gate (.580 PB / .704 PRMB, as eval.py), true-pi sC gate (context arm, uses dev labels).
Run: HD_ROOT=$WS/hd python dev_table.py $WS > out.json"""
import sys, json, numpy as np
WS = sys.argv[1]
sys.path.insert(0, f'{WS}/agentX1')
from common import load_dev, RA, L
PIHAT = {'processbench': .580, 'prmbench': .704}
R = {}
for bench in ('processbench', 'prmbench'):
    B = load_dev(bench, draws=2000)
    Cd, f0, sC, bd, gid = B['Cd'], B['f0'], B['sC'], B['bd'], B['gid']
    Z = np.load(f'{WS}/agentS2/out/RULES_{bench}.npz'); assert (Z['uids'] == Cd.uids).all()
    ex = lambda op: f0 & L.expand(op, Cd.off)
    pt = float(Cd.y_ans.mean())  # evaluation-only (context arm)
    arms = {'C': Z['flagsC'].astype(bool), 'no_gate': f0, 'ref_pihat_sC': ex(RA.gate_open(sC, PIHAT[bench])),
            'ctx_true_pi': ex(RA.gate_open(sC, pt))}
    M = {k: RA.score_flags(Cd, v, bd.W, gid) for k, v in arms.items()}
    keys = ['F1', 'SLA'] if bench == 'processbench' else ['PRMScore']
    res = {'true_pi': pt, 'n_answers': int(len(Cd.uids))}
    for k, v in arms.items():
        m = M[k]; d = {'zero_flag_share': float(np.mean(np.add.reduceat(v.astype(int), Cd.off[:-1]) == 0))}
        for kk in keys: d[kk] = float(m[kk]); d[kk + '_ci'] = L.ci(m['_boot'][kk])
        if bench == 'processbench': d.update(err_acc=float(m['err_acc']), clean_acc=float(m['clean_acc']))
        res[k] = d
    res['C_minus'] = {ref: {kk: RA.diff_ci(M['C'], M[ref], kk) for kk in keys} for ref in ('no_gate', 'ref_pihat_sC', 'ctx_true_pi')}
    R[bench] = res
print(json.dumps(RA.rnd(R, 4), indent=1, default=str))
