"""Stage 1 (label-free): scores, per-answer gap quantities, null p-values and every declared policy's per-answer
count. Writes STAGE1.npz. Reads no correctness label (PRMBench classification is read only in stage 2)."""
import json
import numpy as np
from gt_common import *  # noqa
from common_ec import load_tele_bank, answer_z  # noqa: E402

out = {}
# ---------------- PRMBench ----------------
bk = load_tele_bank('N16'); off = bk['offsets'].astype(np.int64); ids = bk['idx']
Sb = dict(np.load(BOOST_NPZ))
PB = {'B_CHAIN_N16': answer_z(Sb['CHAIN_N16'], off), 'T_LSML_CONT': bk['S__LSML_CONT'], 'T_ALL_EQ': bk['S__ALL_EQ_ORIENTED']}
# ---------------- Socratic ----------------
SOC = {}
for model in ('qwen_prm', 'qwen_prm_72b'):
    S = dict(np.load(SOC_DIR / f'SCORES_{model}.npz'))
    ne = S['nonempty'].astype(bool); lens = S['nonempty_lengths']; st = np.r_[0, np.cumsum(lens)]
    offn = np.r_[0, np.cumsum(np.add.reduceat(ne.astype(int), st[:-1]))]
    for arm in ('CHAIN_N16', 'TELE_N16'):
        SOC[f'{model}__{arm}'] = dict(z=answer_z(S[f'score__{arm}'], offn), raw=S[f'score__{arm}'], off=offn)

all_n = np.r_[np.diff(off), np.concatenate([np.diff(v['off']) for v in SOC.values()])]
null = load_null(all_n)
tau = {'prmbench': pooled_null_quantile(np.diff(off), null),
       'socratic': pooled_null_quantile(np.diff(next(iter(SOC.values()))['off']), null)}
print('tau_null', tau)


def per_arm(key, z, off_, q80_flags):
    Q = answer_quantities(z, off_, null)
    q80c = np.add.reduceat(q80_flags.astype(int), off_[:-1])
    T = tests(Q)
    T['DW'] = (Q['n'] >= 3) & (np.nan_to_num(Q['G1'], nan=-1) >= tau['prmbench' if key.startswith('PB') else 'socratic'])
    for q, v in Q.items():
        out[f'{key}__Q__{q}'] = v
    out[f'{key}__Q80count'] = q80c
    for t, m in T.items():
        out[f'{key}__pass__{t}'] = m
        for fb in ('F0', 'FQ', 'F1'):
            out[f'{key}__count__{t}-{fb}'] = policy_counts(Q, m, fb, q80c)
    out[f'{key}__count__GAP'] = Q['kstar']
    out[f'{key}__count__K2'] = np.full(len(Q['n']), 2)
    return Q


for arm, z in PB.items():
    per_arm(f'PB__{arm}', z, off, z > np.quantile(z, .8))
    out[f'PB__{arm}__z'] = z
for key, d in SOC.items():
    z = d['z']
    az = np.zeros_like(d['raw'])
    for a, b in zip(d['off'][:-1], d['off'][1:]):
        v = d['raw'][a:b]; az[a:b] = (v - v.mean()) / max(v.std(), 1e-8)
    per_arm(f'SOC__{key}', z, d['off'], az >= np.quantile(az, .8))   # az_q80 of eval_socratic_ablation5.py
    out[f'SOC__{key}__z'] = z; out[f'SOC__{key}__off'] = d['off']
out['PB__off'] = off; out['PB__ids'] = ids
out['tau_null_prmbench'] = tau['prmbench']; out['tau_null_socratic'] = tau['socratic']
np.savez_compressed(HERE / 'STAGE1.npz', **out)
# label-free summary: pass shares (all answers)
for k in sorted(k for k in out if '__pass__' in k):
    print(f'{k:45s} pass share (all answers) = {out[k].mean():.3f}')
