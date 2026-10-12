"""How much of the true-count PRMScore gain is answer-specific information vs answer length? Fixed IU-PCR order (N16,
corrected prefix, results/thinkfix_rescore_v1/prmscore_inputs). Count rules: Q80 (answer-z, global q.8), true count (oracle),
best count from length only (label-fitted oracle), best count from error type only (oracle), proportional c*n (c fitted).
Labels for evaluation/oracles only. PRMScore pooled over noncontrols (classification != correct)."""
import importlib.util, io, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd

MAIN = Path(r'C:\Users\omris\TAU\hallucination_detection'); sys.path.insert(0, str(MAIN))
import spectral_utils.fusion_utils  # noqa
spec = importlib.util.spec_from_file_location('dr', MAIN / 'results/monday_nested_banks_v2/dev_run.py'); dr = importlib.util.module_from_spec(spec); spec.loader.exec_module(dr)

z = np.load(MAIN / 'results/thinkfix_rescore_v1/prmscore_inputs/bank_N16.npz', allow_pickle=True)
OFF = z['offsets']; IDX = z['idx'].astype(str)
A = pd.read_csv(dr.SE / 'OOF_ANSWERS.csv', encoding='utf-8-sig')
with np.load(dr.SE / 'OOF_STEP_SCORES.npz') as q: O2 = q['offsets'].astype(np.int64); L2 = q['labels'].astype(bool)
pos = {i: k for k, i in enumerate(A.id.astype(str))}
csv = subprocess.run(['git', '-C', str(MAIN), 'show', '07cb7a45f:results/advisor_completion_2026-10-11/controls/PER_ANSWER_COUNTS.csv'], capture_output=True, text=True).stdout
PC = pd.read_csv(io.StringIO(csv)); PC = PC[(PC.method == 'IU_PCR') & (PC.rule == 'ANSWER_Z_Q80')].set_index('idx')
N = len(IDX); n = np.diff(OFF); lab = np.zeros(OFF[-1], bool)
for a, i in enumerate(IDX):
    k = pos[i]; y = L2[O2[k]:O2[k + 1]]
    if len(y) != n[a]: raise SystemExit(f'len mismatch {i}')
    lab[OFF[a]:OFF[a + 1]] = y
cls = PC.loc[IDX, 'classification'].to_numpy(); tc = np.add.reduceat(lab.astype(int), OFF[:-1])
if not np.array_equal(tc, PC.loc[IDX, 'true_count'].to_numpy()): print('WARN true_count differs from v11 CSV in', int((tc != PC.loc[IDX, 'true_count'].to_numpy()).sum()), 'answers')
nonc = cls != 'correct'


def run(score_key):
    s = z[score_key].astype(float)
    order = [np.argsort(-s[OFF[a]:OFF[a + 1]], kind='stable') for a in range(N)]
    def prmscore(k):
        fl = np.zeros(OFF[-1], bool)
        for a in range(N):
            kk = int(np.clip(k[a], 0, n[a])); fl[OFF[a] + order[a][:kk]] = True
        m = np.repeat(nonc, n); y, f = lab[m], fl[m]
        tp = (y & f).sum(); fp = (~y & f).sum(); fn = (y & ~f).sum(); tn = (~y & ~f).sum()
        f1e = 2 * tp / (2 * tp + fp + fn); f1v = 2 * tn / (2 * tn + fn + fp)
        mae = np.abs(np.clip(k, 0, n) - tc)[nonc].mean()
        return 100 * (f1e + f1v) / 2, mae
    zs = np.concatenate([(s[OFF[a]:OFF[a + 1]] - s[OFF[a]:OFF[a + 1]].mean()) / (s[OFF[a]:OFF[a + 1]].std() or 1.) for a in range(N)])
    thr = np.quantile(zs, .8); k_q80 = np.add.reduceat((zs > thr).astype(int), OFF[:-1])
    rows = {'Q80 (deployed)': k_q80, 'True count per answer (oracle)': tc}
    # length-only oracle: mean true count per exact n_steps among noncontrols, rounded
    dfn = pd.DataFrame({'n': n, 'tc': tc, 'cls': cls})[nonc]
    mlen = dfn.groupby('n').tc.mean(); rows['Best count from length only (oracle)'] = np.floor(pd.Series(n).map(mlen).fillna(dfn.tc.mean()).to_numpy() + .5)
    best = max(((prmscore(np.floor(c * n + .5))[0], c) for c in np.arange(.05, .40, .005)))
    rows[f'Proportional c*n, best c={best[1]:.3f} (oracle)'] = np.floor(best[1] * n + .5)
    mtyp = dfn.groupby('cls').tc.mean(); rows['Best count from error type only (oracle)'] = np.floor(pd.Series(cls).map(mtyp).fillna(0).to_numpy() + .5)
    mlt = dfn.groupby(['cls', 'n']).tc.mean()
    rows['Best count from type + length (oracle)'] = np.floor(np.array([mlt.get((c, m), mtyp.get(c, 0)) for c, m in zip(cls, n)]) + .5)
    out = []
    for name, k in rows.items():
        p, mae = prmscore(np.asarray(k, float)); out.append((name, round(p, 3), round(mae, 3), round(float(np.clip(k, 0, n)[nonc].sum() / n[nonc].sum() * 100), 2)))
    return out


from scipy.stats import spearmanr
r_len = spearmanr(n[nonc], tc[nonc])[0]
print(f'noncontrols {nonc.sum()} answers; true-count mean {tc[nonc].mean():.2f}, Spearman(true count, n_steps) {r_len:.3f}')
print('true count distribution (noncontrols):', dict(zip(*np.unique(tc[nonc], return_counts=True))))
for key in ('S__IU_PCR', 'S__LSML_CONT'):
    print('\n==', key, ' [rule | PRMScore x100 | count MAE | % steps flagged]')
    for r in run(key): print('  ', r)

print('\n== constant k per answer (label-free rule; TOP_k) [PRMScore | MAE | % flagged]')
for key in ('S__IU_PCR', 'S__LSML_CONT'):
    s = z[key].astype(float); order = [np.argsort(-s[OFF[a]:OFF[a + 1]], kind='stable') for a in range(N)]
    for kc in (1, 2, 3):
        fl = np.zeros(OFF[-1], bool)
        for a in range(N): fl[OFF[a] + order[a][:min(kc, n[a])]] = True
        m = np.repeat(nonc, n); y, f = lab[m], fl[m]
        tp = (y & f).sum(); fp = (~y & f).sum(); fn = (y & ~f).sum(); tn = (~y & ~f).sum()
        p = 100 * (2 * tp / (2 * tp + fp + fn) + 2 * tn / (2 * tn + fn + fp)) / 2
        print(f'   {key} k={kc}: {p:.3f} | {np.abs(np.minimum(kc, n) - tc)[nonc].mean():.3f} | {f.mean() * 100:.2f}')
