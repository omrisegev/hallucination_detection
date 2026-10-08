"""Independent check of CONTRASTS.csv: point deltas, aggregation definition, bootstrap."""
import pickle
from pathlib import Path
import numpy as np, pandas as pd

OUT = Path(r'C:\Users\omris\TAU\hallucination_detection\results\monday_fusion_competitor_extension_v1\run_20261006')
SCR = Path(__file__).parent
R = pickle.load(open(SCR / 'records.pkl', 'rb')); rec = R['rec']
main = pd.read_csv(SCR / 'main_table_mine.csv').set_index(['method', 'decision_rule'])
BANKS = ['B13', 'B16', 'B20', 'B23', 'B32', 'B35', 'B51', 'B54']
RULES = ['ANSWER_Z_Q80', 'R2_COUNT', 'ENTROPY_GATE_Q80']
METHODS = ['HEM_MOM_cc', 'MEAN_MOM_cc', 'MEAN_MOM_ga', 'TCI_ONLY_ga']
PRM = 'prmbench_qwen3_8b'
target, groups, cells_all = R['target'], R['groups'], R['cells']
PBC = sorted(c for c in set(cells_all) if c != PRM)
C = pd.read_csv(OUT / 'CONTRASTS.csv')

prm_ans = np.flatnonzero(cells_all == PRM); pb_ans = np.flatnonzero(cells_all != PRM)
pg, pinv = np.unique(groups[prm_ans], return_inverse=True); bg, binv = np.unique(groups[pb_ans], return_inverse=True)
print('source groups: PRMB', len(pg), 'answers', len(prm_ans), '| PB', len(bg), 'answers', len(pb_ans),
      '| groups shared PRMB/PB', len(set(pg) & set(bg)))


def ps(tp, fp, tn, fn):
    a = np.divide(2 * tp, 2 * tp + fp + fn, out=np.full(np.shape(tp), -1.), where=(2 * tp + fp + fn) != 0)
    b = np.divide(2 * tn, 2 * tn + fp + fn, out=np.full(np.shape(tp), -1.), where=(2 * tn + fp + fn) != 0)
    return .5 * (a + b)


def per_answer_pb(bk, m, a):
    """Arrays aligned to pb_ans order."""
    hit = np.zeros(len(pb_ans), bool); flag = hit.copy(); ah = hit.copy(); cellidx = np.zeros(len(pb_ans), int)
    pos = {i: k for k, i in enumerate(pb_ans)}
    for ci, cell in enumerate(PBC):
        d = rec[(bk, cell, m, a)]; k = np.array([pos[i] for i in d['ans']]); t = target[d['ans']]
        hit[k] = d['first'] == t; flag[k] = d['nflag'] > 0; ah[k] = d['argmax'] == t; cellidx[k] = ci
    return hit, flag, ah, target[pb_ans] >= 0, cellidx


def pooled_point(m, a):
    f1s, aes, ams = [], [], []
    for bk in BANKS:
        h, f, ah, e, _ = per_answer_pb(bk, m, a)
        ae = h[e].mean(); ac = h[~e].mean(); f1s.append(2 * ae * ac / (ae + ac)); aes.append(ae); ams.append(ah[e].mean())
    return {'pb_official_f1': np.mean(f1s), 'pb_decision_error_accuracy': np.mean(aes), 'pb_argmax_first_error_accuracy': np.mean(ams)}


# ---- point deltas: CONTRASTS vs mine (pooled definition) vs main table (cell-equal)
rows = []
for a in RULES:
    pp = {m: pooled_point(m, a) for m in METHODS}
    for m in METHODS[:-1]:
        for metric in ['within_answer_auc', 'prmscore', 'pb_official_f1', 'pb_decision_error_accuracy', 'pb_argmax_first_error_accuracy']:
            c = C[(C.method_a == m) & (C.method_b == 'TCI_ONLY_ga') & (C.rule == a) & (C.metric == metric)].iloc[0]
            col = {'pb_argmax_first_error_accuracy': 'pb_first_error_accuracy'}.get(metric, metric)
            cell_eq = main.loc[('TCI_ONLY_ga', a), col] - main.loc[(m, a), col]
            pooled = (pp['TCI_ONLY_ga'][metric] - pp[m][metric]) if metric.startswith('pb') else cell_eq
            rows.append({'rule': a, 'vs': m, 'metric': metric, 'codex_TCI_minus': -c.delta_a_minus_b, 'mine_pooled': pooled,
                         'main_table_cell_equal': cell_eq, 'codex_lo': -c.hi95, 'codex_hi': -c.lo95})
D = pd.DataFrame(rows); D['pooled_vs_codex'] = (D.codex_TCI_minus - D.mine_pooled).abs()
D['cell_eq_minus_pooled'] = D.main_table_cell_equal - D.mine_pooled
pd.set_option('display.width', 250); print(D.round(4).to_string())
print('max |codex - mine pooled| =', D.pooled_vs_codex.max())

# ---- bootstrap replication: same seed/order as protocol; my own aggregation code
rng = np.random.default_rng(20261006); draws = 5000
Wp = rng.multinomial(len(pg), np.full(len(pg), 1 / len(pg)), size=draws).astype(float)
Wb = rng.multinomial(len(bg), np.full(len(bg), 1 / len(bg)), size=draws).astype(float)


def gsum(x, inv, n):
    out = np.zeros((n,) + x.shape[1:]); np.add.at(out, inv, x); return out


def boot(m, a, cell_equal=False):
    auc_b, ps_b, f1_b, ae_b, am_b = [], [], [], [], []
    for bk in BANKS:
        d = rec[(bk, PRM, m, a)]; assert np.array_equal(d['ans'], prm_ans)
        au = d['auc']; s = Wp @ gsum(np.nan_to_num(au), pinv, len(pg)); n = Wp @ gsum(np.isfinite(au).astype(float), pinv, len(pg))
        auc_b.append(s / n)
        c = Wp @ gsum(d['conf'].astype(float), pinv, len(pg)); ps_b.append(ps(*c.T))
        h, f, ah, e, ci = per_answer_pb(bk, m, a)
        if not cell_equal:
            z = Wb @ gsum(np.column_stack([h & e, e, h & ~e, ~e, ah & e]).astype(float), binv, len(bg))
            ae = z[:, 0] / z[:, 1]; ac = z[:, 2] / z[:, 3]; f1_b.append(2 * ae * ac / (ae + ac)); ae_b.append(ae); am_b.append(z[:, 4] / z[:, 1])
        else:
            f1c, aec, amc = [], [], []
            for k in range(len(PBC)):
                sel = (ci == k).astype(float)
                z = Wb @ gsum(np.column_stack([(h & e) * sel, e * sel, (h & ~e) * sel, (~e) * sel, (ah & e) * sel]), binv, len(bg))
                ae = z[:, 0] / z[:, 1]; ac = z[:, 2] / z[:, 3]
                f1c.append(np.where(ae + ac > 0, 2 * ae * ac / np.where(ae + ac > 0, ae + ac, 1), 0)); aec.append(ae); amc.append(z[:, 4] / z[:, 1])
            f1_b.append(np.nanmean(f1c, 0)); ae_b.append(np.nanmean(aec, 0)); am_b.append(np.nanmean(amc, 0))
    return {'within_answer_auc': np.mean(auc_b, 0), 'prmscore': np.mean(ps_b, 0), 'pb_official_f1': np.nanmean(f1_b, 0),
            'pb_decision_error_accuracy': np.nanmean(ae_b, 0), 'pb_argmax_first_error_accuracy': np.nanmean(am_b, 0)}


out = []
for a in RULES:
    B = {m: boot(m, a) for m in METHODS}
    Bc = {m: boot(m, a, cell_equal=True) for m in METHODS}
    for m in METHODS[:-1]:
        for metric in B[m]:
            dd = B['TCI_ONLY_ga'][metric] - B[m][metric]; lo, hi = np.nanquantile(dd, [.025, .975])
            c = C[(C.method_a == m) & (C.method_b == 'TCI_ONLY_ga') & (C.rule == a) & (C.metric == metric)].iloc[0]
            dc = Bc['TCI_ONLY_ga'][metric] - Bc[m][metric]; loc_, hic = np.nanquantile(dc, [.025, .975])
            out.append({'rule': a, 'vs': m, 'metric': metric, 'my_lo': lo, 'my_hi': hi, 'codex_lo': -c.hi95, 'codex_hi': -c.lo95,
                        'cellEq_lo': loc_, 'cellEq_hi': hic})
O = pd.DataFrame(out); O['ci_diff'] = np.maximum((O.my_lo - O.codex_lo).abs(), (O.my_hi - O.codex_hi).abs())
print(O.round(4).to_string()); print('max CI endpoint diff vs codex', O.ci_diff.max())
# MTG contrasts present in CONTRASTS.csv
print(C[C.method_b == 'MTG_NATIVE_ARGMAX'].round(4).to_string())
D.to_csv(SCR / 'contrast_points.csv', index=False); O.to_csv(SCR / 'contrast_boot.csv', index=False)
