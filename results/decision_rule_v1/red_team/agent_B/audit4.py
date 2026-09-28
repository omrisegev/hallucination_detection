exec(open('load.py').read())
from collections import Counter
from sklearn.metrics import roc_auc_score
RULES = ['R0_frozen', 'R1_global', 'R2_allocate', 'R3_ds_map', 'R4_ds_expected_f1', 'R5_ds_count']
wk = np.load('work.npz'); lab_meta = wk['lab_meta']
pb = ans.cell.str.startswith('pb_').to_numpy(); prm = ~pb; fold = ans.fold.to_numpy(); groups = ans.source_group.to_numpy()
cells = ans.cell.to_numpy(); target = ans.target.to_numpy(); ids = ans.id.to_numpy()
aid = np.repeat(np.arange(n), ns); step_fold = fold[aid]; prm_step = prm[aid]
post = D['posterior'].astype(float)
print('posterior AUC: PRMB error vs correct steps %.4f; PB-vs-PRMB membership (all steps) %.4f' % (
    roc_auc_score(lab_meta[prm_step], post[prm_step]), roc_auc_score(~prm_step, post)))
G = D['G'].astype(float)
print('G AUC: PRMB error vs correct steps %.4f; PB-vs-PRMB membership %.4f' % (roc_auc_score(lab_meta[prm_step], G[prm_step]), roc_auc_score(~prm_step, G)))
print('step flag rates PB / PRMB-all:')
for r in RULES: print(' ', r, '%.3f / %.3f' % (D[r][~prm_step].mean(), D[r][prm_step].mean()))
cellstep = cells[aid]
for c in sorted(set(cells[pb])):
    m = cellstep == c
    print(' ', c, 'mean steps %.1f' % ns[cells == c].mean(), ' '.join('%s %.3f' % (r[:2], D[r][m].mean()) for r in RULES))
# PB groups and folds per cell; are q4 and q8 the same answers?
for c in sorted(set(cells[pb])):
    m = cells == c
    print(c, 'answers', m.sum(), 'groups', len(set(groups[m])), 'folds', sorted(Counter(fold[m]).items()))
q4 = set(ids[cells == 'pb_math_q4']); q8 = set(ids[cells == 'pb_math_q8'])
print('pb_math q4 vs q8 same answer ids:', q4 == q8, len(q4 & q8))
print('PB groups total', len(set(groups[pb])))
# PB first flagged step position relative to answer for correct answers (R1): where are flags?
for r in ['R0_frozen', 'R1_global']:
    firsts = []
    for i in np.flatnonzero(pb & (target >= 0)):
        f = np.flatnonzero(D[r][off[i]:off[i + 1]])
        if len(f): firsts.append((f[0], target[i]))
    a = np.array(firsts)
    print(r, 'erroneous PB: first flag index median %.1f vs true first-error median %.1f; share first-flag==0: %.3f' % (np.median(a[:, 0]), np.median(a[:, 1]), (a[:, 0] == 0).mean()))
