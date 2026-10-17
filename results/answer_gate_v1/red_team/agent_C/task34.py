"""Task 2 extra (isolate count vs gate), Task 3 PRMBench confusion, Task 4 math checks."""
from load import *
import pickle as pk
from scipy.stats import spearmanr, rankdata
from spectral_utils.prmbench import prmbench_evaluate

pd.set_option('display.width', 250)
zA = pk.load(open('zA_recon.pkl', 'rb'))['zA']
Pb = np.flatnonzero(pb); err = target >= 0
cellix = {c: np.flatnonzero(cells == c) for c in PBc}


def nflag(fl): return np.bincount(aid, weights=fl, minlength=n)


def first_flag(fl):
    nf = nflag(fl); pos = np.arange(S_) - off[aid]; big = np.where(fl, pos, 10**9)
    ff = np.minimum.reduceat(big, off[:-1]); return np.where(nf > 0, ff, -1).astype(int), nf


def pbF1(fl):
    ff, nf = first_flag(fl); f = []; aes = []; acs = []
    for c, ix in cellix.items():
        e = ix[err[ix]]; o = ix[~err[ix]]; ae = (ff[e] == target[e]).mean(); ac = (ff[o] == -1).mean()
        f.append(0.0 if ae == 0 and ac == 0 else 2 * ae * ac / (ae + ac)); aes.append(ae); acs.append(ac)
    return np.mean(f), np.mean(aes), np.mean(acs)


R0 = DEC['R0_frozen']
# ---- isolate: R0 flags, but with OFFSET_D1's exact zero-flag answer set
for r in ('OFFSET_D1_upcr_full', 'OFFSET_D2_lsml_cont_good5'):
    un = nflag(DEC[r]) == 0; g = R0.copy(); g[un[aid]] = False
    print(f'R0 steps inside {r} zero-flag set removed: PB F1/acc_err/acc_cor =', np.round(pbF1(g), 4), ' vs', r, np.round(pbF1(DEC[r]), 4))
    # and: OFFSET's count in flagged answers but first-flag from R0 when both flag
    ffo, nfo = first_flag(DEC[r]); ff0, nf0 = first_flag(R0)
    both = pb & (nfo > 0) & (nf0 > 0) & err
    print(f'   among PB erroneous answers flagged by both: n={both.sum()}  hit R0 {np.mean(ff0[both]==target[both]):.4f}  hit {r} {np.mean(ffo[both]==target[both]):.4f}  '
          f'mean flags R0 {nf0[both].mean():.2f} vs {nflag(DEC[r])[both].mean():.2f}; first flag earlier than R0 in {np.mean(ffo[both]<ff0[both]):.3f}, later {np.mean(ffo[both]>ff0[both]):.3f}')

# ================= Task 3: PRMBench confusion counts (positive = correct step kept valid)
good = ~labels
P = np.flatnonzero(prm)


def conf(fl, mask_ans):
    v = ~fl; m = mask_ans[aid]
    TP = int((v & good & m).sum()); FP = int((v & ~good & m).sum()); TN = int((~v & ~good & m).sum()); FN = int((~v & good & m).sum())
    f1 = 2 * TP / (2 * TP + FP + FN); f1e = 2 * TN / (2 * TN + FN + FP)
    return {'TP(correct kept)': TP, 'FP(error kept)': FP, 'TN(error flagged)': TN, 'FN(correct flagged)': FN, 'F1_correct': f1, 'F1_error': f1e,
            'PRMScore': (f1 + f1e) / 2, 'flag_rate': (TN + FN) / (TP + FP + TN + FN), 'zero_flag_answers': float((nflag(fl)[mask_ans] == 0).mean()),
            'err_answers_zero_flag': float((nflag(fl)[mask_ans & has_err] == 0).mean()),
            'err_answers_with_error_flagged': float((np.bincount(aid, weights=(fl & labels), minlength=n)[mask_ans & has_err] > 0).mean())}


def official(fl):
    v = ~fl; res = prmbench_evaluate([{'idx': ids[i], 'labels': v[off[i]:off[i + 1]].astype(int).tolist()} for i in P], [meta[ids[i]] for i in P])
    return 0.5 * (res['total']['f1'] + res['total']['negative_f1'])


# PRMBench gates: calibrated D1 gate on R0 (label-free) and random gate matched per fold
def matched_gate_prm(score, ref, base=R0):
    nfr = nflag(DEC[ref]); nfb = nflag(base); un = np.zeros(n, bool)
    for k in range(5):
        ix = np.flatnonzero(prm & (fold == k)); need = int((nfr[ix] == 0).sum()) - int((nfb[ix] == 0).sum())
        if need <= 0: continue
        cand = ix[nfb[ix] > 0]; un[cand[np.argsort(score[cand], kind='stable')][:need]] = True
    g = base.copy(); g[un[aid]] = False; return g


prm_rules = {r: DEC[r] for r in ['R0_frozen', 'R2_allocate', 'OFFSET_D1_upcr_full', 'OFFSET_D2_lsml_cont_good5', 'OFFSET_D4_equal_full', 'OFFSET_D5_epr', 'OFFSET_D6_length']}
prm_rules['GATE_D1_matched_on_R0'] = matched_gate_prm(DEC['A_D1_upcr_full'].astype(float), 'OFFSET_D1_upcr_full')
un = nflag(DEC['OFFSET_D1_upcr_full']) == 0; g = R0.copy(); g[un[aid]] = False; prm_rules['R0_minus_OFFSET_D1_zeroflag_set'] = g
rng = np.random.default_rng(11)
rnd = [conf(matched_gate_prm(rng.random(n), 'OFFSET_D1_upcr_full'), noncontrol) for _ in range(50)]
rows = []
for r, fl in prm_rules.items():
    c = conf(fl, noncontrol); rows.append({'rule': r, **c})
rows.append({'rule': 'RANDOM_GATE_D1share_on_R0 (mean 50)', **pd.DataFrame(rnd).mean().to_dict()})
C = pd.DataFrame(rows)
print('\nPRMBench non-control (6211 answers) confusion counts; positive = correct step kept')
print(C.round(4).to_string(index=False))
for r in ('R0_frozen', 'OFFSET_D1_upcr_full', 'OFFSET_D2_lsml_cont_good5', 'R2_allocate'):
    print(r, 'official PRMScore', official(prm_rules[r]))
base = C.set_index('rule').loc['R0_frozen']
print('\nChanges vs R0:')
for r in C.rule:
    d = C.set_index('rule').loc[r] - base
    print(f'  {r:40s} dTP {d["TP(correct kept)"]:+7.0f} dFP {d["FP(error kept)"]:+7.0f} dTN {d["TN(error flagged)"]:+7.0f} dFN {d["FN(correct flagged)"]:+7.0f}  dPRMScore {d["PRMScore"]:+.4f}  dF1c {d["F1_correct"]:+.4f} dF1e {d["F1_error"]:+.4f}')

# where do OFFSET flags move on PRMBench? per answer, error-step density vs zA
dens = np.bincount(aid, weights=labels, minlength=n) / ns
A1 = DEC['A_D1_upcr_full'].astype(float)
m = err_nc
print('\nPRMBench erroneous non-control: Spearman(zA_D1, error-step density) =', round(spearmanr(A1[m], dens[m])[0], 3),
      ' Spearman(zA_D1, n steps) =', round(spearmanr(A1[m], ns[m])[0], 3), ' Spearman(zA_D1, n error steps)=', round(spearmanr(A1[m], np.bincount(aid, weights=labels, minlength=n)[m])[0], 3))
q = pd.qcut(A1[noncontrol], 5, labels=False)
nf0 = nflag(R0); nf1 = nflag(DEC['OFFSET_D1_upcr_full']); nerr = np.bincount(aid, weights=labels, minlength=n)
hit0 = np.bincount(aid, weights=R0 & labels, minlength=n); hit1 = np.bincount(aid, weights=DEC['OFFSET_D1_upcr_full'] & labels, minlength=n)
nc = np.flatnonzero(noncontrol)
T = pd.DataFrame({'zA_quintile': q, 'err': has_err[nc], 'steps': ns[nc], 'err_density': dens[nc], 'flags_R0': nf0[nc], 'flags_OFF': nf1[nc],
                  'errflag_R0': hit0[nc], 'errflag_OFF': hit1[nc], 'zero_R0': nf0[nc] == 0, 'zero_OFF': nf1[nc] == 0})
print(T.groupby('zA_quintile').agg(['mean']).round(3).to_string())
print('sum over quintiles of error flags (TN):', T.groupby('zA_quintile')[['errflag_R0', 'errflag_OFF', 'flags_R0', 'flags_OFF']].sum().to_string())
print('AUROC zA_D1 err vs multi_solutions (non-control):', round(((rankdata(A1[noncontrol])[has_err[noncontrol]]).sum() - err_nc.sum() * (err_nc.sum() + 1) / 2) / (err_nc.sum() * ms.sum()), 4))

# ================= Task 4 checks
print('\nTASK 4')
zS = np.empty(S_)
for i in range(n):
    s = S[off[i]:off[i + 1]]; zS[off[i]:off[i + 1]] = (s - s.mean()) / max(s.std(), 1e-8)
off_rules = [k for k in DEC if k.startswith('OFFSET') or k in ('R0_frozen', 'R2_allocate', 'R2pb_allocate')]
for r in off_rules:
    fl = DEC[r]; viol = 0; viol_ans = 0; tie_viol = 0
    Smin_fl = np.where(fl, S, np.inf); Smax_un = np.where(~fl, S, -np.inf)
    mn = np.minimum.reduceat(Smin_fl, off[:-1]); mx = np.maximum.reduceat(Smax_un, off[:-1])
    strict = mx > mn; ties = mx == mn
    print(f'  {r:32s} answers violating upper set (unflagged S > flagged S): {int(strict.sum())}; exact-tie boundary answers: {int(ties.sum())}')
# recompute OFFSET flags exactly and compare
for d in ('D1_upcr_full', 'D2_lsml_cont_good5', 'D4_equal_full', 'D5_epr', 'D6_length'):
    f = np.zeros(S_, bool)
    for k in range(5):
        c = (k + 1) % 5; Dv = zA[d][k][aid] + zS
        for bm in (prm_step, pb_step):
            calm = bm & (step_fold == c); evm = bm & (step_fold == k); f[evm] = Dv[evm] >= np.quantile(Dv[calm], .8)
    print(f'  recomputed OFFSET_{d} equals saved: {np.array_equal(f, DEC["OFFSET_" + d])}')
print('\n  flag rate per benchmark x fold (evaluation steps):')
fr = []
for r in ['R0_frozen', 'R2_allocate', 'R2pb_allocate', 'OFFSET_D1_upcr_full', 'OFFSET_D2_lsml_cont_good5', 'OFFSET_D4_equal_full', 'OFFSET_D5_epr']:
    row = {'rule': r}
    for bn, bm in (('prm', prm_step), ('prm_noncontrol', noncontrol[aid]), ('pb', pb_step)):
        row[bn + '_all'] = DEC[r][bm].mean()
        for k in range(5): row[f'{bn}_f{k}'] = DEC[r][bm & (step_fold == k)].mean()
    fr.append(row)
print(pd.DataFrame(fr).round(4).to_string(index=False))
# PB per cell flag rates
print(pd.DataFrame({r: {c: DEC[r][cells[aid] == c].mean() for c in PBc} for r in ['R0_frozen', 'OFFSET_D1_upcr_full', 'OFFSET_D2_lsml_cont_good5']}).round(4).to_string())
# polarity: correlation of A_D1 with epr per cell (evaluation answers)
e = X[:, names.index('epr')]
pc = {}
for c in sorted(set(cells)):
    m = cells == c
    pc[c] = {d: spearmanr(DEC['A_' + d][m], e[m])[0] for d in ('D1_upcr_full', 'D2_lsml_cont_good5', 'D3_lsml_full', 'D4_equal_full')}
    pc[c].update({f'D1_fold{k}': spearmanr(DEC['A_D1_upcr_full'][m & (fold == k)], e[m & (fold == k)])[0] for k in range(5)})
print('\nSpearman(A_d, epr) on evaluation answers per cell'); print(pd.DataFrame(pc).T.round(3).to_string())
# single-step and constant-S answers
const = np.array([S[off[i]:off[i + 1]].std() < 1e-12 for i in range(n)])
print('\nanswers with 1 step:', int((ns == 1).sum()), ' constant-S answers:', int(const.sum()), ' PB among them:', int((const & pb).sum()))
