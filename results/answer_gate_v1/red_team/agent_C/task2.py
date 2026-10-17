"""Task 2: PB F1 decomposition, matched gates, random gate; Task 3: PRMBench confusion counts; Task 4 checks."""
from load import *
import pickle as pk
from spectral_utils.prmbench import prmbench_evaluate

zA = pk.load(open('zA_recon.pkl', 'rb'))['zA']
pd.set_option('display.width', 250)
Pb = np.flatnonzero(pb); err = target >= 0
cellix = {c: np.flatnonzero(cells == c) for c in PBc}


def first_flag(fl):
    nf = np.bincount(aid, weights=fl, minlength=n)
    # first flagged step index per answer
    pos = np.arange(S_) - off[aid]
    big = np.where(fl, pos, 10**9)
    ff = np.minimum.reduceat(big, off[:-1]) if True else None
    ff = np.where(nf > 0, ff, -1)
    return ff.astype(int), nf


def pb_panel(fl, name):
    ff, nf = first_flag(fl)
    rows = []
    for c, ix in cellix.items():
        e = ix[err[ix]]; o = ix[~err[ix]]
        ae = (ff[e] == target[e]).mean(); ac = (ff[o] == -1).mean()
        fe = (nf[e] > 0).mean(); cond = (ff[e] == target[e])[nf[e] > 0].mean() if (nf[e] > 0).any() else np.nan
        f1 = 0.0 if ae == 0 and ac == 0 else 2 * ae * ac / (ae + ac)
        rows.append({'cell': c, 'F1': f1, 'acc_err': ae, 'acc_correct(=unflagged share of correct)': ac, 'erroneous_flagged_share': fe,
                     'first_flag_hits_error|flagged': cond, 'unflagged_share_all': (nf[ix] == 0).mean(),
                     'early(first<target)|err': (ff[e][(nf[e] > 0)] < target[e][nf[e] > 0]).mean(), 'late|err': (ff[e][(nf[e] > 0)] > target[e][nf[e] > 0]).mean()})
    T = pd.DataFrame(rows); m = T.drop(columns='cell').mean()
    return {'rule': name, **m.to_dict()}, T


def pbf1(fl):
    return pb_panel(fl, '')[0]['F1']


R0 = DEC['R0_frozen']; R2pb = DEC['R2pb_allocate']
ff0, nf0 = first_flag(R0)
rules = ['R0_frozen', 'R2pb_allocate', 'R2_allocate', 'OFFSET_D1_upcr_full', 'OFFSET_D2_lsml_cont_good5', 'OFFSET_D3_lsml_full', 'OFFSET_D4_equal_full',
         'OFFSET_D5_epr', 'OFFSET_D6_length', 'OFFSET_D6b_length_anchored', 'OFFSETw0.5_D1_upcr_full', 'OFFSETw2.0_D1_upcr_full']
summ = [pb_panel(DEC[r], r)[0] for r in rules]


def gated(unflag_answers, base=R0):
    fl = base.copy(); m = unflag_answers[aid]; fl[m] = False; return fl


# ---------- matched gates
def matched_gate(score, ref_rule, level='pbfold', base=R0):
    """unflag lowest-score answers (on top of base-unflagged) until total unflagged count matches ref_rule, per PB fold or per cell x fold (evaluation-matched)."""
    _, nfr = first_flag(DEC[ref_rule]); _, nfb = first_flag(base)
    un = np.zeros(n, bool)
    units = [(pb & (fold == k)) for k in range(5)] if level == 'pbfold' else [(cells == c) & (fold == k) for c in PBc for k in range(5)]
    for u in units:
        ix = np.flatnonzero(u); target_cnt = int((nfr[ix] == 0).sum()); already = nfb[ix] == 0
        need = target_cnt - int(already.sum())
        if need <= 0: continue
        cand = ix[~already]; order = cand[np.argsort(score[cand], kind='stable')]; un[order[:need]] = True
    return gated(un, base)


def random_gate(ref_rule, rng, level='cellfold', base=R0):
    return matched_gate(rng.random(n), ref_rule, level, base)


def calibrated_gate(d, ref_det, base=R0, bench='pb'):
    """label-free, deployable: on calibration fold c with the fold-k model, find the OFFSET unflagged share, choose the zA threshold that
    reproduces it (gate U R0-unflagged), apply that threshold to evaluation answers. Per benchmark, like OFFSET's calibration."""
    bm_ans = pb if bench == 'pb' else prm
    bstep = bm_ans[aid]; _, nfb = first_flag(base)
    zS = np.empty(S_)
    for i in range(n):
        s = S[off[i]:off[i + 1]]; zS[off[i]:off[i + 1]] = (s - s.mean()) / max(s.std(), 1e-8)
    un = np.zeros(n, bool)
    for k in range(5):
        c = (k + 1) % 5; zr = zA[ref_det][k]; Dv = zr[aid] + zS
        calm = bstep & (step_fold == c); tau = np.quantile(Dv[calm], .8)
        fl_c = np.zeros(S_, bool); fl_c[calm] = Dv[calm] >= tau
        cal_ans = np.flatnonzero(bm_ans & (fold == c)); nf_c = np.bincount(aid, weights=fl_c, minlength=n)
        target_share = (nf_c[cal_ans] == 0).mean()
        # threshold t on zA[d][k] so that share of (zA<t or R0-unflagged) among calibration answers = target_share
        sc = zA[d][k][cal_ans]; base_un = nfb[cal_ans] == 0
        cands = np.sort(sc); best = None
        lo, hi = 0, len(cands)
        for t in cands:
            sh = ((sc < t) | base_un).mean()
            if sh >= target_share: best = t; break
        ev = np.flatnonzero(bm_ans & (fold == k)); un[ev] = zA[d][k][ev] < best
    return gated(un, base)


extra = {}
extra['GATE_D1_matched_pbfold'] = matched_gate(DEC['A_D1_upcr_full'].astype(float), 'OFFSET_D1_upcr_full', 'pbfold')
extra['GATE_D1_matched_cellfold'] = matched_gate(DEC['A_D1_upcr_full'].astype(float), 'OFFSET_D1_upcr_full', 'cellfold')
extra['GATE_D1_calibrated'] = calibrated_gate('D1_upcr_full', 'D1_upcr_full')
extra['GATE_D2_matched_pbfold'] = matched_gate(DEC['A_D2_lsml_cont_good5'].astype(float), 'OFFSET_D2_lsml_cont_good5', 'pbfold')
extra['GATE_D5epr_matched_D1share_pbfold'] = matched_gate(DEC['A_D5_epr'].astype(float), 'OFFSET_D1_upcr_full', 'pbfold')
extra['GATE_length_matched_D1share_pbfold'] = matched_gate(X[:, names.index('trace_length')], 'OFFSET_D1_upcr_full', 'pbfold')
extra['GATE_stepcount_matched_D1share_pbfold'] = matched_gate(ns + 1e-3 * np.random.default_rng(1).random(n), 'OFFSET_D1_upcr_full', 'pbfold')
extra['GATE_D1_matched_pbfold_on_R2pb'] = matched_gate(DEC['A_D1_upcr_full'].astype(float), 'OFFSET_D1_upcr_full', 'pbfold', base=R2pb)
for k_, v in extra.items(): summ.append(pb_panel(v, k_)[0])
rng = np.random.default_rng(20260930)
for lvl in ('cellfold', 'pbfold'):
    rs = [pb_panel(random_gate('OFFSET_D1_upcr_full', rng, lvl), f'RANDOM_GATE_{lvl}')[0] for _ in range(50)]
    df = pd.DataFrame(rs).drop(columns='rule'); r = {'rule': f'RANDOM_GATE_D1share_{lvl} (mean of 50)', **df.mean().to_dict()}
    r['F1_sd'] = df.F1.std(); r['F1_min'] = df.F1.min(); r['F1_max'] = df.F1.max(); summ.append(r)
rs = [pb_panel(random_gate('OFFSET_D1_upcr_full', rng, 'cellfold', base=R2pb), 'x')[0] for _ in range(50)]
df = pd.DataFrame(rs).drop(columns='rule'); summ.append({'rule': 'RANDOM_GATE_D1share_cellfold_on_R2pb (mean of 50)', **df.mean().to_dict(), 'F1_sd': df.F1.std()})
TT = pd.DataFrame(summ)
cols = ['rule', 'F1', 'acc_err', 'acc_correct(=unflagged share of correct)', 'erroneous_flagged_share', 'first_flag_hits_error|flagged', 'unflagged_share_all', 'early(first<target)|err', 'late|err', 'F1_sd']
print('PROCESSBENCH macro-8 decomposition')
print(TT[cols].round(4).to_string(index=False))
# per-cell for key rules
for r in ['R0_frozen', 'R2pb_allocate', 'OFFSET_D1_upcr_full', 'OFFSET_D2_lsml_cont_good5']:
    print('\n', r); print(pb_panel(DEC[r], r)[1].round(3).to_string(index=False))
for r in ['GATE_D1_calibrated', 'GATE_D1_matched_pbfold']:
    print('\n', r); print(pb_panel(extra[r], r)[1].round(3).to_string(index=False))

# analytic random-gate curve: with R0 flags, unflag a random share u of R0-flagged answers
print('\nRandom-gate curve on R0 (analytic, per-cell expectation, macro): u -> F1')
for u in (0.3, 0.4, 0.5, 0.6, 0.7):
    f = []
    for c, ix in cellix.items():
        e = ix[err[ix]]; o = ix[~err[ix]]
        ae = (1 - u) * (ff0[e] == target[e]).mean(); ac = (nf0[o] == 0).mean() + u * (nf0[o] > 0).mean()
        f.append(2 * ae * ac / (ae + ac))
    print(f'  u={u:.1f}  F1={np.mean(f):.4f}')

pk.dump({'extra': extra}, open('task2_flags.pkl', 'wb'))
