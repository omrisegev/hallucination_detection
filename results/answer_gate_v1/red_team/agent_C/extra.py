from load import *
import pickle as pk, json
err = target >= 0
ex = pk.load(open('task2_flags.pkl', 'rb'))['extra']
cellix = {c: np.flatnonzero(cells == c) for c in PBc}
PBc_ = PBc


def nflag(fl): return np.bincount(aid, weights=fl, minlength=n)


def first_flag(fl):
    nf = nflag(fl); pos = np.arange(S_) - off[aid]; big = np.where(fl, pos, 10**9)
    return np.where(nf > 0, np.minimum.reduceat(big, off[:-1]), -1).astype(int)


Pb = np.flatnonzero(pb); ug, gi = np.unique(groups[Pb], return_inverse=True); cix = np.array([PBc.index(c) for c in cells[Pb]])


def counts(fl):
    ff = first_flag(fl); a = np.zeros((len(ug), 8, 4)); e = err[Pb]; hit = ff[Pb] == target[Pb]
    np.add.at(a, (gi, cix, 0), (hit & e)); np.add.at(a, (gi, cix, 1), e); np.add.at(a, (gi, cix, 2), (hit & ~e)); np.add.at(a, (gi, cix, 3), ~e)
    return a


def f1(a):
    ae = a[..., 0] / a[..., 1]; ac = a[..., 2] / a[..., 3]; f = np.where((ae == 0) & (ac == 0), 0, 2 * ae * ac / np.maximum(ae + ac, 1e-300)); return f.mean(-1)


rules = {'OFFSET_D1': DEC['OFFSET_D1_upcr_full'], 'OFFSET_D2': DEC['OFFSET_D2_lsml_cont_good5'], 'R2pb': DEC['R2pb_allocate'], 'R0': DEC['R0_frozen'],
         'GATE_D1_calibrated': ex['GATE_D1_calibrated'], 'GATE_epr_matched': ex['GATE_D5epr_matched_D1share_pbfold'], 'OFFSET_D5': DEC['OFFSET_D5_epr']}
C = {k: counts(v) for k, v in rules.items()}
rng = np.random.default_rng(9); Wb = rng.multinomial(len(ug), np.full(len(ug), 1 / len(ug)), size=2000).astype(float)
bs = {k: f1(np.einsum('bg,gkc->bkc', Wb, v)) for k, v in C.items()}
for a, b in (('GATE_D1_calibrated', 'OFFSET_D1'), ('OFFSET_D1', 'R2pb'), ('OFFSET_D2', 'R2pb'), ('OFFSET_D5', 'OFFSET_D1'), ('GATE_D1_calibrated', 'R2pb')):
    d = bs[a] - bs[b]; print(f'PB F1 {a} - {b}: {f1(C[a].sum(0)[None])[0]-f1(C[b].sum(0)[None])[0]:+.4f}  CI95 [{np.quantile(d,.025):+.4f}, {np.quantile(d,.975):+.4f}]')

# PRMBench zero-flag composition
for r in ('OFFSET_D1_upcr_full', 'OFFSET_D2_lsml_cont_good5', 'R0_frozen', 'R2_allocate'):
    z = nflag(DEC[r]) == 0
    print(f'{r}: noncontrol zero-flag answers {int((z & noncontrol).sum())} = erroneous {int((z & err_nc).sum())} + multi_solutions {int((z & ms).sum())} + other clean {int((z & noncontrol & ~err_nc & ~ms).sum())};'
          f' error steps inside silenced erroneous answers {int(labels[(z & err_nc)[aid]].sum())}; controls zero-flag {int((z & control).sum())}/{int(control.sum())}')
print('non-control answers without in-range error:', int((noncontrol & ~err_nc).sum()), 'of', int(noncontrol.sum()))

# U-PCR weight on trace_length and length-scaling views: pull from a reconstructed fit is costly; instead correlation of each view with log length per PB cell
tl = np.log(X[:, names.index('trace_length')])
from scipy.stats import spearmanr
cor = pd.DataFrame({c: {nm: spearmanr(X[ix, j], tl[ix])[0] for j, nm in enumerate(names)} for c, ix in cellix.items() if c.endswith('q4')})
print(cor.round(2).to_string())
