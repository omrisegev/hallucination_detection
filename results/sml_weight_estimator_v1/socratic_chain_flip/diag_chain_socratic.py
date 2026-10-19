"""Diagnose why CHAIN / PRMG fall below chance on Socratic-PRMBench (CPU only, no inference).

Recomputes TELE / PRMG / CHAIN exactly as results/prm_boost_ablation5_v1/ablation5_socratic.py
(same EV.az / EV.risk, same Population.cell_bank, same guarded lsml_continuous, same pooled-correlation
orientation), then prints LABEL-FREE diagnostics. Only the function `read_gold_labels` touches labels, and it
is called only in `evaluate_with_gold`, after every score has been computed and frozen.

    python diag_chain_socratic.py --repo <repo root> --bank <dir with FEATURES.npz, REWARDS.npz> \
        [--gold <socratic_gold.json>] [--banks N13 N16] [--parity <SCORES npz from ablation5_socratic.py>]

Without --gold the script stops after the label-free part.

Label-free variants (all computed before labels are read):
  CHAIN            as shipped: SML over [zp, az(g_1..g_K)], sign-majority eigvec, flipped if corr(s, zp) < 0
  CHAIN_RAW        the SML output before the anchor flip
  CHAIN_ABS        same views, weights |v| (every view forced positive: the inputs are pre-oriented risks)
  CHAIN_CLIP0      negative weights set to 0
  CHAIN_WITHIN     CHAIN_RAW oriented per answer by the sign of its within-answer correlation with zp
  CHAIN_TELESIGN   CHAIN_RAW oriented by pooled corr with TELE (the telemetry anchor) instead of zp
  PRMG_ABS         PRMG with |cross weights| (PRM its own group)
"""
import argparse, importlib.util, json, sys, types
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--repo', required=True)
ap.add_argument('--bank', required=True)
ap.add_argument('--gold', default=None)
ap.add_argument('--banks', nargs='+', default=['N13', 'N16'])
ap.add_argument('--parity', default=None)
ap.add_argument('--out', default=None, help='optional JSON with all printed numbers')
A = ap.parse_args()
M = Path(A.repo).resolve(); sys.path.insert(0, str(M))

# ---- identical loading to ablation5_socratic.py ---------------------------------------------------------
_src = M / 'results/prm_boosting_generalization_v1/run_v1/evaluate_amended.py'
EV = types.ModuleType('ev'); EV.__file__ = str(M / 'results/prm_boosting_generalization_v1/evaluate_amended.py')
sys.modules['ev'] = EV
exec(compile(_src.read_text(), str(_src), 'exec'), EV.__dict__)
from spectral_utils.fusion_utils import lsml_continuous_guarded as lsml_continuous  # noqa: E402
from spectral_utils.fusion_utils import sml_fuse_signed  # noqa: E402

bank = Path(A.bank)
F = dict(np.load(bank / 'FEATURES.npz')); R = dict(np.load(bank / 'REWARDS.npz'))
assert np.array_equal(F['uids'], R['uids']) and np.array_equal(F['nonempty_lengths'], R['nonempty_lengths'])
ne = F['nonempty'].astype(bool); o = F['offsets'].astype(np.int64)
F['cells'] = np.array([EV.CELL] * len(F['uids'])); F['row_ids'] = F['uids']
fs = importlib.util.spec_from_file_location('prm_fuse', M / 'results/prm_boosting_v1/fuse.py')
FU = importlib.util.module_from_spec(fs); fs.loader.exec_module(FU); ext = FU.ext
ans = np.arange(len(F['uids']))
pop = ext.Population(F, list(A.banks))
prm = EV.risk(np.clip(R['rewards'][ne], 1e-6, 1 - 1e-6)); zp = EV.az(prm, o)
nst = np.diff(o); pos = np.concatenate([(np.arange(k) + .5) / k for k in nst]); zpos = EV.az(pos, o)
OUT = {'bank': str(bank), 'n_answers': int(len(ans)), 'n_rows': int(len(zp))}


def lsml(V, groups=None, anchor=None):
    s, d = lsml_continuous(*[V[:, j] for j in range(V.shape[1])], groups=groups)
    s = np.asarray(s, float); raw = s.copy()
    ref = V[:, 0] if anchor is None else anchor
    flipped = bool(np.corrcoef(s, ref)[0, 1] < 0)
    if flipped: s = -s
    return s, d, raw, flipped


def C(a, b): return float(np.corrcoef(a, b)[0, 1])


def per_answer_corr(a, b):
    r = []
    for i, j in zip(o[:-1], o[1:]):
        x, y = a[i:j], b[i:j]
        if j - i >= 3 and x.std() > 1e-12 and y.std() > 1e-12: r.append(np.corrcoef(x, y)[0, 1])
    r = np.asarray(r)
    return {'mean': float(r.mean()), 'median': float(np.median(r)), 'frac_neg': float((r < 0).mean()), 'n': int(len(r))}


def orient_within(s, ref):
    s = s.copy()
    for i, j in zip(o[:-1], o[1:]):
        if j - i >= 2 and s[i:j].std() > 1e-12 and ref[i:j].std() > 1e-12 and np.corrcoef(s[i:j], ref[i:j])[0, 1] < 0:
            s[i:j] = -s[i:j]
    return s


arms = {'zp': zp, 'POS': zpos}
for bk in A.banks:
    b = pop.cell_bank(bk, ans); assert np.array_equal(b['off'], o)
    V = np.asarray(b['V'], float)
    tele, dt, tele_raw, tele_flip = lsml(V, anchor=V.mean(1))
    cn = np.asarray(dt['c'], int)
    prmg, dg, prmg_raw, prmg_flip = lsml(np.column_stack([zp, V]), groups=np.r_[cn.max() + 1, cn])
    vg = np.asarray(dt['virtual_classifiers'], float)
    G = [EV.az(vg[:, g], o) for g in range(vg.shape[1])]
    VC = np.column_stack([zp] + G)
    chain, dc, chain_raw, chain_flip = lsml(VC, groups=np.arange(VC.shape[1]))
    w = np.asarray(dc['cross_weights'], float)
    # with groups = arange every group is a singleton, so CHAIN_RAW = VC @ w exactly; check it
    assert np.allclose(chain_raw, VC @ w), 'CHAIN cross stage is not VC @ cross_weights'
    _, w_direct = sml_fuse_signed(*[VC[:, j] for j in range(VC.shape[1])])
    Roff = np.cov(VC.T); np.fill_diagonal(Roff, 0)
    ev = np.linalg.eigvalsh(Roff)
    sgn = -1. if chain_flip else 1.
    chain_abs = VC @ np.abs(w); chain_clip = VC @ np.clip(sgn * w, 0, None)
    chain_within = orient_within(chain_raw, zp)
    chain_tsign = chain_raw * (1. if C(chain_raw, tele) >= 0 else -1.)
    wg = np.asarray(dg['cross_weights'], float); vgg = np.asarray(dg['virtual_classifiers'], float)
    prmg_abs = vgg @ np.abs(wg)
    gw_tele = np.asarray(dt['cross_weights'], float)
    d = {
        'channels': list(b['names']), 'zero_var_channels': [str(n) for n, sd in zip(b['names'], V.std(0)) if sd < 1e-9],
        'tele_groups': cn.tolist(), 'tele_K': int(dt['K']),
        'tele_cross_weights': gw_tele.tolist(), 'tele_flipped_by_anchor': tele_flip,
        'tele_within_group_weights': [{'channels': [b['names'][k] for k in idx], 'w': np.asarray(ww).round(4).tolist()}
                                      for idx, ww in dt['group_weights']],
        'corr_TELE_zp': C(tele, zp), 'corr_MEAN_zp': C(V.mean(1), zp), 'corr_TELE_MEAN': C(tele, V.mean(1)),
        'corr_TELE_POS': C(tele, zpos), 'corr_zp_POS': C(zp, zpos), 'corr_MEAN_POS': C(V.mean(1), zpos),
        'per_answer_corr_TELE_zp': per_answer_corr(tele, zp),
        'CHAIN_view_names': ['zp'] + [f'az(g{g})' for g in range(len(G))],
        'CHAIN_cross_weights_signed_as_returned': w.tolist(),
        'CHAIN_cross_weights_after_anchor_flip': (sgn * w).tolist(),
        'CHAIN_direct_sml_weights': np.asarray(w_direct).tolist(),
        'CHAIN_flipped_by_zp_anchor': chain_flip,
        'CHAIN_Roff_eigvals': ev.round(4).tolist(),
        'CHAIN_view_corr_matrix': np.corrcoef(VC.T).round(4).tolist(),
        'corr_group_zp_pooled': [C(g, zp) for g in G],
        'corr_group_zp_per_answer': [per_answer_corr(g, zp) for g in G],
        'corr_group_POS': [C(g, zpos) for g in G],
        'corr_CHAIN_zp_pooled': C(chain, zp), 'corr_CHAIN_zp_per_answer': per_answer_corr(chain, zp),
        'corr_CHAIN_TELE': C(chain, tele), 'corr_CHAIN_MEAN': C(chain, V.mean(1)), 'corr_CHAIN_POS': C(chain, zpos),
        'PRMG_cross_weights_signed': wg.tolist(), 'PRMG_flipped_by_zp_anchor': prmg_flip,
        'corr_PRMG_zp': C(prmg, zp), 'corr_PRMG_TELE': C(prmg, tele),
    }
    OUT[bk] = d
    arms.update({f'TELE_{bk}': tele, f'MEAN_{bk}': V.mean(1), f'HALF_{bk}': .5 * zp + .5 * V.mean(1),
                 f'EQ_{bk}': np.column_stack([zp, V]).mean(1), f'PRMG_{bk}': prmg, f'PRMG_ABS_{bk}': prmg_abs,
                 f'CHAIN_{bk}': chain, f'CHAIN_RAW_{bk}': chain_raw, f'CHAIN_ABS_{bk}': chain_abs,
                 f'CHAIN_CLIP0_{bk}': chain_clip, f'CHAIN_WITHIN_{bk}': chain_within, f'CHAIN_TELESIGN_{bk}': chain_tsign})
    for g in range(len(G)): arms[f'G{g}_{bk}'] = G[g]

print('==== LABEL-FREE DIAGNOSTICS ====')
for bk in A.banks:
    d = OUT[bk]
    print(f'\n--- {bk}: K={d["tele_K"]} groups={d["tele_groups"]} zero-var={d["zero_var_channels"]}')
    print(f'TELE cross weights {np.round(d["tele_cross_weights"], 4).tolist()}  flipped by mean anchor: {d["tele_flipped_by_anchor"]}')
    print(f'corr(TELE,zp)={d["corr_TELE_zp"]:+.4f}  corr(MEAN,zp)={d["corr_MEAN_zp"]:+.4f}  corr(TELE,MEAN)={d["corr_TELE_MEAN"]:+.4f}')
    print(f'corr with position: zp {d["corr_zp_POS"]:+.4f}  TELE {d["corr_TELE_POS"]:+.4f}  MEAN {d["corr_MEAN_POS"]:+.4f}')
    print('per-answer corr(TELE,zp):', {k: round(v, 4) for k, v in d['per_answer_corr_TELE_zp'].items()})
    print('CHAIN views', d['CHAIN_view_names'])
    print('CHAIN weights as returned by SML (sign-majority):', np.round(d['CHAIN_cross_weights_signed_as_returned'], 4).tolist())
    print(f'CHAIN flipped by zp anchor: {d["CHAIN_flipped_by_zp_anchor"]}  -> effective weights',
          np.round(d['CHAIN_cross_weights_after_anchor_flip'], 4).tolist())
    print('R_off eigenvalues', d['CHAIN_Roff_eigvals'])
    print('corr(az(g), zp) pooled  :', np.round(d['corr_group_zp_pooled'], 4).tolist())
    print('corr(az(g), zp) per-ans :', [round(x['mean'], 4) for x in d['corr_group_zp_per_answer']])
    print('corr(az(g), position)   :', np.round(d['corr_group_POS'], 4).tolist())
    print('CHAIN view corr matrix:'); [print('   ', r) for r in d['CHAIN_view_corr_matrix']]
    print(f'corr(CHAIN,zp) pooled {d["corr_CHAIN_zp_pooled"]:+.4f}; per-answer',
          {k: round(v, 4) for k, v in d['corr_CHAIN_zp_per_answer'].items()})
    print(f'corr(CHAIN,TELE) {d["corr_CHAIN_TELE"]:+.4f}  corr(CHAIN,MEAN) {d["corr_CHAIN_MEAN"]:+.4f}  corr(CHAIN,pos) {d["corr_CHAIN_POS"]:+.4f}')
    print(f'PRMG cross weights {np.round(d["PRMG_cross_weights_signed"], 4).tolist()} flipped: {d["PRMG_flipped_by_zp_anchor"]}  corr(PRMG,TELE) {d["corr_PRMG_TELE"]:+.4f}')

if A.parity:
    P = dict(np.load(A.parity))
    for k in [k for k in arms if k.split('_')[0] in ('TELE', 'PRMG', 'CHAIN', 'HALF', 'MEAN', 'EQ') and f'score__{k}' in P]:
        print(f'PARITY {k}: max|diff| = {np.max(np.abs(P["score__" + k] - arms[k])):.2e}')


# ---- labels: read only here, after all scores above are frozen -------------------------------------------
def read_gold_labels(path):
    """The ONLY function that reads gold. Returns {uid: record} with correct/include per original step."""
    return {g['uid']: g for g in json.loads(Path(path).read_text(encoding='utf8'))}


def evaluate_with_gold(gold_path):
    from spectral_utils.external_generalization.evaluation import within_auc
    gold = read_gold_labels(gold_path)
    uids = [str(u) for u in F['uids']]; lens = F['nonempty_lengths']; starts = np.r_[0, np.cumsum(lens)]
    assert set(uids) <= set(gold), 'uid mismatch between bank and gold'
    res = {}
    for name, s in arms.items():
        full = np.full(int(lens.sum()), np.nan); full[np.flatnonzero(ne)] = s; A_ = []
        for i, u in enumerate(uids):
            g = gold[u]; a, b = starts[i], starts[i + 1]; inc = np.asarray(g['include'], bool)
            assert len(inc) == b - a, f'step count mismatch for {u}'
            v = within_auc(g['correct'], full[a:b], inc & ne[a:b]); A_.append(np.nan if v is None else v)
        res[name] = float(np.nanmean(A_))
    print('\n==== WITHIN-ANSWER AUC (gold read now; higher score = more likely error) ====')
    for k, v in res.items(): print(f'{k:22s} {v:.4f}')
    return res


if A.gold:
    OUT['within_auc'] = evaluate_with_gold(A.gold)
if A.out:
    Path(A.out).write_text(json.dumps(OUT, indent=1))
