"""Shared loaders for the paper-metric parity package (CPU only, existing data only).

Builds per-step scores for the four PRM-boosting arms on PRMBench (Qwen2.5-Math-PRM-7B backbone):
  PRM_native  : prm_risk = logit(1 - reward); native decision = prm_risk >= 0 (reward <= 0.5)
  LSML_CONT   : L-SML over [answer_z(prm_risk), energy_level, hist_entropy_cusum_abs_series, realized_drv]
                (label-chosen channels; replays results/remote_claims_replay_2026-10-15 S3_HEAD_LSML)
  CHAIN_N16   : CHAIN recipe of results/prm_boosting_bank_sweep_v1/lsml_bank_sweep.py on the nested N16 bank
                (label-free channel list, contains 3 digit channels)
  CHAIN_N13   : same on N13 (digit-free)
CHAIN is RE-FITTED here from the local exported step channels (float32), because the sweep's per-step scores
exist only on AIRCC. `parity()` compares the refit against the sweep's reported numbers before anything is used.

Every fusion here is fitted on the POOLED steps of all 6,969 PRMBench answers, without labels (transductive,
label-free); labels are read only by the metric code. Decision rule for fused arms: ANSWER_Z_Q80
(flag steps whose answer-standardized score is above the 80th percentile of all answer-z scores).
"""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_k, '2')
import csv, json, pickle, sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from spectral_utils.fusion_utils import lsml_continuous  # noqa: E402

RAW = ROOT / 'results/remote_claims_replay_2026-10-15/inputs/unpacked/results/prm_boosting_v1'
POP = ROOT / 'results/advisor_completion_2026-10-11/inputs/population.npz'
META = ROOT / 'results/advisor_completion_2026-10-11/inputs/prmbench_meta.json'
PRM_PKL = ROOT / 'dataset_cache/four_localization/prmbench_qwen25math7b_full/prmbench_prm.pkl'
SWEEP_CSV = ROOT / 'results/prm_boosting_bank_sweep_v1/SWEEP_RESULTS_qwen_prm_7b.csv'   # read only
REPLAY_CSV = ROOT / 'results/remote_claims_replay_2026-10-15/PRM7B_METRICS.csv'
SCORES_NPZ = HERE / 'ARM_SCORES.npz'

N11 = ['q15_H1', 'q15_VE1', 'chosen_surprisal', 'logprob_margin', 'true_tail50', 'energy_level',
       'energy_innovation', 'top15_turnover', 'top50_js', 'dominant_freq16', 'bocpd_p0']
N13 = N11 + ['realized_z', 'realized_drv']
N16 = N13 + ['digit_alternative', 'digit_spread', 'digit_alternative_innovation']
S3 = ['energy_level', 'hist_entropy_cusum_abs_series', 'realized_drv']
ARMS = ['PRM_native', 'LSML_CONT', 'CHAIN_N16', 'CHAIN_N13']
ARM_NOTE = {
    'PRM_native': 'Qwen2.5-Math-PRM-7B alone; supervised PRM, native reward<=0.5 rule',
    'LSML_CONT': 'L-SML of PRM + 3 LABEL-CHOSEN channels; Q80 rule',
    'CHAIN_N16': 'L-SML CHAIN of PRM + 16 label-free channels (3 digit channels); Q80 rule',
    'CHAIN_N13': 'L-SML CHAIN of PRM + 13 label-free digit-free channels; Q80 rule',
}


def answer_z(score, off):
    out = np.zeros(len(score))
    for a, b in zip(off[:-1], off[1:]):
        s = score[a:b]
        out[a:b] = (s - s.mean()) / max(s.std(), 1e-8)
    return out


def q80_flags(score, off):
    z = answer_z(score, off)
    return z > np.quantile(z, .8)


def auc(y, s):
    y = np.asarray(y, bool); n1 = int(y.sum()); n0 = len(y) - n1
    if not n1 or not n0:
        return np.nan
    return float((rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def load_population():
    with np.load(RAW / 'CHANNEL_CONDCORR_prm_backbone.npz', allow_pickle=True) as z:
        names = z['names'].astype(str).tolist(); X = z['S'].astype(float)
        y = z['y'].astype(bool); off = z['offsets'].astype(np.int64); ids = z['idx'].astype(str)
    signs = {r['channel']: float(r['sign']) for r in csv.DictReader((RAW / 'CHANNEL_ATLAS_prm_backbone.csv').open())}
    X *= np.array([signs[c] for c in names])          # undo the label-orientation of the exported atlas
    with np.load(RAW / 'SCORES_prm_backbone.npz', allow_pickle=True) as z:
        assert np.array_equal(z['idx'].astype(str), ids) and np.array_equal(z['offsets'], off)
        prm = z['prm_risk'].astype(float)
    with np.load(POP, allow_pickle=True) as z:
        assert np.array_equal(z['idx'].astype(str), ids) and np.array_equal(z['offsets'], off)
        assert np.array_equal(z['y'], y)
        cls = z['classification'].astype(str); group = z['group'].astype(str)
        tok = z['token_lengths'].astype(float); atok = z['answer_token_lengths'].astype(float)
    meta = {r['idx']: r for r in json.loads(META.read_text())['rows']}
    rec = np.zeros_like(y)                              # independent label rebuild from raw error_steps
    for j, i in enumerate(ids):
        a, b = off[j:j + 2]
        es = meta[i]['error_steps']; es = json.loads(es) if isinstance(es, str) else es
        rec[a:b] = np.isin(np.arange(1, b - a + 1), es)
    assert np.array_equal(rec, y), 'label rebuild mismatch'
    return dict(names=names, X=X, y=y, off=off, ids=ids, prm=prm, cls=cls, group=group, tok=tok, atok=atok)


def rewards_by_id():
    d = pickle.load(open(PRM_PKL, 'rb'))
    order = sorted(d)                                    # loader (Hub) order, see README
    rows = [d[k] for k in order]
    L = lambda v: v if isinstance(v, list) else json.loads(v)
    return {r['idx']: np.asarray(L(r['rewards']), float) for r in rows}, [r['idx'] for r in rows], \
        {r['idx']: r['classification'] for r in rows}


def lsml(V, groups=None, anchor=None):
    s, d = lsml_continuous(*[V[:, j] for j in range(V.shape[1])], groups=groups)
    s = np.asarray(s, float)
    ref = V[:, 0] if anchor is None else anchor
    if np.corrcoef(s, ref)[0, 1] < 0:
        s = -s
    return s, d


def build_scores(P):
    X, off, names = P['X'], P['off'], P['names']
    zp = answer_z(P['prm'], off)
    out = {'PRM_native': P['prm']}
    V4 = np.column_stack([zp] + [X[:, names.index(c)] for c in S3])
    out['LSML_CONT'], _ = lsml(V4)
    diag = {}
    for bk, chans in (('N16', N16), ('N13', N13)):
        V = X[:, [names.index(c) for c in chans]]
        _, dt = lsml(V, anchor=V.mean(1))
        vg = np.asarray(dt['virtual_classifiers'], float)
        VC = np.column_stack([zp] + [answer_z(vg[:, g], off) for g in range(vg.shape[1])])
        out[f'CHAIN_{bk}'], dc = lsml(VC, groups=np.arange(VC.shape[1]))
        diag[bk] = {'tele_K': int(dt['K']), 'tele_groups': np.asarray(dt['c']).tolist(),
                    'chain_cross_weights': np.asarray(dc['cross_weights'], float).tolist()}
    return out, diag


def flags_for(arm, score, off):
    return (score >= 0) if arm == 'PRM_native' else q80_flags(score, off)


def prmscore_counts(y, flag, nonc_step):
    """Official PRMBench confusion with POSITIVE = valid step (labels==1 means 'valid').
    TP: valid kept valid; FP: error predicted valid; TN: error flagged; FN: valid flagged."""
    m = nonc_step
    tp = int((~y & ~flag & m).sum()); fp = int((y & ~flag & m).sum())
    tn = int((y & flag & m).sum()); fn = int((~y & flag & m).sum())
    return tp, fp, tn, fn


def prmscore(tp, fp, tn, fn):
    f1p = 2 * tp / (2 * tp + fp + fn) if tp else 0.
    f1n = 2 * tn / (2 * tn + fp + fn) if tn else 0.
    return 100 * .5 * (f1p + f1n)


def within_auc(y, s, off, answers=None):
    vals = []
    for j, (a, b) in enumerate(zip(off[:-1], off[1:])):
        if answers is not None and not answers[j]:
            continue
        v = auc(y[a:b], s[a:b])
        if np.isfinite(v): vals.append(v)
    return float(np.mean(vals)), len(vals)


def get_scores(P=None, force=False):
    if SCORES_NPZ.exists() and not force:
        with np.load(SCORES_NPZ) as z:
            return {a: z[a] for a in ARMS}
    P = P or load_population()
    S, diag = build_scores(P)
    np.savez_compressed(SCORES_NPZ, **S)
    (HERE / 'FUSION_DIAG.json').write_text(json.dumps(diag, indent=1))
    return S


def parity(P, S):
    """Full-population PRMScore and within-AUC of each arm vs the reported sources."""
    y, off, cls = P['y'], P['off'], P['cls']
    nonc_step = np.repeat(cls != 'correct', np.diff(off))
    rep = {}
    for r in csv.DictReader(SWEEP_CSV.open()):
        if r['rule'] == 'Q80': rep[r['arm']] = (float(r['prmscore']), float(r['within_auc']))
    rows = []
    for a in ARMS:
        f = flags_for(a, S[a], off)
        ps = prmscore(*prmscore_counts(y, f, nonc_step)); wa, n = within_auc(y, S[a], off)
        rp, rw = rep[a]
        rows.append({'arm': a, 'prmscore_local': ps, 'prmscore_reported': rp, 'd_prmscore': ps - rp,
                     'within_auc_local': wa, 'within_auc_reported': rw, 'd_within_auc': wa - rw, 'n_auc': n,
                     'source': str(SWEEP_CSV.relative_to(ROOT))})
    return rows


if __name__ == '__main__':
    P = load_population()
    S = get_scores(P, force='--force' in sys.argv)
    rows = parity(P, S)
    with (HERE / 'PARITY.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(r['arm'], round(r['prmscore_local'], 4), round(r['prmscore_reported'], 4),
              round(r['within_auc_local'], 5), round(r['within_auc_reported'], 5))
