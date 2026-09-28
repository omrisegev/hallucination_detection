"""Red-team C: nulls, length-only / shuffled-count allocation, rate-vs-allocation decomposition for decision_rule_v1.
Reads only DECISIONS.npz of the run plus raw inputs. Writes nothing tracked."""
import json, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd

MAIN = Path(r'C:/Users/omris/TAU/hallucination_detection')
R = MAIN / '.worktrees/readout-quickest-detection-v1/results/step_evidence_v1'
W = MAIN / '.worktrees/decision-rule-v1'
SSL = MAIN / '.worktrees/ssl-pseudolabel-residual-v1/results/expectation_realization_v1'
OUT = Path(r'C:/Users/DELL/AppData/Local/Temp/claude/c--Users-omris-TAU-hallucination-detection/41041b9e-b636-4238-bb1a-111fd1f803fe/scratchpad/rt_dr_C')

ans = pd.read_csv(R / 'OOF_ANSWERS.csv', encoding='utf-8-sig'); Z = np.load(R / 'OOF_STEP_SCORES.npz')
off = Z['offsets']; lab = Z['labels'].astype(bool); n = len(ans); ns = np.diff(off); S_ = int(off[-1])
aid = np.repeat(np.arange(n), ns); fold = ans.fold.to_numpy(); ids = ans.id.to_numpy()
prm = ~ans.cell.str.startswith('pb_').to_numpy()
meta_path = Path(json.loads((R / 'INPUT_FREEZE.json').read_text(encoding='utf-8-sig'))['prm_metadata']['path'])
meta = {m['idx']: m for m in pickle.load(open(meta_path, 'rb')).values()}
cls = np.array([meta[ids[i]]['classification'] if prm[i] else '' for i in range(n)])
control = prm & (cls == 'correct'); nc = prm & ~control
print('prm', prm.sum(), 'controls', control.sum(), 'noncontrol', nc.sum())
D = np.load(W / 'results/decision_rule_v1/run_20260929/DECISIONS.npz')
assert (D['offsets'] == off).all()
F = {r: D[r].astype(bool) for r in ['R0_frozen', 'R1_global', 'R2_allocate', 'R3_ds_map', 'R4_ds_expected_f1', 'R5_ds_count']}
S = np.load(SSL / 'run_20260927_stage_b/STEP_SCORES.npz')['S_equal'].astype(float)
tauS = {int(k): float(v) for k, v in json.loads((SSL / 'run_20260927_stage_b_thr/THRESHOLDS.json').read_text(encoding='utf8'))['S_equal'].items()}
ncs = nc[aid]; cts = control[aid]; prms = prm[aid]


def parts(flag, y=lab, mask=ncs):
    v = ~flag[mask]; g = ~y[mask]
    tp = (v & g).sum(); fp = (v & ~g).sum(); tn = (~v & ~g).sum(); fn = (~v & g).sum()
    f1 = 2 * tp / (2 * tp + fp + fn); f1e = 2 * tn / (2 * tn + fn + fp)
    return dict(prm=(f1 + f1e) / 2, f1c=f1, f1e=f1e, TP=int(tp), FP=int(fp), TN=int(tn), FN=int(fn), rate=float((~v).mean()))


def prmscore(flag, y=lab):
    return parts(flag, y)['prm']


def ctrl_share(flag):
    c = np.bincount(aid, weights=flag, minlength=n); return float((c[control] > 0).mean())


# --- within-answer rank of S (stable, earlier step first on ties), and count-based flags
rank = np.empty(S_, int)
for i in range(n):
    a, b = off[i], off[i + 1]; o = np.argsort(-S[a:b], kind='stable'); rank[a + o] = np.arange(b - a)
def by_count(cnt): return rank < np.asarray(cnt)[aid]
cnt = {r: np.bincount(aid, weights=F[r], minlength=n).astype(int) for r in F}

P = {r: parts(F[r]) for r in F}
