"""Red-team C loader: raw inputs only (no METRICS/CONTRASTS/HOLM/ANSWER_DETECTION/SHUFFLED_A)."""
import json, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd

MAIN = Path(r'C:/Users/omris/TAU/hallucination_detection')
W = MAIN / '.worktrees/decision-rule-v1'
R = MAIN / '.worktrees/readout-quickest-detection-v1/results/step_evidence_v1'
SSL = MAIN / '.worktrees/ssl-pseudolabel-residual-v1'
RUN = W / 'results/answer_gate_v1/run_20260930'
sys.path.insert(0, str(W))

ans = pd.read_csv(R / 'OOF_ANSWERS.csv', encoding='utf-8-sig', usecols=['uid', 'id', 'source_group', 'fold', 'cell', 'target'])
Zs = np.load(R / 'OOF_STEP_SCORES.npz'); off = Zs['offsets'].astype(np.int64); labels = Zs['labels'].astype(bool)
n = len(ans); ns = np.diff(off); S_ = int(off[-1])
cells = ans.cell.to_numpy(); fold = ans.fold.to_numpy(); ids = ans.id.to_numpy(); target = ans.target.to_numpy(); groups = ans.source_group.to_numpy()
pb = pd.Series(cells).str.startswith('pb_').to_numpy(); prm = ~pb
meta_path = Path(json.loads((R / 'INPUT_FREEZE.json').read_text(encoding='utf-8-sig'))['prm_metadata']['path'])
meta = {m['idx']: m for m in pickle.load(open(meta_path, 'rb')).values()}
cls = np.array([meta[ids[i]]['classification'] if prm[i] else '' for i in range(n)])
control = prm & (cls == 'correct'); noncontrol = prm & ~control
aid = np.repeat(np.arange(n), ns); step_fold = fold[aid]; prm_step = prm[aid]; pb_step = pb[aid]
has_err = np.bincount(aid, weights=labels, minlength=n) > 0
has_err_prm = has_err & prm
ms = noncontrol & (cls == 'multi_solutions'); err_nc = has_err & noncontrol
S = np.load(SSL / 'results/expectation_realization_v1/run_20260927_stage_b/STEP_SCORES.npz')['S_equal'].astype(float)
D = np.load(RUN / 'DECISIONS.npz'); DEC = {k: D[k] for k in D.files}
F = np.load(W / 'results/answer_gate_v1/ANSWER_FEATURES.npz'); X = F['X'].astype(float); names = [str(x) for x in F['names']]
assert np.array_equal(F['ids'], ids.astype(str)) and np.array_equal(F['folds'], fold)
assert np.array_equal(DEC['offsets'], off)
PBc = sorted(set(cells[pb]))
