"""CPU controls for PRM boosting (Omri, 2026-10-18; plan docs/post_advisor_2026-10-17/05_PRM_BOOSTING_EVIDENCE_PLAN_HE.md, axis C).

Same inputs, orientation, flag rule (ANSWER_Z_Q80) and paired source-group bootstrap as
results/prm_boosting_generalization_v1/run_v1/eval_lsml4/lsml_four.py. Only the fused views change.
All fusion is label-free; labels are used only to score.

Arms (per PRM, fitted on its pooled PRMBench steps):
  PRM_native, LSML_CONT                  replays of lsml_four (parity check)
  C1 label-free channel choice (no channel picked with labels):
    HALF_N16        1/2 z(PRM) + 1/2 mean of the 16 N16 channels
    LSML_N16_PRMG   continuous L-SML on PRM + N16, PRM forced into its own group, N16 groups detected label-free
    LSML_N16_AUTO   continuous L-SML on PRM + N16 with automatic grouping (PRM may be absorbed)
  C2 ablation of the 3 label-chosen channels:
    DROP_<ch>       L-SML on PRM + the other two channels
    ONLY_<ch>       1/2 z(PRM) + 1/2 z(ch)
  C3 null: PERM_ANS    the 3-channel block is moved to another answer of the same step count
              (keeps the channels' joint structure and position profile, breaks step-level alignment)
  C4 position/length: LSML_RESID  each channel residualised on relative position and log step length (label-free OLS)
  C5 decision rule: NATIVECOUNT   LSML_CONT ranking, with the number of flags per answer taken from the PRM's native rule
Contrasts: every arm minus PRM_native and minus LSML_CONT, within-AUC and PRMScore, 95% bootstrap.

    python3 lsml_controls.py <tag> <FEATURES.npz> <telemetry.pkl> <out_dir>
"""
import csv, importlib.util, json, sys
from pathlib import Path
import numpy as np

M = Path('/shared/cycle2_tau_averbuch_prj/omrisegev1/eval_mirror_prm_gen_v1')
spec = importlib.util.spec_from_file_location('ev', M / 'results/prm_boosting_generalization_v1/evaluate_amended.py')
EV = importlib.util.module_from_spec(spec); spec.loader.exec_module(EV)
from spectral_utils.fusion_utils import lsml_continuous  # noqa: E402

tag, feat, tele, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
SE = M / '.worktrees/readout-quickest-detection-v1/results/step_evidence_v1'
out.mkdir(parents=True, exist_ok=True)
with open(SE / 'OOF_ANSWERS.csv', newline='', encoding='utf-8-sig') as f: A = list(csv.DictReader(f))
F = dict(np.load(feat)); off_all = F['offsets'].astype(np.int64)
ans = np.flatnonzero(F['cells'].astype(str) == EV.CELL)
ids = F['row_ids'][ans].astype(str); o = np.r_[0, np.cumsum(np.diff(off_all)[ans])]
T = EV.metadata_from(tele)
meta = [dict(T[i], idx=i) for i in ids]
for r in meta:
    if 'native_labels' not in r: r['native_labels'] = [int(v > .5) for v in r['rewards']]
prm = np.concatenate([EV.risk(r['rewards']) for r in meta])
native = np.concatenate([np.asarray(r['native_labels'], int) == 0 for r in meta])
with np.load(SE / 'OOF_STEP_SCORES.npz') as z:
    lab_all = z['labels'].astype(bool); assert np.array_equal(z['offsets'], off_all)
step_idx = np.concatenate([np.arange(off_all[i], off_all[i + 1]) for i in ans])
labels = lab_all[step_idx]
fs = importlib.util.spec_from_file_location('prm_fuse', M / 'results/prm_boosting_v1/fuse.py')
FU = importlib.util.module_from_spec(fs); fs.loader.exec_module(FU); ext = FU.ext
ext.BN = dict(ext.BN, TRANSFER=list(EV.S3))
pop = ext.Population(F, ['TRANSFER', 'N16'])
bt, bn = pop.cell_bank('TRANSFER', ans), pop.cell_bank('N16', ans)
C3 = np.column_stack([bt['V'][:, bt['names'].index(c)] for c in EV.S3])
N16 = np.asarray(bn['V'], float)
zp = EV.az(prm, o)
n_steps = np.diff(o)
pos = np.concatenate([(np.arange(k) + .5) / k for k in n_steps])
# step token length: count of nonempty tokens per step is not stored per step; use the answer-z of
# log(step token count) from the telemetry spans when present, else position only.
spans = [m.get('step_token_spans') for m in meta]
if all(s is not None for s in spans):
    loglen = np.concatenate([np.log1p([max(b - a, 0) for a, b in s]) for s in spans])
    if len(loglen) != o[-1]: loglen = None
else:
    loglen = None


def lsml(V, groups=None):
    s, d = lsml_continuous(*[V[:, j] for j in range(V.shape[1])], groups=groups)
    s = np.asarray(s, float)
    if np.corrcoef(s, V[:, 0])[0, 1] < 0: s = -s
    return s, d


arms, diag = {'PRM_native': prm}, {}
V4 = np.column_stack([zp, C3])
arms['LSML_CONT'], d = lsml(V4); diag['LSML_CONT'] = {'K': int(d['K']), 'groups': np.asarray(d['c']).tolist()}
# C1
arms['HALF_N16'] = .5 * zp + .5 * N16.mean(1)
_, dn = lsml_continuous(*[N16[:, j] for j in range(N16.shape[1])])
cn = np.asarray(dn['c'], int)
arms['LSML_N16_PRMG'], d = lsml(np.column_stack([zp, N16]), groups=np.r_[cn.max() + 1, cn])
diag['LSML_N16_PRMG'] = {'K': int(d['K']), 'groups': np.r_[cn.max() + 1, cn].tolist()}
arms['LSML_N16_AUTO'], d = lsml(np.column_stack([zp, N16]))
diag['LSML_N16_AUTO'] = {'K': int(d['K']), 'groups': np.asarray(d['c']).tolist()}
# C2
for j, ch in enumerate(EV.S3):
    keep = [k for k in range(3) if k != j]
    arms[f'DROP_{ch}'], _ = lsml(np.column_stack([zp, C3[:, keep]]))
    arms[f'ONLY_{ch}'] = .5 * zp + .5 * C3[:, j]
# C3: move the channel block to another answer with the same number of steps
rng = np.random.default_rng(20261018)
perm_rows = np.arange(o[-1])
for k in np.unique(n_steps):
    js = np.flatnonzero(n_steps == k)
    if len(js) < 2: continue
    tgt = js[rng.permutation(len(js))]
    for a, b in zip(js, tgt):
        perm_rows[o[a]:o[a + 1]] = np.arange(o[b], o[b + 1])
arms['PERM_ANS'], _ = lsml(np.column_stack([zp, C3[perm_rows]]))
# C4
X = np.column_stack([np.ones(o[-1]), pos] + ([loglen] if loglen is not None else []))
R = C3 - X @ np.linalg.lstsq(X, C3, rcond=None)[0]
R = np.column_stack([EV.az(R[:, j], o) for j in range(3)])
arms['LSML_RESID'], _ = lsml(np.column_stack([zp, R]))
diag['LSML_RESID'] = {'covariates': ['position'] + (['log_step_tokens'] if loglen is not None else [])}

flags = {n: native if n == 'PRM_native' else EV.flags_for(s, o) for n, s in arms.items()}
# C5: native count, fused order
nc = np.zeros(o[-1], bool)
for j in range(len(ids)):
    a, b = o[j], o[j + 1]; k = int(native[a:b].sum())
    if k: nc[a + np.argsort(-arms['LSML_CONT'][a:b], kind='stable')[:k]] = True
arms['NATIVECOUNT'] = arms['LSML_CONT']; flags['NATIVECOUNT'] = nc

rows = [{'model': tag, 'arm': n, **EV.measure(s, flags[n], labels, o, ids, meta)[0]} for n, s in arms.items()]
groups = np.array([A[i]['source_group'] for i in ans])
unique, inv = np.unique(groups, return_inverse=True)
rb = np.random.default_rng(EV.SEED)
W = np.vstack([np.ones(len(unique)), rb.multinomial(len(unique), np.full(len(unique), 1 / len(unique)), size=EV.DRAWS)])
err = np.zeros(o[-1], bool)
for j, m in enumerate(meta):
    for e in m['error_steps']:
        if 1 <= e <= o[j + 1] - o[j]: err[o[j] + e - 1] = True
B = {}
for n, s in arms.items():
    _, au = EV.measure(s, flags[n], labels, o, ids, meta)
    c4 = np.zeros((len(ids), 4))
    for j, m in enumerate(meta):
        if m['classification'] == 'correct': continue
        y, f = err[o[j]:o[j + 1]], flags[n][o[j]:o[j + 1]]
        c4[j] = [(~y & ~f).sum(), (y & ~f).sum(), (y & f).sum(), (~y & f).sum()]
    G = np.zeros((len(unique), 6)); np.add.at(G, inv, np.column_stack([np.where(np.isfinite(au), au, 0), np.isfinite(au), c4]))
    t = W @ G; tp, fp, tn, fn = t[:, 2:].T
    B[n] = (t[:, 0] / t[:, 1], 50 * (tp / (tp + .5 * (fp + fn)) + tn / (tn + .5 * (fp + fn))))
con = []
for a in arms:
    for b in ('PRM_native', 'LSML_CONT'):
        if a == b: continue
        for j, mt in enumerate(('within_auc', 'prmscore')):
            d = B[a][j] - B[b][j]
            con.append({'model': tag, 'contrast': f'{a} - {b}', 'metric': mt, 'delta': d[0],
                        'ci_low': np.quantile(d[1:], .025), 'ci_high': np.quantile(d[1:], .975)})
for path, rr in ((out / f'CONTROLS_RESULTS_{tag}.csv', rows), (out / f'CONTROLS_BOOTSTRAP_{tag}.csv', con)):
    with open(path, 'w', newline='') as f:
        wr = csv.DictWriter(f, list(dict.fromkeys(k for r in rr for k in r))); wr.writeheader(); wr.writerows(rr)
(out / f'CONTROLS_DIAG_{tag}.json').write_text(json.dumps(diag, indent=1))
for r in rows: print(tag, r['arm'], round(r['within_auc'], 4), round(r['prmscore'], 2))
