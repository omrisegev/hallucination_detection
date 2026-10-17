"""Label-free fusion of the PRM score with the 3 channels (Omri, 2026-10-14): CONT L-SML instead of the fixed 1/2 weight.

Views per answer step (answer-z, oriented +1 = riskier): PRM risk, energy_level, hist_entropy_cusum_abs_series,
realized_drv, the last three from the PRM's own backbone bank (same construction as evaluate.py).
Arms, each fitted label-free on the pooled PRMBench steps of one model:
  LSML_CONT    spectral_utils.fusion_utils.lsml_continuous, automatic group detection (the monday_cont_baselines path)
  SML4         lsml_continuous with every view its own group (conditional independence across all 4 views)
  EQ4          plain mean of the 4 views (PRM weight 1/4)
  S3           the earlier fixed rule (1/2 PRM + 1/2 mean of the 3 channels), as a reference
The fused score's sign is fixed label-free so that it correlates positively with the PRM view. Weights are
reported as least-squares coefficients of the fused score on the 4 views. The decision rule is ANSWER_Z_Q80,
and the bootstrap is the paired source-group one used by evaluate.py.

    python3 lsml_four.py <tag> <adapter-for-labels-meta> <FEATURES.npz> <telemetry.pkl> <out_dir>
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
    if 'native_labels' not in r: r['native_labels'] = [int(v > .5) for v in r['rewards']]   # v1 Qwen-7B cell
prm = np.concatenate([EV.risk(r['rewards']) for r in meta])
native = np.concatenate([np.asarray(r['native_labels'], int) == 0 for r in meta])
with np.load(SE / 'OOF_STEP_SCORES.npz') as z:
    lab_all = z['labels'].astype(bool); assert np.array_equal(z['offsets'], off_all)
labels = lab_all[np.concatenate([np.arange(off_all[i], off_all[i + 1]) for i in ans])]
fs = importlib.util.spec_from_file_location('prm_fuse', M / 'results/prm_boosting_v1/fuse.py')
FU = importlib.util.module_from_spec(fs); fs.loader.exec_module(FU); ext = FU.ext
ext.BN = dict(ext.BN, TRANSFER=list(EV.S3))
bank = ext.Population(F, ['TRANSFER']).cell_bank('TRANSFER', ans)
V = np.column_stack([EV.az(prm, o)] + [bank['V'][:, bank['names'].index(c)] for c in EV.S3])
names = ['prm_risk'] + list(EV.S3)
arms = {'PRM_native': prm, 'PRM_Q80': prm, 'S3': .5 * V[:, 0] + .5 * V[:, 1:].mean(1), 'EQ4': V.mean(1)}
diag = {}
for name, groups in (('LSML_CONT', None), ('SML4', [0, 1, 2, 3])):
    s, d = lsml_continuous(*[V[:, j] for j in range(4)], groups=groups)
    s = np.asarray(s, float)
    if np.corrcoef(s, V[:, 0])[0, 1] < 0: s = -s
    w = np.linalg.lstsq(np.column_stack([V, np.ones(len(V))]), s, rcond=None)[0][:4]
    arms[name] = s
    diag[name] = {'K': int(d['K']), 'groups': np.asarray(d['c']).tolist(), 'degenerate': bool(d.get('degenerate', False)),
                  'weights_normalized': dict(zip(names, (w / np.abs(w).sum()).round(4).tolist()))}
flags = {n: native if n == 'PRM_native' else EV.flags_for(s, o) for n, s in arms.items()}
rows = [{'model': tag, 'arm': n, **EV.measure(s, flags[n], labels, o, ids, meta)[0]} for n, s in arms.items()]
groups = np.array([A[i]['source_group'] for i in ans])
unique, inv = np.unique(groups, return_inverse=True)
rng = np.random.default_rng(EV.SEED)
W = np.vstack([np.ones(len(unique)), rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)), size=EV.DRAWS)])
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
for a in ('LSML_CONT', 'SML4', 'EQ4', 'S3'):
    for b in ('PRM_native', 'S3'):
        if a == b: continue
        for j, mt in enumerate(('within_auc', 'prmscore')):
            d = B[a][j] - B[b][j]
            con.append({'model': tag, 'contrast': f'{a} - {b}', 'metric': mt, 'delta': d[0],
                        'ci_low': np.quantile(d[1:], .025), 'ci_high': np.quantile(d[1:], .975)})
for path, rr in ((out / f'LSML4_RESULTS_{tag}.csv', rows), (out / f'LSML4_BOOTSTRAP_{tag}.csv', con)):
    with open(path, 'w', newline='') as f:
        wr = csv.DictWriter(f, list(dict.fromkeys(k for r in rr for k in r))); wr.writeheader(); wr.writerows(rr)
(out / f'LSML4_DIAG_{tag}.json').write_text(json.dumps(diag, indent=1))
for r in rows: print(tag, r['arm'], round(r['within_auc'], 4), round(r['prmscore'], 2))
print(json.dumps(diag))
