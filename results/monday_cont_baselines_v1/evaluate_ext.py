"""Stage 2: evaluate the sealed external predictions with the frozen evaluator (same code path as
monday_nested_banks_v2/external_oriented.py). Refuses to run if a sealed file changed."""
import hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np, pandas as pd

OUT = Path(__file__).resolve().parent; MAIN = OUT.parents[1]; EXT = MAIN / 'results/monday_external_panel_v1'; V2 = MAIN / 'results/monday_nested_banks_v2'
seal = json.loads((OUT / 'ext/SEAL.json').read_text(encoding='utf8'))
for cell, v in seal['cells'].items():
    if hashlib.sha256((OUT / v['path']).read_bytes()).hexdigest() != v['sha256']: raise SystemExit(f'seal broken: {cell}')
spec = importlib.util.spec_from_file_location('ext_score', EXT / 'score.py'); ext = importlib.util.module_from_spec(spec); spec.loader.exec_module(ext)
sys.path.insert(0, str(MAIN / '.worktrees/ssl-pseudolabel-residual-v1'))
from spectral_utils.external_generalization.evaluation import confusion, within_auc
from spectral_utils.family_external_metrics import summary

NB = json.loads((V2 / 'BANKS.json').read_text(encoding='utf8')); rows = []
for cell, bench in ext.CELLS.items():
    P = np.load(OUT / seal['cells'][cell]['path']); uids = [str(u) for u in P['uids']]; lens = P['lengths']; ne = P['nonempty'].astype(bool)
    starts = np.r_[0, np.cumsum(lens)]
    gold = {g['uid']: g for g in json.loads((ext.INPUTS / 'evaluator_only' / (bench + '.json')).read_text(encoding='utf8'))}
    for bk in NB:
        for name in ('LSML_CONT', 'IU_PCR'):
            key = f'{bk}__{name}__score'
            if key not in P.files: rows.append({'cell': cell, 'bank': bk, 'scorer': name, 'status': 'FAILED'}); continue
            S = P[key]; Pr = P[f'{bk}__{name}__pred_correct']; counts, aucs = [], []
            for i, u in enumerate(uids):
                g = gold[u]; a, b = starts[i], starts[i + 1]; inc = np.asarray(g['include'], bool)
                counts.append(confusion(g['correct'], Pr[a:b], inc)); v = within_auc(g['correct'], S[a:b], inc & ne[a:b]); aucs.append(np.nan if v is None else v)
            rows.append({'cell': cell, 'bank': bk, 'scorer': name, 'rule': 'ANSWER_Z_Q80', 'official_metric': summary(np.array(counts), bench)['metric'],
                         'within_auc': float(np.nanmean(aucs)), 'status': 'OK'})
    print(cell, 'evaluated', flush=True)
pd.DataFrame(rows).to_csv(OUT / 'EXTERNAL_METRICS.csv', index=False)
(OUT / 'ext/EVALUATION_STATUS.json').write_text(json.dumps({'seal_verified': True, 'rows': len(rows)}, indent=1), encoding='utf8')
