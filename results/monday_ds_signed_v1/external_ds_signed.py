"""DS-signed average on the three exploratory external cells. Mirrors monday_nested_banks_v2/external_oriented.py exactly
(same features, ANSWER_Z_Q80 rule, frozen evaluator); only the sign rule is the Dawid-Skene rule of PROTOCOL.json.
Computed AFTER the seal, no tuning; labelled as such."""
import importlib.util, json, sys
from pathlib import Path
import numpy as np, pandas as pd

OUT = Path(__file__).resolve().parent; MAIN = OUT.parents[1]; V2 = MAIN / 'results/monday_nested_banks_v2'; EXT = MAIN / 'results/monday_external_panel_v1'
spec = importlib.util.spec_from_file_location('ext_score', EXT / 'score.py'); ext = importlib.util.module_from_spec(spec); spec.loader.exec_module(ext)
NB = json.loads((V2 / 'BANKS.json').read_text(encoding='utf8')); ext.BN = NB
sys.path.insert(0, str(MAIN / '.worktrees/ssl-pseudolabel-residual-v1'))
from spectral_utils.external_generalization.evaluation import confusion, within_auc
from spectral_utils.family_external_metrics import summary
from spectral_utils.monday_fusion_decision import rules as _r  # noqa: F401 (same runtime as external_oriented.py)
SA = ext.SA
rows = []
for cell, bench in ext.CELLS.items():
    F = np.load(ext.FE / cell / 'FEATURES.npz'); uids = [str(u) for u in F['uids']]; lens = F['nonempty_lengths']; ne = F['nonempty'].astype(bool)
    starts = np.r_[0, np.cumsum(lens)]; pop = ext.Population(F, list(NB)); off = pop.off
    gold = {g['uid']: g for g in json.loads((ext.INPUTS / 'evaluator_only' / (bench + '.json')).read_text(encoding='utf8'))}
    for bk in NB:
        bank = pop.cell_bank(bk, np.arange(len(uids))); V = bank['V']; lev = pop.level[bank['rows']]
        lev_sg = np.array([(np.sign(np.corrcoef(V[:, j], lev)[0, 1]) or 1.) if V[:, j].std() > 0 else 1. for j in range(V.shape[1])])
        sg = np.where(SA.em_estimate(bank['marks'], 'ds')['pi'] > .5, 1., -1.)
        s = (V * sg).mean(1); az = np.zeros_like(s)
        for a, b in zip(off[:-1], off[1:]): v = s[a:b]; az[a:b] = (v - v.mean()) / max(v.std(), 1e-8)
        flag = az >= np.quantile(az, .8)
        full_s = np.full(int(lens.sum()), np.nan); full_s[np.flatnonzero(ne)] = s
        full_p = np.zeros(int(lens.sum()), int); full_p[np.flatnonzero(ne)] = (~flag).astype(int)
        counts = []; aucs = []
        for i, u in enumerate(uids):
            g = gold[u]; a, b = starts[i], starts[i + 1]; inc = np.asarray(g['include'], bool)
            counts.append(confusion(g['correct'], full_p[a:b], inc)); v = within_auc(g['correct'], full_s[a:b], inc & ne[a:b]); aucs.append(np.nan if v is None else v)
        rows.append({'cell': cell, 'bank': bk, 'scorer': 'DS_SIGNED', 'rule': 'ANSWER_Z_Q80', 'official_metric': summary(np.array(counts), bench)['metric'],
                     'within_auc': float(np.nanmean(aucs)), 'n_flipped': int((sg < 0).sum()), 'n_flipped_level_anchor': int((lev_sg < 0).sum()),
                     'same_flips_as_level_anchor': bool(np.array_equal(sg, lev_sg)), 'computed_after_seal': True})
    print(cell, 'done', flush=True)
pd.DataFrame(rows).to_csv(OUT / 'EXTERNAL_DS_SIGNED.csv', index=False)
