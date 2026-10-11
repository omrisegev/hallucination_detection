"""Step 4: re-run the self-generated B16 behaviour check (claude/self-generated-step-labels-v1
scripts/self_generated_labels/b16_eval.py) against the think-fix teacher-forced ProcessBench reference.

Identical to b16_eval.main() (its own functions are imported and used unchanged) except: the ProcessBench
side reads STEP_SCORES_<variant>.npz (B16 BASE/GRP refitted per dataset on the rebuilt bank by
step4_pb_reference.py) instead of per_dataset_fit_v1/run_20260929/STEP_SCORES.npz, and results are written here.
The self-generated side (b16/STEP_SCORES.npz on the GENERATED answers) is unchanged: generation does not use
teacher forcing. Variant old_rebuild replays the frozen reference (Step 462) to <= 2.7e-9.
    python run_b16_eval_thinkfix.py
"""
import importlib.util, json, os
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent; MAIN = HERE.parents[2]
SG = MAIN / '.worktrees/self-generated-step-labels-v1'
spec = importlib.util.spec_from_file_location('b16_eval', SG / 'scripts/self_generated_labels/b16_eval.py')
E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)


def run(pb_path, use_final):
    cons = E.final_labels() if use_final else E.consensus()
    fit = json.load(open(E.ROOT / 'b16/FIT.json')); own_sc = np.load(E.ROOT / 'b16/STEP_SCORES.npz'); ooff = own_sc['offsets']
    ans = pd.read_csv(MAIN / '.worktrees/readout-quickest-detection-v1/results/step_evidence_v1/OOF_ANSWERS.csv', encoding='utf-8-sig')
    pb_sc = np.load(pb_path); poff = pb_sc['offsets']
    rng = np.random.default_rng(E.SEED)
    res = {'equivalence_margin_within_auc': E.MARGIN, 'draws': E.DRAWS, 'seed': E.SEED, 'pb_reference': str(pb_path), 'cells': {}}
    for arm in E.ARMS:
        pooled = {'own': {}, 'pb': {}}
        for tag, (cell, pbcell) in E.PAIRS.items():
            own = defaultdict(list)
            for i, m in enumerate(fit['answers']):
                if m['cell'] != cell or m['item_id'] not in cons or cons[m['item_id']] < 0:
                    continue
                own[f"{cell}::{m['src_idx']}"].append(E.answer_row(own_sc[arm][ooff[i]:ooff[i + 1]], cons[m['item_id']]))
            pb = defaultdict(list)
            for i in np.flatnonzero((ans.cell == pbcell).to_numpy()):
                t = int(ans.target.iloc[i])
                if t >= 0:
                    pb[ans.source_group.iloc[i]].append(E.answer_row(pb_sc[arm][poff[i]:poff[i + 1]], t))
            pooled['own'].update(own); pooled['pb'].update({f'{pbcell}::{k}': v for k, v in pb.items()})
            res['cells'].setdefault(tag, {})[arm] = E.compare(own, pb, rng)
        res.setdefault('pooled', {})[arm] = E.compare(pooled['own'], pooled['pb'], rng)
    res['labels'] = 'FINAL_LABELS_V1 (890)' if use_final else 'round-1 consensus (856)'
    return res


rows = []
for variant in ('old_rebuild', 'thinkfix', 'thinkfix_mask1'):
    for use_final in (False, True):
        r = run(HERE / f'STEP_SCORES_{variant}.npz', use_final)
        tag = 'FINAL890' if use_final else 'CONSENSUS856'
        (HERE / f'B16_BEHAVIOUR_{variant}_{tag}.json').write_text(json.dumps(r, indent=1))
        for arm in E.ARMS:
            for cell in list(E.PAIRS) + ['pooled']:
                c = r['pooled'][arm] if cell == 'pooled' else r['cells'][cell][arm]
                rows.append({'labels': tag, 'variant': variant, 'arm': arm, 'cell': cell, 'own_within': c['own_within'],
                             'pb_within': c['pb_within'], 'd_within': c['d_within'], 'd_lo': c['d_within_ci95'][0], 'd_hi': c['d_within_ci95'][1],
                             'equivalent_within': c['equivalent_within'], 'own_hit': c['own_hit'], 'pb_hit': c['pb_hit']})
df = pd.DataFrame(rows); df.to_csv(HERE / 'B16_SELFGEN_VS_PB_TF.csv', index=False)
print(df[(df.cell == 'pooled')].round(4).to_string(index=False))
