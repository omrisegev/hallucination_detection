"""Review precompute of the three-rule decision table (plus top-k controls) for the highlighted comparison,
CONTENT position only: original banks, groups_answer vs channels_cell, recipes HEM_MOM, MEAN_MOM, HEM_EQ, MEAN_EQ,
DS_FILTER_EQ, ALL_EQ. Uses Codex's committed functions (input_state, raw_pool, raw_variant, entropy_feature, rules,
evaluate) unchanged. No nulls and no position variants: those are the slow parts of Codex's layer stage.

Gate: (1) CONTENT/ANSWER_Z_Q80 rows equal the sealed estimator-input content METRICS; (2) the groups_answer rows of every
rule equal Codex's sealed level-reduction layer rows for variant 'original' (identical scores) where those exist.
Writes only into this review folder. raw_pool's replay file is redirected here. Run with python -B."""
import os
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[k] = '2'
import sys, json, time, datetime
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(r'C:\Users\omris\TAU\hallucination_detection'); OUT = Path(__file__).resolve().parent / 'decision_rules_precompute'
OUT.mkdir(exist_ok=True)
CODE = ROOT / 'results/monday_fusion_analysis_v1/code'
sys.path.insert(0, str(CODE)); sys.path.insert(0, str(CODE / 'scripts/experiments'))
import spectral_utils.monday_fusion_decision as D
from spectral_utils.monday_fusion_protocol import RUNS
from spectral_utils.monday_fusion_replay import HistoricalData
from spectral_utils.monday_fusion_layers import input_state
from spectral_utils.monday_fusion_metrics import evaluate, top_flags
import calfix_common as CC

_real_dump = D.dump
def _redirect(path, obj):
    _real_dump(OUT / Path(path).name, obj)
D.dump = _redirect  # RAW_BANK_REPLAY.json / RAW_INPUT_MANIFEST_V3 comparisons stay read-only for Codex's folder

BANKS = ['B13', 'B16', 'B20', 'B23', 'B32', 'B35', 'B51', 'B54']
VARIANTS = ['groups_answer', 'channels_cell']
RECIPES = ['HEM_MOM', 'MEAN_MOM', 'HEM_EQ', 'MEAN_EQ', 'DS_FILTER_EQ', 'ALL_EQ']
RULES = ['ANSWER_Z_Q80', 'R2_COUNT', 'ENTROPY_GATE_Q80', 'TOP_1', 'TOP_2', 'TOP_3']
manifest = json.loads((RUNS['level_reduction_v1'] / 'INPUT_MANIFEST.json').read_text(encoding='utf8'))
pop = CC.Population(); cells = sorted(set(pop.cells))
rows = []; per = {}; t0 = time.time(); pool = entropy = None
for bk in BANKS:
    data = HistoricalData(manifest, pop, banks=[bk])
    if pool is None:
        pool = D.raw_pool(data); entropy = D.entropy_feature(data)
    for ce in cells:
        for variant in VARIANTS:
            state = input_state(data, 'estimator_input_v1', bk, ce, variant); bank = state['bank']
            raw = D.raw_variant(data, bank, bk, pool); ans = bank['answers']
            for recipe in RECIPES:
                score = state['scores'][recipe]; rr = raw if recipe == 'ALL_EQ' else raw[:, state['surv']]
                flags, detectors, dd = D.rules(score, bank, rr, entropy[ans], pop.uids[ans])
                for k in (1, 2, 3): flags['TOP_' + str(k)] = top_flags(score, bank['off'], np.full(len(ans), k))
                for rule in RULES:
                    m, det = evaluate(pop, bank, score, flags[rule], detectors.get(rule))
                    rows.append({'bank': bk, 'cell': ce, 'variant': variant, 'recipe': recipe, 'decision_rule': rule, **m,
                                 'q80': dd['q80'], 'entropy_closed_fraction': dd['entropy_closed_fraction']})
                    per[(bk, ce, variant, recipe, rule)] = {k: det[k] for k in ['confusion', 'first', 'argmax', 'flag_count', 'erroneous']}
            print(f'{bk} {ce} {variant} {time.time() - t0:.0f}s', flush=True)
    del data
R = pd.DataFrame(rows); R.to_csv(OUT / 'DECISION_METRICS.csv', index=False)

# ---- gate 1: content q80 rows vs sealed estimator-input METRICS ----
fields = ['within_answer_auc', 'prmscore', 'pb_first_error_accuracy', 'pb_official_f1', 'clean_flag_rate', 'wrong_answer_auroc']
sealed = pd.read_csv(RUNS['estimator_input_v1'] / 'METRICS.csv')
sealed = sealed[(sealed.position_variant == 'CONTENT') & (sealed.decision_rule == 'ANSWER_Z_Q80')]
g1 = R[R.decision_rule == 'ANSWER_Z_Q80'].merge(sealed, on=['bank', 'cell', 'variant', 'recipe'], suffixes=('', '_sealed'))
d1 = max(float(np.nanmax(np.abs(g1[f] - g1[f + '_sealed']))) if g1[f].notna().any() else 0. for f in fields)
# ---- gate 2: groups_answer rows of every rule vs Codex's sealed level layers, variant original ----
lay = []
for p in sorted((RUNS['level_reduction_v1'] / 'layers').glob('*__original.complete.json')):
    stem = p.name.removesuffix('.complete.json'); t = pd.read_csv(p.parent / (stem + '.metrics.csv.gz'))
    lay.append(t[(t.position_variant == 'CONTENT') & t.recipe.isin(RECIPES)])
lay = pd.concat(lay); lay['variant'] = 'groups_answer'
g2 = R[R.variant == 'groups_answer'].merge(lay, on=['bank', 'cell', 'variant', 'recipe', 'decision_rule'], suffixes=('', '_layer'))
d2 = max(float(np.nanmax(np.abs(g2[f] - g2[f + '_layer']))) if g2[f].notna().any() else 0. for f in fields)
gate = {'content_rows_compared': int(len(g1)), 'content_max_diff': d1, 'layer_rows_compared': int(len(g2)), 'layer_max_diff': d2,
        'status': 'PASS' if len(g1) == 2 * 8 * 9 * 6 and d1 < 1e-12 and len(g2) > 0 and d2 < 1e-12 else 'FAIL'}
print('GATE', json.dumps(gate), flush=True)

# ---- table: equal mean over banks (PRMB) and over banks x 8 PB cells ----
prm = R[R.cell == 'prmbench_qwen3_8b']; pb = R[R.cell != 'prmbench_qwen3_8b']
tab = prm.groupby(['variant', 'recipe', 'decision_rule'])[['within_answer_auc', 'prmscore', 'clean_flag_rate', 'wrong_answer_auroc']].mean().add_prefix('PRMB_')
tab = tab.join(pb.groupby(['variant', 'recipe', 'decision_rule'])[['pb_first_error_accuracy', 'pb_decision_error_accuracy', 'pb_official_f1', 'clean_flag_rate', 'wrong_answer_auroc']].mean().add_prefix('PB_'))
tab.to_csv(OUT / 'DECISION_TABLE.csv')

# ---- paired source-question bootstrap: HEM_MOM channels_cell minus groups_answer, per rule ----
def prm_score(c):
    tp, fp, tn, fn = c
    return .5 * (2 * tp / (2 * tp + fp + fn) + 2 * tn / (2 * tn + fp + fn))
rng = np.random.default_rng(20261004); draws = 5000; contrasts = {}
prm_ans = np.flatnonzero(pop.cells == 'prmbench_qwen3_8b'); sg, sgi = np.unique(pop.groups[prm_ans], return_inverse=True)
Wm = rng.multinomial(len(sg), np.full(len(sg), 1 / len(sg)), size=draws).astype(float)
pb_cells = [c for c in cells if c != 'prmbench_qwen3_8b']
pb_ans = {c: np.flatnonzero(pop.cells == c) for c in pb_cells}
allpb = np.concatenate([pb_ans[c] for c in pb_cells]); pg, pgi_all = np.unique(pop.groups[allpb], return_inverse=True)
off = 0; pgi = {}
for c in pb_cells: pgi[c] = pgi_all[off:off + len(pb_ans[c])]; off += len(pb_ans[c])
Wp = rng.multinomial(len(pg), np.full(len(pg), 1 / len(pg)), size=draws).astype(float)
for recipe in ['HEM_MOM', 'MEAN_MOM']:
    for rule in RULES:
        res = {}
        # PRMScore: per bank, confusion summed per source group
        diffs = np.zeros(draws); point = 0.
        for bk in BANKS:
            vals = []
            for v in VARIANTS:
                c = per[(bk, 'prmbench_qwen3_8b', v, recipe, rule)]['confusion']
                G = np.zeros((len(sg), 4)); np.add.at(G, sgi, c); vals.append((G, prm_score(G.sum(0))))
            point += (vals[1][1] - vals[0][1]) / 8
            b1 = Wm @ vals[1][0]; b0 = Wm @ vals[0][0]
            diffs += (np.array([prm_score(x) for x in b1]) - np.array([prm_score(x) for x in b0])) / 8
        res['PRMB_prmscore'] = {'delta': point, 'lo95': float(np.quantile(diffs, .025)), 'hi95': float(np.quantile(diffs, .975))}
        # PB: official F1 and clean flag rate, equal over banks x cells
        f1d = np.zeros(draws); cfd = np.zeros(draws); f1p = cfp = 0.
        for bk in BANKS:
            for c in pb_cells:
                tgt = pop.target[pb_ans[c]]
                stats = []
                for v in VARIANTS:
                    d = per[(bk, c, v, recipe, rule)]; err = d['erroneous']; hit = d['first'] == tgt
                    S = np.zeros((len(pg), 5))
                    np.add.at(S, pgi[c], np.column_stack([hit & err, err, hit & ~err, ~err, (d['flag_count'] > 0) & ~err]).astype(float))
                    stats.append(S)
                def f1cf(S):
                    ae = S[..., 0] / S[..., 1]; ac = S[..., 2] / S[..., 3]
                    return np.where(ae + ac > 0, 2 * ae * ac / np.where(ae + ac > 0, ae + ac, 1), 0.), S[..., 4] / S[..., 3]
                (a1, c1), (a0, c0) = f1cf(stats[1].sum(0)), f1cf(stats[0].sum(0)); f1p += (a1 - a0) / 64; cfp += (c1 - c0) / 64
                (b1f, b1c), (b0f, b0c) = f1cf(Wp @ stats[1]), f1cf(Wp @ stats[0]); f1d += (b1f - b0f) / 64; cfd += (b1c - b0c) / 64
        res['PB_official_f1'] = {'delta': float(f1p), 'lo95': float(np.nanquantile(f1d, .025)), 'hi95': float(np.nanquantile(f1d, .975))}
        res['PB_clean_flag_rate'] = {'delta': float(cfp), 'lo95': float(np.nanquantile(cfd, .025)), 'hi95': float(np.nanquantile(cfd, .975))}
        contrasts[f'{recipe} {rule}'] = res
summary = {'time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'gate': gate, 'bootstrap_draws': draws,
           'status': 'REVIEW_PRECOMPUTE; CONTENT position only; no nulls; Omri chooses the rule', 'contrasts_channels_cell_minus_groups_answer': contrasts}
(OUT / 'SUMMARY.json').write_text(json.dumps(summary, indent=1, default=float), encoding='utf8')
pd.set_option('display.width', 250)
print(tab.round(4).to_string())
print(json.dumps(contrasts, indent=1, default=lambda x: round(float(x), 5)))
