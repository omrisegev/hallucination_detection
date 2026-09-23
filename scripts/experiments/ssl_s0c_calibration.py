"""S0-C: ranking vs calibration on the existing nested PRMBench scores of Step 432 (A1).

Per-answer monotone transforms (RAW / Z / ECDF) leave the within-answer ranking untouched;
only the between-answer scale changes. Thresholds are calibrated on inner-held scores
(jobs fitted without folds k and j) and frozen before touching fold k.  No fit, no inference.
Protocol: results/ssl_pseudolabel_residual_v1/S0C/PROTOCOL.json (frozen before this ran).
"""
from pathlib import Path
import json, pickle, sys, time
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
MAIN = Path(r'C:\Users\omris\TAU\hallucination_detection')
R = MAIN / '.worktrees/readout-quickest-detection-v1/results/step_evidence_v1'
OUT = ROOT / 'results/ssl_pseudolabel_residual_v1/S0C'; OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(MAIN))
from spectral_utils.prmbench import prmbench_evaluate

P = json.loads((OUT / 'PROTOCOL.json').read_text(encoding='utf8'))
METHODS = P['methods']; PRIMARY = P['primary_family']['method']
SEED = 20260923; DRAWS = 100_000
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(p, v): Path(p).write_text(json.dumps(v, indent=2, ensure_ascii=False, default=lambda x: x.item() if isinstance(x, np.generic) else x.tolist() if isinstance(x, np.ndarray) else str(x)), encoding='utf8')
def within_auc(y, s):
    y = np.asarray(y, bool); n1 = int(y.sum()); n0 = len(y) - n1
    if not n1 or not n0: return np.nan
    return float((rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
def T_raw(s): return np.asarray(s, float)
def T_z(s): s = np.asarray(s, float); return (s - s.mean()) / max(s.std(), 1e-8)
def T_ecdf(s): s = np.asarray(s, float); return (rankdata(s, method='average') - .5) / len(s)
TRANSFORMS = {'C_RAW': T_raw, 'C_Z': T_z, 'C_ECDF': T_ecdf}

t0 = time.perf_counter()
ans = pd.read_csv(R / 'OOF_ANSWERS.csv', encoding='utf-8-sig'); Z = np.load(R / 'OOF_STEP_SCORES.npz')
off = Z['offsets']; labels = Z['labels']; n = len(ans); nsteps = np.diff(off)
prm = ~ans.cell.str.startswith('pb_').to_numpy(); fold = ans.fold.to_numpy(); groups = ans.source_group.to_numpy(); ids = ans.id.to_numpy()
freeze = read(R / 'INPUT_FREEZE.json'); meta = {m['idx']: m for m in pickle.load(open(freeze['prm_metadata']['path'], 'rb')).values()}
prm_idx = np.flatnonzero(prm)
noncontrol = np.zeros(n, bool)
for i in prm_idx: noncontrol[i] = meta[ids[i]]['classification'] != 'correct'
step_answer = np.repeat(np.arange(n), nsteps)
elig_steps = np.repeat(noncontrol, nsteps)
prm_steps = np.repeat(prm, nsteps)
qgrid = np.round(np.linspace(.5, .99, 50), 2)

# ---------------------------------------------------------------- evaluator: vectorised total PRMScore, verified on all 114 saved decision sets
def counts(valid, y, elig):
    v = np.asarray(valid, bool)[..., elig]; good = (np.asarray(y)[elig] == 0)
    tp = (v & good).sum(-1); fp = (v & ~good).sum(-1); tn = (~v & ~good).sum(-1); fn = (~v & good).sum(-1)
    return tp, fp, tn, fn
def prmscore_from_counts(tp, fp, tn, fn):
    def ratio(a, b): return np.divide(a, b, out=np.full(np.shape(a), -1., float), where=b != 0)
    p = ratio(tp, tp + fp); r = ratio(tp, tp + fn); f = ratio(2 * p * r, p + r)
    p = ratio(tn, tn + fn); r = ratio(tn, tn + fp); nf = ratio(2 * p * r, p + r)
    return (f + nf) / 2
def total_prmscore(valid, y, elig): return float(prmscore_from_counts(*counts(valid, y, elig)))
def official(valid):
    preds = [{'idx': ids[i], 'labels': valid[off[i]:off[i+1]].astype(int).tolist()} for i in prm_idx]
    r = prmbench_evaluate(preds, [meta[ids[i]] for i in prm_idx]); t = r['total']
    return {'prmscore': .5 * (t['f1'] + t['negative_f1']), 'f1': t['f1'], 'negative_f1': t['negative_f1'], 'correct_step_acc': t['correct_step_acc'], 'wrong_step_acc': t['wrong_step_acc']}
dec = np.load(R / 'PRM_OOF_DECISIONS.npz'); saved = read(R / 'PRMSCORE.json')
fixture = []
for key in dec.files:
    m, kind = key.rsplit('__', 1); want = saved[m]['quantile_0.8' if kind == 'q80' else 'inner_selected']['prmscore']
    got = total_prmscore(dec[key][prm_steps], labels[prm_steps], elig_steps[prm_steps]); fixture.append(abs(got - want))
fixture_max = max(fixture)
assert fixture_max < 1e-12, f'vectorised PRMScore disagrees with official evaluator: {fixture_max}'
print(f'evaluator fixture: {len(fixture)} saved decision sets replay, max error {fixture_max:.1e}', flush=True)

# ---------------------------------------------------------------- calibration + evaluation
results = {}; choices = []; hygiene = []; identity = {}; decisions = {}
for m in METHODS:
    enc = m.split('__')[2]; arm = f'{enc}__equal'; raw_scores = Z[m]
    for tname, T in TRANSFORMS.items():
        valid_sel = np.zeros(int(off[-1]), bool); valid_q80 = valid_sel.copy(); tscore = np.full(int(off[-1]), np.nan)
        for k in range(5):
            cal_s, cal_y, cal_e = [], [], []
            for j in range(5):
                if j == k: continue
                stem = f'prm__fold{k}__evidence__all__spectral__inner{j}'
                zi = np.load(R / 'jobs' / f'{stem}.npz'); info = read(R / 'jobs' / f'{stem}.json'); idx = zi['indices']; soff = zi['step_offsets']; s = zi[arm + '__scores']
                held_ok = bool((fold[idx] == j).all() and prm[idx].all())
                train_groups = set(info['train_source_groups']); leak = len(train_groups & set(groups[prm & np.isin(fold, [k, j])]))
                hygiene.append({'method': m, 'transform': tname, 'outer': k, 'inner': j, 'held_answers': len(idx), 'held_all_in_fold_j': held_ok, 'train_groups_overlapping_folds_k_or_j': leak, 'test_groups_equal_fold_j': set(info['test_source_groups']) == set(groups[idx])})
                assert held_ok and leak == 0
                for ii, i in enumerate(idx):
                    a, b = off[i:i+2]; assert b - a == soff[ii+1] - soff[ii]
                    cal_s.append(T(s[soff[ii]:soff[ii+1]])); cal_y.append(labels[a:b]); cal_e.append(np.full(b - a, noncontrol[i]))
            cal_s = np.concatenate(cal_s); cal_y = np.concatenate(cal_y); cal_e = np.concatenate(cal_e)
            thresholds = np.quantile(cal_s, qgrid); tau80 = float(np.quantile(cal_s, .8))
            grid = prmscore_from_counts(*counts(cal_s[None, :] < thresholds[:, None], cal_y, cal_e)); best = int(np.argmax(grid))
            test = np.flatnonzero(prm & (fold == k))
            for i in test:
                a, b = off[i:i+2]; t = T(raw_scores[a:b]); tscore[a:b] = t
                valid_sel[a:b] = t < thresholds[best]; valid_q80[a:b] = t < tau80
            choices.append({'method': m, 'transform': tname, 'outer_fold': k, 'calibration_steps': len(cal_s), 'selected_quantile': float(qgrid[best]), 'selected_threshold': float(thresholds[best]),
                            'q80_threshold': tau80, 'calibration_prmscore_at_selected': float(grid[best]), 'calibration_prmscore_at_q80': float(grid[np.argmin(np.abs(qgrid - .8))])})
        # identity check
        auc_raw = np.array([within_auc(labels[off[i]:off[i+1]], raw_scores[off[i]:off[i+1]]) for i in prm_idx])
        auc_t = np.array([within_auc(labels[off[i]:off[i+1]], tscore[off[i]:off[i+1]]) for i in prm_idx])
        identity[f'{m}::{tname}'] = float(np.nanmax(np.abs(auc_raw - auc_t)))
        assert identity[f'{m}::{tname}'] < 1e-12, 'within-answer ranking changed: transform/alignment bug'
        row = {'method': m, 'transform': tname, 'within_auc': float(np.nanmean(auc_t)),
               'pooled_step_auroc': float(roc_auc_score(labels[prm_steps], tscore[prm_steps])),
               'answer_mean_sd_between_answers': float(np.std([tscore[off[i]:off[i+1]].mean() for i in prm_idx])),
               'answer_sd_sd_between_answers': float(np.std([tscore[off[i]:off[i+1]].std() for i in prm_idx]))}
        for kind, valid in [('inner', valid_sel), ('q80', valid_q80)]:
            o = official(valid); vec = total_prmscore(valid[prm_steps], labels[prm_steps], elig_steps[prm_steps]); assert abs(o['prmscore'] - vec) < 1e-12
            row[f'prmscore_{kind}'] = o['prmscore']; row[f'f1_{kind}'] = o['f1']; row[f'negative_f1_{kind}'] = o['negative_f1']
            row[f'correct_step_acc_{kind}'] = o['correct_step_acc']; row[f'wrong_step_acc_{kind}'] = o['wrong_step_acc']
            ctrl = prm_steps & ~elig_steps
            row[f'clean_control_false_alarm_{kind}'] = float((~valid[ctrl]).mean())         # control answers: fraction of steps flagged invalid
            row[f'flagged_fraction_eligible_{kind}'] = float((~valid[prm_steps & elig_steps]).mean())
            decisions[f'{m}__{tname}__{kind}'] = valid
        row['original_run_prmscore_inner'] = saved[m]['inner_selected']['prmscore']; row['original_run_prmscore_q80'] = saved[m]['quantile_0.8']['prmscore']
        results[f'{m}::{tname}'] = row
        print(f'{m} {tname}: PRMScore inner {row["prmscore_inner"]:.6f} q80 {row["prmscore_q80"]:.6f} within {row["within_auc"]:.6f}  ({time.perf_counter()-t0:.0f}s)', flush=True)
res = pd.DataFrame(results.values()); res.to_csv(OUT / 'S0C_CALIBRATION.csv', index=False)
pd.DataFrame(choices).to_csv(OUT / 'THRESHOLDS_PER_FOLD.csv', index=False); pd.DataFrame(hygiene).to_csv(OUT / 'CALIBRATION_HYGIENE.csv', index=False)
dump(OUT / 'WITHIN_RANK_IDENTITY.json', {'max_abs_within_auc_change': identity, 'tolerance': 1e-12, 'evaluator_fixture_max_error': fixture_max, 'fixture_sets': len(fixture)})
np.savez_compressed(OUT / 'DECISIONS.npz', **decisions)
ct7_row = {'method': 'ct7', 'transform': 'original_run', 'prmscore_inner': saved['ct7']['inner_selected']['prmscore'], 'prmscore_q80': saved['ct7']['quantile_0.8']['prmscore']}

# ---------------------------------------------------------------- paired source-group bootstrap of PRMScore (pooled counts per group)
g_prm = groups[prm_idx]; G, ginv = np.unique(g_prm, return_inverse=True)
def group_counts(valid):
    tp = np.zeros(len(G)); fp = tp.copy(); tn = tp.copy(); fn = tp.copy()
    for gi, i in zip(ginv, prm_idx):
        if not noncontrol[i]: continue
        a, b = off[i:i+2]; v = valid[a:b]; good = labels[a:b] == 0
        tp[gi] += (v & good).sum(); fp[gi] += (v & ~good).sum(); tn[gi] += (~v & ~good).sum(); fn[gi] += (~v & good).sum()
    return np.stack([tp, fp, tn, fn], 1)
def paired_bootstrap(cA, cB, draws=DRAWS, seed=SEED, chunk=5000):
    rng = np.random.default_rng(seed); deltas = np.empty(draws); pos = 0
    while pos < draws:
        w = rng.multinomial(len(G), np.full(len(G), 1 / len(G)), size=min(chunk, draws - pos)).astype(float)
        sA = prmscore_from_counts(*(w @ cA).T); sB = prmscore_from_counts(*(w @ cB).T)
        deltas[pos:pos + len(w)] = sA - sB; pos += len(w)
    return deltas
boot = []; primary = []
for m in METHODS:
    for kind in ['inner', 'q80']:
        cRAW = group_counts(decisions[f'{m}__C_RAW__{kind}'])
        for tname in ['C_Z', 'C_ECDF']:
            cT = group_counts(decisions[f'{m}__{tname}__{kind}']); d = paired_bootstrap(cT, cRAW)
            point = results[f'{m}::{tname}'][f'prmscore_{kind}'] - results[f'{m}::C_RAW'][f'prmscore_{kind}']
            row = {'method': m, 'kind': kind, 'contrast': f'{tname} - C_RAW', 'delta_prmscore': point, 'ci95': [float(np.quantile(d, .025)), float(np.quantile(d, .975))],
                   'ci_bonferroni_97.5': [float(np.quantile(d, .0125)), float(np.quantile(d, .9875))], 'p_two_sided': float(2 * min((d <= 0).mean(), (d >= 0).mean())),
                   'draws': DRAWS, 'primary': (m == PRIMARY and kind == 'inner'), 'unit': 'source_group', 'groups': len(G)}
            boot.append(row)
    # C_RAW recalibrated vs the original run's threshold selection: same scores, different calibration pool
    orig = dec[m + '__inner']; d = paired_bootstrap(group_counts(decisions[f'{m}__C_RAW__inner']), group_counts(orig))
    boot.append({'method': m, 'kind': 'inner', 'contrast': 'C_RAW (recalibrated) - original run', 'delta_prmscore': results[f'{m}::C_RAW']['prmscore_inner'] - saved[m]['inner_selected']['prmscore'],
                 'ci95': [float(np.quantile(d, .025)), float(np.quantile(d, .975))], 'ci_bonferroni_97.5': None, 'p_two_sided': float(2 * min((d <= 0).mean(), (d >= 0).mean())), 'draws': DRAWS, 'primary': False, 'unit': 'source_group', 'groups': len(G)})
# best transform of the primary method vs CT7 original (secondary, descriptive)
cCT7 = group_counts(dec['ct7__inner'])
for tname in ['C_RAW', 'C_Z', 'C_ECDF']:
    d = paired_bootstrap(group_counts(decisions[f'{PRIMARY}__{tname}__inner']), cCT7)
    boot.append({'method': PRIMARY, 'kind': 'inner', 'contrast': f'{tname} - ct7 (original)', 'delta_prmscore': results[f'{PRIMARY}::{tname}']['prmscore_inner'] - saved['ct7']['inner_selected']['prmscore'],
                 'ci95': [float(np.quantile(d, .025)), float(np.quantile(d, .975))], 'ci_bonferroni_97.5': None, 'p_two_sided': float(2 * min((d <= 0).mean(), (d >= 0).mean())), 'draws': DRAWS, 'primary': False, 'unit': 'source_group', 'groups': len(G)})
dump(OUT / 'BOOTSTRAP.json', {'seed': SEED, 'draws': DRAWS, 'unit': 'source_group', 'groups': int(len(G)), 'contrasts': boot, 'ct7_reference': ct7_row, 'seconds': time.perf_counter() - t0})
print(pd.DataFrame(boot)[['method', 'kind', 'contrast', 'delta_prmscore', 'ci95', 'p_two_sided', 'primary']].to_string())
print(f'done in {time.perf_counter()-t0:.0f}s')
