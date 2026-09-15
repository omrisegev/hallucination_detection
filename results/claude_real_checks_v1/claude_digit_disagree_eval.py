"""Full-benchmark evaluation of one cheap new view: digit disagreement between the provided token and the scorer's top-1.

Token stream d_t = 1 if the provided token is a single digit AND the scorer's top-1 token is a different digit, else 0.
Declared before running: step auxiliary = per-step Top10 mean of d_t (same readout as every stream);
primary candidate = innovation5 base + .25*std(base)*z(aux) (the project's residual readout, gamma=.25),
secondary gamma=1; standalone aux is a diagnostic. Frozen tail15 gate, frozen Top10, same evaluator, all 13,769 answers.
Label-free construction; labels enter only in evaluation. Development data.
"""
import sys, io, json, sqlite3, pathlib, time
import numpy as np
ROOT = pathlib.Path(r'C:\Users\omris\TAU\hallucination_detection\.worktrees\temporal-research-20260915'); sys.path.insert(0, str(ROOT))
SRC = pathlib.Path(r'C:\Users\omris\TAU\hallucination_detection'); OUT = ROOT / 'results/claude_real_checks_v1'
from scripts import run_temporal_research_baseline as base
from scripts import run_direct_probability_temporal as ev
from spectral_utils.direct_probability_fusion import step_top_mean
from spectral_utils.temporal_context_models import residual_step_score
from spectral_utils.temporal_research_features import BASELINE

def main():
    t0 = time.perf_counter()
    records, joined = base.load_contract(SRC); off = joined['offsets']; total = int(off[-1]); target = joined['target']
    digit = set(json.load(open(OUT / 'digit_token_ids.json')))
    with np.load(ROOT / 'results/temporal_research_baseline_v1/SCORES_FROZEN.npz') as f:
        gate = f['gate_percentile'] >= .33; base_scores = f['steps__append_innovation__H0lim']; orig4 = f['steps__' + BASELINE]
    aux = np.full(total, np.nan); count = np.zeros(len(records)); digits = np.zeros(len(records)); covered = np.zeros(len(records), bool)
    for cell, path, kind, dataset in ev.source_specs():
        idx = [i for i, r in enumerate(records) if r['cell'] == cell]
        print('[load]', cell, len(idx), flush=True)
        rows = ev.old._source_row_map(ev.old.load_pickle(path), kind=kind, dataset=dataset)
        for i in idx:
            row = rows[records[i]['row_id']]
            if 'gen_token_ids' not in row or row['gen_token_ids'] is None: continue
            ids = np.asarray(ev.old._topk_payload(row)['ids']); gen = np.asarray(row['gen_token_ids'])
            spans = np.asarray(row['step_token_spans'], dtype=int)
            isdig = np.isin(gen, list(digit)); d = (isdig & np.isin(ids[:, 0], list(digit)) & (ids[:, 0] != gen)).astype(float)
            aux[off[i]:off[i + 1]] = step_top_mean(d, spans[:, 0], spans[:, 1], 10)
            count[i] = d.sum(); digits[i] = isdig.sum(); covered[i] = True
        del rows
    print('covered answers', int(covered.sum()), 'of', len(records), 'seconds', round(time.perf_counter() - t0, 1), flush=True)
    if not covered.all(): raise ValueError('incomplete coverage; refusing to evaluate a partial roster')
    scores = {'digit_aux_standalone': aux.copy()}
    for g in (.25, 1.):
        s = np.empty(total)
        for i in range(len(records)):
            sl = slice(off[i], off[i + 1]); s[sl] = residual_step_score(base_scores[sl], aux[sl], g)
        scores[f'innovation5_plus_digit_g{g:g}'] = s
    refs = {'innovation5': base_scores, 'original4': orig4}
    metrics, per = base.evaluator.evaluate_arrays(records, joined, scores, fold_auc=True, pb_gate_open=gate)
    rm, rp = base.evaluator.evaluate_arrays(records, joined, refs, fold_auc=True, pb_gate_open=gate); metrics.update(rm); per.update(rp)
    primary = [('innovation5_plus_digit_g0.25', 'innovation5')]
    pairs = primary + [('innovation5_plus_digit_g1', 'innovation5'), ('digit_aux_standalone', 'innovation5')]
    contrasts = base.evaluator.paired_bootstrap(records, joined, per, draws=10000, pairs=pairs, primary_pairs=set(primary), primary_ci=.975)
    for a, b in pairs: contrasts[a + '_minus_' + b]['pb_delta'] = metrics[a]['pb_all8'] - metrics[b]['pb_all8']
    # strata and hit accounting vs innovation5 on PB errors
    cells = np.array([r['cell'] for r in records]); pb = np.char.startswith(cells, 'pb_'); err = pb & (target >= 0); clean = pb & (target == -1)
    steps = np.array([r['steps'] for r in records]); rel = np.where(err, target / np.maximum(steps - 1, 1), np.nan)
    strata = {'early': err & (rel < 1 / 3), 'middle': err & (rel >= 1 / 3) & (rel < 2 / 3), 'late': err & (rel >= 2 / 3), 'all': err}
    hit = lambda n: per[n]['decision_valid'] & (per[n]['prediction'] == target)
    acct = {s: {n: dict(exact=int((m & hit(n)).sum()), gained=int((m & hit(n) & ~hit('innovation5')).sum()), lost=int((m & ~hit(n) & hit('innovation5')).sum()))
                for n in scores} for s, m in strata.items()}
    # gate-side descriptive diagnostic: does the answer-level disagreement count separate clean from erroneous PB answers?
    def auc(x, y):
        pos, neg = x[y], x[~y]; return float(((pos[:, None] > neg).sum() + .5 * (pos[:, None] == neg).sum()) / (len(pos) * len(neg)))
    gate_diag = {}
    for c in sorted(set(cells[pb])):
        m = cells == c; y = (target[m] >= 0)
        gate_diag[c] = dict(auc_count=auc(count[m], y), auc_rate=auc(count[m] / np.maximum(digits[m], 1), y), zero_count_clean=float(np.mean(count[m][~y] == 0)), zero_count_error=float(np.mean(count[m][y] == 0)))
    table = {k: dict(pb=metrics[k]['pb_all8'], within=metrics[k]['prm_within'], prmscore_default=metrics[k]['prmscore_q08']) for k in list(scores) + list(refs)}
    out = dict(table=table, contrasts=contrasts, strata=acct, gate_side_diagnostic=gate_diag, seconds=time.perf_counter() - t0,
               note='gamma=.25 declared primary before evaluation; PRMScore uses evaluator default calibration; development data.')
    (OUT / 'DIGIT_DISAGREE_EVAL.json').write_text(json.dumps(out, indent=1, default=float), encoding='utf8')
    np.savez_compressed(OUT / 'DIGIT_DISAGREE_SCORES.npz', **{'steps__' + k: v for k, v in scores.items()}, aux=aux)
    print('\n| method | PB % | within | PRMScore(default) |'); print('|---|---:|---:|---:|')
    for k, v in table.items(): print(f"| {k} | {100*v['pb']:.4f} | {v['within']:.6f} | {v['prmscore_default']:.6f} |")
    print('\ncontrasts'); [print(k, 'pb %+.4f' % (100 * v['pb_delta']), [round(100 * x, 4) for x in v['pb_ci']], 'within', [round(x, 6) for x in v['prm_within_ci']] if v.get('prm_within_ci') else None, v['ci_level']) for k, v in contrasts.items()]
    print('\nstrata (exact/gained/lost vs innovation5)'); [print(s, {n: acct[s][n] for n in scores}) for s in acct]
    print('\ngate-side diagnostic'); [print(c, v) for c, v in gate_diag.items()]

if __name__ == '__main__': main()
