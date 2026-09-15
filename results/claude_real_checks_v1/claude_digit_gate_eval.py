"""Step 396 (Claude): digit disagreement as answer-gate evidence. Protocol: docs/experiments/DIGIT_GATE_EVIDENCE_20260916.md.

Locators frozen (innovation5 primary; innovation5+digit.25 secondary). Every detector uses the same transductive
within-cell midrank >= .33 rule. Labels only in evaluation. Development data.
"""
import sys, json, pathlib, time
import numpy as np
ROOT = pathlib.Path(r'C:\Users\omris\TAU\hallucination_detection\.worktrees\temporal-research-20260915'); sys.path.insert(0, str(ROOT))
SRC = pathlib.Path(r'C:\Users\omris\TAU\hallucination_detection'); OUT = ROOT / 'results/claude_real_checks_v1'
from scripts import run_temporal_research_baseline as base
from scripts import run_direct_probability_temporal as ev
from spectral_utils.math_gate_selection import percentile_by_cell
from spectral_utils.temporal_research_features import BASELINE
Q = .33

def main():
    t0 = time.perf_counter()
    records, joined = base.load_contract(SRC); target = joined['target']
    cells = np.array([r['cell'] for r in records]); pb = np.char.startswith(cells, 'pb_')
    digit = set(json.load(open(OUT / 'digit_token_ids.json')))
    with np.load(ROOT / 'results/temporal_research_baseline_v1/SCORES_FROZEN.npz') as f:
        tail15 = f['gate_raw']; frozen_pct = f['gate_percentile']; base_scores = f['steps__append_innovation__H0lim']
    with np.load(OUT / 'DIGIT_DISAGREE_SCORES.npz') as f: digit025 = f['steps__innovation5_plus_digit_g0.25']
    count = np.zeros(len(records)); ndig = np.zeros(len(records)); seen = np.zeros(len(records), bool)
    for cell, path, kind, dataset in ev.source_specs():
        if kind != 'pb': continue  # gate is a PB-only decision; PRMB rows are unaffected
        idx = [i for i, r in enumerate(records) if r['cell'] == cell]
        rows = ev.old._source_row_map(ev.old.load_pickle(path), kind=kind, dataset=dataset)
        for i in idx:
            row = rows[records[i]['row_id']]; ids = np.asarray(ev.old._topk_payload(row)['ids']); gen = np.asarray(row['gen_token_ids'])
            isdig = np.isin(gen, list(digit)); count[i] = float((isdig & np.isin(ids[:, 0], list(digit)) & (ids[:, 0] != gen)).sum()); ndig[i] = float(isdig.sum()); seen[i] = True
        del rows
    assert seen[pb].all()
    rate = count / np.maximum(ndig, 1)
    def midrank(x):
        p = np.full(len(records), np.nan); p[pb] = percentile_by_cell(x[pb], cells[pb]); return p
    r_tail = midrank(tail15)
    np.testing.assert_allclose(r_tail[pb], frozen_pct[pb], atol=1e-12)  # replay of the frozen gate percentiles
    detectors = {'tail15': r_tail, 'digit_count': midrank(count), 'digit_rate': midrank(rate), 'digit_presence': midrank(ndig)}
    detectors['equal_rank_tail15_digit_rate'] = midrank(np.where(pb, (r_tail + detectors['digit_rate']) / 2, np.nan))
    detectors['equal_rank_tail15_digit_count'] = midrank(np.where(pb, (r_tail + detectors['digit_count']) / 2, np.nan))
    gates = {k: (v >= Q) for k, v in detectors.items()}
    locators = {'innovation5': base_scores, 'digit025': digit025}
    metrics, per = {}, {}
    for lname, scores in locators.items():
        for gname, opened in gates.items():
            m, p = base.evaluator.evaluate_arrays(records, joined, {f'{lname}@{gname}': scores}, fold_auc=(gname == 'tail15'), pb_gate_open=opened)
            metrics.update(m); per.update(p)
    np.testing.assert_allclose(metrics['innovation5@tail15']['pb_all8'], 0.39831353198166894, atol=1e-12)
    np.testing.assert_allclose(metrics['digit025@tail15']['pb_all8'], 0.41329967, atol=1e-7)
    primary = [('innovation5@equal_rank_tail15_digit_rate', 'innovation5@tail15'), ('innovation5@digit_rate', 'innovation5@tail15')]
    pairs = primary + [(f'innovation5@{g}', 'innovation5@tail15') for g in gates if g not in ('tail15', 'digit_rate', 'equal_rank_tail15_digit_rate')]
    pairs += [(f'digit025@{g}', 'digit025@tail15') for g in gates if g != 'tail15']
    for n in per: per[n]['within'] = per['innovation5@tail15']['within'] if n.startswith('innovation5') else per['digit025@tail15']['within']
    contrasts = base.evaluator.paired_bootstrap(records, joined, per, draws=10000, pairs=pairs, primary_pairs=set(primary), primary_ci=.975)
    for a, b in pairs: contrasts[a + '_minus_' + b]['pb_delta'] = metrics[a]['pb_all8'] - metrics[b]['pb_all8']
    err = pb & (target >= 0); clean = pb & (target == -1)
    def acct(n):
        p = per[n]; hit = p['decision_valid'] & (p['prediction'] == target); cs = p['decision_valid'] & (p['prediction'] == -1)
        return dict(error_hits=int((err & hit).sum()), clean_successes=int((clean & cs).sum()))
    table = {}
    for n, m in metrics.items():
        lname, gname = n.split('@'); ref = per[f'{lname}@tail15']; p = per[n]
        hit = p['decision_valid'] & (p['prediction'] == target); rhit = ref['decision_valid'] & (ref['prediction'] == target)
        cs = p['decision_valid'] & (p['prediction'] == -1); rcs = ref['decision_valid'] & (ref['prediction'] == -1)
        table[n] = dict(pb_all8=m['pb_all8'], pb_q4=m['pb_q4'], pb_q8=m['pb_q8'], clean_accuracy=m['pb_clean_accuracy'], error_exact=m['pb_error_exact_accuracy'],
                        gate_open=int(gates[gname][pb].sum()), error_hits=int((err & hit).sum()), clean_successes=int((clean & cs).sum()),
                        error_hits_gained=int((err & hit & ~rhit).sum()), error_hits_lost=int((err & ~hit & rhit).sum()),
                        clean_gained=int((clean & cs & ~rcs).sum()), clean_lost=int((clean & ~cs & rcs).sum()),
                        correct_peaks_suppressed=m['pb_correct_peaks_suppressed'])
    # label-using descriptive: detector AUCs for erroneous-vs-clean per cell
    def auc(x, y):
        pos, neg = x[y], x[~y]; return float(((pos[:, None] > neg).sum() + .5 * (pos[:, None] == neg).sum()) / (len(pos) * len(neg)))
    aucs = {c: {k: auc(v[cells == c], target[cells == c] >= 0) for k, v in detectors.items()} for c in sorted(set(cells[pb]))}
    out = dict(protocol='docs/experiments/DIGIT_GATE_EVIDENCE_20260916.md', q=Q, table=table, contrasts=contrasts, detector_auc_by_cell=aucs, seconds=time.perf_counter() - t0)
    (OUT / 'DIGIT_GATE_EVAL.json').write_text(json.dumps(out, indent=1, default=float), encoding='utf8')
    np.savez_compressed(OUT / 'DIGIT_GATE_DETECTORS.npz', tail15=tail15, digit_count=count, digit_rate=rate, digit_presence=ndig)
    print('| locator@gate | PB all-8 | q4 | q8 | clean acc | error exact | open | err hits (+/-) | clean succ (+/-) | suppressed |'); print('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for n, v in table.items():
        print(f"| {n} | {100*v['pb_all8']:.4f} | {100*v['pb_q4']:.2f} | {100*v['pb_q8']:.2f} | {100*v['clean_accuracy']:.2f} | {100*v['error_exact']:.2f} | {v['gate_open']} | {v['error_hits']} (+{v['error_hits_gained']}/-{v['error_hits_lost']}) | {v['clean_successes']} (+{v['clean_gained']}/-{v['clean_lost']}) | {v['correct_peaks_suppressed']} |")
    print('\ncontrasts (PB)'); [print(k, '%+.4f pp' % (100 * v['pb_delta']), [round(100 * x, 4) for x in v['pb_ci']], v['ci_level']) for k, v in contrasts.items()]
    print('\ndetector AUC erroneous-vs-clean, mean over cells:', {k: round(float(np.mean([aucs[c][k] for c in aucs])), 4) for k in detectors})

if __name__ == '__main__': main()
