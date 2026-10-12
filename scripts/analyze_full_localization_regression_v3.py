"""Explain the single-answer pilot/full gap from reviewed, frozen outputs."""
import hashlib
import html
import importlib.util
import json
from pathlib import Path
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/localization_full_benchmark_v3/evaluation'
PILOT = ROOT / 'results/fusion_entropy_sampling_v1/EVALUATION.json'
OUT = ROOT / 'results/localization_full_regression_audit_v3'
CORE = ROOT / 'spectral_utils/fusion_regression_audit.py'
spec = importlib.util.spec_from_file_location('fusion_regression_audit', CORE)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


def load(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [CORE, Path(__file__).resolve(), PILOT] + [SOURCE / n for n in
        ('JOINED.json', 'JOINED.npz', 'METRICS.json', 'REVIEW.json', 'MANIFEST.json')]
    hashes = {str(p): sha(p) for p in paths}
    join = load(SOURCE / 'JOINED.json')
    assert join['arrays_sha256'] == sha(SOURCE / 'JOINED.npz')
    assert load(SOURCE / 'REVIEW.json')['status'] == 'PASS'
    assert load(SOURCE / 'MANIFEST.json')['hashes'][str(PILOT)] == sha(PILOT)
    records, arms = join['records'], join['arms']
    with np.load(SOURCE / 'JOINED.npz', allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    pilot = load(PILOT)
    previous = {r['uid']: r for r in pilot['rows']}
    pilot_mask = np.asarray([r['reuse_original110'] for r in records], bool)
    assert len(records) == 13769 and pilot_mask.sum() == 110
    checks = 0
    for i in np.flatnonzero(pilot_mask):
        r = records[i]; old = previous[r['uid']]
        for k in ('row_id', 'group_id', 'cell', 'tokens', 'steps'):
            assert r[k] == old[k]
        lo, hi = arrays['offsets'][i:i+2]
        if r['cell'].startswith('prm'):
            np.testing.assert_array_equal(arrays['labels'][lo:hi], old['target'])
        else:
            assert arrays['target'][i] == old['target']
        for j, arm in enumerate(arms):
            assert arrays['valid'][i, j] == old['valid'][arm]
            assert arrays['decision'][i, j] == old['decision_valid'][arm]
            if old['valid'][arm]:
                np.testing.assert_array_equal(arrays['scores'][lo:hi, j], old['scores'][arm])
                assert arrays['peaks'][i, j] == old['peaks'][arm]
            if old['decision_valid'][arm]:
                assert arrays['predictions'][i, j] == old['predictions'][arm]
            checks += 1
    masks = dict(pilot=pilot_mask, additional=~pilot_mask, full=np.ones(len(records), bool))
    bundles = {arm: {name: core.pb_panel(records, arrays, j, mask) for name, mask in masks.items()}
               for j, arm in enumerate(arms)}
    full = load(SOURCE / 'METRICS.json')['metrics']
    for arm, b in bundles.items():
        assert abs(b['pilot']['macro_f1'] - pilot['metrics'][arm]['pb']['macro_f1']) < 1e-14
        for cell, entry in b['full']['cells'].items():
            for key in ('answers', 'clean', 'erroneous', 'valid_decisions', 'clean_accuracy', 'error_exact_accuracy', 'f1'):
                assert abs(entry[key] - full[arm]['pb']['cells'][cell][key]) < 1e-14
    # Hand-constructed invalid/gate cases verify denominator and accounting rules.
    toy = core.outcome_summary([-1,-1,0,0,0,0], [-1,-2,0,-1,-2,1], [-2,-2,0,0,0,1],
                               [1,0,1,1,1,1], [1,0,1,1,0,1])
    assert toy['clean_accuracy'] == .5 and toy['error_exact_accuracy'] == .25
    assert toy['peak_correct'] == 3 and toy['correct_peak_suppressed'] == 1
    assert toy['correct_peak_invalid_decision'] == 1 and abs(toy['f1'] - 1/3) < 1e-14
    iu = bundles['dual__iu']
    drop = iu['pilot']['macro_f1'] - iu['full']['macro_f1']
    contributions = {c: (iu['pilot']['cells'][c]['f1'] - v['f1']) / 4 for c, v in iu['full']['cells'].items()}
    assert abs(sum(contributions.values()) - drop) < 1e-14
    pending_names = ('dual__cond100_graph010', 'traj_iu_joint_graph__mean',
                     'traj_iu_joint_graph__gls', 'sample_risk_top__equal_graph_perm')
    pending = {k: dict(pilot_pb=pilot['metrics'][k]['pb']['macro_f1'],
                      full_status='NOT_IN_ANCHOR_PASS; see separate shortlist/sampling run') for k in pending_names}
    support = {name: core.support_summary(records, mask) for name, mask in masks.items()}
    result = dict(status='POSTHOC_DESCRIPTIVE_REVIEWED', created_unix=time.time(),
        model='Qwen3-8B', metric='equal four-cell macro of ProcessBench first-error/no-error harmonic score',
        new_fits=0, changed_predictions=0, population_support=support, metrics=bundles,
        iu_drop=drop, iu_cell_drop_contributions=contributions,
        omnimath_fraction_of_macro_drop=contributions['pb_omnimath_q8']/drop,
        newer_pilot_candidates=pending, source_hashes=hashes,
        limitations=['Descriptive population expansion; no causal attribution or significance claim.',
            'Pilot was source-disjoint and length-stratified with quotas; not a representative full-population estimate.',
            'Historical multiple-answer refits and new full-shortlist results are not supplied by this audit.'])
    (OUT / 'FINDINGS.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    pct = lambda x: f'{100*x:.2f}%'
    def table(headers, rows):
        return '<table><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in headers)+'</tr>'+''.join(
            '<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</table>'
    selected = ('dual__iu', 'dual__joint0', 'dual__graph010', 'dual__graph_perm', 'dual__equal', 'moment__iu', 'context__iu')
    methods = table(['Identical method', 'Pilot: 86 PB answers', 'Additional: 3314', 'Full: 3400'],
                    [[a]+[pct(bundles[a][n]['macro_f1']) for n in ('pilot','additional','full')] for a in selected])
    cells = table(['IU subset', 'Pilot n', 'Pilot score', 'Full n', 'Full score', 'Contribution to macro drop (pp)'],
        [[c.replace('pb_','').replace('_q8',''), iu['pilot']['cells'][c]['answers'], pct(iu['pilot']['cells'][c]['f1']),
          v['answers'], pct(v['f1']), f'{100*contributions[c]:.2f}'] for c,v in iu['full']['cells'].items()])
    peaks = table(['OmniMath IU outcome', 'Pilot', 'Full'], [
        [key, str(iu['pilot']['cells']['pb_omnimath_q8'][key]), str(iu['full']['cells']['pb_omnimath_q8'][key])]
        for key in ('clean', 'erroneous', 'clean_correct', 'error_correct', 'peak_correct', 'correct_peak_suppressed')])
    more = table(['Newer method', 'Pilot score', 'Full result in this anchor pass'],
                 [[k, pct(v['pilot_pb']), 'Not included'] for k,v in pending.items()])
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Why the single-answer score fell on full ProcessBench</title><style>body{font:16px/1.6 system-ui;max-width:1100px;margin:30px auto;padding:20px;color:#243449}table{border-collapse:collapse;width:100%;margin:20px 0}th,td{padding:8px;border-bottom:1px solid #cbd5e1;text-align:left}.note{padding:18px;background:#fff1d6}h1{line-height:1.25}</style>
<h1>We did see better single-answer scores. They were pilot results.</h1>
<p class="note">The same IU method scores 30.16% on the original 86 ProcessBench answers and 20.38% on all 3,400 Qwen3-8B answers.
The original pilot scores and decisions replay exactly. The other 3,314 answers score 20.12%.
This weakens the evidence for the current method; it is not an algorithm improvement.</p>
<p>Every row below still fits within one answer. The scorer, step mapping, no-error rule and metric stay fixed.
The pilot sampled source-disjoint questions with quotas across trace-length bins. Its small cells did not estimate full-population performance reliably.</p>
METHODS<h2>Most of the IU gap comes from OmniMath</h2>CELLS
<p>OmniMath contributes 7.14 percentage points, about 73% of the 9.78-point IU macro drop. This is an arithmetic decomposition, not a causal finding.
Each subset contributes one quarter of the macro score, even though the pilot has only 24 OmniMath answers.</p>
<p>In OmniMath, clean-answer accuracy changes from 8/10 (80%) to 85/241 (35.27%).
Exact first-error decisions change from 5/14 (35.71%) to 112/759 (14.76%).
Before the no-error gate, the peak is correct in 7/14 (50%) versus 191/759 (25.16%).
Both localization and the no-error decision weaken across these populations.</p>PEAKS
<h2>The newer promising pilot methods still need full results</h2>MORE
<p>Original Joint and graph rows above use condition1000. The newer graph uses condition100.
Do not substitute one row for the other or claim the whole new shortlist has already failed at 21%.
The 33.92% row is a simple equal-fusion plus permuted-graph control, not evidence of learned graph superiority.</p>
<h2>Research decision</h2><p>Do not promote the answer-only recipe from its pilot score. Finish the registered full shortlist and restore corrected historical comparisons before opening another feature or parameter sweep.
Investigate the full error/gate breakdown on a fixed development split, then require improvement over the matched incumbent on both tasks.
Full cached data are exposed development data; untouched confirmation remains required.</p>
<p>Review: 110 pilot rows × 19 original methods replayed, all 19 pilot macros and 76 full Q8 cell summaries recomputed, invalid/gate accounting fixture passed. No new fit or prediction.
<a href="FINDINGS.json">All 19 methods and population support</a> · <a href="REVIEW.json">Review and hashes</a></p></html>'''
    for token, value in [('METHODS',methods), ('CELLS',cells), ('PEAKS',peaks), ('MORE',more)]:
        page = page.replace(token, value)
    (OUT / 'REPORT.html').write_text(page, encoding='utf-8')
    for p,h in hashes.items():
        assert sha(Path(p)) == h
    review = dict(status='PASS', pilot_rows_replayed=110, original_method_replays=checks,
        pilot_macro_replays=len(arms), full_q8_cell_replays=4*len(arms), invalid_gate_fixture='PASS',
        unchanged_source_hashes=hashes, report_sha256=sha(OUT/'REPORT.html'),
        findings_sha256=sha(OUT/'FINDINGS.json'), scope='Same-session frozen-output audit; no external or browser review.')
    (OUT/'REVIEW.json').write_text(json.dumps(review, indent=2), encoding='utf-8')
    print(json.dumps(dict(status='PASS', iu_pilot=iu['pilot']['macro_f1'], iu_full=iu['full']['macro_f1'],
                         additional=iu['additional']['macro_f1'], omnimath_fraction=result['omnimath_fraction_of_macro_drop'],
                         support=support, report=str(OUT/'REPORT.html')), indent=2))


if __name__ == '__main__':
    main()
