"""Independent saved-array review and transparent report; no candidate fitting."""
import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/fusion_window_sampling_pilot_v1'
PARENT = ROOT/'results/answer_localization_representation_pilot_v1'
REP = 'moments27_local8'


def load(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p, d): Path(p).write_text(json.dumps(d, indent=2, allow_nan=False), encoding='utf-8')


def auc(rows, arm):
    available = [r for r in rows if r['cell'].startswith('prm') and r['valid'][arm]]
    if not available: return None
    y = np.concatenate([r['target'] for r in available]); x = np.concatenate([r['scores'][arm] for r in available])
    a, b = x[y == 1], x[y == 0]
    if not len(a) or not len(b): return None
    return float(np.mean(a[:, None] > b[None, :])+.5*np.mean(a[:, None] == b[None, :]))


def pb(rows, arm, fixed=False):
    values = []
    for cell in sorted({r['cell'] for r in rows if r['cell'].startswith('pb_')}):
        hits = {True: [], False: []}
        for r in (r for r in rows if r['cell'] == cell):
            pred = r['fixed_parent_predictions'][arm] if fixed else r['predictions'][arm]
            hits[r['target'] == -1].append(bool(r['decision_valid'][arm] and pred == r['target']))
        if not hits[True] or not hits[False]: return None
        ca, ea = np.mean(hits[True]), np.mean(hits[False])
        values.append(2*ca*ea/(ca+ea) if ca+ea else 0.)
    return float(np.mean(values)) if values else None


def review():
    manifest, frozen, evaluation = [load(OUT/name) for name in ('MANIFEST.json', 'SCORES_FROZEN.json', 'EVALUATION.json')]
    for p, h in {**manifest['hashes'], **frozen['files']}.items(): assert sha(p) == h, p
    assert frozen['manifest_sha256'] == sha(OUT/'MANIFEST.json')
    assert evaluation['scores_sha256'] == sha(OUT/'SCORES_FROZEN.json')
    rows = evaluation['rows']; cores, selectors = manifest['cores'], manifest['readouts']
    assert {r['uid'] for r in rows} == {r['uid'] for r in manifest['selected']}
    assert len(rows) == len(manifest['selected']) == 58
    counts = Counter(); failures = Counter(); parent_failures = Counter(); valid_counts = Counter(); max_mapping_error = 0.
    selection_summary = {s: {'jaccards': [], 'seed_jaccards': [], 'gaps': [], 'seconds': [], 'selected_fraction': []} for s in selectors}
    release = load(PARENT/'RELEASE.json')
    for cell in {r['cell'] for r in rows}:
        info = release['cells'][cell]
        assert sha(info['label_path']) == info['label_opaque_sha256']
        with np.load(info['label_path'], allow_pickle=False) as labels:
            for row in (r for r in rows if r['cell'] == cell):
                indices = np.flatnonzero(labels['row_ids'] == row['row_id']); assert len(indices) == 1
                i = int(indices[0])
                if cell.startswith('prm'):
                    a, b = labels['step_flag_offsets'][i:i+2]; target = labels['step_error_flags'][a:b]
                else: target = labels['first_error'][i]
                np.testing.assert_array_equal(target, row['target']); counts['label_joins'] += 1
    for row in rows:
        uid = row['uid']; meta = load(OUT/'scores'/f'{uid}.json'); parent = load(PARENT/'scores'/f'{uid}.json')
        with np.load(OUT/'scores'/f'{uid}.npz', allow_pickle=False) as z, np.load(PARENT/'scores'/f'{uid}.npz', allow_pickle=False) as old:
            values = old[REP+'__features']; original = old[REP+'__fit_indices']; n = len(original)
            quota = min(n, max(32, (n+1)//2))
            assert row['sampling_eligible'] == (quota < n)
            names = [s+'__'+op for s in ('entropy_series','spilled_series','energy_series','top1_logprob_series',
                'logprob_margin_series','topk_entropy_series','topk_varentropy_series','topk_renyi2_series','topk_tail_mass_series')
                for op in ('level','sd','slope')]
            for selector in selectors:
                diag = meta['sampling']['selectors'][selector]
                if selector+'__selected' not in z:
                    assert diag['status'] == 'FAILED'; counts['selector_failures'] += 1; continue
                indices = z[selector+'__selected']; expected_count = n if selector == 'full' else quota
                assert len(indices) == len(np.unique(indices)) == expected_count
                assert np.isin(indices, original).all() and np.all(np.diff(indices) > 0)
                if expected_count == n: np.testing.assert_array_equal(indices, original)
                elif selector == 'uniform':
                    np.testing.assert_array_equal(indices, original[np.rint(np.linspace(0, n-1, quota)).astype(int)])
                elif selector in ('risk_top', 'dufs_transposed', 'dufs_permuted'):
                    probabilities = values[original, 0] if selector == 'risk_top' else np.asarray(diag['probabilities'])
                    ordered = sorted(range(n), key=lambda j: (-probabilities[j], j))[:quota]
                    np.testing.assert_array_equal(indices, original[sorted(ordered)])
                starts, ends = old[REP+'__starts'], old[REP+'__ends']
                if selector == 'dufs_permuted' and quota < n:
                    source = meta['sampling']['selectors']['dufs_transposed']['probabilities']
                    np.testing.assert_array_equal(diag['probabilities'], np.asarray(source)[diag['permutation']])
                    assert len(set(diag['permutation'])) == n
                if row['sampling_eligible']:
                    summary = selection_summary[selector]
                    summary['jaccards'] += [p['jaccard'] for p in diag['block_perturbation'] if p['jaccard'] is not None]
                    summary['seed_jaccards'] += diag.get('seed_pair_jaccard', [])
                    summary['gaps'].append(diag['largest_start_gap_tokens'])
                    summary['seconds'].append(diag['seconds']); summary['selected_fraction'].append(len(indices)/n)
                if row['cell'].startswith('pb_') and row['target'] >= 0:
                    a, b = z['step_starts'][row['target']], z['step_ends'][row['target']]
                    overlap = any(max(a, starts[i]) < min(b, ends[i]) for i in indices)
                    assert bool(overlap) == row['first_error_support'][selector]; counts['error_support_checks'] += 1
                counts['selection_contracts'] += 1
                shared = meta['sampling']['fits'][selector]['shared']
                for core in cores:
                    arm = core+'@@'+selector; detail = meta['methods'][arm]
                    assert row['valid'][arm] == row['decision_valid'][arm] == detail['valid']
                    if not detail['valid']:
                        failures[detail.get('reason', detail.get('status', 'UNKNOWN'))] += 1
                        if detail.get('parent_replay'):
                            prior = parent['report']['methods'][core]
                            parent_failures[prior.get('reason', prior.get('status', 'UNKNOWN'))] += 1
                        assert row['predictions'][arm] is None
                        continue
                    valid_counts[arm] += 1
                    window = z[arm+'__window']; steps = z[arm+'__risk']
                    total = np.zeros(row['tokens']); support = np.zeros(row['tokens'])
                    for i in range(len(starts)):
                        total[starts[i]:ends[i]] += window[i]; support[starts[i]:ends[i]] += 1
                    assert np.all(support > 0)
                    mapped = [np.max((total/support)[a:b]) for a,b in zip(z['step_starts'], z['step_ends'])]
                    max_mapping_error = max(max_mapping_error, float(np.max(np.abs(mapped-steps))))
                    np.testing.assert_allclose(mapped, steps, atol=1e-12, rtol=1e-12)
                    np.testing.assert_array_equal(steps, row['scores'][arm]); counts['span_maps'] += 1
                    gate = detail['gate_readout']; expected = int(np.argmax(steps)) if gate['prediction'] != -1 else -1
                    assert expected == detail['prediction'] == row['predictions'][arm]
                    if detail['parent_replay']:
                        np.testing.assert_array_equal(window, old[core+'__window'])
                        np.testing.assert_array_equal(steps, old[core+'__step'])
                        assert gate == parent['report']['methods'][core]['readout']; counts['parent_replays'] += 1
                    else:
                        columns = [names.index(s) for s in shared['active_features']]
                        selected = values[np.ix_(indices, columns)]
                        np.testing.assert_allclose(selected.mean(axis=0), shared['mean'], atol=1e-12)
                        np.testing.assert_allclose(selected.std(axis=0), shared['sd'], atol=1e-12)
                        features = (values[:, columns]-shared['mean'])/shared['sd']
                        features -= features[indices].mean(axis=0); features *= shared['feature_signs']
                        reconstructed = -(features @ np.asarray(detail['standardized_weights']))
                        np.testing.assert_allclose(reconstructed, window, atol=1e-11, rtol=1e-11)
                        counts['selected_fit_weight_replays'] += 1
                        if 'joint_' in core:
                            fit = shared['joint']; jac = fit['jacobian']
                            assert fit['converged'] and fit['multistart']['status'] == 'PASS'
                            assert jac['full_global_rank'] and jac['condition_number'] <= 1e8
                            counts['joint_validity_checks'] += 1
    fixed = {}; eligible_metrics = {}
    eligible_rows = [r for r in rows if r['sampling_eligible']]
    for arm, expected in evaluation['metrics'].items():
        observed = auc(rows, arm)
        assert observed is None and expected['prm']['auroc'] is None or abs(observed-expected['prm']['auroc']) < 1e-12
        assert abs(pb(rows, arm)-expected['pb']['macro_f1']) < 1e-12
        fixed[arm] = pb(rows, arm, fixed=True)
        eligible_metrics[arm] = {'prm_available_auc': auc(eligible_rows, arm), 'pb_macro_f1': pb(eligible_rows, arm)}
        counts['endpoint_checks'] += 1
    parent_peak = load(ROOT/'results/fused_trajectory_readout_pilot_v1/EVALUATION.json')
    for core in cores:
        assert evaluation['metrics'][core+'@@full'] == parent_peak['metrics'][core+'@@parent_peak']
        counts['matched_previous_stage_endpoints'] += 1
    support = {}
    for selector in selectors:
        support[selector] = {}
        for short in (False, True):
            targets = [r for r in eligible_rows if r['cell'].startswith('pb_') and r['target'] >= 0
                       and (not short or r['first_error_tokens'] <= 32)]
            support[selector]['short' if short else 'all'] = {'n': len(targets),
                'retained': sum(r['first_error_support'][selector] for r in targets)}
    summary = {}
    for s, d in selection_summary.items():
        summary[s] = {k+'_mean': float(np.mean(v)) if v else None for k, v in d.items()}
    report = {'status': 'PASS', 'counts': dict(counts), 'max_mapping_error': max_mapping_error,
        'valid_counts': dict(valid_counts), 'failures': dict(failures), 'replay_failures_original_reason': dict(parent_failures),
        'eligible_answers': len(eligible_rows), 'eligible_prm_answers': sum(r['cell'].startswith('prm') for r in eligible_rows),
        'fixed_parent_gate_pb': fixed, 'eligible_metrics': eligible_metrics, 'sampling_summary': summary,
        'first_error_step_support': support, 'evaluation_sha256': sha(OUT/'EVALUATION.json'), 'review_script_sha256': sha(__file__)}
    save(OUT/'REVIEW.json', report)
    print(json.dumps({k:report[k] for k in ('status','counts','failures','eligible_answers')}, indent=2), flush=True)


def render():
    evaluation, audit, contrasts, manifest = [load(OUT/n) for n in ('EVALUATION.json','REVIEW.json','CONTRASTS.json','MANIFEST.json')]
    assert contrasts['state'] == 'COMPLETE' and len(contrasts['pairs']) == 57
    assert audit['evaluation_sha256'] == contrasts['evaluation_sha256'] == sha(OUT/'EVALUATION.json')
    cores, selectors = manifest['cores'], manifest['readouts']
    short = lambda s: s.replace(REP+'__', '')
    lines = ['# Window sampling to support our fusion', '', 'Completed 2026-09-07. Adaptive development; no confirmed winner.', '',
        'IU-PCR and Joint L-SML remain the core. We test which windows should supply the data used to fit their weights. The selector has no access to correctness labels. Every selector is shared with equal aggregation and both graph controls.', '',
        '## What changed', '',
        'One answer becomes N nonoverlapping 8-token windows by 27 moment measurements. The selected fitting budget is min(N, max(32, ceil(N/2))). Short answers use all rows with exact parent replay. The same fitted fusion still scores every window and official step. This is a fitting-data experiment; feature extraction is not reduced.', '',
        'Full, uniform and raw-entropy top-risk controls accompany two graph ideas. Transposed DUFS gates windows but graphs feature coordinates; window diffusion graphs windows and chooses geometric representatives. Its geometry is not a learned measure of correctness. A shuffled-DUFS control tests whether learned placement matters.', '',
        'Every arm uses the same mixture error-gate rule on all original nonoverlapping fused scores and the same peak locator. Refitting may change both risk and the gate. A separate fixed-parent-gate diagnostic below isolates the location change.', '',
        '## ProcessBench macro F1 (%) — all 46 fixed answers', '',
        '| Core | Full | Uniform | Top risk | DUFS windows | Shuffled DUFS | Window diffusion |', '|---|---:|---:|---:|---:|---:|---:|']
    for core in cores:
        lines.append('| '+short(core)+' | '+' | '.join(f"{100*evaluation['metrics'][core+'@@'+s]['pb']['macro_f1']:.2f}" for s in selectors)+' |')
    lines += ['', '## PRMBench pooled step AUROC — availability shown', '',
        'Each entry is AUROC (valid answers). Cells with different coverage are not a common-population leaderboard. Use the paired contrasts to judge method differences.', '',
        '| Core | Full | Uniform | Top risk | DUFS windows | Shuffled DUFS | Window diffusion |', '|---|---:|---:|---:|---:|---:|---:|']
    for core in cores:
        metrics = [evaluation['metrics'][core+'@@'+s]['prm'] for s in selectors]
        lines.append('| '+short(core)+' | '+' | '.join(f"{m['auroc']:.5f} ({m['answers']})" if m['auroc'] is not None else 'unavailable' for m in metrics)+' |')
    lines += ['', '## Paired evidence', '',
        'Exploratory, unadjusted 95% intervals from 1,000 source-group bootstrap draws. PRMB compares common valid IDs; PB includes all fixed IDs with failure penalties. Some resamples lack one class within a PB subset and have undefined macro F1; only defined draws form the interval, and their count is logged per comparison. All 57 registered comparisons are saved, including comparisons not displayed here.', '',
        '| Left minus right | Common PRMB N | PRMB difference [95% CI] | PB difference in points [95% CI] |', '|---|---:|---|---|']
    for name, pair in contrasts['pairs'].items():
        left, right = pair['left'], pair['right']
        show = (left.startswith(cores[1]) and right == cores[1]+'@@full') or (
            left.startswith(cores[3]) and right == cores[3]+'@@full') or (
            left.endswith('@@dufs_transposed') and right.endswith('@@dufs_transposed'))
        if not show: continue
        a,b = pair['left_prm']['auroc'], pair['right_prm']['auroc']; uncertainty = pair['uncertainty']
        def interval(value, ci, scale):
            if value is None or ci is None: return 'undefined'
            return f'{scale*value:+.4f} ['+', '.join(f'{scale*x:+.4f}' for x in ci)+']'
        prm = interval(a-b if a is not None and b is not None else None, uncertainty['prm_common_valid_ci95'], 1)
        pb_diff = pair['left_pb']['macro_f1']-pair['right_pb']['macro_f1']
        lines.append(f"| {short(name)} | {pair['left_prm']['answers']} | {prm} | {interval(pb_diff,uncertainty['pb_all_population_ci95'],100)} |")
    lines += ['', '## Sampling stability and potential short-error support', '',
        f"Sampling reduces rows in {audit['eligible_answers']}/58 answers, including {audit['eligible_prm_answers']} PRMB answers. Stability below is measured only on these eligible answers. Short-event support means any selected fitting window overlaps the first erroneous official step; it is not sparse-detector recall.", '',
        '| Selector | Mean selected fraction | Block-perturbation Jaccard | DUFS seed Jaccard | Selection seconds | First-error support | <=32-token error support |', '|---|---:|---:|---:|---:|---|---|']
    for s in selectors:
        d = audit['sampling_summary'][s]; support = audit['first_error_step_support'][s]
        fmt = lambda v: '-' if v is None else f'{v:.3f}'
        lines.append('| '+s+' | '+' | '.join(fmt(d[k]) for k in ('selected_fraction_mean','jaccards_mean','seed_jaccards_mean','seconds_mean'))+
            f" | {support['all']['retained']}/{support['all']['n']} | {support['short']['retained']}/{support['short']['n']} |")
    lines += ['', 'There are zero <=32-token first-error steps in the sampling-eligible PB group. The 0/0 entries therefore mean no evidence, not perfect retention. Selection seconds exclude fusion fitting and perturbation repeats; shuffled DUFS shares the DUFS training cost.', '',
        'Two-token block bootstrap stays within the original eight-token window. It tests measurement sensitivity, adds no independent observations and does not reproduce LOCA. The DUFS selector uses fixed 120-epoch optimization; a negative result does not close longer optimization or different task-aware sampling objectives.', '',
        '## Fit coverage and gate diagnostic', '',
        '| Core / selector | Valid answers / 58 | PB with refitted-score gate (%) | PB with fixed parent gate (%) |', '|---|---:|---:|---:|']
    for core in (cores[1], cores[3]):
        for s in selectors:
            arm = core+'@@'+s
            lines.append(f"| {short(arm)} | {audit['valid_counts'].get(arm,0)} | {100*evaluation['metrics'][arm]['pb']['macro_f1']:.2f} | {100*audit['fixed_parent_gate_pb'][arm]:.2f} |")
    lines += ['', 'Fixed-parent-gate values are diagnostics; unavailable parent gates or current fits earn no credit. The full cohort remains primary. Sampling-eligible point estimates, within-answer PRMB AUROC and every subset accuracy are retained in REVIEW.json and EVALUATION.json.', '',
        '## Review, cost and historical bridge', '',
        'Six contract tests passed. Independent review validates hashes, selected-only normalization and weight reconstruction, exact parent replays, official span mapping, valid Joint fit requirements, direct label-ID joins and every endpoint. Detailed counts are in REVIEW.json. Replay metadata uses PARENT_READOUT_INVALID for an unavailable parent path, including a parent fit that never produced a readout; the original failure causes are separately recovered in REVIEW.json. The independent span sum needed a 1e-12 floating-point tolerance instead of bitwise equality; frozen scores were unchanged.', '',
        f"Scoring, selection and two perturbation repetitions took {load(OUT/'SCORES_FROZEN.json')['seconds']:.1f} seconds on three CPU workers. Bootstrap time is additional. This does not establish a runtime saving: most selectors add work, and every window feature is still computed.", '',
        'The five full arms exactly match the previous readout pilot peak endpoints. The same-cohort entropy peak control remains PB 19.85% and PRMB 0.62587; it is a diagnostic control, not a replacement for our method. Earlier 30-long-answer IU 0.70070 and Claude pooled-fit figures use different populations and fit contracts. They remain historical context and cannot be used as matched improvement claims.', '',
        'Source: local DUFS arXiv:2007.04728v3 (2020), equations 6–7 and the full paper digest. The later NeurIPS 2021 paper has a different author list and equation numbering. Both selectors here are explicitly adaptations for our fusion architecture.', '']
    interpretation = OUT/'INTERPRETATION.md'
    if interpretation.exists(): lines += interpretation.read_text(encoding='utf-8').splitlines()
    (OUT/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    body=[]; table=False
    for line in lines:
        if line.startswith('|---'): continue
        if line.startswith('|'):
            tag='td' if table else 'th'
            if not table: body.append('<div class="scroll"><table>'); table=True
            body.append('<tr>'+''.join(f'<{tag}>{html.escape(c.strip())}</{tag}>' for c in line.strip('|').split('|'))+'</tr>'); continue
        if table: body.append('</table></div>');table=False
        if line.startswith('# '): body.append('<h1>'+html.escape(line[2:])+'</h1>')
        elif line.startswith('## '): body.append('<h2>'+html.escape(line[3:])+'</h2>')
        elif line: body.append('<p>'+html.escape(line)+'</p>')
    diagram='<div class="flow"><span>One answer → N windows × P features</span><b>→</b><span>Choose fitting windows</span><b>→</b><strong>Our IU-PCR / Joint L-SML fusion</strong><b>→</b><span>Score every step + no-error decision</span></div>'
    page='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Window sampling supports our fusion</title><style>body{font:17px/1.6 system-ui;background:#f4f7f8;color:#183d48;margin:0}main{max-width:1180px;margin:auto;padding:32px}h1{font-size:36px;line-height:1.2}h2{margin-top:42px}.scroll{overflow:auto}table{border-collapse:collapse;background:white;width:100%;font-size:14px}th,td{border:1px solid #cfdce1;padding:10px;text-align:left}th{background:#dcebe8}.flow{display:flex;flex-wrap:wrap;gap:12px;align-items:center;padding:20px;background:#e4eeef;border-radius:12px}.flow strong{background:#175f5a;color:white;padding:15px;border-radius:8px}.flow span{max-width:230px}</style><main>'+diagram+''.join(body)+'</main></html>'
    (OUT/'REPORT.html').write_text(page, encoding='utf-8')
    save(OUT/'REPORT_PROVENANCE.json', {'evaluation_sha256':sha(OUT/'EVALUATION.json'),'review_sha256':sha(OUT/'REVIEW.json'),
        'contrasts_sha256':sha(OUT/'CONTRASTS.json'),'report_script_sha256':sha(__file__),
        'interpretation_sha256':sha(interpretation) if interpretation.exists() else None})
    print('Written REPORT.md and REPORT.html', flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--phase',choices=('review','report'),required=True)
    {'review':review,'report':render}[parser.parse_args().phase]()
