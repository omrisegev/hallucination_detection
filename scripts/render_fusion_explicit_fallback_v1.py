"""Simple-English visual report from frozen fallback results and review."""
from html import escape
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/fusion_explicit_fallback_pilot_v1'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def fmt(x, percent=False): return 'unavailable' if x is None else (f'{100*x:.2f}%' if percent else f'{x:.5f}')
def ci(x, percent=False):
    if x is None: return 'undefined'
    scale, suffix = (100, ' pp') if percent else (1, '')
    return f'[{scale*x[0]:+.4f}, {scale*x[1]:+.4f}]' + suffix
def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join('<th>' + escape(h) + '</th>' for h in headers) + \
        '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + escape(str(v)) + '</td>' for v in row) +
        '</tr>' for row in rows) + '</tbody></table></div>'
def mdtable(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join('---' for _ in headers) + ' |'] +
                     ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in rows])


def render():
    e, r, c, a = [load(OUT / n) for n in ('EVALUATION.json', 'REVIEW.json', 'CONTRASTS.json', 'ADDITIONAL_COMPARISONS.json')]
    assert r['status'] == 'PASS' and c['state'] == 'COMPLETE' and len(c['pairs']) == 30
    assert r['review_script_sha256'] == sha(ROOT / 'scripts/review_fusion_explicit_fallback_v1.py')
    for path, expected in r['source_hashes'].items(): assert sha(path) == expected
    assert r['additional_comparisons_sha256'] == sha(OUT / 'ADDITIONAL_COMPARISONS.json')
    m = e['metrics']; summaries = []; markdown = []
    title = 'Joint L-SML with an explicit IU fallback'
    lead = ('The fallback gives a score for all 58 development answers. Original Joint with IU fallback improves '
            'both headline point estimates over original IU. It does not establish a winner: context equal fusion '
            'has higher pooled PRMB and PB points, and the paired intervals against IU include zero.')
    summaries.append('<header><p class="eyebrow">Fusion research / 07 September 2026</p><h1>' + title + '</h1><p>' + lead + '</p></header>')
    markdown += ['# ' + title, '', lead, '']

    def section(heading, paragraphs=(), headers=None, rows=None):
        body = ''.join('<p>' + escape(p) + '</p>' for p in paragraphs)
        if headers: body += table(headers, rows)
        summaries.append('<section><h2>' + escape(heading) + '</h2>' + body + '</section>')
        markdown.extend(['## ' + heading, '', *[p + '\n' for p in paragraphs]])
        if headers: markdown.extend([mdtable(headers, rows), ''])

    flow = '<section><h2>The fusion core stays in place</h2><p>One answer supplies the N windows by P features matrix. The route depends only on whether Joint can be fitted.</p><div class="flow">'
    for heading, text in [('1. Original Joint', 'Use the moment bank when its Joint fit is valid: 43 answers.'),
                          ('2. Context Joint', 'Dual policy only: try context when the first fit is invalid. It rescues 13 answers.'),
                          ('3. IU-PCR', 'Single policy uses IU on 15 answers; dual uses IU on the remaining two.')]:
        flow += '<div class="card"><h3>' + heading + '</h3><p>' + text + '</p></div>'
    flow += '</div><p>The single policy skips box 2. Both policies copy the selected source score and gate unchanged. A no-error prediction stays selected; it is not a reason to try another method.</p></section>'
    summaries.append(flow)
    markdown.extend(['## Routing', '', 'Single: moment Joint -> moment IU. Dual: moment Joint -> context Joint -> moment IU. '
                     'Use fit validity only; a no-error prediction or readout failure does not trigger a new route.', ''])
    labels = {'moment__iu': 'Original IU', 'moment__equal': 'Original equal fusion', 'context__equal': 'Context equal fusion',
              'single__joint0': 'Original Joint -> IU', 'single__graph010': 'Original graph 0.1 -> IU',
              'single__graph_perm': 'Permuted graph -> IU', 'dual__joint0': 'Original Joint -> context Joint -> IU'}
    key_rows = [[labels[arm], fmt(m[arm]['prm']['auroc']), fmt(m[arm]['prm']['within_answer_auc']), fmt(m[arm]['pb']['macro_f1'], True)]
                for arm in labels]
    section('Matched results: all 12 PRMB and all 46 PB answers', [
        'These rows all have full fit and decision coverage. PRMB pooled AUC and average within-answer AUC answer different questions. '
        'Only 10 of the 12 PRMB answers contain both classes and contribute to the within-answer average.',
        'The always-context equal control is essential. Its higher headline points prevent a claim that the current learned fusion is best. '
        'Joint has a higher within-answer mean, so the comparison is not uniform across metrics.'],
        ['Method', 'PRMB pooled AUC', 'Within-answer AUC', 'PB macro F1'], key_rows)
    section('What the second bank actually changes', [
        'Dual routes seven PRMB answers through original Joint and five through context Joint. For PB the counts are 36 original Joint, eight context Joint and two IU.',
        'Compared with the single policy, each dual Joint variant changes four PB predictions and three peak locations. '
        'Every answer retains the same exact-success status: changed predictions are still wrong. This is why the PB metric and its paired difference interval are unchanged.',
        'Dual Joint without a graph has a lower pooled PRMB AUC than single Joint, but a higher within-answer mean. '
        'Neither difference establishes a reliable advantage on this small, repeatedly inspected sample.'])
    hit_rows = [[labels[arm], r['hit_counts'][arm]['clean_hits'], r['hit_counts'][arm]['exact_error_hits'],
                 r['hit_counts'][arm]['error_peak_hits']] for arm in ('moment__iu', 'single__joint0', 'single__graph010')]
    section('The PB improvement has a tradeoff', [
        'PB is the average across four subsets of the harmonic mean of clean-answer accuracy and exact first-error accuracy. '
        'It is not the fraction of all answers predicted correctly.',
        'Original Joint -> IU finds two more exact errors than IU, but recognizes four fewer clean answers. Its total exact successes fall from 16 to 14 of 46, '
        'while the registered PB macro F1 rises. Report both facts; this is not improvement on every answer type. Peak hits ignore the no-error gate and are diagnostic only.'],
        ['Method', 'Clean hits / 21', 'Exact error hits / 25', 'Error peak hits / 25'], hit_rows)
    cell_rows = [[cell, *[fmt(m[arm]['pb']['cells'][cell]['f1'], True) for arm in ('moment__iu', 'single__joint0', 'single__graph010')]]
                 for cell in m['moment__iu']['pb']['cells']]
    section('PB subset results', ['Joint -> IU improves GSM8K and OlympiadBench, ties MATH and worsens Omni-MATH relative to IU.'],
            ['Subset', 'IU', 'Joint -> IU', 'Graph -> IU'], cell_rows)

    pair_names = [('single__joint0', 'moment__iu'), ('single__joint0', 'moment__equal'), ('dual__joint0', 'single__joint0'),
                  ('dual__joint0', 'dual__equal'), ('dual__graph010', 'dual__joint0'), ('dual__graph010', 'dual__graph_perm')]
    pair_rows = []
    for left, right in pair_names:
        pair = c['pairs'][left + ' minus ' + right]; u = pair['uncertainty']
        pair_rows.append([left + ' minus ' + right, ci(u['prm_common_valid_ci95']),
                         ci(u['prm_within_answer_common_valid_ci95']), ci(u['pb_all_population_ci95'], True)])
    section('Paired uncertainty: registered comparisons', [
        'These are 95% percentile intervals for left minus right, using 1,000 source-group draws stratified by cell. '
        'They are exploratory and unadjusted for multiple comparisons and repeated development. The displayed full-coverage pairs have 1,000 defined PRMB draws and 982 defined PB draws.',
        'Original Joint -> IU versus IU changes PRMB AUC by +0.01402 and PB by +9.51 percentage points; both intervals include zero. '
        'The graph still has no established advantage over lambda zero or the permuted graph.'],
        ['Pair', 'PRMB AUC difference CI', 'Within-answer difference CI', 'PB difference CI'], pair_rows)
    extra_rows = [[key, ci(pair['uncertainty']['prm_common_valid_ci95']), ci(pair['uncertainty']['pb_all_population_ci95'], True)]
                  for key, pair in a['pairs'].items()]
    section('Review finding: include the stronger simple incumbent', [
        'The original 30-pair roster contained routed equal controls but omitted a paired comparison with always-context equal. '
        'Review added these two comparisons in a separate, explicitly post-evaluation artifact. The original protocol and 30 contrasts are unchanged.',
        'These additional intervals also include zero for the two headline endpoints. No candidate is promoted from them.'],
        ['Post-evaluation comparison', 'PRMB AUC difference CI', 'PB difference CI'], extra_rows)
    all_rows = []
    for arm, mm in m.items():
        valid_pb = sum(v['valid_decisions'] for v in mm['pb']['cells'].values())
        all_rows.append([arm, str(r['hit_counts'][arm]['fit_valid']) + '/58', mm['prm']['answers'],
                         fmt(mm['prm']['auroc']), fmt(mm['prm']['within_answer_auc']), fmt(mm['pb']['macro_f1'], True),
                         str(valid_pb) + '/46', fmt(mm['pb_common_iu_gate']['macro_f1'], True)])
    section('All 25 arms, including all 17 previous anchors', [
        'Pure Joint rows keep their fit failures. Their PRMB AUC uses fewer answers and cannot be compared directly with a full-coverage row as an algorithmic gain. '
        'PB includes all 46 answers and counts invalid decisions as failures. Expanded-K arms are unchanged references, not additional routing candidates.',
        'The last column holds the same answer-fitted parent-IU binary gate fixed for every method. It is a diagnostic, separate from the primary native-gate decision.'],
        ['Arm', 'Fit coverage', 'PRMB answers', 'Pooled AUC', 'Within AUC', 'PB native', 'PB decision coverage', 'PB common-IU gate'], all_rows)
    section('What was verified', [
        'Five scientific routing tests pass: all eligibility cases, clean/no-error behavior, failed readouts, exhausted fallbacks, exact source copying and invalid-source rejection.',
        'Independent review reconstructs the routing truth table and metrics without importing the composite implementation or evaluator. '
        'It verifies the provenance of previously audited parent fits, without refitting them. Three registered 1,000-draw comparisons are independently reconstructed by explicitly resampling rows, '
        'including all four endpoint intervals and defined-draw counts.',
        'A Windows file-access error interrupted the first contrast checkpoint write. The process exited, and the identical frozen runner resumed its completed checkpoints; '
        'all 30 contrasts now exist and pass review. No scientific code or frozen parent artifact was changed.'],
        ['Check', 'Count'], [[k, v] for k, v in r['counts'].items()])
    score_time = load(OUT / 'SCORES_FROZEN.json')['seconds']
    section('Cost and fitting scope', [
        f'Composing the cached score bundles took {score_time:.2f} seconds on one CPU process. This excludes the earlier feature, grouping, graph and fusion fits, '
        'and is not an end-to-end speed claim. The resumed contrast invocation took ' + f"{c['seconds_this_invocation']:.2f}" + ' seconds; that excludes work in the interrupted invocation.',
        'Every fitted quantity comes from the same answer under the previously declared rules and negative-entropy anchor. '
        'There is one model generation pass and offline processing of saved gray-box telemetry. The routed simple controls still use Joint eligibility and therefore do not avoid its fitting cost.'])
    section('The next useful step', [
        'Keep Original Joint -> IU as a viable fusion candidate, retain IU and both equal-fusion banks, and retain lambda-zero and permuted-graph controls. '
        'The dual bank improves Joint coverage but does not justify replacing the simpler route.',
        'Before more tuning on these 58 answers, audit exposure and source-group overlap, then freeze a small disjoint-group replication with the same recipes and evaluator. '
        'The existing release explicitly says development_previously_evaluated_by_v2. A different subset from it is development replication, not an untouched publication test.',
        'Joint-specific features/grouping, further graph hypotheses, supporting temporal/geometry/sampling methods, the wider comparator registry, untouched two-task confirmation '
        'and the requested 24-cell transfer remain open. This stage does not complete the research goal.'])
    summaries.append('<footer><p><a href="../../docs/experiments/FUSION_EXPLICIT_FALLBACK_PILOT_V1.md">Frozen protocol</a> · '
        '<a href="../fusion_context_bank_pilot_v1/REPORT.html">Previous context-bank experiment</a> · '
        '<a href="CONTRASTS.json">30 registered comparisons</a> · <a href="ADDITIONAL_COMPARISONS.json">Two post-evaluation review comparisons</a> · '
        '<a href="REVIEW.json">Independent review</a></p><p>Evidence from 58 exposed development answers. No confirmed winner.</p></footer>')
    style = 'body{margin:0;background:#f4f7f6;color:#15323c;font:17px/1.65 system-ui,sans-serif}main{max-width:1180px;margin:auto;padding:24px}header{background:#153f49;color:white;padding:32px;border-radius:16px}h1{font-size:clamp(30px,5vw,48px);line-height:1.15}h2{font-size:26px;line-height:1.3}.eyebrow{letter-spacing:2px;font-size:12px;text-transform:uppercase}section{margin:30px 0;padding:24px;background:white;border:1px solid #d4e2de;border-radius:14px}p{max-width:1000px}.flow{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}.card{padding:18px;border:2px solid #16877b;border-radius:10px;background:#edf8f4}.card h3{margin-top:0}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;padding:12px;border-bottom:1px solid #d4e2de;vertical-align:top}th{background:#edf3f2}td:first-child{font-weight:600}a{color:#086c86}footer{font-size:14px;padding:20px}@media(max-width:700px){main{padding:12px}section,header{padding:18px}.flow{grid-template-columns:1fr}}@media print{body{background:white;font-size:11pt}header{background:white;color:#15323c}section{border:0;padding:5px}.scroll{overflow:visible}table{font-size:8pt}th,td{padding:5px}h2{break-after:avoid}}'
    html = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>' + title + '</title><style>' + style + '</style></head><body><main>' + ''.join(summaries) + '</main></body></html>'
    (OUT / 'REPORT.html').write_text(html, encoding='utf-8')
    (OUT / 'REPORT.md').write_text('\n'.join(markdown), encoding='utf-8', newline='\n')
    files = {str(OUT / name): sha(OUT / name) for name in ('EVALUATION.json', 'CONTRASTS.json', 'REVIEW.json',
             'ADDITIONAL_COMPARISONS.json', 'REPORT.html', 'REPORT.md')}
    files[str(Path(__file__))] = sha(__file__)
    (OUT / 'REPORT_PROVENANCE.json').write_text(json.dumps({'hashes': files}, indent=2), encoding='utf-8')
    print('HTML / Markdown report rendered with seven provenance hashes.')


if __name__ == '__main__': render()
