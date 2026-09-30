"""Render reviewed fixed-recipe replication, with matched and historical panels."""
from collections import Counter
import hashlib
from html import escape
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/fusion_replication_v1'


def load(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def f(x, scale=1, signed=False):
    return 'undefined' if x is None else format(x * scale, '+.5f' if signed else '.5f')


def pct(x):
    return 'undefined' if x is None else f'{100*x:.2f}%'


def interval(x, scale=1):
    return '[' + ', '.join(f(v, scale, True) for v in x) + ']'


def table(headers, rows):
    return '<div class="table"><table><thead><tr>' + ''.join('<th scope="col">' + escape(h) + '</th>' for h in headers) + \
        '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + escape(str(v)) + '</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def mdtable(headers, rows):
    return '\n' + '| ' + ' | '.join(headers) + ' |\n|' + '|'.join(['---'] * len(headers)) + '|\n' + \
        '\n'.join('| ' + ' | '.join(str(v).replace('|', '/') for v in row) + ' |' for row in rows) + '\n'


def render():
    manifest, scores, evaluation, contrasts, review = [load(n) for n in
        ('MANIFEST.json', 'SCORES_FROZEN.json', 'EVALUATION.json', 'CONTRASTS.json', 'REVIEW.json')]
    assert review['status'] == 'PASS' and len(review['bootstrap_checks']) == 5
    assert review['review_script_sha256'] == sha(ROOT / 'scripts/review_fusion_replication_v1.py')
    for path, digest in {**review['hashes'], **review['review_dependencies']}.items():
        assert sha(path) == digest, path
    rows, metrics = evaluation['rows'], evaluation['metrics']
    chunks, markdown = [], ['# Fixed fusion recipes on additional source questions\n\n2026-09-07. Development replication; independent review PASS.\n']

    def section(title, prose, headers=None, data=None, extra=''):
        chunks.append('<section><h2>' + escape(title) + '</h2><p>' + escape(prose) + '</p>' +
                      (table(headers, data) if headers else '') + extra + '</section>')
        markdown.append('\n## ' + title + '\n\n' + prose + '\n' + (mdtable(headers, data) if headers else ''))

    section('What we learned',
        'The earlier Joint0 -> IU advantage did not recur on this additional cohort. Dual-bank IU has encouraging point estimates, '
        'but it does not establish a consistent advantage on both benchmarks or over its matched equal-fusion control. '
        'Joint with graph lambda 0.1 also improves over its zero-graph version at the point-estimate level. Its PB interval against '
        'the permuted graph excludes zero, but the zero-graph and IU comparisons do not establish a two-task advantage. No winner is promoted.')
    section('Our method is still fusion',
        'We develop IU-PCR and Joint L-SML on the N windows x P features matrix of one answer. Representation, grouping, '
        'graph penalties, sampling and temporal models serve this fusion. Any addition must beat the same fusion without it and '
        'be compared with simple aggregation using the same addition. This stage changes the development cohort, not the 19 fixed recipes.',
        extra='<div class="flow" aria-label="Fusion architecture">' + ''.join('<div>' + t + '</div>' for t in [
            'One fixed official answer<br><small>One teacher-forced gray-box model pass</small>',
            'N windows x P measurements<br><small>Moment or context representation</small>',
            '<strong>IU-PCR / Joint L-SML</strong><br><small>The central learned fusion</small>',
            'Fused trajectory<br><small>Official step scores and no-error decision</small>']) + '</div>')
    section('Exactly what was fixed',
        'Both banks have nine primitive telemetry streams and 27 window features. Moment uses mean, SD and slope; context uses '
        'level and EMA8/EMA32. Full fitting windows have eight tokens; a final end-anchored full window is scored when needed. '
        'Groups use K={3,4,6,8}, minimum size three and the existing stability/validity checks. The native inverse has condition '
        'target 1000; graph lambda is 0.1 with zero and node-permutation controls. Normalization, signs, gates, groups and fusion '
        'weights are fitted within each answer, with the declared fixed negative-entropy anchor. The peak and no-error rule are unchanged. '
        'This is offline processing of a full answer, not a claim of causal online detection.')

    definitions = [
        ('moment / context', 'The feature bank; these are alternative inputs to the same fusion family.'),
        ('equal / iu', 'Equal-weight fusion or IU-PCR on that bank.'),
        ('joint0 / graph010 / graph_perm', 'Joint L-SML with lambda 0 / graph lambda 0.1 / permuted graph lambda 0.1.'),
        ('single', 'Use moment Joint when valid; otherwise moment IU.'),
        ('dual Joint variants', 'Use moment Joint, then context Joint if needed, then moment IU.'),
        ('dual__iu / dual__equal', 'Use context only when moment Joint is invalid and context Joint is valid; otherwise moment. Apply IU or equal fusion on the selected bank.'),
        ('fixed-IU gate', 'Diagnostic: retain each arm\'s location peak, but use the original moment-IU binary error/no-error decision.')]
    section('Read the method names',
        'Routing is based on fit validity, never correctness labels, peak scores or the no-error decision. A selected readout failure '
        'stays a failure. Dual IU is a fixed control defined before scoring; its bank rule still requires Joint eligibility calculations '
        'and must not be described as the cost of plain IU.', ['Name', 'Meaning'], definitions)

    support = [[r['cell'], str(r['length_bin']), r['eligible_groups_at_bin_entry'], r['selected'], r['shortfall']]
               for r in manifest['length_support']]
    section('110 different source-question groups',
        '24 PRMB answers and 86 ProcessBench answers were selected without labels or fit outcomes, with at most eight groups per '
        'length bin and no filling of shortages. All 110 corrected groups are distinct and excluded from the documented 94-component '
        'Codex pilot inventory. This inventory is not a complete project exposure history. Claude already evaluated the whole cache, '
        'so this is development replication, not untouched publication confirmation. The sample stresses length ranges; it is not '
        'a prevalence-representative random sample. The corrected release is localization-cached-v2-sourcegroups-20260907; '
        'the v1 scoring namespace is retained to preserve the fixed graph-permutation seeds.',
        ['Cell', 'Tokens in answer', 'Eligible groups at bin entry', 'Selected', 'Quota shortfall'], support)

    current = []
    for arm in manifest['arms']:
        m = metrics[arm]
        current.append([arm, str(review['coverage'][arm]) + '/110', str(m['prm']['answers']) + '/24',
                        f(m['prm']['auroc']), f(m['prm']['within_answer_auc']), pct(m['pb']['macro_f1']), pct(m['pb_common_iu_gate']['macro_f1'])])
    section('Matched results: this 110-answer cohort',
        'Higher is better. PRMB pooled AUC ranks steps across answers; within-answer AUC measures local ranking only on answers '
        'with both labels. Full-coverage arms have 16 such PRMB answers. PB is the macro of four subset-level harmonic means of '
        'clean-answer accuracy and exact first-error accuracy. All 86 PB answers remain in the denominator, with invalid decisions '
        'counted as failures. Pure Joint PRMB rows use different valid populations; use the common-ID contrasts below to compare them. '
        'All valid fits in this run also have valid native and fixed-IU decisions.',
        ['Arm', 'Fit coverage', 'Valid PRMB', 'PRMB AUC', 'Within-answer AUC', 'PB native F1', 'PB fixed-IU gate (diagnostic)'], current)

    historic = []
    for arm in manifest['arms']:
        a, b = evaluation['previous_cohort_metrics'][arm], metrics[arm]
        historic.append([arm, a['prm']['answers'], f(a['prm']['auroc']), pct(a['pb']['macro_f1']),
                         b['prm']['answers'], f(b['prm']['auroc']), pct(b['pb']['macro_f1'])])
    section('History stays visible: two separate cohorts',
        'The earlier cohort has 58 answers: 12 PRMB (11 corrected source groups) and 46 PB. The new cohort has 110 answers: '
        '24 PRMB and 86 PB. These columns are historical context, not a paired cross-cohort comparison. All 19 old metric bundles '
        'were checked against the corrected Step 305 bridge. The higher or lower score of a recipe on different questions is not '
        'an algorithmic improvement. Expanded-K arms remain in the previous report and were not included in this fixed legacy-K replication.',
        ['Arm', 'Old valid PRMB', 'Old PRMB AUC', 'Old PB F1', 'New valid PRMB', 'New PRMB AUC', 'New PB F1'], historic)

    paired = []
    for pair in contrasts['pairs'].values():
        u = pair['uncertainty']; p, q = pair['left_prm'], pair['right_prm']
        paired.append([pair['left'] + ' minus ' + pair['right'], p['answers'], f(p['auroc'] - q['auroc'], signed=True),
                       interval(u['prm_common_valid_ci95']),
                       f(pair['left_pb']['macro_f1'] - pair['right_pb']['macro_f1'], 100, True),
                       interval(u['pb_all_population_ci95'], 100), u['pb_all_population_valid_draws']])
    section('38 registered paired comparisons',
        'Intervals use 1,000 source-group bootstrap draws, stratified by cell. These are exploratory, unadjusted 95% intervals '
        'across many comparisons. Undefined draws remain excluded and counted; most full-coverage PB comparisons have 999 defined '
        'draws, because one draw loses a required class in a subset. All four intervals, including within-answer AUC and the '
        'fixed-IU diagnostic, are stored in CONTRASTS.json. The highlighted graph-permutation result is one development signal; '
        'it is not confirmation of graph superiority over Joint0 or IU.',
        ['Left minus right', 'Common valid PRMB', 'Delta AUC', 'AUC CI', 'Delta PB (pp)', 'PB CI (pp)', 'Defined PB draws'], paired)

    selected_arms = ['moment__iu', 'single__joint0', 'dual__joint0', 'dual__graph010', 'dual__graph_perm', 'dual__equal', 'dual__iu', 'context__equal']
    pbrows = []
    for arm in selected_arms:
        for cell, v in metrics[arm]['pb']['cells'].items():
            rr = [r for r in rows if r['cell'] == cell]
            hit = lambda clean: sum(r['decision_valid'][arm] and r['predictions'][arm] == r['target'] for r in rr if (r['target'] == -1) == clean)
            pbrows.append([arm, cell, str(hit(True)) + '/' + str(v['clean']),
                           str(hit(False)) + '/' + str(v['erroneous']), pct(v['f1']), v['valid_decisions']])
    section('ProcessBench: clean answers and exact error locations',
        'The new PB cohort has 33 clean and 53 erroneous answers. Moment IU gets 15 clean and 11 exact errors right; dual IU gets '
        '18 and 12, while dual Joint graph gets 17 and 13. Raw peaks hit 17, 20 and 18 of the 53 erroneous answers respectively. '
        'Dual IU has the same GSM score and higher subset F1 in the other three subsets than moment IU, but Omni-Math exact-error '
        'hits decrease from six to five. Its higher clean-answer accuracy offsets this in the harmonic score. Do not call this '
        'uniform improvement of error localization. Every arm and subset, including those not expanded here, is in EVALUATION.json.',
        ['Arm', 'PB subset', 'Clean hits', 'Exact-error hits', 'Subset F1', 'Valid decisions'], pbrows)

    route_rows = []
    for cell in dict.fromkeys(r['cell'] for r in manifest['selected']):
        rr = [r for r in rows if r['cell'] == cell]; counts = Counter(r['routing']['routes']['dual'] for r in rr)
        route_rows.append([cell, len(rr), sum(r['valid']['moment__joint0'] for r in rr), sum(r['valid']['context__joint0'] for r in rr),
                           counts['moment_joint'], counts['context_joint'], counts['moment_iu']])
    section('Coverage and routing are part of the result',
        'Moment Joint is valid for 78/110 answers; context Joint for 102/110. Context rescues 29 moment failures but loses five '
        'moment-valid fits. The dual route therefore uses 78 moment Joint, 29 context Joint and three IU fallbacks. Single uses '
        '78 Joint and 32 IU. Moment has 31 inadmissible partitions and one unconverged/blocked multistart fit; context has seven '
        'inadmissible partitions and one such fit. These failures are retained. Context Joint has higher pooled AUC than context '
        'equal on their unmatched full rows, but on the same 21 valid PRMB answers it is worse: 0.68397 versus 0.69658. '
        'Their paired AUC interval is [-0.02420, -0.00289], also exploratory.',
        ['Cell', 'Answers', 'Moment Joint valid', 'Context Joint valid', 'Dual: moment Joint', 'Dual: context Joint', 'Dual: IU fallback'], route_rows)

    section('Review and compute',
        'Independent reconstruction passed for selection and group exclusions, all 110 raw-input/span and label-ID joins, 220 '
        'feature banks and normalizations, 980 weight projections, 1,970 step maps, 1,090 GMM decisions, 880 exact fallback '
        'inheritances, 19 complete metric/history bundles and all 38 paired point bundles. Five explicit 1,000-draw bootstrap '
        'reconstructions match all four intervals and defined counts. Five representative Joint refits reproduce 15 native '
        'inverse heads; the original Joint optimizer and graph-builder kernels were reused. This is not an independent '
        'implementation of those two algorithms. Four scientific tests passed before cohort freeze (two scorer/replay and two '
        'cohort-selection tests). The report is structurally checked; it has not been visually inspected in a browser.',
        ['Measurement', 'Value'], [
            ('Complete 19-arm scoring, three CPU workers', f'{scores["seconds_this_invocation"]:.2f} s wall time'),
            ('All 38 paired contrasts', f'{contrasts["seconds_this_invocation"]:.2f} s'),
            ('Independent review', f'{review["seconds"]:.2f} s'),
            ('Largest independently reconstructed feature difference', str(review['max_feature_difference'])),
            ('Largest reconstructed weight-projection difference', str(review['max_projection_difference'])),
            ('Largest refitted inverse-weight difference', str(review['max_refitted_inverse_difference']))])
    section('What this stage supports next',
        'Keep moment IU, dual IU, dual Joint0 and dual Joint graph as research anchors, with both always-on equal banks and '
        'the matched routed equal control. Do not tune these recipes on this cohort. Next audit Claude\'s proposed minimum '
        'FEATURE-group size two: distinguish identifiability of each latent loading from identifiability of the covariance '
        'and fusion weights, then check solver behavior before any new measured arm. This concerns our Joint fusion core. '
        'Benchmark SOURCE-question groups are a separate correction: Claude\'s multi-answer fits and selections still require '
        'corrected-fold reruns. Further feature/graph development and supporting temporal/geometry/sampling ideas remain open. '
        'A complete comparator registry, untouched confirmation and frozen-method transfer to the historical 24 cells are still pending.')

    links = [('MANIFEST.json', 'Frozen cohort, protocol/source hashes and comparison roster'),
             ('SCORES_FROZEN.json', 'Predictions frozen before label evaluation'), ('EVALUATION.json', 'All target joins, scores and endpoint bundles'),
             ('CONTRASTS.json', 'All 38 comparisons and four intervals'), ('REVIEW.json', 'Independent numerical review'),
             ('../localization_source_group_audit_v1/REPORT.html', 'Previous source-group correction and score bridge'),
             ('../fusion_explicit_fallback_pilot_v1/REPORT.html', 'Historical 58-answer fallback experiment'),
             ('../../docs/experiments/FUSION_SOURCE_DISJOINT_REPLICATION_V1.md', 'Frozen protocol'),
             ('../../spectral_utils/fusion_replication.py', 'Fusion scorer and explicit routing'),
             ('../../scripts/review_fusion_replication_v1.py', 'Independent reviewer'),
             ('../../docs/reviews/joint_lsml_visual_guide_2026-09-06.html', 'Undergraduate guide to our fusion architecture')]
    chunks.append('<section><h2>Evidence and code</h2><ul>' + ''.join('<li><a href="' + escape(p) + '">' + escape(t) + '</a></li>' for p, t in links) + '</ul></section>')
    markdown.append('\n## Evidence and code\n\n' + '\n'.join('- [' + t + '](' + p + ')' for p, t in links) + '\n')
    css = '''*{box-sizing:border-box}body{margin:0;background:#f3f6f5;color:#183b42;font:16px/1.65 system-ui,Segoe UI,sans-serif}header{background:#123f49;color:white;padding:44px max(22px,calc((100vw - 1180px)/2))}h1{font-size:clamp(30px,4vw,50px);line-height:1.15;margin:10px 0}header p{max-width:850px}main{max-width:1224px;margin:auto;padding:18px 22px 50px}section{background:white;border:1px solid #dbe5e1;border-radius:13px;padding:24px;margin:22px 0}h2{font-size:25px;line-height:1.3;margin-top:0}p{max-width:1000px}.badge{display:inline-block;background:#e1eee8;color:#164d40;padding:4px 12px;border-radius:20px;font-weight:650;font-size:13px}.table{overflow-x:auto;margin:20px 0}table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:10px;border-bottom:1px solid #dbe5e1;vertical-align:top}th{background:#e8f1ed}td:first-child{font-weight:650;white-space:nowrap}a{color:#075c90}a:focus-visible,summary:focus-visible{outline:3px solid #b17013;outline-offset:3px}.flow{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.flow>div{padding:18px;background:#eaf3f0;border-radius:9px;border-top:4px solid #087d72}.flow>div:nth-child(3){background:#123f49;color:white}small{font-size:13px}@media(max-width:760px){.flow{grid-template-columns:1fr}main{padding:12px}section{padding:17px}th,td{padding:8px}}@media print{body{background:white;font-size:11pt}header{background:white;color:#183b42;padding:15px}section{break-inside:auto;border:0;padding:10px}h2{break-after:avoid}.table{overflow:visible}table{font-size:8pt}th,td{padding:5px;white-space:normal!important}.flow{grid-template-columns:repeat(4,1fr)}}'''
    html = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Fusion replication: 110 additional source questions</title><style>' + css + \
        '</style></head><body><header><span class="badge">07 September 2026 · Reviewed development evidence</span><h1>Testing our fusion on<br>additional source questions</h1><p>19 fixed recipes. 110 different source groups. The fusion stays central; the results decide which additions earn their place.</p></header><main>' + ''.join(chunks) + '</main></body></html>\n'
    (OUT / 'REPORT.html').write_text(html, encoding='utf-8', newline='\n')
    (OUT / 'REPORT.md').write_text('\n'.join(markdown), encoding='utf-8', newline='\n')
    provenance = {'status': 'REVIEWED', 'source_hashes': {str(OUT/n): sha(OUT/n) for n in
        ('MANIFEST.json', 'SCORES_FROZEN.json', 'EVALUATION.json', 'CONTRASTS.json', 'REVIEW.json')},
        'renderer_sha256': sha(__file__), 'report_hashes': {str(OUT/n): sha(OUT/n) for n in ('REPORT.html', 'REPORT.md')}}
    (OUT / 'REPORT_PROVENANCE.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')

    class Checker(HTMLParser):
        def __init__(self):
            super().__init__(); self.stack = []; self.links = []; self.headings = 0

        def handle_starttag(self, tag, attrs):
            if tag not in ('meta', 'br', 'hr', 'input', 'img', 'link'):
                self.stack.append(tag)
            self.headings += tag == 'h1'
            if tag == 'a': self.links.append(dict(attrs)['href'])

        def handle_endtag(self, tag):
            assert self.stack.pop() == tag, tag

    checker = Checker(); checker.feed(html); checker.close()
    assert not checker.stack and checker.headings == 1
    for link in checker.links:
        target = urlsplit(link)
        assert not target.scheme
        assert (OUT / unquote(target.path)).resolve().exists(), link
    assert len(metrics) == 19 and len(contrasts['pairs']) == 38
    (OUT / 'ARTIFACT_VALIDATION.json').write_text(json.dumps({'status': 'PASS', 'html_structure': 'PASS',
        'local_links': len(checker.links), 'all_19_arms_and_38_contrasts': True, 'browser_visual_check': 'NOT_RUN',
        'provenance_sha256': sha(OUT / 'REPORT_PROVENANCE.json'), 'report_hashes': provenance['report_hashes']}, indent=2), encoding='utf-8')
    print('Rendered and structurally validated REPORT.html / REPORT.md; 19 arms, 38 contrasts,', len(checker.links), 'local links.')


if __name__ == '__main__':
    render()
