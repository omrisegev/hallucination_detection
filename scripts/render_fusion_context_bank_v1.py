"""Render reviewed context-bank results, with coverage and matched controls."""
import hashlib
import html
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/fusion_context_bank_pilot_v1'
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fmt(x,places=5):return 'unavailable' if x is None else f'{x:.{places}f}'
def percent(x):return 'unavailable' if x is None else f'{100*x:.2f}'
def ci(x,scale=1):return 'undefined' if x is None else '['+', '.join(f'{scale*v:+.4f}' for v in x)+']'
def table(head,rows):return '\n'.join(['| '+' | '.join(head)+' |','|'+'|'.join('---' for _ in head)+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])


def render():
    e,c,a=[load(OUT/n) for n in ('EVALUATION.json','CONTRASTS.json','REVIEW.json')]
    assert a['status']=='PASS' and a['evaluation_sha256']==c['evaluation_sha256']==sha(OUT/'EVALUATION.json')
    assert a['contrasts_sha256']==sha(OUT/'CONTRASTS.json')
    assert a['review_script_sha256']==sha(ROOT/'scripts/review_fusion_context_bank_v1.py')
    m=e['metrics'];rows=e['rows'];manifest=load(OUT/'MANIFEST.json')
    fit_geometry={};offdiag_checks=0
    for rec in manifest['selected']:
        meta=load(OUT/'scores'/f"{rec['uid']}.json")['diagnostics']
        with np.load(OUT/'scores'/f"{rec['uid']}.npz",allow_pickle=False) as arrays:
            for family,j in meta['joint_fits'].items():
                if not j['valid']:continue
                bank='context' if family.startswith('context') else 'moment'
                z=arrays[bank+'__normalized'][arrays['fit_indices']];observed=np.cov(z.T)
                model=arrays[family+'__model_covariance'];mask=~np.eye(len(model),dtype=bool)
                misfit=float(np.linalg.norm((observed-model)[mask])/max(np.linalg.norm(observed[mask]),1e-12))
                assert abs(misfit-j['relative_offdiag_misfit'])<1e-12;offdiag_checks+=1
                sizes=meta['groupings'][family]['group_sizes']
                fit_geometry.setdefault(family,[]).append({'uid':rec['uid'],'smallest_group':min(sizes),
                    'group_sizes':sizes,'jacobian_condition':j['jacobian']['condition_number'],'offdiag_misfit':misfit})
    geometry_audit={'status':'PASS','offdiag_checks':offdiag_checks,'records':fit_geometry,
        'scores_sha256':sha(OUT/'SCORES_FROZEN.json'),'main_review_sha256':sha(OUT/'REVIEW.json'),
        'renderer_sha256':sha(__file__)}
    (OUT/'FIT_GEOMETRY_AUDIT.json').write_text(json.dumps(geometry_audit,indent=2),encoding='utf-8')
    key_pairs=['context__equal minus moment__equal','context__iu minus moment__iu',
        'context__joint0 minus moment__joint0','context__graph010 minus moment__graph010',
        'context__joint0 minus context__iu','context__graph010 minus context__joint0',
        'context__graph010 minus context__graph_perm','context_allk__joint0 minus context__joint0']
    lines=['# Context features change Joint coverage and localization','',
        'Completed development pilot, 2026-09-07. Same 58 cached answers, 17 arms, 31 paired contrasts. IU-PCR / Joint L-SML are still the fusion cores. All six parent controls reproduce their original scores and 12 task endpoints.','',
        'The context bank raises the observed ProcessBench result for Joint lambda-zero from 12.50% to 27.43%, and valid fits from 43/58 to 50/58. It also loses some previously valid fits, performs worse on the four common valid PRMB answers, and does not establish a learned-fusion advantage over its simple control. There is no consistent two-task winner.','',
        '## What changed','',
        table(['Factor','Parent','New comparison'],[
            ['Feature definitions','Nine primitive streams x {window mean, SD, slope}','Same nine streams x {window mean, mean EMA8, mean EMA32}'],
            ['Nominal P / fitting windows','27 features / original non-overlapping eight-token windows','27 features / exactly the same fitting and scoring windows'],
            ['Grouping search','K in {3,4,6,8}','Also test all K from 3 to floor(active P / 3)'],
            ['Fusion controls','Equal, IU, Joint native inverse at lambda 0, graph 0.1, permuted graph','Same cores and numerical validity rules'],
            ['Primary decision','Answer-only GMM gate, then peak official step','Same rule fitted to the new score'],
            ['Gate diagnostic','One frozen parent IU binary decision per answer','Use that identical gate for every bank and core']]),'',
        'EMA is an existing project idea. We reuse its linear update from the causal DSP work, initialize it with the first observed value, and import none of that historical pipeline\'s label-selected rosters, signs or fitted references. All nine streams remain. The feature definitions are fixed engineering choices, not a learned optimum. The declared negative-entropy sign anchor remains.','',
        'The new features bring trajectory context into the feature matrix before fusion. They do not add independent observations or constitute a separately learned second trajectory-fusion stage. Full-answer fitting and step decisions remain offline, using the existing trace with no new inference or other-answer fit.','',
        '## Primary results and the gate diagnostic','',
        'PB macro-F1 (%) always uses all 46 PB answers, including failed fits. PRMB AUROCs use available valid answers; see the common-ID comparisons below before comparing different coverage. The common-IU-gate column is diagnostic, not a newly selected candidate.','',
        table(['Arm','Valid fits / 58','PRMB valid answers','PRMB pooled AUC','PRMB mean within-answer AUC','PB native gate %','PB common IU gate %'],[
            [arm,a['coverage'].get(arm,0),v['prm']['answers'],fmt(v['prm']['auroc']),fmt(v['prm']['within_answer_auc']),percent(v['pb']['macro_f1']),percent(v['pb_common_iu_gate']['macro_f1'])] for arm,v in m.items()]),'',
        'Context equal fusion reaches PB 27.67%, context IU 16.97%, context Joint 27.43%, and context Joint graph 20.19%. A larger PB score for a Joint variant does not by itself demonstrate that learning its weights helped.','',
        'Descriptive matched simple controls within each bank: all methods below use that bank\'s same Joint-valid PRMB answers. Different rows of this table still have different answer populations. This panel is an attribution diagnostic; the 31 registered intervals are saved separately.','',
        table(['Bank','Common PRMB answers','Equal','IU','Joint lambda 0','Graph lambda 0.1','Permuted graph'],[
            [bank,p['common_prm_answers']]+[fmt(p['auc'][bank+'__'+core]) for core in ('equal','iu','joint0','graph010','graph_perm')]
            for bank,p in a['descriptive_matched_simple_controls'].items()]),'',
        '## Coverage is not a simple increase','',
        table(['Population','Valid in both banks','Rescued by context','Lost with context','Invalid in both'],[
            [task]+[v[k] for k in ('both_valid','rescued','lost','both_invalid')] for task,v in a['coverage_transitions'].items()]),'',
        'Joint gains 13 valid answers and loses six, leaving 50 instead of 43. On PRMB, its old seven and new nine valid answers overlap in only four. The common-ID Joint comparison is 0.66823 (context) versus 0.72613 (moment), not the unmatched 0.63826 versus 0.66171. The exploratory difference interval is negative, but it rests on four development answers and is not a population-level confirmation.','',
        'Peak-only counts among the same 25 erroneous PB answers; these ignore the binary gate, and invalid fits still count as misses. They help separate a locator change from a change in clean/error decisions.','',
        table(['Arm','Exact peak hits / 25'],[[arm,a['pb_error_peak_hits'].get(arm,0)] for arm in manifest['arms']]),'',
        '## What the broader K search found','',
        table(['Bank / roster','Selected K counts, plus blocked cases'],[[family,str(v)] for family,v in a['group_counts'].items()]),'',
        'The expanded search selects K=5 in one moment-bank answer and three context-bank answers. K counts groups, not features: a three-group solution can contain many features per group. Every accepted group still has at least three coordinates. Larger K is an option, not a target or a guarantee of better localization.','',
        'The expanded moment roster leaves all reported endpoints unchanged. In the context bank it changes pooled Joint AUC from 0.63826 to 0.64808 and graph AUC from 0.64297 to 0.65161, with PB unchanged. These are small development differences; no K setting is selected by labels.','',
        'The context bank has more concentrated covariance. The mean participation rank of the normalized matrix is '+fmt(np.mean([r['participation_rank'] for r in a['geometry']['moment']]),2)+' for moments and '+fmt(np.mean([r['participation_rank'] for r in a['geometry']['context']]),2)+' for context. This measures concentration of covariance eigenvalues; it is not a literal count of independent features or correctness information. The nominal feature count remains 27.','',
        'Both banks retain 27 active coordinates for every answer; fitting-row counts range from 13 to 176. Valid-model geometry below is descriptive and uses each family\'s available fits. Its off-diagonal residuals are independently recomputed from the empirical and saved model covariances; Jacobian conditions were checked in the main review. Reused identical partitions are counted once per reported family.','',
        table(['Family','Valid models','Smallest group: min / median','Jacobian condition: median / max','Off-diagonal relative misfit: median [Q25, Q75]'],[
            [family,len(v),str(min(x['smallest_group'] for x in v))+' / '+fmt(np.median([x['smallest_group'] for x in v]),1),
             fmt(np.median([x['jacobian_condition'] for x in v]),1)+' / '+fmt(max(x['jacobian_condition'] for x in v),1),
             fmt(np.median([x['offdiag_misfit'] for x in v]),4)+' '+str([round(q,4) for q in np.quantile([x['offdiag_misfit'] for x in v],[.25,.75])])]
            for family,v in fit_geometry.items()]),'',
        '## Paired evidence','',
        'Exploratory, unadjusted 95% source-group intervals, 1,000 draws. PRMB uses common valid IDs; PB uses the fixed full population. Some bootstrap draws omit a class and remain undefined, with counts retained. Do not promote an isolated positive interval from this development grid.','',
        table(['Left minus right','Common PRMB N','PRMB difference [CI]','Within-answer difference [CI]','PB difference, percentage points [CI]'],[
            [key,c['pairs'][key]['left_prm']['answers'],
             fmt(c['pairs'][key]['left_prm']['auroc']-c['pairs'][key]['right_prm']['auroc'])+' '+ci(c['pairs'][key]['uncertainty']['prm_common_valid_ci95']),
             fmt(c['pairs'][key]['left_prm']['within_answer_auc']-c['pairs'][key]['right_prm']['within_answer_auc'])+' '+ci(c['pairs'][key]['uncertainty']['prm_within_answer_common_valid_ci95']),
             percent(c['pairs'][key]['left_pb']['macro_f1']-c['pairs'][key]['right_pb']['macro_f1'])+' '+ci(c['pairs'][key]['uncertainty']['pb_all_population_ci95'],100)] for key in key_pairs]),'',
        'The context graph has a small positive within-answer interval versus context Joint lambda-zero, but its native PB score is lower, and its within-answer comparison to the permuted graph includes zero. This does not establish a learned-graph advantage on both tasks.','',
        '## The next short experiment','',
        'Test an explicit fallback policy using the existing fusion fits: retain moment-bank Joint when it is valid; try context-bank Joint when the first fit fails; use answer-only IU if both fail. Keep pure Joint rows with their failures visible. Compare against moment Joint with the same IU fallback and against plain IU. Include equal aggregation under the same selected-bank routing to measure whether learned fusion adds value.','',
        'This is a proposed, not yet implemented policy. Freeze eligibility, routing, gate handling and comparator budgets before evaluating it. It is motivated by 13 rescued fits and six regressions; it must not become a label-chosen winner per answer. All fitting remains within the current answer. It could make the comparison cover every answer, but improved accuracy is unproven.','',
        'Keep the two banks as separate evidence rather than replacing the old one. Feature information beyond repeated telemetry transforms, remaining temporal/geometry/sampling support, the full comparator replay, untouched two-task confirmation and historical 24-cell transfer remain open.','',
        '## Review and execution','',
        'Six scientific tests pass, including a cached Joint/permuted-graph replay. The pre-freeze smoke test caught an omitted release prefix in the graph permutation identity; it was fixed before scoring and is now guarded by an exact parent-seed assertion. A floating-point reduction-order difference was also fixed before freezing so the unchanged level columns replay bit for bit. No parent artifact was changed.','',
        table(['Independent check','Count'],[[k,v] for k,v in a['counts'].items()]),'',
        'The review independently reconstructs EMA features with a linear filter, normalization, grouping admissibility/ARI selection, native inverse weights, span mappings, GMM decisions, labels and endpoints. It reuses the unchanged source-bound nearest-neighbor graph builder and IU kernel; the Laplacian, inverse and metrics are separately reconstructed. Numerical checks establish implementation consistency, not scientific success.','',
        f"Scoring: {load(OUT/'RUN_STATE.json')['seconds']:.1f} seconds with three CPU workers. All 31 contrasts: {c['seconds_this_invocation']:.1f} seconds using cached pair-count statistics on one process. The review additionally replayed three representative 1,000-draw contrasts with the old explicit implementation; all original pooled-PRMB/PB intervals and valid-draw counts match to 1e-12. That reference check took {sum(v['explicit_seconds'] for v in a['bootstrap_reference_checks']):.1f} seconds. These different workloads are not a direct speed ratio.",'',
        'Maximum raw-feature, weight and span discrepancies: '+', '.join(f"{a[k]:.3g}" for k in ('max_raw_feature_error','max_weight_error','max_span_error'))+f'. The separate geometry audit verifies {offdiag_checks} covariance residuals. All bound source and result hashes match. Scores froze before this stage decoded labels; these answers were already exposed in development. All scoring, contrasts and reviews are complete.','',
        'Historical context: the earlier 30-long-answer IU 0.70070 and Claude pooled-fit results use different populations and fitting contracts. The six current parents provide the exact bridge here. A new bank is not a new benchmark release, and this pilot is not publication confirmation.','']
    markdown='\n'.join(lines);(OUT/'REPORT.md').write_text(markdown,encoding='utf-8')
    body=[];in_table=False
    for line in lines:
        if line.startswith('|'):
            if not in_table:body.append('<div class="scroll"><table>');in_table=True;header=True
            if not line.replace('|','').replace('-','').strip():continue
            tag='th' if header else 'td';body.append('<tr>'+''.join(f'<{tag}>{html.escape(v.strip())}</{tag}>' for v in line.strip('|').split('|'))+'</tr>');header=False
        else:
            if in_table:body.append('</table></div>');in_table=False
            if line.startswith('# '):body.append('<h1>'+html.escape(line[2:])+'</h1>')
            elif line.startswith('## '):body.append('<h2>'+html.escape(line[3:])+'</h2>')
            elif line:body.append('<p>'+html.escape(line)+'</p>')
    if in_table:body.append('</table></div>')
    diagram='''<section class="visual"><h2>Same fusion, two feature banks</h2><div class="pipeline"><div class="card">One answer<br><strong>N fixed windows</strong><br>9 primitive streams</div><div class="card">Moment bank<br><strong>mean / SD / slope</strong><hr>Context bank<br><strong>level / EMA8 / EMA32</strong></div><div class="card">Fit within this answer<br><strong>IU-PCR / Joint L-SML</strong><br>Joint: test both K rosters</div><div class="card">Fused trajectory<br><strong>step + no-error decision</strong><br>Show gate and locator separately</div></div><p>Two controlled factors: the feature definitions and the grouping search. Graph and permutation controls stay beside the native Joint inverse.</p></section>'''
    index=next(i for i,v in enumerate(body) if v.startswith('<h2>What changed'))
    body.insert(index,diagram)
    html_text='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Fusion context-bank pilot</title><style>
body{margin:0;background:#f4f7fb;color:#152c43;font:17px/1.6 system-ui,sans-serif}main{max-width:1250px;margin:auto;padding:30px 24px 70px}h1{font-size:2.2rem;line-height:1.2}h2{line-height:1.3;margin-top:2em}p{max-width:100ch}.scroll{overflow:auto}table{border-collapse:collapse;background:white;width:100%;font-size:.88rem}td,th{padding:10px;border:1px solid #d4dfed;text-align:left}th{background:#e2edf7}tr:nth-child(even){background:#f8fafc}.visual{border:2px solid #8aaecd;background:white;border-radius:12px;padding:22px;margin:28px 0}.visual h2{margin-top:0}.pipeline{display:grid;grid-template-columns:repeat(4,1fr);gap:15px}.card{background:#edf4fa;padding:15px;border-left:4px solid #286b9a}a{color:#12639b}@media(max-width:750px){.pipeline{grid-template-columns:1fr 1fr}main{padding:17px}}@media print{body{background:white}.visual{break-inside:avoid}}</style></head><body><main>'''
    html_text+='\n'.join(body)+'''<p><a href="CONTRASTS.json">All 31 paired contrasts</a> | <a href="REVIEW.json">Independent review</a> | <a href="../fusion_gate_interface_audit_v1/REPORT.html">Previous gate audit</a> | <a href="../../docs/reviews/joint_lsml_visual_guide_2026-09-06.html">Joint visual guide</a></p></main></body></html>'''
    (OUT/'REPORT.html').write_text(html_text,encoding='utf-8')
    provenance={'evaluation_sha256':sha(OUT/'EVALUATION.json'),'contrasts_sha256':sha(OUT/'CONTRASTS.json'),
        'review_sha256':sha(OUT/'REVIEW.json'),'fit_geometry_audit_sha256':sha(OUT/'FIT_GEOMETRY_AUDIT.json'),'renderer_sha256':sha(__file__),
        'markdown_sha256':sha(OUT/'REPORT.md'),'html_sha256':sha(OUT/'REPORT.html')}
    (OUT/'REPORT_PROVENANCE.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print('Reviewed HTML / Markdown report and provenance written.')


if __name__=='__main__':render()
