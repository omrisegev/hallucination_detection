"""Render the grouping correction and its limits in simple English."""
from html import escape
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/localization_source_group_audit_v1'
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def interval(v,scale=1):return '['+', '.join(f'{scale*x:+.4f}' for x in v)+']'
def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+escape(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join(
        '<tr>'+''.join('<td>'+escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def markdown_table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join('---' for _ in headers)+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])


def render():
    a,r,c,e=[load(OUT/n) for n in ('AUDIT.json','REVIEW.json','CONTRASTS_V2.json','EVALUATION_V2.json')]
    assert r['status']=='PASS' and c['status']=='COMPLETE'
    assert r['review_script_sha256']==sha(ROOT/'scripts/review_localization_source_groups_v2.py')
    for p,h in r['hashes'].items():assert sha(p)==h
    body=[];md=[]
    def section(title,paragraphs=(),headers=None,rows=None):
        body.append('<section><h2>'+escape(title)+'</h2>'+''.join('<p>'+escape(x)+'</p>' for x in paragraphs)+(table(headers,rows) if headers else '')+'</section>')
        md.extend(['## '+title,'',*list(paragraphs),''])
        if headers:md.extend([markdown_table(headers,rows),''])
    title='A necessary benchmark correction: keep source questions together'
    intro='The exposure audit found that different versions of the same question were assigned to different evaluation groups. '
    intro+='We created a corrected release and folds, and recomputed uncertainty for the existing answer-only pilot. No fusion winner is established.'
    body.append('<header><p class="eyebrow">Benchmark integrity / 07 September 2026</p><h1>'+title+'</h1><p>'+intro+'</p></header>')
    md.extend(['# '+title,'',intro,''])
    section('What was wrong',[
        'PRMB source_idx identifies a perturbation record. Names such as confidence_prm_train_p1_7 and circular_prm_train_p1_7 can describe versions of the same source question. '
        'The previous grouping kept these names separate. ProcessBench also has different answer IDs with identical problem text.',
        'These are groups of source questions used to evaluate experiments. Joint feature groups are a separate part of the fusion method; their fitted values are unchanged by this repair.'],
        ['Local cache','Answer rows','Old groups','Corrected source-question groups','Groups crossing old outer folds'],[
            ['PRMB / Qwen3-8B',a['rows'],a['old_groups'],a['canonical_components'],a['canonical_groups_spanning_outer_folds']],
            ['PB / each of Qwen3-4B and 8B',a['pb_rows'],a['pb_rows'],a['pb_exact_question_groups'],a['pb_repeated_groups_spanning_outer_folds']]])
    body.append('<section><h2>How the correction works</h2><div class="flow"><div class="card"><h3>One source question</h3><p>A base reasoning problem may have several answers or deliberate modifications.</p></div><div class="card"><h3>Find the family</h3><p>PRMB: source-seed suffix plus identical question text. PB: identical problem text. Link shared text across both tasks.</p></div><div class="card"><h3>Keep it together</h3><p>All linked versions share one source group, one outer fold and one inner-fold assignment.</p></div></div><p>Only whitespace is collapsed for text matching. Numbers and mathematical notation are preserved. This does not identify every paraphrase.</p></section>')
    section('Evidence from the actual saved data',[
        f"PRMB contains {a['source_seeds']} source-seed IDs. Exact question matches join some seeds into {a['canonical_components']} components. "
        f"There are {a['identical_question_hashes_spanning_old_groups']} distinct question hashes occurring under multiple old groups.",
        f"Direct text evidence: {r['prmb_identical_question_hashes_crossing_old_folds']} identical PRMB question hashes cross old outer folds. "
        'This observation does not depend on interpreting the source-seed suffix.',
        f"PB has {a['pb_repeated_question_groups']} repeated-question groups covering {a['pb_repeated_question_rows']} answers. "
        f"{a['pb_repeated_groups_spanning_outer_folds']} of those groups cross old folds. No identical PB question crosses its four subsets in these caches.",
        f"There are {a['pb_prmb_identical_question_hashes']} exact whitespace-normalized question hashes shared by PB and PRMB. The corrected global folds keep them aligned across tasks.",
        'The metadata reader skips NumPy payloads instead of constructing the large telemetry arrays. It reads IDs and question text from the nine local pickle files. '
        'Correctness fields exist in those containers but are not used to construct groups. All 3,400 PB question texts match exactly between the 4b and 8b caches.'])
    body.append('<p class="source">The <a href="https://github.com/ssmisya/PRMBench#-data-format-for-prmbench">official PRMB data format</a> documents original/modified questions and perturbation-prefixed IDs. '
                'The counts above come from our frozen local cache, not the current remote dataset.</p>')
    md.extend(['Official schema: https://github.com/ssmisya/PRMBench#-data-format-for-prmbench . Counts are from the frozen local cache.',''])
    section('What this means for the experiments',[
        'Claude v2 fitted and selected methods across multiple answers using the old folds. Every corrected PRMB component appears in more than one old outer fold. '
        'Those results do not establish performance on unseen source questions. The amount and direction of any score bias have not been measured.',
        'Changing group names in an existing prediction file cannot repair those trained fits. We prepared corrected folds; the relevant multi-answer fits, configuration selections and predictions still need to be recomputed.',
        'Our recent window experiments fit each answer independently. Their frozen scores and predictions remain valid for those answers. '
        'Their source-group uncertainty and disjointness claims require correction. Twelve PRMB pilot answers form 11 corrected groups, and two of these groups overlap earlier short-cycle cohorts. '
        'The 46 PB pilot answers remain 46 distinct groups.',
        'The existing cache was already evaluated by v2. A new sample from it is development replication. Truly untouched publication confirmation still needs a separate exposure audit and data source.'])
    rows=[]
    for arm,label in [('moment__iu','Original IU'),('single__joint0','Original Joint -> IU'),('context__equal','Context equal fusion')]:
        m=e['metrics'][arm];rows.append([label,f"{m['prm']['auroc']:.5f}",f"{m['prm']['within_answer_auc']:.5f}",f"{100*m['pb']['macro_f1']:.2f}%"])
    section('All 25 point-metric bundles are unchanged',[
        'The score bridge reuses the same 58 answers, all scores, targets, validity flags and predictions. Only the resampling group IDs change. '
        'The matched table still gives no clear winner: Joint -> IU has promising points over original IU, but context equal has higher headline points.'],
        ['Method','PRMB pooled AUC','Within-answer AUC','PB macro F1'],rows)
    cis=[]
    for key in ['single__joint0 minus moment__iu','single__joint0 minus context__equal','dual__graph010 minus dual__joint0']:
        p=c['pairs'][key]
        for endpoint,label,scale in [('prm_common_valid_ci95','PRMB AUC',1),('pb_all_population_ci95','PB percentage points',100)]:
            cis.append([key,label,interval(p['old_uncertainty'][endpoint],scale),interval(p['corrected_uncertainty'][endpoint],scale)])
    section('Uncertainty recomputed under corrected source groups',[
        'All 32 bridge comparisons are complete: the 30 original registered pairs and the two previously added context-equal comparisons. '
        'The selected intervals below still include zero. These are retrospective, unadjusted 95% percentile intervals from 1,000 source-group draws.',
        'The existing bootstrap draws PRMB before PB from one random generator. Changing the PRMB group count changes subsequent PB draws. '
        'Small PB interval changes here reflect that Monte Carlo effect; the 46 PB pilot groups themselves remain distinct.'],
        ['Comparison','Endpoint','Old interval','Corrected interval'],cis)
    section('What has been delivered',[
        'A new immutable release, localization-cached-v2-sourcegroups-20260907, keeps the original rows and raw telemetry hashes and stores both corrected and legacy group IDs.',
        'New five-outer / five-inner assignments preserve source-question isolation across all benchmark/model cells. They have been checked, but multi-answer methods have not yet been refitted on them.',
        'The exposure inventory excludes 94 corrected components represented in the documented earlier Codex short cycles and 58-answer pilot. '
        'This is an exclusion list for further development, not a list of all exposure in the project.',
        'No previous release, frozen score file, or Claude worktree was modified. The fixed-recipe replication prototype passed two tests, including replays of all 19 retained arms '
        'in three prior routing cases, but no new replication cohort or experiment has been launched.'])
    checks=[[key,r[key]] for key in ['metadata_identity_rows','source_pickle_hashes','release_rows_checked','fold_isolation_checks',
                                    'score_target_decision_row_replays','unchanged_independent_metric_bundles']]
    section('Independent review',[
        'A separate reviewer rebuilds source components with a sparse graph algorithm, checks every corrected release row and verifies fold isolation. '
        'It verifies the original telemetry and label file hashes, all nine metadata-source pickle hashes, and the unchanged 25 metric bundles.',
        'Three corrected 1,000-draw comparisons are independently reproduced by explicitly resampling rows. All four endpoint intervals and defined-draw counts match. '
        'Two metadata-decoder tests compare the extracted strings with standard pickle across protocols 4 and 5, shared references and large binary frames.'],
        ['Check','Count'],checks)
    section('Next action',[
        'Use the corrected identity map to freeze a small source-disjoint development replication. Retain IU, both equal-fusion banks, Joint with explicit IU fallback, and graph lambda-zero/permuted controls. '
        'The grouping correction must be applied at data loading before reusing any multi-answer fitting code.',
        'Keep the planned rerun of relevant Claude contenders on corrected folds in the benchmark queue. Do not present the old grouped-CV numbers as source-question-disjoint evidence.',
        'Claude\'s latest report also proposes minimum feature-group size two. That is a separate Joint identifiability question to audit before changing the existing minimum-three recipe. '
        'Joint features/graphs, IU improvements, temporal/geometry/sampling support, the wider comparator panel and the historical 24-cell transfer remain active research work.'])
    links=[('Frozen repair protocol','../../docs/experiments/LOCALIZATION_SOURCE_GROUP_REPAIR_V2.md'),('Corrected release','RELEASE_V2.json'),
           ('Corrected folds, not yet fitted','FOLDS_V2.json'),('Audit evidence','AUDIT.json'),('32 interval bridges','CONTRASTS_V2.json'),
           ('Independent review','REVIEW.json'),('Previous fusion experiment','../fusion_explicit_fallback_pilot_v1/REPORT.html')]
    body.append('<footer>'+''.join('<p><a href="'+escape(url)+'">'+escape(label)+'</a></p>' for label,url in links)+'</footer>')
    md.extend([*['- ['+label+']('+url+')' for label,url in links],''])
    style='body{margin:0;background:#f4f7f7;color:#173640;font:17px/1.65 system-ui,sans-serif}main{max-width:1140px;margin:auto;padding:24px}header{background:#173d48;color:white;padding:32px;border-radius:16px}h1{font-size:clamp(30px,4.5vw,45px);line-height:1.15}h2{font-size:26px;line-height:1.3}section{padding:24px;margin:24px 0;background:white;border:1px solid #d4e1df;border-radius:14px}.eyebrow{letter-spacing:2px;font-size:12px;text-transform:uppercase}.flow{display:grid;grid-template-columns:repeat(3,1fr);gap:15px}.card{padding:18px;background:#eaf6f0;border:2px solid #138070;border-radius:10px}.card h3{margin-top:0}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:12px;text-align:left;border-bottom:1px solid #d9e4e2;vertical-align:top}th{background:#eaf1ef}a{color:#086d88}.source,footer{font-size:14px}footer{padding:20px}@media(max-width:700px){main{padding:12px}header,section{padding:18px}.flow{grid-template-columns:1fr}}@media print{body{background:white;font-size:11pt}header{background:white;color:#173640}section{border:0;padding:5px}.scroll{overflow:visible}table{font-size:8pt}h2{break-after:avoid}}'
    html='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><style>'+style+'</style></head><body><main>'+''.join(body)+'</main></body></html>'
    (OUT/'REPORT.html').write_text(html,encoding='utf-8');(OUT/'REPORT.md').write_text('\n'.join(md),encoding='utf-8',newline='\n')
    hashes={str(OUT/n):sha(OUT/n) for n in ('AUDIT.json','REVIEW.json','CONTRASTS_V2.json','EVALUATION_V2.json','REPORT.html','REPORT.md')}
    hashes[str(Path(__file__))]=sha(__file__)
    (OUT/'REPORT_PROVENANCE.json').write_text(json.dumps({'hashes':hashes},indent=2),encoding='utf-8')
    print('Source-group correction report rendered.')


if __name__=='__main__':render()
