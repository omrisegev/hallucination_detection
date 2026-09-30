"""Independent numerical review and evidence-based report for frozen readouts."""
import argparse
from collections import Counter
import hashlib
import html
import json
import re
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/fused_trajectory_readout_pilot_v1'
PARENT=ROOT/'results/answer_localization_representation_pilot_v1'


def load(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value): Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def review():
    manifest,frozen,evaluation=[load(OUT/name) for name in ('MANIFEST.json','SCORES_FROZEN.json','EVALUATION.json')]
    for p,h in {**manifest['hashes'],**frozen['files']}.items(): assert sha(p)==h,p
    assert frozen['manifest_sha256']==sha(OUT/'MANIFEST.json')
    assert evaluation['scores_sha256']==sha(OUT/'SCORES_FROZEN.json')
    rows=evaluation['rows']; cores=manifest['cores']; readouts=manifest['readouts']
    replay_count=gate_checks=mapping_checks=filter_checks=label_joins=0
    invalid=Counter(); coverage=Counter(); prediction_counts={}; ceilings={}
    release=load(PARENT/'RELEASE.json')
    for cell in sorted({r['cell'] for r in rows}):
        info=release['cells'][cell]
        assert sha(info['label_path'])==info['label_opaque_sha256']
        with np.load(info['label_path'],allow_pickle=False) as labels:
            for row in (r for r in rows if r['cell']==cell):
                index=np.flatnonzero(labels['row_ids']==row['row_id'])
                assert len(index)==1; i=int(index[0])
                if cell.startswith('prm'):
                    a,b=labels['step_flag_offsets'][i:i+2]; target=labels['step_error_flags'][a:b]
                else: target=labels['first_error'][i]
                np.testing.assert_array_equal(target,row['target']);label_joins+=1
    for row in rows:
        uid=row['uid']; meta=load(OUT/'scores'/f'{uid}.json'); parent=load(PARENT/'scores'/f'{uid}.json')
        with np.load(OUT/'scores'/f'{uid}.npz',allow_pickle=False) as arrays, np.load(PARENT/'scores'/f'{uid}.npz',allow_pickle=False) as old:
            for core in cores:
                parent_method=parent['report']['methods'][core]
                for readout in readouts:
                    name=core+'@@'+readout; detail=meta['methods'][name]
                    if not detail['valid']:
                        invalid[detail['reason']]+=1
                        assert not row['valid'][name] and not row['decision_valid'][name]
                        assert row['predictions'][name] is None
                        continue
                    coverage[name]+=1
                    assert (detail['prediction']!=-1)==(parent_method['readout']['prediction']!=-1)
                    assert row['predictions'][name]==detail['prediction'];gate_checks+=1
                    risk,onset=arrays[name+'__risk'],arrays[name+'__onset']
                    np.testing.assert_array_equal(risk,row['scores'][name])
                    if readout=='parent_first':
                        np.testing.assert_array_equal(risk,old[core+'__step'])
                        assert detail['prediction']==parent_method['readout']['prediction'];replay_count+=1
                    elif detail['gate']:
                        assert detail['prediction']==int(np.argmax(onset))
                    if readout.startswith('parent'): continue
                    for which in ('risk','onset'):
                        curve=arrays[name+'__window_'+which]
                        # Independent token-index calculation (no repeat/pad helper).
                        token_curve=curve[np.minimum(np.arange(row['tokens'])//8,len(curve)-1)]
                        mapped=[max(token_curve[a:b]) for a,b in zip(arrays['step_starts'],arrays['step_ends'])]
                        np.testing.assert_allclose(mapped,arrays[name+'__'+which],atol=1e-12);mapping_checks+=1
                    if readout in ('imm_level','kalman_level'):
                        assert np.all(arrays[name+'__variance']>0)
                        np.testing.assert_allclose(arrays[name+'__mode_probability'].sum(axis=1),1.,atol=1e-12)
                        assert np.isfinite(arrays[name+'__log_predictive']).all();filter_checks+=1
                    if readout=='bocpd_rise':
                        p=arrays[name+'__reset_probability'];assert np.all((p>=0)&(p<=1))
                        assert np.isfinite(arrays[name+'__log_predictive']).all();filter_checks+=1
                    if readout=='hmm_entry':
                        model=detail['selected']
                        assert model['valid'] and model['n_iter_used']<120
                        assert model['means'][1]>model['means'][0]
                        assert np.all((risk>=0)&(risk<=1)) and np.all((onset>=0)&(onset<=1))
    # Pairwise positive-vs-negative rank probability, independent of sklearn.
    for name,metric in evaluation['metrics'].items():
        prm=[r for r in rows if r['cell'].startswith('prm') and r['valid'][name]]
        if prm:
            labels=np.concatenate([r['target'] for r in prm]); scores=np.concatenate([r['scores'][name] for r in prm])
            pos,neg=scores[labels==1],scores[labels==0]
            if len(pos) and len(neg):
                auc=float(np.mean(pos[:,None]>neg[None,:])+.5*np.mean(pos[:,None]==neg[None,:]))
                assert abs(auc-metric['prm']['auroc'])<1e-12
        f1s=[]
        for cell,expected in metric['pb']['cells'].items():
            hits={True:[],False:[]}
            for r in (r for r in rows if r['cell']==cell):
                hits[r['target']==-1].append(int(r['decision_valid'][name] and r['predictions'][name]==r['target']))
            ca,ea=np.mean(hits[True]),np.mean(hits[False])
            f1=2*ca*ea/(ca+ea) if ca+ea else 0.
            assert abs(f1-expected['f1'])<1e-12;f1s.append(f1)
        assert abs(np.mean(f1s)-metric['pb']['macro_f1'])<1e-12
        prediction_counts[name]=dict(Counter(str(r['predictions'][name]) for r in rows if r['cell'].startswith('pb_')))
    # A perfect locator cannot rescue errors rejected by the frozen gate.
    for core in cores:
        name=core+'@@parent_first'; cell_results={}
        for cell in sorted({r['cell'] for r in rows if r['cell'].startswith('pb_')}):
            subset=[r for r in rows if r['cell']==cell]
            clean=[r for r in subset if r['target']==-1]; error=[r for r in subset if r['target']!=-1]
            ca=sum(r['valid'][name] and r['predictions'][name]==-1 for r in clean)/len(clean)
            eligible=sum(r['valid'][name] and r['predictions'][name]!=-1 for r in error)
            ea=eligible/len(error);f1=2*ca*ea/(ca+ea) if ca+ea else 0.
            cell_results[cell]={'clean_accuracy':ca,'eligible_errors':eligible,'all_errors':len(error),'max_f1':f1}
        ceilings[core]={'cells':cell_results,'macro_ceiling':float(np.mean([d['max_f1'] for d in cell_results.values()]))}
    audit={'status':'PASS','parent_replays':replay_count,'gate_invariance_checks':gate_checks,
           'span_replays':mapping_checks,'filter_distribution_checks':filter_checks,'direct_label_joins':label_joins,
           'endpoint_checks':len(evaluation['metrics']),'coverage':dict(coverage),'invalid_reasons':dict(invalid),
           'pb_prediction_counts':prediction_counts,'frozen_gate_ceilings':ceilings,
           'evaluation_sha256':sha(OUT/'EVALUATION.json'),'review_script_sha256':sha(__file__)}
    save(OUT/'REVIEW.json',audit)
    print(json.dumps({k:v for k,v in audit.items() if k not in ('pb_prediction_counts','coverage','frozen_gate_ceilings')},indent=2),flush=True)


def render():
    evaluation,audit,contrasts,manifest=[load(OUT/name) for name in ('EVALUATION.json','REVIEW.json','CONTRASTS.json','MANIFEST.json')]
    assert audit['evaluation_sha256']==contrasts['evaluation_sha256']==sha(OUT/'EVALUATION.json')
    assert contrasts['state']=='COMPLETE' and len(contrasts['pairs'])==58
    cores,readouts=manifest['cores'],manifest['readouts']
    short=lambda core: core.replace('moments27_local8__','')
    lines=['# Fused trajectory readout pilot v1','',
           '2026-09-07. Completed adaptive development pilot; fusion remains the core. No winner promoted.','',
           'A peak locator repairs part of the previous first-crossing failure. None of the tested chronological additions improves the strongest simple readout on both tasks. Learned fusion still has no demonstrated advantage over its simple controls on this small cohort.','',
           '## Scope and attribution','',
           'The exact same 58 answers (12 PRMB, 46 PB), feature fits, signs, weights, graphs and binary error gates are retained from the representation pilot. Six cores × seven readouts were frozen before this version was evaluated. The parent labels were already known: this is adaptive development, not fresh confirmation. The error gate stays fixed; this stage tests where to localize an accepted error, not a better clean/error classifier. All full-pipeline PB scores count failed fits/readouts as misses.','',
           'IU-PCR and Joint L-SML generate every proposed fused input. Entropy and equal averaging are controls. IMM, HMM and BOCPD process those saved scores; they are supporting components, not replacement detectors.','',
           '## ProcessBench macro F1 (%) — same 46-answer population','',
           '| Fusion core | First crossing | Raw peak | Hold peak | HMM entry | Kalman level | IMM level | BOCPD rise |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for core in cores:
        values=[100*evaluation['metrics'][core+'@@'+r]['pb']['macro_f1'] for r in readouts]
        lines.append('| '+short(core)+' | '+' | '.join(f'{v:.2f}' for v in values)+' |')
    lines += ['',
              'IU rises from 0 to 17.71% with the raw peak locator; equal fusion is 17.76% and entropy is 19.85%. The same IU gate and same PRMB ranking are retained. This is a readout repair, not evidence that IU learned superior fusion weights. IMM gives IU 14.79%, BOCPD 13.69%, and HMM 0%. Joint lambda-zero raw peak gives 12.50%, meaningful graph 8.33%, permutation 4.17%; their small-sample differences need the paired uncertainty below.','',
              '## PRMB step AUROC — availability is different','',
              'Each entry is AUROC (number of valid answers). Do not rank entries with different availability as a common-population leaderboard. HMM uses high-state occupancy for PRMB ranking and entry probability for PB onset; this distinction was declared before scoring.','',
              '| Core | Raw peak | Hold peak | HMM | Kalman | IMM | BOCPD |','|---|---:|---:|---:|---:|---:|---:|']
    for core in cores:
        values=[evaluation['metrics'][core+'@@'+r]['prm'] for r in readouts[1:]]
        lines.append('| '+short(core)+' | '+' | '.join(f"{d['auroc']:.5f} ({d['answers']})" if d['auroc'] is not None else 'unavailable' for d in values)+' |')
    lines += ['','Raw peak leaves original PRMB rankings untouched. The hold control drops the extra overlapping end window, so temporal additions must be compared with hold_peak as well. This mapping change alone reduces IU PRMB AUROC from 0.62261 to 0.60272, illustrating why implementation-level anchors matter.','',
              '## Paired contrasts','',
              'Exploratory, unadjusted, 1,000 source-group bootstrap draws. PRMB uses common valid IDs; PB uses all 46 answers with failed-readout penalties. Tiny or degenerate intervals cannot establish population equivalence. All 58 registered contrasts are retained in CONTRASTS.json.','',
              '| Left minus right | PRMB common N | PRMB delta [95% CI] | PB delta in percentage points [95% CI] |',
              '|---|---:|---|---|']
    selected=[]
    for name,pair in contrasts['pairs'].items():
        left,right=pair['left'],pair['right']
        if (left.startswith(cores[1]+'@@') and right.startswith(cores[1]+'@@')) or (left.endswith('@@parent_peak') and right.endswith('@@parent_peak')):
            selected.append((name,pair))
    def interval(ci,scale): return 'undefined' if ci is None else '['+', '.join(f'{scale*x:+.4f}' for x in ci)+']'
    for name,pair in selected:
        a,b=pair['left_prm']['auroc'],pair['right_prm']['auroc']; u=pair['uncertainty']
        delta=None if a is None or b is None else a-b
        pb=pair['left_pb']['macro_f1']-pair['right_pb']['macro_f1']
        prm='undefined' if delta is None else f'{delta:+.5f} '+interval(u['prm_common_valid_ci95'],1)
        pb_ci=u['pb_all_population_ci95']
        lines.append(f"| {name.replace('moments27_local8__','')} | {pair['left_prm']['answers']} | {prm} | {100*pb:+.2f} {interval(pb_ci,100)} |")
    lines += ['','## Gate limitation and availability','',
              'The following ceiling assumes a perfect locator but preserves every core’s frozen clean/error decision. It is an oracle diagnostic, not an achieved score or a supervised candidate. HMM failures can further reduce coverage below this ceiling.','',
              '| Core | Parent valid / 58 | HMM valid / 58 | PB F1 ceiling with perfect locator (%) |',
              '|---|---:|---:|---:|']
    for core in cores:
        lines.append(f"| {short(core)} | {audit['coverage'].get(core+'@@parent_first',0)} | {audit['coverage'].get(core+'@@hmm_entry',0)} | {100*audit['frozen_gate_ceilings'][core]['macro_ceiling']:.2f} |")
    lines += ['', 'On this cohort, even a perfect locator cannot take the present Joint graph gate above 28.24% PB F1, whereas the IU gate ceiling is 55.18%. Both fit availability and clean/error decisions constrain the result. Improving the locator alone cannot solve every failure. These ceilings depend on the pilot labels and are diagnostics only.']
    lines += ['','## Review and historical continuity','',
              f"Seven scientific-contract tests passed, including exact HMM state-path probabilities, exhaustive Gaussian partition evidence, IMM mixing covariance and the single-Kalman limit. Independent review verifies {audit['parent_replays']} exact parent replays, {audit['gate_invariance_checks']} unchanged error gates, {audit['span_replays']} span mappings, {audit['filter_distribution_checks']} filter probability/covariance checks, all 42 endpoints and 58 direct source-label joins. Source and score hashes match. Scoring took about 34 seconds on three CPU workers; bootstrap time is additional.",'',
              'HISTORY Step 246 already tested pooled token-level IU-HMM and found it worse than ordinary IU. This new single-answer window version also fails to improve PB in the tested form. It does not close all chronological models or Joint/graph feature development. The historical 30-long-answer IU 0.70070 and Claude’s pooled/calibrated PB figures belong to different cohorts/access contracts; parent_first replays the compatible current baseline exactly.','',
              'The old temporal_models BOCPD code mixes reset-before-observation prediction with an unupdated reset branch. Its claim that constant P(r=0) always indicates a bug is also too broad: the original paper uses an after-observation boundary convention. This pilot uses a consistent reset-before-observation Gaussian adaptation, verified against all short-sequence partitions. Historical numerical outputs remain intact.','',
              'Sources: [Adams and MacKay](https://arxiv.org/html/0710.3742v1), [FilterPy IMM](https://filterpy.readthedocs.io/en/latest/_modules/filterpy/kalman/IMM.html). The localization readouts, fixed parameters and boundary heuristic are our adaptations, not published hallucination detectors.','',
              '## Next bounded direction','',
              'Retain raw peak as the current simple localization control. Do not expand these losing temporal configurations into a full sweep. Next investigate the representation and reliability inputs to fusion, including label-free Joint regularization/stability and the user’s token/window sampling idea, with IU/equal and meaningful/permuted graph anchors. Separately address the within-answer clean/error gate; a different gate needs a new frozen version. KalmanNet, LOCA and flow-derived views remain supporting hypotheses. Full matched comparator coverage, untouched confirmation on both tasks and frozen-candidate 24-cell transfer remain open.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    save(OUT/'REPORT_PROVENANCE.json',{'evaluation_sha256':sha(OUT/'EVALUATION.json'),'contrasts_sha256':sha(OUT/'CONTRASTS.json'),
                                      'review_sha256':sha(OUT/'REVIEW.json'),'report_script_sha256':sha(__file__)})
    # A standalone, simple-English HTML companion; no external scripts/assets.
    def inline(text):
        chunks=[];end=0
        for match in re.finditer(r'\[([^\]]+)\]\((https?://[^)]+)\)',text):
            chunks.append(html.escape(text[end:match.start()]))
            chunks.append('<a href="'+html.escape(match[2],quote=True)+'">'+html.escape(match[1])+'</a>')
            end=match.end()
        chunks.append(html.escape(text[end:]));return ''.join(chunks)
    body=[];in_table=False
    for line in lines:
        if line.startswith('|---'): continue
        if line.startswith('|'):
            header=not in_table
            if header: body.append('<div class="scroll"><table>');in_table=True
            cells=[c.strip() for c in line.strip('|').split('|')]
            tag='th' if header else 'td'
            body.append('<tr>'+''.join('<'+tag+'>'+html.escape(c)+'</'+tag+'>' for c in cells)+'</tr>');continue
        if in_table: body.append('</table></div>');in_table=False
        if line.startswith('# '): body.append('<h1>'+html.escape(line[2:])+'</h1>')
        elif line.startswith('## '): body.append('<h2>'+html.escape(line[3:])+'</h2>')
        elif line: body.append('<p>'+inline(line)+'</p>')
    if in_table: body.append('</table></div>')
    page='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Fusion trajectory pilot</title><style>body{font:17px/1.6 system-ui;background:#f3f6f5;color:#193e46;margin:0}main{max-width:1100px;margin:auto;padding:32px}h1{font-size:38px;line-height:1.15}h2{margin-top:40px}p{max-width:950px}.scroll{overflow:auto}table{border-collapse:collapse;background:white;width:100%;font-size:14px}td,th{padding:12px;border:1px solid #d6e1df;text-align:left}tr:first-child{font-weight:bold;background:#dceee8}a{color:#075bb0}</style><main>'+''.join(body)+'</main></html>'
    (OUT/'REPORT.html').write_text(page,encoding='utf-8')
    print('REPORT.md and REPORT.html written; paired evidence complete.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=('review','report'),required=True)
    args=parser.parse_args();{'review':review,'report':render}[args.phase]()
