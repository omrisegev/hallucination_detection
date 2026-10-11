"""Build a portable advisor page from pinned branch CSVs and independent audits.

Run with the main repository Python environment. No inference, installs or Git writes.
All plotted values come from copied source files, with reported versus recomputed
evidence distinguished. SVG/PNG figures are also exportable separately.
"""
import argparse
import csv
import hashlib
import html
import json
import re
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HEAD = '080877eb9a8369341d647f4115a3b475c040b46e'
URL = 'https://github.com/omrisegev/hallucination_detection/blob/' + HEAD + '/'
ROSTER = ['ALL_EQ','DS_FILTER_EQ','LSML_CONT','IU_PCR','HEM_MOM_cc',
          'MEAN_MOM_cc','MEAN_MOM_ga','TCI_ONLY_ga']
NAMES = ['Plain average','DS-filter average','Continuous L-SML','IU-PCR',
         'HEM / cc','MEAN / cc','MEAN / ga','TCI / ga']
BANKS = ['N11','N13','N22','N40','N53','N14','N16','N25','N43','N56']
BLUE, ORANGE, GREY = '#246c9c', '#bf6235', '#737c85'


def rows(p):
    with p.open(newline='') as f:
        return list(csv.DictReader(f))


def link(path, label=None, line=None):
    return '<a href="'+URL+path+('' if line is None else '#L'+str(line))+'">'+html.escape(label or path)+'</a>'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--snapshot', type=Path, required=True)
    ap.add_argument('--audit-root', type=Path, default=Path('/private/tmp'))
    a = ap.parse_args()
    out = Path(__file__).resolve().parent
    evidence = out/'evidence'; evidence.mkdir(exist_ok=True)
    sources = out/'sources'; sources.mkdir(exist_ok=True)
    figs = out/'figures'; figs.mkdir(exist_ok=True)
    src = a.snapshot/'results/thinkfix_rescore_v1'
    for tag in ['A','B','C']:
        dest = evidence/tag; dest.mkdir(exist_ok=True)
        for p in (a.audit_root/f'thinkfix-audit-{tag}').iterdir():
            if p.is_file() and p.name not in ['per_answer.csv','null_draws.npz']:
                shutil.copyfile(p,dest/p.name)
    tocopy = ['PRMBENCH_COMPARISON.csv','PROCESSBENCH_COMPARISON.csv','COMPARISON_SUMMARY.json','README.md',
              'thinkfix_mask1/nested_banks/PANEL_A_ABSOLUTE.csv',
              'thinkfix_mask1/nested_banks/PANEL_B_PRMBENCH_TYPES.csv',
              'thinkfix_mask1/cont_baselines/ABSOLUTE.csv',
              'thinkfix_mask1/cont_baselines/PROTOCOL.json',
              'thinkfix_mask1/nested_banks/BANKS.json',
              'thinkfix_mask1/lr_reference/LR_SCORES.csv',
              'thinkfix_mask1/lr_reference/GAPS.csv',
              'external/EXTERNAL_PANEL_COMPARISON.csv','external/EXTERNAL_LSML_COMPARISON.csv',
              'selfgen/B16_SELFGEN_VS_PB_TF.csv','selfgen/B16_BEHAVIOUR_thinkfix_mask1_FINAL890.json',
              'prmscore_inputs/bank_N16.npz','prmscore_inputs/prmbench_meta.json']
    for rel in tocopy:
        dest = sources/rel; dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src/rel,dest)
    shutil.copyfile(src/'thinkfix_mask1/nested_banks/BANKS.json', sources/'BANKS.json')
    shutil.copyfile(src/'thinkfix_mask1/cont_baselines/PROTOCOL.json', sources/'CONT_BASELINES_PROTOCOL.json')
    for rel in ['docs/meetings/advisor_meeting_2026-10-08/README_HE.md',
                'docs/HANDOFF_prmbench_meeting_and_consolidation_2026-10-10.md',
                'docs/HANDOFF_thinkfix_rescore_2026-10-11.md',
                'docs/reviews/consolidation_20261003/sources/b16_eval.py']:
        dest=sources/Path(rel).name; shutil.copyfile(a.snapshot/rel,dest)
    shutil.copyfile(a.snapshot/'FETCH_MANIFEST.json',sources/'FETCH_MANIFEST.json')
    for p,name in [(a.audit_root/'prmbench-upstream-task.py','upstream_prmbench_task.py'),
                   (a.audit_root/'prmbench-upstream-parity.json','UPSTREAM_PARITY.json'),
                   (a.audit_root/'thinkfix-prm7b-manifest.json','PRM7B_MANIFEST.json')]:
        shutil.copyfile(p,sources/name)
    audit=json.loads((evidence/'A/audit.json').read_text())
    parity=json.loads((sources/'UPSTREAM_PARITY.json').read_text())
    prm=json.loads((sources/'PRM7B_MANIFEST.json').read_text())
    metrics = {b:{} for b in BANKS}
    for r in rows(src/'thinkfix_mask1/nested_banks/PANEL_A_ABSOLUTE.csv'):
        if r['scorer'] in ROSTER:
            metrics[r['bank']][r['scorer']]=float(r['prmb_within_auc'])
    for r in rows(src/'thinkfix_mask1/cont_baselines/ABSOLUTE.csv'):
        if r['scorer'] in ROSTER:
            metrics[r['bank']][r['scorer']]=float(r['prmb_within_auc'])
    lr={r['bank']:float(r['prmb_within_auc']) for r in rows(src/'thinkfix_mask1/lr_reference/LR_SCORES.csv') if r['scorer']=='LR'}
    for b in BANKS:
        assert set(metrics[b])==set(ROSTER)
    for s in ROSTER:
        assert abs(metrics['N16'][s]-audit['metrics'][s]['within_answer_auc'])<1e-12
    gaps=rows(src/'thinkfix_mask1/lr_reference/GAPS.csv')
    comp=rows(src/'PRMBENCH_COMPARISON.csv')
    corrections={}
    for r in comp:
        if r['bank']=='N16' and r['metric']=='prmb_within_auc' and r['scorer'] in ROSTER and r['table'] in ['panel_a','cont_baselines']:
            assert r['old_replay'] and r['thinkfix'] and r['thinkfix_mask1']
            corrections[r['scorer']]=[float(r[x]) for x in ['old_replay','thinkfix','thinkfix_mask1']]
    assert set(corrections)==set(ROSTER)
    replay=[]
    for fn in ['PRMBENCH_COMPARISON.csv','PROCESSBENCH_COMPARISON.csv']:
        for r in rows(src/fn):
            if r['replay_minus_frozen']:
                replay.append((abs(float(r['replay_minus_frozen'])),r))
    replay.sort(key=lambda p:p[0],reverse=True)
    (evidence/'REPLAY_LARGEST.json').write_text(json.dumps([dict(abs_difference=v,row=r) for v,r in replay[:15]],indent=2))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.edgecolor':'#b8c3cb','axes.labelcolor':'#344451',
                         'text.color':'#21323d','xtick.color':'#50616b','ytick.color':'#50616b',
                         'svg.fonttype':'none','figure.facecolor':'white','axes.facecolor':'white'})
    def save(fig,name):
        fig.savefig(figs/(name+'.svg'),bbox_inches='tight')
        fig.savefig(figs/(name+'.png'),bbox_inches='tight',dpi=180)
        plt.close(fig)
    def figure(name,caption):
        svg=(figs/(name+'.svg')).read_text(); svg=svg[svg.index('<svg'):]
        ids=re.findall(r'\bid="([^"]+)"',svg)
        for identifier in ids:
            svg=svg.replace('id="'+identifier+'"','id="'+name+'-'+identifier+'"')
            svg=svg.replace('#'+identifier+'"','#'+name+'-'+identifier+'"')
            svg=svg.replace('#'+identifier+')','#'+name+'-'+identifier+')')
        return '<figure>'+svg+'<figcaption>'+caption+'</figcaption></figure>'
    # Corrected versus old: three stages, never attribute the combined delta to one intervention.
    fig,ax=plt.subplots(figsize=(10.5,4.7))
    y=np.arange(8)
    for i,s in enumerate(ROSTER):
        v=corrections[s]
        ax.plot(v,[i]*3,color='#c9d3da',lw=2,zorder=1)
    for j,(col,label,marker) in enumerate([(GREY,'Old replay (missing block)','o'),(ORANGE,'Block corrected; unmasked','s'),(BLUE,'Block corrected + mask token 0','o')]):
        ax.scatter([corrections[s][j] for s in ROSTER],y,color=col,label=label,s=45,marker=marker,zorder=3)
    ax.set_yticks(y,NAMES);ax.invert_yaxis();ax.set_xlim(.758,.806)
    ax.set_xlabel('Mean within-answer AUROC · N16 · higher is better')
    ax.grid(axis='x',color='#edf0f2');ax.set_title('Correcting the input lowers the N16 scores',loc='left',fontweight='bold')
    ax.legend(loc='lower right',fontsize=8);save(fig,'prefix_correction')
    # Bank family and supervised reference.
    fig,axes=plt.subplots(1,2,figsize=(11,4.0),sharey=True)
    for ax,family,title in zip(axes,[BANKS[:5],BANKS[5:]],['Digit-free banks','Banks with digit channels']):
        x=np.arange(5)
        ax.plot(x,[metrics[b]['ALL_EQ'] for b in family],color=GREY,marker='o',label='Plain average')
        ax.plot(x,[metrics[b]['DS_FILTER_EQ'] for b in family],color=BLUE,marker='o',label='DS-filter average')
        ax.plot(x,[np.mean([metrics[b][s] for s in ROSTER]) for b in family],color=BLUE,ls='--',marker='.',label='Mean of 8 label-free scorers')
        ax.plot(x,[lr[b] for b in family],color=ORANGE,marker='s',label='Supervised LR (OOF, reported)')
        ax.set_xticks(x,family);ax.set_xlabel('Nested feature bank · channels increase →');ax.set_ylim(.72,.83)
        ax.grid(axis='y',color='#edf0f2');ax.set_title(title,loc='left',fontweight='bold')
    axes[0].set_ylabel('Mean within-answer AUROC')
    axes[1].legend(fontsize=8,loc='lower right');save(fig,'banks_headroom')
    fig,ax=plt.subplots(figsize=(10,3.7))
    order=['N11','N14','N13','N16','N22','N25','N40','N43','N53','N56']
    gm={r['bank']:r for r in gaps};xx=np.arange(10)
    point=np.array([float(gm[b]['gap_vs_best']) for b in order]);lo=np.array([float(gm[b]['gap_vs_best_lo95']) for b in order]);hi=np.array([float(gm[b]['gap_vs_best_hi95']) for b in order])
    ax.errorbar(xx,point,yerr=[point-lo,hi-point],fmt='o',color=ORANGE,capsize=4)
    ax.axhline(0,color=GREY,lw=1);ax.set_xticks(xx,order);ax.set_xlabel('Feature bank');ax.set_ylabel('LR − best of 4 pipeline variants\nAUROC difference (95% interval)')
    ax.set_title('The branch reports positive LR headroom in every bank',loc='left',fontweight='bold');save(fig,'lr_intervals')
    # Fixed scorer by type, no selecting a different winner in each category.
    types=['redundency','circular','domain_inconsistency','step_contradiction','missing_condition','confidence','counterfactual','deception']
    ours={r['group']:float(r['within_auc']) for r in rows(src/'thinkfix_mask1/nested_banks/PANEL_B_PRMBENCH_TYPES.csv') if r['bank']=='N16' and r['scorer']=='MEAN_MOM_ga'}
    # Comparator AUC values shown on linked advisor artifact; rounded, raw predictions absent.
    comparator=dict(zip(types,[.654,.763,.828,.852,.720,.923,.857,.812]))
    fig,ax=plt.subplots(figsize=(10,4.5));y=np.arange(8)
    for i,t in enumerate(types):ax.plot([ours[t],comparator[t]],[i,i],color='#c4cdd3',lw=2)
    ax.scatter([ours[t] for t in types],y,color=BLUE,label='N16 MEAN / ga · corrected CSV',s=45)
    ax.scatter([comparator[t] for t in types],y,color=ORANGE,label='Qwen2.5-Math-PRM-7B · artifact values',s=45,marker='s')
    ax.set_yticks(y,[t.replace('_',' ').replace('redundency','redundancy') for t in types]);ax.invert_yaxis()
    ax.set_xlabel('Mean within-answer AUROC · each error type');ax.set_xlim(.62,.96);ax.grid(axis='x',color='#edf0f2')
    ax.legend(fontsize=8,loc='lower right');ax.set_title('A fixed scorer is strongest on simplicity errors',loc='left',fontweight='bold');save(fig,'error_types')
    # Category F1 and clean answer behavior.
    cat=parity['by_class'];pcat=prm['results']['by_classification']
    vals={t:cat[t]['half_f1_plus_negative_f1'] for t in types}
    pvals={t:.5*(pcat['f1'][t]+pcat['negative_f1'][t]) for t in types}
    catgroups=[types[:2],['counterfactual','step_contradiction','domain_inconsistency','confidence'],['missing_condition','deception']]
    categoryours=[np.mean([vals[t] for t in g]) for g in catgroups]
    categoryprm=[np.mean([pvals[t] for t in g]) for g in catgroups]
    fig,axes=plt.subplots(1,2,figsize=(11,4),gridspec_kw={'width_ratios':[1.25,1]})
    ax=axes[0];x=np.arange(3);w=.34
    ax.bar(x-w/2,categoryours,w,color=BLUE,label='N16 MEAN / ga · global Q80')
    ax.bar(x+w/2,categoryprm,w,color=ORANGE,label='PRM-7B · threshold 0.5')
    for i,v in enumerate(categoryours):ax.text(i-w/2,v+.012,f'{v:.3f}',ha='center',fontsize=9)
    for i,v in enumerate(categoryprm):ax.text(i+w/2,v+.012,f'{v:.3f}',ha='center',fontsize=9)
    ax.set_xticks(x,['Simplicity','Soundness','Sensitivity*']);ax.set_ylim(0,.86);ax.set_ylabel('Mean subtype pooled macro-F1');ax.legend(fontsize=7,loc='upper left')
    ax.set_title('The gain is category-specific',loc='left',fontweight='bold')
    ax=axes[1];falseflag=audit['metrics']['MEAN_MOM_ga']['synthetic_controls']['q80_gt']['answer_false_flag_rate']
    ax.bar(['At least one\nfalse error flag','No flag\n(clean accepted)'],[falseflag,1-falseflag],color=[BLUE,GREY]);ax.set_ylim(0,1.12)
    ax.text(0,falseflag+.03,f'{falseflag:.1%}',ha='center',fontweight='bold');ax.text(1,1-falseflag+.03,f'{1-falseflag:.1%}',ha='center')
    ax.set_ylabel('Fraction of 758 synthetic correct answers');ax.set_title('Relative ranking does not abstain',loc='left',fontweight='bold');save(fig,'flags_categories')
    # External protocols separately, mixed benchmark metrics are labelled rather than averaged.
    ext=rows(src/'external/EXTERNAL_PANEL_COMPARISON.csv')
    picked=[r for r in ext if r['table']=='nested' and r['bank']=='N16' and r['scorer']=='MEAN_MOM_ga' and r['rule']=='ANSWER_Z_Q80' and r['measure']=='official_metric']
    assert len(picked)==2, picked
    lock=[r for r in rows(src/'external/EXTERNAL_LSML_COMPARISON.csv') if r['group']=='ALL' and r['measure']=='metric' and r['arm']=='frozen_lsml']
    assert len(lock)==2,lock
    fig,axes=plt.subplots(1,2,figsize=(11,3.5))
    for ax,data,title in zip(axes,[lock,picked],['Locked L-SML · pre-fix source bundle','Monday N16 MEAN / ga · refit on target cell']):
        data=sorted(data,key=lambda r:r['cell']);y=np.arange(2)
        for i,r in enumerate(data):ax.plot([float(r['old']),float(r['thinkfix_mask1'])],[i,i],color='#c4cdd3',lw=3)
        ax.scatter([float(r['old']) for r in data],y,color=GREY,label='Old',s=50)
        ax.scatter([float(r['thinkfix_mask1']) for r in data],y,color=BLUE,label='Corrected + mask1',s=50)
        for i,r in enumerate(data):ax.annotate(f"{float(r['old']):.3f} → {float(r['thinkfix_mask1']):.3f}",(float(r['thinkfix_mask1']),i),xytext=(0,14),textcoords='offset points',ha='center',fontsize=9)
        ax.set_yticks(y,['Hard2Verify\nbalanced F1','Socratic\npooled macro-F1']);ax.set_ylim(-.5,1.6);ax.set_xlim(.32,.69);ax.set_xlabel('Benchmark-specific score · no cross-benchmark average')
        ax.set_title(title,fontsize=10,loc='left',fontweight='bold');ax.legend(fontsize=8,loc='lower right')
    save(fig,'external_protocols')
    # First-error selfgen estimand and equivalence counterexample.
    selfgen=json.loads((src/'selfgen/B16_BEHAVIOUR_thinkfix_mask1_FINAL890.json').read_text())
    sg=selfgen['pooled']
    fig,ax=plt.subplots(figsize=(10,3.2));yy=np.arange(3)
    v=[sg['B16__BASE']['d_within'],sg['B16__GRP']['d_within'],sg['B16__BASE']['d_hit']]
    cis=[sg['B16__BASE']['d_within_ci95'],sg['B16__GRP']['d_within_ci95'],sg['B16__BASE']['d_hit_ci95']]
    ax.axvspan(-.03,.03,color='#e2edf5',label='Declared equivalence margin ±0.03 (AUC only)')
    for i,(vv,ci) in enumerate(zip(v,cis)):
        ax.errorbar(vv,i,xerr=[[vv-ci[0]],[ci[1]-vv]],fmt='o',color=ORANGE if i==2 else BLUE,capsize=5)
        ax.text(ci[1]+.005,i,f'{vv:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]',fontsize=9,va='center')
    ax.axvline(0,color=GREY,lw=1);ax.set_yticks(yy,['BASE first-error AUC','GRP first-error AUC','BASE exact first-error hit']);ax.invert_yaxis();ax.set_xlim(-.24,.36)
    ax.set_xlabel('Self-generated − teacher-forced ProcessBench · difference, 95% interval')
    ax.set_title('Similar point estimates do not establish equivalence',loc='left',fontweight='bold')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.27),fontsize=8,frameon=False)
    save(fig,'selfgen_intervals')
    # All copied evidence hashes, actual definitions and data used for browser interaction.
    data=dict(head=HEAD,banks=BANKS,roster=ROSTER,names=NAMES,metrics=metrics,lr=lr,
              corrected_n16_recomputed=audit['integrity'],category_f1=vals,
              error_type_auc=ours,artifact_comparator_auc=comparator,
              external_refit=picked,external_locked=lock,selfgen=sg,
              largest_replay_difference=replay[0][0],corrections=corrections)
    (out/'DATA.json').write_text(json.dumps(data,indent=2))
    def caption(path,extra=''):
        return 'Source: '+link('results/thinkfix_rescore_v1/'+path,Path(path).name)+'. '+extra
    charts={
      'fix':figure('prefix_correction',caption('PRMBENCH_COMPARISON.csv','N16; old_replay, thinkfix and thinkfix_mask1. Corrected + mask1 values independently reproduced; raw old/new extraction not replayed here.')),
      'banks':figure('banks_headroom',caption('thinkfix_mask1/nested_banks/PANEL_A_ABSOLUTE.csv','Combined with cont_baselines/ABSOLUTE and LR_SCORES. Development data; LR values reported by branch, not independently refitted.')),
      'lr':figure('lr_intervals',caption('thinkfix_mask1/lr_reference/GAPS.csv','Reported paired bootstrap; best means best of FOUR pipeline variants, not all eight scorers. Source-fold file unavailable for independent disjointness audit.')),
      'types':figure('error_types',caption('thinkfix_mask1/nested_banks/PANEL_B_PRMBENCH_TYPES.csv','Fixed MEAN_MOM_ga throughout. PRM AUC dots are rounded values displayed in the linked artifact; comparator raw scores not replayed.')),
      'flags':figure('flags_categories','Independent N16 reconstruction; <a href="sources/UPSTREAM_PARITY.json">upstream numerical evaluator parity</a>; <a href="sources/PRM7B_MANIFEST.json">PRM comparator manifest</a>. *Sensitivity here covers missing_condition and deception only. It omits multi_solutions and is not the complete leaderboard category.'),
      'external':figure('external_protocols',caption('external/EXTERNAL_PANEL_COMPARISON.csv','Left uses EXTERNAL_LSML_COMPARISON. Corrected external results are branch reports; raw records/predictions unavailable for fresh scoring.')),
      'selfgen':figure('selfgen_intervals',caption('selfgen/B16_BEHAVIOUR_thinkfix_mask1_FINAL890.json','FINAL890; 417 self-generated and 1,602 ProcessBench erroneous answers. AUC uses first error versus preceding correct steps and excludes first-step errors. Different populations; no causal teacher-forcing equivalence claim.'))}
    issue_rows=[
      ('P1','False equivalence claim','The saved BASE interval [+−0.021, +0.052] extends beyond the +0.03 margin. The evaluator tests a small point estimate plus an interval crossing zero. This is not equivalence. A numerical counterexample returns “equivalent” even for CI [−1,+1].',link('docs/reviews/consolidation_20261003/sources/b16_eval.py','Evaluator',126),'Withdraw equivalence wording; use a predeclared TOST or an interval fully inside the margin.'),
      ('P2','Held-out labels enter restricted LR selection','R1/R2/R3 feature sets are selected from mean coefficients across all five outer models, then evaluated with nested_lr. Other outer models trained on each held-out fold. Full LR reference is a separate result and is not affected.',link('results/thinkfix_rescore_v1/thinkfix/lr_weights/run_patched.py','Selection',163),'Keep these as supervised diagnostics, or repeat selection within each outer training fold.'),
      ('P2','Resume can mix prompt variants','Both teacher-forced drivers skip cached rows by idx, ignoring prompt/model/template identity. Fixture proves an old none-prefix row survives a new empty_block request. Separate fresh directories mitigate this campaign.',link('cluster/run_teacher_forced.py','PB driver',423)+'; '+link('cluster/run_prmbench_teacher_forced.py','PRMBench driver',219),'Bind a run digest before reuse and reject mismatches; verify per-record telemetry source digests externally.'),
      ('P2','Entropy gate is only partly rescored','ENTROPY_GATE_Q80 loads frozen old answer-feature epr, whereas step channels use corrected inputs. Central within-answer AUC and argmax rows are unaffected.',link('results/thinkfix_rescore_v1/thinkfix_mask1/nested_banks/dev_run_patched.py','Input binding',81),'Rebuild answer entropy for each variant, or label this gate an old-entropy continuity control.'),
      ('P2','“Official Sensitivity” is incomplete','The displayed mean omits multi_solutions although the taxonomy includes it. Upstream numeric functions reproduce overall MEAN_ga Q80 = 0.652258; evaluator parity on this slice does not establish full leaderboard parity.',link('spectral_utils/prmbench.py','Port')+'; <a href="sources/UPSTREAM_PARITY.json">fresh parity evidence</a>','Label the two-subtype mean explicitly. Include multi-solution correctness behavior and publish the exact aggregation script/outputs.'),
      ('P3','Replay/documentation drift','The final combined comparison has 4,165 replayed values, 66 beyond 1e−9 and max deviation 0.001266; the page quotes the earlier panel-only 3.3e−4 bound. The generic submission selects old spans, although the actual mask1 resubmission repaired it. Mask diagnostics retain old token count.',link('results/thinkfix_rescore_v1/COMPARISON_SUMMARY.json','Final replay summary'),'Report replay limits by component; fix the generic span helper and refresh trace diagnostics.')]
    issues=''.join('<article class="issue"><span class="priority">'+p+'</span><div><h3>'+title+'</h3><p>'+text.replace('[+−0.021, +0.052]','[−0.021, +0.052]')+'</p><p class="small">'+source+'</p><p><b>Action:</b> '+action+'</p></div></article>' for p,title,text,source,action in issue_rows)
    plan=[
      ('Done; verified in part','0a · Prefix correction + masking','New telemetry, old-bank replay, corrected dev scoring and LR rerun are recorded. N16 exported scores and mask/index fixtures pass independently. Raw GPU replay remains report-level evidence.'),
      ('Partial','0b · Understand PRMBench','Taxonomy and dataset construction reviewed. Redundancy and circularity can preserve final correctness; their labels test simplicity. Still need worked examples, repetition/length controls and the original strong-model claim audit.'),
      ('Partial','1 · Performance and flag rules','Corrected bank/method panels and a fresh N16 threshold evaluation are available. Exact full-category leaderboard parity, aggregation provenance and corrected entropy gates remain.'),
      ('Partial','2 · Generalization','Hard2Verify and Socratic were rescored in two protocols; self-generation comparison updated. Source-group half-selection/half-test, leave-one-error-type-out and PRMBench 4B/8B comparison remain.'),
      ('Open','3 · Conditional feature statistics','Need correct-versus-error correlation matrices, DS/MoM estimated-versus-real accuracy and corrected per-family/single-channel controls. Existing historical analyses are not a corrected completion.'),
      ('Partial','4 · Teacher forcing versus own answers','Rebuilt ProcessBench reference available. Withdraw equivalence; report the clear first-error-hit deficit separately. Full matched teacher-forced panel remains blocked by missing frozen VALIDATION inputs.'),
      ('Partial','5 · Supervised LR comparison','Full LR reference and weights rerun. Correct restricted-feature selection; compute error overlap by type/position/length and compare effective weights on the same scale.'),
      ('Open / partial','6 · Confounds, uncertainty and practical cost','Digit twins and some paired intervals exist. Need position-controlled comparisons, paired source-group CIs for fixed primary contrasts and matched runtime/cost measurements.'),
      ('Separate unfinished work','Consolidation','The dated handoff still lists missing HISTORY entries, integration gaps and unresolved test-regression evidence. This branch review does not establish a clean merged project.')]
    plans=''.join('<div class="plan"><span class="status">'+s+'</span><div><h3>'+t+'</h3><p>'+d+'</p></div></div>' for s,t,d in plan)
    bank_options=''.join('<option '+('selected ' if b=='N16' else '')+'value="'+b+'">'+b+'</option>' for b in BANKS)
    page='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PRMBench · corrected results and review</title>
<style>
:root{--ink:#21323d;--muted:#5f707a;--line:#d6dfe4;--paper:#fafbf9;--blue:#246c9c;--orange:#bf6235;--tint:#edf4f8}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.6 system-ui,-apple-system,sans-serif}a{color:var(--blue);text-underline-offset:3px}.wrap{max-width:1180px;margin:auto;padding:0 34px 80px}nav{position:sticky;top:0;z-index:5;background:var(--paper);display:flex;gap:20px;align-items:center;flex-wrap:wrap;padding:14px 0;border-bottom:1px solid var(--line);font-size:13px}nav a{text-decoration:none}nav button{margin-left:auto}button,select{font:inherit;border:1px solid var(--line);border-radius:4px;padding:6px 12px;background:white;color:var(--ink);cursor:pointer}.kicker{text-transform:uppercase;letter-spacing:.15em;font-size:12px;color:var(--muted);margin-top:34px}h1,h2{font-family:Georgia,serif;font-weight:500;line-height:1.14;text-wrap:balance}h1{font-size:clamp(32px,4vw,54px);max-width:850px;margin:14px 0 18px}h2{font-size:34px;margin:0 0 20px}h3{font-size:17px;margin:0 0 6px}p{margin:0 0 16px;max-width:90ch}.lede{font-size:20px;max-width:770px}.verdict{padding:18px 22px;border-left:4px solid var(--blue);background:var(--tint);font-size:18px;margin:24px 0}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:24px;border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:22px 0;margin:28px 0}.metric strong{display:block;font:36px Georgia,serif;line-height:1.15}.metric span{font-size:13px;color:var(--muted);display:block;margin-top:8px}.metric .label{font-size:14px;color:var(--ink);margin:0 0 8px}.section{padding:56px 0;border-top:2px solid var(--ink);scroll-margin-top:76px}.section header{display:flex;gap:24px;align-items:baseline}.num{font:22px Georgia,serif;color:var(--muted)}.two{display:grid;grid-template-columns:1.15fr 1fr;gap:34px}.takeaway{font-size:19px;border-top:1px solid var(--line);padding-top:15px}.small,figcaption{font-size:12px;color:var(--muted)}figure{margin:22px 0 28px;background:white;padding:12px 12px 8px;border:1px solid var(--line)}figure svg{display:block;width:100%;height:auto}figcaption{line-height:1.6;padding:8px}.warning{border-left:3px solid var(--orange);padding:10px 18px;margin:20px 0;background:#f7f0eb}.pipeline{display:flex;gap:10px;align-items:stretch;margin:25px 0;flex-wrap:wrap}.pipeline div{flex:1;min-width:135px;padding:12px;border-top:2px solid var(--blue);background:white;font-size:13px}.pipeline b{display:block;margin-bottom:7px}.pipeline span{color:var(--muted)}details{margin:18px 0;border-top:1px solid var(--line);padding-top:14px}summary{cursor:pointer;font-weight:600}.issue{display:grid;grid-template-columns:48px 1fr;gap:14px;border-top:1px solid var(--line);padding:20px 0}.priority{font-weight:600;color:var(--orange)}.issue p{font-size:14px}.plan{display:grid;grid-template-columns:180px 1fr;gap:25px;border-top:1px solid var(--line);padding:18px 0}.status{font-size:13px;color:var(--blue);font-weight:600}.plan p{font-size:14px;margin:0}.tools{display:flex;gap:15px;align-items:center;margin-bottom:14px}.barrow{display:grid;grid-template-columns:180px 1fr 65px;gap:12px;align-items:center;height:36px;font-size:13px}.bartrack{position:relative;height:16px;background:#e8edf0}.barfill{height:100%;background:var(--blue)}.barrow:last-child .barfill{background:var(--orange)}.barvalue{font-variant-numeric:tabular-nums;text-align:right}.explore{background:white;padding:22px;border:1px solid var(--line)}.legend{font-size:12px;color:var(--muted);margin-top:12px}.matrix{overflow:auto}table{border-collapse:collapse;width:100%;font-size:12px;font-variant-numeric:tabular-nums}th,td{padding:8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}th:first-child{text-align:left}tr.focus{background:var(--tint)}.speaker{font-size:15px;border-left:3px solid var(--blue);padding:8px 20px;max-width:85ch}.footer{font-size:12px;color:var(--muted)}
@media(max-width:700px){.wrap{padding:0 18px 40px}.metrics{grid-template-columns:1fr 1fr}.two{grid-template-columns:1fr}.plan{grid-template-columns:1fr;gap:4px}h2{font-size:27px}.section{padding:32px 0}nav{gap:8px 14px;font-size:12px}.barrow{grid-template-columns:130px 1fr 45px;font-size:11px}.metric strong{font-size:30px}}
@media print{nav,.tools,button{display:none}.wrap{padding:0;max-width:none}body{background:white;font-size:11px}.section{break-before:page;padding:20px 0;border:0}h1{font-size:32px}h2{font-size:25px}figure{break-inside:avoid}.metrics strong{font-size:28px}a{color:inherit;text-decoration:none}details{display:block}details:not([open])>*:not(summary){display:none}.footer{font-size:9px}}
</style></head><body><div class="wrap">
<nav><a href="#top">Conclusion</a><a href="#fix">Input correction</a><a href="#fusion">Fusion &amp; banks</a><a href="#types">Error types</a><a href="#flags">Flag rules</a><a href="#transfer">Generalization</a><a href="#plan">Plan</a><a href="#review">Review</a><button onclick="window.print()">Print / save PDF</button></nav>
<main><div class="kicker" id="top">Advisor meeting · 15 October 2026 · audited 11 October</div>
<h1>A useful ranking signal.<br>A narrower claim to validate.</h1>
<p class="lede">Correcting the Qwen3 input reduces the scores. The signal survives, but the evidence supports error localization on development data more strongly than reliable error detection in new answers.</p>
<p class="small">Pinned branch: <a href="https://github.com/omrisegev/hallucination_detection/tree/HEADVALUE">claude/thinkfix-rescore-v1 @ 080877eb</a>. Primary input: empty think block + first content token masked. PRMBench scored with Qwen3-8B. All bank/method choices remain development choices.</p>
<div class="verdict"><b>Review verdict:</b> the central corrected N16 arithmetic passes independent recomputation. The branch is not fully validated end to end: one material equivalence claim is invalid, and several diagnostics and replay helpers need correction or clearer scope.</div>
<div class="metrics"><div class="metric"><span class="label">N16 label-free ranking</span><strong>0.778</strong><span>Mean AUROC across 8 scorers<br>Independently recomputed</span></div><div class="metric"><span class="label">N16 signed/filter control</span><strong>0.780</strong><span>DS-filter average<br>Pipeline gains are small</span></div><div class="metric"><span class="label">N16 supervised headroom</span><strong>0.801</strong><span>Full LR cross-fit, branch report<br>PRM-7B artifact AUC also ≈0.801</span></div><div class="metric"><span class="label">Clean-answer false flags</span><strong>99.2%</strong><span>MEAN / ga · global answer-z Q80<br>758 synthetic correct controls</span></div></div>
<p class="speaker" lang="he" dir="rtl"><b>מסר לפגישה:</b> אחרי תיקון הקלט, האות נשאר אבל המספרים נמוכים יותר. מיזוג מורכב כמעט לא מוסיף על ממוצע עם סינון. היתרון המעניין הוא בשגיאות פשטות — יתירות ומעגליות — אבל עדיין צריך להראות שהוא נשאר אחרי בקרת מיקום ובדיקה על אוכלוסייה שלא שימשה לבחירה.</p>
<section class="section" id="fix"><header><span class="num">01</span><h2>The input correction matters</h2></header>
<p>Teacher forcing scores a supplied answer; it generates no new answer. The old prefix omitted the empty think block expected along the selected Qwen3 no-think path. The campaign added it, then separately masked token 0. Those are two interventions: the three-stage chart keeps their effects distinct.</p>
<div class="pipeline"><div><b>1 · Supplied answer</b><span>Question + corrected assistant prefix</span></div><div><b>2 · One forward pass</b><span>Token probabilities / written-token surprise</span></div><div><b>3 · Mask token 0</b><span>Shift telemetry fields and step spans together</span></div><div><b>4 · Step channels</b><span>Feature bank, answer normalization</span></div><div><b>5 · Fit &amp; rank</b><span>One cell; no labels in label-free fitting</span></div><div><b>6 · Decide flags</b><span>Separate threshold or quota rule</span></div></div>
CHARTFIX
<p class="takeaway">MEAN / ga falls from 0.791 to 0.777 on N16. The correction weakens the headline score; it does not erase the signal.</p>
<details><summary>What was actually verified?</summary><p>Independent pairwise AUC from saved corrected scores, exact IDs and offsets, finite columns, answer normalization, masking edge cases and a 1,000-draw within-answer label null. MEAN / ga null mean is 0.4999; null 95% range 0.4930–0.5064; observed AUC is 0.7767. This is a saved-score check, not a fresh GPU or full fitting null.</p><p>The final development manifests retain 13,769 answers / 145,597 steps. External manifests retain 200 Hard2Verify + 2,995 Socratic answers. Three originally empty Socratic steps are correctly retained after the repair. These full-campaign counts are manifest-level evidence; raw caches were not available for enumeration.</p></details>
</section>
<section class="section" id="fusion"><header><span class="num">02</span><h2>More channels do not automatically improve fusion</h2></header>
<p>The eight label-free scorers are close on N16: 0.771–0.780. A filtered equal average reaches the same region as the more elaborate methods. Across larger banks, LR improves while label-free curves flatten or fall. This is evidence of supervised headroom; the curve alone does not identify its cause.</p>
CHARTBANKS
<div class="two"><div><h3>N13 → N16: make the digit contribution visible</h3><p>The mean across eight methods rises from <b>0.764 to 0.778</b> when adding the digit channels in this pair. This is a descriptive development comparison. It needs a paired, source-group interval before a significance claim.</p><p>Digit-inclusive banks are allowed by the later September 30 decision. Report both N13 and N16; do not hide the dependency or reinstate the older prohibition.</p></div><div class="explore"><div class="tools"><label for="bank">Inspect bank</label><select id="bank">BANKOPTIONS</select></div><div id="bars" aria-live="polite"></div><div class="legend">Horizontal bars: within-answer AUROC; axis range 0.70–0.83. Blue = label-free; orange = supervised LR. Values from the pinned corrected CSVs.</div></div></div>
CHARTLR
<p class="small">The displayed +0.019–+0.057 intervals compare LR against the best of four tail-pipeline variants. Against the best of all eight N16 methods, the point gap is ≈0.021. Restricted-feature LR analyses have selection leakage; they cannot validate a minimal unbiased subset.</p>
<details><summary>Full bank × scorer matrix</summary><div class="matrix" id="matrix"></div><p class="small">Arithmetic means across eight fixed scorers. No per-bank winner replacement. N16 was chosen after inspecting development results.</p></details>
</section>
<section class="section" id="types"><header><span class="num">03</span><h2>The strongest claim is specific to error type</h2></header>
<p>A fixed MEAN / ga scorer performs well on redundancy and circularity, while the PRM is stronger on several fluent but invalid statements. Keep one method fixed across the chart; choosing a different winner in each error type would overstate a deployable result.</p>
CHARTTYPES
<div class="two"><div><h3>What PRMBench calls an error</h3><p>Redundant and circular steps may preserve the final answer’s correctness. They test efficiency and logical simplicity. The benchmark modifies correct solutions into typed cases, then applies quality checks. This is a broader target than detecting arithmetic mistakes. <a href="https://arxiv.org/html/2501.03124v1">PRMBench paper, §§3–4</a>.</p></div><div><h3>What remains a hypothesis</h3><p>Repeated text, step position or length may explain part of the advantage. The paper describes general PRM weaknesses here; it does not establish that this dataset was targeted against our particular PRM comparator. Validate the advantage after controlling those cues.</p></div></div>
<p class="takeaway">Candidate research claim: one label-free pass provides complementary simplicity-error signal. The current evidence does not establish superiority across all error types.</p>
</section>
<section class="section" id="flags"><header><span class="num">04</span><h2>Ranking and clean-answer detection tell different stories</h2></header>
<p>The N16 MEAN / ga Q80 score is <b>0.6523</b>. I reproduced it both independently and with the upstream evaluator’s numerical functions on the exported slice. Synthetic correct controls are excluded from this pooled score. Their false flags therefore need a separate panel.</p>
CHARTFLAGS
<div class="warning"><b>Category naming needs correction.</b> The original page calls the two-subtype Sensitivity mean “official,” while omitting multi-solution consistency. Multi-solutions have no error-versus-correct AUC, but their valid-step judgments still matter. This page labels that mean explicitly and does not claim full leaderboard parity.</div>
<p>For the 160 multi-solution answers, upstream numerical parity gives valid-step F1 = <b>0.8854</b> for MEAN / ga Q80. Applying the same half-F1 formula yields 0.4427 because error F1 is zero. Including it in a three-subtype arithmetic mean gives 0.5466; this is a declared reconstruction, not a certified leaderboard aggregation. The complete official reporting contract needs to be pinned.</p>
<details><summary>Threshold, quota, and population contract</summary><p><b>R0:</b> answer-normalize scores, then take one global 80th percentile over all scored steps. It does not guarantee 20% of every answer. <b>R1:</b> one bank-wide DS-survivor mean, globally standardized; not a replay of the earlier decision-rule implementation. <b>R2:</b> take each answer’s R1 flag count and choose steps by the method ranking. <b>TOP1/TOP2:</b> forced counts; TOP2 is benchmark-informed and remains diagnostic.</p><p>Scorers fit pooled cells, not isolated answers. No labels in the fitted label-free formula does not mean bank selection, feature design, quotas and presentation choices were held out. This evaluation is transductive development evidence.</p><p>Export: 6,969 answers / 94,203 steps. Within-AUC: 6,030 mixed-label answers; top-one error hit: 6,035 erroneous answers. Noncontrol pooled F1: 6,211 answers; synthetic controls: 758; all actual error-free answers: 934. Five raw duplicate rows are dropped by the loader. One hundred rows contain out-of-range error annotations; these stay inert when matching existing steps.</p></details>
</section>
<section class="section" id="transfer"><header><span class="num">05</span><h2>External rescoring is done; confirmation is not</h2></header>
<p>The linked page says external benchmarks have not been rescored. The current branch contains them. Two protocols answer different questions, so keep them separate:</p>
CHARTEXTERNAL
<p><b>Locked bundle:</b> target telemetry is corrected, but the source-trained L-SML bundle remains fitted on pre-fix data because its source inputs are gone. <b>Monday panel:</b> label-free fitting is repeated on each corrected target cell. The latter is exploratory target adaptation, not frozen source-only transfer. QwQ was not rerun.</p>
<h3>Own-generation comparison: withdraw “equivalent”</h3>
CHARTSELFGEN
<p>The BASE AUC interval crosses zero but extends beyond the +0.03 margin. Equivalence is unestablished. Exact first-error hit is clearly lower in self-generated answers: <b>0.264 versus 0.428</b>, a difference of <b>−0.164 [−0.210, −0.114]</b>. These are different cohorts and a first-error task, not PRMBench dense-label AUC.</p>
<details><summary>ProcessBench stays context only</summary><p>Corrected N16 macro first-error hit: plain average ≈0.404; MEAN / cc ≈0.407; MEAN / ga ≈0.375; LR transfer ≈0.373. This does not motivate restoring ProcessBench as the meeting’s focus. Do not relabel these as official ProcessBench F1; clean-answer behavior is not included in this hit metric. Missing Mind the Gap inputs prevent a corrected comparison.</p></details>
</section>
<section class="section" id="plan"><header><span class="num">06</span><h2>What is complete, and what the plan still requires</h2></header>
<p>Mapped to the October 8 advisor plan and the October 10–11 handoffs. “Done” means the branch contains the result; independent review coverage is stated separately.</p>
PLANS
<div class="verdict"><b>Recommended next sequence:</b> correct the equivalence and category claims; freeze one primary scorer/bank/flag rule; run source-group held-out selection plus position controls; then investigate the remaining LR gap. Avoid another broad sweep before these questions are answered.</div>
<p class="speaker" lang="he" dir="rtl"><b>שלוש שאלות למנחים:</b> האם יתרון בשגיאות פשטות הוא הטענה המרכזית שנרצה לבדוק? מהו מבחן ההכללה שיספיק אחרי הקפאת השיטה? האם משקיעים בשיפור הייצוג ובניגודים בתוך קבוצות, או קודם בהחלטה שתשובה נקייה?</p>
</section>
<section class="section" id="review"><header><span class="num">07</span><h2>Review findings and evidence boundary</h2></header>
<details><summary>Six findings and required corrections</summary>ISSUES</details>
<details><summary>Scope and reproducibility</summary><p>Reviewed the five focused commits from base 5efe8761 to 080877eb, changed code, scoring dependencies, guides, manifests and result tables. The branch also carries a long inherited history; this is not a fresh review of all 99 commits relative to master. Retrieved 679 files using Git blob-hash verification. Raw GPU caches, most feature arrays, OOF source-group assignments, several removed-worktree dependencies and official category-page generator inputs were absent.</p><p>Independent audit A: direct pairwise AUC and flag arithmetic. Audit B: coverage/manifests and masking/resume fixtures. Audit C: mathematics, selection audit and saved-score permutation null. Code defects and numerical checks have different evidence scopes; no full repository regression-suite attestation is claimed.</p><p><a href="REVIEW.md">Detailed review and disposition</a> · <a href="DATA.json">Chart data</a> · <a href="evidence/A/audit.json">Metric audit</a> · <a href="evidence/B/REVIEW_B.md">Coverage audit</a> · <a href="evidence/C/NULL_REPORT.json">Null audit</a> · <a href="sources/FETCH_MANIFEST.json">Pinned snapshot manifest</a>.</p></details>
<p class="footer">Local replacement artifact, not an edit to the published Claude page. No research code, branch, existing results or frozen evidence modified. No commit, push, package install or cluster run. Layout inspired by the October 8 findings page: serif headings, restrained palette, reading order and drill-downs. Replaced primary wide tables with comparison graphs; historical material is kept in expandable context. Earlier private slide links require sign-in; repository HTML/deep dives and the supplied live artifact were used directly.</p>
</section></main></div>
<script id="data" type="application/json">EMBEDDATA</script>
<script>
const data=JSON.parse(document.getElementById('data').textContent); const selector=document.getElementById('bank');
function draw(){const b=selector.value;const series=data.roster.map((s,i)=>[data.names[i],data.metrics[b][s]]);series.push(['LR · supervised',data.lr[b]]);document.getElementById('bars').innerHTML=series.map(([name,value])=>`<div class="barrow"><span>${name}</span><div class="bartrack"><div class="barfill" style="width:${Math.max(0,Math.min(100,(value-.70)/.13*100))}%"></div></div><span class="barvalue">${value.toFixed(3)}</span></div>`).join('');}
selector.addEventListener('change',draw);draw();
document.getElementById('matrix').innerHTML='<table><caption>Corrected PRMBench · mean within-answer AUROC</caption><thead><tr><th>Bank</th>'+data.names.map(n=>'<th>'+n+'</th>').join('')+'<th>Mean, 8</th><th>LR</th></tr></thead><tbody>'+data.banks.map(b=>'<tr class="'+(b==='N16'?'focus':'')+'"><th>'+b+'</th>'+data.roster.map(s=>'<td>'+data.metrics[b][s].toFixed(3)+'</td>').join('')+'<td>'+(data.roster.reduce((v,s)=>v+data.metrics[b][s],0)/8).toFixed(3)+'</td><td>'+data.lr[b].toFixed(3)+'</td></tr>').join('')+'</tbody></table>';
</script></body></html>'''
    replacements={'HEADVALUE':HEAD,'CHARTFIX':charts['fix'],'CHARTBANKS':charts['banks'],'CHARTLR':charts['lr'],
                  'CHARTTYPES':charts['types'],'CHARTFLAGS':charts['flags'],'CHARTEXTERNAL':charts['external'],
                  'CHARTSELFGEN':charts['selfgen'],'PLANS':plans,'ISSUES':issues,'BANKOPTIONS':bank_options,
                  'EMBEDDATA':json.dumps({k:data[k] for k in ['banks','roster','names','metrics','lr']}).replace('</','<\\/')}
    for key,value in replacements.items():page=page.replace(key,value)
    page=page.replace('scroll-behavior:smooth','scroll-behavior:auto')
    page=page.replace('</style>','#top{scroll-margin-top:95px}</style>')
    page=page.replace('N16 signed/filter control','N16 filter control')
    # Preserve the complete technical audit as supporting material. The meeting
    # presentation follows the user's priority: full corrected PRMBench results,
    # with Continuous L-SML and IU-PCR first; input-history and review in appendix.
    (out/'audit_detail.html').write_text(page)
    from build_presentation import build
    build(out)
    manifest={str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in out.rglob('*') if p.is_file() and p.name!='DELIVERABLE_MANIFEST.json'}
    (out/'DELIVERABLE_MANIFEST.json').write_text(json.dumps(dict(reviewed_head=HEAD,files=manifest),indent=2))
    print(json.dumps(dict(output=str(out/'advisor_review.html'),n_figures=len(list(figs.glob('*.svg'))),n_files=len(manifest),max_replay_abs_diff=replay[0][0]),indent=2))


if __name__=='__main__':
    main()
