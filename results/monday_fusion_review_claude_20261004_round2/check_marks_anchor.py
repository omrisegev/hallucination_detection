"""Q4 anchor and Q5 marks-histogram checks from the sealed diagnostics (no refits)."""
import json
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path(r'C:\Users\omris\TAU\hallucination_detection'); OUT=Path(__file__).resolve().parent
LEVEL=('q15_H1','q15_VE1','logprob_margin','true_tail50','energy_level')
lev=ROOT/'results/level_reduction_v1/run_20261003/chunks'; inp=ROOT/'results/estimator_input_v1/run_20261003/chunks'
# ---- Q5: histogram unit = answer x column
hist_rows=[]; bad=0
for p in sorted(inp.glob('*.diagnostics.json')):
    ce,bk,var=p.name.removesuffix('.diagnostics.json').split('__'); d=json.loads(p.read_text(encoding='utf8'))['diagnostics']
    for within in ['MEAN','HEM']:
        md=d.get(within+'_marks_distribution')
        if not md: continue
        for kind,h in md.items():
            tot=sum(h['histogram'].values()); ok=tot==h['n_answers']*h['n_columns']; bad+=not ok
            hist_rows.append({'cell':ce,'bank':bk,'variant':var,'within':within,'kind':kind,'n_answers':h['n_answers'],'n_columns':h['n_columns'],'total':tot,'unit_is_answer_x_column':ok,
                              'zero_fraction':h['histogram'].get('0',0)/tot,'mean_marks':sum(int(k)*n for k,n in h['histogram'].items())/tot})
hist=pd.DataFrame(hist_rows); hist.to_csv(OUT/'MARKS_HISTOGRAM_UNITS.csv',index=False)
q5={'histograms_checked':len(hist),'unit_mismatches':int(bad),'cells_covered':sorted(hist.cell.unique().tolist()),
    'prm_clean_zero_fraction_by_variant':hist[(hist.cell=='prmbench_qwen3_8b')&(hist.kind=='clean')].groupby('variant').zero_fraction.mean().round(5).to_dict(),
    'prm_clean_mean_marks_by_variant':hist[(hist.cell=='prmbench_qwen3_8b')&(hist.kind=='clean')].groupby('variant').mean_marks.mean().round(4).to_dict(),
    'note':'zero_fraction counts answer x column units with zero marks; whole-answer no-mark counts are not stored anywhere in the sealed artifacts'}
# ---- Q4: anchor. Grouping call: lsml_fit_scaled(tt[:,surv], ai, x, ...) -> lsml_continuous(*x.T) receives no anchor (code). Here: data checks.
rows=[]; alt_anchor={}
for p in sorted(lev.glob('*.diagnostics.json')):
    ce,bk,var=p.name.removesuffix('.diagnostics.json').split('__'); d=json.loads(p.read_text(encoding='utf8'))['diagnostics']
    if var in ('remove_2','remove_3'): alt_anchor[(ce,bk,var)]=d['anchor']
for p in sorted(lev.glob('*.diagnostics.json')):
    ce,bk,var=p.name.removesuffix('.diagnostics.json').split('__'); d=json.loads(p.read_text(encoding='utf8'))['diagnostics']
    surv=d['survivors']; groups=d.get('groups_after')
    for within in ['MEAN','HEM']:
        for est in ['MOM','SML','DS']:
            k=within+'_'+est
            if k not in d or 'estimator' not in d[k]: rows.append({'cell':ce,'bank':bk,'variant':var,'recipe':k,'status':'absent'}); continue
            e=d[k]['estimator']; r={'cell':ce,'bank':bk,'variant':var,'recipe':k,'status':'ok','anchor':d['anchor'],'anchor_fallback':d['anchor_fallback']}
            if est=='MOM':
                t=np.asarray(e['t'],float); r.update(t_sum=float(t.sum()),t_pos_majority=bool((t>0).sum()>len(t)/2),t_min=float(t.min()),n_groups=len(t))
                ag=int(groups[surv.index(d['anchor'])]); r['anchor_group_t']=float(t[ag])
                # counterfactual: in the ORIGINAL bank, sign of t for the group holding the anchor that the removal variant actually used
                if var=='original':
                    for rv in ('remove_2','remove_3'):
                        alt=alt_anchor.get((ce,bk,rv))
                        if alt in surv: r[f'alt_anchor_{rv}']=alt; r[f'alt_anchor_group_t_{rv}']=float(t[int(groups[surv.index(alt)])])
                        else: r[f'alt_anchor_{rv}']=alt; r[f'alt_anchor_group_t_{rv}']=None
            elif est=='SML':
                r.update(anchor_flipped=e.get('anchor_flipped'),anchor_spearman=e.get('anchor_spearman'),small_m_guard=e.get('small_m_guard'))
            else:
                r.update(orientation=e.get('orientation'),prevalence=e.get('prevalence'))
            rows.append(r)
A=pd.DataFrame(rows); A.to_csv(OUT/'ANCHOR_SIGN_AUDIT.csv',index=False)
mom=A[(A.recipe.str.endswith('_MOM'))&(A.status=='ok')]; sml=A[(A.recipe.str.endswith('_SML'))&(A.status=='ok')]; ds=A[(A.recipe.str.endswith('_DS'))&(A.status=='ok')]
orig=mom[mom.variant=='original']
q4={'mom_fits':int(len(mom)),'mom_t_sum_positive':int((mom.t_sum>0).sum()),'mom_t_pos_majority':int(mom.t_pos_majority.sum()),'mom_anchor_group_t_positive':int((mom.anchor_group_t>0).sum()),
    'mom_any_negative_group_t':int((mom.t_min<0).sum()),
    'original_mom_fits':int(len(orig)),
    'original_alt_anchor_remove_2_group_t_positive':int((orig.alt_anchor_group_t_remove_2.astype(float)>0).sum()),'original_alt_anchor_remove_2_defined':int(orig.alt_anchor_group_t_remove_2.notna().sum()),
    'original_alt_anchor_remove_3_group_t_positive':int((orig.alt_anchor_group_t_remove_3.astype(float)>0).sum()),'original_alt_anchor_remove_3_defined':int(orig.alt_anchor_group_t_remove_3.notna().sum()),
    'sml_fits':int(len(sml)),'sml_anchor_flipped':int(sml.anchor_flipped.astype(bool).sum()),'sml_min_anchor_spearman':float(sml.anchor_spearman.min()),
    'ds_fits':int(len(ds)),'ds_orientation_values':sorted(ds.orientation.unique().tolist()),
    'anchor_changed_in_removal':int(((A.variant.isin(['remove_2','remove_3']))&(A.anchor!='q15_H1')).groupby([A.cell,A.bank,A.variant]).any().sum()),
    'alt_anchors_used':sorted(set(alt_anchor.values()))}
json.dump({'q4_anchor':q4,'q5_marks':q5},open(OUT/'MARKS_ANCHOR_CHECKS.json','w',encoding='utf8'),indent=1,default=str)
print(json.dumps({'q4_anchor':q4,'q5_marks':q5},indent=1,default=str))
