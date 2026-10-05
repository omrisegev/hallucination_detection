"""Redraw F11/F12 with category and legend fixes; no metrics are recomputed."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(r'C:\Users\omris\TAU\hallucination_detection')
OUT = ROOT / 'results/monday_fusion_competitor_extension_v1/run_20261006'
FIG = OUT / 'figures'
METHODS = ['HEM_MOM_cc', 'MEAN_MOM_cc', 'MEAN_MOM_ga', 'TCI_ONLY_ga']
COLORS = {'HEM_MOM_cc':'#2864a5', 'MEAN_MOM_cc':'#e17c05', 'MEAN_MOM_ga':'#299447',
          'TCI_ONLY_ga':'#c73e3a', 'MTG native argmax':'#353535'}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()

def draw(dark=False):
    bg='#17191d' if dark else '#ffffff'; fg='#f4f4f4' if dark else '#222222'
    suffix='dark' if dark else 'light'
    cat=pd.read_csv(OUT/'F11_PRM_CATEGORY_METRICS.csv')
    cat=cat[(cat.method.isin(METHODS)) & (cat.rule=='ANSWER_Z_Q80') & (cat.category!='control')]
    if set(cat.category.unique()) != {'sensitivity','simplicity','soundness'}:
        raise RuntimeError('F11 error-category coverage is incomplete')
    fig,axes=plt.subplots(1,2,figsize=(14,5.5),facecolor=bg)
    for ax,metric,title,limits in zip(axes,['within_answer_auc','prmscore'],
        ['Within-answer AUC','PRMScore'],[(.62,.86),(.55,.75)]):
        x=cat.groupby(['category','method'])[metric].mean().unstack('method')
        x=x.reindex(['sensitivity','simplicity','soundness'])
        for method in METHODS:
            ax.plot(range(len(x)),x[method],marker='o',markersize=6,linewidth=1.8,
                    color=COLORS[method],label=method)
        ax.set_xticks(range(len(x)),x.index,rotation=15); ax.set_ylim(*limits)
        ax.set_title(title,color=fg); ax.set_ylabel(metric,color=fg); ax.grid(axis='y',alpha=.25)
        ax.tick_params(colors=fg); ax.set_facecolor(bg)
        for sp in ax.spines.values(): sp.set_color(fg)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=4,frameon=False,labelcolor=fg,bbox_to_anchor=(.5,-.015))
    fig.suptitle('F11 · PRMBench by annotated error category · answer-z q80',color=fg)
    fig.tight_layout(rect=(0,.09,1,.93))
    for ext in ('png','svg'):
        fig.savefig(FIG/f'F11_PRM_categories_revised_{suffix}.{ext}',dpi=180,facecolor=bg,bbox_inches='tight')
    plt.close(fig)

    pb=pd.read_csv(OUT/'F12_MTG_BREAKDOWN.csv')
    pb=pb[(pb.rule=='R2_COUNT') & pb.method.isin(METHODS) &
          pb.stratum.isin(['error_position_quartile','answer_length'])]
    fig,axes=plt.subplots(2,2,figsize=(15,9),facecolor=bg)
    specifications=[('error_position_quartile','candidate_argmax_hit','Score argmax · error-position quartile', ['0','1','2','3']),
                    ('answer_length','candidate_argmax_hit','Score argmax · answer length', ['1-5','6-10','11-20','21+']),
                    ('error_position_quartile','candidate_decision_hit','First flagged step · error-position quartile', ['0','1','2','3']),
                    ('answer_length','candidate_decision_hit','First flagged step · answer length', ['1-5','6-10','11-20','21+'])]
    for ax,(stratum,metric,title,order) in zip(axes.flat,specifications):
        sub=pb[pb.stratum==stratum].copy()
        grouped=sub.groupby(['value','method'],dropna=False)[[metric,'MTG_native_argmax_hit']].mean()
        vals=grouped[metric].unstack('method').reindex(order)
        for method in METHODS:
            ax.plot(range(len(order)),vals[method],marker='o',linewidth=1.7,color=COLORS[method],label=method)
        # MTG has one native argmax series shared by candidate rows.
        mtg=sub.groupby('value')['MTG_native_argmax_hit'].mean().reindex(order)
        if metric=='candidate_argmax_hit':
            ax.plot(range(len(order)),mtg,marker='s',linestyle='--',linewidth=2,color=COLORS['MTG native argmax'],label='MTG native argmax')
        ax.set_xticks(range(len(order)),order); ax.set_ylim(0,1); ax.grid(axis='y',alpha=.25)
        ax.set_title(title,color=fg); ax.set_ylabel('First-error hit rate',color=fg)
        ax.tick_params(colors=fg); ax.set_facecolor(bg)
        for sp in ax.spines.values(): sp.set_color(fg)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,labelcolor=fg,bbox_to_anchor=(.5,-.01))
    fig.suptitle('F12 · ProcessBench localization vs Mind the Gap · R2_COUNT',color=fg)
    fig.tight_layout(rect=(0,.09,1,.93))
    for ext in ('png','svg'):
        fig.savefig(FIG/f'F12_MTG_position_length_revised_{suffix}.{ext}',dpi=180,facecolor=bg,bbox_inches='tight')
    plt.close(fig)

def main():
    FIG.mkdir(exist_ok=True)
    for dark in (False,True): draw(dark)
    entries=[]
    for p in sorted(FIG.glob('*_revised_*')):
        entries.append({'path':str(p.relative_to(OUT)).replace('\\','/'),'bytes':p.stat().st_size,'sha256':sha(p)})
    manifest={'status':'RENDERED_FROM_FROZEN_TABLES','source_tables':{
        'F11_PRM_CATEGORY_METRICS.csv':sha(OUT/'F11_PRM_CATEGORY_METRICS.csv'),
        'F12_MTG_BREAKDOWN.csv':sha(OUT/'F12_MTG_BREAKDOWN.csv')},
        'renderer_sha256':sha(Path(__file__)),'figures':entries,
        'visual_change':'Exclude correct-control category from error-category chart; separate score-argmax and first-flag hit rates; move legend below axes.',
        'metrics_changed':False}
    tmp=OUT/'FIGURES_MANIFEST.json.tmp'; tmp.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8'); tmp.replace(OUT/'FIGURES_MANIFEST.json')
    report=(OUT/'REPORT.md').read_text(encoding='utf8')
    report += '\n## Corrected figures\n\n'
    report += 'Use the `*_revised_light` or `*_revised_dark` F11/F12 PNG/SVG files. The first-pass F12 plot had an obstructive legend; the corrected F11 plot omits the correct-control category, which has no error steps. No metric or interval changed.\n'
    (OUT/'REPORT.md').write_text(report,encoding='utf8')
    print('Revised figures rendered from frozen tables; no metric values changed.')

if __name__=='__main__': main()
