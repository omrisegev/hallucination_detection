"""Reviewed tables, a standard exported figure, and explicitly post-hoc diagnostics."""
import html
from html.parser import HTMLParser
import importlib.util
from pathlib import Path
from urllib.parse import urlsplit,unquote
from collections import Counter
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
d=module(ROOT/'scripts/run_fusion_prediction_quality_v1.py','prediction_quality_report_driver')
OUT=d.OUT


def diagnostics(e):
    result={'status':'POST_EVALUATION_DESCRIPTIVE_DIAGNOSTICS','labels_used_for_hit_counts':True,
        'no_new_candidate_or_selection_rule':True,'evaluation_sha256':d.sha(OUT/'EVALUATION.json'),
        'feature_changes':{},'hit_counts':{}}
    for kind in d.KINDS:
        flips=[];mass=[];rho=[];changed=Counter()
        for row in e['rows']:
            m=d.load(OUT/'scores'/(row['uid']+'.json'));bank=m['diagnostics']['bank'];sh=m['diagnostics']['variants'][kind]
            old=d.load(d.ORIGINAL/'scores'/(row['uid']+'.json'))['diagnostics']['banks'][bank]['shared']
            signs=dict(zip(sh['active_features'],sh['feature_signs']));prior=dict(zip(old['active_features'],old['feature_signs']))
            assert all(name in signs for name in prior)
            diff=[name for name in prior if signs[name]!=prior[name]];flips.append(len(diff));changed.update(diff)
            w=np.asarray(m['methods'][kind+'__graph010']['standardized_weights'])
            extra=np.array([name.endswith('__prediction_abs_mean') for name in sh['active_features']])
            mass.append(float(np.abs(w[extra]).sum()/np.abs(w).sum()))
            with np.load(OUT/'scores'/(row['uid']+'.npz'),allow_pickle=False) as a,np.load(d.PARENT/'scores'/(row['uid']+'.npz'),allow_pickle=False) as b:
                fi=a['fit_indices'];rho.append(float(np.corrcoef(a[kind+'__graph010__window'][fi],b['dual__cond100_graph010__window'][fi])[0,1]))
        result['feature_changes'][kind]={'answers_with_original_sign_changes':sum(x>0 for x in flips),
            'median_original_sign_changes':float(np.median(flips)),'sign_change_feature_counts':dict(changed),
            'median_added_absolute_standardized_weight_share':float(np.median(mass)),
            'median_graph_window_pearson_to_original_graph100':float(np.median(rho)),
            'weight_share_is_not_causal_importance':True}
    pb=[r for r in e['rows'] if r['cell'].startswith('pb_')]
    result['pb_population']={'clean':sum(r['target']==-1 for r in pb),'error':sum(r['target']!=-1 for r in pb)}
    for arm in d.ARMS:
        result['hit_counts'][arm]={'clean_hits':sum(r['decision_valid'][arm] and r['predictions'][arm]==-1 for r in pb if r['target']==-1),
            'error_exact_hits':sum(r['decision_valid'][arm] and r['predictions'][arm]==r['target'] for r in pb if r['target']!=-1),
            'error_peak_hits':sum(r['valid'][arm] and r['peaks'][arm]==r['target'] for r in pb if r['target']!=-1),
            'score_coverage':sum(r['valid'][arm] for r in e['rows'])}
    return result


def fmt(v,percent=False):return '—' if v is None else (f'{100*v:.2f}%' if percent else f'{v:.5f}')
def interval(v,scale=1.):return 'undefined' if v is None else f'[{scale*v[0]:+.4f}, {scale*v[1]:+.4f}]'
def metric_row(arm,e,diag):
    x=e['metrics'][arm];h=diag['hit_counts'][arm]
    return f'<tr><td><code>{html.escape(arm)}</code></td><td>{fmt(x["prm"]["auroc"])}</td><td>{x["prm"]["answers"]}</td><td>{fmt(x["prm"]["within_answer_auc"])}</td><td>{fmt(x["pb"]["macro_f1"],True)}</td><td>{fmt(x["pb_common_iu_gate"]["macro_f1"],True)}</td><td>{h["score_coverage"]}/110</td></tr>'


def figure(e):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cores=('equal','iu','joint0','graph010','graph_perm','equal_graph010','equal_graph_perm')
    labels=('Equal','IU','Joint λ0','Joint graph','Joint perm.','Equal graph','Equal perm.')
    fig,axes=plt.subplots(1,2,figsize=(13,5.2),layout='constrained')
    groups=[('Original',None,'#173b43'),('AR residual','ar1','#d47843'),('Last residual','last','#4b86b4'),('EMA32 residual','ema32','#7c68a6')]
    points=[]
    for ax,endpoint,title in [(axes[0],'prm','PRMBench step AUROC (24 answers)'),(axes[1],'pb','ProcessBench macro F1 (86 answers)')]:
        for offset,(label,kind,color) in enumerate(groups):
            arms=[d.ANCHORS[c] if kind is None else kind+'__'+c for c in cores]
            values=[e['metrics'][a][endpoint]['auroc' if endpoint=='prm' else 'macro_f1'] for a in arms]
            if endpoint=='pb':values=[v*100 for v in values]
            pos=np.arange(7)+(offset-1.5)*.18
            ax.plot(pos,values,'o',color=color,label=label,markersize=5)
            points.extend({'endpoint':endpoint,'arm':a,'value':v} for a,v in zip(arms,values))
        ax.set_title(title);ax.set_xticks(np.arange(7),labels,rotation=40,ha='right')
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].set_ylabel('AUROC');axes[1].set_ylabel('Macro F1 (%)');axes[1].legend(fontsize=9)
    fig.suptitle('Adding residual columns did not improve the existing leaders on both tasks',fontsize=13)
    fig.savefig(OUT/'quality_comparison.png',dpi=170);fig.savefig(OUT/'quality_comparison.svg');plt.close(fig)
    d.save(OUT/'FIGURE_DATA.json',{'points':points,'source_sha256':d.sha(OUT/'EVALUATION.json'),
        'note':'Point estimates only; all 74 paired uncertainty comparisons are in the report tables.'})


class Parser(HTMLParser):
    def __init__(self):super().__init__();self.links=[];self.ids=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id' in a:self.ids.append(a['id'])
        if tag=='a' and 'href' in a:self.links.append(a['href'])
        if tag=='img' and 'src' in a:self.links.append(a['src'])


def main():
    d.verify();review=d.load(OUT/'REVIEW.json');assert review['status']=='PASS'
    assert d.sha(ROOT/'scripts/review_fusion_prediction_quality_v1.py')==review['review_script_sha256']
    for p,h in {**review['hashes'],**review['review_dependencies']}.items():assert d.sha(p)==h,p
    e,c=d.load(OUT/'EVALUATION.json'),d.load(OUT/'CONTRASTS.json')
    diag=diagnostics(e);d.save(OUT/'DIAGNOSTICS.json',diag);figure(e)
    score=d.load(OUT/'SCORES_FROZEN.json');tests=d.load(OUT/'TEST_EXECUTION.json')
    table_head='<thead><tr><th>Method</th><th>PRMB AUROC</th><th>PRMB N</th><th>Within-answer AUC</th><th>PB macro F1</th><th>Fixed-IU-gate PB</th><th>Score coverage</th></tr></thead>'
    main_arms=['dual__equal','dual__iu','dual__cond100','dual__cond100_graph010','ar1__equal','ar1__iu','ar1__joint0','ar1__graph010','ar1__equal_graph010','last__iu','last__graph010','ema32__iu','ema32__graph010']
    primary=''.join(metric_row(a,e,diag) for a in main_arms)
    full=''.join(metric_row(a,e,diag) for a in d.ARMS)
    contrasts=[];md_contrasts=[]
    for p in c['pairs'].values():
        delta=p['left_prm']['auroc']-p['right_prm']['auroc'] if p['left_prm']['auroc'] is not None and p['right_prm']['auroc'] is not None else None
        pb=p['left_pb']['macro_f1']-p['right_pb']['macro_f1'] if p['left_pb']['macro_f1'] is not None and p['right_pb']['macro_f1'] is not None else None
        u=p['uncertainty'];name=html.escape(p['left']+' − '+p['right'])
        contrasts.append(f'<tr><td><code>{name}</code><br>{p["scope"]}; N={p["selected_answers"]}</td><td>{fmt(delta)}<br>{interval(u["prm_common_valid_ci95"])}</td><td>{"—" if pb is None else f"{pb*100:+.3f}"}<br>{interval(u["pb_all_population_ci95"],100)}</td><td>{interval(u["prm_within_answer_common_valid_ci95"])}</td><td>{interval(u["pb_common_iu_gate_all_population_ci95"],100)}</td><td>{u["prm_common_valid_valid_draws"]}/{u["prm_within_answer_common_valid_valid_draws"]}/{u["pb_all_population_valid_draws"]}/{u["pb_common_iu_gate_all_population_valid_draws"]}</td></tr>')
        md_contrasts.append(f'| {p["left"]} minus {p["right"]} | {p["scope"]} ({p["selected_answers"]}) | {fmt(delta)} {interval(u["prm_common_valid_ci95"])} | {fmt(pb,True)} {interval(u["pb_all_population_ci95"],100)} |')
    krows=[]
    for kind in d.KINDS:
        krows.append(f'<tr><td>{kind}</td><td>{review["native_joint_coverage"][kind]}/110</td>'+''.join(f'<td>{review["K_counts"].get(kind+"/"+str(k),0)}</td>' for k in (3,4,6,8))+'</tr>')
    changes=[]
    for kind,x in diag['feature_changes'].items():
        changes.append(f'<tr><td>{kind}</td><td>{x["answers_with_original_sign_changes"]}/110</td><td>{x["median_graph_window_pearson_to_original_graph100"]:.4f}</td><td>{100*x["median_added_absolute_standardized_weight_share"]:.1f}%</td></tr>')
    hitrows=[]
    for arm in ('dual__iu','dual__cond100_graph010','ar1__iu','ar1__graph010','last__iu','ema32__graph010'):
        h=diag['hit_counts'][arm];hitrows.append(f'<tr><td><code>{arm}</code></td><td>{h["clean_hits"]}/33</td><td>{h["error_exact_hits"]}/53</td><td>{h["error_peak_hits"]}/53</td></tr>')
    text=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Prediction views inside IU/Joint — Step 312 quality results</title>
<style>*{{box-sizing:border-box}}body{{margin:0;background:#f3f6f5;color:#183640;font:17px/1.6 system-ui,sans-serif}}main{{max-width:1200px;margin:auto;padding:34px 24px 70px}}h1{{font-size:clamp(30px,5vw,46px);line-height:1.15}}h2{{font-size:26px}}section{{background:white;border:1px solid #d7e3df;border-radius:13px;padding:24px;margin:24px 0}}a{{color:#165f94}}.note{{background:#e4f2ec;border-left:5px solid #087c72;padding:18px}}.warn{{background:#fff0df;border-color:#b36d22}}.small{{font-size:14px;color:#526b75}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #d9e3df;vertical-align:top}}th{{background:#edf4f0}}code{{font-size:.9em;overflow-wrap:anywhere}}img{{max-width:100%;height:auto}}details{{margin:20px 0}}summary{{cursor:pointer;font-weight:650}}.flow{{display:flex;flex-wrap:wrap;gap:10px}}.flow span{{padding:16px;border:1px solid #c5d8d0;border-radius:9px;background:#f3f8f5;flex:1;min-width:150px}}.flow strong{{background:#087c72;color:white;padding:16px;border-radius:9px;flex:1.3;min-width:210px}}@media print{{body{{background:white}}main{{padding:0}}details{{display:block}}}}</style>
</head><body><main>
<p class="small">7 September 2026 · Step 312 · 110 development answers · 98 arms · 74 registered comparisons · Review PASS</p>
<h1>Joint fits better.<br>The added residual views do not locate errors better.</h1>
<p class="note warn"><strong>Decision: keep the existing IU/Joint references.</strong> None of the 21 augmented recipes exceeds both existing dual IU and graph100 Joint on both primary point metrics. Several regressions have exploratory intervals excluding zero. This result does not close Joint, graph methods, KalmanNet or Flows; it rejects promoting this fixed residual-column augmentation.</p>
<div class="flow"><span>One answer<br>Same cached telemetry</span><span>Original 27 features<br>+ nine residual columns</span><strong>Our fusion core<br>IU-PCR / Joint L-SML</strong><span>Same step mapping<br>and no-error decision</span></div>
<p>We tested the addition inside our method. Equal aggregation and equal-graph controls use the same new columns. Last-value and EMA32 residuals test whether fitting AR helps beyond simple dynamics. All old 77 arms remain exact frozen references.</p>

<section><h2>The matched results</h2>
<p>Same 24 PRMBench and 86 ProcessBench answers throughout this main table. Joint maps use condition100, with graph λ=0.1 where shown. PB macro F1 is the mean of four subset harmonic means of clean-answer accuracy and exact first-error accuracy; it is not ordinary binary F1.</p>
<img src="quality_comparison.png" alt="Point comparisons for all seven cores with original features, AR, last-value and EMA32 residual columns. None of the new methods improves both existing leaders on both tasks.">
<p class="small">Point estimates only; paired intervals are below. <a href="quality_comparison.svg">Exportable SVG</a> · <a href="FIGURE_DATA.json">Exact figure data</a>.</p>
<div class="scroll"><table>{table_head}<tbody>{primary}</tbody></table></div>
<p>AR + IU changes PRMB AUROC by −0.02464 (95% CI [−0.05027, −0.00477]) and PB by −6.663 percentage points (CI [−16.424,+1.426]). AR + Joint graph changes PRMB by −0.01492 (CI [−0.03538,+0.00334]) and PB by −12.208 points (CI [−23.824,−0.764]). These are unadjusted exploratory intervals from 74 comparisons.</p>
<p>The graph adds +0.00520 AUROC and +2.465 PB points to AR + Joint λ0, but both intervals include zero. AR + Joint graph versus matched AR + equal-graph is +0.01243 AUROC / −8.398 PB points, with both intervals including zero. There is no two-task fusion advantage from this addition.</p>
<p>Last-value + IU and EMA32 + Joint graph are less harmful than the corresponding AR recipes on PB, but neither exceeds the existing IU/Joint references on both tasks. A different best row for each benchmark is not one leading method.</p></section>

<section><h2>The user's feature/grouping hypothesis was tested</h2>
<p><strong>All 330 augmented Joint fits passed the frozen guards.</strong> Each of the three residual families fits 110/110 answers. The original selected-bank Joint fits were valid on 107/110. Every candidate keeps the same original bank: {review['fixed_banks']['moment']} moment, {review['fixed_banks']['context']} context. No augmented-IU fallback was used.</p>
<div class="scroll"><table><thead><tr><th>Added residual</th><th>Valid native Joint</th><th>K=3</th><th>K=4</th><th>K=6</th><th>K=8</th></tr></thead><tbody>{''.join(krows)}</tbody></table></div>
<p>AR yields more than three groups in 76/110 answers. So representation can change admissible groups and stable fitting. It did not improve localization here. K is the number of groups, not the number of features; every group still has at least three features.</p>
<p>Restricting AR + Joint graph versus old Joint graph100 to the 107 answers with both native fits valid still gives a PB drop of −12.555 points, CI [−24.466,−1.072]. Therefore the three newly valid fits do not explain away the observed regression. This is a conditional diagnostic, not a new full-population score.</p></section>

<section><h2>Both the peak and the no-error decision matter</h2>
<div class="scroll"><table><thead><tr><th>Method</th><th>Clean answers correct</th><th>Errors exactly localized with native gate</th><th>Raw peak at first error</th></tr></thead><tbody>{''.join(hitrows)}</tbody></table></div>
<p>With AR, clean-answer decisions and raw first-error peaks both weaken. Keeping the original IU gate does not restore the original result: AR + Joint graph gives 22.72% fixed-gate PB versus 29.53% for the original graph. PRMB within-answer AUC also falls (0.61029 versus 0.66244). This is more than a pooled score-calibration change.</p>
<p class="small">Hit counts are post-evaluation descriptions, not another selected candidate or an ensemble.</p></section>

<section><h2>Keeping original columns does not freeze their role</h2>
<p>The original 27 raw columns replay exactly. But refitting fusion on 36 columns can change feature signs, grouping and weights. The following diagnostics were added after evaluation and are descriptive.</p>
<div class="scroll"><table><thead><tr><th>Residual</th><th>Any original feature sign changes</th><th>Median new/old graph trajectory Pearson</th><th>Median added |weight| share</th></tr></thead><tbody>{''.join(changes)}</tbody></table></div>
<p>The trajectories are highly correlated, yet their maxima and GMM decisions can differ. Absolute standardized-weight share is not causal importance, and these observations do not prove that sign changes caused the loss. The next useful check is to inspect the changed first-error peaks and clean decisions against actual benchmark spans/text before proposing another component.</p></section>

<section><h2>All 98 arms and all 74 comparisons</h2>
<p>Pure historical Joint rows can have lower coverage and cannot be ranked against full-coverage methods as if their populations matched. Paired PRMB comparisons use common valid IDs. PB retains its full selected population and counts missing decisions as failures. Native scopes condition on fit validity; native-and-original scopes require both fits.</p>
<details><summary>Complete 98-arm metric table</summary><div class="scroll"><table>{table_head}<tbody>{full}</tbody></table></div></details>
<details><summary>Complete 74-comparison uncertainty table</summary><p>Columns show point delta and 95% CI. PB intervals use percentage points. Four draw counts are PRMB / within-answer / native PB / fixed-IU-gate PB, out of 1000. Intervals are unadjusted for the exploratory comparison roster.</p><div class="scroll"><table><thead><tr><th>Comparison and scope</th><th>PRMB Δ / CI</th><th>PB Δ points / CI</th><th>Within-answer Δ CI</th><th>Fixed-IU PB Δ CI</th><th>Defined draws</th></tr></thead><tbody>{''.join(contrasts)}</tbody></table></div></details>
<p>Older 58-answer results remain in <a href="EVALUATION.json">EVALUATION.json</a> and the <a href="../fusion_graph_conditioning_v1/REPORT.html">previous history panel</a>. Cross-answer token B3/CIW/Local-Online work has different fitting and evaluation contracts; see the <a href="../fusion_prediction_view_audit_v1/REPORT.html">innovation-history audit</a>. These are historical context, not matched gains.</p></section>

<section><h2>Review and next decision</h2>
<p>Three policy/algebra tests passed. Review checked 8,470 exact historical method rows, 330 exact augmented matrices, 330 covariance and Jacobian reconstructions, 330 IU refits, 660 Laplacians and 2,310 weight/step/GMM paths. It reproduced all 98 metric bundles and 74 paired point/scope bundles, plus six explicit 1000-draw comparisons. Representative grouping, Joint and DUFS refits cover all five cells × three residual families.</p>
<p>Maximum reconstructed risk discrepancy: {review['maximum_risk_difference']:.2e}. Scoring: {score['seconds_this_invocation']:.2f} s with three CPU workers; paired comparisons: {c['seconds_this_invocation']:.2f} s; review: {review['seconds']:.2f} s. Original eligibility fitting and cached feature generation are reused here, so this is incremental experiment time, not complete deployment time. No new model inference.</p>
<p>The review reused the scientific fitting/graph/GMM kernels and independently reconstructed covariance, inverse/graph algebra, routes, step maps and metrics in this session. Two review-code issues were corrected: a numeric helper could not compare lists of grouping dictionaries, and a bit-exact check was inappropriate for independent sums with different floating-point order (first discrepancy4.44e-16). Frozen scientific sources and predictions were unchanged.</p>
<p class="note"><strong>Continue with failure analysis, not another residual sweep.</strong> Inspect actual first-error spans, near-tied peaks and clean-decision changes on the frozen original and augmented trajectories. Use that evidence to choose a narrowly defined fusion/readout repair. No evidence here justifies switching the method or expanding to a costly learned predictor. Named supporting tracks remain open, along with corrected-fold cross-answer refits, full comparator coverage, untouched confirmation and historical24 transfer.</p>
<p>This cohort was already used for development. No consistent winner or publication confirmation is established. HTML structure/local links and the exported figure are checked separately; no browser rendering check is claimed.</p>
<p><a href="REVIEW.json">Review evidence</a> · <a href="DIAGNOSTICS.json">Post-evaluation diagnostics</a> · <a href="CONTRASTS.json">All comparisons</a> · <a href="MANIFEST.json">Frozen contract</a> · <a href="TESTS.txt">Tests</a> · <a href="../../docs/experiments/FUSION_PREDICTION_QUALITY_V1.md">Protocol</a> · <a href="../../spectral_utils/fusion_prediction_quality.py">Fusion implementation</a> · <a href="../../docs/reviews/joint_lsml_visual_guide_2026-09-06.html">Joint visual guide</a></p></section>
</main></body></html>'''
    (OUT/'REPORT.html').write_text(text,encoding='utf-8')
    md=['# Prediction views inside IU/Joint — Step312','',
        'Decision: retain original IU/Joint. All21 augmented recipes are below both original dual IU and graph100 Joint on both primary point metrics. No consistent winner.',
        '', '| Method | PRMB AUROC | PB macro F1 |','|---|---:|---:|']
    for a in (*d.ANCHORS.values(),*d.NEW_ARMS):
        x=e['metrics'][a];md.append(f'| {a} | {fmt(x["prm"]["auroc"])} | {fmt(x["pb"]["macro_f1"],True)} |')
    md += ['', 'All330 native Joint fits pass; original selected-bank native coverage107/110. Fixed81moment/29context banks; no fallback used. AR has K3/4/6/8 in34/32/39/5 answers. Better fitting did not give better localization.',
        '', 'AR+IU minus original IU: PRMB -0.02464 CI[-0.05027,-0.00477]; PB -6.663pp CI[-16.424,+1.426]. AR+Joint graph minus original graph100: PRMB -0.01492 CI[-0.03538,+0.00334]; PB -12.208pp CI[-23.824,-0.764]. Conditional107 fit-valid PB difference remains negative. All intervals exploratory/unadjusted for74 comparisons.',
        '', 'Original27 raw columns replay exactly, but refitting changes their signs/weights/groups. Post-evaluation sign-change/trajectory-correlation/weight-share and hit diagnostics are descriptive, not causal explanation or new candidate.',
        '', f'Review PASS: {review["counts"]}. Max risk difference {review["maximum_risk_difference"]:.3e}. Scoring {score["seconds_this_invocation"]:.2f}s; contrasts {c["seconds_this_invocation"]:.2f}s; review {review["seconds"]:.2f}s. Three tests; six explicit bootstraps. Scientific kernels reused for specified replays; independent algebra/metrics in same session. No new model inference.',
        '', 'Review harness fixes: recursive grouping-record comparison; established1e-12 independent projection tolerance for direct overlap sums versus difference-array cumsum (first mismatch4.44e-16). Frozen scientific code/predictions unchanged.',
        '', 'Next bounded work: actual benchmark span/text and frozen-score failure analysis (first-error peaks, near ties and clean decisions). Avoid another residual-dose sweep before identifying a mechanism. Keep fusion central and all wider authorized tracks pending.',
        '', 'Full historical98 table and all74 uncertainty/scope comparisons appear in [REPORT.html](REPORT.html); original58 and cross-answer results remain separate context.',
        '', '| Comparison | Scope (N) | PRMB delta / CI | PB delta / CI in pp |','|---|---|---:|---:|',*md_contrasts]
    (OUT/'REPORT.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    parser=Parser();parser.feed(text);assert len(parser.ids)==len(set(parser.ids))
    for href in parser.links:
        url=urlsplit(href);assert not url.scheme and not url.netloc
        path=(OUT/unquote(url.path)).resolve();assert path.is_relative_to(ROOT) and path.is_file(),href
    d.save(OUT/'ARTIFACT_VALIDATION.json',{'status':'PASS','arms_in_full_table':len(d.ARMS),'paired_rows':len(contrasts),
        'local_links_and_images':len(parser.links),'no_browser_rendering_check':True,'report_sha256':d.sha(OUT/'REPORT.html')})
    files=[Path(__file__),OUT/'REVIEW.json',OUT/'DIAGNOSTICS.json',OUT/'REPORT.html',OUT/'REPORT.md',OUT/'quality_comparison.png',OUT/'quality_comparison.svg',OUT/'FIGURE_DATA.json',OUT/'ARTIFACT_VALIDATION.json']
    d.save(OUT/'REPORT_PROVENANCE.json',{'status':'PASS','hashes':{str(p):d.sha(p) for p in files}})
    print('Reports:98 arms,74 comparisons;',len(parser.links),'links/images verified.',flush=True)


if __name__=='__main__':main()
