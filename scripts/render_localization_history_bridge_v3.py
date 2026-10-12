"""Make historical corrections reviewable without combining unmatched cohorts."""
from html import escape
import importlib.util
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('history_report_driver',ROOT/'scripts/bridge_localization_history_v3.py')
d=importlib.util.module_from_spec(s);s.loader.exec_module(d)
OUT=d.OUT
def num(x,n=5):return 'n/a' if x is None else f'{x:.{n}f}'
def pct(x):return 'n/a' if x is None else f'{x*100:.2f}%'
def ci(x,scale=1):return 'n/a' if x is None else '['+', '.join(f'{v*scale:+.4f}' for v in x)+']'
def table(headers,rows,identity):
    return '<div class="scroll"><table id="'+identity+'"><thead><tr>'+''.join('<th>'+escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def main():
    m=d.verify();reg=d.load(OUT/'REGISTRY.json');review=d.load(OUT/'REVIEW.json');assert review['status']=='PASS'
    for p,h in {**review['hashes'],**review['dependencies']}.items():assert d.sha(p)==h,p
    lanes={lane:d.load(OUT/(lane+'_EVALUATION_V3.json')) for lane in m['lanes']}
    contrasts={lane:d.load(OUT/(lane+'_CONTRASTS_V3.json')) for lane in m['lanes']}
    gate=d.load(OUT/'gate_DIAGNOSTICS_V3.json');current=d.load(reg['current110_external_evaluation'])
    def entry(row):
        a=row['metric'];old=row['previous_metric'];return [row['lane'],row['arm'],a['prm']['answers'],num(old['prm']['auroc']),
            num(a['prm']['auroc']),num(a['prm']['within_answer_auc']),pct(a['pb']['macro_f1']),
            sum(v['valid_decisions'] for v in a['pb']['cells'].values())]
    headers=['Lane','Arm','Valid PRMB /12','Old-label PRMB AUC','Corrected PRMB AUC','Corrected within-answer AUC','PB F1: unchanged','Valid PB decisions /46']
    early=table(headers,[entry(x) for x in reg['entries']],'earlier-table')
    current_rows=[]
    for arm,bundle in current['metrics'].items():current_rows.append([arm,bundle['prm']['answers'],num(bundle['prm']['auroc']),num(bundle['prm']['within_answer_auc']),pct(bundle['pb']['macro_f1']),sum(v['valid_decisions'] for v in bundle['pb']['cells'].values())])
    pairs=[]
    for lane,items in contrasts.items():
        for p in items['pairs'].values():
            u=p['uncertainty'];a,b=p['left_prm']['auroc'],p['right_prm']['auroc'];pa,pb=p['left_pb']['macro_f1'],p['right_pb']['macro_f1']
            pairs.append([lane,p['left']+' minus '+p['right'],p['left_prm']['answers'],num(a-b if a is not None and b is not None else None),
                ci(u['prm_common_valid_ci95']),num((pa-pb)*100,3),ci(u['pb_all_population_ci95'],100),ci(u['prm_within_answer_common_valid_ci95'])])
    shortlist=[('readout','moments27_local8__iu@@parent_first'),('readout','moments27_local8__iu@@parent_peak'),
        ('readout','moments27_local8__iu@@hmm_entry'),('readout','moments27_local8__iu@@kalman_level'),
        ('readout','moments27_local8__iu@@imm_level'),('readout','moments27_local8__iu@@bocpd_rise'),
        ('sampling','moments27_local8__iu@@risk_top'),('sampling','moments27_local8__iu@@dufs_transposed'),
        ('sampling','moments27_local8__iu@@dufs_permuted'),('context','context__equal'),('context','context__joint0')]
    selected=[]
    for lane,arm in shortlist:
        q=lanes[lane]['metrics'][arm];selected.append([lane,arm,q['prm']['answers'],num(q['prm']['auroc']),num(q['prm']['within_answer_auc']),pct(q['pb']['macro_f1'])])
    g=gate['summaries']['iu_parent']['prm'];sampling=lanes['sampling'];pair=contrasts['sampling']['pairs']['moments27_local8__joint_lambda0@@dufs_transposed minus moments27_local8__joint_lambda0@@full']
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
    vals=[sampling['metrics']['moments27_local8__joint_lambda0@@full']['prm']['auroc'],pair['left_prm']['auroc'],pair['right_prm']['auroc']]
    axes[0].barh([2,1,0],vals,color=['#6687a9','#c97a33','#265b87']);axes[0].set_yticks([2,1,0],['Original Joint: 7 valid answers','DUFS-selected Joint: 4 answers','Original Joint: same 4 answers'])
    for y,v in zip([2,1,0],vals):axes[0].text(v+.007,y,f'{v:.3f}',va='center')
    axes[0].set_xlim(0,.88);axes[0].set_xlabel('Corrected PRMB pooled AUROC');axes[0].set_title('Coverage changes the comparison')
    for x,kind in enumerate(['risk','origin_projection']):
        axes[1].bar(x-.17,g[kind]['pooled_auc'],width=.3,color='#265b87',label='Pooled AUC' if x==0 else None)
        axes[1].bar(x+.17,g[kind]['mean_within_answer_auc'],width=.3,color='#c97a33',label='Within-answer AUC' if x==0 else None)
    axes[1].set_xticks([0,1],['IU score','Same score + answer offset']);axes[1].set_ylim(0,.85);axes[1].set_ylabel('Corrected PRMB AUROC');axes[1].set_title('A pooled gain can leave localization unchanged');axes[1].legend(loc='upper left',fontsize=9)
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Historical58 development answers — label correction is not a method gain',fontsize=12)
    fig.savefig(OUT/'comparison_pitfalls.svg',bbox_inches='tight');fig.savefig(OUT/'comparison_pitfalls.png',dpi=170,bbox_inches='tight');plt.close(fig)
    gate_rows=[]
    for arm,summary in gate['summaries'].items():
        q=summary['prm'];gate_rows.append([arm,num(q['risk']['pooled_auc']),num(q['origin_projection']['pooled_auc']),num(q['risk']['mean_within_answer_auc']),num(q['origin_projection']['mean_within_answer_auc']),pct(q['risk']['cross_pair_fraction'])])
    recovery=d.load(OUT/'EXECUTION_RECOVERY.json');counts=review['counts']
    text=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Corrected historical fusion evidence — Step316</title>
<style>body{{margin:0;background:#f4f7fa;color:#21344b;font:17px/1.6 system-ui,sans-serif}}main{{max-width:1150px;margin:auto;padding:32px 24px 70px}}h1{{font-size:2.15rem;line-height:1.2}}h2{{margin-top:2rem}}a{{color:#185b94}}.card{{background:white;border:1px solid #d3dfeb;border-left:5px solid #b36b2f;padding:20px;border-radius:10px;margin:22px 0}}.muted{{font-size:.93rem;color:#53677c}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:10px;border-bottom:1px solid #d9e2ec;text-align:left;vertical-align:top}}th{{background:#eaf0f6}}td:nth-child(2){{overflow-wrap:anywhere;min-width:180px}}.scroll{{overflow:auto;background:white;border:1px solid #d4e0eb;border-radius:9px;margin:20px 0}}img{{max-width:100%;height:auto}}input{{font:inherit;padding:10px;border:1px solid #8197ae;border-radius:6px;width:360px;max-width:90%}}[hidden]{{display:none!important}}details{{margin-top:25px}}summary{{cursor:pointer;font-weight:650}}code{{background:#e7edf4;padding:2px 5px}}@media print{{body{{background:white}}.search{{display:none}}.scroll{{overflow:visible}}}}</style></head>
<body><main><p class="muted">Step316 · September7, 2026 · Reviewed historical correction · No new model fits or predictions</p>
<h1>Our older fusion experiments now use the corrected benchmark</h1>
<div class="card"><strong>131 earlier method entries and199 comparisons repaired; no new winner established.</strong><p>The repair covers representation, chronological readouts, window sampling, regularization and context banks on the original58 answers. It also retains25 previously corrected fallback references and repairs seven gate diagnostics. The separate current110 panel contains107 entries. Repeated controls are kept for traceability; these counts are not independent new methods.</p></div>
<p>PRMB's raw error-step indices are one-based. The earlier derived labels placed errors one step late. This bridge changes11/12 PRMB target arrays in every early lane and replaces all58 group identities with the canonical source-question map. Scores, weights, windows, failures, predictions and PB metrics are unchanged. Group renaming is not58 new independent observations.</p>
<h2>What the corrected evidence says</h2>
{table(['Lane','Method','Valid PRMB /12','PRMB AUC','Within-answer AUC','PB F1 /46 answers'],selected,'selected-results')}
<p><strong>Readout matters:</strong> first-threshold crossing and peak selection have identical IU score arrays and PRMB AUC0.64753, but PB rises from0% to17.71% when the peak is returned. That is an explicit readout change, not an improvement in feature-fusion weights. HMM, ordinary Kalman, IMM and BOCPD adaptations do not establish a two-task advantage over the peak/hold controls. These are the actual Step299 supporting readouts; they do not stand in for actual KalmanNet, LOCA or Diverging Flows.</p>
<p><strong>Sampling deserves a controlled follow-up, not promotion:</strong> IU fitted on the highest-entropy windows gives0.66850/19.05% versus full-grid0.64753/17.71%. Paired intervals are[−0.00827,+0.07477] for PRMB and[0,+8.929] PB points; they include zero. The mean within-answer AUC falls0.70397→0.69799. Four PB predictions change: one correct answer is lost and one gained. Clean/error successes remain12/4, so the macro-F1 increase reflects which subset benefits, not more overall correct decisions. Equal fusion under the same risk selection has higher PRMB0.67903 but lower PB11.13%; the permuted-DUFS IU control has higher PB21.88% but lower PRMB0.64487.</p>
<p>Only37/58 answers were sampling-eligible (29 PB,8 PRMB). All window features were still computed. The old pilot had no≤32-token first-error spans among eligible PB answers. It does not establish sparse end-to-end compute savings or short-error retention. Uniform, risk-based and graph/permuted controls all remain visible below.</p>
<h2>Two easy ways to overstate a result</h2><img src="comparison_pitfalls.png" alt="Joint DUFS comparison changes when using the same four valid answers; adding an answer offset changes pooled IU AUC but not within-answer AUC.">
<p><strong>Coverage:</strong> Joint with transposed-DUFS fitting rows reaches PRMB0.76333 on just four valid answers. The original Joint score on those SAME four answers is0.75000, rather than its0.62579 over seven valid answers. The paired difference interval[−0.04167,+0.06667] includes zero. PB still counts all46 answers and their failed fits:13.26% versus12.50%. A high partially covered row is not evidence of a winning method.</p>
<p><strong>Score offsets:</strong> restoring an answer-specific constant raises pooled IU AUC0.64753→0.69240 while within-answer AUC remains0.70397 and the gate/peak decisions remain unchanged. About89.34% of the positive-negative pairs in this pooled metric come from different answers. This is useful evidence about the metric; it is not improved localization.</p>
{table(['Diagnostic arm','Risk pooled AUC','Offset pooled AUC','Risk within-answer AUC','Offset within-answer AUC','Cross-answer pair fraction'],gate_rows,'gate-table')}
<h2>What changes next</h2><p>Keep IU-PCR and Joint L-SML as the cores. The next bounded experiment should test the observation-selection idea on the already fixed current110 answers: start from the original feature banks and routes, retain full-grid, uniform, entropy-risk and graph/permutation controls, and measure exact-error retention and fit coverage. First freeze a small matched design and inspect target-free budget feasibility; preserve the inherited current107 anchors. This is a development replication, not an untouched confirmation or a promised gain.</p>
<p>Do not widen the earlier lambda grid or treat improved grouping coverage as correctness evidence. The corrected larger-lambda and regularization results do not establish consistent improvement. Context equal fusion remains a strong earlier58 control; its scores are not a substitute for a comparison on the current110 answers.</p>
<p>The full goal remains open: Joint/graph representation and IU improvements, feature and trajectory fusion, the named supporting tracks, corrected-label/fold multi-answer refits, full relevant comparators, untouched confirmation and later historical24 transfer. This bridge is scoped to the listed sources, not all repository/Claude/Drive history or the earliest short-cycle formats.</p>
<h2>How this was checked</h2><p>Review PASS:58 unique raw-annotation joins through the official PRMB port/PB contract;{counts['exact_unchanged_row_payloads']} full unchanged row payloads;{counts['source_array_and_prediction_records']} original stored score/prediction records;131 independent metric bundles plus25 repaired anchor bundles;199 paired scope/point checks; five explicit1000-draw source-group bootstrap replays;14 independent flattened AUC decompositions. Original PB oracle/gate diagnostics are unchanged and remain diagnostic.</p>
<p>Twelve archived diagnostic representation scores remain present with invalid fit/decision flags (three FINITE_UNCONVERGED_DESCRIPTIVE and nine FIT_DIAGNOSTIC_ONLY). The review initially assumed invalid scores were absent, then allowed only one status; source inspection corrected both assumptions and verified those arrays remain excluded from primary metrics. No scientific output was changed to make the review pass. The source metadata reader, official port and frozen scores are shared; this was a same-session review, not an external reviewer.</p>
<p>Two Windows atomic checkpoint writes failed. A separate bounded retry wrapper resumed the saved results without changing frozen scientific code; all199 comparisons are now complete. An early review attempt stopped on the missing completion marker. Correction took{reg['seconds']:.2f}s; the final recovery invocation took{recovery['seconds_this_resume_invocation']:.2f}s, excluding earlier partial contrast invocations; successful review{review['seconds']:.2f}s. No exact aggregate runtime is claimed.</p>
<h2>Full earlier58 ledger:156 entries</h2><p>12 PRMB +46 PB answers. Old-label AUC is shown only to document the correction. An increase between old and corrected columns is not an algorithm gain. Pure Joint rows have varying coverage. Fallback-reference rows were already corrected in Step313.</p>
<p class="search"><label for="early-search">Find an earlier method: </label><input id="early-search" type="search" placeholder="For example: imm, sampling, joint"><span id="early-count" aria-live="polite"></span></p>{early}
<details><summary>Current110 context:107 entries — different answers, no cross-cohort gain claim</summary><p>24 PRMB +86 PB answers. These are the unchanged Step315 corrected reference/result rows. A later higher number can reflect a different cohort.</p><p class="search"><label for="current-search">Find a current method: </label><input id="current-search" type="search" placeholder="For example: dual__iu, gap__"><span id="current-count" aria-live="polite"></span></p>
{table(['Arm','Valid PRMB /24','PRMB AUC','Within-answer AUC','PB F1','Valid PB decisions /86'],current_rows,'current-table')}</details>
<details><summary>All199 corrected paired comparisons</summary><p>PRMB uses common valid answers; PB includes failures on all46 answers. Source-group resampling uses1000 draws with seed2026090706. The original contrast rosters and row order are preserved. Intervals are exploratory and unadjusted. The representation <code>common_ids</code> field does not mean its PB endpoint used that subset; source code and numerical replay confirm full-population PB.</p>
{table(['Lane','Left minus right','Common valid PRMB','PRMB ΔAUC','PRMB95% interval','PB Δpoints','PB95% interval, points','Within-answer95% interval'],pairs,'pairs-table')}</details>
<h2>Source files</h2><ul><li><a href="../../docs/experiments/LOCALIZATION_HISTORY_BRIDGE_V3.md">Frozen correction protocol</a></li><li><a href="REGISTRY.json">156-entry registry</a></li><li><a href="REVIEW.json">Review evidence</a></li><li><a href="EXECUTION_AMENDMENT.json">Execution recovery amendment</a></li><li><a href="comparison_pitfalls.svg">Export figure</a></li><li><a href="../localization_prm_label_audit_v1/REPORT.html">Raw-label correction and earlier32 fallback contrasts</a></li><li><a href="../fusion_token_gap_v1/REPORT.html">Current110 results</a></li><li><a href="../../scripts/bridge_localization_history_v3.py">Bridge source</a></li><li><a href="../../scripts/review_localization_history_bridge_v3.py">Review source</a></li></ul>
<p class="muted">HTML structure, displayed metrics, links, hashes and actual JavaScript filters are checked separately. No browser visual rendering is claimed.</p>
</main><script>function attachFilter(inputId,tableId,countId){{const input=document.getElementById(inputId);const rows=Array.from(document.querySelectorAll('#'+tableId+' tbody tr'));function apply(){{let count=0;const q=input.value.trim().toLowerCase();for(const row of rows){{row.hidden=!row.textContent.toLowerCase().includes(q);if(!row.hidden)count++;}}document.getElementById(countId).textContent=' '+count+' / '+rows.length;}}input.addEventListener('input',apply);apply();}}attachFilter('early-search','earlier-table','early-count');attachFilter('current-search','current-table','current-count');</script></body></html>'''
    (OUT/'REPORT.html').write_text(text,encoding='utf-8')
    lines=['# Historical localization bridge — Step316','','Review PASS.131 earlier method entries,25 repaired fallback anchors,199 contrasts and7 gate diagnostics.','Same original58 answers; current110/107 entries shown separately. No new fits or predictions.','',
        '| Lane | Arm | Valid PRMB | Corrected AUC | Within-answer AUC | PB F1 |','|---|---|---:|---:|---:|---:|']
    lines+=['| '+' | '.join(map(str,row))+' |' for row in selected]
    lines+=['','Joint DUFS AUC0.76333 covers only4 answers; original Joint is0.75000 on those same4.','Entropy-risk IU shows0.66850/19.05% versus0.64753/17.71%, but both paired intervals include0, within-answer AUC falls, and PB loses one success/gains one. No winner.','An answer offset raises pooled IU AUC0.64753->0.69240 without changing within-answer ranking or decisions.','Next: bounded sampling replication on the current110 with original banks/routes, matched controls, budget feasibility and short-error retention reporting. Full research goal remains open.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    d.save(OUT/'REPORT_PROVENANCE.json',{'status':'REVIEWED_CORRECTION','renderer_sha256':d.sha(Path(__file__)),
        'source_hashes':{str(OUT/n):d.sha(OUT/n) for n in ('MANIFEST.json','REGISTRY.json','REVIEW.json','gate_DIAGNOSTICS_V3.json','EXECUTION_RECOVERY.json')},
        'report_sha256':d.sha(OUT/'REPORT.html'),'figure_sha256':d.sha(OUT/'comparison_pitfalls.svg')})
    print('Rendered156 earlier entries,107 current entries,199 contrasts and7 gate diagnostics.',flush=True)


if __name__=='__main__':main()
