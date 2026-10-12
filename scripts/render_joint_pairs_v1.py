"""Simple-English visual explanation of the reviewed Joint pair extension."""
from collections import Counter
import hashlib
from html import escape
from html.parser import HTMLParser
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/joint_pair_identifiability_audit_v1'
def load(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def table(head,rows):
    return '<div class="table"><table><thead><tr>'+''.join('<th scope="col">'+escape(x)+'</th>' for x in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(str(x))+'</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table></div>'
def mdtable(head,rows):return '\n| '+' | '.join(head)+' |\n|'+'|'.join(['---']*len(head))+'|\n'+'\n'.join('| '+' | '.join(str(x) for x in r)+' |' for r in rows)+'\n'


def render():
    m,f,s,a,review=[load(n) for n in ('MANIFEST.json','FROZEN.json','SUMMARY.json','ALGEBRA.json','REVIEW.json')]
    assert review['status']=='PASS' and not review['labels_decoded']
    assert not review['jacobian_amendment']['guard_changes']
    for p,h in {**review['hashes'],**review['dependencies']}.items():assert sha(p)==h,p
    assert sha(ROOT/'scripts/review_joint_pairs_v1.py')==review['review_script_sha256']
    rows=[load('rows/'+r['uid']+'.json') for r in m['selected']]
    sections=[];md=['# Allowing feature pairs inside Joint L-SML\n\n2026-09-07. Reviewed mathematical and unlabeled structural audit.\n']
    def section(title,text,head=None,data=None,extra=''):
        sections.append('<section><h2>'+escape(title)+'</h2><p>'+escape(text)+'</p>'+(table(head,data) if head else '')+extra+'</section>')
        md.append('\n## '+title+'\n\n'+text+'\n'+(mdtable(head,data) if head else ''))
    section('The useful result',
        'A careful extension can admit pairs of features without replacing our fusion. Joint fit coverage rises from 78 to 106 '
        'of 110 answers for the moment bank, and from 102 to 108 for context. At least one bank has a valid Joint fit for every '
        'answer, versus 107 previously. This earns a localization experiment; it does not yet show higher AUC or better error detection. '
        'No error labels were read in this stage.')
    section('Where this fits in our method',
        'The input is still N windows from one answer by P feature measurements. Joint estimates a global factor shared across '
        'features and a local dependence factor inside each feature group. Its native fusion weights solve a regularized covariance '
        'system. We change how a two-feature group is represented inside that same covariance model.',
        extra='<div class="flow"><div>N x P matrix<br><small>One answer; existing moment/context features</small></div><div>Feature groups<br><small>Allow size 2 under explicit checks</small></div><div><strong>Joint L-SML fusion</strong><br><small>Same global/group covariance idea</small></div><div>Fused trajectory<br><small>Next: evaluate steps and no-error decisions</small></div></div>')
    section('One equation cannot identify two loadings',
        'After removing the global contribution v_i*v_j from the covariance between two features, Joint fits the residual '
        'r = u_i*u_j. A residual of 0.20 can come from (0.5, 0.4), (1.0, 0.2), or many other pairs. The product is determined, '
        'but the individual loadings are not. Choosing equal-magnitude loadings is a convention. Claude\'s report and R1 amendment '
        'describe this as an exactly determined system; that explanation is too strong.',
        extra='<div class="formula">u_i &rarr; t u_i &nbsp;&nbsp; u_j &rarr; u_j / t<br><strong>The product stays fixed for any positive t.</strong></div>')
    section('Why this matters for the code',
        'An arbitrary loading ratio is harmless to the native fusion if its covariance stays unchanged. However, the old constructor '
        'computes diagonal noise as observed variance minus factor variance, then clips negative values to zero. A rescaled pair '
        'can exceed the available variance, inflate the covariance diagonal and change the final weights while keeping the same '
        'off-diagonal fitting objective. The existing global-loading Jacobian still passes in our counterexample. That test '
        'checks global loadings after removing nuisance directions; it is not a full covariance/head invariance test. The frozen '
        'minimum-three experiment is preserved; the counterexample tests what would happen if pairs were admitted naively.')
    family=[[x['pair_scale'],f"{x['legacy_diagonal'][0]:.5f}",f"{x['legacy_diagonal'][1]:.5f}",
             f"{x['legacy_weights'][0]:.5f}",f"{x['legacy_weights'][1]:.5f}",
             f"{x['pair_covariance_weights'][0]:.5f}",f"{x['pair_covariance_weights'][1]:.5f}"] for x in a['family']]
    section('A numerical counterexample you can inspect',
        'This is a synthetic covariance with six features in three pairs, not a benchmark answer. All rows below have identical '
        'off-diagonal covariances and pass the global Jacobian check. The old clipped weights vary; the new pair covariance '
        'and its weights stay fixed. The full off-diagonal Jacobian has rank 9 for 12 latent loadings: three pair-scale directions '
        'remain undetermined.',
        ['Scale t','Old variance 1','Old variance 2','Old weight 1','Old weight 2','New weight 1','New weight 2'],family,
        extra='<div class="explore"><label for="example">Inspect pair scale: </label><select id="example">'+''.join('<option value="'+str(i)+'"'+(' selected' if x['pair_scale']==1 else '')+'>'+str(x['pair_scale'])+'</option>' for i,x in enumerate(a['family']))+'</select><p id="example-reading" aria-live="polite">At scale 1, both constructors use the same observed variances. Choose another scale to see the difference.</p></div>')
    section('The pair representation we implemented',
        'Let b_i = S_ii - v_i^2 be the variance left after the global factor, and similarly b_j. The pair is feasible exactly '
        'when both budgets are nonnegative and |r| <= sqrt(b_i*b_j). We allocate the same fraction of each budget to the group '
        'factor. This preserves the fitted product and the observed diagonal. It is a declared representative of equivalent '
        'loadings, not recovered latent truth. The invariant statement is conditional on the same global loadings, pair product '
        'and other groups. Near-roundoff adjustments are recorded; substantive infeasibility is rejected.',
        extra='<div class="formula">u_i<sup>2</sup> = |r| &radic;(b_i / b_j)<br>u_j<sup>2</sup> = |r| &radic;(b_j / b_i)</div><p>Zero products and zero budgets have explicit handling. Groups of size three or more retain the legacy constructor. A pair fit also needs agreement of the resulting native covariances and weights across converged starts, plus the existing convergence and profiled-global-Jacobian checks.</p>')
    section('A second review finding: zero pair products',
        'At u_i=u_j=0, the old first-order nuisance derivative vanishes. That can make its global-identification check pass '
        'even though a different global loading can be absorbed by a changed pair product. The amended check represents '
        'each pair by its product directly, including at zero. A synthetic counterexample passes the old check and correctly '
        'fails the amended one. This was added transparently during review, after the 110 fits completed; the original '
        'frozen prototype and fitted arrays are preserved. All 219 fitted records were checked again, with an independent '
        'product-profile reconstruction on 69 pair fits. No current eligibility status changes, and none of those pair products '
        'is exactly zero. Three additional tests pass. Future experiments must call fit_joint_pairs_checked from '
        'joint_pair_jacobian.py, rather than the earlier prototype directly.')
    section('What was held fixed',
        'The 110-answer cohort, both 27-coordinate feature banks, width-eight windows, answer-only normalization and fixed '
        'negative-entropy anchor, four chronological blocks, seed, K={3,4,6,8}, held-block admissibility fraction 0.95, five '
        'optimizer starts, 5,000-sweep cap and inverse condition target 1000 were retained. We lowered the group minimum to two '
        'and added the pair covariance/validity treatment. We did not simultaneously widen K or run a condition-number sweep. '
        'The lower minimum also applies to held-block partitions: a final partition without a pair can become admissible '
        'because one held-block partition contains a pair. No inference, graph-quality or hierarchical-head experiment was run.')
    coverage=[[bank,b['old_valid'],b['new_valid'],b['rescued'],b['lost'],b['valid_pair_fits'],b['same_partition_as_parent']]
              for bank,b in s['banks'].items()]
    section('Matched structural comparison on the same 110 answers',
        'Higher coverage is useful, but not a correctness metric. Changing the selected partition can lose an old successful '
        'fit, so rescues and losses are both shown. The union across banks is 110/110; this is a potential route, not a new '
        'measured localization score. The previous quality results and all benchmark targets remain unchanged.',
        ['Bank','Old valid','New valid','Rescued','Lost','Valid fits with a pair','Same selected partition'],coverage)
    ks=[]
    for bank,b in s['banks'].items():
        old=Counter(r['banks'][bank]['old_grouping'].get('K') for r in rows)
        for k in (3,4,6,8):ks.append([bank,k,old.get(k,0),b['selected_K'].get(str(k),0),b['valid_K'].get(str(k),0)])
    section('More groups now become possible',
        'For moment features, 60 of 110 selected partitions now use six or eight groups; previously every admissible partition '
        'had three or four. There are 55 valid moment fits and 12 valid context fits whose final partition actually includes '
        'a pair. Context still selects only three or four groups. A larger K is not itself evidence of better fusion.',
        ['Bank','K','Old selected','New selected','New valid'],ks)
    cells=[]
    for cell in dict.fromkeys(r['cell'] for r in m['selected']):
        rr=[r for r in rows if r['cell']==cell]
        cells.append([cell,len(rr)]+[str(sum(r['banks'][b]['old_valid'] for r in rr))+' -> '+str(sum(r['banks'][b]['valid'] for r in rr)) for b in ('moment','context')])
    section('Coverage by benchmark cell',
        'All counts use the same fixed source questions as the preceding replication. The current cache is development data '
        'already evaluated elsewhere in the project; the lack of label reads in this audit does not make it untouched confirmation.',
        ['Cell','Answers','Moment: old -> new','Context: old -> new'],cells)
    section('The remaining failures are real',
        'Five fits fail convergence/multistart guards: three moment and two context. One additional moment fit has an infeasible '
        'pair. Its variance budgets are 0.0299446 and 0.00212325, permitting a residual product magnitude at most 0.00797371; '
        'the fitted product is 0.0145291. The extension rejects it instead of inflating a diagonal. We do not pick another '
        'partition after seeing this outcome in this stage. This feasibility guard deliberately changes how a pair-related '
        'diagonal conflict is handled; it is not a general repair of all historical diagonal clipping.')
    section('Review and limits',
        'Eight original scientific tests and three review-amendment tests pass, including a real pair fit, exact minimum-three replay, scaling invariance, signed and zero '
        'products, infeasible variances, permutation/sign equivariance and a finite-difference Jacobian. Independent review '
        'reconstructs 220 normalized feature covariances, 880 candidate-group guards/ARI summaries, all 220 selections, 219 '
        'native inverse maps, 107 pair products/variance allocations, the one infeasibility, and 117 unchanged-partition parent '
        'maps. Nine representative pair refits match. The original clustering labels and optimizer are reused; this is not '
        'an independent clustering/optimizer implementation. The mathematical example demonstrates a possible failure of '
        'naive pair admission, not that every unconstrained pair fit in practice would fail.',
        ['Check','Evidence'],[['Scoring wall time, three CPU workers',f"{f['seconds_this_invocation']:.2f} s"],
            ['Independent review',f"{review['seconds']:.2f} s"],['Maximum native-weight reconstruction difference',review['max_native_weight_difference']],
            ['Benchmark error-label reads in this stage','None'],['Browser visual inspection','Not run; structural/link checks only']])
    section('Next: measure the contribution to localization',
        'Freeze one pair-enabled native Joint recipe with lambda zero, graph lambda 0.1 and permuted-graph controls. Compare '
        'it on shared development answers with the frozen minimum-three versions, moment/dual IU, both equal banks and '
        'matched routing controls. Preserve native no-error gates and fixed-IU diagnostics separately. Report failures and '
        'common-ID ranking alongside full-population decisions. Only those results can tell us whether the added fitting '
        'capacity helps. Hierarchical heads, wider hyperparameter work, supporting temporal/geometry/sampling ideas, corrected-fold '
        'multi-answer contenders, full comparator coverage, untouched confirmation and historical 24-cell transfer remain open.')
    links=[('SUMMARY.json','Structural results'),('ALGEBRA.json','Counterexample arrays and weights'),('REVIEW.json','Independent review'),('TESTS.txt','Eight original scientific tests'),
        ('JACOBIAN_TESTS.txt','Three zero-product review tests'),('JACOBIAN_AMENDMENT.json','Post-fit review amendment and source hashes'),
        ('MANIFEST.json','Frozen inputs and scientific source hashes'),('../fusion_replication_v1/REPORT.html','Previous quality results and historical comparisons'),
        ('../../spectral_utils/joint_pair_extension.py','Pair covariance construction and optimizer wrapper'),
        ('../../spectral_utils/joint_pair_jacobian.py','Checked entry point for future pair-fusion experiments'),
        ('../../docs/experiments/JOINT_PAIR_IDENTIFIABILITY_AUDIT_V1.md','Frozen audit protocol'),
        ('../../docs/reviews/joint_lsml_visual_guide_2026-09-06.html','Visual guide to the existing fusion method')]
    sections.append('<section><h2>Evidence and code</h2><ul>'+''.join('<li><a href="'+p+'">'+escape(t)+'</a></li>' for p,t in links)+'</ul></section>')
    md.append('\n## Evidence and code\n\n'+'\n'.join('- ['+t+']('+p+')' for p,t in links))
    # The interactive selector displays only precomputed algebra examples.
    script='const cases='+json.dumps(a['family'])+''';const select=document.getElementById('example');select.addEventListener('change',()=>{const c=cases[Number(select.value)];document.getElementById('example-reading').textContent='Scale '+c.pair_scale+': old pair variances '+c.legacy_diagonal.slice(0,2).map(x=>x.toFixed(5)).join(', ')+'. Old weights '+c.legacy_weights.slice(0,2).map(x=>x.toFixed(5)).join(', ')+'. Pair-aware weights '+c.pair_covariance_weights.slice(0,2).map(x=>x.toFixed(5)).join(', ')+'. Off-diagonal covariances stay fixed.';});'''
    css='''*{box-sizing:border-box}body{margin:0;background:#f4f6f3;color:#193d45;font:17px/1.65 system-ui,Segoe UI,sans-serif}header{background:#153f48;color:white;padding:45px max(24px,calc((100vw - 1100px)/2))}h1{font-size:clamp(32px,5vw,52px);line-height:1.12}main{max-width:1148px;margin:auto;padding:20px 24px 50px}section{margin:24px 0;padding:25px;background:white;border:1px solid #dce5df;border-radius:14px}h2{line-height:1.25;font-size:27px}.flow{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.flow>div,.formula,.explore{background:#e8f2ec;border-radius:8px;padding:18px}.flow>div{border-top:4px solid #137f70}.flow>div:nth-child(3){background:#153f48;color:white}.formula{font:18px/1.8 ui-monospace,Consolas,monospace}.table{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}td,th{text-align:left;padding:11px;border-bottom:1px solid #dce5df;vertical-align:top}th{background:#e8f2ec}td:first-child{font-weight:650}a{color:#075c90}select{font:inherit;padding:6px}a:focus-visible,select:focus-visible{outline:3px solid #b57715}small{font-size:14px}@media(max-width:720px){.flow{grid-template-columns:1fr}main{padding:12px}section{padding:18px}}@media print{body{background:white;font-size:11pt}header{background:white;color:#193d45;padding:15px}section{border:0;padding:12px}h2{break-after:avoid}.table{overflow:visible}td,th{padding:6px}table{font-size:9pt}.explore{display:none}}'''
    html='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Feature pairs inside Joint L-SML</title><style>'+css+'</style></head><body><header><p>07 September 2026 · Reviewed structural experiment</p><h1>More flexible groups.<br>The same Joint fusion method.</h1><p>Why pairs need care, what we changed, and what the 110-answer audit actually proves.</p></header><main>'+''.join(sections)+'</main><script>'+script+'</script></body></html>\n'
    (OUT/'REPORT.html').write_text(html,encoding='utf-8',newline='\n');(OUT/'REPORT.md').write_text('\n'.join(md)+'\n',encoding='utf-8',newline='\n')
    class Check(HTMLParser):
        def __init__(self):super().__init__();self.stack=[];self.links=[]
        def handle_starttag(self,t,attrs):
            if t not in ('meta','br','input','hr','link','img'):self.stack.append(t)
            if t=='a':self.links.append(dict(attrs)['href'])
        def handle_endtag(self,t):assert self.stack.pop()==t,t
    check=Check();check.feed(html);check.close();assert not check.stack
    for p in check.links:assert (OUT/p).resolve().exists(),p
    provenance={'status':'PASS','hashes':{str(OUT/n):sha(OUT/n) for n in ('MANIFEST.json','FROZEN.json','SUMMARY.json','ALGEBRA.json','REVIEW.json','REPORT.html','REPORT.md')},'renderer_sha256':sha(__file__)}
    (OUT/'REPORT_PROVENANCE.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    (OUT/'ARTIFACT_VALIDATION.json').write_text(json.dumps({'status':'PASS','html_structure':'PASS','local_links':len(check.links),'browser_visual_check':'NOT_RUN','provenance_sha256':sha(OUT/'REPORT_PROVENANCE.json')},indent=2),encoding='utf-8')
    print('Rendered REPORT.html / REPORT.md. Structure and',len(check.links),'local links pass.')


if __name__=='__main__':render()
