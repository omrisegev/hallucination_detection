"""Simple-English advisor-facing feasibility report, generated from reviewed data."""
import html
from html.parser import HTMLParser
import importlib.util
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prediction_report_io', ROOT/'scripts/audit_fusion_prediction_view_v1.py')
io = importlib.util.module_from_spec(spec); spec.loader.exec_module(io)
OUT = io.OUT


class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.links=[]; self.ids=[]; self.titles=0
    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == 'a' and 'href' in values: self.links.append(values['href'])
        if 'id' in values: self.ids.append(values['id'])
        if tag == 'title': self.titles += 1


def main():
    io.verify(); review = io.load(OUT/'REVIEW.json'); assert review['status']=='PASS'
    for p,h in review['hashes'].items(): assert io.sha(p)==h,p
    summary = io.load(OUT/'SUMMARY.json'); frozen = io.load(OUT/'AUDIT_FROZEN.json')
    for p,h in frozen['files'].items(): assert io.sha(p)==h,p
    tests = io.load(OUT/'TEST_EXECUTION.json')
    ratios = summary['prediction_ratios']
    table=[]; mdrows=[]
    for a,b in zip(ratios['ar1_over_last'], ratios['ar1_over_ema32']):
        assert a['primitive']==b['primitive']
        label=a['primitive'].removesuffix('_series')
        table.append(f'<tr><td>{html.escape(label)}</td><td>{a["median"]:.3f}</td><td>{b["median"]:.3f}</td><td>{b["ar1_lower_mse"]}/{b["defined"]}</td></tr>')
        mdrows.append(f'| {label} | {a["median"]:.3f} | {b["median"]:.3f} | {b["ar1_lower_mse"]}/{b["defined"]} |')
    diversity=[]; mddiv=[]
    for key,d in summary['feature_diagnostics'].items():
        bank,kind=key.split('__'); corr=d['closest_original_abs_spearman']
        diversity.append(f'<tr><td>{bank}</td><td>{kind}</td><td>{corr["median"]:.3f}</td><td>{d["near_original_at_090"]}/{corr["defined"]}</td></tr>')
        mddiv.append(f'| {bank} | {kind} | {corr["median"]:.3f} | {d["near_original_at_090"]}/{corr["defined"]} |')
    support=summary['support']
    text=f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fusion stays central — prediction-view feasibility, Step 311</title>
<style>
:root{{--bg:#f3f6f5;--ink:#17343e;--teal:#087c72;--muted:#536b73;--line:#d3dfdf}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:17px/1.6 system-ui,sans-serif}}
main{{max-width:1040px;margin:auto;padding:36px 24px 65px}}h1{{font-size:clamp(30px,5vw,48px);line-height:1.12}}h2{{font-size:25px}}
a{{color:#126494}}.card,section{{background:white;border:1px solid var(--line);border-radius:14px;padding:24px;margin:22px 0}}
.note{{border-left:5px solid var(--teal);padding:16px 22px;background:#e2f2ed}}.caution{{background:#fff1df;border-color:#bc7c26}}
.flow{{display:grid;grid-template-columns:1fr 1fr 1.4fr 1fr;gap:12px;margin:24px 0}}.box{{padding:20px 16px;border:1px solid var(--line);border-radius:12px;background:white}}
.core{{background:var(--teal);color:white;border:3px solid #074f49}}.box strong{{display:block;margin-bottom:12px}}
.small{{font-size:14px;color:var(--muted)}}table{{border-collapse:collapse;width:100%;font-size:15px}}th,td{{text-align:left;padding:10px;border-bottom:1px solid var(--line)}}th{{background:#edf4f1}}
.scroll{{overflow:auto}}code{{font-size:.9em;overflow-wrap:anywhere}}details{{margin:18px 0}}summary{{cursor:pointer;font-weight:650}}
@media(max-width:700px){{.flow{{grid-template-columns:1fr}}section{{padding:18px}}}}@media print{{body{{background:white}}main{{padding:0}}section{{break-inside:avoid}}}}
</style></head><body><main>
<p class="small">7 September 2026 · Step 311 · Existing 110 development answers · Review passed</p>
<h1>Improve our fusion method.<br>Add prediction information where it helps.</h1>
<p class="note"><strong>IU-PCR and Joint L-SML stay at the centre.</strong> This study builds extra measurements for their matrix.
It does not replace fusion with an AR predictor, KalmanNet, a flow model or another detector.</p>
<div class="flow" aria-label="One answer feeds measurements, then fusion, then a step decision">
<div class="box"><strong>1. One answer</strong>Reuse its cached token trace. No new model call.</div>
<div class="box"><strong>2. Measurements</strong>Keep the original 27 window features. Add nine prediction-error columns.</div>
<div class="box core"><strong>3. Our fusion core</strong>IU-PCR or Joint L-SML combines the measurements into a risk trajectory.</div>
<div class="box"><strong>4. Localization</strong>Map the fused trajectory to official steps and a no-error decision.</div></div>
<p class="small">Steps 3–4 describe the next quality test. This feasibility stage only computes and reviews Step 2.</p>

<section><h2>What is the new measurement?</h2>
<p>For each telemetry stream, predict the next token value from earlier values in the same answer. Then measure how far the observed value was from that prediction. Average the absolute error within each eight-token window.</p>
<p>The small AR(1) predictor learns a slope and a mean change. With little history, it stays close to the simple prediction “repeat the last value”. We keep that simple predictor and EMA32 as controls.</p>
<p><strong>The original matrix is still there.</strong> Both moment27 and context27 keep their original columns exactly. The proposed matrix has N windows × 36 columns. An extra feature is a candidate input to fusion, not a correctness label.</p>
<details><summary>The fitting rule, precisely</summary>
<p>At token t, use pairs (x[k−1], x[k]) for k=1,…,t−1. The current target is excluded. Let b be the OLS slope clipped to [−1,1], mₓ/mᵧ the two prefix means, n the number of past pairs, and η=n/(n+16).</p>
<p><code>prediction[t] = x[t−1] + η × ((mᵧ−mₓ) + (b−1) × (x[t−1]−mₓ))</code></p>
<p>The first token has no prediction and is masked. The first window averages seven residuals; later windows average eight. The fixed 16-pair shrinkage is an engineering choice, not an optimal value.</p>
<p>The predictor uses only past tokens. Full-answer normalization and fusion fitting would still make the complete localizer offline.</p></details></section>

<section><h2>The history matters</h2>
<p>Prediction errors have already appeared in several parts of this project. The new question is their role in the current answer-only fusion architecture.</p>
<div class="scroll"><table><thead><tr><th>Earlier work</th><th>What it actually tested</th><th>Why the new question remains open</th></tr></thead><tbody>
<tr><td>Scalar AR / ordinary Kalman</td><td>One scalar score for each complete answer.</td><td>It did not test an added chronological view in today's answer-only matrix. Ordinary Kalman is not learned KalmanNet.</td></tr>
<tr><td>Local/Online IU dynamics</td><td>EMA innovations inside token matrices; calibration answers stacked to fit IU.</td><td>The fitting scope was multiple answers. The final selected local representation used level features.</td></tr>
<tr><td>Token temporal innovation B3</td><td>Lag predictors, optional cross-stream support, and a B3 addition learned from donor questions.</td><td>Code/protocol/tests exist. No matching named local result freeze was located in the scoped filename search; execution elsewhere is unverified.</td></tr>
<tr><td>CIW cross-scale localization</td><td>Predict token coordinates using whole-answer information; donor fitting and an IU input blend.</td><td>No improvement on both tasks under that response/token contract. It is not a paired comparison with these 110 answers.</td></tr>
</tbody></table></div>
<p class="small">The <a href="../../docs/experiments/FUSION_PREDICTION_VIEW_AUDIT_V1.md">audit protocol</a> identifies the exact code and older results. This is a correction to an incomplete history search, not a claim that prediction-error features are new.</p></section>

<section><h2>Can one answer support prediction?</h2>
<p>Yes, for this small predictor on these traces. The following numbers compare token prediction error, <strong>not hallucination detection</strong>. Each ratio is the median over 110 answers. Below 1 means less prediction error. Only tokens with at least 16 earlier fitting pairs enter this diagnostic.</p>
<div class="scroll"><table><thead><tr><th>Stream</th><th>AR MSE / last-value MSE</th><th>AR MSE / EMA32 MSE</th><th>AR beats EMA32</th></tr></thead><tbody>{''.join(table)}</tbody></table></div>
<p>AR predicts entropy slightly better than EMA32 at the median, but loses on 32 of 110 answers. Its larger prediction gain for spilled energy does not authorize selecting that stream based on correctness. Predicting the telemetry well and finding a wrong reasoning step are different objectives.</p></section>

<section><h2>Do the extra columns differ from existing features?</h2>
<p>They are usable in all 110 answers, but many remain similar to existing columns. For each added column, we find its strongest absolute Spearman correlation with any of the original 27. The median below pools 9 streams × 110 answers; these 990 entries are not independent examples.</p>
<div class="scroll"><table><thead><tr><th>Original bank</th><th>Residual predictor</th><th>Median closest-column correlation</th><th>Correlation ≥ 0.90</th></tr></thead><tbody>{''.join(diversity)}</tbody></table></div>
<p>AR residuals are less similar than last-value differences to both banks. EMA32 is less similar to the moment bank than AR, so AR is not uniformly the least redundant option. Low correlation alone does not show useful new error information.</p>
<p class="note caution">The fitting matrix still needs care: <strong>{support['n_less_than_36']}/110 answers have fewer than 36 independent grid windows</strong>. The median is {support['fit_windows']['median']:.0f}, with range {support['fit_windows']['minimum']:.0f}–{support['fit_windows']['maximum']:.0f}.
In 31 answers the original centered matrix already fills its N−1 row-rank limit. Adding columns cannot raise that rank. This does not by itself prove Joint is invalid; its actual fitting guards must be checked.</p></section>

<section><h2>How we will attribute progress to fusion</h2>
<div class="scroll"><table><thead><tr><th>Required comparison</th><th>What it tells the advisors</th></tr></thead><tbody>
<tr><td>Current IU / Joint versus the same core with residual columns</td><td>Whether the addition improves our existing method.</td></tr>
<tr><td>Augmented IU / Joint versus equal aggregation of the same columns</td><td>Whether learned fusion contributes beyond the new measurements.</td></tr>
<tr><td>AR residuals versus last-value and EMA residual controls</td><td>Whether the fitted predictor contributes beyond simple dynamics.</td></tr>
<tr><td>Joint graph versus lambda zero and a permuted graph</td><td>Whether graph structure helps beyond the shared features and fitting recipe.</td></tr>
</tbody></table></div>
<p><strong>Next bounded stage:</strong> freeze this comparison roster and its failure/fallback rules, then evaluate both PRMBench and ProcessBench. Keep the existing historical anchors, matched coverage, within-answer ranking and paired uncertainty. Do not select different winners for the two benchmarks.</p>
<p>IMM, HMM, BOCPD, LOCA, Diverging Flows, KalmanNet and token sampling remain supporting research tracks. This AR feasibility test does not implement or settle those named methods.</p>
<p class="note"><strong>Advisor-ready claim today:</strong> we have built and mechanically verified a way to add prediction information from the same answer to our fusion matrix. We have not yet shown that this addition improves localization. The previous <a href="../fusion_graph_conditioning_v1/REPORT.html">graph/conditioning results</a> remain the latest quality evidence, with no consistent winner established.</p></section>

<section><h2>Evidence and review</h2>
<p>All 110 answers / {support['tokens']:,} tokens were processed. Three scientific tests passed. Review independently reconstructed all 2,970 predictor/stream traces with batch prefix fits, 330 residual-window/MSE arrays, 660 exact original-column replays and 660 correlation/rank bundles.</p>
<p>Largest prediction difference: {review['maximum_prediction_difference']:.2e}. Audit runtime: {frozen['seconds_this_invocation']:.2f} s; review: {review['seconds']:.2f} s; test process: {tests['seconds']:.2f} s. No new localization heads or model inference were run. The review uses independent algebra in this session, with shared SciPy rank handling. HTML structure and local links are checked; no browser visual inspection.</p>
<p><a href="SUMMARY.json">All summaries</a> · <a href="REVIEW.json">Review</a> · <a href="MANIFEST.json">Input/source manifest</a> · <a href="AUDIT_FROZEN.json">Frozen artifacts</a> · <a href="TESTS.txt">Tests</a> · <a href="../../spectral_utils/fusion_prediction_view.py">New supporting-view code</a> · <a href="../../docs/reviews/joint_lsml_visual_guide_2026-09-06.html">Joint L-SML visual guide</a></p></section>
</main></body></html>'''
    (OUT/'REPORT.html').write_text(text, encoding='utf-8')
    md = f'''# Fusion prediction view — feasibility only, Step 311

IU-PCR / Joint L-SML remains the core. The new AR(1) component appends nine
window error measurements to the original 27 columns; it is not a detector
replacement, KalmanNet or Diverging Flows. No new localization quality scores.

All 110 answers support finite, varying added columns. This predictor fits
past pairs within one answer; old token B3/local-online/CIW variants used
donor/calibration answers. See the source audit protocol for exact boundaries.

| Stream | AR / last MSE median | AR / EMA32 MSE median | AR beats EMA32 |
|---|---:|---:|---:|
{chr(10).join(mdrows)}

These are telemetry-prediction results, not localization quality. Tokens t>=17
are used, with at least 16 previous fitting pairs. First-token residual is
excluded from every window mean. AR is not uniformly better than EMA32.

| Original bank | Residual predictor | Median closest-column abs Spearman | >=0.90 |
|---|---|---:|---:|
{chr(10).join(mddiv)}

990 entries = 9 streams x 110 answers, not independent observations. Added
columns remain substantially redundant; lower correlation may also be noise.
40/110 have N<36, while 31 original matrices already reach centered rank N-1.
No blanket fit-validity or quality conclusion follows from those facts.

The next quality study must include unchanged IU/Joint, the same cores with
the addition, equal aggregation with the same addition, simple residual
controls and graph-zero/permutation controls. Register coverage/fallback
before labels; retain both benchmarks and matched historical anchors.

Review PASS: {dict(review['counts'])}. Maximum prediction discrepancy
{review['maximum_prediction_difference']:.3e}. Three tests pass. Audit
{frozen['seconds_this_invocation']:.2f} s, review {review['seconds']:.2f} s.
Same-session independent algebra, shared ranking primitive; no browser visual
inspection. This is an already-exposed development cohort. No winner promoted.

See [visual report](REPORT.html), [protocol](../../docs/experiments/FUSION_PREDICTION_VIEW_AUDIT_V1.md),
[summary](SUMMARY.json), [review](REVIEW.json), and the existing
[quality anchors](../fusion_graph_conditioning_v1/REPORT.html).
'''
    (OUT/'REPORT.md').write_text(md, encoding='utf-8')
    parser=Links(); parser.feed(text); assert parser.titles==1 and len(set(parser.ids))==len(parser.ids)
    count=0
    for href in parser.links:
        url=urlsplit(href)
        assert not url.scheme and not url.netloc
        path=(OUT/unquote(url.path)).resolve(); assert path.is_relative_to(ROOT) and path.is_file(),href
        count+=1
    io.save(OUT/'ARTIFACT_VALIDATION.json', {'status':'PASS','local_links':count,
        'no_browser_visual_inspection':True,'report_sha256':io.sha(OUT/'REPORT.html')})
    paths=[Path(__file__), OUT/'MANIFEST.json', OUT/'AUDIT_FROZEN.json', OUT/'SUMMARY.json',
           OUT/'REVIEW.json', OUT/'REPORT.html', OUT/'REPORT.md', OUT/'ARTIFACT_VALIDATION.json']
    io.save(OUT/'REPORT_PROVENANCE.json', {'status':'PASS','hashes':{str(p):io.sha(p) for p in paths}})
    print('Reports generated; structural/local link checks PASS:',count,flush=True)


if __name__=='__main__': main()
