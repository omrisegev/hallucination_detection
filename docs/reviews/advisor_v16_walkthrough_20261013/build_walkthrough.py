"""Apply the agreed meeting edits to the pinned v16 page; no model fitting."""
from pathlib import Path
import hashlib
import json
import re
import html
import base64

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'source_v16_2cf1209c.html'
EXPECTED_BLOB = '9aaab92066600edb593b7050cd7f042acf7f1719'
raw = SOURCE.read_bytes()
if raw.endswith(b'\n'):
    without_newline = raw[:-1]
    blob = hashlib.sha1(b'blob ' + str(len(without_newline)).encode() + b'\0' + without_newline).hexdigest()
    if blob == EXPECTED_BLOB:
        raw = without_newline
        SOURCE.write_bytes(raw)
blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
assert blob == EXPECTED_BLOB, f'Source does not match the pinned GitHub blob: {blob}'
s = raw.decode()
original_s0 = re.search(r'<section id="s0">.*?</section>', s, flags=re.S).group()
old_figure = re.search(r'<figure>.*?</figure>', original_s0, flags=re.S).group()
source_links = re.search(r'<p class="source">.*?</p>', original_s0, flags=re.S).group()
svg = (HERE / 'digit_banks_preview.svg').read_text()
svg = svg[svg.index('<svg'):]
svg = re.sub(r'<svg\b', '<svg role="img" aria-label="Within-answer AUC and gain over the plain average on the strictly nested digit banks N14, N16, N25, N43 and N56"', svg, count=1)
new_s0 = '''<section id="s0"><div class="eyebrow">0 · WHERE WE STOPPED ON 8 OCTOBER</div>
<h2>Adding features helps LR, but does not consistently help label-free fusion</h2>
<div class="exp"><span class="kt">Experiment · what we tested and why</span>
We compare methods on the same corrected Qwen3-8B input and 6,030 answers. We show only banks that keep all three digit channels: N14 ⊂ N16 ⊂ N25 ⊂ N43 ⊂ N56. Each expansion only adds channels (+2, +9, +18, +13); no channel is removed. Left: within-answer AUC. Right: each method's gain over the plain average in the same bank.</div>
<figure>''' + svg + '''<figcaption>Strictly nested digit banks, corrected prefix + mask1. LR uses the same branch-reported cross-fitted reference as v16. Faint curves show the other label-free methods, excluding duplicate LEVELSIGN variants. These are point estimates; no confidence intervals are shown. Bank expansion does not guarantee monotone performance.</figcaption></figure>
<ul class="takeaways concl"><span class="kt">Conclusions</span>
<li><b>Supervised LR improves at every expansion; label-free fusion does not improve consistently.</b> From N16 to N43, L-SML falls from 0.779 to 0.762 and IU-PCR from 0.780 to 0.754, while LR rises from 0.801 to 0.816.</li>
<li><b>L-SML keeps a larger margin over the plain average than IU-PCR in the larger banks.</b> At N56 the gains are +1.68 and +1.04 AUC percentage points. The wider margin mainly reflects a larger decline of the plain average; L-SML itself remains below its N16 result.</li>
<li><b>The added channels give supervised fusion an improvement that the label-free methods do not recover.</b> From N16 to N56, LR rises from 0.801 to 0.821 while L-SML and IU-PCR decline. Which added features, contrasts or weighting changes account for this divergence is an open question.</li></ul>
<div class="box open"><span class="kt">Question for follow-up</span>
<p><b>Focus on N25 → N43.</b> Eighteen channels are added. LR improves from 0.8071 to 0.8159 (+0.88 AUC points), while L-SML falls from 0.7740 to 0.7616 (−1.24 points), and IU-PCR from 0.7730 to 0.7544 (−1.86 points). Investigate which added features help LR, which changes degrade label-free fusion, and why. The bank contrast alone does not identify an individual feature's contribution.</p></div>
''' + source_links + '<details><summary>All ten banks: original v16 overview</summary>' + old_figure + '</details></section>'
s = s.replace(original_s0, new_s0, 1)
original_s1 = re.search(r'<section id="s1">.*?</section>', s, flags=re.S).group()
without_tf = re.search(r'<div class="box"><span class="kt">Without teacher forcing</span>.*?</div>', original_s1, flags=re.S).group()
new_s1 = original_s1.replace(without_tf, '')
new_s1 = re.sub(r'<div class="box warn"><span class="kt">Claims we withdraw</span>.*?</div>', '', new_s1, flags=re.S)
root = HERE.parents[2]
ex = json.loads((root / 'results/advisor_completion_2026-10-11/controls/DATASET_EXAMPLES.json').read_text())[0]
question = ex['question']
user_message = ('Solve the following competition math problem. '
                'Show all your work step by step, then give your final answer in \\boxed{}.\n\n'
                + question + ' /no_think')
old_prompt = '<|im_start|>user\n' + user_message + '<|im_end|>\n<|im_start|>assistant\n'
new_prompt = old_prompt + '<think>\n\n</think>\n\n'
diagnostic = base64.b64encode((HERE / 'prefix_token_diagnostic.png').read_bytes()).decode()
prefix_detail = '''<div class="box"><span class="kt">What the bug was · a concrete fixed-answer example</span>
<p>Teacher forcing scores an existing answer token by token. It does not generate a new solution. In our old /no_think path the prompt stopped at the assistant header; the fixed answer started immediately. Qwen3 instead predicted the structural &lt;think&gt; token at that boundary. The corrected prompt supplies the empty think block before the answer content.</p>
<p><b>Actual PRMBench question:</b> ''' + html.escape(question) + '''</p>
<p><b>First two steps of the fixed answer:</b></p><pre style="white-space:pre-wrap;background:#f5f8fb;padding:14px">''' + html.escape('\n\n'.join(ex['steps'][:2])) + '''</pre>
<p><b>Old end of prompt:</b></p><pre style="white-space:pre-wrap;background:#f5f8fb;padding:14px">&lt;|im_start|&gt;assistant
[fixed answer starts here]</pre><p><b>Corrected end of prompt:</b></p><pre style="white-space:pre-wrap;background:#f5f8fb;padding:14px">&lt;|im_start|&gt;assistant
&lt;think&gt;

&lt;/think&gt;

[the same fixed answer starts here]</pre>
<details><summary>Complete reconstructed prompts and the full fixed answer</summary><p>Reconstruction from the saved official row and the frozen formatter; this is not a raw captured prompt from a selected inference record. Bracketed annotations above are explanatory labels and are not supplied to the model.</p><h3>Old conditioning prompt</h3><pre style="white-space:pre-wrap">''' + html.escape(old_prompt) + '''</pre><h3>Corrected conditioning prompt</h3><pre style="white-space:pre-wrap">''' + html.escape(new_prompt) + '''</pre><h3>Same fixed answer in both runs</h3><pre style="white-space:pre-wrap">''' + html.escape('\n\n'.join(ex['steps'])) + '''</pre></details></div>
<figure><img alt="Saved PRMBench first-token surprisal and first-position top-prediction diagnostics before and after the prefix fix" style="width:100%;height:auto" src="data:image/png;base64,''' + diagnostic + '''"><figcaption>Aggregate diagnostics on all 6,969 PRMBench rows; these are not measurements of the example above. Median token-0 surprisal: 34.6 → 24.1 nats. First-position top prediction &lt;think&gt;: 100% → 0%. Residual token-0 surprisal remains high. mask1 removes that original content token and shifts aligned telemetry; it is a separate preprocessing decision. Its cause is not established by these aggregate statistics.</figcaption></figure>
<p class="source">Evidence: dataset_cache/four_localization/prmbench_qwen3_8b_telemetry_thinkfix/THINKFIX_STATS.json; spectral_utils/processbench.py::teacher_forced_prompt; spectral_utils/prmbench.py::prmbench_prompt; spectral_utils/data_loaders.py::math_prompt; controls/DATASET_EXAMPLES.json.</p>'''
new_s1 = new_s1.replace('<figure>', prefix_detail + '<figure>', 1)
new_s1 = new_s1.replace('<h2>All methods went down by a similar amount; the picture stays</h2>', '<h2>The missing think block changed the conditioning at the answer boundary</h2>', 1)
new_s1 = new_s1.replace('The decline is not a first-token effect. Without step 0 the decline is larger (−2.2 vs −1.6 points for L-SML). The first-token mask changes AUC by less than 0.1 point.', 'The aggregate AUC decline is not explained by the first step alone. Without step 0 the decline is larger (−2.2 vs −1.6 points for L-SML). The separate mask1 change at N16 is −0.075 AUC points for L-SML and −0.013 for IU-PCR.', 1)
new_s1 = new_s1.replace('The same conclusions hold on Qwen3-4B (N16 L-SML 0.775 vs 0.779; section 6c) and on a held-out half of the source groups (IU-PCR 0.786; appendix A).', 'The corrected input preserves the main comparison: LR stays above label-free fusion, and the plain average stays below it. Feature-level conclusions must use the corrected scores. Model-size and self-generation checks answer separate transfer questions and are shown with robustness evidence.', 1)
s = s.replace(original_s1, new_s1, 1)
s = s.replace('<section id="appendix-v13">', '<section id="appendix-v13"><details><summary>F · Does the signal transfer to answers the model generated itself?</summary><p>This tests transfer from externally supplied answers to model-generated answers. It is distinct from reconstructing the correct teacher-forcing prefix. Different data and error populations prevent this contrast from isolating a causal teacher-forcing effect.</p>' + without_tf + '</details>', 1)
original_s10 = re.search(r'<section id="s10">.*?</section>', s, flags=re.S).group()
extra = '''<li><b>Why added features help LR but degrade label-free fusion.</b> Start with the eighteen channels added between N25 and N43. Identify useful supervised features or contrasts, and inspect how the added channels change signs, weights, redundancy and L-SML grouping. This is a proposed follow-up question; no ablation has been run in this walkthrough.</li>'''
s = s.replace(original_s10, original_s10.replace('</ol>', extra + '</ol>'), 1)
s = s.replace('<title>PRMBench Advisor Review</title>', '<title>PRMBench Advisor Review · meeting walkthrough draft</title>', 1)
banner = '''<div class="box"><span class="kt">Local meeting walkthrough draft · 13 October 2026</span><p>Based on remote v16 at commit 2cf1209c. Agreed edits: the nested digit-bank graph in section 0; the concrete prompt and first-token explanation in section 1; removal of the withdrawn-claims box; and the N25 → N43 question in section 10. The proposed 40-minute order is a separate proposal, awaiting discussion.</p></div>'''
s = s.replace('<main>', '<main>' + banner, 1)
dest = HERE / 'advisor_walkthrough_draft.html'
dest.write_text(s)
banks = json.loads((HERE.parents[2] / 'results/advisor_completion_2026-10-11/sources/research/results/monday_nested_banks_v2/BANKS.json').read_text())
chain = ['N14', 'N16', 'N25', 'N43', 'N56']
transitions = [{'from':a, 'to':b, 'added': sorted(set(banks[b])-set(banks[a])), 'removed': sorted(set(banks[a])-set(banks[b]))} for a,b in zip(chain,chain[1:])]
assert all(not t['removed'] for t in transitions)
(HERE / 'BANK_TRANSITIONS.json').write_text(json.dumps(transitions, indent=2) + '\n')
print(f'Saved {dest}; pinned source blob matches; five digit banks form a strict inclusion chain.')
