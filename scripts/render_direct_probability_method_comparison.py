"""Describe existing DPF v2 results; no model fitting or new experiment."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join(
        '<th>' + html.escape(str(x)) + '</th>' for x in headers
    ) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join('<td>' + html.escape(str(x)) + '</td>' for x in row) + '</tr>'
        for row in rows
    ) + '</tbody></table></div>'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = root / 'results/direct_probability_fusion_v2_selected_tail'
    paths = {
        'localization': out / 'LOCALIZATION.json',
        'historical': out / 'HISTORICAL_24.json',
        'token_references': args.source_root / 'results/token_level_readout_v1/METRICS.json',
        'historical_references': args.source_root / 'results/hard_filter_dufs_liu_24cell/per_cell_metrics.csv',
        'comparison': out / 'V2_V1_COMPARISON.json',
    }
    load = lambda key: json.loads(paths[key].read_text(encoding='utf-8-sig'))
    loc, hist, old = load('localization'), load('historical'), load('token_references')['results']
    cmp = load('comparison')
    refs = list(csv.DictReader(paths['historical_references'].open(encoding='utf-8-sig')))
    liu = {r['cell']: r for r in refs if r['contract'] == 'mixed_v2' and r['filter'] == 'full' and r['solver'] == 'dufs_liu'}
    iu = {r['cell']: r for r in refs if r['contract'] == 'mixed_v2' and r['filter'] == 'full' and r['solver'] == 'iu_pcr'}
    assert set(liu) == set(iu) == set(hist['cells']) and len(liu) == 24
    for c, v in hist['cells'].items():
        assert int(liu[c]['n']) == int(iu[c]['n']) == v['n_answers']
        assert abs(float(iu[c]['auroc']) - v['historical_iu_pcr']) < 1e-12
    for k in ('pb_all8', 'prm_within', 'prm_pooled', 'prmscore_q08'):
        assert old['token_entropy'][k] == loc['methods']['entropy'][k]
    names = [('DPF-Equal v2', 'augmented_equal'), ('DPF-IU v2', 'augmented_iu'), ('DPF-Joint v2', 'augmented_joint_lw')]
    entries = [(name, loc['methods'][key]) for name, key in names] + [
        ('Previous token fusion: IU-PCR, 9 streams', old['token_iu9']),
        ('Token Entropy (fixed anchor)', old['token_entropy']),
        ('Token Varentropy (earlier exploratory result)', old['token_varentropy']),
        ('Token Margin (earlier PB point maximum)', old['token_margin']),
        ('Previous window fusion: IU-PCR, 27 features', old['window_iu27_top10']),
        ('Mind the Gap locator + common gate', loc['comparators']['mindgap_paper_locator_common_gate']),
    ]
    f = lambda x: '\u2014' if x is None else f'{x:.6f}'
    pct = lambda x: '\u2014' if x is None else f'{100*x:.4f}%'
    macro_rows = [[n, pct(v['pb_all8']), pct(v['pb_q4']), pct(v['pb_q8']), f(v.get('prm_within')), f(v.get('prm_pooled')), f(v.get('prmscore_q08'))] for n, v in entries]
    per_rows = []
    cell_labels = lambda c: c.removeprefix('pb_').replace('_q4', ' / Qwen3-4B').replace('_q8', ' / Qwen3-8B')
    for c in sorted(loc['methods']['entropy']['pb_cells']):
        per_rows.append([cell_labels(c)] + [pct(v.get('pb_cells', v.get('cells', {}))[c]) for _, v in entries])
    hrows = []
    hist_detail = []
    for c, v in sorted(hist['cells'].items(), key=lambda x: (x[1]['domain'], x[0])):
        d = v['auroc']['augmented_equal'] - v['historical_iu_pcr']
        hrows.append([c, v['domain'], v['n_answers'], f(v['auroc']['augmented_equal']), f(v['auroc']['augmented_iu']), f(v['auroc']['augmented_joint_lw']), f(v['historical_iu_pcr']), f(float(liu[c]['auroc'])), f'{d:+.6f}'])
        hist_detail.append({'cell': c, 'domain': v['domain'], 'n': v['n_answers'], **v['auroc'], 'historical_iu': v['historical_iu_pcr'], 'historical_dufs_liu': float(liu[c]['auroc'])})
    hmacro = []
    for n, k in names + [('Historical IU-PCR', 'historical_iu_pcr'), ('Token Entropy', 'entropy')]:
        m = hist['macro'][k]
        hmacro.append([n, f(m['all24']), f(m['qa9']), f(m['math15'])])
    avg = lambda domain: sum(float(r['auroc']) for c, r in liu.items() if domain is None or hist['cells'][c]['domain'] == domain) / (24 if domain is None else 9 if domain == 'QA' else 15)
    hmacro.append(['Historical DUFS-LIU', f(avg(None)), f(avg('QA')), f(avg('math'))])
    summary = {
        'scope': 'Comparison of saved results only; no new fits or inference',
        'names': dict(names),
        'localization': dict(entries),
        'historical_cells': hist_detail,
        'historical_macro_rows': hmacro,
        'source_files': {k: {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for k, p in paths.items()},
        'interpretation': {
            'family': 'Direct Probability Fusion (DPF), descriptive working name, not a novelty claim',
            'single_answer_limit': 'Localization weights and normalization are answer-local; ProcessBench gate and PRMScore calibration use other answers. Historical normalization and learned weights use multiple answers per cell.',
            'tail_identity': 'At token level, unclipped tail = rank_1_risk - sum(rank_2_risk ... rank_15_risk). It adds no raw information. Top-10 aggregation per answer is nonlinear, so an added tail summary can change the answer-level representation.',
            'baseline_limit': 'Point gaps to token fusion/varentropy/margin and DUFS-LIU are descriptive; no new paired confidence intervals were computed for them.',
        },
    }
    (out / 'METHOD_COMPARISON.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    bars = ''.join(f'<div class="bar"><span>{html.escape(n)}</span><div><i style="width:{100*v["pb_all8"]:.3f}%"></i></div><b>{pct(v["pb_all8"])}</b></div>' for n, v in entries)
    body = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Direct Probability Fusion: exact comparison</title>
<style>body{font:16px/1.55 system-ui,sans-serif;background:#f4f7fb;color:#18283f;margin:0}main{max-width:1450px;margin:auto;padding:30px}h1{font-size:32px}h2{margin-top:36px}.box,.scroll{background:white;padding:18px;border:1px solid #dce4ef;border-radius:10px;margin:18px 0}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:9px;border-bottom:1px solid #e2e8f0;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{background:#eef3fa}.bar{display:grid;grid-template-columns:360px 1fr 100px;gap:12px;align-items:center;margin:12px 0}.bar div{background:#e1e7f0;height:16px}.bar i{display:block;height:100%;background:#2463ad}code{background:#e8eef5;padding:3px}small{color:#526176}</style><main>
<h1>Direct Probability Fusion: exact comparison</h1><p>Saved full-development results, 10 September 2026. No new experiment or method selection in this report. DPF is a descriptive working name.</p>
<div class="box"><b>The demonstrated result:</b> replacing the handcrafted feature bank with simple probability coordinates gives competitive performance. The Equal Weights version is the strongest new overall candidate on the historical panel. It does not beat the earlier token baselines on ProcessBench. The data do not establish a breakthrough or a new confirmed leader.</div>
<h2>1. What is the method?</h2>'''
    body += table(['Working label', 'Label in the original report', 'Internal key', 'Fusion'], [[n, 'Selected + Tail Probability Fusion - ' + suffix, k, desc] for (n,k),suffix,desc in zip(names, ['Equal Weights','IU-PCR','Joint Shrinkage'], ['Mean of standardized coordinates; no IU-PCR','IU-PCR learned weights','IU-PCR with Joint-inspired covariance shrinkage'])])
    body += '''<p>Per token: <code>[1-p1, p2, ..., p15, -log p(actual token), residual tail]</code>. K=15, 17 columns, no token window. Localization fits over all tokens in one answer, then uses the top-10 token mean in each reasoning step. This avoids the former spectral/moment feature bank, but still transforms, normalizes and aggregates probabilities. Entropy also remains the orientation anchor and the ProcessBench gate.</p>'''
    body += table(['Component', 'Localization', 'Historical 24 cells'], [
        ['Observation matrix','T tokens from one answer x 17','N answer summaries in one cell x 17'],
        ['Normalization','Within this answer','Across answers in the cell, including evaluated answers'],
        ['IU / Joint weights','Learned within this answer, without labels','Learned across answers in the cell, without labels'],
        ['Equal weights','Fixed 1/17 after answer-local normalization','Fixed 1/17 after cell-level normalization'],
        ['Calibration','PB entropy q=0.3 and PRMScore q=0.8 use other folds','AUROC measures ranking; no classification gate'],
        ['Strictly one-answer end to end?','No: calibration is external','No: normalization and learned weights use the cell'],
    ])
    body += '<h2>2. Full localization comparison</h2><p>ProcessBench: harmonic mean of clean-answer accuracy and exact first-error accuracy in each cell, then an unweighted mean across cells. Q4 / Q8 mean Qwen3-4B / Qwen3-8B, not quantization. PRMBench here is one Qwen3-8B cell (6,969 answers); within-answer AUC averages only answers with both step classes. PB has 6,800 model-answer rows.</p>'
    body += table(['Method','PB all 8','PB Q4','PB Q8','PRMB within AUC','PRMB pooled AUC','PRMScore'], macro_rows)
    body += '<div class="box"><b>Correction to the previous headline:</b> Token Entropy is the fixed anchor, not the highest earlier ProcessBench score. Token Margin has the largest PB point estimate in this token panel; Token Varentropy is the strongest earlier balance of PB, within-answer AUC and PRMScore. The previous token fusion is IU-PCR on 9 streams. No single method leads every endpoint. All new arms and token references cover 13,769 rows; the old window row has 13,748 valid fits, with failures retained in PB evaluation.</div>'
    body += '<div class="box">' + bars + '<small>Bars start at zero; full track = 100% F1. Labels show exact stored scores rounded to four decimal places.</small></div>'
    body += table(['PB cell']+[n for n,_ in entries], per_rows)
    body += '<p><b>Mind the Gap:</b> this is our paper-form locator adaptation with the same frozen entropy gate. It is not the paper\'s native SLA metric and not a reproduced published F1. DPF-IU exceeds this adaptation by 7.9589 percentage points, with saved paired 97.5% CI [5.9858, 9.9673]. This does not establish a win over the original paper under its native protocol.</p>'
    body += '<h2>3. Complete-answer detection: all 24 cells</h2><p>All values below are AUROC, higher is better. The 24-cell macro weights each cell equally (9 QA, 15 reasoning). The saved complete-case populations, labels and answer order are matched to Historical IU-PCR. DUFS-LIU is taken from the same historical release and row counts; its weights were not refit in this report.</p>'
    body += table(['Method','All 24','QA 9','Reasoning 15'], hmacro)
    body += table(['Cell','Domain','Answers','DPF-Equal v2','DPF-IU v2','DPF-Joint v2','Historical IU-PCR','Historical DUFS-LIU','Equal minus historical IU'], hrows)
    wins = {domain: sum(v['auroc']['augmented_equal'] > v['historical_iu_pcr'] for v in hist['cells'].values() if v['domain'] == domain) for domain in ('math', 'QA')}
    body += f'<p>DPF-Equal beats Historical IU-PCR in {sum(wins.values())}/24 cells: {wins["math"]}/15 reasoning and {wins["QA"]}/9 QA. It is stronger on the reasoning macro but slightly weaker on QA. HotpotQA remains weak (~0.58); high performance on selected math or TriviaQA cells is not uniform domain-wide success.</p>'
    body += '<p><b>Historical coverage:</b> this is a matched comparison with the available IU-PCR and DUFS-LIU releases, not a complete refit of every historical method. The DUFS-LIU feature contract was selected retrospectively in an earlier development search. Older fixed L-SML GOOD5 / GOOD6 reports give macro AUROC 0.754729 / 0.763225, but differ in at least one cell population (OPT-30B / TriviaQA: 4,993 versus 5,000 answers) and retain rounded scores; they are historical context, not a strict matched ranking. A matched original Joint L-SML result for these 24 cells was not located. Earlier 0.791369 IU and 0.793561 DUFS figures are from eight ProcessBench answer-clean/error detection cells, a different task from this historical24 panel.</p>'
    ci = cmp['historical_exploratory']['augmented_equal_minus_historical_iu_pcr']
    body += f'<p>Equal minus Historical IU-PCR: {ci["delta"]:+.6f}, exploratory paired-cell 97.5% CI [{ci["paired_cell_ci97_5"][0]:+.6f}, {ci["paired_cell_ci97_5"][1]:+.6f}]. This interval includes zero. The positive paired IU v2-v1 result is improvement over the weaker direct-matrix IU, not proof of superiority over historical IU or DUFS-LIU. New paired intervals against token Varentropy, token Margin, token IU9 and DUFS-LIU have not been computed.</p>'
    body += '''<h2>4. What is genuinely established?</h2><div class="box"><ul>
<li>The localization fusion fit can be learned from one answer. The reported calibrated decision system still uses other answers for its gate.</li>
<li>The direct representation is compact and avoids the handcrafted feature bank. It does not eliminate all feature transformations or entropy use.</li>
<li>The historical result is competitive, especially in reasoning, but was not obtained with strict single-answer fitting.</li>
<li>Residual tail is determined by the retained raw probabilities: <code>tail = (1-p1) - (p2+...+p15)</code>, apart from clipping. At token level it adds a dependent coordinate, not new information. In the historical route, top-10 aggregation happens separately per column; the extra tail summary can therefore change what is retained at answer level.</li>
<li>The actual-token coordinate adds which probability belongs to the scored token, information lost by sorting alone. This run added both coordinates together and cannot identify the cause of the observed improvement.</li>
<li>A simplicity advantage is worth studying. Equivalent accuracy, end-to-end speed and novelty still need direct evidence; they are not established by close point estimates.</li>
</ul></div><p><a href="REPORT.html">Original frozen experiment report</a> · <a href="METHOD_COMPARISON.json">Exact values and source hashes</a></p>
<h2>Sources</h2>'''
    body += table(['Artifact','SHA-256'], [[str(p), hashlib.sha256(p.read_bytes()).hexdigest()] for p in paths.values()])
    body += '</main></html>'
    (out / 'METHOD_COMPARISON.html').write_text(body, encoding='utf-8')
    print(out / 'METHOD_COMPARISON.html')
    print('Verified: 24 IU/LIU cell names and counts; IU scores and shared entropy endpoint continuity.')


if __name__ == '__main__':
    main()
