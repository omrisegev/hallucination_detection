"""Post-hoc metric diagnosis from frozen scores; no fitting or selection."""
from collections import Counter
import hashlib
import html
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/fusion_entropy_sampling_v1'
SELECTORS = ('full', 'risk_top', 'entropy_tails', 'entropy_quantiles')
LABELS = ('All windows', 'High entropy only', 'Half low + half high', 'Entropy quantiles')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pb_summary(rows, peak_arm, gate_arm):
    """Swap saved peak and binary gate decisions; never refit a threshold."""
    cells = {}
    for cell in sorted({r['cell'] for r in rows}):
        subset = [r for r in rows if r['cell'] == cell]
        counts = Counter(clean=0, erroneous=0, clean_correct=0, exact_correct=0,
                         peak_exact=0, peak_early=0, peak_late=0,
                         error_gate_closed=0, gated_early=0, gated_late=0)
        for r in subset:
            assert r['valid'][peak_arm] and r['decision_valid'][gate_arm]
            target = r['target']
            peak = r['peaks'][peak_arm]
            opened = r['predictions'][gate_arm] != -1
            prediction = peak if opened else -1
            if target == -1:
                counts['clean'] += 1
                counts['clean_correct'] += prediction == -1
            else:
                counts['erroneous'] += 1
                counts['exact_correct'] += prediction == target
                counts['peak_exact'] += peak == target
                counts['peak_early'] += peak < target
                counts['peak_late'] += peak > target
                counts['error_gate_closed'] += not opened
                counts['gated_early'] += opened and peak < target
                counts['gated_late'] += opened and peak > target
        clean_acc = counts['clean_correct'] / counts['clean']
        error_acc = counts['exact_correct'] / counts['erroneous']
        score = 2 * clean_acc * error_acc / (clean_acc + error_acc) if clean_acc + error_acc else 0.
        cells[cell] = dict(counts=counts, score=score)
    totals = Counter()
    for detail in cells.values():
        totals.update(detail['counts'])
    assert totals['clean'] == 33 and totals['erroneous'] == 53
    assert totals['peak_exact'] + totals['peak_early'] + totals['peak_late'] == 53
    assert totals['exact_correct'] + totals['error_gate_closed'] + totals['gated_early'] + totals['gated_late'] == 53
    return dict(counts=totals, cells=cells, macro_score=sum(d['score'] for d in cells.values()) / len(cells))


def main():
    evaluation = json.loads((OUT / 'EVALUATION.json').read_text())
    assert json.loads((OUT / 'REVIEW.json').read_text())['status'] == 'PASS'
    rows = [r for r in evaluation['rows'] if r['cell'].startswith('pb_')]
    result = dict(status='POSTHOC_DESCRIPTIVE_ONLY', evaluation_sha256=sha(OUT / 'EVALUATION.json'),
                  script_sha256=sha(Path(__file__)), new_fits=0, new_score_arrays=0, methods={})
    display = []
    swaps = []
    for core in ('iu', 'graph010'):
        base = f'sample_full__{core}'
        for selector, label in zip(SELECTORS, LABELS):
            arm = f'sample_{selector}__{core}'
            m = evaluation['metrics'][arm]
            native = pb_summary(rows, arm, arm)
            assert math.isclose(native['macro_score'], m['pb']['macro_f1'], abs_tol=1e-14)
            for cell, d in native['cells'].items():
                assert math.isclose(d['score'], m['pb']['cells'][cell]['f1'], abs_tol=1e-14)
            changed_peak = pb_summary(rows, arm, base)
            changed_gate = pb_summary(rows, base, arm)
            prefix = []
            for r in rows:
                t = r['target']
                if t > 0:
                    scores = r['scores'][arm]
                    prefix.append(sum((scores[t] > s) + .5 * (scores[t] == s) for s in scores[:t]) / t)
            assert len(prefix) == 44
            result['methods'][arm] = dict(native=native, new_peaks_full_gate=changed_peak,
                full_peaks_new_gate=changed_gate, prefix_pair_ranking=sum(prefix)/len(prefix),
                prefix_answers=44, excluded_step_zero_errors=9,
                stored_common_moment_iu_gate_score=m['pb_common_iu_gate']['macro_f1'])
            c = native['counts']
            display.append(f'<tr><td>{core}</td><td>{label}</td><td>{m["prm"]["auroc"]:.4f}</td>'
                f'<td>{m["prm"]["within_answer_auc"]:.4f}</td><td>{100*native["macro_score"]:.2f}%</td>'
                f'<td>{c["clean_correct"]}/33</td><td>{c["exact_correct"]}/53</td><td>{c["peak_exact"]}/53</td></tr>')
            if selector != 'full':
                swaps.append(f'<tr><td>{core}: {label}</td><td>{100*evaluation["metrics"][base]["pb"]["macro_f1"]:.2f}%</td>'
                    f'<td>{100*changed_peak["macro_score"]:.2f}%</td><td>{100*changed_gate["macro_score"]:.2f}%</td>'
                    f'<td>{100*native["macro_score"]:.2f}%</td></tr>')
    result['limitations'] = [
        'PRMB and PB use different answers, labels and endpoints; no cross-dataset correlation is estimated.',
        'Peak/gate swaps are post-hoc output diagnostics, not promoted methods or causal identification.',
        'PB prefix ranking compares the annotated first error only with its correct prefix. Later steps are excluded as unlabeled; clean answers and errors at step zero do not enter this diagnostic.',
        'Raw counts are pooled; PB headline scores average four cell-specific harmonic scores.',
        'The stored common gate is original moment IU, not necessarily the full-selector native gate.',
        'No new uncertainty intervals or independent confirmation from these already exposed 110 answers.'
    ]
    (OUT / 'METRIC_GAP_DIAGNOSTIC.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    body = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Why AUC and ProcessBench move differently</title><style>body{font:17px/1.6 system-ui;max-width:1150px;margin:32px auto;padding:20px;color:#203448;background:#f5f7fa}table{border-collapse:collapse;background:white;width:100%;font-size:14px}td,th{padding:10px;border-bottom:1px solid #ccd5df;text-align:left}.scroll{overflow:auto}.flow{padding:18px;background:#e4eff8}.note{border-left:5px solid #ac762f;padding:16px;background:white}a{color:#175a96}</style>
<h1>Why AUC can rise while ProcessBench falls</h1>
<p>This is a post-hoc diagnosis of the frozen 110-answer sampling experiment. No fusion fit, score, threshold or benchmark endpoint changed.</p>
<p>Our PRMB AUC asks whether an erroneous step ranks above a correct step. Pooled AUC includes pairs from different answers. Within-answer AUC only uses pairs within the same answer, averaged across 16 mixed-label answers. PB evaluates different answers and demands the exact FIRST error, or a no-error decision.</p>
<div class="flow">Same-answer windows → selected-row normalization and fusion → dense risk curve → step maximum → <strong>highest-risk step</strong><br>Dense risk distribution → mixture gate → <strong>error / no error</strong><br>Final output: highest-risk step if the gate opens; otherwise no error.</div>
<p>The current locator picks the strongest peak. The first reasoning error need not produce the strongest peak. AUC also does not judge whether the mixture gate correctly accepts an entirely clean answer.</p>
<h2>Observed results and decision counts</h2><p>Joint graph is condition100, lambda0.1, including its declared IU fallback. PB scores average four subsets; the counts below are pooled and should not be combined directly to reconstruct the macro score.</p>
<div class="scroll"><table><thead><tr><th>Core</th><th>Selection</th><th>PRMB pooled AUC</th><th>Within-answer AUC</th><th>PB score</th><th>Clean correct</th><th>First error correct</th><th>Peak correct, gate ignored</th></tr></thead><tbody>ROWS</tbody></table></div>
<h2>Separate the saved peak and gate decisions</h2><p>Each row uses its own full-selector core as the baseline. These diagnostic swaps do not train a new localizer. They differ from the original report's common moment-IU gate control.</p>
<div class="scroll"><table><thead><tr><th>Core / new selector</th><th>Full peak + full gate</th><th>New peak + full gate</th><th>Full peak + new gate</th><th>New peak + new gate</th></tr></thead><tbody>SWAPS</tbody></table></div>
<div class="note">For quantile IU, keeping the old gate still gives only 22.24%, versus 30.16% with the original peaks. The native quantile result is 20.77%. The decline is therefore not explained by the no-error gate alone. These swaps do not establish why the new fusion curve moved its peaks.</div>
<p>A second PB-only diagnostic compares each annotated first-error score with scores of earlier, correct steps: full IU 0.7094, high-only 0.7113, tails 0.7033, quantiles 0.6911. It uses 44 erroneous answers with a nonempty prefix. Nine step-zero errors, all clean answers and every post-error step are excluded. This is not dense step AUROC or a new primary benchmark.</p>
<p>Interpretation: quantile selection gives a local-ranking gain on PRMB; this gain does not carry over to these PB answers. Retain full/high-only references. Complete matched full-population evaluation and historical refits first; inspect lost first-error peaks before proposing a bounded fusion/readout change. No new sweep or publication winner is selected here.</p>
<p><a href="REPORT.html">Frozen experiment report</a> · <a href="METRIC_GAP_DIAGNOSTIC.json">Counts and diagnostic details</a> · <a href="https://arxiv.org/html/2412.06559v1#S4.SS1">Official PB metric definition</a></p>
<p>Small development cohort; no new confidence intervals for this diagnosis. No browser rendering or external review claimed.</p></html>'''
    report = body.replace('ROWS', ''.join(display)).replace('SWAPS', ''.join(swaps))
    (OUT / 'METRIC_GAP_DIAGNOSTIC.html').write_text(report, encoding='utf-8')
    print(json.dumps({a: {k: d[k] for k in ('prefix_pair_ranking',)} | d['native']['counts']
                      for a, d in result['methods'].items()}, indent=2))


if __name__ == '__main__':
    main()
