"""Rebuild the meeting presentation from the portable review package.

No model inference or fusion refit. Error-type AUROCs are reconstructed from
the saved N16 risk scores with the independently checked pairwise definition.
Run with the main checkout's .venv/bin/python; no private/tmp inputs required.
"""
import csv
import html
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PRIMARY = ['ALL_EQ', 'DS_FILTER_EQ', 'LSML_CONT', 'IU_PCR']
LABEL = {'ALL_EQ': 'Plain average', 'DS_FILTER_EQ': 'DS-filter average',
         'LSML_CONT': 'Continuous L-SML', 'IU_PCR': 'IU-PCR'}
COLOR = {'ALL_EQ': '#8b969d', 'DS_FILTER_EQ': '#77703d',
         'LSML_CONT': '#246c9c', 'IU_PCR': '#278577', 'LR': '#bf6235', 'PRM': '#a75e70'}
TYPES = ['redundency', 'circular', 'domain_inconsistency', 'step_contradiction',
         'missing_condition', 'confidence', 'counterfactual', 'deception']
GROUPS = {'Simplicity': ['redundency', 'circular'],
          'Soundness': ['confidence', 'counterfactual', 'domain_inconsistency', 'step_contradiction'],
          'Sensitivity*': ['missing_condition', 'deception']}


def read_rows(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def build(out=None):
    out = Path(out or Path(__file__).resolve().parent)
    data = json.loads((out/'DATA.json').read_text())
    audit = json.loads((out/'evidence/A/audit.json').read_text())
    prm = json.loads((out/'sources/PRM7B_MANIFEST.json').read_text())
    banks = json.loads((out/'sources/BANKS.json').read_text())
    comparison = read_rows(out/'sources/PRMBENCH_COMPARISON.csv')
    metrics = data['metrics']
    corrected = {'mask1': metrics, 'unmasked': {b: {} for b in data['banks']}}
    lr = {'mask1': data['lr'], 'unmasked': {}}
    for row in comparison:
        if row['metric'] != 'prmb_within_auc':
            continue
        if row['table'] in ['panel_a', 'cont_baselines'] and row['scorer'] in data['roster']:
            corrected['unmasked'][row['bank']][row['scorer']] = float(row['thinkfix'])
        elif row['table'] == 'lr_reference' and row['scorer'] == 'LR':
            lr['unmasked'][row['bank']] = float(row['thinkfix'])
    for b in data['banks']:
        assert set(corrected['unmasked'][b]) == set(data['roster'])
        assert b in lr['unmasked']
    # Independent fixed-method error-type breakdown, not per-type winner selection.
    meta = {r['idx']: r for r in json.loads((out/'sources/prmscore_inputs/prmbench_meta.json').read_text())['rows']}
    type_auc = {}
    with np.load(out/'sources/prmscore_inputs/bank_N16.npz', allow_pickle=False) as bank:
        for scorer in PRIMARY:
            scores = bank['S__'+scorer]
            values = defaultdict(list)
            for idx, start, end in zip(bank['idx'].astype(str), bank['offsets'][:-1], bank['offsets'][1:]):
                row = meta[idx]
                err = np.isin(np.arange(1, end-start+1), row['error_steps'])
                bad, good = scores[start:end][err], scores[start:end][~err]
                if len(bad) and len(good):
                    pairs = bad[:, None] - good[None, :]
                    values[row['classification']].append(float(((pairs > 0).sum()+.5*(pairs == 0).sum())/pairs.size))
            type_auc[scorer] = {t: float(np.mean(v)) for t, v in values.items()}
            total_n = sum(map(len, values.values()))
            weighted = sum(sum(v) for v in values.values()) / total_n
            assert total_n == audit['metrics'][scorer]['auc_n_answers']
            assert abs(weighted - metrics['N16'][scorer]) < 1e-12
    pcat = prm['results']['by_classification']
    category = {s: {g: float(np.mean([audit['metrics'][s]['by_class']['q80_gt'][t]['pooled_macro_f1'] for t in ts]))
                    for g, ts in GROUPS.items()} for s in ['LSML_CONT', 'IU_PCR']}
    category['PRM'] = {g: float(np.mean([.5*(pcat['f1'][t]+pcat['negative_f1'][t]) for t in ts]))
                       for g, ts in GROUPS.items()}
    mask_stats = {}
    for s in PRIMARY:
        diffs = [corrected['mask1'][b][s]-corrected['unmasked'][b][s] for b in data['banks']]
        mask_stats[s] = dict(n16_delta=metrics['N16'][s]-corrected['unmasked']['N16'][s],
                             mean_across_banks=float(np.mean(diffs)), improved=int(sum(x > 0 for x in diffs)),
                             worsened=int(sum(x < 0 for x in diffs)))
    present = dict(primary=PRIMARY, names=LABEL, corrected=corrected, lr=lr, banks=data['banks'],
                   bank_features=banks, type_auc=type_auc, category_f1=category, mask_only=mask_stats,
                   main_input='mask1', main_input_reason='Retain the campaign primary protocol; unmasked is a separate visible ablation.',
                   source_head=data['head'], type_auc_contract='Mean of within-answer error/correct pairwise AUC; half credit for ties.',
                   prm_auc_source='Rounded values from the supplied advisor artifact; raw PRM scores not replayed.')
    (out/'PRESENTATION_DATA.json').write_text(json.dumps(present, indent=2))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#b8c3cb',
                         'text.color': '#21323d', 'axes.labelcolor': '#344451', 'figure.facecolor': 'white'})
    def save(fig, name):
        for ext in ['svg', 'png']:
            fig.savefig(out/'figures'/f'{name}.{ext}', bbox_inches='tight', dpi=180)
        plt.close(fig)
    def figure(name, caption):
        svg = (out/'figures'/f'{name}.svg').read_text()
        svg = svg[svg.index('<svg'):]
        for identifier in re.findall(r'\bid="([^"]+)"', svg):
            svg = svg.replace('id="'+identifier+'"', 'id="'+name+'-'+identifier+'"')
            svg = svg.replace('#'+identifier+'"', '#'+name+'-'+identifier+'"')
            svg = svg.replace('#'+identifier+')', '#'+name+'-'+identifier+')')
        return '<figure>'+svg+'<figcaption>'+caption+'</figcaption></figure>'
    # Direct primary-method and reference comparison, without an eight-scorer mean.
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={'width_ratios': [1.4, 1]})
    ss = PRIMARY + ['LR', 'PRM']; vv = [metrics['N16'][s] for s in PRIMARY] + [data['lr']['N16'], .801]
    ax = axes[0]; yy = np.arange(len(ss))
    ax.barh(yy, np.array(vv)-.70, left=.70, color=[COLOR[s] for s in ss], height=.62)
    ax.set_yticks(yy, [LABEL.get(s, {'LR': 'Supervised LR · OOF', 'PRM': 'Qwen2.5-Math-PRM-7B'}.get(s)) for s in ss])
    ax.invert_yaxis(); ax.set_xlim(.70, .827); ax.set_xlabel('Mean within-answer AUROC')
    for y, v in zip(yy, vv): ax.text(v+.0015, y, f'{v:.3f}', va='center', fontsize=10)
    ax.set_title('N16 · primary methods and references', loc='left', fontweight='bold')
    qss = ['LSML_CONT', 'IU_PCR', 'PRM']
    qvv = [audit['metrics'][s]['rules']['q80_gt']['pooled_macro_f1'] for s in qss[:-1]] + [.5*(prm['results']['total']['f1']+prm['results']['total']['negative_f1'])]
    ax = axes[1]; ax.bar(range(3), qvv, color=[COLOR[s] for s in qss], width=.60)
    ax.set_xticks(range(3), ['L-SML\nQ80', 'IU-PCR\nQ80', 'PRM-7B\nthreshold 0.5'])
    ax.set_ylim(0, .77); ax.set_ylabel('Pooled macro-F1 (PRMScore)')
    for x, v in enumerate(qvv): ax.text(x, v+.012, f'{v:.4f}', ha='center')
    ax.set_title('A different metric, a similar score', loc='left', fontweight='bold', fontsize=10)
    fig.tight_layout(); save(fig, 'primary_comparison')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    for ax, family, title in zip(axes, [data['banks'][:5], data['banks'][5:]], ['Digit-free feature banks', 'Matched banks + 3 digit channels']):
        xx = np.arange(5)
        for s in PRIMARY:
            ax.plot(xx, [metrics[b][s] for b in family], color=COLOR[s], marker='o',
                    ls='--' if s in PRIMARY[:2] else '-', lw=1.3 if s in PRIMARY[:2] else 2, label=LABEL[s])
        ax.plot(xx, [data['lr'][b] for b in family], color=COLOR['LR'], marker='s', label='LR · supervised, reported')
        ax.set_xticks(xx, family); ax.set_ylim(.725, .83); ax.set_title(title, loc='left', fontweight='bold', fontsize=11)
        ax.set_xlabel('Feature bank · channel count increases →'); ax.grid(axis='y', color='#edf0f2')
    axes[0].set_ylabel('Mean within-answer AUROC')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, -.09), ncol=3, fontsize=9)
    save(fig, 'primary_banks')
    fig, ax = plt.subplots(figsize=(10.7, 4.8)); yy = np.arange(len(TYPES))
    for s, offset, marker in [('LSML_CONT', -.11, 'o'), ('IU_PCR', .11, 'D')]:
        ax.scatter([type_auc[s][t] for t in TYPES], yy+offset, color=COLOR[s], marker=marker, s=42, label=LABEL[s], zorder=3)
    ax.scatter([data['artifact_comparator_auc'][t] for t in TYPES], yy, color=COLOR['PRM'], marker='s', s=44, label='PRM-7B · artifact, rounded', zorder=3)
    for i, t in enumerate(TYPES):
        v = [type_auc[s][t] for s in ['LSML_CONT', 'IU_PCR']] + [data['artifact_comparator_auc'][t]]
        ax.plot([min(v), max(v)], [i, i], color='#d0d9de', lw=2)
    ax.set_yticks(yy, [t.replace('_', ' ').replace('redundency', 'redundancy') for t in TYPES]); ax.invert_yaxis()
    ax.set_xlim(.62, .96); ax.set_xlabel('Mean within-answer AUROC · fixed methods across all error types')
    ax.grid(axis='x', color='#edf0f2'); ax.legend(fontsize=9, loc='lower right')
    ax.set_title('Where do Continuous L-SML and IU-PCR help?', loc='left', fontweight='bold')
    save(fig, 'primary_error_types')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={'width_ratios': [1.35, 1]})
    ax = axes[0]; xx = np.arange(3); width=.24
    for j, s in enumerate(['LSML_CONT', 'IU_PCR', 'PRM']):
        v = [category[s][g] for g in GROUPS]
        ax.bar(xx+(j-1)*width, v, width, color=COLOR[s], label=LABEL.get(s, 'PRM-7B'))
        for x, value in zip(xx+(j-1)*width, v): ax.text(x, value+.012, f'{value:.3f}', ha='center', fontsize=8)
    ax.set_xticks(xx, list(GROUPS)); ax.set_ylim(0, .9); ax.set_ylabel('Mean subtype pooled macro-F1')
    ax.set_title('Complementarity appears in simplicity', loc='left', fontweight='bold', fontsize=10)
    ax.legend(loc='upper left', fontsize=8)
    ax = axes[1]; clean = [audit['metrics'][s]['synthetic_controls']['q80_gt']['answer_false_flag_rate'] for s in ['LSML_CONT', 'IU_PCR']]
    ax.bar(range(2), clean, color=[COLOR[s] for s in ['LSML_CONT', 'IU_PCR']], width=.50)
    for x, v in enumerate(clean): ax.text(x, v+.024, f'{v:.1%}', ha='center', fontweight='bold')
    ax.set_xticks(range(2), ['Continuous\nL-SML', 'IU-PCR']); ax.set_ylim(0, 1.13)
    ax.set_ylabel('Clean answers with ≥1 false error flag')
    ax.set_title('Q80 still needs an abstention rule', loc='left', fontweight='bold', fontsize=10)
    fig.tight_layout(); save(fig, 'primary_categories')
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, bank_list, title in zip(axes, [['N16'], data['banks']], ['N16 alone', 'Mean across the 10 development banks']):
        ds = [np.mean([corrected['mask1'][b][s]-corrected['unmasked'][b][s] for b in bank_list]) for s in PRIMARY]
        ax.barh(range(4), ds, color=[COLOR[s] for s in PRIMARY], height=.58); ax.axvline(0, color='#657782', lw=1)
        ax.set_yticks(range(4), [LABEL[s] for s in PRIMARY]); ax.invert_yaxis(); ax.set_xlim(-.0019, .0065)
        for y, v in enumerate(ds): ax.text(v+(.00012 if v >= 0 else -.00012), y, f'{v:+.5f}', ha='left' if v >= 0 else 'right', va='center', fontsize=9)
        ax.set_title(title, loc='left', fontweight='bold', fontsize=10)
        ax.set_xlabel('Mask-only change in AUROC · corrected masked − corrected unmasked')
    fig.tight_layout(); save(fig, 'mask_only')
    original = (out/'audit_detail.html').read_text()
    style = original.split('<style>', 1)[1].split('</style>', 1)[0]
    plans = original.split('<section class="section" id="plan">', 1)[1].split('</section>', 1)[0]
    plans = plans.replace('0a · Prefix correction + masking', '0a · Repair scoring input')
    plans = plans.replace('New telemetry, old-bank replay, corrected dev scoring and LR rerun are recorded.', 'Prefix-corrected telemetry, corrected PRMBench panels (including Continuous L-SML and IU-PCR), and LR rerun are recorded. Masking is a separate ablation.')
    plans = plans.replace('freeze one primary scorer/bank/flag rule', 'freeze the primary L-SML / IU-PCR comparisons, bank and flag rule')
    plans = plans.replace('correct the equivalence and category claims;', 'complete the PRMBench result discussion and correct category reporting;')
    plan_mapping = plans.split('</header>', 1)[1].split('<div class="verdict">', 1)[0]
    transfer = original.split('<section class="section" id="transfer">', 1)[1].split('</section>', 1)[0]
    transfer = transfer.replace('<span class="num">05</span>', '<span class="num">A3</span>')
    transfer = transfer.replace('<p>The linked page says external benchmarks have not been rescored. The current branch contains them. Two protocols answer different questions, so keep them separate:</p>', '<p>These corrected results are context for the remaining generalization work. They use different metrics and protocols from the main PRMBench comparison.</p>')
    options = ''.join('<option value="'+b+'" '+('selected' if b=='N16' else '')+'>'+b+'</option>' for b in data['banks'])
    figures = {
        'COMPARE': figure('primary_comparison', 'N16, corrected prefix + mask1. Label-free AUC and Q80 arithmetic independently reproduced from exported scores; LR is branch-reported OOF. PRM AUC ≈0.801 comes from the supplied artifact; PRMScore 0.6546 from its evaluator manifest. Different decision thresholds; development comparison, no superiority/equivalence claim.'),
        'BANKS': figure('primary_banks', 'All 10 corrected mask1 banks; same development cohort. L-SML and IU-PCR use native label-free orientation. Sources: cont_baselines/ABSOLUTE.csv, PANEL_A_ABSOLUTE.csv and LR_SCORES.csv. These are point estimates; no significance claim is made from this plot.'),
        'TYPES': figure('primary_error_types', 'L-SML and IU-PCR type AUCs freshly reconstructed from the N16 exported scores with the audited pairwise definition. Comparator dots are rounded AUCs from the supplied artifact, not raw-score replay. Multi-solutions have no error/correct contrast and therefore no within-answer AUC.'),
        'CATEGORIES': figure('primary_categories', 'L-SML and IU-PCR: global answer-z Q80; PRM-7B: threshold 0.5. Synthetic controls excluded from pooled PRMScore and shown separately (758 answers). *Sensitivity mean covers missing-condition and deception only; full taxonomy also includes multi-solutions. This is not a certified full leaderboard category score.'),
        'MASK': figure('mask_only', 'Same corrected think prefix in both arms. Mask-only deltas are distinct from the old → corrected-prefix change. Equal mean across banks is descriptive, not a new selection objective. Source: PRMBENCH_COMPARISON.csv.'),
        'PREFIX': figure('prefix_correction', 'Historical audit: old replay → corrected unmasked → corrected + mask1. Keep the two input changes separate. Main presentation uses corrected results throughout.')}
    page = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PRMBench · Continuous L-SML and IU-PCR</title><style>STYLE
.methodcards{display:grid;grid-template-columns:1fr 1fr;gap:24px}.methodcard{background:white;padding:24px;border:1px solid var(--line)}.flow{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0}.flow span{padding:9px 13px;border:1px solid var(--line);background:var(--tint);font-size:13px}.flow i{align-self:center;font-style:normal;color:var(--muted)}.question{border-top:3px solid var(--blue);padding:18px;background:white}.note{padding:12px 18px;border-left:3px solid #278577;background:#edf5f2}.taxonomy{display:grid;grid-template-columns:repeat(3,1fr);gap:20px;margin:26px 0}.taxonomy div{border-top:2px solid var(--blue);padding:18px 0}.taxonomy strong{font:26px Georgia,serif}.tags{font-size:12px;color:var(--muted)}.appendix{margin-top:50px}.appendix>details{padding:24px 0;border-top:2px solid var(--line)}.appendix>details>summary{font:24px Georgia,serif}.appendix h2{font-size:27px}.mini{font-variant-numeric:tabular-nums}.protocol{display:flex;gap:16px;align-items:center;flex-wrap:wrap}.summaryline{font-size:18px;margin:18px 0}.two .question p{font-size:15px}.methodcard svg{width:100%;height:auto;display:block}.barrow.primary{font-weight:650}.barrow.primary .barfill{background:#278577}.hero-note{font-size:14px;color:var(--muted)}
@media(max-width:700px){.methodcards,.taxonomy{grid-template-columns:1fr}.methodcard{padding:18px}}@media print{.appendix>details:not([open])>*:not(summary){display:none}.appendix>details{break-before:page}.methodcard,.taxonomy{break-inside:avoid}}
</style></head><body><div class="wrap"><nav><a href="#top">Overview</a><a href="#task">PRMBench</a><a href="#methods">L-SML &amp; IU-PCR</a><a href="#results">Results</a><a href="#banks">Feature banks</a><a href="#types">Error types</a><a href="#decisions">Detection</a><a href="#plan">Plan</a><a href="#appendix">Appendix</a><button onclick="window.print()">Print / save PDF</button></nav><main>
<div class="kicker" id="top">Advisor discussion · PRMBench results · updated 11 October 2026</div>
<h1>PRMBench: what do<br>Continuous L-SML and IU-PCR show?</h1>
<p class="lede">Complete the result discussion we barely reached on October 8: compare the established label-free methods, understand their strengths by error type, and identify what still blocks the project’s claim.</p>
<p class="hero-note">All main results use the corrected Qwen3-8B prefix. The campaign’s primary mask1 input is retained for consistency; corrected unmasked results and the rationale for masking are explicit in Appendix A1. Pipeline variants and the rescoring audit are supporting material.</p>
<div class="metrics"><div class="metric"><span class="label">Continuous L-SML · N16</span><strong>0.779</strong><span>Within-answer AUROC<br>0.6538 Q80 PRMScore</span></div><div class="metric"><span class="label">IU-PCR · N16</span><strong>0.780</strong><span>Within-answer AUROC<br>0.6536 Q80 PRMScore</span></div><div class="metric"><span class="label">PRM-7B · reference</span><strong>≈0.801</strong><span>AUC reported in supplied artifact<br>0.6546 PRMScore from manifest</span></div><div class="metric"><span class="label">Supervised LR · N16</span><strong>0.801</strong><span>Group-OOF AUC, branch report<br>≈0.021 ranking headroom</span></div></div>
<p class="speaker" lang="he" dir="rtl"><b>פתיחה לפגישה:</b> בפגישה הקודמת כמעט לא הספקנו לדון ב־PRMBench. היום נציג את התמונה המלאה עם הקלט המתוקן: L-SML הרציף ו־IU-PCR, השוואה ל־PRM, הבדלים בין סוגי שגיאות, ומה נדרש כדי להפוך את התוצאות לטענה מחקרית מבוססת.</p>
<section class="section" id="task"><header><span class="num">01</span><h2>What does PRMBench ask us to detect?</h2></header>
<p>Rank and judge individual steps in supplied mathematical solutions. Its errors cover logical simplicity, soundness and sensitivity—not only a wrong final answer. We extract token-level signals in a teacher-forced forward pass, aggregate them into step channels, and fuse them without fitting to error labels.</p>
<div class="taxonomy"><div><h3>Simplicity</h3><strong>1,516 answers</strong><p>Redundancy and circularity. A step can be unhelpful even when the final answer remains correct.</p></div><div><h3>Soundness</h3><strong>3,029 answers</strong><p>Confidence, counterfactuals, domain inconsistency and step contradiction.</p></div><div><h3>Sensitivity</h3><strong>1,666 answers</strong><p>Missing conditions, deception and multiple valid solutions. The last subtype needs valid-step judgment.</p></div></div>
<div class="flow"><span>Supplied solution</span><i>→</i><span>Corrected Qwen3-8B telemetry</span><i>→</i><span>Step feature bank</span><i>→</i><span>Continuous L-SML / IU-PCR</span><i>→</i><span>Rank steps</span><i>→</i><span>Choose error flags</span></div>
<div class="two"><div class="question"><h3>Ranking question</h3><p>Does an annotated error receive a higher risk score than a correct step in the same answer? Mean within-answer AUROC uses 6,030 answers with both kinds of step.</p></div><div class="question"><h3>Detection question</h3><p>Which steps should be flagged, and can the method accept a clean answer? PRMScore pools valid/error F1 on the 6,211 noncontrol answers. Clean controls are checked separately.</p></div></div>
<p class="small">Export: 6,969 answers / 94,203 steps, including 758 synthetic correct controls. Counts independently checked against copied metadata. Taxonomy: <a href="https://arxiv.org/html/2501.03124v1">PRMBench paper</a>. Labels are used in evaluation; bank design and method choices remain development choices.</p>
</section>
<section class="section" id="methods"><header><span class="num">02</span><h2>Put the established methods first</h2></header>
<div class="methodcards"><article class="methodcard"><h3>Continuous L-SML</h3><p>Estimate dependent groups of channels, build a continuous score within each group, then fuse those group scores. Keep the continuous signal throughout.</p>
<svg viewBox="0 0 440 145" role="img" aria-label="Conceptual Continuous L-SML diagram: channels form groups, each group creates a continuous score, then group scores are fused"><g fill="#edf4f8" stroke="#246c9c"><rect x="8" y="10" width="80" height="37" rx="4"/><rect x="8" y="54" width="80" height="37" rx="4"/><rect x="8" y="98" width="80" height="37" rx="4"/><rect x="155" y="10" width="99" height="37" rx="4"/><rect x="155" y="54" width="99" height="37" rx="4"/><rect x="155" y="98" width="99" height="37" rx="4"/><rect x="328" y="48" width="104" height="49" rx="4"/></g><g stroke="#246c9c" fill="none"><path d="M88 28H155 M88 72H155 M88 116H155 M254 28L328 72 M254 72H328 M254 116L328 72"/></g><g fill="#21323d" font-family="system-ui" font-size="12" text-anchor="middle"><text x="48" y="33">Group A</text><text x="48" y="77">Group B</text><text x="48" y="121">Group C</text><text x="205" y="33">Group score</text><text x="205" y="77">Group score</text><text x="205" y="121">Group score</text><text x="380" y="77">Fused score</text></g></svg>
<p class="tags">Conceptual diagram, not a fitted N16 grouping. Native sign; continuous grouping and fusion; no added tail binarization or external filtering. This is the implemented continuous extension, not a theorem for arbitrary continuous channels.</p></article>
<article class="methodcard"><h3>IU-PCR</h3><p>Estimate feature–target covariance from the observed channel covariance under its moment assumptions. Form a weighted score in the two-component covariance subspace.</p>
<svg viewBox="0 0 440 145" role="img" aria-label="Conceptual IU-PCR diagram: observed feature covariance estimates target covariance; two principal components provide fusion weights"><g fill="#edf5f2" stroke="#278577"><rect x="7" y="38" width="112" height="66" rx="4"/><rect x="165" y="13" width="109" height="48" rx="4"/><rect x="165" y="86" width="109" height="48" rx="4"/><rect x="329" y="38" width="103" height="66" rx="4"/></g><g stroke="#278577" fill="none"><path d="M119 70L165 37 M119 70L165 110 M274 37L329 70 M274 110L329 70"/></g><g fill="#21323d" font-family="system-ui" font-size="12" text-anchor="middle"><text x="63" y="65">Observed</text><text x="63" y="82">covariance C</text><text x="220" y="34">Estimate ρ</text><text x="220" y="49">without labels</text><text x="220" y="106">First two</text><text x="220" y="121">components</text><text x="380" y="65">Weights w</text><text x="380" y="82">Score = wᵀF</text></g></svg>
<p class="tags">Conceptual diagram. Frozen two-component L2 configuration, scale ratio 0.25; native orientation; no exclusion, difficulty gate or average fallback. Assumptions and settings matter.</p></article></div>
<div class="note"><b>Fair comparison:</b> the same step channels, same bank and cell fitting scope. Plain average and DS-filter average are controls. Four newer tail-pipeline variants test whether extra filtering, clustering and merging help; they do not define the main story.</div>
<p class="small"><a href="sources/CONT_BASELINES_PROTOCOL.json">Frozen October 7 comparison protocol</a> · <a href="https://github.com/omrisegev/hallucination_detection/blob/HEADVALUE/results/thinkfix_rescore_v1/thinkfix_mask1/cont_baselines/run_patched.py">Exact corrected implementations</a>. “Label-free fitting” does not make the development-selected bank or presentation held out.</p>
</section>
<section class="section" id="results"><header><span class="num">03</span><h2>The overall comparison: useful signal, modest fusion gains</h2></header>
COMPARE
<div class="two"><div class="question"><h3>L-SML and IU-PCR improve on a plain average</h3><p>N16 AUROC increases from 0.7708 to 0.7792 / 0.7805: descriptive gains of +0.0084 / +0.0097. DS-filter average reaches 0.7801, so neither complex method has established an overall advantage over that control.</p></div><div class="question"><h3>PRMScore and ranking answer different questions</h3><p>Our Q80 pooled scores are close to the PRM reference, although the reported PRM ranking AUC is higher. Different flag thresholds, error-type mixtures and pooling can produce this pattern. Close point estimates do not establish equivalence.</p></div></div>
<p class="takeaway">The project has a label-free ranking signal worth studying. The new tail-pipeline variants do not establish a better N16 fusion method: their AUC range is 0.7767–0.7789.</p>
<p class="speaker" lang="he" dir="rtl"><b>מסר:</b> השיטות הרציפות נותנות אות ברור, אבל התוספת של הפייפליינים החדשים לא משפרת את התמונה ב־N16. ההשוואה החשובה היא בין השיטות הקיימות, הממוצעים וה־PRM, ולא מי ניצח באחד הווריאנטים החדשים.</p>
</section>
<section class="section" id="banks"><header><span class="num">04</span><h2>More features help LR more consistently than label-free fusion</h2></header>
BANKS
<div class="two"><div><h3>N13 → N16 isolates the added digit channels</h3><p>Continuous L-SML rises by <b>LSMLDIGIT</b>; IU-PCR by <b>IUDIGIT</b>. Larger banks do not sustain that improvement. Preserve the digit-free comparison and test the added value on held-out source groups.</p><h3>The remaining LR gap is a research question</h3><p>On N16, LR exceeds L-SML by 0.0223 and IU-PCR by 0.0210 AUROC. Investigate feature redundancy, useful contrasts and which errors LR recovers. These are hypotheses, not conclusions from the curve.</p></div><div class="explore"><div class="tools"><label for="bank">Inspect bank</label><select id="bank">OPTIONS</select></div><div id="bars" aria-live="polite"></div><div class="legend">Main methods and controls · corrected + mask1 · axis 0.70–0.83. Supervised LR is branch-reported.</div><details><summary>Exact channels in this bank</summary><div id="channels" class="small"></div></details></div></div>
<details><summary>Full bank × scorer matrix, including pipeline variants</summary><div class="matrix" id="matrix"></div><p class="small">No per-type or per-bank winner replacement. Main and appendix values are development evidence; no mean-of-eight headline.</p></details>
</section>
<section class="section" id="types"><header><span class="num">05</span><h2>The comparison changes with the kind of error</h2></header>
TYPES
<div class="two"><div class="question"><h3>Simplicity is the strongest complementary signal</h3><p>Both continuous methods outperform the reported PRM on redundancy and circularity. Those labels include unnecessary reasoning that can preserve final-answer correctness. This is a candidate contribution for the project.</p></div><div class="question"><h3>The advantage needs controls</h3><p>The PRM is stronger on several soundness and sensitivity types. Repetition, step position or length may explain some of our simplicity signal. Test these explanations before claiming a general quality detector.</p></div></div>
CATEGORIES
<p class="summaryline"><b>Ranking success does not solve abstention:</b> global Q80 falsely flags at least one step in 99.5% / 99.3% of synthetic correct answers for L-SML / IU-PCR.</p>
<p class="small">Multiple valid solutions: valid-step F1 is 0.8840 / 0.8859 for L-SML / IU-PCR. There are no error steps in that subtype, so its AUC is undefined. It remains part of the full detection taxonomy; the two-subtype Sensitivity plot above is explicitly partial. See the appendix for aggregation details.</p>
</section>
<section class="section" id="decisions"><header><span class="num">→</span><h2>What does this mean for the project?</h2></header>
<div class="two"><div class="question"><h3>Supported on the current development data</h3><p>Continuous L-SML and IU-PCR rank errors above correct steps. Their simplicity signal complements the PRM. Added pipeline complexity has not established a better N16 method, and supervised LR leaves ranking headroom.</p></div><div class="question"><h3>Still required for the intended claim</h3><p>Freeze the comparisons, validate on untouched source groups, control position and repetition, and separate localization from accepting a clean answer. External target refitting is useful exploration but does not complete source-only transfer validation.</p></div></div>
<p class="speaker" lang="he" dir="rtl"><b>השאלה למנחים:</b> האם נתמקד בהוכחת האות המשלים בשגיאות פשטות, עם בקרות והכללה, או ננסה קודם לסגור את הפער מול LR? במקביל נדרשת החלטת הימנעות כדי שהשיטה תדע לא לסמן שגיאה בתשובה נקייה.</p>
</section>
<section class="section" id="plan"><header><span class="num">06</span><h2>From corrected results to a defensible research claim</h2></header>
<div class="taxonomy"><div><h3>Completed / available</h3><strong>Corrected result panel</strong><p>Continuous L-SML, IU-PCR, controls and LR across 10 banks. Full PRMBench discussion, type comparisons and independently checked N16 score arithmetic.</p></div><div><h3>Next priority</h3><strong>Validate the signal</strong><p>Freeze methods, input and bank before source-group confirmation. Control repetition, position and length; verify the simplicity advantage and full category reporting.</p></div><div><h3>Remaining project work</h3><strong>Explain and generalize</strong><p>Analyze the LR gap and conditional feature correlations. Develop clean-answer abstention; test source-only transfer and own-answer behavior under matched protocols.</p></div></div>
<p class="takeaway">First finish the PRMBench result discussion. Then test the current primary methods on held-out data before investing in another broad fusion sweep.</p>
<details><summary>Full completion map against the October 8 advisor plan</summary>PLANMAPPING</details>
</section>
<section class="appendix" id="appendix"><div class="kicker">Supporting evidence · open only when needed</div><h2>Appendix</h2>
<details><summary>A1 · Does masking help, and why consider it?</summary><p>Masking drops the first <b>content token of the answer</b> before step-channel extraction, with aligned spans. It is distinct from repairing the missing empty think block. The rationale is to test whether a prompt-boundary effect contaminates telemetry; the token can also contain useful error signal.</p>
MASK
<p><b>For N16: yes, it slightly hurts both primary methods.</b> Continuous L-SML changes by −0.00075; IU-PCR by −0.00013 AUROC. Across all 10 banks the picture differs: L-SML averages −0.00085 (4 improve / 6 worsen), IU-PCR +0.00480 (8 improve / 2 worsen).</p>
<div class="note"><b>Interpretation:</b> masking is an optional preprocessing ablation, not a required consequence of the think-prefix fix or a demonstrated universal improvement. Choose the preprocessing contract before confirmation; do not select it by the highest development score.</div>
<div class="explore"><div class="protocol"><label for="ablation-bank">Bank</label><select id="ablation-bank">OPTIONS</select><label for="input">Corrected input</label><select id="input"><option value="mask1">Corrected + mask1 · campaign primary</option><option value="unmasked">Corrected · unmasked</option></select></div><div id="ablation-bars" aria-live="polite"></div><p class="small" id="ablation-note"></p></div>
PREFIX
<p class="small">The large old → new score drop mainly belongs to prefix repair, not masking. Original extraction diagnostics suggested a first-token boundary mismatch, but did not prove that every first content token is irrelevant. Main panels retain the campaign primary mask1 input transparently; no protocol promotion was made.</p></details>
<details><summary>A2 · Review, flag rules and reproducibility</summary><p>The central corrected N16 score arithmetic passes independent checks. Raw telemetry replay and full end-to-end validation remain unavailable from the published artifacts. The code review still flags false equivalence wording, supervised leakage in restricted LR feature selection, stale-resume risk, an old-input entropy gate and incomplete Sensitivity reporting.</p><p>Q80 is one global 80th percentile over answer-normalized steps; it is not a fixed 20% quota per answer. TOP1/TOP2 force flags and are diagnostics. Noncontrol pooled PRMScore includes multi-solutions; the partial Sensitivity chart does not. No full official leaderboard aggregation parity is claimed.</p><p><a href="REVIEW.md">Detailed code and numerical review</a> · <a href="audit_detail.html#review">Six findings and evidence boundary</a> · <a href="audit_detail.html#flags">Full flag-rule and category details</a> · <a href="PRESENTATION_DATA.json">Presentation data and reconstructed type AUCs</a> · <a href="evidence/A/audit.json">Independent metric audit</a>.</p></details>
<details><summary>A3 · External benchmarks and own-generation context</summary>TRANSFER</details>
</section><p class="footer">Pinned branch <a href="https://github.com/omrisegev/hallucination_detection/tree/HEADVALUE">claude/thinkfix-rescore-v1 @ 080877eb</a>. Revised meeting narrative follows the October 11 clarification: present the full corrected PRMBench results omitted from the October 8 discussion. Main focus: Continuous L-SML and IU-PCR. Local replacement artifact; published Claude artifact and research results unchanged. <a href="sources/README_HE.md">October 8 advisor plan</a> · <a href="build_presentation.py">Portable presentation builder</a>.</p>
</main></div><script id="presentation-data" type="application/json">JSONDATA</script><script id="full-data" type="application/json">FULLDATA</script>
<script>
const data=JSON.parse(document.getElementById('presentation-data').textContent), full=JSON.parse(document.getElementById('full-data').textContent);
const bank=document.getElementById('bank'), abank=document.getElementById('ablation-bank'), input=document.getElementById('input');
function bars(b, arm, target){const series=data.primary.map(s=>[data.names[s],data.corrected[arm][b][s],s]);series.push(['LR · supervised',data.lr[arm][b],'LR']);document.getElementById(target).innerHTML=series.map(([name,value,s])=>`<div class="barrow ${s==='LSML_CONT'||s==='IU_PCR'?'primary':''}"><span>${name}</span><div class="bartrack"><div class="barfill" style="width:${Math.max(0,Math.min(100,(value-.70)/.13*100))}%"></div></div><span class="barvalue">${value.toFixed(4)}</span></div>`).join('');}
function draw(){bars(bank.value,'mask1','bars');document.getElementById('channels').textContent=data.bank_features[bank.value].join(' · ');}
function ablation(){bars(abank.value,input.value,'ablation-bars');document.getElementById('ablation-note').textContent='Explorer only: '+(input.value==='mask1'?'corrected + mask1':'corrected unmasked')+'. All main figures retain the mask1 protocol. Label-free values reported by the pinned branch; N16 mask1 independently checked.';}
bank.addEventListener('change',draw);abank.addEventListener('change',ablation);input.addEventListener('change',ablation);draw();ablation();
document.getElementById('matrix').innerHTML='<table><caption>All corrected mask1 methods · mean within-answer AUROC</caption><thead><tr><th>Bank</th>'+full.names.map(n=>'<th>'+n+'</th>').join('')+'<th>Supervised LR</th></tr></thead><tbody>'+full.banks.map(b=>'<tr class="'+(b==='N16'?'focus':'')+'"><th>'+b+'</th>'+full.roster.map(s=>'<td>'+full.metrics[b][s].toFixed(4)+'</td>').join('')+'<td>'+full.lr[b].toFixed(4)+'</td></tr>').join('')+'</tbody></table>';
</script></body></html>'''
    replacements = dict(STYLE=style, OPTIONS=options, PLANMAPPING=plan_mapping, TRANSFER=transfer,
                        HEADVALUE=data['head'], JSONDATA=json.dumps(present).replace('</', '<\\/'),
                        FULLDATA=json.dumps(data).replace('</', '<\\/'),
                        LSMLDIGIT=f"{metrics['N16']['LSML_CONT']-metrics['N13']['LSML_CONT']:+.4f}",
                        IUDIGIT=f"{metrics['N16']['IU_PCR']-metrics['N13']['IU_PCR']:+.4f}", **figures)
    # Only uppercase marker words: replacing BANKS inside embedded data or SVG text would corrupt it.
    pattern = re.compile(r'\b('+'|'.join(map(re.escape, replacements))+r')\b')
    page = pattern.sub(lambda match: replacements[match.group()], page)
    (out/'advisor_review.html').write_text(page)
    return present


if __name__ == '__main__':
    result = build()
    print(json.dumps(dict(primary=result['primary'], mask_only=result['mask_only'], output=str(Path(__file__).resolve().parent/'advisor_review.html')), indent=2))
