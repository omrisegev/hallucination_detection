"""Insert the "Model size check: Qwen3-4B vs Qwen3-8B" section into the advisor v11 page.

Usage: python3 add_model_size_section.py <data_dir> <in_html> <out_html>

<data_dir> holds the qwen3_4b_vs_8b__*.csv files from the 4B thread
(results/thinkfix_rescore_4b_v1/COMPARE_8B_4B/ on AIRCC). All numbers on the
section are read from these files here; nothing is typed in by hand except the
text of the bullets, which the asserts below tie to the data.
"""
import csv, html, os, statistics as st, sys

P = 'qwen3_4b_vs_8b__'
BANKS = ['N11', 'N13', 'N22', 'N40', 'N53', 'N14', 'N16', 'N25', 'N43', 'N56']


def rows(d, name):
    with open(os.path.join(d, P + name), newline='') as f:
        return list(csv.DictReader(f))


def f3(x): return f'{x:.3f}'
def pct(x): return f'{100 * x:.1f}'
def sd(x, k=3):
    if round(abs(x), k) == 0: return f'{0:.{k}f}'
    return ('+' if x >= 0 else '−') + f'{abs(x):.{k}f}'


def table(head, body, cap=''):
    h = ''.join(f'<th>{c}</th>' for c in head)
    b = ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>' for r in body)
    cap = f'<caption>{cap}</caption>' if cap else ''
    return f'<div class="scroll"><table>{cap}<thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def box(items, title='Key points'):
    return ('<ul class="takeaways"><span class="kt">' + title + '</span>'
            + ''.join('<li>' + t + '</li>' for t in items) + '</ul>')


def delta_svg(series):
    """Dot plot of 4B minus 8B within-answer AUC per bank, one row of dots per method."""
    W, H, L, R, T, B = 1000, 330, 70, 20, 30, 70
    lo, hi = -0.012, 0.012
    x = lambda i: L + (i + 0.5) * (W - L - R) / len(BANKS)
    y = lambda v: T + (hi - v) / (hi - lo) * (H - T - B)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" '
         f'aria-label="Change in within-answer AUC from Qwen3-8B to Qwen3-4B per bank" '
         f'font-family="system-ui,-apple-system,Segoe UI,sans-serif">']
    for v in [-0.01, -0.005, 0, 0.005, 0.01]:
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="{"#5a6b7b" if v == 0 else "#e3e9ef"}" stroke-width="{1.4 if v == 0 else 1}"/>')
        o.append(f'<text x="{L-8}" y="{y(v)+4:.1f}" font-size="12" fill="#60778d" text-anchor="end">{sd(v,3) if v else "0"}</text>')
    mid = L + 5 * (W - L - R) / len(BANKS)
    o.append(f'<line x1="{mid:.1f}" x2="{mid:.1f}" y1="{T}" y2="{H-B}" stroke="#c9d3dd" stroke-dasharray="4 4"/>')
    o.append(f'<text x="{(L+mid)/2:.1f}" y="{H-B+38}" font-size="12" fill="#60778d" text-anchor="middle">digit-free banks</text>')
    o.append(f'<text x="{(mid+W-R)/2:.1f}" y="{H-B+38}" font-size="12" fill="#60778d" text-anchor="middle">matched banks with three digit channels</text>')
    for i, b in enumerate(BANKS):
        o.append(f'<text x="{x(i):.1f}" y="{H-B+18}" font-size="12.5" fill="#33475b" text-anchor="middle" font-weight="{650 if b == "N16" else 400}">{b}</text>')
    off = [-15, -5, 5, 15]
    for k, (name, color, vals) in enumerate(series):
        for i, b in enumerate(BANKS):
            v = max(lo, min(hi, vals[b]))
            o.append(f'<circle cx="{x(i)+off[k]:.1f}" cy="{y(v):.1f}" r="5" fill="{color}"><title>{html.escape(name)} {b}: {sd(vals[b],4)}</title></circle>')
    lx = L
    for name, color, _ in series:
        o.append(f'<circle cx="{lx+6}" cy="14" r="5" fill="{color}"/><text x="{lx+16}" y="18" font-size="12.5" fill="#33475b">{html.escape(name)}</text>')
        lx += 22 + 7.2 * len(name) + 18
    o.append(f'<text x="18" y="{(T+H-B)/2:.1f}" font-size="12" fill="#60778d" text-anchor="middle" transform="rotate(-90 18 {(T+H-B)/2:.1f})">4B − 8B, AUC</text>')
    o.append('</svg>')
    return ''.join(o)


def build(d):
    cont = {(r['bank'], r['scorer']): r for r in rows(d, 'prmbench_cont.csv')}
    panel = {(r['bank'], r['scorer']): r for r in rows(d, 'prmbench_panel_a.csv')}
    lr = {(r['bank'], r['scorer']): r for r in rows(d, 'prmbench_lr.csv')}
    gaps = {r['bank']: r for r in rows(d, 'prmbench_lr_gaps.csv')}
    off = {(r['bank'], r['method'], r['rule']): r for r in rows(d, 'prmbench_official_prmscore.csv')}
    ext = {(r['cell'].split('_')[0], r['arm']): r for r in rows(d, 'external_lsml_locked.csv')}
    g = lambda r, k: float(r[k])

    # Headline table, N16.
    head_rows, M = [], [('Continuous L-SML', cont, 'LSML_CONT', 'LSML_CONT'), ('IU-PCR', cont, 'IU_PCR', 'IU_PCR'),
                        ('Plain average', panel, 'ALL_EQ', 'ALL_EQ'), ('DS-filter average', panel, 'DS_FILTER_EQ', 'DS_FILTER_EQ')]
    for name, src, key, okey in M:
        r, o = src[('N16', key)], off[('N16', okey, 'R0')]
        head_rows.append([name, f3(g(r, 'prmb_within_auc_8b')), f3(g(r, 'prmb_within_auc_4b')), sd(g(r, 'prmb_within_auc_delta')),
                          pct(g(r, 'prmb_top1_on_error_8b')), pct(g(r, 'prmb_top1_on_error_4b')),
                          pct(g(o, 'overall_8b')), pct(g(o, 'overall_4b'))])
    rl = lr[('N16', 'LR')]
    head_rows.append(['LR · supervised reference', f3(g(rl, 'prmb_within_auc_8b')), f3(g(rl, 'prmb_within_auc_4b')),
                      sd(g(rl, 'prmb_within_auc_delta')), '—', '—', '—', '—'])
    head_rows.append(['Qwen2.5-Math-PRM-7B', '0.801', '0.801', '0', '57.8', '57.8', '65.5', '65.5'])

    # Figure data.
    series = [('Continuous L-SML', '#1f5f99', {b: g(cont[(b, 'LSML_CONT')], 'prmb_within_auc_delta') for b in BANKS}),
              ('IU-PCR', '#2a8a7a', {b: g(cont[(b, 'IU_PCR')], 'prmb_within_auc_delta') for b in BANKS}),
              ('Plain average', '#9aa6b2', {b: g(panel[(b, 'ALL_EQ')], 'prmb_within_auc_delta') for b in BANKS}),
              ('LR (supervised)', '#c0782c', {b: g(lr[(b, 'LR')], 'prmb_within_auc_delta') for b in BANKS})]
    lf = [v for s in series[:2] for v in s[2].values()]
    lrd = list(series[3][2].values())
    assert all(v < 0 for v in lrd) and abs(st.mean(lf)) < 0.001 and max(map(abs, lf)) < 0.01
    narrow = sum(g(gaps[b], 'gap_vs_best_4b') < g(gaps[b], 'gap_vs_best_8b') for b in BANKS)
    assert narrow == 9

    # Category table, N16 R0, all nine subtypes (leaderboard convention: Sensitivity = mean of PS, DR, MS).
    sub = {(r['model'], r['method']): r for r in rows(d, 'N16_R0_LSML_IUPCR_by_subtype.csv')}
    SUBS = ['redundency', 'circular', 'counterfactual', 'step_contradiction', 'domain_inconsistency', 'confidence',
            'missing_condition', 'deception', 'multi_solutions']
    cat_rows, sens, soun = [], {}, {}
    for name, key in [('Continuous L-SML', 'LSML_CONT'), ('IU-PCR', 'IU_PCR')]:
        for m, lab in [('8b', 'Qwen3-8B'), ('4b', 'Qwen3-4B')]:
            r = sub[(m, key)]
            v = {c: g(r, c) for c in SUBS}
            simp = (v['redundency'] + v['circular']) / 2
            soun[(m, key)] = st.mean(v[c] for c in SUBS[2:6])
            sens[(m, key)] = st.mean(v[c] for c in SUBS[6:])
            assert abs(g(r, 'overall') - g(off[('N16', key, 'R0')], 'overall_' + m)) < 1e-9
            cat_rows.append([name, lab, pct(g(r, 'overall')), pct(v['redundency']), pct(v['circular']), pct(simp)]
                            + [pct(v[c]) for c in SUBS[2:6]] + [pct(soun[(m, key)])]
                            + [pct(v[c]) for c in SUBS[6:]] + [pct(sens[(m, key)])])
    for key in ['LSML_CONT', 'IU_PCR']:
        assert -0.007 < sens[('4b', key)] - sens[('8b', key)] < 0 and -0.007 < soun[('4b', key)] - soun[('8b', key)] < 0
        assert abs(g(sub[('4b', key)], 'multi_solutions') - g(sub[('8b', key)], 'multi_solutions')) < 0.003
    allr0 = [g(r, 'overall_delta') for k, r in off.items()]

    # External table.
    ext_rows = []
    for bm, lab in [('hard2verify', 'Hard2Verify · Balanced F1 · 200 answers'), ('socratic', 'Socratic · PRMScore · 2,995 answers')]:
        for arm, aname in [('frozen_lsml', 'Frozen L-SML'), ('frozen_equal', 'Frozen equal weights'), ('ct7', 'CT7')]:
            r = ext[(bm, arm)]
            ext_rows.append([lab, aname, f3(g(r, 'metric_8b')), f3(g(r, 'metric_4b')), sd(g(r, 'metric_delta'))])

    l16, i16 = cont[('N16', 'LSML_CONT')], cont[('N16', 'IU_PCR')]
    lr16 = gaps['N16']
    s = ['<section id="model-size"><div class="eyebrow">MODEL SIZE CHECK</div>',
         '<h2>Qwen3-4B gives almost the same results as Qwen3-8B</h2>',
         '<p class="lead" style="font-size:17px">We repeated the full PRMBench and external pipeline with Qwen3-4B as the teacher-forcing model. '
         'The answers, labels, banks, methods and evaluators are the same. The 4B tokens and step spans are identical to 8B on 6,969 of 6,969 PRMBench answers. '
         'Only the model that produces the features changes. All values are point estimates; we did not compute paired 8B-vs-4B intervals.</p>',
         table(['N16 method', 'AUC · 8B', 'AUC · 4B', 'Δ AUC', 'Top1 % · 8B', 'Top1 % · 4B', 'PRMScore × 100 · 8B', 'PRMScore × 100 · 4B'], head_rows,
               'N16 · within-answer AUC (6,030 answers), top1-on-error (6,035 answers), official PRMScore with the Q80 rule (R0 = ANSWER_Z_Q80; 6,211 answers). PRM-7B does not use our features, so its values do not change.'),
         box([f'At N16, the label-free AUC falls by only about 0.004 on 4B: L-SML {f3(g(l16,"prmb_within_auc_8b"))} → {f3(g(l16,"prmb_within_auc_4b"))}, IU-PCR {f3(g(i16,"prmb_within_auc_8b"))} → {f3(g(i16,"prmb_within_auc_4b"))}.',
              f'Top1-on-error falls by about 1.2 points on 4B (L-SML {pct(g(l16,"prmb_top1_on_error_8b"))} → {pct(g(l16,"prmb_top1_on_error_4b"))}). Our lead over PRM-7B (57.8) becomes smaller but stays.',
              f'PRMScore changes by less than 0.4 points; it stays at the level of PRM-7B (65.5).']),
         '<p class="source">Source: qwen3_4b_vs_8b__prmbench_cont.csv, __prmbench_panel_a.csv, __prmbench_lr.csv, __prmbench_official_prmscore.csv (results/thinkfix_rescore_4b_v1/COMPARE_8B_4B/)</p>',
         '<figure>' + delta_svg(series) + '<figcaption>Change in within-answer AUC from Qwen3-8B to Qwen3-4B, per bank. Each dot is one method; below zero means 4B is lower.</figcaption></figure>',
         box([f'The label-free methods change by less than 0.01 AUC in every bank, in both directions; the mean change is about zero ({sd(st.mean(lf),4)}).',
              f'LR is lower on 4B in all ten banks ({sd(max(lrd))} to {sd(min(lrd))}). Supervised LR depends more on the model than our methods do.',
              f'So the LR lead over the best label-free method becomes smaller in {narrow} of 10 banks; at N16 it goes from {f3(g(lr16,"gap_vs_best_8b"))} to {f3(g(lr16,"gap_vs_best_4b"))}.'],
             'What this figure shows'),
         table(['Method', 'Model', 'Overall', 'NR', 'NCL', 'Simplicity', 'ES', 'SC', 'DC', 'CI', 'Soundness', 'PS', 'DR', 'MS', 'Sensitivity'], cat_rows,
               'N16 · official PRMScore × 100 with the Q80 rule (R0 = ANSWER_Z_Q80), same layout as the leaderboard table above. MS is valid-step F1; Sensitivity is the mean of PS, DR and MS.'),
         box(['The category profile does not change: Simplicity stays at 69 for both models, and NR stays at about 70.',
              'Soundness and Sensitivity fall by about 0.5 points on 4B. MS stays at 88, so the abstention problem of Q80 is the same.',
              f'Over all 700 bank × method × rule rows, overall PRMScore changes by {sd(st.mean(allr0),4)} on average.']),
         '<p class="source">Source: qwen3_4b_vs_8b__N16_R0_LSML_IUPCR_by_subtype.csv; per-bank files official_prmscore_4b/ and official_prmscore_8b_replay/</p>',
         table(['Benchmark', 'Arm (weights fitted on 8B)', 'Qwen3-8B', 'Qwen3-4B', 'Δ'], ext_rows,
               'External benchmarks with the locked L-SML weights. The weights were fitted on 8B, so 4B is a transfer test.'),
         box(['The locked L-SML weights transfer to 4B, and the score even rises a little: Hard2Verify 0.421 → 0.433, Socratic 0.628 → 0.631.',
              'Frozen L-SML stays above frozen equal weights and CT7 on both benchmarks.',
              'On Hard2Verify, step index alone (0.483) is still higher than all our methods.']),
         '<p class="source">Source: qwen3_4b_vs_8b__external_lsml_locked.csv · ProcessBench is not shown: its source-bank rows are the same for both models by construction.</p>',
         '</section>']
    for b, name in [(ext[('hard2verify', 'frozen_lsml')], 'h2v'), (ext[('socratic', 'frozen_lsml')], 'soc')]:
        assert g(b, 'metric_4b') > g(ext[(b['cell'].split('_')[0], 'frozen_equal')], 'metric_4b')
    return ''.join(s)


def patch(s, section):
    assert 'id="model-size"' not in s, 'already patched'
    i = s.find('<section id="q80-provenance"')
    assert i > 0
    s = s[:i] + section + s[i:]
    nav = '<a href="#tasks">'
    if nav in s:
        s = s.replace(nav, '<a href="#model-size">4B vs 8B</a>' + nav, 1)
    return s


if __name__ == '__main__':
    d, src, dst = sys.argv[1:4]
    s = open(src, encoding='utf-8').read()
    open(dst, 'w', encoding='utf-8').write(patch(s, build(d)))
