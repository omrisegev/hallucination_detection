"""Add takeaway bullets to the Codex advisor v11 page.

Usage: python3 add_v11_takeaways.py <in_html> <out_html>

Each figure gets 2-3 bullets right after its <figcaption>. A few key tables get a
"Key points" box. All numbers come from the v11 package files; see
/mnt/project-files/advisor_page/V11_REVIEW_2026-10-12.md for the source of each number.
"""
import sys

CSS = ('.takeaways{background:#f3f7fb;border-left:3px solid #1768ac;border-radius:6px;'
       'margin:10px 0 18px;padding:10px 16px 10px 30px;font-size:14.5px;line-height:1.55;'
       'color:#223548;max-width:1000px}.takeaways li{margin:3px 0}'
       '.takeaways .kt{display:block;margin-left:-14px;font-size:11px;letter-spacing:1.2px;'
       'font-weight:750;color:#1768ac;text-transform:uppercase}')

# Figure bullets, keyed by the start of the figure caption text.
FIGS = {
    'Actual category scores for three evaluated methods': [
        'We are best in Simplicity: NR 70.0 and NCL 68.0, against 49.0 and 55.0 for PRM-7B. No row in paper Table 3 is higher.',
        'PRM-7B is better in Soundness (71.0 vs 66.9) and in Sensitivity (75.5 vs 69.3).',
        'Our MS is low (88.4 vs 99.9), because Q80 flags at least one step in almost every clean answer.',
    ],
    'Digit-free banks': [
        'LR is above both label-free methods in every bank.',
        'The AUC gap grows with bank size, from 0.025 (N11) to 0.063 (N40), and is 0.055 at N53.',
    ],
    'Matched banks with three digit channels': [
        'Adding the three digit channels makes the gap smaller in all five matched pairs, for example N13 0.026 \u2192 N16 0.021.',
        'N16 has the smallest gap of all ten banks (0.021). The new group-strict fits repeat it: LR 0.802 vs 0.801 historical.',
    ],
    'New OOF gap across all ten banks': [
        'LR leads on AUC in all ten banks. The gap grows with bank size (r = 0.89), but it is largest at N40 (0.063), not N56 (0.051).',
        'On top1 the result is opposite up to N25: at N16, L-SML leads LR by 2.0 points [1.0, 3.0].',
    ],
    'N16 and N56 crossed diagnostic': [
        'LR does best on its own half: 0.797 at N16, against 0.788 / 0.791 on the label-free halves.',
        'On the same LR half, label-free weighting stays well below LR: 0.774 (L-SML) against 0.797.',
        'The LR half raises L-SML AUC at N16 (+1.0) but lowers its top1 (−4.5) and lowers AUC at N56 (−0.6). It is not a label-free fix.',
    ],
    'N16 common-scale signed coefficients': [
        'LR puts +0.44 and −0.44 on two digit channels that are almost copies (r = 0.9998). The two weights cancel.',
        'All other LR weights are 0.025 or less.',
        'L-SML and IU-PCR spread the weight evenly (largest about 0.10).',
    ],
    'N56 common-scale signed coefficients': [
        'LR keeps the same digit contrast (+0.30 / −0.31).',
        'LR has 21 negative weights of 56; IU-PCR has none.',
    ],
    'Weight mass by feature family': [
        'At N16, LR puts 89% of its absolute weight on the digit family; L-SML puts 15% there.',
        'L-SML and IU-PCR put the most weight on uncertainty (F1): 48% and 44%.',
        'Weight mass shows where the coefficients are, not how much each family predicts. See the omission figure.',
    ],
    'Family omission with paired intervals': [
        'At N16, removing uncertainty (F1) or surprise (F2) costs about 1.6 AUC points; removing digits costs 1.0.',
        'Removing only one channel of the digit pair costs nothing, because the pair works as a contrast.',
        'At N56, F1 still costs the most (−1.55). Rhythm (F5) costs almost nothing in both banks.',
    ],
    'N16 weight agreement': [
        'L-SML and IU-PCR weights agree with each other (cosine 0.96).',
        'They do not agree with LR (cosine about 0.03). The two digit points at ±0.44 control the LR axis.',
    ],
    'N56 weight agreement': [
        'Agreement with LR is even lower at N56 (cosine about 0.02).',
        'L-SML and IU-PCR still agree with each other, but less than at N16 (cosine 0.74).',
    ],
    'Four-way top1 outcomes by error type': [
        'L-SML alone finds the error in 485 answers that LR misses; LR alone finds 366. Net +119 for L-SML.',
        'We win most on redundancy (90 vs 27), domain (79 vs 32) and circular (77 vs 42).',
        'LR wins on confidence (60 vs 28) and deception (75 vs 58).',
    ],
    'Matched AUC controls': [
        'On all answers, position matching removes PRM’s AUC lead (+2.2 → L-SML +1.0 [−0.2, 2.2]) and our lead over the plain average (+0.8 → −0.3 [−0.8, 0.2]). The matched set is 4,691 of 6,030 answers.',
        'On redundancy, our lead over PRM stays at about +18 to +20 AUC points under every control.',
        'On circular, our lead shrinks to +4.3 points, and the interval crosses zero.',
    ],
    'Budget curves and oracle diagnostics': [
        'At the same flag rate, PRM’s curve is above ours: 68.3 vs 65.7 at about 18% of steps flagged.',
        'Our Q80 point (20%) is almost at the top of our curve. A better global threshold adds only 0.3 points.',
        'PRM’s native 0.5 threshold flags only 8.7% of steps. That is why its default score equals ours.',
    ],
    '16 × 16 Pearson correlations': [
        'Features correlate more on error steps than on valid steps: mean |r| 0.30 vs 0.24.',
        'The largest change is chosen_surprisal with energy_level: 0.50 on error steps vs 0.17 on valid steps.',
        'Most pairs change by less than 0.1. Small changes do not prove conditional independence.',
    ],
    'All 56 channels plus step index': [
        'No channel wins every type. Uncertainty channels lead redundancy (0.84), domain (0.81) and circular (0.77). digit_spread leads deception (0.73).',
        'Step index alone reaches 0.87 on confidence, more than any channel. It is near chance on domain (0.54) and circular (0.57).',
        'One channel (rolling spectral entropy) is below 0.5 in all eight types, so its label-free sign is wrong. realized_drv no longer leads any type.',
    ],
}

# Key-point boxes after a table, keyed by a unique text that ends the anchor element.
TABLES = [
    ('</a> · 5,000 paired resamples; seed 20261011', [
        'At the default rules, our PRMScore equals PRM-7B (difference −0.08, CI [−0.77, +0.64]).',
        'PRM-7B ranks better: at the same flag budget (about 18%), PRM gets 68.3 and L-SML 65.7.',
        'Our top-ranked step is an error more often than PRM’s: 59.9% vs 57.8%.',
    ]),
    ('</a> · N16; 6,030 AUC-eligible answers', [
        'The prefix fix lowers L-SML AUC by 1.6 points; the interval excludes zero.',
        'The decline is larger without step 0 (−2.2), so it is not a first-token artifact.',
        'mask1 changes AUC by less than 0.1 point; the interval includes zero.',
    ]),
    ('</a> · 6,969 answers including controls; PRMScore excludes', [
        'A better global threshold adds only 0.3 points for us, but 2.9 points for PRM-7B.',
        'The true error count per answer adds 6.3 points; perfect ranking at our own counts adds 16.4.',
        'So we must improve both the number of flags in each answer and the ranking.',
    ]),
    ('</a> · 6,211 noncontrols / 83,371 steps; clean-rate', [
        'R2_COUNT lowers false flags on clean answers from 99.5% to 58.6%, for only −0.05 PRMScore.',
        'ENTROPY_GATE_Q80 raises MS from 88.4 to 97.6 and removes the count bias, for −0.5 PRMScore.',
        'PRM-7B falsely flags 19.0% of clean answers. R2_COUNT closes about half of our gap to it.',
    ]),
    ('</a> · 94,203 steps / 6,969 answers', [
        'All estimators give too high an error rate. Group MoM gives 22.9%; the truth is 14.0%.',
        'Each vote marks 23.0% of steps by construction, so the estimate follows the vote rate, not the error rate (our inference).',
    ]),
    ('</a> · 6,211 pooled answers; 758 synthetic controls', [
        'The MoM quota lowers PRMScore to 64.2 (Q80: 65.4).',
        'It still flags 99.9% of clean answers (Q80: 99.5%).',
    ]),
    ('</a> · Donor 3464 answers / 353 groups', [
        'On the held-out half, N16 IU-PCR keeps AUC 0.786 and top1 60.1%.',
        'This is internal validation on data that we already looked at. "PASS" in these tables means that the fit ran.',
    ]),
]


def box(items, title='Key points'):
    return ('<ul class="takeaways"><span class="kt">' + title + '</span>'
            + ''.join('<li>' + t + '</li>' for t in items) + '</ul>')


def patch(s):
    assert 'class="takeaways"' not in s, 'already patched'
    s = s.replace('</style>', CSS + '</style>', 1)
    for key, items in FIGS.items():
        i = s.find('<figcaption>' + key)
        assert i >= 0, key
        j = s.find('</figure>', i) + len('</figure>')
        s = s[:j] + box(items, 'What this figure shows') + s[j:]
    for key, items in TABLES:
        i = s.find(key)
        assert i >= 0, key
        j = s.find('</p>', i) + len('</p>')
        s = s[:j] + box(items) + s[j:]
    return s


if __name__ == '__main__':
    src, dst = sys.argv[1:3]
    s = open(src, encoding='utf-8').read()
    open(dst, 'w', encoding='utf-8').write(patch(s))
