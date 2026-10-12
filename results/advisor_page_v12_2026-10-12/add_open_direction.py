"""Add the "Open direction: how many steps to flag per answer" section to the v11 page (v12).

Usage: python3 add_open_direction.py <published_v11.html> <out_v12.html>

Follows PAGE_REVIEW_TASKS_2026-10-11.md @ cbb498e (Omri's decision 2026-10-12): one short
section, existing numbers only, no new experiments, no arm or quota tables in the main flow.
- Inserts the section before #tasks and adds a nav link.
- Moves the MoM quota section (#mom-quota) after #remaining-work, as supporting material.
- Updates the notice, the footer and one stale "not pushed" bullet.
"""
import sys

GH = 'https://github.com/omrisegev/hallucination_detection/blob/'
CEIL = GH + '07cb7a45fb01ab62ade51fa5e59717c1f27fd8ff/results/advisor_completion_2026-10-11/controls/DIAGNOSTIC_CEILINGS.json'
COUNT = GH + '17aa1e473cf189ad9b2d76eb277e64d3ba3bd447/results/count_structure_check_v1/README.md'
AD = GH + '6a0b870/results/label_free_error_count_v1_2026-10-12/FINDINGS_HE.md'
RULES = GH + '07cb7a45fb01ab62ade51fa5e59717c1f27fd8ff/results/advisor_completion_2026-10-11/controls/EXACT_DECISION_RULES.csv'


def a(href, text):
    return f'<a href="{href}" target="_blank" rel="noopener">{text}</a>'


ROWS = [
    ('Q80 · frozen rule', '65.4', '—', 'no'),
    ('Best global cutoff', '65.7', '+0.3', 'yes'),
    ('Constant 2 flags per answer (TOP_2, post hoc)', '67.9', '+2.5', 'post hoc choice'),
    ('Count from error type only', '68.9', '+3.5', 'yes'),
    ('True count per answer', '71.7', '+6.3', 'yes'),
    ('Perfect ranking at our own counts', '81.8', '+16.4', 'yes'),
]

SECTION = (
    '<section id="open-direction"><div class="eyebrow">OPEN DIRECTION · FOR DISCUSSION</div>'
    '<h2>Open direction: how many steps to flag per answer</h2>'
    '<p>We show this as a research direction, not as a result. This section uses only saved analyses; '
    'we did not run new experiments for it.</p>'
    '<h3>1 · Why this direction</h3>'
    '<div class="scroll"><table><caption>N16, same fused ranking (L-SML / IU-PCR); only the number of flags per answer changes. '
    'Label-oracle rows are ceilings, not methods, and the gains do not add.</caption>'
    '<thead><tr><th>Operating point</th><th>PRMScore × 100</th><th>Δ vs Q80</th><th>Uses labels</th></tr></thead><tbody>'
    + ''.join(f'<tr><th>{r[0]}</th><td>{r[1]}</td><td>{r[2]}</td><td>{r[3]}</td></tr>' for r in ROWS)
    + '</tbody></table></div>'
    '<p>A better global cutoff gives almost nothing. The true count per answer gives +6.3 with the same ranking. '
    'Perfect ranking gives more (+16.4), but the LR gap shows that it needs information our label-free methods do not find today. '
    'The count is the part with a large gap that does not require a change to the ranking, which is already close to PRM-7B.</p>'
    '<p class="source">Source: ' + a(CEIL, 'controls/DIAGNOSTIC_CEILINGS.json') + ' (commit 07cb7a4) · '
    + a(COUNT, 'results/count_structure_check_v1') + ' (commit 17aa1e4) · 6,211 noncontrol answers</p>'
    '<h3>2 · What we already tried</h3><ul class="remaining">'
    '<li>Registered rules (R2_COUNT, entropy gate): false flags on clean answers go from 99.5% to 58.6%, '
    'but PRMScore does not improve (R2_COUNT 65.33 vs Q80 65.38). ' + a(RULES, 'EXACT_DECISION_RULES.csv') + '</li>'
    '<li>MoM on the L-SML group votes, summed posterior as the count: 64.2, below Q80. Details are in the supporting section below.</li>'
    '<li>Codex label-free count estimators A–D: no arm met the combined success gate. Arm A failed in 4 of 5 N16 folds '
    '(1,269 of 6,211 answers covered). The best arm (LOCAL_MOM_MEAN) loses 0.48 PRMScore points against Q80. '
    + a(AD, 'codex/label-free-error-count-v1 @ 6a0b870') + '</li></ul>'
    '<p><b>What we learned:</b></p><ul class="remaining">'
    '<li>The MoM error rate follows the vote quota, not the data: quotas of 10% / 20% / 30% give 13.6% / 22.9% / 32.6%.</li>'
    '<li>In PRMBench, the number of errors per answer almost does not depend on length (mean 2.12; Spearman with step count 0.09). '
    'Q80 and all tested estimators give counts that grow with length. This is a simple explanation for why they failed.</li>'
    '<li>Two flags per answer give 67.9, above Q80 and PRM-7B. <b>This is not a method:</b> TOP_2 is a post hoc choice and matches how '
    'PRMBench inserts errors. It shows that the correct count is small, almost constant, and depends mainly on the error type.</li></ul>'
    '<h3>3 · How to continue (proposal for discussion)</h3><ol class="remaining">'
    '<li>First check if the information exists: does anything in the fused score inside an answer (gap between the top two steps, '
    'number of outstanding steps) predict the error count beyond length? Compare to a supervised baseline as a ceiling. '
    'If that also fails, move the effort to ranking.</li>'
    '<li>If the information exists: an estimator on the continuous score (for example a mixture model inside each answer), '
    'not on votes after a threshold.</li>'
    '<li>Separate two problems: how many errors a wrong answer has (what PRMScore rewards), and when an answer is clean '
    '(what matters for ProcessBench and the clean controls; PRMScore rewards it very little).</li></ol>'
    '<ul class="takeaways"><span class="kt">Key points</span>'
    '<li>The global threshold is not the problem: the best global cut adds only 0.3 points.</li>'
    '<li>The per-answer count is worth up to 6.3 points with the same ranking, but all tested label-free estimators failed.</li>'
    '<li>PRMBench errors per answer are few and almost independent of length; length-proportional rules cannot match this.</li></ul>'
    '<details><summary>Speaker note · עברית</summary><p lang="he" dir="rtl">את בחירת מספר הסימונים בכל תשובה אנחנו מציגים ככיוון מחקר פתוח, '
    'לא כתוצאה. הדירוג כבר קרוב ל-PRM, והסף הגלובלי כמעט לא משנה. הכמות הנכונה בכל תשובה שווה עד 6.3 נקודות, '
    'אבל כל האומדים בלי תוויות שניסינו נכשלו, כי הם נותנים כמות שגדלה עם האורך, וב-PRMBench מספר השגיאות כמעט קבוע.</p></details>'
    '</section>'
)

NOTICE_OLD = 'Version 11, 12 October 2026.'
NOTICE_NEW = ('Version 12, 12 October 2026. Same as version 11, plus the section "Open direction: how many steps to flag per answer" '
              '(existing results only; no new experiments). Version 11:')
FOOTER_OLD = '<footer><b>Evidence state:</b> v11 analysis'
FOOTER_NEW = ('<footer><b>Evidence state:</b> v12 adds the open-direction section from commits 17aa1e4 and 6a0b870. v11 analysis')
STALE = ('GitHub push and Claude Artifact replacement were not executed in this local implementation; '
         'Claude reachability remains unverified.')
STALE_NEW = 'The v11 package is on GitHub (commit 07cb7a4) and this page is its published version.'


def once(s, old, new):
    assert s.count(old) == 1, old
    return s.replace(old, new, 1)


def section(s, sid):
    i = s.index(f'<section id="{sid}">')
    j = s.index('</section>', i) + len('</section>')
    return i, j


def patch(s):
    assert 'id="open-direction"' not in s
    i, j = section(s, 'mom-quota')
    mom = s[i:j].replace('<div class="eyebrow">COMPLETED ANALYSIS</div>',
                         '<div class="eyebrow">SUPPORTING ANALYSIS · OPEN DIRECTION</div>', 1)
    s = s[:i] + s[j:]
    s = once(s, '<section id="tasks">', SECTION + '<section id="tasks">')
    _, j = section(s, 'remaining-work')
    s = s[:j] + mom + s[j:]
    s = once(s, '<a href="#tasks">', '<a href="#open-direction">Open direction</a><a href="#tasks">')
    s = once(s, NOTICE_OLD, NOTICE_NEW)
    s = once(s, FOOTER_OLD, FOOTER_NEW)
    s = once(s, STALE, STALE_NEW)
    return s


if __name__ == '__main__':
    src, dst = sys.argv[1:3]
    s = open(src, encoding='utf-8').read()
    open(dst, 'w', encoding='utf-8').write(patch(s))
