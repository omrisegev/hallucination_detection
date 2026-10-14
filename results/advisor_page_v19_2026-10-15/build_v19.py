"""v19 = v18 with slides 12-13 (sections s12, s13) filled with the same-checkpoint L-SML PRM results.

Steps (run from this directory, PYTHONUTF8=1):
  python derive_prm_update.py   # base/PRM_UPDATE.json from sources/eval_slide12 (AIRCC job 284086)
  python patch_base.py          # base/build_base.py = unchanged v17 builder + slide 12/13 renderers and notes
  (cd base && python build_base.py)
  python build_v19.py
Every other section is copied byte-for-byte from the pinned v18 HTML. Only s12/s13, their SLIDE_MAP
records, the version labels and the regenerated speaker-notes file change.
"""
from pathlib import Path
import hashlib, json, re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V18 = ROOT / 'results/advisor_page_v18_2026-10-15'
BASE = HERE / 'base'
v18_path = V18 / 'published_v18_2026-10-15.html'
v18_hash = hashlib.sha256(v18_path.read_bytes()).hexdigest()
assert v18_hash == '931e68a61d943a265e959365696cba1b02e6440ea57097158bf205c72e21535d', v18_hash
page = v18_path.read_text(encoding='utf8')
base = (BASE / 'published_v17_2026-10-15.html').read_text(encoding='utf8')
U = json.loads((BASE / 'PRM_UPDATE.json').read_text(encoding='utf8'))
assert U['boosting']['status'] == U['generalization']['status'] == 'complete'


def section(ident, html_text):
    found = re.findall(r'<section class="slide[^"]*" id="' + re.escape(ident) + r'".*?</section>', html_text, re.S)
    assert len(found) == 1, (ident, len(found))
    return found[0]


records = json.loads((V18 / 'SLIDE_MAP.json').read_text(encoding='utf8'))
base_records = {r['id']: r for r in json.loads((BASE / 'SLIDE_MAP.json').read_text(encoding='utf8'))}
for ident in ('s12', 's13'):
    new = section(ident, base).replace('on slide 11', 'on slide 13')      # v18 renumbering of the external slide
    old = section(ident, page)
    page = page.replace(old, new, 1)
    for r in records:
        if r['id'] == ident:
            for k in ('notes', 'question', 'limit', 'sources', 'old', 'title'):
                r[k] = base_records[ident][k].replace('on slide 11', 'on slide 13')
assert 'RESERVED FOR CLAUDE' not in page and 'PRM model runs in progress' not in page

for a, b in [('Advisor meeting · 15 Oct 2026 · v18 · restored visuals', 'Advisor meeting · 15 Oct 2026 · v19 · PRM results'),
             ('Advisor meeting · v18', 'Advisor meeting · v19')]:
    assert a in page, a
    page = page.replace(a, b)

destination = HERE / 'published_v19_2026-10-15.html'
destination.write_text(page, encoding='utf8', newline='\n')
(HERE / 'SLIDE_MAP.json').write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n', encoding='utf8', newline='\n')
book = '# v19 — הערות דובר ומיפוי תכנים\n\nזהה ל־v18, חוץ משקופיות 14–15 (תוצאות PRM). 34 דקות תוכן + 6 דקות דיון.\n\n'
for r in records:
    book += (f'## {"נספח " if r["appendix"] else ""}{r["number"]} — {r["title"]}\n\nמקור: {r["old"]}. זמן: {r["minutes"]} דקות.\n\n'
             f'{r["notes"]}\n\n**איך קוראים:** {r["question"]}\n\n**גבול המסקנה:** {r["limit"]}\n\nמקורות: {r["sources"]}\n\n')
(HERE / 'SPEAKER_NOTES_HE.md').write_text(book.rstrip() + '\n', encoding='utf8', newline='\n')

ids18 = re.findall(r'<section class="slide[^"]*" id="([^"]+)"', v18_path.read_text(encoding='utf8'))
unchanged = [i for i in ids18 if i not in ('s12', 's13')]
v18_text = v18_path.read_text(encoding='utf8')
assert all(section(i, page) == section(i, v18_text) for i in unchanged)
receipt = {'html': destination.name, 'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
           'v18_html_sha256': v18_hash, 'changed_sections': ['s12', 's13'],
           'unchanged_sections_byte_identical_to_v18': unchanged,
           'prm_update_sha256': hashlib.sha256((BASE / 'PRM_UPDATE.json').read_bytes()).hexdigest(),
           'results_source': 'AIRCC job 284086, results/prm_boost_gen_v1/eval_slide12 (copied to sources/eval_slide12)',
           'figure_note': 's12/s13 charts were rendered with local matplotlib 3.10.1; other figures are v18 originals',
           'main_slides': 16, 'appendix_slides': 7, 'minutes': sum(r['minutes'] for r in records if not r['appendix'])}
(HERE / 'BUILD_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf8', newline='\n')
print('Built v19:', receipt['sha256'][:12], 'changed', receipt['changed_sections'], 'minutes', receipt['minutes'])
