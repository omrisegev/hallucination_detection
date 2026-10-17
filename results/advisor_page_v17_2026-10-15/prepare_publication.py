"""Freeze only the additional builder assets needed for the Git reading delivery."""
from pathlib import Path
import hashlib,json,re,shutil

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BASE=ROOT/'results/advisor_completion_2026-10-11'
WALK=ROOT/'docs/reviews/advisor_v16_walkthrough_20261013'

def main():
    prior_hash=hashlib.sha256((HERE/'published_v17_2026-10-15.html').read_bytes()).hexdigest()
    (HERE/'sources/runtime').mkdir(parents=True,exist_ok=True)
    copies=[(BASE/'controls/ALL_BANK_Q80_PRMSCORE.csv',HERE/'sources/ALL_BANK_Q80_PRMSCORE.csv'),
            (WALK/'prefix_token_diagnostic.png',HERE/'sources/prefix_token_diagnostic.png'),
            (ROOT/'spectral_utils/prmbench.py',HERE/'sources/runtime/prmbench.py'),
            (ROOT/'spectral_utils/advisor_controls.py',HERE/'sources/runtime/advisor_controls.py')]
    for source,destination in copies:shutil.copyfile(source,destination)
    draft=(WALK/'advisor_walkthrough_draft.html').read_text()
    match=re.search(r'<div class="box"><span class="kt">What the bug was.*?</details></div>',draft,re.S)
    assert match,'Missing original correction example'
    (HERE/'sources/correction_example.html').write_text(match[0])
    (HERE/'PUBLICATION_INPUTS.json').write_text(json.dumps({'original_html_sha256':prior_hash,
        'copied_builder_sources':[{'from':str(s.relative_to(ROOT)),'to':str(t.relative_to(HERE)),
                                  'sha256':hashlib.sha256(t.read_bytes()).hexdigest()} for s,t in copies],
        'correction_example':{'source':str((WALK/'advisor_walkthrough_draft.html').relative_to(ROOT)),
                              'sha256':hashlib.sha256((HERE/'sources/correction_example.html').read_bytes()).hexdigest()},
        'base':'2cf1209cda253a50b53897fa14c99b4431cb0fcd','branch':'codex/advisor-v17-2026-10-15',
        'excluded':'Raw NPZ/PKL input arrays and local font/runtime caches; see SOURCE_MANIFEST.json'},indent=2)+'\n')
    print('Prepared frozen Git builder assets; original HTML:',prior_hash)

if __name__=='__main__':main()
