"""Record Step390 while preserving prior document bytes."""
from pathlib import Path
import json,html
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/predictor_error_profiles_v1'
def run():
    d=json.loads((OUT/'REVIEW.json').read_text(encoding='utf8'));summary=d['summary_en'].replace('\n','\r\n')+'\r\n'
    for name in ['PROGRESS.md','Research_Directions.md']:
        p=ROOT/name;s=p.read_bytes().decode('utf8');assert not s.startswith('## Step390')
        s=s.replace('LATEST — Step389','PREVIOUS — Step389',1)
        p.write_bytes(('## Step390 — common missed-error profiles\r\n\r\n'+summary+'\r\n'+s).encode('utf8'))
    p=ROOT/'HISTORY.md';s=p.read_bytes().decode('utf8');assert '### Step 390' not in s
    entry='\r\n\r\n### Step 390 — Analyze shared missed errors and plot feature/predictor trajectories [Codex]\r\n\r\n**What**: User requested histograms and full feature/predictor plots, then clarified that the central question is common failures across all available methods. Audit both current combinations and the broader aligned archive; compare gate failures and localization failures separately.\r\n\r\n**Why**: Determine whether repeated fusion attempts miss the same kind of evidence, without assuming another weighting method will recover it.\r\n\r\n**Result**:\r\n'+summary+'\r\n**Files changed**:\r\n- scripts/analyze_predictor_error_profiles.py, analyze_broad_common_misses.py, report_predictor_error_profiles.py and documentation updater.\r\n- docs/experiments/PREDICTOR_ERROR_PROFILES_20260915.md — initial diagnostic contract.\r\n- results/predictor_error_profiles_v1/ — all figures, inventories, profiles, counts, source hashes and interpretation.\r\n- PROGRESS, Research_Directions and central execution report — results and limits.\r\n\r\n---\r\n'
    p.write_bytes((s.rstrip('\r\n')+entry).encode('utf8'))
    p=ROOT/'docs/reviews/temporal_research_execution_2026-09-15.md';s=p.read_bytes().decode('utf8');i=s.index('\n')+1
    text='\n## ההחמצות המשותפות — Step390\n\n'+'\n\n'.join(d['paragraphs_he'])+'\n\n[הדוח והגרפים](../../results/predictor_error_profiles_v1/REPORT.html) · [כל ההחמצות](../../results/predictor_error_profiles_v1/COMMON_MISSES.csv).\n\n'
    p.write_bytes((s[:i]+text+s[i:]).encode('utf8'))
    p=ROOT/'docs/reviews/temporal_research_execution_2026-09-15.html';s=p.read_bytes().decode('utf8');i=s.index('</h1>')+5
    text='<section id="common-misses-step390"><h2>ההחמצות המשותפות — Step390</h2>'+''.join('<p>'+html.escape(t)+'</p>' for t in d['paragraphs_he'])+'<p><a href="../../results/predictor_error_profiles_v1/REPORT.html">הדוח והגרפים</a> · <a href="../../results/predictor_error_profiles_v1/COMMON_MISSES.csv">כל ההחמצות</a></p></section>'
    p.write_bytes((s[:i]+text+s[i:]).encode('utf8'))
if __name__=='__main__':run()
