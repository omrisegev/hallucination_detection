"""Create concise machine-readable and Markdown completion notes after both reviews."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_dufs_moment_selection import OUT,NAMES,BANK_NAMES,atomic_json_retry
from spectral_utils.dufs_moment_selection import METHODS
NAMES['shrinkage__rbm_shrinkage']='RBM with weight shrinkage (saved)'


def main():
    state=json.loads((OUT/'RUN_STATE.json').read_text())
    assert state['status']=='COMPLETE' and state['review']=='PASS'
    audit=json.loads((OUT/'STATE_REVIEW.json').read_text());assert audit['status']=='PASS'
    data=json.loads((OUT/'METRICS.json').read_text());M=data['metrics']
    lines=['# DUFS moment selection before RBM','',
        'Full matched development benchmark. Fusion and selection fit within each answer; the common entropy gate and PRMScore calibration are external.',
        'Same top15 moment bank, same Top10 token readout. No new moment orders or graph hyperparameter search. Both metric and saved-state reviews PASS.','',
        '| Bank | Model | ProcessBench macro F1 (%) | PRMB within-answer AUC | PRMB pooled AUC | PRMScore | Valid answers |',
        '|---|---|---:|---:|---:|---:|---:|']
    for m in METHODS:
        bank,solver=m.split('__');x=M[m]
        lines.append(f"| {BANK_NAMES[bank]} | {'Trained RBM' if solver=='rbm' else 'Before learning'} | {100*x['pb_all8']:.4f} | {x['prm_within']:.6f} | {x['prm_pooled']:.6f} | {x['prmscore_q08']:.6f} | {x['valid_answers']} |")
    lines+=['','## Primary paired comparisons','',
            '10,000 canonical source-group draws; 97.5% intervals for each primary comparison. These intervals do not include earlier research selection.','']
    for name,c in data['contrasts'].items():
        if not c['primary']:continue
        a,b=name.split('_minus_');pb=c['pb_ci'];prm=c['prm_within_ci']
        lines.append(f"- {NAMES[a]} minus {NAMES[b]}: PB {100*c['pb_delta']:+.4f} percentage points [{100*pb[0]:+.4f}, {100*pb[1]:+.4f}]; within-answer AUC {c['prm_within_delta_common']:+.6f} [{prm[0]:+.6f}, {prm[1]:+.6f}]. Exact PB decisions gained {c['gained']}, lost {c['lost']}.")
    lines+=['','## Saved references','', '| Reference | PB macro (%) | Within-answer AUC | PRMScore |','|---|---:|---:|---:|']
    for m in ('ref__entropy','ref__k15__raw','ref__k50__raw','ref__k15__equal','ref__k15__iu','shrinkage__rbm_shrinkage'):
        if m in M:
            x=M[m];lines.append(f"| {NAMES[m]} | {100*x['pb_all8']:.4f} | {x['prm_within']:.6f} | {x['prmscore_q08']:.6f} |")
    lines+=['','Per-cell results: SUMMARY.csv. All metrics, coverage, selected-column frequencies, seed agreement and coefficients: METRICS.json. Gains/losses: ERROR_CASES.json.',
            'History and Mind-the-Gap reference metadata retain their original comparability limits. This does not establish a literature-wide winner or untouched confirmation.',
            'No automatic follow-up experiment. Review whether selection helps relative to both full-bank and matched six-column controls, and whether trained RBM improves over initialization.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf8')
    note='\n\n## DUFS moment selection [Codex] - COMPLETE 2026-09-11\n\n'
    note+='Full13769 answers/eight arms. Metric arithmetic and saved-state review PASS.\n'
    for m in METHODS:
        x=M[m];note+=f"{NAMES[m]}: PB{100*x['pb_all8']:.4f}%, within{x['prm_within']:.6f}, PRMScore{x['prmscore_q08']:.6f}.\n"
    note+='Question: select useful columns before fusion after covariance fitting improved without task gains.\nFixed order6 bank, original6 mean orientation, common entropy q0.3 gate/Top10 readout.\nPrimary DUFS-vs-low-correlation and DUFS-vs-all12 comparisons in METRICS.json;\nno new search or subsequent experiment authorized by this completion.\nEvidence remains full-development, not untouched confirmation.\nArtifacts results/dufs_moment_selection_v1/{REPORT.md,SUMMARY.csv,METRICS.json,STATE_REVIEW.json,RESULT_REVIEW.json}.\n'
    marker=b'## DUFS moment selection [Codex] - COMPLETE 2026-09-11'
    for name in ('HISTORY.md','Research_Directions.md','PROGRESS.md'):
        path=ROOT/name;raw=path.read_bytes()
        if marker in raw:continue
        if name=='PROGRESS.md':
            first,rest=raw.split(b'\n',1);raw=first+b'\n'+note.encode('utf8')+b'\n'+rest
        else:raw+=note.encode('utf8')
        path.write_bytes(raw)
    state['state_review']='PASS';state['report']='REPORT.md';atomic_json_retry(OUT/'RUN_STATE.json',state)
    print('COMPLETE: full benchmark, both reviews and concise results saved.',flush=True)


if __name__=='__main__':main()
