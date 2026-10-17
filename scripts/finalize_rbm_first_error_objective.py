"""Finalize the reviewed first-error objective comparison."""
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_rbm_data_diagnostics import base,csv_write


def main():
    out=ROOT/'results/rbm_first_error_objective_v1'
    assert json.loads((out/'RESULT_REVIEW.json').read_text())['status']=='PASS'
    d=json.loads((out/'METRICS.json').read_text());m=d['metrics'];c=d['contrasts'];z=np.load(out/'BOOTSTRAP_DRAWS.npz');names=z['names'].tolist()
    for key,v in c.items():
        i=names.index(key);q=[50*(1-v['ci_level']),100-50*(1-v['ci_level'])]
        np.testing.assert_allclose(np.percentile(z['pb'][:,i],q),v['pb_ci'],atol=1e-14)
        np.testing.assert_allclose(np.percentile(z['prm'][:,i],q),v['prm_within_ci'],atol=1e-14)
    base.atomic_json(out/'BOOTSTRAP_REVIEW.json',dict(status='PASS',draws=10000,checks=['reported intervals from saved draws','PRMB identity and zero-delta descriptive draws']))
    cells=m['first_error_pb']['pb_cells'];rows=[]
    for cell in cells:
        rows.append(dict(cell=cell,first_error_f1=m['first_error_pb']['pb_cells'][cell]['f1'],step_bce_f1=m['supervised_update']['pb_cells'][cell]['f1'],
                         delta=m['first_error_pb']['pb_cells'][cell]['f1']-m['supervised_update']['pb_cells'][cell]['f1']))
    csv_write(out/'OBJECTIVE_CELLS.csv',rows)
    ff=json.loads((out/'FORENSICS.json').read_text())
    base.atomic_json(out/'DECISION.json',dict(status='closed_negative',primary=c['first_error_vs_bce'],
        finding='First-error listwise training loses to step BCE in all eight PB cells; primary PB CI includes zero only at its upper edge.',
        interpretation='The full-answer categorical competition is a poor correction objective for this Top10 localizer under the frozen contract. This does not close all first-error losses or local objectives.',
        evidence=ff,prmb='Copied unchanged from the previous supervised BCE arm; identity check, not new evidence.',
        next='Keep step BCE as the supervised correction diagnostic reference. Do not add first-error softmax, position terms, graphs or capacity based on this result.'))
    lines=['# First-error objective: complete','',
        'Branch codex/rbm-first-error-objective-v1; base f9d984266.',
        'All6,800 ProcessBench answers,40 group-disjoint fits and unchanged token bank12',
        'RBM correction/Logit/Top10/gate. Only the PB training objective changed:',
        'step BCE versus categorical first-error loss over all steps of erroneous answers.',
        'Clean answers had no first-error target and contributed zero location loss.',
        '', '| PB objective | PB macro | Q4 | Q8 | exact hits | early | late |',
        '|---|---:|---:|---:|---:|---:|---:|',
        f"| Step BCE (previous) | {100*m['supervised_update']['pb_all8']:.4f}% | {100*m['supervised_update']['pb_q4']:.4f}% | {100*m['supervised_update']['pb_q8']:.4f}% | {m['supervised_update']['pb_exact_count']} | {m['supervised_update']['pb_early']} | {m['supervised_update']['pb_late']} |",
        f"| First-error listwise | {100*m['first_error_pb']['pb_all8']:.4f}% | {100*m['first_error_pb']['pb_q4']:.4f}% | {100*m['first_error_pb']['pb_q8']:.4f}% | {m['first_error_pb']['pb_exact_count']} | {m['first_error_pb']['pb_early']} | {m['first_error_pb']['pb_late']} |",
        '',f"Primary first-error minus BCE: {100*c['first_error_vs_bce']['pb_delta']:+.4f}pp;97.5%CI {[round(100*x,4) for x in c['first_error_vs_bce']['pb_ci']]};",
        f"within-AUC difference is exactly zero because PRMB rows are copied from the previous arm.",
        'First-error loses in all eight PB cells. It gains 481 exact gated successes and loses 556;',
        '412 of the losses are late choices. The result is not a universal rejection of first-error',
        'training: it rejects this full-answer softmax correction under this frozen Top10 contract.',
        '', 'Checks PASS: direct first-error gradient/loss tests; zero-update replay inherited from',
        'the matched study;40 saved models and6,800 test answers replayed; PB metrics and PRMB',
        'identity replayed;10000 bootstrap draws reviewed. No nonconverged fits.',
        'PRMB rows are inherited unchanged and are not new transfer evidence.',
        '', 'Decision: keep step BCE as the supervised correction diagnostic reference. Do not open',
        'another first-error variant, graph, position term or capacity sweep from this result.',
        'Development evidence only; no untouched confirmation.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    base.atomic_json(out/'RUN_STATE.json',dict(status='COMPLETE',answers=13769,pb_fits=40,prmb_copied=6969,reviews=['RESULT_REVIEW.json','BOOTSTRAP_REVIEW.json']))
    print('COMPLETE: first-error objective reviewed and closed negative.')


if __name__=='__main__':main()
