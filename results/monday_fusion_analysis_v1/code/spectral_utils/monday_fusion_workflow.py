"""Continue approved stages after the active LEVEL run, stopping at Omri's choice."""
import json
import time
from spectral_utils.monday_fusion_protocol import WORK,RUNS,dump

def wait_level():
    path=RUNS['level_reduction_v1']/'RUN_STATUS.json'
    while True:
        status=json.loads(path.read_text(encoding='utf8')) if path.exists() else {}
        if status.get('status') in ('CONTENT_COMPLETE_LAYERS_PENDING','EXPERIMENT_COMPLETE_REVIEW_PENDING','COMPLETE'): return
        gate=WORK/'GATE_STATUS.json'
        if gate.exists():
            g=json.loads(gate.read_text(encoding='utf8'))
            if g.get('status')=='STOPPED' and g.get('stage')=='level': raise RuntimeError('LEVEL stopped: '+g['reason'])
        print('Waiting for LEVEL checkpoint: '+str(status.get('completed_cells',0))+'/360',flush=True)
        time.sleep(30)

def continue_workflow(data):
    from spectral_utils.monday_fusion_estimators import run_estimators
    from spectral_utils.monday_fusion_statistics import core_contrasts
    from spectral_utils.monday_fusion_secondary import layer_contrasts
    from spectral_utils.monday_fusion_layers import run_position_decision,run_lr
    from spectral_utils.monday_fusion_figures import render
    from spectral_utils.monday_fusion_report import report
    from spectral_utils.monday_fusion_combined import run_combined
    from spectral_utils.monday_fusion_artifacts import package
    stages=[('estimator_a',lambda:run_estimators(data,'answer')),
            ('estimator_b',lambda:run_estimators(data,'cell')),
            ('level_contrasts_review_corrected',lambda:core_contrasts(data.pop,'level_reduction_v1')),
            ('input_contrasts_review_corrected',lambda:core_contrasts(data.pop,'estimator_input_v1')),
            ('position_decision',lambda:run_position_decision(data)),('lr',lambda:run_lr(data)),
            ('level_secondary',lambda:layer_contrasts(data.pop,'level_reduction_v1')),
            ('input_secondary',lambda:layer_contrasts(data.pop,'estimator_input_v1')),
            ('level_package',lambda:package(data,'level_reduction_v1')),('input_package',lambda:package(data,'estimator_input_v1')),
            ('level_figures',lambda:render('level_reduction_v1')),('input_figures',lambda:render('estimator_input_v1')),
            ('level_report',lambda:report('level_reduction_v1')),('input_report',lambda:report('estimator_input_v1')),
            ('combined',lambda:run_combined(data))]
    ledger=WORK/'WORKFLOW_STATUS.json'
    previous=json.loads(ledger.read_text(encoding='utf8')) if ledger.exists() else {'completed':[]}
    completed=previous.get('completed',[])
    for name,run in stages:
        if name in completed: continue
        dump(ledger,{'status':'RUNNING','stage':name,'completed':completed})
        try: run()
        except Exception as exc:
            dump(ledger,{'status':'STOPPED','stage':name,'completed':completed,'reason':str(exc)})
            raise
        completed.append(name)
        dump(ledger,{'status':'RUNNING','stage':name,'completed':completed})
    dump(ledger,{'status':'AWAITING_OMRI_DECISION_RULE','completed':completed,'independent_review':'CLAUDE_PENDING'})
