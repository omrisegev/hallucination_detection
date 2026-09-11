"""Index existing full experiments without dropping RBM or mixing access claims."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_rbm_literature_completion as run


SOURCES=[
 ('moment-rbm-fusion-v1','moment_rbm_fusion_v1_staged/combined','Answer-local moment fusion; includes B3 and reference rows'),
 ('rbm-m3-powers-v1','rbm_m3_powers_v1','Answer-local m3 ablation and raw rank-power RBM'),
 ('higher-moment-fusion-v1','higher_moment_fusion_v1','Answer-local moments through order6; includes initial controls'),
 ('rbm-weight-shrinkage-v1','rbm_weight_shrinkage_v1','Answer-local weight shrinkage'),
 ('rbm-diagonal-variance-v1','rbm_diagonal_variance_v1','Answer-local variance shared across latent states'),
 ('rbm-data-diagnostics-v1','rbm_data_diagnostics_v1','Frozen state diagnosis, old/near readout; no refit'),
 ('rbm-logit-readout-v1','rbm_logit_readout_v1','Frozen models; posterior/logit and maximum/near ablations'),
 ('rbm-position-fusion-v1','rbm_position_fusion_v1_overlap_fix','Answer-local unlabeled position correction and shuffled control'),
 ('rbm-supervised-position-diagnostic-v1','rbm_supervised_position_diagnostic_v1','Supervised other-answer diagnostic on STEP MEANS; not matched RBM input'),
 ('rbm-supervision-matched-v1','rbm_supervision_matched_v1','Unlabeled and labeled other-answer corrections above the frozen answer-local RBM'),
 ('rbm-first-error-objective-v1','rbm_first_error_objective_v1','Supervised other-answer objective comparison; PRMB inherited unchanged'),
 ('varentropy-contribution-fusion-v1','varentropy_contribution_fusion_v1','Answer-local contribution fusion and scalar controls'),
 ('direct-probability-temporal-v3','direct_probability_temporal_v3','Earlier direct-probability temporal/fusion bank; full matched localization'),
 ('deem-b3-probability-moments-v1','deem_b3_probability_moments_v1','Earlier DEEM/B3 adaptations; inspect per-arm input semantics'),
 ('binary-moment-fusion-v1','binary_moment_fusion_v1','Binary detector fusion on moment banks'),
]


def main():
    source=ROOT.parents[1];out=run.PROGRAM;out.mkdir(parents=True,exist_ok=True)
    rows=[];inventory=[]
    for wt,folder,scope in SOURCES:
        directory=source/'.worktrees'/wt/'results'/folder;path=directory/'METRICS.json'
        if not path.exists():
            if wt=='deem-b3-probability-moments-v1':
                inventory.append(dict(experiment=wt,path=str(directory),status='STOPPED_INPUT_SEMANTICS',scope=scope,
                    evidence=str(source/'.worktrees'/wt/'PROGRESS.md'),
                    reason='Only3 smoke rows; user challenged vocabulary probabilities as soft hallucination decisions. Native DEEM full run was not started; later Gaussian RBM is a distinct approach.',
                    unresolved_prior_request='Continuous B3 direct-probability lane has no full result in this artifact; six-moment B3 is complete in the later staged run.'))
            else:inventory.append(dict(experiment=wt,path=str(path),status='MISSING_AT_EXPECTED_PATH',scope=scope))
            continue
        data=json.loads(path.read_text());metrics=data.get('metrics',{})
        if not isinstance(metrics,dict):metrics={}
        n=0
        for method,m in metrics.items():
            if not isinstance(m,dict) or 'pb_all8' not in m or 'prm_within' not in m:continue
            rows.append(dict(experiment=wt,method=method,source_path=str(path),source_sha256=run.base.old.sha256_file(path),
                experiment_scope=scope,readout='near' if '__near' in method or method.endswith('_near') else 'see method/source',
                **{k:m.get(k) for k in run.METRIC_KEYS},prm_fold_auc=m.get('prm_fold_auc'),
                valid_answers=m.get('valid_answers'),prm_within_n=m.get('prm_within_n'),
                comparison_note='Retain experiment/arm contract; reference rows may differ from the source experiment fitting scope.'))
            n+=1
        state=directory/'RUN_STATE.json';state=json.loads(state.read_text()) if state.exists() else {}
        review=directory/'RESULT_REVIEW.json';review=json.loads(review.read_text()) if review.exists() else {}
        inventory.append(dict(experiment=wt,path=str(path),status=state.get('status','STATE_NOT_LOCATED'),
            source_review=review.get('status','REVIEW_NOT_LOCATED'),indexed_rows=n,scope=scope))
    dufs=source/'.worktrees/dufs-moment-selection-v1/results/dufs_moment_selection_v1'
    ds=json.loads((dufs/'RUN_STATE.json').read_text())
    inventory.append(dict(experiment='dufs-moment-selection-v1',path=str(dufs),reported_state=ds,
        live_handle_check_required=True,scope='Select6 of12 FEATURES, then RBM; not token selection. No partial performance ranking.'))
    run.csv_write(out/'PRIOR_METHODS.csv',rows)
    run.base.atomic_json(out/'PRIOR_EXPERIMENTS.json',inventory)
    run.base.atomic_json(out/'REQUIREMENT_LEDGER.json',dict(
        cleanup=dict(status='COMPLETE',audit=str(source/'scratch/cleanup_20260912/verified_duplicates.jsonl'),
                     scope='104 SHA256-identical data copies, originals preserved'),
        dufs=dict(status='PENDING_FULL_REVIEW',path=str(dufs)),
        experiments=[dict(suite=s,path=str(out/s),required=True) for s in run.SUITES],
        diagnostic_correction=dict(path=str(out/'diagnostic_correction/CORRECTED_DIAGNOSTICS.json'),required=True),
        comparison_index=dict(indexed_rows=len(rows),inventory=str(out/'PRIOR_EXPERIMENTS.json'),
            note='An index is not completion of missing historical refits. Never merge incompatible metrics into one claimed leaderboard.'),
        remaining_optional_scope=[
            'More than moment order6 was discussed as optional continuation, not added to this frozen literature-mechanism roster.',
            'No LOCA, Diverging Flows, KalmanNet or new external inference in this registered completion program.',
            'DUFS token/window selection is distinct from the current DUFS feature-selection experiment.'
        ],
        completion_rule='All required full stages and DUFS must have checked outputs, interpretation and documentation; green code tests alone do not complete the goal.'))
    print(json.dumps(dict(indexed_rows=len(rows),experiments=len(inventory),
        missing=[r['experiment'] for r in inventory if r.get('status')=='MISSING_AT_EXPECTED_PATH']),indent=2))


if __name__=='__main__':main()
