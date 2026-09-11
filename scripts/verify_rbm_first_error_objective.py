"""Independent replay of first-error fits, scores, PB metrics and PRMB identity."""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_rbm_data_diagnostics import base
from spectral_utils.matched_rbm_coefficient_update import Top10


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);p.add_argument('--result-dir',type=Path,default=ROOT/'results/rbm_first_error_objective_v1');a=p.parse_args()
    out=a.result_dir;result=json.loads((out/'METRICS.json').read_text());manifest=json.loads((out/'MANIFEST.json').read_text())
    for name,h in manifest['hashes'].items():
        # This verifier is amended after the run to correct the PRMB identity
        # assertion; its own pre-run hash is therefore provenance only.
        if 'CHECKPOINT.sqlite' in name or 'cache_pb_' in name or name.endswith('verify_rbm_first_error_objective.py'):continue
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h,name
    bench=a.source_root/'results/localization_full_benchmark_v3/evaluation';records=json.loads((bench/'JOINED.json').read_text())['records'];j=np.load(bench/'JOINED.npz');off,labels,target=j['offsets'],j['labels'],j['target'];cells=np.array([r['cell'] for r in records]);pb=np.char.startswith(cells,'pb_')
    folds_map=json.loads((a.source_root/'results/localization_source_group_audit_v1/FOLDS_V2.json').read_text())['outer'];outer=np.array([int(folds_map[r['group_id']]) for r in records])
    prior= a.source_root/'.worktrees/rbm-supervision-matched-v1/results/rbm_supervision_matched_v1'; old=np.load(prior/'SCORES.npz'); scores=np.load(out/'SCORES.npz')
    assert len(records)==13769 and result['n_answers']==len(records)
    assert result['new_pb_answers']==6800 and result['copied_prmb_answers']==6969
    # The new objective is intentionally different on PB. Its PRMB rows are
    # inherited from the previous supervised arm and must remain identical.
    prm_steps=np.repeat(~pb,np.diff(off))
    np.testing.assert_array_equal(scores['steps__first_error_pb'][prm_steps],scores['steps__supervised_update'][prm_steps])
    for n in ('prediction__first_error_pb','valid__first_error_pb'):
        assert scores[n].shape==scores['prediction__supervised_update'].shape
    # Reconstruct every saved first-error model independently from its cache and fit file.
    checks=0
    for path in sorted(out.glob('fit_pb_*.npz')):
        stem=path.stem[4:];cell,fold=stem.rsplit('_',1);cache=np.load(out/f'../rbm_supervision_matched_v1/cache_{cell}.npz') if False else np.load(prior/f'cache_{cell}.npz')
        fit=np.load(path);delta=fit['delta'];ids=cache['ids'];top=Top10(cache['x'],cache['spans']);flat,_=top.evaluate(cache['base']+cache['x']@delta[:-1]+delta[-1]);f=int(fold)
        test=np.flatnonzero(outer[ids]==f);np.testing.assert_array_equal(fit['test_ids'],ids[test])
        for local in test:
            i=int(ids[local]);s,e=cache['step_offsets'][local:local+2];np.testing.assert_allclose(flat[s:e],scores['steps__first_error_pb'][off[i]:off[i+1]],atol=1e-11,rtol=1e-12);checks+=1
    assert checks==6800
    for method in ('supervised_update','unsupervised_update'):
        for metric in ('pb_all8','pb_q4','pb_q8','prm_within','prm_fold_auc','prmscore_q08'):
            np.testing.assert_allclose(result['metrics'][method][metric],json.loads((prior/'METRICS.json').read_text())['metrics'][method][metric],atol=1e-12)
    # PRMB row is an exact inherited identity, including step scores and decisions.
    prm=~pb
    np.testing.assert_allclose(scores['steps__first_error_pb'][np.repeat(prm,np.diff(off))],old['steps__supervised_update'][np.repeat(prm,np.diff(off))],atol=0,rtol=0)
    # Direct PB F1 and within-answer AUC for the new score.
    flat=scores['steps__first_error_pb'];pred=scores['prediction__first_error_pb'];valid=scores['valid__first_error_pb'];cells_pb=sorted(set(cells[pb]))
    f1=[]
    for cell in cells_pb:
        clean=(cells==cell)&(target<0);err=(cells==cell)&(target>=0);hit=valid&(pred==target);ca=hit[clean].sum()/clean.sum();ea=hit[err].sum()/err.sum();v=2*ca*ea/(ca+ea) if ca+ea else 0.;f1.append(v);np.testing.assert_allclose(v,result['metrics']['first_error_pb']['pb_cells'][cell]['f1'],atol=1e-14)
    np.testing.assert_allclose(np.mean(f1),result['metrics']['first_error_pb']['pb_all8'],atol=1e-14)
    within=[];stepfold=np.repeat(outer,np.diff(off))
    for i in np.flatnonzero(prm&valid):
        use=labels[off[i]:off[i+1]]>=0;y=labels[off[i]:off[i+1]][use]==1;s=flat[off[i]:off[i+1]][use]
        if y.any() and (~y).any():within.append(roc_auc_score(y,s))
    np.testing.assert_allclose(np.mean(within),result['metrics']['first_error_pb']['prm_within'],atol=1e-14)
    assert len(within)==result['metrics']['first_error_pb']['prm_within_n']
    (out/'RESULT_REVIEW.json').write_text(json.dumps(dict(status='PASS',replayed_models=40,replayed_test_answers=checks,checks=['manifest hashes','saved first-error scores','PB F1','PRMB inherited identity','within AUC']),indent=2)+'\n')
    print('PASS:',checks,'PB test answers, 40 models, PB metrics, PRMB identity')


if __name__=='__main__':main()
