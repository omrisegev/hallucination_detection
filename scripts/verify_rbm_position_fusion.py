"""Full score replay and independent normalized-mixture likelihood audit."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import sys
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_rbm_data_diagnostics import base
from spectral_utils.higher_moment_fusion import representation_order


def mixture_loss(X,c,a,w0,b,delta):
    values=np.zeros(len(X))
    for sign in np.unique(c):
        mask=c==sign;x=X[mask];w=w0+sign*delta;s=b+a@w+.5*w@w
        l0=-np.logaddexp(0,s)-.5*np.sum((x-a)**2,axis=1)
        l1=-np.logaddexp(0,-s)-.5*np.sum((x-a-w)**2,axis=1)
        values[mask]=-np.logaddexp(l0,l1)
    return float(values.mean())


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    source=args.source_root.resolve();out=ROOT/'results/rbm_position_fusion_v1_overlap_fix';base.old.configure_source_root(source)
    records=json.loads((base.old.BENCH/'evaluation/JOINED.json').read_text())['records']
    db=source/'.worktrees/higher-moment-fusion-v1/results/higher_moment_fusion_v1/CHECKPOINT.sqlite'
    src=sqlite3.connect(db.as_uri()+'?mode=ro',uri=True)
    con=sqlite3.connect((out/('SMOKE.sqlite' if args.smoke else 'CHECKPOINT.sqlite')).as_uri()+'?mode=ro',uri=True)
    done={i for i, in con.execute('SELECT idx FROM answers')};checked=0;failed=0;errors=[]
    with threadpool_limits(limits=1):
        for cell,path,kind,dataset in base.source_specs():
            indices=[i for i,r in enumerate(records) if r['cell']==cell and i in done]
            if not indices:continue
            rows=base.old._source_row_map(base.old.load_pickle(path),kind=kind,dataset=dataset)
            for i in indices:
                r=records[i];row=rows[r['row_id']];spans=np.asarray(row['step_token_spans'],int)
                sb,si=src.execute('SELECT payload,info FROM answers WHERE idx=?',(i,)).fetchone();si=json.loads(si)
                ob,oi=con.execute('SELECT payload,info FROM answers WHERE idx=?',(i,)).fetchone();oi=json.loads(oi)
                assert si['uid']==oi['uid']==r['uid']
                X=representation_order(np.asarray(base.old._topk_payload(row)['logprobs'],float),row['token_spilled_energies'],6)
                d=si['diagnostics']['d6__rbm'];cols=np.asarray(d['columns']);Z=(X[:,cols]-np.asarray(d['normalization_mean'])[cols])/np.asarray(d['normalization_scale'])[cols]
                signs=np.array([-1. if j<(len(spans)+1)//2 else 1. for j in range(len(spans))])
                seed=int.from_bytes(hashlib.sha256(('rbm-position-v1:'+r['uid']).encode()).digest()[:8],'little')
                perm=np.random.default_rng(seed).permutation(signs)
                np.testing.assert_array_equal(signs,oi['partition']['step_signs']);np.testing.assert_array_equal(perm,oi['partition']['permuted_step_signs'])
                # Independent assignment: each later step owns tokens from its start;
                # gaps follow the previous step; prefix follows the first.
                step=np.zeros(len(Z),int)
                for j,(start,end) in enumerate(spans):step[start:]=j
                counts=[int(np.sum(signs[step]==s)) for s in (-1,1)]
                ridge=.1+len(cols)/max(1,min(counts))
                with np.load(io.BytesIO(sb)) as saved,np.load(io.BytesIO(ob)) as state:
                    a,w,b=saved['d6__rbm::a'],saved['d6__rbm::w'],float(saved['d6__rbm::b'])
                    np.testing.assert_array_equal(a,state['a']);np.testing.assert_array_equal(w,state['w']);assert b==state['b'];assert oi['orientation']==d['orientation']
                    for mode,c in [('position',signs[step]),('shared',np.ones(len(Z))),('permuted',perm[step])]:
                        actual=state[mode+'__max']
                        if mode in oi['failures']:
                            assert np.isnan(actual).all();failed+=1;continue
                        delta=state[mode+'::delta'];diag=oi['diagnostics'][mode]
                        token=d['orientation']*(b+np.sum(Z*(w+c[:,None]*delta),axis=1))
                        expected=np.array([np.mean(np.sort(token[start:end])[-10:]) for start,end in spans])
                        np.testing.assert_allclose(expected,actual,atol=1e-9,rtol=1e-11)
                        nll=mixture_loss(Z,c,a,w,b,delta);initial=mixture_loss(Z,c,a,w,b,np.zeros_like(delta))
                        np.testing.assert_allclose(nll+.5*ridge*delta@delta,diag['objective_final'],atol=1e-8,rtol=1e-10)
                        np.testing.assert_allclose(initial,diag['objective_initial'],atol=1e-8,rtol=1e-10)
                        assert diag['ridge']==ridge
                        if len(spans)==1:np.testing.assert_array_equal(delta,np.zeros_like(delta))
                        errors.append(float(np.max(np.abs(expected-actual))));checked+=1
            print('[review]',cell,checked,flush=True);del rows
    con.close();src.close()
    if not args.smoke:
        assert len(done)==13769
        assert json.loads((out/'RESULT_REVIEW.json').read_text())['status']=='PASS'
    report=dict(status='PASS',answers=len(done),checked_models=checked,failed_models=failed,
        max_score_error=max(errors,default=0),checks=['saved base model identity','fixed step split/permutation','manual sorted top10 score','independent normalized Gaussian mixture objective','same per-answer ridge','full denominator preservation'],
        scope='Independent arithmetic in the same session, not external replication.')
    (out/('SMOKE_REVIEW.json' if args.smoke else 'MODEL_REVIEW.json')).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
