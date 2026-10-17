"""Audit frozen original RBM covariance on every answer, without refitting/labels."""
import io,json,sqlite3,sys,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_direct_probability_temporal import old,source_specs
from spectral_utils.moment_rbm_fusion import representation
from spectral_utils.direct_probability_fusion import zscore_columns
from spectral_utils.rbm_diagonal_variance import covariance_diagnostics
OUT=ROOT/'results/rbm_covariance_assumptions_v1'


def main():
    source=ROOT.parents[1];old.configure_source_root(source);OUT.mkdir(parents=True,exist_ok=True)
    previous=source/'.worktrees/higher-moment-fusion-v1/results/higher_moment_fusion_v1'
    assert json.loads((previous/'RUN_STATE.json').read_text())['review']=='PASS'
    rec=json.loads((old.BENCH/'evaluation/JOINED.json').read_text())['records']
    con=sqlite3.connect('file:'+str(previous/'CHECKPOINT.sqlite')+'?mode=ro',uri=True)
    findings=[];t=time.perf_counter()
    for cell,path,kind,dataset in source_specs():
        rawrows=old._source_row_map(old.load_pickle(path),kind=kind,dataset=dataset)
        for i,r in enumerate(rec):
            if r['cell']!=cell:continue
            payload,info=con.execute('SELECT payload,info FROM answers WHERE idx=?',(i,)).fetchone()
            diag=json.loads(info);assert diag['uid']==r['uid']
            row=rawrows[r['row_id']]
            Z,_,_,_=zscore_columns(representation(old._topk_payload(row)['logprobs'],row['token_spilled_energies']))
            with np.load(io.BytesIO(payload),allow_pickle=False) as z:
                state={key:z['d3__rbm::'+key].copy() for key in ('a','w','b')}
            findings.append(dict(uid=r['uid'],cell=cell,**covariance_diagnostics(Z,state)))
        del rawrows
        print(cell,len(findings),round(time.perf_counter()-t,1),flush=True)
    con.close();assert len(findings)==13769
    keys=('variance_rmse','mean_rmse','covariance_relative_error','offdiag_relative_error')
    summary={key:np.quantile([x[key] for x in findings],[0,.25,.5,.75,1]).tolist() for key in keys}
    variances=np.concatenate([x['predicted_variance'] for x in findings])
    summary['predicted_variance_quantiles']=np.quantile(variances,[0,.25,.5,.75,1]).tolist()
    summary['observed_variance_max_abs_error_from_one']=max(np.max(np.abs(np.asarray(x['observed_variance'])-1)) for x in findings)
    inputs={str(previous/'CHECKPOINT.sqlite'):old.sha256_file(previous/'CHECKPOINT.sqlite'),
            str(old.BENCH/'evaluation/JOINED.json'):old.sha256_file(old.BENCH/'evaluation/JOINED.json')}
    (OUT/'AUDIT.json').write_text(json.dumps(dict(status='COMPLETE',n_answers=len(findings),scope='Descriptive full-population assumption audit; no labels, refits, p-values or causal claim.',summary=summary,inputs=inputs,records=findings))+'\n',encoding='utf8')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
