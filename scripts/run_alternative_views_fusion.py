"""Frozen seven-view experiment: extraction, answer-local fitting and scoring."""
from pathlib import Path
import sys,json,sqlite3,io,time,gc
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
from spectral_utils.alternative_views_fusion import METHODS,extract,score,shrink_covariance
from spectral_utils.context_training import FeatureBundle
from spectral_utils.temporal_research_features import prefix_innovation
from spectral_utils.predictor_subset_fusion import top10
from spectral_utils.step_evidence_fusion import standardize
from spectral_utils.upcr import upcr_fit_covariance
from spectral_utils.laplacian_upcr import IU_FIT_DEFAULTS
from scripts.run_predictor_subset_study import sha,write
from scripts import run_temporal_research_baseline as base

OUT=ROOT/'results/alternative_views_fusion_v1'
DATA=ROOT/'results/temporal_context_data_v1'
REF=ROOT/'results/digit_fusion_v1'

def run():
    OUT.mkdir(exist_ok=True);start=time.perf_counter()
    bundle=FeatureBundle(DATA,'innovation5');meta=bundle.metadata
    base.evaluator.old.configure_source_root(ROOT.parents[1])
    ids=[{k:r[k] for k in ('uid','cell','row_id')} for r in json.loads((base.evaluator.old.BENCH/'evaluation/JOINED.json').read_text())['records']]
    assert [r['uid'] for r in ids]==[m['uid'] for m in meta]
    sources=list(base.evaluator.source_specs())
    files=[Path(__file__),ROOT/'spectral_utils/alternative_views_fusion.py',ROOT/'docs/experiments/ALTERNATIVE_VIEWS_FUSION_20260915.md',
           ROOT/'scripts/evaluate_alternative_views_fusion.py',DATA/'METADATA.json',DATA/'step_spans.npy',REF/'SCORES_FROZEN.npz']
    manifest=dict(methods=METHODS,code_inputs={p.relative_to(ROOT).as_posix():sha(p) for p in files},
        sources={str(p):dict(bytes=p.stat().st_size,sha256=sha(p)) for _,p,_,_ in sources},
        flow_state_sha256=sha(ROOT/'results/temporal_neural_queue_seed0_v1/RUN_STATE.json'),labels_in_fit=False)
    manifest=json.loads(json.dumps(manifest))
    if (OUT/'MANIFEST.json').exists():assert json.loads((OUT/'MANIFEST.json').read_text())==manifest
    write(OUT/'MANIFEST.json',manifest)
    with np.load(REF/'SCORES_FROZEN.npz') as f:b=f['innovation5'];digit=f['digit025']
    with np.load(REF/'AUXILIARY.npz') as f:rawdigit=f['aux'][:,0]
    cp=sqlite3.connect(OUT/'CHECKPOINT.sqlite');cp.execute('create table if not exists answers(idx INTEGER PRIMARY KEY,payload BLOB,diagnostic TEXT)')
    done={r[0] for r in cp.execute('select idx from answers')}
    rawbank=sqlite3.connect('file:'+(ROOT/'results/temporal_research_baseline_v1/CHECKPOINT.sqlite').as_posix()+'?mode=ro',uri=True)
    for cell,path,kind,ds in sources:
        selected=[i for i,r in enumerate(ids) if r['cell']==cell and i not in done]
        if not selected:continue
        print('[extract]',cell,len(selected),flush=True)
        rows=base.evaluator.old._source_row_map(base.evaluator.old.load_pickle(path),kind=kind,dataset=ds)
        for j,i in enumerate(selected):
            m=meta[i];row=rows[ids[i]['row_id']];sl=slice(m['step_start'],m['step_stop'])
            payload=base.evaluator.old._topk_payload(row);lp=np.asarray(payload['logprobs'],float)
            spans=np.asarray(row['step_token_spans'],int)
            np.testing.assert_array_equal(spans,np.asarray(bundle.spans[sl])-m['offset'])
            new,validation=extract(row['gen_token_ids'],payload['ids'],lp,row['token_spilled_energies'])
            assert len(new)==m['tokens']
            # Independent full-population tail expression and digit reference.
            for k,c in ((15,7),(50,8)):
                np.testing.assert_allclose(new[:,c],np.clip(1-np.exp(lp[:,:k]).sum(1),0,1),atol=1e-15,rtol=0)
            blob=rawbank.execute('select payload from answers where idx=?',(i,)).fetchone()[0]
            with np.load(io.BytesIO(blob),allow_pickle=False) as f:features=np.asarray(f['features'],float)
            bank=np.column_stack((features[:,:4],prefix_innovation(features[:,0])[0]))
            np.testing.assert_allclose(top10(bank,spans).mean(1),b[sl],atol=1e-10,rtol=0)
            scored,diagnostic=score(bank,new,spans,b[sl])
            np.testing.assert_array_equal(scored[:,6],rawdigit[sl])
            np.testing.assert_allclose(scored[:,15],digit[sl],atol=2e-12,rtol=0)
            maxdelta=0.
            if j<3:
                for name,x,groups in [('new7',new[:,:7],np.array([0]*4+[1]*2+[2])),
                                      ('augmented12',np.column_stack((bank,new[:,:7])),np.array([0]*5+[1]*4+[2]*2+[3]))]:
                    z,sd=standardize(x);live=np.array(diagnostic[name]['live']);z=z[:,live];C=np.array(diagnostic[name]['covariance'])
                    for head in ('iu','diag','block'):
                        info=diagnostic[name]['heads'][head]
                        if info['fallback']:continue
                        fit=C if head=='iu' else shrink_covariance(z,C,groups[live],head)[0]
                        w=upcr_fit_covariance(fit,**IU_FIT_DEFAULTS).w
                        if w@C@np.ones(len(w))<0:w=-w
                        got=np.array(info['weights'])[live]
                        np.testing.assert_allclose(got,w,atol=2e-7,rtol=2e-7)
                        maxdelta=max(maxdelta,float(np.max(np.abs(got-w))))
            validation.update(uid=m['uid'],canonical_weight_delta=maxdelta,banks=diagnostic)
            buf=io.BytesIO();np.savez_compressed(buf,scores=scored)
            cp.execute('insert into answers values(?,?,?)',(i,buf.getvalue(),json.dumps(validation)));done.add(i)
            if len(done)%250==0:
                cp.commit();write(OUT/'RUN_STATE.json',dict(status='SCORING',answers=len(done),expected=len(meta),seconds=time.perf_counter()-start))
                print('[scored]',len(done),round(time.perf_counter()-start),flush=True)
        cp.commit();del rows;gc.collect()
    rawbank.close();assert len(done)==len(meta)
    output=np.empty((len(b),len(METHODS)));diagnostics=[]
    for i,blob,d in cp.execute('select idx,payload,diagnostic from answers order by idx'):
        m=meta[i]
        with np.load(io.BytesIO(blob)) as f:output[m['step_start']:m['step_stop']]=f['scores']
        diagnostics.append(json.loads(d))
    cp.close();assert np.isfinite(output).all()
    np.savez_compressed(OUT/'NEW_SCORES.npz',**{n:output[:,j] for j,n in enumerate(METHODS)})
    write(OUT/'DIAGNOSTICS.json',diagnostics)
    write(OUT/'SCORING_AUDIT.json',dict(status='PASS',answers=len(meta),tokens=sum(m['tokens'] for m in meta),
        digit_reference_exact=True,tail_formula_checked_all_tokens=True,baseline_reconstructed_all_answers=True,
        provided_probability_max_delta=max(d['provided_probability_delta'] for d in diagnostics),
        canonical_weight_max_delta=max(d['canonical_weight_delta'] for d in diagnostics),scores_sha256=sha(OUT/'NEW_SCORES.npz')))
    write(OUT/'RUN_STATE.json',dict(status='SCORED_PENDING_EVALUATION',answers=len(meta),scoring_seconds=time.perf_counter()-start))
    print('[complete scoring]',time.perf_counter()-start,flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):run()
