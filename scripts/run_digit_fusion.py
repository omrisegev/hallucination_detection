"""Independent raw-token extraction and full, source-excluded digit fusion."""
from pathlib import Path
import sys,json,sqlite3,io,time,gc
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
from spectral_utils.context_training import FeatureBundle
from spectral_utils.digit_fusion import METHODS,digit_streams,summarize,direct_bank,score_auxiliary
from spectral_utils.temporal_research_features import prefix_innovation
from spectral_utils.predictor_subset_fusion import top10
from spectral_utils.step_evidence_fusion import standardize
from spectral_utils.upcr import upcr_fit_covariance
from spectral_utils.laplacian_upcr import IU_FIT_DEFAULTS
from scripts.run_predictor_subset_study import sha,write,JOBS,key
from scripts import run_temporal_research_baseline as base

OUT=ROOT/'results/digit_fusion_v1'
DATA=ROOT/'results/temporal_context_data_v1'
REF=ROOT/'results/tcn_aligned_predictor_seed0_v1'
MODELS=ROOT/'results/temporal_context_models_v1'


def ro(path):return sqlite3.connect('file:'+path.as_posix()+'?mode=ro',uri=True)


def run():
    OUT.mkdir(exist_ok=True);began=time.perf_counter()
    bundle=FeatureBundle(DATA,'innovation5');meta=bundle.metadata
    base.evaluator.old.configure_source_root(ROOT.parents[1])
    # Read only identity fields from the established benchmark record map.
    ids=[{k:r[k] for k in ('uid','cell','row_id')} for r in json.loads((base.evaluator.old.BENCH/'evaluation/JOINED.json').read_text())['records']]
    assert [r['uid'] for r in ids]==[m['uid'] for m in meta]
    inputs=ROOT.parents[1]/'results/automatic_group_free_phase_a6_s0a_v1/inputs'
    tokenizers={};digits=None
    for p in inputs.glob('qwen*/tokenizer.json'):
        vocab=json.loads(p.read_text(encoding='utf8'))['model']['vocab']
        found=[vocab[str(i)] for i in range(10)]
        if digits is not None and found!=digits:raise ValueError('Tokenizer digit mismatch')
        digits=found;tokenizers[p.parent.name]=dict(sha256=sha(p),digit_ids=found)
    if len(tokenizers)!=2 or digits!=list(range(15,25)):raise ValueError('Missing/mismatched tokenizer evidence')
    sources=list(base.evaluator.source_specs())
    files=[Path(__file__),ROOT/'spectral_utils/digit_fusion.py',ROOT/'spectral_utils/predictor_subset_fusion.py',
           ROOT/'docs/experiments/DIGIT_FUSION_20260915.md',DATA/'METADATA.json',DATA/'step_spans.npy',
           REF/'SCORES_FROZEN.npz',ROOT/'results/temporal_research_baseline_v1/CHECKPOINT.sqlite',
           ROOT/'results/claude_real_checks_v1/DIGIT_DISAGREE_SCORES.npz']
    provenance={}
    for e in JOBS:
        folder=MODELS/('tcn__innovation5__seed0__'+key(e))
        spec=json.loads((folder/'MANIFEST.json').read_text())
        assert spec['excluded_folds']==list(e) and not spec['smoke']
        for name,rows in zip(('training','validation','held'),bundle.split(e)):
            assert set(spec[name+'_groups'])=={meta[i]['group_id'] for i in rows}
        provenance[key(e)]=dict(manifest_sha256=sha(folder/'MANIFEST.json'),scores_sha256=sha(folder/'scoring/STEP_SCORES.npz'))
    manifest=dict(methods=METHODS,tokenizers=tokenizers,source_files={str(p):dict(bytes=p.stat().st_size,sha256=sha(p)) for _,p,_,_ in sources},
        code_inputs={p.relative_to(ROOT).as_posix():sha(p) for p in files},predictor_scores=provenance,
        flow_state_sha256=sha(ROOT/'results/temporal_neural_queue_seed0_v1/RUN_STATE.json'),labels_used_in_fitting=False)
    manifest=json.loads(json.dumps(manifest));mp=OUT/'MANIFEST.json'
    if mp.exists() and json.loads(mp.read_text())!=manifest:raise ValueError('Manifest drift')
    write(mp,manifest)
    with np.load(REF/'SCORES_FROZEN.npz') as f:b=f['innovation5'];tcn=f['tcn__real']
    with np.load(ROOT/'results/claude_real_checks_v1/DIGIT_DISAGREE_SCORES.npz') as f:
        claude={k:f[k] for k in f.files}
    cp=sqlite3.connect(OUT/'EXTRACTED.sqlite')
    cp.execute('create table if not exists answers(idx INTEGER PRIMARY KEY, payload BLOB, diagnostic TEXT)')
    done={r[0] for r in cp.execute('select idx from answers')};rawbank=ro(ROOT/'results/temporal_research_baseline_v1/CHECKPOINT.sqlite')
    maxdelta=0.;maxweights=0.;tokens=0
    for cell,path,kind,ds in sources:
        selected=[i for i,r in enumerate(ids) if r['cell']==cell and i not in done]
        if not selected:continue
        print('[raw]',cell,len(selected),flush=True)
        rows=base.evaluator.old._source_row_map(base.evaluator.old.load_pickle(path),kind=kind,dataset=ds)
        for j,i in enumerate(selected):
            m=meta[i];row=rows[ids[i]['row_id']]
            gen=np.asarray(row['gen_token_ids']);payload=base.evaluator.old._topk_payload(row)
            top=np.asarray(payload['ids']);lp=np.asarray(payload['logprobs']);spans=np.asarray(row['step_token_spans'],int)
            assert len(gen)==len(top)==m['tokens'] and top.shape==lp.shape
            if not np.isfinite(lp).all() or np.any(np.diff(lp,axis=1)>2e-6):raise ValueError('Unsorted or invalid top-k')
            sl=slice(m['step_start'],m['step_stop'])
            np.testing.assert_array_equal(spans,np.asarray(bundle.spans[sl])-m['offset'])
            d,o,null=digit_streams(gen,top[:,0],digits,m['uid'])
            # Independent scalar token predicate over every token.
            scalar=np.fromiter((int(int(g) in digits and int(p) in digits and int(g)!=int(p)) for g,p in zip(gen,top[:,0])),float,count=len(gen))
            np.testing.assert_array_equal(d,scalar)
            aux,count,opp=summarize(d,o,null,spans)
            manual=np.array([min(float(d[a:c].sum()),min(10,c-a))/min(10,c-a) for a,c in spans])
            np.testing.assert_array_equal(aux[:,0],manual);np.testing.assert_array_equal(aux[:,0],claude['aux'][sl])
            blob=rawbank.execute('select payload from answers where idx=?',(i,)).fetchone()[0]
            with np.load(io.BytesIO(blob),allow_pickle=False) as f:features=np.asarray(f['features'],float)
            bank=np.column_stack((features[:,:4],prefix_innovation(features[:,0])[0]))
            np.testing.assert_allclose(top10(bank,spans).mean(1),b[sl],atol=1e-10,rtol=0)
            bank_scores,diag=direct_bank(bank,d,spans,b[sl])
            out=score_auxiliary(b[sl],tcn[sl],aux,bank_scores)
            for k,name in ((1,'steps__innovation5_plus_digit_g0.25'),(2,'steps__innovation5_plus_digit_g1')):
                np.testing.assert_allclose(out[:,k],claude[name][sl],atol=2e-12,rtol=0)
                maxdelta=max(maxdelta,float(np.max(np.abs(out[:,k]-claude[name][sl]))))
            if j<5:
                for dd in diag:
                    if not dd['native']:continue
                    C=np.array(dd['covariance']);ref=upcr_fit_covariance(C,**IU_FIT_DEFAULTS).w
                    if ref@C@np.ones(len(ref))<0:ref=-ref
                    got=np.asarray(dd['weights'])[dd['live']]
                    np.testing.assert_allclose(got,ref,atol=2e-7,rtol=2e-7);maxweights=max(maxweights,float(np.max(np.abs(got-ref))))
            z,_=standardize(np.column_stack((bank,d)));C=z.T@z/len(z)
            diagnostic=dict(uid=m['uid'],tokens=len(d),disagreements=int(d.sum()),provided_digits=int(o.sum()),
                digit_constant=bool(d.std()<=1e-12),banks=diag,covariance6=C.tolist(),claude_max_delta=maxdelta,canonical_max_delta=maxweights)
            buf=io.BytesIO();np.savez_compressed(buf,aux=aux,counts=count,opportunities=opp,bank_scores=bank_scores)
            cp.execute('insert into answers values(?,?,?)',(i,buf.getvalue(),json.dumps(diagnostic)));done.add(i);tokens+=len(d)
            if len(done)%250==0:
                cp.commit();write(OUT/'RUN_STATE.json',dict(status='EXTRACTING',answers=len(done),expected=len(meta)))
        cp.commit();del rows;gc.collect()
    rawbank.close()
    assert len(done)==len(meta)
    aux=np.empty((len(b),4));bank_scores=np.empty_like(aux);counts=np.empty(len(b));opp=np.empty(len(b));diagnostics=[]
    for i,blob,spec in cp.execute('select idx,payload,diagnostic from answers order by idx'):
        m=meta[i];sl=slice(m['step_start'],m['step_stop'])
        with np.load(io.BytesIO(blob),allow_pickle=False) as f:
            aux[sl]=f['aux'];bank_scores[sl]=f['bank_scores'];counts[sl]=f['counts'];opp[sl]=f['opportunities']
        diagnostics.append(json.loads(spec))
    cp.close();np.savez_compressed(OUT/'AUXILIARY.npz',aux=aux,counts=counts,opportunities=opp,bank_scores=bank_scores)
    write(OUT/'DIAGNOSTICS.json',diagnostics)
    for e in JOBS:
        folder=MODELS/('tcn__innovation5__seed0__'+key(e))
        with np.load(folder/'scoring/STEP_SCORES.npz') as f:context=f['real__signed_residual_0.25']
        selected=[i for i,m in enumerate(meta) if m['fold'] in e and (len(e)==1 or not m['cell'].startswith('pb_'))]
        scored=np.full((len(b),len(METHODS)),np.nan)
        for i in selected:
            m=meta[i];sl=slice(m['step_start'],m['step_stop'])
            scored[sl]=score_auxiliary(b[sl],context[sl],aux[sl],bank_scores[sl])
        p=OUT/(key(e)+'.npz');np.savez_compressed(p,scores=scored,ids=selected)
        write(OUT/(key(e)+'_AUDIT.json'),dict(status='PASS',answers=len(selected),scores_sha256=sha(p)))
    write(OUT/'SCORING_AUDIT.json',dict(status='PASS',answers=len(meta),tokens=sum(m['tokens'] for m in meta),
        digit_auxiliary_exact=True,claude_score_max_delta=max(d['claude_max_delta'] for d in diagnostics),
        canonical_weight_max_delta=max(d['canonical_max_delta'] for d in diagnostics),
        tokenizer_verified=True,scalar_token_predicate_all_tokens=True,topk_order_and_spans_verified=True,
        alignment_scope='Producer code uses logits[plen-1:plen-1+T]; raw token/span correspondence checked; no new forward pass.',
        jobs={key(e):json.loads((OUT/(key(e)+'_AUDIT.json')).read_text()) for e in JOBS}))
    write(OUT/'RUN_STATE.json',dict(status='SCORED_PENDING_EVALUATION',answers=len(meta),scoring_seconds=time.perf_counter()-began))
    print('[digit]scoring complete',time.perf_counter()-began,flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):run()
