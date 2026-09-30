"""Independent source/likelihood/rank checks and the frozen advancement screen."""
import hashlib
import importlib.util
from pathlib import Path
import time
import numpy as np
from scipy.special import logsumexp

ROOT=Path(__file__).resolve().parents[1]


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


d=module(ROOT/'scripts/run_fusion_gate_calibration_v1.py','calibration_review_driver')
vector=module(ROOT/'tests/test_fusion_trajectory_imm.py','calibration_vector_reference')
OUT=d.OUT


def norm(x):return (x-x.mean())/x.std()


def ar(z,rho):
    x=np.empty_like(z);x[0]=z[0]
    for i in range(1,len(x)):x[i]=rho*x[i-1]+np.sqrt(1-rho*rho)*z[i]
    return x


def noise(x):return float(np.clip((np.median(np.abs(np.diff(x)))/(.67448975*np.sqrt(2)))**2,.05,1))


def gate_algebra(x,detail):
    if not detail['valid']:return False
    g=detail['gate'];mu=np.array(g['means']);v=np.array(g['variances']);w=np.array(g['weights'])
    assert np.all(v>0) and np.all(w>0);np.testing.assert_allclose(w.sum(),1.,atol=1e-12)
    ll2=float(logsumexp(np.log(w)[None,:]-.5*np.log(2*np.pi*v)[None,:]-(x[:,None]-mu[None,:])**2/(2*v[None,:]),axis=1).sum())
    variance=x.var()+1e-4;ll1=float((-.5*np.log(2*np.pi*variance)-(x-x.mean())**2/(2*variance)).sum())
    bic=np.array([-2*ll1+2*np.log(len(x)),-2*ll2+5*np.log(len(x))])
    np.testing.assert_allclose(g['log_likelihood'],[ll1,ll2],atol=1e-9,rtol=1e-11)
    np.testing.assert_allclose(g['bic'],bic,atol=1e-9,rtol=1e-11)
    np.testing.assert_allclose(g['bic_gain'],bic[0]-bic[1],atol=1e-9,rtol=1e-11)
    opened=bool(bic[1]<bic[0] and np.any(x>mu.mean()))
    assert opened==detail['native_open']==(g['prediction']!=-1)
    return True


def main():
    start=time.monotonic();m=d.verify();f=d.load(OUT/'FROZEN.json')
    assert f['trials']==384 and f['manifest_sha256']==d.sha(OUT/'MANIFEST.json')
    for p,h in f['files'].items():assert d.sha(p)==h,p
    records=[];checks=dict(sources=0,lag_regressions=0,mixture_likelihoods=0,pvalue_decisions=0,direct_vector_imm=0,actual_mixture_refits=0)
    missing=0
    for rec in m['selected']:
        uid=d.identity(rec);meta=d.load(OUT/'trials'/(uid+'.json'));assert meta['manifest_sha256']==f['manifest_sha256']
        for key,value in rec.items():assert meta[key]==value
        n,rho=rec['n'],rec['rho']
        es=int(hashlib.sha256(('fusion-gate-calibration-v1/evaluation/'+f'{n}/{rec["replicate"]}').encode()).hexdigest()[:32],16)
        cs=int(hashlib.sha256(('fusion-gate-calibration-v1/calibration/'+uid).encode()).hexdigest()[:32],16)
        assert es==meta['evaluation_seed'] and cs==meta['calibration_seed'] and es!=cs
        z=np.random.default_rng(es).normal(size=n);nulls=np.random.default_rng(cs).normal(size=(39,n))
        x=ar(z,rho)
        if rec['jump']:x[n//2:]+=3
        regression=np.linalg.lstsq(np.column_stack((np.ones(n-1),x[:-1])),x[1:],rcond=None)[0]
        np.testing.assert_allclose(regression[1],meta['rho_estimate']['raw'],atol=1e-12,rtol=1e-12)
        fitted=float(np.clip(regression[1],-.95,.95))
        np.testing.assert_allclose(fitted,meta['rho_estimate']['value'],atol=1e-12,rtol=1e-12)
        checks['lag_regressions']+=1
        with np.load(OUT/'trials'/(uid+'.npz'),allow_pickle=False) as arrays:
            np.testing.assert_array_equal(arrays['observed_innovations'],z)
            np.testing.assert_array_equal(arrays['null_innovations'],nulls)
            np.testing.assert_array_equal(arrays['observed_source'],x)
            batches=[('observed',[x],[meta['observed']])]
            for kind,param in [('fitted',meta['rho_estimate']['value']),('known',rho)]:
                batches.append((kind,[ar(z,param) for z in nulls],meta['nulls'][kind]))
            for kind,sources,details in batches:
                assert len(sources)==len(details)
                for i,(source,info) in enumerate(zip(sources,details)):
                    normalized=norm(source);r=noise(normalized)
                    np.testing.assert_allclose([info['source_scale']['mean'],info['source_scale']['sd']],[source.mean(),source.std()],atol=1e-12,rtol=1e-12)
                    np.testing.assert_allclose(info['R'],r,atol=1e-12,rtol=1e-12);checks['sources']+=1
                    for name in ('raw','imm'):
                        curve=arrays[kind+'_'+name] if kind=='observed' else arrays[kind+'_'+name][i]
                        np.testing.assert_allclose([curve.mean(),curve.std()],[0.,1.],atol=1e-12)
                        if name=='raw':np.testing.assert_allclose(curve,norm(normalized),atol=1e-12,rtol=1e-12)
                        if name=='imm' and rec['replicate']==0 and (kind=='observed' or i==0):
                            expected=vector.vector_imm(normalized[:,None],np.array([[r]]),[.01*r,r])['level']
                            np.testing.assert_allclose(curve,norm(expected),atol=1e-10,rtol=1e-10)
                            np.testing.assert_allclose([info['scales'][name]['mean'],info['scales'][name]['sd']],[expected.mean(),expected.std()],atol=1e-11,rtol=1e-11)
                            checks['direct_vector_imm']+=1
                        valid=gate_algebra(curve,info['methods'][name]);checks['mixture_likelihoods']+=int(valid);missing+=int(not valid)
                        if valid and rec['replicate']==0 and (kind=='observed' or i==0):
                            actual=d.core.inspect_mixture(curve,curve)
                            np.testing.assert_allclose(actual['bic'],info['methods'][name]['gate']['bic'],atol=1e-9,rtol=1e-11)
                            checks['actual_mixture_refits']+=1
            for name in ('raw','imm'):
                observed=meta['observed']['methods'][name]
                assert meta['decisions'][name+'__native']['valid']==observed['valid']
                assert meta['decisions'][name+'__native']['open']==observed.get('native_open')
                for kind in ('fitted','known'):
                    null=[r['methods'][name] for r in meta['nulls'][kind]];decision=meta['decisions'][name+'__'+kind]
                    valid=observed['valid'] and len(null)==39 and all(r['valid'] for r in null)
                    assert decision['valid']==valid
                    if valid:
                        stat=observed['gate']['bic_gain'];tol=100*np.finfo(float).eps*abs(stat)
                        ge=sum(r['gate']['bic_gain']>=stat-tol for r in null);p=(1+ge)/40
                        eligible=bool(np.any(arrays['observed_'+name]>np.mean(observed['gate']['means'])))
                        assert decision['pvalue']==p and decision['exceedances']==ge and decision['eligible']==eligible
                        assert decision['open']==bool(p<=.05 and eligible)
                    checks['pvalue_decisions']+=1
        records.append(meta)
    summaries=[];screen={}
    for n in (16,64,256):
        for rho,jump in [(0.,False),(.6,False),(.9,False),(0.,True)]:
            subset=[r for r in records if (r['n'],r['rho'],r['jump'])==(n,rho,jump)]
            assert len(subset)==32
            for name in ('raw','imm'):
                native=[r['decisions'][name+'__native']['open'] for r in subset]
                counts={kind:sum(bool(r['decisions'][name+'__'+kind].get('open')) for r in subset) for kind in ('native','fitted','known')}
                valid={kind:sum(r['decisions'][name+'__'+kind]['valid'] for r in subset) for kind in ('native','fitted','known')}
                retained=sum(bool(a and r['decisions'][name+'__fitted'].get('open')) for a,r in zip(native,subset))
                passed=(retained>=.75*counts['native']) if jump else (counts['native']<.2*32 or counts['fitted']<=.5*counts['native'])
                passed=passed and valid['native']==valid['fitted']==32
                summaries.append(dict(n=n,rho=rho,jump=jump,readout=name,counts=counts,valid=valid,
                                      retained_native_detections=retained,advancement_cell_pass=passed))
    for name in ('raw','imm'):
        subset=[r for r in summaries if r['readout']==name]
        screen[name]=dict(pass_all_cells=all(r['advancement_cell_pass'] for r in subset),
                         failed_cells=[{k:r[k] for k in ('n','rho','jump','counts','retained_native_detections')} for r in subset if not r['advancement_cell_pass']])
    results=dict(status='COMPLETE_REVIEWED_SIMULATION',conditions=summaries,advancement_screen=screen,
                 real_benchmark_scores_changed=False,semantic_null_established=False,missing_mixture_fits=missing,
                 interpretation='Engineering advancement screen, not hypothesis-test confirmation; known rho is diagnostic only.')
    d.save(OUT/'RESULTS.json',results)
    d.save(OUT/'REVIEW.json',dict(status='PASS',checks=checks,missing_mixture_fits=missing,seconds=time.monotonic()-start,
          results_sha256=d.sha(OUT/'RESULTS.json'),manifest_sha256=d.sha(OUT/'MANIFEST.json'),frozen_sha256=d.sha(OUT/'FROZEN.json'),
          scope='Same-session independent regression, source and likelihood/rank algebra; representative shared-kernel GMM refits and direct vector IMM.'))
    print('ReviewPASS',checks,'advancement',screen,flush=True)


if __name__=='__main__':main()
