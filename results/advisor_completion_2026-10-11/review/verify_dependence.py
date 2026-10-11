"""Independent covariance-by-sums and vote-contingency replay."""
from pathlib import Path
import csv,json
import numpy as np
from scipy.stats import rankdata
P=Path(__file__).resolve().parents[1]
p=np.load(P/'inputs/population.npz');off=p['offsets'];y=p['y'].astype(bool)
models=json.loads((P/'statistics/ESTIMATORS.json').read_text())
checks=[]
for scope in ['channels','groups','features']:
 if scope=='features':
  X=np.load(P/'inputs/bank_V_N16.npz')['V'].copy();target=np.load(P/'statistics/CONDITIONAL_CORRELATIONS.npz')
 else:
  target=np.load(P/f'statistics/{scope}_VOTE_CONDITIONAL.npz');X=target['votes'].astype(float)
 for a,b in zip(off[:-1],off[1:]):X[a:b]-=X[a:b].sum(0)/(b-a)
 matrices=[]
 for name,mask in [('error',y),('valid',~y)]:
  Z=X[mask];N=len(Z);cross=Z.T@Z-np.outer(Z.sum(0),Z.sum(0))/N
  den=np.sqrt(np.diag(cross));corr=cross/np.outer(den,den)
  discrepancy=float(np.nanmax(np.abs(corr-target[name])))
  assert discrepancy<1e-10,(scope,name,discrepancy)
  matrices.append(corr);checks.append(dict(scope=scope,subset=name,max_abs=discrepancy))
 assert np.allclose(matrices[0]-matrices[1],target['difference'],atol=1e-10)
 if scope!='features':
  votes=target['votes']>0;sens=np.count_nonzero(votes[y],axis=0)/int(y.sum());spec=np.count_nonzero(~votes[~y],axis=0)/int((~y).sum())
  bal=(sens+spec)/2
  for method in ['MOM','DS']:
   m=models[f'{scope}/{method}'];truth=m['truth'];summ=m['summary'];est=m['estimate']
   for key,value in [('psi',sens),('eta',spec),('pi',bal)]:assert np.allclose(truth[key],value,rtol=0,atol=1e-14)
   er=rankdata(est['pi']);tr=rankdata(bal);er-=er.mean();tr-=tr.mean()
   rho=float(er@tr/np.sqrt((er@er)*(tr@tr)))
   assert abs(rho-summ['reliability_spearman'])<1e-14
   assert abs(np.mean(np.abs(np.asarray(est['psi'])-sens))-summ['psi_MAE'])<1e-14
   assert abs(np.mean(np.abs(np.asarray(est['eta'])-spec))-summ['eta_MAE'])<1e-14
   assert abs(summ['prevalence_true']-y.sum()/len(y))<1e-14
   checks.append(dict(scope=scope,estimator=method,truth_and_calibration='PASS'))
result=dict(status='PASS',checks=checks,interpretation='Diagnostic dependence/calibration replay; does not prove conditional independence or estimator identifiability')
(P/'review/DEPENDENCE_RECEIPT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
