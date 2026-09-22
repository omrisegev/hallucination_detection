"""Additional direct loss replay; does not refit or select parameters."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from scipy.special import logsumexp,expit
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_rbm_data_diagnostics import csv_write,base
from spectral_utils.higher_moment_fusion import feature_names


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);args=p.parse_args()
    out=ROOT/'results/rbm_supervised_position_diagnostic_v1';bench=args.source_root/'results/localization_full_benchmark_v3/evaluation'
    records=json.loads((bench/'JOINED.json').read_text())['records'];j=np.load(bench/'JOINED.npz');off=j['offsets'];y=j['labels']
    f=np.load(out/'FEATURES.npz');owner=np.repeat(np.arange(len(records)),np.diff(off));checks=[];weights=[]
    with threadpool_limits(limits=1):
        for path in sorted(out.glob('fit_*.npz')):
            info=json.loads(path.with_suffix('.json').read_text());z=np.load(path);pb=info['cell'].startswith('pb_')
            x=(f['x']-z['mean'])/z['scale'];c=f['context'];train=z['train_answers']
            for mode in ('static','prior','conditional'):
                key='supervised_'+mode;theta=z[key+'_theta'];design=x
                if mode=='prior':design=np.c_[x,c]
                if mode=='conditional':design=np.c_[x,c,x*c[:,None]]
                w=theta if pb else theta[:-1];bias=0. if pb else theta[-1];score=design@w+bias
                if pb:
                    eligible=train[j['target'][train]>=0];losses=[];grad=np.zeros_like(w)
                    for i in eligible:
                        a,b=off[i:i+2];v=score[a:b];t=int(j['target'][i]);l=logsumexp(v)
                        losses.append(l-v[t]);p=np.exp(v-l);p[t]-=1;grad+=design[a:b].T@p/len(eligible)
                    loss=np.mean(losses)+.005*(w@w);grad+=.01*w
                else:
                    rows=z['train_steps'];rows=rows[y[rows]>=0];truth=(y[rows]==1).astype(float)
                    count={i:int(np.sum(owner[rows]==i)) for i in train};mass=np.array([1/count[i] for i in owner[rows]])
                    mass=np.where(truth==1,.5*mass/mass[truth==1].sum(),.5*mass/mass[truth==0].sum())
                    v=score[rows];loss=np.sum(mass*np.where(truth==1,np.logaddexp(0,-v),np.logaddexp(0,v)))+.005*(w@w)
                    residual=mass*(expit(v)-truth);grad=np.r_[design[rows].T@residual+.01*w,residual.sum()]
                np.testing.assert_allclose(loss,info['fits'][key]['loss'],atol=1e-11,rtol=1e-11)
                np.testing.assert_allclose(np.max(np.abs(grad)),info['fits'][key]['gradient_max'],atol=1e-11,rtol=1e-8)
                assert loss<=info['fits'][key]['initial_loss']+1e-10
                checks.append(dict(cell=info['cell'],fold=info['fold'],method=key,loss=float(loss),gradient_max=float(np.max(np.abs(grad)))))
                if mode=='conditional':
                    # Coefficients in the original answer-normalized coordinate system.
                    for k,name in enumerate(feature_names(6)):
                        weights.append(dict(cell=info['cell'],fold=info['fold'],feature=name,
                            early=float((w[k]-w[13+k])/z['scale'][k]),late=float((w[k]+w[13+k])/z['scale'][k]),
                            position_intercept=float(w[12])))
    assert len(checks)==135
    base.atomic_json(out/'OBJECTIVE_REVIEW.json',dict(status='PASS',models=135,max_gradient=max(v['gradient_max'] for v in checks),checks=checks))
    csv_write(out/'CONDITIONAL_WEIGHTS.csv',weights)
    print('PASS: 135 objectives, gradients, train-only loss and saved conditional coefficients')


if __name__=='__main__':main()
