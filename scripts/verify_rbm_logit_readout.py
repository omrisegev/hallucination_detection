"""Separate scalar replay on fixed cases and bootstrap algebra audit; no fitting."""
import argparse
import io
import json
from pathlib import Path
import sqlite3
import sys
import numpy as np
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_rbm_data_diagnostics import base
from spectral_utils.higher_moment_fusion import representation_order


def near(s):
    out=s.copy();sd=float(np.std(s));maximum=float(max(s))
    ids=[i for i,x in enumerate(s) if x>=maximum-.25*sd]
    for rank,i in enumerate(ids):out[i]=maximum+1e-6*max(sd,1e-12)*(len(ids)-rank)
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);a=p.parse_args()
    source=a.source_root;out=ROOT/'results/rbm_logit_readout_v1';base.old.configure_source_root(source)
    records=json.loads((base.old.BENCH/'evaluation/JOINED.json').read_text())['records']
    with np.load(base.old.BENCH/'evaluation/JOINED.npz') as z:offsets=z['offsets']
    with np.load(out/'SCORES.npz') as z:scores={k:z[k] for k in z.files if k.startswith('steps__')}
    db=source/'.worktrees/higher-moment-fusion-v1/results/higher_moment_fusion_v1/CHECKPOINT.sqlite'
    con=sqlite3.connect(db.as_uri()+'?mode=ro',uri=True);checks=0;negative=0
    for cell,path,kind,dataset in base.source_specs():
        rows=base.old._source_row_map(base.old.load_pickle(path),kind=kind,dataset=dataset)
        ids=sorted([i for i,r in enumerate(records) if r['cell']==cell],key=lambda i:len(rows[records[i]['row_id']]['token_entropies']))
        for index in sorted({0,len(ids)//2,int(.95*(len(ids)-1))}):
            i=ids[index];r=records[i];row=rows[r['row_id']]
            blob,info=con.execute('SELECT payload,info FROM answers WHERE idx=?',(i,)).fetchone();info=json.loads(info)
            lp=np.asarray(base.old._topk_payload(row)['logprobs'],float);spans=row['step_token_spans']
            with np.load(io.BytesIO(blob)) as state:
                for degree,bank in ((3,6),(6,12)):
                    x=representation_order(lp,row['token_spilled_energies'],degree)
                    for solver,root in [('rbm',f'rbm{bank}'),('rbm_initial',f'initial{bank}')]:
                        key=f'd{degree}__{solver}';d=info['diagnostics'][key];columns=np.asarray(d['columns'])
                        mean=np.asarray(d['normalization_mean']);scale=np.asarray(d['normalization_scale'])
                        z=(x[:,columns]-mean[columns])/scale[columns]
                        # Reconstruct each dot product separately; no scoring helper reused.
                        ell=np.array([float(state[key+'::b'])+float(np.dot(v,state[key+'::w'])) for v in z])
                        orientation=d['orientation'];negative+=int(orientation<0)
                        token=orientation*ell;post=expit(ell) if orientation==1 else 1-expit(ell)
                        np.testing.assert_allclose(expit(token),post,atol=1e-15)
                        for token_scores,suffix in [(post,''),(token,'logit_')]:
                            step=np.array([np.mean(sorted(token_scores[start:end])[-10:]) for start,end in spans])
                            sl=slice(offsets[i],offsets[i+1])
                            for readout,v in [('old',step),('near',near(step))]:
                                np.testing.assert_allclose(v,scores['steps__'+root+'__'+suffix+readout][sl],atol=1e-10,rtol=1e-12)
                                checks+=1
        del rows
    con.close()
    with np.load(out/'BOOTSTRAP_DRAWS.npz') as z:
        names=z['names'].tolist()
        for root in ('rbm6','initial6','rbm12','initial12'):
            for metric in ('pb','prm'):
                np.testing.assert_allclose(z[metric][:,names.index(root+'_interaction')],
                    z[metric][:,names.index(root+'_near_logit')]-z[metric][:,names.index(root+'_near_posterior')],atol=1e-14)
    assert json.loads((out/'RESULT_REVIEW.json').read_text())['status']=='PASS'
    report=dict(status='PASS',scope='Separate arithmetic in the same session, not external scientific review.',
        scalar_step_vector_checks=checks,negative_orientation_cases=negative,bootstrap_interaction_draw_checks=80000,
        checks=['27 fixed length-stratified cases, both banks and saved states','sigmoid identity','manual sorted top10 and nearmax','per-draw interaction identities','32-arm independent metric replay'])
    (out/'REPLAY_REVIEW.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':main()
