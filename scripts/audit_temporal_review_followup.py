"""Read-only audits of frozen selectors and completed context checkpoints."""
import hashlib
import itertools
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from spectral_utils.context_training import FeatureBundle
from spectral_utils.temporal_context_models import ConditionalFlow,flow_condition,probability_pgd,flow_objective

OUT=ROOT/'results/temporal_review_followup_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf8')

def selectors():
    path=ROOT/'results/temporal_dufs31_v1/SELECTORS.json'
    rows=json.loads(path.read_text());result={}
    for name,r in rows.items():
        d=r['diagnostics'];probs=np.asarray(d['per_seed_probabilities'])
        sets=[set(np.argsort(-p,kind='stable')[:4]) for p in probs]
        pairs=list(itertools.combinations(sets,2))
        result[name]=dict(effective_features=d['effective_feature_count'],near_one=d['near_one_fraction'],
            raw_gate_range=float(np.ptp(d['raw_probabilities'])),
            top4_seed_overlap=float(np.mean([len(a&b)/4 for a,b in pairs])),
            top4_seed_jaccard=float(np.mean([len(a&b)/len(a|b) for a,b in pairs])))
    summary={k:dict(min=min(v[k] for v in result.values()),median=float(np.median([v[k] for v in result.values()])),
                   max=max(v[k] for v in result.values()),mean=float(np.mean([v[k] for v in result.values()]))) for k in next(iter(result.values()))}
    save(OUT/'DUFS_AUDIT.json',dict(status='SATURATED_SELECTION_NOT_VALIDATED',selectors=len(rows),
        source_sha256=sha(path),summary=summary,selectors_detail=result,
        conclusion='Dense gates and unstable top-k across seeds; historical top-k scores are valid observations, not a validated sparse-selection result.'))
    print('DUFS',summary,flush=True)

def checkpoint(path):
    manifest=json.loads((path/'MANIFEST.json').read_text());bank=manifest['bank']
    bundle=FeatureBundle(ROOT/'results/temporal_context_data_v1',bank)
    train,validation,held=bundle.split(manifest['excluded_folds'])
    assert sorted({bundle.metadata[i]['group_id'] for i in validation})==manifest['validation_groups']
    d=len(bundle.columns);model=ConditionalFlow(17*d+18,d)
    state=torch.load(path/'BEST.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(state['model']);model.eval()
    rng=np.random.default_rng(20260915);torch.manual_seed(20260915);rows=[]
    for _ in range(16):
        ids,pos=bundle.sampler(validation,rng,256,flow=True);b=bundle.batch(ids,pos,flow=True,device='cpu')
        noise=torch.randn_like(b['target']);tau=torch.rand(len(ids),1)
        x=(1-tau)*noise+tau*b['target'];u=b['target']-noise
        c=flow_condition(b['history'],b['mask'],b['position'])
        nc,adversary=probability_pgd(model,x,tau,u,b['source_logits'],lambda z:bundle.condition_from_probabilities(b,z))
        v=model(x,tau,c);nv=model(x,tau,nc)
        pe=(u-v).norm(dim=-1);ne=(u-nv).norm(dim=-1)
        rh=pe-ne+1.;ch=(1-F.cosine_similarity(u,v,dim=-1))-(1-F.cosine_similarity(u,nv,dim=-1))+.9
        fm=(v-u).square().sum(-1).mean();repel=F.relu(rh).mean();curve=F.relu(ch).mean()
        losses=[fm,.1*repel,.1*curve];grads=[]
        for loss in losses:
            g=torch.autograd.grad(loss,tuple(model.parameters()),retain_graph=True)
            grads.append(torch.cat([t.flatten() for t in g]))
        total,parts=flow_objective(model,x,tau,u,c,nc)
        torch.testing.assert_close(total,fm+.1*repel+.1*curve)
        identity,iparts=flow_objective(model,x,tau,u,c,c)
        torch.testing.assert_close(iparts['repel'],torch.tensor(1.));torch.testing.assert_close(iparts['curve'],torch.tensor(.9))
        rows.append(dict(fm=float(fm.detach()),repel=float(repel.detach()),curve=float(curve.detach()),
            weighted_total=float(total.detach()),repel_active=float((rh>0).float().mean()),curve_active=float((ch>0).float().mean()),
            target_velocity_norm=float(u.norm(dim=-1).mean()),positive_error=float(pe.detach().mean()),negative_error=float(ne.detach().mean()),
            condition_delta_norm=float((nc-c).norm(dim=-1).mean()),max_logit_delta=float((adversary-b['source_logits']).abs().max()),
            grad_fm=float(grads[0].norm()),grad_weighted_repel=float(grads[1].norm()),grad_weighted_curve=float(grads[2].norm()),
            grad_aux_sum=float((grads[1]+grads[2]).norm()),
            cosine_fm_aux=float(F.cosine_similarity(grads[0][None],(grads[1]+grads[2])[None]).item())))
    summary={k:float(np.mean([r[k] for r in rows])) for k in rows[0]}
    result=dict(checkpoint_sha256=sha(path/'BEST.pt'),source_manifest_sha256=sha(path/'MANIFEST.json'),best_step=state['step'],
        method=manifest['method'],bank=bank,validation_groups=manifest['validation_groups'],batches=16,batch_size=256,
        summary=summary,batches_detail=rows,training_modified=False,historical_loss_reconstruction=False,
        fm_note='Diagnostic PGD/repel/curve are counterfactual for FM; FM training used only FM loss.',
        identity_negative_check='PASS: same-condition hinge losses equal margins; activity alone does not prove useful negative separation.')
    save(OUT/(path.name+'_CHECKPOINT_AUDIT.json'),result);print(path.name,summary,flush=True)

def sensitivity():
    results={}
    for method in ('fm','diflo','tcn'):
        p=ROOT/f'results/temporal_context_models_v1/{method}__innovation5__seed0__exclude0/scoring/STEP_SCORES.npz'
        if not p.exists():continue
        z=np.load(p);out={}
        for readout in (['dot'] if method!='tcn' else ['signed','squared','variance_fusion']):
            for control in ('zero','shuffled'):
                a,b=z['real__'+readout],z[control+'__'+readout];ok=np.isfinite(a)&np.isfinite(b)
                out[readout+'__'+control]=dict(steps=int(ok.sum()),correlation=float(np.corrcoef(a[ok],b[ok])[0,1]),
                    rms_difference_over_real_std=float(np.sqrt(np.mean((a[ok]-b[ok])**2))/np.std(a[ok])))
        results[method]=dict(scores_sha256=sha(p),comparisons=out)
    save(OUT/'HISTORY_SENSITIVITY.json',dict(quality_evaluation=False,results=results,
        warning='Inference interventions only; zero preserves current innovation, masks and position. Pooled step correlation does not establish task utility or absence of within-answer effects.'))

def main():
    torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=True)
    paths=[p for p in sorted((ROOT/'results/temporal_context_models_v1').glob('*'))
           if (p/'RUN_STATE.json').exists() and json.loads((p/'RUN_STATE.json').read_text()).get('status')=='TRAINED'
           and (p/'MANIFEST.json').exists() and json.loads((p/'MANIFEST.json').read_text())['method'] in ('diflo','fm')]
    save(OUT/'AUDIT_MANIFEST.json',dict(schema='temporal-review-checkpoint-audit-v1',checkpoints=[p.name for p in paths],
        contract_sha256=sha(ROOT/'docs/experiments/TEMPORAL_REVIEW_FOLLOWUP_20260915.md'),code_sha256=sha(Path(__file__)),
        paper_deviations=dict(width=dict(ours=128,paper_synthetic=512),max_updates=dict(ours=50000,paper_synthetic=200000),
            margins=dict(repel=1.,curve=.9,task_scale_justified=False),early_stop='6 validation checkpoints without improvement'),
        adaptations=['q15-logit-constrained PGD','reasoning telemetry condition','DOT assigned to current token','residual readout'],
        no_new_training=True))
    selectors();sensitivity()
    for p in paths:checkpoint(p)
    save(OUT/'RUN_STATE.json',dict(status='COMPLETE_CHECKPOINT_DIAGNOSTICS',checkpoints=len(paths),new_training=False))

if __name__=='__main__':main()
