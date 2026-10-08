"""Same-answer native soft DEEM and continuous B3 on two frozen input banks."""
from dataclasses import asdict, replace
from importlib import metadata
import hashlib
import time
import numpy as np
from .direct_probability_fusion import zscore_columns
from .direct_probability_fusion_v2 import augmented_probability_risk
from .varentropy_contribution_fusion import contributions
from .deem_adapter import (DEEM_PINNED_VERSION, repaired_soft_adapter020_config,
                           continuous_to_deem_soft, risk_consensus_align, _jsonable)
from .deem_b3_contract_ablation import PreparedArm, GenericEnergy, fit_generic
from .residual_graph_deem import ContinuousDeemConfig, set_determinism

BANKS=('prob17','var15')
SOLVERS=('continuous_equal','soft_equal','deem','b3_initial','b3')
METHODS=tuple(f'{b}__{s}' for b in BANKS for s in SOLVERS)


def seed_for(uid):
    return int.from_bytes(hashlib.sha256(('deem-b3-v1:'+uid).encode()).digest()[:4],'little') % (2**31-1)


def inputs(logprobs,chosen):
    raw={'prob17':augmented_probability_risk({'logprobs':logprobs},chosen,k=15),
         'var15':contributions(logprobs,15)}
    out={}
    for bank,X in raw.items():
        if len(X)<3:raise ValueError('need three token rows')
        Z,keep,mean,scale=zscore_columns(X)
        if Z.shape[1]<3:raise ValueError('need three varying columns')
        if bank=='prob17':
            U=X.copy();U[:,15]=-np.expm1(-U[:,15])
            if np.any(U < -5e-7) or np.any(U > 1+5e-7):raise ValueError('invalid probability input')
            U=np.clip(U[:,keep],0.,1.)
            soft=np.stack((1-U,U),axis=1)
        else:
            # Existing rank adapter: ordinal pseudo-probabilities, not raw p.
            soft=continuous_to_deem_soft(X[:,keep])
        np.testing.assert_allclose(soft.sum(axis=1),1.,atol=1e-14)
        out[bank]=dict(raw=X,Z=Z,soft=soft,keep=keep,mean=mean,scale=scale)
    return out


def native_deem(soft,Z,seed,epochs=100):
    from deem import DEEM
    import torch
    if metadata.version('deem')!=DEEM_PINNED_VERSION:raise RuntimeError('DEEM version drift')
    cfg=repaired_soft_adapter020_config(device='cpu',epochs=epochs,alignment_ambiguous='identity')
    set_determinism(seed);torch.set_num_threads(1)
    model=DEEM(n_classes=2,hidden_dim=cfg.hidden_dim,cd_k=cfg.cd_k,
        deterministic=cfg.deterministic,learning_rate=cfg.learning_rate,
        momentum=cfg.momentum,epochs=cfg.epochs,batch_size=min(cfg.batch_size,len(Z)),
        device='cpu',auto_hyperparameters=False,random_state=seed,
        use_preprocessing=cfg.use_preprocessing,preprocessing_layers=cfg.preprocessing_layers,
        preprocessing_activation=cfg.preprocessing_activation,preprocessing_init=cfg.preprocessing_init,
        sampler_steps=cfg.sampler_steps,sampler_oh_mode=True,
        use_weighted=cfg.use_weighted,init_method=cfg.init_method)
    model.fit(soft,verbose=False)
    raw=np.asarray(model.predict(soft,return_probs=True),float)
    if raw.shape!=(len(Z),2) or not np.isfinite(raw).all():raise ValueError('invalid DEEM posterior')
    if np.any(raw<0) or np.any(raw>1):raise ValueError('out-of-range posterior')
    np.testing.assert_allclose(raw.sum(axis=1),1.,atol=1e-5)
    aligned,mapping,margin=risk_consensus_align(raw,Z,ambiguous='identity')
    state={k:v.detach().cpu().numpy().copy() for k,v in model.model_.state_dict().items()}
    if any(not np.isfinite(v).all() for v in state.values()):raise ValueError('nonfinite learned state')
    score=aligned[:,1]
    return score,dict(config=asdict(cfg),class_map=mapping,alignment_margin=float(margin),
        posterior_sd=float(np.std(score)),collapsed=bool(np.std(score)<1e-3),
        history=_jsonable(model.history_)),state


def fit_all(logprobs,chosen,uid,*,epochs=100):
    banks=inputs(logprobs,chosen);fits={};failures={};seconds={}
    seed=seed_for(uid)
    for bank,data in banks.items():
        X,Z,soft=data['raw'],data['Z'],data['soft'];indices=np.flatnonzero(data['keep'])
        for solver in SOLVERS:
            m=f'{bank}__{solver}';started=time.perf_counter();state={}
            diag=dict(seed=seed,active_columns=len(indices),columns=indices.tolist(),
                      normalization_mean=data['mean'].tolist(),normalization_scale=data['scale'].tolist(),
                      input_mapping='raw probability risks' if bank=='prob17' else 'rank pseudo-probabilities')
            try:
                if solver=='continuous_equal':score=Z @ (np.ones(Z.shape[1])/Z.shape[1])
                elif solver=='soft_equal':score=soft[:,1,:].mean(axis=1)
                elif solver=='deem':
                    score,extra,state=native_deem(soft,Z,seed,epochs);diag.update(extra)
                else:
                    names=tuple(f'coordinate_{i}' for i in indices)
                    groups={'input_bank':tuple(range(len(indices)))}
                    prepared=PreparedArm(Z,names,groups,frozenset(),data['mean'],data['scale'])
                    config=replace(ContinuousDeemConfig(),epochs=epochs)
                    if solver=='b3_initial':
                        import torch
                        model=GenericEnergy(names,groups,config,seed)
                        with torch.no_grad():
                            ell,_,_=model.logit(torch.as_tensor(Z,dtype=torch.float64))
                            raw=torch.sigmoid(ell).numpy()
                        anchor=Z.mean(axis=1)
                        high=float(np.sum(raw*anchor)/max(np.sum(raw),1e-12))
                        low=float(np.sum((1-raw)*anchor)/max(np.sum(1-raw),1e-12))
                        orientation=1 if high-low>0 else -1
                        score=raw if orientation>0 else 1-raw;state=model.state()
                        diag.update(orientation=orientation,alignment_margin=high-low,trained=False,
                                    groups=groups,config=asdict(config))
                    else:
                        result=fit_generic(prepared,seed=seed,config=config)
                        score=result.score;state=result.state
                        diag.update(health=result.health,orientation=result.orientation,
                                    aligned_bias=float(result.aligned_bias),groups=groups,
                                    config=asdict(config))
                if not np.isfinite(score).all():raise ValueError('nonfinite score')
                fits[m]=dict(score=score,diagnostics=diag,state=state)
            except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as e:
                failures[m]=f'{type(e).__name__}: {e}'
            seconds[m]=time.perf_counter()-started
    return fits,failures,seconds
