"""Conservative original+innovation input blend for continuous B3."""
from __future__ import annotations
import time
import numpy as np
from .deem_b3_contract_ablation import GenericResult, PreparedArm
from .deem_b3_crossed_innovation import InnovationEnergy, build_innovation_map
from .residual_graph_deem import ContinuousDeemConfig, EPS, persistent_mala, set_determinism

ARMS=("L1_ROOK_BLEND_A25","L2_NONROOK_BLEND_A25")
BLEND=0.25

class BlendedInnovationEnergy(InnovationEnergy):
    def input_transform(self,X):
        innovation=super().input_transform(X)
        return (1.0-BLEND)*X+BLEND*innovation

def fit_blended_innovation(prepared:PreparedArm,arm:str,*,seed:int=0,config:ContinuousDeemConfig|None=None):
    import torch
    if arm not in ARMS: raise KeyError(arm)
    source_arm="K1_ROOK_INNOVATION" if arm.startswith("L1_") else "K2_NONROOK_INNOVATION"
    config=config or ContinuousDeemConfig(); innovation=build_innovation_map(prepared,source_arm); X=prepared.X_risk; started=time.perf_counter()
    set_determinism(seed); model=BlendedInnovationEnergy(prepared,config,seed,innovation); tensor=torch.as_tensor(X,dtype=torch.float64); buffer=tensor.clone(); generator=torch.Generator(device="cpu").manual_seed(seed+1_000_003)
    parameters=list(model.parameters()); optimizer=torch.optim.SGD(parameters,lr=config.learning_rate,momentum=config.momentum); acceptances=[]
    for _epoch in range(config.epochs):
        refresh=torch.rand(len(X),generator=generator)<config.replay_refresh
        if bool(refresh.any()): buffer[refresh]=tensor[torch.randint(len(X),(int(refresh.sum()),),generator=generator)]
        buffer,acceptance=persistent_mala(model,buffer,delta=config.mala_delta,steps=config.mala_steps,generator=generator)
        loss=model.free_energy(tensor).mean()-model.free_energy(buffer).mean(); optimizer.zero_grad(); loss.backward(); optimizer.step()
        if not bool(torch.isfinite(loss)) or not all(bool(torch.isfinite(p).all()) for p in parameters): raise FloatingPointError("non-finite blended innovation fit")
        acceptances.append(float(acceptance))
    with torch.no_grad(): ell_t,atomic_t,family_t=model.logit(tensor); transformed=model.input_transform(tensor).numpy()
    ell,atomic=ell_t.numpy(),atomic_t.numpy(); families={name:value.numpy() for name,value in family_t.items()}; q=1/(1+np.exp(-np.clip(ell,-700,700)))
    block_values=[]
    for indices in prepared.groups.values():
        usable=[i for i in indices if prepared.feature_names[i] not in prepared.anchor_exclusions]
        if usable: block_values.append(X[:,usable].mean(axis=1))
    anchor=np.mean(block_values,axis=0); high=float(np.sum(q*anchor)/max(np.sum(q),EPS)); low=float(np.sum((1-q)*anchor)/max(np.sum(1-q),EPS)); orientation=1 if high-low>0 else -1
    if orientation<0: q,ell,atomic=1-q,-ell,-atomic; families={name:-value for name,value in families.items()}
    reconstruction=float(np.max(np.abs(orientation*float(model.b.detach())+atomic.sum(1)-ell)))
    health={"healthy":bool(np.std(q)>=config.posterior_sd_min and reconstruction<=1e-8),"posterior_sd":float(np.std(q)),"reconstruction":reconstruction,"mala_acceptance":float(np.mean(acceptances)),"runtime_seconds":float(time.perf_counter()-started),"input_change_sd":float(np.std(transformed-X)),**innovation.diagnostics}
    return GenericResult(q,ell,atomic,families,orientation*float(model.b.detach()),orientation,health,model.state()),innovation

__all__=["ARMS","BLEND","fit_blended_innovation"]
