"""Actuated structured input adapter: frozen B3 SGD + adapter-only Adam."""

from __future__ import annotations

import time
import numpy as np

from .deem_b3_contract_ablation import GenericEnergy, GenericResult, PreparedArm
from .deem_b3_input_adapter import ADAPTER_SCALE, AdapterEnergy, build_adapter_bases
from .residual_graph_deem import ContinuousDeemConfig, EPS, persistent_mala, set_determinism


ARMS = ("J1_CROSS_ADAM_R2", "J2_BLOCK_PLUS_CROSS_ADAM_R2")
ADAPTER_LR = 0.002
ADAPTER_L2 = 1e-4


def fit_adapter_adam(prepared: PreparedArm, arm: str, *, seed: int = 0, config: ContinuousDeemConfig | None = None):
    import torch

    if arm not in ARMS: raise KeyError(arm)
    config = config or ContinuousDeemConfig(); started = time.perf_counter(); bases = build_adapter_bases(prepared)
    internal_arm = "I2_CROSS_R2" if arm == "J1_CROSS_ADAM_R2" else "I3_BLOCK_PLUS_CROSS_R2"
    X = prepared.X_risk; set_determinism(seed); model = AdapterEnergy(prepared, config, seed, bases, internal_arm)
    tensor = torch.as_tensor(X, dtype=torch.float64); buffer = tensor.clone(); generator = torch.Generator(device="cpu").manual_seed(seed + 1_000_003)
    base_parameters = GenericEnergy.parameters(model)
    adapter_parameters = [model.theta_cross] if arm == "J1_CROSS_ADAM_R2" else [model.theta_within, model.theta_cross]
    base_optimizer = torch.optim.SGD(base_parameters, lr=config.learning_rate, momentum=config.momentum)
    adapter_optimizer = torch.optim.Adam(adapter_parameters, lr=ADAPTER_LR)
    acceptances = []
    for _epoch in range(config.epochs):
        refresh = torch.rand(len(X), generator=generator) < config.replay_refresh
        if bool(refresh.any()): buffer[refresh] = tensor[torch.randint(len(X), (int(refresh.sum()),), generator=generator)]
        buffer, acceptance = persistent_mala(model, buffer, delta=config.mala_delta, steps=config.mala_steps, generator=generator)
        contrastive = model.free_energy(tensor).mean() - model.free_energy(buffer).mean()
        l2 = sum(parameter.square().sum() for parameter in adapter_parameters)
        loss = contrastive + ADAPTER_L2 * l2
        base_optimizer.zero_grad(); adapter_optimizer.zero_grad(); loss.backward()
        base_optimizer.step(); adapter_optimizer.step()
        if not bool(torch.isfinite(loss)) or not all(bool(torch.isfinite(p).all()) for p in base_parameters + adapter_parameters): raise FloatingPointError("non-finite Adam adapter")
        acceptances.append(float(acceptance))
    with torch.no_grad(): ell_t, atomic_t, family_t = model.logit(tensor); transformed = model.input_transform(tensor).numpy()
    ell, atomic = ell_t.numpy(), atomic_t.numpy(); families = {name:value.numpy() for name,value in family_t.items()}; q = 1/(1+np.exp(-np.clip(ell,-700,700)))
    block_values=[]
    for indices in prepared.groups.values():
        usable=[i for i in indices if prepared.feature_names[i] not in prepared.anchor_exclusions]
        if usable: block_values.append(X[:,usable].mean(axis=1))
    anchor=np.mean(block_values,axis=0); high=float(np.sum(q*anchor)/max(np.sum(q),EPS)); low=float(np.sum((1-q)*anchor)/max(np.sum(1-q),EPS)); orientation=1 if high-low>0 else -1
    if orientation<0: q,ell,atomic=1-q,-ell,-atomic; families={name:-value for name,value in families.items()}
    reconstruction=float(np.max(np.abs(orientation*float(model.b.detach())+atomic.sum(1)-ell)))
    diagnostics={
        "theta_within_norm":float(np.linalg.norm(model.theta_within.detach().numpy())),"theta_cross_norm":float(np.linalg.norm(model.theta_cross.detach().numpy())),
        "input_correction_sd":float(np.std(transformed-X)),"input_correction_max_abs":float(np.max(np.abs(transformed-X))),
    }
    health={"healthy":bool(np.std(q)>=config.posterior_sd_min and reconstruction<=1e-8),"posterior_sd":float(np.std(q)),"reconstruction":reconstruction,"mala_acceptance":float(np.mean(acceptances)),"runtime_seconds":float(time.perf_counter()-started),**diagnostics}
    result=GenericResult(q,ell,atomic,families,orientation*float(model.b.detach()),orientation,health,model.state())
    return result,bases,diagnostics


__all__=["ADAPTER_L2","ADAPTER_LR","ARMS","fit_adapter_adam"]
