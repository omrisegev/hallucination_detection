"""SU/IU-style unsupervised reliability gate for B3 input innovations.

Each universal 3x3 core coordinate is ridge-predicted from four structured
peers. Group/length cross-fitting estimates coordinate reliability without
targets.  A bounded gate then mixes the original coordinate with its
standardized innovation before the unchanged B3 family subnetworks.
"""
from __future__ import annotations
from dataclasses import dataclass
import time
from typing import Sequence
import numpy as np
from .deem_b3_contract_ablation import GenericEnergy,GenericResult,PreparedArm
from .deem_b3_crossed_innovation import CORE_GRID,RIDGE
from .residual_graph_deem import ContinuousDeemConfig,EPS,assign_grouped_length_folds,persistent_mala,set_determinism

ARMS=("M1_ROOK_STATIC_R2","M2_ROOK_CONDITIONAL_R2","M3_NONROOK_CONDITIONAL_R2")
MAX_GATE=0.5

@dataclass(frozen=True)
class GateMap:
 prediction_matrix:np.ndarray
 innovation_scale:np.ndarray
 residual_scale:np.ndarray
 reliability:np.ndarray
 core_indices:np.ndarray
 diagnostics:dict[str,float]

def _peers(core,target_position,rook:bool):
 target_row,target_col,_=core[target_position]
 if rook: return [(position,index) for position,(row,col,index) in enumerate(core) if position!=target_position and (row==target_row or col==target_col)]
 return [(position,index) for position,(row,col,index) in enumerate(core) if row!=target_row and col!=target_col]

def build_gate_map(prepared:PreparedArm,group_ids:Sequence[str],raw_lengths:np.ndarray,arm:str)->GateMap:
 if arm not in ARMS: raise KeyError(arm)
 X=np.asarray(prepared.X_risk,dtype=np.float64); groups=np.asarray(group_ids,dtype=str); lengths=np.asarray(raw_lengths,dtype=float)
 if groups.shape!=(len(X),) or lengths.shape!=(len(X),): raise ValueError("unaligned group/length metadata")
 lookup={name:i for i,name in enumerate(prepared.feature_names)}
 if any(name not in lookup for row in CORE_GRID for name in row): raise ValueError("universal crossed core is incomplete")
 core=[(row,col,lookup[name]) for row,names in enumerate(CORE_GRID) for col,name in enumerate(names)]; rook=arm!="M3_NONROOK_CONDITIONAL_R2"
 folds=assign_grouped_length_folds(groups,lengths,n_folds=5); prediction_matrix=np.zeros((X.shape[1],X.shape[1]),dtype=np.float64); innovation_scale=np.ones(X.shape[1]); residual_scale=np.ones(X.shape[1]); reliability=np.zeros(X.shape[1]); oof_r2=[]
 for target_position,(_,_,target_index) in enumerate(core):
  peer_indices=[index for _,index in _peers(core,target_position,rook)]; oof=np.empty(len(X),dtype=np.float64)
  for fold in sorted(set(folds.tolist())):
   held=folds==fold; donor=~held; P=X[donor][:,peer_indices]; y=X[donor,target_index]; beta=np.linalg.solve(P.T@P/max(int(donor.sum()),1)+RIDGE*np.eye(4),P.T@y/max(int(donor.sum()),1)); oof[held]=X[held][:,peer_indices]@beta
  target=X[:,target_index]; residual=target-oof; r2=max(0.0,min(1.0,1.0-float(np.mean(residual**2))/max(float(np.var(target)),EPS))); oof_r2.append(r2); reliability[target_index]=r2; residual_scale[target_index]=max(float(np.std(residual)),EPS)
  P=X[:,peer_indices]; beta=np.linalg.solve(P.T@P/len(X)+RIDGE*np.eye(4),P.T@target/len(X)); prediction_matrix[target_index,peer_indices]=beta; innovation_scale[target_index]=max(float(np.std(target-P@beta)),EPS)
 core_indices=np.asarray([index for _,_,index in core],dtype=np.int64); static=MAX_GATE*reliability[core_indices]
 return GateMap(prediction_matrix,innovation_scale,residual_scale,reliability,core_indices,{"mean_oof_r2":float(np.mean(oof_r2)),"min_oof_r2":float(np.min(oof_r2)),"max_oof_r2":float(np.max(oof_r2)),"mean_static_gate":float(np.mean(static)),"max_gate":MAX_GATE,"rook_support":float(rook)})

class GatedInnovationEnergy(GenericEnergy):
 def __init__(self,prepared:PreparedArm,config:ContinuousDeemConfig,seed:int,gate_map:GateMap,arm:str):
  super().__init__(prepared.feature_names,prepared.groups,config,seed); torch=self.torch; self.arm=arm; self.prediction_matrix=torch.as_tensor(gate_map.prediction_matrix,dtype=torch.float64); self.innovation_scale=torch.as_tensor(gate_map.innovation_scale,dtype=torch.float64); self.residual_scale=torch.as_tensor(gate_map.residual_scale,dtype=torch.float64); self.reliability=torch.as_tensor(gate_map.reliability,dtype=torch.float64); self.core_indices=torch.as_tensor(gate_map.core_indices,dtype=torch.long)
 def input_transform_and_gate(self,X):
  prediction=X@self.prediction_matrix.T; innovation=(X-prediction)/self.innovation_scale; static=MAX_GATE*self.reliability
  if self.arm=="M1_ROOK_STATIC_R2": gate=static.unsqueeze(0).expand_as(X)
  else: gate=static.unsqueeze(0)*self.torch.exp(-0.5*((X-prediction)/self.residual_scale).square())
  mask=self.torch.zeros_like(gate); mask[:,self.core_indices]=gate[:,self.core_indices]; transformed=(1-mask)*X+mask*innovation
  return transformed,mask
 def input_transform(self,X): return self.input_transform_and_gate(X)[0]
 def contributions(self,X): return super().contributions(self.input_transform(X))
 def state(self):
  output=super().state(); output["gate::prediction_matrix"]=self.prediction_matrix.detach().numpy().copy(); output["gate::innovation_scale"]=self.innovation_scale.detach().numpy().copy(); output["gate::residual_scale"]=self.residual_scale.detach().numpy().copy(); output["gate::reliability"]=self.reliability.detach().numpy().copy(); return output

def fit_unsupervised_gate(prepared:PreparedArm,group_ids:Sequence[str],raw_lengths:np.ndarray,arm:str,*,seed:int=0,config:ContinuousDeemConfig|None=None):
 import torch
 config=config or ContinuousDeemConfig(); gate_map=build_gate_map(prepared,group_ids,raw_lengths,arm); X=prepared.X_risk; started=time.perf_counter(); set_determinism(seed); model=GatedInnovationEnergy(prepared,config,seed,gate_map,arm); tensor=torch.as_tensor(X,dtype=torch.float64); buffer=tensor.clone(); generator=torch.Generator(device="cpu").manual_seed(seed+1_000_003); parameters=list(model.parameters()); optimizer=torch.optim.SGD(parameters,lr=config.learning_rate,momentum=config.momentum); acceptances=[]
 for _epoch in range(config.epochs):
  refresh=torch.rand(len(X),generator=generator)<config.replay_refresh
  if bool(refresh.any()): buffer[refresh]=tensor[torch.randint(len(X),(int(refresh.sum()),),generator=generator)]
  buffer,acceptance=persistent_mala(model,buffer,delta=config.mala_delta,steps=config.mala_steps,generator=generator); loss=model.free_energy(tensor).mean()-model.free_energy(buffer).mean(); optimizer.zero_grad(); loss.backward(); optimizer.step()
  if not bool(torch.isfinite(loss)) or not all(bool(torch.isfinite(p).all()) for p in parameters): raise FloatingPointError("non-finite unsupervised gate fit")
  acceptances.append(float(acceptance))
 with torch.no_grad(): ell_t,atomic_t,family_t=model.logit(tensor); transformed,gate=model.input_transform_and_gate(tensor)
 ell,atomic=ell_t.numpy(),atomic_t.numpy(); families={name:value.numpy() for name,value in family_t.items()}; q=1/(1+np.exp(-np.clip(ell,-700,700))); block_values=[]
 for indices in prepared.groups.values():
  usable=[i for i in indices if prepared.feature_names[i] not in prepared.anchor_exclusions]
  if usable: block_values.append(X[:,usable].mean(axis=1))
 anchor=np.mean(block_values,axis=0); high=float(np.sum(q*anchor)/max(np.sum(q),EPS)); low=float(np.sum((1-q)*anchor)/max(np.sum(1-q),EPS)); orientation=1 if high-low>0 else -1
 if orientation<0: q,ell,atomic=1-q,-ell,-atomic; families={name:-value for name,value in families.items()}
 reconstruction=float(np.max(np.abs(orientation*float(model.b.detach())+atomic.sum(1)-ell))); gate_np=gate.numpy()[:,gate_map.core_indices]
 health={"healthy":bool(np.std(q)>=config.posterior_sd_min and reconstruction<=1e-8),"posterior_sd":float(np.std(q)),"reconstruction":reconstruction,"mala_acceptance":float(np.mean(acceptances)),"runtime_seconds":float(time.perf_counter()-started),"input_change_sd":float(np.std(transformed.numpy()-X)),"mean_gate":float(np.mean(gate_np)),"gate_sd":float(np.std(gate_np)),"gate_max":float(np.max(gate_np)),**gate_map.diagnostics}
 return GenericResult(q,ell,atomic,families,orientation*float(model.b.detach()),orientation,health,model.state()),gate_map

__all__=["ARMS","MAX_GATE","GateMap","build_gate_map","fit_unsupervised_gate"]
