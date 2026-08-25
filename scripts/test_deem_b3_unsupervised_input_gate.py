#!/usr/bin/env python3
"""Mechanical checks for the cross-fitted unsupervised input gate."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from spectral_utils.deem_b3_contract_ablation import PreparedArm
from spectral_utils.deem_b3_crossed_innovation import CORE_GRID
from spectral_utils.deem_b3_unsupervised_input_gate import ARMS,MAX_GATE,GatedInnovationEnergy,build_gate_map
from spectral_utils.residual_graph_deem import ContinuousDeemConfig
def main():
 rng=np.random.default_rng(77); n=250; shared=rng.normal(size=(n,3)); X=np.column_stack([.55*shared[:,r]+.35*shared[:,c]+.3*rng.normal(size=n) for r in range(3) for c in range(3)]); X=(X-X.mean(0))/X.std(0); names=tuple(name for row in CORE_GRID for name in row); prepared=PreparedArm(X,names,{f"g{i}":tuple(range(3*i,3*i+3)) for i in range(3)},frozenset(),np.zeros(9),np.ones(9)); group_ids=np.asarray([f"q{i//2}" for i in range(n)]); lengths=np.asarray(16+(np.arange(n)%31))
 for arm in ARMS:
  gate_map=build_gate_map(prepared,group_ids,lengths,arm); assert np.all((gate_map.reliability>=0)&(gate_map.reliability<=1)); assert gate_map.diagnostics["mean_static_gate"]<=MAX_GATE
  model=GatedInnovationEnergy(prepared,ContinuousDeemConfig(epochs=1,mala_steps=1),0,gate_map,arm)
  import torch
  tensor=torch.as_tensor(X,dtype=torch.float64); transformed,gate=model.input_transform_and_gate(tensor); assert torch.isfinite(transformed).all(); assert float(gate.min())>=0 and float(gate.max())<=MAX_GATE+1e-12; assert torch.isfinite(model.free_energy(tensor)).all()
 print("deem_b3_unsupervised_input_gate mechanical tests: PASS")
if __name__=="__main__": main()
