#!/usr/bin/env python3
"""Mechanical invariants for the crossed-innovation input layer."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from spectral_utils.deem_b3_contract_ablation import PreparedArm
from spectral_utils.deem_b3_crossed_innovation import ARMS,CORE_GRID,InnovationEnergy,build_innovation_map
from spectral_utils.deem_b3_crossed_innovation_blend import BLEND,BlendedInnovationEnergy
from spectral_utils.residual_graph_deem import ContinuousDeemConfig

def main():
 rng=np.random.default_rng(20260825); latent=rng.normal(size=(256,3)); columns=[]
 for source in range(3):
  for operator in range(3): columns.append(.55*latent[:,source]+.35*latent[:,operator]+.25*rng.normal(size=256))
 X=np.column_stack(columns); X=(X-X.mean(0))/X.std(0); names=tuple(name for row in CORE_GRID for name in row); groups={f"source_{i}":tuple(range(3*i,3*i+3)) for i in range(3)}
 prepared=PreparedArm(X,names,groups,frozenset(),np.zeros(9),np.ones(9))
 rook=build_innovation_map(prepared,ARMS[0]); nonrook=build_innovation_map(prepared,ARMS[1])
 assert rook.coefficients.shape==(9,9) and nonrook.coefficients.shape==(9,9)
 for target in range(9):
  row,col=divmod(target,3); rook_support=set(np.flatnonzero(np.abs(rook.coefficients[target])>1e-12)); nonrook_support=set(np.flatnonzero(np.abs(nonrook.coefficients[target])>1e-12))
  assert rook_support=={peer for peer in range(9) if peer!=target and (peer//3==row or peer%3==col)}
  assert nonrook_support=={peer for peer in range(9) if peer//3!=row and peer%3!=col}
 assert np.isfinite(rook.matrix).all() and rook.diagnostics["matrix_condition"]<1e6
 config=ContinuousDeemConfig(epochs=1,mala_steps=1); full=InnovationEnergy(prepared,config,0,rook); blend=BlendedInnovationEnergy(prepared,config,0,rook)
 import torch
 tensor=torch.as_tensor(X,dtype=torch.float64); expected=(1-BLEND)*tensor+BLEND*full.input_transform(tensor)
 assert torch.equal(blend.input_transform(tensor),expected)
 assert torch.isfinite(blend.free_energy(tensor)).all()
 print("deem_b3_crossed_innovation mechanical tests: PASS")
if __name__=="__main__": main()
