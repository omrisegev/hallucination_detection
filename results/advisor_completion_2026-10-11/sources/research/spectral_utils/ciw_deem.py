"""Canonical public alias for CIW-DEEM v1.

CIW-DEEM = Cross-fitted Innovation-Weighted DEEM.  The v1 method is the
five-seed M1 rook/static-R2 arm frozen by the input-gate experiment.
"""
from __future__ import annotations
from typing import Sequence
import numpy as np
from .deem_b3_contract_ablation import PreparedArm
from .deem_b3_unsupervised_input_gate import fit_unsupervised_gate
from .residual_graph_deem import ContinuousDeemConfig

METHOD_ID="CIW_DEEM_V1"
DISPLAY_NAME="CIW-DEEM"
FROZEN_ARM="M1_ROOK_STATIC_R2"

def fit_ciw_deem(prepared:PreparedArm,group_ids:Sequence[str],raw_lengths:np.ndarray,*,seed:int=0,config:ContinuousDeemConfig|None=None):
 return fit_unsupervised_gate(prepared,group_ids,raw_lengths,FROZEN_ARM,seed=seed,config=config)

__all__=["DISPLAY_NAME","FROZEN_ARM","METHOD_ID","fit_ciw_deem"]
