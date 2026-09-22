#!/usr/bin/env python3
"""Complete required statistical-summary fields for Phase-2C claims."""
import json,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from spectral_utils.reconstruction_benchmark.io import atomic_write_json  # noqa:E402
from scripts.reasoning_localization import run_phase1_baseline as p1  # noqa:E402
p=p1.PROGRAM_ROOT/"CLAIMS.json";d=json.loads(p.read_text());by={r["claim_id"]:r for r in d["claims"]}
by["CLAIM_P2C_LOAD_BEARING_FAMILIES"]["statistical_summary"]={"metric":"macro_f1","point_delta":0.024645749097507053,"ci_low":0.00750596102920069,"ci_high":0.04203096383838138,"benefit_bound":0.003,"harm_bound":-0.005,"bound_basis":"Registered Phase-2C conditional contribution and practical-harm bounds; entropy-level row shown, with top-k distribution separately recorded in claim text.","multiplicity":"Bonferroni simultaneous interval across the frozen thirteen-contrast family."}
by["CLAIM_P2C_UNRESOLVED_REMOVALS"]["statistical_summary"]={"metric":"macro_f1","point_delta":0.010632515138614407,"ci_low":-0.0006757899226087197,"ci_high":0.02232298445751211,"benefit_bound":0.003,"harm_bound":-0.005,"bound_basis":"Registered Phase-2C conditional contribution and practical-harm bounds.","multiplicity":"Bonferroni simultaneous interval across the frozen thirteen-contrast family."}
atomic_write_json(p,d);print("repaired")
