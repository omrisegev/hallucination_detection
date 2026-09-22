#!/usr/bin/env python3
"""Register the complete Phase-2C verdict and update its plot contract."""
import json,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from spectral_utils.reconstruction_benchmark.io import atomic_write_json  # noqa:E402
from scripts.reasoning_localization import run_phase1_baseline as p1  # noqa:E402
cp=p1.PROGRAM_ROOT/"CLAIMS.json";c=json.loads(cp.read_text());cid="CLAIM_P2C_C8_OUTER_EXPERT"
if any(r["claim_id"]==cid for r in c["claims"]):raise RuntimeError("claim exists")
c["claims"].append({"claim_id":cid,"text":"The C8 outer expert is the raw-best Phase-2C candidate with a positive one-point macro-F1 delta, but its simultaneous interval crosses zero and clean abstention regresses; it is promising unconfirmed rather than promoted or rejected.",
 "verdict":"PROMISING_UNCONFIRMED","task_scope":"Current common eight-Qwen ProcessBench first-error localization development population.","claim_boundary":"Does not establish a supported improvement, PRMBench transfer, or eligibility for outcome-selected Phase-3 fusion; independent confirmation is required.","fresh_confirmation_required":True,
 "worst_case_behavior":"Three of eight cells lose, the worst-cell macro-F1 delta is -0.00919, and clean abstention falls by 0.02249.",
 "statistical_summary":{"metric":"macro_f1","point_delta":0.010735353272928738,"ci_low":-0.002481387280222966,"ci_high":0.024254992411723665,"benefit_bound":0.003,"harm_bound":-0.005,"bound_basis":"Registered Phase-2C candidate and practical boundaries.","multiplicity":"Bonferroni simultaneous interval across the frozen thirteen-contrast family."},
 "evidence_refs":["PLOT_P2C_REMOVAL_FOREST","CONTRAST:P2C_F6_PLUS_C8_OUTER_EXPERT:P2C_F6_TOP10_REFERENCE","TABLE_GATES"]})
atomic_write_json(cp,c)
pp=p1.PROGRAM_ROOT/"PLOT_MANIFEST.json";p=json.loads(pp.read_text());plot=next(r for r in p["plots"] if r["plot_id"]=="PLOT_P2C_REMOVAL_FOREST")
plot.update({"title":"Phase 2C complete conditional-variant forest","caption":"All thirteen candidate-minus-parent deltas under the frozen simultaneous family. Removal rows are interpreted in reverse for component contribution; zero-crossing intervals remain unresolved, not rejected.",
 "selection_rule":"All thirteen completed family/view removals, structural control, formulation swap, and C7/C8 insertions versus the exact five-family/top-ten parent.",
 "legend":["Point and line = candidate minus parent and simultaneous interval","Removal rows reverse sign for component contribution","Zero-crossing intervals are promising or inconclusive, never automatic rejection"]})
atomic_write_json(pp,p);print(cid)
