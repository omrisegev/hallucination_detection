#!/usr/bin/env python3
"""Register official Phase-2C removal claims and report plot contracts."""
from __future__ import annotations
import json,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from spectral_utils.reconstruction_benchmark.io import atomic_write_json  # noqa:E402
from scripts.reasoning_localization import run_phase1_baseline as p1  # noqa:E402

def main():
 cp=p1.PROGRAM_ROOT/"CLAIMS.json";c=json.loads(cp.read_text());ids={r["claim_id"] for r in c["claims"]}
 additions=[
  {"claim_id":"CLAIM_P2C_LOAD_BEARING_FAMILIES","text":"On the frozen current five-family/top-ten ProcessBench parent, entropy level and top-k distribution have supported positive aggregate conditional contributions, but both expose material exact-error versus clean-abstention tradeoffs and therefore do not pass the full conditional promotion gate.",
   "verdict":"SUPPORTED","task_scope":"Current common eight-Qwen ProcessBench first-error localization development population.","claim_boundary":"Supports aggregate conditional contribution only; does not establish a universal compact feature set, PRMBench transfer, or clean-abstention-safe promotion.","fresh_confirmation_required":True,
   "worst_case_behavior":"Entropy-level and top-k retention each reduce clean-abstention accuracy by about 5.5 percentage points despite improving exact-error localization.",
   "statistical_summary":{"metric":"macro_f1","entropy_level_contribution":0.024645749097507053,"entropy_level_ci":[0.00750596102920069,0.04203096383838138],"topk_distribution_contribution":0.02266947612363779,"topk_distribution_ci":[0.004707831496145314,0.040472579653874456],"multiplicity":"Bonferroni simultaneous intervals across the frozen thirteen-contrast family."},
   "evidence_refs":["PLOT_P2C_REMOVAL_FOREST","TABLE_VARIANTS","TABLE_GATES"]},
  {"claim_id":"CLAIM_P2C_UNRESOLVED_REMOVALS","text":"Partition energy is promising but unconfirmed, while entropy dynamics, sampled energy and the four targeted individual views remain inconclusive under the frozen conditional-removal study; none is excluded merely because its interval crosses zero.",
   "verdict":"PROMISING_UNCONFIRMED","task_scope":"Current common eight-Qwen ProcessBench first-error localization development population.","claim_boundary":"No unresolved family or view earns Phase-3 promotion on this evidence, but uncertainty is not evidence of zero effect or harm.","fresh_confirmation_required":True,
   "worst_case_behavior":"Several point estimates favor removal, while their simultaneous intervals remain compatible with both useful contribution and harm.",
   "statistical_summary":{"metric":"macro_f1","partition_energy_contribution":0.010632515138614407,"partition_energy_ci":[-0.0006757899226087197,0.02232298445751211],"multiplicity":"Bonferroni simultaneous intervals across the frozen thirteen-contrast family."},
   "evidence_refs":["PLOT_P2C_REMOVAL_FOREST","TABLE_VARIANTS","TABLE_GATES"]},
 ]
 for row in additions:
  if row["claim_id"] in ids:raise RuntimeError("claim already registered")
 c["claims"].extend(additions);atomic_write_json(cp,c)
 pp=p1.PROGRAM_ROOT/"PLOT_MANIFEST.json";p=json.loads(pp.read_text());pids={r["plot_id"] for r in p["plots"]}
 plot={"plot_id":"PLOT_P2C_REMOVAL_FOREST","phase":"P2C","kind":"contrast_forest","title":"Phase 2C conditional removal effects","caption":"Candidate-minus-parent deltas for all completed removals; conditional component contribution is the sign-reversed quantity. An interval crossing zero is inconclusive, not rejection.",
  "source_table":"CONTRASTS_LONG.csv","selection":{"experiment_id":"P2_CONDITIONAL_ABLATION","metric_id":"macro_f1","status":"COMPLETE"},
  "legend":["Point and line = ablated candidate minus full parent and simultaneous interval","Negative candidate delta means positive component contribution","Zero-crossing intervals remain unresolved"],
  "comparison_group":"exact current eight-Qwen five-family/top-ten ProcessBench group","bootstrap_definition":"20,000 paired whole-question grouped draws; Bonferroni simultaneous intervals across the frozen thirteen-contrast family.",
  "selection_rule":"All completed family and targeted-view leave-one-out contrasts; insertion and formulation-swap rows are excluded until separately opened.","series_field":"evidence_status","x_field":"delta","y_field":"left_variant_id"}
 if plot["plot_id"] in pids:raise RuntimeError("plot already registered")
 p["plots"].append(plot);atomic_write_json(pp,p);print(json.dumps({"claims":2,"plot_id":plot["plot_id"]},indent=2))
if __name__=="__main__":main()
