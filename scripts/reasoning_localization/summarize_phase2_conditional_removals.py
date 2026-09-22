#!/usr/bin/env python3
"""Build the deterministic Phase-2C removal summary table and plot."""

from __future__ import annotations

import csv, json, sys
from pathlib import Path

REPO=Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from spectral_utils.reconstruction_benchmark.io import atomic_write_json, sha256_file  # noqa:E402
from scripts.reasoning_localization import run_phase1_baseline as p1  # noqa:E402
from scripts.reasoning_localization import run_phase2_conditional as runner  # noqa:E402
from scripts.reasoning_localization.build_reasoning_localization_report import REPORTING  # noqa:E402

ROSTER=(
 ("Entropy level","P2C_F6_MINUS_ENTROPY_LEVEL","family"),
 ("Entropy dynamics","P2C_F6_MINUS_ENTROPY_DYNAMICS","family"),
 ("Sampled energy","P2C_F6_MINUS_SAMPLED_ENERGY","family"),
 ("Partition energy","P2C_F6_MINUS_PARTITION_ENERGY","family"),
 ("Top-k distribution","P2C_F6_MINUS_TOPK_DISTRIBUTION","family"),
 ("Entropy SWVar16 view","P2C_F6_MINUS_ENTROPY_SWVAR16_VIEW","view"),
 ("Entropy CUSUM view","P2C_F6_MINUS_ENTROPY_CUSUM_VIEW","view"),
 ("Sampled level view","P2C_F6_MINUS_SAMPLED_LEVEL_VIEW","view"),
 ("Partition level view","P2C_F6_MINUS_PARTITION_LEVEL_VIEW","view"),
)
OUT=runner.ROOT/"removal_summary"

def read_csv(path):
 with path.open(newline="") as h:return list(csv.DictReader(h))

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 rows=[]
 for label,variant,scope in ROSTER:
  er=runner.output_root(variant)/"evaluation";s=json.loads((er/"SUMMARY.json").read_text());c=s["primary_contrast"]
  point=-float(c["candidate_minus_parent_delta"]);lo=-float(c["ci_high"]);hi=-float(c["ci_low"])
  exact=-float(s["exact_error_delta"]);clean=-float(s["clean_abstention_delta"])
  if lo>.003:stat="SUPPORTED_CONTRIBUTION"
  elif point>0:stat="PROMISING_UNCONFIRMED"
  else:stat="INCONCLUSIVE"
  # CI crossing zero is explicitly uncertainty, never rejection.
  if lo<=0<=hi: stat="PROMISING_UNCONFIRMED" if point>0 else "INCONCLUSIVE"
  rows.append({"label":label,"variant_id":variant,"scope":scope,"parent_macro_f1":s["parent_macro_f1"],"ablated_macro_f1":s["candidate_macro_f1"],
   "contribution_macro_f1":point,"simultaneous_ci_low":lo,"simultaneous_ci_high":hi,"exact_error_contribution":exact,"clean_abstention_contribution":clean,
   "cell_support":int(c["losses"])+int(c["ties"]),"statistical_interpretation":stat,"source_summary":str((er/"SUMMARY.json").relative_to(REPO)),"source_sha256":sha256_file(er/"SUMMARY.json")})
 csv_path=OUT/"REMOVAL_SUMMARY.csv";fields=list(rows[0])
 with csv_path.open("w",newline="") as h:w=csv.DictWriter(h,fieldnames=fields,lineterminator="\n");w.writeheader();w.writerows(rows)

 def esc(x):return str(x).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
 width,height=1200,650;left,right=255,1130;top,row_h=105,49;xmin,xmax=-4.5,4.5
 sx=lambda v:left+(float(v)-xmin)/(xmax-xmin)*(right-left)
 parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
  '<rect width="100%" height="100%" fill="white"/>','<style>text{font-family:Arial,sans-serif;fill:#222}.small{font-size:12px}.label{font-size:13px}.title{font-size:20px;font-weight:700}.sub{font-size:14px}.family{stroke:#2b6f9f;fill:#2b6f9f}.view{stroke:#777;fill:#777}</style>',
  '<text x="40" y="35" class="title">Phase 2C removal study — component contribution to ProcessBench macro-F1</text>',
  '<text x="40" y="60" class="sub">Contribution = full five-family/top-10 parent (F1 0.35426) minus ablated variant; 13-contrast simultaneous intervals</text>']
 for tick in range(-4,5):
  x=sx(tick);parts.append(f'<line x1="{x:.1f}" y1="80" x2="{x:.1f}" y2="555" stroke="#e1e1e1"/>');parts.append(f'<text x="{x:.1f}" y="580" text-anchor="middle" class="small">{tick:+d}</text>')
 parts.append(f'<line x1="{sx(0):.1f}" y1="80" x2="{sx(0):.1f}" y2="555" stroke="#111" stroke-width="1.5"/>')
 parts.append(f'<line x1="{sx(.3):.1f}" y1="80" x2="{sx(.3):.1f}" y2="555" stroke="#8b5a2b" stroke-dasharray="5 4"/><text x="{sx(.3)+5:.1f}" y="92" class="small">+0.3 practical bound</text>')
 for i,r in enumerate(rows):
  y=top+i*row_h;klass="family" if r["scope"]=="family" else "view";p=100*r["contribution_macro_f1"];lo=100*r["simultaneous_ci_low"];hi=100*r["simultaneous_ci_high"]
  parts.append(f'<text x="245" y="{y+5}" text-anchor="end" class="label">{i+1}. {esc(r["label"])}</text>')
  parts.append(f'<line x1="{sx(lo):.1f}" y1="{y}" x2="{sx(hi):.1f}" y2="{y}" class="{klass}" stroke-width="2"/><line x1="{sx(lo):.1f}" y1="{y-5}" x2="{sx(lo):.1f}" y2="{y+5}" class="{klass}"/><line x1="{sx(hi):.1f}" y1="{y-5}" x2="{sx(hi):.1f}" y2="{y+5}" class="{klass}"/><circle cx="{sx(p):.1f}" cy="{y}" r="5" class="{klass}"/>')
  parts.append(f'<text x="1140" y="{y+4}" class="small">{p:+.2f} [{lo:+.2f}, {hi:+.2f}]</text>')
 parts.extend(['<text x="690" y="610" text-anchor="middle" class="sub">Contribution to macro-F1 (percentage points)</text>',
  '<text x="40" y="635" class="small">Blue: whole family · Gray: individual view · Source: frozen eight-Qwen common population, 20,000 paired grouped bootstrap draws.</text>',
  '<text x="40" y="650" class="small">An interval crossing zero is uncertainty (INCONCLUSIVE or PROMISING_UNCONFIRMED), not rejection.</text>','</svg>'])
 svg=OUT/"REMOVAL_CONTRIBUTION.svg";svg.write_text("".join(parts),encoding="utf-8")
 manifest={"schema":"reasoning-localization-p2c-removal-summary-v1","status":"COMPLETE","population_id":"current_common_eight_qwen","parent":runner.PARENT,
  "rows":len(rows),"interpretation_rule":"CI crossing zero is INCONCLUSIVE or PROMISING_UNCONFIRMED, never rejection",
  "artifacts":[{"path":p.name,"sha256":sha256_file(p),"bytes":p.stat().st_size} for p in (csv_path,svg)]}
 manifest["payload_sha256"]=runner.c1.payload_sha(manifest);atomic_write_json(OUT/"MANIFEST.json",manifest)
 ep=p1.PROGRAM_ROOT/"EXPERIMENT_REGISTRY.json";ex=json.loads(ep.read_text());e=next(r for r in ex["experiments"] if r["experiment_id"]=="P2_CONDITIONAL_ABLATION");e["removal_substage_status"]="COMPLETE";e["next_variant"]="P2C_F6_PLUS_STRUCTURAL_CONTROL";atomic_write_json(ep,ex)
 build=REPORTING.prepare_build(p1.PROGRAM_ROOT,REPO);REPORTING.write_build(p1.PROGRAM_ROOT,build)
 print(json.dumps({"rows":len(rows),"plot":str(svg),"manifest":str(OUT/"MANIFEST.json"),"report_sha256":build.manifest["output"]["sha256"]},indent=2))
if __name__=="__main__":main()
