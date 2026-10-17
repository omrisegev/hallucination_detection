#!/usr/bin/env python3
"""Finalize the complete thirteen-arm Phase-2C conditional study and plot."""
from __future__ import annotations
import csv,json,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from spectral_utils.reconstruction_benchmark.io import atomic_write_json,sha256_file  # noqa:E402
from scripts.reasoning_localization import run_phase1_baseline as p1  # noqa:E402
from scripts.reasoning_localization import run_phase2_conditional as runner  # noqa:E402
from scripts.reasoning_localization.build_reasoning_localization_report import REPORTING  # noqa:E402

ROSTER=(
 ("Remove entropy level","P2C_F6_MINUS_ENTROPY_LEVEL","family removal"),("Remove entropy dynamics","P2C_F6_MINUS_ENTROPY_DYNAMICS","family removal"),
 ("Remove sampled energy","P2C_F6_MINUS_SAMPLED_ENERGY","family removal"),("Remove partition energy","P2C_F6_MINUS_PARTITION_ENERGY","family removal"),
 ("Remove top-k distribution","P2C_F6_MINUS_TOPK_DISTRIBUTION","family removal"),("Add structural family","P2C_F6_PLUS_STRUCTURAL_CONTROL","control"),
 ("Remove SWVar16 view","P2C_F6_MINUS_ENTROPY_SWVAR16_VIEW","view removal"),("Remove CUSUM view","P2C_F6_MINUS_ENTROPY_CUSUM_VIEW","view removal"),
 ("Remove sampled-level view","P2C_F6_MINUS_SAMPLED_LEVEL_VIEW","view removal"),("Remove partition-level view","P2C_F6_MINUS_PARTITION_LEVEL_VIEW","view removal"),
 ("Swap exact C1 SWVar16","P2C_F6_SWAP_C1_SWVAR16","swap"),("Insert C7 EDIS view","P2C_F6_PLUS_C7_EDIS_VIEW","insertion"),
 ("Add C8 outer expert","P2C_F6_PLUS_C8_OUTER_EXPERT","insertion"),)
OUT=runner.ROOT/"final_summary"
COLORS={"family removal":"#2b6f9f","view removal":"#777777","control":"#8b5a2b","swap":"#6d4c8d","insertion":"#2f7d4a"}
def esc(x):return str(x).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
def main():
 OUT.mkdir(parents=True,exist_ok=True);rows=[]
 for label,v,kind in ROSTER:
  sp=runner.output_root(v)/"evaluation/SUMMARY.json";s=json.loads(sp.read_text());c=s["primary_contrast"];point=float(c["candidate_minus_parent_delta"]);lo=float(c["ci_low"]);hi=float(c["ci_high"])
  if lo>.003:status="SUPPORTED_IMPROVEMENT"
  elif hi<-.005:status="SUPPORTED_HARM"
  elif point>0:status="PROMISING_UNCONFIRMED"
  else:status="INCONCLUSIVE"
  rows.append({"label":label,"variant_id":v,"kind":kind,"candidate_macro_f1":s["candidate_macro_f1"],"parent_macro_f1":s["parent_macro_f1"],"candidate_minus_parent_delta":point,
   "simultaneous_ci_low":lo,"simultaneous_ci_high":hi,"exact_error_delta":s["exact_error_delta"],"clean_abstention_delta":s["clean_abstention_delta"],"wins":c["wins"],"ties":c["ties"],"losses":c["losses"],
   "statistical_interpretation":status,"source_summary":str(sp.relative_to(REPO)),"source_sha256":sha256_file(sp)})
 csvp=OUT/"PHASE2C_SUMMARY.csv";fields=list(rows[0])
 with csvp.open("w",newline="") as h:w=csv.DictWriter(h,fieldnames=fields,lineterminator="\n");w.writeheader();w.writerows(rows)
 W,H=1450,820;forest_l,forest_r=255,860;scatter_l,scatter_r=980,1395;top,rowh=105,47;xmin,xmax=-4.5,3.0;sx=lambda v:forest_l+(v-xmin)/(xmax-xmin)*(forest_r-forest_l)
 symin,symax=-6.5,2.5;xxmin,xxmax=-4.0,4.0;ssx=lambda v:scatter_l+(v-xxmin)/(xxmax-xxmin)*(scatter_r-scatter_l);ssy=lambda v:690-(v-symin)/(symax-symin)*520
 p=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><rect width="100%" height="100%" fill="white"/>',
 '<style>text{font-family:Arial,sans-serif;fill:#222}.title{font-size:21px;font-weight:700}.sub{font-size:13px}.label{font-size:12px}.small{font-size:11px}</style>',
 '<text x="35" y="34" class="title">Phase 2C complete — conditional variants versus frozen five-family/top-10 parent</text>',
 '<text x="35" y="58" class="sub">Left: macro-F1 paired deltas and 13-contrast simultaneous intervals. Right: exact-error versus clean-abstention deltas.</text>']
 for t in range(-4,4):x=sx(t);p.extend([f'<line x1="{x:.1f}" y1="80" x2="{x:.1f}" y2="710" stroke="#e3e3e3"/>',f'<text x="{x:.1f}" y="735" text-anchor="middle" class="small">{t:+d}</text>'])
 p.append(f'<line x1="{sx(0):.1f}" y1="80" x2="{sx(0):.1f}" y2="710" stroke="#111" stroke-width="1.4"/>')
 for i,r in enumerate(rows):
  y=top+i*rowh;pt=100*r["candidate_minus_parent_delta"];lo=100*r["simultaneous_ci_low"];hi=100*r["simultaneous_ci_high"];col=COLORS[r["kind"]]
  p.extend([f'<text x="245" y="{y+4}" text-anchor="end" class="label">{i+1}. {esc(r["label"])}</text>',f'<line x1="{sx(lo):.1f}" y1="{y}" x2="{sx(hi):.1f}" y2="{y}" stroke="{col}" stroke-width="2"/>',
   f'<line x1="{sx(lo):.1f}" y1="{y-5}" x2="{sx(lo):.1f}" y2="{y+5}" stroke="{col}"/><line x1="{sx(hi):.1f}" y1="{y-5}" x2="{sx(hi):.1f}" y2="{y+5}" stroke="{col}"/><circle cx="{sx(pt):.1f}" cy="{y}" r="5" fill="{col}"/>',
   f'<text x="870" y="{y+4}" class="small">{pt:+.2f} [{lo:+.2f},{hi:+.2f}]</text>'])
 # secondary scatter
 for t in range(-4,5,2):x=ssx(t);p.extend([f'<line x1="{x:.1f}" y1="80" x2="{x:.1f}" y2="690" stroke="#ececec"/>',f'<text x="{x:.1f}" y="714" text-anchor="middle" class="small">{t:+d}</text>'])
 for t in range(-6,3,2):y=ssy(t);p.extend([f'<line x1="{scatter_l}" y1="{y:.1f}" x2="{scatter_r}" y2="{y:.1f}" stroke="#ececec"/>',f'<text x="{scatter_l-8}" y="{y+4:.1f}" text-anchor="end" class="small">{t:+d}</text>'])
 p.extend([f'<line x1="{ssx(0):.1f}" y1="80" x2="{ssx(0):.1f}" y2="690" stroke="#111"/><line x1="{scatter_l}" y1="{ssy(0):.1f}" x2="{scatter_r}" y2="{ssy(0):.1f}" stroke="#111"/>'])
 for i,r in enumerate(rows):
  x=100*float(r["exact_error_delta"]);y=100*float(r["clean_abstention_delta"]);col=COLORS[r["kind"]];p.extend([f'<circle cx="{ssx(x):.1f}" cy="{ssy(y):.1f}" r="6" fill="{col}"/>',f'<text x="{ssx(x)+7:.1f}" y="{ssy(y)-7:.1f}" class="small">{i+1}</text>'])
 p.extend(['<text x="557" y="770" text-anchor="middle" class="sub">Candidate − parent macro-F1 (percentage points)</text>',
  '<text x="1188" y="750" text-anchor="middle" class="sub">Exact-error delta (percentage points)</text>',f'<text x="955" y="385" transform="rotate(-90 955 385)" text-anchor="middle" class="sub">Clean-abstention delta (percentage points)</text>',
  '<text x="35" y="800" class="small">Blue family removal · Gray view removal · Brown control · Purple swap · Green insertion. CI crossing zero is uncertainty, not rejection.</text>','</svg>'])
 svg=OUT/"PHASE2C_COMPLETE.svg";svg.write_text("".join(p),encoding="utf-8")
 manifest={"schema":"reasoning-localization-p2c-final-summary-v1","status":"COMPLETE","parent":runner.PARENT,"primary_family_size":13,"verdict":"NO_FULL_CONDITIONAL_PROMOTION",
  "raw_best":"P2C_F6_PLUS_C8_OUTER_EXPERT","raw_best_macro_f1":0.3649965053124179,"artifacts":[{"path":q.name,"sha256":sha256_file(q),"bytes":q.stat().st_size} for q in (csvp,svg)]}
 manifest["payload_sha256"]=runner.c1.payload_sha(manifest);atomic_write_json(OUT/"MANIFEST.json",manifest)
 ep=p1.PROGRAM_ROOT/"EXPERIMENT_REGISTRY.json";e=json.loads(ep.read_text());r=next(x for x in e["experiments"] if x["experiment_id"]=="P2_CONDITIONAL_ABLATION");r["execution_status"]="COMPLETE";r["next_variant"]=None;r["verdict"]="NO_FULL_CONDITIONAL_PROMOTION";r["raw_best"]="P2C_F6_PLUS_C8_OUTER_EXPERT";atomic_write_json(ep,e)
 build=REPORTING.prepare_build(p1.PROGRAM_ROOT,REPO);REPORTING.write_build(p1.PROGRAM_ROOT,build);print(json.dumps({"verdict":manifest["verdict"],"raw_best":manifest["raw_best"],"plot":str(svg),"report_sha256":build.manifest["output"]["sha256"]},indent=2))
if __name__=="__main__":main()
