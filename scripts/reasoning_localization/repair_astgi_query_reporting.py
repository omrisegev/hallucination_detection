#!/usr/bin/env python3
"""Remove the phase-wide duplicate ASTGI plot; global lineage remains."""
from __future__ import annotations
import json,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from spectral_utils.reconstruction_benchmark.io import atomic_write_json  # noqa:E402
from scripts.reasoning_localization import run_phase1_baseline as p1  # noqa:E402
from scripts.reasoning_localization.build_reasoning_localization_report import REPORTING  # noqa:E402
def main():
 pp=p1.PROGRAM_ROOT/"PLOT_MANIFEST.json";plots=json.loads(pp.read_text());before=len(plots["plots"]);plots["plots"]=[x for x in plots["plots"] if x["plot_id"]!="PLOT_P3_ASTGI_QUERY_LADDER"]
 if len(plots["plots"])!=before-1:raise RuntimeError("ASTGI duplicate plot not found exactly once")
 atomic_write_json(pp,plots)
 ep=p1.PROGRAM_ROOT/"EXPERIMENT_REGISTRY.json";experiments=json.loads(ep.read_text());row=next(x for x in experiments["experiments"] if x["experiment_id"]=="P3_ASTGI_QUERY_HEADS");row["report_sections"]=["p3_tensor_pipeline","p3_parent_fusion"];atomic_write_json(ep,experiments)
 build=REPORTING.prepare_build(p1.PROGRAM_ROOT,REPO);REPORTING.write_build(p1.PROGRAM_ROOT,build);print(json.dumps({"status":"REPAIRED","report_sha256":build.manifest["output"]["sha256"]},indent=2))
if __name__=="__main__":main()
