#!/usr/bin/env python3
"""Repair P3 inner-IU source selectors after interrupted report build."""
from __future__ import annotations
import csv,json,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts.reasoning_localization import run_phase1_baseline as p1  # noqa:E402
from scripts.reasoning_localization.build_reasoning_localization_report import REPORTING  # noqa:E402
from scripts.reasoning_localization.register_phase3_inner_iu import EXP  # noqa:E402
def main():
 path=p1.PROGRAM_ROOT/"CONTRASTS_LONG.csv"
 with path.open(newline="") as h:rows=list(csv.DictReader(h));fields=list(rows[0])
 changed=0
 for row in rows:
  if row["experiment_id"]!=EXP:continue
  selector=row["source_row_selector"]
  selector=selector.replace("left=","left_variant_id=").replace("right=","right_variant_id=").replace("metric=","metric_id=")
  if selector!=row["source_row_selector"]:row["source_row_selector"]=selector;changed+=1
 with path.open("w",newline="") as h:w=csv.DictWriter(h,fieldnames=fields,lineterminator="\n");w.writeheader();w.writerows(rows)
 build=REPORTING.prepare_build(p1.PROGRAM_ROOT,REPO);REPORTING.write_build(p1.PROGRAM_ROOT,build);print(json.dumps({"changed":changed,"report_sha256":build.manifest["output"]["sha256"]},indent=2))
if __name__=="__main__":main()
