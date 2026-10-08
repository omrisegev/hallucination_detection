"""Validate local links and example/source alignment in the Codex slide supplement."""
from pathlib import Path
from html.parser import HTMLParser
import hashlib, json

ROOT=Path(r"C:\Users\omris\TAU\hallucination_detection")
HTML=ROOT/"docs/meetings/advisor_prep_2026-10-05/NEXT_STAGE_SUPPLEMENT_HE.html"
EXAMPLES=ROOT/"results/monday_presentation_panels_v1/run_20261006/WORKED_EXAMPLES.json"

class Links(HTMLParser):
    def __init__(self): super().__init__(); self.local=[]
    def handle_starttag(self,tag,attrs):
        for k,v in attrs:
            if k in ("href","src") and v and not v.startswith(("#","https://","http://","mailto:")):
                self.local.append(v)

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    parser=Links(); content=HTML.read_text(encoding="utf-8"); parser.feed(content)
    missing=[]
    for ref in parser.local:
        target=(HTML.parent/ref).resolve()
        if not target.exists(): missing.append(ref)
    record=json.loads(EXAMPLES.read_text(encoding="utf-8"))
    assert record["status"]=="PASS"
    assert record["alignment_checks"]=={"roster_rows":400,"score_indices_exact":True,"step_lengths_exact":True,"first_error_labels_exact":True,"raw_records_joined":400}
    assert "steps" not in record["examples"]["error"] and "problem" not in record["examples"]["error"]
    for rel,digest in record["source_hashes"].items():
        paths={"OOF_ANSWERS.csv":ROOT/".worktrees/readout-quickest-detection-v1/results/step_evidence_v1/OOF_ANSWERS.csv",
               "OOF_STEP_SCORES.npz":ROOT/".worktrees/readout-quickest-detection-v1/results/step_evidence_v1/OOF_STEP_SCORES.npz",
               "candidate_scores.npz":ROOT/"results/monday_candidate_panel_v1/run_20261004/candidate_scores/pb_gsm8k_q8__B16__channels_cell.npz",
               "processbench_gsm8k.pkl":ROOT/"dataset_cache/repgrid/pb_qwen3_8b/processbench_gsm8k.pkl"}
        if sha(paths[rel])!=digest: raise RuntimeError("source hash changed: "+rel)
    report={"status":"PASS" if not missing else "FAIL","html":str(HTML.relative_to(ROOT)).replace("\\","/"),
            "local_links_checked":len(parser.local),"missing_links":missing,
            "worked_examples_alignment":record["alignment_checks"],"source_hashes_match":True,
            "artifacts":{"html_sha256":sha(HTML),"claims_ledger_sha256":sha(ROOT/"docs/meetings/advisor_prep_2026-10-05/CLAIMS_LEDGER_NEXT_STAGE_HE.md"),"worked_examples_sha256":sha(EXAMPLES)}}
    out=ROOT/"docs/meetings/advisor_prep_2026-10-05/NEXT_STAGE_VALIDATION_20261006.json"
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if missing: raise RuntimeError("Missing local links")

if __name__=="__main__": main()
