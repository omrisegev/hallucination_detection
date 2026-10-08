"""Extract two traceable ProcessBench teaching examples from frozen arrays."""
from pathlib import Path
import hashlib, json, pickle
import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\omris\TAU\hallucination_detection")
RUN = ROOT / "results/monday_presentation_panels_v1/run_20261006"
EVIDENCE = ROOT / ".worktrees/readout-quickest-detection-v1/results/step_evidence_v1"
SCORES = ROOT / "results/monday_candidate_panel_v1/run_20261004/candidate_scores/pb_gsm8k_q8__B16__channels_cell.npz"
RAW = ROOT / "dataset_cache/repgrid/pb_qwen3_8b/processbench_gsm8k.pkl"

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    roster = pd.read_csv(EVIDENCE / "OOF_ANSWERS.csv")
    roster = roster[roster.cell.astype(str).eq("pb_gsm8k_q8")]
    with np.load(EVIDENCE / "OOF_STEP_SCORES.npz", allow_pickle=False) as z:
        gold_offsets = z["offsets"].astype(np.int64)
        mindgap = z["mindgap"].astype(float)
        ids = roster.id.astype(str).tolist()
        targets = roster.target.astype(int).to_numpy()
    with np.load(SCORES, allow_pickle=False) as z:
        idx = z["answer_indices"].astype(int)
        off = z["offsets"].astype(np.int64)
        if not np.array_equal(idx, roster.index.to_numpy(dtype=int)):
            raise RuntimeError("candidate-score indices do not match the frozen Q8 roster")
        score = z["HEM_MOM__score"].astype(float)
        az = z["HEM_MOM__ANSWER_Z_Q80__flags"].astype(bool)
        r2 = z["HEM_MOM__R2_COUNT__flags"].astype(bool)
    with RAW.open("rb") as f:
        raw = pickle.load(f)
    by_id = {str(v["id"]): v for v in raw.values()}

    records=[]
    for j, (global_i, target) in enumerate(zip(idx, targets)):
        a,b=map(int,off[j:j+2]); ga,gb=map(int,gold_offsets[global_i:global_i+2])
        answer_id=ids[j].split("::")[-1]
        item=by_id.get(answer_id)
        if item is None or len(item.get("steps",[])) != b-a or len(item.get("steps",[])) != gb-ga:
            raise RuntimeError(f"raw answer alignment failed: {ids[j]}")
        if int(item["label"]) != int(target):
            raise RuntimeError(f"label mismatch for {ids[j]}")
        s=score[a:b]; f=az[a:b]; f2=r2[a:b]; m=mindgap[ga:gb]
        records.append({
            "j":j,"id":ids[j],"uid":str(roster.iloc[j].uid),"source_group":str(roster.iloc[j].source_group),
            "target_first_error_step_0based":int(target),"n_steps":int(b-a),
            "hem_argmax_0based":int(np.argmax(s)),"hem_q80_flags_0based":np.flatnonzero(f).astype(int).tolist(),
            "hem_r2_flags_0based":np.flatnonzero(f2).astype(int).tolist(),"mtg_argmax_0based":int(np.argmax(m)),
            "hem_scores_rounded":np.round(s,4).tolist(),
            "raw_record_sha256":hashlib.sha256(json.dumps({"problem":item["problem"],"steps":item["steps"]},ensure_ascii=False).encode("utf-8")).hexdigest(),
            "final_answer_correct":bool(item.get("final_answer_correct",False)),
        })
    errors=[r for r in records if r["target_first_error_step_0based"]>0 and r["hem_argmax_0based"]==r["target_first_error_step_0based"] and r["mtg_argmax_0based"]!=r["target_first_error_step_0based"]]
    if not errors:
        errors=[r for r in records if r["target_first_error_step_0based"]>0 and r["hem_argmax_0based"]==r["target_first_error_step_0based"]]
    if not errors:
        raise RuntimeError("No nontrivial HEM first-error hit found")
    # Stable lowest numeric GSM8K ID among examples meeting the stated teaching criterion.
    def numeric_id(r): return int(r["id"].rsplit("-",1)[1])
    error=min(errors,key=numeric_id)
    clean=[r for r in records if r["target_first_error_step_0based"]==-1 and r["hem_q80_flags_0based"]]
    if not clean: raise RuntimeError("No clean-answer q80 false-flag example found")
    clean_without_r2=[r for r in clean if not r["hem_r2_flags_0based"]]
    if clean_without_r2: clean=clean_without_r2
    clean=min(clean,key=lambda r:(r["n_steps"],numeric_id(r)))
    out={
        "status":"PASS",
        "purpose":"Teaching examples only; selected by deterministic criteria, not representative performance estimates.",
        "method":"HEM_MOM_cc (B16, ProcessBench GSM8K Qwen3-8B cell)",
        "alignment_checks":{"roster_rows":len(records),"score_indices_exact":True,"step_lengths_exact":True,"first_error_labels_exact":True,"raw_records_joined":len(records)},
        "selection":{"error":"lowest numeric GSM8K id with first error after step 0, HEM argmax exactly at first error; prefer a case where MTG argmax differs.","clean":"prefer a clean answer with >=1 ANSWER_Z_Q80 flag and no R2 flags; then fewest steps and lowest numeric GSM8K id."},
        "examples":{"error":error,"clean_false_flag":clean},
        "source_hashes":{"OOF_ANSWERS.csv":sha(EVIDENCE/"OOF_ANSWERS.csv"),"OOF_STEP_SCORES.npz":sha(EVIDENCE/"OOF_STEP_SCORES.npz"),"candidate_scores.npz":sha(SCORES),"processbench_gsm8k.pkl":sha(RAW)},
    }
    (RUN/"WORKED_EXAMPLES.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    for name,r in out["examples"].items():
        print(name,r["id"],"target",r["target_first_error_step_0based"],"hem_argmax",r["hem_argmax_0based"],"mtg_argmax",r["mtg_argmax_0based"],"q80",r["hem_q80_flags_0based"],"r2",r["hem_r2_flags_0based"],"steps",r["n_steps"])

if __name__=="__main__": main()
