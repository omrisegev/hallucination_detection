"""Label-blind agent telemetry: one teacher-forced pass, atomic resume, no evaluation."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"cluster"))
from spectral_utils.agent_bench.contracts import validate
from spectral_utils.agent_bench.render_qwen3 import tokenize
from spectral_utils.external_generalization.artifacts import RecordStore, atomic_json, file_hash
from spectral_utils.external_generalization.contracts import digest
from spectral_utils.external_generalization.budget import authorize, smoke_indices
from run_external_telemetry import alignment_gate, collect_quantities as external_quantities

MODEL = "Qwen/Qwen3-8B"
REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
STOP = False


def stop(*_):
    global STOP
    STOP = True


def code_identity():
    paths = ("cluster/run_agent_telemetry.py", "cluster/run_external_telemetry.py", "cluster/backfill_views.py",
             "spectral_utils/agent_bench/contracts.py", "spectral_utils/agent_bench/render_qwen3.py",
             "spectral_utils/external_generalization/artifacts.py", "spectral_utils/external_generalization/budget.py")
    return {p:file_hash(ROOT/p) for p in paths}


def make_items(answers, tokenizer, max_context=32768):
    return [tokenize(validate(row),tokenizer,max_context=max_context) for row in answers]


def validate_record(payload, item):
    import numpy as np
    n=len(item["gen_ids"])
    for k in ("actual_token_logprobs", "token_entropy_full", "token_entropies", "token_logsumexp", "token_spilled_energies"):
        a=np.asarray(payload[k])
        if a.shape!=(n,) or not np.isfinite(a).all():
            raise ValueError(f"{item['uid']}: invalid {k}")
    indices=[i for a,b in item["step_token_spans"] for i in range(a,b)]
    if payload["top_k_token_indices"]!=indices:
        raise ValueError("top-k token map differs")
    top=payload["top_k_logprobs"]
    expected=(len(indices),50)
    actual_ids=np.asarray(top["ids"]).shape if indices else (0,50) if top["ids"]==[] else None
    actual_lps=np.asarray(top["logprobs"]).shape if indices else (0,50) if top["logprobs"]==[] else None
    if actual_ids!=expected or actual_lps!=expected:
        raise ValueError("top-k shape mismatch")
    if not np.isfinite(top["logprobs"]).all():
        raise ValueError("nonfinite top-k telemetry")
    for k in ("token_offsets", "step_char_spans", "step_token_spans", "context_spans"):
        if digest(payload[k])!=digest(item[k]):
            raise ValueError(f"record span mismatch: {k}")
    np.testing.assert_allclose(-np.asarray(payload["actual_token_logprobs"]),payload["token_spilled_energies"],atol=0,rtol=0)


def collect_quantities(model, item):
    import numpy as np
    # Same numerical implementation as the established external collection.
    payload=external_quantities(model,item)
    indices=[i for a,b in item["step_token_spans"] for i in range(a,b)]
    top=payload["top_k_logprobs"]
    payload["top_k_logprobs"]={k:np.asarray(v)[indices].reshape(len(indices),50).tolist() for k,v in top.items()}
    payload.update({k:item[k] for k in ("full_text","gen_token_offsets","context_spans",
                    "span_coordinate_system","first_message_token_count","step_details","benchmark","subset",
                    "source_id","group","render_mode","step_message_indices","step_entry_indices","entry_to_history")})
    payload["top_k_token_indices"]=indices
    payload["step_think_block_inserted"]=[d["think_block_inserted"] for d in item["step_details"]]
    payload["step_has_source_reasoning"]=[d["has_source_reasoning"] for d in item["step_details"]]
    payload["reasoning_span"]=[d["reasoning_token_span"] for d in item["step_details"]]
    payload["content_span"]=[d["content_token_span"] for d in item["step_details"]]
    validate_record(payload,item)
    return payload


def run(items, store, scorer, selected, *, memory=lambda:0):
    measurements=[]
    for index in selected:
        if STOP:
            return False,measurements
        item=items[index]
        previous=store.get(item["uid"])
        if previous is not None:
            validate_record(previous["telemetry"],item)
            measurements.append(previous["measurement"])
            continue
        start,cpu=time.perf_counter(),time.process_time()
        payload=scorer(item)
        validate_record(payload,item)
        measurement={"uid":item["uid"],"context_tokens":len(item["prompt_ids"])+len(item["gen_ids"]),
            "answer_tokens":len(item["gen_ids"]),"seconds":time.perf_counter()-start,
            "cpu_seconds":time.process_time()-cpu,"peak_gpu_bytes":memory(),"generation_measured":False,
            "bytes":len(json.dumps(payload).encode())}
        store.put(item["uid"],{"telemetry":payload,"measurement":measurement})
        measurements.append(measurement)
        print(json.dumps(measurement),flush=True)
    return True,measurements


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--answers",type=Path,required=True)
    p.add_argument("--benchmark",choices=("apb","tprm","aft"),required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--mode",choices=("smoke","full"),default="smoke")
    p.add_argument("--preflight",type=Path,required=True)
    p.add_argument("--protocol",type=Path,required=True)
    p.add_argument("--budget-decision",type=Path)
    p.add_argument("--estimate",type=Path)
    p.add_argument("--model-path",type=Path,required=True,help="Offline pinned snapshot directory")
    p.add_argument("--max-context",type=int,default=32768)
    args=p.parse_args()
    if args.max_context!=32768:
        raise ValueError("v1 requires exactly 32768 tokens, no rope scaling")
    protocol=json.loads(args.protocol.read_text())
    if protocol.get("models",{}).get(MODEL)!=REVISION:
        raise ValueError("protocol model revision mismatch")
    preflight=json.loads(args.preflight.read_text())
    if preflight.get("code_identity")!=code_identity():
        raise ValueError("preflight covers different code")
    if preflight.get("answers_sha256",{}).get(args.benchmark)!=file_hash(args.answers):
        raise ValueError("preflight covers different inputs")
    if any(preflight.get("gates",{}).get(g)!="PASS" for g in ("G1","G2","G3","G4","G5","G6","LENGTH")):
        raise ValueError("all CPU gates must pass before model load")
    if preflight.get("reviewed_by")!="Claude" or preflight.get("gpu_approved_by")!="user":
        raise ValueError("GPU collection requires recorded Claude review and Omri's go")
    estimate=json.loads(args.estimate.read_text()) if args.estimate else None
    decision=json.loads(args.budget_decision.read_text()) if args.budget_decision else None
    if args.mode=="full" and (estimate is None or decision is None or
            estimate["estimated_gpu_hours"]>decision.get("max_gpu_hours",0)):
        raise ValueError("estimate missing or exceeds approved budget")
    authorize(args.mode,preflight=preflight,protocol_hash=file_hash(args.protocol),
              estimate_hash=file_hash(args.estimate) if args.estimate else None,decision=decision)
    if args.mode=="smoke" and (int(os.environ.get("SLURM_GPUS_ON_NODE","1"))!=1 or
            int(os.environ.get("SLURM_TIMELIMIT","60"))>60):
        raise ValueError("smoke limited to one GPU-hour")
    # Snapshot pin is checked from the HF snapshot path, not a mutable model alias.
    if args.model_path.name!=REVISION:
        raise ValueError("model-path must be the pinned HF snapshots/<revision> directory")
    if preflight.get("tokenizer_sha256") != file_hash(args.model_path/"tokenizer.json"):
        raise ValueError("tokenizer differs from CPU preflight")
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    tok=AutoTokenizer.from_pretrained(args.model_path,local_files_only=True,use_fast=True)
    answers=[json.loads(l) for l in args.answers.read_text(encoding="utf8").splitlines()]
    if not answers or len({r["uid"] for r in answers})!=len(answers) or any(r["benchmark"]!=args.benchmark for r in answers):
        raise ValueError("empty/duplicate/mixed-benchmark input")
    items=make_items(answers,tok,args.max_context)
    excluded=[{"uid":i["uid"],"tokens":len(i["prompt_ids"])+len(i["gen_ids"])} for i in items if i["overlength"]]
    for subset in {i["subset"] for i in items}:
        rows=[i for i in items if i["subset"]==subset]
        if sum(i["overlength"] for i in rows)>.05*len(rows):
            raise ValueError(f"subset exceeds 5% overlength: {subset}")
    items=[i for i in items if not i["overlength"]]
    lengths=[len(i["prompt_ids"])+len(i["gen_ids"]) for i in items]
    selected=smoke_indices(lengths,[i["uid"] for i in items]) if args.mode=="smoke" else list(range(len(items)))
    identity={"schema":1,"benchmark":args.benchmark,"model":MODEL,"revision":REVISION,"dtype":"bfloat16","attn":"sdpa",
        "answers_sha256":file_hash(args.answers),"protocol_sha256":file_hash(args.protocol),"code_identity":code_identity(),
        "chat_template_sha256":digest(tok.chat_template),"tokenizer_sha256":file_hash(args.model_path/"tokenizer.json"),
        "mode":args.mode,"max_context":args.max_context,"selected_uids":[items[i]["uid"] for i in selected]}
    # Acquire the writer lock BEFORE any run reports can be replaced.
    with RecordStore(args.out/"records",identity) as store:
        atomic_json(args.out/"TOKENIZATION.json",{"identity":identity,"lengths":lengths,"excluded":excluded,"truncated":0})
        loaded=time.perf_counter()
        model=AutoModelForCausalLM.from_pretrained(args.model_path,local_files_only=True,
            torch_dtype=torch.bfloat16,device_map="auto",attn_implementation="sdpa").eval()
        if getattr(model.config,"rope_scaling",None):
            raise ValueError("rope scaling prohibited")
        load_seconds=time.perf_counter()-loaded
        torch.cuda.reset_peak_memory_stats()
        gate=alignment_gate(model,items[min(range(len(items)),key=lambda i:lengths[i])],model_id=MODEL,revision=REVISION)
        atomic_json(args.out/"ALIGNMENT_GATE.json",gate)
        complete,measurements=run(items,store,lambda i:collect_quantities(model,i),selected,memory=torch.cuda.max_memory_allocated)
        atomic_json(args.out/"TIMING.json",{"identity":identity,"complete":complete,"load_seconds":load_seconds,
                    "measurements":measurements,"quality_evaluated":False,"job_id":os.environ.get("SLURM_JOB_ID")})
    return 0 if complete else 85


if __name__=="__main__":
    raise SystemExit(main())
