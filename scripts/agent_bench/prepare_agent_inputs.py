"""Verify source freeze, prepare blind JSONL, and run CPU rendering gates only."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from spectral_utils.agent_bench.contracts import adapt_apb, adapt_tprm, adapt_aft, blind_keys
from spectral_utils.agent_bench.render_qwen3 import render, tokenize, native, HEADER, EMPTY
from spectral_utils.external_generalization.artifacts import atomic_json, file_hash

COUNTS = {"apb": {"bfcl": (250,2590), "gaia_dev": (250,1628), "hotpotqa": (250,734), "tau2": (250,3557)},
          "tprm": {"GTA":118, "bfcl":111, "tooltalk":86, "ToolSandbox":130}}


def verify_sources(directory):
    directory = Path(directory).resolve()
    m = json.loads((directory/"SOURCES.json").read_text())
    paths = {}
    # Verify EVERY byte hash before any source file is parsed.
    for f in m["files"]:
        p = (directory/f["local_path"]).resolve()
        if not p.is_relative_to(directory) or p.stat().st_size != f["bytes"] or file_hash(p) != f["sha256"]:
            raise ValueError(f"source integrity failure: {f['local_path']}")
        paths[(f["key"],f["path"])] = p
    return m, paths


def prepare(sources, out):
    m, paths = verify_sources(sources)
    if out.exists():
        raise FileExistsError("input directory exists; choose a new version")
    out.mkdir(parents=True)
    items, annotations, diagnostics, failures, counts = [], [], [], [], {}
    def add(fn, *args):
        try:
            result = fn(*args)
            a, g = result
            items.extend(a if isinstance(a,list) else [a])
            annotations.extend(g if isinstance(g,list) else [g])
        except (ValueError, KeyError, TypeError) as e:
            source=args[0]
            identity=source.get("total_index",source.get("sample_id",str(source.get("task"))+":"+str(source.get("index"))))
            failures.append(f"{fn.__name__}:{identity}: {e}")
    for subset, (n, steps) in COUNTS["apb"].items():
        rows = [json.loads(line) for line in paths[("apb",f"data/AgentProcessBench/{subset}.jsonl")].read_text().splitlines()]
        actual = (len(rows), sum(len(r["step_labels"]) for r in rows))
        counts[f"apb:{subset}"] = {"source_items":actual[0],"source_steps":actual[1]}
        if actual != (n,steps):
            failures.append(f"apb:{subset}: counts {actual} != {(n,steps)}")
        for row in rows:
            add(adapt_apb,row,subset,diagnostics)
    for subset, n in COUNTS["tprm"].items():
        rows = json.loads(paths[("tprm",f"data/prmbench_{subset}.json")].read_text())
        counts[f"tprm:{subset}"] = {"source_cases":len(rows),"items":2*len(rows),"steps":2*len(rows)}
        if len(rows) != n:
            failures.append(f"tprm:{subset}: count {len(rows)} != {n}")
        for case_index,row in enumerate(rows):
            add(adapt_tprm,row,subset,diagnostics,case_index)
    import pyarrow.parquet as pq
    split = json.loads(paths[("aft","splits_test.json")].read_text())
    if len(split["test_safe"]) != 169 or len(split["test_unsafe"]) != 163:
        failures.append("aft: official test split count mismatch")
    test_ids = set(split["test_safe"]+split["test_unsafe"])
    if len(test_ids)!=332:
        failures.append("aft: duplicate official test IDs")
    domains, unsafe_sources, test_seen = Counter(), Counter(), set()
    for safe, expected in ((True,1162),(False,1114)):
        rows = pq.read_table(paths[("aft",f"aftraj_{'safe' if safe else 'unsafe'}.parquet")]).to_pylist()
        if len(rows)!=expected:
            failures.append(f"aft: split size {len(rows)} != {expected}")
        for row in rows:
            domains[(row["domain"],safe)] += 1
            if not safe:
                unsafe_sources[row["unsafe_source"]] += 1
            if row["conv_id"] in test_ids:
                test_seen.add(row["conv_id"])
            add(adapt_aft,row,safe,test_ids)
    for domain, ns, nu in (("math",396,397),("coding",361,247),("agentic",405,470)):
        if (domains[domain,True],domains[domain,False])!=(ns,nu):
            failures.append(f"aft:{domain}: domain counts mismatch")
    if test_seen != test_ids:
        failures.append("aft: test IDs missing from release")
    ids = [a.uid for a in items]
    if len(ids)!=len(set(ids)):
        failures.append("duplicate item UIDs")
    for domain in ("math","coding","agentic"):
        selected = [i for i in items if i.benchmark=="aft" and i.subset==domain]
        counts[f"aft:{domain}"] = {"items":len(selected),"steps":sum(len(i.step_entry_indices) for i in selected),
             "safe":domains[domain,True], "unsafe":domains[domain,False]}
    report = {"status":"FAIL" if failures else "PASS", "counts":counts,"failures":failures,
              "unsafe_source_counts":dict(unsafe_sources),"aft_test_items":len(test_seen),
              "aft_mistake_step_exceptions":[e for e in failures if "mistake_step" in e],
              "source_manifest_sha256":file_hash(Path(sources)/"SOURCES.json"),"diagnostics":dict(Counter(d['kind'] for d in diagnostics))}
    atomic_json(out/"PREPARATION.json",report)
    atomic_json(out/"ADAPTER_DIAGNOSTICS.json",diagnostics)
    if failures:
        raise ValueError(f"preparation failed; see {out/'PREPARATION.json'}")
    # Source order retained for APB/TPRM; AFT safe/unsafe partitions combined and UID sorted.
    items = [i for i in items if i.benchmark!="aft"] + sorted([i for i in items if i.benchmark=="aft"],key=lambda i:i.uid)
    (out/"evaluator_only").mkdir()
    for benchmark in ("apb","tprm","aft"):
        with (out/f"{benchmark}_answers.jsonl").open("w",encoding="utf8") as f:
            for item in items:
                if item.benchmark==benchmark:
                    f.write(json.dumps(item.to_dict(),ensure_ascii=False,allow_nan=False)+"\n")
        with (out/"evaluator_only"/f"{benchmark}.jsonl").open("w",encoding="utf8") as f:
            for a in annotations:
                if a["uid"].startswith(benchmark+":"):
                    f.write(json.dumps(a,ensure_ascii=False,allow_nan=False)+"\n")
    atomic_json(out/"INPUTS.json",{b:file_hash(out/f"{b}_answers.jsonl") for b in ("apb","tprm","aft")})
    return report


def cpu_gates(answers, tokenizer, out, max_context=32768):
    import numpy as np
    out.mkdir(parents=True,exist_ok=True)
    gates = {name:{"status":"PASS","failures":[],"checked":0} for name in ("G1","G2","G3","G4")}
    lengths, excluded, per_subset, rendered_pairs, empty = {}, [], defaultdict(list), {}, []
    sample_rng = random.Random(20261012)
    apb = [row for row in answers if row["benchmark"]=="apb" and
           "source_reasoning" not in row["messages"][row["step_message_indices"][0]]]
    parity_ids = set(row["uid"] for row in sample_rng.sample(apb,min(20,len(apb))))
    for row in answers:
        uid = row["uid"]
        gates["G4"]["checked"] += 1
        bad = blind_keys(row)
        if bad:
            gates["G4"]["failures"].append({"uid":uid,"keys":bad})
        try:
            text, spans = render(row,tokenizer)
            n = len(tokenizer(text,add_special_tokens=False,truncation=False,verbose=False)["input_ids"])
            lengths[uid] = n
            per_subset[row["benchmark"]+":"+row["subset"]].append(n)
            if n>max_context:
                excluded.append({"uid":uid,"subset":row["benchmark"]+":"+row["subset"],"tokens":n})
            gates["G2"]["checked"] += 1
            try:
                tokenize(row,tokenizer,max_context=max_context)
            except ValueError as e:
                gates["G2"]["failures"].append({"uid":uid,"reason":str(e)})
            empty.extend({"uid":uid,"step":i} for i,(a,b) in enumerate(spans["step_char_spans"]) if a==b)
            if row["benchmark"]=="tprm":
                left,right = spans["step_char_spans"][0]
                pair = uid.rsplit(":",1)[0]
                if pair in rendered_pairs:
                    prefix,action,other_uid = rendered_pairs.pop(pair)
                    gates["G3"]["checked"] += 1
                    if prefix!=text[:left] or action==text[left:right]:
                        gates["G3"]["failures"].append({"uid":uid,"other_uid":other_uid,"reason":"prefix differs or actions identical"})
                else:
                    rendered_pairs[pair]=(text[:left],text[left:right],uid)
            if uid in parity_ids or row["benchmark"]=="tprm":
                # One specified prefix per selected APB item; action prefix for TPRM.
                # The first scored APB turn avoids later-prefix native/corrected ambiguity.
                k = row["step_message_indices"][0]
                if k:
                    prefix_messages = [dict(m) for m in row["messages"][:k]]
                    for m in prefix_messages:
                        if m.get("think_literal"):
                            m["reasoning_content"]=""
                    actual = text[:spans["step_char_spans"][0][0]]
                    expected = native(tokenizer,prefix_messages,row["tools"],generation=True)
                    gates["G1"]["checked"] += 1
                    if actual!=expected:
                        position=next((j for j,(a,b) in enumerate(zip(actual,expected)) if a!=b),min(len(actual),len(expected)))
                        gates["G1"]["failures"].append({"uid":uid,"step_message_index":k,"first_difference_char":position})
        except (ValueError,KeyError,TypeError) as e:
            gates["G2"]["failures"].append({"uid":uid,"reason":str(e)})
        if len(lengths)%200==0:
            print(json.dumps({"tokenized":len(lengths),"excluded":len(excluded),"span_failures":len(gates['G2']['failures'])}),flush=True)
    for pair,(_,_,uid) in rendered_pairs.items():
        gates["G3"]["failures"].append({"uid":uid,"reason":"unmatched pair"})
    summary = {key:{"items":len(values),"quantiles":dict(zip(("min","q25","median","q75","q95","max"),
                    np.quantile(values,[0,.25,.5,.75,.95,1]).tolist())),
                    "excluded":sum(e["subset"]==key for e in excluded)} for key,values in per_subset.items()}
    overlimit = [key for key,s in summary.items() if s["excluded"]>.05*s["items"]]
    for gate in gates.values():
        gate["status"]="FAIL" if gate["failures"] else "PASS"
    gates["LENGTH"]={"status":"FAIL" if overlimit else "PASS","subsets_over_5_percent":overlimit}
    atomic_json(out/"LENGTHS.json",{"tokens_per_item":lengths,"subsets":summary,"max_context":max_context,"truncated":0})
    atomic_json(out/"EXCLUDED_OVERLENGTH.json",excluded)
    atomic_json(out/"EMPTY_STEPS.json",empty)
    atomic_json(out/"CPU_GATES.json",gates)
    return gates


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sources",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--gates",action="store_true")
    args=p.parse_args()
    prepare(args.sources,args.out)
    if args.gates:
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(args.sources/"tokenizer",local_files_only=True,use_fast=True)
        rows=[json.loads(l) for b in ("apb","tprm","aft") for l in (args.out/f"{b}_answers.jsonl").read_text().splitlines()]
        g=cpu_gates(rows,tok,args.out)
        raise SystemExit(0 if all(v["status"]=="PASS" for v in g.values()) else 2)
