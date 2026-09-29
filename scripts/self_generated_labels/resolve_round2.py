"""Apply the declared round-2 rule: RESOLVED iff fable-5.1 == astra-6 == one of the two round-1 answers.
Everything else goes to debate. Accuracy of the rule on the 16 ProcessBench items vs human labels.
Writes results/self_generated_step_labels_v1/round2/RESOLUTION.json."""
import json, os, collections
R = "results/self_generated_step_labels_v1"; O = f"{R}/round2"
def load(d):
    out = {}
    for n in sorted(os.listdir(d)):
        if n.startswith("shard_"):
            for l in open(f"{d}/{n}", encoding="utf-8"):
                x = json.loads(l); out[x["item_id"]] = x["first_error_step"]
    return out
r1 = {j: load(f"{R}/labels/{j}") for j in ("claude-opus-5.5", "gpt-6-sol")}
r2 = {j: load(f"{O}/labels/{j}") for j in ("fable-5.1", "astra-6")}
kmap = {k["item_id"]: k["round1_item_id"] for k in map(json.loads, open(f"{O}/private/ROUND2_KEY.jsonl"))}
key = {k["item_id"]: k for k in map(json.loads, open(f"{R}/private/ITEM_KEY.jsonl", encoding="utf-8"))}
rows, c = [], collections.Counter()
for kid, jid in sorted(kmap.items()):
    a, b = r1["claude-opus-5.5"][jid], r1["gpt-6-sol"][jid]; f, s = r2["fable-5.1"][kid], r2["astra-6"][kid]
    if f == s and f in (a, b): status, final = "resolved", f
    elif f == s: status, final = "debate_new_third_answer", None
    else: status, final = "debate_round2_split", None
    k = key[jid]
    rows.append({"round1_item_id": jid, "round2_item_id": kid, "source": k["source"], "votes": {"claude-opus-5.5": a, "gpt-6-sol": b, "fable-5.1": f, "astra-6": s},
                 "status": status, "label": final, **({"pb_label": k["pb_label"]} if k["source"] == "processbench" else {})})
    c[(k["source"], status)] += 1
pb = [r for r in rows if r["source"] == "processbench"]
res = {"counts": {f"{a}/{b}": v for (a, b), v in sorted(c.items())},
       "processbench_check": {"resolved": sum(r["status"] == "resolved" for r in pb),
                              "resolved_correct": sum(r["status"] == "resolved" and r["label"] == r["pb_label"] for r in pb),
                              "fable_correct": sum(r["votes"]["fable-5.1"] == r["pb_label"] for r in pb),
                              "astra_correct": sum(r["votes"]["astra-6"] == r["pb_label"] for r in pb), "n": len(pb)},
       "items": rows}
json.dump(res, open(f"{O}/RESOLUTION.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "items"}, indent=1))
own = [r for r in rows if r["source"] == "own" and r["status"] == "resolved"]
print("own resolved: sided with claude", sum(r["label"] == r["votes"]["claude-opus-5.5"] for r in own), "with gpt", sum(r["label"] == r["votes"]["gpt-6-sol"] for r in own))
