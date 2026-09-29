"""Round 2: the 118 items the two round-1 judges disagree on (102 own + 16 ProcessBench validation),
re-packed blind for two NEW independent judges: new opaque ids (K001...), new seeded order, same item
content, same instructions and checker. Round-1 labels and the fact that these are disagreements are
not in the packet. Mapping K-id -> J-id is in round2/private/ROUND2_KEY.jsonl.
"""
import hashlib, json, os, random, shutil
R = "results/self_generated_step_labels_v1"; O = f"{R}/round2"; SEED = 20260930
lab = {}
for j in ("claude-opus-5.5", "gpt-6-sol"):
    for n in sorted(os.listdir(f"{R}/labels/{j}")):
        if n.startswith("shard_"):
            for l in open(f"{R}/labels/{j}/{n}", encoding="utf-8"):
                x = json.loads(l); lab.setdefault(x["item_id"], {})[j] = x["first_error_step"]
dis = sorted(i for i, v in lab.items() if v["claude-opus-5.5"] != v["gpt-6-sol"])
items = {}
for n in sorted(os.listdir(f"{R}/packet/shards")):
    for l in open(f"{R}/packet/shards/{n}", encoding="utf-8"):
        x = json.loads(l)
        if x["item_id"] in dis: items[x["item_id"]] = x
assert len(items) == len(dis) == 118, len(items)
order = sorted(items); random.Random(SEED).shuffle(order)
os.makedirs(f"{O}/packet/shards", exist_ok=True); os.makedirs(f"{O}/private", exist_ok=True)
shards, keyl = [], []
for s in range(0, len(order), 25):
    name = f"shard_{s // 25:03d}.jsonl"; lines = []
    for k, jid in enumerate(order[s:s + 25]):
        kid = f"K{s + k + 1:03d}"; x = dict(items[jid]); x["item_id"] = kid
        lines.append(json.dumps(x, ensure_ascii=False)); keyl.append(json.dumps({"item_id": kid, "round1_item_id": jid}))
    body = "\n".join(lines) + "\n"
    open(f"{O}/packet/shards/{name}", "w", encoding="utf-8", newline="\n").write(body)
    shards.append({"shard": name, "n_items": len(lines), "sha256": hashlib.sha256(body.encode()).hexdigest()})
open(f"{O}/private/ROUND2_KEY.jsonl", "w", newline="\n").write("\n".join(keyl) + "\n")
json.dump({"packet_version": "self-generated-step-labels-round2-v1", "shard_size": 25, "n_items": len(order),
           "n_shards": len(shards), "shards": shards}, open(f"{O}/packet/PACKET_MANIFEST.json", "w"), indent=1)
for f in ("JUDGE_INSTRUCTIONS.md", "FORMAT_EXAMPLE_ITEM.json", "FORMAT_EXAMPLE_LABELS.jsonl", "check_labels.py"):
    shutil.copy(f"{R}/packet/{f}", f"{O}/packet/{f}")
print(len(order), "items,", len(shards), "shards")
