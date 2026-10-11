"""Position-0..4 surprisal and first-token stats for think-fix teacher-forced outputs.
Writes THINKFIX_STATS.json beside each output. Old (pre-fix) dirs are read-only inputs.
Usage: python thinkfix_stats.py <name>=<path> ...   (path: pkl, dir of pkls, or external run dir)"""
import glob, json, os, pickle, sys
import numpy as np
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
THINK = tok.convert_tokens_to_ids("<think>")
WE = set(tok.encode("We", add_special_tokens=False)[:1])


def rows_from(path):
    if os.path.isdir(os.path.join(path, "records")) or os.path.basename(path) == "records":
        rec = path if os.path.basename(path) == "records" else os.path.join(path, "records")
        for f in sorted(glob.glob(os.path.join(rec, "**", "*"), recursive=True)):
            if os.path.isfile(f) and f.endswith((".pkl", ".npz", ".json")):
                yield from external_rows(f)
        return
    files = [path] if os.path.isfile(path) else sorted(glob.glob(os.path.join(path, "*.pkl")))
    for f in files:
        c = pickle.load(open(f, "rb"))
        for e in (c.values() if isinstance(c, dict) else c):
            yield os.path.basename(f), e


def external_rows(f):
    if f.endswith(".pkl"):
        c = pickle.load(open(f, "rb"))
        for e in (c.values() if isinstance(c, dict) and "gen_token_ids" not in c else [c]):
            yield os.path.basename(f), e
    elif f.endswith(".record.json"):
        yield os.path.basename(f), json.load(open(f, encoding="utf8"))["payload"]["telemetry"]
    elif f.endswith(".npz"):
        z = np.load(f, allow_pickle=True)
        yield os.path.basename(f), {k: z[k] for k in z.files}


def key(e, *names):
    for n in names:
        if n in e:
            return e[n]
    raise KeyError(f"none of {names} in {sorted(e)[:20]}")


def stats(rows):
    first_ids, s = [], [[] for _ in range(5)]
    top1_think = []
    for _, e in rows:
        ids = list(np.asarray(key(e, "gen_token_ids", "gen_ids", "answer_token_ids")).tolist())
        if not ids:
            continue
        sur = np.asarray(key(e, "token_spilled_energies", "spilled_energies", "surprisal"), float)
        first_ids.append(ids[0])
        for p in range(5):
            if len(sur) > p:
                s[p].append(float(sur[p]))
        tk = e.get("top_k_logprobs", e.get("top_k_logprobs_raw"))
        if isinstance(tk, dict):
            top1_think.append(int(np.asarray(tk["ids"])[0][0]) == THINK)
    out = {"n_rows": len(first_ids),
           "frac_first_token_We": float(np.mean([i in WE for i in first_ids])) if first_ids else None,
           "frac_first_position_top1_think": float(np.mean(top1_think)) if top1_think else None}
    for p in range(5):
        out[f"surprisal_median_pos{p}"] = float(np.median(s[p])) if s[p] else None
    out["surprisal_median_pos1_4_pooled"] = float(np.median(sum(s[1:], []))) if any(s[1:]) else None
    return out


res = {}
for arg in sys.argv[1:]:
    name, path = arg.split("=", 1)
    try:
        res[name] = {"path": path, **stats(rows_from(path))}
    except Exception as ex:  # report, do not hide
        res[name] = {"path": path, "error": repr(ex)}
    print(name, json.dumps(res[name]), flush=True)
out = os.environ.get("STATS_OUT")
if out:
    with open(out + ".tmp", "w") as f:
        json.dump(res, f, indent=2)
    os.replace(out + ".tmp", out)
