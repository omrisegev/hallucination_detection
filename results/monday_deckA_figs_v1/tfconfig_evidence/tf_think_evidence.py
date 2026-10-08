"""CPU-only: does the teacher-forced Qwen3 pass expect a <think> block at answer position 0?"""
import pickle, sys, json, numpy as np
R = r"c:\Users\omris\TAU\hallucination_detection\dataset_cache"
THINK, ENDTHINK = 151667, 151668
out = {}

def prof(arrs, n=8):
    m = [float(np.mean([a[i] for a in arrs if len(a) > i])) for i in range(n)]
    late = float(np.mean([np.mean(a[20:]) for a in arrs if len(a) > 40]))
    return m, late

# (A) generated /no_think cell: what does Qwen3-8B itself emit first?
g = pickle.load(open(R + r"\repgrid\evdrop_gsm8k_qwen3_8b\raw_gsm8k_T0.0.pkl", "rb"))
first, ent_g, sp_g, starts_think = [], [], [], 0
keys = sorted(g)
for k in keys:
    c = g[k]["candidates"][0] if "candidates" in g[k] else g[k]
    ids = list(c["gen_token_ids"]); first.append(ids[:5])
    starts_think += int(ids[:1] == [THINK])
    ent_g.append(np.asarray(c["token_entropies"])); sp_g.append(np.asarray(c.get("token_spilled_energies", [])))
out["gen_n"] = len(keys); out["gen_starts_with_think"] = starts_think
out["gen_first5_example"] = first[0]
# offset where content starts: after </think> and following newline token(s)
def content_start(ids):
    ids = list(ids)
    if ENDTHINK in ids[:6]:
        j = ids.index(ENDTHINK) + 1
        return j + 1  # '\n\n' is one token in Qwen tokenizer (checked below)
    return 0
cs = [content_start(g[k]["candidates"][0]["gen_token_ids"]) for k in keys]
out["gen_content_start_values"] = sorted(set(cs))
ent_c = [e[s:] for e, s in zip(ent_g, cs)]; sp_c = [e[s:] for e, s in zip(sp_g, cs) if len(e)]
out["gen_entropy_think_positions_mean"] = [float(np.mean([e[i] for e in ent_g])) for i in range(4)]
out["gen_spilled_think_positions_mean"] = [float(np.mean([e[i] for e in sp_g if len(e) > i])) for i in range(4)]
out["gen_content_entropy_pos0_7"], out["gen_content_entropy_late"] = prof(ent_c)
out["gen_content_spilled_pos0_7"], out["gen_content_spilled_late"] = prof(sp_c)
del g

# (B) teacher-forced ProcessBench gsm8k cells, Qwen3-4B and 8B
for m in ["4b", "8b"]:
    t = pickle.load(open(R + rf"\repgrid\pb_qwen3_{m}\processbench_gsm8k.pkl", "rb"))
    ks = sorted(t)
    ent = [np.asarray(t[k]["token_entropies"]) for k in ks]
    sp = [np.asarray(t[k]["token_spilled_energies"]) for k in ks]
    top1 = [int(t[k]["top_k_logprobs"]["ids"][0][0]) for k in ks]
    top1lp = [float(t[k]["top_k_logprobs"]["logprobs"][0][0]) for k in ks]
    d = {"n": len(ks), "pos0_top1_is_think": int(sum(x == THINK for x in top1)),
         "pos0_top1_prob_median_when_think": float(np.median([np.exp(l) for x, l in zip(top1, top1lp) if x == THINK])) if THINK in top1 else None}
    d["entropy_pos0_7"], d["entropy_late"] = prof(ent)
    d["spilled_pos0_7"], d["spilled_late"] = prof(sp)
    # step 0 vs later steps: mean surprisal per step
    s0, sl = [], []
    for k in ks:
        sp_k = np.asarray(t[k]["token_spilled_energies"])
        spans = t[k]["step_token_spans"]
        vals = [float(sp_k[a:b].mean()) for (a, b) in [s for s in spans if s]]
        s0.append(vals[0]); sl += vals[1:]
        # without the first 2 tokens
    s0x = []
    for k in ks:
        sp_k = np.asarray(t[k]["token_spilled_energies"]); a, b = t[k]["step_token_spans"][0]
        if b - a > 3: s0x.append(float(sp_k[a + 2:b].mean()))
    d["step0_mean_surprisal"] = float(np.mean(s0)); d["step0_mean_surprisal_excl_first2tok"] = float(np.mean(s0x))
    d["later_steps_mean_surprisal"] = float(np.mean(sl))
    labs = [t[k]["label"] for k in ks]
    d["n_with_error"] = int(sum(l != -1 for l in labs)); d["n_first_error_step0"] = int(sum(l == 0 for l in labs))
    out[f"pb_gsm8k_qwen3_{m}"] = d
    del t
print(json.dumps(out, indent=1))
