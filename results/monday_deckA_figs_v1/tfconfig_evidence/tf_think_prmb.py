import pickle, json, numpy as np
THINK=151667
out={}
def run(name, path):
    t=pickle.load(open(path,"rb")); ks=sorted(t)
    top1=np.array([int(t[k]["top_k_logprobs"]["ids"][0][0]) for k in ks])
    p0=np.array([float(np.exp(t[k]["top_k_logprobs"]["logprobs"][0][0])) for k in ks])
    sp=[np.asarray(t[k]["token_spilled_energies"]) for k in ks]
    out[name]={"n":len(ks),"pos0_top1_is_think":int((top1==THINK).sum()),
      "pos0_think_prob_median":float(np.median(p0[top1==THINK])) if (top1==THINK).any() else None,
      "spilled_pos0_mean":float(np.mean([s[0] for s in sp])),
      "spilled_pos1_mean":float(np.mean([s[1] for s in sp if len(s)>1])),
      "spilled_pos2_mean":float(np.mean([s[2] for s in sp if len(s)>2])),
      "spilled_late_mean":float(np.mean([np.mean(s[20:]) for s in sp if len(s)>40]))}
    del t
R=r"c:\Users\omris\TAU\hallucination_detection\dataset_cache"
for m in ["4b","8b"]:
    run(f"pb_math_qwen3_{m}", R+rf"\repgrid\pb_qwen3_{m}\processbench_math.pkl")
run("prmbench_qwen3_8b", R+r"\four_localization\prmbench_qwen3_8b_telemetry_full\prmbench_telemetry.pkl")
print(json.dumps(out,indent=1))
