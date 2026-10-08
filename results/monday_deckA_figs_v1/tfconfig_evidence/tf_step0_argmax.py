import pickle, json, numpy as np
R=r"c:\Users\omris\TAU\hallucination_detection\dataset_cache\repgrid"
out={}
def top10(x): return float(np.sort(x)[-10:].mean()) if len(x) else np.nan
for m in ["4b","8b"]:
  for sub in ["gsm8k","math"]:
    t=pickle.load(open(fr"{R}\pb_qwen3_{m}\processbench_{sub}.pkl","rb"))
    a0=a0x=n=0; err=e0=0
    for k,e in t.items():
        sp=np.asarray(e["token_spilled_energies"]); spans=[s for s in e["step_token_spans"] if s]
        if len(spans)<2: continue
        n+=1
        v=[top10(sp[a:b]) for a,b in spans]
        sp2=sp.copy(); sp2[:3]=np.nan
        vx=[top10(sp2[a:b][~np.isnan(sp2[a:b])]) for a,b in spans]
        a0+=int(np.nanargmax(v)==0); a0x+=int(np.nanargmax(vx)==0)
        if e["label"]!=-1: err+=1; e0+=int(e["label"]==0)
    out[f"{sub}_{m}"]={"n_answers_ge2_steps":n,"top10_surprisal_argmax_step0":a0,"same_after_dropping_first3_tokens":a0x,"n_error":err,"n_first_error_step0":e0}
    del t
print(json.dumps(out,indent=1))
