import pickle, collections
p=r'C:\Users\omris\TAU\hallucination_detection\dataset_cache\four_localization\prmbench_qwen25math7b_full\prmbench_prm.pkl'
d=pickle.load(open(p,'rb'))
print(type(d), len(d))
k=next(iter(d)); v=d[k]
print(k, type(v), list(v.keys()) if isinstance(v,dict) else v)
for kk,vv in v.items():
    s=repr(vv); print(kk, '::', s[:300])
print(collections.Counter(m['classification'] for m in d.values()))
