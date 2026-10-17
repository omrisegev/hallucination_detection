import json, collections
p=r'C:\Users\DELL\.cache\huggingface\hub\datasets--hitsmy--PRMBench_Preview\snapshots\5cc7683d0ae5797f84d7aeac0607966f277c39e1\prmbench_preview.jsonl'
rows=[json.loads(l) for l in open(p,encoding='utf8')]
print(len(rows)); print(list(rows[0].keys()))
ms=[r for r in rows if r['classification']=='multi_solutions']
print(len(ms))
r=ms[0]
for k,v in r.items():
    print('==',k,'::',repr(v)[:700])
# idx pattern
print([r['idx'] for r in ms[:10]])
red=[r for r in rows if r['classification']=='redundency']
print([r['idx'] for r in red[:5]])
