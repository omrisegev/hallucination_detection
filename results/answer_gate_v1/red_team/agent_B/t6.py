import json, collections
p=r'C:\Users\DELL\.cache\huggingface\hub\datasets--hitsmy--PRMBench_Preview\snapshots\5cc7683d0ae5797f84d7aeac0607966f277c39e1\prmbench_preview.jsonl'
c=collections.Counter()
for l in open(p,encoding='utf8'):
    r=json.loads(l); c[(r['classification'],'original_response' in r and r['original_response'] not in (None,''),'ground_truth' in r and r['ground_truth'] not in (None,''), r['original_process']==r['modified_process'])]+=1
for k,v in sorted(c.items()): print(k,v)
