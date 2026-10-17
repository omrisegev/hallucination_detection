"""Independent masked-raw telemetry identity and derived nuisance check."""
from pathlib import Path
import hashlib,json,pickle,re
import numpy as np

P=Path(__file__).resolve().parents[1];I=P/'inputs'
path=I/'prmbench_telemetry.pkl'
h=hashlib.sha256()
with path.open('rb') as f:
    for b in iter(lambda:f.read(8<<20),b''):h.update(b)
expected=json.loads((I/'MASK.json').read_text())['files'][0]['output_sha256']
assert h.hexdigest()==expected
records=pickle.load(path.open('rb'))
records=list(records.values()) if isinstance(records,dict) else records
records={str(r['idx']):r for r in records}
p=np.load(I/'population.npz');off=p['offsets'];idx=p['idx'];nuis=np.load(P/'controls/NUISANCE_INPUTS.npz')
assert set(records)==set(idx) and len(records)==6969
maxentropy=0.;maxspan=0.;mismatch=0
for i,u in enumerate(idx):
    row=records[u];a,b=off[i:i+2]
    entropy=np.asarray(row['token_entropies'],float)
    maxentropy=max(maxentropy,abs(float(entropy.mean())-nuis['answer_entropy'][i]))
    span=np.asarray(row['step_token_spans'],int)
    assert span.shape==(b-a,2) and len(row['steps'])==b-a
    maxspan=max(maxspan,int(np.max(np.abs(np.diff(span,axis=1).ravel()-nuis['length'][a:b]))))
    assert row['first_token_mask']
    # Independently check six example answers and deterministic every97th
    # answer, including short-text/first-step behavior.
    if i%97==0 or i in (0,1,2):
        words=[re.findall(r'\w+',str(t).casefold()) for t in row['steps']]
        for j,cur in enumerate(words):
            val=0.
            for pre in words[:j]:
                k=3 if min(len(cur),len(pre))>=3 else 1
                cs=set(zip(*[cur[v:] for v in range(k)]))
                ps=set(zip(*[pre[v:] for v in range(k)]))
                val=max(val,len(cs&ps)/len(cs|ps) if cs|ps else 0.)
            mismatch+=abs(val-float(nuis['repetition'][a+j]))>1e-12
assert maxentropy<1e-12 and maxspan==0 and mismatch==0
receipt=dict(status='PASS',raw_sha256=h.hexdigest(),n_answers=len(records),
             n_steps=int(off[-1]),entropy_max_abs=float(maxentropy),
             token_length_max_abs=int(maxspan),repetition_sample_mismatches=int(mismatch),
             repetition_sample='Every97th answer plus first3; exact short-pair definition',
             raw_masked_metadata_present=True)
(P/'review/TELEMETRY_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
