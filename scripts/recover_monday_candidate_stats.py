"""One-time repair of this report's own source-group string storage only."""
from pathlib import Path
import datetime
import hashlib
import json
import numpy as np
ROOT=Path(r'C:\Users\omris\TAU\hallucination_detection')
OUT=ROOT/'results/monday_candidate_panel_v1/run_20261004'
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
note={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'problem':'Panel sufficient-statistic group names were object dtype, so safe np.load refused them.',
    'scope':'Own derived report files only. No frozen experiment, labels, fits, scores or flag arrays changed.',
    'artifacts':[],'content_generation_code_sha256':{}}
archive=OUT/'content_generation_code'; archive.mkdir(exist_ok=True)
for source in [ROOT/'spectral_utils/monday_candidate_panel.py',ROOT/'scripts/build_monday_candidate_panel.py']:
    target=archive/source.name
    if target.exists(): raise RuntimeError('Recovery code archive already exists')
    target.write_bytes(source.read_bytes()); note['content_generation_code_sha256'][str(source)]=digest(source)
for path in sorted((OUT/'sufficient').glob('*.npz')):
    old=digest(path)
    # These are trusted artifacts just written by this report, not imported inputs.
    with np.load(path,allow_pickle=True) as z: values={k:z[k] for k in z.files}
    groups=values['groups']
    assert groups.ndim==1 and all(isinstance(x,str) for x in groups)
    assert all(v.dtype.kind!='O' for k,v in values.items() if k!='groups')
    values['groups']=groups.astype(str)
    assert np.array_equal(groups,values['groups'])
    np.savez_compressed(path,**values)
    with np.load(path) as z:
        assert all(np.array_equal(v,z[k],equal_nan=True) if v.dtype.kind not in 'US' else np.array_equal(v,z[k]) for k,v in values.items())
    note['artifacts'].append({'path':str(path),'old_sha256':old,'new_sha256':digest(path),'only_change':'groups dtype object -> Unicode; values exactly equal'})
assert len(note['artifacts'])==144
(OUT/'STORAGE_RECOVERY.json').write_text(json.dumps(note,indent=2)+'\n',encoding='utf8')
print('144 own statistic files repaired; all numerical arrays and group names exactly preserved')
