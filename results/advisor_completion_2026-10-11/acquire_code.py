"""Acquire immutable Git blobs by their inspected tree SHA; verify Git identity."""
from pathlib import Path
import base64, concurrent.futures, hashlib, json, subprocess
OUT=Path(__file__).resolve().parent
tree=json.loads(Path('/private/tmp/thinkfix-tree.json').read_text())['tree']
names={'monday_fusion_protocol.py','monday_fusion_core.py','monday_fusion_decision.py','monday_fusion_metrics.py','lsml_gate_locator_research.py','er_stage_a.py','er_stage_b.py','ds_group_weights.py','calfix_common.py','tensor_mom_stage_a_run.py','monday_fusion_data.py','monday_fusion_estimator.py','monday_fusion_partition.py'}
selected=[t for t in tree if t['type']=='blob' and t['path'].startswith('results/monday_fusion_analysis_v1/code/') and (Path(t['path']).name in names or '/cvf_v2/' in t['path'])]
selected += [t for t in tree if t['path']=='results/monday_combined_recipe_v1/run_20261003/PROTOCOL.json']
def get(t):
    p=OUT/'sources/research'/t['path'];p.parent.mkdir(parents=True,exist_ok=True)
    if not p.exists():
        r=json.loads(subprocess.check_output(['gh','api','repos/omrisegev/hallucination_detection/git/blobs/'+t['sha']],text=True));p.write_bytes(base64.b64decode(r['content']))
    b=p.read_bytes(); h=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest();assert h==t['sha'],p
    return {'path':t['path'],'git_blob':h,'sha256':hashlib.sha256(b).hexdigest()}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(get,selected))
(OUT/'CODE_ACQUISITION.json').write_text(json.dumps(rows,indent=2)+'\n')
print('verified',len(rows),'pinned Git blobs')
