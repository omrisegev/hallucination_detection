"""Package immutable evaluation evidence locally; upload is an explicit separate command."""
import hashlib,json,tarfile,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/lsml_external_generalization_v1/evaluation'
DEST=ROOT/'scratch/external_generalization_private/evidence_archives'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def main():
    for name in ('ALL_CELLS_SEALED.json','OFFICIAL_METRIC_REPLAY.json','RED_TEAM.md','METRICS.json','CONTRASTS.json'):
        assert (OUT/name).is_file(),name
    assert not list(OUT.rglob('WRITER.lock')), 'active writer cannot be archived'
    DEST.mkdir(parents=True,exist_ok=True)
    paths=sorted(p for p in OUT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='EVALUATION_ARCHIVE.json')
    entries=[{'path':str(p.relative_to(OUT)).replace('\\','/'),'bytes':p.stat().st_size,'sha256':sha(p)} for p in paths]
    manifest={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root':'evaluation','files':entries,'file_count':len(entries),'uncompressed_bytes':sum(e['bytes'] for e in entries)}
    mf=DEST/'MANIFEST.json';mf.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8',newline='\n')
    tmp=DEST/'evaluation.pending.tar.gz'
    if tmp.exists():raise FileExistsError(tmp)
    with tarfile.open(tmp,'w:gz',compresslevel=6) as tf:
        tf.add(mf,arcname='MANIFEST.json')
        for p in paths:tf.add(p,arcname='evaluation/'+str(p.relative_to(OUT)).replace('\\','/'))
    digest=sha(tmp);archive=DEST/f'external_evaluation_{digest[:16]}.tar.gz'
    if archive.exists():raise FileExistsError(archive)
    tmp.rename(archive)
    result={'archive':str(archive),'sha256':digest,'bytes':archive.stat().st_size,'manifest_sha256':sha(mf),'file_count':len(entries),'uncompressed_bytes':manifest['uncompressed_bytes'],'remote':'gdrive:hallucination_detection/cluster_results/lsml_external_generalization_v1/evaluation_archives/'+archive.name,'remote_verified':False,'reason_local_upload':'AIRCC SSH probes timed out; completed CPU evaluation and authorized Drive storage use local fallback'}
    (OUT/'EVALUATION_ARCHIVE.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8',newline='\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
