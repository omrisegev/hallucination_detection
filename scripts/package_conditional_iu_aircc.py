"""Package reviewed code only; reuse the earlier frozen AIRCC source directory."""
import argparse,hashlib,io,json,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--prior-manifest',type=Path,required=True)
    p.add_argument('--review',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-root',required=True);args=p.parse_args()
    review=json.loads(args.review.read_text());assert review['status']=='PASS'
    for relative,digest in review['reviewed_file_hashes'].items():
        assert sha(ROOT/relative)==digest,('reviewed code changed',relative)
    previous=json.loads(args.prior_manifest.read_text())
    inputs=[f for f in previous['files'] if f['path'].startswith('source/')]
    # The inherited 43-entry experiment freeze includes code; its separately
    # staged source directory contains these 22 data/provenance files.
    assert len(inputs)==22 and sum('/cache_' in f['path'] for f in inputs)==9
    files=sorted(set([*ROOT.glob('scripts/*.py'),*ROOT.glob('spectral_utils/**/*.py'),
                      *ROOT.glob('docs/experiments/*.md'),ROOT/'cluster/conditional_iu_aircc.sbatch']))
    entries=[dict(path='code/'+f.relative_to(ROOT).as_posix(),bytes=f.stat().st_size,sha256=sha(f)) for f in files]
    review_bytes=json.dumps(review,indent=2).encode()
    entries.append(dict(path='INDEPENDENT_REVIEW.json',bytes=len(review_bytes),sha256=hashlib.sha256(review_bytes).hexdigest()))
    commit=subprocess.check_output(['git','-c','safe.directory='+ROOT.as_posix(),'-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    manifest=dict(schema='conditional-iu-aircc-v1',commit=commit,files=entries,source_files=inputs,
                  source_root=args.source_root,source_manifest_sha256=sha(args.prior_manifest),
                  note='Code-only archive; source files are verified at their original remote location. No writes to prior job.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(args.output,'w:gz') as archive:
        for f in files:archive.add(f,arcname='code/'+f.relative_to(ROOT).as_posix(),recursive=False)
        for name,data in [('INDEPENDENT_REVIEW.json',review_bytes),('BUNDLE_MANIFEST.json',json.dumps(manifest,indent=2).encode())]:
            item=tarfile.TarInfo(name);item.size=len(data);archive.addfile(item,io.BytesIO(data))
    args.output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2))
    result=dict(status='PASS',archive=str(args.output),sha256=sha(args.output),bytes=args.output.stat().st_size,
                code_files=len(files),source_files_reused=len(inputs),copied_data_bytes=0)
    args.output.with_suffix('.review.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
