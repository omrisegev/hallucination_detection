"""Read-only byte validation, independent-review gate and runtime provenance."""
import argparse,hashlib,json,platform,sys
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle-root',type=Path,required=True)
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--family',choices=['position','graph_local','graph_tv'],required=True);args=p.parse_args()
    bundle=args.bundle_root.resolve();source=args.source_root.resolve()
    manifest=json.loads((bundle/'BUNDLE_MANIFEST.json').read_text())
    assert source==Path(manifest['source_root']).resolve(),'wrong remote source root'
    for group,root in [(manifest['files'],bundle),(manifest['source_files'],source)]:
        for item in group:
            relative=item['path'] if root==bundle else item['path'].removeprefix('source/')
            path=(root/relative).resolve();assert path.is_relative_to(root)
            assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],str(path)
    review=json.loads((bundle/'INDEPENDENT_REVIEW.json').read_text());assert review['status']=='PASS'
    for relative,digest in review['reviewed_file_hashes'].items():
        assert sha(bundle/'code'/relative)==digest,relative
    import numpy,scipy,sklearn
    out=bundle/'code/results/conditional_iu_fusion_v1'/args.family;out.mkdir(parents=True,exist_ok=True)
    result=dict(status='PASS',family=args.family,code_files=len(manifest['files']),source_files=len(manifest['source_files']),
                python=sys.version,platform=platform.platform(),numpy=numpy.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,
                read_only_source=str(source),independent_review='PASS',source_data_mutations=False)
    (out/'TRANSFER_REVIEW.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
