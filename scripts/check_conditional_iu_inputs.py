"""Read-only readiness check for a second machine; never downloads data."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args();source=args.source_root.resolve()
    manifest=json.loads((ROOT/'docs/experiments/CONDITIONAL_IU_INPUTS_V1.json').read_text())
    failures=[];matched=0
    for item in manifest['source_files']:
        path=(source/item['path'].removeprefix('source/')).resolve()
        assert path.is_relative_to(source)
        if not path.is_file():failures.append(dict(path=str(path),reason='MISSING'));continue
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
        if h.hexdigest()!=item['sha256']:failures.append(dict(path=str(path),reason='HASH_MISMATCH'))
        else:matched+=1
    print(json.dumps(dict(status='FAIL' if failures else 'PASS',source_root=str(source),
                         matched=matched,expected=len(manifest['source_files']),failures=failures),indent=2))
    raise SystemExit(bool(failures))


if __name__=='__main__':main()
