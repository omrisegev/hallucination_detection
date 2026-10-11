"""Check the real manifest validator rejects output corruption before replay."""
from pathlib import Path
import ast,hashlib,importlib.util,json,tempfile
P=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('manifest_review',P/'package_manifest.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
tree=ast.parse((P/'reproduce.py').read_text())
calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='verify']
assert len(calls)==1
assert any(k.arg=='full' and isinstance(k.value,ast.Constant) and k.value.value is True for k in calls[0].keywords)
with tempfile.TemporaryDirectory(prefix='replay_guard_',dir=P/'review') as d:
 fake=Path(d);out=fake/'SAVED_RESULT.csv';out.write_text('score\n0.779\n')
 entry=dict(path=out.name,role='output',size=out.stat().st_size,sha256=hashlib.sha256(out.read_bytes()).hexdigest())
 manifest=fake/'MANIFEST.json';identity=json.loads((P/'PACKAGE_MANIFEST.json').read_text());fixture={**identity,'files':[entry],'runtime_files':[]};manifest.write_text(json.dumps(fixture))
 m.PACKAGE=fake;m.ROOT=fake;m.MANIFEST=manifest
 assert m.verify(full=True)['status']=='PASS'
 original_python=dict(fixture['python']);fixture['python']={'version':'deliberately incompatible'};manifest.write_text(json.dumps(fixture))
 try:m.verify(full=True)
 except RuntimeError as exc:assert 'Python version differs' in str(exc)
 else:raise AssertionError('Changed Python identity accepted')
 fixture['python']=original_python;name=next(iter(fixture['versions']));version=fixture['versions'][name];fixture['versions'][name]='999.invalid';manifest.write_text(json.dumps(fixture))
 try:m.verify(full=True)
 except RuntimeError as exc:assert 'Changed numerical dependency' in str(exc)
 else:raise AssertionError('Changed dependency identity accepted')
 fixture['versions'][name]=version;manifest.write_text(json.dumps(fixture))
 out.write_text('score\n0.999\n')
 rejected=False
 try:m.verify(full=True)
 except RuntimeError as exc:rejected='Changed/missing package identity' in str(exc)
 assert rejected, 'Corrupt saved output was accepted'
result=dict(status='PASS',all_wrapper_modes_full_verify=True,corrupt_saved_csv_rejected=True,changed_python_rejected=True,changed_dependency_rejected=True,
 wrapper_sha256=hashlib.sha256((P/'reproduce.py').read_bytes()).hexdigest(),
 validator_sha256=hashlib.sha256((P/'package_manifest.py').read_bytes()).hexdigest(),
 scope='Independent temporary fixture using the actual manifest verify function; original saved results were not modified')
(P/'review/REPLAY_GUARD_RECEIPT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
