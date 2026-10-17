"""Read-only Git/worktree inventory; writes only audit evidence beside this script."""
import collections, datetime, hashlib, json, pathlib, subprocess, sys
sys.stdout.reconfigure(encoding='utf-8')
ROOT=pathlib.Path(r'C:\Users\omris\TAU\hallucination_detection')
OUT=pathlib.Path(__file__).parent
PLAN=['claude/ssl-pseudolabel-residual-v1','claude/estimator-provenance-collection-2026-09-29','claude/decision-rule-v1','claude/self-generated-step-labels-v1','codex/family15-tail20-transfer-v1','rescue/main-checkout-loose-files-2026-10-01','claude/readout-quickest-detection-v1','lsml-ct7-levers-run','claude/whitebox-layer-views-v1','origin/claude/token-axis-fusion-sampling-3i9r2u','claude/token-probability-fusion-v1','claude/depth-feature-fusion-v1','codex/claude-feature-bank-token-lsml-v1']
def git(*args, root=ROOT):
 r=subprocess.run(['git','-C',str(root),*args],capture_output=True)
 if r.returncode: raise RuntimeError((args,r.stderr.decode('utf8','replace')))
 return r.stdout.decode('utf8','replace')
def tree(ref):
 return {s.split('\t',1)[1]:s.split()[2] for s in git('ls-tree','-r',ref).splitlines() if '\t' in s}
def blob(d):return hashlib.sha1(b'blob '+str(len(d)).encode()+b'\0'+d).hexdigest()
trees={b:tree(b) for b in PLAN};union=collections.defaultdict(set)
for t in trees.values():
 for path,h in t.items():union[path].add(h)
reach=set(git('rev-list',*PLAN).splitlines())
refs=[dict(zip(['ref','sha'],s.split('\t'))) for s in git('for-each-ref','--format=%(refname)\t%(objectname)','refs/heads','refs/remotes/origin').splitlines()]
unc=[r for r in refs if r['sha'] not in reach]
for r in unc:
 t=tree(r['sha']);r['absent_paths']=[f for f in t if f not in union]
 r['distinct_blob_paths']=[f for f,h in t.items() if f in union and h not in union[f]]
 r['unreachable_commits']=int(git('rev-list','--count',r['sha'],'--not',*PLAN))
 r['last_commit']=git('log','-1','--format=%cs %s',r['sha']).strip()
loose=git('ls-files','-z','--others','--exclude-standard','--','spectral_utils','scripts','tests','docs','.claude/agents').split('\0')
checked=[]
for rel in filter(None,loose):
 f=ROOT/rel
 if not f.is_file():continue
 d=f.read_bytes();hs={blob(d),blob(d.replace(b'\r\n',b'\n'))}
 checked.append({'path':rel,'bytes':len(d),'covered_by':[b for b,t in trees.items() if t.get(rel) in hs]})
orphans=[]
for name in ['binary-moment-fusion-v1','literature-data-diagnostics-v1','rbm-hierarchical-time-v1-setup-recovery']:
 for f in (ROOT/'.worktrees'/name).rglob('*'):
  if not f.is_file():continue
  rel=f.relative_to(ROOT/'.worktrees'/name).as_posix();d=f.read_bytes();hs={blob(d),blob(d.replace(b'\r\n',b'\n'))}
  orphans.append({'folder':name,'path':rel,'bytes':len(d),'sha256':hashlib.sha256(d).hexdigest(),'covered_by':[b for b,t in trees.items() if t.get(rel) in hs]})
worktrees=[]
for block in git('worktree','list','--porcelain').strip().split('\n\n'):
 fields=dict(line.split(' ',1) for line in block.splitlines() if ' ' in line)
 w=pathlib.Path(fields['worktree']);print('inventory',w.name,flush=True)
 fields['tracked_changes']=git('status','--porcelain','--untracked-files=no',root=w).splitlines()
 ignored=git('ls-files','-z','--others','--ignored','--exclude-standard','--','results',root=w).split('\0')
 stats=[]
 for rel in filter(None,ignored):
  f=w/rel
  if f.is_file():stats.append({'path':rel,'bytes':f.stat().st_size})
 fields['ignored_result_files']=stats;fields['ignored_result_count']=len(stats);fields['ignored_result_bytes']=sum(s['bytes'] for s in stats)
 worktrees.append(fields)
u=pathlib.Path(r'C:\Users\DELL\AppData\Local\Temp\claude\c--Users-omris-TAU-hallucination-detection\ae2dd164-1ddb-4b21-9e3e-cb18f86482aa\scratchpad\upload')
ll=set((u/'loose_results.txt').read_text().splitlines());heavy=(u/'heavy_dirs.txt').read_text().splitlines()
mi=worktrees[0]['ignored_result_files'];omitted=[s for s in mi if s['path'] not in ll and not any(s['path'].startswith('results/'+h+'/') for h in heavy)]
rescue=trees['rescue/main-checkout-loose-files-2026-10-01'];dirty={}
for rel in ['HISTORY.md','PROGRESS.md','LESSONS.md']:
 d=(ROOT/rel).read_bytes();dirty[rel]=rescue.get(rel) in {blob(d),blob(d.replace(b'\r\n',b'\n'))}
result={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'proposed 13 sources; all local/origin refs; registered worktrees; three orphan folders','plan':[{'ref':b,'sha':git('rev-parse',b).strip(),'status_files':[f for f in trees[b] if f.startswith('docs/line_status/')]} for b in PLAN],'refs_total':len(refs),'refs_covered':len(refs)-len(unc),'uncovered_refs':unc,'loose_code_docs':checked,'dirty_main_docs_match_rescue':dirty,'orphan_files':orphans,'worktrees':worktrees,'main_ignored_results_omitted_from_oct1_upload':omitted,'upload_plan':{'heavy':heavy,'loose_count':len(ll),'log':(u/'upload.log').read_text(),'bundle_md5':(u/'bundle.md5').read_text().strip()}}
(OUT/'INVENTORY.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps({'refs':len(refs),'covered':len(refs)-len(unc),'loose_checked':len(checked),'loose_missing':sum(not x['covered_by'] for x in checked),'worktrees':len(worktrees),'ignored_omitted_main':len(omitted),'ignored_omitted_main_bytes':sum(x['bytes'] for x in omitted),'orphan_files':len(orphans)},indent=2))
