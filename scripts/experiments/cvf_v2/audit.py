"""Full ID/span/cache audit and independent legacy Llama score replay."""
import argparse
import csv
import gc
import json
import pickle
import subprocess
import sys
from pathlib import Path
import numpy as np
from .data import config,Dataset,digest,dump

def raw_audit(d,root):
    from evidence_drop import shannon_evidence
    from localization_metrics import step_drop_scores
    bank=np.load(d.c['paths']['tokens']);tokens=bank['tokens'];toff=bank['token_offsets'];spans=bank['step_spans']
    evidence=np.load(d.c['paths']['mindgap'])['evidence'];raw_scores=np.empty(int(d.off[-1]));ledger=[]
    specs=[]
    for scorer in ['q4','q8']:
        for subset in ['gsm8k','math','olympiadbench','omnimath']:
            path=root/'dataset_cache/repgrid'/('pb_qwen3_4b' if scorer=='q4' else 'pb_qwen3_8b')/f'processbench_{subset}.pkl'
            specs.append((f'pb_{subset}_{scorer}',subset,path))
    specs.append(('prmbench_qwen3_8b',None,root/'dataset_cache/four_localization/prmbench_qwen3_8b_telemetry_full/prmbench_telemetry.pkl'))
    inspect=Path(__file__).resolve().parents[2]/'inspect_cell.py'
    for cell,subset,path in specs:
        manifest=path.with_name('manifest.json');stat=path.stat()
        info={'cell':cell,'path':str(path),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,'sha256':digest(path),
              'manifest_sha256':digest(manifest),'manifest':json.loads(manifest.read_text(encoding='utf8'))}
        # Required standard probe first. It expects generation-candidate schema;
        # teacher-forced flat rows intentionally need the explicit audit below.
        probe=subprocess.run([sys.executable,str(inspect),str(path)],capture_output=True,text=True,encoding='utf8',errors='replace')
        info['standard_schema_probe']={'returncode':probe.returncode,'stdout':probe.stdout,'stderr':probe.stderr}
        with open(path,'rb') as f:payload=pickle.load(f)
        rows={}
        for row in payload.values():
            key=f'{subset}::{row["id"]}' if subset else str(row['idx'])
            if key in rows:raise ValueError('duplicate raw ID '+key)
            rows[key]=row
        idx=np.flatnonzero(d.cells==cell)
        assert set(rows)==set(d.ids[idx]),cell
        maxdiff=0.
        for i in idx:
            row=rows[d.ids[i]];a,b=d.off[i:i+2];ta,tb=toff[i:i+2]
            assert np.array_equal(row['step_token_spans'],spans[a:b]),d.ids[i]
            assert len(row['token_entropies'])==tb-ta
            np.testing.assert_array_equal(np.asarray(row['token_spilled_energies'],np.float32),tokens[ta:tb,2])
            np.testing.assert_array_equal(-np.asarray(row['token_logsumexp'],np.float32),tokens[ta:tb,5])
            if subset:assert row['label']==d.target[i],d.ids[i]
            e=shannon_evidence(row['top_k_logprobs'],20)
            diff=float(np.max(abs(e-evidence[ta:tb])));maxdiff=max(maxdiff,diff)
            assert diff<3e-7,(d.ids[i],diff)
            raw_scores[a:b]=step_drop_scores(e,spans[a:b],ema_span=5)
        info.update(answers=len(idx),max_float32_evidence_difference=maxdiff,ids_spans_token_values_labels_verified=True)
        ledger.append(info);del rows,payload;gc.collect()
        print('Raw audit complete '+cell,flush=True)
    assert np.isfinite(raw_scores).all()
    path=d.out/'MINDGAP_ADAPTER_REPLAY.npz';cached=np.load(path)['scores']
    archive=d.out/'MINDGAP_CACHED_EVIDENCE_REPLAY.npz'
    if not archive.exists():archive.write_bytes(path.read_bytes())
    changed=int(np.sum(d.peaks(cached)!=d.peaks(raw_scores)))
    np.savez(path,scores=raw_scores)
    dump(d.out/'RAW_CACHE_AUDIT.json',{'sources':ledger,'answers':d.n,'raw_replay_peak_changes_from_float32_evidence':changed,
      'max_replay_score_difference':float(np.max(abs(cached-raw_scores)))})
    mf=json.loads((d.out/'MINDGAP_REPLAY_MANIFEST.json').read_text(encoding='utf8'))
    mf.update(source_precision='top20 logprobs from original raw caches, stable log-space float64 entropy; same existing adapter',
      raw_cache_audit_sha256=digest(d.out/'RAW_CACHE_AUDIT.json'),scores_sha256=digest(path))
    dump(d.out/'MINDGAP_REPLAY_MANIFEST.json',mf)

def llama(d,directory):
    path=directory/'PREDICTIONS.csv';report=directory/'REPORT.json'
    all_rows=list(csv.DictReader(path.open(encoding='utf8')));old=json.loads(report.read_text(encoding='utf8'));results={}
    assert len(all_rows)==old['n_records']
    rows=[r for r in all_rows if int(r['label'])>=0]
    for subset in sorted(set(r['subset'] for r in rows)) + ['all']:
        take=[r for r in rows if subset=='all' or r['subset']==subset];label=np.array([int(r['label']) for r in take])
        assert len(take)==old['fusion'][subset]['n']
        results[subset]={}
        for rule,record in old['fusion'][subset]['rules'].items():
            delta=np.array([int(r[rule]) for r in take])-label
            values={'sla':float((delta==0).mean()),'sla_tol1':float((abs(delta)<=1).mean()),'early':float((delta<0).mean()),'late':float((delta>0).mean()),'mean_offset':float(delta.mean())}
            for k,v in values.items():assert abs(v-record[k])<1e-12,(subset,rule,k)
            results[subset][rule]=values
    dump(d.out/'LLAMA_APPENDIX.json',{'scope':'historical prediction-score replay only; not a refit or full-population comparison',
      'original_answers':old['n_records'],'erroneous_predictions':len(rows),'tau':old['tau'],'pilot_only':True,
      'excluded_from_qwen_macro':True,'predictions_sha256':digest(path),'report_sha256':digest(report),'source_directory':str(directory),
      'all_archived_metrics_match':True,'results':results})
    print('Llama archived predictions replayed:',len(rows),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--raw-root');p.add_argument('--llama-directory');args=p.parse_args()
    d=Dataset(config(args.config))
    if args.raw_root:raw_audit(d,Path(args.raw_root))
    if args.llama_directory:llama(d,Path(args.llama_directory))

if __name__=='__main__':main()
