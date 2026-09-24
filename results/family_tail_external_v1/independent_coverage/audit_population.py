"""Independent coverage/provenance/algebra audit; deliberately never opens labels or metrics."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
EXP = OUT.parent
OLD = ROOT/'results/lsml_external_generalization_v1/evaluation'
PRIVATE = ROOT/'scratch/external_generalization_private'
CELLS = {'hard2verify_qwen3_8b': ('hard2verify', 200, 1860, 389770),
         'socratic_qwen3_8b': ('socratic', 2995, 26055, 2505065),
         'socratic_qwq32b': ('socratic', 2995, 26055, 2505065)}
ANCHORS = {'B11_lsml': 'frozen_lsml', 'B11_equal': 'frozen_equal',
           'B11_partition_equal': 'frozen_partition_equal', 'ct7': 'ct7'}
POOL = Path('C:/Users/DELL/AppData/Local/Temp/claude/c--Users-omris-TAU-hallucination-detection/2d14a8c9-8b3f-489b-b26f-812b4b84a8b3/scratchpad')


def load(path):
    return json.loads(path.read_text(encoding='utf8'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False,
                     separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n',
                    encoding='utf8', newline='\n')


def standard(x, cutoff=1e-12):
    x = np.asarray(x, dtype=np.float64)
    sd = x.std(0)
    return np.divide(x-x.mean(0), sd, out=np.zeros_like(x), where=sd>cutoff)


def manual_scores(x, lock):
    """Independent expression of the contract, without importing project scoring helpers."""
    recipe = lock['recipe']; names = recipe['channels_48']
    raw = standard(x)
    signed = raw * np.array([recipe['source_signs'][name] for name in names])
    family = np.column_stack([signed[:, [names.index(n) for n in members]].mean(1)
                              for members in recipe['families_15'].values()])
    reps = {'B11': raw[:, [names.index(n) for n in recipe['bank11']]],
            'F15': standard(family),
            'K28': signed[:, [names.index(n) for n in recipe['channels_28']]],
            'A48o': signed}
    columns = {'B11': recipe['bank11'], 'F15': list(recipe['families_15']),
               'K28': recipe['channels_28'], 'A48o': names}
    result = {}
    for arm in lock['rows']:
        if arm == 'ct7':
            continue
        prefix = arm.split('_')[0]
        weights = np.array([lock['deployment'][arm]['weights'][n] for n in columns[prefix]])
        assert abs(np.abs(weights).sum()-1)<1e-12
        result[arm] = standard(reps[prefix]@weights, cutoff=1e-8)
    return result


def source_and_identity(pooldir):
    gatepath = EXP/'source_full_v1/GATE.json'; gate = load(gatepath)
    assert (gate['status'], gate['scope'], gate['n_checked'], gate['steps']) == ('PASS', 'FULL', 13769, 145597)
    lockpath = ROOT/'results/family_tail_transfer_v1/TRANSFER_LOCK_V1.json'; lock = load(lockpath)
    assert sha(lockpath) == '65b35336fcc2f66b7843ec040d3bdafbbaa03bb44ae5f61f1d00335abfaea5cf'
    observationpath = EXP/'SOURCE_CODE_OBSERVATION.json'; observation = load(observationpath)
    completion = load(EXP/'SOURCE_CODE_COMPLETION.json')
    freeze = load(EXP/'IMPLEMENTATION_FREEZE.json')
    assert completion['status'] == 'PASS'
    assert completion['source_gate_sha256'] == sha(gatepath)
    assert completion['observation_sha256'] == sha(observationpath)
    assert completion['implementation_freeze_sha256'] == sha(EXP/'IMPLEMENTATION_FREEZE.json')
    assert set(completion['files']) == set(observation['files'])
    for rel, expected in freeze['files'].items():
        # Frozen evaluator-only annotation files are hashed as opaque bytes, never decoded.
        assert sha(ROOT/rel) == expected, rel
    for rel, item in observation['files'].items():
        done = completion['files'][rel]
        assert done['unchanged_hash_and_mtime']
        assert done['sha256'] == item['sha256'] == sha(ROOT/rel)
        assert done['mtime_ns'] == item['mtime_ns'] == (ROOT/rel).stat().st_mtime_ns
        assert item['mtime_ns'] <= observation['source_run_initial_gate_mtime_ns']
        assert freeze['files'][rel] == item['sha256']
    assert sha(pooldir/'pool_z.npy') == gate['pool_sha256'] == 'd9abccfb561f459d2f3a6c5b4346a1f6766df7fdbaddb230338aa40349077a16'
    pool = np.load(pooldir/'pool_z.npy', mmap_mode='r')
    poolnames = load(pooldir/'pool_names.json')
    names = lock['recipe']['channels_48']; columns = [poolnames.index(n) for n in names]
    roster = load(ROOT/'results/localization_full_benchmark_v3/evaluation/JOINED.json')['records']
    off = np.r_[0, np.cumsum([r['steps'] for r in roster])]
    assert len(roster) == 13769 and off[-1] == 145597
    seen = set(); max_error = np.zeros(48); source_files = {}
    for item in gate['features']:
        path = Path(item['path']); assert sha(path) == item['sha256']
        source_files[path.name] = item['sha256']
        with np.load(path, allow_pickle=False) as values:
            assert list(values['names']) == names
            x = values['features']; indexes = values['indexes']; local = 0
            assert np.isfinite(x).all()
            for idx in indexes:
                idx = int(idx); assert idx not in seen; seen.add(idx)
                n = int(off[idx+1]-off[idx]); block = x[local:local+n]; local += n
                assert len(block) == n
                delta = np.max(np.abs(standard(block)-pool[off[idx]:off[idx+1]][:,columns]), axis=0)
                assert np.max(delta) <= 1e-6, (idx, delta.max())
                max_error = np.maximum(max_error, delta)
            assert local == len(x)
    assert seen == set(range(13769))
    execution = load(EXP/'CPU_EXECUTION.json'); identity = execution['identity']
    assert identity['source_gate'] == sha(gatepath)
    assert identity['lock'] == sha(lockpath)
    assert identity['implementation_freeze'] == sha(EXP/'IMPLEMENTATION_FREEZE.json')
    assert identity['code'] == freeze['files']
    assert identity['arms'] == list(lock['rows'])
    return lock, digest(identity), {'answers': len(seen), 'steps': int(off[-1]),
             'maximum_error': float(max_error.max()), 'per_channel_error': dict(zip(names, map(float,max_error))),
             'saved_feature_files': source_files, 'gate_sha256': sha(gatepath),
             'source_code_observation_sha256': sha(observationpath),
             'source_code_completion_sha256': sha(EXP/'SOURCE_CODE_COMPLETION.json'),
             'freeze_sha256': sha(EXP/'IMPLEMENTATION_FREEZE.json'),
             'provenance_limit': 'Source code hashes first observed mid-run; all mtimes precede initial gate and hashes/mtimes unchanged through completion.'}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--pool-dir', type=Path, default=POOL)
    args = parser.parse_args(); started = time.perf_counter()
    if (OUT/'AUDIT.json').exists():
        raise FileExistsError('refuse overwrite of completed independent audit')
    if not (EXP/'ALL_CELLS_SEALED.json').exists():
        raise RuntimeError('wait for all external predictions to be sealed')
    assert not list(EXP.glob('*/shard_*/WRITER.lock')), 'active writer'
    lock, identity, source = source_and_identity(args.pool_dir)
    names = lock['recipe']['channels_48']; arms = list(lock['rows'])
    assert len(arms)==10 and len(lock['external_primary_contrasts'])==6
    plan = load(EXP/'EXECUTION_PLAN.json')
    assert plan['methods']==arms and plan['primary_family']==18 and plan['draws']==100000 and plan['seed']==20260924
    allseals = load(EXP/'ALL_CELLS_SEALED.json'); counts = {}; seals = {}
    for cell, (benchmark, expected_answers, expected_steps, expected_tokens) in CELLS.items():
        inputrows = load(PRIVATE/'inputs'/benchmark/'answers.json')
        inputs = {r['uid']:r for r in inputrows}; assert len(inputs)==len(inputrows)==expected_answers
        rawdir = PRIVATE/'evaluation_archives'/cell/'records'
        rawpaths = {p.name:p for p in rawdir.glob('*.record.json')}
        paths = sorted((EXP/cell).glob('shard_*/*.record.json'))
        assert len(paths)==len(rawpaths)==expected_answers
        assert {p.name for p in paths}==set(rawpaths)
        seen=set(); rows={}; steps=tokens=empty=checked_decisions=0
        errors={arm:0. for arm in arms if arm!='ct7'}; anchor_errors={arm:0. for arm in ANCHORS}
        ledger=[]
        for k,path in enumerate(paths):
            record=load(path); uid=record['uid']; row=record['payload']
            assert uid not in seen and uid in inputs; seen.add(uid)
            assert record['run_identity']==identity==load(path.parent/'RUN.json')['identity']
            assert path.name==digest(uid)+'.record.json'
            rawpath=rawpaths[path.name]; rawbytes=rawpath.read_bytes(); rawhash=hashlib.sha256(rawbytes).hexdigest()
            raw=json.loads(rawbytes); assert raw['uid']==uid
            assert row['telemetry_sha256']==rawhash
            telemetry=raw['payload']['telemetry']; spans=np.array(telemetry['step_token_spans'])
            assert spans.dtype.kind in 'iu' and spans.ndim==2 and spans.shape[1]==2
            t=len(telemetry['gen_token_ids']); n=len(spans)
            assert n==len(inputs[uid]['steps'])
            assert (spans>=0).all() and (spans[:,0]<=spans[:,1]).all() and (spans[:,1]<=t).all()
            valid=spans[:,1]>spans[:,0]; assert valid.any(), 'all-empty answer must be explicitly handled'
            assert row['nonempty']==valid.tolist()
            assert row['feature_names']==names
            x=np.array(row['features'], dtype=float); assert x.shape==(int(valid.sum()),48) and np.isfinite(x).all()
            assert row['tokens']==t
            assert set(row['scores'])==set(row['predictions'])==set(arms)
            referencepath=OLD/cell/'shard_000'/path.name
            referencehash=sha(referencepath); oldrecord=load(referencepath); old=oldrecord['payload']
            assert oldrecord['uid']==uid and old['telemetry_sha256']==rawhash
            assert row['reference_record_sha256']==referencehash
            assert row['nonempty']==old['nonempty']
            replay=manual_scores(x,lock)
            for arm in arms:
                score=row['scores'][arm]; pred=row['predictions'][arm]
                assert len(score)==len(pred)==n and set(pred)<={0,1}
                assert [v is not None for v in score]==valid.tolist()
                s=np.array([float(v) for v in score if v is not None])
                p=np.array(pred); assert np.isfinite(s).all() and not p[~valid].any()
                tau=lock['deployment'][arm]['q80_threshold_fold4']
                assert np.array_equal(p[valid],s<tau), (cell,uid,arm,'threshold')
                if arm!='ct7':
                    err=float(np.max(abs(s-replay[arm]))); errors[arm]=max(errors[arm],err)
                    assert err<=1e-10, (cell,uid,arm,err)
                    assert np.array_equal(p[valid],replay[arm]<tau), (cell,uid,arm,'replay decision')
                    assert abs(s.mean())<1e-10
                    assert abs(s.std()-1)<1e-8 or np.max(abs(s))<1e-10
                if arm in ANCHORS:
                    prior=ANCHORS[arm]; ref=np.array([v for v in old['scores'][prior] if v is not None])
                    err=float(np.max(abs(s-ref))); anchor_errors[arm]=max(anchor_errors[arm],err)
                    assert err<=1e-6 and pred==old['predictions'][prior]
                    if arm=='ct7': assert score==old['scores'][prior]
                checked_decisions+=n
            steps+=n; tokens+=t; empty+=int((~valid).sum()); rows[uid]=row
            ledger.append({'uid':uid,'record_sha256':sha(path),'telemetry_sha256':rawhash,
                           'reference_sha256':referencehash,'steps':n,'tokens':t,'empty_steps':int((~valid).sum())})
            if (k+1)%500==0: print(cell,k+1,'/',len(paths),flush=True)
        assert seen==set(inputs)
        assert (len(seen),steps,tokens)==(expected_answers,expected_steps,expected_tokens)
        seal=load(EXP/cell/'SEAL.json')
        assert seal['prediction_sha256']==digest(rows)
        assert seal['answers']==expected_answers and seal['arms']==arms
        assert seal['lock_sha256']==sha(ROOT/'results/family_tail_transfer_v1/TRANSFER_LOCK_V1.json')
        assert seal['implementation_freeze_sha256']==sha(EXP/'IMPLEMENTATION_FREEZE.json')
        assert allseals['seals'][cell]==seal
        seals[cell]={'seal_sha256':sha(EXP/cell/'SEAL.json'),'recomputed_prediction_sha256':digest(rows)}
        counts[cell]={'answers':len(seen),'steps':steps,'tokens':tokens,'empty_steps':empty,
                      'finite_feature_values':(steps-empty)*48,'checked_arm_decisions':checked_decisions,
                      'maximum_score_errors':errors,'anchor_maximum_score_errors':anchor_errors}
        with (OUT/(cell+'_HASHES.csv')).open('w',encoding='utf8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(ledger[0]));writer.writeheader();writer.writerows(ledger)
        print('PASS',cell,len(seen),steps,flush=True)
    result={'status':'PASS','n_checked':sum(v['answers'] for v in counts.values()),'n_total':6190,
            'source':source,'cells':counts,'seals':seals,'arms':arms,'primary_contrasts':lock['external_primary_contrasts'],
            'all_cells_seal_sha256':sha(EXP/'ALL_CELLS_SEALED.json'),'code_sha256':sha(Path(__file__)),
            'labels_decoded':False,'aggregate_quality_read':False,'other_reviews_read':False,
            'scope':'Full population coverage, raw/source/reference hashes and independent saved-feature score algebra; no duplicate raw feature extraction.',
            'elapsed_seconds':time.perf_counter()-started}
    assert result['n_checked']==6190
    write(OUT/'AUDIT.json',result)
    print('FULL INDEPENDENT AUDIT PASS',result['n_checked'],flush=True)


if __name__=='__main__':
    main()
