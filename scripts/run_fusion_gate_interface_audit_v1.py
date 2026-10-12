"""Phased, source-bound audit of existing fusion normalization and gating."""
import os
for option in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[option]='1'
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'results/answer_localization_representation_pilot_v1'
LATEST=ROOT/'results/fusion_reliability_regularization_v1'
OUT=ROOT/'results/fusion_gate_interface_audit_v1'
sys.path.insert(0,str(ROOT/'local_cache/short_cycle01_code'))
import spectral_utils
spectral_utils.__path__.append(str(ROOT/'spectral_utils'))
from spectral_utils.answer_localization_v2 import prepare_local, PRIMITIVES, mixture_readout
from spectral_utils.fusion_gate_interface_audit import (
    ARMS,REP,PARENT_ARMS,projection,inspect_mixture,comparison_parts,oracle_predictions,pb_from_predictions)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def safe(value):
    if isinstance(value,dict): return {str(k):safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)): return [safe(v) for v in value]
    if isinstance(value,(bool,np.bool_)): return bool(value)
    if isinstance(value,np.integer): return int(value)
    if isinstance(value,(float,np.floating)): return float(value) if np.isfinite(value) else None
    return value
def save(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(safe(value),indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)


def verify():
    manifest=load(OUT/'MANIFEST.json')
    for path,digest in manifest['hashes'].items(): assert sha(path)==digest,path
    return manifest


def prepare():
    if (OUT/'MANIFEST.json').exists(): verify();print('Existing manifest verified.');return
    parent=load(LATEST/'MANIFEST.json'); frozen=load(LATEST/'SCORES_FROZEN.json')
    assert frozen['manifest_sha256']==sha(LATEST/'MANIFEST.json')
    hashes={**parent['hashes'],**frozen['files']}
    for path in (LATEST/'MANIFEST.json',LATEST/'SCORES_FROZEN.json',LATEST/'EVALUATION.json',
                 Path(__file__),ROOT/'spectral_utils/fusion_gate_interface_audit.py',
                 ROOT/'tests/test_fusion_gate_interface_audit.py',
                 ROOT/'docs/experiments/FUSION_GATE_INTERFACE_AUDIT_V1.md'):
        hashes[str(path)]=sha(path)
    for path,digest in hashes.items(): assert sha(path)==digest,path
    save(OUT/'MANIFEST.json',{'release_id':parent['release_id'],'selected':parent['selected'],
         'arms':ARMS,'hashes':hashes,'created_unix':time.time(),
         'status':'ADAPTIVE_DEVELOPMENT_DIAGNOSTIC','targets_used_for_new_fitting':False})
    print('Frozen audit protocol, code and parents; 58 existing answers.',flush=True)


def measure_one(record):
    uid=record['uid']; original=load(PARENT/'scores'/f'{uid}.json')['report']
    latest=load(LATEST/'scores'/f'{uid}.json')
    normal=original['representations'][REP]['shared']
    names=[s+'__'+op for s in PRIMITIVES for op in ('level','sd','slope')]
    with np.load(PARENT/'scores'/f'{uid}.npz',allow_pickle=False) as p:
        values=p[REP+'__features']; fit=p[REP+'__fit_indices']
        z,anchor,again=prepare_local(values,names,fit)
        for key in ('mean','sd','feature_signs'): np.testing.assert_allclose(again[key],normal[key],atol=1e-12)
        feature_sd=values[fit].std(axis=0)
        changed=values*np.linspace(.7,1.3,values.shape[1])+feature_sd*np.linspace(-1.,1.,values.shape[1])
        zz,other_anchor,other=prepare_local(changed,names,fit)
        assert again['active_features']==other['active_features'] and anchor==other_anchor
        np.testing.assert_allclose(z,zz,atol=1e-8,rtol=1e-8)
        affine_error=float(np.max(np.abs(z-zz)))
        raw_entropy=values[:,names.index('entropy_series__level')]
        raw_spilled=values[:,names.index('spilled_series__level')]
        arrays={}; details={}
        with np.load(LATEST/'scores'/f'{uid}.npz',allow_pickle=False) as a:
            for arm in ARMS:
                method=latest['methods'][arm]
                if not method.get('valid'):
                    details[arm]={'valid':False,'reason':method.get('reason',method.get('status','PARENT_INVALID'))};continue
                risk,steps=a[arm+'__window'],a[arm+'__risk']
                if arm=='entropy_parent':
                    uncentered=raw_entropy/raw_entropy[fit].std()
                    offset=float(raw_entropy[fit].mean()/raw_entropy[fit].std())
                    centered=uncentered-offset
                else:
                    weights=a[arm+'__weights_grid'][method['selected_index']] if arm=='iu__dufs_graph' else original['methods'][PARENT_ARMS[arm]]['standardized_weights']
                    centered,uncentered,offset=projection(values,names,fit,normal,weights)
                np.testing.assert_allclose(centered,risk,atol=1e-9,rtol=1e-9)
                np.testing.assert_allclose(uncentered,risk+offset,atol=1e-9,rtol=1e-9)
                assert abs(risk[fit].mean())<1e-9 and abs(risk[fit].std()-1)<1e-9
                gate=inspect_mixture(risk[fit],steps); reference=mixture_readout(risk[fit],steps)
                assert gate['prediction']==reference['prediction']==method['gate_readout']['prediction']
                np.testing.assert_allclose(gate['bic'],reference['bic'],atol=1e-9)
                shifted=inspect_mixture(risk[fit]+10,steps+10)
                assert shifted['prediction']==gate['prediction']
                assert shifted['two_components_selected']==gate['two_components_selected']
                assert abs(shifted['bic_gain']-gate['bic_gain'])<1e-6
                if gate['threshold'] is not None: assert abs(shifted['threshold']-gate['threshold']-10)<1e-7
                peak=int(np.argmax(steps)); opened=gate['prediction']!=-1
                prediction=peak if opened else -1
                assert prediction==method['prediction']
                arrays[arm+'__risk']=steps; arrays[arm+'__origin_projection']=steps+offset
                details[arm]={'valid':True,'gate':gate,'gate_open':opened,'peak':peak,'prediction':prediction,
                    'fixed_parent_prediction':method['fixed_parent_gate_prediction'],
                    'fit_mean':float(risk[fit].mean()),'fit_sd':float(risk[fit].std()),
                    'origin_offset':offset,'score_replay_error':float(np.max(np.abs(centered-risk))),
                    'translation_bic_error':abs(shifted['bic_gain']-gate['bic_gain'])}
    return arrays,{**record,'affine_matrix_error':affine_error,'methods':details,
        'raw_entropy_mean':float(raw_entropy[fit].mean()),'raw_spilled_mean':float(raw_spilled[fit].mean()),
        'labels_decoded':False}


def measure():
    manifest=verify(); digest=sha(OUT/'MANIFEST.json'); started=time.monotonic()
    if (OUT/'DIAGNOSTICS_FROZEN.json').exists():
        frozen=load(OUT/'DIAGNOSTICS_FROZEN.json')
        assert frozen['manifest_sha256']==digest
        for path,h in frozen['files'].items(): assert sha(path)==h,path
        print('Frozen diagnostics verified; no rerun.');return
    for i,record in enumerate(manifest['selected'],1):
        uid=record['uid']; dest=OUT/'measurements'/f'{uid}.npz'; meta=dest.with_suffix('.json')
        if meta.exists():
            info=load(meta); assert info['manifest_sha256']==digest and info['array_sha256']==sha(dest)
        else:
            arrays,detail=measure_one(record);dest.parent.mkdir(exist_ok=True,parents=True)
            with dest.with_suffix('.npz.tmp').open('wb') as stream: np.savez_compressed(stream,**arrays)
            dest.with_suffix('.npz.tmp').replace(dest)
            save(meta,{**detail,'manifest_sha256':digest,'array_sha256':sha(dest)})
        save(OUT/'RUN_STATE.json',{'state':'RUNNING','pid':os.getpid(),'completed':i,'total':len(manifest['selected'])})
        if i%10==0 or i==len(manifest['selected']): print(f'Measured {i}/{len(manifest["selected"])}',flush=True)
    verify();paths=sorted((OUT/'measurements').glob('*.npz'))+sorted((OUT/'measurements').glob('*.json'))
    assert len(paths)==116
    save(OUT/'DIAGNOSTICS_FROZEN.json',{'manifest_sha256':digest,'files':{str(p):sha(p) for p in paths},
        'seconds':time.monotonic()-started,'labels_decoded':False})
    save(OUT/'RUN_STATE.json',{'state':'COMPLETE','completed':58,'total':58,'seconds':time.monotonic()-started})
    print('Unlabeled diagnostics frozen.',flush=True)


def binary_auc(y,x): return comparison_parts([y],[x])['pooled_auc']


def evaluate():
    manifest=verify(); frozen=load(OUT/'DIAGNOSTICS_FROZEN.json')
    assert frozen['manifest_sha256']==sha(OUT/'MANIFEST.json') and frozen['labels_decoded'] is False
    for p,h in frozen['files'].items(): assert sha(p)==h,p
    release=load(PARENT/'RELEASE.json'); source=load(LATEST/'EVALUATION.json'); prior={r['uid']:r for r in source['rows']}
    rows=[];counts=Counter()
    for cell in sorted({r['cell'] for r in manifest['selected']}):
        info=release['cells'][cell];assert sha(info['label_path'])==info['label_opaque_sha256']
        with np.load(info['label_path'],allow_pickle=False) as labels:
            positions={str(row):i for i,row in enumerate(labels['row_ids'])};assert len(positions)==len(labels['row_ids'])
            for rec in (r for r in manifest['selected'] if r['cell']==cell):
                i=positions[rec['row_id']]
                if cell.startswith('prm'):
                    lo,hi=labels['step_flag_offsets'][i:i+2];target=labels['step_error_flags'][lo:hi]
                else: target=int(labels['first_error'][i])
                np.testing.assert_array_equal(target,prior[rec['uid']]['target']);counts['label_joins']+=1
                meta=load(OUT/'measurements'/f"{rec['uid']}.json")
                with np.load(OUT/'measurements'/f"{rec['uid']}.npz",allow_pickle=False) as arrays:
                    methods={}
                    for arm,detail in meta['methods'].items():
                        assert detail['valid']==prior[rec['uid']]['decision_valid'][arm]
                        methods[arm]=dict(detail)
                        if detail['valid']:
                            np.testing.assert_array_equal(arrays[arm+'__risk'],prior[rec['uid']]['scores'][arm]);counts['score_replays']+=1
                            methods[arm]['risk']=arrays[arm+'__risk'];methods[arm]['origin_projection']=arrays[arm+'__origin_projection']
                rows.append({**rec,'target':target,'methods':methods,
                    'raw_entropy_mean':meta['raw_entropy_mean'],'raw_spilled_mean':meta['raw_spilled_mean'],
                    'affine_matrix_error':meta['affine_matrix_error']})
    summaries={}
    for arm in ARMS:
        pbrows=[]; gate_counts=Counter();location=Counter();duplication=Counter()
        for row in (r for r in rows if r['cell'].startswith('pb_')):
            d=row['methods'][arm];valid=d['valid'];target=row['target'];error=target!=-1
            preds=oracle_predictions(target,valid,d.get('gate_open',False),d.get('peak',0))
            preds['fixed_parent_gate']=d.get('fixed_parent_prediction') if valid else None
            pbrows.append({'cell':row['cell'],'uid':row['uid'],'target':target,'predictions':preds})
            label='error' if error else 'clean'
            gate_counts[label+'_total']+=1
            if not valid:
                gate_counts[label+'_invalid']+=1
                if error: location['invalid']+=1
                continue
            gate_counts[label+('_gate_open' if d['gate_open'] else '_gate_closed')]+=1
            gate_counts[label+'_two_components']+=int(d['gate']['two_components_selected'])
            if error:
                key='exact' if d['peak']==target else ('before' if d['peak']<target else 'after')
                location[key]+=1
                gate_counts['exact_peak_gated_out']+=int(key=='exact' and not d['gate_open'])
            duplication['valid']+=1
            duplication['original_two_components']+=int(d['gate']['two_components_selected'])
            duplication['duplicated_fixed_parameter_two_components']+=int(d['gate']['duplicated_fixed_parameter_bic_gain']>0)
        pb={key:pb_from_predictions(pbrows,key) for key in ('actual','perfect_gate','perfect_locator','both_perfect','fixed_parent_gate')}
        assert abs(pb['actual']['macro_f1']-source['metrics'][arm]['pb']['macro_f1'])<1e-12;counts['endpoint_replays']+=1
        available=[r for r in rows if r['cell'].startswith('prm') and r['methods'][arm]['valid']]
        prm={key:comparison_parts([r['target'] for r in available],[r['methods'][arm][key] for r in available])
             for key in ('risk','origin_projection')}
        assert abs(prm['risk']['pooled_auc']-source['metrics'][arm]['prm']['auroc'])<1e-12;counts['endpoint_replays']+=1
        assert abs(prm['risk']['mean_within_answer_auc']-prm['origin_projection']['mean_within_answer_auc'])<1e-12
        probes={}
        for cell in sorted({r['cell'] for r in pbrows}):
            subset=[r for r in rows if r['cell']==cell and r['methods'][arm]['valid']]
            y=[int(r['target']!=-1) for r in subset]
            probes[cell]={'answers':len(subset),'erroneous':sum(y),'clean':len(y)-sum(y),
                'bic_gain_auc':binary_auc(y,[r['methods'][arm]['gate']['bic_gain'] for r in subset]),
                'raw_entropy_mean_auc':binary_auc(y,[r['raw_entropy_mean'] for r in subset]),
                'raw_spilled_mean_auc':binary_auc(y,[r['raw_spilled_mean'] for r in subset])}
        summaries[arm]={'pb':pb,'pb_rows':pbrows,'gate_counts':dict(gate_counts),'peak_location':dict(location),
            'duplication_stress_pb':dict(duplication),'prm':prm,'prm_valid_answers':len(available),'pb_probe_auc':probes}
    save(OUT/'EVALUATION.json',{'status':'RETROSPECTIVE_DIAGNOSTIC_ONLY','manifest_sha256':sha(OUT/'MANIFEST.json'),
        'diagnostics_sha256':sha(OUT/'DIAGNOSTICS_FROZEN.json'),'rows':rows,'summaries':summaries,'checks':dict(counts)})
    print(json.dumps({'checks':dict(counts),'pb':{a:{k:round(v['macro_f1'],5) for k,v in s['pb'].items()}
        for a,s in summaries.items()},'peak_counts':{a:s['peak_location'] for a,s in summaries.items()}},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=('prepare','measure','evaluate'),required=True)
    args=parser.parse_args();{'prepare':prepare,'measure':measure,'evaluate':evaluate}[args.phase]()
