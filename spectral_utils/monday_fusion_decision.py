"""Recover raw bank values for R2; entropy-only gate and fixed top-k controls."""
import json
import numpy as np
from spectral_utils.monday_fusion_protocol import ROOT,SSL,WORK,sha,dump
from spectral_utils.monday_fusion_metrics import answer_z,top_flags

def raw_pool(data):
    """Rebuild the original pool before answer normalization; verify its full replay."""
    ref=SSL/'results/external_banks_v4/reference_arrays'
    hashes=json.loads((ref/'HASHES.json').read_text(encoding='utf8'))
    historical=json.loads((SSL/'results/indbank_lsml_prmbench_v1/run_20260924/INPUT_MANIFEST.json').read_text(encoding='utf8'))
    freeze=json.loads((ROOT/'.worktrees/readout-quickest-detection-v1/results/step_evidence_v1/INPUT_FREEZE.json').read_text(encoding='utf-8-sig'))
    inputs={
        'union':{'path':str(ref/'union_top10_profiles.npy'),'sha256':hashes['union_top10_profiles.npy']['sha256']},
        'hist':{'path':str(ref/'hist29_top10_profiles.npy'),'sha256':hashes['hist29_top10_profiles.npy']['sha256']},
        'profiles':historical['profiles_full'],'risk':freeze['mindgap'],'tokens':freeze['tokens']}
    for name,spec in inputs.items():
        actual=sha(spec['path'])
        if actual!=spec['sha256']: raise RuntimeError('Raw-bank input mismatch '+name)
    manifest={'version':3,'parent':data.manifest,'additional_inputs':inputs}; path=WORK/'RAW_INPUT_MANIFEST_V3.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf8'))!=manifest: raise RuntimeError('Raw manifest changed; do not overwrite frozen V3')
    else: dump(path,manifest)
    union=np.load(inputs['union']['path'],mmap_mode='r'); hist=np.load(inputs['hist']['path'],mmap_mode='r')
    profiles=np.load(inputs['profiles']['path'],mmap_mode='r')
    shape=np.ascontiguousarray(profiles[:,0,[14,11,12,10]]).astype(float)
    with np.load(inputs['risk']['path']) as ed: risk=ed['risk_token'].astype(float)
    with np.load(inputs['tokens']['path']) as tk: toff=tk['token_offsets']; spans=tk['step_spans']
    p=data.pop; ev=np.empty(p.total)
    for i in range(p.n):
        r=risk[toff[i]:toff[i+1]]
        for step in range(p.off[i],p.off[i+1]):
            a,b=spans[step]; v=r[a:b]; k=min(10,len(v))
            if k==0: raise ValueError('Empty evidence-drop span')
            ev[step]=np.partition(v,len(v)-k)[-k:].mean()
    x=np.column_stack([data.raw13[:,:11],union[:,11:],shape,ev,hist])
    if x.shape!=data.env['POOL'].shape: raise ValueError('Raw pool shape mismatch')
    normalized=np.empty_like(x)
    for a,b in zip(p.off[:-1],p.off[1:]):
        block=x[a:b]
        if np.isnan(block).any():
            mean=np.nanmean(block,0); mean=np.where(np.isfinite(mean),mean,0.)
            where=np.where(np.isnan(block)); block[where]=mean[where[1]]
        mean=block.mean(0); sd=block.std(0)
        normalized[a:b]=np.divide(block-mean,sd,out=np.zeros_like(block),where=sd>1e-12)
    deviation=float(np.max(np.abs(normalized-data.env['POOL'])))
    if not np.isfinite(x).all() or deviation>1e-9: raise RuntimeError(f'Raw bank replay differs from frozen pool: {deviation}')
    dump(WORK/'RAW_BANK_REPLAY.json',{'status':'PASS','max_abs_difference':deviation,'n_values':int(x.size),'shape':list(x.shape)})
    return x

def raw_variant(data,bank,bk,pool_raw):
    e=data.env; columns=[]
    for name in bank['names']:
        if name in e['names11']+['realized_z','realized_drv'] and bk in ('B13','B16'):
            value=data.raw13[bank['rows'],(e['names11']+['realized_z','realized_drv']).index(name)]
        elif name in e['DIG']:
            value=data.digit_raw[bank['rows'],e['DIG'].index(name)]
        else:
            underlying=name[4:] if name.startswith('lf__') else name
            value=pool_raw[bank['rows'],e['PN'].index(underlying)]
            if name.startswith('lf__'): value=value*bank['diagnostics']['lf_signs'][name]['sign']
        columns.append(value)
    out=np.column_stack(columns)
    if not np.isfinite(out).all(): raise ValueError('Non-finite raw decision bank')
    return out

def entropy_feature(data):
    z=np.load(data.manifest['inputs']['answer_features']['path'])
    if not np.array_equal(z['ids'].astype(str),data.pop.ids) or not np.array_equal(z['cells'].astype(str),data.pop.cells) or not np.array_equal(z['folds'],data.pop.fold):
        raise RuntimeError('Answer entropy feature identity mismatch')
    names=list(map(str,z['names'])); value=z['X'][:,names.index('epr')]
    if not np.isfinite(value).all(): raise ValueError('Non-finite mean token entropy')
    return value

def rules(score,bank,raw_survivors,entropy,uids):
    off=bank['off']; ns=np.diff(off); az=answer_z(score,off); tau=float(np.quantile(az,.8))
    base=az>=tau
    mean=raw_survivors.mean(0); sd=np.maximum(raw_survivors.std(0),1e-12)
    global_score=((raw_survivors-mean)/sd).mean(1); tau_g=float(np.quantile(global_score,.8))
    count=np.add.reduceat((global_score>=tau_g).astype(int),off[:-1])
    e=(entropy-entropy.mean())/max(entropy.std(),1e-12)
    offset=az+np.repeat(e,ns); offset_flags=offset>=np.quantile(offset,.8)
    offset_zero=~np.add.reduceat(offset_flags.astype(int),off[:-1]).astype(bool)
    close=int(np.ceil(offset_zero.mean()*len(ns)))
    order=np.lexsort((np.asarray(uids,str),entropy)); gate=np.ones(len(ns),bool); gate[order[:close]]=False
    flags={'ANSWER_Z_Q80':base,'R2_COUNT':top_flags(score,off,count),
           'ENTROPY_GATE_Q80':base&np.repeat(gate,ns)}
    detectors={'ANSWER_Z_Q80':np.array([score[a:b].max() for a,b in zip(off[:-1],off[1:])]),
               'R2_COUNT':np.array([global_score[a:b].mean() for a,b in zip(off[:-1],off[1:])]),'ENTROPY_GATE_Q80':entropy}
    return flags,detectors,{'q80':tau,'q80_raw_global':tau_g,'entropy_offset_zero_fraction':float(offset_zero.mean()),
                            'entropy_closed_count':close,'entropy_closed_fraction':float((~gate).mean()),
                            'actual_final_zero_fraction':float(np.mean(np.add.reduceat(flags['ENTROPY_GATE_Q80'].astype(int),off[:-1])==0)),
                            'adaptation':'pure entropy gate; target close rate from entropy OFFSET; raw-score maxima/mean-G/entropy are continuous answer detectors'}
