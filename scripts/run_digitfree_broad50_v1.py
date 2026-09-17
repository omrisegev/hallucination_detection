"""One frozen full-population broad50 experiment. Resumable cell/fold outputs."""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
from spectral_utils.digitfree_broad50 import NAMES, ANCHOR, step_bank, masked_answer_standardize
from spectral_utils.joint_block_balanced import discover_sourcefold_groups, fit_block_balanced
from spectral_utils.joint_lsml import covariance_matrix, hierarchical_joint_weights
from spectral_utils.joint_pair_jacobian import fit_joint_pairs_checked
from spectral_utils.lsml_gate_locator_research import FusionRecipe, fit_fusion_weights, _orient
from scripts.run_lsml_gate_locator_research_v1 import dump, score_locator, scalar_metrics, rank_gate_score

OUT = ROOT/'results/digitfree_broad50_v1'
ATLAS = ROOT/'results/fusion_independence_atlas_v1'
SEED = 399170
ARMS = ('entropy_H1', 'equal50', 'continuous', 'joint', 'joint_balanced')


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def state(status, **extra):
    dump(OUT/'RUN_STATE.json', dict(status=status, time=time.strftime('%Y-%m-%dT%H:%M:%S'), **extra))
    print(status, extra, flush=True)


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    files = [Path(__file__), ROOT/'spectral_utils/digitfree_broad50.py', ROOT/'spectral_utils/joint_block_balanced.py',
        ROOT/'docs/experiments/DIGITFREE_BROAD50_V1.md', SOURCE/'results/localization_full_benchmark_v3/evaluation/JOINED.json',
        ATLAS/'dependence/EVALUATION.npz', ATLAS/'baseline_replay/SCORES_FROZEN.npz',
        ATLAS/'bundle/data/METADATA.json']
    files += [ROOT/'spectral_utils'/name for name in ('joint_lsml.py','joint_pair_extension.py',
        'joint_pair_jacobian.py','fusion_utils.py','lsml_gate_locator_research.py','renyi_alpha_sweep.py',
        'renyi_view_fusion.py','historical_fusion_evaluation.py')]
    manifest = dict(schema='digitfree-broad50-v1', names=NAMES, seed=SEED, arms=ARMS,
                    hashes={str(p):sha(p) for p in files})
    path = OUT/'MANIFEST.json'
    canonical = json.loads(json.dumps(manifest))
    if path.exists() and json.loads(path.read_text()) != canonical:
        raise ValueError('Frozen run changed; do not overwrite its results')
    dump(path, manifest)


def extract():
    from scripts import run_direct_probability_temporal as e
    from scripts.run_fusion_independence_atlas_v1 import RAW_SOURCE_HASHES
    e.old.configure_source_root(SOURCE)
    records = json.loads((SOURCE/'results/localization_full_benchmark_v3/evaluation/JOINED.json').read_text())['records']
    metadata = json.loads((ATLAS/'bundle/data/METADATA.json').read_text())
    assert len(records) == len(metadata) == 13769
    folder = OUT/'extracted'; folder.mkdir(exist_ok=True)
    for cell, path, kind, dataset in e.source_specs():
        dest = folder/f'{cell}.npz'
        if dest.exists(): continue
        relative = path.relative_to(SOURCE).as_posix()
        state('VERIFY_SOURCE', cell=cell, bytes=path.stat().st_size)
        digest = sha(path)
        if digest != RAW_SOURCE_HASHES[relative]: raise ValueError(f'raw source hash differs: {relative}')
        source = e.old._source_row_map(e.old.load_pickle(path), kind=kind, dataset=dataset)
        indexes = [i for i,r in enumerate(records) if r['cell'] == cell]
        blocks, masks, signs = [], [], []
        for pos, i in enumerate(indexes):
            record = records[i]; row = source[record['row_id']]; m = metadata[i]
            if m['uid'] != record['uid']: raise ValueError('metadata identity mismatch')
            payload = e.old._topk_payload(row)
            if len(payload['logprobs']) != record['tokens']: raise ValueError('token count mismatch')
            spans = np.asarray(row['step_token_spans'], int)
            if spans.shape != (record['steps'], 2): raise ValueError('step count mismatch')
            block, available, sign = step_bank(payload['logprobs'], payload['ids'], row['gen_token_ids'],
                                               row['token_spilled_energies'], spans)
            blocks.append(block); masks.append(available); signs.append(sign)
            if (pos+1) % 250 == 0: state('EXTRACTING', cell=cell, completed=pos+1, expected=len(indexes))
        with dest.with_suffix('.tmp').open('wb') as f:
            np.savez_compressed(f, indexes=indexes, values=np.vstack(blocks).astype(np.float32),
                                available=np.vstack(masks), signs=signs, source_sha256=digest)
        os.replace(dest.with_suffix('.tmp'), dest)
        del source, blocks, masks
        state('CELL_COMPLETE', cell=cell, answers=len(indexes))


def load_data():
    with np.load(ATLAS/'dependence/EVALUATION.npz') as z:
        data = {k:z[k] for k in ('target','labels','offsets','folds','cells','groups','tokens','steps')}
    with np.load(ATLAS/'baseline_replay/SCORES_FROZEN.npz') as z:
        raw = z['gate_raw'].copy()
        data['references'] = {name:z[key].copy() for name,key in (
            ('original4', 'steps__mean__H0lim_VE0_VE075_VE1'), ('innovation5', 'steps__append_innovation__H0lim'))}
    pb = np.char.startswith(data['cells'].astype(str), 'pb_')
    _, data['gate'] = rank_gate_score(raw, data['cells'], pb)
    offsets = data['offsets']; x = np.full((offsets[-1],50), np.nan); mask = np.zeros(x.shape,bool)
    completed = np.zeros(len(offsets)-1,bool)
    for path in sorted((OUT/'extracted').glob('*.npz')):
        with np.load(path) as z:
            values = z['values']; available = z['available']; indexes = z['indexes']
            cursor = 0
            for i in indexes:
                a,b = offsets[i:i+2]; n=b-a
                if completed[i]: raise ValueError('duplicate answer')
                x[a:b] = values[cursor:cursor+n]; mask[a:b] = available[cursor:cursor+n]
                cursor += n; completed[i] = True
    if not completed.all(): raise ValueError('incomplete population')
    data['available'] = mask
    data['x'] = masked_answer_standardize(x, mask, offsets)
    for g in np.unique(data['groups']):
        if len(np.unique(data['folds'][data['groups']==g])) != 1: raise ValueError('source leakage')
    return data


def run_folds(data):
    x = data['x']; offsets = data['offsets']
    rowfolds = np.repeat(data['folds'], np.diff(offsets))
    scores = {arm:np.empty(len(x)) for arm in ARMS}; details=[]
    for outer in range(5):
        dest = OUT/f'fold_{outer}.npz'; info = OUT/f'fold_{outer}.json'
        test = rowfolds == outer
        if dest.exists() and info.exists():
            with np.load(dest) as z:
                for arm in ARMS: scores[arm][test] = z[arm]
            details.append(json.loads(info.read_text())); continue
        start = time.perf_counter(); train = x[~test]
        weights = {'entropy_H1':np.eye(50)[ANCHOR], 'equal50':np.ones(50)/50}
        meta = {'outer':outer, 'arms':{}, 'train_answers':int(np.sum(data['folds']!=outer))}
        state('FIT_CONTINUOUS', outer=outer)
        try:
            weights['continuous'], m = fit_fusion_weights(train, FusionRecipe('broad50',NAMES,'continuous',anchor=ANCHOR),seed=SEED+outer)
            meta['arms']['continuous'] = dict(valid=True, details=m)
        except (ValueError, np.linalg.LinAlgError, RuntimeError) as exc:
            meta['arms']['continuous'] = dict(valid=False, failure=str(exc))
        state('GROUP_DISCOVERY', outer=outer)
        discovery = discover_sourcefold_groups(train, rowfolds[~test],seed=SEED+100+outer)
        meta['discovery'] = discovery
        if discovery['status'] == 'SELECTED':
            labels = discovery['labels']; cov = covariance_matrix(train)
            for arm in ('joint','joint_balanced'):
                state('FIT_JOINT', outer=outer, arm=arm, sizes=discovery['group_sizes'])
                try:
                    if arm == 'joint':
                        fit = fit_joint_pairs_checked(cov,labels,anchor_index=ANCHOR,seed=SEED+200+outer,starts=5)
                        joint = fit.joint; v=joint.global_loading
                        audit = dict(valid=bool(joint.converged and joint.multistart_audit['status']=='PASS'
                            and joint.jacobian_audit['full_global_rank'] and joint.jacobian_audit['condition_number']<=1e8),
                            converged_starts=joint.converged_starts, multistart=joint.multistart_audit,
                            jacobian=joint.jacobian_audit, pair=fit.pair_audit,
                            relative_offdiag_misfit=joint.relative_offdiag_misfit,
                            sweeps=[r.sweeps for r in joint.starts])
                    else:
                        v, _, _, audit = fit_block_balanced(cov,labels,anchor_index=ANCHOR,seed=SEED+200+outer)
                    if audit['valid']:
                        _, w, readout = hierarchical_joint_weights(train,labels,v,anchor_index=ANCHOR,small_m_guard=True)
                        weights[arm], orientation = _orient(train,w,ANCHOR)
                        audit['readout'] = readout; audit['orientation'] = orientation
                    meta['arms'][arm] = audit
                except (ValueError, np.linalg.LinAlgError, RuntimeError) as exc:
                    meta['arms'][arm] = dict(valid=False, failure=f'{type(exc).__name__}: {exc}')
        for arm in ARMS:
            if arm not in weights:
                weights[arm] = weights['entropy_H1'].copy()
                meta['arms'].setdefault(arm,dict(valid=False,failure='NO_ADMISSIBLE_PARTITION'))
                meta['arms'][arm]['fallback'] = 'entropy_H1'
            scores[arm][test] = x[test]@weights[arm]
        meta['weights']=weights; meta['seconds']=time.perf_counter()-start
        dump(info,meta)
        np.savez_compressed(dest, **{arm:scores[arm][test] for arm in ARMS})
        details.append(meta); state('FOLD_COMPLETE',outer=outer,seconds=meta['seconds'])
    return scores,details


def uncertainty(data, results):
    """Vectorized group sufficient statistics for paired source bootstrap."""
    target=data['target']; cells=data['cells'].astype(str)
    unique,g=np.unique(data['groups'],return_inverse=True); ng=len(unique)
    arms=('joint','joint_balanced'); cellnames=sorted(c for c in set(cells) if c.startswith('pb_'))
    stats=np.zeros((ng,len(cellnames),6))
    for j,c in enumerate(cellnames):
        clean=(cells==c)&(target==-1); error=(cells==c)&(target>=0)
        vectors=[clean,error]
        for arm in arms:
            correct=results[arm]['prediction']==target
            vectors += [clean&correct,error&correct]
        for k,v in enumerate(vectors): stats[:,j,k]=np.bincount(g,weights=v,minlength=ng)
    within=np.zeros((ng,3)); v0=results[arms[0]]['within_values']; valid=np.isfinite(v0)
    within[:,0]=np.bincount(g,weights=valid,minlength=ng)
    for j,arm in enumerate(arms):
        v=results[arm]['within_values']; assert np.array_equal(valid,np.isfinite(v))
        within[:,j+1]=np.bincount(g,weights=np.nan_to_num(v),minlength=ng)
    rng=np.random.default_rng(SEED+999); pb=[]; wa=[]
    for begin in range(0,10000,100):
        counts=rng.multinomial(ng,np.full(ng,1/ng),size=100)
        total=np.einsum('bg,gck->bck',counts,stats); accuracy=[]
        for j in range(2):
            ca=total[:,:,2+2*j]/total[:,:,0]; ea=total[:,:,3+2*j]/total[:,:,1]
            accuracy.append(np.divide(2*ca*ea,ca+ea,out=np.zeros_like(ca),where=(ca+ea)>0).mean(axis=1))
        pb.extend(accuracy[1]-accuracy[0]); sums=counts@within
        wa.extend((sums[:,2]-sums[:,1])/sums[:,0])
    out={}
    for endpoint,draws in (('pb',pb),('within',wa)):
        out[endpoint]=dict(point=results[arms[1]][endpoint]-results[arms[0]][endpoint],
            low=float(np.quantile(draws,.0125)),high=float(np.quantile(draws,.9875)),confidence=.975,draws=10000)
    return out


def main():
    freeze(); started=time.perf_counter(); extract(); data=load_data()
    state('EXTRACTION_COMPLETE',answers=len(data['target']),steps=len(data['x']),missing=int((~data['available']).sum()))
    scores,details=run_folds(data); scores.update(data['references'])
    state('EVALUATING')
    results={arm:score_locator(s,data['gate'],data) for arm,s in scores.items()}
    metrics={arm:scalar_metrics(r) for arm,r in results.items()}
    coverage={arm:sum(int(np.sum(data['folds']==f['outer'])) for f in details
        if f['arms'].get(arm,{}).get('valid',arm in ('entropy_H1','equal50'))) for arm in ARMS}
    contrast=uncertainty(data,results)
    report=dict(status='COMPLETE',scope='full development; pooled other-source fitting; transductive non-digit gate',
        metrics=metrics,native_answers=coverage,primary_contrast=contrast,
        answers=len(data['target']),steps=len(data['x']),missing_by_feature=dict(zip(NAMES,(~data['available']).sum(axis=0))),
        seconds_this_launch=time.perf_counter()-started)
    np.savez_compressed(OUT/'SCORES.npz',**scores,gate=data['gate'])
    dump(OUT/'RESULTS.json',report)
    lines=['# Broad50 without digit features — development result','',
        'Full 13,769 answers. Source-fold pooled fitting; fixed non-digit tail15 gate.', '',
        '| Arm | PB macro % | PRMB within-answer AUC | Native answers |','|---|---:|---:|---:|']
    for arm,m in metrics.items():
        lines.append(f"| {arm} | {100*m['pb']:.4f} | {m['within']:.6f} | {coverage.get(arm,'historical')} |")
    lines += ['', 'Primary paired contrast (balanced minus ordinary Joint), 97.5% source-group intervals:',
              '', '```json',json.dumps(contrast,indent=2),'```', '',
              'Invalid fits use the predeclared H1 fallback; inspect native coverage before attributing performance to Joint.',
              'This single broad-bank test is development evidence, not untouched confirmation or an exact redundancy-invariance test.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    state('COMPLETE',metrics=metrics,primary_contrast=contrast,native_answers=coverage)


if __name__ == '__main__':
    try: main()
    except Exception as exc:
        state('FAILED',error=f'{type(exc).__name__}: {exc}')
        raise
