"""Resumable, fold-excluded conditional IU follow-ups on frozen token caches.

No experiment starts on import. Example:
python scripts/run_conditional_iu_fusion.py --source-root ROOT --family position --phase smoke
"""
from __future__ import annotations

import os
for _variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_variable] = '1'

import argparse
from contextlib import contextmanager
import gc
import hashlib
import importlib
import json
from pathlib import Path
import signal
import sqlite3
import sys
import time
import traceback

import numpy as np
import scipy
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_answer_position_fusion as h
from spectral_utils import conditional_iu_fusion as model

hy, old, base = h.hy, h.old, h.base
OUT = ROOT / 'results/conditional_iu_fusion_v1'
START = 0.0
FAMILY = 'position'
METHODS = model.FAMILIES[FAMILY]
PRIMARY_CI = 1.0 - 0.05 / 3.0
DRAW_COUNT = 10000
EXPECTED_ANSWERS = 13769
INVOCATION_SECONDS = 8 * 3600
STOP_REQUESTED = False


class EndpointUnavailable(Exception):
    """The population was scored, but a registered endpoint is undefined."""


def json_ready(value):
    """Keep unavailable numerical summaries explicit as JSON null, never zero."""
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def emit(name, value):
    base.atomic_json(OUT / name, json_ready(value))


def cap_check():
    if STOP_REQUESTED or (START > 0 and time.monotonic() - START >= INVOCATION_SECONDS):
        raise hy.InvocationCap()


def progress(stage, **kwargs):
    emit('RUN_STATE.json', dict(status=stage, family=FAMILY, pid=os.getpid(),
         elapsed_seconds=time.monotonic() - START, **kwargs))
    print(stage, kwargs, flush=True)
    cap_check()


def pairs_for(family):
    methods = model.FAMILIES[family]
    proposed = [model.PRIMARY[family], ('pooled', 'baseline'), ('position', 'pooled')]
    proposed += [(name, name + '_shuffled') for name in methods if name + '_shuffled' in methods]
    proposed += [(name, 'baseline') for name in methods if name != 'baseline']
    proposed += [(name, 'top10') for name in methods]
    return list(dict.fromkeys(pair for pair in proposed if all(n in methods + ('top10',) for n in pair)))


def manifest(source, smoke, selected):
    # h.manifest binds the existing nine caches, original model DB, all 13
    # reference rows, benchmark labels, groups/folds, and entropy gate files.
    hashes = dict(h.manifest(source, smoke)['hashes'])
    dependencies = [Path(__file__), ROOT / 'spectral_utils/conditional_iu_fusion.py',
        ROOT / 'spectral_utils/conditional_iu_graph.py', ROOT / 'spectral_utils/direct_probability_temporal.py',
        ROOT / 'spectral_utils/shrinkage_iu.py', ROOT / 'scripts/test_conditional_iu_fusion.py',
        ROOT / 'scripts/test_conditional_iu_graph.py', ROOT / 'scripts/test_conditional_iu_driver.py',
        ROOT / 'scripts/complete_conditional_iu_fusion.py',
        ROOT / 'docs/experiments/CONDITIONAL_IU_FUSION_V1.md',
        ROOT / 'docs/experiments/CONDITIONAL_IU_FOLLOWUPS_V1.md']
    hashes.update({str(path): base.old.sha256_file(path) for path in dependencies})
    return dict(schema='conditional-iu-fusion-v1', family=FAMILY, smoke=bool(smoke),
        selected_ids=selected, source_root=str(source), methods=METHODS, reference_methods=old.REFS,
        feature_names=h.FEATURE_NAMES, bins=model.BINS, alpha=model.ALPHA, eta=model.ETA,
        baseline='answer-local native Shrinkage IU; fixed marginal rho and top-two eigenspace',
        training='within-answer regional centered covariances; equal groups then answers',
        exclusion='outer held fold; PRMB nested two-fold exclusion for q=.8 calibration',
        position='whole answer; official step boundaries only in Top10 readout',
        graph=dict(nodes='tokens',window=16,bandwidth_neighbor=8,bandwidth_floor=1e-8,
                   shuffle='SHA256 UID-seeded graph vertex permutation; no feature/label permutation',
                   penalty='eta sum a_ij ||theta_i-theta_j||_2; sum a=T/2',
                   solver='Chambolle-Pock',max_iter=3000,tol=1e-6),
        gate='frozen mean-entropy q=.3',readout='Top10 mean; earliest argmax tie',
        primary_pairs=[model.PRIMARY[FAMILY]],pairs=pairs_for(FAMILY),
        bootstrap=dict(draws=DRAW_COUNT,primary_ci=PRIMARY_CI,secondary_ci=.95,unit='canonical_source_group'),
        invocation_seconds=INVOCATION_SECONDS,
        software=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__),hashes=hashes)


def connect(freeze, smoke):
    con = sqlite3.connect(OUT / ('SMOKE.sqlite' if smoke else 'CHECKPOINT.sqlite'))
    con.execute('pragma journal_mode=WAL')
    for statement in (
        'create table if not exists manifest(payload text not null)',
        'create table if not exists stats(idx integer primary key,uid text not null,payload blob not null)',
        'create table if not exists models(key text primary key,payload blob not null,info text not null)',
        'create table if not exists scores(idx integer primary key,payload blob not null)',
        'create table if not exists health(idx integer primary key,info text not null)',
        'create table if not exists weightmaps(idx integer primary key,payload blob not null)',
    ):
        con.execute(statement)
    previous = con.execute('select payload from manifest').fetchone()
    if previous is not None:
        if json.loads(previous[0]) != json_ready(freeze):
            con.close()
            raise ValueError('frozen code/source manifest changed; use a new output directory')
    else:
        con.execute('insert into manifest values(?)', (base.dumps(json_ready(freeze)),))
    con.commit()
    return con


def training_metadata(records, fold):
    """Strip records to the only fields that unlabelled prior fitting accepts."""
    return {i: dict(uid=str(row['uid']),cell=str(row['cell']),group_id=str(row['group_id']),fold=int(fold[i]))
            for i, row in enumerate(records)}


def fit_prior(stats, metadata, cell, excluded):
    """Fit using saved statistics and allow-listed identity/fold metadata only.

    Neither benchmark records nor a label array is an argument. Every supplied
    metadata row is validated so a future caller cannot accidentally pass the
    original label-bearing records through this API.
    """
    allowed = {'uid', 'cell', 'group_id', 'fold'}
    if any(set(row) != allowed for row in metadata.values()):
        raise ValueError('prior metadata must contain exactly uid/cell/group_id/fold')
    excluded = tuple(sorted(set(map(int, excluded))))
    if not excluded or len(excluded) > 2:
        raise ValueError('prior requires one or two excluded folds')
    ids = sorted(i for i in stats if metadata[i]['cell'] == cell)
    train_ids = [i for i in ids if metadata[i]['fold'] not in excluded]
    if not train_ids:
        raise ValueError('no training answers remain after fold exclusion')
    train_groups = sorted({metadata[i]['group_id'] for i in train_ids})
    held_groups = sorted({metadata[i]['group_id'] for i in ids if metadata[i]['fold'] in excluded})
    if set(train_groups).intersection(held_groups):
        raise ValueError('a canonical source group crosses the training exclusion boundary')
    weights = hy.group_weights(metadata, train_ids)
    prior = {key: np.zeros((model.BINS, 12, 12)) for key in ('real', 'shuffle')}
    digest = hashlib.sha256()
    for i in train_ids:
        digest.update(str(i).encode() + b'\0')
        digest.update(np.float64(weights[i]).tobytes())
        for key in prior:
            covariance = np.asarray(stats[i][key], dtype=np.float64)
            if covariance.shape != (model.BINS, 12, 12) or not np.isfinite(covariance).all():
                raise ValueError('invalid saved regional covariance')
            prior[key] += weights[i] * covariance
            digest.update(np.ascontiguousarray(covariance).tobytes())
    if not all(np.isfinite(a).all() for a in prior.values()):
        raise FloatingPointError('nonfinite external covariance prior')
    info = dict(cell=cell,excluded_folds=list(excluded),training_ids=train_ids,
        training_groups=train_groups,excluded_groups=held_groups,training_answers=len(train_ids),
        training_group_count=len(train_groups),answer_weights={str(i):weights[i] for i in train_ids},
        training_group_sha256=hashlib.sha256('\n'.join(train_groups).encode()).hexdigest(),
        fit_input_sha256=digest.hexdigest(),status='OK',fit_api_accepts_no_labels=True)
    return prior, info


def excluded_sets(metadata, selected, cell):
    folds = sorted({metadata[i]['fold'] for i in selected if metadata[i]['cell'] == cell})
    result = [(f,) for f in folds]
    if not cell.startswith('pb_'):
        result += [(f, g) for f in folds for g in folds if f < g]
    return result


def load_cache(path):
    while hy.free_gib() < 4.0:
        progress('WAITING_FOR_RAM',cache=path.name,available_gib=hy.free_gib())
        time.sleep(30)
    cap_check()
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def extract(con, paths, selected, records, joined, reference):
    selected_set = set(selected)
    done = {i for i, in con.execute('select idx from stats')}
    for path in paths:
        with np.load(path, allow_pickle=False) as saved:
            ids = saved['ids'].copy()
        todo = [k for k, i in enumerate(ids) if int(i) in selected_set and int(i) not in done]
        if not todo:
            continue
        cache = load_cache(path)
        for k in todo:
            cap_check()
            i, x, spans, top = h.answer(cache, k, records, joined, reference)
            stat = model.regional_covariances(x, records[i]['uid'])
            if not all(np.isfinite(value).all() for value in stat.values()):
                raise FloatingPointError('invalid regional covariance for ' + records[i]['uid'])
            with con:
                con.execute('insert into stats values(?,?,?)', (i,records[i]['uid'],base.packed(**stat)))
            done.add(i)
            if len(done) % 100 == 0:
                progress('EXTRACTING_REGIONAL_COVARIANCES',completed=len(done),expected=len(selected))
        del cache
        gc.collect()
    if done != selected_set:
        raise ValueError('statistics coverage differs from frozen selected roster')
    progress('STATISTICS_COMPLETE',completed=len(done))


def train(con, metadata, selected):
    saved = {key for key, in con.execute('select key from models')}
    for cell in sorted({metadata[i]['cell'] for i in selected}):
        ids = [i for i in selected if metadata[i]['cell'] == cell]
        pending = [ex for ex in excluded_sets(metadata,selected,cell) if h.model_key(cell,ex) not in saved]
        if not pending:
            continue
        stats = {i: hy.load_blob(con.execute('select payload from stats where idx=?',(i,)).fetchone()[0]) for i in ids}
        for excluded in pending:
            cap_check()
            key = h.model_key(cell, excluded)
            prior, info = fit_prior(stats, metadata, cell, excluded)
            with con:
                con.execute('insert into models values(?,?,?)',(key,base.packed(**prior),base.dumps(info)))
            progress('FITTING_COVARIANCE_PRIORS',model=key,training_answers=len(info['training_ids']))
        del stats
        gc.collect()


def review_exclusions(con, metadata, selected):
    """Re-derive every exclusion, weight, and covariance from saved statistics."""
    expected_keys = {h.model_key(cell, ex) for cell in {metadata[i]['cell'] for i in selected}
                     for ex in excluded_sets(metadata,selected,cell)}
    seen = set()
    checked = []
    for cell in sorted({metadata[i]['cell'] for i in selected}):
        stats = {i:hy.load_blob(con.execute('select payload from stats where idx=?',(i,)).fetchone()[0])
                 for i in selected if metadata[i]['cell'] == cell}
        for excluded in excluded_sets(metadata,selected,cell):
            cap_check()
            key = h.model_key(cell,excluded)
            row = con.execute('select payload,info from models where key=?',(key,)).fetchone()
            if row is None:
                raise ValueError('missing model ' + key)
            saved, info = hy.load_blob(row[0]), json.loads(row[1])
            replay, replay_info = fit_prior(stats, metadata, cell, excluded)
            if info != replay_info:
                raise ValueError('model provenance differs from independent exclusion replay: ' + key)
            for name in replay:
                np.testing.assert_array_equal(saved[name], replay[name])
            seen.add(key)
            checked.append(dict(model=key,training_answers=len(info['training_ids']),
                                training_groups=len(info['training_groups']),excluded_folds=list(excluded)))
    actual = {key for key, in con.execute('select key from models')}
    if seen != expected_keys or actual != expected_keys:
        raise ValueError('unexpected or missing excluded-fold models')
    return dict(status='PASS',models=checked,all_priors_recomputed=True,
                all_exclusions_rederived=True,fit_api_accepts_no_labels=True)


def label_firewall_review(con, records, joined, metadata, selected):
    """Alter labels/targets, rebuild stripped metadata, and refit held examples."""
    fold = {i:row['fold'] for i,row in metadata.items()}
    changed_records = [{**row,'label':'FORBIDDEN','target':'FORBIDDEN','labels':['FORBIDDEN']} for row in records]
    rebuilt = training_metadata(changed_records,fold)
    if rebuilt != metadata:
        raise ValueError('label-bearing record fields reached fit metadata')
    checks = []
    for key, blob, text in con.execute('select key,payload,info from models order by key'):
        info = json.loads(text)
        category = ('pb' if info['cell'].startswith('pb_') else 'prm',len(info['excluded_folds']))
        if any(tuple(row['category']) == category for row in checks):
            continue
        ids = [i for i in selected if metadata[i]['cell'] == info['cell']]
        stats = {i:hy.load_blob(con.execute('select payload from stats where idx=?',(i,)).fetchone()[0]) for i in ids}
        held = [i for i in ids if metadata[i]['fold'] in info['excluded_folds']]
        changed_labels = joined['labels'].copy()
        changed_target = joined['target'].copy()
        for i in held:
            a,b = joined['offsets'][i:i+2]
            known = changed_labels[a:b] >= 0
            changed_labels[a:b][known] = 1-changed_labels[a:b][known]
            changed_target[i] += 17
            # Stronger than a label flip: held-feature covariance cannot enter
            # an external prior either. Change those statistics drastically.
            stats[i] = {name:value + 12345.0 for name,value in stats[i].items()}
        if not held or np.array_equal(changed_target,joined['target']):
            raise ValueError('label-firewall fixture failed to alter held target')
        replay, replay_info = fit_prior(stats,rebuilt,info['cell'],info['excluded_folds'])
        saved = hy.load_blob(blob)
        for name in saved:
            np.testing.assert_array_equal(saved[name],replay[name])
        if replay_info != info:
            raise ValueError('held labels/features changed training provenance')
        checks.append(dict(category=list(category),model=key,held_answers=len(held),
            held_labels_changed=not np.array_equal(changed_labels,joined['labels']),
            held_targets_changed=True,held_covariances_changed=True,prior_bitwise_unchanged=True))
    return dict(status='PASS',fit_metadata_fields=['uid','cell','group_id','fold'],
                fit_api_accepts_no_labels=True,refitted=checks)


def score(con, paths, selected, records, joined, reference, metadata):
    selected_set = set(selected)
    done = {i for i, in con.execute('select idx from scores')}
    priors = {key:hy.load_blob(blob) for key,blob in con.execute('select key,payload from models')}
    external = tuple(name for name in METHODS if name != 'baseline')
    for path in paths:
        with np.load(path,allow_pickle=False) as saved:
            ids = saved['ids'].copy()
        todo = [k for k,i in enumerate(ids) if int(i) in selected_set and int(i) not in done]
        if not todo:
            continue
        cache = load_cache(path)
        for k in todo:
            cap_check()
            started = time.perf_counter()
            i,x,spans,top = h.answer(cache,k,records,joined,reference)
            row = metadata[i]
            cell,f = row['cell'],row['fold']
            contexts = [(f,)]
            if not cell.startswith('pb_'):
                contexts += [tuple(sorted((f,g))) for g in sorted({metadata[j]['fold'] for j in selected
                             if metadata[j]['cell']==cell and metadata[j]['fold']!=f})]
            payload = dict(top10=top)
            health = dict(uid=row['uid'],cell=cell,outer_fold=f,tokens=len(x),steps=len(spans),contexts={},fits={})
            maps = {}
            try:
                prepared = model.prepare_answer(x,row['uid'],METHODS)
                health['baseline_status'] = 'OK'
            except Exception as exc:
                prepared = None
                health.update(baseline_status='FAILED',baseline_failure=f'{type(exc).__name__}: {exc}')
            for excluded in contexts:
                suffix = '' if len(excluded)==1 else '__inner_for_'+str(next(g for g in excluded if g!=f))
                names = METHODS if not suffix else external
                key = h.model_key(cell,excluded)
                if key not in priors:
                    raise ValueError('missing required external prior: '+key)
                health['contexts'][suffix or 'outer'] = dict(model=key,excluded_folds=list(excluded),methods=list(names))
                if prepared is None:
                    values = {name:np.full(len(spans),np.nan) for name in names}
                    detail = {name:dict(status='FAILED',reason='native baseline failure: '+health['baseline_failure']) for name in names}
                    weight = {}
                else:
                    values,detail,weight = model.score_answer(prepared,priors[key],spans,names)
                for name in names:
                    vector = np.asarray(values[name],dtype=float)
                    if vector.shape != (len(spans),):
                        raise ValueError('score vector changed the official step roster')
                    payload[name+suffix] = vector
                    health['fits'][name+suffix] = detail[name]
                    if not suffix:
                        maps[name] = weight.get(name,np.full((model.BINS,12),np.nan))
            health.update(seconds=time.perf_counter()-started,weightmaps='outer predictions only; sixteen bins by twelve features',
                          exclusion_invariant_methods=['top10','baseline'])
            # Checkpoint all of an answer's outer/inner predictions together.
            with con:
                con.execute('insert into scores values(?,?)',(i,base.packed(**payload)))
                con.execute('insert into health values(?,?)',(i,base.dumps(json_ready(health))))
                con.execute('insert into weightmaps values(?,?)',(i,base.packed(**maps)))
            done.add(i)
            del prepared
            if len(done)%25==0 or len(selected)==27:
                progress('SCORING_TOP10',completed=len(done),expected=len(selected),last_answer_seconds=health['seconds'])
        del cache
        gc.collect()
    if done != selected_set:
        raise ValueError('score coverage differs from frozen answer roster')
    progress('SCORES_COMPLETE',completed=len(done))


def score_review(con, metadata, selected, records, joined, reference):
    """Audit every outer/inner vector and its exact held-fold dependency."""
    roster = set(selected)
    for table in ('stats','scores','health','weightmaps'):
        if {i for i, in con.execute('select idx from '+table)} != roster:
            raise ValueError('incomplete atomic answer coverage in '+table)
    counts = {name:dict(outer_answers=0,outer_failed=0,inner_predictions=0,inner_failed=0,
                       nonconverged_outer=0,nonconverged_inner=0) for name in METHODS}
    baseline_failures = []
    for i,blob in con.execute('select idx,payload from scores order by idx'):
        values = hy.load_blob(blob)
        info = json.loads(con.execute('select info from health where idx=?',(i,)).fetchone()[0])
        maps = hy.load_blob(con.execute('select payload from weightmaps where idx=?',(i,)).fetchone()[0])
        a,b = joined['offsets'][i:i+2]
        np.testing.assert_allclose(values['top10'],reference['rbm12__logit_old'][a:b],atol=1e-12,rtol=0)
        row = metadata[i]
        others = sorted({metadata[j]['fold'] for j in selected if metadata[j]['cell']==row['cell']
                         and metadata[j]['fold']!=row['fold']}) if not row['cell'].startswith('pb_') else []
        expected = {'top10',*METHODS}
        expected.update(name+'__inner_for_'+str(g) for g in others for name in METHODS if name!='baseline')
        if set(values) != expected or set(maps) != set(METHODS):
            raise ValueError('outer/inner score or weight-map roster mismatch')
        for suffix,context in info['contexts'].items():
            ex = [row['fold']] if suffix=='outer' else sorted((row['fold'],int(suffix.split('_')[-1])))
            if context['excluded_folds'] != ex or context['model'] != h.model_key(row['cell'],ex):
                raise ValueError('score context violates nested held-fold exclusions')
            saved = json.loads(con.execute('select info from models where key=?',(context['model'],)).fetchone()[0])
            if row['group_id'] in saved['training_groups']:
                raise ValueError('current answer group entered its external prior')
        for name,vector in values.items():
            if vector.shape != (records[i]['steps'],):
                raise ValueError('prediction failed to preserve official steps')
            if name=='top10':
                continue
            method = name.split('__inner_for_')[0]
            inner = '__inner_for_' in name
            valid = bool(np.isfinite(vector).all())
            count = counts[method]
            count['inner_predictions' if inner else 'outer_answers'] += 1
            count['inner_failed' if inner else 'outer_failed'] += int(not valid)
            count['nonconverged_inner' if inner else 'nonconverged_outer'] += int(info['fits'][name].get('converged') is False)
            if not valid and not np.isnan(vector).all():
                raise ValueError('partial nonfinite answer scores must be an explicit whole-answer failure')
            if not inner and maps[method].shape != (model.BINS,12):
                raise ValueError('invalid compact weight-map shape')
        if info['baseline_status']=='FAILED':
            baseline_failures.append(dict(idx=i,uid=row['uid'],reason=info['baseline_failure']))
            if any(np.isfinite(values[name]).any() for name in values if name!='top10'):
                raise ValueError('a dependent method substituted scores after native baseline failure')
    return dict(status='PASS',answers=len(selected),methods=counts,native_baseline_failures=baseline_failures,
                all_outer_inner_vectors_audited=True,original_top10_reproduced=True,
                baseline_and_top10_exclusion_invariant=True,all_failures_preserved=True)


def write_health(con, coverage):
    # SQLite retains every per-answer/per-context detail. JSON summarizes that
    # audit without assembling all health or token weights in memory at once.
    seconds=[]
    for text, in con.execute('select info from health'):
        seconds.append(json.loads(text)['seconds'])
    emit('FIT_HEALTH.json',dict(coverage=coverage,details='health table in checkpoint SQLite',
        compact_weight_maps='weightmaps table; every outer answer, 16 x 12 per method',
        scoring_seconds_total=float(np.sum(seconds)),scoring_seconds_median=float(np.median(seconds)),
        scoring_seconds_max=float(np.max(seconds))))


@contextmanager
def endpoint_configuration():
    """Configure imported uncertainty helpers in this process only."""
    settings=dict(ALL_SCORED=METHODS,METHODS=tuple(n for n in METHODS if n!='baseline'),
                  PRIMARY=[model.PRIMARY[FAMILY]],PRIMARY_CI=PRIMARY_CI,PAIRS=pairs_for(FAMILY))
    previous={name:getattr(h,name) for name in settings}
    hy_methods=hy.METHODS
    try:
        for name,value in settings.items():
            setattr(h,name,value)
        hy.METHODS=('top10',)+METHODS
        yield
    finally:
        for name,value in previous.items():
            setattr(h,name,value)
        hy.METHODS=hy_methods


def endpoint_uncertainty(records, joined, scores, metrics, thresholds, fold):
    # Keep h's common-cohort definition and exact bootstrap routine. Report
    # unavailable intervals explicitly if any fold lacks both label classes.
    offsets=joined['offsets']
    common=np.array([not row['cell'].startswith('pb_') for row in records])
    for name in ('top10',)+METHODS:
        common &= np.array([np.isfinite(scores[name][offsets[i]:offsets[i+1]]).all() for i in range(len(records))])
    details=dict(common_prm_answers=int(common.sum()),expected_prm_answers=sum(not row['cell'].startswith('pb_') for row in records))
    missing=[]
    for f in sorted(set(fold.values())):
        ids=[i for i in range(len(records)) if common[i] and fold[i]==f]
        y=np.concatenate([joined['labels'][offsets[i]:offsets[i+1]] for i in ids]) if ids else np.array([])
        if len(np.unique(y[y>=0]))<2:
            missing.append(int(f))
    if missing:
        return dict(status='UNAVAILABLE',reason='common PRMB cohort lacks both classes in held folds',
                    unavailable_folds=missing,**details)
    return dict(status='AVAILABLE',**details,contrasts=h.endpoint_uncertainty(records,joined,scores,metrics,thresholds,fold))


def build_calibration(predictions, scores, records, joined, fold, selected, methods):
    """Frozen q=.8 thresholds from outer-blind predictions, with coverage.

    For an answer in fold h and threshold for held fold f, external methods
    must use its saved ``__inner_for_f`` vector from the (f,h)-excluded fit.
    Answer-local methods and frozen references use their invariant scores.
    Missing predictions remain failures; an entirely unavailable threshold
    is None and is never used to classify all steps as clean.
    """
    offsets=np.asarray(joined['offsets'])
    np.testing.assert_array_equal(offsets,np.concatenate([[0],np.cumsum([row['steps'] for row in records])]))
    if any(np.asarray(value).shape!=(int(offsets[-1]),) for value in scores.values()):
        raise ValueError('calibration scores do not match the official step roster')
    external=set(methods)-{'baseline'}
    thresholds={name:{} for name in scores}
    calibration=[]
    unavailable=[]
    for f in sorted(set(fold.values())):
        ids=[i for i in selected if not records[i]['cell'].startswith('pb_') and fold[i]!=f]
        excluded_groups=sorted({records[i]['group_id'] for i in selected if not records[i]['cell'].startswith('pb_') and fold[i]==f})
        for name in scores:
            vectors=[];used=[]
            for i in ids:
                a,b=offsets[i:i+2]
                values=predictions[i][name+'__inner_for_'+str(f)] if name in external else scores[name][a:b]
                if np.isfinite(values).all():
                    vectors.append(values);used.append(i)
            groups=sorted({records[i]['group_id'] for i in used})
            if set(groups).intersection(excluded_groups):
                raise ValueError('PRMScore calibration source groups overlap held fold')
            if vectors:
                thresholds[name][str(f)]=float(np.quantile(np.concatenate(vectors),.8))
            else:
                thresholds[name][str(f)]=None
                unavailable.append(dict(method=name,outer_fold=int(f)))
            calibration.append(dict(method=name,outer_fold=int(f),answers=len(used),expected=len(ids),
                training_ids=used,training_groups=groups,excluded_groups=excluded_groups,
                prediction_scope='nested excluded folds' if name in external else 'answer-local fixed scores'))
    return thresholds,calibration,unavailable


def evaluate(con, records, joined, reference, fold, selected, source, exclusion_review, coverage):
    """Same endpoints/calibration/bootstrap as h.evaluate, with this roster."""
    if len(selected)!=EXPECTED_ANSWERS or set(selected)!=set(range(len(records))):
        raise ValueError('full evaluation requires the complete 13,769-answer roster')
    np.testing.assert_array_equal(joined['offsets'],np.concatenate([[0],np.cumsum([r['steps'] for r in records])]))
    total=int(joined['offsets'][-1])
    scores={name:np.full(total,np.nan) for name in ('top10',)+METHODS}
    predictions={i:hy.load_blob(blob) for i,blob in con.execute('select idx,payload from scores')}
    for i,values in predictions.items():
        a,b=joined['offsets'][i:i+2]
        for name in scores:
            scores[name][a:b]=values[name]
    if set(reference)!=set(old.REFS) or len(reference)!=13:
        raise ValueError('the thirteen frozen reference rows are required')
    scores.update({'reference__'+name:values for name,values in reference.items()})
    np.testing.assert_allclose(scores['top10'],reference['rbm12__logit_old'],atol=1e-12,rtol=0)
    np.savez_compressed(OUT/'SCORES.npz',**scores)
    emit('SCORE_REVIEW.json',coverage)
    thresholds,calibration,unavailable=build_calibration(predictions,scores,records,joined,fold,selected,METHODS)
    emit('CALIBRATION.json',dict(thresholds=thresholds,coverage=calibration,unavailable=unavailable))
    if unavailable:
        raise EndpointUnavailable('no finite inner calibration for '+str(unavailable))
    with endpoint_configuration():
        metrics,per=base.evaluate_arrays(records,joined,scores,calibration_thresholds=thresholds,fold_auc=True)
        prm=np.array([not row['cell'].startswith('pb_') for row in records])
        step_prm=np.repeat(prm,np.diff(joined['offsets']))
        for name,flat in scores.items():
            valid=step_prm & np.repeat(per[name]['valid'],np.diff(joined['offsets'])) & (joined['labels']>=0) & np.isfinite(flat)
            metrics[name]['prm_pooled']=base.old.auc(joined['labels'][valid]==1,flat[valid])
        previous=json.loads((old.parent(source)/'METRICS.json').read_text(encoding='utf8'))['metrics']
        for name in reference:
            for endpoint in ('pb_all8','prm_within','prm_pooled','prmscore_q08'):
                np.testing.assert_allclose(metrics['reference__'+name][endpoint],previous[name][endpoint],atol=1e-12,rtol=0)
        emit('METRICS.json',metrics)
        progress('BOOTSTRAP',draws=DRAW_COUNT)
        contrasts=base.paired_bootstrap(records,joined,per,draws=DRAW_COUNT,pairs=pairs_for(FAMILY),
                                       primary_pairs={model.PRIMARY[FAMILY]},primary_ci=PRIMARY_CI)
        independent=hy.independent_review(records,joined,scores,metrics,thresholds,per)
        endpoint_review=h.independent_endpoint_review(records,joined,scores,metrics,fold)
        extended=endpoint_uncertainty(records,joined,scores,metrics,thresholds,fold)
        transitions=hy.error_and_weight_tables(records,joined,{},[],per)
    emit('CONTRASTS.json',contrasts)
    emit('ENDPOINT_CONTRASTS.json',extended)
    emit('ERROR_TRANSITIONS.json',transitions)
    old.csv_write(OUT/'COMPARISON.csv',[dict(method=name,label=model.LABELS.get(name,name),
        **{k:v for k,v in values.items() if not isinstance(v,dict)}) for name,values in metrics.items()])
    old.csv_write(OUT/'PER_CELL.csv',[dict(method=name,cell=cell,**values)
        for name,summary in metrics.items() for cell,values in summary['pb_cells'].items()])
    emit('RESULT_REVIEW.json',dict(status='PASS',family=FAMILY,answers=len(selected),
        independent_metrics=independent,independent_all_endpoints=endpoint_review,
        frozen_references_reproduced=True,exclusions=exclusion_review,coverage=coverage,
        performance_thresholds_applied=False,endpoint_uncertainty_status=extended['status']))
    def display(value, percent=False):
        return 'unavailable' if value is None else f'{value*(100 if percent else 1):.5f}'
    lines=['# Conditional IU follow-up: '+FAMILY,'',
        'Full cached development benchmark. External covariance fitting excludes held source-group folds.',
        'Native IU rho and its two-dimensional eigenspace stay fixed within each answer. No model-performance threshold is applied.',
        '',f'Primary contrast: {model.PRIMARY[FAMILY][0]} minus {model.PRIMARY[FAMILY][1]}; {100*PRIMARY_CI:.5f}% interval; 10,000 grouped draws.',
        '', '| Method | PB % | PRMB within AUC | PRMScore | Coverage |','|---|---:|---:|---:|---:|']
    for name in ('top10',)+METHODS:
        values=metrics[name]
        lines.append(f'| {model.LABELS.get(name,name)} | {display(values["pb_all8"],True)} | {display(values["prm_within"])} | {display(values["prmscore_q08"])} | {values["valid_answers"]}/{EXPECTED_ANSWERS} |')
    lines+=['','All thirteen frozen reference rows are in COMPARISON.csv. Failures remain in the coverage denominator.',
            'Conditional endpoints and common-cohort uncertainty coverage are recorded explicitly in METRICS.json and ENDPOINT_CONTRASTS.json.',
            'Per-answer outer weight maps and every outer/inner fit diagnostic are retained in the checkpoint SQLite.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    progress('COMPLETE_REVIEWED',completed=EXPECTED_ANSWERS)


def require_smoke(freeze):
    smoke_dir=OUT/'smoke'
    review=json.loads((smoke_dir/'SMOKE_REVIEW.json').read_text(encoding='utf8'))
    previous=json.loads((smoke_dir/'MANIFEST.json').read_text(encoding='utf8'))
    if review.get('status')!='PASS' or review.get('answers')!=27:
        raise ValueError('full phase requires this family\'s completed 27-answer smoke review')
    for key in freeze:
        if key not in {'smoke','selected_ids'} and json_ready(freeze[key])!=previous[key]:
            raise ValueError('smoke and full freezes differ at '+key)


def main():
    global START,OUT,FAMILY,METHODS,STOP_REQUESTED
    START=time.monotonic()
    STOP_REQUESTED=False
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--family',choices=tuple(model.FAMILIES),required=True)
    parser.add_argument('--phase',choices=('smoke','full'),default='smoke')
    args=parser.parse_args()
    source=args.source_root.resolve()
    FAMILY=args.family;METHODS=model.FAMILIES[FAMILY]
    smoke=args.phase=='smoke'
    OUT=ROOT/'results/conditional_iu_fusion_v1'/FAMILY
    if smoke:
        OUT=OUT/'smoke'
    OUT.mkdir(parents=True,exist_ok=True)
    lock=OUT/'RUN.lock'
    descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    try:
        os.write(descriptor,str(os.getpid()).encode())
    finally:
        os.close(descriptor)
    con=None
    previous_handler=signal.getsignal(signal.SIGTERM)
    def stop_at_checkpoint(signum, frame):
        # Latch instead of raising inside a numerical routine's error handler:
        # finish the current answer transaction, then stop at cap_check().
        global STOP_REQUESTED
        STOP_REQUESTED=True
    signal.signal(signal.SIGTERM,stop_at_checkpoint)
    try:
        with threadpool_limits(limits=1):
            fixtures=importlib.import_module('scripts.test_conditional_iu_fusion')
            emit('UNIT_REVIEW.json',fixtures.run())
            records,joined,reference=old.load_contract(source)
            folds=json.loads(base.old.FOLDS.read_text(encoding='utf8'))['outer']
            fold={i:int(folds[row['group_id']]) for i,row in enumerate(records)}
            metadata=training_metadata(records,fold)
            paths=sorted(old.caches(source).glob('cache_*.npz'))
            if len(paths)!=9:
                raise ValueError('exactly nine frozen token caches are required')
            selected=sorted(i for path in paths for i in h.selection(path,records,fold,smoke))
            expected=27 if smoke else EXPECTED_ANSWERS
            if len(selected)!=expected or len(set(selected))!=expected:
                raise ValueError('smoke/full selected answer count differs from contract')
            freeze=manifest(source,smoke,selected)
            if not smoke:
                require_smoke(freeze)
            con=connect(freeze,smoke)
            emit('MANIFEST.json',freeze)
            progress('STARTED',phase=args.phase,expected=expected)
            extract(con,paths,selected,records,joined,reference)
            train(con,metadata,selected)
            exclusions=review_exclusions(con,metadata,selected)
            emit('EXCLUSION_REVIEW.json',exclusions)
            emit('LABEL_FIREWALL_REVIEW.json',label_firewall_review(con,records,joined,metadata,selected))
            score(con,paths,selected,records,joined,reference,metadata)
            coverage=score_review(con,metadata,selected,records,joined,reference)
            emit('SCORE_REVIEW.json',coverage)
            write_health(con,coverage)
            if smoke:
                failure_count=sum(row['outer_failed']+row['inner_failed'] for row in coverage['methods'].values())
                review=dict(status='FAIL' if failure_count else 'PASS',family=FAMILY,answers=len(selected),
                    coverage=coverage,exclusions=exclusions,finite_outer_and_inner_required=True,
                    purpose='numerical feasibility and exclusion validation only; no benchmark ranking',
                    performance_thresholds_applied=False)
                emit('SMOKE_REVIEW.json',review)
                if failure_count:
                    raise ValueError(f'smoke has {failure_count} nonfinite outer/inner prediction vectors')
                progress('SMOKE_COMPLETE',completed=len(selected))
            else:
                evaluate(con,records,joined,reference,fold,selected,source,exclusions,coverage)
    except EndpointUnavailable as exc:
        emit('RESULT_REVIEW.json',dict(status='UNAVAILABLE_ENDPOINTS',reason=str(exc),all_scores_checkpointed=True))
        emit('RUN_STATE.json',dict(status='COMPLETE_WITH_UNAVAILABLE_ENDPOINTS',family=FAMILY,reason=str(exc)))
    except hy.InvocationCap:
        emit('RUN_STATE.json',dict(status='CHECKPOINTED_INVOCATION_CAP',family=FAMILY,pid=os.getpid(),
             elapsed_seconds=time.monotonic()-START))
    except BaseException:
        emit('RUN_STATE.json',dict(status='FAILED',family=FAMILY,traceback=traceback.format_exc()))
        raise
    finally:
        if con is not None:
            con.close()
        signal.signal(signal.SIGTERM,previous_handler)
        # Only remove the lock created by this process.
        if lock.exists() and lock.read_text(encoding='utf8')==str(os.getpid()):
            lock.unlink()


if __name__=='__main__':
    main()
