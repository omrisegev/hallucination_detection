"""Finish full anchor evaluation from immutable saved scores, without fits."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import csv
import hashlib
import html
import io
import json
from pathlib import Path
import time
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results/localization_full_benchmark_v3'
OUT = RUN / 'evaluation'
PROTOCOL = ROOT / 'docs/experiments/FULL_ANCHOR_EVALUATION_V3.md'
RELEASE = ROOT / 'results/localization_prm_label_audit_v1/RELEASE_V3.json'
PILOT = ROOT / 'results/fusion_entropy_sampling_v1/EVALUATION.json'
DRAWS, SEED = 1000, 2026090707


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def state(phase, **extra):
    save(OUT / 'RUN_STATE.json', dict(phase=phase, pid=os.getpid(), updated_unix=time.time(), **extra))


def rank_auc(y, s):
    y = np.asarray(y, dtype=bool)
    p, n = y.sum(), (~y).sum()
    if not p or not n:
        return None
    return float((rankdata(s)[y].sum() - p*(p+1)/2) / (p*n))


def auc_plan(y, s, groups):
    order = np.argsort(s, kind='stable')
    scores = s[order]
    starts = np.r_[0, np.flatnonzero(scores[1:] != scores[:-1]) + 1]
    return np.asarray(y, bool)[order], groups[order], starts


def weighted_auc(plan, weights):
    y, groups, starts = plan
    row_weights = weights[groups]
    pos = np.add.reduceat(row_weights*y, starts)
    neg = np.add.reduceat(row_weights*(~y), starts)
    p, n = pos.sum(), neg.sum()
    return float(np.dot(pos, np.cumsum(neg) - .5*neg) / (p*n)) if p and n else np.nan


def bootstrap_preflight():
    y = np.array([0, 1, 0, 1, 1, 0])
    s = np.array([.2, .2, .8, .6, .9, .1])
    groups = np.array([0, 0, 1, 2, 1, 2])
    plan = auc_plan(y, s, groups)
    for w in [np.ones(3, int), np.array([0, 1, 2]), np.array([3, 0, 0]), np.array([2, 1, 0])]:
        ix = np.repeat(np.arange(6), w[groups])
        assert abs(weighted_auc(plan, w) - roc_auc_score(y[ix], s[ix])) < 1e-14


def prepare_inputs(manifest, frozen):
    state('VERIFYING_AND_JOINING', completed=0, total=len(manifest['selected']))
    release = load(RELEASE)
    labels = {}
    for cell, info in release['cells'].items():
        label_path = Path(info['label_path'])
        assert sha(label_path) == manifest['hashes'][str(label_path)]
        with np.load(label_path, allow_pickle=False) as a:
            ids = list(map(str, a['row_ids']))
            assert len(set(ids)) == len(ids)
            declared = {x['row_id']: x for x in info['rows']}
            for i, row_id in enumerate(ids):
                if cell.startswith('prm'):
                    lo, hi = a['step_flag_offsets'][i:i+2]
                    target = a['step_error_flags'][lo:hi].copy()
                else:
                    target = int(a['first_error'][i])
                labels[cell, row_id] = (target, declared[row_id])
    records, arms = manifest['selected'], manifest['arms']
    n, k = len(records), len(arms)
    offsets = np.r_[0, np.cumsum([r['steps'] for r in records])]
    a = dict(scores=np.full((offsets[-1], k), np.nan), labels=np.full(offsets[-1], -2, dtype=np.int8),
             offsets=offsets, target=np.full(n, -2, dtype=np.int32),
             valid=np.zeros((n,k), bool), decision=np.zeros((n,k), bool),
             predictions=np.full((n,k), -2, dtype=np.int32), peaks=np.full((n,k), -2, dtype=np.int32))
    sources = {arm: {} for arm in arms}
    for i, rec in enumerate(records):
        uid = rec['uid']; meta_path = RUN/'scores'/(uid+'.json'); arrays_path = meta_path.with_suffix('.npz')
        assert sha(meta_path) == frozen['files'][str(meta_path)]
        d = load(meta_path)
        assert sha(arrays_path) == frozen['files'][str(arrays_path)] == d['array_sha256']
        for key in ('uid', 'cell', 'row_id', 'group_id', 'tokens', 'steps'):
            assert d[key] == rec[key], (uid, key)
        target, declared = labels[rec['cell'], rec['row_id']]
        for key in ('row_id', 'group_id', 'tokens', 'steps'):
            assert rec[key] == declared[key]
        lo, hi = offsets[i:i+2]
        if rec['cell'].startswith('prm'):
            assert len(target) == hi-lo and np.isin(target, [0, 1]).all()
            a['labels'][lo:hi] = target
        else:
            assert -1 <= target < rec['steps']; a['target'][i] = target
        assert set(d['methods']) == set(arms)
        with np.load(arrays_path, allow_pickle=False) as z:
            for j, arm in enumerate(arms):
                method = d['methods'][arm]
                a['valid'][i,j] = method['valid']; a['decision'][i,j] = method['decision_valid']
                if method['valid']:
                    risk = z[arm+'__risk']
                    assert risk.shape == (hi-lo,) and np.isfinite(risk).all()
                    a['scores'][lo:hi,j] = risk
                    a['peaks'][i,j] = method['peak']
                    assert method['peak'] == int(np.argmax(risk))
                if method['decision_valid']:
                    assert method['valid']
                    a['predictions'][i,j] = method['prediction']
                    assert method['prediction'] in (-1, method['peak'])
                source = method.get('source_arm') or 'INVALID'
                sources[arm][source] = sources[arm].get(source, 0) + 1
        if (i+1) % 500 == 0:
            state('VERIFYING_AND_JOINING', completed=i+1, total=n); print('Joined',i+1,'/',n,flush=True)
    a['within'] = np.full((n,k), np.nan)
    for i, rec in enumerate(records):
        if not rec['cell'].startswith('prm'): continue
        lo, hi = offsets[i:i+2]
        for j in range(k):
            if a['valid'][i,j]:
                value = rank_auc(a['labels'][lo:hi], a['scores'][lo:hi,j])
                if value is not None: a['within'][i,j] = value
    tmp = OUT / 'JOINED.npz.tmp'
    with tmp.open('wb') as f: np.savez_compressed(f, **a)
    tmp.replace(OUT / 'JOINED.npz')
    save(OUT / 'JOINED.json', dict(records=records, arms=arms, sources=sources,
         scores_freeze_sha256=sha(RUN/'SCORES_FROZEN.json'), arrays_sha256=sha(OUT/'JOINED.npz')))
    return records, arms, sources, a


def metric(records, a, j, selected=None):
    mask = np.ones(len(records), bool) if selected is None else selected
    prm = mask & np.array([r['cell'].startswith('prm') for r in records]) & a['valid'][:,j]
    owner = np.repeat(np.arange(len(records)), np.diff(a['offsets']))
    step_mask = prm[owner]
    pooled = rank_auc(a['labels'][step_mask], a['scores'][step_mask,j]) if step_mask.any() else None
    within = a['within'][prm,j]; within = within[np.isfinite(within)]
    cells = {}
    for cell in sorted({r['cell'] for r in records if r['cell'].startswith('pb_')}):
        rows = mask & np.array([r['cell']==cell for r in records])
        clean, error = rows & (a['target']==-1), rows & (a['target']>=0)
        valid = a['decision'][:,j]; success = valid & (a['predictions'][:,j]==a['target'])
        ca = float(success[clean].mean()) if clean.any() else None
        ea = float(success[error].mean()) if error.any() else None
        f1 = (2*ca*ea/(ca+ea) if ca+ea else 0.) if ca is not None and ea is not None else None
        cells[cell] = dict(answers=int(rows.sum()), clean=int(clean.sum()), erroneous=int(error.sum()),
             clean_accuracy=ca, error_exact_accuracy=ea, f1=f1, valid_decisions=int((rows & valid).sum()))
    macros = {}
    for panel in ('q4', 'q8', 'all'):
        values = [d['f1'] for cell,d in cells.items() if panel=='all' or cell.endswith(panel)]
        macros[panel] = float(np.mean(values)) if values and None not in values else None
    return dict(prm=dict(answers=int(prm.sum()), auroc=pooled,
                within_answer_auc=float(np.mean(within)) if len(within) else None, mixed_answers=len(within)),
                pb=dict(cells=cells, macros=macros))


def intervals(records, arms, a, metrics):
    state('BOOTSTRAP', completed=0)
    group_ids = sorted({r['group_id'] for r in records}); lookup = {g:i for i,g in enumerate(group_ids)}
    gi = np.array([lookup[r['group_id']] for r in records]); ng = len(group_ids)
    weights = np.random.default_rng(SEED).multinomial(ng, np.full(ng, 1/ng), size=DRAWS)
    owner = np.repeat(np.arange(len(records)), np.diff(a['offsets']))
    prm_rows = np.array([r['cell'].startswith('prm') for r in records]); cache = {}
    def prm_distribution(j, common):
        key = (j, np.packbits(common).tobytes())
        if key in cache: return cache[key]
        selected = common & prm_rows; steps = selected[owner]
        plan = auc_plan(a['labels'][steps], a['scores'][steps,j], gi[owner[steps]])
        mixed = selected & np.isfinite(a['within'][:,j])
        counts = np.bincount(gi[mixed], minlength=ng)
        totals = np.bincount(gi[mixed], weights=a['within'][mixed,j], minlength=ng)
        draws = np.array([weighted_auc(plan, w) for w in weights])
        means = (weights @ totals) / (weights @ counts)
        point = weighted_auc(plan, np.ones(ng, int))
        cache[key] = (draws, means, dict(answers=int(selected.sum()), mixed_answers=int(mixed.sum()),
              auroc=point, within_answer_auc=float(a['within'][mixed,j].mean())))
        return cache[key]
    def ci(x):
        finite = x[np.isfinite(x)]
        return dict(ci95=np.quantile(finite, [.025,.975]).tolist(), valid_draws=len(finite))
    pb_distributions = {}
    for j, arm in enumerate(arms):
        cell_draws = {}
        for cell in metrics[arm]['pb']['cells']:
            rows = np.array([r['cell']==cell for r in records]); c = rows & (a['target']==-1); e = rows & (a['target']>=0)
            success = a['decision'][:,j] & (a['predictions'][:,j]==a['target'])
            def count(mask): return weights @ np.bincount(gi[mask], minlength=ng)
            ca, ea = count(c & success)/count(c), count(e & success)/count(e)
            cell_draws[cell] = np.divide(2*ca*ea, ca+ea, out=np.zeros_like(ca), where=ca+ea!=0)
        pb_distributions[arm] = {panel:np.mean([v for c,v in cell_draws.items() if panel=='all' or c.endswith(panel)], axis=0)
                                for panel in ('q4','q8','all')}
    output = dict(status='IN_PROGRESS', draws=DRAWS, seed=SEED, source_groups=ng,
        unit='Joint canonical-source bootstrap; all tasks, repeated answers and scorers share group weights', absolute={}, paired={})
    for j, arm in enumerate(arms):
        draws, within, _ = prm_distribution(j, a['valid'][:,j])
        output['absolute'][arm] = dict(prm=ci(draws), within=ci(within), pb={p:ci(v) for p,v in pb_distributions[arm].items()})
        save(OUT/'INTERVALS.json', output); state('BOOTSTRAP_ABSOLUTE', completed=j+1,total=len(arms))
        print('Intervals', j+1, '/', len(arms),flush=True)
    pairs = [(arm, 'dual__iu') for arm in arms if arm!='dual__iu']
    pairs += [(f'{scope}__graph010', f'{scope}__{reference}') for scope in ('moment','context','single','dual') for reference in ('joint0','graph_perm')]
    for n, (left,right) in enumerate(pairs):
        j,k = arms.index(left),arms.index(right); common = a['valid'][:,j] & a['valid'][:,k]
        dl,wl,pl = prm_distribution(j, common); dr,wr,pr = prm_distribution(k, common)
        output['paired'][left+' minus '+right] = dict(left=left,right=right,prm_left=pl,prm_right=pr,
            prm_delta=pl['auroc']-pr['auroc'], within_delta=pl['within_answer_auc']-pr['within_answer_auc'],
            prm=ci(dl-dr),within=ci(wl-wr),pb={p:dict(delta=metrics[left]['pb']['macros'][p]-metrics[right]['pb']['macros'][p],
              **ci(pb_distributions[left][p]-pb_distributions[right][p])) for p in ('q4','q8','all')})
        save(OUT/'INTERVALS.json',output);state('BOOTSTRAP_PAIRED',completed=n+1,total=len(pairs))
    output['status']='COMPLETE';save(OUT/'INTERVALS.json',output)
    return output


def main():
    OUT.mkdir(exist_ok=True);start=time.time();bootstrap_preflight()
    manifest, frozen = load(RUN/'MANIFEST.json'),load(RUN/'SCORES_FROZEN.json')
    assert frozen['status']=='COMPLETE_ANCHOR_PASS' and frozen['rows']==13769
    assert frozen['manifest_sha256']==sha(RUN/'MANIFEST.json')
    hashes = {str(p):sha(p) for p in (Path(__file__),PROTOCOL,RELEASE,PILOT,RUN/'MANIFEST.json',RUN/'SCORES_FROZEN.json')}
    if (OUT/'MANIFEST.json').exists(): assert load(OUT/'MANIFEST.json')['hashes']==hashes
    else: save(OUT/'MANIFEST.json',dict(hashes=hashes,draws=DRAWS,seed=SEED,created_unix=time.time()))
    if (OUT/'JOINED.json').exists():
        joined=load(OUT/'JOINED.json');assert sha(OUT/'JOINED.npz')==joined['arrays_sha256']
        records,arms,sources=joined['records'],joined['arms'],joined['sources']
        with np.load(OUT/'JOINED.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
    else: records,arms,sources,a=prepare_inputs(manifest,frozen)
    state('METRICS');metrics={arm:metric(records,a,j) for j,arm in enumerate(arms)}
    pilot=load(PILOT);by_uid={r['uid']:i for i,r in enumerate(records)}
    subset=np.zeros(len(records),bool)
    for row in pilot['rows']:
        i=by_uid[row['uid']];subset[i]=True;lo,hi=a['offsets'][i:i+2]
        for j,arm in enumerate(arms):
            assert bool(a['valid'][i,j])==row['valid'][arm]
            assert bool(a['decision'][i,j])==row['decision_valid'][arm]
            if row['valid'][arm]:np.testing.assert_array_equal(a['scores'][lo:hi,j],row['scores'][arm])
            if row['decision_valid'][arm]:assert a['predictions'][i,j]==row['predictions'][arm]
    for j,arm in enumerate(arms):
        got=metric(records,a,j,subset);old=pilot['metrics'][arm]
        for key in ('auroc','within_answer_auc'):
            assert abs(got['prm'][key]-old['prm'][key])<1e-14
        assert got['prm']['answers']==old['prm']['answers']
        assert abs(got['pb']['macros']['q8']-old['pb']['macro_f1'])<1e-14
        mask=(a['labels']>=0)&np.repeat(a['valid'][:,j],np.diff(a['offsets']))
        assert abs(roc_auc_score(a['labels'][mask],a['scores'][mask,j])-metrics[arm]['prm']['auroc'])<1e-14
    save(OUT/'METRICS.json',dict(status='POINT_METRICS_VERIFIED_INTERVALS_PENDING',metrics=metrics,sources=sources))
    print('All19 point metrics and110-answer bridge verified.',flush=True)
    uncertainty=intervals(records,arms,a,metrics)
    review=dict(status='PASS',joined_rows=len(records),frozen_score_hashes=len(frozen['files']),
        pilot_rows_replayed=110,pilot_metric_bundles_replayed=19,independent_full_auc_checks=19,
        weighted_auc_tie_tests=4,source_groups=uncertainty['source_groups'],
        note='Same-session metric/input review. No new fits, external review or untouched confirmation.',seconds=time.time()-start)
    for path,h in hashes.items():assert sha(path)==h
    save(OUT/'REVIEW.json',review)
    rendered=[];records_csv=[]
    for arm,m in metrics.items():
        row=dict(method=arm,prm_auc=m['prm']['auroc'],prm_within=m['prm']['within_answer_auc'],
          prm_valid=m['prm']['answers'],prm_mixed=m['prm']['mixed_answers'],pb_q4=m['pb']['macros']['q4'],
          pb_q8=m['pb']['macros']['q8'],pb_all=m['pb']['macros']['all'],
          pb_valid=sum(d['valid_decisions'] for d in m['pb']['cells'].values()))
        records_csv.append(row)
        rendered.append('<tr><td>'+arm+'</td><td>'+f'{row["prm_auc"]:.4f}'+'</td><td>'+f'{row["prm_within"]:.4f}'+'</td><td>'+str(row['prm_valid'])+'/6969</td><td>'+f'{100*row["pb_q4"]:.2f}%'+'</td><td>'+f'{100*row["pb_q8"]:.2f}%'+'</td><td>'+f'{100*row["pb_all"]:.2f}%'+'</td><td>'+str(row['pb_valid'])+'/6800</td></tr>')
    buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=list(records_csv[0]));w.writeheader();w.writerows(records_csv)
    (OUT/'METRICS.csv').write_text(buf.getvalue(),encoding='utf-8')
    pair_rows=[]
    for key,d in uncertainty['paired'].items():
        pair_rows.append('<tr><td>'+html.escape(key)+'</td><td>'+str(d['prm_left']['answers'])+'</td><td>'+str(d['prm']['ci95'])+'</td><td>'+str(d['within']['ci95'])+'</td><td>'+str(d['pb']['q8']['ci95'])+'</td></tr>')
    report='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Full cached localization: anchor evaluation</title><style>body{font:16px/1.5 system-ui;max-width:1250px;margin:30px auto;padding:20px;color:#23354a}table{border-collapse:collapse;font-size:14px;width:100%}th,td{padding:9px;border-bottom:1px solid #ccd5dd;text-align:left}.scroll{overflow:auto}.note{background:#fff3d8;padding:20px}</style><h1>Full cached localization: 19 anchor methods</h1><p class="note">Pass1 evaluated on all13,769 registered model-answer rows; same-session review PASS. This is development evidence. New shortlist candidates and corrected historical refits are still pending. These Joint arms retain their original conditioning, not the newer condition100 recipe. Native Joint and fallback routes have different coverage.</p><p>PRMB:6969 answers, corrected v3 labels. PB:3400 answers scored by each Qwen3-4B andQwen3-8B. Unsupported/invalid PB decisions count as failures. PRMB invalid scores are disclosed. Pooled AUC is not official PRMScore. Within-answer AUC averages mixed-label answers.</p><div class="scroll"><table><tr><th>Method</th><th>PRMB pooled AUC</th><th>Within AUC</th><th>PRMB valid</th><th>PB Q4</th><th>PB Q8</th><th>PB all</th><th>PB valid</th></tr>TABLE</table></div><h2>Paired exploratory95% intervals</h2><p>Left minus right; native0-1 units. PRMB uses common-valid support. All1000 draws resample canonical source groups jointly across both tasks and repeated scorers. Intervals are not multiplicity adjusted. Separate point differences and Q4/all-cell intervals are in INTERVALS.json.</p><div class="scroll"><table><tr><th>Comparison</th><th>Common PRMB answers</th><th>Pooled delta CI</th><th>Within delta CI</th><th>PB Q8 delta CI</th></tr>PAIRS</table></div><p><a href="METRICS.csv">Summary CSV</a> · <a href="METRICS.json">All cell metrics and source routes</a> · <a href="INTERVALS.json">All absolute/paired intervals</a> · <a href="REVIEW.json">Review</a> · <a href="MANIFEST.json">Evaluation freeze</a></p></html>'''
    (OUT/'REPORT.html').write_text(report.replace('TABLE',''.join(rendered)).replace('PAIRS',''.join(pair_rows)),encoding='utf-8')
    save(OUT/'METRICS.json',dict(status='COMPLETE_ANCHOR_EVALUATION',metrics=metrics,sources=sources))
    save(OUT/'ARTIFACT_CHECKS.json',dict(status='PASS',metric_rows=len(rendered),paired_rows=len(pair_rows),
           report_sha256=sha(OUT/'REPORT.html'),browser_rendered=False))
    state('COMPLETE_ANCHOR_EVALUATION',rows=len(records),methods=len(arms),historical_comparison_complete=False,seconds=time.time()-start)
    print('Full anchor evaluation COMPLETE:',OUT/'REPORT.html',flush=True)


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    try:main()
    except Exception as exc:
        state('FAILED_CHECKPOINTS_PRESERVED',reason=repr(exc));raise
