"""Read-only independent audit of frozen Claude Steps 428-429; writes only this audit."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import sys, json, hashlib, math, ast, types
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
WT = ROOT / '.worktrees/readout-quickest-detection-v1'
OUT = ROOT / 'scratch/claude_readout_review_20260922'
OUT.mkdir(exist_ok=True)
R = WT / 'results/readout_family_v1'
result = {}
def read(p): return json.loads(p.read_text(encoding='utf8'))
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()

summary = pd.read_csv(R/'SUMMARY.csv').set_index('method')
ans = pd.read_csv(R/'OOF_ANSWERS.csv')
pb = ans.cell.str.startswith('pb_').to_numpy()
target = ans.target.to_numpy()
gate = ans.ct7_gate.to_numpy().astype(bool)
groups = ans.source_group.to_numpy()
result['population'] = {'rows':len(ans), 'pb':int(pb.sum()), 'prm':int((~pb).sum()),
    'pb_errors':int((pb & (target >= 0)).sum()),
    'groups_crossing_folds':int((ans.groupby('source_group').fold.nunique()>1).sum())}
errors = []
for method, row in summary.iterrows():
    col = method+'__mode'
    if col not in ans: continue
    pred = ans[col].to_numpy()
    slas, f1s = [], []
    for cell in sorted(set(ans.cell[pb])):
        mask = (ans.cell==cell).to_numpy() & np.isfinite(pred)
        e, c = mask & (target >= 0), mask & (target < 0)
        sla = (pred[e] == target[e]).mean()
        ca = (~gate[c]).mean()
        ea = ((pred[e] == target[e]) & gate[e]).mean()
        slas.append(sla); f1s.append(2*ca*ea/(ca+ea) if ca+ea else 0.)
    errors.append({'method':method,'sla_error':float(abs(np.mean(slas)-row.pb_sla_macro8)),
                   'f1_error':float(abs(np.mean(f1s)-row.pb_f1_ct7_gate))})
result['pb_metric_replay'] = {'methods':len(errors), 'max_sla_error':max(r['sla_error'] for r in errors),
                             'max_f1_error':max(r['f1_error'] for r in errors)}

jobs = [read(f) for f in (R/'jobs').glob('*.json')]
result['jobs'] = {'outer':sum(j['inner_fold'] is None for j in jobs),
    'inner':sum(j['inner_fold'] is not None for j in jobs),
    'train_test_group_overlap':sum(bool(set(j['train_source_groups']) & set(j['test_source_groups'])) for j in jobs)}
freeze = read(R/'RUN_FREEZE.json')
result['frozen_source_hashes'] = {n:digest(R/'source_snapshot'/n)==h for n,h in freeze.items() if n!='inputs'}
result['input_freeze_hash'] = digest(R/'INPUT_FREEZE.json')==freeze['inputs']
unc = read(R/'UNCERTAINTY.json')
result['multiplicity'] = {'comparisons':len(unc['contrasts']), 'draws':unc['draws'],
    'min_raw_p':min(r['p_bootstrap'] for r in unc['contrasts']),
    'min_holm_p':min(r['p_holm'] for r in unc['contrasts']),
    'holm_below_005':sum(r['p_holm']<.05 for r in unc['contrasts']),
    'unadjusted_ci_excludes_zero':sum(r['ci95'][0]>0 or r['ci95'][1]<0 for r in unc['contrasts'])}

# Load only function definitions for the synthetic causality fixture; no experiment code executes.
def defs(path,names,namespace):
    tree=ast.parse(path.read_text(encoding='utf8'))
    tree.body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names]
    exec(compile(tree,str(path),'exec'),namespace)
ns={'np':np,'DEFAULTS':{'page_k':.5}}
defs(WT/'spectral_utils/changepoint_step_readout_v1.py',{'_series','zscore_within','page_cusum'},ns)
defs(WT/'spectral_utils/step_readouts_v1.py',{'_check','_reduce','step_page_wmax'},ns)
defs(WT/'scripts/diagnostics/quickest_detection_diagnostics_v1.py',{'causal_z'},ns)
prefix=np.array([0.,1.,2.,3.,4.,5.])[:,None]
x=np.concatenate([prefix,np.full((30,1),100.)])
y=np.concatenate([prefix,np.full((30,1),-100.)])
spans=np.array([[0,3],[3,6],[6,36]])
cx,cy=ns['causal_z'](x),ns['causal_z'](y)
px,py=ns['step_page_wmax'](cx,spans),ns['step_page_wmax'](cy,spans)
result['causality_fixture']={'same_raw_prefix':True,'causal_z_prefix_max_diff':float(abs(cx[:6]-cy[:6]).max()),
    'claimed_causal_page_prefix_max_diff':float(abs(px[:2]-py[:2]).max()),
    'page_prefix_a':px[:2,0].tolist(),'page_prefix_b':py[:2,0].tolist()}

cfg=read(WT/'configs/readout_family_v1.json')
joined=np.load(cfg['paths']['joined'])
off, labels = joined['offsets'],joined['labels']
steps=np.diff(off)
oof=np.load(R/'OOF_STEP_SCORES.npz')
selected=['ct7','token_lsml','top5__all__soft__equal','top30__all__soft__equal',
    'consensus__all__soft__equal','selected_all__all__hard__ds',
    'top5__all__pmf__continuous_lsml','selected__all__soft__equal']
result['prm_metric_replay']=[]
result['competition']=[]
for method in selected:
    s=oof[method]
    aucs=[]
    for i in np.flatnonzero(~pb):
        a,b=off[i:i+2];yy=labels[a:b].astype(bool);n1=yy.sum();n0=len(yy)-n1
        if n0 and n1:aucs.append(float((rankdata(s[a:b])[yy].sum()-n1*(n1+1)/2)/(n0*n1)))
    result['prm_metric_replay'].append({'method':method,'eligible':len(aucs),
       'auc':float(np.mean(aucs)),'error':float(abs(np.mean(aucs)-summary.loc[method,'within_auc']))})
    if method not in ['ct7','token_lsml']:continue
    for name,lo,hi in [('2_to_5',2,5),('6_to_10',6,10),('11_plus',11,9999)]:
        rows=[]
        for i in np.flatnonzero(pb & (target>=0) & (steps>=lo) & (steps<=hi)):
            a,b=off[i:i+2];v=s[a:b];t=int(target[i]);ids=np.arange(len(v));others=ids!=t
            before=ids<t;after=ids>t
            outrank=(v>v[t])|((v==v[t])&(ids<t))
            good=int((others & ~outrank).sum());n=len(v)-1
            # Expected exact success after retaining target and 3 uniformly drawn competitors.
            fixed=float(math.comb(good,3)/math.comb(n,3)) if n>=3 and good>=3 else (0. if n>=3 else np.nan)
            rows.append({'exact':float(np.argmax(v)==t),'pairwise_all':float(np.mean((v[t]>v[others])+.5*(v[t]==v[others]))),
                'before':float(np.mean((v[t]>v[before])+.5*(v[t]==v[before]))) if before.any() else np.nan,
                'after':float(np.mean((v[t]>v[after])+.5*(v[t]==v[after]))) if after.any() else np.nan,
                'retained_3_competitors_success':fixed})
        frame=pd.DataFrame(rows)
        result['competition'].append({'method':method,'stratum':name,'n':len(rows),**frame.mean().to_dict()})

curves=pd.read_csv(WT/'results/quickest_detection_diagnostics_v1/B1_DELAY_CURVES.csv')
m=curves[(curves.scope=='macro8_all') & (curves.false_alarm<=.30)]
result['first_crossing_30pct']={'locators':int(m.locator.nunique()),
    'beats_argmax':int((m.groupby('locator').apply(lambda d:(d.exact-d.argmax_sla).max(),include_groups=False)>0).sum())}
result['reported_headline_rows']=summary.loc[selected,['pb_sla_macro8','pb_f1_ct7_gate','within_auc']].reset_index().to_dict('records')
(OUT/'AUDIT.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result,indent=2))
