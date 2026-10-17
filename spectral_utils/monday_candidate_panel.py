"""Read-only post-review candidate comparison, without fitting or selecting a recipe.

The active experiment package supplies the metric definitions, not its fit paths.
All reconstructed scores must replay the sealed q80 content rows before reporting.
"""
from pathlib import Path
import datetime
import hashlib
import html
import json
import sys
import gc
import numpy as np
import pandas as pd

ROOT = Path(r'C:\Users\omris\TAU\hallucination_detection')
CODE = ROOT / 'results/monday_fusion_analysis_v1/code'
OUT = ROOT / 'results/monday_candidate_panel_v1/run_20261004'
INPUT = ROOT / 'results/estimator_input_v1/run_20261004_resume'
LEVEL = ROOT / 'results/level_reduction_v1/run_20261004_resume'
BANKS = ['B13','B16','B20','B23','B32','B35','B51','B54']
VARIANTS = ['groups_answer','channels_cell']
RECIPES = ['MEAN_MOM','HEM_MOM','DS_FILTER_EQ','ALL_EQ','MEAN_EQ','HEM_EQ']
RULES = ['ANSWER_Z_Q80','R2_COUNT','ENTROPY_GATE_Q80','TOP_1','TOP_2','TOP_3']
FIELDS = ['within_answer_auc','prmscore','pb_first_error_accuracy','pb_decision_error_accuracy',
          'pb_official_f1','clean_flag_rate','wrong_answer_auroc']
SCOPE = 'pre-selection development comparison; not a locked recipe; not untouched confirmation'

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def dump(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist())+'\n',encoding='utf8')

class Evidence:
    def __init__(self): self.used={}
    def use(self,path,expected=None):
        path=Path(path); digest=sha(path)
        if expected is not None and digest!=expected: raise RuntimeError('Input hash mismatch: '+str(path))
        if str(path) in self.used and self.used[str(path)]!=digest: raise RuntimeError('Input changed: '+str(path))
        self.used[str(path)]=digest
        return path
    def sealed(self,directory,stem):
        stamp=directory/(stem+'.complete.json')
        if not stamp.exists(): return False
        record=json.loads(self.use(stamp).read_text(encoding='utf8'))
        if not record.get('artifacts'): raise RuntimeError('Missing artifact seal: '+str(stamp))
        for suffix,digest in record['artifacts'].items(): self.use(directory/(stem+suffix),digest)
        return True
    def verify(self):
        for path,digest in self.used.items():
            if sha(path)!=digest: raise RuntimeError('Input changed during panel: '+path)

def setup():
    # Only imports metric/reconstruction helpers. Never import a script that fits.
    sys.path.insert(0,str(CODE/'scripts/experiments'))
    import spectral_utils
    spectral_utils.__path__.append(str(CODE/'spectral_utils'))
    from calfix_common import Population
    return Population()

class Inputs:
    """Small normalized arrays plus read-only memory maps; no HistoricalData fits."""
    def __init__(self,pop,evidence):
        from spectral_utils.lsml_gate_locator_research import answer_standardize
        self.p=pop; self.ev=evidence
        self.manifest=json.loads(evidence.use(INPUT/'INPUT_MANIFEST.json').read_text(encoding='utf8'))
        self.paths={k:evidence.use(v['path'],v['sha256']) for k,v in self.manifest['inputs'].items()}
        z=np.load(self.paths['level_bank']); self.names11=list(map(str,z['channels']))
        prof=np.load(self.paths['ct7_profiles'],mmap_mode='r')
        pn=json.loads(self.paths['ct7_profile_validation'].read_text(encoding='utf8'))['channels']
        self.raw13=np.column_stack([z['level'].astype(float),prof[:,pn.index('chosen_token_z_despiked')],z['derivative'][:,self.names11.index('chosen_surprisal')].astype(float)])
        z.close(); self.norm13=answer_standardize(self.raw13,pop.off)
        with np.load(self.paths['digit_features']) as d:
            self.digit_raw=d['values']; act=d['active'].astype(bool)
        self.digit=np.zeros_like(self.digit_raw,dtype=float)
        for a,b in zip(pop.off[:-1],pop.off[1:]):
            for j in range(3):
                valid=act[a:b,j]; y=self.digit_raw[a:b,j][valid]
                if len(y) and y.std()>1e-12: self.digit[a:b,j][valid]=(y-y.mean())/y.std()
        self.pool=np.load(self.paths['pool_z'],mmap_mode='r')
        self.pn=json.loads(self.paths['pool_names'].read_text(encoding='utf8'))
        raw_manifest=json.loads(evidence.use(ROOT/'results/monday_fusion_analysis_v1/RAW_INPUT_MANIFEST_V3.json').read_text(encoding='utf8'))
        rawpaths={k:evidence.use(v['path'],v['sha256']) for k,v in raw_manifest['additional_inputs'].items()}
        self.union=np.load(rawpaths['union'],mmap_mode='r'); self.hist=np.load(rawpaths['hist'],mmap_mode='r')
        self.profiles=np.load(rawpaths['profiles'],mmap_mode='r')
        with np.load(rawpaths['risk']) as z: risk=z['risk_token'].astype(float); self.mtg=z['step_m5'].astype(float)
        if self.mtg.shape!=(pop.total,): raise RuntimeError('MTG population shape mismatch')
        with np.load(rawpaths['tokens']) as z: toff=z['token_offsets']; spans=z['step_spans']
        ev=np.empty(pop.total)
        for i in range(pop.n):
            r=risk[toff[i]:toff[i+1]]
            for s in range(pop.off[i],pop.off[i+1]):
                a,b=spans[s]; v=r[a:b]; k=min(10,len(v))
                if not k: raise RuntimeError('Empty risk span')
                ev[s]=np.partition(v,len(v)-k)[-k:].mean()
        self.risk=ev
        with np.load(self.paths['answer_features']) as z:
            if not np.array_equal(z['ids'].astype(str),pop.ids): raise RuntimeError('Entropy identity mismatch')
            if not np.array_equal(z['cells'].astype(str),pop.cells) or not np.array_equal(z['folds'],pop.fold): raise RuntimeError('Entropy population mismatch')
            self.entropy=z['X'][:,list(map(str,z['names'])).index('epr')]

    def raw_column(self,j,rows):
        # Exact column concatenation from monday_fusion_decision.raw_pool.
        if j<11: return self.raw13[rows,j]
        n=self.union.shape[1]-11
        if j<11+n: return self.union[rows,j]
        j-=11+n
        if j<4: return self.profiles[rows,0,[14,11,12,10][j]]
        if j==4: return self.risk[rows]
        return self.hist[rows,j-5]

    def bank(self,bk,ce,record):
        from spectral_utils.lsml_gate_locator_research import answer_standardize
        p=self.p; ans=np.flatnonzero(p.cells==ce); rows=p.rows(p.cells==ce); off=np.r_[0,np.cumsum(p.ns[ans])]
        names=self.manifest['banks'][bk]; v=[]; raw=[]; need=[]
        digit=['digit_alternative','digit_spread','digit_alternative_innovation']
        names13=self.names11+['realized_z','realized_drv']
        for name in names:
            if name in names13 and bk in ('B13','B16'):
                j=names13.index(name); v.append(self.norm13[rows,j]); raw.append(self.raw13[rows,j])
            elif name in digit:
                j=digit.index(name); v.append(self.digit[rows,j]); raw.append(self.digit_raw[rows,j])
            else:
                j=self.pn.index(name[4:] if name.startswith('lf__') else name)
                v.append(self.pool[rows,j]); raw.append(self.raw_column(j,rows)); need.append(len(v)-1)
        if need:
            norm=answer_standardize(np.column_stack([v[j] for j in need]),off)
            for k,j in enumerate(need): v[j]=norm[:,k]
        for j,name in enumerate(names):
            if name.startswith('lf__'):
                sign=record['diagnostics']['lf_signs'][name]['sign']; v[j]=v[j]*sign; raw[j]=raw[j]*sign
        return {'V':np.column_stack(v),'raw':np.column_stack(raw),'answers':ans,'rows':rows,'off':off,'names':names}

def reconstruct(bank,record):
    import ds_group_weights as GW
    from spectral_utils.lsml_gate_locator_research import answer_standardize
    surv=np.asarray(record['surv'],int); x=bank['V'][:,surv]; groups=np.asarray(record['groups'],int)
    scores={'ALL_EQ':bank['V'].mean(1),'DS_FILTER_EQ':x@np.full(len(surv),1/len(surv))}
    for recipe in ['MEAN_MOM','HEM_MOM','MEAN_EQ','HEM_EQ']:
        r=record['recipes'][recipe]
        wi=None if r['within'] is None else [None if w is None else np.asarray(w,float) for w in r['within']]
        z=GW.group_matrix(x,bank['off'],groups,wi,answer_standardize)
        scores[recipe]=GW.weighted_group_score(z,np.asarray(r['weights'],float))
    return scores

def statistics(pop,bank,per):
    ans=bank['answers']; groups,gi=np.unique(pop.groups[ans],return_inverse=True)
    err=per['erroneous']; target=pop.target[ans]; auc=per['auc']; valid=np.isfinite(auc)
    # Per-source columns: AUC sum/count, confusion4, correct first/error count,
    # correct clean/clean count, clean flags, argmax hits.
    values=np.column_stack([np.nan_to_num(auc),valid,per['confusion'],
        (per['first']==target)&err,err,(per['first']==target)&~err,~err,
        (per['flag_count']>0)&~err,(per['argmax']==target)&err]).astype(float)
    sums=np.zeros((len(groups),12)); np.add.at(sums,gi,values)
    return groups,sums

def decision_metrics(pop,bank,score,flags,detector,reference):
    """Re-use decision-invariant ranks; independently aggregate each new flag rule.

    reference is the official evaluate() result on exactly this score and bank.
    No ranking, label or answer detector is inferred from another recipe.
    """
    from sklearn.metrics import roc_auc_score
    from spectral_utils.monday_fusion_metrics import prmscore_counts
    metric0,per0=reference; ans=bank['answers']; off=bank['off']; starts=off[:-1]
    count=np.add.reduceat(flags.astype(int),starts)
    first=np.minimum.reduceat(np.where(flags,np.arange(len(score)),len(score)),starts)-starts
    first=np.where(count>0,first,-1)
    err=per0['erroneous']; counts=np.zeros_like(per0['confusion'])
    metric=dict(metric0)
    metric['clean_flag_rate']=float((count[~err]>0).mean()) if (~err).any() else None
    if pop.prm[ans[0]]:
        y=pop.labels[bank['rows']]; valid=~flags; gold=~y
        for j,v in enumerate([valid&gold,valid&~gold,~valid&~gold,~valid&gold]): counts[:,j]=np.add.reduceat(v.astype(int),starts)
        counts[~pop.noncontrol[ans]]=0
        metric['prmscore']=float(prmscore_counts(counts.sum(0))) if counts.sum()>0 else None
    else:
        hit=first==pop.target[ans]
        ae=float(hit[err].mean()) if err.any() else np.nan
        ac=float(hit[~err].mean()) if (~err).any() else np.nan
        metric['pb_decision_error_accuracy']=ae if err.any() else None
        metric['pb_official_f1']=(2*ae*ac/(ae+ac) if ae+ac>0 else 0.) if err.any() and (~err).any() else None
    detector=np.maximum.reduceat(score,starts) if detector is None else detector
    metric['wrong_answer_auroc']=float(roc_auc_score(err,detector)) if min(int(err.sum()),int((~err).sum()))>=10 else None
    return metric,dict(auc=per0['auc'],confusion=counts,first=first,argmax=per0['argmax'],flag_count=count,erroneous=err)

def assert_metric_parity(actual,expected):
    for field in FIELDS+['n_answers','n_steps','n_auc_answers','n_erroneous','n_clean']:
        a,b=actual[0][field],expected[0][field]
        if a is None or b is None:
            if a is not b: raise RuntimeError('Missing metric mismatch '+field)
        elif abs(a-b)>1e-12: raise RuntimeError('Metric mismatch '+field+': '+str((a,b)))
    for field in ['auc','confusion','first','argmax','flag_count','erroneous']:
        if not np.array_equal(actual[1][field],expected[1][field],equal_nan=True): raise RuntimeError('Per-answer mismatch '+field)

def content(pop,data,evidence,smoke=False):
    from spectral_utils.monday_fusion_decision import rules
    from spectral_utils.monday_fusion_metrics import evaluate,top_flags
    rows=[]; errors=[]; direct_checks=0; srcdir=INPUT/'chunks'
    statsdir=OUT/'sufficient'; statsdir.mkdir(exist_ok=True)
    score_dir=OUT/'candidate_scores'; score_dir.mkdir(exist_ok=True)
    for bk in BANKS[:1] if smoke else BANKS:
        for ce in [sorted(set(pop.cells))[0],'prmbench_qwen3_8b'] if smoke else sorted(set(pop.cells)):
            stem0=ce+'__'+bk+'__groups_answer'
            if not evidence.sealed(srcdir,stem0): raise RuntimeError('Missing core '+stem0)
            record0=json.loads((srcdir/(stem0+'.diagnostics.json')).read_text(encoding='utf8'))
            bank=data.bank(bk,ce,record0); ans=bank['answers']
            for variant in VARIANTS:
                stem=ce+'__'+bk+'__'+variant
                if not evidence.sealed(srcdir,stem): raise RuntimeError('Missing core '+stem)
                record=json.loads((srcdir/(stem+'.diagnostics.json')).read_text(encoding='utf8'))
                scores=reconstruct(bank,record); sealed=pd.read_csv(srcdir/(stem+'.metrics.csv'))
                arrays={'answer_indices':ans,'offsets':bank['off']}; stats_arrays={}
                with np.load(srcdir/(stem+'.stats.npz')) as old:
                    assert np.array_equal(old['answer_indices'],ans)
                    for recipe in RECIPES:
                        score=scores[recipe]; rr=bank['raw'] if recipe=='ALL_EQ' else bank['raw'][:,record['surv']]
                        flags,detectors,dd=rules(score,bank,rr,data.entropy[ans],pop.uids[ans])
                        for k in (1,2,3): flags['TOP_'+str(k)]=top_flags(score,bank['off'],np.full(len(ans),k))
                        reference=evaluate(pop,bank,score,flags['ANSWER_Z_Q80'],detectors['ANSWER_Z_Q80'])
                        if recipe+'__step_score' in old:
                            diff=float(np.max(np.abs(score-old[recipe+'__step_score'])))
                            if diff>1e-9: raise RuntimeError(f'Score replay {stem}/{recipe}: {diff}')
                        for rule in RULES:
                            m,per=decision_metrics(pop,bank,score,flags[rule],detectors.get(rule),reference)
                            if smoke and recipe=='MEAN_MOM':
                                assert_metric_parity((m,per),evaluate(pop,bank,score,flags[rule],detectors.get(rule)))
                                direct_checks+=1
                            row=dict(bank=bk,cell=ce,variant=variant,recipe=recipe,position_variant='CONTENT',decision_rule=rule,
                                role='CANDIDATE' if recipe.endswith('_MOM') else 'CONTROL',top2_post_hoc=rule=='TOP_2',
                                source_file=str(srcdir/(stem+'.complete.json')),scope=SCOPE,n_total=len(ans),
                                n_prmscore_answers=int(pop.noncontrol[ans].sum()),**m)
                            rows.append(row)
                            if rule=='ANSWER_Z_Q80':
                                match=sealed[(sealed.recipe==recipe)&(sealed.decision_rule==rule)].iloc[0]
                                for field in FIELDS:
                                    if m[field] is not None:
                                        delta=abs(float(m[field])-float(match[field])); errors.append(delta)
                                        if delta>1e-12: raise RuntimeError(f'Metric replay {stem}/{recipe}/{field}: {delta}')
                            groups,sums=statistics(pop,bank,per); stats_arrays['groups']=groups
                            stats_arrays[recipe+'__'+rule]=sums
                            if recipe.endswith('_MOM') and rule in RULES[:3]: arrays[recipe+'__'+rule+'__flags']=flags[rule]
                        if recipe.endswith('_MOM'): arrays[recipe+'__score']=score
                np.savez_compressed(statsdir/(stem+'.npz'),**stats_arrays)
                np.savez_compressed(score_dir/(stem+'.npz'),**arrays)
            print('Candidate content replay '+bk+'/'+ce,flush=True)
            gc.collect()
    frame=pd.DataFrame(rows); frame.to_csv(OUT/'CONTENT_METRICS.csv',index=False)
    return frame,{'sealed_metric_checks':len(errors),'max_difference':max(errors,default=0.),'direct_all_rule_evaluate_checks':direct_checks}

def check_claude(frame,evidence):
    path=ROOT/'results/monday_fusion_review_claude_20261004_round2/decision_rules_precompute/DECISION_METRICS.csv'
    reference=pd.read_csv(evidence.use(path))
    keys=['bank','cell','variant','recipe','decision_rule']; merged=frame.merge(reference,on=keys,suffixes=('','_claude'),validate='one_to_one')
    errors=[]
    for f in FIELDS:
        a=merged[f].to_numpy(float); b=merged[f+'_claude'].to_numpy(float)
        if not np.array_equal(np.isnan(a),np.isnan(b)): raise RuntimeError('Claude missing-value mismatch '+f)
        errors.extend(np.abs(a[np.isfinite(a)]-b[np.isfinite(b)]).tolist())
    largest=max(errors,default=0.)
    if largest>1e-12: raise RuntimeError('Claude decision-table mismatch '+str(largest))
    return {'rows_compared':len(merged),'numeric_values_compared':len(errors),'max_difference':largest,'source':str(path),'status':'PASS'}

def aggregates(frame):
    frame=frame.copy(); frame['benchmark']=np.where(frame.cell.str.startswith('prm'),'PRMBench','ProcessBench')
    rows=[]
    for (benchmark,variant,recipe,rule),part in frame.groupby(['benchmark','variant','recipe','decision_rule']):
        expected_cells=1 if benchmark=='PRMBench' else 8
        for bk in BANKS+['EQUAL_8_BANK_MEAN']:
            selected=part if bk=='EQUAL_8_BANK_MEAN' else part[part.bank==bk]
            expected=expected_cells*(8 if bk=='EQUAL_8_BANK_MEAN' else 1)
            valid=selected[selected.coverage==1]
            complete=len(selected)==expected and len(valid)==expected
            # Counts are unique model answers per bank, never multiplied by number of banks.
            unique=selected.drop_duplicates('cell')
            row=dict(benchmark=benchmark,bank=bk,variant=variant,recipe=recipe,decision_rule=rule,
                n_expected_bank_cells=expected,n_valid_bank_cells=len(valid),complete=complete,
                n_answers=int(unique.n_total.sum()),n_auc_answers=int(unique.n_auc_answers.sum()),
                n_prmscore_answers=int(unique.n_prmscore_answers.sum()),
                n_erroneous=int(unique.n_erroneous.sum()),n_clean=int(unique.n_clean.sum()),
                source_file='CONTENT_METRICS.csv',scope=SCOPE,top2_post_hoc=rule=='TOP_2')
            for f in FIELDS: row[f]=float(selected[f].mean()) if complete and selected[f].notna().any() else np.nan
            rows.append(row)
    result=pd.DataFrame(rows); result.to_csv(OUT/'CANDIDATE_PANEL.csv',index=False)
    return result

def endpoints(s):
    from spectral_utils.monday_fusion_metrics import prmscore_counts
    def div(a,b): return np.divide(a,b,out=np.full(np.shape(a),np.nan),where=b>0)
    ae=div(s[...,6],s[...,7]); ac=div(s[...,8],s[...,9])
    return {'within_answer_auc':div(s[...,0],s[...,1]),'prmscore':prmscore_counts(s[...,2:6]),
        'pb_first_error_accuracy':div(s[...,11],s[...,7]),'pb_decision_error_accuracy':ae,
        'pb_official_f1':np.where((s[...,7]>0)&(s[...,9]>0),np.where(ae+ac>0,div(2*ae*ac,ae+ac),0.),np.nan),
        'clean_flag_rate':div(s[...,10],s[...,9])}

def canonical_peak(score):
    """Earliest numerical tie, exactly matching monday_fusion_metrics.evaluate."""
    return int(np.flatnonzero(score>=score.max()-8*np.finfo(float).eps)[0])

def bootstrap(pop,draws=5000):
    comparisons=[]
    for variant in VARIANTS:
        for recipe in ['MEAN_MOM','HEM_MOM']:
            comparisons.append((variant,recipe,variant,'DS_FILTER_EQ','candidate_minus_DS_FILTER_EQ'))
        comparisons.append((variant,'HEM_MOM',variant,'MEAN_MOM','HEM_minus_MEAN'))
    for recipe in ['MEAN_MOM','HEM_MOM']:
        comparisons.append(('channels_cell',recipe,'groups_answer',recipe,'cell_channels_minus_answer_groups'))
    rows=[]
    prm_has_error=np.array([bool(pop.labels[a:b].any()) if pop.prm[i] else False for i,(a,b) in enumerate(zip(pop.off[:-1],pop.off[1:]))])
    for benchmark in ['PRMBench','ProcessBench']:
        is_prm=benchmark=='PRMBench'; mask=pop.prm if is_prm else pop.pb
        universe=np.unique(pop.groups[mask]); group_index={g:i for i,g in enumerate(universe)}
        cells=sorted(set(pop.cells[mask])); kinds=['within_answer_auc','prmscore','clean_flag_rate'] if is_prm else ['pb_first_error_accuracy','pb_decision_error_accuracy','pb_official_f1','clean_flag_rate']
        # Same bootstrap weights for all banks, cells, recipes and rules. Store only
        # resampled endpoints, not per-step or per-answer bootstrap matrices.
        distributions={}; points={}
        for bk in BANKS:
            for ce in cells:
                for variant in VARIANTS:
                    path=OUT/'sufficient'/(ce+'__'+bk+'__'+variant+'.npz')
                    with np.load(path) as z:
                        local=np.array([group_index[g] for g in z['groups']]); keys=[r+'__'+d for r in ['MEAN_MOM','HEM_MOM','DS_FILTER_EQ'] for d in RULES]
                        matrices=np.column_stack([z[k] for k in keys]); sums=matrices.sum(0).reshape(len(keys),12)
                    point=endpoints(sums); sample={k:np.empty((draws,len(keys))) for k in kinds}
                    rng=np.random.default_rng(2026100418 if is_prm else 2026100419)
                    for start in range(0,draws,100):
                        n=min(100,draws-start)
                        weights=rng.multinomial(len(universe),np.full(len(universe),1/len(universe)),size=n)[:,local].astype(float)
                        ep=endpoints((weights@matrices).reshape(n,len(keys),12))
                        for kind in kinds: sample[kind][start:start+n]=ep[kind]
                    for j,key in enumerate(keys):
                        recipe,rule=key.split('__')
                        for kind in kinds:
                            k=(bk,variant,recipe,rule,kind)
                            distributions[k]=distributions.get(k,0)+sample[kind][:,j]/len(cells)
                            points[k]=points.get(k,0)+float(point[kind][j])/len(cells)
                print('Paired candidate intervals '+bk+'/'+ce,flush=True)
        for bk in BANKS+['EQUAL_8_BANK_MEAN']:
            use=BANKS if bk=='EQUAL_8_BANK_MEAN' else [bk]
            for va,ra,vb,rb,comparison in comparisons:
                for rule in RULES:
                    for kind in kinds:
                        delta=sum(points[b,va,ra,rule,kind]-points[b,vb,rb,rule,kind] for b in use)/len(use)
                        dist=sum(distributions[b,va,ra,rule,kind]-distributions[b,vb,rb,rule,kind] for b in use)/len(use)
                        valid=np.isfinite(dist)
                        lo,hi=np.quantile(dist[valid],[.025,.975]) if valid.any() else (np.nan,np.nan)
                        n_endpoint=int((mask & (pop.eligible if kind=='within_answer_auc' else pop.noncontrol if kind=='prmscore' else
                            ~prm_has_error if is_prm and kind=='clean_flag_rate' else pop.target<0 if kind=='clean_flag_rate' else
                            pop.target>=0 if kind in ['pb_first_error_accuracy','pb_decision_error_accuracy'] else np.ones(pop.n,bool))).sum())
                        rows.append(dict(benchmark=benchmark,bank=bk,variant=va,recipe=ra,control_variant=vb,control_recipe=rb,
                            comparison=comparison,decision_rule=rule,endpoint=kind,delta=delta,lo95=lo,hi95=hi,draws=draws,
                            valid_draws=int(valid.sum()),n_source_groups=len(universe),n_answers=n_endpoint,n_population_answers=int(mask.sum()),
                            multiplicity='Unadjusted descriptive 95%; post-review exploratory; no confirmatory significance claim',
                            source_file='sufficient/*.npz',scope=SCOPE))
    result=pd.DataFrame(rows); result.to_csv(OUT/'PAIRED_CONTRASTS.csv',index=False)
    return result

def checkpoint_panels(pop,evidence):
    positions=[]; lr=[]; coverage=[]
    for bk in BANKS:
        for ce in sorted(set(pop.cells)):
            n=int((pop.cells==ce).sum())
            for variant in VARIANTS:
                stem=ce+'__'+bk+'__'+variant; directory=INPUT/'layers'
                # Existing LEVEL-original layers are exactly the groups_answer score contract.
                if not (directory/(stem+'.complete.json')).exists() and variant=='groups_answer':
                    directory=LEVEL/'layers'; stem=ce+'__'+bk+'__original'
                present=evidence.sealed(directory,stem)
                coverage.append(dict(stage='position',bank=bk,cell=ce,variant=variant,n_total=n,status='SEALED' if present else 'PENDING'))
                if present:
                    part=pd.read_csv(directory/(stem+'.metrics.csv.gz'))
                    part=part[part.recipe.isin(RECIPES)&part.position_variant.isin(['POS','PRIOR','PRIOR_CF'])].copy()
                    part['variant']=variant; part['source_file']=str(directory/(stem+'.complete.json')); positions.extend(part.to_dict('records'))
                else:
                    for recipe in RECIPES:
                        for pos in ['POS','PRIOR','PRIOR_CF']:
                            for rule in RULES:
                                positions.append(dict(bank=bk,cell=ce,variant=variant,recipe=recipe,position_variant=pos,decision_rule=rule,
                                    status='PENDING',n_total=n,n_answers=0,coverage=0,failures=0,failure_reason='Checkpoint not completed; not an arm failure'))
            for roster in ['full','survivors']:
                for pos in [0,1]:
                    stem=f'{ce}__{bk}__original__{roster}__pos{pos}'
                    directory=LEVEL/'lr'
                    present=evidence.sealed(directory,stem)
                    coverage.append(dict(stage='LR',bank=bk,cell=ce,variant='original',roster=roster,position=pos,n_total=n,status='SEALED' if present else 'PENDING'))
                    if present:
                        part=pd.read_csv(directory/(stem+'.metrics.csv')); part['source_file']=str(directory/(stem+'.complete.json')); lr.extend(part.to_dict('records'))
                    else:
                        lr.append(dict(bank=bk,cell=ce,variant='original',recipe='LR_'+roster,position_variant='POS' if pos else 'CONTENT',
                            score_scope='OOF',status='PENDING',n_total=n,n_answers=0,coverage=0,failures=0))
    pd.DataFrame(positions).to_csv(OUT/'POSITION_DIAGNOSTICS.csv',index=False)
    pd.DataFrame(lr).to_csv(OUT/'LR_REFERENCES.csv',index=False)
    pd.DataFrame(coverage).to_csv(OUT/'CHECKPOINT_COVERAGE.csv',index=False)
    return pd.DataFrame(positions),pd.DataFrame(lr),pd.DataFrame(coverage)

def competitors(pop,data):
    from spectral_utils.monday_fusion_metrics import evaluate,top_flags
    overlap=[]; pb=[]; category=[]; ov_summaries=[]; pb_summaries={k:[] for k in ['error_position_quartile','length_bin']}; misses=[]
    n_prm=n_pb=0
    keys=['bank','variant','recipe','decision_rule']
    for bk in BANKS:
        for ce in sorted(set(pop.cells)):
            reference_cache={}
            for variant in VARIANTS:
                with np.load(OUT/'candidate_scores'/(ce+'__'+bk+'__'+variant+'.npz')) as z:
                    ans=z['answer_indices']; off=z['offsets']; rr=pop.rows(pop.cells==ce)
                    for recipe in ['MEAN_MOM','HEM_MOM']:
                        score=z[recipe+'__score']
                        for rule in RULES[:3]:
                            flags=z[recipe+'__'+rule+'__flags']; ident=dict(bank=bk,variant=variant,recipe=recipe,decision_rule=rule)
                            if ce.startswith('prm'):
                                local=[]
                                for j,i in enumerate(ans):
                                    a,b=off[j:j+2]; meta=pop.meta[pop.ids[i]]; gold=pop.labels[pop.off[i]:pop.off[i+1]]
                                    critic=np.asarray(meta['labels'])==0; own=flags[a:b]
                                    overlap.append(dict(**ident,classification=meta['classification'],category=meta['category'],answer=pop.ids[i],source_group=pop.groups[i],
                                        n_steps=int(b-a),n_errors=int(gold.sum()),both=int((gold&own&critic).sum()),only_ours=int((gold&own&~critic).sum()),
                                        only_PRM=int((gold&~own&critic).sum()),neither=int((gold&~own&~critic).sum()),
                                        ours_clean_flag=bool(own.any()) if not gold.any() else None,PRM_clean_flag=bool(critic.any()) if not gold.any() else None))
                                for cls in sorted(set(pop.meta[pop.ids[i]]['classification'] for i in ans)):
                                    choose=np.array([pop.meta[pop.ids[i]]['classification']==cls for i in ans]); selected=ans[choose]
                                    steps=np.flatnonzero(np.repeat(choose,np.diff(off))); suboff=np.r_[0,np.cumsum(pop.ns[selected])]
                                    bank={'answers':selected,'rows':rr[steps],'off':suboff}
                                    cs=np.concatenate([-np.asarray(pop.meta[pop.ids[i]]['rewards'],float) for i in selected])
                                    cf=np.concatenate([np.asarray(pop.meta[pop.ids[i]]['labels'])==0 for i in selected])
                                    for method,s,f in [('ours',score[steps],flags[steps]),('supervised_PRM',cs,cf)]:
                                        cache_key=(variant,recipe,cls,method) if method=='ours' else ('shared','shared',cls,method)
                                        if cache_key not in reference_cache: reference_cache[cache_key]=evaluate(pop,bank,s,f)
                                        for readout in ['NATIVE','TOP_1','TOP_2','TOP_3']:
                                            ff=f if readout=='NATIVE' else top_flags(s,suboff,np.full(len(selected),int(readout[-1])))
                                            m,_=decision_metrics(pop,bank,s,ff,None,reference_cache[cache_key])
                                            category.append(dict(**ident,classification=cls,method=method,readout=readout,top2_post_hoc=readout=='TOP_2',**m))
                            else:
                                for j,i in enumerate(ans):
                                    a,b=off[j:j+2]; target=int(pop.target[i]); ns=int(b-a)
                                    own_peak=canonical_peak(score[a:b]); mtg_peak=int(np.argmax(data.mtg[pop.off[i]:pop.off[i+1]]))
                                    first=int(np.flatnonzero(flags[a:b])[0]) if flags[a:b].any() else -1
                                    pb.append(dict(**ident,cell=ce,answer=pop.ids[i],source_group=pop.groups[i],target=target,n_steps=ns,
                                        length_bin='1-5' if ns<=5 else '6-10' if ns<=10 else '11-20' if ns<=20 else '21+',
                                        error_position_quartile='clean' if target<0 else str(min(3,int(4*target/max(ns-1,1)))),
                                        ours_argmax_hit=own_peak==target if target>=0 else None,ours_decision_hit=first==target,
                                        MTG_native_argmax_hit=mtg_peak==target if target>=0 else None,
                                        ours_miss=('no_flag' if first<0 else 'early' if first<target else 'late' if first>target else 'hit') if target>=0 else None,
                                        MTG_miss=('early' if mtg_peak<target else 'late' if mtg_peak>target else 'hit') if target>=0 else None,
                                        ours_clean_flag=first>=0 if target<0 else None,MTG_native_clean_flag=True if target<0 else None))
                print('Descriptive comparator panel '+bk+'/'+ce,flush=True)
            # Flush one bank/cell at a time: bounded memory while retaining answer audit trails.
            if overlap:
                ov=pd.DataFrame(overlap); ov['cell']=ce
                ov.to_csv(OUT/'PRM_OVERLAP.csv.gz',mode='w' if n_prm==0 else 'a',header=n_prm==0,index=False,compression='gzip')
                n_prm+=len(ov)
                ov_summaries.append(ov.groupby(keys+['cell','classification','category'],dropna=False).agg(n_answers=('answer','size'),n_steps=('n_steps','sum'),n_errors=('n_errors','sum'),
                    both=('both','sum'),only_ours=('only_ours','sum'),only_PRM=('only_PRM','sum'),neither=('neither','sum'),
                    n_clean=('ours_clean_flag','count'),ours_clean_flag_rate=('ours_clean_flag','mean'),PRM_clean_flag_rate=('PRM_clean_flag','mean')).reset_index())
                overlap=[]; del ov
            if pb:
                pp=pd.DataFrame(pb)
                pp.to_csv(OUT/'MTG_FORENSICS.csv.gz',mode='w' if n_pb==0 else 'a',header=n_pb==0,index=False,compression='gzip')
                n_pb+=len(pp)
                for field in pb_summaries:
                    pb_summaries[field].append(pp.groupby(keys+['cell',field],dropna=False).agg(n_answers=('answer','size'),n_erroneous=('ours_argmax_hit','count'),n_clean=('ours_clean_flag','count'),
                        ours_argmax_hit=('ours_argmax_hit','mean'),ours_decision_hit=('ours_decision_hit','mean'),MTG_native_argmax_hit=('MTG_native_argmax_hit','mean'),
                        ours_clean_flag_rate=('ours_clean_flag','mean'),MTG_native_clean_flag_rate=('MTG_native_clean_flag','mean')).reset_index())
                for method,field in [('ours','ours_miss'),('MTG','MTG_miss')]:
                    mm=pp.groupby(keys+['cell',field],dropna=False).size().rename('n_answers').reset_index().rename(columns={field:'miss'})
                    mm['method']=method; misses.append(mm)
                pb=[]; del pp
    # Compress answer-level audit trails, preserving the full population without large flat CSVs.
    pd.DataFrame(category).to_csv(OUT/'PRM_CATEGORY_METRICS.csv',index=False)
    pd.concat(ov_summaries,ignore_index=True).to_csv(OUT/'PRM_OVERLAP_SUMMARY.csv',index=False)
    for field,frames in pb_summaries.items(): pd.concat(frames,ignore_index=True).to_csv(OUT/('MTG_BY_'+field.upper()+'.csv'),index=False)
    pd.concat(misses,ignore_index=True).to_csv(OUT/'MTG_MISS_COUNTS.csv',index=False)
    return {'prm_answer_bank_candidate_rule_rows':n_prm,'pb_answer_bank_candidate_rule_rows':n_pb,
        'scope':SCOPE,'comparator':'cached supervised PRM; our teacher-forced MTG top5-drop mean with argmax/no abstention, NOT the paper first-threshold rule',
        'source_code':str(CODE/'spectral_utils/monday_fusion_competitors.py'),'population_alignment':'Input hashes, frozen offsets and shape; original comparator lacks standalone answer ids'}

def lr_summary(lr):
    """Show incomplete LR rosters without averaging only the completed cells."""
    rows=[]
    for benchmark in ['PRMBench','ProcessBench']:
        expected_cells=1 if benchmark=='PRMBench' else 8
        for roster in ['full','survivors']:
            for pos in ['CONTENT','POS']:
                for scope in ['OOF','IN_SAMPLE_DIAGNOSTIC']:
                    for bk in BANKS+['EQUAL_8_BANK_MEAN']:
                        part=lr[(lr.recipe=='LR_'+roster)&(lr.position_variant==pos)&(lr.score_scope==scope)].copy()
                        part=part[part.cell.str.startswith('prm') if benchmark=='PRMBench' else part.cell.str.startswith('pb_')]
                        if bk!='EQUAL_8_BANK_MEAN': part=part[part.bank==bk]
                        if 'decision_rule' in part: part=part[(part.decision_rule=='ANSWER_Z_Q80')|part.decision_rule.isna()]
                        expected=expected_cells*(8 if bk=='EQUAL_8_BANK_MEAN' else 1)
                        valid=part[part.coverage==1]
                        complete=len(valid)==expected and len(part)==expected
                        unique=part.drop_duplicates('cell')
                        n_total=unique.get('n_total',pd.Series(np.nan,index=unique.index)).fillna(unique.n_answers)
                        row=dict(benchmark=benchmark,bank=bk,recipe='LR_'+roster,position_variant=pos,score_scope=scope,decision_rule='ANSWER_Z_Q80',
                            expected_bank_cells=expected,completed_bank_cells=len(valid),complete=complete,n_answers=int(n_total.sum()),
                            status='COMPLETE' if complete else 'PENDING_OR_FAILED_DO_NOT_RANK',source_file='LR_REFERENCES.csv')
                        for f in FIELDS: row[f]=float(part[f].mean()) if complete and f in part and part[f].notna().any() else np.nan
                        rows.append(row)
    frame=pd.DataFrame(rows); frame.to_csv(OUT/'LR_PANEL.csv',index=False)
    return frame

def render(panel,contrast,positions,lr,coverage,audit):
    def table(frame): return frame.to_html(index=False,escape=True,float_format=lambda v:f'{v:.4f}',na_rep='—',border=0)
    primary=panel[(panel.bank=='EQUAL_8_BANK_MEAN')&panel.decision_rule.isin(RULES[:3])]
    highlight=contrast[(contrast.bank=='EQUAL_8_BANK_MEAN')&(contrast.recipe=='HEM_MOM')&(contrast.comparison=='cell_channels_minus_answer_groups')&contrast.decision_rule.isin(RULES[:3])]
    sections=[]; lrt=lr_summary(lr)
    for benchmark in ['PRMBench','ProcessBench']:
        fields=['within_answer_auc','prmscore','clean_flag_rate','wrong_answer_auroc'] if benchmark=='PRMBench' else ['pb_first_error_accuracy','pb_decision_error_accuracy','pb_official_f1','clean_flag_rate','wrong_answer_auroc']
        t=primary[primary.benchmark==benchmark][['variant','recipe','decision_rule']+fields+['n_answers','n_erroneous','n_clean','complete']]
        sections.append(f'<h2>{benchmark}: ארבעת המועמדים והבקרות</h2><p>מקור: <a href="CONTENT_METRICS.csv">CONTENT_METRICS.csv</a>; ממוצע שווה בין 8 בנקים וב־ProcessBench גם בין 8 תאים.</p>'+table(t))
        lrcols=['bank','recipe','position_variant']+fields+['n_answers','completed_bank_cells','expected_bank_cells','status']
        sections.append('<h3>ייחוס לינארי מפוקח — מחוץ לאימון (OOF)</h3><p>מלא ושורדים; עם ובלי POSZ. מקור: <a href="LR_REFERENCES.csv">LR_REFERENCES.csv</a>. מוצג q80; בקרות top-k נמצאות בקובץ. אין ציוני R2 או שער אנטרופיה עבור LR בריצה הזאת. ציוני תשובות מקיפולים שונים אינם מאוחדים ל־AUC גלובלי.</p>'+table(lrt[(lrt.benchmark==benchmark)&(lrt.score_scope=='OOF')][lrcols]))
        sections.append('<details><summary>ביצועי LR בתוך האימון — אבחון נפרד</summary><p>אלה אינם ציוני ההערכה של הייחוס המפוקח ואינם תקרה מתמטית.</p>'+table(lrt[(lrt.benchmark==benchmark)&(lrt.score_scope=='IN_SAMPLE_DIAGNOSTIC')][lrcols])+'</details>')
    failures=positions[positions.failures.fillna(0)>0]
    failcols=[c for c in ['bank','cell','variant','recipe','position_variant','decision_rule','n_total','failures','failure_reason'] if c in failures]
    files=['CANDIDATE_PANEL.csv','CONTENT_METRICS.csv','PAIRED_CONTRASTS.csv','POSITION_DIAGNOSTICS.csv','LR_REFERENCES.csv','LR_PANEL.csv','CHECKPOINT_COVERAGE.csv','PRM_OVERLAP_SUMMARY.csv','PRM_CATEGORY_METRICS.csv','MTG_BY_ERROR_POSITION_QUARTILE.csv','MTG_BY_LENGTH_BIN.csv','MTG_MISS_COUNTS.csv','AUDIT.json']
    links=' · '.join(f'<a href="{name}">{name}</a>' for name in files if (OUT/name).exists())
    text='''<!doctype html><html lang="he" dir="rtl"><meta charset="utf-8"><title>השוואת מועמדים לפני בחירה</title><style>
body{font:17px Arial,sans-serif;max-width:1600px;margin:32px auto;padding:20px;background:#fff;color:#17212e}p{line-height:1.65}table{direction:ltr;border-collapse:collapse;font-size:13px;width:100%;margin:18px 0}th,td{padding:7px;border:1px solid #91a0ad}th{background:#e8eef5;color:#17212e}a{color:#185ea5}.note{background:#fff1c7;color:#352700;padding:16px}details{margin:20px 0}img{max-width:100%;background:white} @media(prefers-color-scheme:dark){body{background:#17212e;color:#e7eef5}a{color:#8fc8ff}td{border-color:#8a98a4}}
</style><h1>ארבעה מועמדים, בלי בחירה אוטומטית</h1><p class="note">השוואת פיתוח לפני בחירה; אין מתכון נעול ואין כאן אימות על נתונים שלא נחשפו. עמרי בוחר את המתכון ואת כלל ההחלטה. לא בוצע אימון נוסף של שיטות האיחוד.</p>'''
    text+=f'<p>עודכן: {html.escape(audit["time_utc"])}. ארבעת המועמדים משתמשים בבנקים המקוריים, ללא הסרת LEVEL. מקור הטבלאות וה־N מצוין בכל קובץ. LR הוא ייחוס לינארי מפוקח, לא תקרה מתמטית.</p>'
    text+='<p>groups_answer: אומדן מהצבעות הקבוצות עם מכסה לכל תשובה. channels_cell: אומדן מהצבעות הערוצים עם מכסה לכל התא. MEAN ו־HEM מתארים את המשקול בתוך קבוצה; MoM את המשקול בין הקבוצות.</p>'
    text+='<p>q80 מסמן צעדים שחוצים סף לאחר תקנון בתוך תשובה; ההחלטה היא הסימון הראשון. R2 קובע מספר סימונים לכל תשובה מן הבנק הגולמי, ואז מסמן את הצעדים המדורגים גבוה ביותר. שער האנטרופיה סוגר תשובות לפי ממוצע האנטרופיה; בתוך תשובה פתוחה נשמר q80. איתור לפי מקסימום מדווח בנפרד מהחלטת הסימון הראשון.</p>'
    text+='<h2>מה משתנה ב־HEM+MoM כאשר מחליפים את קלט האומדנים</h2><p>הפרש channels_cell פחות groups_answer. רווחי 95% מזווגים לפי שאלת מקור, חקרניים וללא תיקון ריבוי השוואות. אין להפוך אותם בדיעבד לבדיקה מאשרת. מקור: PAIRED_CONTRASTS.csv.</p>'
    text+=table(highlight[['benchmark','decision_rule','endpoint','delta','lo95','hi95','n_answers','n_source_groups']])
    if (OUT/'figures/HEM_MOM_DECISION.svg').exists(): text+='<img src="figures/HEM_MOM_DECISION.svg" alt="HEM MOM decision differences with paired intervals">'
    text+=''.join(sections)
    top=panel[(panel.bank=='EQUAL_8_BANK_MEAN')&panel.decision_rule.str.startswith('TOP_')]
    text+='<details><summary>בקרות top-1 / top-2 / top-3 לכל מועמד ולכל בקרה</summary><p>בחירת top-2 ההיסטורית נעשתה בדיעבד. ב־ProcessBench סימון קבוע של לפחות צעד אחד מסמן גם כל תשובה תקינה, ולכן ה־F1 הרשמי הוא אפס.</p>'+table(top[['benchmark','variant','recipe','decision_rule']+FIELDS+['n_answers']])+'</details>'
    text+='<h2>כיסוי המיקום ו־LR</h2><p>תאים חסרים נשארים במכנה המתוכנן; אין ממוצע מנצח על תת־אוכלוסייה. PENDING אינו כשל באמידה. וריאציות המיקום הן אבחון נפרד; PRIOR_CF עשוי לשנות גם את ציון התוכן.</p>'+table(coverage.groupby(['stage','status'],dropna=False).size().rename('checkpoints').reset_index())
    text+='<p><a href="POSITION_DIAGNOSTICS.csv">כל אבחוני המיקום</a> · <a href="LR_REFERENCES.csv">כל ייחוסי LR: מלא/שורדים, עם/בלי מיקום, OOF בנפרד מאימון</a></p>'
    if len(failures): text+='<details><summary>כשלים מפורשים בשכבת המיקום</summary>'+table(failures[failcols])+'</details>'
    text+='<h2>השוואת מתחרים תיאורית לפני בחירה</h2><p>קובצי PRM מפרטים קטגוריות וחפיפת צעדי שגיאה: שניהם, רק שלנו, רק PRM, אף אחד; גם הבקרות top-k קיימות לשתי השיטות. קובצי MTG מפרטים אורך, מיקום והחמצה מוקדמת/מאוחרת/ללא דגל. ההשוואה היא לשחזור שלנו עם argmax וללא הימנעות: זו אינה החלטת החצייה הראשונה המתוארת במאמר Mind the Gap. ציון ה־PRM נשמר ברמת השגיאות המלאות; ProcessBench אינו מספק אותן.</p>'
    if audit.get('competitors'): text+='<p>'+html.escape(json.dumps(audit['competitors'],ensure_ascii=False))+'</p>'
    text+='<h2>בדיקות ומקורות</h2><p>'+html.escape(json.dumps(audit.get('checks',{}),ensure_ascii=False))+'</p><p>'+links+'</p></html>'
    (OUT/'REPORT_HE.html').write_text(text,encoding='utf8')

def figures(contrast):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory=OUT/'figures'; directory.mkdir(exist_ok=True)
    selected=contrast[(contrast.bank=='EQUAL_8_BANK_MEAN')&(contrast.recipe=='HEM_MOM')&(contrast.comparison=='cell_channels_minus_answer_groups')&contrast.decision_rule.isin(RULES[:3])]
    fig,axes=plt.subplots(1,3,figsize=(14,4),facecolor='white')
    for ax,endpoint,title in zip(axes,['pb_first_error_accuracy','pb_official_f1','clean_flag_rate'],['ProcessBench argmax accuracy','ProcessBench official F1','ProcessBench clean flags (lower better)']):
        t=selected[(selected.benchmark=='ProcessBench')&(selected.endpoint==endpoint)].set_index('decision_rule').loc[RULES[:3]]
        ax.errorbar(t.delta*100,np.arange(3),xerr=np.array([t.delta-t.lo95,t.hi95-t.delta])*100,fmt='o',capsize=4,color='#7628a0')
        ax.axvline(0,color='black',lw=1); ax.set_yticks(np.arange(3),['q80','R2 count','Entropy gate']); ax.set_title(title); ax.set_xlabel('Change in percentage points; paired 95% CI'); ax.grid(axis='x',alpha=.3)
    fig.suptitle('HEM + MoM: channel/cell input minus group/answer input | exploratory, unadjusted')
    fig.tight_layout(); fig.savefig(directory/'HEM_MOM_DECISION.svg',facecolor='white'); fig.savefig(directory/'HEM_MOM_DECISION.png',dpi=150,facecolor='white'); plt.close(fig)

def main():
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('--smoke',action='store_true'); parser.add_argument('--refresh',action='store_true'); parser.add_argument('--draws',type=int,default=5000)
    args=parser.parse_args(); OUT.mkdir(parents=True,exist_ok=True)
    evidence=Evidence(); pop=setup()
    audit={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':SCOPE,'status':'BUILDING','selection':'NONE; Omri chooses candidate and rule','draws':args.draws}
    for path in [Path(__file__),ROOT/'scripts/build_monday_candidate_panel.py',CODE/'spectral_utils/monday_fusion_metrics.py',CODE/'spectral_utils/monday_fusion_decision.py',CODE/'scripts/experiments/ds_group_weights.py']:
        evidence.use(path)
    if args.refresh:
        old=json.loads((OUT/'AUDIT.json').read_text(encoding='utf8'))
        own_code={str(Path(__file__)),str(ROOT/'scripts/build_monday_candidate_panel.py')}
        transitions=[]
        for path,digest in old['input_sha256'].items():
            if path in own_code:
                if sha(path)!=digest: transitions.append(dict(path=path,old_sha256=digest,new_sha256=sha(path)))
            else: evidence.use(path,digest)
        audit.update({k:v for k,v in old.items() if k not in ['time_utc','input_sha256']})
        audit.setdefault('content_generation_code_sha256',{p:h for p,h in old['input_sha256'].items() if p in own_code})
        audit['refresh_code_transitions']=transitions
        frame=pd.read_csv(OUT/'CONTENT_METRICS.csv'); contrast=pd.read_csv(OUT/'PAIRED_CONTRASTS.csv')
    else:
        data=Inputs(pop,evidence); frame,gate=content(pop,data,evidence,args.smoke)
        audit['checks']={'sealed_content_replay':gate,'claude_precompute':check_claude(frame,evidence)}
        if args.smoke:
            dump(OUT/'SMOKE.json',dict(audit,status='FEASIBILITY_ONLY_NO_QUALITY_CLAIM')); print('SMOKE PASS',flush=True); return
        contrast=bootstrap(pop,args.draws); audit['competitors']=competitors(pop,data); del data; gc.collect()
    panel=aggregates(frame); positions,lr,coverage=checkpoint_panels(pop,evidence)
    audit['position_checkpoints']=coverage[coverage.stage=='position'].status.value_counts().to_dict()
    audit['lr_checkpoints']=coverage[coverage.stage=='LR'].status.value_counts().to_dict()
    audit['status']='CONTENT_AND_PRESELECTION_FORENSICS_COMPLETE_LAYER_LR_SNAPSHOT'
    audit['time_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    for module in list(sys.modules.values()):
        value=getattr(module,'__file__',None)
        if value:
            path=Path(value)
            if path.suffix=='.py' and path.is_file() and path.is_relative_to(ROOT): evidence.use(path)
    evidence.verify(); audit['input_sha256']=evidence.used
    audit['checks']['input_files_unchanged']=len(evidence.used)
    figures(contrast); dump(OUT/'AUDIT.json',audit); render(panel,contrast,positions,lr,coverage,audit)
    # Mutable report, versioned audit snapshots; sealed experiment files are never edited.
    snapshots=OUT/'snapshots'; snapshots.mkdir(exist_ok=True)
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')
    dump(snapshots/(stamp+'_AUDIT.json'),audit)
    print('CANDIDATE PANEL READY '+str(OUT/'REPORT_HE.html'),flush=True)
