"""Internal source-disjoint validation on previously inspected development data."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import csv,json,sys,time
import numpy as np
OUT=Path(__file__).resolve().parents[1];ROOT=OUT.parents[1];sys.path.insert(0,str(ROOT))
from spectral_utils.advisor_lr import fit_labelfree,as_dict
from spectral_utils.advisor_statistics import answer_auc
from prepare_population import answer_std
BANKS=['N11','N13','N22','N40','N53','N14','N16','N25','N43','N56'];METHODS=['LSML_CONT','IU_PCR']
def write(p,x):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def csvwrite(p,rows):
    if not rows:return
    with p.open('w',newline='') as f:w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def evaluate(y,s,off,am):
    auc,_=answer_auc(y,s,off);top=np.full(len(am),np.nan)
    for i,(a,b) in enumerate(zip(off[:-1],off[1:])):
        if am[i] and y[a:b].any() and np.isfinite(s[a:b]).all():top[i]=float(y[a+np.flatnonzero(s[a:b]>=s[a:b].max()-8*np.finfo(float).eps)[0]])
    good=am&np.isfinite(auc);tg=am&np.isfinite(top)
    return {'within_answer_auc':float(auc[good].mean()) if good.any() else None,'n_auc':int(good.sum()),'top1_on_error':float(top[tg].mean()) if tg.any() else None,'n_top1':int(tg.sum())},auc,top
def main():
    t=time.time();dest=OUT/'validation';dest.mkdir(exist_ok=True);cp=dest/'checkpoints';cp.mkdir(exist_ok=True);I=OUT/'inputs';p=np.load(I/'population.npz');y=p['y'];off=p['offsets'];n=np.diff(off);ai=np.repeat(np.arange(len(n)),n);groups=p['group'];fold=p['fold'];cls=p['classification'];ug=np.unique(groups)
    shuffled=np.random.default_rng(20261011).permutation(ug);donor_groups=set(shuffled[:len(ug)//2]);donor=np.array([g in donor_groups for g in groups]);test=~donor;assert not set(groups[donor])&set(groups[test]);np.savez_compressed(dest/'SPLIT.npz',idx=p['idx'],donor=donor,fold=fold,group=groups)
    inputs={bk:np.load(I/f'bank_V_{bk}.npz')['V'] for bk in BANKS};candidate=[]
    for bk in BANKS:
        X=inputs[bk]
        for method in METHODS:
            key=bk+'_'+method;path=cp/(key+'.json');scorepath=cp/(key+'.npz')
            if path.exists() and scorepath.exists():state=json.loads(path.read_text());scores=np.load(scorepath)['scores']
            else:
                scores=np.full(len(y),np.nan);failures=[];fits=[]
                for f in range(5):
                    tr=donor&(fold!=f);va=donor&(fold==f);assert not set(groups[tr])&set(groups[va]);ts=tr[ai];vs=va[ai]
                    try:
                        model=fit_labelfree(X[ts],method);scores[vs]=model.predict(X[vs]);fits.append({'fold':f,'n_train_groups':len(set(groups[tr])),'n_valid_groups':len(set(groups[va])),'model':as_dict(model)})
                    except Exception as exc:failures.append({'fold':f,'reason':repr(exc)})
                result,_,_=evaluate(y,scores,off,donor);state={'bank':bk,'method':method,'metrics':result,'failures':failures,'fits':fits,'status':'PASS' if not failures else 'FAILED'};np.savez_compressed(scorepath,scores=scores);write(path,state)
            candidate.append({'bank':bk,'method':method,'status':state['status'],**state['metrics']});print('donor',key,state['metrics'],flush=True)
    eligible=[r for r in candidate if r['status']=='PASS'];assert eligible,'No full-coverage donor candidate';winner=max(eligible,key=lambda r:r['within_answer_auc'])
    tests=[];testscore={}
    for bk,method in dict.fromkeys([(winner['bank'],winner['method']),('N16','LSML_CONT'),('N16','IU_PCR')]):
        X=inputs[bk];model=fit_labelfree(X[donor[ai]],method);s=np.full(len(y),np.nan);s[test[ai]]=model.predict(X[test[ai]]);r,aa,top=evaluate(y,s,off,test);tests.append({'bank':bk,'method':method,'role':'donor_selected' if (bk,method)==(winner['bank'],winner['method']) else 'fixed_N16','status':'PASS',**r});testscore[bk+'_'+method]=s;write(cp/('TEST_'+bk+'_'+method+'.json'),as_dict(model))
    csvwrite(dest/'HALF_DONOR_CANDIDATES.csv',candidate);csvwrite(dest/'HALF_TEST.csv',tests);np.savez_compressed(dest/'HALF_TEST_SCORES.npz',idx=p['idx'],**testscore)
    summary={'status':'COMPLETE','design':'353donor/354test canonicalgroups; donor5foldcrossfit bank/method selection by meanwithinanswerAUC; fixedtieorder; selectedrefitdonoronly','seed':20261011,'n_donor_groups':len(donor_groups),'n_test_groups':len(ug)-len(donor_groups),'n_donor_answers':int(donor.sum()),'n_test_answers':int(test.sum()),'winner':winner,'test':tests,'source_overlap':0,'development_only':True}
    write(dest/'HALF_SUMMARY.json',summary)
    # Locked N16 transfer across error types: never fit on the target source fold or target type.
    X=inputs['N16'];rows=[];lotoscores={}
    for typ in sorted(set(cls)-{'correct','multi_solutions'}):
        for method in METHODS:
            key=typ+'_'+method;path=cp/('LOTO_'+key+'.json');scorepath=cp/('LOTO_'+key+'.npz')
            if path.exists() and scorepath.exists():st=json.loads(path.read_text());s=np.load(scorepath)['score']
            else:
                s=np.full(len(y),np.nan);fails=[];fits=[]
                for f in range(5):
                    tr=(fold!=f)&(cls!=typ);te=(fold==f)&(cls==typ);assert not set(groups[tr])&set(groups[te]);assert not (cls[tr]==typ).any()
                    try:
                        model=fit_labelfree(X[tr[ai]],method);s[te[ai]]=model.predict(X[te[ai]]);fits.append({'fold':f,'n_train_answers':int(tr.sum()),'n_test_answers':int(te.sum()),'model':as_dict(model)})
                    except Exception as exc:fails.append({'fold':f,'reason':repr(exc)})
                metrics,_,_=evaluate(y,s,off,cls==typ);st={'type':typ,'method':method,'bank':'N16','status':'PASS' if not fails else 'FAILED','failures':fails,'fits':fits,**metrics};write(path,st);np.savez_compressed(scorepath,score=s)
            rows.append({k:v for k,v in st.items() if k not in ['fits','failures']});lotoscores[key]=s;print('LOTO',key,st['within_answer_auc'],flush=True)
    csvwrite(dest/'LEAVE_ONE_TYPE_OUT.csv',rows);np.savez_compressed(dest/'LEAVE_ONE_TYPE_OUT_SCORES.npz',idx=p['idx'],**lotoscores)
    write(dest/'STATUS.json',{'status':'COMPLETE','half':summary,'loto_status':'COMPLETE','loto_design':'LockedN16;5sourcefolds; trainexcludesentiretestfold andheldtype; labelsforreportonly; no type-winnerselection','n_loto_rows':len(rows),'runtime_s':time.time()-t,'limits':['Previouslyinspecteddevelopmentdata; notuntouchedconfirmation','Answerlocalstandardizationofeachtestanswerislabel-free; modelweightsandglobalstatsdonoronly','LOTOlocksN16; halfexperimentselectsbankandmethod','UndefinedarmsremainFAILEDwithcoverage; nohiddenfallback']})
    print('VALIDATION_COMPLETE',time.time()-t,flush=True)
if __name__=='__main__':main()
