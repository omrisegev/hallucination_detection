"""Independent arithmetic/identity verification; not the production scorer.

Run with the repository .venv. All inputs are read-only. Verification results
alone are written to this review directory. Pairwise AUC is computed directly,
rather than using the production rank-sum helper.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import csv, hashlib, json, pickle, sys
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PKG = Path(__file__).resolve().parents[1]
REPO = PKG.parents[1]
sys.path.insert(0, str(REPO))


def table(path):
    return list(csv.DictReader(path.open()))


def pair_auc(y, score):
    e, v = score[y], score[~y]
    if not len(e) or not len(v):
        return np.nan
    diff = e[:, None] - v[None, :]
    return (np.sum(diff > 0) + .5 * np.sum(diff == 0)) / diff.size


def prmscore(y, flag, ms=False):
    tp, fp, fn, tn = [int(q.sum()) for q in
                      (y & flag, ~y & flag, y & ~flag, ~y & ~flag)]
    ef = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
    vf = 2*tn/(2*tn+fp+fn) if 2*tn+fp+fn else 0.
    return (vf if ms else .5*(ef+vf)), [tp, fp, fn, tn]


def zlocal(x, off):
    z = np.zeros_like(x, dtype=float)
    for a, b in zip(off[:-1], off[1:]):
        sd = np.std(x[a:b], axis=0)
        z[a:b] = np.divide(x[a:b]-np.mean(x[a:b], axis=0), sd,
                          out=np.zeros_like(x[a:b], dtype=float), where=sd > 1e-12)
    return z


def hashfile(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def run():
    checks = []
    def check(name, ok, **detail):
        checks.append(dict(check=name, status='PASS' if ok else 'FAIL', **detail))
        if not ok:
            raise AssertionError((name, detail))
    I = PKG/'inputs'
    p = np.load(I/'population.npz')
    ids, off, y, cls, groups, fold = (p[k] for k in
                                    ('idx','offsets','y','classification','group','fold'))
    ai = np.repeat(np.arange(len(ids)), np.diff(off))
    meta = {r['idx']: r for r in json.loads((I/'prmbench_meta.json').read_text())['rows']}
    truth = np.concatenate([np.isin(np.arange(1,b-a+1), meta[u]['error_steps'])
                            for u,a,b in zip(ids, off[:-1], off[1:])])
    check('canonical_labels_and_identity', len(ids)==6969 and off[-1]==94203
          and len(np.unique(groups))==707 and np.array_equal(truth,y))
    check('canonical_population_counts', sum(cls=='correct')==758 and
          sum(cls=='multi_solutions')==160 and sum(cls!='correct')==6211 and
          sum(cls[ai]!='correct')==83371,
          counts={c:int(sum(cls==c)) for c in np.unique(cls)})
    mapping = {r['row_id']:r['canonical_group_id'] for r in
               json.loads((I/'CANONICAL_GROUPS.json').read_text())['rows']}
    splits = json.loads((I/'FOLDS_V2.json').read_text())['prmbench']
    check('canonical_group_fold_mapping', all(mapping[u]==g and splits['outer'][g]==int(f)
                                               for u,g,f in zip(ids,groups,fold)))
    check('fold_group_disjointness', all(not set(groups[fold==k]) & set(groups[fold!=k])
                                        for k in range(5)))
    check('true_error_prevalence', abs(float(y.mean())-.13958154198910863)<1e-15,
          prevalence=float(y.mean()))
    bank = np.load(I/'bank_N16.npz')
    prm = pickle.load((I/'prmbench_prm.pkl').open('rb'))
    prm = list(prm.values()) if isinstance(prm,dict) else prm
    prm = {str(r['idx']):r for r in prm}
    check('PRM_identity', set(prm)==set(ids) and all(
        prm[u]['classification']==meta[u]['classification'] and
        prm[u]['error_steps']==meta[u]['error_steps'] and
        prm[u]['n_steps']==int(b-a) for u,a,b in zip(ids,off[:-1],off[1:])))
    risk = 1-np.concatenate([np.asarray(prm[u]['rewards'],float) for u in ids])
    scores = {m:bank['S__'+m] for m in ['LSML_CONT','IU_PCR','ALL_EQ','DS_FILTER_EQ']}
    scores['PRM_NATIVE'] = risk
    expected_auc = {'LSML_CONT':.779193338180572,'IU_PCR':.7804590822159837}
    fullrows = table(PKG/'controls/ALL_BANK_TOP1_AUC.csv')
    for method, score in scores.items():
        aa=np.array([pair_auc(y[a:b],score[a:b]) for a,b in zip(off[:-1],off[1:])])
        tt=np.array([int(y[a+np.argmax(score[a:b])]) for a,b in zip(off[:-1],off[1:])])
        error=np.add.reduceat(y.astype(int),off[:-1])>0
        auc=float(np.nanmean(aa)); top=float(tt[error].mean())
        row=next(r for r in fullrows if r['method']==method and r['classification']=='ALL_NONCONTROL'
                 and r['bank']==('REFERENCE' if method=='PRM_NATIVE' else 'N16'))
        check('direct_pair_AUC_top1_'+method, abs(auc-float(row['within_auc']))<1e-12
              and abs(top-float(row['top1_on_error']))<1e-12,
              auc=auc,top1=top,n_auc=int(np.isfinite(aa).sum()),n_top1=int(error.sum()))
    flags=np.load(PKG/'controls/DECISION_FLAGS.npz')
    rules=table(PKG/'controls/EXACT_DECISION_RULES.csv')
    bv=np.load(I/'bank_V_N16.npz');names=list(bv['columns'].astype(str))
    survivors=list(bank['G_survivors'].astype(str))
    raw=bv['raw'][:,[names.index(c) for c in survivors]]
    raw_z=(raw-raw.mean(0))/np.maximum(raw.std(0),1e-12)
    G=raw_z.mean(1)
    check('raw_survivor_G_parity', np.max(np.abs(G-bank['G']))<1e-9,
          max_abs=float(np.max(np.abs(G-bank['G']))))
    rawquota=np.add.reduceat((G>=np.quantile(G,.8)).astype(int),off[:-1])
    nuisance=np.load(PKG/'controls/NUISANCE_INPUTS.npz')
    entropy=nuisance['answer_entropy']
    for method,score in scores.items():
        if method=='PRM_NATIVE':
            calculated={'NATIVE_0_5':risk>.5}
        else:
            az=zlocal(score,off);base=az>=np.quantile(az,.8)
            r2=np.zeros(len(y),bool)
            for a,b,k in zip(off[:-1],off[1:],rawquota):
                r2[a+np.argsort(-score[a:b],kind='stable')[:k]]=True
            e=(entropy-entropy.mean())/max(entropy.std(),1e-12)
            shifted=az+e[ai];sf=shifted>=np.quantile(shifted,.8)
            nclose=int(np.ceil(np.mean(np.add.reduceat(sf.astype(int),off[:-1])==0)*len(ids)))
            order=np.lexsort((ids,entropy));gate=np.ones(len(ids),bool);gate[order[:nclose]]=False
            calculated={'ANSWER_Z_Q80':base,'R2_COUNT':r2,'ENTROPY_GATE_Q80':base&gate[ai]}
        for rule,flag in calculated.items():
            check('exact_flags_'+method+'_'+rule,np.array_equal(flag,flags[method+'__'+rule]))
            for typ in ['ALL_NONCONTROL']+list(np.unique(cls)):
                selection=(cls!='correct' if typ=='ALL_NONCONTROL' else cls==typ)[ai]
                value,confusion=prmscore(y[selection],flag[selection],typ=='multi_solutions')
                row=next(r for r in rules if r['method']==method and r['rule']==rule
                         and r['classification']==typ)
                check('confusion_'+method+'_'+rule+'_'+typ,
                      abs(value-float(row['category_score']))<1e-12,
                      score=value,confusion=confusion,n_steps=int(selection.sum()))
            ss=[float(next(r['category_score'] for r in rules if r['method']==method
                          and r['rule']==rule and r['classification']==c))
                for c in ('missing_condition','deception','multi_solutions')]
            sr=float(next(r['category_score'] for r in rules if r['method']==method
                          and r['rule']==rule and r['classification']=='sensitivity'))
            check('PS_DR_MS_sensitivity_'+method+'_'+rule,abs(np.mean(ss)-sr)<1e-12)
    controls=table(PKG/'controls/NUISANCE_CONTROLS.csv')
    bins=np.minimum((5*np.concatenate([(np.arange(b-a)+.5)/(b-a)
                     for a,b in zip(off[:-1],off[1:])])).astype(int),4)
    cuts=np.quantile(nuisance['length'],[1/3,2/3])
    lb=np.searchsorted(cuts,nuisance['length'],side='right');strata=3*bins+lb
    for method,score in scores.items():
        aa=np.full(len(ids),np.nan);paircount=np.zeros(len(ids),int)
        for i,(a,b) in enumerate(zip(off[:-1],off[1:])):
            wins=0.;npair=0
            for h in np.unique(strata[a:b]):
                use=strata[a:b]==h; yy=y[a:b][use];s=score[a:b][use]
                den=int(yy.sum()*(len(yy)-yy.sum()))
                if den:
                    wins+=pair_auc(yy,s)*den;npair+=den
            if npair:aa[i]=wins/npair;paircount[i]=npair
        for typ in ['ALL_NONCONTROL','redundency','circular','counterfactual','step_contradiction']:
            use=(cls!='correct' if typ=='ALL_NONCONTROL' else cls==typ)&np.isfinite(aa)
            row=next(r for r in controls if r['method']==method and
                     r['control']=='position_length' and r['classification']==typ)
            check('matched_position_length_'+method+'_'+typ,
                  abs(float(aa[use].mean())-float(row['within_auc']))<1e-12
                  and int(use.sum())==int(row['n_auc_answers']),
                  auc=float(aa[use].mean()),n_answers=int(use.sum()),n_pairs=int(paircount[use].sum()))
    q=np.load(PKG/'statistics/MOM_POSTERIOR_QUOTA.npz')
    estimate=json.loads((PKG/'statistics/ESTIMATORS.json').read_text())['groups/MOM']['estimate']
    votes=np.load(PKG/'statistics/groups_VOTE_CONDITIONAL.npz')['votes']>0
    pi=estimate['prevalence'];psi=np.asarray(estimate['psi']);eta=np.asarray(estimate['eta'])
    # Direct Bayes product in log-space, independent of the odds implementation.
    loge=np.log(pi)+np.sum(np.where(votes,np.log(psi),np.log1p(-psi)),axis=1)
    logv=np.log1p(-pi)+np.sum(np.where(votes,np.log1p(-eta),np.log(eta)),axis=1)
    posterior=np.exp(loge-np.logaddexp(loge,logv))
    count=np.floor(np.add.reduceat(posterior,off[:-1])+.5).astype(int)
    check('MoM_posterior_Bayes_and_quota',np.max(np.abs(q['posterior']-posterior))<1e-12
          and np.array_equal(q['count'],count) and np.all(count<=np.diff(off)),
          posterior_max_abs=float(np.max(np.abs(q['posterior']-posterior))))
    half=np.load(PKG/'validation/SPLIT.npz');donor=half['donor']
    dg=set(groups[donor]);tg=set(groups[~donor]);check('half_group_disjoint_353_354',
        not dg&tg and len(dg)==353 and len(tg)==354)
    for path in (PKG/'validation/checkpoints').glob('LOTO_*.json'):
        st=json.loads(path.read_text());typ=st['type']
        for f in st['fits']:
            k=f['fold'];tr=(fold!=k)&(cls!=typ);te=(fold==k)&(cls==typ)
            check('LOTO_disjoint_'+typ+'_'+st['method']+'_'+str(k),
                  not set(groups[tr])&set(groups[te]) and not (cls[tr]==typ).any())
    # Refit one full outer LR without calling the project fit helper. Selected
    # C was obtained only from its inner folds; this checks saved affine scores.
    path=PKG/'lr/checkpoints/N16/outer_0/MODELS.json'
    models=json.loads(path.read_text());X=bv['V'];sf=fold[ai];tr=sf!=0;te=sf==0
    spec=models['arms']['LR_FULL'];C=spec['C']
    scaler=StandardScaler().fit(X[tr])
    fit=LogisticRegression(C=C,class_weight='balanced',solver='lbfgs',
                           max_iter=3000,random_state=20261003).fit(scaler.transform(X[tr]),y[tr])
    z=np.load(path.with_name('SCORES.npz'))
    prediction=fit.decision_function(scaler.transform(X[te]))
    check('independent_outer0_N16_LR_refit',np.array_equal(z['rows'],np.flatnonzero(te))
          and np.max(np.abs(z['LR_FULL']-prediction))<1e-9,
          C=C,max_abs=float(np.max(np.abs(z['LR_FULL']-prediction))))
    for path in (PKG/'lr/checkpoints').glob('*/*/MODELS.json'):
        m=json.loads(path.read_text());bankname=m['bank'];outer=m['outer_fold']
        check('LR_saved_group_partitions_'+bankname+'_'+str(outer),
              not set(m['train_groups'])&set(m['test_groups']) and
              set(m['train_groups'])==set(groups[fold!=outer]) and
              set(m['test_groups'])==set(groups[fold==outer]))
        iv=[set(v) for v in m['inner_valid_groups'].values()]
        check('LR_inner_disjoint_'+bankname+'_'+str(outer),
              all(not a&b for i,a in enumerate(iv) for b in iv[i+1:]) and
              set.union(*iv)==set(m['train_groups']))
        check('LR_affine_precision_'+bankname+'_'+str(outer),all(
            a.get('model',{}).get('diagnostic',{}).get('affine_max_abs',0)<1e-9
            for a in m['arms'].values()))
    receipt=dict(status='PASS',n_checks=len(checks),checks=checks,
                 scope='Independent direct pair/contingency/decision/control/Bayes replay; one LR fit; all saved group partitions',
                 not_rerun=['Every nested LR C/selector fit','all raw entropy values','full final HTML pending'])
    (PKG/'review/ARITHMETIC_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(status=receipt['status'],n_checks=len(checks)),indent=2))


if __name__=='__main__':
    run()
