"""Full saved-vector readout audit and explicit matched choice comparisons."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-root',type=Path,required=True);args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];out=root/'results/rbm_logit_readout_v1'
    bench=args.source_root/'results/localization_full_benchmark_v3/evaluation'
    records=json.loads((bench/'JOINED.json').read_text())['records']
    with np.load(bench/'JOINED.npz') as z:offsets,target=z['offsets'],z['target']
    with np.load(out/'SCORES.npz') as z:data={k:z[k] for k in z.files}
    result=json.loads((out/'METRICS.json').read_text());comparisons={};vectors=0
    for bank in (6,12):
        for state in ('rbm','initial'):
            name=f'{state}{bank}'
            for prefix in ('','logit_'):
                old=data['steps__'+name+'__'+prefix+'old'];new=data['steps__'+name+'__'+prefix+'near']
                for i in range(len(records)):
                    sl=slice(offsets[i],offsets[i+1]);s=old[sl];v=s.copy()
                    if not np.isfinite(s).all():continue
                    maximum=float(max(s));sd=float(np.std(s));near=[j for j,t in enumerate(s) if t>=maximum-.25*sd]
                    for rank,j in enumerate(near):v[j]=maximum+1e-6*max(sd,1e-12)*(len(near)-rank)
                    np.testing.assert_array_equal(v,new[sl]);vectors+=1
        for prefix in ('old','near','logit_old','logit_near'):
            trained='rbm'+str(bank)+'__'+prefix;initial='initial'+str(bank)+'__'+prefix
            a=data['prediction__'+trained];b=data['prediction__'+initial]
            pb=np.array([r['cell'].startswith('pb_') for r in records]);error=pb&(target>=0)
            ahit=data['valid__'+trained]&(a==target);bhit=data['valid__'+initial]&(b==target)
            comparisons[f'bank{bank}_learning_{prefix}']=dict(
                gained=int(np.sum(error&ahit&~bhit)),lost=int(np.sum(error&~ahit&bhit)),
                changed_pb_decisions=int(np.sum(pb&(a!=b))))
    labels={'rbm6':'Trained RBM, 6 features','rbm12':'Trained RBM, 12 features',
        'initial6':'Initial RBM, 6 features','initial12':'Initial RBM, 12 features',
        'entropy':'Entropy','var15':'Varentropy, top 15','var50':'Varentropy, top 50',
        'var15_iu':'Varentropy contributions + IU-PCR','var15_equal':'Varentropy contributions + equal weights',
        'shrinkage':'RBM with shrinkage','diagonal':'RBM with shared diagonal variance','length':'Step length control','random':'Random control'}
    rows=[]
    for key,m in result['metrics'].items():
        name,suffix=key.split('__');rows.append(dict(method=key,method_label=labels.get(name,name),
            token_score='Logit' if 'logit' in suffix else ('Posterior' if name.startswith(('rbm','initial')) else 'Original score'),
            step_readout='First near maximum' if suffix.endswith('near') else 'Original maximum',
            **{k:m[k] for k in ('pb_all8','pb_q4','pb_q8','prm_within','prm_pooled','prmscore_q08','valid_answers','prm_within_n')}))
    with (out/'COMPARISON.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    mechanisms={}
    with (out/'ANSWER_DIAGNOSTICS.csv').open(newline='',encoding='utf8') as f:
        for row in csv.DictReader(f):
            if not row['cell'].startswith('pb_'):continue
            root=row['method'];m=mechanisms.setdefault(root,dict(pb_answers=0,original_max_changed=0,
                rescues=0,rescues_same_original_max=0,rescues_same_max_smaller_near_set=0,
                posterior_exact_step_max_ties=0,logit_exact_step_max_ties=0))
            m['pb_answers']+=1;m['original_max_changed']+=int(row['old_peak']!=row['logit_old_peak'])
            m['posterior_exact_step_max_ties']+=int(int(row.get('old_exact_max_count') or 0)>1)
            m['logit_exact_step_max_ties']+=int(int(row.get('logit_old_exact_max_count') or 0)>1)
            if row['rescued_by_logit_near']=='True':
                m['rescues']+=1;same=row['old_peak']==row['logit_old_peak'];m['rescues_same_original_max']+=int(same)
                m['rescues_same_max_smaller_near_set']+=int(same and int(row['logit_old_near_count'])<int(row['old_near_count']))
    report=dict(status='PASS',exact_near_vectors=vectors,learning_choices=comparisons,mechanisms=mechanisms,
        scope='Full saved-vector arithmetic and matched gains/losses; no refit or protocol change.')
    (out/'CHOICE_REVIEW.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
