import csv
R='results/prm_boost_ablation5_v1/'
files={p:R+f'ABL5_RESULTS_{p}.csv' for p in ['qwen_prm_7b','qwen_prm_72b','pure_prm','rlhflow_prm','math_shepherd']}
files['math_shepherd_DIGITFIX']=R+'digitfix_v1/digitfix/ABL5_RESULTS_math_shepherd.csv'
for p,f in files.items():
    rows=list(csv.DictReader(open(f)))
    g={(r['arm'],r['rule']):r for r in rows}
    def get(a):
        c=[r for r in rows if r['arm']==a]
        return c
    out=[]
    for arm in ['PRM_native','PRM_Q80','PRM_Q80_PROB','CHAIN_N16','CHAIN_N13','EQ_N16','MEAN_N16','TELE_N16']:
        c=get(arm)
        out.append((arm,[(r['rule'],round(float(r['within_auc']),4),round(float(r['prmscore']),2)) for r in c]))
    ch=float(get('CHAIN_N16')[0]['within_auc']); pr=float(get('PRM_Q80_PROB')[0]['within_auc']); pn=float(get('PRM_native')[0]['within_auc'])
    print(p, 'CHAIN_N16-PRM_Q80_PROB within = %+.4f'%(ch-pr), ' (vs PRM_native %+.4f)'%(ch-pn))
    for o in out: print('   ',o)
