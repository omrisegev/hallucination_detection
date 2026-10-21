import json, glob, os
D='results/sml_weight_estimator_v1/socratic_chain_flip'
tot=0; flips=[]; negs=[]
for f in sorted(glob.glob(D+'/DIAG_*.json')):
    d=json.load(open(f)); p=os.path.basename(f)[5:-5]
    print(p, d.get('bank'))
    for bk in ('N13','N16'):
        b=d[bk]
        for arm,key,wkey in (('CHAIN','CHAIN_flipped_by_zp_anchor','CHAIN_cross_weights_after_anchor_flip'),
                             ('PRMG','PRMG_flipped_by_zp_anchor','PRMG_cross_weights_signed'),
                             ('TELE','tele_flipped_by_anchor','tele_cross_weights')):
            fl=b.get(key); w=b.get(wkey)
            # PRM weight is index 0 for CHAIN (zp first); for PRMG find where PRM sits
            if arm!='TELE': tot+=1
            if fl: flips.append((p,bk,arm))
            if arm=='CHAIN' and w is not None:
                print(f'  {bk} {arm} flipped={fl} final PRM weight={w[0]:+.3f} weights={[round(x,3) for x in w]}')
            elif arm=='PRMG':
                print(f'  {bk} {arm} flipped={fl} signed weights={[round(x,3) for x in w]}')
            else:
                print(f'  {bk} {arm} flipped={fl}')
        if 'PRMG_cross_weights_after_anchor_flip' in b: print('   PRMG after flip', b['PRMG_cross_weights_after_anchor_flip'])
print('PRM-including fits:', tot, 'flips (CHAIN/PRMG):', [x for x in flips if x[2]!='TELE'], len([x for x in flips if x[2]!='TELE']))
print('TELE flips:', [x for x in flips if x[2]=='TELE'])
