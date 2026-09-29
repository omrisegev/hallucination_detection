exec(open('load.py').read())
from sklearn.metrics import roc_auc_score
# text-derived style stats
P=np.flatnonzero(prm)
style=pd.DataFrame(index=P)
style['cls']=cls[P]
st=[raw[ids[i]]['steps'] for i in P]
assert all(len(s)==ns[i] for s,i in zip(st,P)), 'step count mismatch'
style['n_steps']=ns[P]
style['chars']=[sum(len(x) for x in s) for s in st]
style['chars_per_step']=style.chars/style.n_steps
style['step_prefix_share']=[np.mean([bool(re.match(r'^\s*Step\s*\d+',x)) for x in s]) for s in st]
style['indent8_share']=[np.mean(['        ' in x for x in s]) for s in st]
style['ntok']=X[P,names.index('trace_length')]
style['tok_per_step']=style.ntok/style.n_steps
grp=np.full(n,'',object)
grp[ms]='multi_solutions'; grp[control]='controls'
for c in np.unique(cls[err_nc]): grp[err_nc&(cls==c)]='err_'+c
grp[noncontrol&~ms&~err_nc]='noncontrol_no_inrange_err'
style['grp']=grp[P]
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
print(style.groupby('grp')[['n_steps','chars','chars_per_step','ntok','tok_per_step','step_prefix_share','indent8_share']].median().round(3))
print('count',style.groupby('grp').size().to_dict())
print('share of answers with >=50% steps prefixed "Step N"', style.assign(v=style.step_prefix_share>=.5).groupby('grp').v.mean().round(3).to_dict())
style.to_pickle('style.pkl')
