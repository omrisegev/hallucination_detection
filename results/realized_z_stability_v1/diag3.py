import numpy as np, json
from scipy.stats import spearmanr
from fast import wauc, get
from pm import pm_auc, az
D=get(); pbd=D['PB']; pr=D['PRMB']
G={g['uid']:g for g in json.load(open('/home/user/hallucination_detection/results/prm_boost_processbench_v1/local/processbench_gold.json'))}
gen=np.array([G[u]['generator'] for u in pbd['ids']]); gs=np.repeat(gen,np.diff(pbd['off']))
fam=np.where(np.char.startswith(gs.astype(str),'Qwen2.5-Math'),'Qwen2.5-Math (scorer family)',np.where(np.char.find(gs.astype(str),'Llama')>=0,'Llama','Qwen2/2.5 general'))
def wcorr(d,a,b,m):
    za=az(a,d['off']); zb=az(b,d['off']); return spearmanr(za[m],zb[m])[0]
print('within-answer spearman(realized_z, q15_H1) on correct steps; mean realized_z-sign: excess surprisal per token not avail')
for k,d in D.items():
    m=d['keep']&~d['err']; print(f' {k}: corr(rz,H1)={wcorr(d,d["col"]("realized_z"),d["col"]("q15_H1"),m):+.3f}  corr(rz,chosen_surprisal)={wcorr(d,d["col"]("realized_z"),d["col"]("chosen_surprisal"),m):+.3f}  corr(cs,H1)={wcorr(d,d["col"]("chosen_surprisal"),d["col"]("q15_H1"),m):+.3f}')
for fm in sorted(set(fam)):
    mm=fam==fm; m=pbd['keep']&~pbd['err']&mm
    print(f' PB {fm:30s} corr(rz,H1)={wcorr(pbd,pbd["col"]("realized_z"),pbd["col"]("q15_H1"),m):+.3f}  within rz={wauc(pbd["err"],pbd["col"]("realized_z"),pbd["off"],pbd["keep"]&mm):.3f} cs={wauc(pbd["err"],pbd["col"]("chosen_surprisal"),pbd["off"],pbd["keep"]&mm):.3f} drv={wauc(pbd["err"],pbd["col"]("realized_drv"),pbd["off"],pbd["keep"]&mm):.3f}')
# ct7_chosen_std_excess = token std-excess top10 (related): 
for k,d in D.items():
    print(k,'ct7_chosen_std_excess within',round(wauc(d['err'],d['col']('ct7_chosen_std_excess'),d['off'],d['keep']),3),'PM',round(pm_auc(d,d['col']('ct7_chosen_std_excess')),3))
# PRMBench by error classification
print('PRMBench within-AUC by classification')
for c in sorted(set(pr['cat'])):
    if c=='correct': continue
    mm=np.repeat(pr['cat']==c,np.diff(pr['off']))
    print(f'  {c:22s}', ' '.join(f'{f[:10]}={wauc(pr["err"],pr["col"](f),pr["off"],pr["keep"]&mm):.3f}' for f in ['chosen_surprisal','realized_z','realized_drv','q15_H1']))
