exec(open('load.py').read())
from sklearn.metrics import roc_auc_score
style=pd.read_pickle('style.pkl')
def auc(pos,neg):
    y=np.r_[np.ones(len(pos)),np.zeros(len(neg))]; s=np.r_[pos,neg]; return roc_auc_score(y,s)
feats=['epr','trace_length','epr_spilled','epr_energy','varentropy','logprob_margin','tail50_mass','spectral_entropy','low_band_power','sw_var_peak','cusum_max']
redc=err_nc&(cls=='redundency'); circ=err_nc&(cls=='circular')
rows=[]
for f in feats:
    v=X[:,names.index(f)]
    rows.append(dict(feature=f, med_ms=np.median(v[ms]), med_err=np.median(v[err_nc]), med_ctrl=np.median(v[control]), med_err_red=np.median(v[redc]), med_err_circ=np.median(v[circ]),
      auc_err_vs_ms=auc(v[err_nc],v[ms]), auc_ctrl_vs_ms=auc(v[control],v[ms]), auc_err_vs_ctrl=auc(v[err_nc],v[control]),
      auc_red_vs_ctrl=auc(v[redc],v[control]), auc_circ_vs_ctrl=auc(v[circ],v[control])))
nsv=ns.astype(float)
rows.append(dict(feature='n_steps',med_ms=np.median(nsv[ms]),med_err=np.median(nsv[err_nc]),med_ctrl=np.median(nsv[control]),med_err_red=np.median(nsv[redc]),med_err_circ=np.median(nsv[circ]),
  auc_err_vs_ms=auc(nsv[err_nc],nsv[ms]),auc_ctrl_vs_ms=auc(nsv[control],nsv[ms]),auc_err_vs_ctrl=auc(nsv[err_nc],nsv[control]),auc_red_vs_ctrl=auc(nsv[redc],nsv[control]),auc_circ_vs_ctrl=auc(nsv[circ],nsv[control])))
tps=X[:,names.index('trace_length')]/ns
rows.append(dict(feature='tokens_per_step',med_ms=np.median(tps[ms]),med_err=np.median(tps[err_nc]),med_ctrl=np.median(tps[control]),med_err_red=np.median(tps[redc]),med_err_circ=np.median(tps[circ]),
  auc_err_vs_ms=auc(tps[err_nc],tps[ms]),auc_ctrl_vs_ms=auc(tps[control],tps[ms]),auc_err_vs_ctrl=auc(tps[err_nc],tps[control]),auc_red_vs_ctrl=auc(tps[redc],tps[control]),auc_circ_vs_ctrl=auc(tps[circ],tps[control])))
for d in ['D1_upcr_full','D2_lsml_cont_good5','D3_lsml_full','D4_equal_full','D5_epr','D6_length','D6b_length_anchored']:
    v=D['A_'+d].astype(float)
    rows.append(dict(feature='A_'+d, med_ms=np.median(v[ms]), med_err=np.median(v[err_nc]), med_ctrl=np.median(v[control]), med_err_red=np.median(v[redc]), med_err_circ=np.median(v[circ]),
      auc_err_vs_ms=auc(v[err_nc],v[ms]), auc_ctrl_vs_ms=auc(v[control],v[ms]), auc_err_vs_ctrl=auc(v[err_nc],v[control]),auc_red_vs_ctrl=auc(v[redc],v[control]), auc_circ_vs_ctrl=auc(v[circ],v[control])))
pd.set_option('display.width',260); pd.set_option('display.max_columns',30)
T=pd.DataFrame(rows).set_index('feature'); print(T.round(4).to_string())
T.to_csv('t1_feature_table.csv')
# per erroneous class AUROC vs ms and vs ctrl for D1,D2,D5
print()
for d in ['D1_upcr_full','D2_lsml_cont_good5','D5_epr']:
    v=D['A_'+d].astype(float); out={}
    for c in sorted(np.unique(cls[err_nc])):
        m=err_nc&(cls==c); out[c]=(round(auc(v[m],v[ms]),3), round(auc(v[m],v[control]),3))
    print(d,'(vs ms, vs ctrl):',out)
