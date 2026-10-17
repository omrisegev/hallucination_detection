import numpy as np, json, pickle, pandas as pd
R='C:/Users/omris/TAU/hallucination_detection/.worktrees/'
P=dict(
 ans=R+'readout-quickest-detection-v1/results/step_evidence_v1/OOF_ANSWERS.csv',
 oof=R+'readout-quickest-detection-v1/results/step_evidence_v1/OOF_STEP_SCORES.npz',
 frz=R+'readout-quickest-detection-v1/results/step_evidence_v1/INPUT_FREEZE.json',
 S=R+'ssl-pseudolabel-residual-v1/results/expectation_realization_v1/run_20260927_stage_b/STEP_SCORES.npz',
 thr=R+'ssl-pseudolabel-residual-v1/results/expectation_realization_v1/run_20260927_stage_b_thr/THRESHOLDS.json',
 der=R+'token-probability-fusion-v1/results/token_probability_fusion_v1/DERIVATIVE_CHANNELS.npz',
 prof=R+'cumulative-vote-fusion-v2/results/cumulative_vote_fusion_v2/ct7_profiles_v1/profiles.npy',
 pval=R+'cumulative-vote-fusion-v2/results/cumulative_vote_fusion_v2/ct7_profiles_v1/PROFILE_VALIDATION.json')
def load():
    a=pd.read_csv(P['ans'],encoding='utf-8-sig',usecols=['id','source_group','fold','cell','target'])
    oof=np.load(P['oof']); off=oof['offsets']; lab=oof['labels']
    se=np.load(P['S']); assert np.array_equal(se['offsets'],off)
    S=se['S_equal']
    tau=json.load(open(P['thr']))['S_equal']; tau=np.array([tau[str(k)] for k in range(5)])
    d=np.load(P['der']); ch=[str(c) for c in d['channels']]
    prof=np.load(P['prof']); pch=json.load(open(P['pval'],encoding='utf-8-sig'))['channels']
    raw=np.column_stack([d['level'], prof[:,pch.index('chosen_token_z_despiked')], d['derivative'][:,ch.index('chosen_surprisal')]])
    names=ch+['realized_z','realized_drv']
    surv=[i for i,n in enumerate(names) if n not in ('energy_innovation','top50_js')]
    frz=json.load(open(P['frz'],encoding='utf-8-sig'))
    meta=pickle.load(open(frz['prm_metadata']['path'],'rb'))
    return a,off,lab,S,tau,raw,names,surv,meta,frz['prm_metadata']['path']
