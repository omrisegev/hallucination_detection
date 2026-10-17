"""Label-free interventions; no labels or quality metrics are accepted here."""
import numpy as np
from scipy.stats import spearmanr
from spectral_utils.lsml_gate_locator_research import answer_standardize

LEVEL = ('q15_H1','q15_VE1','logprob_margin','true_tail50','energy_level')

def removal_names(level_values, count, random=False):
    if count not in (0,2,3): raise ValueError('Unregistered removal count')
    if count==0: return []
    if random:
        return np.random.default_rng(20261003).permutation(sorted(LEVEL))[:count].tolist()
    remaining=list(LEVEL); removed=[]
    while len(removed)<count:
        scores={c:np.mean([abs(float(spearmanr(level_values[:,LEVEL.index(c)],level_values[:,LEVEL.index(d)]).statistic)) for d in remaining if d!=c]) for c in remaining}
        if not all(np.isfinite(v) for v in scores.values()): raise ValueError('Undefined LEVEL correlation')
        chosen=min(remaining,key=lambda c:(-scores[c],c))
        removed.append(chosen); remaining.remove(chosen)
    return removed

def variant_bank(data,bk,cell,variant):
    e=data.env; p=data.pop
    ans=np.flatnonzero(p.cells==cell); rows=p.rows(p.cells==cell)
    off=np.r_[0,np.cumsum(p.ns[ans])]
    names=e['BN'][bk]
    count=0 if variant=='original' else int(variant.rsplit('_',1)[1])
    removed=removal_names(e['POOL'][rows][:,[e['PN'].index(c) for c in LEVEL]],count,variant.startswith('random'))
    idx=np.array([i for i,c in enumerate(names) if c not in removed],int)
    surviving_level=[c for c in LEVEL if c not in removed]
    level=e['POOL'][rows][:,[e['PN'].index(c) for c in surviving_level]].mean(1)
    v=e['VB'][bk][rows][:,idx].copy(); signs={}; flips=[]; pooled_flips=[]
    for j,i in enumerate(idx):
        c=names[i]
        if c.startswith('lf__'):
            raw=e['POOL'][rows,e['PN'].index(c[4:])]
            r=float(np.corrcoef(raw,level)[0,1])
            if not np.isfinite(r): raise ValueError('Undefined orientation '+c)
            old=float(np.sign(e['PS'].loc[c[4:],'r_level_marginal'])) or 1.
            sign=float(np.sign(r)) or 1.
            r0=float(np.corrcoef(raw,e['LEVEL'][rows])[0,1]); sign0=float(np.sign(r0)) or 1.
            v[:,j]*=sign/old
            if sign!=old: pooled_flips.append(j)
            signs[c]={'correlation':r,'sign':sign,'original_cell_sign':sign0,'changed_from_original':sign!=sign0}
            if sign!=sign0: flips.append(c)
    keys=e['KEYB'][bk][rows][:,idx]
    # Preserve the historical summation layout: unchanged columns retain their
    # original marks; changed-sign columns are computed as a column slice (F order).
    # Recomputing every column in a C-order cell matrix changes some tied-column
    # centering values by 1e-16, violating exact zero-removal replay.
    marks=e['MARKS'][bk][rows][:,idx].copy()
    tt=e['TTA'][bk][rows][:,idx].copy()
    if pooled_flips:
        marks[:,pooled_flips]=e['SB'].random_tie_marks(v[:,pooled_flips],off,.2,keys[:,pooled_flips])
        tt[:,pooled_flips]=e['tail_marks'](v[:,pooled_flips],off,.2,tie_aware=True,centred=True)[0]
    hist={c:abs(float(spearmanr(v[:,j],level).statistic)) for j,c in enumerate(np.array(names)[idx]) if 'hist_' in c}
    return {'V':v,'marks':marks,'TT':tt,'key':keys,'off':off,'rows':rows,'answers':ans,
            'names':[names[i] for i in idx], 'diagnostics':{'removed':removed,'remaining_LEVEL':surviving_level,'sign_flips':flips,'lf_signs':signs,'hist_abs_spearman_LEVEL':hist}}

def quota_marks(v,off,key,cell_wide=False):
    import er_stage_b as SB
    reference=SB.random_tie_marks(v,off,.2,key)
    if not cell_wide: return reference
    out=-np.ones(v.shape,np.int8)
    for j in range(v.shape[1]):
        k=int((reference[:,j]>0).sum())
        order=np.lexsort((key[:,j],-v[:,j]))
        out[order[:k],j]=1
    assert np.array_equal((out>0).sum(0),(reference>0).sum(0))
    return out

def choose_anchor(names,surv,pi):
    if 'q15_H1' in names and names.index('q15_H1') in surv: return names.index('q15_H1'),'original'
    options=[i for i in surv if names[i] in LEVEL]
    reason='highest_DS_pi_surviving_LEVEL'
    if not options: options=list(surv); reason='highest_DS_pi_any_survivor'
    if not options: raise ValueError('No survivor for anchor')
    return min(options,key=lambda i:(-pi[i],names[i])),reason

def between_weights(method,z,votes,anchor_score,anchor_group,channel_votes=None,groups=None,channel_anchor=None):
    import er_stage_a as SA, er_stage_b as SB
    from spectral_utils.fusion_utils import sml_fuse_signed
    g=z.shape[1]
    if g<3: raise ValueError(f'{g} groups: minimum is 3')
    est={}
    if method=='EQ': w=np.ones(g)
    elif method=='SML':
        _,w=sml_fuse_signed(*z.T,small_m_guard=True); w=np.asarray(w,float)
        rho=float(spearmanr(z@w,anchor_score).statistic)
        if not np.isfinite(rho): raise ValueError('Undefined SML anchor orientation')
        if rho<0: w=-w
        est={'small_m_guard':g==3,'anchor_flipped':rho<0,'anchor_spearman':abs(rho),'binary_quota_affects_weights':False}
    else:
        inputs=votes if channel_votes is None else channel_votes
        anchor=anchor_group if channel_votes is None else channel_anchor
        if anchor is None: raise ValueError('Channel estimator requires explicit surviving anchor index')
        if inputs.shape[1]<3: raise ValueError('Fewer than 3 votes')
        est=SA.em_estimate(inputs,'ds') if method=='DS' else SA.tensor_mom_estimate(inputs,anchor)
        w=SB.mle_weights(est['psi'],est['eta'])
        if channel_votes is not None: w=np.array([w[groups==h].mean() for h in range(g)])
    if not np.isfinite(w).all() or np.abs(w).sum()<=0: raise ValueError('Invalid/all-zero between weights')
    return w/np.abs(w).sum(),est

def fit_content(data,bank,recipes=None):
    import er_stage_a as SA, ds_group_weights as GW, lsml_merge_step as MS, tail_calib_common as TC
    import er_stage_b as SB
    v,m,tt,off=bank['V'],bank['marks'],bank['TT'],bank['off']
    d=dict(bank['diagnostics']); failures={}; scores={'ALL_EQ':v.mean(1)}
    est=SA.em_estimate(m,'ds'); surv=np.flatnonzero(est['pi']>.5)
    d.update(filter_estimator=est,survivors=[bank['names'][i] for i in surv])
    state={'scores':scores,'diagnostics':d,'failures':failures,'bank':bank,'surv':surv,'recipes':{}}
    if len(surv)==0:
        failures['DS_FILTER_EQ']='No DS survivors'; return state
    x=v[:,surv]; scores['DS_FILTER_EQ']=x@np.full(len(surv),1/len(surv))
    anchor,reason=choose_anchor(bank['names'],surv,est['pi']); ai=int(np.flatnonzero(surv==anchor)[0])
    d.update(anchor=bank['names'][anchor],anchor_fallback=reason!='original',anchor_reason=reason)
    state.update(X=x,anchor=ai)
    try:
        gb=MS.canon(TC.lsml_fit_scaled(tt[:,surv],ai,x,standardize=True,loading_scale='unit')['groups'])
        if len(set(gb))<3: raise ValueError(f'{len(set(gb))} groups before merge: minimum is 3')
        groups,merge=MS.absorb_merge(np.corrcoef(tt[:,surv],rowvar=False),gb,min_groups=3)
        if len(set(groups))<3: raise ValueError('Fewer than 3 groups after merge')
        d.update(groups_before=gb,groups_after=groups,merge=merge,
                 level_groups_before=sorted(set(int(gb[j]) for j,i in enumerate(surv) if bank['names'][i] in LEVEL)),
                 level_groups_after=sorted(set(int(groups[j]) for j,i in enumerate(surv) if bank['names'][i] in LEVEL)))
        state['groups']=groups
    except (ValueError,ArithmeticError,np.linalg.LinAlgError) as exc:
        failures['GROUPING']=str(exc); return state
    n_groups=len(set(groups)); key=data.env['KEYG'][bank['rows'],:n_groups]
    if key.shape[1]!=n_groups: raise ValueError('Group key capacity exceeded')
    for within in ['MEAN','HEM']:
        if recipes is not None and not any(r.startswith(within+'_') for r in recipes): continue
        try:
            hem=None if within=='MEAN' else GW.hem_fit(m[:,surv],groups)
            wi=None if hem is None else hem['within']
            if hem is not None: d['HEM']=hem
            z=GW.group_matrix(x,off,groups,wi,answer_standardize)
            votes=SB.random_tie_marks(z,off,.2,key)
            d[within+'_group_correlation']=np.corrcoef(z,rowvar=False)
        except (ValueError,ArithmeticError,np.linalg.LinAlgError) as exc:
            failures[within]=str(exc); continue
        for between in ['EQ','DS','SML','MOM']:
            name=within+'_'+between
            if recipes is not None and name not in recipes: continue
            try:
                w,ef=between_weights(between,z,votes,x[:,ai],int(groups[ai]))
                scores[name]=GW.weighted_group_score(z,w,signed=between=='SML')
                state['recipes'][name]={'Z':z,'votes':votes,'weights':w,'within':wi,'estimator':ef}
                p=np.abs(w)/np.abs(w).sum()
                d[name]={'weights':w,'estimator':ef,'effective_groups':float(1/(p@p)),'negative_weights':int((w<0).sum())}
                if between=='DS':
                    top=int(np.argmax(w)); d[within+'_top_DS_group']={'group':top,'members':[bank['names'][surv[j]] for j in np.flatnonzero(groups==top)]}
            except (ValueError,ArithmeticError,np.linalg.LinAlgError) as exc: failures[name]=str(exc)
    return state
