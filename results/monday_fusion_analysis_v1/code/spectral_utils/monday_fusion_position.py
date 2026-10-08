"""Position diagnostics and the explicitly new grouped cross-fitted slope extension."""
import hashlib
import numpy as np
from spectral_utils.monday_fusion_core import between_weights,choose_anchor
from spectral_utils.monday_fusion_metrics import answer_z

def half_groups(groups):
    sizes=np.bincount(groups); assignment={}; totals=[0,0]
    for g in sorted(range(len(sizes)),key=lambda g:(-sizes[g],g)):
        side=0 if totals[0]<=totals[1] else 1
        assignment[g]=side; totals[side]+=int(sizes[g])
    if min(totals)<3: raise ValueError('Cross-fit half has fewer than 3 channels; no alternate split')
    return [np.array([g for g in range(len(sizes)) if assignment[g]==s],int) for s in range(2)]

def prior(votes,off,cache=None):
    import er_stage_a as SA,position_prior_ds as PP
    key=hashlib.sha256(votes.tobytes()+off.tobytes()).hexdigest()
    if cache is not None and key in cache: return cache[key]
    if votes.shape[1]<3: raise ValueError('Position latent fit needs at least 3 votes')
    est=SA.em_estimate(votes,'ds')
    one=PP.fit_pds(votes,np.zeros(len(votes),int),1,est['psi'],est['eta'],est['prevalence'])
    dev=max(np.abs(one['psi']-est['psi']).max(),np.abs(one['eta']-est['eta']).max(),abs(one['pi_bins'][0]-est['prevalence']))
    rel=(one['loglik']-one['loglik_start'])/abs(one['loglik_start'])
    if not(one['converged'] and one['oriented'] and rel<=1e-6 and dev<=1e-2):
        raise ValueError(f'One-bin DS optimum gate: move={dev}, relative_likelihood={rel}')
    bins=PP.position_bins(off,10)
    fit=PP.fit_pds(votes,bins,10,one['psi'],one['eta'],float(one['pi_bins'][0]))
    if not(fit['converged'] and fit['oriented']): raise ValueError('Position prior did not converge/orient')
    if fit['bin_rows'].min()<50: raise ValueError('Position bin has fewer than 50 rows')
    lp=(np.log(fit['pi_bins'])-np.log1p(-fit['pi_bins']))[bins]
    info={k:v for k,v in fit.items() if k!='q'}
    info.update(one_bin_max_move=float(dev),one_bin_relative_likelihood=float(rel),
                clipped_bins=int(((fit['pi_bins']<=PP.EPS*1.001)|(fit['pi_bins']>=1-PP.EPS*1.001)).sum()))
    result=(lp,fit['q'],info)
    if cache is not None: cache[key]=result
    return result

def calibrated(score,q):
    import position_prior_ds as PP
    a,parts=PP.latent_slope(score,q)
    if not np.isfinite(a) or a<=0: raise ValueError(f'Nonpositive/nonfinite latent slope {a}')
    return a*(score-(parts['mu1']+parts['mu0'])/2),float(a),parts

def variants(data,state,recipe,cache=None):
    import er_stage_a as SA,er_stage_b as SB,lsml_merge_step as MS,tail_calib_common as TC
    b=state['bank']; off=b['off']; rows=b['rows']; score=state['scores'][recipe]
    pos=data.env['POSZ'][rows]; posm=data.env['POSM'][rows]
    if b.get('quota')=='cell':
        from spectral_utils.monday_fusion_core import quota_marks
        keyp=np.random.default_rng(20260931).random((data.pop.total,1))[rows]
        posm=quota_marks(pos[:,None],off,keyp,True)
    result={'CONTENT':score,'POSZ':pos,'EQUAL_CONTENT_POSITION':.5*answer_z(score,off)+.5*pos}
    diag={}; failures={}
    grouped=recipe in state['recipes']
    if grouped:
        r=state['recipes'][recipe]; z=r['Z']; votes=r['votes']; weights=r['weights']; groups=state['groups']
        channel_votes=b['marks'][:,state['surv']]; between=recipe.split('_')[1]
        ai=state['anchor']; anchor_group=int(groups[ai]); anchor_score=state['X'][:,ai]
    else:
        idx=np.arange(b['V'].shape[1]) if recipe=='ALL_EQ' else state['surv']
        z=b['V'][:,idx]; votes=b['marks'][:,idx]; channel_votes=votes
        weights=np.ones(len(idx))/len(idx)
        if recipe=='ALL_EQ':
            names=[b['names'][i] for i in idx]
            try:
                anchor,_=choose_anchor(names,state['surv'],state['diagnostics']['filter_estimator']['pi'])
                gb=MS.canon(TC.lsml_fit_scaled(b['TT'][:,idx],anchor,z,standardize=True,loading_scale='unit')['groups'])
                if len(set(gb))<3: raise ValueError('ALL_EQ has fewer than 3 full-bank groups')
                groups,_=MS.absorb_merge(np.corrcoef(b['TT'][:,idx],rowvar=False),gb,min_groups=3)
            except (ValueError,ArithmeticError,np.linalg.LinAlgError) as exc:
                groups=None; failures['PRIOR_CF']=str(exc)
        else: groups=state.get('groups')
    try:
        if grouped:
            zp=np.column_stack([z,pos]); mp=np.column_stack([votes,posm])
            is_channel=b.get('vote_type')=='channels'
            wp,ep=between_weights(between,zp,mp,anchor_score,anchor_group,
                                 channel_votes=np.column_stack([channel_votes,posm]) if is_channel else None,
                                 groups=np.r_[groups,z.shape[1]] if is_channel else None,channel_anchor=ai)
            result['POS']=zp@wp; diag['POS']={'weights':wp,'estimator':ep}
        elif recipe=='ALL_EQ':
            result['POS']=np.column_stack([b['V'],pos]).mean(1)
            diag['POS']={'position_kept':True,'filter':'none; matched unfiltered control'}
        else:
            mp=np.column_stack([b['marks'],posm]); ep=SA.em_estimate(mp,'ds'); keep=np.flatnonzero(ep['pi']>.5)
            if len(keep)==0: raise ValueError('No survivors after adding position')
            result['POS']=np.column_stack([b['V'],pos])[:,keep]@np.full(len(keep),1/len(keep))
            diag['POS']={'position_kept':bool(b['V'].shape[1] in keep),'survivors':keep,'estimator':ep}
    except (ValueError,ArithmeticError,np.linalg.LinAlgError) as exc: failures['POS']=str(exc)
    try:
        prior_votes=channel_votes if grouped and b.get('vote_type')=='channels' else votes
        lp,q,fit=prior(prior_votes,off,cache); _,slope,parts=calibrated(score,q)
        result['PRIOR']=score+lp/slope; diag['PRIOR']={'slope':slope,'parts':parts,'fit':fit}
    except (ValueError,ArithmeticError,np.linalg.LinAlgError) as exc:
        failures['PRIOR']=str(exc); failures['PRIOR_CF']='Prior failure: '+str(exc)
        return result,diag,failures
    try:
        if groups is None: raise ValueError('No partition for cross-fit')
        halves=half_groups(groups); half_score=[]; posterior=[]; details=[]
        for h in halves:
            channels=np.flatnonzero(np.isin(groups,h))
            _,qh,fit_h=prior(channel_votes[:,channels],off,cache); posterior.append(qh)
            selected=h if grouped else channels
            wh=weights[selected]; denom=np.abs(wh).sum()
            if denom<=0: raise ValueError('Cross-fit side has zero score weight')
            half_score.append(z[:,selected]@(wh/denom))
            details.append({'groups':h,'channels':channels,'weights':wh/denom,'fit':fit_h})
        a,sa,pa=calibrated(half_score[0],posterior[1]); c,sb,pb=calibrated(half_score[1],posterior[0])
        content=a+c; result['PRIOR_CF']=content+lp; result['CF_CONTENT_ONLY']=content
        diag['PRIOR_CF']={'halves':details,'slopes':[sa,sb],'parts':[pa,pb],
                          'content_correlation':float(np.corrcoef(score,content)[0,1]),'new_grouped_extension':grouped}
    except (ValueError,ArithmeticError,np.linalg.LinAlgError) as exc: failures['PRIOR_CF']=str(exc)
    return result,diag,failures
