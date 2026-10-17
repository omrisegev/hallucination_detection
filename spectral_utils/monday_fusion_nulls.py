"""Full-cell score-fixed label nulls; never fabricate ProcessBench step labels."""
import numpy as np
from spectral_utils.monday_fusion_metrics import within_auc

def nulls(pop,bank,scores,permutations=200):
    import er_stage_b2 as C2
    names=list(scores); matrix=np.column_stack([scores[n] for n in names])
    off=bank['off']; ns=np.diff(off); answers=bank['answers']
    singleton=sum(int(np.sum(ns==n))==1 for n in np.unique(ns))
    if pop.prm[answers[0]]:
        eligible=np.array([pop.eligible[i] for i in answers]); ei=np.flatnonzero(eligible)
        rank,loc=C2.within_ranks(matrix,off,ei)
        y=np.concatenate([pop.labels[pop.off[answers[j]]:pop.off[answers[j]+1]] for j in ei]).astype(float)
        observed=np.nanmean(C2.auc_from_ranks(rank,loc,y),0)
        result={n:{'observed_within_auc':float(observed[j]),'n_checked':len(ei),'scope':'eligible PRMB answers',
                   'same_length_singletons':int(sum(np.sum(ns[ei]==n)==1 for n in np.unique(ns[ei])))} for j,n in enumerate(names)}
        for label,func,seed in [('within_answer_shuffle',C2.shuffle_within,11),('whole_answer_same_length_swap',C2.swap_same_length,12)]:
            rng=np.random.default_rng(seed)
            sample=np.array([np.nanmean(C2.auc_from_ranks(rank,loc,func(y,loc,rng)),0) for _ in range(permutations)])
            for j,n in enumerate(names):
                result[n][label]={'mean':float(sample[:,j].mean()),'sd':float(sample[:,j].std()),
                                  'p_plus_one':float((1+np.sum(sample[:,j]>=observed[j]))/(permutations+1))}
    else:
        target=pop.target[answers].copy(); erroneous=target>=0
        pred=np.stack([np.argmax(matrix[a:b],axis=0) for a,b in zip(off[:-1],off[1:])])
        observed=np.mean(pred[erroneous]==target[erroneous,None],axis=0)
        result={n:{'observed_first_error_accuracy':float(observed[j]),'n_checked':len(answers),
                   'same_length_singletons':int(singleton),'scope':'valid first-error targets; no full-step labels'} for j,n in enumerate(names)}
        for mode,seed in [('within_answer_shuffle',11),('whole_answer_same_length_swap',12)]:
            rng=np.random.default_rng(seed); sample=[]
            for _ in range(permutations):
                shuffled=target.copy()
                if mode=='within_answer_shuffle':
                    shuffled[erroneous]=rng.integers(0,ns[erroneous])
                else:
                    for length in np.unique(ns):
                        ix=np.flatnonzero(ns==length)
                        if len(ix)>1:
                            perm=rng.permutation(ix); shifted=np.roll(perm,1)
                            shuffled[perm]=target[shifted]
                mask=shuffled>=0
                sample.append(np.mean(pred[mask]==shuffled[mask,None],axis=0))
            sample=np.asarray(sample)
            for j,n in enumerate(names):
                result[n][mode]={'mean':float(sample[:,j].mean()),'sd':float(sample[:,j].std()),
                                'p_plus_one':float((1+np.sum(sample[:,j]>=observed[j]))/(permutations+1))}
    return result
