"""First-error step selection on the unchanged saved RBM token/Top10 scores."""
import numpy as np
from .matched_rbm_coefficient_update import Top10,RIDGE


class FirstErrorObjective:
    def __init__(self,x,base,spans,step_offsets,targets):
        self.x=x;self.base=base;self.top=Top10(x,spans)
        self.offsets=np.asarray(step_offsets,int);self.targets=np.asarray(targets,int)
        lengths=np.diff(self.offsets)
        if len(lengths)!=len(self.targets) or self.offsets[0]!=0 or self.offsets[-1]!=len(spans) or np.any(lengths<=0):
            raise ValueError('invalid answer-step offsets')
        if np.any(self.targets < -1) or np.any(self.targets>=lengths):raise ValueError('invalid first-error labels')
        self.eligible=self.targets>=0;self.n=int(self.eligible.sum())
        if not self.n:raise ValueError('no first-error labels in training fold')
        self.owner=np.repeat(np.arange(len(lengths)),lengths);self.lengths=lengths

    def __call__(self,delta):
        score,jac=self.top.evaluate(self.base+self.x@delta[:-1]+delta[-1],True)
        maxima=np.maximum.reduceat(score,self.offsets[:-1])
        e=np.exp(score-maxima[self.owner]);total=np.add.reduceat(e,self.offsets[:-1])
        indices=self.offsets[:-1][self.eligible]+self.targets[self.eligible]
        loss=np.mean(maxima[self.eligible]+np.log(total[self.eligible])-score[indices])
        residual=e/total[self.owner];residual[~self.eligible[self.owner]]=0.;residual[indices]-=1.
        gradient=jac.T@residual/self.n
        gradient[-1]=0. # Common step-score shift cancels exactly.
        return float(loss+.5*RIDGE*(delta@delta)),gradient+RIDGE*delta
