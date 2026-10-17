"""Source-question-disjoint nested LR reference, balanced and fit-only standardized."""
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

C_GRID=(.001,.01,.1,1.,10.,100.,1000.)

def training_labels(pop,bank):
    y=np.full(len(bank['V']),-1,int)
    for i,a,b in zip(bank['answers'],bank['off'][:-1],bank['off'][1:]):
        if pop.prm[i]: y[a:b]=pop.labels[pop.off[i]:pop.off[i+1]].astype(int)
        elif pop.target[i]<0: y[a:b]=0
        else:
            t=int(pop.target[i]); y[a:a+t]=0; y[a+t]=1
    return y

def model_fit(x,y,c):
    if len(np.unique(y))!=2: raise ValueError('Training fold has only one class')
    scaler=StandardScaler().fit(x)
    model=LogisticRegression(C=c,class_weight='balanced',solver='lbfgs',max_iter=3000,random_state=20261003)
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        model.fit(scaler.transform(x),y)
    return scaler,model

def balanced_loss(y,prob):
    counts=np.bincount(y,minlength=2)
    if (counts==0).any(): raise ValueError('Inner validation lacks a class')
    weight=len(y)/(2*counts[y])
    return float(log_loss(y,prob,sample_weight=weight,labels=[0,1]))

def nested_lr(pop,bank,columns,position=None):
    x=bank['V'][:,columns]
    if position is not None: x=np.column_stack([x,position])
    y=training_labels(pop,bank); ns=np.diff(bank['off'])
    folds=np.repeat(pop.fold[bank['answers']],ns)
    oof=np.full(len(x),np.nan); diagnostics=[]
    for outer in range(5):
        losses=[]
        for c in C_GRID:
            trial=[]
            for inner in range(5):
                if inner==outer: continue
                train=(folds!=outer)&(folds!=inner)&(y>=0); valid=(folds==inner)&(y>=0)
                scale,fit=model_fit(x[train],y[train],c)
                trial.append(balanced_loss(y[valid],fit.predict_proba(scale.transform(x[valid]))[:,1]))
            losses.append(float(np.mean(trial)))
        best=min(range(len(C_GRID)),key=lambda j:(losses[j],C_GRID[j])); c=C_GRID[best]
        train=(folds!=outer)&(y>=0); test=folds==outer
        scale,fit=model_fit(x[train],y[train],c)
        oof[test]=fit.decision_function(scale.transform(x[test]))
        diagnostics.append({'outer_fold':outer,'C':c,'inner_balanced_log_loss':losses,'n_train_steps':int(train.sum()),
                            'n_test_steps':int(test.sum()),'coef':fit.coef_.tolist(),'intercept':fit.intercept_.tolist(),
                            'mean':scale.mean_.tolist(),'scale':scale.scale_.tolist()})
    if not np.isfinite(oof).all(): raise ValueError('Incomplete OOF scores')
    # In-sample reference uses median selected C; does not affect any OOF prediction.
    selected_c=float(np.median([d['C'] for d in diagnostics]))
    scale,fit=model_fit(x[y>=0],y[y>=0],selected_c)
    insample=fit.decision_function(scale.transform(x))
    return {'oof':oof,'in_sample':insample,'diagnostics':diagnostics,'folds':folds,
            'n_unknown_post_error_steps':int((y<0).sum()),'in_sample_C':selected_c}
