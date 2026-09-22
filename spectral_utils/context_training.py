"""Training access and batches for unlabeled chronological feature prediction."""
from __future__ import annotations
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .temporal_context_models import probability_features, flow_condition

METADATA_KEYS={'uid','cell','group_id','fold','offset','tokens','step_start','step_stop','mean','scale','signs'}


class FeatureBundle:
    def __init__(self,path,bank='original4'):
        path=Path(path)
        self.manifest=json.loads((path/'MANIFEST.json').read_text())
        self.metadata=json.loads((path/'METADATA.json').read_text())
        if any(set(m)!=METADATA_KEYS for m in self.metadata):
            raise ValueError('training bundle contains undeclared metadata; correctness labels are forbidden')
        self.bank=bank;self.columns=np.array(self.manifest['banks'][bank],int)
        self.features=np.load(path/'features.npy',mmap_mode='r')
        self.logits=np.load(path/'logprobs15.npy',mmap_mode='r')
        self.prefix=np.load(path/'h0_prefix_sum.npy',mmap_mode='r')
        self.spans=np.load(path/'step_spans.npy',mmap_mode='r')
        self.offset=np.array([m['offset'] for m in self.metadata],int)
        self.length=np.array([m['tokens'] for m in self.metadata],int)
        self.mean=np.array([m['mean'] for m in self.metadata])[:,self.columns]
        self.scale=np.array([m['scale'] for m in self.metadata])[:,self.columns]
        self.signs=np.array([m['signs'] for m in self.metadata])

    def split(self,excluded):
        excluded=set(excluded)
        held=[i for i,m in enumerate(self.metadata) if m['fold'] in excluded]
        allowed=[i for i,m in enumerate(self.metadata) if m['fold'] not in excluded]
        held_groups={self.metadata[i]['group_id'] for i in held}
        allowed_groups={self.metadata[i]['group_id'] for i in allowed}
        if held_groups&allowed_groups:raise ValueError('source-group overlap in model split')
        ordered=sorted(allowed_groups,key=lambda g:hashlib.sha256(('context-validation/'+g).encode()).digest())
        validation_groups=set(ordered[:max(1,len(ordered)//10)])
        validation=[i for i in allowed if self.metadata[i]['group_id'] in validation_groups]
        training=[i for i in allowed if self.metadata[i]['group_id'] not in validation_groups]
        if not training or not validation or not held:raise ValueError('empty context split')
        return training,validation,held

    def sampler(self,ids,rng,batch_size,flow=False):
        grouped=defaultdict(list)
        for i in ids:
            if self.length[i]>int(flow):grouped[self.metadata[i]['group_id']].append(i)
        groups=sorted(grouped)
        if not groups:raise ValueError('no eligible observations for context training')
        answers=np.array([rng.choice(grouped[groups[j]]) for j in rng.integers(len(groups),size=batch_size)],int)
        positions=np.array([rng.integers(self.length[i]-int(flow)) for i in answers],int)
        return answers,positions

    def batch(self,answers,positions,flow=False,device='cpu',with_target=True):
        answers=np.asarray(answers,int);positions=np.asarray(positions,int)
        if answers.shape!=positions.shape or np.any(positions<0) or np.any(positions>=self.length[answers]):
            raise ValueError('invalid answer/position batch')
        width=17 if flow else 16
        local=positions[:,None]+int(flow)-np.arange(width,0,-1)[None,:]
        mask=local>=0
        global_index=self.offset[answers,None]+np.maximum(local,0)
        raw=np.asarray(self.features[global_index])[:,:,self.columns]
        z=(raw-self.mean[answers,None,:])/self.scale[answers,None,:]
        z[~mask]=0
        tensor=lambda x,dtype=torch.float32:torch.tensor(x,dtype=dtype,device=device)
        result=dict(history=tensor(z),mask=tensor(mask,torch.bool),
            position=tensor((positions+.5)/self.length[answers]),
            local=tensor(local,torch.float64),source_logits=tensor(self.logits[global_index],torch.float64),
            signs=tensor(self.signs[answers,None,:],torch.float64),
            mean=tensor(self.mean[answers,None,:],torch.float64),scale=tensor(self.scale[answers,None,:],torch.float64),
            prefix=tensor(self.prefix[self.offset[answers]+np.maximum(local[:,0],0)],torch.float64))
        if with_target:
            target_position=positions+int(flow)
            if np.any(target_position>=self.length[answers]):raise ValueError('future training observation missing')
            target=(self.features[self.offset[answers]+target_position][:,self.columns]-self.mean[answers])/self.scale[answers]
            result['target']=tensor(target)
        return result

    def condition_from_probabilities(self,batch,logits):
        raw=probability_features(logits,batch['signs'])
        if self.bank=='innovation5':
            h0=torch.where(batch['mask'],raw[:,:,0],torch.zeros_like(raw[:,:,0]))
            before=batch['prefix'][:,None]+torch.cumsum(h0,dim=1)-h0
            innovation=torch.where(batch['local']>0,h0-before/batch['local'].clamp_min(1),torch.zeros_like(h0))
            raw=torch.cat((raw,innovation[:,:,None]),dim=-1)
        else:raw=raw[:,:,self.columns]
        z=(raw-batch['mean'])/batch['scale']
        z=torch.where(batch['mask'][:,:,None],z,torch.zeros_like(z)).float()
        return flow_condition(z,batch['mask'],batch['position'])
