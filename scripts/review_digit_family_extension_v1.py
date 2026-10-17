"""Separate saved-prediction metric and learned-weight replay, full population."""
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_digit_alternative_probability_v1 import sha,rows
from spectral_utils.digit_feature_family import answer_standardize,NAMES
from spectral_utils.external_generalization.fusion import answer_z


def main():
    out=ROOT/'results/digit_family_extension_v1';old=ROOT/'results/digit_alternative_probability_v1';source=ROOT.parents[1]
    freeze=json.loads((out/'FREEZE.json').read_text());seal=json.loads((out/'SEAL.json').read_text())
    for p,h in freeze['hashes'].items():assert sha(ROOT/p)==h
    assert sha(out/'PREDICTIONS.npz')==seal['prediction_sha256']
    assert sha(out/'FITS.json')==seal['fits_sha256']
    meta=json.loads((out/'FIT_INPUTS.json').read_text());assert sha(Path(meta['level']['path']))==meta['level']['sha256']
    rec=json.loads((old/'RECORDS.json').read_text());m=json.loads((out/'METRICS.json').read_text())
    f=np.load(out/'PREDICTIONS.npz');data={k:f[k] for k in f.files};off=data['offsets'];sf=np.repeat(data['folds'],np.diff(off))
    raw=np.load(old/'SCORES.npz');y=raw['labels'];target=raw['target'];feat=np.load(out/'FEATURES.npz')
    digit=answer_standardize(feat['values'],off,feat['active'])
    base=answer_standardize(np.load(meta['level']['path'])['level'],off)
    banks={'bank11':base,'plus1':np.column_stack((base,digit[:,0])),
           'plus3':np.column_stack((base,digit)),'copies3':np.column_stack((base,np.repeat(digit[:,0,None],3,axis=1)))}
    maxerr=0.;checked=0
    for fit in json.loads((out/'FITS.json').read_text()):
        for name,bank in banks.items():
            score=bank@np.array(fit['fits'][name]['weights']);z=np.empty_like(score)
            for a,b in zip(off[:-1],off[1:]):z[a:b]=answer_z(score[a:b])
            tau=np.quantile(z[sf==fit['calibration']],.8);assert abs(tau-fit['thresholds'][name+'_lsml'])<1e-12
            held=sf==fit['test'];maxerr=max(maxerr,float(np.max(np.abs(z[held]-data[name+'_lsml_score'][held]))))
            np.testing.assert_array_equal((z[held]<tau),data[name+'_lsml_pred'][held]);checked+=int(held.sum())
    assert maxerr<1e-12
    prm={r['idx']:r for r in rows(source/'dataset_cache/four_localization/prmbench_qwen25math7b_full/prmbench_prm.pkl')}
    checked_auc=0
    for name in seal['names']:
        score=data[name+'_score'];pred=data[name+'_pred'];counts=np.zeros(4,int);aucvals=[];hit={}
        for i,r in enumerate(rec):
            a,b=off[i:i+2];s=score[a:b];truth=y[a:b];p=pred[a:b].astype(bool)
            if r['cell'].startswith('pb_'):
                if target[i]>=0:
                    peak=next(k for k,v in enumerate(s) if v>=max(s)-8*np.finfo(float).eps)
                    hit.setdefault(r['cell'],[]).append(int(peak==target[i]))
            else:
                if prm[r['row_id']]['classification']!='correct':
                    correct=truth==0;counts+=np.array([np.count_nonzero(correct&p),np.count_nonzero(~correct&p),
                                                       np.count_nonzero(~correct&~p),np.count_nonzero(correct&~p)])
                if 0 in truth and 1 in truth:
                    pos=s[truth==1];neg=s[truth==0]
                    aucvals.append(float(((pos[:,None]>neg).sum()+.5*(pos[:,None]==neg).sum())/(len(pos)*len(neg))))
                    checked_auc+=1
        tp,fp,tn,fn=counts;ps=.5*(2*tp/(2*tp+fp+fn)+2*tn/(2*tn+fp+fn))
        assert abs(ps-m['methods'][name]['prmscore'])<1e-12
        assert abs(np.mean(aucvals)-m['methods'][name]['within_auc'])<1e-12
        assert abs(np.mean([np.mean(v) for v in hit.values()])-m['methods'][name]['pb_macro_exact'])<1e-12
    result={'status':'PASS','n_checked':len(rec),'n_total':13769,'steps':int(off[-1]),'methods':len(seal['names']),
            'lsml_prediction_steps_replayed':checked,'max_score_replay_error':maxerr,'pairwise_auc_rows_checked':checked_auc,
            'source_protocol_hashes':'PASS','scope':'separate implementation, same agent/session'}
    (out/'REVIEW.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
