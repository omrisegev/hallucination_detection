"""Separate full-population saved-score replay and historical-mask parity."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spectral_utils.digit_alternative_probability import digit_innovation_step_max
from scripts.run_digit_alternative_probability_v1 import sha,write,METHODS


def main():
    out=ROOT/'results/digit_alternative_probability_v1'
    reference=ROOT.parents[1]/'.worktrees/lsml-ct7-levers-run/spectral_utils/fusion_signal_registry.py'
    spec=importlib.util.spec_from_file_location('historical_digit_registry',reference)
    registry=importlib.util.module_from_spec(spec);sys.modules[spec.name]=registry;spec.loader.exec_module(registry)
    rng=np.random.default_rng(17)
    for _ in range(200):
        given=rng.integers(0,40,size=100);preferred=rng.integers(0,40,size=100)
        opp=np.isin(given,np.arange(15,25));event=(opp&np.isin(preferred,np.arange(15,25))&(given!=preferred)).astype(float)
        spans=np.array([[0,9],[9,20],[20,48],[48,100]])
        v,active=registry.digit_token_clock_innovation(event,opp)
        score,available=registry.readout_steps(v,spans,'top1',active_mask=active)
        got,got_active=digit_innovation_step_max(event,given,range(15,25),spans)
        np.testing.assert_allclose(got,score,atol=1e-14,rtol=0);np.testing.assert_array_equal(got_active,available)
    met=json.loads((out/'METRICS.json').read_text(encoding='utf-8'))
    audit=json.loads((out/'EXTRACTION_AUDIT.json').read_text(encoding='utf-8'))
    run=json.loads((out/'RUN.json').read_text(encoding='utf-8'))
    for p,h in run['code_sha256'].items():assert sha(ROOT/p)==h
    assert sha(ROOT/'docs/experiments/DIGIT_ALTERNATIVE_PROBABILITY_V1.md')==run['protocol_sha256']
    assert sha(out/'SCORES.npz')==met['scores_sha256']==audit['scores_sha256']
    records=json.loads((out/'RECORDS.json').read_text(encoding='utf-8'))
    f=np.load(out/'SCORES.npz');scores=f['scores'];offsets=f['offsets'];truth=f['labels'];target=f['target']
    with np.load(out/'ANSWER_METRICS.npz') as saved:
        saved_peaks=saved['peaks'];saved_within=saved['within']
    assert len(records)==13769 and offsets[-1]==145597
    assert np.all(np.isfinite(scores)) and np.all(scores[:,0]<=scores[:,-1]+1e-15)
    independent={};max_auc_diff=0.;mixed_counts=[]
    for j,name in enumerate(METHODS):
        pb={};within=[]
        for i,r in enumerate(records):
            x=scores[offsets[i]:offsets[i+1],j]
            # Python stable max returns the first tied index, matching declared rule.
            peak=max(range(len(x)),key=lambda k:float(x[k]));assert peak==saved_peaks[i,j]
            if r['cell'].startswith('pb_'):
                if target[i]>=0:pb.setdefault(r['cell'],[]).append(int(peak==target[i]))
            else:
                y=truth[offsets[i]:offsets[i+1]]
                if 0 in y and 1 in y:
                    pos=x[y==1];neg=x[y==0]
                    correct=sum(float(p>q)+.5*float(p==q) for p in pos for q in neg)/(len(pos)*len(neg))
                    max_auc_diff=max(max_auc_diff,abs(correct-saved_within[i,j]));within.append(correct)
        vals={'pb_macro_exact':sum(sum(v)/len(v) for v in pb.values())/len(pb),'prmb_within_auc':sum(within)/len(within)}
        for key,value in vals.items():assert abs(value-met['methods'][name][key])<1e-12
        independent[name]=vals;mixed_counts.append(len(within))
    assert max_auc_diff<1e-12
    result={'status':'PASS','n_checked':len(records),'n_total':13769,'steps_checked':int(offsets[-1]),
            'historical_mask_fixture_traces':200,'historical_registry_sha256':sha(reference),
            'prmb_mixed_answers_per_method':mixed_counts,'pairwise_auc_max_error':max_auc_diff,
            'score_protocol_code_hashes_verified':True,'metrics':independent,
            'scope':'separate implementation in the same agent/session; not an independent-agent review'}
    write(out/'REVIEW.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
