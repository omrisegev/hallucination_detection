"""Independent raw/alignment, projection, outcome and oracle review."""
from collections import Counter,defaultdict
from copy import deepcopy
import importlib.util
from pathlib import Path
import time
import numpy as np
from scipy.stats import spearmanr
from tokenizers import Tokenizer

ROOT=Path(__file__).resolve().parents[1]


def module(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


d=module(ROOT/'scripts/audit_fusion_localization_forensics_v3.py','forensics_v3_driver')
OUT=d.OUT
load,save,sha=d.load,d.save,d.sha


class TokenizerAPI:
    def __init__(self,path):self.tokenizer=Tokenizer.from_file(path)
    def __call__(self,text,**kwargs):
        e=self.tokenizer.encode(text,add_special_tokens=kwargs.get('add_special_tokens',False))
        return {'input_ids':e.ids,'offset_mapping':e.offsets}


def outcome(target,pred,peak,tied):
    clean=target==-1
    if clean:category='clean_correct' if pred==-1 else 'clean_false_alarm'
    elif pred==target:category='error_exact'
    elif pred==-1:category='error_gate_closed'
    else:category='error_wrong_step'
    return dict(category=category,correct=pred==target,raw_peak_exact=None if clean else peak==target,
        raw_peak_before=None if clean else peak<target,raw_peak_after=None if clean else peak>target,
        target_in_top_tie=None if clean else target in tied,exact_peak_hidden=not clean and peak==target and pred==-1)


def main():
    start=time.monotonic();m=d.verify();a=load(OUT/'AUDIT.json');assert a['manifest_sha256']==sha(OUT/'MANIFEST.json')
    e=load(m['corrected_evaluation_path']);ei={r['uid']:r for r in e['rows']};ri={r['uid']:r for r in a['records']}
    meta=load(m['raw_metadata_path']);mi={r['uid']:r for r in meta['rows']};assert set(ei)==set(ri)==set(mi)
    align=module(ROOT/'spectral_utils/processbench.py','forensics_independent_alignment')
    prm=module(ROOT/'spectral_utils/prmbench.py','forensics_official_prm_labels')
    metrics=module(ROOT/'scripts/review_fusion_explicit_fallback_v1.py','forensics_independent_metrics')
    tok=TokenizerAPI(m['tokenizer_path']);counts=Counter();maxdiff=0.
    # Re-read the raw containers; don't trust the extraction file for joins.
    for cell,path in d.old.RAW_FILES.items():
        with path.open('rb') as f:raw=d.old.metadata.MetadataUnpickler(f).load()
        index={(v['idx'] if cell.startswith('prm') else cell.split('_')[1]+'::'+str(v['id'])):v for v in raw.values()}
        for rec in (r for r in a['records'] if r['cell']==cell):
            uid=rec['uid'];source=index[rec['row_id']];extracted=mi[uid];row=ei[uid]
            assert rec['text_steps']==source['steps'];assert rec['question']==source.get('question',source.get('problem'))
            text,chars=align.build_chain(source['steps']);ids,spans=align.step_token_spans(tok,text,chars)
            assert ids==source['gen_token_ids']==extracted['gen_token_ids'];assert list(map(list,spans))==rec['spans']==extracted['raw_spans']
            assert list(map(list,chars))==rec['char_spans'];assert tok(text,add_special_tokens=False)['offset_mapping']==list(map(tuple,rec['token_offsets']))
            diag=align.assert_alignment(ids,spans,source['steps'],strict=True);assert not diag['problems']
            if cell.startswith('prm'):
                flags=1-np.array(prm.eval_on_hallucination_step(source['error_steps'],[1]*len(spans))['total_step_acc_list'])
                np.testing.assert_array_equal(flags,rec['target'])
            else:assert source['label']==rec['target']
            assert rec['target']==row['target']
            with np.load(d.ORIGINAL/'inputs'/(uid+'.npz'),allow_pickle=False) as inp:
                for name,col in [('token_entropies',1),('token_spilled_energies',15),('token_logsumexp',19)]:
                    np.testing.assert_array_equal(source[name],inp['raw'][:,col]);counts['raw_stream_replays']+=1
            counts['raw_text_token_span_label_replays']+=1
        del raw,index
    for uid,rec in ri.items():
        token_step=np.full(rec['tokens'],-1,int)
        for i,(lo,hi) in enumerate(rec['spans']):token_step[lo:hi]=i
        assert int(np.sum(token_step<0))==rec['separator_tokens']
        for arm,info in rec['methods'].items():
            step=np.array(ei[uid]['scores'][arm]);np.testing.assert_array_equal(step,info['step_scores'])
            starts,ends=np.array(info['window_starts']),np.array(info['window_ends']);windows=np.array(info['window_risk'])
            incidence=(np.arange(rec['tokens'])[:,None]>=starts)&(np.arange(rec['tokens'])[:,None]<ends)
            assert (incidence.sum(axis=1)>0).all();token=(incidence@windows)/incidence.sum(axis=1)
            projected=np.array([max(token[x:y]) for x,y in rec['spans']]);maxdiff=max(maxdiff,float(np.max(np.abs(projected-step))))
            np.testing.assert_allclose(projected,step,atol=1e-12,rtol=1e-12)
            order=sorted(range(len(step)),key=lambda i:(-step[i],i));peak=order[0];tolerance=1e-12*max(1.,abs(step[peak]))
            exact=[i for i,v in enumerate(step) if v==step[peak]];near=[i for i,v in enumerate(step) if step[peak]-v<=tolerance]
            assert peak==info['peak']==ei[uid]['peaks'][arm];assert near==info['numerical_top_steps'] and exact==info['exact_top_steps']
            margin=float(step[order[0]]-step[order[1]]) if len(step)>1 else None
            metrics.check_equal(info['top_two_margin'],margin);assert info['margin_le_0_1']==(margin is not None and margin<=.1)
            sets=defaultdict(set);top_tokens=[]
            for t,value in enumerate(token):
                if token_step[t] in near and abs(value-step[peak])<=tolerance:
                    sets[tuple(np.flatnonzero(incidence[t]))].add(int(token_step[t]));top_tokens.append(t)
            plates=[dict(windows=list(k),steps=sorted(v)) for k,v in sorted(sets.items()) if len(v)>1]
            assert plates==info['shared_peak_plateaus'];assert top_tokens==info['peak_token_indices']
            overlaps=[set(token_step[lo:hi])-{-1} for lo,hi in zip(starts,ends)]
            assert [i for i,v in enumerate(overlaps) if len(v)>1]==info['cross_step_windows']
            assert [sum(i in v for v in overlaps) for i in range(rec['steps'])]==info['window_counts_per_step']
            lengths=np.array([hi-lo for lo,hi in rec['spans']]);assert lengths.tolist()==info['step_lengths']
            rho=float(spearmanr(lengths,step).statistic) if len(step)>1 and np.ptp(lengths)>0 and np.ptp(step)>0 else None
            metrics.check_equal(info['length_score_spearman'],rho)
            detail=load((d.old.GRAPH if arm in d.old.OLD_ARMS else d.old.PARENT)/'scores'/(uid+'.json'))['methods'][arm]
            assert detail['prediction']==info['prediction']==ei[uid]['predictions'][arm]
            assert detail['gate']['bic']==info['bic'];assert info['two_components_selected']==(info['bic'][1]<info['bic'][0])
            assert info['bic_advantage_two']==info['bic'][0]-info['bic'][1]
            if rec['cell'].startswith('pb'):
                assert info['outcome']==outcome(rec['target'],info['prediction'],peak,near)
                if rec['target']!=-1:
                    target=rec['target'];assert info['true_step_rank_best']==1+sum(step>step[target])
                    assert info['peak_minus_true_score']==step.max()-step[target]
                    assert info['true_step_length']==lengths[target] and info['peak_step_length']==lengths[peak]
            counts['independent_trajectory_geometry_replays']+=1
    for arm,bundle in a['historical_metrics'].items():
        metrics.check_equal(bundle['prm'],metrics.prm(e['rows'],arm));metrics.check_equal(bundle['pb'],metrics.pb(e['rows'],arm))
        counts['independent_historical_metrics']+=1
    # Recompute native and oracle counts independently of the production taxonomy.
    for arm,summary in a['summary'].items():
        pb=[r for r in a['records'] if r['cell'].startswith('pb')];err=[r for r in pb if r['target']!=-1]
        categories=Counter(outcome(r['target'],r['methods'][arm]['prediction'],r['methods'][arm]['peak'],r['methods'][arm]['numerical_top_steps'])['category'] for r in pb)
        assert dict(categories)==summary['native_categories']
        for field,test in [('raw_error_peak_exact',lambda r:r['methods'][arm]['peak']==r['target']),
            ('raw_error_peak_before',lambda r:r['methods'][arm]['peak']<r['target']),('raw_error_peak_after',lambda r:r['methods'][arm]['peak']>r['target']),
            ('error_top_tie_contains_target',lambda r:r['target'] in r['methods'][arm]['numerical_top_steps']),
            ('exact_peak_hidden_by_gate',lambda r:r['methods'][arm]['peak']==r['target'] and r['methods'][arm]['prediction']==-1)]:
            assert summary[field]==sum(test(r) for r in err),field
        for field,test in [('all_pb_multiple_exact_top_steps',lambda q:len(q['exact_top_steps'])>1),
            ('all_pb_multiple_numerical_top_steps',lambda q:len(q['numerical_top_steps'])>1),('all_pb_shared_top_plateaus',lambda q:bool(q['shared_peak_plateaus'])),
            ('all_pb_small_margin',lambda q:q['margin_le_0_1'])]:assert summary[field]==sum(test(r['methods'][arm]) for r in pb)
        for kind,target_metric in summary['oracle_diagnostics'].items():
            rows=deepcopy(e['rows'])
            for row in rows:
                if not row['cell'].startswith('pb'):continue
                q=ri[row['uid']]['methods'][arm];gold=row['target'];pred=q['prediction']
                if kind=='perfect_binary_gate':pred=q['peak'] if gold>=0 else -1
                if kind=='perfect_locator' and pred>=0 and gold>=0:pred=gold
                if kind=='top_tie_locator' and pred>=0 and gold in q['numerical_top_steps']:pred=gold
                row['predictions'][arm]=pred
            metrics.check_equal(target_metric,metrics.pb(rows,arm));counts['independent_oracle_metrics']+=1
        counts['native_aggregate_replays']+=1
    for pair,transition in a['transitions'].items():
        new,old=pair.split(' minus ');pb=[r for r in a['records'] if r['cell'].startswith('pb')]
        expected={field:[] for field in transition}
        for r in pb:
            n,o=r['methods'][new],r['methods'][old];target=r['target'];uid=r['uid']
            tests={'native_lost_correct':o['prediction']==target and n['prediction']!=target,'native_gained_correct':n['prediction']==target and o['prediction']!=target,
                'raw_peak_lost':target!=-1 and o['peak']==target and n['peak']!=target,'raw_peak_gained':target!=-1 and n['peak']==target and o['peak']!=target,
                'gate_changed':(n['prediction']==-1)!=(o['prediction']==-1),'peak_changed':n['peak']!=o['peak'],
                'old_margin_small_when_peak_changed':n['peak']!=o['peak'] and o['margin_le_0_1']}
            for field,yes in tests.items():
                if yes:expected[field].append(uid)
        assert transition==expected;counts['transition_bundles']+=1
    dependencies=[Path(__file__),ROOT/'spectral_utils/processbench.py',ROOT/'spectral_utils/prmbench.py',ROOT/'scripts/review_fusion_explicit_fallback_v1.py']
    save(OUT/'REVIEW.json',{'status':'PASS','counts':dict(counts),'maximum_projection_discrepancy':maxdiff,
        'scope':'Same-session review. Original raw containers, existing character/token-overlap API, incidence-matrix token projection, independent pairwise-AUC/PB counts and oracle reconstruction. Tokenizer, saved fusion scores, official PRMB port and metadata reader shared; no new inference, original logit-position or top-K fidelity audit.',
        'hashes':{str(OUT/n):sha(OUT/n) for n in ('MANIFEST.json','AUDIT.json')},
        'review_dependencies':{str(p):sha(p) for p in dependencies},'seconds':time.monotonic()-start})
    print('Review PASS',dict(counts),'max',maxdiff,flush=True)


if __name__=='__main__':main()
