"""Audit descriptive claims, preserving Claude's source files and results."""
from pathlib import Path
import sys,json,csv,sqlite3,io,importlib.util
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts import run_temporal_research_baseline as base
from scripts.run_predictor_subset_study import sha,write
OUT=ROOT/'results/readout_claim_audit_v1';SRC=ROOT/'results/claude_real_checks_v1'

def run():
    OUT.mkdir(exist_ok=True)
    source=SRC/'claude_miss_readout_ceiling.py';spec=importlib.util.spec_from_file_location('claude_readout_audit_source',source);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    records,joined=base.load_contract(ROOT.parents[1]);target=joined['target'];idx={r['uid']:i for i,r in enumerate(records)}
    with (SRC/'MISS_READOUT_CEILING_ROWS.csv').open(encoding='utf8') as f:old={int(r['idx']):r for r in csv.DictReader(f)}
    expected=[i for i,r in enumerate(records) if r['cell'].startswith('pb_') and target[i]>=0];assert set(old)==set(expected)
    con=sqlite3.connect('file:'+str(ROOT/'results/temporal_research_baseline_v1/CHECKPOINT.sqlite').replace('\\','/')+'?mode=ro',uri=True)
    saved=json.loads((SRC/'MISS_READOUT_CEILING.json').read_text());values={g:[] for g in saved};replay={};maxdiff=0.
    with np.load(ROOT/'results/predictor_subset_iu_v1/SCORES_FROZEN.npz') as f:base_scores=f['innovation5']
    for i in expected:
        blob=con.execute('SELECT payload FROM answers WHERE idx=?',(i,)).fetchone()[0]
        with np.load(io.BytesIO(blob),allow_pickle=False) as f:M=f['features'].astype(float);spans=f['spans']
        streams=np.column_stack((M[:,:4],mod.prefix_innovation(M[:,0])[0]));T=len(streams);t=int(target[i]);a,b=map(int,spans[t]);L=b-a
        # Exact scalar readout replay: labels enter evaluation and oracle selection.
        per=[mod.step_readouts(streams[:,s],spans) for s in range(5)]
        fused={k:np.mean([p[k] for p in per],axis=0) for k in mod.READOUTS}
        ranks={k:mod.rank_of(s,t) for k,s in fused.items()}
        for k,v in ranks.items():assert v==int(old[i]['rank_'+k]),(i,k,v,old[i]['rank_'+k])
        assert min(ranks.values())==int(old[i]['rank_oracle'])
        sl=slice(joined['offsets'][i],joined['offsets'][i+1]);delta=float(np.max(np.abs(fused['top10']-base_scores[sl])));maxdiff=max(maxdiff,delta);assert delta<1e-10
        z=(streams-streams.mean(0))/np.maximum(streams.std(0),1e-12);p=np.argmax(z,axis=0)
        observed=bool(np.any((p>=a)&(p<b)));assert observed==(old[i]['contains_max']=='True')
        # One common cyclic shift of ALL streams. Exact enumeration preserves
        # peak coincidences, pairwise offsets and chronological shape.
        covered=np.zeros(T,bool)
        for q in p:covered[(np.arange(a,b)-int(q))%T]=True
        shared_shift=float(covered.mean())
        # One common arbitrary permutation preserves identical peak tokens.
        distinct=len(set(map(int,p)));not_hit=1.
        for q in range(distinct):not_hit*=max(T-L-q,0)/(T-q)
        independent=1-(1-L/T)**5
        values[old[i]['cohort']].append(dict(idx=i,observed=observed,share=L/T,independent=independent,
            common_circular_shift=shared_shift,common_permutation=1-not_hit,distinct_peak_tokens=distinct,
            last4_hit=ranks['last4']==1,eight_readout_oracle_hit=min(ranks.values())==1))
    con.close()
    for group,rows in values.items():
        replay[group]=dict(n=len(rows),observed_max_hit=float(np.mean([r['observed'] for r in rows])),
            independent_five_peak_reference=float(np.mean([r['independent'] for r in rows])),
            common_shift_reference=float(np.mean([r['common_circular_shift'] for r in rows])),
            common_permutation_reference=float(np.mean([r['common_permutation'] for r in rows])),
            single_uniform_peak_reference=float(np.mean([r['share'] for r in rows])),
            mean_distinct_peak_tokens=float(np.mean([r['distinct_peak_tokens'] for r in rows])),
            last4_hits=sum(r['last4_hit'] for r in rows),eight_readout_oracle_hits=sum(r['eight_readout_oracle_hit'] for r in rows))
        np.testing.assert_allclose(replay[group]['observed_max_hit'],saved[group]['contains_answer_max_token_any_stream'],atol=1e-14)
        np.testing.assert_allclose(replay[group]['independent_five_peak_reference'],saved[group]['chance_contains_max'],atol=1e-14)
    metrics=json.loads((ROOT/'results/predictor_subset_iu_v1/METRICS.json').read_text())['metrics']
    gates={n:dict(correct_peak_suppressed=int(metrics[n]['pb_correct_peaks_suppressed'])) for n in ['innovation5','ridge','tcn__real','iu__ridge+tcn+noreset']}
    data=dict(status='PASS',answers=len(expected),all8_readout_ranks_replayed=True,max_innovation5_score_delta=maxdiff,
        cohorts=replay,actual_method_gate_headroom=gates,source_sha256=sha(source),source_rows_sha256=sha(SRC/'MISS_READOUT_CEILING_ROWS.csv'),
        source_summary_sha256=sha(SRC/'MISS_READOUT_CEILING.json'),code_sha256=sha(Path(__file__)),
        null_scope='Geometric references conditional on fixed previously selected cohorts. Shared shifts preserve dependence, but neither reference accounts for selection as misses, step-boundary nonstationarity or semantic labels. No below-chance significance or information-impossibility claim.',
        all_common_misses_last4_hits=replay['loc_miss_open']['last4_hits']+replay['loc_miss_closed']['last4_hits'],
        all_common_misses_eight_readout_oracle_hits=replay['loc_miss_open']['eight_readout_oracle_hits']+replay['loc_miss_closed']['eight_readout_oracle_hits'])
    write(OUT/'AUDIT.json',data)
    print(json.dumps(data,indent=2),flush=True)

if __name__=='__main__':run()
