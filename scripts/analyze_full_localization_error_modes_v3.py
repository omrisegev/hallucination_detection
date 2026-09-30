"""Full frozen-output failure accounting; descriptive, with no candidate fitting."""
import hashlib
import html
import json
from pathlib import Path
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'results/historical_fusion_refit_v3'
OUT=ROOT/'results/localization_full_error_modes_v3'
ARMS=('dual__equal','dual__iu','dual__joint0','dual__graph010','dual__graph_perm',
      'iu_c2_s25_l2_exoff','fixed_family_cont_unguarded','equal_all23')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path,data):
    tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)


def localization_curves(meta,a):
    """Use the actual PB decision readout, which differs from PRMB spanmax."""
    columns=[meta['arms'].index(arm) for arm in ARMS]
    curves=a['scores'][:,columns].copy();records=meta['records']
    lookup={(r['cell'],r['row']):i for i,r in enumerate(records)}
    frozen=load(SOURCE/'SCORES_FROZEN.json');files={};count=0
    for cell in sorted({r['cell'] for r in records if r['cell'].startswith('pb_')}):
        for outer in range(5):
            path=SOURCE/'fits'/cell/('outer'+str(outer))/'test.npz'
            for p in (path,path.with_suffix('.json')):
                files[str(p)]=sha(p);assert files[str(p)]==frozen['files'][str(p)]
            with np.load(path,allow_pickle=False) as z:
                rows=z['rows'];ix=[lookup[cell,int(r)] for r in rows]
                offsets=np.r_[0,np.cumsum([records[i]['steps'] for i in ix])]
                for arm in ARMS[5:]:
                    values=z[arm+'__top10'];assert len(values)==offsets[-1]
                    for q,i in enumerate(ix):
                        lo,hi=a['offsets'][i:i+2];u,v=offsets[q:q+2]
                        curves[lo:hi,ARMS.index(arm)]=values[u:v]
                count+=len(ix)
    assert count==6800
    return curves,files


def main():
    OUT.mkdir(parents=True,exist_ok=True);started=time.time()
    assert load(SOURCE/'REVIEW_SUPPLEMENT.json')['status']=='PASS'
    meta=load(SOURCE/'JOINED.json');assert sha(SOURCE/'JOINED.npz')==meta['arrays_sha256']
    paths=[Path(__file__),SOURCE/'JOINED.json',SOURCE/'JOINED.npz',SOURCE/'METRICS.json',SOURCE/'REVIEW_SUPPLEMENT.json',SOURCE/'SCORES_FROZEN.json']
    hashes={str(p):sha(p) for p in paths}
    if (OUT/'MANIFEST_V2.json').exists():assert load(OUT/'MANIFEST_V2.json')['hashes']==hashes
    else:save(OUT/'MANIFEST_V2.json',dict(hashes=hashes,arms=ARMS,scope='full frozen development outputs; actual PB readout curves',
        correction='Initial diagnostic assertion caught historical PB top10 peaks versus saved PRMB spanmax curves; no scientific output changed.'))
    records=meta['records'];assert len(records)==13769
    with np.load(SOURCE/'JOINED.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
    curves,readout_hashes=localization_curves(meta,a)
    cells=np.array([r['cell'] for r in records]);target=a['target'];metrics=load(SOURCE/'METRICS.json')['metrics']
    pb_cells=sorted(set(cells[np.char.startswith(cells,'pb_')]))
    prefix=np.full((len(records),len(ARMS)),np.nan)
    for i,rec in enumerate(records):
        if not rec['cell'].startswith('pb') or target[i]<=0:continue
        lo,hi=a['offsets'][i:i+2];truth=target[i]
        for column,arm in enumerate(ARMS):
            j=meta['arms'].index(arm)
            if not a['valid'][i,j]:continue
            risk=curves[lo:hi,column];value=risk[truth]
            prefix[i,column]=float(np.mean((value>risk[:truth])+.5*(value==risk[:truth])))
    result={};checks=0
    for column,arm in enumerate(ARMS):
        j=meta['arms'].index(arm);valid=a['valid'][:,j];decision=a['decision'][:,j]
        peak=a['peaks'][:,j];prediction=a['predictions'][:,j];table={}
        for cell in pb_cells:
            mask=cells==cell;error=mask & (target>=0);clean=mask & (target==-1)
            count=lambda condition:int(condition.sum())
            d=dict(answers=count(mask),erroneous=count(error),clean=count(clean),
                error_step_zero=count(error & (target==0)),
                peak_invalid=count(error & ~valid),peak_exact=count(error & valid & (peak==target)),
                peak_before=count(error & valid & (peak<target)),peak_after=count(error & valid & (peak>target)),
                error_decision_invalid=count(error & ~decision),
                error_final_exact=count(error & decision & (prediction==target)),
                error_called_clean=count(error & decision & (prediction==-1)),
                error_wrong_step=count(error & decision & (prediction>=0) & (prediction!=target)),
                exact_peak_suppressed=count(error & valid & (peak==target) & decision & (prediction==-1)),
                exact_peak_invalid_decision=count(error & valid & (peak==target) & ~decision),
                clean_correct=count(clean & decision & (prediction==-1)),
                clean_false_alarm=count(clean & decision & (prediction>=0)),
                clean_invalid=count(clean & ~decision))
            values=prefix[mask,column];values=values[np.isfinite(values)]
            d['first_error_vs_prefix_answers']=len(values)
            d['first_error_vs_prefix_auc']=float(values.mean()) if len(values) else None
            # Independent row-by-row peak categories, rather than the vector masks above.
            counted=dict(invalid=0,exact=0,before=0,after=0)
            for i in np.flatnonzero(error):
                if not valid[i]:category='invalid'
                else:
                    lo,hi=a['offsets'][i:i+2];actual=int(np.argmax(curves[lo:hi,column]))
                    assert actual==peak[i]
                    category='exact' if actual==target[i] else ('before' if actual<target[i] else 'after')
                counted[category]+=1
            for key,value in counted.items():assert d['peak_'+key]==value
            assert sum(d[k] for k in ('peak_invalid','peak_exact','peak_before','peak_after'))==d['erroneous']
            assert sum(d[k] for k in ('error_decision_invalid','error_final_exact','error_called_clean','error_wrong_step'))==d['erroneous']
            assert d['peak_exact']==d['error_final_exact']+d['exact_peak_suppressed']+d['exact_peak_invalid_decision']
            assert sum(d[k] for k in ('clean_correct','clean_false_alarm','clean_invalid'))==d['clean']
            old=metrics[arm]['pb']['cells'][cell]
            assert d['error_final_exact']/d['erroneous']==old['error_exact_accuracy']
            assert d['clean_correct']/d['clean']==old['clean_accuracy']
            table[cell]=d;checks+=1
        result[arm]=dict(access=metrics[arm]['access'],cells=table)
    rows=[]
    for arm,table in result.items():
        for cell,d in table['cells'].items():
            values=(arm,cell,d['erroneous'],d['peak_before'],d['peak_exact'],d['peak_after'],
                    d['peak_invalid'],d['exact_peak_suppressed'],d['error_final_exact'],
                    str(d['clean_false_alarm'])+'/'+str(d['clean']),
                    f"{d['first_error_vs_prefix_auc']:.4f}" if d['first_error_vs_prefix_auc'] is not None else 'N/A')
            rows.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in values)+'</tr>')
    report='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Full localization error modes</title><style>body{font:16px/1.55 system-ui;max-width:1400px;margin:30px auto;padding:20px;color:#20344b}table{border-collapse:collapse;font-size:13px}th,td{padding:8px;border-bottom:1px solid #ccd6e0;text-align:left}.scroll{overflow:auto}.note{padding:20px;background:#fff2d5}</style>
<h1>Where does localization fail on the full benchmark?</h1><p class="note">All 6,800 cached ProcessBench model-answer rows, inside the verified 13,769-record benchmark. Eight fixed references. Descriptive analysis of already measured outputs: no new model, tuning, oracle method or performance claim.</p>
<p>Before/exact/after/invalid partition all erroneous answers by their raw peak. Suppressed means a correct raw peak was followed by a no-error decision. Final exact counts include the gate. A clean false alarm is an answer with no annotated error that the method calls erroneous. The CSV-style source counts are in the JSON below.</p>
<p>First-error versus prefix AUC compares the first erroneous step only with its earlier correct steps, then averages over supported erroneous answers with a nonempty prefix. Step-zero errors are excluded explicitly. Later steps have no added correctness labels. This conditional diagnostic differs from exact first-error/no-error accuracy and from PRMB's per-step ranking.</p>
<p>The five dual methods fit one answer alone; Joint graphs here use the original condition1000. Historical IU, fixed-family CONT and equal_all23 fit other training answers and use nested PB label calibration. Their PB location uses the mean of the ten largest token risks within each step; their saved PRMB ranking curve uses span maximum. This analysis reloads the actual historical PB top-10 readout before checking peaks or prefix rankings. Differences in representation, fit scope and readout prevent attributing their whole gap to a fusion formula or to the gate alone.</p>
<div class="scroll"><table><tr><th>Method</th><th>Cell</th><th>Errors</th><th>Peak before</th><th>Exact peak</th><th>Peak after</th><th>Invalid</th><th>Exact suppressed</th><th>Final exact</th><th>Clean false alarms</th><th>First error vs prefix</th></tr>ROWS</table></div>
<p><a href="FINDINGS.json">All counts and denominators</a> · <a href="REVIEW.json">Review</a> · <a href="../historical_fusion_refit_v3/REPORT.html">Matched benchmark and uncertainty</a></p></html>'''.replace('ROWS',''.join(rows))
    (OUT/'REPORT.html').write_text(report,encoding='utf-8')
    save(OUT/'FINDINGS.json',dict(scope='descriptive full-data accounting; no causal attribution',methods=result,records=13769,pb_records=6800,
        readout='answer-only dense window-to-step risks; historical actual PB top10, not PRMB spanmax'))
    for path,h in hashes.items():assert sha(path)==h,path
    assert report.count('<tr>')==65
    save(OUT/'REVIEW.json',dict(status='PASS',method_cell_bundles=checks,independent_peak_counts=True,
        exact_final_metric_replays=True,partition_checks=True,source_hashes=hashes,readout_file_hashes=readout_hashes,seconds=time.time()-started,
        report_sha256=sha(OUT/'REPORT.html'),findings_sha256=sha(OUT/'FINDINGS.json'),
        scope='same-session arithmetic/provenance review; descriptive, not causal or external confirmation'))
    print('Full error-mode accounting PASS:',checks,'method/cell bundles',flush=True)


if __name__=='__main__':main()
