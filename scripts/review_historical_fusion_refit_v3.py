"""Independent arithmetic/artifact review of the complete first historical panel."""
import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import time
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/historical_fusion_refit_v3'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))


class Tables(HTMLParser):
    def __init__(self):super().__init__();self.tables=[];self.current=None;self.row=None;self.cell=None;self.links=[]
    def handle_starttag(self,tag,attrs):
        if tag=='table':self.current=[];self.tables.append(self.current)
        elif tag=='tr':self.row=[]
        elif tag in ('th','td'):self.cell=[]
        elif tag=='a':
            self.links.extend(v for k,v in attrs if k=='href')
    def handle_data(self,data):
        if self.cell is not None:self.cell.append(data)
    def handle_endtag(self,tag):
        if tag in ('th','td') and self.cell is not None:self.row.append(''.join(self.cell));self.cell=None
        elif tag=='tr' and self.row is not None:self.current.append(self.row);self.row=None


def main():
    started=time.time();manifest=load(OUT/'MANIFEST.json');joined=load(OUT/'JOINED.json');reported=load(OUT/'METRICS.json')['metrics']
    assert sha(OUT/'JOINED.npz')==joined['arrays_sha256']
    assert load(OUT/'REVIEW.json')['status']=='PASS'
    with np.load(OUT/'JOINED.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
    records,arms=joined['records'],joined['arms'];cells=np.array([r['cell'] for r in records]);prm=np.char.startswith(cells,'prm')
    owner=np.repeat(np.arange(len(records)),np.diff(a['offsets']));checks=0
    for j,arm in enumerate(arms):
        d=reported[arm]
        for fold in range(5):
            mask=(prm & (a['outer']==fold) & a['valid'][:,j])[owner]
            expected=roc_auc_score(a['labels'][mask],a['scores'][mask,j])
            np.testing.assert_allclose(expected,d['prm']['fold_aucs'][fold],atol=1e-14);checks+=1
        for cell,c in d['pb']['cells'].items():
            ix=np.flatnonzero(cells==cell);nc=ne=gc=ge=gp=nv=0
            for i in ix:
                target=int(a['target'][i]);prediction=int(a['predictions'][i,j]);valid=bool(a['decision'][i,j])
                nv+=int(valid)
                if target==-1:nc+=1;gc+=int(valid and prediction==-1)
                else:ne+=1;ge+=int(valid and prediction==target);gp+=int(a['valid'][i,j] and a['peaks'][i,j]==target)
            ca,ea=gc/nc,ge/ne;f1=2*ca*ea/(ca+ea) if ca+ea else 0.
            np.testing.assert_allclose([ca,ea,f1,gp/ne],[c['clean_accuracy'],c['error_exact_accuracy'],c['f1'],c['raw_peak_accuracy']],atol=1e-14)
            assert nv==c['valid_decisions'] and nc+ne==c['answers'];checks+=1
    # Verify 25 selected calibration thresholds by independent direct counting.
    thresholds=load(OUT/'THRESHOLDS.json');canonical=load(ROOT/'results/localization_source_group_audit_v1/FOLDS_V2.json')
    row_lookup={(r['cell'],r['row']):i for i,r in enumerate(records)};calibration_checks=0
    for fold in range(5):
        accum={arm:[] for arm in manifest['arms']}
        for job in manifest['jobs']:
            if job['outer']!=fold or job['inner'] is None:continue
            name=job['cell']+'/outer'+str(fold)+'/inner'+str(job['inner']);path=OUT/'fits'/name
            with np.load(path.with_suffix('.npz'),allow_pickle=False) as z:
                indices=[row_lookup[job['cell'],int(r)] for r in z['rows']];local=np.r_[0,np.cumsum([records[i]['steps'] for i in indices])]
                for arm in manifest['arms']:
                    for q,i in enumerate(indices):
                        assert canonical['outer'][records[i]['group_id']]!=fold
                        lo,hi=local[q:q+2];scores=z[arm+'__top10'][lo:hi]
                        accum[arm].append((job['cell'],float(z[arm+'__detector'][q]),int(np.argmax(scores)),int(a['target'][i])))
        for arm,rows in accum.items():
            threshold=thresholds[str(fold)+'/'+arm]
            grid=np.quantile([r[1] for r in rows],np.linspace(.01,.99,99));values=[]
            cell_order=sorted(set(r[0] for r in rows))
            for tau in grid:
                counts={cell:[0,0,0,0] for cell in cell_order}
                for cell,detector,peak,target in rows:
                    clean=target==-1;prediction=peak if detector>=tau else -1
                    counts[cell][0 if clean else 1]+=1
                    counts[cell][2 if clean else 3]+=int(prediction==target)
                cell_values=[]
                for nc,ne,gc,ge in counts.values():
                    ca,ea=gc/nc,ge/ne;cell_values.append(2*ca*ea/(ca+ea) if ca+ea else 0.)
                values.append(sum(cell_values)/len(cell_values))
            selected=int(np.argmax(values));assert selected==threshold['grid_index']
            assert float(grid[selected])==threshold['threshold']
            np.testing.assert_allclose(values[selected],threshold['training_macro'],atol=1e-14);calibration_checks+=1
    parser=Tables();parser.feed((OUT/'REPORT.html').read_text(encoding='utf-8'))
    assert len(parser.tables)==3 and len(parser.tables[0])==len(arms)+1
    for row,arm in zip(parser.tables[0][1:],arms):
        d=reported[arm];assert row[0]==arm
        expected=[f"{d['prm']['fold_mean_auc']:.4f}",f"{d['prm']['within_answer_auc']:.4f}",
                  str(d['prm']['valid_answers'])+'/'+str(d['prm']['total_answers'])]
        expected += [f"{100*d['pb']['macros'][panel]:.2f}%" for panel in ('q4','q8','all')]
        assert row[2:]==expected
    for link in parser.links:assert (OUT/link).resolve().exists()
    inputs=[OUT/'MANIFEST.json',OUT/'JOINED.json',OUT/'JOINED.npz',OUT/'METRICS.json',OUT/'THRESHOLDS.json',OUT/'REPORT.html']
    result=dict(status='PASS',scope='Independent full metric arithmetic, calibration search and HTML table checks',
        metric_bundles=checks,thresholds=calibration_checks,html_metric_rows=len(arms),local_links=len(parser.links),
        seconds=time.time()-started,external_review=False,browser_review=False,source_hashes={str(p):sha(p) for p in inputs})
    (OUT/'REVIEW_SUPPLEMENT.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


def review_within():
    """Compare all new within-answer AUCs with explicit positive/negative pairs."""
    started=time.time();joined=load(OUT/'JOINED.json');reported=load(OUT/'METRICS.json')['metrics']
    assert sha(OUT/'JOINED.npz')==joined['arrays_sha256']
    # Materialize once: repeated NPZ indexing would decompress for every answer.
    with np.load(OUT/'JOINED.npz',allow_pickle=False) as z:
        a={key:z[key] for key in ('scores','labels','offsets','valid','within')}
    checked=0
    for arm in load(OUT/'MANIFEST.json')['arms']:
        j=joined['arms'].index(arm);values=[]
        for i,rec in enumerate(joined['records']):
            if not rec['cell'].startswith('prm') or not a['valid'][i,j]:continue
            lo,hi=a['offsets'][i:i+2];labels=a['labels'][lo:hi];scores=a['scores'][lo:hi,j]
            pos=scores[labels==1];neg=scores[labels==0]
            if len(pos) and len(neg):
                value=((pos[:,None]>neg).sum()+.5*(pos[:,None]==neg).sum())/(len(pos)*len(neg))
                assert abs(value-a['within'][i,j])<1e-14;values.append(float(value));checked+=1
            else:assert not np.isfinite(a['within'][i,j])
        assert abs(np.mean(values)-reported[arm]['prm']['within_answer_auc'])<1e-14
    review=load(OUT/'REVIEW_SUPPLEMENT.json');review['independent_within_answer_pair_checks']=checked
    review['within_check_seconds']=time.time()-started
    (OUT/'REVIEW_SUPPLEMENT.json').write_text(json.dumps(review,indent=2),encoding='utf-8')
    print('Explicit positive/negative-pair AUC review PASS:',checked,'answers',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--within-only',action='store_true');args=parser.parse_args()
    if args.within_only:review_within()
    else:main()
