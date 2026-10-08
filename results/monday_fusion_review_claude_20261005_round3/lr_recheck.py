import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='2'
import sys, json, io
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path(r'C:\Users\omris\TAU\hallucination_detection'); CODE=ROOT/'results/monday_fusion_analysis_v1/code'
sys.path.insert(0,str(CODE)); sys.path.insert(0,str(CODE/'scripts/experiments'))
from spectral_utils.monday_fusion_replay import HistoricalData
from spectral_utils.monday_fusion_execution import restore_content
from spectral_utils.monday_fusion_layers import lr_rows
import calfix_common as CC
from spectral_utils.monday_fusion_execution import native
R=ROOT/'results/level_reduction_v1/run_20261004_resume'; L=R/'lr'
manifest=json.loads((R/'INPUT_MANIFEST.json').read_text(encoding='utf8')); pop=CC.Population()
stems=sys.argv[1].split(',')
helper={s['stem'] for s in json.load(open(ROOT/'results/monday_fusion_analysis_v1/lr_helper_20261004T160144Z_455924/PROVENANCE.json'))['written']}
for stem in stems:
    ce,bk,variant,roster,pos=stem.split('__')
    data=HistoricalData(manifest,pop,banks=[bk]); state=restore_content(data,bk,ce,variant)
    rows,folds,diag=lr_rows(data,state,bk,ce,variant,roster,pos=='pos1')
    new=pd.DataFrame(rows).drop(columns=['runtime_s'],errors='ignore')
    buf=io.StringIO(); new.to_csv(buf,index=False); new=pd.read_csv(io.StringIO(buf.getvalue()))
    old=pd.read_csv(L/(stem+'.metrics.csv')).drop(columns=['runtime_s'])
    same_csv=old.equals(new)
    oldf=json.loads((L/(stem+'.fits.json')).read_text()); oldm=json.loads((L/(stem+'.fold_metrics.json')).read_text())
    same_f=json.dumps(native(diag),sort_keys=True)==json.dumps(oldf,sort_keys=True); same_m=json.dumps(native(folds),sort_keys=True)==json.dumps(oldm,sort_keys=True)
    num=old.select_dtypes('number').columns
    print(stem, 'writer=helper' if stem in helper else 'writer=runner', 'metrics_equal',same_csv,'fits_equal',same_f,'folds_equal',same_m,
          'max_numeric_diff', float(np.nanmax(np.abs(old[num].values-new[num].values))) if len(num) else 0, flush=True)
    del data
