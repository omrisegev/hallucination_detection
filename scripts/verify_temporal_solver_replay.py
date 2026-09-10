"""Compare original and optimized real-answer smoke fits without reading labels."""
import io
import json
from pathlib import Path
import sqlite3
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/direct_probability_temporal_v3'


def load(path):
    con=sqlite3.connect(path)
    result={i:(b,json.loads(s)) for i,b,s in con.execute('SELECT * FROM answers')}
    con.close();return result


def main():
    a=load(ROOT/'results/direct_probability_temporal_v3_original_solver/SMOKE.sqlite')
    b=load(OUT/'SMOKE.sqlite')
    assert set(a)==set(b) and len(a)==27
    worst={};oldtime=[];newtime=[]
    for i in a:
        ab,ai=a[i];bb,bi=b[i]
        assert ai['uid']==bi['uid'] and ai['failures']==bi['failures']=={}
        oldtime.append(ai['wall_seconds']);newtime.append(bi['wall_seconds'])
        with np.load(io.BytesIO(ab)) as az,np.load(io.BytesIO(bb)) as bz:
            assert set(az.files)==set(bz.files)
            for key in az.files:
                np.testing.assert_allclose(az[key],bz[key],atol=1e-9,rtol=1e-8,equal_nan=True)
                worst[key]=max(worst.get(key,0),float(np.nanmax(np.abs(az[key]-bz[key]))))
    result=dict(status='PASS',answers=27,arms=18,max_abs_errors=worst,
        median_original_seconds=float(np.median(oldtime)),median_optimized_seconds=float(np.median(newtime)),
        paired_total_runtime_ratio=float(sum(oldtime)/sum(newtime)),
        scope='Numerical/runtime replay only, no benchmark ranking.')
    (OUT/'SOLVER_REPLAY.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result))


if __name__=='__main__':main()
