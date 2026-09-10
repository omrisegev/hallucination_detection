"""Replay preserved original-solver checkpoints against completed optimized rows."""
import io
import json
from pathlib import Path
import sqlite3
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/direct_probability_temporal_v3'


def main():
    original=sqlite3.connect(ROOT/'results/direct_probability_temporal_v3_original_solver/CHECKPOINT.sqlite')
    current=sqlite3.connect(OUT/'CHECKPOINT.sqlite')
    worst=0.;count=0
    for idx,blob,info in original.execute('SELECT * FROM answers ORDER BY idx'):
        match=current.execute('SELECT payload,info FROM answers WHERE idx=?',(idx,)).fetchone()
        if match is None:raise RuntimeError(f'optimized row {idx} is not ready')
        a,b=json.loads(info),json.loads(match[1])
        assert a['uid']==b['uid'] and a['failures']==b['failures']
        with np.load(io.BytesIO(blob)) as old,np.load(io.BytesIO(match[0])) as new:
            assert set(old.files)==set(new.files)
            for key in old.files:
                np.testing.assert_allclose(old[key],new[key],atol=1e-9,rtol=1e-8,equal_nan=True)
                worst=max(worst,float(np.nanmax(np.abs(old[key]-new[key]))))
        count+=1
    result=dict(status='PASS',answers=count,arms=18,max_abs_score_or_weight_error=worst,
        scope='Numerical parity, no labels or accuracy metrics read.')
    (OUT/'ORIGINAL_CHECKPOINT_REPLAY.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(result);original.close();current.close()


if __name__=='__main__':main()
