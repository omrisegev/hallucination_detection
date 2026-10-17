"""Verify the boundary-token amendment did not change already scored answers."""
import io
import json
from pathlib import Path
import sqlite3
import numpy as np


def main():
    root=Path(__file__).resolve().parents[1]/'results'
    old=root/'rbm_position_fusion_v1';new=root/'rbm_position_fusion_v1_overlap_fix'
    a=sqlite3.connect((old/'CHECKPOINT.sqlite').as_uri()+'?mode=ro',uri=True)
    b=sqlite3.connect((new/'CHECKPOINT.sqlite').as_uri()+'?mode=ro',uri=True)
    n=arrays=0
    for i,blob,info in a.execute('SELECT idx,payload,info FROM answers ORDER BY idx'):
        other,oi=b.execute('SELECT payload,info FROM answers WHERE idx=?',(i,)).fetchone()
        assert json.loads(info)['uid']==json.loads(oi)['uid']
        with np.load(io.BytesIO(blob)) as x,np.load(io.BytesIO(other)) as y:
            assert x.files==y.files
            for key in x.files:np.testing.assert_array_equal(x[key],y[key]);arrays+=1
        n+=1
    a.close();b.close()
    result=dict(status='PASS',same_answers=n,exact_arrays=arrays,original_freeze='b5614775e',corrected_freeze='258393249',
        scope='Every completed answer from the stopped run has identical saved parameters, corrections and step scores in the restarted full run.')
    (new/'AMENDMENT_REVIEW.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
