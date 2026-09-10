"""Driver tests independent of real benchmark outcomes."""
import io
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import run_direct_probability_temporal as run


class DriverTests(unittest.TestCase):
    def test_checkpoint_commit_rollback_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'checkpoint.sqlite';manifest={'hashes':{'x':'abc'}}
            con=run.connect(p,manifest)
            rows=[(i,run.packed(steps=np.full((2,18),i)),json.dumps({'uid':str(i)})) for i in range(3)]
            con.execute('INSERT INTO answers VALUES (?,?,?)',rows[0]);con.commit()
            con.execute('INSERT INTO answers VALUES (?,?,?)',rows[1]);con.close() # interrupted uncommitted batch
            con=run.connect(p,manifest)
            self.assertEqual(con.execute('SELECT idx FROM answers').fetchall(),[(0,)])
            con.executemany('INSERT INTO answers VALUES (?,?,?)',rows[1:]);con.commit()
            for i,b,_ in con.execute('SELECT * FROM answers ORDER BY idx'):
                with np.load(io.BytesIO(b)) as z:np.testing.assert_array_equal(z['steps'],np.full((2,18),i))
            con.close()
            with self.assertRaisesRegex(ValueError,'manifest mismatch'):run.connect(p,{'hashes':{'x':'changed'}})

    def test_invalid_clean_is_failure_and_prmscore_fold_isolation(self):
        records=[dict(cell='pb_test_q8',group_id=f'g{i}',uid=str(i),row_id=str(i)) for i in range(4)]
        records += [dict(cell='prmbench_qwen3_8b',group_id=f'p{i//2}',uid=f'prm{i}',row_id=f'calculation_{i}') for i in range(4)]
        offsets=np.arange(0,17,2);target=np.array([-1,-1,0,1,-1,-1,-1,-1])
        labels=np.tile([0,1],8)
        values=np.array([0,0,np.nan,np.nan,2,1,1,2, 1,2,3,4, 10,20,30,40],float)
        joined=dict(offsets=offsets,target=target,labels=labels)
        detector=np.array([0,0,2,2,np.nan,np.nan,np.nan,np.nan]);threshold=np.ones(8)
        folds={'outer':{r['group_id']:(0 if r['group_id'] in ('p0','g0','g1') else 1) for r in records}}
        meta={i:dict(idx=f'calculation_{i}',classification='calculation',error_steps=[1]) for i in range(4)}
        with tempfile.TemporaryDirectory() as directory:
            fp=Path(directory)/'folds.json';fp.write_text(json.dumps(folds))
            with patch.object(run.old,'FOLDS',fp),patch.object(run.old,'_gate_contract',return_value=(detector,threshold)),patch.object(run.old,'load_pickle',return_value=meta):
                m,p=run.evaluate_arrays(records,joined,{'test':values})
        self.assertEqual(m['test']['pb_clean_accuracy'],.5)
        self.assertEqual(m['test']['pb_all8'],2*.5/(1+.5))
        self.assertEqual(m['test']['pb_invalid'],1)
        self.assertAlmostEqual(m['test']['prmscore_thresholds']['0'],34.)
        self.assertAlmostEqual(m['test']['prmscore_thresholds']['1'],3.4)

    def test_bootstrap_against_explicit_source_resampling(self):
        records=[];target=[]
        for cell in ('a','b','c','d'):
            for model in ('q4','q8'):
                for i in range(20):
                    records.append(dict(cell=f'pb_{cell}_{model}',group_id=f'{cell}_{i}'))
                    target.append(-1 if i<10 else i%2)
        for i in range(12):records.append(dict(cell='prmbench_qwen3_8b',group_id=f'p{i//2}'));target.append(-1)
        target=np.array(target);n=len(target);rng=np.random.default_rng(7)
        per={}
        for j,m in enumerate(run.METHODS):
            prediction=target.copy();prediction[rng.random(n)<.2]=-2
            valid=np.ones(n,bool);valid[3+j]=False
            within=np.full(n,np.nan);within[-12:]=rng.uniform(.4,.8,12)
            per[m]=dict(prediction=prediction,decision_valid=valid,within=within)
        # Ensure zero common within coverage produces JSON null, not NaN.
        per['delta__equal']['within'][:]=np.nan
        out=run.paired_bootstrap(records,dict(target=target),per,draws=30)
        self.assertIsNone(out['delta__equal_minus_current__iu']['prm_within_ci'])
        json.dumps(out,allow_nan=False)
        _,inv=np.unique([r['group_id'] for r in records],return_inverse=True);ng=inv.max()+1
        W=np.random.default_rng(20260910136).multinomial(ng,np.full(ng,1/ng),size=30)
        cells=np.array([r['cell'] for r in records]);pb=np.char.startswith(cells,'pb_')
        a,b='lag8__iu','current__iu';values=[]
        for draw in W:
            weights=draw[inv];v=[]
            for m in (a,b):
                v.append(run.pb_metrics(target[pb],per[m]['prediction'][pb],per[m]['decision_valid'][pb],cells[pb],weights[pb])['macros']['all'])
            values.append(v[0]-v[1])
        np.testing.assert_allclose(out[a+'_minus_'+b]['pb_ci'],np.nanpercentile(values,[1.25,98.75]),atol=1e-14)
        custom=run.paired_bootstrap(records,dict(target=target),{'k50__iu':per[a],'k50__raw':per[b]},
            draws=30,pairs=[('k50__iu','k50__raw')],primary_pairs={('k50__iu','k50__raw')})
        np.testing.assert_allclose(custom['k50__iu_minus_k50__raw']['pb_ci'],out[a+'_minus_'+b]['pb_ci'],atol=1e-14)
        self.assertEqual(custom['k50__iu_minus_k50__raw']['ci_level'],.975)


if __name__=='__main__':
    with threadpool_limits(limits=1):unittest.main()
