"""Scheduling-only change: split fits and checkpoint projections must be exact."""
import sys
from pathlib import Path
import io,json,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_moment_rbm_staged as staged
from spectral_utils import moment_rbm_fusion as core

class Tests(unittest.TestCase):
    def test_reader_does_not_block_checkpoint(self):
        import sqlite3,tempfile
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'checkpoint.sqlite'
            writer=sqlite3.connect(p);staged.configure_checkpoint(writer)
            writer.execute('CREATE TABLE rows (i INTEGER)')
            writer.executemany('INSERT INTO rows VALUES (?)',[(1,),(2,)]);writer.commit()
            reader=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True)
            cursor=reader.execute('SELECT i FROM rows');self.assertEqual(cursor.fetchone(),(1,))
            writer.execute('INSERT INTO rows VALUES (3)');writer.commit()
            self.assertEqual(cursor.fetchall(),[(2,)])
            reader.close();self.assertEqual(writer.execute('SELECT count(*) FROM rows').fetchone()[0],3)
            writer.close()

    def test_fit_selection_preserves_scores(self):
        import torch
        torch.set_num_threads(1)
        rng=np.random.default_rng(715)
        lp=np.log(np.sort(rng.dirichlet(np.ones(50),size=65),axis=1)[:,::-1]);a=rng.uniform(.1,6,65)
        core.METHODS=staged.ALL_METHODS
        both,fail,_=core.fit_all(lp,a,'stage-test',epochs=2);self.assertFalse(fail)
        try:
            for methods in (staged.FAST,staged.SLOW):
                core.METHODS=methods
                fits,fail,_=core.fit_all(lp,a,'stage-test',epochs=2)
                self.assertFalse(fail);self.assertEqual(set(fits),set(methods))
                for m in methods:
                    np.testing.assert_array_equal(fits[m]['score'],both[m]['score'])
                    for key,val in fits[m]['state'].items():np.testing.assert_array_equal(val,both[m]['state'][key])
        finally:core.METHODS=staged.ALL_METHODS

    def test_checkpoint_roundtrip_and_failed_arm(self):
        S=np.arange(18,dtype=float).reshape(3,6);S[:,1]=np.nan
        W=np.arange(36,dtype=float).reshape(6,6)
        info=staged.base.dumps(dict(uid='uid',n_tokens=10,failures={'iu':'test failure'},
            seconds={m:.1 for m in staged.ALL_METHODS},diagnostics={m:{'seed':1} for m in staged.ALL_METHODS if m!='iu'},raw50_max_error=0.))
        data=staged.base.packed(steps=S,weights=W,**{m+'::w':np.arange(6) for m in staged.ALL_METHODS})
        f,fi=staged.subset_row(data,info,staged.FAST);b,bi=staged.subset_row(data,info,staged.SLOW)
        merged,mi=staged.merge_rows(f,fi,b,bi)
        with np.load(io.BytesIO(data)) as old,np.load(io.BytesIO(merged)) as new:
            self.assertEqual(set(old.files),set(new.files))
            for k in old.files:np.testing.assert_array_equal(old[k],new[k])
        self.assertEqual(json.loads(mi),json.loads(info))

if __name__=='__main__':unittest.main()
