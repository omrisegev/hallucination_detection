"""Numerical provenance of H1 refits; fixed smoke cases, no model selection."""
import io
import json
from pathlib import Path
import sqlite3
import sys
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_rbm_literature_completion as run
from spectral_utils import rbm_literature_completion as model
from spectral_utils.moment_rbm_fusion import fit_rbm


def main():
    source=ROOT.parents[1];out=run.PROGRAM/'capacity'
    records,joined,reference=run.load_contract(source)
    db=sqlite3.connect((out/'SMOKE.sqlite').as_uri()+'?mode=ro',uri=True)
    src=sqlite3.connect(run.modeldb(source).as_uri()+'?mode=ro',uri=True)
    ids={int(i) for i, in db.execute('select idx from answers')};rows=[]
    with threadpool_limits(limits=1):
        for path in sorted(run.caches(source).glob('cache_*.npz')):
            with np.load(path) as z:cache={k:z[k] for k in z.files if k!='labels'}
            for k,i in enumerate(cache['ids']):
                i=int(i)
                if i not in ids:continue
                _,uid,spans,anchor,banks=run.prepare_answer(k,cache,src,records,joined,reference)
                blob,info=db.execute('select payload,info from answers where idx=?',(i,)).fetchone();info=json.loads(info)
                with np.load(io.BytesIO(blob)) as z:states={k:z[k] for k in z.files}
                for bank in (6,12):
                    x=banks[bank]['x'];old=banks[bank]['theta'];key=f'b{bank}_exact1';new=states[key+'::theta']
                    candidates={}
                    for layout,xx in [('C',np.ascontiguousarray(x)),('F',np.asfortranarray(x))]:
                        _,s,d=fit_rbm(xx,maxiter=100)
                        t=model.pack(s['a'],s['w'][:,None],[float(s['b'])]);candidates[layout]=t
                    # The current experiment must implement the actual historical
                    # objective/optimizer on its exact current input layout.
                    np.testing.assert_allclose(new,candidates['C'],atol=1e-12,rtol=1e-12)
                    sl=slice(joined['offsets'][i],joined['offsets'][i+1])
                    row=dict(uid=uid,bank=bank,tokens=len(x),
                        current_vs_legacy_same_input_max_theta=float(np.max(np.abs(new-candidates['C']))),
                        current_C_vs_frozen_max_theta=float(np.max(np.abs(new-old))),
                        legacy_F_vs_frozen_max_theta=float(np.max(np.abs(candidates['F']-old))),
                        refit_NLL_minus_frozen=float(model.ExactRBM(x,1)(new)[0]-model.ExactRBM(x,1)(old)[0]))
                    for mode,alias in [('logit','logit_old'),('posterior','old')]:
                        a=states['score::'+key+'_'+mode];b=reference[f'rbm{bank}__'+alias][sl]
                        row[mode+'_max_step_difference']=float(np.max(np.abs(a-b)))
                        row[mode+'_peak_changed']=bool(np.argmax(a)!=np.argmax(b))
                    rows.append(row)
            print('[H1 provenance]',path.stem,len(rows),flush=True)
    assert len(rows)==2*len(ids)==54
    run.csv_write(out/'H1_NUMERICAL_PROVENANCE.csv',rows)
    run.base.atomic_json(out/'H1_NUMERICAL_PROVENANCE.json',dict(status='PASS',answers=len(ids),fits_compared=len(rows),
        current_legacy_same_input_max_theta=max(r['current_vs_legacy_same_input_max_theta'] for r in rows),
        current_C_vs_frozen_max_theta=max(r['current_C_vs_frozen_max_theta'] for r in rows),
        legacy_F_vs_frozen_max_theta=max(r['legacy_F_vs_frozen_max_theta'] for r in rows),
        logit_max_step_difference=max(r['logit_max_step_difference'] for r in rows),
        posterior_max_step_difference=max(r['posterior_max_step_difference'] for r in rows),
        logit_peak_changes=sum(r['logit_peak_changed'] for r in rows),
        posterior_peak_changes=sum(r['posterior_peak_changed'] for r in rows),
        max_absolute_NLL_difference=max(abs(r['refit_NLL_minus_frozen']) for r in rows),
        purpose='Fixed smoke-case numerical provenance, not a performance sample or model selection. Full frozen-reference comparison remains required.',
        script_sha256=run.base.old.sha256_file(Path(__file__))))
    src.close();db.close()


if __name__=='__main__':main()
