"""Compare completed v1 checkpoints with the independent full v2 rerun."""
import sys
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_energy_context_stability import OUT,ARMS,save,sha

def main():
    previous=ROOT/'results/energy_context_stability_v1';checked=0;max_weight=0.;checksums={}
    for done in sorted(previous.glob('*__exclude*/COMPLETE.json')):
        record=json.loads(done.read_text());name=done.parent.name
        for path,expected in record['hashes'].items():assert sha(done.parent/path)==expected
        with np.load(done.parent/'DIAGNOSTICS.npz') as old,np.load(OUT/name/'DIAGNOSTICS.npz') as new:
            assert set(old.files)==set(new.files)
            for key in old.files:
                if '__qp_' in key or '__bootstrap_qp_' in key:
                    max_weight=max(max_weight,float(np.max(abs(old[key]-new[key]))))
                else:np.testing.assert_array_equal(old[key],new[key],err_msg=name+'/'+key)
        checked+=1;checksums[name]=record['hashes']
    assert checked==33
    save(OUT/'NUMERICAL_REVISION_AUDIT.json',dict(status='PASS',old_completed_fits=checked,
        non_simplex_arrays_bitwise_equal=True,max_simplex_coefficient_change=max_weight,
        old_checkpoint_hashes=checksums,old_provenance_sha256=sha(previous/'PROVENANCE.json')))
    print('Revision audit PASS:',checked,'fits, max simplex delta',max_weight)

if __name__=='__main__':main()
