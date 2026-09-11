"""Independent structural/reconstruction audit of saved diagnostic outputs."""
import argparse
import io
import json
from pathlib import Path
import sqlite3
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_direct_probability_temporal as base


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);a=p.parse_args()
    out=ROOT/'results/rbm_data_diagnostics_v1';base.old.configure_source_root(a.source_root)
    records=json.loads((base.old.BENCH/'evaluation/JOINED.json').read_text())['records']
    joined=np.load(base.old.BENCH/'evaluation/JOINED.npz');z=np.load(out/'SCORES.npz')
    c=sqlite3.connect((out/'CHECKPOINT.sqlite').as_uri()+'?mode=ro',uri=True)
    assert c.execute('SELECT COUNT(*) FROM answers').fetchone()[0]==len(records)==13769
    checked=0;missing={6:0,12:0}
    for i,payload,info in c.execute('SELECT idx,payload,info FROM answers ORDER BY idx'):
        info=json.loads(info);assert info['uid']==records[i]['uid']
        sl=slice(joined['offsets'][i],joined['offsets'][i+1]);labels=joined['labels'][sl]
        with np.load(io.BytesIO(payload)) as row:
            assert len(row['lengths'])==records[i]['steps']
            for bank in (6,12):
                pre=f'b{bank}__'
                if pre+'columns' not in row:missing[bank]+=1;continue
                sm=row[pre+'step_means'];counts=row[pre+'class_counts'];var=row[pre+'class_variance']
                if records[i]['cell'].startswith('pb_'):
                    assert not counts.any() and np.isnan(var).all()
                else:
                    for si,mask in enumerate(row['masks']):
                        for label in (0,1):
                            selected=sm[mask&(labels==label)]
                            assert counts[si,label]==len(selected)
                            if counts[si].min()>=2:
                                # Independent pairwise sample-variance identity.
                                mean=selected.sum(axis=0)/len(selected)
                                manual=((selected-mean)**2).sum(axis=0)/(len(selected)-1)
                                np.testing.assert_allclose(var[si,label],manual,atol=1e-12,rtol=1e-12)
                for state in ('old','near'):
                    s=z[f'steps__rbm{bank}__{state}'][sl]
                    assert len(s)==len(sm)
                old=z[f'steps__rbm{bank}__old'][sl];new=z[f'steps__rbm{bank}__near'][sl]
                near=np.flatnonzero(old>=old.max()-.25*old.std())
                if old.std()>1e-10:
                    assert int(np.argmax(new))==int(near[0])
                assert row[pre+'residual_synthetic'].shape==(4,5)
                checked+=1
    review=dict(status='PASS',n_answers=len(records),banks_checked=checked,missing=missing,
        checks=['all identities and spans','PB has no invented step labels','PRMB sample variances reconstructed',
                'new RBM peaks prefer first near maximum','four conditional model replicates per saved fit'])
    base.atomic_json(out/'DIAGNOSTIC_REVIEW.json',review);print(base.dumps(review))


if __name__=='__main__':main()
