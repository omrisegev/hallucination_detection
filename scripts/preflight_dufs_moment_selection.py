"""Check all four historical score anchors on smoke rows; no quality scores."""
import io,json,sqlite3,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spectral_utils.dufs_moment_selection import METHODS
from scripts.run_direct_probability_temporal import old
source=ROOT.parents[1];old.configure_source_root(source)
out=ROOT/'results/dufs_moment_selection_v1'
refs={}
with np.load(source/'.worktrees/higher-moment-fusion-v1/results/higher_moment_fusion_v1/SCORES.npz') as z:
    for bank,degree in (('original6',3),('all12',6)):
        for solver in ('rbm','rbm_initial'):refs[bank+'__'+solver]=z[f'steps__d{degree}__{solver}']
offsets=np.load(old.BENCH/'evaluation/JOINED.npz')['offsets']
con=sqlite3.connect('file:'+str(out/'SMOKE.sqlite')+'?mode=ro',uri=True)
rows=con.execute('SELECT idx,payload,info FROM answers').fetchall();con.close()
assert len(rows)==27
seconds=[];checks=0
for i,blob,info in rows:
    d=json.loads(info);assert not d['failures'];seconds.append(sum(d['seconds'].values()))
    with np.load(io.BytesIO(blob),allow_pickle=False) as z:
        for m,flat in refs.items():
            np.testing.assert_array_equal(z['steps'][:,METHODS.index(m)],flat[offsets[i]:offsets[i+1]])
            checks+=1
summary=dict(status='PASS',scope='Mechanics and runtime only; not a population performance estimate',
    answers=27,historical_score_replays=checks,failures=0,
    seconds_per_answer_quantiles=np.quantile(seconds,[0,.25,.5,.75,1]).tolist())
(out/'PREFLIGHT_REVIEW.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary))
