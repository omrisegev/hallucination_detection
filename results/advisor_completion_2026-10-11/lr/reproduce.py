"""Replay the task-7 package and reject stale checkpoint identities first."""
import datetime,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
INP=OUT.parent/'inputs'
PYTHON=ROOT/'.venv/bin/python'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    identities={'source_sha256':sha(ROOT/'scripts/run_advisor_v11_lr.py'),
                'helper_sha256':sha(ROOT/'spectral_utils/advisor_lr.py'),
                'population_sha256':sha(INP/'population.npz')}
    bank_hash={}
    for p in (OUT/'checkpoints').glob('*/outer_*/DONE.json'):
        bank=p.parents[1].name
        if bank not in bank_hash:bank_hash[bank]=sha(INP/f'bank_V_{bank}.npz')
        done=json.loads(p.read_text())
        for key,value in identities.items():
            if done[key]!=value:raise RuntimeError(f'Stale checkpoint {p}: {key} changed. Use a fresh output package.')
        if done['bank_input_sha256']!=bank_hash[bank]:raise RuntimeError(f'Stale bank input: {p}')
    logdir=OUT/'logs';logdir.mkdir(exist_ok=True)
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    with (logdir/f'reproduce_{stamp}.log').open('w') as log:
        for script in [ROOT/'scripts/run_advisor_v11_lr.py',OUT/'replay_native_weights.py',
                       OUT/'supplement_and_verify.py',OUT/'plot_diagnostics.py']:
            log.write(f'Running {script}\n');log.flush()
            subprocess.run([str(PYTHON),str(script)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('Task7 replay PASS; log',logdir/f'reproduce_{stamp}.log')

if __name__=='__main__':main()
