"""Write JOB_<bench>_<V>.json (RC_HANDOFF D.5) from the job's own records. No labels.
    python job_record.py W BENCH V JOBDIR
JOBDIR holds steps.tsv (step, exit code, seconds, utc end), gpu.txt (nvidia-smi name/memory), gpumem.log (MiB samples).
"""
import json, os, socket, sys
from pathlib import Path

W, bench, V, jd = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
steps = [dict(zip(('step', 'exit_code', 'seconds', 'utc_end'), l.rstrip('\n').split('\t')))
         for l in open(jd / 'steps.tsv')] if (jd / 'steps.tsv').exists() else []
mem = [int(x) for x in open(jd / 'gpumem.log').read().split() if x.strip().isdigit()] if (jd / 'gpumem.log').exists() else []
rec = dict(bench=bench, variant=V, job_id=os.environ.get('SLURM_JOB_ID'), restart_count=os.environ.get('SLURM_RESTART_COUNT', '0'),
           node=socket.gethostname(), gpu=(jd / 'gpu.txt').read_text().strip() if (jd / 'gpu.txt').exists() else None,
           peak_gpu_memory_mib_sampled_20s=max(mem) if mem else None, steps=steps,
           wall_sec_this_attempt=sum(float(s['seconds']) for s in steps if s.get('seconds')))
man = W / 'runs' / f'{bench}_{V}' / 'manifest.json'
if man.exists():
    m = json.load(open(man))
    rec['full_run'] = dict(status=m.get('status'), n_rows=m.get('n_rows'), n_failed_rows=m.get('n_failed_rows'),
                           elapsed_sec=m.get('elapsed_sec'), s_per_row=round(m['elapsed_sec'] / max(1, m['n_rows']), 3),
                           max_input_tokens=m.get('max_input_tokens'), model_revision=m.get('model_revision'))
json.dump(rec, open(W / f'JOB_{bench}_{V}.json', 'w'), indent=1)
print('JOB RECORD', json.dumps(rec)[:600], flush=True)
