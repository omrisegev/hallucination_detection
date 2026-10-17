"""Durable completion of the two already frozen runs; stop before new experiments.

Uses existing drivers unchanged. An existing historical driver can be adopted
read-only, invocation-cap checkpoints resume, unexpected failures stop visibly.
"""
from __future__ import annotations

import json
import msvcrt
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import psutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/research_consolidation_v1'
PYTHON = Path(sys.executable)
JOBS = [
    dict(name='historical_joint', folder=ROOT/'results/historical_joint_refit_v3',
         driver=ROOT/'scripts/run_historical_joint_refit_v3.py', args=[],
         complete='COMPLETE_REVIEWED_JOINT_EXTENSION', review='REVIEW.json', total=245),
    dict(name='full_sampling', folder=ROOT/'results/localization_full_sampling_v3',
         driver=ROOT/'scripts/run_full_sampling_v3.py', args=['--phase','run'],
         complete='COMPLETE_REVIEWED_FULL_SAMPLING', review='evaluation/REVIEW.json', total=13769),
]


def load(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def state(phase, **fields):
    value = dict(phase=phase, supervisor_pid=os.getpid(), updated_unix=time.time(),
                 stage_boundary='RETURN_BEFORE_NEW_IMPROVEMENT_EXPERIMENT', **fields)
    tmp=OUT/'RUN_STATE.json.tmp'
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    tmp.replace(OUT/'RUN_STATE.json')


def matching_process(pid, driver):
    if not pid:
        return None
    try:
        proc=psutil.Process(int(pid))
        command=proc.cmdline()
        if any(str(driver).lower()==str(x).replace('/','\\').lower() for x in command):
            return proc
        return None
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return None
    # AccessDenied deliberately propagates: do not assume an inaccessible process is absent.


def find_driver(driver):
    matches=[]
    for proc in psutil.process_iter(['pid','name']):
        if 'python' not in (proc.info['name'] or '').lower():
            continue
        candidate=matching_process(proc.pid,driver)
        if candidate is not None:
            matches.append(candidate)
    if len(matches)>1:
        raise RuntimeError('Duplicate driver processes already exist: '+str(driver))
    return matches[0] if matches else None


def verified_complete(job):
    current=load(job['folder']/'RUN_STATE.json')
    review=load(job['folder']/job['review'])
    return current.get('phase')==job['complete'] and review.get('status')=='PASS'


def wait_for(proc, job):
    last=None
    while proc.is_running():
        current=load(job['folder']/'RUN_STATE.json')
        key=(current.get('phase'),current.get('completed'),current.get('done'))
        state('WAITING_FOR_'+job['name'].upper(), child_pid=proc.pid, child_state=current,
              free_disk_bytes=shutil.disk_usage(ROOT).free)
        if key!=last:
            print(job['name'],current,flush=True); last=key
        try:
            proc.wait(timeout=30)
            break
        except psutil.TimeoutExpired:
            pass
        except psutil.NoSuchProcess:
            break
    # The child closes its outputs before exiting. Its final phase/review governs success.


def finish_job(job):
    first=True
    while not verified_complete(job):
        existing=find_driver(job['driver'])
        current=load(job['folder']/'RUN_STATE.json')
        if existing:
            print('Adopting existing driver',job['name'],existing.pid,flush=True)
            wait_for(existing,job)
        else:
            phase=current.get('phase','')
            allowed={'CHECKPOINTED_INVOCATION_CAP','FITTING','SCORING','DRAINING_INVOCATION_CAP',
                     'SCORING_COMPLETE','PREPARED_FULL_SAMPLING','SCORES_COMPLETE_WAITING_FOR_SHORTLIST_REVIEW'}
            if phase not in allowed:
                raise RuntimeError(f"Refusing automatic restart of {job['name']} in phase {phase!r}; inspect failure first")
            if not first and phase!='CHECKPOINTED_INVOCATION_CAP':
                raise RuntimeError(f"Unexpected exit of {job['name']} in phase {phase}; checkpoints preserved")
            if shutil.disk_usage(ROOT).free<1024**3:
                raise RuntimeError('Less than1GiB disk free before launching the next invocation; no deletion authorized')
            logfile=OUT/(job['name']+f'_{int(time.time())}.log')
            with logfile.open('ab',buffering=0) as stream:
                child=subprocess.Popen([str(PYTHON),'-B','-u',str(job['driver']),*job['args']],
                    cwd=str(ROOT),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW)
            state('STARTED_'+job['name'].upper(),child_pid=child.pid,log=str(logfile))
            wait_for(psutil.Process(child.pid),job)
            code=child.wait()
            if code:
                raise RuntimeError(f'{job["name"]} exited{code}; see {logfile}')
        first=False
        if verified_complete(job):
            print('Reviewed completion:',job['name'],flush=True)
            return
        current=load(job['folder']/'RUN_STATE.json')
        if current.get('phase')!='CHECKPOINTED_INVOCATION_CAP':
            raise RuntimeError(f'{job["name"]} ended without reviewed completion or a resumable time cap: {current}')


def document_completion():
    ledger=load(OUT/'LEDGER.json')
    assert ledger['status']=='COMPLETE_REVIEWED_CONSOLIDATION'
    marker='## Step332 completion — consolidation reviewed'
    block=('\n'+marker+'\n\n'
           'Both previously paused frozen runs are now complete and reviewed: historical Joint245/245 '
           'and full sampling13769/13769. The separate entropy-gate review also passed. '
           'These statements supersede earlier live/paused checkpoints.\n\n'
           'Hebrew reflection: docs/reviews/research_consolidation_2026-09-08.html. '
           'Machine-readable evidence and review: results/research_consolidation_v1/. '
           'Original reports and protocol distinctions are preserved. Full cached results remain development evidence. '
           'Completion of these obligations does not mean all historical leaders or untouched confirmation are finished.\n\n'
           '**Stage boundary reached: return to Omri before a new improvement experiment.** '
           'Shared-q0.3 gate and max/top10 trials are specified, not started. No expanded search.\n\n')
    for name in ('PROGRESS.md','Research_Directions.md'):
        path=ROOT/name
        original=path.read_bytes();text=original.decode('utf-8-sig')
        if marker in text:
            continue
        split=text.find('\n')
        updated=text[:split+1]+block+text[split+1:]
        if path.read_bytes()!=original:
            raise RuntimeError('Concurrent documentation edit; do not overwrite '+name)
        encoding='utf-8-sig' if original.startswith(b'\xef\xbb\xbf') else 'utf-8'
        path.write_text(updated,encoding=encoding,newline='')
    history=ROOT/'HISTORY.md'
    if marker not in history.read_text(encoding='utf-8-sig'):
        with history.open('a',encoding='utf-8') as f:
            f.write('\n---\n'+block)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'supervisor.lock').open('a+b') as lock:
        lock.seek(0)
        if not lock.read(1):
            lock.write(b'0');lock.flush()
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:
            state('STARTING_COMPLETION_ONLY')
            audit=load(ROOT/'results/fixed_gate_completion_review_v1/REVIEW.json')
            if audit.get('status')!='PASS':
                raise RuntimeError('Gate review must finish before consolidation supervisor starts')
            for job in JOBS:
                finish_job(job)
            state('BUILDING_FINAL_REFLECTION')
            subprocess.run([str(PYTHON),'-B','-u',str(ROOT/'scripts/build_research_consolidation_v1.py')],
                           cwd=str(ROOT),check=True,creationflags=subprocess.CREATE_NO_WINDOW)
            document_completion()
            state('COMPLETE_REVIEWED_RETURN_TO_USER',report='docs/reviews/research_consolidation_2026-09-08.html',
                  new_improvement_experiments_started=False)
            print('Completion stage finished. No new improvement experiment started.',flush=True)
        except BaseException as error:
            state('STOPPED_REQUIRES_REVIEW',error=repr(error),traceback=traceback.format_exc())
            raise
        finally:
            lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)


if __name__=='__main__':
    main()
