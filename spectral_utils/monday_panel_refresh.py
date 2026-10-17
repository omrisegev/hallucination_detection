"""Refresh the descriptive panel once preceding stages finish; never select/freeze."""
from pathlib import Path
import datetime
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'results/monday_fusion_analysis_v1'
OUT = WORK / 'candidate_panel_refresh'
PANEL = ROOT / 'results/monday_candidate_panel_v1/run_20261004'


def read(path):
    try:
        return json.loads(path.read_text(encoding='utf8'))
    except (FileNotFoundError, json.JSONDecodeError, PermissionError):
        return {}


def ready(guard, panel):
    return guard.get('status') == 'AWAITING_OMRI_CANDIDATE_AND_RULE' and panel.get('status') == 'CONTENT_AND_PRESELECTION_FORENSICS_COMPLETE_LAYER_LR_SNAPSHOT'


def original_lr_ready(prm_only=False):
    run = ROOT / 'results/level_reduction_v1/run_20261004_resume'
    originals = list((run / 'chunks').glob('*__original.complete.json'))
    originals = [p for p in originals if not prm_only or p.name.startswith('prmbench_')]
    if len(originals) != (8 if prm_only else 72):
        return False
    return all((run / 'lr' / (p.name.removesuffix('.complete.json') + f'__{roster}__pos{pos}.complete.json')).exists()
               for p in originals for roster in ('full', 'survivors') for pos in (0, 1))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    status = {'pid': os.getpid(), 'status': 'WAITING_FOR_COMPLETED_STAGES',
              'started': datetime.datetime.now().astimezone().isoformat(),
              'command': [sys.executable, '-u', '-B', str(ROOT / 'scripts/build_monday_candidate_panel.py'), '--refresh'],
              'refreshed': [],
              'policy': 'Refresh after all original PRMBench LR, all original LR, and final layer/report boundary. Read sealed checkpoints only; no selection, freeze or external evaluation.'}
    def save():
        tmp = OUT / 'STATUS.json.tmp'
        tmp.write_text(json.dumps(status, indent=2) + '\n', encoding='utf8')
        os.replace(tmp, OUT / 'STATUS.json')
    save()
    while not (OUT / 'STOP').exists():
        guard = read(WORK / 'candidate_panel_guard/STATUS.json')
        panel = read(PANEL / 'AUDIT.json')
        phase = None
        if panel.get('status') == 'CONTENT_AND_PRESELECTION_FORENSICS_COMPLETE_LAYER_LR_SNAPSHOT':
            if ready(guard, panel):
                phase = 'ALL_STAGES'
            elif original_lr_ready():
                phase = 'ALL_ORIGINAL_LR'
            elif original_lr_ready(prm_only=True):
                phase = 'PRMBENCH_ORIGINAL_LR'
        if phase and phase not in status['refreshed']:
            status.update(status='REFRESHING', phase=phase)
            save()
            with (OUT / (phase + '.log')).open('w', encoding='utf8') as log:
                result = subprocess.run(status['command'], stdout=log, stderr=subprocess.STDOUT)
            status.update(returncode=result.returncode, checked=datetime.datetime.now().astimezone().isoformat())
            if result.returncode:
                status['status'] = 'REFRESH_FAILED'
                save()
                return
            status['refreshed'].append(phase)
            if phase == 'ALL_STAGES':
                status['status'] = 'COMPLETE'
                save()
                return
            status['status'] = 'WAITING_FOR_COMPLETED_STAGES'
            save()
        if guard.get('status') in ('GUARD_STOP_REQUESTED', 'RUNNER_EXITED_BEFORE_BOUNDARY'):
            status.update(status='NEEDS_RUNNER_FOLLOWUP', guard_status=guard)
            save()
            return
        time.sleep(15)
    status['status'] = 'STOP_REQUESTED'
    save()


if __name__ == '__main__':
    main()
