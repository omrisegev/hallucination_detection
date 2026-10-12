"""Stop only the identified legacy workflow at its superseded combined stage."""
import argparse
import datetime
import json
import os
from pathlib import Path
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'results/monday_fusion_analysis_v1'
OUT = WORK / 'candidate_panel_guard'


def dump(path, value):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n', encoding='utf8')
    os.replace(tmp, path)


def should_stop(ledger):
    # input_report is recorded as completed only after it returns. Checking this
    # boundary also catches the few milliseconds before stage becomes combined.
    return ledger.get('stage') == 'combined' or 'input_report' in ledger.get('completed', [])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pid', required=True, type=int)
    parser.add_argument('--created', required=True, type=float)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    runner = psutil.Process(args.pid)
    expected = (WORK / 'code/scripts/experiments/monday_fusion_run.py').resolve()
    if abs(runner.create_time() - args.created) > 0.01:
        raise RuntimeError('Runner PID identity changed')
    cmd = runner.cmdline()
    if str(expected).lower() not in [v.lower() for v in cmd] or cmd[-1] != 'continue':
        raise RuntimeError('Not the authorized Monday continue process')
    record = {'guard_pid': os.getpid(), 'runner_pid': args.pid,
              'runner_created': args.created, 'runner_command': cmd,
              'authorization_source': 'results/monday_fusion_review_claude_20261004_round2/MESSAGE_TO_CODEX_2_CANDIDATE_PANEL.md',
              'started': datetime.datetime.now().astimezone().isoformat(),
              'status': 'WATCHING', 'stop_file': str(OUT / 'STOP'),
              'policy': 'Preserve preceding stages; terminate only at completed input_report / combined boundary; never select or freeze.'}
    dump(OUT / 'STATUS.json', record)
    while True:
        if (OUT / 'STOP').exists():
            record['status'] = 'GUARD_STOP_REQUESTED'
            break
        if not runner.is_running():
            record['status'] = 'RUNNER_EXITED_BEFORE_BOUNDARY'
            break
        try:
            ledger = json.loads((WORK / 'WORKFLOW_STATUS.json').read_text(encoding='utf8'))
        except (FileNotFoundError, json.JSONDecodeError, PermissionError):
            time.sleep(0.25)
            continue
        if should_stop(ledger):
            if abs(runner.create_time() - args.created) > 0.01:
                raise RuntimeError('Runner PID reused before stop')
            record.update(status='STOPPING_AT_SUPERSEDED_COMBINED', observed_ledger=ledger)
            dump(OUT / 'STATUS.json', record)
            runner.terminate()
            runner.wait(timeout=30)
            record['status'] = 'AWAITING_OMRI_CANDIDATE_AND_RULE'
            record['runner_stopped'] = True
            record['note'] = 'Legacy WORKFLOW_STATUS may still say RUNNING. No legacy combined artifact constitutes a chosen recipe. Use candidate panel.'
            break
        time.sleep(0.25)
    record['finished'] = datetime.datetime.now().astimezone().isoformat()
    dump(OUT / 'STATUS.json', record)
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
