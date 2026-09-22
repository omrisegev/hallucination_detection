"""One resumable launch: full scoring, metric review, model review, concise results."""
import json,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/dufs_moment_selection_v1'


def main():
    for stage,script,args in (
        ('scoring_and_metrics','run_dufs_moment_selection.py',sys.argv[1:]),
        ('saved_state_review','review_dufs_moment_selection.py',[]),
        ('completion_notes','finish_dufs_moment_selection.py',[])):
        print('[pipeline]',stage,datetime.now(timezone.utc).isoformat(),flush=True)
        try:subprocess.run([sys.executable,str(ROOT/'scripts'/script),*args],check=True)
        except BaseException as error:
            OUT.mkdir(parents=True,exist_ok=True)
            (OUT/'PIPELINE_STATE.json').write_text(json.dumps(dict(status='FAILED',stage=stage,error=str(error)),indent=2)+'\n')
            raise
    (OUT/'PIPELINE_STATE.json').write_text(json.dumps(dict(status='COMPLETE',finished_utc=datetime.now(timezone.utc).isoformat()),indent=2)+'\n')


if __name__=='__main__':main()
