"""Resume only the candidate-report worker when physical RAM permits.

This small supervisor never signals the experiment runner or the LR helper.
"""
import argparse
import csv
import ctypes
import datetime
import hashlib
import html
import json
from pathlib import Path
import re
import time
import psutil

def trim_paused_cache(process):
    """Evict only this stopped worker's resident cache; its allocations survive."""
    from ctypes import wintypes
    if process.status()!=psutil.STATUS_STOPPED: raise RuntimeError('Cache trim requires paused candidate worker')
    if not any(str(x).endswith('build_monday_candidate_panel.py') for x in process.cmdline()): raise RuntimeError('Unexpected cache-trim target')
    kernel=ctypes.WinDLL('kernel32',use_last_error=True); kernel.OpenProcess.restype=wintypes.HANDLE
    handle=kernel.OpenProcess(0x0500,False,process.pid)
    if not handle: raise ctypes.WinError(ctypes.get_last_error())
    try:
        api=ctypes.WinDLL('psapi',use_last_error=True); api.EmptyWorkingSet.argtypes=[wintypes.HANDLE]
        if not api.EmptyWorkingSet(handle): raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.CloseHandle.argtypes=[wintypes.HANDLE]; kernel.CloseHandle(handle)

def validate_finished(directory):
    """Check complete output coverage even if the foreground agent has yielded."""
    try:
        from finalize_monday_candidate_report import finalize,regression_check
        regression_check(); finalize(directory)
        audit=json.loads((directory/'AUDIT.json').read_text(encoding='utf8'))
        assert audit['status']=='CONTENT_AND_PRESELECTION_FORENSICS_COMPLETE_LAYER_LR_SNAPSHOT'
        assert audit['checks']['claude_precompute']['rows_compared']==5184
        assert audit['checks']['claude_precompute']['max_difference']<=1e-12
        with (directory/'CONTENT_METRICS.csv').open(encoding='utf8',newline='') as stream: rows=list(csv.DictReader(stream))
        assert len(rows)==5184
        identities={(r['bank'],r['cell'],r['variant'],r['recipe'],r['decision_rule']) for r in rows}
        assert len(identities)==5184
        assert len({r['bank'] for r in rows})==8 and len({r['cell'] for r in rows})==9
        assert all(float(r['coverage'])==1 and int(float(r['failures']))==0 for r in rows)
        with (directory/'PAIRED_CONTRASTS.csv').open(encoding='utf8',newline='') as stream: contrasts=list(csv.DictReader(stream))
        assert len(contrasts)==3024
        page=(directory/'REPORT_HE.html').read_text(encoding='utf8')
        assert 'charset="utf-8"' in page and 'LR_REFERENCES.csv' in page and 'argmax' in page
        assert sum('\u0590'<=c<='\u05ff' for c in page)>500
        links=[html.unescape(x) for x in re.findall(r'(?:href|src)="([^"]+)"',page)]
        assert all((directory/link).is_file() for link in links)
        assert (directory/'figures/HEM_MOM_DECISION.png').read_bytes()[:8]==b'\x89PNG\r\n\x1a\n'
        hashes={str(p.relative_to(directory)):hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.glob('*.csv')}
        result={'status':'PASS','metric_rows':len(rows),'contrast_rows':len(contrasts),'unique_metric_keys':len(identities),
            'local_report_links_checked':len(links),'csv_sha256':hashes,'visual_review':'PNG signature, SVG/HTML links and explicit white chart background checked; manual visual inspection pending',
            'MTG_summary_denominator':'Corrected from unchanged saved answer rows; see MTG_SUMMARY_REPAIR.json',
            'review_status':'Scientific review pending; automated checks are not independent Claude signoff'}
    except Exception as exc:
        result={'status':'FAIL_OR_INCOMPLETE','error':repr(exc)}
    result['time_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    result['validator_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (directory/'AUTOMATED_VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    return result['status']

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--pid',type=int,required=True)
    args=parser.parse_args(); process=psutil.Process(args.pid)
    if not any(str(x).endswith('build_monday_candidate_panel.py') for x in process.cmdline()):
        raise RuntimeError('Refusing to signal a different process')
    created=process.create_time(); paused=process.status()==psutil.STATUS_STOPPED
    last_trim=0.; trim_count=0; trim_error=None
    out=Path(r'C:\Users\omris\TAU\hallucination_detection\results\monday_candidate_panel_v1\run_20261004\MEMORY_STATUS.json')
    while True:
        available=psutil.virtual_memory().available/1048576
        try:
            if process.create_time()!=created or not process.is_running(): break
            if paused and available>=900:
                process.resume(); paused=False
            elif not paused and available<700:
                process.suspend(); paused=True
            if paused and available<900 and time.monotonic()-last_trim>60:
                try:
                    trim_paused_cache(process); trim_count+=1; trim_error=None
                except (OSError,RuntimeError) as exc:
                    trim_error=repr(exc)
                last_trim=time.monotonic()
            record={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'worker_pid':args.pid,
                'worker_create_time':created,'status':'PAUSED_FOR_RAM' if paused else 'RUNNING',
                'available_mib':available,'worker_rss_mib':process.memory_info().rss/1048576,
                'resume_threshold_mib':900,'pause_threshold_mib':700,
                'working_set_cache_trims':trim_count,'last_trim_error':trim_error,
                'scope':'Candidate report only; main runner and LR helper are never signaled'}
            out.write_text(json.dumps(record,indent=2)+'\n',encoding='utf8')
        except psutil.NoSuchProcess:
            break
        time.sleep(10)
    checked=validate_finished(out.parent)
    out.write_text(json.dumps({'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'worker_pid':args.pid,'status':'WORKER_EXITED','automated_validation':checked},indent=2)+'\n',encoding='utf8')

if __name__=='__main__': main()
