"""Retry a transient Windows atomic-replace failure; frozen math unchanged."""
import importlib.util
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('history_resume_driver',ROOT/'scripts/bridge_localization_history_v3.py')
d=importlib.util.module_from_spec(s);s.loader.exec_module(d)


def main():
    d.verify();original=d.save;events=[];start=time.monotonic()
    def retry_save(path,value):
        Path(path).resolve().relative_to(d.OUT.resolve())
        for attempt in range(8):
            try:return original(path,value)
            except PermissionError:
                events.append({'file':Path(path).name,'attempt':attempt+1})
                if attempt==7:raise
                time.sleep(min(.1*2**attempt,2.))
    d.save=retry_save
    checkpoint=d.load(d.OUT/'context_CONTRASTS_V3.json')
    retry_save(d.OUT/'EXECUTION_AMENDMENT.json',{'status':'BOUNDED_ATOMIC_WRITE_RETRY_ONLY',
        'runner_sha256':d.sha(Path(__file__)),'frozen_manifest_sha256':d.sha(d.OUT/'MANIFEST.json'),
        'completed_context_pairs_before':len(checkpoint['pairs']),
        'reason':'Two terminal invocations hit Windows WinError5 during atomic checkpoint replacement; earlier lane checkpoints and25 context pairs were preserved. A premature review attempt stopped on missing completion marker.',
        'scientific_driver_or_sources_changed':False,'retry_limit':8,'maximum_retry_delay_seconds':2})
    d.contrasts();d.verify()
    retry_save(d.OUT/'EXECUTION_RECOVERY.json',{'status':'COMPLETE','events':events,
        'seconds_this_resume_invocation':time.monotonic()-start,
        'timing_scope':'Excludes two earlier partial invocations; no exact total runtime claim.',
        'runner_sha256':d.sha(Path(__file__))})


if __name__=='__main__':main()
