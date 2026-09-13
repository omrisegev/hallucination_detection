"""Linux lifecycle test using a fake worker; never starts scientific fitting."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


def run():
    if sys.platform != 'linux':
        return dict(status='NOT_APPLICABLE',reason='Slurm signal lifecycle is Linux only')
    supervisor = Path(__file__).with_name('complete_conditional_iu_fusion.py')
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);out=root/'results/conditional_iu_fusion_v1/position'
        (root/'scripts').mkdir();(out/'smoke').mkdir(parents=True)
        (root/'scripts/run_conditional_iu_fusion.py').write_text(
            'import os,time\nfrom pathlib import Path\n'
            'out=Path(__file__).resolve().parents[1]/"results/conditional_iu_fusion_v1/position/smoke"\n'
            '(out/"RUN.lock").write_text(str(os.getpid()))\n'
            '(out/"saved_checkpoint").write_text("preserve")\n'
            'time.sleep(120)\n')
        loader=('import importlib.util,sys\nfrom pathlib import Path\n'
                'spec=importlib.util.spec_from_file_location("supervisor",sys.argv[1])\n'
                'm=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)\n'
                'm.ROOT=Path(sys.argv[2]);m.OUT=m.ROOT/"results/conditional_iu_fusion_v1/position"\n'
                'sys.argv=["supervisor","--source-root",sys.argv[2],"--family","position"];m.main()\n')
        parent=subprocess.Popen([sys.executable,'-c',loader,str(supervisor),str(root)])
        def wait_for(predicate):
            deadline=time.monotonic()+15
            while time.monotonic()<deadline:
                if predicate():return
                time.sleep(.05)
            raise AssertionError('supervisor test timed out')
        child=None
        try:
            wait_for(lambda:(out/'smoke/saved_checkpoint').exists())
            child=int((out/'smoke/RUN.lock').read_text())
            parent.send_signal(signal.SIGTERM)
            wait_for(lambda:not (out/'PROGRAM.lock').exists())
            state=json.loads((out/'PROGRAM_STATE.json').read_text())
            assert state['status']=='CHECKPOINTED_SCHEDULER_STOP'
            assert not (out/'smoke/RUN.lock').exists()
            assert (out/'smoke/saved_checkpoint').read_text()=='preserve'
            try:os.kill(child,0)
            except ProcessLookupError:pass
            else:raise AssertionError('worker remains alive after checkpoint stop')
        finally:
            if parent.poll() is None:parent.kill()
            parent.wait(timeout=10)
            if child:
                try:os.kill(child,signal.SIGKILL)
                except ProcessLookupError:pass
    return dict(status='PASS',terminated_owned_worker=True,released_owned_locks=True,
                preserved_checkpoint=True,scientific_training=False)


if __name__=='__main__':print(run(),flush=True)
