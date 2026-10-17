"""Configure correctly typed Windows priority calls; run the frozen driver.

The preflight driver's untyped HANDLE call could not set priority on64-bit
Windows. This execution supplement changes no scorer, selector or metric.
"""
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/fusion_entropy_sampling_v1'
status='one worker; default priority'
if os.name=='nt':
    k=ctypes.windll.kernel32
    k.GetCurrentProcess.restype=wintypes.HANDLE
    k.SetPriorityClass.argtypes=(wintypes.HANDLE,wintypes.DWORD)
    k.SetPriorityClass.restype=wintypes.BOOL
    k.GetPriorityClass.argtypes=(wintypes.HANDLE,)
    k.GetPriorityClass.restype=wintypes.DWORD
    assert k.SetPriorityClass(k.GetCurrentProcess(),0x4000)
    assert k.GetPriorityClass(k.GetCurrentProcess())==0x4000
    status='BELOW_NORMAL_VERIFIED'
record=dict(pid=os.getpid(),priority=status,launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            manifest_sha256=hashlib.sha256((OUT/'MANIFEST.json').read_bytes()).hexdigest(),
            scope='Typed Windows HANDLE declarations only; frozen scientific driver unchanged.')
(OUT/'EXECUTION_ENVIRONMENT.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
sys.argv=[str(ROOT/'scripts/run_fusion_entropy_sampling_v1.py'),'--phase','run']
runpy.run_path(sys.argv[0],run_name='__main__')
