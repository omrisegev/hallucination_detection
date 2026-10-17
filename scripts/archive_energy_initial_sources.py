"""Recover initial source bytes only when they match the pre-run SHA256."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/energy_context_stability_v1'
manifest=json.loads((OUT/'PROVENANCE.json').read_text())
for name,expected in manifest.items():
    original=(ROOT/name).read_bytes()
    if hashlib.sha256(original).hexdigest()!=expected:
        content=original.decode('utf8').replace('\r\n','\n')
        if name=='spectral_utils/energy_context_stability.py':
            content=content.replace('from itertools import combinations\n','')
            content=content.replace('from .cca_iu_isolation import iu_moments\n',
                                    'from .cca_iu_isolation import iu_moments, simplex_qp\n')
            start=content.index('def simplex_qp(');end=content.index('def history_landmarks(')
            content=content[:start]+content[end:]
        elif name=='scripts/run_energy_context_stability.py':
            content=content.replace("OUT=ROOT/'results/energy_context_stability_v2'","OUT=ROOT/'results/energy_context_stability_v1'")
            content=content.replace("           'docs/experiments/ENERGY_CONTEXT_STABILITY_NUMERICAL_AMENDMENT_20260915.md',\n",'')
        candidates=[content.encode('utf8'),content.replace('\n','\r\n').encode('utf8')]
        matches=[b for b in candidates if hashlib.sha256(b).hexdigest()==expected]
        if not matches:raise ValueError('Cannot recover original source bytes: '+name)
        original=matches[0]
    dest=OUT/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(original)
    print('Verified original source',name)
