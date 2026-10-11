"""Summarize the archived protocol/result history without inferring private dates."""
from pathlib import Path
import hashlib,json

ROOT=Path(__file__).resolve().parent

def main():
    sources=ROOT/'sources'
    protocol=json.loads((sources/'REGISTERED_PROTOCOL.json').read_text())
    history=json.loads((sources/'PROTOCOL_COMMITS.json').read_text())
    result_history=json.loads((sources/'THINKFIX_RESULT_COMMITS.json').read_text())
    access=json.loads((sources/'REPOSITORY_ACCESS.json').read_text())
    earliest=min(history,key=lambda r:r['commit']['committer']['date'])
    firstresult=min(result_history,key=lambda r:r['commit']['committer']['date'])
    out={'status':'CHRONOLOGY_VERIFIED_WITH_LIMITS','protocol_run_directory_date':'2026-10-03',
         'earliest_versioned_protocol_commit':{'sha':earliest['sha'],'committer_date':earliest['commit']['committer']['date']},
         'earliest_versioned_corrected_results_commit':{'sha':firstresult['sha'],'committer_date':firstresult['commit']['committer']['date']},
         'protocol_precedes_corrected_result_export':earliest['commit']['committer']['date']<firstresult['commit']['committer']['date'],
         'first_private_registration_datetime_verified':False,
         'original_public_push_datetime_verified':False,
         'q80_never_selected_on_historical_results_verified':False,
         'rules':protocol['decisions'],
         'interpretation':'The currently public Git history dates the earliest protocol commit to 8 October and corrected-result export commit to 11 October, using committer timestamps. Its run directory names 3 October. This verifies versioned commit chronology, not the original public push time, private registration time or absence of prior historical selection. TOP_2 is expressly post hoc. The protocol leaves final choice to Omri after comparison.',
         'repository_access':access,
         'advisor_membership_required':access['private'],
         'claude_artifact_access':'not independently verified during this run; web reader cannot access the URL',
         'publication_state':'New v11 local only; no commit, push, or Claude publication performed',
         'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [sources/'REGISTERED_PROTOCOL.json',sources/'PROTOCOL_COMMITS.json',sources/'THINKFIX_RESULT_COMMITS.json',sources/'REPOSITORY_ACCESS.json']}}
    (ROOT/'Q80_PROVENANCE.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:out[k] for k in ['earliest_versioned_protocol_commit','earliest_versioned_corrected_results_commit','protocol_precedes_corrected_result_export','repository_access']}))

if __name__=='__main__': main()
