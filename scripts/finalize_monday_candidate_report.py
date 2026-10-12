"""Match ProcessBench forensics denominators using the saved answer rows only."""
from pathlib import Path
import collections
import csv
import datetime
import gzip
import hashlib
import json

KEYS=['bank','variant','recipe','decision_rule','cell']
FIELDS=['error_position_quartile','length_bin']

def flag(value):
    if value not in ('True','False'): raise ValueError('Missing/invalid decision flag: '+repr(value))
    return int(value=='True')

def add(counts,row):
    """All hit rates below use erroneous answers; clean flag rates have their own N."""
    counts[0]+=1
    if int(row['target'])>=0:
        counts[1]+=1; counts[3]+=flag(row['ours_argmax_hit']); counts[4]+=flag(row['ours_decision_hit']); counts[5]+=flag(row['MTG_native_argmax_hit'])
    else:
        counts[2]+=1; counts[6]+=flag(row['ours_clean_flag']); counts[7]+=flag(row['MTG_native_clean_flag'])

def values(counts):
    n,ne,nc,ownpeak,owndec,mtgpeak,ownclean,mtgclean=counts
    return dict(n_answers=n,n_erroneous=ne,n_clean=nc,ours_argmax_hit=ownpeak/ne if ne else None,
        ours_decision_hit=owndec/ne if ne else None,MTG_native_argmax_hit=mtgpeak/ne if ne else None,
        ours_clean_flag_rate=ownclean/nc if nc else None,MTG_native_clean_flag_rate=mtgclean/nc if nc else None)

def finalize(directory):
    directory=Path(directory)
    audit=json.loads((directory/'AUDIT.json').read_text(encoding='utf8'))
    if audit['status']!='CONTENT_AND_PRESELECTION_FORENSICS_COMPLETE_LAYER_LR_SNAPSHOT': raise RuntimeError('Report not complete')
    source=directory/'MTG_FORENSICS.csv.gz'
    source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    record_path=directory/'MTG_SUMMARY_REPAIR.json'
    if record_path.exists() and json.loads(record_path.read_text())['source_sha256']==source_hash: return
    tables={field:collections.defaultdict(lambda:[0]*8) for field in FIELDS}; checked=0
    with gzip.open(source,'rt',encoding='utf8',newline='') as stream:
        for row in csv.DictReader(stream):
            for field,table in tables.items(): add(table[tuple(row[k] for k in KEYS+[field])],row)
            checked+=1
    if checked!=audit['competitors']['pb_answer_bank_candidate_rule_rows']: raise RuntimeError('Forensics coverage mismatch')
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')
    archive=directory/'snapshots'/('mtg_denominator_'+stamp); archive.mkdir(parents=True)
    corrected=[]
    for field,table in tables.items():
        target=directory/('MTG_BY_'+field.upper()+'.csv')
        old=target.read_bytes(); (archive/target.name).write_bytes(old)
        rows=[dict(zip(KEYS+[field],key),**values(counts)) for key,counts in sorted(table.items())]
        with target.open('w',encoding='utf8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        corrected.append({'file':target.name,'old_sha256':hashlib.sha256(old).hexdigest(),'new_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
            'summary_rows':len(rows),'answers_checked':sum(r['n_answers'] for r in rows)})
    record={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_file':str(source),'source_sha256':source_hash,
        'n_checked':checked,'correction':'All first-error hit means now condition on target>=0 for both methods; clean flag rates retain their separate denominator.',
        'unchanged':'All answer-level predictions, scores, core metrics, contrasts and benchmark inputs.',
        'files':corrected,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    record_path.write_text(json.dumps(record,indent=2)+'\n',encoding='utf8')

def regression_check():
    counts=[0]*8
    add(counts,dict(target='-1',ours_clean_flag='False',MTG_native_clean_flag='True'))
    add(counts,dict(target='2',ours_argmax_hit='True',ours_decision_hit='False',MTG_native_argmax_hit='False'))
    result=values(counts)
    assert result['n_erroneous']==result['n_clean']==1
    assert result['ours_decision_hit']==0. and result['ours_argmax_hit']==1.
    assert result['ours_clean_flag_rate']==0. and result['MTG_native_clean_flag_rate']==1.
    clean=[0]*8; add(clean,dict(target='-1',ours_clean_flag='False',MTG_native_clean_flag='True'))
    assert values(clean)['ours_decision_hit'] is None

if __name__=='__main__':
    regression_check()
    finalize(Path(r'C:\Users\omris\TAU\hallucination_detection\results\monday_candidate_panel_v1\run_20261004'))
