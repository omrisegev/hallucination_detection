"""Inventory frozen outputs without calling a scorer or choosing new methods."""
import csv
import hashlib
import io
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'results/fusion_trajectory_imm_v1/EVALUATION.json'
OUT=ROOT/'results/localization_evidence_ledger_v1'


def category(arm):
    if arm.startswith('sample_'):return 'window_selection'
    if arm.startswith('traj_'):return 'curve_fusion_and_readout'
    if arm.startswith('gap'):return 'confidence_reparameterization'
    if arm.startswith(('ar1__','last__','ema32__')):return 'prediction_residual_features'
    if arm.startswith('pair_'):return 'pair_grouping'
    if 'cond' in arm:return 'conditioning_and_graph' if 'graph' in arm else 'conditioning'
    if 'graph' in arm:return 'graph_and_controls'
    return 'original_bank_and_routing_reference'


def main():
    e=json.loads(SOURCE.read_text(encoding='utf-8'));rows=e['rows'];metrics=e['metrics']
    assert len(rows)==110 and len(metrics)==176
    assert len({r['uid'] for r in rows})==110
    ordered=sorted(rows,key=lambda r:r['uid']);records=[];groups={}
    for arm,m in metrics.items():
        payload=[dict(uid=r['uid'],valid=r['valid'][arm],decision_valid=r['decision_valid'][arm],
                      prediction=r['predictions'].get(arm),scores=r['scores'].get(arm)) for r in ordered]
        # Equality is only of saved final scores/validity/decisions on current110.
        # It does not identify implementations or claim equal unseen behavior.
        h=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        groups.setdefault(h,[]).append(arm)
        prm=[r for r in rows if r['cell'].startswith('prm')]
        pb=[r for r in rows if r['cell'].startswith('pb_')]
        assert sum(r['valid'][arm] for r in prm)==m['prm']['answers']
        assert sum(c['answers'] for c in m['pb']['cells'].values())==86
        records.append(dict(method=arm,category=category(arm),cohort='current110_v3labels_v2groups',
            prm_population=24,prm_valid=m['prm']['answers'],prm_mixed=m['prm']['mixed_answers'],
            prm_pooled_auc=m['prm']['auroc'],prm_within_answer_auc=m['prm']['within_answer_auc'],
            pb_population=86,pb_valid_decisions=sum(r['decision_valid'][arm] for r in pb),pb_macro_f1=m['pb']['macro_f1'],
            output_fingerprint=h,first_identical_output_entry=groups[h][0],
            full_population_measured=False,runtime_by_method=None,
            significance='Consult registered paired contrasts; no new uncertainty computed',
            source=str(SOURCE)))
    OUT.mkdir(parents=True,exist_ok=True)
    buf=io.StringIO(newline='');writer=csv.DictWriter(buf,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    (OUT/'CURRENT110.csv').write_text(buf.getvalue(),encoding='utf-8')
    result=dict(status='FROZEN_OUTPUT_INVENTORY_ONLY',source=str(SOURCE),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                displayed_entries=176,distinct_saved_output_fingerprints=len(groups),records=records,
                identical_output_groups=[v for v in groups.values() if len(v)>1],
                limitations=['Only current110; earlier58 and full historical contracts remain separate and pending in this ledger.',
                             'Equal fingerprints mean identical saved final output, not algorithm equivalence.',
                             'No new scoring, causal attribution, uncertainty, runtime claim or winner selection.'])
    (OUT/'CURRENT110.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    (OUT/'README.md').write_text('# Evidence ledger: current110 inventory\n\n'
        '176 displayed entries on the same corrected development cohort. '
        f'{len(groups)} distinct saved-output fingerprints; aliases are exposed in the JSON. '
        'This is not a count of independent methods or experiments.\n\n'
        '[CSV](CURRENT110.csv) | [JSON and provenance](CURRENT110.json)\n\n'
        'Coverage stays visible. PRMB AUROC is conditional on valid scores; do not subtract '
        'two values with different coverage as a matched improvement. PB retains all86 rows. '
        'For paired effects consult the original contrast artifacts. Per-method runtime is '
        'unknown here, not zero. Earlier58, historical refits and full-population results '
        'remain separate additions. No new model or score was fitted.\n',encoding='utf-8')
    print(json.dumps(dict(entries=176,output_fingerprints=len(groups),repeated_output_groups=sum(len(v)>1 for v in groups.values()),
                          source_hash=result['source_sha256'])))


if __name__=='__main__':main()
