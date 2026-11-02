"""sub5_sla.py: ProcessBench SLA (Mind the Gap Table 3 metric) of SUB5 WITHOUT the PRM, standalone and backbone.

SUB5 = L-SML over 6 of the 16 channels (q15_H1, chosen_surprisal, top15_turnover, dominant_freq16, bocpd_p0,
digit_spread), built exactly as in results/prm_sub5_all_benchmarks_v1/sub5_all.py (sub5_scores, with_prm=False).
Arms: 'SUB5 alone' (L-SML), 'SUB5 mean' (equal-weight average of the 6), and TELE_N16 for parity with RESULTS.csv.
Rule: GAP first flag (and Argmax). SUB5 was selected with PRMBench+Socratic labels, so ProcessBench is out of sample.
Run on the PC (needs hd_collection_data, same as sub5_all.py):
    python results/processbench_gap_sla_standalone_v1/sub5_sla.py   -> SUB5_SLA.csv
"""
import csv, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'prm_sub5_all_benchmarks_v1'))
from evaluate_sla_standalone import gap_predict, SUBSETS, MIND_THE_GAP  # noqa: E402
import sub5_all as A  # noqa: E402

rows = []
ref = {(r['model'], r['mode'], r['arm'], r['rule']): float(r['macro_sla']) for r in csv.DictReader(open(HERE / 'RESULTS.csv'))}
for mode in ('sa', 'prm'):
    for prm in A.PRMS:
        if A.cab_dir(mode, prm, 'processbench') is None:
            print('MISSING', mode, prm); continue
        key = f'cab|{mode}|processbench'
        A.REG[key] = dict(mode=mode, bench='processbench', common=A.common_uids('processbench'))
        c, sc, info = A.sub5_scores(key, prm, False)
        assert c.off[-1] == sum(A.PB_GOLD[u]['n_steps'] for u in c.ids)    # no empty steps on ProcessBench
        lab = np.array([A.PB_GOLD[u]['label'] for u in c.ids]); sub = np.array([A.PB_GOLD[u]['subset'] for u in c.ids])
        e = lab != -1
        arms = {'TELE_N16': c.arms['TELE_N16'], 'SUB5 alone': sc['SUB5 alone'], 'SUB5 mean': sc['SUB5 mean']}
        for arm, s in arms.items():
            if s is None: print('invalid', mode, prm, arm, info); continue
            P = np.array([gap_predict(np.asarray(s[a:b], float)) for a, b in zip(c.off[:-1], c.off[1:])])
            for rule, col in (('GAP_first_flag', 0), ('Argmax_top1', 1)):
                hit = (P[:, col] == lab)[e]; se = sub[e]
                per = {s_: 100 * hit[se == s_].mean() for s_ in SUBSETS}
                macro = float(np.mean(list(per.values())))
                m = {'sa': 'standalone', 'prm': 'backbone (Step 483 input)'}[mode]
                r0 = ref.get((A.PFILE.get(prm, prm), m, arm, rule))
                rows.append(dict(mode='standalone' if mode == 'sa' else 'backbone', prm=prm, arm=arm, rule=rule,
                                 macro_sla=round(macro, 2), overall_sla=round(100 * float(hit.mean()), 2),
                                 **{f'{k}_sla': round(v, 2) for k, v in per.items()},
                                 parity_ref=r0, parity_ok=None if r0 is None else abs(round(macro, 2) - r0) < 0.006,
                                 lsml_K=info.get('alone_K')))
                print(rows[-1], flush=True)
with open(HERE / 'SUB5_SLA.csv', 'w', newline='', encoding='utf-8') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print('MtG Qwen3-8B macro', MIND_THE_GAP['Qwen3-8B (Shannon Drop)']['macro'])
