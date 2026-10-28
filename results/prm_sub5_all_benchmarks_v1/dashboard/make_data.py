"""SUB5_ALL.csv -> dashboard/scores.csv (one row per benchmark, telemetry source, PRM, arm and metric, with labels)."""
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = [('prmbench', 'PRMBench'), ('socratic', 'Socratic'), ('hard2verify', 'Hard2Verify'), ('processbench', 'ProcessBench'),
         ('crv_arith', 'CRV-Arith'), ('crv_gsm8k', 'CRV-GSM8K'), ('prm800k', 'PRM800K'), ('deltabench', 'DeltaBench')]
PRM = {'qwen_prm_7b': 'Qwen-PRM-7B', 'qwen_prm_72b': 'Qwen-PRM-72B', 'pure_prm': 'PURE-PRM-7B', 'rlhflow_prm': 'RLHFlow-PRM-8B'}
MAIN_LABEL = {'prmbench': 'PRMScore', 'socratic': 'Step macro-F1', 'hard2verify': 'Step balanced F1', 'processbench': 'Average F1 (NC rule)',
              'crv_arith': 'Step macro-F1', 'crv_gsm8k': 'Step macro-F1', 'prm800k': 'Step macro-F1', 'deltabench': 'Step macro-F1'}
IN_SAMPLE = {'prmbench', 'socratic'}


def main():
    order = {b: i for i, (b, _) in enumerate(BENCH)}; name = dict(BENCH); out = []
    for r in csv.DictReader(open(HERE.parent / 'SUB5_ALL.csv', encoding='utf-8')):
        if r['value'] in ('', 'nan', 'None'): continue
        b = r['benchmark']
        out.append(dict(benchmark=b, benchmark_label=name[b], bench_order=order[b], mode=r['mode'], prm=r['prm'], prm_label=PRM[r['prm']],
                        sample='Used to choose SUB5' if b in IN_SAMPLE else 'Out of sample', arm=r['arm'], metric=r['metric'],
                        value=r['value'], label_main=MAIN_LABEL[b], label_within_auc='Within-answer AUC',
                        label_crop_kept_a05='CROP kept % (α 0.05)', label_crop_auroc='CROP step AUROC'))
    with open(HERE / 'scores.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, list(out[0])); w.writeheader(); w.writerows(out)
    print(len(out), 'rows')


if __name__ == '__main__':
    main()
