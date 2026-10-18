"""Collect every paper metric into METRICS.csv (run after the four metric scripts)."""
import csv, json
import numpy as np
from common import HERE, ARMS, ARM_NOTE
from prism_metrics import PRISM_T2, PRISM_T7, T7_COLS, PRISM_T6

R = lambda n: json.load(open(HERE / n))
COLS = ['paper', 'metric', 'definition', 'their_method', 'their_value', 'our_arm', 'our_value', 'N',
        'access_labels_used', 'match_status', 'protocol_notes']
ACC_OURS = {'PRM_native': 'PRM trained with labels by its authors; no labels used by us',
            'LSML_CONT': 'label-free fusion; 3 channels CHOSEN WITH PRMBench labels; PRM itself label-trained',
            'CHAIN_N16': 'label-free fusion and channel list; PRM itself label-trained; contains 3 digit channels',
            'CHAIN_N13': 'label-free fusion and channel list (digit-free); PRM itself label-trained'}
rows = []
add = lambda **k: rows.append({c: k.get(c, '') for c in COLS})




# ---------------- PRISM ----------------
P = R('PRISM_RESULTS.json'); NA, NS = P['n_answers_nonctrl'], P['n_steps_nonctrl']
N = f'{NA} answers / {NS} steps (non-control PRMBench)'
DEF = {'PRMScore': 'official PRMBench: mean of F1(valid) and F1(error), pooled over non-control steps',
       'FPR': 'error steps passed as valid / all error steps (positive class = valid step)',
       'TNR': 'error steps flagged / all error steps (Neg-Acc)', 'TPR': 'valid steps kept / all valid steps (Pos-Acc)',
       'FNR': 'valid steps flagged / all valid steps', 'Precision': 'valid-predicted steps that are valid'}
for met in ['PRMScore', 'FPR', 'TNR', 'TPR', 'FNR', 'Precision']:
    pool_note = ('PRISM Table 2 FNR/Precision are only reproduced when the synthetic correct-control rows are pooled in '
                 '(their TPR/TNR/PRMScore match the official non-control pool; TPR+FNR != 100 in their table). '
                 'Ours shown on the official pool; control-row pool value in notes.') if met in ('FNR', 'Precision') else ''
    for a in ARMS:
        ours = P['rates'][a][met]; alt = P['rates'][a + '__with_control_rows'][met]
        base = a == 'PRM_native'
        tm = 'Baseline (Qwen-PRM-7B)' if base else 'PRISM'
        status = 'SAME' if met in ('PRMScore', 'FPR', 'TNR', 'TPR') else 'RECOMPUTED-ON-THEIR-PROTOCOL'
        note = ('parity row: our native PRM vs their baseline. ' if base else
                'Different decision rule: ours flags the top 20% answer-z steps (Q80, label-free); PRISM uses reward<0.5 '
                'after re-training the PRM with labels. ') + pool_note + (f' With control rows: {alt:.2f}.' if pool_note else '')
        add(paper='PRISM 2606.09078', metric=f'PRMBench {met} (Table 2)', definition=DEF[met], their_method=tm,
            their_value=PRISM_T2[tm][met], our_arm=a, our_value=round(ours, 2), N=N, access_labels_used=ACC_OURS[a],
            match_status=status, protocol_notes=note)
MT = P['matched_tpr_87_30']
for met, tv in (('FPR', PRISM_T2['PRISM']['FPR']), ('PRMScore', PRISM_T2['PRISM']['PRMScore']), ('Precision', PRISM_T2['PRISM']['Precision'])):
    for a in ARMS:
        nm = 'PRM_native (re-thresholded)' if a == 'PRM_native' else a
        add(paper='PRISM 2606.09078', metric=f"PRMBench {met} at PRISM's TPR (87.30)",
            definition=f"{DEF[met]}; global threshold set so that TPR = {MT[a]['TPR']:.2f} (closest to 87.30)",
            their_method='PRISM', their_value=tv, our_arm=nm, our_value=round(MT[a][met], 2), N=N,
            access_labels_used=ACC_OURS[a] + '; operating point chosen to match their TPR (label-informed reading, not a deployable rule)',
            match_status='APPROXIMATE',
            protocol_notes=('control: the untouched PRM with only its threshold moved (raw risk, global cut)' if a == 'PRM_native'
                            else 'global cut on the answer-z fused score, not our deployed Q80 rule'))
CATN = {'NR': 'redundency', 'NCL': 'circular', 'ES': 'counterfactual', 'SC': 'step_contradiction', 'DC': 'domain_inconsistency',
        'CI': 'confidence', 'PS': 'missing_condition', 'DR': 'deception', 'MS': 'multi_solutions'}
for k, col in enumerate(T7_COLS[1:], start=1):
    for a in ARMS:
        base = a == 'PRM_native'; tm = 'Baseline' if base else '132k Curriculum-4 (best FPR)'
        note = 'MS has no error steps: valid-step F1 (published scale ~99.7). ' if col == 'MS' else ''
        note += 'Category average = plain mean of its sub-categories (checked against their table). ' if col in ('Simplicity', 'Soundness', 'Sensitivity') else ''
        if not base: note += 'Our arms use Q80 (flags 20% of steps everywhere, including MS where nothing is wrong).'
        add(paper='PRISM 2606.09078', metric=f'PRMBench {col} (Table 7)',
            definition=f'official per-classification PRMScore ({CATN.get(col, "category mean")})', their_method=tm,
            their_value=PRISM_T7[tm][k], our_arm=a, our_value=round(P['categories'][a][col], 2), N=N,
            access_labels_used=ACC_OURS[a], match_status='SAME', protocol_notes=note)
for sub, (tpr, tnr, f1) in PRISM_T6.items():
    o = P['processbench_native'][sub]
    for met, tv, ov in (('TPR (acc. on erroneous samples)', tpr, o['TPR']), ('TNR (acc. on correct samples)', tnr, o['TNR']), ('F1', f1, o['F1'])):
        add(paper='PRISM 2606.09078', metric=f'ProcessBench {sub} {met} (Table 6)',
            definition='official ProcessBench: predicted first error = first step with reward < 0.5', their_method='Baseline',
            their_value=tv, our_arm='PRM_native', our_value=round(ov, 2), N=f"{o['n_err']} err + {o['n_cor']} correct samples",
            access_labels_used=ACC_OURS['PRM_native'], match_status='SAME',
            protocol_notes=('Their Omni-MATH/OlympiadBench cells look transposed: ours give olympiad 54.8/85.3/66.7, '
                            'omnimath 54.7/83.4/66.1; their omnimath TNR 66.1 and F1 83.4 are not consistent with TPR 54.8.')
            if sub in ('omnimath', 'olympiadbench') else 'parity with their baseline')
        for a in ARMS[1:]:
            add(paper='PRISM 2606.09078', metric=f'ProcessBench {sub} {met} (Table 6)', definition='as above',
                their_method='PRISM CL1-CL3 rows', their_value='', our_arm=a, our_value='', N='',
                access_labels_used=ACC_OURS[a], match_status='GAP',
                protocol_notes='GAP: no PRM-backbone telemetry was collected on ProcessBench (needs new GPU inference)')

# ---------------- CROP ----------------
C = R('CROP_RESULTS.json'); sb = C['subset']
NC = f"{sb['n_traces']} traces / {sb['n_steps']} steps; 10 splits x 20% test"
CROP_T6 = {'Random': (0.4994, 0.0050, 8.7, 0.5, 4.7, 0.4), 'Token/format': (0.7751, 0.0034, 16.2, 1.0, 5.0, 0.7),
           'Trace features': (0.7812, 0.0034, 15.9, 0.7, 4.6, 0.5), 'Direct PRM': (0.8010, 0.0049, 4.8, 0.2, 0.6, 0.1),
           'Trace+PRM': (0.8336, 0.0027, 21.8, 0.9, 5.0, 0.6)}
MAP = {'Random': ('Random', 'SAME', 'label-free; uniform scores', ''),
       'Direct PRM (1-reward)': ('Direct PRM', 'RECOMPUTED-ON-THEIR-PROTOCOL', ACC_OURS['PRM_native'] + '; CROP threshold uses calibration labels', 'fixed 101-point grid as in their code'),
       'Direct PRM (1-reward), quantile grid': ('Direct PRM', 'APPROXIMATE', ACC_OURS['PRM_native'] + '; CROP threshold uses calibration labels', 'bridge: same scores, 201 train-quantile grid (the grid our fused arms need). Shows that their 4.8% is mostly a grid-resolution effect'),
       'Position/length LR (supervised)': ('Token/format', 'APPROXIMATE', 'SUPERVISED: LR trained on train-split step labels; CROP threshold uses calibration labels', 'position/length features only (no step text available locally: no char/digit/punctuation counts)'),
       'TELE13+PRM LR (supervised)': ('Trace+PRM', 'APPROXIMATE', 'SUPERVISED: LR trained on train-split step labels; CROP threshold uses calibration labels', 'our features (PRM risk + 13 digit-free backbone channels), not their Llama-3.1-8B trace features'),
       'LSML_CONT': ('Trace+PRM', 'RECOMPUTED-ON-THEIR-PROTOCOL', ACC_OURS['LSML_CONT'] + '; CROP threshold uses calibration labels', 'fused score fitted label-free on all 6,969 answers (incl. these traces); 201 train-quantile grid'),
       'CHAIN_N16': ('Trace+PRM', 'RECOMPUTED-ON-THEIR-PROTOCOL', ACC_OURS['CHAIN_N16'] + '; CROP threshold uses calibration labels', 'fused score fitted label-free on all 6,969 answers; 201 train-quantile grid'),
       'CHAIN_N13': ('Trace+PRM', 'RECOMPUTED-ON-THEIR-PROTOCOL', ACC_OURS['CHAIN_N13'] + '; CROP threshold uses calibration labels', 'fused score fitted label-free on all 6,969 answers; 201 train-quantile grid')}
common = 'Subset = first 5,000 non-control rows in Hub order (APPROXIMATE: 67,465 vs their 67,445 steps; our loader drops 5 duplicate rows they keep). Our splits re-implement crop/splits.py; exact row assignment may differ. '
for nm, (tm, st, acc, note) in MAP.items():
    a = C['arms'][nm]; t = CROP_T6[tm]
    for met, tv, ov, oci, d in (('Step AUROC (pooled)', t[0], a['auroc'], a['auroc_ci95'], 'AUROC of step error vs score, pooled over all test-split steps'),
                                ('Prefix kept % (alpha=0.05)', t[2], a['kept'], a['kept_ci95'], 'mean over test traces of retained-prefix length / trace length'),
                                ('Prefix risk % (alpha=0.05)', t[4], a['risk'], a['risk_ci95'], 'share of test traces whose retained prefix contains an annotated error')):
        add(paper='CROP 2605.30085', metric=f'PRMBench {met} (Table 6)', definition=d, their_method=tm, their_value=tv,
            our_arm=nm, our_value=f'{ov:.4f} +- {oci:.4f}' if 'AUROC' in met else f'{ov:.1f} +- {oci:.1f}', N=NC,
            access_labels_used=acc, match_status=st, protocol_notes=common + note)
for a in ARMS:
    add(paper='CROP 2605.30085', metric='PRMBench within-answer AUC on the same 5,000 traces (NOT a CROP metric)',
        definition='mean over answers with both classes of the per-answer AUC', their_method='(not reported)', their_value='',
        our_arm=a, our_value=round(C['within_auc_subset'][a], 4), N='5,000 traces (answers with both classes)',
        access_labels_used=ACC_OURS[a], match_status='GAP',
        protocol_notes='shown so the pooled-vs-within difference is visible: our fused scores are partly answer-standardized')

# ---------------- SLIDER ----------------
S = R('SLIDER_RESULTS.json')
NSL = f"{S['n_problems']} redundancy problems / {S['n_steps']} steps ({S['n_redundant_steps']} redundant, {100*S['redundant_rate']:.1f}%)"
for a in ARMS:
    r = S['arms'][a]
    for key, label, st, acc in (
            ('all_steps__LR_balanced__acc', 'accuracy, class-balanced LR threshold', 'APPROXIMATE', 'threshold fitted on held-out labels (as SLIDER)'),
            ('all_steps__LR_balanced__bal_acc', 'balanced accuracy, class-balanced LR threshold', 'APPROXIMATE', 'threshold fitted on held-out labels (as SLIDER)'),
            ('all_steps__LR_unweighted__acc', 'accuracy, unweighted LR threshold', 'APPROXIMATE', 'threshold fitted on held-out labels (as SLIDER)'),
            ('steps_ge2__LR_balanced__acc', 'accuracy, class-balanced LR threshold, steps >= 2 only', 'APPROXIMATE', 'threshold fitted on held-out labels (as SLIDER)'),
            ('all_steps__label_free_Q80__acc', 'accuracy, our label-free Q80 rule (no threshold calibration)', 'APPROXIMATE', 'no labels for the threshold'),
            ('all_steps__label_free_Q80__bal_acc', 'balanced accuracy, our label-free Q80 rule', 'APPROXIMATE', 'no labels for the threshold')):
        v = r[key]; ci = r.get(key + '_ci95')
        add(paper='SLIDER 2610.00571', metric=f'PRMBench redundancy step {label} (Fig. 7)',
            definition='positive = step annotated redundant (error step of the redundency class); test steps of 10 random 50/50 trace splits',
            their_method='Step-RRI (InfoGain 48.38, Cosine 51.60)', their_value=61.91, our_arm=a,
            our_value=f'{v:.2f} +- {ci:.2f}' if ci is not None else round(v, 2), N=NSL,
            access_labels_used=ACC_OURS[a] + '; ' + acc, match_status=st,
            protocol_notes=f"SLIDER does not state split, step scope, class weighting or whether accuracy is balanced. Always-'not redundant' gives {r['all_steps__majority_acc']:.2f}% plain accuracy, so their 61.91% cannot be plain accuracy with an unweighted threshold at the natural base rate.")

# ---------------- Optimal Aggregation ----------------
OA = {'Optimal (oracle)': 66.5, 'BoN': 61.8, 'MV': 57.1, 'Vanilla WV': 60.8, 'KDE WV': 60.2, 'Linear WV': 62.9, 'Logit WV': 63.3}
for tm, tv in OA.items():
    for a in ARMS:
        add(paper='OptAgg 2510.13918', metric='MATH answer accuracy at N=32 (Table 1, Qwen2.5-PRM-7B, mean of 5 LLMs)',
            definition='final-answer accuracy of the response chosen by BoN / (weighted) majority vote over 32 samples',
            their_method=tm, their_value=tv, our_arm=a, our_value='', N='', access_labels_used='',
            match_status='GAP', protocol_notes='GAP: needs N sampled responses per question with graded final answers; PRMBench has one fixed trace per perturbation and no final-answer grading. Needs new generation + PRM/telemetry inference on e.g. MATH500.')
G = R('AGGREGATION_PROXY.json')
for a in ARMS:
    add(paper='OptAgg 2510.13918', metric='PROXY (not their metric): pick-the-clean-trace rate within a PRMBench source group',
        definition='per source group pick the trace with the lowest max-step risk; hit if it has no annotated error',
        their_method='(no counterpart)', their_value='', our_arm=a, our_value=round(G['arms'][a], 2),
        N=f"{G['n_groups']} source groups (random pick {G['random_pick_hit_rate']:.1f}%)", access_labels_used=ACC_OURS[a],
        match_status='GAP', protocol_notes='different task; fused arms are partly answer-standardized and are not designed to rank answers against each other')

with open(HERE / 'METRICS.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, COLS); w.writeheader(); w.writerows(rows)
print(len(rows), 'rows')
