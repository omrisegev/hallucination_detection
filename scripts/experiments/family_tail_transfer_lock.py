"""Source-side transfer lock for the section-8 shortlist (handoff 2026-09-24): deployment fits on
source folds 0-3 (PB+PRMB, unlabeled), q80 thresholds on fold 4 (pooled, unlabeled), frozen channel
lists, source signs, families and the external comparison family.  Touches no external data.

    python -B scripts/experiments/family_tail_transfer_lock.py
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from calfix_common import MAIN, Population, dump, extgen_hashes, lsml_fit, partition_equal, sha, tail_marks  # noqa: E402

STAGE = ROOT / 'results/family_tail_calfix_v1'; RUN = STAGE / 'run_20260924_1542'
SCR = Path('C:/Users/DELL/AppData/Local/Temp/claude/c--Users-omris-TAU-hallucination-detection/2d14a8c9-8b3f-489b-b26f-812b4b84a8b3/scratchpad')
DROP = ['ct7_ve1', 'hist_entropy_series', 'hist_spilled_series', 'hist_trace_length_series']
pop = Population(); off = pop.off
POOL = np.load(SCR / 'pool_z.npy'); PN = json.loads((SCR / 'pool_names.json').read_text(encoding='utf8'))
assert sha(SCR / 'pool_z.npy') == json.loads((RUN / 'INPUT_MANIFEST.json').read_text())['pool_z']['sha256']
names = [c for c in PN if c not in DROP]; ix = {c: j for j, c in enumerate(names)}
Z = pop.answer_standardize(POOL[:, [PN.index(c) for c in names]])
D = json.loads((RUN / 'DESIGN.json').read_text(encoding='utf8'))['retrospective']; sign = {c: float(D['orientation'][c]) for c in names}; kept = D['kept']
S3 = json.loads((ROOT / 'results/named_group_fusion_v1/PROTOCOL.json').read_text(encoding='utf8'))['splits']['S3_M15']
B11N = PN[:11]
F15 = pop.answer_standardize(np.column_stack([np.column_stack([Z[:, ix[c]] * sign[c] for c in mem]).mean(1) for mem in S3.values()]))
X = {'B11': Z[:, [ix[c] for c in B11N]], 'K28': np.column_stack([Z[:, ix[c]] * sign[c] for c in kept]), 'A48o': np.column_stack([Z[:, ix[c]] * sign[c] for c in names]), 'F15': F15}
fam_anchor = list(S3).index('level_entropy')
T, degen = tail_marks(F15, off, .2, tie_aware=True)
fit_rows = pop.rows(pop.fold < 4); cal_rows = pop.rows(pop.fold == 4)

methods = {}
def add(name, Xb, w, cols, extra):
    s = pop.answer_z(Xb @ w); methods[name] = {'weights': dict(zip(cols, np.asarray(w, float))), 'q80_threshold_fold4': float(np.quantile(s[cal_rows], .8)), **extra}
fits = {'B11_lsml': lsml_fit(X['B11'][fit_rows], 0), 'K28_cov_lsml': lsml_fit(X['K28'][fit_rows], kept.index('q15_H1')),
        'F15_cov_lsml': lsml_fit(F15[fit_rows], fam_anchor), 'F15_tailtie_lsml': lsml_fit(T[fit_rows], fam_anchor, F15[fit_rows])}
for nm, b, cols in (('B11_lsml', 'B11', B11N), ('K28_cov_lsml', 'K28', kept), ('F15_cov_lsml', 'F15', list(S3)), ('F15_tailtie_lsml', 'F15', list(S3))):
    f = fits[nm]; add(nm, X[b], f['weights'], cols, {'K': f['K'], 'groups': dict(zip(cols, f['groups'].tolist())), 'small_m_guarded': f['small_m_guarded'], 'anchor_spearman': f['anchor_spearman']})
add('B11_equal', X['B11'], np.ones(11) / 11, B11N, {})
add('B11_partition_equal', X['B11'], partition_equal(fits['B11_lsml']['groups']), B11N, {'groups_from': 'B11_lsml deployment fit'})
add('K28_equal', X['K28'], np.ones(28) / 28, kept, {}); add('F15_equal', F15, np.ones(15) / 15, list(S3), {}); add('A48o_equal', X['A48o'], np.ones(48) / 48, names, {})
ct7z = pop.answer_z(pop.ct7); methods['ct7'] = {'recipe': 'frozen CT7 (external port spectral_utils/external_generalization/ct7.py), answer-z', 'q80_threshold_fold4': float(np.quantile(ct7z[cal_rows], .8))}
# the frozen external bundle fitted bank11 on the same folds 0-3 / fold 4: weights and thresholds must match exactly
cb = json.loads((MAIN / 'results/lsml_external_generalization_v1/evaluation/source/BUNDLE.json').read_text())
check = {'B11_weights_max_abs_diff': float(np.abs(np.array(cb['fit']['weights']) - fits['B11_lsml']['weights']).max()),
         'B11_groups_equal': cb['fit']['groups'] == fits['B11_lsml']['groups'].tolist(),
         'threshold_diffs': {a: abs(cb['thresholds'][c] - methods[a]['q80_threshold_fold4']) for a, c in (('B11_lsml', 'frozen_lsml'), ('B11_equal', 'frozen_equal'), ('B11_partition_equal', 'frozen_partition_equal'), ('ct7', 'ct7'))}}
check['status'] = 'PASS' if check['B11_weights_max_abs_diff'] < 1e-9 and check['B11_groups_equal'] and max(check['threshold_diffs'].values()) < 1e-9 else 'FAIL'
assert check['status'] == 'PASS', check

lock = {
 'lock': 'family_tail_transfer_lock_v1', 'date': datetime.now().isoformat(timespec='seconds'),
 'status': 'LOCKED (source side). External feature extraction and scoring have NOT started; they are gated by the parity gate below. Any change to rows, recipes or comparisons requires a new lock version with its reason, written before external scoring.',
 'basis': {'source_run': str(RUN.relative_to(ROOT)), 'metrics_sha256': sha(RUN / 'METRICS.csv'), 'contrasts_sha256': sha(RUN / 'CONTRASTS.csv'), 'models_sha256': sha(RUN / 'MODELS.jsonl'),
           'report': 'results/family_tail_calfix_v1/REPORT_HE.md', 'protocol_sha256': sha(STAGE / 'PROTOCOL.json')},
 'external_access_label': 'Socratic-PRMBench and Hard2Verify were already evaluated for bank11 and have influenced this discussion: any evaluation of this shortlist on them is an EXPLORATORY FOLLOW-UP, not confirmation. A confirmation claim needs a source-disjoint set not yet exposed to selection. Hard2Verify keeps its own official metric (balanced F1 of correct/incorrect step recall), never called PRMScore and never averaged with Socratic.',
 'rows': {
  'B11_lsml': 'frozen bank11 L-SML: learned anchor, identical to the existing external bundle',
  'F15_tailtie_lsml': 'new research candidate: equal mean of oriented members within each of 15 families, answer-z, then L-SML fitted on tie-aware top-20% marks and applied to the continuous family features. NOTE: the fit discovers K=2 groups (energy_cusum+entropy_cusum vs the other 13) on every source fold; the between-group split is fixed by eigenvector normalization, not learned. What is learned is the within-group SML weighting of the 13-family group. Describe it as tail-weighted family fusion, not as evidence for L-SML between-group learning.',
  'F15_cov_lsml': 'learned control: same families, continuous L-SML between families (learning-object question)',
  'K28_cov_lsml': 'learned control: L-SML directly on the 28 oriented channels (does the family layer add beyond filtering)',
  'K28_equal': 'control', 'F15_equal': 'control (matched equal of the candidate)', 'B11_equal': 'control', 'B11_partition_equal': 'control',
  'A48o_equal': 'explanatory filtering control: equal on all 48 channels with the same frozen source signs (section 8, last paragraph)',
  'ct7': 'reference (developed on the source data; not an unbiased transfer estimate)'},
 'recipe': {
  'channels_48': names, 'channels_28': kept, 'bank11': B11N, 'source_signs': {c: int(sign[c]) for c in names},
  'families_15': S3, 'family_feature': 'mean of the sign-oriented member channels, then within-answer z-score (constant -> 0)',
  'channel_normalization': 'each channel: Top10 token-mean step readout of its token stream exactly as the source pool, NaN step values replaced by the answer column mean, then within-answer z-score (constant -> 0)',
  'tail_marks': 'tie-aware top ceil(0.2 n) steps per answer and family, centred within answer (calfix_common.tail_marks tie_aware=True)',
  'lsml_fit': 'calfix_common.lsml_fit == extgen fusion.fit_weights semantics (small_m_guard, failure recording raises, Spearman orientation with the anchor on continuous features, sum|w|=1)',
  'final_score': 'weights applied to the continuous features, then within-answer z-score',
  'deployment_fit': 'source folds 0-3 (PB+PRMB, 13,769-answer population, unlabeled)', 'calibration': 'q80 (numpy linear) of the method\'s scores on all source fold-4 steps, unlabeled; the same threshold transfers to every external cell',
  'decision': 'step valid iff score < threshold', 'empty_external_steps': 'predicted incorrect, null score (as the existing external lock)', 'fit_failure': 'raise; no silent equal fallback'},
 'deployment': methods, 'bank11_replay_of_frozen_external_bundle': check, 'tail_mark_degeneracy_source': degen,
 'parity_gate': {
  'required_before_external_scoring': True,
  'rule': 'the external extractor, run on SOURCE raw telemetry, must reproduce every one of the 48 pool channels after within-answer z-score with max |diff| <= 1e-6 on all 13,769 source answers (or, if raw source telemetry is unavailable for some cells, on a deterministic sample of >= 300 answers covering all 9 cells, declared before running). A channel that fails gives the pipeline a new identity: source validation is rerun with the replayable definition and a new lock is written before any external score.',
  'known_risks': ['15 of the 28 channels are not computed by the external pipeline today; 5 CT7 streams are computed inside the external CT7 port but not output', 'the builder of the pooled ct7 step channels (union_top10_profiles.npy) was not found: whether ct7_chosen_std_excess is a Top10 of the token z-stream and whether step 0 was replaced must be recovered', 'H1_frac_above_z is profiles_full column 10 and is not a Top10 readout', 'hist_* rolling streams need >= 16 tokens (shorter answers are constant -> 0)']},
 'external_primary_contrasts': [['F15_tailtie_lsml', 'B11_lsml'], ['F15_tailtie_lsml', 'F15_equal'], ['F15_tailtie_lsml', 'F15_cov_lsml'], ['F15_cov_lsml', 'K28_cov_lsml'], ['F15_equal', 'K28_equal'], ['K28_equal', 'A48o_equal']],
 'external_multiplicity': 'per dataset/backbone cell; paired source-question bootstrap, 100,000 draws, seed 20260924; Bonferroni over 6 contrasts x the number of cells evaluated. B11_lsml vs its two controls and vs CT7 keep the existing lock. Secondary: within-answer AUC, class recalls, category panels. No row is selected, dropped or re-tuned after external labels are seen; all rows are reported.',
 'code_hashes': {'calfix_common': sha(HERE / 'calfix_common.py'), 'lock_script': sha(Path(__file__)), **extgen_hashes()},
}
dump(STAGE / 'TRANSFER_LOCK_V1.json', lock)
print(json.dumps(check, indent=1)); print({k: (v.get('K'), round(v['q80_threshold_fold4'], 4)) for k, v in methods.items()})
print('lock sha256', sha(STAGE / 'TRANSFER_LOCK_V1.json'))
