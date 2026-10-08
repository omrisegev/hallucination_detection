"""Build figures + S0_AUDIT.md from the S0 audit CSVs (no recomputation)."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/ssl_pseudolabel_residual_v1/S0'
log = json.loads((OUT / 'S0_LOG.json').read_text(encoding='utf8'))
board = pd.read_csv(OUT / 'SCOREBOARD_REPLAY.csv').set_index('method')
q = pd.read_csv(OUT / 'PLANNED_CONTRASTS_BY_QUESTION.csv')
pbc = pd.read_csv(OUT / 'PB_DECISION_CHANGES.csv')
prc = pd.read_csv(OUT / 'PRM_PAIRED_CHANGES.csv')
depth = pd.read_csv(OUT / 'PB_BY_DEPTH.csv'); rel = pd.read_csv(OUT / 'PB_BY_RELATIVE_POSITION.csv'); comp = pd.read_csv(OUT / 'PB_COMPETITION_ORACLE.csv')
parity = pd.read_csv(OUT / 'SEED_PARITY.csv')
kind = pd.read_csv(OUT / 'PRM_WITHIN_AUC_BY_KIND.csv', header=[0, 1], index_col=0)

NICE = {'ct7': 'CT7 (frozen anchor)', 'token_lsml': 'token L-SML', 'token_equal': 'token equal',
        'evidence__all__seed__equal': 'top5 seed (teacher)', 'evidence__all__plain__equal': 'top5 plain evidence',
        'evidence__all__position__equal': 'top5 position evidence', 'evidence__all__plain2__equal': 'top5 plain, iter 2',
        'evidence__all__position2__equal': 'top5 position, iter 2', 'evidence__all__randomseed__equal': 'top5 random-seed null',
        'evidence__all__randomseed_position__equal': 'top5 random-seed + position', 'evidence__all__prioronly__equal': 'top5 prior only',
        'evidence__all__ceiling__equal': 'top5 TRUE-label ceiling', 'evidence__all__plain__continuous_lsml': 'top5 plain + L-SML',
        'evidence30__all__seed__equal': 'top30 seed', 'evidence30__all__plain__equal': 'top30 plain evidence', 'evidence30__all__position__equal': 'top30 position evidence'}
KEY = list(NICE)

# ---- Figure 1: scoreboard, PB SLA vs PRMB within-AUC
fig, ax = plt.subplots(figsize=(8.5, 6))
colors = {'anchor': '#444444', 'teacher': '#1f77b4', 'evidence': '#d62728', 'control': '#999999', 'ceiling': '#2ca02c'}
def group(m):
    if m in ('ct7', 'token_lsml', 'token_equal'): return 'anchor'
    if 'seed__' in m and 'random' not in m: return 'teacher'
    if 'random' in m or 'prior' in m: return 'control'
    if 'ceiling' in m: return 'ceiling'
    return 'evidence'
for m in KEY:
    r = board.loc[m]; g = group(m)
    ax.scatter(r.within_auc, 100 * r.pb_sla_macro8, c=colors[g], s=60 if g != 'anchor' else 90, marker='D' if g == 'anchor' else 'o', zorder=3)
    ax.annotate(NICE[m], (r.within_auc, 100 * r.pb_sla_macro8), fontsize=7, xytext=(4, 3), textcoords='offset points')
ax.set_xlabel('PRMBench within-answer AUROC (higher = better ranking of error steps)')
ax.set_ylabel('ProcessBench SLA macro8, % (first-error exact hit)')
ax.set_title('S0 scoreboard replay: every arm of Step 432 (development data, 13,769 answers)')
ax.grid(alpha=.3)
for g, c in colors.items(): ax.scatter([], [], c=c, label=g)
ax.legend(loc='lower right', fontsize=8)
fig.tight_layout(); fig.savefig(OUT / 'FIG1_scoreboard.png', dpi=150); plt.close(fig)

# ---- Figure 2: PB SLA by depth bin for key methods
fig, ax = plt.subplots(figsize=(8, 4.5))
order = ['1', '2-5', '6-10', '11+']
for m in ['ct7', 'token_lsml', 'evidence__all__seed__equal', 'evidence__all__plain__equal', 'evidence__all__position__equal', 'evidence30__all__plain__equal', 'evidence__all__ceiling__equal']:
    d = depth[depth.method == m].set_index('depth_bin').reindex(order)
    ax.plot(order, 100 * d.sla, marker='o', label=NICE[m])
d = depth[depth.method == 'ct7'].set_index('depth_bin').reindex(order)
for i, b in enumerate(order):
    if not pd.isna(d.n.iloc[i]): ax.text(i, 2, f'n={int(d.n.iloc[i])}', ha='center', fontsize=8, color='#555')
ax.set_ylabel('SLA, % (exact first-error hit)'); ax.set_xlabel('number of steps in the answer'); ax.set_title('ProcessBench exact hit rate by chain depth (erroneous answers only)')
ax.grid(alpha=.3); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(OUT / 'FIG2_pb_by_depth.png', dpi=150); plt.close(fig)

# ---- Figure 3: decision changes vs teacher / CT7 (PB) as paired bars
fig, ax = plt.subplots(figsize=(9, 4.5))
sel = pbc[[(a, b) in {('evidence__all__plain__equal', 'evidence__all__seed__equal'), ('evidence__all__position__equal', 'evidence__all__seed__equal'),
                       ('evidence__all__position__equal', 'evidence__all__plain__equal'), ('evidence__all__ceiling__equal', 'evidence__all__plain__equal'),
                       ('evidence30__all__plain__equal', 'evidence30__all__seed__equal'), ('evidence__all__plain__equal', 'ct7'), ('evidence__all__position__equal', 'ct7')}
           for a, b in zip(pbc.candidate, pbc.reference)]]
x = np.arange(len(sel)); w = .38
ax.bar(x - w / 2, sel.wrong_to_correct, w, color='#2ca02c', label='rescued (reference wrong, candidate right)')
ax.bar(x + w / 2, -sel.correct_to_wrong, w, color='#d62728', label='damaged (reference right, candidate wrong)')
for i, r in enumerate(sel.itertuples()): ax.text(i, max(r.wrong_to_correct, 0) + 15, f'net {r.net_gain:+d}\nagree {100*r.agreement:.0f}%', ha='center', fontsize=7)
ax.set_xticks(x); ax.set_xticklabels([f'{NICE[a]}\nvs {NICE[b]}' for a, b in zip(sel.candidate, sel.reference)], fontsize=7)
ax.axhline(0, color='k', lw=.8); ax.set_ylabel('erroneous ProcessBench answers (of 4,442)'); ax.set_title('Where the decisions actually changed'); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / 'FIG3_pb_decision_changes.png', dpi=150); plt.close(fig)

# ---- Figure 4: planned contrasts with CI (primary questions)
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=False)
for ax, ep, scale, title in [(axes[0], 'pb_sla', 100, 'ProcessBench SLA, pp'), (axes[1], 'prm_within_auc', 1, 'PRMBench within-answer AUROC')]:
    d = q[q.endpoint == ep].reset_index(drop=True)
    lab = [f"{NICE.get(a, a)} - {NICE.get(b, b)}" for a, b in zip(d.a, d.b)]
    y = np.arange(len(d))
    ax.errorbar(scale * d.delta, y, xerr=[scale * (d.delta - d.ci_lo), scale * (d.ci_hi - d.delta)], fmt='o', color='#1f77b4', capsize=3)
    for i, r in d.iterrows():
        if r.p_holm < .05: ax.text(scale * r.ci_hi, i, '  Holm<.05', va='center', fontsize=7, color='#d62728')
    ax.axvline(0, color='k', lw=.8); ax.set_yticks(y); ax.set_yticklabels(lab, fontsize=7); ax.set_title(title); ax.grid(alpha=.3, axis='x'); ax.invert_yaxis()
fig.suptitle('Planned paired contrasts (source-group bootstrap, 10,000 draws, uncorrected 95% CI; Holm over 312 tests)', fontsize=10)
fig.tight_layout(); fig.savefig(OUT / 'FIG4_contrasts.png', dpi=150); plt.close(fig)

# ---- S0_AUDIT.md
def f(x, p=4): return '-' if pd.isna(x) else f'{x:.{p}f}'
md = ['# S0 audit — Claude Step 432 (A1), read-only replay', '',
      f"Source: `{'.worktrees/readout-quickest-detection-v1/results/step_evidence_v1'}`. Recomputed with `scripts/experiments/ssl_s0_audit.py` in {log['seconds']:.0f}s. No fit, no inference.", '',
      '## Exit condition', '']
inv = log['inventory']; cov = log['coverage']; sr = log['scoreboard_replay']
maxerr = max(v for v in sr['max_abs_error_per_metric'].values() if not (isinstance(v, float) and np.isnan(v)))
valid = (inv['stale_jobs_newer_than_summary'] == 0 and all(v['sha_ok'] for v in log['input_freeze_today'].values()) and cov['groups_crossing_folds'] == 0
         and cov['fold_matches_FOLDS_V2'] and maxerr < 1e-9 and cov['prm_two_class_answers'] == 6030)
md += [f"**{'VALIDATED' if valid else 'INCOMPLETE'}** — jobs {inv['jobs']} ({inv['outer']} outer / {inv['inner']} inner), stale jobs {inv['stale_jobs_newer_than_summary']}, "
       f"input hashes today all match: {all(v['sha_ok'] for v in log['input_freeze_today'].values())}, groups crossing folds {cov['groups_crossing_folds']}, "
       f"folds match FOLDS_V2: {cov['fold_matches_FOLDS_V2']}, max |replay - report| over {sr['methods']} methods x 7 metrics = {maxerr:.2e}.", '',
       '## Population / contract', '',
       f"answers {cov['answers']} (PB {cov['pb']}: {cov['pb_erroneous']} erroneous + {cov['pb_clean']} clean; PRMB {cov['prm']}, two-class {cov['prm_two_class_answers']}), steps {cov['steps']}. "
       f"CT7 gate opens on {cov['ct7_gate_open_pb']} PB answers and {cov['ct7_gate_open_prm']} PRMB answers. Non-finite score methods: {cov['methods_with_nonfinite_scores'] or 'none'}; "
       f"PB predictions missing: {cov['pb_predictions_missing'] or 'none'}.", '',
       '## Provenance deviations', '',
       f"- PRMB pseudo rule(s): {inv['prm_pseudo_rules']}; PB rule: seed argmax of gate-open training answers. Min pseudo-positive mass PRMB {inv['prm_min_pseudo_positive']:.0f}, PB {inv['pb_min_pseudo_positive']:.0f}.",
       f"- RUN_FREEZE `step_evidence_v1.py` = module hash; driver hash NOT frozen (basename collision). Driver today: `{log['source_freeze']['driver_vs_module']['scripts/experiments/step_evidence_v1.py'][:12]}`, module: `{log['source_freeze']['driver_vs_module']['spectral_utils/step_evidence_v1.py'][:12]}`.",
       f"- Seed parity: raw replay reproduces the stored seed exactly = {log['seed_parity']['raw_replay_reproduces_stored']}; the protocol-described step-z seed changes {log['seed_parity']['step_z_changes_pb_argmax']} PB argmaxes. Two method IDs are kept (see SEED_PARITY.csv).", '',
       '## Seed parity', '', parity.to_markdown(index=False, floatfmt='.4f'), '',
       '## Scoreboard replay (key arms)', '',
       board.loc[KEY, ['pb_sla_macro8', 'pb_f1_ct7_gate', 'within_auc', 'prmscore_q80', 'prmscore_inner', 'pb_covered', 'prm_covered']].rename(index=NICE).to_markdown(floatfmt='.4f'), '',
       f"Max replay error per metric: {json.dumps({k: (None if (isinstance(v, float) and np.isnan(v)) else float(f'{v:.2e}')) for k, v in sr['max_abs_error_per_metric'].items()})}", '',
       '## Planned contrasts by question', '',
       q.assign(a=q.a.map(lambda s: NICE.get(s, s)), b=q.b.map(lambda s: NICE.get(s, s)))[['question', 'endpoint', 'a', 'b', 'delta', 'ci_lo', 'ci_hi', 'p_holm']].to_markdown(index=False, floatfmt='.4f'), '',
       f"Family: {log['multiplicity']['family_size']} contrasts, {log['multiplicity']['draws']} draws, unit = {log['multiplicity']['unit']} ({log['multiplicity']['source_groups']} groups). "
       f"Tail resolution {log['multiplicity']['minimum_raw_p_resolution']:.1e} so the Holm floor is {log['multiplicity']['min_p_holm']:.4f}; {log['multiplicity']['contrasts_with_p_holm_below_0.05']} contrasts sit at that floor.", '',
       '## Decision changes on ProcessBench (erroneous answers)', '',
       pbc.assign(candidate=pbc.candidate.map(NICE), reference=pbc.reference.map(NICE))[['candidate', 'reference', 'agreement', 'wrong_to_correct', 'correct_to_wrong', 'net_gain', 'moved_earlier', 'moved_later', 'cand_early_miss', 'cand_late_miss', 'long11_wrong_to_correct', 'long11_correct_to_wrong']].to_markdown(index=False, floatfmt='.3f'), '',
       '## Paired per-answer changes on PRMBench (6,030 two-class answers)', '',
       prc.assign(candidate=prc.candidate.map(NICE), reference=prc.reference.map(NICE))[['candidate', 'reference', 'mean_delta_auc', 'answers_improved', 'answers_worsened', 'answers_unchanged', 'pairs_corrected', 'pairs_destroyed', 'net_pairs']].to_markdown(index=False, floatfmt='.4f'), '',
       '## PRMBench within-AUC by answer kind', '', kind.xs('mean', axis=1, level=1)[KEY].rename(columns=NICE).T.to_markdown(floatfmt='.4f'), '',
       '## ProcessBench by depth', '', depth[depth.method.isin(KEY)].assign(method=depth.method.map(NICE)).to_markdown(index=False, floatfmt='.3f'), '',
       '## ProcessBench by relative error position', '', rel[rel.method.isin(KEY)].assign(method=rel.method.map(NICE)).to_markdown(index=False, floatfmt='.3f'), '',
       '## Competition oracle (S>=4; keep truth + 3 uniformly chosen steps; label-using diagnostic)', '', comp.assign(method=comp.method.map(NICE)).to_markdown(index=False, floatfmt='.3f'), '']
(OUT / 'S0_AUDIT.md').write_text('\n'.join(md), encoding='utf8')
print('VALIDATED' if valid else 'INCOMPLETE', maxerr)
