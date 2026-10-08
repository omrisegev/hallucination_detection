from pathlib import Path
import ast,json,subprocess,sys,datetime
ROOT=Path(r'C:\Users\omris\TAU\hallucination_detection')
WORK=ROOT/'results/monday_fusion_analysis_v1'; CODE=WORK/'code'
def dump(p,x):
    p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
notes={'schema':'monday-implementation-clarifications-v1','frozen_before':'experiment2, position/decision/LR, combined and comparator arms',
 'content_selection':'Keep committed matched control: equal between-group weights on the same group scores and within weights. Also report DS-filtered simple feature average; optional Omri clarification has no reply yet.',
 'secondary_intervals':'50,000 shared source-question bootstrap draws, seed 20261003; descriptive 95% only. Original/reference and selected content variant, all recipes: position versus CONTENT under q80, decision rules versus q80 for CONTENT, HEM versus MEAN under each between method, and matched non-AUC endpoints. Combined recipe and component controls receive the same intervals.',
 'ALL_EQ_POS':'Append position to the unfiltered bank and average. The broad ungrouped-DS phrase in PROTOCOL applies to DS_FILTER_EQ; ALL_EQ remains its unfiltered matched control.',
 'raw_digit_R2':'Use saved pre-answer-standardization digit values as raw channels in the per-cell global standardization. No new digit extraction.',
 'position_choice':'Registered primary endpoint is content-only. Position variants are diagnostic; the combined candidate retains CONTENT. No label-based position selection is introduced.',
 'combined_selection':'Keep the within/between recipe from LEVEL selection; choose its full-coverage input variant by equal-weight eight-bank matched advantage. Test BASE, LEVEL_ONLY, INPUT_ONLY, COMBINED on all cells. Highest full-coverage matched advantage wins, deterministic tie order BASE, LEVEL_ONLY, INPUT_ONLY, COMBINED.',
 'runtime':'Content runtime_s is elapsed shared cell fit/evaluation time, not isolated recipe runtime; do not interpret it as method speed.',
 'step_retention':'Core chunks keep original-bank main scores and experiment-2 MEAN_DS/MEAN_MOM input comparisons. Experiment STEP_SCORES.npz holds the selected development content recipe and matched EQ for all banks; final combined scores are separately locked only after Omri chooses the decision.',
 'undefined_metrics':'Correct-control-only PRMB subsets have undefined PRMScore; PB F1 requires both clean and erroneous answers. Keep unknown entries blank, not zero.',
 'disk':'Notify Omri: projected next stages 250-400 MB, compress intermediate metric tables and stop before free space drops below 100 MB. No frozen deletion, overwrite or worktree move.',
 'review':'Every scientific result remains CLAUDE_PENDING; no independent review is fabricated.'}
for exp in ['level_reduction_v1','estimator_input_v1']:
    p=ROOT/'results'/exp/'run_20261003'/'IMPLEMENTATION_NOTES_V1.json'
    if p.exists() and json.loads(p.read_text(encoding='utf8'))!=notes: raise RuntimeError('Registered notes changed')
    if not p.exists(): dump(p,notes)
combined=ROOT/'results/monday_combined_recipe_v1/run_20261003'
protocol=json.loads((ROOT/'results/level_reduction_v1/run_20261003/PROTOCOL.json').read_text(encoding='utf8'))
protocol.update(experiment='monday_combined_recipe_v1',parent_protocol='results/level_reduction_v1/run_20261003/PROTOCOL.json',implementation_notes=notes)
if (combined/'PROTOCOL.json').exists() and json.loads((combined/'PROTOCOL.json').read_text(encoding='utf8'))!=protocol: raise RuntimeError('Combined protocol changed')
if not (combined/'PROTOCOL.json').exists(): dump(combined/'PROTOCOL.json',protocol)
files=list((CODE/'spectral_utils').glob('monday_fusion*.py'))+[CODE/'scripts/experiments/monday_fusion_run.py',CODE/'tests/test_monday_fusion.py']
for p in files: ast.parse(p.read_text(encoding='utf8'))
test=subprocess.run([sys.executable,'-B',str(CODE/'tests/test_monday_fusion.py')],capture_output=True,text=True)
dump(WORK/'TEST_RESULTS.json',{'status':'PASS' if test.returncode==0 else 'FAIL','syntax_files':len(files),'output':test.stdout+test.stderr})
if test.returncode: raise RuntimeError(test.stdout+test.stderr)
progress=subprocess.check_output(['git','-C',str(ROOT),'show','54a2c2a8f:PROGRESS.md'])
prefix='''## Monday fusion implementation and active full-population LEVEL run (Codex) - 2026-10-03

User-approved branch `codex/monday-fusion-analysis-20261003`, based exactly on
`54a2c2a8f463c1c48e8f943405cb1bb7f657c85b`. Isolated source export at
`results/monday_fusion_analysis_v1/code`; no full checkout, no moved worktrees,
main checkout/index preserved. Frozen protocols committed before scores.

Input/labels/source-fold audit PASS for 13,769 records, 145,597 steps. Step 457
replay max 8.881784197001252e-16 (8 comparisons); Step 462 max
2.220446049250313e-16 (72 comparisons), tolerance 1e-9. All 72 zero-removal
bank/cell matrices and marks exact; new BASE/MEAN_DS score replay passed.
Raw pre-normalization pool replay: 7,571,044 values exact; entropy identities
checked for all 13,769 answers. 15 focused contract tests pass.

LEVEL full population is RUNNING from code commit f37f0341, checkpointed by
cell/bank/variant. It has not yet produced a complete scientific conclusion.
Later runners implement channel/group votes, exact cell quotas, position/nulls,
three decision rules, nested OOF LR, paired source intervals, combined controls
and post-freeze competitor forensics. These later data stages are not yet run.
`continue --wait-for-level` enforces their order and stops for Omri's decision.
Read live `results/monday_fusion_analysis_v1/WORKFLOW_STATUS.json` and experiment
`RUN_STATUS.json`; see `docs/experiments/MONDAY_FUSION_ANALYSIS_20261003.md`.

All quality findings are pending Claude's independent review; nothing is deck
approved. Final decision choice is Omri's, with no automatic external evaluation.

'''
(CODE/'PROGRESS.md').write_bytes(prefix.encode()+progress)
mapping={p.relative_to(CODE).as_posix():str(p) for p in files}
for p in [CODE/'PROGRESS.md',CODE/'LESSONS.md',CODE/'docs/experiments/MONDAY_FUSION_ANALYSIS_20261003.md']:
    mapping[p.relative_to(CODE).as_posix()]=str(p)
for p in [WORK/'INPUT_AUDIT.json',WORK/'REPLAY_GATE.json',WORK/'RAW_BANK_REPLAY.json',WORK/'RAW_INPUT_MANIFEST_V3.json',WORK/'TEST_RESULTS.json',combined/'PROTOCOL.json']:
    mapping[p.relative_to(ROOT).as_posix()]=str(p)
for exp in ['level_reduction_v1','estimator_input_v1']:
    p=ROOT/'results'/exp/'run_20261003'/'IMPLEMENTATION_NOTES_V1.json'; mapping[p.relative_to(ROOT).as_posix()]=str(p)
dump(WORK/'CODE_COMMIT_PATHS.json',mapping)
print(json.dumps({'syntax_files':len(files),'tests':'PASS','commit_paths':len(mapping),'combined_protocol':'FROZEN_PENDING_COMMIT'}))
