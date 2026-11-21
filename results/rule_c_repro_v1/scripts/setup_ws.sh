# How $WS was built on the PC (2026-11-21). Bundle originals stay untouched in rule_c_ws/originals/.
O=C:/Users/omris/TAU/rule_c_ws/originals/review_bundle; WS=C:/Users/omris/TAU/rule_c_ws/run; WT=C:/Users/omris/TAU/wt_rule_c_repro
# 1. copy ssl_ws and repo_results/ssl_step_threshold_v1 (macOS ._ files and .DS_Store left out)
# 2. bundle outputs moved aside inside the working copy: agentS2/out -> out_bundle, repo_results/.../out -> out_bundle (fresh empty out/)
# 3. hd = spectral_utils/ + results/ssl_one_experiment_v1/ of commit 4acda6f6 (sparse worktree)
# 4. W/steps, W/inputs = NTFS junctions to C:/Users/omris/TAU/hd_collection_data/ssl_one_experiment_v1/{steps,inputs} (C: had < 1 GB free)
#    W/processbench_gold.json = copy of hd_collection_data/.../analysis_v1/code/results/prm_boost_processbench_v1/local/processbench_gold.json (byte-equal to the repo file)
# 5. Mac path -> $WS in the working copy only:
P=/private/tmp/claude-501/-Users-osegev-Desktop-hallucination-detection/d15d4489-a581-4d83-8971-0b5a5692d733/scratchpad/ssl_ws
grep -rl "$P" "$WS" --include=*.py | xargs sed -i "s#$P#$WS#g"
# 6. run from $WS with HD_ROOT=$WS/hd: agentS2/parity.py, agentS2/pseudo.py, agentS2/eval_pc.py (eval.py with the bitwise sC check relaxed),
#    repo_results/ssl_step_threshold_v1/{phase1_labelfree,phase2_evaluate,prmbench_phase1_labelfree,prmbench_phase2_evaluate,size_curve}.py,
#    then results/rule_c_repro_v1/scripts/dev_table.py $WS
