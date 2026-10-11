set -e
export SLURM_CONF_SERVER=controller-primary
SH=/shared/cycle2_tau_averbuch_prj/omrisegev1; C=$SH/code_tf_thinkfix_153589e; cd $C
grep -o '"prompt_suffix"[^,]*\|no_think[^,]*' $SH/results/evdrop_gsm8k_qwen3_8b/manifest.json | head -3
for m in 8b 4b; do
  M=$([ $m = 8b ] && echo Qwen/Qwen3-8B || echo Qwen/Qwen3-4B)
  sbatch --chdir=$C -p power-gpu -A cycle3_tau_averbuch_prj --qos=owner_940 --time=01:00:00 cluster/submit_teacher_forced.sbatch \
    --model $M --validate --validate-pkl $SH/results/evdrop_gsm8k_qwen3_$m/raw_gsm8k_T0.0.pkl \
    --validate-dataset gsm8k --validate-n 20 --think-prefix empty_block
done
sbatch --chdir=$C -p power-gpu -A cycle3_tau_averbuch_prj --qos=owner_940 --time=01:00:00 cluster/submit_prmbench_tf.sbatch \
  --model Qwen/Qwen3-8B --n-samples 60 --think-prefix empty_block \
  --out $SH/results/prmbench_qwen3_8b_telemetry_thinkfix_pilot
