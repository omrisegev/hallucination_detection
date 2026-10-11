set -e
export SLURM_CONF_SERVER=controller-primary
SH=/shared/cycle2_tau_averbuch_prj/omrisegev1; C=$SH/code_tf_thinkfix_153589e; cd $C
R=$SH/results
for d in prmbench_qwen3_8b_telemetry_thinkfix pb_qwen3_8b_thinkfix pb_qwen3_4b_thinkfix; do
  test ! -e $R/$d || { echo "EXISTS $d"; exit 1; }
done
SB="sbatch --parsable --chdir=$C -p power-gpu -A cycle3_tau_averbuch_prj --qos=owner_940"
chain() {  # sbatch file + args; two linear links (resume of a finished cell is a no-op)
  local f=$1; shift
  local j1=$($SB $f "$@"); local j2=$($SB --dependency=afterany:$j1 $f "$@")
  echo "$j1 -> $j2 : $f $*"
}
chain cluster/submit_prmbench_tf.sbatch --model Qwen/Qwen3-8B --think-prefix empty_block \
  --out $R/prmbench_qwen3_8b_telemetry_thinkfix
chain cluster/submit_teacher_forced.sbatch --model Qwen/Qwen3-8B --think-prefix empty_block \
  --out $R/pb_qwen3_8b_thinkfix
chain cluster/submit_teacher_forced.sbatch --model Qwen/Qwen3-4B --think-prefix empty_block \
  --out $R/pb_qwen3_4b_thinkfix
