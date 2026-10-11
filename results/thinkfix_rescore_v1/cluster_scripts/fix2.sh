set -e
SH=/shared/cycle2_tau_averbuch_prj/omrisegev1; C=$SH/code_tf_thinkfix_153589e; cd $C
ls $SH/wheelhouse/tf_thinkfix_20261010 | grep narwhals
sed -i -e "s|'typing_extensions==4.16.0'|'typing_extensions==4.16.0' 'narwhals==2.27.1'|" \
       -e 's|import numpy, pyarrow, datasets, transformers, torch;|import numpy, pyarrow, datasets, transformers, torch, sklearn.metrics, scipy, spectral_utils;|' \
       cluster/install_offline_tf_thinkfix.sh
grep -n -E 'narwhals|import numpy' cluster/install_offline_tf_thinkfix.sh
export SLURM_CONF_SERVER=controller-primary
sbatch --chdir=$C -p power-gpu -A cycle3_tau_averbuch_prj --qos=owner_940 --time=01:00:00 cluster/submit_prmbench_tf.sbatch \
  --model Qwen/Qwen3-8B --n-samples 60 --think-prefix empty_block \
  --out $SH/results/prmbench_qwen3_8b_telemetry_thinkfix_pilot
