"""Run the shipped ablation5.py (PRMBench) with a chosen SML sign mode (see nonneg_sml.py). For AIRCC, in the
mirror M (cwd = M), the same way run_ablation5.sh calls ablation5.py:

    python run_prmbench.py <mirror M> <mode> <tag> <FEATURES.npz> <telemetry.pkl> <out dir>

mode 'signed' must reproduce ablation5_v1 exactly. Non-negative modes disable the final anchor flip in lsml().
"""
import json, sys
from pathlib import Path

M, mode = Path(sys.argv[1]).resolve(), sys.argv[2]
sys.path.insert(0, str(M)); sys.path.insert(0, str(Path(__file__).parent))
import spectral_utils.fusion_utils as FU  # noqa: E402
import nonneg_sml as NN  # noqa: E402

FU.sml_fuse_signed = NN.make(mode, FU.sml_fuse_signed)
src_path = M / 'ablation5.py'
src = src_path.read_text()
flip = "        if np.corrcoef(s, ref)[0, 1] < 0: s = -s\n"
assert src.count(flip) == 1
if mode != 'signed':
    src = src.replace(flip, "        pass  # no anchor flip: non-negative weights on risk-oriented views\n")
sys.argv = [str(src_path)] + sys.argv[3:]
g = {'__file__': str(src_path), '__name__': '__main__'}
exec(compile(src, str(src_path), 'exec'), g)
out = Path(sys.argv[4])
(out / f'NONNEG_FLAGS_{sys.argv[1]}_{mode}.json').write_text(json.dumps({
    'mode': mode, 'n_sml_calls': len(NN.FLAGS), 'n_negative_component': sum(f['negative_component'] for f in NN.FLAGS),
    'calls': NN.FLAGS}, indent=1))
