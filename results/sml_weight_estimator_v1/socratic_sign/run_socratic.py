"""Run the shipped ablation5_socratic.py with a chosen SML sign mode (see nonneg_sml.py).

    python run_socratic.py <repo> <mode> <bank dir> <out SCORES npz>

mode 'signed' is the shipped code path (must reproduce shipped scores). For the non-negative modes the final
anchor flip in lsml() is disabled: with non-negative weights on risk-oriented views there is no sign to resolve.
"""
import json, sys
from pathlib import Path

repo, mode, bank, out = Path(sys.argv[1]).resolve(), sys.argv[2], sys.argv[3], sys.argv[4]
sys.path.insert(0, str(repo)); sys.path.insert(0, str(Path(__file__).parent))
import spectral_utils.fusion_utils as FU  # noqa: E402
import nonneg_sml as NN  # noqa: E402

FU.sml_fuse_signed = NN.make(mode, FU.sml_fuse_signed)
src_path = repo / 'results/prm_boost_ablation5_v1/ablation5_socratic.py'
src = src_path.read_text()
flip = "    if np.corrcoef(s, ref)[0, 1] < 0: s = -s\n"
assert src.count(flip) == 1
if mode != 'signed':
    src = src.replace(flip, "    pass  # no anchor flip: non-negative weights on risk-oriented views\n")
sys.argv = [str(src_path), bank, out]
exec(compile(src, str(src_path), 'exec'), {'__file__': str(src_path), '__name__': '__main__'})
flags = NN.FLAGS
Path(out).with_suffix('.flags.json').write_text(json.dumps({
    'mode': mode, 'n_sml_calls': len(flags), 'n_negative_component': sum(f['negative_component'] for f in flags),
    'calls': flags}, indent=1))
