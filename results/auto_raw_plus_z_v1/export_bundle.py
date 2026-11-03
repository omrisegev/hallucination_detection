"""Write one PRM's self-contained CAT34 input file (INPUTS_<tag>.npz) with the same code path as the run.

Contents (one row per PRMBench step, 94k-95k steps; answers in the run's order):
  row_ids [answers], offsets [answers+1], step_index [steps] (0-based within answer), answer_of_step [steps]
  step_error [steps]    the step labels used for within/pooled AUC (OOF_STEP_SCORES 'labels')
  classification [answers], source_group [answers], error_steps_flat + error_steps_offsets (1-based, PRMBench)
  reward [steps]        PRM step reward r (validity probability); prm_native_valid [steps] = the PRM's own labels
  raw17 [steps,17]      untransformed: column 0 = log((1-r)/r), columns 1..16 = FEATURES.npz values of N16
  digit_active [steps,3]
  X34 [steps,34], names34   the exact L-SML inputs: 0..16 answer-z (Z17), 17..33 global rank->normal (raw)

    python3 export_bundle.py <tag> <FEATURES.npz> <telemetry.pkl> <out_dir>
"""
import sys
from pathlib import Path
import numpy as np
import auto_raw_plus_z as AR


def main():
    tag, feat, tele, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    out.mkdir(parents=True, exist_ok=True)
    _, I = AR.build_inputs(feat, tele)
    o, meta = I['o'], I['meta']
    names17 = ['PRM'] + AR.N16
    es = [np.asarray(m['error_steps'], int) for m in meta]
    np.savez_compressed(
        out / f'INPUTS_{tag}.npz', row_ids=I['ids'], offsets=o,
        step_index=np.concatenate([np.arange(b - a) for a, b in zip(o[:-1], o[1:])]),
        answer_of_step=np.repeat(np.arange(len(o) - 1), np.diff(o)),
        step_error=I['labels'], classification=np.array([m['classification'] for m in meta]),
        source_group=np.array([I['A'][i]['source_group'] for i in I['ans']]),
        error_steps_flat=np.concatenate(es) if len(es) else np.zeros(0, int),
        error_steps_offsets=np.r_[0, np.cumsum([len(e) for e in es])],
        reward=I['rewards'], prm_native_valid=~I['native'],
        raw17=np.column_stack([I['prm'], I['raw']]), raw17_names=np.array(names17),
        digit_active=I['act'], digit_names=np.array(AR.DIG),
        X34=I['V34'], names34=np.array(['z:' + c for c in names17] + ['raw:' + c for c in names17]))
    print(tag, 'wrote', out / f'INPUTS_{tag}.npz', I['V34'].shape, flush=True)


if __name__ == '__main__':
    main()
