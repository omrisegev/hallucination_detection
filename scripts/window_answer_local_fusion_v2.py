"""Versioned repair of the frozen window experiment's unmatched native controls.

Uses the original checkout read-only; writes ONLY --output under this checkout.
Requires its complete dependencies and cached inputs. No GPU/LLM inference.
python -B scripts/window_answer_local_fusion_v2.py --source-root .worktrees/lsml-ct7-levers-run --output results/window_representation_b3_v2
"""
import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10000)
    args = parser.parse_args()
    source = (ROOT / args.source_root).resolve()
    out = (ROOT / args.output).resolve()
    if not out.is_relative_to(ROOT / "results") or out.exists():
        raise ValueError("Choose a NEW result directory inside this checkout/results")
    # The source freezes ~several MB of bootstrap draws plus per-step scores.
    if shutil.disk_usage(ROOT).free < 256 * 1024**2:
        raise OSError("Need 256 MiB free for versioned scores, bootstrap draws and safety margin")
    protocol = load("runtime_protocol", ROOT / "spectral_utils/runtime_fusion_protocol.py")
    old = load("window_fusion_original", source / "scripts/experiments/window_answer_local_fusion_v1.py")
    old.SCHEMA = "window-answer-local-fusion-v2-matched-native"
    original_build, original_planned, original_run = old.build_methods, old.planned, old.run
    def build(d, scores, native):
        methods = original_build(d, scores, native)
        protocol.add_native_controls(methods, native)
        return methods
    def planned(names):
        pairs = [p for p in original_planned(names) if p[2] != "learned_minus_equal_native_rows"]
        for arm in ("iu", "shrink_iu", "lsml"):
            # Preserve the original contrast family size: Top10 native only.
            a, b = f"window_{arm}_top10_native", f"window_equal_top10_on_{arm}_native"
            if a in names and b in names:
                pairs.append((a, b, "learned_minus_equal_native_rows"))
        return pairs
    def run(d, mats, spans, views):
        scores, native, summary = original_run(d, mats, spans, views)
        np.savez_compressed(d.out / "SCORES_AND_NATIVE.npz", **scores,
                            **{f"native_{k}": v for k, v in native.items()}, offsets=d.off)
        return scores, native, summary
    old.build_methods, old.planned, old.run = build, planned, run
    original_freeze = old.L.run_freeze
    def freeze(output, code, inputs, metadata):
        return original_freeze(output, code + [Path(__file__), ROOT / "spectral_utils/runtime_fusion_protocol.py"], inputs, metadata)
    old.L.run_freeze = freeze
    config_path = source / "configs/window_representation_b3_v1.json"
    config = json.loads(config_path.read_text(encoding="utf8"))
    config["paths"]["output"] = str(out)
    config["bootstrap_draws"] = args.draws
    gate = source / "results/window_representation_b3_v1/WINDOW_PR.json"
    if not json.loads(gate.read_text(encoding="utf8")).get("gate_passed"):
        raise ValueError("Original measurement gate did not pass")
    out.mkdir(parents=True)
    shutil.copyfile(gate, out / gate.name)
    new_config = out / "CONFIG.json"
    new_config.write_text(json.dumps(config, indent=2), encoding="utf8")
    sys.argv = [str(Path(__file__)), "--config", str(new_config), "--draws", str(args.draws)]
    old.main()


if __name__ == "__main__":
    main()
