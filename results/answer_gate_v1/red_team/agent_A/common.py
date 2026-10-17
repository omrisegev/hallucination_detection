"""Red-team A: independent loaders. Does NOT import scripts/experiments/answer_gate_run.py."""
import importlib.util, pickle
import numpy as np, pandas as pd

ROOT = "C:/Users/omris/TAU/hallucination_detection"
W = ROOT + "/.worktrees/decision-rule-v1"
SE = ROOT + "/.worktrees/readout-quickest-detection-v1/results/step_evidence_v1"
DEC = W + "/results/answer_gate_v1/run_20260930/DECISIONS.npz"
FEAT = W + "/results/answer_gate_v1/ANSWER_FEATURES.npz"
META = ROOT + "/dataset_cache/four_localization/prmbench_qwen25math7b_full/prmbench_prm.pkl"
PRMCELL = "prmbench_qwen3_8b"


def load_official_scorer():
    spec = importlib.util.spec_from_file_location("prmbench_official", W + "/spectral_utils/prmbench.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_all():
    df = pd.read_csv(SE + "/OOF_ANSWERS.csv", encoding="utf-8-sig",
                     usecols=["uid", "id", "source_group", "fold", "cell", "target"])
    s = np.load(SE + "/OOF_STEP_SCORES.npz")
    off = s["offsets"].astype(np.int64)
    lab = s["labels"].astype(np.int8)
    d = np.load(DEC, allow_pickle=True)
    assert np.array_equal(d["offsets"], off), "DECISIONS offsets != OOF_STEP_SCORES offsets"
    assert len(df) == len(off) - 1
    meta = pickle.load(open(META, "rb"))
    bymeta = {v["idx"]: v for v in meta.values()}
    return df, off, lab, d, bymeta
