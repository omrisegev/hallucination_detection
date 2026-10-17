"""Load decision_rule_run.py as a module and extract its nested helpers (answer_z, top_by_score, build_rules)
so they can be exercised on synthetic data with the SAME source text."""
import ast, sys, importlib.util, textwrap
from pathlib import Path
import numpy as np

RUN = Path(r'C:/Users/omris/TAU/hallucination_detection/.worktrees/decision-rule-v1/scripts/experiments/decision_rule_run.py')
spec = importlib.util.spec_from_file_location('drr', RUN); drr = importlib.util.module_from_spec(spec); spec.loader.exec_module(drr)

SRC = RUN.read_text(encoding='utf8'); TREE = ast.parse(SRC)
MAIN = [n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == 'main'][0]
NESTED = {n.name: ast.get_source_segment(SRC, n) for n in MAIN.body if isinstance(n, ast.FunctionDef)}


def make_env(off, fold, prm):
    """namespace mirroring main()'s closure variables for the rule builder."""
    n = len(off) - 1; ns = np.diff(off); S_ = int(off[-1]); aid = np.repeat(np.arange(n), ns)
    env = dict(np=np, off=off, n=n, ns=ns, S_=S_, aid=aid, step_fold=fold[aid], prm_step=prm[aid], RULES=drr.RULES,
               zt=drr.zt, ds_fit=drr.ds_fit, ds_posterior=drr.ds_posterior, prm_parts=drr.prm_parts)
    for name in ('answer_z', 'top_by_score', 'build_rules'):
        exec(textwrap.dedent(NESTED[name]), env)
    return env


def synth(n=1500, seed=0, d=11):
    """answers of 3-20 steps; latent error steps raise all channels (so DS has structure); PB/PRM mix; 5 folds."""
    rng = np.random.default_rng(seed)
    ns = rng.integers(3, 21, n); off = np.r_[0, np.cumsum(ns)]; S = int(off[-1])
    fold = rng.integers(0, 5, n); prm = rng.random(n) < 0.5
    y = rng.random(S) < 0.15
    ans_shift = np.repeat(rng.normal(0, .5, n), ns)[:, None]
    X = rng.normal(0, 1, (S, d)) + 1.0 * y[:, None] + ans_shift
    Sc = X.mean(1) + rng.normal(0, .3, S)
    return off, fold, prm, X, Sc, y
