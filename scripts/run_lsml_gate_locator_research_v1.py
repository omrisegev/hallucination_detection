#!/usr/bin/env python3
"""Run the development-only Continuous/Joint L-SML gate/locator study."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_utils.fusion_signal_registry import READOUT_NAMES, readout_steps
from spectral_utils.historical_fusion_evaluation import auc, pb_metrics
from spectral_utils.lsml_gate_locator_research import (
    FusionRecipe,
    answer_standardize,
    cell_midranks,
    effective_rank,
    fit_fusion_weights,
    weight_diagnostics,
)


SCHEMA = "lsml-gate-locator-research-v1"
ATLAS = ROOT / "results/fusion_independence_atlas_v1"
OUTPUT = ROOT / "results/lsml_gate_locator_research_v1"


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_ready(value), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)): return bool(value)
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def answer_rows(offsets: np.ndarray, indexes: Sequence[int]) -> np.ndarray:
    return np.concatenate([
        np.arange(int(offsets[i]), int(offsets[i + 1]), dtype=np.int64) for i in indexes
    ]) if len(indexes) else np.empty(0, dtype=np.int64)


def prepare() -> Path:
    path = OUTPUT / "prepared/PREPARED.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as cached:
            if {"digit025_scores", "current_gate", "q15_atlas"}.issubset(cached.files):
                return path
    OUTPUT.joinpath("prepared").mkdir(parents=True, exist_ok=True)
    with np.load(ATLAS / "dependence/CONSOLIDATED.npz", allow_pickle=False) as z:
        names = z["names"].astype(str)
        points = z["points"].astype(str)
        consolidated_steps = z["step_scores"].astype(np.float32)
        gate_names = names[points == "answer_gate"]
        gate_values = z["answer_scores"][:, points == "answer_gate"].astype(np.float32)
        eligible_step_names = names[points == "step_post_readout"]
        eligible_steps = consolidated_steps[:, points == "step_post_readout"]
    with np.load(ATLAS / "baseline_replay/SCORES_FROZEN.npz", allow_pickle=False) as z:
        innovation5 = z["steps__append_innovation__H0lim"].astype(np.float64)
        tail15_raw = z["gate_raw"].astype(np.float32)
    extract = ATLAS / "extract/answers"
    top10 = READOUT_NAMES.index("top10")
    total_steps = len(consolidated_steps)
    q15 = np.empty((total_steps, 4), dtype=np.float32)
    digit = np.empty(total_steps, dtype=np.float32)
    energy = np.empty(total_steps, dtype=np.float32)
    with np.load(ATLAS / "dependence/EVALUATION.npz", allow_pickle=False) as z:
        offsets = z["offsets"].astype(np.int64)
    for i in range(len(offsets) - 1):
        with np.load(extract / f"{i:05d}.npz", allow_pickle=False) as z:
            a, b = int(offsets[i]), int(offsets[i + 1])
            q15[a:b] = np.column_stack([
                z[f"step__{name}__readouts"][:, top10]
                for name in ("q15.H0lim", "q15.VE0", "q15.VE0.75", "q15.VE1")
            ])
            digit[a:b] = z["step__digit.disagreement__readouts"][:, top10]
            energy[a:b] = z["step__direct_probability.selected_token_surprisal__readouts"][:, top10]
        if (i + 1) % 2000 == 0:
            print(f"prepare extraction {i + 1}/13769", flush=True)
    # TCN-4: the four learned source-excluded primitive residual views.
    levels = np.load(ATLAS / "bundle/data/primitive_levels.npy", mmap_mode="r")
    tcn_bg = np.load(ATLAS / "predictors/learned_oof/backgrounds.npy", mmap_mode="r")
    tcn_active_raw = np.load(ATLAS / "predictors/learned_oof/active.npy", mmap_mode="r")
    spans = np.load(ATLAS / "bundle/data/step_spans.npy", mmap_mode="r")
    tcn4 = np.empty((len(q15), 4), dtype=np.float32)
    for j in range(4):
        active = np.asarray(tcn_active_raw if tcn_active_raw.ndim == 1 else tcn_active_raw[:, 1], bool)
        score, available = readout_steps(
            np.asarray(levels[:, j], np.float64) - np.asarray(tcn_bg[:, 1, j], np.float64),
            spans, "top10", active_mask=active,
        )
        if not available.all():
            raise RuntimeError(f"TCN primitive {j} has inactive steps")
        tcn4[:, j] = score
        print(f"prepare TCN {j + 1}/4", flush=True)
    base_names = np.asarray([
        "bank6.q15.H0lim", "bank6.q15.VE0", "bank6.q15.VE0.75", "bank6.q15.VE1",
        "bank6.digit.disagreement.top10", "bank6.energy.top10",
        "tcn.H0lim.residual.top10", "tcn.VE0.residual.top10",
        "tcn.VE0.75.residual.top10", "tcn.VE1.residual.top10",
    ])
    base_steps = np.column_stack([q15, digit, energy, tcn4]).astype(np.float32)
    from spectral_utils.temporal_context_models import residual_step_score
    digit025 = np.empty(total_steps, dtype=np.float64)
    for i in range(len(offsets) - 1):
        sl = slice(int(offsets[i]), int(offsets[i + 1]))
        digit025[sl] = residual_step_score(innovation5[sl], digit[sl], .25)
    with np.load(ATLAS / "dependence/EVALUATION.npz", allow_pickle=False) as evaluation:
        current_gate = evaluation["incumbent_gate_open"].astype(bool)
    np.savez_compressed(
        path, base_names=base_names, base_steps=base_steps,
        eligible_step_names=eligible_step_names, eligible_steps=eligible_steps,
        gate_names=gate_names, gate_values=gate_values, tail15_raw=tail15_raw,
        q15_atlas=q15, digit025_scores=digit025, current_gate=current_gate,
        digit_rate=gate_values[:, list(gate_names).index("answer_gate::digit_rate")],
    )
    dump(OUTPUT / "prepared/MANIFEST.json", {
        "schema": SCHEMA + "/prepared-v1", "labels_used": False,
        "steps": len(q15), "answers": len(gate_values),
        "base_step_members": base_names.tolist(),
        "eligible_step_members": eligible_step_names.tolist(),
        "gate_members": gate_names.tolist(),
    })
    return path


def load_inputs() -> dict[str, Any]:
    prepared = prepare()
    with np.load(prepared, allow_pickle=False) as z:
        data = {k: z[k].copy() for k in z.files}
    with np.load(ATLAS / "dependence/EVALUATION.npz", allow_pickle=False) as z:
        data.update({k: z[k].copy() for k in z.files})
    with np.load(ATLAS / "baseline_replay/SCORES_FROZEN.npz", allow_pickle=False) as z:
        data["innovation5_scores"] = z["steps__append_innovation__H0lim"].copy()
    data["tail_gate"] = np.zeros(len(data["target"]), dtype=bool)
    return data


def merge_step_bank(data: Mapping[str, Any]) -> tuple[np.ndarray, list[str]]:
    names: list[str] = []
    cols: list[np.ndarray] = []
    for name, col in zip(data["base_names"].astype(str), np.asarray(data["base_steps"]).T):
        names.append(str(name)); cols.append(col)
    for name, col in zip(data["eligible_step_names"].astype(str), np.asarray(data["eligible_steps"]).T):
        if str(name) not in names:
            names.append(str(name)); cols.append(col)
    return np.column_stack(cols), names


def locator_recipes(names: Sequence[str]) -> list[FusionRecipe]:
    ix = {name: i for i, name in enumerate(names)}
    def recipe(name: str, members: Sequence[str], mode="continuous", groups=None, anchor=0):
        return FusionRecipe(name, tuple(members), mode, None if groups is None else tuple(groups), anchor)
    bank6 = [
        "bank6.q15.H0lim", "bank6.q15.VE0", "bank6.q15.VE0.75", "bank6.q15.VE1",
        "bank6.digit.disagreement.top10", "bank6.energy.top10",
    ]
    diverse8 = [
        "step::digit.disagreement::top2", "step::digit.token_clock_innovation::top1",
        "step::q15.VE0.75.prefix_mean_innovation::top10", "step::renyi_escort.a0.25::top10",
        "step::direct_probability.rank_1_risk::top10", "step::direct_probability.rank_9_risk::top8",
        "step::step395.logtail15::top10", "step::step395.mass_above::top10",
    ]
    plus10 = bank6 + [
        "step::q15.VE0.75.prefix_mean_innovation::top10",
        "step::digit.token_clock_innovation::top1",
        "step::step395.logtail15::top10", "step::step395.mass_above::top10",
    ]
    joint14 = [
        "step::digit.disagreement::top1", "step::digit.disagreement::top2",
        "step::digit.token_clock_innovation::top1",
        "step::q15.VE0.75.prefix_mean_innovation::top10",
        "step::q15.H0lim.prefix_mean_innovation::top10", "step::renyi_escort.a0.25::top10",
        "step::renyi_escort.a8::top10",
        "step::direct_probability.rank_1_risk::top10", "step::direct_probability.rank_3_risk::top10",
        "step::direct_probability.rank_9_risk::top10", "step::direct_probability.rank_10_risk::top8",
        "step::step395.logtail15::top10", "step::step395.logtail50::top10",
        "step::step395.mass_above::top10",
    ]
    tcn1 = "derived.tcn4_equal"
    recipes = [
        recipe("L06_bank6_continuous", bank6, anchor=4),
        recipe("L08_atlas_diverse_continuous", diverse8, anchor=0),
        recipe("L10_incumbent_plus_continuous", plus10, anchor=4),
        recipe("L14_valid_joint", joint14, mode="joint", groups=[0]*3+[1]*4+[2]*4+[3]*3, anchor=0),
        recipe("L14_valid_fixed_groups_continuous", joint14, groups=[0]*3+[1]*4+[2]*4+[3]*3, anchor=0),
        recipe("L10_bank6_plus_tcn4_continuous", bank6 + [
            "tcn.H0lim.residual.top10", "tcn.VE0.residual.top10",
            "tcn.VE0.75.residual.top10", "tcn.VE1.residual.top10"], anchor=4),
        recipe("L07_bank6_plus_tcn1_continuous", bank6 + [tcn1], anchor=4),
    ]
    eligible = [name for name in names if name.startswith("step::")]
    recipes.append(recipe("LALL24_continuous", eligible, anchor=eligible.index("step::digit.disagreement::top2")))
    for row in recipes:
        missing = sorted(set(row.members) - (set(ix) | {tcn1}))
        if missing: raise KeyError(f"{row.name} missing {missing}")
    return recipes


def gate_recipes(names: Sequence[str]) -> list[FusionRecipe]:
    all_names = list(names)
    g6 = [
        "answer_gate::digit_rate", "answer_gate::raw_neglogp1::token_top10",
        "answer_gate::q15_VE0.75::token_q90", "answer_gate::q15_raw4_mean::rolling8_max",
        "answer_gate::tail15.answer_prominence", "answer_gate::entropy_native::mean_step_top10",
    ]
    g10 = [
        "answer_gate::digit_rate", "answer_gate::raw_neglogp1::token_top10",
        "answer_gate::raw_neglogp1::mean_step_top10", "answer_gate::q15_VE0.75::token_q90",
        "answer_gate::q15_VE0.75::rolling8_max", "answer_gate::q15_raw4_mean::rolling8_max",
        "answer_gate::entropy_native::mean_step_top10", "answer_gate::tail15.answer_prominence",
        "answer_gate::tail15_mass::mean_step_top10", "answer_gate::tail50_mass::token_q90",
    ]
    valid12 = [
        "answer_gate::raw_neglogp1::token_top10", "answer_gate::raw_neglogp1::mean_step_top10",
        "answer_gate::raw_neglogp1::token_q95",
        "answer_gate::q15_H0lim::token_q75", "answer_gate::q15_H0lim::token_q95",
        "answer_gate::q15_raw4_mean::token_top10",
        "answer_gate::q15_VE0.75::token_q90", "answer_gate::q15_VE0.75::rolling8_max",
        "answer_gate::q15_VE0::token_q90",
        "answer_gate::tail15.answer_prominence", "answer_gate::tail15_mass::mean_step_top10",
        "answer_gate::tail50_mass::token_q90",
    ]
    groups = [0]*3+[1]*3+[2]*3+[3]*3
    digit = "answer_gate::digit_rate"
    return [
        FusionRecipe("G06_continuous", tuple(g6), "continuous", anchor=0),
        FusionRecipe("G10_continuous", tuple(g10), "continuous", anchor=0),
        FusionRecipe("G12_valid_joint_no_digit", tuple(valid12), "joint", tuple(groups), 0),
        FusionRecipe(
            "G13_joint_virtuals_plus_digit", tuple(valid12 + [digit]),
            "joint_then_continuous", tuple(groups), 0, tuple(range(12)), 12,
        ),
        FusionRecipe("GALL22_continuous", tuple(all_names), "continuous", anchor=all_names.index(digit)),
    ]


def recipe_matrix(recipe: FusionRecipe, matrix: np.ndarray, names: Sequence[str], *, tcn1=None) -> np.ndarray:
    index = {name: i for i, name in enumerate(names)}
    columns = []
    for member in recipe.members:
        if member == "derived.tcn4_equal":
            if tcn1 is None: raise ValueError("TCN-1 is unavailable")
            columns.append(tcn1)
        else:
            columns.append(matrix[:, index[member]])
    return np.column_stack(columns)


def score_locator(step_scores: np.ndarray, gate: np.ndarray, data: Mapping[str, Any], indexes=None) -> dict[str, Any]:
    offsets = data["offsets"]; target = data["target"]; labels = data["labels"]
    cells = data["cells"].astype(str)
    answers = np.arange(len(target)) if indexes is None else np.asarray(indexes, dtype=np.int64)
    peaks = np.empty(len(answers), dtype=np.int32)
    within = np.full(len(answers), np.nan)
    for pos, i in enumerate(answers):
        sl = slice(int(offsets[i]), int(offsets[i + 1])); local = step_scores[sl]
        peaks[pos] = int(np.argmax(local))
        if cells[i].startswith("prmbench_"):
            valid = labels[sl] >= 0; y = labels[sl][valid] == 1
            if y.any() and (~y).any(): within[pos] = auc(y, local[valid])
    local_gate = np.asarray(gate, bool)[answers]
    prediction = np.where(local_gate, peaks, -1)
    metric = pb_metrics(target[answers], prediction, np.ones(len(answers), bool), cells[answers])
    return {
        "pb": metric["macros"]["all"], "within": float(np.nanmean(within)),
        "within_n": int(np.isfinite(within).sum()), "pb_cells": metric["cells"],
        "answer_indexes": answers, "peaks": peaks, "prediction": prediction, "within_values": within,
    }


def scalar_metrics(row: Mapping[str, Any]) -> dict[str, Any]:
    return {k: json_ready(row[k]) for k in ("pb", "within", "within_n", "pb_cells")}


def fit_predict_locator(recipe, x, offsets, folds, train_answers, test_answers, seed):
    train_rows = answer_rows(offsets, train_answers); test_rows = answer_rows(offsets, test_answers)
    weight, meta = fit_fusion_weights(x[train_rows], recipe, seed=seed)
    return test_rows, x[test_rows] @ weight, weight, meta


def evaluate_locator_recipes(data: Mapping[str, Any], x: np.ndarray, names: list[str], recipes):
    offsets = data["offsets"]; folds = data["folds"]
    current_gate = data["current_gate"]
    candidate_rows, oof_scores, fold_fits = [], {}, {}
    for recipe in recipes:
        matrix = recipe_matrix(recipe, x, names, tcn1=x[:, [names.index(n) for n in [
            "tcn.H0lim.residual.top10", "tcn.VE0.residual.top10",
            "tcn.VE0.75.residual.top10", "tcn.VE1.residual.top10"]]].mean(axis=1))
        out = np.empty(len(x)); fits = []
        for outer in range(5):
            train = np.flatnonzero(folds != outer); test = np.flatnonzero(folds == outer)
            rows, pred, weight, meta = fit_predict_locator(recipe, matrix, offsets, folds, train, test, 396150 + outer)
            out[rows] = pred
            fits.append({"outer": outer, "weights": weight, "meta": meta, **weight_diagnostics(weight)})
        metric = score_locator(out, current_gate, data)
        oof_scores[recipe.name] = out; fold_fits[recipe.name] = fits
        candidate_rows.append({"name": recipe.name, "members": recipe.members, "mode": recipe.mode,
                               **scalar_metrics(metric), "effective_rank": effective_rank(matrix)})
        print("locator", recipe.name, metric["pb"], metric["within"], flush=True)
    return candidate_rows, oof_scores, fold_fits


def stitch_inner_locator_selection(data, x, names, recipes):
    offsets=data["offsets"]; folds=data["folds"]; gate=data["current_gate"]
    selected=[]; final=np.empty(len(x)); outer_details=[]
    tcn1=x[:, [names.index(n) for n in ["tcn.H0lim.residual.top10","tcn.VE0.residual.top10",
        "tcn.VE0.75.residual.top10","tcn.VE1.residual.top10"]]].mean(axis=1)
    for outer in range(5):
        inner_rows=[]
        for recipe in recipes:
            matrix=recipe_matrix(recipe,x,names,tcn1=tcn1); scores=np.full(len(x),np.nan)
            val_answers=[]
            for inner in range(5):
                if inner==outer: continue
                train=np.flatnonzero((folds!=outer)&(folds!=inner)); val=np.flatnonzero(folds==inner)
                rows,pred,_,_=fit_predict_locator(recipe,matrix,offsets,folds,train,val,396150+outer*10+inner)
                scores[rows]=pred; val_answers.extend(val.tolist())
            m=score_locator(scores,gate,data,np.asarray(sorted(val_answers)))
            inner_rows.append((recipe,m))
        admissible=[r for r in inner_rows if r[1]["within"]>=.774036]
        choice=max(admissible or inner_rows,key=lambda r:(r[1]["pb"],r[1]["within"]))[0]
        selected.append(choice.name)
        matrix=recipe_matrix(choice,x,names,tcn1=tcn1)
        train=np.flatnonzero(folds!=outer); test=np.flatnonzero(folds==outer)
        rows,pred,w,meta=fit_predict_locator(choice,matrix,offsets,folds,train,test,396150+outer)
        final[rows]=pred
        outer_details.append({"outer":outer,"selected":choice.name,"weights":w,"meta":meta,
                              "inner":{r.name:scalar_metrics(m) for r,m in inner_rows}})
    return selected, final, outer_details


def rank_gate_score(raw: np.ndarray, cells: np.ndarray, pb: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    ranked=cell_midranks(raw,cells,pb)[:,0]; opened=np.zeros(len(pb),bool);opened[pb]=ranked[pb]>=.33
    return ranked,opened


def score_gate(opened, locator_scores, data, indexes=None):
    metric=score_locator(locator_scores,opened,data,indexes)
    answers=metric["answer_indexes"];cells=data["cells"].astype(str);target=data["target"]
    pb=np.char.startswith(cells[answers],"pb_"); err=target[answers]>=0; op=np.asarray(opened)[answers]
    return {**metric,"false_open":float(np.mean(op[pb & ~err])) if np.any(pb&~err) else None,
            "false_close":float(np.mean(~op[pb & err])) if np.any(pb&err) else None,
            "open_rate":float(np.mean(op[pb])) if np.any(pb) else None}


def evaluate_gate_recipes(data, gate_x, gate_names, recipes, locator_scores):
    folds=data["folds"];cells=data["cells"].astype(str);pb=np.char.startswith(cells,"pb_")
    rows=[];oof_raw={};fits={}
    for recipe in recipes:
        matrix=recipe_matrix(recipe,gate_x,gate_names); raw=np.full(len(cells),np.nan); ff=[]
        for outer in range(5):
            train=np.flatnonzero((folds!=outer)&pb);test=np.flatnonzero((folds==outer)&pb)
            w,meta=fit_fusion_weights(matrix[train],recipe,seed=396250+outer)
            raw[test]=matrix[test]@w;ff.append({"outer":outer,"weights":w,"meta":meta,**weight_diagnostics(w)})
        _,opened=rank_gate_score(raw,cells,pb);m=score_gate(opened,locator_scores,data)
        risk=(data["target"]>=0)[pb]; ga=auc(risk,raw[pb])
        rows.append({"name":recipe.name,"members":recipe.members,"mode":recipe.mode,
                     **scalar_metrics(m),"gate_auc":ga,"false_open":m["false_open"],
                     "false_close":m["false_close"],"open_rate":m["open_rate"],
                     "effective_rank":effective_rank(matrix[pb])})
        oof_raw[recipe.name]=raw;fits[recipe.name]=ff
        print("gate",recipe.name,m["pb"],ga,flush=True)
    return rows,oof_raw,fits


def stitch_inner_gate_selection(data,gate_x,gate_names,recipes,locator_scores):
    folds=data["folds"];cells=data["cells"].astype(str);pb=np.char.startswith(cells,"pb_")
    selected=[];final_raw=np.full(len(cells),np.nan);details=[]
    for outer in range(5):
        candidates=[]
        for recipe in recipes:
            matrix=recipe_matrix(recipe,gate_x,gate_names);raw=np.full(len(cells),np.nan);valid=[]
            for inner in range(5):
                if inner==outer:continue
                train=np.flatnonzero((folds!=outer)&(folds!=inner)&pb);val=np.flatnonzero((folds==inner)&pb)
                w,_=fit_fusion_weights(matrix[train],recipe,seed=396250+outer*10+inner)
                raw[val]=matrix[val]@w;valid.extend(val.tolist())
            subset=np.asarray(sorted(valid));_,opened=rank_gate_score(raw,cells,np.isin(np.arange(len(cells)),subset))
            m=score_gate(opened,locator_scores,data,subset);candidates.append((recipe,m))
        choice=max(candidates,key=lambda r:r[1]["pb"])[0];selected.append(choice.name)
        matrix=recipe_matrix(choice,gate_x,gate_names);train=np.flatnonzero((folds!=outer)&pb);test=np.flatnonzero((folds==outer)&pb)
        w,meta=fit_fusion_weights(matrix[train],choice,seed=396250+outer);final_raw[test]=matrix[test]@w
        details.append({"outer":outer,"selected":choice.name,"weights":w,"meta":meta,
                        "inner":{r.name:scalar_metrics(m) for r,m in candidates}})
    return selected,final_raw,details


def report(results):
    loc=results["locator_candidates"];gate=results["gate_candidates"];inter=results["interaction"]
    lines=["# L-SML gate/locator research v1","","Development-only; no untouched confirmation.","",
           "## Locator cross-fitted candidates","",
           "| Candidate | PB | Within | Effective rank |","|---|---:|---:|---:|"]
    for r in sorted(loc,key=lambda x:x["pb"],reverse=True):
        lines.append(f"| {r['name']} | {100*r['pb']:.4f}% | {r['within']:.6f} | {r['effective_rank']:.2f} |")
    lines += ["","## Gate cross-fitted candidates","",
              "| Candidate | PB with frozen locator | Gate AUC | False open | False close |","|---|---:|---:|---:|---:|"]
    for r in sorted(gate,key=lambda x:x["pb"],reverse=True):
        lines.append(f"| {r['name']} | {100*r['pb']:.4f}% | {r['gate_auc']:.4f} | {r['false_open']:.4f} | {r['false_close']:.4f} |")
    lines += ["","## Frozen 2x2 interaction","","| Locator | Gate | PB | Within |","|---|---|---:|---:|"]
    for r in inter:
        lines.append(f"| {r['locator']} | {r['gate']} | {100*r['pb']:.4f}% | {r['within']:.6f} |")
    lines += ["",f"Interaction delta (PB): {100*results['interaction_delta_pb']:+.4f} pp.","",
              "## Stability","",f"Locator selections: `{results['locator_nested_selections']}`.",
              f"Gate selections: `{results['gate_nested_selections']}`.",""]
    (OUTPUT/"REPORT.md").write_text("\n".join(lines))


def run_diagnostics(draws: int = 10_000) -> None:
    """Finalist-only ablations and paired source-group uncertainty."""
    data=load_inputs(); step_raw,names=merge_step_bank(data);x=answer_standardize(step_raw,data["offsets"])
    base=next(r for r in locator_recipes(names) if r.name=="L08_atlas_diverse_continuous")
    recipes=[FusionRecipe("L08_equal",base.members,"equal",anchor=0)]
    recipes += [FusionRecipe(f"L08_minus_{i}_{member.split('::')[-1]}",
        tuple(v for j,v in enumerate(base.members) if j!=i),"continuous",anchor=0)
        for i,member in enumerate(base.members)]
    rows,scores,fits=evaluate_locator_recipes(data,x,names,recipes)
    selections,nested_score,nested_details=stitch_inner_locator_selection(data,x,names,recipes)
    counts=Counter(selections);stable=[name for name,count in counts.items() if count>=4]
    if stable:
        selected=max(stable,key=lambda name:next(r["pb"] for r in rows if r["name"]==name))
        finalist=np.asarray(scores[selected],float)
    else:
        selected="nested_unstable_stitched";finalist=np.asarray(nested_score,float)
    dump(OUTPUT/"ABLATIONS.json",{"schema":SCHEMA+"/ablations-v2","rows":rows,"fits":fits,
        "nested_selections":selections,"stable":stable,"selected":selected,"nested_details":nested_details})

    saved=np.load(OUTPUT/"OOF_SCORES.npz",allow_pickle=False)
    original_l08=np.asarray(saved["final_locator"],float)
    current=np.asarray(data["digit025_scores"],float);gate=np.asarray(data["current_gate"],bool)
    cur=score_locator(current,gate,data);fin=score_locator(finalist,gate,data)
    base_fin=score_locator(original_l08,gate,data)
    cells=data["cells"].astype(str);groups=data["groups"].astype(str);target=data["target"]
    unique,group_id=np.unique(groups,return_inverse=True);rng=np.random.default_rng(396151)
    pb_delta=np.empty(draws);within_delta=np.empty(draws)
    base_pb_delta=np.empty(draws);base_within_delta=np.empty(draws)
    cur_pred=np.full(len(target),-999,np.int32);fin_pred=np.full(len(target),-999,np.int32)
    cur_pred[cur["answer_indexes"]]=cur["prediction"];fin_pred[fin["answer_indexes"]]=fin["prediction"]
    base_pred=np.full(len(target),-999,np.int32);base_pred[base_fin["answer_indexes"]]=base_fin["prediction"]
    cur_with=np.full(len(target),np.nan);fin_with=np.full(len(target),np.nan)
    cur_with[cur["answer_indexes"]]=cur["within_values"];fin_with[fin["answer_indexes"]]=fin["within_values"]
    base_with=np.full(len(target),np.nan);base_with[base_fin["answer_indexes"]]=base_fin["within_values"]
    valid=np.ones(len(target),bool)
    for draw in range(draws):
        count=np.bincount(rng.integers(0,len(unique),len(unique)),minlength=len(unique))
        weights=count[group_id].astype(float)
        cm=pb_metrics(target,cur_pred,valid,cells,weights=weights)["macros"]["all"]
        fm=pb_metrics(target,fin_pred,valid,cells,weights=weights)["macros"]["all"]
        bm=pb_metrics(target,base_pred,valid,cells,weights=weights)["macros"]["all"]
        pb_delta[draw]=fm-cm
        base_pb_delta[draw]=bm-cm
        c=np.isfinite(cur_with)&(weights>0);f=np.isfinite(fin_with)&(weights>0)
        within_delta[draw]=np.average(fin_with[f],weights=weights[f])-np.average(cur_with[c],weights=weights[c])
        b=np.isfinite(base_with)&(weights>0)
        base_within_delta[draw]=np.average(base_with[b],weights=weights[b])-np.average(cur_with[c],weights=weights[c])
        if (draw+1)%2000==0: print(f"bootstrap {draw+1}/{draws}",flush=True)
    uncertainty={
        "schema":SCHEMA+"/paired-source-bootstrap-v1","draws":draws,"seed":396151,
        "groups":len(unique),"contrast":selected+" - current digit025",
        "pb_delta":{"point":fin["pb"]-cur["pb"],"low":float(np.quantile(pb_delta,.025)),
                    "high":float(np.quantile(pb_delta,.975)),"probability_positive":float(np.mean(pb_delta>0))},
        "within_delta":{"point":fin["within"]-cur["within"],"low":float(np.quantile(within_delta,.025)),
                        "high":float(np.quantile(within_delta,.975)),"probability_positive":float(np.mean(within_delta>0))},
        "stable_l08_contrast": {
            "contrast":"L08_atlas_diverse_continuous - current digit025",
            "pb_delta":{"point":base_fin["pb"]-cur["pb"],"low":float(np.quantile(base_pb_delta,.025)),
                        "high":float(np.quantile(base_pb_delta,.975)),"probability_positive":float(np.mean(base_pb_delta>0))},
            "within_delta":{"point":base_fin["within"]-cur["within"],"low":float(np.quantile(base_within_delta,.025)),
                            "high":float(np.quantile(base_within_delta,.975)),"probability_positive":float(np.mean(base_within_delta>0))},
        },
    }
    dump(OUTPUT/"BOOTSTRAP.json",uncertainty)
    # Idempotent report generation: replace the base report before appending
    # diagnostics so reruns never accumulate stale sections.
    report(json.loads((OUTPUT/"RESULTS.json").read_text()))
    report_path=OUTPUT/"REPORT.md"
    with report_path.open("a") as handle:
        handle.write("\n## Finalist diagnostics\n\n")
        handle.write(f"Nested leave-one selection: `{selections}`; frozen diagnostic finalist: `{selected}`.\n\n")
        stable_boot=uncertainty["stable_l08_contrast"]
        handle.write(f"Stable L08 paired source bootstrap: PB delta {100*stable_boot['pb_delta']['point']:+.4f} pp "
            f"[{100*stable_boot['pb_delta']['low']:+.4f}, {100*stable_boot['pb_delta']['high']:+.4f}]; "
            f"within delta {stable_boot['within_delta']['point']:+.6f} "
            f"[{stable_boot['within_delta']['low']:+.6f}, {stable_boot['within_delta']['high']:+.6f}].\n\n")
        handle.write(f"Paired source bootstrap ({draws:,} draws): PB delta "
            f"{100*uncertainty['pb_delta']['point']:+.4f} pp "
            f"[{100*uncertainty['pb_delta']['low']:+.4f}, {100*uncertainty['pb_delta']['high']:+.4f}]; "
            f"within delta {uncertainty['within_delta']['point']:+.6f} "
            f"[{uncertainty['within_delta']['low']:+.6f}, {uncertainty['within_delta']['high']:+.6f}].\n\n")
        handle.write("| Ablation | PB | Within |\n|---|---:|---:|\n")
        for row in sorted(rows,key=lambda v:v["pb"],reverse=True):
            handle.write(f"| {row['name']} | {100*row['pb']:.4f}% | {row['within']:.6f} |\n")


def run() -> None:
    data=load_inputs(); step_raw,names=merge_step_bank(data)
    x=answer_standardize(step_raw,data["offsets"])
    lrecipes=locator_recipes(names)
    lrows,lscores,lfits=evaluate_locator_recipes(data,x,names,lrecipes)
    lsel,final_locator,lnested=stitch_inner_locator_selection(data,x,names,lrecipes)
    lcount=Counter(lsel); stable=[n for n,c in lcount.items() if c>=4]
    if stable:
        locator_name=max(stable,key=lambda n:next(r["pb"] for r in lrows if r["name"]==n))
        final_locator=lscores[locator_name]
    else:
        locator_name="nested_unstable_stitched"
    cells=data["cells"].astype(str);pb=np.char.startswith(cells,"pb_")
    gate_raw=np.asarray(data["gate_values"],float);gate_names=data["gate_names"].astype(str).tolist()
    gate_x=cell_midranks(gate_raw,cells,pb)
    grecipes=gate_recipes(gate_names)
    grows,gscores,gfits=evaluate_gate_recipes(data,gate_x,gate_names,grecipes,final_locator)
    gsel,final_gate_raw,gnested=stitch_inner_gate_selection(data,gate_x,gate_names,grecipes,final_locator)
    gcount=Counter(gsel);gstable=[n for n,c in gcount.items() if c>=4]
    if gstable:
        gate_name=max(gstable,key=lambda n:next(r["pb"] for r in grows if r["name"]==n));final_gate_raw=gscores[gate_name]
    else: gate_name="nested_unstable_stitched"
    _,final_gate=rank_gate_score(final_gate_raw,cells,pb)
    arms=[]
    for lname,lscore in [("digit025",data["digit025_scores"]),(locator_name,final_locator)]:
        for gname,gopen in [("current_tail15_digit_rate",data["current_gate"]),(gate_name,final_gate)]:
            m=score_locator(lscore,gopen,data);arms.append({"locator":lname,"gate":gname,**scalar_metrics(m)})
    lookup={(r["locator"],r["gate"]):r for r in arms}
    a00=lookup[("digit025","current_tail15_digit_rate")]["pb"]
    a10=lookup[(locator_name,"current_tail15_digit_rate")]["pb"]
    a01=lookup[("digit025",gate_name)]["pb"];a11=lookup[(locator_name,gate_name)]["pb"]
    results={"schema":SCHEMA,"development_only":True,"locator_candidates":lrows,"gate_candidates":grows,
             "locator_nested_selections":lsel,"gate_nested_selections":gsel,"locator_finalist":locator_name,
             "gate_finalist":gate_name,"interaction":arms,"interaction_delta_pb":a11-a10-a01+a00,
             "locator_fits":lfits,"gate_fits":gfits,"locator_nested":lnested,"gate_nested":gnested}
    dump(OUTPUT/"RESULTS.json",results)
    np.savez_compressed(OUTPUT/"OOF_SCORES.npz",final_locator=final_locator,final_gate_raw=final_gate_raw,
                        final_gate_open=final_gate,locator_names=np.asarray(list(lscores)),
                        gate_names=np.asarray(list(gscores)),**{f"locator__{k}":v for k,v in lscores.items()},
                        **{f"gate__{k}":v for k,v in gscores.items()})
    report(results)
    print(OUTPUT/"REPORT.md")


def main(argv: Sequence[str] | None=None):
    parser=argparse.ArgumentParser();parser.add_argument("stage",choices=("prepare","run","diagnostics"),nargs="?",default="run")
    args=parser.parse_args(argv)
    if args.stage=="prepare": print(prepare())
    elif args.stage=="diagnostics": run_diagnostics()
    else: run()


if __name__=="__main__": main()
