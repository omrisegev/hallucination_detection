"""One-time label evaluation of the two frozen candidates against N56 (spec section 11 B, C, D).

Stage 1 (no labels): per source-group outer fold, fit T and every fusion on training steps,
score test steps, set the Q80 flag threshold from training scores. Predictions are sealed.
Stage 2: read labels, compute metrics, paired source-group bootstrap, diagnostics.
Usage: python evaluate.py predict | python evaluate.py evaluate
"""
import hashlib, json, pickle, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
HERE = Path(__file__).parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from spectral_utils.advisor_lr import fit_labelfree, fit_lr  # noqa: E402
from spectral_utils.advisor_controls import answer_z, metrics, confusion_by_group, score_from_confusion, bootstrap_answer_mean  # noqa: E402
from spectral_utils.advisor_statistics import answer_auc  # noqa: E402
from spectral_utils.prmbench import prmbench_evaluate  # noqa: E402
from spectral_utils.fusion_utils import lsml_continuous  # noqa: E402
import transforms as T  # noqa: E402

D = ROOT / 'results/advisor_completion_2026-10-11/inputs'
OUT = HERE / 'evaluation'; PRED = OUT / 'PREDICTIONS.npz'
ARMS = {'N56': None, 'A_copies_pc1': False, 'B_copies_pc1_lenpos': True}
METHODS = ('LSML', 'IU', 'EQ', 'LSML_PRM')
CATEGORIES = {'simplicity': ['redundency', 'circular'],
              'soundness': ['counterfactual', 'step_contradiction', 'domain_inconsistency', 'confidence'],
              'sensitivity': ['missing_condition', 'deception', 'multi_solutions']}


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cut(off, answers):
    return (np.concatenate([np.arange(off[i], off[i + 1]) for i in answers]), np.r_[0, np.cumsum(np.diff(off)[answers])])


def load_inputs():
    Zf = np.load(D / 'bank_V_N56.npz'); P = np.load(D / 'population.npz')
    off = Zf['offsets']; assert np.array_equal(off, P['offsets'])
    folds = json.loads((D / 'FOLDS_V2.json').read_text())['prmbench']['outer']
    assert all(int(folds[str(g)]) == int(f) for g, f in zip(P['group'], P['fold']))
    prm = pickle.load(open(D / 'prmbench_prm.pkl', 'rb')); by = {str(r['idx']): r for r in prm.values()}
    reward = np.concatenate([np.asarray(by[str(i)]['rewards'], float) for i in P['idx']])
    assert len(reward) == off[-1]
    return Zf['V'], [str(c) for c in Zf['columns']], off, P, reward


def predict():
    freeze = json.loads((HERE / 'FREEZE.json').read_text())
    assert all(sha(HERE / p) == h for p, h in freeze['code'].items() if p != 'freeze.py'), 'frozen code changed'
    X, names, off, P, reward = load_inputs()
    cov = T.step_covariates(off, P['token_lengths']); risk = 1 - reward
    n = off[-1]; pred = {}; diag = {}
    for arm in ARMS:
        for m in METHODS:
            pred[f'{arm}__{m}'] = np.full(n, np.nan); pred[f'{arm}__{m}__flags'] = np.zeros(n, bool)
    pred['PRM__Q80__flags'] = np.zeros(n, bool)
    for f in range(5):
        tr_a = np.flatnonzero(P['fold'] != f); te_a = np.flatnonzero(P['fold'] == f)
        assert not set(P['group'][tr_a]) & set(P['group'][te_a])
        tr, toff = cut(off, tr_a); te, eoff = cut(off, te_a)
        ptr, pte = answer_z(risk[tr], toff), answer_z(risk[te], eoff)
        tau_prm = float(np.quantile(ptr, .8)); pred['PRM__Q80__flags'][te] = pte >= tau_prm
        old = np.load(D.parent / f'lr/checkpoints/N56/outer_{f}/SCORES.npz'); assert np.array_equal(old['rows'], te)
        for arm, res in ARMS.items():
            if res is None:
                Ztr, Zte = X[tr], X[te]
            else:
                tf = T.FrozenT(res).fit(X[tr], names, toff, cov[tr])
                Ztr, Zte = tf.transform(X[tr], toff, cov[tr]), tf.transform(X[te], eoff, cov[te])
                diag[f'{arm}/fold{f}/T'] = tf.describe()
            for m in METHODS:
                key = f'{arm}__{m}'
                if m == 'EQ':
                    s_tr, s_te = Ztr.mean(1), Zte.mean(1)
                else:
                    A_tr = np.column_stack([Ztr, ptr]) if m == 'LSML_PRM' else Ztr
                    A_te = np.column_stack([Zte, pte]) if m == 'LSML_PRM' else Zte
                    try:
                        model = fit_labelfree(A_tr, 'IU_PCR' if m == 'IU' else 'LSML_CONT')
                    except (ValueError, RuntimeError) as exc:
                        diag[f'{key}/fold{f}'] = {'FAILED': repr(exc)}; continue
                    s_tr, s_te = model.predict(A_tr), model.predict(A_te)
                    d = {k: v for k, v in model.diagnostic.items() if k in ('K', 'degenerate', 'cross_weights', 'group')}
                    if m == 'LSML_PRM':
                        contrib = np.abs(model.weight) * A_tr.std(0)
                        d['prm_weight_share'] = float(contrib[-1] / contrib.sum()); d['prm_weight_sign'] = float(np.sign(model.weight[-1]))
                    diag[f'{key}/fold{f}'] = d
                    if arm == 'N56' and m in ('LSML', 'IU'):
                        err = float(np.max(np.abs(s_te - old['LSML_FULL' if m == 'LSML' else 'IU_FULL'])))
                        diag[f'{key}/fold{f}']['baseline_max_abs'] = err; assert err <= 1e-9, (key, f, err)
                tau = float(np.quantile(answer_z(s_tr, toff), .8))
                pred[key][te] = s_te; pred[key + '__flags'][te] = answer_z(s_te, eoff) >= tau
        print('fold', f, 'done', flush=True)
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(PRED, prm_risk=risk, **pred)
    (OUT / 'FIT_DIAGNOSTICS.json').write_text(json.dumps(diag, indent=1))
    (OUT / 'PREDICTION_SEAL.json').write_text(json.dumps(dict(status='SEALED_BEFORE_LABELS', predictions=sha(PRED),
                                                              diagnostics=sha(OUT / 'FIT_DIAGNOSTICS.json')), indent=1))


def ds_prevalence(V, q=.8, iters=200):
    """Two-class Dawid-Skene on per-channel top-20% votes (cell-wide). Returns prevalence, balanced accuracy."""
    votes = (V >= np.quantile(V, q, axis=0)).astype(float)
    t = (votes.mean(1) > .5).astype(float) * .8 + .1
    for _ in range(iters):
        pi = t.mean()
        se = (t[:, None] * votes).sum(0) / t.sum(); sp = ((1 - t)[:, None] * (1 - votes)).sum(0) / (1 - t).sum()
        se, sp = np.clip(se, 1e-4, 1 - 1e-4), np.clip(sp, 1e-4, 1 - 1e-4)
        l1 = np.log(pi) + votes @ np.log(se) + (1 - votes) @ np.log(1 - se)
        l0 = np.log(1 - pi) + votes @ np.log(1 - sp) + (1 - votes) @ np.log(sp)
        t = 1 / (1 + np.exp(l0 - l1))
    return float(t.mean()), (se + sp) / 2


def evaluate():
    seal = json.loads((OUT / 'PREDICTION_SEAL.json').read_text()); assert sha(PRED) == seal['predictions']
    X, names, off, P, reward = load_inputs(); p = np.load(PRED)
    y = P['y'].astype(bool); cls = P['classification'].astype(str); ids = P['idx']   # labels read here, after the seal
    meta = {str(r['idx']): r for r in json.loads((D / 'prmbench_meta.json').read_text())['rows']}
    unique, inv = np.unique(P['group'], return_inverse=True); sg = np.repeat(inv, np.diff(off)); assert len(unique) == 707
    W = np.random.default_rng(20261020).multinomial(len(unique), np.full(len(unique), 1 / len(unique)), size=2000).astype(float)
    L = np.diff(off); pos = np.concatenate([np.arange(l) / max(l - 1, 1) for l in L]); posz = answer_z(pos, off)
    noncontrol = np.repeat(cls != 'correct', L)
    scores = {k: p[k] for k in p.files if '__' in k and not k.endswith('__flags')}
    scores['PRM__alone'] = p['prm_risk']
    flags = {k: p[k + '__flags'] for k in scores if k != 'PRM__alone'}
    flags['PRM__alone'] = reward < .5; flags['PRM__Q80'] = p['PRM__Q80__flags']; scores['PRM__Q80'] = p['prm_risk']
    rows, cats, per_auc, per_conf = [], [], {}, {}
    for key, s in scores.items():
        covered = np.array([np.isfinite(s[a:b]).all() for a, b in zip(off[:-1], off[1:])])
        s0 = np.where(np.isfinite(s), s, 0.); auc, _ = answer_auc(y, s0, off); auc[~covered] = np.nan
        sz = answer_z(s0, off); bpos = np.dot(sz, posz) / np.dot(posz, posz)
        auc_pos, _ = answer_auc(y, sz - bpos * posz, off); auc_pos[~covered] = np.nan
        wrong = np.array([y[a:b].any() for a, b in zip(off[:-1], off[1:])]) & covered & (cls != 'correct')
        top1 = np.array([y[a + int(np.argmax(s0[a:b]))] for a, b in zip(off[:-1], off[1:])])
        fl = flags[key]
        preds = [{'idx': str(ids[i]), 'labels': (~fl[a:b]).astype(int).tolist()} for i, (a, b) in enumerate(zip(off[:-1], off[1:])) if covered[i]]
        ev = prmbench_evaluate(preds, [meta[str(ids[i])] for i in np.flatnonzero(covered)])
        prms = .5 * (ev['total']['f1'] + ev['total']['negative_f1'])
        sel = np.repeat(covered, L) & noncontrol
        assert abs(prms - metrics(y[sel], fl[sel])['prm_score']) < 1e-12
        conf = confusion_by_group(y, fl, sg, sel, len(unique)); prm_draw = score_from_confusion(conf, W)
        draw = bootstrap_answer_mean(auc, inv, covered, W)
        per_auc[key], per_conf[key] = auc, conf
        rows.append(dict(key=key, covered=int(covered.sum()), within_auc=float(np.nanmean(auc)),
                         auc_lo95=float(np.quantile(draw, .025)), auc_hi95=float(np.quantile(draw, .975)),
                         prmscore=float(prms), prms_lo95=float(np.quantile(prm_draw, .025)), prms_hi95=float(np.quantile(prm_draw, .975)),
                         top1_error=float(top1[wrong].mean()), within_auc_position_removed=float(np.nanmean(auc_pos))))
        sub = {}
        for typ in np.unique(cls):
            sm = np.repeat(covered & (cls == typ), L); m = metrics(y[sm], fl[sm])
            sub[typ] = m['f1_valid'] if typ == 'multi_solutions' else m['prm_score'] if typ != 'correct' else None
        for cat, mem in CATEGORIES.items():
            cats.append(dict(key=key, category=cat, score=float(np.mean([sub[t] for t in mem]))))
        for typ in ('redundency', 'circular'):
            cats.append(dict(key=key, category=f'type:{typ}', score=float(sub[typ])))
        print(key, round(rows[-1]['within_auc'], 4), round(prms, 4), flush=True)
    contrasts = []
    for arm in ('A_copies_pc1', 'B_copies_pc1_lenpos'):
        for m in METHODS:
            a, b = f'{arm}__{m}', f'N56__{m}'
            common = np.isfinite(per_auc[a]) & np.isfinite(per_auc[b]); dl = per_auc[a] - per_auc[b]
            draw = bootstrap_answer_mean(np.where(common, dl, 0), inv, common, W)
            dp = score_from_confusion(per_conf[a], W) - score_from_confusion(per_conf[b], W)
            ra = next(r for r in rows if r['key'] == a); rb = next(r for r in rows if r['key'] == b)
            contrasts.append(dict(contrast=f'{a} - {b}', delta_auc=float(np.nanmean(dl[common])), auc_lo95=float(np.quantile(draw, .025)),
                                  auc_hi95=float(np.quantile(draw, .975)), delta_prmscore=ra['prmscore'] - rb['prmscore'],
                                  prms_lo95=float(np.quantile(dp, .025)), prms_hi95=float(np.quantile(dp, .975))))
    # Section 11-B diagnostics on full-data fits (they use labels; not a selection input)
    cov = T.step_covariates(off, P['token_lengths']); diag = {}
    for arm, res in ARMS.items():
        Z = X if res is None else T.FrozenT(res).fit(X, names, off, cov).transform(X, off, cov)
        _, meta_l = lsml_continuous(*[Z[:, j] for j in range(Z.shape[1])]); c = np.asarray(meta_l['c'])
        between = (c[:, None] != c[None, :])
        cond = {lab: float(np.abs(np.corrcoef(Z[y == lab], rowvar=False)[between]).mean()) for lab in (False, True)}
        prev, bacc = ds_prevalence(Z)
        true_auc = np.array([np.nanmean(answer_auc(y, Z[:, j], off)[0]) for j in range(Z.shape[1])])
        lr_oof = np.full(off[-1], np.nan)
        for f in range(5):
            tr, _ = cut(off, np.flatnonzero(P['fold'] != f)); te, _ = cut(off, np.flatnonzero(P['fold'] == f))
            lr_oof[te] = fit_lr(Z[tr], y[tr]).predict(Z[te])
        diag[arm] = dict(m=int(Z.shape[1]), L1_between_group_abs_r_correct_steps=cond[False], L1_between_group_abs_r_error_steps=cond[True],
                         L2_oracle_LR_within_auc=float(np.nanmean(answer_auc(y, lr_oof, off)[0])),
                         L3_DS_prevalence=prev, true_step_error_rate=float(y[noncontrol].mean()), true_step_error_rate_all=float(y.mean()),
                         L4_spearman_DS_accuracy_vs_true_auc=float(spearmanr(bacc, true_auc)[0]),
                         best_single_channel_auc=float(np.nanmax(np.maximum(true_auc, 1 - true_auc))))
    fd = json.loads((OUT / 'FIT_DIAGNOSTICS.json').read_text())
    prm_share = {arm: [fd[f'{arm}__LSML_PRM/fold{f}'].get('prm_weight_share') for f in range(5)] for arm in ARMS}
    prm_sign = {arm: [fd[f'{arm}__LSML_PRM/fold{f}'].get('prm_weight_sign') for f in range(5)] for arm in ARMS}
    res = dict(labels_read_after_seal=True, rows=rows, categories=cats, contrasts=contrasts, diagnostics_B=diag,
               prm_weight_share=prm_share, prm_weight_sign=prm_sign, bootstrap_draws=2000, source_groups=707)
    (OUT / 'RESULTS.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(dict(contrasts=contrasts, diagnostics_B=diag, prm_weight_share=prm_share), indent=1))


if __name__ == '__main__':
    {'predict': predict, 'evaluate': evaluate}[sys.argv[1]]()
