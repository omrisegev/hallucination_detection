"""evaluate_sla_standalone.py: ProcessBench SLA under the GAP rule, STANDALONE telemetry vs Mind the Gap (Table 3).

Standalone counterpart of results/processbench_gap_sla_v1/evaluate_sla.py (Step 483, PRM-backbone telemetry).
Same metric, same rule, same published reference rows. Only the score source changes.

Metric (Mind the Gap, Chen et al., ICML 2026, Table 3):
  SLA = exact match of the predicted first-error step with the gold first-error step (tolerance 0),
  ONLY on the 2,221 erroneous answers (gold label != -1). Macro = mean of the 4 subset SLAs.
GAP rule (label-free, per answer of m steps):
  sort the step risk scores descending; km = min(ceil(m/2), m-1); k = 1 + argmax of the first km sorted gaps;
  flag the top-k steps; prediction = the chronologically first flagged step. Argmax top-1 is a control.

Score files (label-free, from AIRCC; copies on Drive cluster_results/prm_boost_crop_ares_benchmarks_v1/):
  standalone: SCORES_sa_<prm>_processbench.npz  (<prm> names the PRM whose base model produced the telemetry)
  backbone:   SCORES_CHAIN_<prm>.npz             (Step 483 input, used for the parity check only)
Arms (standalone = base-model telemetry only, no PRM signal): TELE_N16 (L-SML), MEAN_N16, TELE_N13, MEAN_N13.
Controls: POS_FIRST (always step 0), POS_LAST (always the last step).
Math-Shepherd is excluded (Omri, 2026-10-28).

Step index mapping. Score arrays can hold only the NON-EMPTY steps of an answer ('nonempty' mask). The gold label is
an index into ALL steps. This script maps the predicted non-empty position back to the full step index. Step 483 used
the non-empty position directly; the script reports both ('raw' = Step 483 convention) so the effect is visible.

    python evaluate_sla_standalone.py            # env: SA_SCORES_DIR, PB_SCORES_DIR (backbone, optional), PB_GOLD_PATH
-> RESULTS.csv, BOOTSTRAP.csv, PARITY.json, DATA_CHECK.json
"""
import csv, json, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STEP483 = ROOT / 'results/processbench_gap_sla_v1'
sys.path.insert(0, str(STEP483))
from evaluate_sla import MIND_THE_GAP, SUBSETS  # noqa: E402  (same published rows as Step 483)


def first(*cands):
    return next((Path(p) for p in cands if p and Path(p).exists()), None)


SA_DIR = first(os.environ.get('SA_SCORES_DIR'),
               ROOT.parent / 'hd_collection_data/crop_ares_v1/scores',
               ROOT / 'results/processbench_gap_sla_standalone_v1/data')
BB_DIR = first(os.environ.get('PB_SCORES_DIR'), ROOT / 'results/prm_boost_processbench_v1/scores',
               r'C:\Users\omris\TAU\wt_pb_integ\results\prm_boost_processbench_v1\scores',
               ROOT.parent / 'hd_collection_data/sub5_all_v1/scores')
GOLD = first(os.environ.get('PB_GOLD_PATH'), ROOT / 'results/prm_boost_processbench_v1/local/processbench_gold.json')

PRMS = ['qwen_prm', 'qwen_prm_72b', 'pure_prm', 'rlhflow_prm']
BASE = {'qwen_prm': 'Qwen2.5-Math-7B-Instruct', 'qwen_prm_72b': 'Qwen2.5-Math-72B-Instruct',
        'pure_prm': 'Qwen2.5-Math-7B', 'rlhflow_prm': 'Llama-3.1-8B-Instruct'}
SA_ARMS = ['TELE_N16', 'MEAN_N16', 'TELE_N13', 'MEAN_N13']
N_BOOT, SEED = 2000, 0


def gap_predict(s):
    """Returns (first-flag position, argmax position) inside one answer's score vector."""
    m = len(s)
    if m == 0: return -1, -1
    if m == 1: return 0, 0
    order = np.argsort(-s, kind='stable'); srt = s[order]
    km = min(int(np.ceil(m / 2)), m - 1)
    k = int(np.argmax((srt[:-1] - srt[1:])[:km])) + 1
    return int(order[:k].min()), int(order[0])


def load_cell(path, gold):
    """Per answer: score segments over non-empty steps + map to full step index."""
    S = np.load(path)
    uids = S['uids'].astype(str); lens = S['nonempty_lengths'].astype(int)
    ne = S['nonempty'].astype(bool) if 'nonempty' in S.files else None
    arms = {k[len('score__'):]: S[k].astype(float) for k in S.files if k.startswith('score__')}
    n_rows = len(next(iter(arms.values())))
    starts = np.r_[0, np.cumsum(lens)]
    chk = dict(file=str(path.name), n_answers=len(uids), sum_lens=int(lens.sum()), n_score_rows=n_rows,
               has_nonempty=ne is not None, n_nonempty=int(ne.sum()) if ne is not None else None,
               gold_nsteps_match=int(sum(gold[u]['n_steps'] == l for u, l in zip(uids, lens))))
    segs, maps = [], []
    if ne is not None and len(ne) == lens.sum() and n_rows == ne.sum():
        layout = 'full_lens + nonempty mask, scores on non-empty rows'
        ne_start = np.r_[0, np.cumsum([ne[a:b].sum() for a, b in zip(starts[:-1], starts[1:])])]
        for j, (a, b) in enumerate(zip(starts[:-1], starts[1:])):
            segs.append((ne_start[j], ne_start[j + 1])); maps.append(np.flatnonzero(ne[a:b]))
    elif n_rows == lens.sum():
        layout = 'scores on lens rows (no remap needed or nonempty mask unusable)'
        for a, b in zip(starts[:-1], starts[1:]):
            if ne is not None and len(ne) == lens.sum():
                maps.append(np.flatnonzero(ne[a:b]) if ne[a:b].sum() == b - a else np.arange(b - a))
            else:
                maps.append(np.arange(b - a))
            segs.append((a, b))
    else:
        raise SystemExit(f'unknown layout {chk}')
    chk['layout'] = layout
    chk['answers_with_empty_steps'] = int(sum(len(mp) < gold[u]['n_steps'] for u, mp in zip(uids, maps)))
    return uids, arms, segs, maps, chk


def predictions(sc, segs, maps):
    out = {k: [] for k in ('gap_raw', 'gap', 'argmax')}
    for (a, b), mp in zip(segs, maps):
        c, t = gap_predict(sc[a:b])
        out['gap_raw'].append(c)
        out['gap'].append(int(mp[c]) if c >= 0 and len(mp) > c else c)
        out['argmax'].append(int(mp[t]) if t >= 0 and len(mp) > t else t)
    return {k: np.array(v) for k, v in out.items()}


def sla(hit, subs, idx=None):
    if idx is not None: hit, subs = hit[idx], subs[idx]
    per = {s: 100 * hit[subs == s].mean() for s in SUBSETS}
    return per, float(np.mean(list(per.values()))), 100 * float(hit.mean())


def row(model, mode, arm, rule, hit, subs):
    per, macro, overall = sla(hit, subs)
    return dict(model=model, base_model=BASE.get(model, ''), mode=mode, arm=arm, rule=rule, n_err=int(len(hit)),
                macro_sla=round(macro, 2), overall_sla=round(overall, 2),
                **{f'{s}_sla': round(per[s], 2) for s in SUBSETS})


def main():
    assert SA_DIR and GOLD, (SA_DIR, GOLD)
    gold = {g['uid']: g for g in json.load(open(GOLD))}
    rows, boot, parity, checks = [], [], [], []
    for name, v in MIND_THE_GAP.items():
        rows.append(dict(model='Mind the Gap paper', base_model=name.split(' ')[0], mode='published', arm=name,
                         rule='Shannon Drop (Table 3)', n_err=2221, macro_sla=v['macro'], overall_sla=v['overall'],
                         **{f'{s}_sla': v[s] for s in SUBSETS}))
    mtg8 = MIND_THE_GAP['Qwen3-8B (Shannon Drop)']['macro']
    rng = np.random.default_rng(SEED)

    # ---- parity: Step 483 backbone numbers with the same code (raw convention)
    ref483 = {(r['model'], r['arm'], r['rule']): float(r['macro_sla'])
              for r in csv.DictReader(open(STEP483 / 'RESULTS.csv'))}
    if BB_DIR:
        for prm in PRMS:
            p = BB_DIR / f'SCORES_CHAIN_{prm}.npz'
            if not p.exists(): continue
            uids, arms, segs, maps, chk = load_cell(p, gold); chk['mode'] = 'backbone'; checks.append(chk)
            lab = np.array([gold[u]['label'] for u in uids]); sub = np.array([gold[u]['subset'] for u in uids])
            e = lab != -1
            for arm in ('TELE_N16', 'MEAN_N16', 'PRM'):
                if arm not in arms: continue
                P = predictions(arms[arm], segs, maps)
                _, m_raw, _ = sla((P['gap_raw'] == lab)[e], sub[e])
                ref = ref483.get((prm, arm, 'GAP_first_flag'))
                parity.append(dict(check=f'backbone {prm} {arm} GAP raw vs Step 483', got=round(m_raw, 2), ref=ref,
                                   ok=ref is not None and abs(round(m_raw, 2) - ref) < 0.006))
                rows.append(row(prm, 'backbone (Step 483 input)', arm, 'GAP_first_flag', (P['gap'] == lab)[e], sub[e]))
    else:
        print('backbone score dir not found: parity with Step 483 skipped')

    # ---- standalone
    for prm in PRMS:
        p = SA_DIR / f'SCORES_sa_{prm}_processbench.npz'
        if not p.exists():
            print('missing', p); continue
        uids, arms, segs, maps, chk = load_cell(p, gold); chk['mode'] = 'standalone'
        chk['arms'] = sorted(arms); checks.append(chk)
        lab = np.array([gold[u]['label'] for u in uids]); sub = np.array([gold[u]['subset'] for u in uids])
        e = lab != -1
        nsteps = np.array([gold[u]['n_steps'] for u in uids])
        rows.append(row(prm, 'standalone', 'POS_FIRST', 'control', (lab == 0)[e], sub[e]))
        rows.append(row(prm, 'standalone', 'POS_LAST', 'control', (lab == nsteps - 1)[e], sub[e]))
        for arm in SA_ARMS:
            if arm not in arms: continue
            P = predictions(arms[arm], segs, maps)
            for rule, key in (('GAP_first_flag', 'gap'), ('GAP_first_flag_raw483', 'gap_raw'), ('Argmax_top1', 'argmax')):
                if rule != 'GAP_first_flag' and arm not in ('TELE_N16', 'MEAN_N16'): continue
                rows.append(row(prm, 'standalone', arm, rule, (P[key] == lab)[e], sub[e]))
            if arm in ('TELE_N16', 'MEAN_N16'):
                hit = (P['gap'] == lab)[e]; s_ = sub[e]
                # stratified answer bootstrap (resample inside each subset), macro SLA
                idx_by = {s: np.flatnonzero(s_ == s) for s in SUBSETS}
                bm = np.array([np.mean([100 * hit[rng.choice(ix, len(ix))].mean() for ix in idx_by.values()])
                               for _ in range(N_BOOT)])
                _, macro, _ = sla(hit, s_)
                boot.append(dict(model=prm, arm=arm, rule='GAP_first_flag', macro_sla=round(macro, 2),
                                 ci_lo=round(float(np.percentile(bm, 2.5)), 2), ci_hi=round(float(np.percentile(bm, 97.5)), 2),
                                 delta_vs_mtg_qwen3_8b=round(macro - mtg8, 2),
                                 ci_excludes_mtg8=bool(np.percentile(bm, 2.5) > mtg8 or np.percentile(bm, 97.5) < mtg8)))

    def dump(name, rs):
        if not rs: return
        keys = list(dict.fromkeys(k for r in rs for k in r))
        with open(HERE / name, 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rs)
    dump('RESULTS.csv', rows); dump('BOOTSTRAP.csv', boot)
    json.dump(parity, open(HERE / 'PARITY.json', 'w'), indent=1)
    json.dump(checks, open(HERE / 'DATA_CHECK.json', 'w'), indent=1)
    for r in rows:
        print(f"{r['model']:<18} {r['mode']:<26} {r['arm']:<26} {r['rule']:<22} macro {r['macro_sla']:>6} overall {r['overall_sla']:>6}")
    for b in boot: print(b)
    for p in parity: print(('PASS ' if p['ok'] else 'FAIL ') + p['check'], p['got'], p['ref'])


if __name__ == '__main__':
    main()
