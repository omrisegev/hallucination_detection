"""CPU feasibility of new per-step PRM features computable from saved caches (2026-10-26). No labels.

Question per candidate: does it compute, how many steps are valid, and how much of it is already in the PRM
channel and the N16 bank? Nothing here measures quality; labels are never loaded (the raw cache carries
error_steps=None, and no gold file is opened).

Population: Socratic-PRMBench, Qwen2.5-Math-PRM-7B same-pass cache (2,995 answers, 26,052 non-empty steps),
plus the reward/feature banks of Qwen-72B, PURE and RLHFlow on the same rows. Development data.

Redundancy measures, all on answer-z values (within-answer z-scoring, as every fusion input):
  r_prm     Pearson with z(logit(1 - reward)) of the same PRM (the CHAIN PRM channel)
  max_n16   largest |Pearson| with one N16 channel, and which channel
  R2_n16    R^2 of OLS of the candidate on the 16 N16 channels
  R2_all    R^2 on z(PRM) + N16 + relative position + log step tokens
  r_pos     Pearson with relative position (position confounds Socratic)
Reference row per N16 channel: the same R2_all with that channel left out of the regressors, so a new
candidate's redundancy can be read against what existing channels already have.

    python feasibility.py <data_dir> <out_dir> [<backbone tokenizer.json>]
With a tokenizer, two reflective-token channels are added (word list fixed below, before any run):
probability mass the backbone puts on self-correction words ("Wait", "But", "However", ...) among its top-50
alternatives, at the scoring state after step k and as a top-10 token mean inside step k.
data_dir holds <prm>_socratic/bank_v2/{FEATURES,REWARDS}.npz and qwen_prm_socratic/external_telemetry.pkl
(copied read-only from gdrive:hallucination_detection/cluster_results/prm_boost_external_v1/).
"""
import csv, json, pickle, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
N16 = json.loads((ROOT / 'results/monday_nested_banks_v2/BANKS.json').read_text())['N16']
OTHER = ('qwen_prm_72b', 'pure_prm', 'rlhflow_prm')
REFLECT = {'wait', 'but', 'however', 'hmm', 'actually', 'alternatively', 'hold', 'oops', 'no', 'wrong', 'incorrect',
           'mistake', 'error', 'correction', 'recheck', 'verify', 'check', 'double', 'reconsider', 'revisit'}


def reflect_ids(tokenizer_json):
    """Vocabulary ids whose decoded text, stripped of spaces and punctuation, is one self-correction word."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(tokenizer_json))
    ids = [i for i in range(tok.get_vocab_size()) if tok.decode([i]).strip(' \n\t,.:;!?*').lower() in REFLECT]
    return np.array(sorted(ids))


def reflect_mass(ids, lp, rid):
    """Per token: renormalised top-50 probability on the reflective ids."""
    p = np.exp(lp); p /= p.sum(1, keepdims=True)
    return (p * np.isin(ids, rid)).sum(1)


def az(v, o):
    v = np.asarray(v, float); out = np.zeros_like(v)
    for a, b in zip(o[:-1], o[1:]):
        x = v[a:b]; s = x.std()
        if b - a > 1 and s > 1e-12: out[a:b] = (x - x.mean()) / s
    return out


def risk(r):
    p = np.clip(1 - np.asarray(r, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fill(v, valid, o):
    """Invalid steps get the answer mean of the valid ones (the bank's own missing-value rule)."""
    v = np.array(v, float)
    for a, b in zip(o[:-1], o[1:]):
        ok = valid[a:b]
        v[a:b][~ok] = v[a:b][ok].mean() if ok.any() else 0.
    return v


def r2(y, X):
    X = np.column_stack([np.ones(len(y)), X])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    res = y - X @ beta
    return 1 - res.var() / y.var() if y.var() > 0 else np.nan


def load_bank(d):
    F = np.load(d / 'FEATURES.npz'); R = np.load(d / 'REWARDS.npz')
    names = [str(n) for n in F['names']]
    assert np.array_equal(F['uids'], R['uids']), 'bank/reward answer order differs'
    keep = F['nonempty'].astype(bool)
    o = F['offsets'].astype(np.int64)  # over non-empty steps; 'nonempty_lengths' counts every step
    assert o[-1] == keep.sum() == len(F['values'])
    return dict(uids=F['uids'].astype(str), V=F['values'][:, [names.index(c) for c in N16]].astype(float),
                rewards=R['rewards'][keep].astype(float), keep=keep, o=o, names=names, all=F['values'])


def main():
    data, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    B = load_bank(data / 'qwen_prm_socratic/bank_v2'); o = B['o']; n = o[-1]
    raw = pickle.load(open(data / 'qwen_prm_socratic/external_telemetry.pkl', 'rb'))
    rows = {str(r['idx']): r for r in raw.values()}
    assert all(r.get('error_steps') is None for r in rows.values()), 'labels present in the raw cache'
    meta = [rows[u] for u in B['uids']]

    # Step-level arrays over the non-empty steps, in bank order
    keep = B['keep']; full_o = np.r_[0, np.cumsum([len(m['step_token_spans']) for m in meta])]
    assert full_o[-1] == len(keep)
    rid = reflect_ids(sys.argv[3]) if len(sys.argv) > 3 else None
    pos_full, loglen_full, bnd_H, bnd_top1, bnd_valid, bnd_ref, step_ref = [], [], [], [], [], [], []
    for m in meta:
        sp = m['step_token_spans']; k = len(sp); H = np.asarray(m['token_entropies'], float)
        lp = np.asarray(m['top_k_logprobs']['logprobs'], float)
        if rid is not None:
            rm = reflect_mass(np.asarray(m['top_k_logprobs']['ids']), lp, rid)
            for j, (a, b) in enumerate(sp):
                bnd_ref.append(rm[sp[j + 1][0]] if j + 1 < k and sp[j + 1][1] > sp[j + 1][0] else np.nan)
                step_ref.append(np.sort(rm[a:b])[-10:].mean() if b > a else np.nan)
        pos_full += list((np.arange(k) + .5) / k); loglen_full += [np.log1p(max(b - a, 0)) for a, b in sp]
        for j in range(k):
            # The first token of step j+1 is predicted from the scoring position (<extra_0>) that closes step j:
            # the lm_head reading of the very hidden state the reward head reads for step j.
            nxt = sp[j + 1][0] if j + 1 < k and sp[j + 1][1] > sp[j + 1][0] else None
            bnd_valid.append(nxt is not None)
            bnd_H.append(H[nxt] if nxt is not None else np.nan)
            bnd_top1.append(np.exp(lp[nxt, 0]) if nxt is not None else np.nan)
    sel = lambda v: np.asarray(v, float)[keep]
    pos, loglen = sel(pos_full), sel(loglen_full)
    bnd_valid = np.asarray(bnd_valid)[keep]

    rw = B['rewards']; rk = risk(rw); zp = az(rk, o)
    nxt_valid = np.ones(n, bool); prv_valid = np.ones(n, bool)
    nxt_valid[o[1:] - 1] = False; prv_valid[o[:-1]] = False
    nxt = np.r_[rk[1:], 0.]; prv = np.r_[0., rk[:-1]]
    run_max = np.empty(n); fut_mean = np.empty(n)
    for a, b in zip(o[:-1], o[1:]):
        run_max[a:b] = np.maximum.accumulate(rw[a:b])
        c = np.cumsum(rk[a:b][::-1])[::-1]; cnt = np.arange(b - a, 0, -1)
        fut_mean[a:b] = np.r_[(c[1:] / cnt[1:]), 0.]
    cands = {
        'reward_next_risk': (nxt, nxt_valid, 'PRM risk of step k+1, read as evidence about step k'),
        'reward_delta_risk': (rk - prv, prv_valid, 'risk_k - risk_{k-1}: how sharply the verdict turns at step k'),
        'reward_drop_from_max': (run_max - rw, np.ones(n, bool), 'best reward so far minus reward_k'),
        'reward_future_mean_risk': (fut_mean, nxt_valid, 'mean PRM risk over steps after k'),
        'boundary_entropy': (sel(bnd_H), bnd_valid, 'full-vocab entropy at the scoring state after step k'),
        'boundary_top1_prob': (-np.log(np.clip(sel(bnd_top1), 1e-12, 1)), bnd_valid,
                               'surprisal of the backbone top-1 continuation at the scoring state after step k'),
    }
    if rid is not None:
        cands['boundary_reflect_mass'] = (sel(bnd_ref), bnd_valid, 'top-50 mass on self-correction words at the scoring state after step k')
        cands['step_reflect_mass'] = (sel(step_ref), np.ones(n, bool), 'top-10 token mean of that mass inside step k')
    others = {}
    for m in OTHER:
        Bm = load_bank(data / f'{m}_socratic/bank_v2')
        assert np.array_equal(Bm['uids'], B['uids']) and np.array_equal(Bm['o'], o), m
        cands[f'xprm_{m}_reward'] = (risk(Bm['rewards']), np.ones(n, bool), f'{m} PRM risk on the same step')
        Zm = np.column_stack([az(Bm['V'][:, j], o) for j in range(16)])
        others[m] = Zm

    Z = np.column_stack([az(B['V'][:, j], o) for j in range(16)])
    zpos, zlen = az(pos, o), az(loglen, o)
    rows_out = []
    def describe(name, z, valid, text):
        c = np.array([np.corrcoef(z, Z[:, j])[0, 1] for j in range(16)])
        jj = int(np.nanargmax(np.abs(c)))
        rows_out.append(dict(candidate=name, what=text, valid_rate=round(float(valid.mean()), 4),
                             answers_constant=int(sum(z[a:b].std() < 1e-12 for a, b in zip(o[:-1], o[1:]) if b - a > 1)),
                             r_prm=round(float(np.corrcoef(z, zp)[0, 1]), 3),
                             max_n16=round(float(c[jj]), 3), max_n16_channel=N16[jj],
                             mean_abs_n16=round(float(np.nanmean(np.abs(c))), 3),
                             R2_n16=round(float(r2(z, Z)), 3),
                             R2_all=round(float(r2(z, np.column_stack([zp, Z, zpos, zlen]))), 3),
                             r_pos=round(float(np.corrcoef(z, zpos)[0, 1]), 3)))
    for name, (v, valid, text) in cands.items():
        v = fill(v, valid, o)
        assert np.isfinite(v).all(), name
        describe(name, az(v, o), valid, text)
    # Cross-backbone telemetry: mean answer-z N16 of another PRM's backbone, as one summary view
    for m, Zm in others.items():
        describe(f'xprm_{m}_n16_mean', az(Zm.mean(1), o), np.ones(n, bool), f'{m} backbone N16 mean (unoriented)')
    # Reference: each existing N16 channel against the PRM + the other 15 + position + length
    ref = []
    for j in range(16):
        X = np.column_stack([zp, np.delete(Z, j, axis=1), zpos, zlen])
        oth = np.array([abs(np.corrcoef(Z[:, j], Z[:, i])[0, 1]) for i in range(16) if i != j])
        ref.append(dict(channel=N16[j], R2_rest=round(float(r2(Z[:, j], X)), 3),
                        max_abs_other_n16=round(float(oth.max()), 3), r_prm=round(float(np.corrcoef(Z[:, j], zp)[0, 1]), 3),
                        r_pos=round(float(np.corrcoef(Z[:, j], zpos)[0, 1]), 3)))
    # Same-PRM boundary entropy vs the existing H1_first_token of the NEXT step (expected near-duplicate)
    h1 = B['all'][:, B['names'].index('H1_first_token')].astype(float)
    h1_next = fill(np.r_[h1[1:], 0.], nxt_valid, o)
    be = fill(cands['boundary_entropy'][0], cands['boundary_entropy'][1], o)
    check = dict(corr_boundary_entropy_vs_next_H1_first_token=round(float(np.corrcoef(az(be, o), az(h1_next, o))[0, 1]), 3))
    with open(out / 'FEASIBILITY.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0])); w.writeheader(); w.writerows(rows_out)
    with open(out / 'N16_REFERENCE.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(ref[0])); w.writeheader(); w.writerows(ref)
    json.dump(dict(population=dict(answers=len(o) - 1, steps=int(n), benchmark='socratic', prm='qwen_prm_7b',
                                   labels_read=False, data='development'),
                   reflect_ids=None if rid is None else dict(n=int(len(rid)), tokenizer=str(sys.argv[3])),
                   checks=check, source='gdrive:hallucination_detection/cluster_results/prm_boost_external_v1/'),
              open(out / 'FEASIBILITY_META.json', 'w'), indent=1)
    for r in rows_out: print(r)
    print(check)
    for r in ref: print(r)


if __name__ == '__main__':
    main()
