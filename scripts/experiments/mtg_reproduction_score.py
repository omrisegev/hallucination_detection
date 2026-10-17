"""mtg_reproduction_v1, stage 2: score the pre-declared grid and compare with Chen et al.'s Table 3.
Frozen protocol: results/mtg_reproduction_v1/PROTOCOL.json. Labels enter evaluation only.
"""
from pathlib import Path
import hashlib, itertools, json, subprocess, sys, time
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.signal import lfilter

MAIN = Path(r'C:\Users\omris\TAU\hallucination_detection'); ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/mtg_reproduction_v1'; P = json.loads((OUT / 'PROTOCOL.json').read_text(encoding='utf8'))
TPF = MAIN / '.worktrees/token-probability-fusion-v1'
VEND = ROOT / 'papers/code/mind_the_gap_evidence_drop_ff14a7d/uncertainty_estimation'
t0 = time.time()
def dump(p, v): Path(p).write_text(json.dumps(v, indent=1, default=lambda x: x.item() if isinstance(x, np.generic) else x.tolist() if isinstance(x, np.ndarray) else str(x)), encoding='utf8')
dump(OUT / 'RUN_STATUS.json', {'status': 'RUNNING', 'started': datetime.now().isoformat(timespec='seconds')})

Z = np.load(OUT / 'TOKEN_SIGNALS.npz', allow_pickle=False)
toff, soff, spans, target = Z['token_offsets'], Z['step_offsets'], Z['step_spans'], Z['target']
cells = Z['cells'].astype(str); n = len(target)
SIG = ['shannon', 'logtoku', 'lns_top1', 'lns_provided']; X = {s: Z[s] for s in SIG}
SPAN, ALPHA = 5.0, 2.0 / (5.0 + 1.0)

def ema(v):
    return lfilter([ALPHA], [1.0, -(1.0 - ALPHA)], v, zi=[(1.0 - ALPHA) * v[0]])[0]   # out[0] = v[0], alpha = 2/(span+1)

def transformed(x, T):
    base = x if T == 'paper' else np.cumsum(x) / np.arange(1, len(x) + 1)
    return ema(base)

# ------------------------------------------------------------ replay check 2: our 'code' transform == their released function
sys.path.insert(0, str(VEND))
from utils.metrics import calculate_risk_with_running_mean_drop as their_risk  # noqa: E402
def our_seq_risk(x, T='code', k=5):
    r = -np.diff(transformed(np.asarray(x, float), T)); r = np.sort(r[r > 0])[::-1][:k]
    return float(r.mean()) if len(r) else 0.0
rng = np.random.default_rng(0); rep2 = []
for i in list(range(0, n, 397))[:15]:
    x = X['lns_top1'][toff[i]:toff[i + 1]]; rep2.append(abs(our_seq_risk(x) - their_risk(list(x), ema_span=5, drop_k=5)))
for _ in range(10):
    x = rng.normal(size=rng.integers(50, 800)); rep2.append(abs(our_seq_risk(x) - their_risk(list(x), ema_span=5, drop_k=5)))
REPLAY_CODE = float(max(rep2)); assert REPLAY_CODE < 1e-12, REPLAY_CODE

# ------------------------------------------------------------ step scores for every configuration
CFG = [('avg', None, None, None)] + [('drop', T, A, G) for T in ('paper', 'code') for A in ('later', 'earlier') for G in ('max', 'm5', 'sum')]
cfg_names = [f"{fam}" if fam == 'avg' else f"drop|{T}|{A}|{G}" for fam, T, A, G in CFG]
cols = [(s, c) for s in SIG for c in cfg_names]
SC = np.zeros((int(soff[-1]), len(cols)))
for i in range(n):
    ta, tb = toff[i], toff[i + 1]; a, b = soff[i], soff[i + 1]; sp = spans[a:b]
    for si, s in enumerate(SIG):
        x = X[s][ta:tb]
        base_c = si * len(cfg_names)
        SC[a:b, base_c] = [-x[p:q].mean() for p, q in sp]                               # Avg: minus mean evidence in the step
        for T in ('paper', 'code'):
            d = -np.diff(transformed(x, T))                                            # risk flux, positive = evidence drop
            for A in ('later', 'earlier'):
                r = np.concatenate([[0.0], d]) if A == 'later' else np.concatenate([d, [0.0]])
                o_max = np.zeros(len(sp)); o_m5 = np.zeros(len(sp)); o_sum = np.zeros(len(sp))
                for k_, (p, q) in enumerate(sp):
                    seg = r[p:q]
                    if not len(seg): continue
                    kk = min(5, len(seg)); top = np.partition(seg, len(seg) - kk)[-kk:]
                    o_max[k_] = top.max(); o_m5[k_] = top.mean(); o_sum[k_] = seg[seg > 0].sum()
                SC[a:b, base_c + cfg_names.index(f'drop|{T}|{A}|max')] = o_max
                SC[a:b, base_c + cfg_names.index(f'drop|{T}|{A}|m5')] = o_m5
                SC[a:b, base_c + cfg_names.index(f'drop|{T}|{A}|sum')] = o_sum
    if i % 1000 == 0: print(f'  scores {i}/{n} {time.time()-t0:.0f}s', flush=True)

# ------------------------------------------------------------ replay check 1: EVIDENCE_DROP step_m5 / step_worst
ED = np.load(TPF / 'results/token_probability_fusion_v1/EVIDENCE_DROP.npz')
J = np.load(TPF / 'results/localization_full_benchmark_v3/evaluation/JOINED.npz', allow_pickle=False); roff = np.asarray(J['offsets'], int)
recs = json.load(open(TPF / 'results/localization_full_benchmark_v3/evaluation/JOINED.json', encoding='utf8'))['records']
uid2r = {r['uid']: k for k, r in enumerate(recs)}; uids = Z['uids'].astype(str)
jm5 = cols.index(('shannon', 'drop|paper|later|m5')); jmx = cols.index(('shannon', 'drop|paper|later|max'))
e1 = e2 = 0.0
for i in range(n):
    r_ = uid2r[uids[i]]; ra, rb = roff[r_], roff[r_ + 1]; a, b = soff[i], soff[i + 1]
    e1 = max(e1, float(np.abs(SC[a:b, jm5] - ED['step_m5'][ra:rb]).max())); e2 = max(e2, float(np.abs(SC[a:b, jmx] - ED['step_worst'][ra:rb]).max()))
REPLAY_ED = {'step_m5_max_abs_diff': e1, 'step_worst_max_abs_diff': e2}; print('replay EVIDENCE_DROP:', REPLAY_ED, '| replay released code:', REPLAY_CODE, flush=True)
assert e1 < 1e-5 and e2 < 1e-5, REPLAY_ED

# ------------------------------------------------------------ decisions and SLA
CELLS = [f'pb_{d}_q{m}' for m in (4, 8) for d in ('gsm8k', 'math', 'olympiadbench', 'omnimath')]
DEC = [('argmax', None)] + [('first_z', c) for c in (0.0, 0.5, 1.0, 1.5, 2.0)] + [('first_q', q) for q in (0.5, 0.7, 0.8, 0.9, 0.95)]
cell_of = {c: np.flatnonzero(cells == c) for c in CELLS}
thr = {}
for c in CELLS:
    rows = np.concatenate([np.arange(soff[i], soff[i + 1]) for i in cell_of[c]])
    for _, q in [d for d in DEC if d[0] == 'first_q']: thr[(c, q)] = np.quantile(SC[rows], q, axis=0)   # all traces: label-free
recs_out = []
for c in CELLS:
    err = [i for i in cell_of[c] if target[i] >= 0]
    hit = {d: np.zeros(len(cols)) for d in DEC}; fired = {d: np.zeros(len(cols)) for d in DEC}
    for i in err:
        V = SC[soff[i]:soff[i + 1]]; t = target[i]; ns = len(V)
        for d in DEC:
            if d[0] == 'argmax':
                pred = np.argmax(V >= V.max(0) - 1e-12, axis=0); ok = np.ones(len(cols), bool)
            else:
                if d[0] == 'first_z':
                    sd = V.std(0); zz = np.where(sd > 1e-12, (V - V.mean(0)) / np.where(sd > 1e-12, sd, 1), 0.0); M_ = zz > d[1]
                else:
                    M_ = V > thr[(c, d[1])]
                ok = M_.any(0); pred = np.where(ok, np.argmax(M_, axis=0), -1)
            hit[d] += (pred == t); fired[d] += ok
    for d in DEC:
        for j, (s, cfg) in enumerate(cols):
            recs_out.append({'cell': c, 'signal': s, 'config': cfg, 'decision': d[0], 'param': d[1], 'dec': d[0] if d[1] is None else f'{d[0]}{d[1]}', 'n_err': len(err),
                             'hits': int(hit[d][j]), 'fired': int(fired[d][j]),
                             'sla_all': 100 * hit[d][j] / len(err), 'sla_fired': 100 * hit[d][j] / max(fired[d][j], 1)})
G = pd.DataFrame(recs_out); G.to_csv(OUT / 'GRID.csv', index=False)
print(f'grid: {len(G):,} rows  {time.time()-t0:.0f}s', flush=True)

# ------------------------------------------------------------ matching against Table 3
TT = P['target_table3']; COLS6 = TT['columns']
tgt = {(f"pb_{d}_q{4 if m == 'Qwen3-4B' else 8}", col): v[k] for m in ('Qwen3-4B', 'Qwen3-8B') for d, v in TT[m].items() for k, col in enumerate(COLS6)}
SIG_OF = {'LogTokU': ['logtoku'], 'Shannon': ['shannon'], 'LN-S': ['lns_top1', 'lns_provided']}
def col_values(sig, fam_cfg, dec, den):
    g = G[(G.signal == sig) & (G.config == fam_cfg) & (G.dec == dec)].set_index('cell')
    return np.array([g.loc[c, 'sla_' + den] for c in CELLS])
per_col = []
for col in COLS6:
    meth, kind = col.split(' ')
    tv = np.array([tgt[(c, col)] for c in CELLS])
    for sig in SIG_OF[meth]:
        cfgs = ['avg'] if kind == 'Avg' else [x for x in cfg_names if x != 'avg']
        for cfg in cfgs:
            for dec in G.dec.unique():
                for den in ('all', 'fired'):
                    if dec == 'argmax' and den == 'fired': continue
                    v = col_values(sig, cfg, dec, den)
                    per_col.append({'column': col, 'signal': sig, 'config': cfg, 'decision': dec, 'denominator': den, 'mad': float(np.abs(v - tv).mean()),
                                    'mean_ours': float(v.mean()), 'mean_theirs': float(tv.mean()), 'max_abs_dev': float(np.abs(v - tv).max()), **{f'sla_{c}': round(float(x), 2) for c, x in zip(CELLS, v)}})
PC = pd.DataFrame(per_col).sort_values(['column', 'mad']); PC.to_csv(OUT / 'MATCH_PER_COLUMN.csv', index=False)
# one shared configuration for all six columns
drop_cfgs = [x for x in cfg_names if x != 'avg']; shared = []
for dcfg, dec, den, lns in itertools.product(drop_cfgs, G.dec.unique(), ('all', 'fired'), ('lns_top1', 'lns_provided')):
    if dec == 'argmax' and den == 'fired': continue
    rowd = {'drop_config': dcfg, 'decision': dec, 'denominator': den, 'lns_reading': lns}; mads = []
    vals = {}
    for col in COLS6:
        meth, kind = col.split(' '); sig = lns if meth == 'LN-S' else SIG_OF[meth][0]
        v = col_values(sig, 'avg' if kind == 'Avg' else dcfg, dec, den); tv = np.array([tgt[(c, col)] for c in CELLS])
        mads.append(float(np.abs(v - tv).mean())); rowd['mad_' + col] = mads[-1]; vals[col] = v
    rowd['mean_mad'] = float(np.mean(mads))
    rowd['shannon_drop_gt_avg_cells'] = int((vals['Shannon Drop'] > vals['Shannon Avg']).sum())
    six = np.stack([vals[c] for c in COLS6]); rowd['shannon_drop_best_cells'] = int((six.argmax(0) == COLS6.index('Shannon Drop')).sum())
    rowd['shannon_drop_mean'] = float(vals['Shannon Drop'].mean()); rowd['shannon_avg_mean'] = float(vals['Shannon Avg'].mean())
    shared.append(rowd)
SH = pd.DataFrame(shared).sort_values('mean_mad'); SH.to_csv(OUT / 'MATCH_SHARED.csv', index=False)
# summaries
best_sd = PC[PC.column == 'Shannon Drop'].iloc[0]
max_sd_mean = G[(G.signal == 'shannon') & (G.config != 'avg')].groupby(['config', 'dec']).apply(lambda g: pd.Series({'all': g.sla_all.mean(), 'fired': g.sla_fired.mean()}))
den_effect = G[(G.decision != 'argmax')].assign(gap=lambda d: d.sla_fired - d.sla_all).gap.describe().to_dict()
S_ = {'replay_evidence_drop': REPLAY_ED, 'replay_released_code': REPLAY_CODE,
      'best_shared': SH.iloc[0].to_dict(), 'best_shared_top5': SH.head(5).to_dict('records'),
      'best_per_column': {col: PC[PC.column == col].iloc[0].to_dict() for col in COLS6},
      'shannon_drop_best_match': best_sd.to_dict(),
      'shannon_drop_highest_8cell_mean': {'all_denominator': float(max_sd_mean['all'].max()), 'fired_denominator': float(max_sd_mean['fired'].max()),
                                          'config_all': str(max_sd_mean['all'].idxmax()), 'config_fired': str(max_sd_mean['fired'].idxmax())},
      'paper_shannon_drop_8cell_mean': float(np.mean([tgt[(c, 'Shannon Drop')] for c in CELLS])),
      'denominator_effect_sla_points': den_effect,
      'transform_preference_shannon_drop': PC[PC.column == 'Shannon Drop'].groupby(PC.config.str.split('|').str[1]).mad.min().to_dict()}
dump(OUT / 'SUMMARY.json', S_)
code = {rel: hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() for rel in ['scripts/experiments/mtg_reproduction_extract.py', 'scripts/experiments/mtg_reproduction_score.py', 'results/mtg_reproduction_v1/PROTOCOL.json']}
git = lambda *a: subprocess.run(['git', *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
dump(OUT / 'CODE_MANIFEST.json', {'files': code, 'git_head': git('rev-parse', 'HEAD'), 'vendored_code_commit': 'ff14a7d5bd0f3969b47555d0daa9a0c8b9dbb71f'})
dump(OUT / 'RUN_STATUS.json', {'status': 'COMPLETE', 'finished': datetime.now().isoformat(timespec='seconds'), 'seconds': round(time.time() - t0, 1), 'grid_rows': len(G), 'shared_configs': len(SH)})
pd.set_option('display.width', 220)
print('\nBEST SHARED CONFIG (one pipeline for all six columns):'); print(SH.head(5)[['drop_config', 'decision', 'denominator', 'lns_reading', 'mean_mad', 'shannon_drop_mean', 'shannon_avg_mean', 'shannon_drop_gt_avg_cells', 'shannon_drop_best_cells']].round(2).to_string(index=False))
print('\nBEST PER COLUMN:'); print(pd.DataFrame([{**{'column': c}, **{k: S_['best_per_column'][c][k] for k in ['signal', 'config', 'decision', 'denominator', 'mad', 'mean_ours', 'mean_theirs', 'max_abs_dev']}} for c in COLS6]).round(2).to_string(index=False))
print('\nShannon Drop highest 8-cell mean:', S_['shannon_drop_highest_8cell_mean'], '| paper', round(S_['paper_shannon_drop_8cell_mean'], 2))
print('transform preference (min MAD, Shannon Drop):', S_['transform_preference_shannon_drop'])
print(f'done {time.time()-t0:.0f}s')
