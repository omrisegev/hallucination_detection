"""Compare independent per-answer records against Codex's frozen tables."""
import gzip, pickle, json
from pathlib import Path
import numpy as np, pandas as pd

OUT = Path(r'C:\Users\omris\TAU\hallucination_detection\results\monday_fusion_competitor_extension_v1\run_20261006')
SCR = Path(__file__).parent
R = pickle.load(open(SCR / 'records.pkl', 'rb')); rec = R['rec']
BANKS = ['B13', 'B16', 'B20', 'B23', 'B32', 'B35', 'B51', 'B54']
RULES = ['ANSWER_Z_Q80', 'R2_COUNT', 'ENTROPY_GATE_Q80']; ACTIONS = RULES + ['TOP_1', 'TOP_2', 'TOP_3']
METHODS = ['HEM_MOM_cc', 'MEAN_MOM_cc', 'MEAN_MOM_ga', 'TCI_ONLY_ga']
PRM = 'prmbench_qwen3_8b'
target, NS, groups, cells_all = R['target'], R['NS'], R['groups'], R['cells']
CELLS = sorted(set(cells_all)); PBC = [c for c in CELLS if c != PRM]
cat, cls = R['category'], R['classification']
print('argmax tolerance disagreements', R['argmax_tol_diff'], '| meta error_steps==labels', R['meta_error_steps_ok'])
print('max |TCI-run copy - panel score|', max(v for k, v in rec.items() if k[0] == '_copydiff'))
mtg_hit_of = dict(zip(R['pb_ans'], R['mtg_arg'] == target[R['pb_ans']]))


def ps(c):
    tp, fp, tn, fn = np.asarray(c, float)
    f1p = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else -1
    f1n = 2 * tn / (2 * tn + fp + fn) if (2 * tn + fp + fn) else -1
    return .5 * (f1p + f1n)


# ---------------- 1. METRICS.csv per row
M = pd.read_csv(OUT / 'METRICS.csv')
rows = []
for bk in BANKS:
    for cell in CELLS:
        for m in METHODS:
            for a in ACTIONS:
                d = rec[(bk, cell, m, a)]; t = target[d['ans']]
                r = {'bank': bk, 'cell': cell, 'method': m, 'decision_rule': a}
                if cell == PRM:
                    r['within_answer_auc'] = np.nanmean(d['auc']); r['prmscore'] = ps(d['conf'].sum(0))
                    err = d['ov'].sum(1) + 0 > -1  # placeholder
                    yerr = np.array([R['n_errors'][i] > 0 for i in d['ans']])
                    r['clean_flag_rate'] = np.mean(d['nflag'][~yerr] > 0)
                else:
                    err = t >= 0; hit = d['first'] == t
                    ae = hit[err].mean(); ac = hit[~err].mean()
                    r.update(pb_first_error_accuracy=(d['argmax'][err] == t[err]).mean(), pb_decision_error_accuracy=ae,
                             pb_official_f1=2 * ae * ac / (ae + ac) if ae + ac else 0., clean_flag_rate=np.mean(d['nflag'][~err] > 0))
                rows.append(r)
mine = pd.DataFrame(rows).set_index(['bank', 'cell', 'method', 'decision_rule'])
J = M.set_index(['bank', 'cell', 'method', 'decision_rule']).join(mine, rsuffix='_me')
for f in ['within_answer_auc', 'prmscore', 'pb_first_error_accuracy', 'pb_decision_error_accuracy', 'pb_official_f1', 'clean_flag_rate']:
    d = (J[f] - J[f + '_me']).abs(); nm = int((J[f].isna() != J[f + '_me'].isna()).sum())
    print(f'METRICS {f}: rows compared {int(d.notna().sum())}, max abs diff {d.max():.3g}, nan mismatch {nm}')

# report main table: bank-equal, PB cell-equal within bank
prm = mine.xs(PRM, level='cell').groupby(['method', 'decision_rule'])[['within_answer_auc', 'prmscore']].mean()
pbm = mine.drop(PRM, level='cell').groupby(['method', 'decision_rule'])[['pb_first_error_accuracy', 'pb_decision_error_accuracy', 'pb_official_f1', 'clean_flag_rate']].mean()
main = prm.join(pbm); main.to_csv(SCR / 'main_table_mine.csv')
print(main.round(4).to_string())

# ---------------- 2. F11 categories
C = pd.read_csv(OUT / 'F11_PRM_CATEGORY_METRICS.csv')
crow = []
for bk in BANKS:
    for m in METHODS:
        for a in ACTIONS:
            d = rec[(bk, PRM, m, a)]; cats = np.array([str(cat[i]) for i in d['ans']])
            for c in sorted(set(cats)):
                loc = cats == c; au = d['auc'][loc]; conf = d['conf'][loc].sum(0)
                crow.append({'bank': bk, 'category': c, 'method': m, 'rule': a, 'n_answers': int(loc.sum()), 'n_auc_answers': int(np.isfinite(au).sum()),
                             'within_answer_auc': np.nanmean(au) if np.isfinite(au).any() else np.nan, 'prmscore': ps(conf) if conf.sum() else np.nan})
cm = pd.DataFrame(crow).set_index(['bank', 'category', 'method', 'rule'])
Jc = C[C.method != 'SUPERVISED_PRM'].set_index(['bank', 'category', 'method', 'rule']).join(cm, rsuffix='_me', how='outer')
for f in ['n_answers', 'n_auc_answers', 'within_answer_auc', 'prmscore']:
    d = (Jc[f] - Jc[f + '_me']).abs(); print(f'F11 cat {f}: rows {len(Jc)}, max diff {d.max():.3g}, nan mismatch {int((Jc[f].isna()!=Jc[f+"_me"].isna()).sum())}')
rep = cm.reset_index(); rep = rep[(rep.rule == 'ANSWER_Z_Q80') & (rep.category != 'control')].groupby(['category', 'method'])[['within_answer_auc', 'prmscore', 'n_auc_answers']].mean()
print(rep.round(4).to_string())
# category n's, PRM answers by category/classification
print('category counts (answers):', pd.Series(list(cat.values())).value_counts().to_dict())
print('classification x category:', pd.crosstab(pd.Series(list(cat.values()), name='cat'), pd.Series(list(cls.values()), name='cls')).to_dict())
# supervised PRM rows sanity (from Codex table)
print(C[C.method == 'SUPERVISED_PRM'].groupby(['rule', 'category'])[['within_answer_auc', 'prmscore']].mean().round(4).to_string())

# ---------------- 3. overlap
OV = pd.read_csv(OUT / 'F11_PRM_OVERLAP_SUMMARY.csv')
orow = []
for bk in BANKS:
    for m in METHODS:
        for a in ACTIONS:
            d = rec[(bk, PRM, m, a)]
            df = pd.DataFrame(d['ov'], columns=['both', 'only_ours', 'only_PRM', 'neither'])
            df['category'] = [cat[i] for i in d['ans']]; df['classification'] = [cls[i] for i in d['ans']]
            g = df.groupby(['category', 'classification'], dropna=False).sum().reset_index(); g['bank'] = bk; g['method'] = m; g['rule'] = a
            orow.append(g)
om = pd.concat(orow).set_index(['bank', 'category', 'classification', 'method', 'rule'])
Jo = OV.set_index(['bank', 'category', 'classification', 'method', 'rule']).join(om, rsuffix='_me', how='outer')
for f in ['both', 'only_ours', 'only_PRM', 'neither']:
    print(f'overlap {f}: rows {len(Jo)} max diff {(Jo[f]-Jo[f+"_me"]).abs().max()}')
tot = om.reset_index(); print(tot[tot.rule == 'ANSWER_Z_Q80'].groupby('method')[['both', 'only_ours', 'only_PRM', 'neither']].sum().to_string())
print('total gold error steps per bank:', int(tot[(tot.rule == 'ANSWER_Z_Q80') & (tot.method == 'TCI_ONLY_ga') & (tot.bank == 'B13')][['both', 'only_ours', 'only_PRM', 'neither']].sum().sum()))
# supervised PRM alone recall
for m in METHODS:
    for a in ['ANSWER_Z_Q80', 'R2_COUNT', 'ENTROPY_GATE_Q80', 'TOP_1']:
        s = tot[(tot.method == m) & (tot.rule == a)][['both', 'only_ours', 'only_PRM', 'neither']].sum()
        n = s.sum(); print(f'{m} {a}: ours recall {(s.both+s.only_ours)/n:.4f} PRM recall {(s.both+s.only_PRM)/n:.4f} union {(s.both+s.only_ours+s.only_PRM)/n:.4f}')

# ---------------- 4. F12 breakdown
B = pd.read_csv(OUT / 'F12_MTG_BREAKDOWN.csv')
brow = []
for bk in BANKS:
    for m in METHODS:
        for a in ACTIONS:
            parts = []
            for cell in PBC:
                d = rec[(bk, cell, m, a)]; t = target[d['ans']]; ns = NS[d['ans']]
                parts.append(pd.DataFrame({'t': t, 'ns': ns, 'first': d['first'], 'argmax': d['argmax'], 'nflag': d['nflag'], 'mtg': [mtg_hit_of[i] for i in d['ans']]}))
            df = pd.concat(parts)
            df['len'] = pd.cut(df.ns, [0, 5, 10, 20, 10 ** 6], labels=['1-5', '6-10', '11-20', '21+']).astype(str)
            df['q'] = np.where(df.t < 0, 'clean', np.minimum(3, (4 * df.t / np.maximum(df.ns - 1, 1)).astype(int)).astype(str))
            df['ahit'] = df.argmax == df.t; df['dhit'] = df['first'] == df.t
            df['miss'] = np.select([df['first'] < 0, df['first'] < df.t, df['first'] > df.t], ['no_flag', 'early', 'late'], 'hit')
            e = df[df.t >= 0]
            for st, col in [('answer_length', 'len'), ('error_position_quartile', 'q')]:
                for v, g in e.groupby(col):
                    brow.append({'bank': bk, 'method': m, 'rule': a, 'stratum': st, 'value': str(v), 'n_answers': len(g), 'candidate_argmax_hit': g.ahit.mean(),
                                 'candidate_decision_hit': g.dhit.mean(), 'MTG_native_argmax_hit': g.mtg.mean(), 'early': (g.miss == 'early').sum(), 'late': (g.miss == 'late').sum(),
                                 'no_flag': (g.miss == 'no_flag').sum(), 'hit': (g.miss == 'hit').sum()})
            # NOTE: Codex's clean stratum is per cell; aggregate by cell here
            for cell, part in zip(PBC, parts):
                cl = part[part.t < 0]
                brow.append({'bank': bk, 'method': m, 'rule': a, 'stratum': 'clean_answers', 'value': cell, 'n_answers': len(cl), 'candidate_clean_flag_rate': (cl.nflag > 0).mean()})
bm = pd.DataFrame(brow); bm['value'] = bm.value.astype(str); B['value'] = B.value.astype(str)
Jb = B.set_index(['bank', 'method', 'rule', 'stratum', 'value']).join(bm.set_index(['bank', 'method', 'rule', 'stratum', 'value']), rsuffix='_me', how='outer')
for f in ['n_answers', 'candidate_argmax_hit', 'candidate_decision_hit', 'MTG_native_argmax_hit', 'early', 'late', 'no_flag', 'hit', 'candidate_clean_flag_rate']:
    d = (Jb[f] - Jb[f + '_me']).abs(); print(f'F12 {f}: rows {int(d.notna().sum())} max diff {d.max():.3g} nan mismatch {int((Jb[f].isna()!=Jb[f+"_me"].isna()).sum())}')
print('F12 rows', len(B), 'mine', len(bm))
# report F12 aggregate (R2, answer-weighted pooled over banks)
e = bm[(bm.rule == 'R2_COUNT') & (bm.stratum == 'error_position_quartile')]
for m, g in e.groupby('method'):
    n = g.n_answers.sum(); w = lambda c: (g[c] * g.n_answers).sum() / n
    print(f'F12 report {m}: argmax {w("candidate_argmax_hit"):.4f} MTG {w("MTG_native_argmax_hit"):.4f} firstflag {w("candidate_decision_hit"):.4f} early {g.early.sum()} late {g.late.sum()} noflag {g.no_flag.sum()} hit {g.hit.sum()} n {n}')
pickle.dump({'main': main}, open(SCR / 'compare_out.pkl', 'wb'))
