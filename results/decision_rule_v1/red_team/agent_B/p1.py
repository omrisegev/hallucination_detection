exec(open('load.py').read())
lvp = MAIN/'.worktrees/token-probability-fusion-v1/results/token_probability_fusion_v1/DERIVATIVE_CHANNELS.npz'
lv = np.load(lvp); print(lv.files, {k: lv[k].shape for k in lv.files})
st = np.load(SSL/'results/expectation_realization_v1/run_20260927_stage_b/STEP_SCORES.npz'); print(st.files, {k: st[k].shape for k in st.files})
print(json.loads((SSL/'results/expectation_realization_v1/run_20260927_stage_b_thr/THRESHOLDS.json').read_text(encoding='utf8'))['S_equal'])
pb = ans.cell.str.startswith('pb_').to_numpy()
# PB label convention
aid = np.repeat(np.arange(n), ns)
tgt = ans.target.to_numpy()
bad=0; conv={'first_only':0,'from_first':0,'other':0,'none':0}
for i in np.flatnonzero(pb):
    l = labels[off[i]:off[i+1]]
    if tgt[i] < 0:
        conv['none'] += int(not l.any()); bad += int(l.any()); continue
    fo = np.zeros_like(l); fo[tgt[i]] = True
    ff = np.zeros_like(l); ff[tgt[i]:] = True
    if np.array_equal(l, fo): conv['first_only'] += 1
    elif np.array_equal(l, ff): conv['from_first'] += 1
    else: conv['other'] += 1
print('PB label convention', conv, 'correct-with-labels', bad)
print('PB prevalence', labels[pb[aid]].mean(), 'PRMB prevalence', labels[~pb[aid]].mean())
