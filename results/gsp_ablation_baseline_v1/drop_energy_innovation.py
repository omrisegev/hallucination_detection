"""Drop energy_innovation from TELE16 (2026-11-12, Omri: 'if you remove it, how does it affect all benchmarks?').
Frozen GSP-T (gsp_v, no weight changes) on 16 vs 15 channels; R1 columns; all 4 benchmarks x 4 base models + Qwen3-8B PB.
Also L-SML + GAP and equal mean + GAP on 16 vs 15 as references. Output RESULTS_drop_ei.csv."""
import sys, runpy
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ablation as AB  # noqa: E402
import r0 as R0  # noqa: E402
import signed_weights as SW  # noqa: E402
T, E, N = AB.T, AB.E, AB.N
DROP = T.N16.index('energy_innovation')


def run(c, X, bench, prm, rows, pos=None):
    for tag, Xs in (('16', X), ('15 no EI', np.delete(X, DROP, 1))):
        tel = SW.gsp_v(c, Xs, False, False)
        arms = {f'GSP-T [{tag}]': tel, f'L-SML [{tag}]': dict(rank=tel['s'], flags=AB.gap(c, tel['s'])),
                f'mean [{tag}]': dict(rank=Xs.mean(1), flags=AB.gap(c, Xs.mean(1)))}
        for name, t in arms.items():
            if bench == 'processbench':
                for k, mk, v in R0.pb_extra(c, t, name, pos):
                    if 'CP' not in k and not ('own count' in k and 'GSP' not in k):
                        rows.append(dict(bench=bench, prm=prm, arm=k, metric=mk, value=v))
            else:
                fl = [(f'{name} + GAP', AB.gap(c, t['rank']))]
                if 'GSP' in name: fl.append((f'{name} own count', t['flags']))
                for k, f in fl:
                    for mk, mv in AB.score(c, bench, t['rank'], f).items():
                        rows.append(dict(bench=bench, prm=prm, arm=k, metric=mk, value=mv))


rows = []
for bench in AB.BENCHES:
    for prm in AB.PRMS:
        c, X, ab = AB.load_cell(bench, prm)
        run(c, X[:, 2:], bench, prm, rows); print(bench, prm, flush=True); del c
g = runpy.run_path(str(HERE / 'qwen3_8b_pb.py'))
run(g['c'], g['X'], 'processbench', 'qwen3_8b_mtg', rows, g['pos'])
E.write_csv(HERE / 'RESULTS_drop_ei.csv', rows)
