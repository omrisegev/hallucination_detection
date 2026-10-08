"""External label-free scoring of the nested banks (Amendment A1: failures recorded, no fallback).

Same steps as monday_external_panel_v1/score.py score_external(), using its validated Population / fit_cell /
decisions; requires HIGHLIGHT.json so the highlight is fixed before any external result. Never opens labels."""
from __future__ import annotations
import hashlib, importlib.util, json, time
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent; EXT = OUT.parent / 'monday_external_panel_v1'
spec = importlib.util.spec_from_file_location('ext_score', EXT / 'score.py'); ext = importlib.util.module_from_spec(spec); spec.loader.exec_module(ext)
NB = json.loads((OUT / 'BANKS.json').read_text(encoding='utf8')); BANKS = list(NB); ext.BN = NB


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    if not (OUT / 'HIGHLIGHT.json').exists(): raise SystemExit('development highlight not decided yet')
    if (OUT / 'SCORING_STATUS.json').exists(): raise SystemExit('already scored')
    if json.loads((OUT / 'PROTOCOL.json').read_text(encoding='utf8'))['status'] != 'FROZEN': raise SystemExit('protocol not frozen')
    status = {'started': time.strftime('%Y-%m-%dT%H:%M:%S'), 'protocol_sha256': sha(OUT / 'PROTOCOL.json'), 'amendment_A1_sha256': sha(OUT / 'AMENDMENT_A1.json'),
              'highlight_sha256': sha(OUT / 'HIGHLIGHT.json'), 'score_script_sha256': sha(EXT / 'score.py'), 'wrapper_sha256': sha(Path(__file__)),
              'external_labels_opened': False, 'source_parity_gate_sha256': sha(OUT / 'SOURCE_PARITY_GATE.json'), 'cells': {}, 'failures': []}
    for ci, (cell, bench) in enumerate(ext.CELLS.items()):
        tc = time.perf_counter(); F = np.load(ext.FE / cell / 'FEATURES.npz'); uids = [str(u) for u in F['uids']]
        lens = F['nonempty_lengths']; ne = F['nonempty'].astype(bool); starts = np.r_[0, np.cumsum(lens)]
        public = {r['uid']: r for r in json.loads((ext.INPUTS / bench / 'answers.json').read_text(encoding='utf8'))}
        if set(public) != set(uids): raise SystemExit(cell + ': population differs')
        pop = ext.Population(F, BANKS); off = pop.off
        for i, u in enumerate(uids):
            if len(public[u]['steps']) != lens[i] or int(ne[starts[i]:starts[i + 1]].sum()) != off[i + 1] - off[i]: raise SystemExit(cell + ': step alignment ' + u)
        answers = np.arange(len(uids)); names_sc, names_pr, SC, PR, diag = [], [], [], [], {}
        for bi, bk in enumerate(BANKS):
            bank = pop.cell_bank(bk, answers)
            try:
                sc, surv, d = ext.fit_cell(pop, bank, 1000 * bi + ci)
            except RuntimeError as exc:
                if 'groups before merge: minimum is 3' not in str(exc) and 'Fewer than 3 groups' not in str(exc): raise   # A2
                st0 = ext.fit_content(pop, bank, recipes=[]); sc = {k: st0['scores'][k] for k in ('ALL_EQ', 'DS_FILTER_EQ') if k in st0['scores']}
                surv = st0['surv']; d = {'FAILED': str(exc)[:300]}
                status['failures'].append({'cell': cell, 'bank': bk, 'reason': str(exc)[:300], 'computed': sorted(sc)})
            diag[bk] = d
            for s in ext.SCORERS:
                if s not in sc: continue
                if not np.isfinite(sc[s]).all(): raise SystemExit(f'{cell} {bk} {s} nonfinite')
                names_sc.append(f'{bk}__{s}'); SC.append(sc[s])
                dd, _ = ext.decisions(sc[s], bank, surv)
                for r in ext.RULES: names_pr.append(f'{bk}__{s}__{r}'); PR.append(dd[r])
            print(f'  {cell} {bk}: {"FAILED " if "FAILED" in d else ""}scorers {sum(1 for s in ext.SCORERS if s in sc)}', flush=True)
        pos = (np.arange(pop.S) - np.repeat(off[:-1], np.diff(off))).astype(float)
        names_sc.append('STEP_INDEX'); SC.append(pos)
        az = np.zeros(pop.S)
        for a, b in zip(off[:-1], off[1:]): v = pos[a:b]; az[a:b] = (v - v.mean()) / max(v.std(), 1e-8)
        names_pr.append('STEP_INDEX__ANSWER_Z_Q80'); PR.append(az >= np.quantile(az, .8))
        names_pr.append('STEP_INDEX__TOP_1'); PR.append(ext.top_flags(pos, off, np.ones(len(uids), int)))
        full_n = int(lens.sum()); idx = np.flatnonzero(ne)
        SCf = np.full((full_n, len(SC)), np.nan); SCf[idx] = np.column_stack(SC)
        PRf = np.zeros((full_n, len(PR)), np.int8); PRf[idx] = (~np.column_stack(PR)).astype(np.int8)
        oldmap = {json.loads(p.read_text(encoding='utf8'))['uid']: p for p in (ext.OLD / cell / 'shard_000').glob('*.record.json')}
        RS = np.full((full_n, len(ext.REFS)), np.nan); RP = np.zeros((full_n, len(ext.REFS)), np.int8); tele = [str(t) for t in F['telemetry_sha256']]
        for i, u in enumerate(uids):
            o = json.loads(oldmap[u].read_text(encoding='utf8'))
            if o['payload']['telemetry_sha256'] != tele[i] or o['payload']['nonempty'] != ne[starts[i]:starts[i + 1]].tolist(): raise SystemExit(cell + ': reference identity ' + u)
            for k, r in enumerate(ext.REFS):
                RS[starts[i]:starts[i + 1], k] = [np.nan if v is None else v for v in o['payload']['scores'][r]]; RP[starts[i]:starts[i + 1], k] = o['payload']['predictions'][r]
        (OUT / cell).mkdir(exist_ok=True)
        np.savez_compressed(OUT / cell / 'PREDICTIONS.npz', uids=np.array(uids), starts=starts, nonempty=ne, scores=SCf, score_names=np.array(names_sc),
                            predictions=PRf, prediction_names=np.array(names_pr), ref_scores=RS, ref_predictions=RP, ref_names=np.array(ext.REFS))
        (OUT / cell / 'SCORING_DIAGNOSTICS.json').write_text(json.dumps(diag, indent=1, default=lambda v: v.item() if isinstance(v, np.generic) else str(v)), encoding='utf8')
        status['cells'][cell] = {'answers': len(uids), 'steps': full_n, 'scores': len(names_sc), 'predictions': len(names_pr),
                                 'predictions_sha256': sha(OUT / cell / 'PREDICTIONS.npz'), 'seconds': round(time.perf_counter() - tc, 1)}
        print(cell, 'scored', status['cells'][cell]['seconds'], flush=True)
    status['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    (OUT / 'SCORING_STATUS.json').write_text(json.dumps(status, indent=1), encoding='utf8')


if __name__ == '__main__':
    main()
