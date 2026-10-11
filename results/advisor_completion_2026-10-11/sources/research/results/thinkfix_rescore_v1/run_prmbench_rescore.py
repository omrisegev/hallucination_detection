"""thinkfix_rescore_v1: re-run the frozen Monday PRMBench/ProcessBench scorers on a rebuilt feature bank.

Usage (from the main checkout, Windows, same environment as the frozen runs):
    python results/thinkfix_rescore_v1/run_prmbench_rescore.py --variant old|thinkfix|thinkfix_mask1 \
        --bank <dir containing source/FEATURES.npz> [--stage dev|cont|both]

What changes relative to the frozen runs, and nothing else:
  * input bank: ext.FE points at --bank (rebuilt from raw telemetry by scripts/rebuild_source_bank_v4.py);
  * output dirs: results/thinkfix_rescore_v1/<variant>/{nested_banks,cont_baselines};
  * monday_cont_baselines_v1 reads its reference rows (ALL_EQ, the four Monday methods) from THIS variant's
    nested_banks run instead of the frozen v2 run, so old-vs-new tables compare like with like;
  * dev_run's B13/PRMBench recheck against the frozen candidate panel stays a HARD gate for variant=old
    (it proves the rebuilt bank + scorer replay the frozen numbers) and is RECORDED, not enforced, for the
    think-fix variants (whose inputs differ by design);
  * an export hook saves the PRMBench per-step scores (answer-z) per bank and method for the PRMScore page.
Each patched script is written next to its outputs (dev_run_patched.py, run_patched.py) for audit.
Labels/folds, method roster, seeds, bootstrap draws: unchanged (taken from the frozen scripts).
"""
from __future__ import annotations
import argparse, importlib.util, json, pickle, shutil, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
MAIN = HERE.parents[1]
V2 = MAIN / 'results/monday_nested_banks_v2'
CONT = MAIN / 'results/monday_cont_baselines_v1'
SPANS = HERE / 'STEP_TOKEN_SPANS_OLD.npz'   # overridden by --spans
EXPORT_METHODS = ['ALL_EQ', 'DS_FILTER_EQ', 'HEM_MOM_cc', 'MEAN_MOM_cc', 'MEAN_MOM_ga', 'TCI_ONLY_ga', 'HEM_MOM_ga',
                  'MEAN_EQ', 'HEM_EQ']


def patch(src: str, old: str, new: str) -> str:
    if src.count(old) != 1:
        raise SystemExit(f'patch anchor found {src.count(old)} times: {old[:80]!r}')
    return src.replace(old, new)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def answer_z(s, off):
    out = np.zeros(len(s))
    for a, b in zip(off[:-1], off[1:]):
        v = s[a:b]; sd = v.std()
        out[a:b] = (v - v.mean()) / sd if sd > 1e-12 else 0.
    return out


class Exporter:
    """Collect PRMBench per-step scores per bank (answer-z) + the global-standardized DS-survivor mean G."""

    def __init__(self, out):
        self.out = out; self.store = {}

    def dev(self, bk, ce, bank, sc, surv):
        if ce != 'prmbench_qwen3_8b':
            return
        d = self.store.setdefault(bk, {'off': bank['off'].copy(), 'answers': bank['answers'].copy()})
        for name in EXPORT_METHODS:
            if name in sc:
                d['S__' + name] = answer_z(np.asarray(sc[name], float), bank['off'])
        raw = np.asarray(bank['raw'][:, surv], float)               # DS survivors, raw (lf__ channels level-signed)
        mu, sd = raw.mean(0), raw.std(0); sd[sd < 1e-12] = 1.
        d['G'] = ((raw - mu) / sd).mean(1)                          # standardized once over ALL PRMBench steps
        d['G_survivors'] = np.asarray([bank['names'][j] for j in surv])

    def cont(self, bk, ce, bank, scores):
        if ce != 'prmbench_qwen3_8b':
            return
        d = self.store.setdefault(bk, {'off': bank['off'].copy(), 'answers': bank['answers'].copy()})
        for name in ('LSML_CONT', 'IU_PCR', 'LSML_CONT_LEVELSIGN', 'IU_PCR_LEVELSIGN', 'ALL_EQ_ORIENTED'):
            if name in scores:
                d['S__' + name] = answer_z(np.asarray(scores[name], float), bank['off'])

    def save(self, ids):
        self.out.mkdir(parents=True, exist_ok=True)
        for bk, d in self.store.items():
            dest = self.out / f'bank_{bk}.npz'
            prev = dict(np.load(dest, allow_pickle=False)) if dest.exists() else {}
            off = d['off']; idx = ids[d['answers']]
            if prev and (not np.array_equal(prev['offsets'], off) or list(prev['idx']) != list(idx)):
                raise SystemExit(f'export identity mismatch for {bk}')
            arrays = {**prev, 'offsets': off.astype(np.int64), 'idx': np.asarray(idx).astype(str),
                      'G_definition': np.asarray('G_def_equal_DS_survivors_global_z (not decision_rule_v1 replay): '
                                                 'mean over the bank\'s DS-filter survivors of each raw channel '
                                                 'standardized once over all PRMBench steps'),
                      'S_definition': np.asarray('S__<method>: the scorer\'s per-step score, answer-z (within each answer)'),
                      'DS_note': np.asarray('DS_SIGNED is not a scorer in these scripts; S__DS_FILTER_EQ = DS filter + equal mean')}
            arrays.update({k: v for k, v in d.items() if k not in ('off', 'answers')})
            np.savez_compressed(dest, **arrays)
        print('exported', sorted(self.store), 'to', self.out, flush=True)


def run_dev(variant, bank_dir, out, exporter):
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(V2 / 'PROTOCOL.json', out / 'PROTOCOL.json')
    src = (V2 / 'dev_run.py').read_text(encoding='utf8')
    src = patch(src, "    if recheck > 1e-9: raise SystemExit(f'recheck failed {recheck}')\n",
                "    if recheck > 1e-9 and _ENFORCE_RECHECK: raise SystemExit(f'recheck failed {recheck}')\n")
    src = patch(src, "'status': 'PASS', 'recheck_B13_PRMBench_MEAN_MOM_ga_max_abs_diff'",
                "'status': ('PASS' if recheck <= 1e-9 else 'RECORDED_NOT_ENFORCED'), 'variant': _VARIANT, "
                "'recheck_B13_PRMBench_MEAN_MOM_ga_max_abs_diff'")
    src = patch(src, "                sc, surv, d = ext.fit_cell(pop, bank, 1000 * bi + ci); diag[f'{bk}/{ce}'] = d\n",
                "                sc, surv, d = ext.fit_cell(pop, bank, 1000 * bi + ci); diag[f'{bk}/{ce}'] = d\n"
                "                _HOOK(bk, ce, bank, sc, surv)\n")
    # Two comparator inputs no longer exist on any machine (token-probability-fusion-v1 worktree is gone):
    # EVIDENCE_DROP.npz (Mind the Gap reference) -> its rows are not recomputed and are stripped below;
    # TOKEN_MATRICES.npz step_spans -> the same spans rebuilt from the old raw cells (STEP_TOKEN_SPANS_OLD.npz).
    src = patch(src, "    with np.load(fr['mindgap']['path']) as z: MTG = z['step_m5'].astype(float)\n",
                "    MTG = None  # thinkfix_rescore_v1: EVIDENCE_DROP.npz is gone; Mind the Gap rows are not recomputed\n")
    src = patch(src, "    with np.load(fr['tokens']['path']) as z: spans = z['step_spans']\n",
                "    with np.load(_SPANS) as z: spans = z['step_spans']\n")
    src = patch(src, "fill(R['MTG'], i, MTG[sl]); ", "")
    header = (f'# PATCHED COPY of results/monday_nested_banks_v2/dev_run.py for thinkfix_rescore_v1 (variant={variant}).\n'
              '# Changes: OUT = this directory; ext.FE set by the wrapper; recheck enforced only for variant=old;\n'
              '# export hook after each fit; Mind the Gap reference not recomputed (input gone); step spans from\n'
              '# STEP_TOKEN_SPANS_OLD.npz. Everything else is identical to the frozen script (see DIFF.txt).\n')
    (out / 'dev_run_patched.py').write_text(header + src, encoding='utf8')
    write_diff(V2 / 'dev_run.py', out / 'dev_run_patched.py', out / 'DIFF.txt')
    dev = load_module('dev_run_patched', out / 'dev_run_patched.py')
    dev._ENFORCE_RECHECK = variant == 'old'; dev._VARIANT = variant; dev._HOOK = exporter.dev; dev._SPANS = SPANS
    dev.ext.FE = bank_dir
    dev.main()
    strip_mtg(out)
    return dev


def write_diff(a, b, dest):
    import difflib
    d = difflib.unified_diff(a.read_text(encoding='utf8').splitlines(True), b.read_text(encoding='utf8').splitlines(True),
                             fromfile=str(a), tofile=str(b))
    dest.write_text(''.join(d), encoding='utf8')


def strip_mtg(out):
    """Remove the not-recomputed Mind the Gap rows/arrays so no empty comparator is ever read as a result."""
    import pandas as pd
    removed = {}
    for p in sorted(out.glob('*.csv')):
        df = pd.read_csv(p)
        bad = df.apply(lambda col: col.astype(str).str.contains('Mind the Gap')).any(axis=1)
        if bad.any():
            removed[p.name] = int(bad.sum()); df[~bad].to_csv(p, index=False)
    pa = out / 'PER_ANSWER.npz'
    if pa.exists():
        z = dict(np.load(pa, allow_pickle=True)); drop = [k for k in z if k.startswith('MTG__')]
        for k in drop: z.pop(k)
        np.savez_compressed(pa, **z); removed['PER_ANSWER.npz'] = drop
    (out / 'NOT_RECOMPUTED.json').write_text(json.dumps({
        'Mind the Gap (local repro.)': 'input EVIDENCE_DROP.npz no longer exists (token-probability-fusion-v1 worktree removed); rows stripped',
        'removed': removed}, indent=1), encoding='utf8')


def run_cont(variant, bank_dir, dev_out, out, exporter):
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(CONT / 'PROTOCOL.json', out / 'PROTOCOL.json')
    src = (CONT / 'run.py').read_text(encoding='utf8')
    src = patch(src, "OUT = Path(__file__).resolve().parent; MAIN = OUT.parents[1]; V2 = MAIN / 'results/monday_nested_banks_v2'\n",
                f"OUT = Path(__file__).resolve().parent; MAIN = Path(r'{MAIN}'); V2 = Path(r'{dev_out}')\n")
    src = patch(src, "spec = importlib.util.spec_from_file_location('dev_run', V2 / 'dev_run.py')",
                "spec = importlib.util.spec_from_file_location('dev_run', V2 / 'dev_run_patched.py')")
    src = patch(src, "            for nm, s in scores.items():\n                st = R[(bk, nm)]\n",
                "            _HOOK(bk, ce, bank, scores)\n            for nm, s in scores.items():\n                st = R[(bk, nm)]\n")
    header = (f'# PATCHED COPY of results/monday_cont_baselines_v1/run.py for thinkfix_rescore_v1 (variant={variant}).\n'
              '# Changes: V2 = this variant\'s nested_banks run; ext.FE set by the wrapper; export hook.\n')
    (out / 'run_patched.py').write_text(header + src, encoding='utf8')
    write_diff(CONT / 'run.py', out / 'run_patched.py', out / 'DIFF.txt')
    cont = load_module('cont_patched', out / 'run_patched.py')
    cont._HOOK = exporter.cont
    cont.dr._ENFORCE_RECHECK = variant == 'old'; cont.dr._VARIANT = variant; cont.dr._HOOK = lambda *a: None
    cont.ext.FE = bank_dir
    cont.main()
    return cont


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=['old', 'thinkfix', 'thinkfix_mask1'], required=True)
    ap.add_argument('--bank', type=Path, required=True)
    ap.add_argument('--stage', choices=['dev', 'cont', 'both'], default='both')
    ap.add_argument('--spans', type=Path, default=SPANS)
    ap.add_argument('--root', type=Path, default=HERE, help='output root (default: next to this script)')
    a = ap.parse_args()
    globals()['SPANS'] = a.spans
    if not (a.bank / 'source/FEATURES.npz').exists():
        raise SystemExit('no source/FEATURES.npz under ' + str(a.bank))
    root = a.root / a.variant
    exporter = Exporter(root / 'prmscore_inputs')
    meta_out = root / 'prmscore_inputs' / 'prmbench_meta.json'
    if not meta_out.exists():
        meta_out.parent.mkdir(parents=True, exist_ok=True)
        pm = pickle.load(open(MAIN / 'dataset_cache/four_localization/prmbench_qwen25math7b_full/prmbench_prm.pkl', 'rb'))
        meta_out.write_text(json.dumps({
            'source': 'dataset_cache/four_localization/prmbench_qwen25math7b_full/prmbench_prm.pkl (load_prmbench output, '
                      'hitsmy/PRMBench_Preview); error_steps are PRMBench raw 1-based step numbers',
            'rows': [{'idx': m['idx'], 'classification': m['classification'], 'category': m.get('category'),
                      'error_steps': [int(x) for x in m['error_steps']], 'n_steps': int(m['n_steps'])} for m in pm.values()]}, indent=0),
            encoding='utf8')
    t0 = time.time()
    if a.stage in ('dev', 'both'):
        dev = run_dev(a.variant, a.bank, root / 'nested_banks', exporter)
        exporter.save(np.asarray(dev.pd.read_csv(dev.SE / 'OOF_ANSWERS.csv', encoding='utf-8-sig').id.astype(str)))
        exporter.store = {}
    if a.stage in ('cont', 'both'):
        cont = run_cont(a.variant, a.bank, root / 'nested_banks', root / 'cont_baselines', exporter)
        exporter.save(np.asarray(cont.pd.read_csv(cont.dr.SE / 'OOF_ANSWERS.csv', encoding='utf-8-sig').id.astype(str)))
    (root / f'RESCORE_STATUS_{a.stage}.json').write_text(json.dumps({
        'variant': a.variant, 'bank': str(a.bank), 'stage': a.stage, 'runtime_s': round(time.time() - t0),
        'bank_manifest': json.loads((a.bank / 'MANIFEST.json').read_text()) if (a.bank / 'MANIFEST.json').exists() else None},
        indent=1, default=str), encoding='utf8')
    print('RESCORE DONE', a.variant, round(time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
