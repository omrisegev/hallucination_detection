---
description: Generate a correct Colab notebook cell for analysis or inference with the three-branch pkl reload + Drive save pattern. Use whenever adding any cell that computes and caches results. Accepts parameters: VAR (variable name), PKL (filename), TYPE (nadler|feats|samples|custom), INCREMENTAL (yes|no).
---

Generate a complete, copy-paste-ready Colab cell based on the parameters provided. If parameters are not specified, ask for them before generating.

## Parameters
- **VAR**: Python variable name for the result (e.g., `MATH500_RES`, `RAG_FEATS`, `P1_SAMPLES`)
- **PKL**: Pickle filename (e.g., `math500_res.pkl`) — will be placed in `OUT_DIR` or `CACHE_DIR`
- **TYPE**: What the cell computes:
  - `nadler` — runs `run_nadler()` per key, result is dict of Nadler result dicts
  - `feats` — runs `extract_feats()` per key, result is dict of (feats_dict, labels) tuples
  - `samples` — inference loop generating K samples per item (SE/SC baselines)
  - `custom` — user provides the computation block
- **INCREMENTAL**: `yes` = save after every key (use for slow loops > 2 min); `no` = save at end

## Output format

Emit a Python code block. Always include:

1. `_valid_res()` helper (unless it was defined earlier in the notebook — check context)
2. Three-branch skip logic with partial-resume support when INCREMENTAL=yes
3. The computation loop with the correct pattern for TYPE
4. A Drive save (after each key if INCREMENTAL=yes, or at end if no)
5. A brief print statement after each key showing progress

---

### Template: TYPE=nadler, INCREMENTAL=no
```python
RES_PATH = os.path.join(OUT_DIR, '<PKL>')
FORCE = False

def _valid_res(res): return bool(res) and any(v for v in res.values() if v)

_skip = False
if not FORCE and '<VAR>' in globals() and _valid_res(<VAR>) and len(<VAR>) == len(DATA):
    print(f'in memory — {len(<VAR>)} results'); _skip = True
elif not FORCE and os.path.exists(RES_PATH):
    with open(RES_PATH, 'rb') as _f: _s = pickle.load(_f)
    _r = _s.get('results', _s)
    if _valid_res(_r):
        <VAR> = _r; print(f'loaded {len(<VAR>)} results'); _skip = True
    else:
        print('stale pkl — recomputing')

if not _skip:
    FEATS, <VAR> = {}, {}
    for key, samps in DATA.items():
        print(f'\n[<VAR> / {key}]')
        fd, lbl = extract_feats(samps, use_adaptive_window=False)
        FEATS[key] = (fd, lbl)
        <VAR>[key] = run_nadler(fd, lbl, key)
    with open(RES_PATH, 'wb') as _f:
        pickle.dump({'results': <VAR>, 'feats': FEATS}, _f)
    print(f'saved {len(<VAR>)} results to {RES_PATH}')
```

---

### Template: TYPE=nadler, INCREMENTAL=yes (for slow 10+ item loops)
```python
RES_PATH = os.path.join(OUT_DIR, '<PKL>')
FORCE = False

def _valid_res(res): return bool(res) and any(v for v in res.values() if v)

_skip = False
if not FORCE and '<VAR>' in globals() and _valid_res(<VAR>) and len(<VAR>) == len(DATA):
    print(f'in memory — all {len(<VAR>)} done'); _skip = True
elif not FORCE and os.path.exists(RES_PATH):
    with open(RES_PATH, 'rb') as _f: _rr = pickle.load(_f)
    if _valid_res(_rr):
        <VAR> = _rr
        if len(<VAR>) == len(DATA):
            print(f'loaded all {len(<VAR>)} from disk'); _skip = True
        else:
            print(f'partial on disk ({len(<VAR>)}/{len(DATA)}) — resuming')
    else:
        print('stale pkl — recomputing'); <VAR> = {}
else:
    <VAR> = {}

if not _skip:
    if '<VAR>' not in globals(): <VAR> = {}
    for key, (fd, lbl) in DATA.items():
        if key in <VAR> and <VAR>[key] is not None:
            print(f'  [cached] {key}'); continue
        <VAR>[key] = run_nadler(fd, lbl, key)
        with open(RES_PATH, 'wb') as _f: pickle.dump(<VAR>, _f)
        print(f'  → checkpoint ({len(<VAR>)}/{len(DATA)})')
    print(f'All {len(<VAR>)} done — {RES_PATH}')
```

---

### Template: TYPE=feats
```python
FEAT_PATH = os.path.join(OUT_DIR, '<PKL>')
FORCE_FEATS = False

_skip = False
if not FORCE_FEATS and '<VAR>' in globals() and len(<VAR>) == len(DATA) and <VAR>:
    print(f'<VAR> in memory ({len(<VAR>)} cells)'); _skip = True
elif not FORCE_FEATS and os.path.exists(FEAT_PATH):
    with open(FEAT_PATH, 'rb') as _f: _rf = pickle.load(_f)
    if len(_rf) == len(DATA) and _rf:
        <VAR> = _rf; print(f'loaded <VAR> ({len(<VAR>)} cells)'); _skip = True
    else:
        print('stale/partial feats pkl — recomputing')

if not _skip:
    <VAR> = {}
    for key, samps in tqdm(DATA.items(), desc='Extracting features'):
        fd, lbl = extract_feats(samps, use_adaptive_window=False)
        <VAR>[key] = (fd, lbl)
    with open(FEAT_PATH, 'wb') as _f: pickle.dump(<VAR>, _f)
    print(f'saved <VAR> ({len(<VAR>)} cells) to {FEAT_PATH}')
```

---

### Template: TYPE=samples (inference for SE/SC baselines)
```python
SAMPLES_PATH = os.path.join(CACHE_DIR, '<PKL>')
FORCE_SAMPLES = False

_skip = False
if not FORCE_SAMPLES and '<VAR>' in globals() and _p12_valid(<VAR>):
    print(f'<VAR> in memory ({len(<VAR>)} items)'); _skip = True
elif not FORCE_SAMPLES and os.path.exists(SAMPLES_PATH):
    with open(SAMPLES_PATH, 'rb') as _f: _cached = pickle.load(_f)
    if _p12_valid(_cached) and len(_cached) == len(ITEMS):
        <VAR> = _cached; print(f'loaded {len(<VAR>)} items'); _skip = True
    else:
        print(f'stale/partial — recomputing ({len(_cached)}/{len(ITEMS)} valid)')
        <VAR> = {k: v for k, v in _cached.items() if _p12_valid({k: v})}

if not _skip:
    if '<VAR>' not in globals(): <VAR> = {}
    for i, item in enumerate(ITEMS):
        if i in <VAR> and <VAR>[i].get('done'): continue
        # ... generate K samples ...
        <VAR>[i] = {'samples': samples, 'label': label, 'done': True}
        if i % 25 == 0:
            with open(SAMPLES_PATH, 'wb') as _f: pickle.dump(<VAR>, _f)
            print(f'  checkpoint {i}/{len(ITEMS)}')
    with open(SAMPLES_PATH, 'wb') as _f: pickle.dump(<VAR>, _f)
    print(f'saved {len(<VAR>)} items to {SAMPLES_PATH}')
```

---

After generating the cell, remind the user:
- Replace `DATA` with the actual data dict variable name
- Replace `OUT_DIR` / `CACHE_DIR` with the correct path variable
- If `_valid_res` is already defined earlier in the notebook, remove the duplicate definition
- If TYPE=nadler, ensure `run_nadler()` and `extract_feats()` are defined in a prior cell
