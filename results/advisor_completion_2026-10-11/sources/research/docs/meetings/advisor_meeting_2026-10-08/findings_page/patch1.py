from pathlib import Path
p = Path(__file__).parent / 'template.html'; s = p.read_text(encoding='utf8')
def R(a, b):
    global s
    assert a in s, 'MISSING: ' + a[:70]
    s = s.replace(a, b)

R('<a href="#pb">ProcessBench</a>', '<a href="#defs">Methods and metrics</a><a href="#signals">Signals</a><a href="#pb">ProcessBench</a>')
R('<a href="#types">PRMBench error types</a>', '<a href="#types">PRMBench error types</a><a href="#lit">Published competitors</a>')
R('<p class="lede">One teacher-forced pass of Qwen3 (4B/8B) over a given answer, token-probability channels per step, fused without labels.',
  '<p class="lede">The task: find the wrong steps in a given answer, with no labels and no training. We do one teacher-forced pass with Qwen3-4B or Qwen3-8B. We read token-probability channels for each step. Then we fuse the channels without labels.')
R("""<div><b>Q1 · Fusion vs plain average</b>Fusion saturates. On PRMBench all label-free rules land close together (pipeline ties the DS-signed average 15/15 of 36 method×bank pairs). On ProcessBench the plain average (0.414 SLA, N16) beats all four pipeline variants (0.396–0.408).</div>""",
  """<div><b>Q1 · Fusion vs plain average</b>Fusion saturates. On PRMBench, all label-free rules give almost the same score. Our pipeline ties the DS-signed average (15 better, 15 worse, of 36 method × bank pairs). On ProcessBench, the plain average (SLA 0.414, N16) is better than all four pipeline variants (0.396–0.408).</div>""")
R("""<div><b>Q2 · Competitors</b>PRMBench: our top step is an error more often than the supervised PRM-7B's, but its within-answer AUC is higher (0.80 vs 0.79). ProcessBench: above our run of Mind the Gap in all 8 cells, mixed vs the paper's numbers, far below the PRM (0.57 top step).</div>""",
  """<div><b>Q2 · Competitors</b>PRMBench: our top step is an error more often than the top step of the supervised PRM-7B. But the PRM has a higher within-answer AUC (0.80 vs 0.79). ProcessBench: we are above our run of Mind the Gap in all 8 cells. Against the paper's numbers, the result is mixed. We are far below the PRM (SLA 0.57, its top step).</div>""")
R("""<div><b>Q3 · Practical value</b>Ranking works, label-free and cheap; error types redundancy and circular logic beat the PRM. Weak point: deciding that an answer is clean (ProcessBench F1 ≈ 0.40 vs PRM 0.73).</div>""",
  """<div><b>Q3 · Practical value</b>The ranking works. It needs no labels and only one pass. On two error types, redundancy and circular logic, we rank errors better than the PRM. Our weak point: the decision that an answer is clean (ProcessBench F1 about 0.40; PRM 0.73).</div>""")
R("""<div><b>Headroom</b>Supervised LR on the same channels gains +0.03 to +0.07 AUC on PRMBench and almost nothing on ProcessBench. The representation is the lever, not the fusion rule.</div>""",
  """<div><b>Headroom</b>A supervised LR on the same channels adds +0.03 to +0.07 AUC on PRMBench. On ProcessBench it adds almost nothing. Thus the channels (the representation) set the score, not the fusion rule.</div>""")
R('\n<h2 id="pb">ProcessBench</h2>', (Path(__file__).parent / 'defs.html').read_text(encoding='utf8') + '\n<h2 id="pb">ProcessBench</h2>')
R('<figure><img src="fig/M2_mtg_per_cell_light.png"', (Path(__file__).parent / 'mtg.html').read_text(encoding='utf8') + '<figure><img src="fig/M2_mtg_per_cell_light.png"')
R('\n<h2 id="types">', (Path(__file__).parent / 'est.html').read_text(encoding='utf8') + '\n<h2 id="types">')
R('\n<h2 id="transfer">', (Path(__file__).parent / 'lit.html').read_text(encoding='utf8') + '\n<h2 id="transfer">')
p.write_text(s, encoding='utf8'); print('patched')
