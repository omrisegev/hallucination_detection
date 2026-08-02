#!/usr/bin/env python
"""
build_shape_page.py — the shareable shape-evidence page for the advisors.

Reads `results/nonmono_v2/shape_curves.json` (produced by shape_curves_export.py)
and emits a self-contained HTML page. No external assets: the Artifact CSP blocks
every remote host, so fonts are system stacks and the curves are inline SVG drawn
from the embedded JSON.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(REPO, "results", "nonmono_v2", "shape_curves.json")
DST = os.path.join(REPO, "results", "nonmono_v2", "shape_evidence.html")

CSS = """
:root{
  color-scheme:light;
  --ground:#F5F6F4; --panel:#FCFDFC; --ink:#111A1C; --ink-2:#415457; --muted:#6B7C7E;
  --rule:#DDE3E1; --rule-2:#C6D0CE;
  --trace:#0F8F84; --assume:#9E5F12; --crit:#B03A34;
  --trace-soft:rgba(15,143,132,.13); --chip:rgba(17,26,28,.05);
}
@media (prefers-color-scheme:dark){
  :root:where(:not([data-theme="light"])){
    color-scheme:dark;
    --ground:#14181A; --panel:#1B2124; --ink:#EDF1F0; --ink-2:#B4C2C2; --muted:#8A9A9B;
    --rule:#2A3336; --rule-2:#3A4548;
    --trace:#1FAA9D; --assume:#C88420; --crit:#E0736C;
    --trace-soft:rgba(31,170,157,.16); --chip:rgba(237,241,240,.07);
  }
}
:root[data-theme="dark"]{
  color-scheme:dark;
  --ground:#14181A; --panel:#1B2124; --ink:#EDF1F0; --ink-2:#B4C2C2; --muted:#8A9A9B;
  --rule:#2A3336; --rule-2:#3A4548;
  --trace:#1FAA9D; --assume:#C88420; --crit:#E0736C;
  --trace-soft:rgba(31,170,157,.16); --chip:rgba(237,241,240,.07);
}
:root[data-theme="light"]{
  color-scheme:light;
  --ground:#F5F6F4; --panel:#FCFDFC; --ink:#111A1C; --ink-2:#415457; --muted:#6B7C7E;
  --rule:#DDE3E1; --rule-2:#C6D0CE;
  --trace:#0F8F84; --assume:#9E5F12; --crit:#B03A34;
  --trace-soft:rgba(15,143,132,.13); --chip:rgba(17,26,28,.05);
}

*{box-sizing:border-box}
body{
  margin:0; background:var(--ground); color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif;
  font-size:16px; line-height:1.62;
  -webkit-font-smoothing:antialiased;
}
.wrap{max-width:1180px;margin:0 auto;padding:0 28px 96px}
.col{max-width:68ch;margin-inline:auto}

h1,h2,h3{font-family:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
  text-wrap:balance;font-weight:600;line-height:1.2;margin:0}
h1{font-size:2.55rem;letter-spacing:-.015em}
h2{font-size:1.62rem;margin-top:0}
h3{font-size:1.12rem}
p{margin:0}
.mono{font-family:ui-monospace,"Cascadia Mono","SF Mono",Consolas,monospace;
  font-variant-numeric:tabular-nums}
code{font-family:ui-monospace,"Cascadia Mono","SF Mono",Consolas,monospace;
  font-size:.87em;background:var(--chip);padding:.1em .34em;border-radius:3px}
.eyebrow{font-family:ui-monospace,"Cascadia Mono","SF Mono",Consolas,monospace;
  font-size:.72rem;letter-spacing:.14em;text-transform:uppercase;color:var(--muted)}

header.masthead{border-bottom:1px solid var(--rule-2);padding:64px 0 30px;margin-bottom:44px}
.masthead .col{display:flex;flex-direction:column;gap:16px}
.standfirst{font-size:1.13rem;color:var(--ink-2);max-width:60ch}
.byline{display:flex;gap:18px;flex-wrap:wrap;color:var(--muted);font-size:.8rem}

section{margin-bottom:56px}
section .col{display:flex;flex-direction:column;gap:18px}
.lede{font-size:1.05rem}

.keyfig{display:flex;gap:34px;flex-wrap:wrap;padding:22px 24px;
  background:var(--panel);border:1px solid var(--rule);border-radius:4px}
.keyfig div{display:flex;flex-direction:column;gap:2px}
.keyfig .v{font-size:1.85rem;font-family:ui-monospace,"Cascadia Mono",Consolas,monospace;
  line-height:1.1;color:var(--trace)}
.keyfig .v.warn{color:var(--crit)}
.keyfig .k{font-size:.74rem;color:var(--muted);letter-spacing:.05em;text-transform:uppercase}

.tablewrap{overflow-x:auto;border:1px solid var(--rule);border-radius:4px;background:var(--panel)}
table{border-collapse:collapse;width:100%;font-size:.87rem}
th,td{text-align:left;padding:9px 14px;border-bottom:1px solid var(--rule)}
tbody tr:last-child td{border-bottom:none}
th{font-size:.7rem;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);font-weight:600}
td.num,th.num{text-align:right;font-family:ui-monospace,"Cascadia Mono",Consolas,monospace;
  font-variant-numeric:tabular-nums}

.note{border-left:2px solid var(--rule-2);padding-left:16px;color:var(--ink-2);font-size:.95rem}
.note.warn{border-left-color:var(--crit)}

.chips{display:flex;gap:8px;flex-wrap:wrap}
.chip{display:inline-flex;align-items:center;gap:6px;font-size:.74rem;
  padding:3px 9px;border-radius:99px;border:1px solid var(--rule-2);color:var(--ink-2)}
.chip .dot{width:8px;height:8px;border-radius:99px;flex:none}

/* ── the small-multiples grid ─────────────────────────────────────── */
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(262px,1fr));gap:16px;margin-top:4px}
.card{background:var(--panel);border:1px solid var(--rule);border-radius:4px;padding:12px 12px 8px;
  display:flex;flex-direction:column;gap:7px}
.card .hd{display:flex;justify-content:space-between;align-items:baseline;gap:8px}
.card .ft{font-family:ui-monospace,"Cascadia Mono",Consolas,monospace;font-size:.79rem;
  color:var(--ink);font-weight:600;overflow-wrap:anywhere}
.card .sh{font-size:.68rem;color:var(--muted);white-space:nowrap}
.card .meta{display:flex;gap:10px;font-size:.68rem;color:var(--muted);
  font-family:ui-monospace,"Cascadia Mono",Consolas,monospace;flex-wrap:wrap}
.card.dim{opacity:.72}
svg.panel{display:block;width:100%;height:auto;overflow:visible}
.gridline{stroke:var(--rule);stroke-width:1}
.axis{stroke:var(--rule-2);stroke-width:1}
.baserate{stroke:var(--muted);stroke-width:1;stroke-dasharray:2 3;opacity:.75}
.fit{fill:none;stroke:var(--assume);stroke-width:2;stroke-dasharray:5 3;stroke-linecap:round}
.obs{fill:none;stroke:var(--trace);stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.err{stroke:var(--trace);stroke-width:1.25;opacity:.5}
.pt{fill:var(--trace);stroke:var(--panel);stroke-width:1.5}
.hit{fill:transparent;cursor:crosshair}
.tick{font-family:ui-monospace,"Cascadia Mono",Consolas,monospace;font-size:8.5px;fill:var(--muted)}

.celltitle{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;
  border-bottom:1px solid var(--rule);padding-bottom:7px;margin:30px 0 12px}
.celltitle .nm{font-family:ui-monospace,"Cascadia Mono",Consolas,monospace;font-size:.94rem;font-weight:600}
.celltitle .sub{font-size:.76rem;color:var(--muted);font-family:ui-monospace,Consolas,monospace}

#tip{position:fixed;pointer-events:none;opacity:0;transition:opacity .1s;z-index:50;
  background:var(--ink);color:var(--ground);font-size:.74rem;padding:6px 9px;border-radius:4px;
  font-family:ui-monospace,"Cascadia Mono",Consolas,monospace;white-space:nowrap;
  box-shadow:0 3px 14px rgba(0,0,0,.22)}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
:focus-visible{outline:2px solid var(--trace);outline-offset:2px}
@media (max-width:640px){h1{font-size:2rem}.wrap{padding:0 18px 64px}.keyfig{gap:22px}}
"""

JS = r"""
const DATA = __DATA__;
const NDEC = 10;
const tip = document.getElementById('tip');

function panelSVG(rec){
  const W=238,H=118,L=26,R=6,T=8,B=17;
  const pts = rec.curve.map((r,i)=>r?{...r,i}:null).filter(Boolean);
  let lo=Math.min(...pts.map(p=>p.p-p.se)), hi=Math.max(...pts.map(p=>p.p+p.se));
  if(rec.iso_min!==undefined){lo=Math.min(lo,rec.iso_min);hi=Math.max(hi,rec.iso_max);}
  const pad=Math.max((hi-lo)*0.14,0.012); lo=Math.max(0,lo-pad); hi=Math.min(1,hi+pad);
  const x=i=>L+(i+0.5)/NDEC*(W-L-R);
  const y=v=>T+(1-(v-lo)/Math.max(hi-lo,1e-9))*(H-T-B);
  const e=(t,a,inner)=>`<${t} ${Object.entries(a).map(([k,v])=>`${k}="${v}"`).join(' ')}>${inner!==undefined?inner+`</${t}>`:''}`;
  let s='';
  // horizontal gridlines + y ticks at lo / mid / hi
  [lo,(lo+hi)/2,hi].forEach(v=>{
    s+=`<line class="gridline" x1="${L}" x2="${W-R}" y1="${y(v).toFixed(1)}" y2="${y(v).toFixed(1)}"/>`;
    s+=`<text class="tick" x="${L-4}" y="${(y(v)+3).toFixed(1)}" text-anchor="end">${(v*100).toFixed(0)}</text>`;
  });
  s+=`<line class="axis" x1="${L}" x2="${L}" y1="${T}" y2="${H-B}"/>`;
  s+=`<line class="axis" x1="${L}" x2="${W-R}" y1="${H-B}" y2="${H-B}"/>`;
  // base rate reference
  if(rec.base_rate>=lo&&rec.base_rate<=hi)
    s+=`<line class="baserate" x1="${L}" x2="${W-R}" y1="${y(rec.base_rate).toFixed(1)}" y2="${y(rec.base_rate).toFixed(1)}"/>`;
  // the best monotone fit — what the fusion assumes the curve looks like
  const fitPts=pts.filter(p=>p.iso!==null&&p.iso!==undefined);
  if(fitPts.length>1)
    s+=`<path class="fit" d="${fitPts.map((p,k)=>(k?'L':'M')+x(p.i).toFixed(1)+' '+y(p.iso).toFixed(1)).join(' ')}"/>`;
  // observed curve + binomial error bars
  pts.forEach(p=>{
    s+=`<line class="err" x1="${x(p.i).toFixed(1)}" x2="${x(p.i).toFixed(1)}" y1="${y(Math.min(1,p.p+p.se)).toFixed(1)}" y2="${y(Math.max(0,p.p-p.se)).toFixed(1)}"/>`;
  });
  s+=`<path class="obs" d="${pts.map((p,k)=>(k?'L':'M')+x(p.i).toFixed(1)+' '+y(p.p).toFixed(1)).join(' ')}"/>`;
  pts.forEach(p=>{
    s+=`<circle class="pt" cx="${x(p.i).toFixed(1)}" cy="${y(p.p).toFixed(1)}" r="3.1"/>`;
    s+=`<rect class="hit" x="${(x(p.i)-(W-L-R)/NDEC/2).toFixed(1)}" y="${T}" width="${((W-L-R)/NDEC).toFixed(1)}" height="${H-T-B}" data-t="decile ${p.decile}/10 &#183; P(correct) ${(p.p*100).toFixed(1)}% &#177;${(p.se*100).toFixed(1)} &#183; n=${p.n}"/>`;
  });
  s+=`<text class="tick" x="${L}" y="${H-5}">low</text>`;
  s+=`<text class="tick" x="${W-R}" y="${H-5}" text-anchor="end">high</text>`;
  s+=`<text class="tick" x="${((L+W-R)/2).toFixed(0)}" y="${H-5}" text-anchor="middle">feature decile</text>`;
  return `<svg class="panel" viewBox="0 0 ${W} ${H}" role="img" aria-label="P(correct) by ${rec.feature} decile on ${rec.cell}">${s}</svg>`;
}

function card(rec){
  const d=document.createElement('div');
  d.className='card'+(rec.dim?' dim':'');
  const g = rec.v2_shape_gain===null?'n/a':(rec.v2_shape_gain>=0?'+':'')+rec.v2_shape_gain.toFixed(3);
  d.innerHTML=`<div class="hd"><span class="ft">${rec.feature}</span><span class="sh">${rec.shape}</span></div>`
    +panelSVG(rec)
    +`<div class="meta"><span>gain ${g}</span><span>n=${rec.n.toLocaleString()}</span><span>n<sub>min</sub>=${rec.n_min.toLocaleString()}</span></div>`;
  return d;
}

function render(hostId, recs){
  const host=document.getElementById(hostId); if(!host) return;
  const byCell={};
  recs.forEach(r=>{(byCell[r.cell]=byCell[r.cell]||[]).push(r)});
  Object.keys(byCell).sort((a,b)=>byCell[b].length-byCell[a].length).forEach(ck=>{
    const rs=byCell[ck];
    const h=document.createElement('div'); h.className='celltitle';
    h.innerHTML=`<span class="nm">${ck}</span><span class="sub">${rs[0].domain} &#183; n=${rs[0].n.toLocaleString()} &#183; ${(rs[0].base_rate*100).toFixed(1)}% correct &#183; ${rs.length} view${rs.length>1?'s':''}</span>`;
    host.appendChild(h);
    const g=document.createElement('div'); g.className='grid';
    rs.sort((a,b)=>(b.v2_shape_gain??-9)-(a.v2_shape_gain??-9)).forEach(r=>g.appendChild(card(r)));
    host.appendChild(g);
  });
}

const flagged = DATA.panels.filter(p=>p.flagged);
render('big',  flagged.filter(p=>p.big));
render('small',flagged.filter(p=>!p.big).map(p=>({...p,dim:true})));
render('ctrl', DATA.panels.filter(p=>!p.flagged).map(p=>({...p,dim:true})));

document.addEventListener('mouseover',e=>{
  const t=e.target.closest('.hit'); if(!t){tip.style.opacity=0;return;}
  tip.innerHTML=t.dataset.t; tip.style.opacity=1;
});
document.addEventListener('mousemove',e=>{
  if(tip.style.opacity==='0')return;
  const w=tip.offsetWidth;
  tip.style.left=Math.min(e.clientX+13,window.innerWidth-w-10)+'px';
  tip.style.top=(e.clientY-34)+'px';
});
"""


def main():
    with open(SRC, encoding="utf-8") as fh:
        data = json.load(fh)

    # y-range needs the isotonic contrast too, so the fit line is never clipped
    for rec in data["panels"]:
        iso = [r["iso"] for r in rec["curve"] if r and r.get("iso") is not None]
        if iso:
            rec["iso_min"], rec["iso_max"] = min(iso), max(iso)

    flagged = [p for p in data["panels"] if p["flagged"]]
    big = [p for p in flagged if p["big"]]
    small = [p for p in flagged if not p["big"]]
    ctrl = [p for p in data["panels"] if not p["flagged"]]
    n_big_cells = len({p["cell"] for p in big})

    html = f"""<title>Non-monotone features: the evidence</title>
<style>{CSS}</style>
<header class="masthead"><div class="wrap"><div class="col">
  <span class="eyebrow">Spectral hallucination detection &#183; interim finding</span>
  <h1>Some features bend. The fusion assumes they don't.</h1>
  <p class="standfirst">Both of our label-free fusion arms are linear in the views, which
  silently assumes every feature is monotone in P(correct). We tested that assumption on
  all {data['n_pairs_total']} feature&#215;cell pairs. Most hold. A handful do not &#8212; and
  the ones that fail are not the simple U-shapes we went looking for.</p>
  <div class="byline mono"><span>Omri Segev Moshe</span><span>2 Aug 2026</span>
    <span>Steps 216&#8211;217 + audit</span></div>
</div></div></header>

<div class="wrap">

<section><div class="col">
  <h2>What was measured</h2>
  <p class="lede">For every feature in every cell, we compared the best <em>monotone</em>
  reading of that feature against an unconstrained one &#8212; both fitted on training folds
  and scored on held-out folds, so neither side gets a free pass on estimation noise. The
  gap between them is how much signal a monotone model cannot reach.</p>
  <div class="keyfig">
    <div><span class="v mono">{data['n_pairs_total']}</span><span class="k">feature&#215;cell pairs</span></div>
    <div><span class="v mono">{data['n_flagged']}</span><span class="k">flagged at 5%</span></div>
    <div><span class="v mono warn">~34</span><span class="k">expected by chance</span></div>
    <div><span class="v mono">{len(big)}</span><span class="k">credible, on {n_big_cells} cells</span></div>
  </div>
  <p>The pair count is {data['n_pairs_total']} rather than 24&#215;30 because a feature is
  only testable where it exists: each cell's live pool runs 19&#8211;30 views, and
  <code>seiclr_triviaqa_opt30b</code> has just 19 &#8212; a three-token answer has no
  spectral content to measure.</p>
  <p class="note warn"><strong>The headline count proves nothing on its own.</strong>
  Running {data['n_pairs_total']} independent tests at the 5% level yields about 34 false
  positives under pure noise. We found {data['n_flagged']}. So the question is not "how
  many fired" but "did they fire where a real effect would".</p>
</div></section>

<section><div class="col">
  <h2>They fired where the data is</h2>
  <p>Splitting the same {data['n_pairs_total']} pairs by cell size settles it. A noise
  process flags ~5% everywhere regardless of sample size. A real effect concentrates where
  there is power to see it.</p>
  <div class="tablewrap"><table>
    <thead><tr><th>Cell size</th><th class="num">Pairs</th><th class="num">Flagged</th>
      <th class="num">Rate</th><th>Against 5% chance</th></tr></thead>
    <tbody>
      <tr><td class="mono">n &#8805; 2000</td><td class="num">188</td><td class="num">22</td>
        <td class="num">11.7%</td><td>more than double</td></tr>
      <tr><td class="mono">n &lt; 2000</td><td class="num">494</td><td class="num">10</td>
        <td class="num">2.0%</td><td>less than half</td></tr>
    </tbody></table></div>
  <p>All 22 credible detections sit on <strong>five QA cells</strong>, and four features
  carry most of them: <code>rpdi</code> (7 cells), <code>pe_mean</code> (6),
  <code>cusum_shift_idx</code> (6), <code>hurst_exponent</code> (3). Math cells show
  essentially nothing, including at n=5000 where there is ample power.</p>
  <p class="note">The largest single number in the whole sweep &#8212;
  <code>spilled_triviaqa_llama8b</code>/<code>rpdi</code> at +0.294 &#8212; comes from a cell
  with <strong>6 correct answers out of 256</strong>. Under 5-fold CV that is about one
  positive per test fold, so each fold's AUROC is near a coin flip. It is measuring fold
  noise, and it is excluded below.</p>
</div></section>

<section><div class="col">
  <h2>The curves</h2>
  <p>Each panel plots the share of answers that were correct, against the feature's own
  decile within that cell. The dashed line is the best monotone fit &#8212; the shape the
  fusion is built to exploit. Where the solid line departs from it, there is signal a
  linear fusion cannot reach.</p>
  <div class="chips">
    <span class="chip"><span class="dot" style="background:var(--trace)"></span>observed P(correct), &#177;1 SE</span>
    <span class="chip"><span class="dot" style="background:var(--assume)"></span>best monotone fit</span>
    <span class="chip"><span class="dot" style="background:var(--muted)"></span>cell base rate</span>
  </div>
</div>
  <div class="col"><h3 style="margin-top:34px">Credible &#8212; large cells (n &#8805; 2000)</h3></div>
  <div id="big"></div>
</section>

<section>
  <div class="col"><h3>Underpowered &#8212; small cells, shown for completeness</h3>
  <p class="note">These fire below the chance rate as a group. Individually they are not
  evidence; they are shown so the record is complete rather than curated.</p></div>
  <div id="small"></div>
</section>

<section>
  <div class="col"><h3>Controls &#8212; the most monotone view on each large cell</h3>
  <p class="note">Measured by the identical procedure. These are what "no defect" looks
  like: the solid and dashed lines sit on top of each other.</p></div>
  <div id="ctrl"></div>
</section>

<section><div class="col">
  <h2>The shapes are the point</h2>
  <p>We tried three transforms to straighten these out &#8212; <code>|x&#8722;median|</code>,
  <code>x&#178;</code>, and <code>|&#934;&#8315;&#185;(rank%)|</code>. All three failed to
  improve the fused score. The curves above show why: <strong>every one of those transforms
  is symmetric and centred on the middle of the distribution</strong>, and almost none of
  these curves are.</p>
  <div class="tablewrap"><table>
    <thead><tr><th>Cell / view</th><th>Actual shape</th><th>Why the transform misses</th></tr></thead>
    <tbody>
      <tr><td class="mono">semenergy&#8230;qwen3_8b / rpdi</td><td>W-shaped</td>
        <td>two bends; not in the family at all</td></tr>
      <tr><td class="mono">semenergy&#8230;qwen3_8b / epr_energy</td><td>inverted-U, peak at decile 6&#8211;7</td>
        <td>mis-centred by a median-centred fold</td></tr>
      <tr><td class="mono">se_squad_v2_llama8b / pe_mean</td><td>dip at decile 2, peak at 9</td>
        <td>asymmetric; folding destroys the ordering</td></tr>
      <tr><td class="mono">se_nq_open_llama8b / rpdi</td><td>argmax at the edge, interior dip</td>
        <td>the gain is a dip, not a peak</td></tr>
    </tbody></table></div>
  <p>So the negative result is narrow: one family of transform, applied one way. It does not
  say reshaping cannot work. The open question is whether the <em>same</em> shape recurs
  across cells &#8212; if it does, a general correction exists; if each cell bends its own
  way, there is nothing to generalise and any fix would be per-cell overfitting.</p>
</div></section>

<section><div class="col">
  <h2>Three measurement defects found along the way</h2>
  <p>All three were in our own code, and each one inflated a number we had been quoting.</p>
  <div class="tablewrap"><table>
    <thead><tr><th>Defect</th><th>Effect</th></tr></thead>
    <tbody>
      <tr><td><strong>Folded AUROC per fold.</strong> <code>gap_ladder.py</code> applied
        <code>max(p,&#8239;1&#8722;p)</code> to each fold's binned score, but a map fitted on
        training data already carries its direction.</td>
        <td>A one-sided noise floor: inflation never negative, up to <strong>+0.200</strong>.
        It credits a view <em>more</em> the closer it sits to chance
        (&#961;=&#8722;0.171, p=7e&#8722;06). <code>pe_mean</code>'s headline +0.044 was
        +0.040 inflation.</td></tr>
      <tr><td><strong>Monotone baseline fitted backwards.</strong> The direction was chosen by
        Pearson correlation, which is ~0 for a U-shape &#8212; so its sign is coin-flip noise.</td>
        <td>A false-positive source aimed at exactly the shapes we hunt. Recomputing with the
        proper constrained fit, <code>math500_qwenmath7b/min_spilled</code> drops from
        <strong>+0.188 to &#8722;0.009</strong> and reclassifies as monotone.</td></tr>
      <tr><td><strong>Sample-size-blind screening.</strong> Only pairs above a fixed threshold
        were tested, while the noise floor shrinks with sample size.</td>
        <td>On the largest cell the floor is 0.016, so a genuine effect of 0.018 was never
        tested at all. Corrected sweep is in progress.</td></tr>
    </tbody></table></div>
  <p class="note warn">Separately, two cells in the benchmark were generating malformed text
  and have been dealt with: <code>seiclr_triviaqa_opt30b</code> was repaired by cropping to
  the answer span (a bug fix &#8212; the grader already cropped, so labels and features
  disagreed), and <code>inside_coqa_llama7b</code> is withdrawn pending regeneration. The
  roster is now <strong>24 cells, not 25</strong>, and the reference constant moved from
  0.7594 to <strong>0.7733</strong>. Every number predating this is stale.</p>
</div></section>

<section><div class="col">
  <h2>What happens next</h2>
  <p>The corrected detection sweep tests all {data['n_pairs_total']} pairs with no screening,
  three statistically independent detectors, a null that holds the relationship
  <em>monotone</em> rather than absent, and family-level false-discovery control &#8212; plus
  a power analysis, so "we found all of them" becomes a quantified claim rather than an
  assertion. Then the decisive test: does a feature bend the <em>same way</em> across cells?
  That question, not the transform search, is what separates a general correction from
  overfitting to the 24 cells we happen to have.</p>
  <p class="note">One structural finding worth flagging early. U-PCR drops a view whose
  estimated &#961; is near zero &#8212; and &#961; is computed from <em>linear</em> covariance,
  so a U-shaped view has &#961;&#8776;0 and is excluded before fusion begins. On four of the
  five cells that carry the detections, last round's transform therefore changed the fused
  score by <em>exactly zero</em>. That is both why the experiment came back null, and the
  channel through which a working transform would have to act.</p>
</div></section>

</div>
<div id="tip" role="status" aria-live="polite"></div>
<script>{JS.replace('__DATA__', json.dumps(data, separators=(',', ':')))}</script>
"""
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    with open(DST, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"wrote {DST}  ({len(html)/1024:.0f} KB)")
    print(f"  {len(big)} credible panels / {len(small)} underpowered / {len(ctrl)} controls")
    return 0


if __name__ == "__main__":
    sys.exit(main())
