#!/usr/bin/env python
"""
build_transform_page.py — the visual justification for every transform choice.

Per candidate (cell, feature): the class-conditional densities of the feature for
CORRECT vs HALLUCINATED answers, shown for the raw view and for every transform
considered, side by side, with each option's cross-fitted AUROC.

WHY THIS IS THE RIGHT PICTURE. A linear fusion can only exploit a view whose two
class densities are SHIFTED versions of each other — that is what "monotone in
P(correct)" means geometrically. A non-monotone view has densities that sit on top
of each other in location but differ in SPREAD: the correct answers piled in the
middle, the hallucinations pushed to both tails (or the reverse). No monotone
reading separates those, which is why AUROC sits at ~0.5. Folding the axis turns
a spread difference into a location difference. The panels show that conversion
happening, or failing to.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(REPO, "results", "nonmono_v2", "transform_selection.json")
DST = os.path.join(REPO, "results", "nonmono_v2", "transform_choices.html")

CSS = """
:root{
  color-scheme:light;
  --ground:#F5F6F4; --panel:#FCFDFC; --sunk:#EEF1EF; --ink:#111A1C; --ink-2:#415457;
  --muted:#6B7C7E; --rule:#DDE3E1; --rule-2:#C6D0CE;
  --ok:#0F8F84; --bad:#9E5F12; --crit:#B03A34;
  --ok-f:rgba(15,143,132,.20); --bad-f:rgba(158,95,18,.20);
  --chip:rgba(17,26,28,.05); --pick:rgba(15,143,132,.09);
}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){
  color-scheme:dark;
  --ground:#14181A; --panel:#1B2124; --sunk:#171C1E; --ink:#EDF1F0; --ink-2:#B4C2C2;
  --muted:#8A9A9B; --rule:#2A3336; --rule-2:#3A4548;
  --ok:#1FAA9D; --bad:#C88420; --crit:#E0736C;
  --ok-f:rgba(31,170,157,.24); --bad-f:rgba(200,132,32,.24);
  --chip:rgba(237,241,240,.07); --pick:rgba(31,170,157,.12);
}}
:root[data-theme="dark"]{
  color-scheme:dark;
  --ground:#14181A; --panel:#1B2124; --sunk:#171C1E; --ink:#EDF1F0; --ink-2:#B4C2C2;
  --muted:#8A9A9B; --rule:#2A3336; --rule-2:#3A4548;
  --ok:#1FAA9D; --bad:#C88420; --crit:#E0736C;
  --ok-f:rgba(31,170,157,.24); --bad-f:rgba(200,132,32,.24);
  --chip:rgba(237,241,240,.07); --pick:rgba(31,170,157,.12);
}
:root[data-theme="light"]{
  color-scheme:light;
  --ground:#F5F6F4; --panel:#FCFDFC; --sunk:#EEF1EF; --ink:#111A1C; --ink-2:#415457;
  --muted:#6B7C7E; --rule:#DDE3E1; --rule-2:#C6D0CE;
  --ok:#0F8F84; --bad:#9E5F12; --crit:#B03A34;
  --ok-f:rgba(15,143,132,.20); --bad-f:rgba(158,95,18,.20);
  --chip:rgba(17,26,28,.05); --pick:rgba(15,143,132,.09);
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif;font-size:16px;line-height:1.6}
.wrap{max-width:1320px;margin:0 auto;padding:0 26px 90px}
.col{max-width:68ch;margin-inline:auto}
h1,h2,h3{font-family:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
  text-wrap:balance;font-weight:600;line-height:1.2;margin:0}
h1{font-size:2.5rem;letter-spacing:-.015em}
h2{font-size:1.6rem}
p{margin:0}
.mono,code{font-family:ui-monospace,"Cascadia Mono","SF Mono",Consolas,monospace;
  font-variant-numeric:tabular-nums}
code{font-size:.87em;background:var(--chip);padding:.1em .34em;border-radius:3px}
.eyebrow{font-family:ui-monospace,Consolas,monospace;font-size:.72rem;letter-spacing:.14em;
  text-transform:uppercase;color:var(--muted)}
header.mast{border-bottom:1px solid var(--rule-2);padding:60px 0 28px;margin-bottom:40px}
.mast .col{display:flex;flex-direction:column;gap:14px}
.standfirst{font-size:1.12rem;color:var(--ink-2)}
section{margin-bottom:48px}
section .col{display:flex;flex-direction:column;gap:16px}
.note{border-left:2px solid var(--rule-2);padding-left:15px;color:var(--ink-2);font-size:.95rem}
.note.warn{border-left-color:var(--crit)}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:.8rem;color:var(--ink-2);
  padding:12px 16px;background:var(--panel);border:1px solid var(--rule);border-radius:4px}
.legend span{display:inline-flex;align-items:center;gap:7px}
.sw{width:15px;height:9px;border-radius:2px;flex:none}
.tablewrap{overflow-x:auto;border:1px solid var(--rule);border-radius:4px;background:var(--panel)}
table{border-collapse:collapse;width:100%;font-size:.86rem}
th,td{text-align:left;padding:8px 13px;border-bottom:1px solid var(--rule);white-space:nowrap}
tbody tr:last-child td{border-bottom:none}
th{font-size:.69rem;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);font-weight:600}
td.num,th.num{text-align:right;font-family:ui-monospace,Consolas,monospace;
  font-variant-numeric:tabular-nums}
td.pick{color:var(--ok);font-weight:600}

/* ── one candidate ─────────────────────────────────────────────── */
.cand{background:var(--panel);border:1px solid var(--rule);border-radius:5px;
  padding:16px 18px 14px;margin-bottom:18px}
.cand .top{display:flex;justify-content:space-between;align-items:baseline;gap:14px;
  flex-wrap:wrap;border-bottom:1px solid var(--rule);padding-bottom:10px;margin-bottom:12px}
.cand .id{font-family:ui-monospace,Consolas,monospace;font-size:.95rem;font-weight:600}
.cand .id .cellnm{color:var(--muted);font-weight:400}
.cand .stats{display:flex;gap:14px;font-size:.74rem;color:var(--muted);
  font-family:ui-monospace,Consolas,monospace;flex-wrap:wrap}
.why{font-size:.85rem;color:var(--ink-2);margin-bottom:12px;padding:9px 12px;
  background:var(--pick);border-radius:4px;border-left:2px solid var(--ok)}
.why b{color:var(--ok)}
.opts{display:grid;grid-template-columns:repeat(auto-fill,minmax(196px,1fr));gap:11px}
.opt{background:var(--sunk);border:1px solid var(--rule);border-radius:4px;padding:9px 9px 6px;
  display:flex;flex-direction:column;gap:5px}
.opt.chosen{border-color:var(--ok);background:var(--pick);box-shadow:0 0 0 1px var(--ok)}
.opt.base{border-style:dashed}
.opt .nm{font-family:ui-monospace,Consolas,monospace;font-size:.75rem;font-weight:600;
  display:flex;justify-content:space-between;gap:6px;align-items:baseline}
.opt .nm .tag{font-size:.6rem;font-weight:400;color:var(--muted);letter-spacing:.05em}
.opt .sc{display:flex;justify-content:space-between;font-size:.71rem;
  font-family:ui-monospace,Consolas,monospace;color:var(--muted)}
.opt .sc .d.up{color:var(--ok);font-weight:600}
.opt .sc .d.dn{color:var(--crit)}
.opt .prm{font-size:.63rem;color:var(--muted);font-family:ui-monospace,Consolas,monospace;
  overflow-wrap:anywhere;min-height:1em}
svg.dens,svg.sweep{display:block;width:100%;height:auto;overflow:visible}
.axis{stroke:var(--rule-2);stroke-width:1}
.gridline{stroke:var(--rule);stroke-width:1}
.tick{font-family:ui-monospace,Consolas,monospace;font-size:7.5px;fill:var(--muted)}
.sweepline{fill:none;stroke:var(--ok);stroke-width:2;stroke-linejoin:round}
.sweepline.pl{stroke:var(--bad);stroke-dasharray:4 3}
.marker{stroke:var(--muted);stroke-width:1;stroke-dasharray:2 2}
.sweepbox{margin-top:13px;padding-top:12px;border-top:1px solid var(--rule);
  display:grid;grid-template-columns:minmax(0,260px) 1fr;gap:16px;align-items:center}
.sweepbox p{font-size:.8rem;color:var(--ink-2)}
#tip{position:fixed;pointer-events:none;opacity:0;transition:opacity .1s;z-index:60;
  background:var(--ink);color:var(--ground);font-size:.73rem;padding:6px 9px;border-radius:4px;
  font-family:ui-monospace,Consolas,monospace;white-space:nowrap;box-shadow:0 3px 14px rgba(0,0,0,.22)}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
:focus-visible{outline:2px solid var(--ok);outline-offset:2px}
@media (max-width:700px){h1{font-size:1.9rem}.wrap{padding:0 16px 60px}
  .sweepbox{grid-template-columns:1fr}}
"""

JS = r"""
const DATA = __DATA__;
const tip = document.getElementById('tip');
const LABEL = {
  identity:'raw feature', squared:'x&#178;', dist_median:'|x &#8722; median|',
  abs_rank:'|&#934;&#8315;&#185;(u)|', mode_centre:'|u &#8722; mode|',
  best_centre:'|u &#8722; c*|', consensus_centre:'|u &#8722; c| consensus',
  loco_centre:'|u &#8722; c| other cells', loco_binmap:'bin map, other cells',
  hinge:'hinge (asym)', consensus_map:'bin map, consensus'
};

function densSVG(d, colOk, colBad){
  const W=178,H=74,L=3,R=3,T=6,B=11;
  if(!d) return `<svg class="dens" viewBox="0 0 ${W} ${H}"><text class="tick" x="${W/2}" y="${H/2}" text-anchor="middle">degenerate</text></svg>`;
  const n=d.grid.length;
  const x=i=>L+i/(n-1)*(W-L-R), y=v=>T+(1-v)*(H-T-B);
  const area=a=>`M ${x(0).toFixed(1)} ${y(0).toFixed(1)} `+a.map((v,i)=>`L ${x(i).toFixed(1)} ${y(v).toFixed(1)}`).join(' ')+` L ${x(n-1).toFixed(1)} ${y(0).toFixed(1)} Z`;
  const line=a=>a.map((v,i)=>(i?'L':'M')+x(i).toFixed(1)+' '+y(v).toFixed(1)).join(' ');
  let s=`<line class="axis" x1="${L}" x2="${W-R}" y1="${y(0).toFixed(1)}" y2="${y(0).toFixed(1)}"/>`;
  s+=`<path d="${area(d.halluc)}" fill="var(--bad-f)"/><path d="${area(d.correct)}" fill="var(--ok-f)"/>`;
  s+=`<path d="${line(d.halluc)}" fill="none" stroke="var(--bad)" stroke-width="1.8" stroke-linejoin="round"/>`;
  s+=`<path d="${line(d.correct)}" fill="none" stroke="var(--ok)" stroke-width="1.8" stroke-linejoin="round"/>`;
  s+=`<text class="tick" x="${L}" y="${H-2}">low</text><text class="tick" x="${W-R}" y="${H-2}" text-anchor="end">high</text>`;
  s+=`<rect class="hit" x="0" y="0" width="${W}" height="${H}" fill="transparent" data-t="correct n=${d.n_correct} &#183; hallucinated n=${d.n_halluc}"/>`;
  return `<svg class="dens" viewBox="0 0 ${W} ${H}" role="img" aria-label="class-conditional densities">${s}</svg>`;
}

function sweepSVG(sw){
  const W=300,H=96,L=30,R=8,T=8,B=18;
  const g=sw.grid, a=sw.auc, ap=sw.auc_pseudo;
  const all=a.concat(ap||[]).filter(Number.isFinite);
  let lo=Math.min(...all), hi=Math.max(...all); const pad=Math.max((hi-lo)*.12,.01);
  lo-=pad; hi+=pad;
  const x=i=>L+i/(g.length-1)*(W-L-R), y=v=>T+(1-(v-lo)/(hi-lo))*(H-T-B);
  let s='';
  [lo,(lo+hi)/2,hi].forEach(v=>{
    s+=`<line class="gridline" x1="${L}" x2="${W-R}" y1="${y(v).toFixed(1)}" y2="${y(v).toFixed(1)}"/>`;
    s+=`<text class="tick" x="${L-4}" y="${(y(v)+3).toFixed(1)}" text-anchor="end">${v.toFixed(2)}</text>`;
  });
  s+=`<line class="axis" x1="${L}" x2="${L}" y1="${T}" y2="${H-B}"/><line class="axis" x1="${L}" x2="${W-R}" y1="${H-B}" y2="${H-B}"/>`;
  if(ap) s+=`<path class="sweepline pl" d="${ap.map((v,i)=>(i?'L':'M')+x(i).toFixed(1)+' '+y(v).toFixed(1)).join(' ')}"/>`;
  s+=`<path class="sweepline" d="${a.map((v,i)=>(i?'L':'M')+x(i).toFixed(1)+' '+y(v).toFixed(1)).join(' ')}"/>`;
  const bi=a.indexOf(Math.max(...a));
  s+=`<line class="marker" x1="${x(bi).toFixed(1)}" x2="${x(bi).toFixed(1)}" y1="${T}" y2="${H-B}"/>`;
  s+=`<text class="tick" x="${x(bi).toFixed(1)}" y="${T-1}" text-anchor="middle">c*=${g[bi]}</text>`;
  s+=`<text class="tick" x="${L}" y="${H-5}">c=0.05</text><text class="tick" x="${W-R}" y="${H-5}" text-anchor="end">c=0.95</text>`;
  g.forEach((c,i)=>{s+=`<rect class="hit" x="${(x(i)-6).toFixed(1)}" y="${T}" width="12" height="${H-T-B}" fill="transparent" data-t="centre ${c} &#183; AUROC ${a[i].toFixed(3)}${ap?' &#183; label-free '+ap[i].toFixed(3):''}"/>`;});
  return `<svg class="sweep" viewBox="0 0 ${W} ${H}" role="img" aria-label="AUROC by fold centre">${s}</svg>`;
}

function optCard(o, chosen){
  const cls='opt'+(o.name===chosen?' chosen':'')+(o.name==='identity'?' base':'');
  const d=o.delta_pp, dc=d>0.05?'up':(d<-0.05?'dn':'');
  const prm=Object.entries(o.params||{}).map(([k,v])=>k==='fitted_on'?v:`${k}=${v}`).join(' &#183; ');
  return `<div class="${cls}">
    <div class="nm"><span>${LABEL[o.name]||o.name}</span><span class="tag">${o.label_free?'label-free':'fitted'}</span></div>
    ${densSVG(o.dens)}
    <div class="sc"><span>AUROC ${o.auc.toFixed(3)}</span><span class="d ${dc}">${d>=0?'+':''}${d.toFixed(1)}pp</span></div>
    <div class="prm">${prm||'&#160;'}</div></div>`;
}

function candCard(p){
  const el=document.createElement('div'); el.className='cand';
  const ch=p.options.find(o=>o.name===p.chosen);
  el.innerHTML=`<div class="top">
      <span class="id">${p.feature} <span class="cellnm">on ${p.cell}</span></span>
      <span class="stats"><span>n=${p.n.toLocaleString()}</span><span>n<sub>min</sub>=${p.n_min.toLocaleString()}</span>
        <span>${(p.base_rate*100).toFixed(1)}% correct</span>
        <span>raw AUROC ${p.auc_mono.toFixed(3)}</span>
        <span>ceiling ${p.auc_ceiling.toFixed(3)}</span>
        <span>headroom ${p.headroom_pp.toFixed(1)}pp</span></span></div>
    <div class="why"><b>Chosen: ${LABEL[p.chosen]||p.chosen}</b> &#8212; ${p.why}</div>
    <div class="opts">${p.options.map(o=>optCard(o,p.chosen)).join('')}</div>
    <div class="sweepbox"><p><b>Where the fold should sit.</b> AUROC of
      |u &#8722; c| as the centre <code>c</code> sweeps the percentile range. Solid is
      scored against the true labels; dashed against the label-free consensus
      pseudo-label. When the two peak together, the centre is recoverable without an
      answer key.</p>${sweepSVG(p.centre_sweep)}</div>`;
  return el;
}

const cands=DATA.panels.filter(p=>p.is_candidate).sort((a,b)=>b.headroom_pp-a.headroom_pp);
const ctrls=DATA.panels.filter(p=>!p.is_candidate).sort((a,b)=>a.headroom_pp-b.headroom_pp);
const hc=document.getElementById('cands'); cands.forEach(p=>hc.appendChild(candCard(p)));
const hx=document.getElementById('ctrls');
ctrls.filter(p=>p.centre_sweep).forEach(p=>hx.appendChild(candCard(p)));

// summary table
const names=[...new Set(DATA.panels.flatMap(p=>p.options.map(o=>o.name)))].filter(n=>n!=='identity');
const tb=document.getElementById('sumtab');
tb.innerHTML=`<thead><tr><th>Transform</th><th>Fitted on</th>
    <th class="num">mean &#916; on candidates</th><th class="num">improves</th>
    <th class="num">mean &#916; on controls</th><th class="num">chosen</th></tr></thead><tbody>`
  +names.map(n=>{
    const c=cands.map(p=>p.options.find(o=>o.name===n)).filter(Boolean);
    const x=ctrls.map(p=>p.options.find(o=>o.name===n)).filter(Boolean);
    if(!c.length) return '';
    const mc=c.reduce((s,o)=>s+o.delta_pp,0)/c.length;
    const mx=x.length?x.reduce((s,o)=>s+o.delta_pp,0)/x.length:NaN;
    const w=c.filter(o=>o.delta_pp>0).length;
    const picked=DATA.panels.filter(p=>p.chosen===n).length;
    const lf=c[0].label_free;
    return `<tr><td class="mono">${LABEL[n]||n}</td><td>${lf?'nothing (label-free)':'labels'}</td>
      <td class="num" style="color:${mc>0?'var(--ok)':'var(--crit)'}">${mc>=0?'+':''}${mc.toFixed(2)}pp</td>
      <td class="num">${w}/${c.length}</td>
      <td class="num" style="color:${mx>0?'var(--ok)':'var(--crit)'}">${isNaN(mx)?'-':(mx>=0?'+':'')+mx.toFixed(2)+'pp'}</td>
      <td class="num ${picked?'pick':''}">${picked}</td></tr>`;}).join('')+'</tbody>';

document.addEventListener('mouseover',e=>{const t=e.target.closest('.hit');
  if(!t){tip.style.opacity=0;return;} tip.innerHTML=t.dataset.t; tip.style.opacity=1;});
document.addEventListener('mousemove',e=>{if(tip.style.opacity==='0')return;
  tip.style.left=Math.min(e.clientX+13,window.innerWidth-tip.offsetWidth-10)+'px';
  tip.style.top=(e.clientY-34)+'px';});
"""


N_CTRL_SHOWN = 6


def main():
    with open(SRC, encoding="utf-8") as fh:
        data = json.load(fh)
    sel = data["selector"]
    cands = [p for p in data["panels"] if p["is_candidate"]]
    ctrls = [p for p in data["panels"] if not p["is_candidate"]]

    # Only candidates and the shown controls need their density arrays embedded;
    # carrying all 99 panels' curves triples the page for nothing the reader sees.
    # The stripped panels stay in `panels` so the summary table's control column is
    # still computed over all 61 of them.
    shown = {id(p) for p in cands}
    shown |= {id(p) for p in sorted(ctrls, key=lambda p: p["headroom_pp"])[:N_CTRL_SHOWN]}
    for p in data["panels"]:
        if id(p) not in shown:
            p["centre_sweep"] = None
            for o in p["options"]:
                o["dens"] = None
    adopted = sum(1 for p in cands if p["chosen"] != "identity")
    ctrl_bad = sum(1 for p in ctrls if p["chosen"] != "identity")

    html = f"""<title>Which transform, and why</title>
<style>{CSS}</style>
<header class="mast"><div class="wrap"><div class="col">
  <span class="eyebrow">Spectral hallucination detection &#183; transform selection</span>
  <h1>Folding the axis turns a spread difference into a shift.</h1>
  <p class="standfirst">A linear fusion can only use a feature whose two class
  distributions are <em>shifted</em> versions of each other. Several of ours differ in
  <em>spread</em> instead &#8212; correct answers piled in the middle, hallucinations
  pushed to both tails. No monotone reading separates those. These panels show each
  candidate transform attempting the conversion, and whether it works.</p>
  <div class="stats mono" style="color:var(--muted);font-size:.8rem">
    <span>Omri Segev Moshe</span> &#183; <span>2 Aug 2026</span> &#183;
    <span>{len(cands)} candidates, {len(ctrls)} controls</span></div>
</div></div></header>

<div class="wrap">
<section><div class="col">
  <h2>How to read a panel</h2>
  <div class="legend">
    <span><span class="sw" style="background:var(--ok-f);border:1.5px solid var(--ok)"></span>
      correct answers</span>
    <span><span class="sw" style="background:var(--bad-f);border:1.5px solid var(--bad)"></span>
      hallucinations</span>
    <span>AUROC is cross-fitted, with the sign chosen on the training folds</span>
  </div>
  <p class="note"><strong>AUROC is invariant under any monotone transform.</strong> So the
  raw feature's AUROC is the ceiling over every monotone reading of it, and any option that
  beats it is necessarily non-monotone. That makes each panel a self-contained test: no
  fusion, no selector, no ensemble.</p>
  <p><strong>The selection rule, fixed before looking:</strong> among options that use
  <em>no labels from any cell</em>, take the highest AUROC, and adopt it only if it beats
  the raw view by at least {data['min_gain_pp']}pp. Label-fitted options
  (<code>|u&#8722;c*|</code>, <code>hinge</code>, the LOCO fits) are computed and shown as
  <em>diagnostics</em> &#8212; they mark the family's own ceiling &#8212; but they are never
  chosen.</p>
</div></section>

<section><div class="col">
  <h2>What each option buys</h2>
  <p>Candidates are views with at least {data['min_headroom']}pp of measured non-monotone
  headroom. Controls are the rest &#8212; views already read correctly, where the same
  transforms should and do <em>hurt</em>.</p>
  <div class="tablewrap"><table id="sumtab"></table></div>
  <p class="note">The controls column is the honest test. A transform that gains on
  candidates and also gains on controls is not detecting anything &#8212; it is just adding
  variance the AUROC estimate happens to reward.</p>
</div></section>

<section><div class="col">
  <h2>Can we tell which views to fold, without labels?</h2>
  <p>This is the binding constraint. Folding a view that is already monotone costs several
  points, so applying it everywhere is worse than doing nothing. Step 217 established that
  the marginal shape cannot decide it &#8212; whether the feature's own histogram has two
  humps is uncorrelated with whether its label curve bends.</p>
  <p>The candidate here builds a <strong>pseudo-label from the other views</strong> (their
  simple average, which needs no answer key) and asks whether the view bends against
  <em>that</em>. Measured across all {sel['n']} views:</p>
  <div class="legend"><span class="mono">Spearman(consensus-detected bend, true headroom)
    = <strong>{sel['rho']:+.3f}</strong>, p = {sel['p']:.2e}, n = {sel['n']}</span></div>
  <p class="note warn">Of the {len(ctrls)} control views, the label-free rule would still
  transform <strong>{ctrl_bad}</strong>. Those are the false positives any deployable rule
  has to pay for, and they are counted against it, not hidden.</p>
</div></section>

<section>
  <div class="col"><h2>The candidates</h2>
  <p>Ordered by how much headroom exists. Each card shows every option, the chosen one
  outlined, and underneath, where the fold's centre should sit &#8212; scored against the
  true labels and against the label-free consensus, so you can see whether the right centre
  is recoverable without an answer key.</p></div>
  <div id="cands"></div>
</section>

<section>
  <div class="col"><h2>Controls &#8212; where the transforms should fail</h2>
  <p class="note">Six views with no measured headroom, run through the identical pipeline.
  The two class densities here are already shifted rather than differently spread, so
  folding the axis destroys a working feature. Adopted on {adopted} of {len(cands)}
  candidates; on these, the rule should and mostly does keep the raw view.</p></div>
  <div id="ctrls"></div>
</section>
</div>
<div id="tip" role="status" aria-live="polite"></div>
<script>{JS.replace('__DATA__', json.dumps(data, separators=(',', ':')))}</script>
"""
    with open(DST, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"wrote {DST}  ({len(html)/1024:.0f} KB)")
    print(f"  {len(cands)} candidates ({adopted} transformed), {len(ctrls)} controls")
    return 0


if __name__ == "__main__":
    sys.exit(main())
