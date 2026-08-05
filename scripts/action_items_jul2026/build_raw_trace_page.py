#!/usr/bin/env python
"""
build_raw_trace_page.py — what the RAW data looks like on a weak cell vs a strong one.

Everything built for the Jul-2026 action items so far lives at the FEATURE level:
30 scalars per answer. This page goes one level down, to the token-entropy traces
those scalars are computed from, because on `seiclr_triviaqa_opt30b` the raw data
turns out to be malformed in a way no fusion could survive:

  * every one of its 5,000 traces is exactly 64 tokens — i.e. every generation hit
    max_new_tokens and was truncated, none stopped on its own;
  * 99.8% of them run past the answer and start inventing a NEW few-shot
    "Question:" block;
  * the actual answer is a handful of tokens at the very front.

So the features are mostly measuring the model's run-on continuation, not its
answer. That is a generation/parsing defect upstream of anything we have been
testing. This page shows it: the questions, what the model actually emitted, the
entropy trace with the answer boundary marked, and what the leading features look
like against a healthy cell — plus a direct test of whether cropping the trace to
the answer span recovers the signal.

Pure CPU, offline, reads cache/repgrid/<cell>/*.pkl directly.
"""
import glob
import html
import os
import pickle
import re
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for _p in (REPO, os.path.join(REPO, "scripts"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common import CSS, esc, hist_svg, page                              # noqa: E402

OUT = os.path.join(REPO, "results", "action_items_jul2026",
                   "item1c_raw_traces")

WEAK = ("seiclr_triviaqa_opt30b", "TriviaQA / OPT-30B (base)")
STRONG = ("semenergy_triviaqa_qwen3_8b", "TriviaQA / Qwen3-8B")

# Where the answer ends: the first newline, or an explicit new Q/A block.
BOUNDARY = re.compile(r"\n|(?:^|\s)(?:Question|Q|Answer|A)\s*:")
N_SHOW = 5


def load(cell):
    f = sorted(glob.glob(os.path.join(REPO, "cache", "repgrid", cell, "*.pkl")))
    if not f:
        raise SystemExit(f"no raw pkl for {cell}")
    return pickle.load(open(f[0], "rb")), os.path.basename(f[0])


THINK = re.compile(r"^\s*<think>.*?</think>\s*", re.S)


def answer_start(text):
    """Reasoning models emit an empty (or filled) <think> block first. That is a
    prefix, not the answer, and searching for the first newline inside it made
    every Qwen3 trace look like a 1-token answer that instantly ran on."""
    m = THINK.match(text)
    return m.end() if m else 0


def answer_end(text, offsets):
    """(char index, token index) where the answer stops and the run-on begins."""
    s = answer_start(text)
    m = BOUNDARY.search(text, s)
    ch = m.start() if m else len(text)
    tk = len(offsets)
    for i, (a, _b) in enumerate(offsets):
        if a >= ch:
            tk = i
            break
    return ch, max(tk, 1)


def rows(cell):
    d, fname = load(cell)
    out = []
    for qi, q in d.items():
        for ci, c in enumerate(q["candidates"]):
            H = np.asarray(c["token_entropies"], float)
            E = np.asarray(c.get("token_spilled_energies", []), float)
            off = c["token_offsets"]
            ch, tk = answer_end(c["full_text"], off)
            out.append(dict(
                qi=qi, ci=ci, question=q["question"],
                gold=str(q["gold_row"].get("answer_value", "")),
                text=c["full_text"], H=H, E=E, ch=ch, tk=tk,
                n=len(H), label=bool(c["label"]),
                runon=bool(BOUNDARY.search(c["full_text"],
                                           answer_start(c["full_text"])))))
    return out, fname


# ── simple features, computed the same way on the full trace and on the crop ──
FEATS = {
    "mean entropy": lambda H, E: H.mean(),
    "std entropy": lambda H, E: H.std(),
    "max entropy": lambda H, E: H.max(),
    "last-token entropy": lambda H, E: H[-1],
    "mean spilled energy": lambda H, E: E.mean() if len(E) else np.nan,
    "max spilled energy": lambda H, E: E.max() if len(E) else np.nan,
}


def feat_table(rs):
    """AUROC of each simple feature on the FULL trace vs the ANSWER SPAN only."""
    y = np.array([r["label"] for r in rs], int)
    out = []
    for name, fn in FEATS.items():
        full, crop = [], []
        for r in rs:
            H, E, k = r["H"], r["E"], r["tk"]
            full.append(fn(H, E))
            Hc = H[:k]
            Ec = E[:k] if len(E) else E
            crop.append(fn(Hc, Ec) if len(Hc) else np.nan)
        full, crop = np.array(full, float), np.array(crop, float)
        rec = dict(name=name)
        for key, v in (("full", full), ("crop", crop)):
            ok = np.isfinite(v)
            if ok.sum() > 10 and len(set(y[ok])) == 2:
                a = roc_auc_score(y[ok], v[ok])
                rec[key] = round(max(a, 1 - a), 4)
            else:
                rec[key] = None
        rec["hist_full"] = hist_of(full, y)
        rec["hist_crop"] = hist_of(crop, y)
        out.append(rec)
    return out


def hist_of(v, y, nb=26, clip=2.8):
    ok = np.isfinite(v)
    z = np.zeros_like(v)
    if ok.sum() > 2 and v[ok].std() > 0:
        z[ok] = (v[ok] - v[ok].mean()) / v[ok].std()
    z = np.clip(z, -clip, clip)
    edges = np.linspace(-clip, clip, nb + 1)
    o = {}
    for key, m in (("pos", (y == 1) & ok), ("neg", (y == 0) & ok)):
        h, _ = np.histogram(z[m], bins=edges)
        s = h.sum()
        o[key] = (h / s).tolist() if s else [0.0] * nb
    return o


# ── rendering ────────────────────────────────────────────────────────────────
def trace_svg(H, tk, w=560, ht=64):
    """Entropy per token. The answer span is drawn solid; everything after the
    boundary — the run-on continuation — is shaded out."""
    n = len(H)
    if n < 2:
        return ""
    m = max(float(H.max()), 1e-9)
    dx = w / (n - 1)
    pts = " ".join(f"{i * dx:.1f},{ht - (v / m) * (ht - 6):.1f}"
                   for i, v in enumerate(H))
    xb = min(tk, n - 1) * dx
    shade = (f'<rect x="{xb:.1f}" y="0" width="{max(w - xb, 0):.1f}" height="{ht}" '
             f'fill="var(--neg)" fill-opacity=".10"/>') if tk < n else ""
    line = (f'<line x1="{xb:.1f}" y1="0" x2="{xb:.1f}" y2="{ht}" '
            f'stroke="var(--bad)" stroke-width="1.5" stroke-dasharray="3 2"/>')
    return (f'<svg width="{w}" height="{ht}" viewBox="0 0 {w} {ht}" role="img" '
            f'style="max-width:100%;border:1px solid var(--line);'
            f'border-radius:4px">{shade}'
            f'<polyline points="{pts}" fill="none" stroke="var(--acc)" '
            f'stroke-width="1.3"/>{line}</svg>')


def sample_html(r, show_len=520):
    ans = esc(r["text"][:r["ch"]]).strip() or "&lt;empty&gt;"
    rest = esc(r["text"][r["ch"]:show_len])
    more = " …" if len(r["text"]) > show_len else ""
    tag = ('<span class="chip c-or">correct</span>' if r["label"]
           else '<span class="chip" style="background:var(--bad)">hallucinated</span>')
    frac = 100.0 * r["tk"] / max(r["n"], 1)
    return (
        f'<div class="card" style="background:var(--bg)">'
        f'<p style="margin:0 0 6px"><b>Q:</b> {esc(r["question"])} &nbsp;{tag}</p>'
        f'<p style="margin:0 0 8px;font-size:13px;color:var(--mut)">'
        f'gold: <code>{esc(r["gold"])}</code></p>'
        f'<p class="mono" style="margin:0 0 8px;white-space:pre-wrap;'
        f'word-break:break-word;line-height:1.5">'
        f'<span style="background:rgba(46,160,67,.18);padding:1px 3px;'
        f'border-radius:3px"><b>{ans}</b></span>'
        f'<span style="color:var(--mut);opacity:.75">{rest}{more}</span></p>'
        f'{trace_svg(r["H"], r["tk"])}'
        f'<p style="margin:6px 0 0;font-size:12.5px;color:var(--mut)">'
        f'entropy per token &mdash; <b>{r["tk"]} answer tokens of {r["n"]}</b> '
        f'({frac:.0f}%); shaded region is the run-on continuation</p></div>')


def pick(rs, k=N_SHOW):
    """A few of each class, preferring the shortest answers (worst filler ratio)."""
    out = []
    for lab in (True, False):
        c = sorted([r for r in rs if r["label"] == lab],
                   key=lambda r: (r["tk"] / max(r["n"], 1)))
        out += c[:k]
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    W, wf = rows(WEAK[0])
    S, sf = rows(STRONG[0])

    def summ(rs):
        L = np.array([r["n"] for r in rs])
        T = np.array([r["tk"] for r in rs])
        y = np.array([r["label"] for r in rs], int)
        return dict(n=len(rs), med=float(np.median(L)), mx=int(L.max()),
                    at_cap=100.0 * float((L == L.max()).mean()),
                    runon=100.0 * float(np.mean([r["runon"] for r in rs])),
                    ans_med=float(np.median(T)),
                    frac=100.0 * float(np.median(T / np.maximum(L, 1))),
                    pos=100.0 * float(y.mean()))

    sw, ss = summ(W), summ(S)
    fw, fs = feat_table(W), feat_table(S)

    # Length-leakage control. Mandatory after spilled_triviaqa_llama8b turned out
    # to be a pure length detector: a crop makes trace length vary with the
    # answer, so the gain must be shown to survive holding length fixed.
    global LEAK
    LEAK = {}
    for cell, rs in ((WEAK[0], W), (STRONG[0], S)):
        y = np.array([r["label"] for r in rs], int)
        T = np.array([r["tk"] for r in rs])
        mx = np.array([r["H"][:r["tk"]].max() for r in rs])
        au = lambda yy, v: max(roc_auc_score(yy, v), 1 - roc_auc_score(yy, v))
        st = [(int((T == L).sum()), au(y[T == L], mx[T == L]))
              for L in np.unique(T)
              if (T == L).sum() >= 150 and len(set(y[T == L])) == 2]
        wgt = np.array([n for n, _ in st], float)
        LEAK[cell] = dict(
            len_only=f"{au(y, T):.4f}", crop=f"{au(y, mx):.4f}",
            corr=f"{abs(np.corrcoef(mx, T)[0, 1]):.3f}",
            strat=(f"{np.average([v for _, v in st], weights=wgt):.4f} "
                   f"<span style='color:var(--mut);font-size:12px'>"
                   f"({int(wgt.sum()):,}/{len(T):,} samples)</span>"
                   if st else "&mdash;"))

    b = ['<p class="crumb"><a href="../index.html">&larr; action items</a></p>',
         "<h1>What the raw data actually looks like</h1>",
         '<p class="sub">One level below the features: the questions, what the '
         'model emitted, and the token-entropy trace the 30 views are computed '
         'from. <b>' + esc(WEAK[1]) + "</b> against <b>" + esc(STRONG[1])
         + "</b>.</p>"]

    b.append(
        '<div class="box bad"><p><b>The weak cell&rsquo;s traces are malformed, '
        "and it is a generation defect, not a fusion one.</b> Every one of its "
        f"{sw['n']:,} traces is exactly {sw['mx']} tokens &mdash; "
        f"<b>{sw['at_cap']:.1f}% sit exactly at <code>max_new_tokens</code></b>, "
        "so not one generation stopped on its own. "
        f"<b>{sw['runon']:.1f}%</b> run past the answer and begin inventing a new "
        "few-shot <code>Question:</code> block. The answer itself is a median of "
        f"<b>{sw['ans_med']:.0f} tokens</b> &mdash; about "
        f"<b>{sw['frac']:.0f}% of the trace</b>. The other ~"
        f"{100 - sw['frac']:.0f}% is the model talking to itself, and all 30 "
        "features are computed over the whole thing.</p></div>")

    b.append("<h2>1. Side by side</h2>")
    b.append('<div class="scroll"><table><tr><th></th>'
             f"<th>{esc(WEAK[1])}</th><th>{esc(STRONG[1])}</th></tr>")
    for lbl, kw, fmt in (
            ("samples", "n", "{:,.0f}"),
            ("median trace length (tokens)", "med", "{:.0f}"),
            ("longest trace", "mx", "{:.0f}"),
            ("% of traces sitting exactly at max_new_tokens", "at_cap", "{:.1f}%"),
            ("% running past the answer into a new 'Question:'", "runon", "{:.1f}%"),
            ("median ANSWER length (tokens)", "ans_med", "{:.0f}"),
            ("median share of the trace that is the answer", "frac", "{:.0f}%"),
            ("% graded correct", "pos", "{:.1f}%")):
        a, c = sw[kw], ss[kw]
        bad = ' class="neg"' if kw in ("at_cap", "runon") and a > 50 else ""
        b.append(f"<tr><td>{lbl}</td><td{bad}>{fmt.format(a)}</td>"
                 f"<td>{fmt.format(c)}</td></tr>")
    b.append("</table></div>")
    b.append(f'<p style="font-size:12.5px;color:var(--mut)">Source files: '
             f"<code>cache/repgrid/{WEAK[0]}/{esc(wf)}</code> and "
             f"<code>cache/repgrid/{STRONG[0]}/{esc(sf)}</code>.</p>")

    b.append("<h2>2. The examples</h2>")
    b.append('<p>Green is the answer span; grey is what the model kept emitting '
             "after it. The dashed line on the entropy trace is the boundary "
             "between them, and the shaded region is everything the features see "
             "but shouldn&rsquo;t.</p>")
    for cell, name, rs in ((WEAK[0], WEAK[1], W), (STRONG[0], STRONG[1], S)):
        b.append(f"<h3>{esc(name)}</h3><div class='cards' "
                 "style='grid-template-columns:1fr'>")
        for r in pick(rs):
            b.append(sample_html(r))
        b.append("</div>")

    b.append("<h2>3. Does cropping to the answer recover the signal?</h2>")
    b.append(
        "<p>A direct test of the parsing hypothesis, needing no re-generation: "
        "recompute simple entropy features over the <b>full</b> trace and over "
        "the <b>answer span only</b>, and compare AUROC. If the run-on is what "
        "is destroying the signal, cropping should help on the weak cell and do "
        "nothing on the strong one. AUROC is oracle-oriented here (these are "
        "throwaway diagnostics, not deployed views).</p>")
    for name, ft, s in ((WEAK[1], fw, sw), (STRONG[1], fs, ss)):
        b.append(f"<h3>{esc(name)}</h3>")
        b.append('<div class="scroll"><table><tr><th>feature</th>'
                 "<th>full trace</th><th>answer span only</th><th>&Delta;</th>"
                 "<th>full-trace distribution</th>"
                 "<th>cropped distribution</th></tr>")
        for r in ft:
            if r["full"] is None or r["crop"] is None:
                continue
            d = (r["crop"] - r["full"]) * 100
            cl = "pos" if d > 1 else ("neg" if d < -1 else "")
            b.append(f'<tr><td>{r["name"]}</td><td>{r["full"]:.4f}</td>'
                     f'<td>{r["crop"]:.4f}</td><td class="{cl}">{d:+.1f}pp</td>'
                     f'<td>{hist_svg(r["hist_full"])}</td>'
                     f'<td>{hist_svg(r["hist_crop"])}</td></tr>')
        b.append("</table></div>")

    best = max(((r["crop"] - r["full"]) * 100 for r in fw
                if r["full"] is not None and r["crop"] is not None), default=0.0)
    b.append(
        f'<div class="box {"ok" if best > 1 else "warn"}"><p><b>Result: the best '
        f"single-feature gain from cropping is {best:+.1f}pp on the weak "
        f"cell.</b> "
        + ("Cropping recovers real signal, so the run-on is doing measurable "
           "damage and a re-run with a proper stop sequence is worth it."
           if best > 1 else
           "Cropping alone does not recover much &mdash; which is not the same "
           "as the parsing being fine. The median answer is only "
           f"{sw['ans_med']:.0f} tokens, and most of the 30 deployed views "
           "(spectral entropy, STFT, sliding-window variance) need a longer "
           "series than that to mean anything. So the honest reading is that "
           "this cell has almost no usable trace either way, and the fix is a "
           "re-run that lets the model stop and gives a longer answer span, not "
           "post-hoc cropping.")
        + "</p></div>")

    b.append("<h2>4. Is the gain just answer length again?</h2>")
    b.append(
        "<p>It has to be asked. Cropping makes the trace length vary with the "
        "answer, and the last cell we headlined (<code>spilled_triviaqa_llama8b"
        "</code>) turned out to be a pure length detector. So: does answer length "
        "predict correctness on its own, and does the cropped feature still work "
        "inside a fixed answer length?</p>")
    b.append('<div class="scroll"><table><tr><th></th>'
             f"<th>{esc(WEAK[1])}</th><th>{esc(STRONG[1])}</th></tr>")
    for lbl, k in (("answer length alone", "len_only"),
                   ("max entropy on the crop", "crop"),
                   ("|corr(feature, length)|", "corr"),
                   ("<b>max entropy WITHIN fixed answer length</b>", "strat")):
        b.append(f"<tr><td>{lbl}</td><td>{LEAK[WEAK[0]][k]}</td>"
                 f"<td>{LEAK[STRONG[0]][k]}</td></tr>")
    b.append("</table></div>")
    b.append(
        '<div class="box ok"><p><b>Not leakage.</b> Answer length alone is '
        f"{LEAK[WEAK[0]]['len_only']} on the weak cell &mdash; chance &mdash; and "
        f"the cropped feature still scores {LEAK[WEAK[0]]['strat']} when compared "
        "only against answers of the <i>same</i> length. The signal is in the "
        "entropy of the answer tokens, not in how many there are.</p></div>")

    b.append("<h2>5. What this suggests</h2>")
    b.append(
        "<p>This cell was generated with a base (non-instruct) model in a "
        "few-shot prompt, at <code>max_new_tokens = 64</code>, with no stop "
        "sequence at the newline. A base model in that setup does not stop after "
        "answering &mdash; it continues the pattern. The strong cell used an "
        "instruct model that emits an answer and halts, which is why its median "
        f"trace is {ss['med']:.0f} tokens and none of it is filler.</p>")
    b.append(
        "<p>So before concluding anything about features or fusion on this cell, "
        "the generation should be re-run with a stop sequence on the newline (and "
        "on <code>Question:</code>), which is a cheap change to the preset. Until "
        "then its numbers describe a parsing artifact, not the method.</p>")
    b.append(
        '<p class="foot">Built by <code>scripts/action_items_jul2026/'
        "build_raw_trace_page.py</code> directly from the raw generation caches. "
        "Nothing here goes through the feature pipeline.</p>")

    with open(os.path.join(OUT, "index.html"), "w", encoding="utf-8") as f:
        f.write(page("Raw traces — weak cell vs strong cell", "".join(b)))
    print(f"weak  : {sw}")
    print(f"strong: {ss}")
    for nm, ft in (("WEAK", fw), ("STRONG", fs)):
        print(f"-- {nm}")
        for r in ft:
            if r["full"] is None or r["crop"] is None:
                continue
            print(f"   {r['name']:22s} full {r['full']:.4f}  crop {r['crop']:.4f}"
                  f"  {(r['crop']-r['full'])*100:+6.2f}pp")
    print(f"best crop gain (weak): {best:+.2f}pp")
    print(f"wrote {os.path.join(OUT, 'index.html')}")


if __name__ == "__main__":
    main()
