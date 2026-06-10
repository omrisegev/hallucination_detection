"""
render_html.py — Generate HTML results table from the local results archive.

Modes:
    --latest        Read results/latest.csv → one table, all cells
    --compare       Read results/archive.jsonl → side-by-side comparison across runs
    --out FILE      Output path (default: results/report.html)

Examples:
    python scripts/render_html.py --latest
    python scripts/render_html.py --compare
    python scripts/render_html.py --compare --out results/compare_report.html
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

REPO_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_DIR, 'results')
ARCHIVE_PATH = os.path.join(RESULTS_DIR, 'archive.jsonl')
LATEST_CSV   = os.path.join(RESULTS_DIR, 'latest.csv')

DOMAIN_ORDER = ['MATH500', 'GSM8K', 'GPQA', 'RAG', 'QA']

CSS = """
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1200px; margin: 40px auto; padding: 0 20px; color: #222; }
  h1   { font-size: 1.4em; border-bottom: 2px solid #444; padding-bottom: 8px; }
  h2   { font-size: 1.1em; margin-top: 2em; color: #444; }
  table { border-collapse: collapse; width: 100%; margin-bottom: 2em; font-size: 0.9em; }
  th   { background: #2c3e50; color: #fff; padding: 8px 12px; text-align: left; }
  td   { padding: 6px 12px; border-bottom: 1px solid #ddd; }
  tr:nth-child(even) { background: #f8f8f8; }
  .auc-high  { color: #1a7a1a; font-weight: bold; }
  .auc-mid   { color: #7a5a00; }
  .auc-low   { color: #999; }
  .domain-tag { display: inline-block; padding: 1px 6px; border-radius: 3px;
                font-size: 0.8em; font-weight: bold; }
  .tag-math500 { background:#dbeafe; color:#1e40af; }
  .tag-gsm8k   { background:#dcfce7; color:#166534; }
  .tag-gpqa    { background:#fef9c3; color:#854d0e; }
  .tag-rag     { background:#fce7f3; color:#9d174d; }
  .tag-qa      { background:#ede9fe; color:#6b21a8; }
  .meta { font-size: 0.8em; color: #666; margin-bottom: 1em; }
  .badge { display:inline-block; background:#e5e7eb; border-radius:3px;
           padding:1px 6px; font-family:monospace; font-size:0.85em; }
</style>
"""


def auc_class(auc):
    if auc >= 0.75:
        return 'auc-high'
    if auc >= 0.60:
        return 'auc-mid'
    return 'auc-low'


def domain_tag(domain):
    key = domain.lower()
    return f'<span class="domain-tag tag-{key}">{domain}</span>'


def fmt_auc(auc, ci_low, ci_high):
    cls = auc_class(auc)
    return f'<span class="{cls}">{100*auc:.1f}%</span> <small>[{ci_low:.3f},{ci_high:.3f}]</small>'


# ---------------------------------------------------------------------------
# Latest mode — single run table
# ---------------------------------------------------------------------------

def render_latest(out_path):
    if not os.path.exists(LATEST_CSV):
        print(f'ERROR: {LATEST_CSV} not found. Run run_lsml_local.py first.')
        sys.exit(1)

    with open(LATEST_CSV, newline='') as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print('latest.csv is empty.')
        sys.exit(1)

    run_id = rows[0].get('run_id', '')
    label  = rows[0].get('label', '')
    feats  = rows[0].get('features', '')

    html = [f'<!DOCTYPE html><html><head><meta charset="utf-8">'
            f'<title>L-SML Results — {label}</title>{CSS}</head><body>']
    html.append(f'<h1>L-SML v2 Results</h1>')
    html.append(f'<p class="meta">Run: <span class="badge">{run_id}</span> &nbsp; '
                f'Label: <span class="badge">{label}</span> &nbsp; '
                f'Features: <span class="badge">{feats}</span></p>')

    # Group by domain
    by_domain = defaultdict(list)
    for r in rows:
        by_domain[r['domain']].append(r)

    for domain in DOMAIN_ORDER:
        domain_rows = by_domain.get(domain, [])
        if not domain_rows:
            continue
        html.append(f'<h2>{domain_tag(domain)} &nbsp; {domain}</h2>')
        html.append('<table><tr><th>Model / Cell</th><th>AUROC</th><th>K</th><th>N</th><th>+correct / −wrong</th></tr>')
        for r in domain_rows:
            auc   = float(r['auroc'])
            ci_lo = float(r['ci_low'])
            ci_hi = float(r['ci_high'])
            html.append(
                f'<tr>'
                f'<td>{r["model_key"]}</td>'
                f'<td>{fmt_auc(auc, ci_lo, ci_hi)}</td>'
                f'<td>{r["K"]}</td>'
                f'<td>{r["n"]}</td>'
                f'<td>+{r["n_pos"]} / −{r["n_neg"]}</td>'
                f'</tr>'
            )
        html.append('</table>')

    beating = sum(1 for r in rows if float(r['auroc']) >= 0.5)
    html.append(f'<p class="meta">Total cells: {len(rows)} &nbsp;|&nbsp; '
                f'Beating chance (≥0.50): {beating}/{len(rows)}</p>')
    html.append('</body></html>')

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html))
    print(f'Saved -> {out_path}')


# ---------------------------------------------------------------------------
# Compare mode — all runs side by side
# ---------------------------------------------------------------------------

def load_archive():
    if not os.path.exists(ARCHIVE_PATH):
        print(f'ERROR: {ARCHIVE_PATH} not found.')
        sys.exit(1)
    runs = []
    with open(ARCHIVE_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                runs.append(json.loads(line))
    return runs


def render_compare(out_path):
    runs = load_archive()
    if not runs:
        print('archive.jsonl is empty.')
        sys.exit(1)

    # Collect all cell keys per domain
    all_keys = defaultdict(set)
    for run in runs:
        for domain, dres in run['results'].items():
            for key in dres:
                all_keys[domain.upper()].add(key)

    html = ['<!DOCTYPE html><html><head><meta charset="utf-8">'
            f'<title>L-SML Comparison — {len(runs)} runs</title>{CSS}</head><body>']
    html.append(f'<h1>L-SML v2 — Run Comparison ({len(runs)} runs)</h1>')

    # Run legend
    html.append('<table><tr><th>#</th><th>Run ID</th><th>Label</th><th>Features</th></tr>')
    for i, run in enumerate(runs):
        html.append(
            f'<tr><td>{i+1}</td>'
            f'<td><span class="badge">{run["run_id"]}</span></td>'
            f'<td>{run["label"]}</td>'
            f'<td><span class="badge">{", ".join(run["features"])}</span></td></tr>'
        )
    html.append('</table>')

    run_headers = ''.join(
        f'<th>#{i+1} {run["label"][:12]}</th>' for i, run in enumerate(runs)
    )

    for domain in DOMAIN_ORDER:
        domain_key = domain.lower()
        keys = sorted(all_keys.get(domain, []))
        if not keys:
            continue
        html.append(f'<h2>{domain_tag(domain)} &nbsp; {domain}</h2>')
        html.append(f'<table><tr><th>Model / Cell</th>{run_headers}</tr>')
        for key in keys:
            cells = [f'<td>{key}</td>']
            for run in runs:
                dres = run['results'].get(domain_key, run['results'].get(domain, {}))
                res  = dres.get(key)
                if res:
                    cells.append(f'<td>{fmt_auc(res["auroc"], res["ci_low"], res["ci_high"])}</td>')
                else:
                    cells.append('<td>—</td>')
            html.append(f'<tr>{"".join(cells)}</tr>')
        html.append('</table>')

    html.append('</body></html>')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html))
    print(f'Saved -> {out_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Render HTML results from L-SML local runs.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--latest',  action='store_true', help='Render the most recent run from latest.csv')
    group.add_argument('--compare', action='store_true', help='Render a comparison of all runs from archive.jsonl')
    parser.add_argument('--out', default='', help='Output HTML file path')
    args = parser.parse_args()

    if args.latest:
        out = args.out or os.path.join(RESULTS_DIR, 'report_latest.html')
        render_latest(out)
    else:
        out = args.out or os.path.join(RESULTS_DIR, 'report_compare.html')
        render_compare(out)


if __name__ == '__main__':
    main()
