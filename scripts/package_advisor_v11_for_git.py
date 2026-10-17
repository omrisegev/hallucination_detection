"""Publish a readable v11 delivery without changing the frozen local bundle."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
REL = Path("results/advisor_completion_2026-10-11")
SOURCE = ROOT / REL
SKIP = {"__pycache__", "mpl_cache", ".pytest_cache"}
BRANCH = "codex/advisor-v11-completion"
DELIVERY_IGNORE = """# Keep readable archive evidence despite the root's generic output ignores.
!**/*.svg
!**/*.png
!**/*.txt
!**/*_SUMMARY.json
!**/*_summary.json
!sources/research/.claude/
!sources/research/.claude/**
# Large raw/derived payloads and runtime caches always stay outside Git.
*.npz
*.pkl
__pycache__/
mpl_cache/
.pytest_cache/
"""

def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def remove_cache_links(text):
    removed = []
    def replace(match):
        url = urlsplit(match.group(1))
        if not url.scheme and not url.netloc and set(Path(unquote(url.path)).parts) & SKIP:
            removed.append(match.group(1))
            return '<span>' + match.group(2) + ' · runtime cache omitted from Git</span>'
        return match.group(0)
    return re.sub(r'<a href="([^"]+)">([^<]*)</a>', replace, text), removed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--repair-cache-links", action="store_true")
    parser.add_argument("--refresh-ignore", action="store_true")
    args = parser.parse_args()
    destroot = args.destination.resolve()
    assert destroot != ROOT.resolve(), "Never overwrite the analysis checkout"
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=destroot, text=True).strip()
    assert head.startswith("89045986"), "Unexpected publication base"
    dest = destroot / REL
    if args.refresh_ignore:
        (dest / ".gitignore").write_text(DELIVERY_IGNORE)
        shutil.copyfile(Path(__file__), destroot / "scripts/package_advisor_v11_for_git.py")
        manifest_path = dest / "GITHUB_DELIVERY_MANIFEST.json"
        receipt = json.loads(manifest_path.read_text())
        receipt["transformations"].append("scoped allowlist for readable figures/text/summary archives; binary exclusion retained")
        receipt["files"] = [{"path": str(f.relative_to(dest)), "size": f.stat().st_size,
                             "sha256": sha(f)} for f in sorted(dest.rglob("*"))
                            if f.is_file() and f != manifest_path]
        receipt["delivered_file_count"] = len(receipt["files"])
        receipt["delivered_bytes"] = sum(r["size"] for r in receipt["files"])
        manifest_path.write_text(json.dumps(receipt, indent=2) + "\n")
        print("Readable evidence allowlist refreshed; NPZ/PKL exclusions retained")
        return
    if args.repair_cache_links:
        reading_path = dest / "advisor_review_v11_github.html"
        reading, removed = remove_cache_links(reading_path.read_text())
        reading_path.write_text(reading)
        shutil.copyfile(Path(__file__), destroot / "scripts/package_advisor_v11_for_git.py")
        manifest_path = dest / "GITHUB_DELIVERY_MANIFEST.json"
        receipt = json.loads(manifest_path.read_text())
        receipt["omitted_runtime_cache_links"] = removed
        receipt["transformations"].append("non-scientific omitted-cache hyperlinks become explicit plain text")
        receipt["reading_html_sha256"] = sha(reading_path)
        receipt["files"] = [{"path": str(f.relative_to(dest)), "size": f.stat().st_size,
                             "sha256": sha(f)} for f in sorted(dest.rglob("*"))
                            if f.is_file() and f != manifest_path]
        receipt["delivered_file_count"] = len(receipt["files"])
        receipt["delivered_bytes"] = sum(r["size"] for r in receipt["files"])
        manifest_path.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps({"status": "CACHE_LINKS_REPAIRED", "links": removed}))
        return
    if dest.exists():
        raise RuntimeError("Use an empty destination; frozen artifacts must not be overwritten")
    dest.mkdir(parents=True)
    source_manifest = json.loads((SOURCE / "PACKAGE_MANIFEST.json").read_text())
    identities = {r["path"]: r for r in source_manifest["files"]}
    omitted = []
    for path in sorted(SOURCE.rglob("*")):
        if not path.is_file() or set(path.relative_to(SOURCE).parts) & SKIP:
            continue
        rel = str(path.relative_to(SOURCE))
        if path.suffix in {".npz", ".pkl"}:
            entry = identities[rel]
            omitted.append({"path": rel, "size": entry["size"], "sha256": entry["sha256"]})
            continue
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    # Preserve the exact analysis README before adding publication guidance.
    shutil.copyfile(dest / "README.md", dest / "README_ANALYSIS_SNAPSHOT.md")
    (dest / ".gitignore").write_text(DELIVERY_IGNORE)
    def anchor(rel):
        return "raw-" + hashlib.sha256(rel.encode()).hexdigest()[:12]
    raw_paths = {r["path"] for r in omitted}
    with (dest / "RAW_DATA_CATALOG.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, ["path", "size", "sha256"])
        writer.writeheader(); writer.writerows(omitted)
    catalog = ["# Large data retained outside Git\n",
               "This Git branch contains code, text evidence, saved fitted-model JSON, tables and figures. "
               "The complete local bundle additionally contains the NPZ/PKL files below. "
               "No omitted file is claimed available from Git alone.\n",
               "The original local directory is `/Users/osegev/Desktop/hallucination_detection/"+str(REL)+"/`. "
               "Downloaded input origins are in INPUT_MANIFEST.json and inputs/*.receipt.json. "
               "Generated scores/checkpoints do not have a verified cloud copy. Full replay requires "
               "the matching full bundle, or a fresh re-execution into a new output directory.\n"]
    for row in omitted:
        catalog.extend([f'<a id="{anchor(row["path"])}"></a>\n',
                        f'## {row["path"]}\n', f'- Bytes: {row["size"]}\n',
                        f'- SHA256: `{row["sha256"]}`\n'])
    (dest / "RAW_DATA_CATALOG.md").write_text("\n".join(catalog))
    original = (dest / "advisor_review_v11.html").read_text()
    rewrites = []
    def replace_link(match):
        href = match.group(1); url = urlsplit(href); rel = unquote(url.path)
        if not url.scheme and not url.netloc and rel in raw_paths:
            rewrites.append(rel)
            return 'href="RAW_DATA_CATALOG.md#' + anchor(rel) + '"'
        return match.group(0)
    reading = re.sub(r'href="([^"]+)"', replace_link, original)
    reading, cache_rewrites = remove_cache_links(reading)
    banner = ('<aside style="margin:20px;padding:16px;border:2px solid #c99a52;background:#fff7e8">'
              '<b>GitHub reading copy.</b> The scientific results and review are the frozen v11 analysis. '
              'Its original local/unpublished disclosures describe the analysis snapshot. '
              'This branch now carries the reading delivery; the Claude Artifact has not been updated. '
              'Large-data links lead to a hash catalog because NPZ/PKL payloads are retained outside Git. '
              '<a href="README.md">Publication and reproduction scope</a>.</aside>')
    reading = reading.replace("<body>", "<body>" + banner, 1)
    assert reading != original and "GitHub reading copy." in reading
    (dest / "advisor_review_v11_github.html").write_text(reading)
    (dest / "README.md").write_text(f'''# Advisor v11: GitHub reading delivery

Branch: `{BRANCH}`. This is a reading delivery of the independently reviewed
PRMBench analysis, plus the next-agent count-estimator prompt. The Claude
Artifact itself has not been replaced by this publication.

Start here:

- [GitHub reading HTML](advisor_review_v11_github.html): identical scientific
  content to v11, with a publication banner and large-data links redirected
  to the catalog. It embeds all figures and opens locally in a browser.
- [Original reviewed HTML](advisor_review_v11.html): preserved byte-for-byte;
  its relative binary-data links require the full local bundle.
- [All sixteen Claude assignments](TASK_STATUS.csv).
- [Hebrew findings](FINDINGS_HE.md) and [independent review](review/INDEPENDENT_REVIEW.md).
- [Next-agent experiment prompt](../../docs/experiments/PRMBENCH_COUNT_ESTIMATOR_AGENT_PROMPT_2026-10-12.md).
- [Large-file catalog](RAW_DATA_CATALOG.md).

## Integrity and reproduction

The original analysis was completed locally on 11/12 October 2026. Its
historical `PUBLICATION_STATUS.json`, task statuses and review receipts remain
unchanged and describe that snapshot. They do not describe the later Git
publication. `README_ANALYSIS_SNAPSHOT.md` preserves its original README.

`PACKAGE_MANIFEST.json` describes the complete local bundle, including binary
files intentionally absent from Git. `GITHUB_DELIVERY_MANIFEST.json` separately
hashes this reading delivery and identifies the presentation-only transformations.

All CSV/JSON results, fitted-model JSON, scientific SVG/PNG figures, archived
source code and protocols are included. NPZ/PKL payloads are outside Git.
Downloaded inputs retain inspected Drive paths, sizes and hashes; generated
large outputs have no verified cloud copy. **Git alone is not a full runnable
numerical bundle.** See RAW_DATA_CATALOG.md before attempting reproduction.

For full replay use the matching full local bundle and the original README's
identity-checked commands. Never weaken its checksum guards merely to reuse
a partial checkout. Fresh computation belongs in a new results directory.

The evidence remains internal validation on previously inspected data. The
one failed N11 arm remains undefined. No new GPU experiment or external
confirmation is claimed. The estimator prompt is proposed future work;
its experiments have not yet been run.
''')
    for rel in ["spectral_utils/advisor_controls.py", "spectral_utils/advisor_lr.py",
                "spectral_utils/advisor_statistics.py", "scripts/run_advisor_v11_lr.py",
                "scripts/package_advisor_v11_for_git.py",
                "docs/experiments/PRMBENCH_COUNT_ESTIMATOR_AGENT_PROMPT_2026-10-12.md"]:
        target = destroot / rel; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, target)
    # Only the v11 addendum, not unrelated dirty progress sections, is copied.
    progress = (ROOT / "PROGRESS.md").read_text()
    start = progress.index("**Advisor CPU completion addendum")
    end = progress.index("**Review addendum", start)
    target = destroot / "PROGRESS.md"; old = target.read_text()
    title, rest = old.split("\n", 1)
    target.write_text(title + "\n\n" + progress[start:end] + rest)
    history = (ROOT / "HISTORY.md").read_text()
    start = history.index("### Step 291 [advisor CPU completion]")
    with (destroot / "HISTORY.md").open("a") as stream:
        stream.write("\n" + history[start:])
    note = ("\n**Git publication follow-up (2026-10-12):** The reading delivery and "
            "next-agent prompt are published on `"+BRANCH+"`. The full binary "
            "bundle remains outside Git; see its RAW_DATA_CATALOG. This is not "
            "a Claude Artifact update or a new numerical experiment.\n")
    with target.open("a") as stream: stream.write(note)
    with (destroot / "HISTORY.md").open("a") as stream: stream.write(note)
    files = [{"path": str(f.relative_to(dest)), "size": f.stat().st_size,
              "sha256": sha(f)} for f in sorted(dest.rglob("*")) if f.is_file()]
    receipt = {"scope": "Git reading delivery; not complete numerical bundle",
               "branch": BRANCH, "base_commit": head,
               "source_html_sha256": sha(SOURCE / "advisor_review_v11.html"),
               "preserved_html_sha256": sha(dest / "advisor_review_v11.html"),
               "reading_html_sha256": sha(dest / "advisor_review_v11_github.html"),
               "transformations": ["README publication guidance; original README retained",
                                   "gitignore excludes all NPZ/PKL",
                                   "separate reading HTML banner and binary-link redirects",
                                   "non-scientific omitted-cache hyperlinks become explicit plain text"],
               "redirected_binary_links": sorted(set(rewrites)),
               "omitted_runtime_cache_links": cache_rewrites,
               "omitted_file_count": len(omitted), "omitted_bytes": sum(r["size"] for r in omitted),
               "delivered_file_count": len(files), "delivered_bytes": sum(r["size"] for r in files),
               "files": files}
    (dest / "GITHUB_DELIVERY_MANIFEST.json").write_text(json.dumps(receipt, indent=2) + "\n")
    assert receipt["source_html_sha256"] == receipt["preserved_html_sha256"]
    print(json.dumps({k: receipt[k] for k in ["delivered_file_count", "delivered_bytes", "omitted_file_count", "omitted_bytes", "source_html_sha256"]}, indent=2))

if __name__ == "__main__": main()
