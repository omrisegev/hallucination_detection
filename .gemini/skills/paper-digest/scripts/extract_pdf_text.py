#!/usr/bin/env python3
"""Mechanical PDF -> markdown text extraction for the paper-digest skill.

Usage:
    python extract_pdf_text.py "papers/EPR.pdf"
    python extract_pdf_text.py "papers/EPR.pdf" --force
    python extract_pdf_text.py "papers/EPR.pdf" --out-dir papers/extracted

Zero-judgment step: pulls raw text per page via PyMuPDF and writes it to a committable
markdown file. Does not summarize or interpret anything - that's the digest step
(see ../SKILL.md Step 3).
"""
import argparse
import datetime
import re
import sys
from pathlib import Path

MIN_PAGE_CHARS = 20  # below this, treat the page as an empty/scanned-image page


def slugify(name: str) -> str:
    slug = name.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    return slug[:60].rstrip("-")


def extract(pdf_path: Path, out_dir: Path, force: bool) -> Path:
    slug = slugify(pdf_path.stem)
    out_path = out_dir / f"{slug}.md"

    if out_path.exists() and not force:
        print(f"already extracted -> {out_path} (skipping; pass --force to regenerate)")
        return out_path

    try:
        import fitz  # PyMuPDF
    except ImportError:
        print(
            "PyMuPDF is not installed in this environment. Install it with:\n"
            "    pip install pymupdf\n"
            "then re-run this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    doc = fitz.open(pdf_path)
    n_pages = doc.page_count
    empty_pages = []
    body_parts = []

    for i, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if len(text) < MIN_PAGE_CHARS:
            empty_pages.append(i)
            body_parts.append(f"## Page {i}\n\n[EMPTY — possible scanned image, no text layer]\n")
        else:
            body_parts.append(f"## Page {i}\n\n{text}\n")

    doc.close()

    out_dir.mkdir(parents=True, exist_ok=True)
    header = (
        "---\n"
        f"source_pdf: {pdf_path.as_posix()}\n"
        f"slug: {slug}\n"
        f"pages: {n_pages}\n"
        f"extracted_on: {datetime.date.today().isoformat()}\n"
        "---\n\n"
        f"# {pdf_path.stem}\n\n"
    )
    out_path.write_text(header + "\n".join(body_parts), encoding="utf-8")

    print(f"extracted {n_pages} pages -> {out_path}")
    if empty_pages:
        print(
            f"WARNING: {len(empty_pages)} near-empty page(s) (possible scanned images, "
            f"no OCR performed): {empty_pages}"
        )
    print(
        "Next: read the extract and write papers/digests/"
        f"{slug}.md using skills/paper-digest/references/digest_template.md, "
        "then update papers/index.md."
    )
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf_path", type=Path, help="Path to the source PDF")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("papers/extracted"),
        help="Directory to write the extracted markdown into (default: papers/extracted)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Regenerate even if the extract already exists",
    )
    args = parser.parse_args()

    if not args.pdf_path.exists():
        print(f"No such file: {args.pdf_path}", file=sys.stderr)
        sys.exit(1)

    extract(args.pdf_path, args.out_dir, args.force)


if __name__ == "__main__":
    main()
