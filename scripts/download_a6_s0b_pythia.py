#!/usr/bin/env python3
"""Download and authenticate the exact prompt-only Pythia input for A6-S0b."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import ssl
import sys
import urllib.request


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils import a6_s0b_input as contract  # noqa: E402


DEFAULT_OUT = REPO / "local_cache" / "a6_s0b_pythia_c4fc8d5"


def download(out: Path) -> dict[str, object]:
    import certifi
    from huggingface_hub import hf_hub_download

    out.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        contract.OFFICIAL_API_URL,
        headers={"Accept": "application/json", "User-Agent": "a6-s0b-input-v1"},
    )
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(request, timeout=60, context=context) as response:
        official = response.read()
    projection = contract.validate_official_tree(official)
    evidence = out / "PYTHIA_OFFICIAL_TREE.json"
    if evidence.exists() and evidence.read_bytes() != official:
        raise RuntimeError("existing official-tree evidence differs from exact response")
    evidence.write_bytes(official)

    rows = []
    for spec in contract.SELECTED_FILES:
        path = Path(hf_hub_download(
            repo_id=contract.REPOSITORY,
            revision=contract.REVISION,
            filename=spec.path,
            local_dir=out,
        ))
        if path.resolve().parent != out.resolve():
            raise RuntimeError(f"download escaped output root: {spec.path}")
        rows.append(contract.verify_selected_bytes(spec, path.read_bytes()))
    result = {
        "repository": contract.REPOSITORY,
        "revision": contract.REVISION,
        "official_projection": projection,
        "files": rows,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    download(args.out)


if __name__ == "__main__":
    main()
