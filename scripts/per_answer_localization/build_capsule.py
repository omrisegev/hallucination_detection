"""Export only the CPU feature-audit code, with exact source hashes.

The capsule has an empty package initializer to avoid importing unrelated model
loaders. Scientific modules are copied byte-for-byte; no features are rewritten.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil

REPO = Path(__file__).resolve().parents[2]
FILES = (
    "spectral_utils/feature_utils.py", "spectral_utils/window_localization.py",
    "scripts/per_answer_localization/feasibility.py",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error("capsule destination already exists; use a new versioned path")
    tree = ast.parse((REPO / "spectral_utils/token_feature_views.py").read_text(encoding="utf-8"))
    broad = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "BROAD_TOKEN_VIEWS" for t in node.targets):
            broad = ast.literal_eval(node.value)
    if broad is None or len(broad) != 28:
        raise RuntimeError("canonical token schema changed")
    manifest = {}
    for name in FILES:
        source = REPO / name
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        manifest[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    initializer = output / "spectral_utils/__init__.py"
    initializer.write_text('"""Isolated CPU window-feature capsule; scientific sources are hash-bound."""\n', encoding="utf-8")
    manifest["spectral_utils/__init__.py"] = hashlib.sha256(initializer.read_bytes()).hexdigest()
    schema = {"stream_names": ["trace_length_series", *broad], "schema_source": "spectral_utils/token_feature_views.py",
              "schema_source_sha256": hashlib.sha256((REPO / "spectral_utils/token_feature_views.py").read_bytes()).hexdigest()}
    (output / "stream_schema.json").write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    manifest["stream_schema.json"] = hashlib.sha256((output / "stream_schema.json").read_bytes()).hexdigest()
    (output / "CAPSULE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
