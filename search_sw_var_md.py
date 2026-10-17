import os
import re
import sys

# Configure output to support UTF-8 encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

root_dir = r"C:\Users\omris\TAU\hallucination_detection"
pattern = re.compile(r"sw_var_peak", re.IGNORECASE)

matches = []
for dirpath, _, filenames in os.walk(root_dir):
    if ".git" in dirpath or ".venv" in dirpath or "node_modules" in dirpath or "brain" in dirpath:
        continue
    for filename in filenames:
        if filename.endswith(".md"):
            filepath = os.path.join(dirpath, filename)
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    for i, line in enumerate(f):
                        if pattern.search(line):
                            matches.append((filepath, i+1, line.strip()))
            except Exception as e:
                pass

print(f"Found {len(matches)} matches in Markdown files:")
for filepath, line_num, content in matches:
    print(f"{os.path.basename(filepath)}:{line_num}: {content[:120]}")
