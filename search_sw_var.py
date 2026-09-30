import os
import re

root_dir = r"C:\Users\omris\TAU\hallucination_detection"
pattern = re.compile(r"sw_var_peak|sw_var", re.IGNORECASE)

results = []
for dirpath, _, filenames in os.walk(root_dir):
    # skip hidden dirs or virtualenvs
    if ".git" in dirpath or ".venv" in dirpath or "node_modules" in dirpath or "brain" in dirpath:
        continue
    for filename in filenames:
        if filename.endswith(".py") or filename.endswith(".md") or filename.endswith(".json") or filename.endswith(".txt"):
            filepath = os.path.join(dirpath, filename)
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    for i, line in enumerate(f):
                        if pattern.search(line):
                            results.append((filepath, i+1, line.strip()))
            except Exception as e:
                pass

print(f"Found {len(results)} matches:")
for filepath, line_num, content in results[:100]:
    print(f"{filepath}:{line_num}: {content}")
