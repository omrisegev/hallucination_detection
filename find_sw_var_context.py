import re
import sys

# Configure output to support UTF-8 encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

filepath = r"C:\Users\omris\TAU\hallucination_detection\HISTORY.md"
with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

lines = text.splitlines()

# Search for sw_var_peak occurrences
pattern = re.compile(r"sw_var_peak", re.IGNORECASE)
hits = []
for i, l in enumerate(lines):
    if pattern.search(l):
        hits.append(i)

print(f"Found {len(hits)} hits in HISTORY.md:")
for idx in hits:
    print(f"Line {idx+1}: {lines[idx]}")
    # print context (3 lines before, 10 lines after)
    print("  CONTEXT:")
    start = max(0, idx-3)
    end = min(len(lines), idx+12)
    for j in range(start, end):
        prefix = "-> " if j == idx else "   "
        print(f"  {prefix}{j+1}: {lines[j]}")
    print("-"*60)
