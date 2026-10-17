import sys

# Configure output to support UTF-8 encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

filepath = r"C:\Users\omris\TAU\hallucination_detection\HISTORY.md"
with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

for i in range(4830, 4915):
    if i < len(lines):
        print(f"{i+1}: {lines[i]}", end="")
