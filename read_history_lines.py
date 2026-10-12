import sys

# Configure output to support UTF-8 encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

filepath = r"C:\Users\omris\TAU\hallucination_detection\HISTORY.md"
with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

print(f"Total lines in HISTORY.md: {len(lines)}")
# Let's print from line 1800 to 1880
for i in range(1790, min(1890, len(lines))):
    print(f"{i+1}: {lines[i]}", end="")
