import os
import sys

# Configure output to support UTF-8 encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

filepath = r"C:\Users\omris\TAU\hallucination_detection\scratch\history_keyword_matches.md"
with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

print(f"Total length of match file: {len(text)} characters")

# Find all headers in the match file (e.g. ## Step ...)
headers = [line for line in text.splitlines() if line.startswith("## Step")]
print("Headers in matches:")
for h in headers:
    print(h)
