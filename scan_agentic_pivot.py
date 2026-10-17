import re
import sys

# Configure output to support UTF-8 encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

filepath = r"C:\Users\omris\TAU\hallucination_detection\HISTORY.md"
with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
    content = f.read()

# Split by "### Step "
steps = content.split("### Step ")

keywords = ["agentic", "agent", "processbench", "sw_var", "pivot", "bocpd", "conformal"]

print(f"Total steps: {len(steps)}")
print("Scanning for agentic/agent pivots and step-level plans...")

matching_steps = []
for step in steps:
    header = step.splitlines()[0] if step.splitlines() else "Unknown"
    # check for exact combinations
    has_agentic = re.search(r"\bagentic\b|\bagent\b", step, re.IGNORECASE)
    has_pivot = re.search(r"\bpivot\b", step, re.IGNORECASE)
    has_processbench = re.search(r"\bprocessbench\b|\bmr-gsm8k\b", step, re.IGNORECASE)
    
    if has_agentic or has_pivot or has_processbench:
        matching_steps.append((header, step))

print(f"Found {len(matching_steps)} matching steps.")
for header, text in matching_steps:
    print(f"Header: {header}")
    # print context of matches
    lines = text.splitlines()
    for l in lines[:10]:
        print(f"  {l}")
    print("...")
    print("-" * 50)
