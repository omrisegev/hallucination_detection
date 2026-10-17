import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

filepath = r"C:\Users\omris\TAU\hallucination_detection\scratch\history_keyword_matches.md"
with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
    content = f.read()

# Let's split by "## Step "
steps = content.split("## Step ")

target_steps = ["32-B", "32-C", "40", "78"]

for step in steps:
    header = step.splitlines()[0] if step.splitlines() else ""
    for target in target_steps:
        if header.startswith(target):
            print("="*80)
            print(f"STEP: {header}")
            print(step)
            print("="*80 + "\n")
