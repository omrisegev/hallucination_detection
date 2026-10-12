import re
import sys

# Configure output to support UTF-8 encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

filepath = r"C:\Users\omris\TAU\hallucination_detection\HISTORY.md"
with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
    content = f.read()

# Let's split by Steps: "### Step "
steps = content.split("### Step ")
print(f"Total steps found: {len(steps)}")

keywords = ["sw_var", "agentic", "pivot", "conformal", "step-level", "processbench", "mr-gsm8k", "change-point", "bocpd", "localiz"]

matches = []
for step in steps:
    # check if step contains any of the keywords
    found_kws = [kw for kw in keywords if re.search(r"\b" + re.escape(kw) + r"\b", step, re.IGNORECASE)]
    if found_kws:
        # get the step header (first line)
        lines = step.splitlines()
        header = lines[0] if lines else "Unknown Header"
        # Let's collect the step's text
        matches.append((header, found_kws, step))

print(f"Found {len(matches)} steps containing keywords:")
for header, kws, text in matches:
    print(f"- Step {header} (Matched: {kws})")
    # print first 5 lines of text
    lines = text.splitlines()
    for l in lines[:5]:
        print(f"    {l}")
    print()

# Write the detailed matches to a scratch file so we can view them in depth
output_path = r"C:\Users\omris\TAU\hallucination_detection\scratch\history_keyword_matches.md"
with open(output_path, "w", encoding="utf-8") as f:
    f.write("# History Matches for Step-Level & Agentic Pivot\n\n")
    for header, kws, text in matches:
        f.write(f"## Step {header}\n")
        f.write(f"**Matched Keywords**: {', '.join(kws)}\n\n")
        f.write(text)
        f.write("\n\n" + "="*80 + "\n\n")

print(f"Wrote detailed matches to {output_path}")
