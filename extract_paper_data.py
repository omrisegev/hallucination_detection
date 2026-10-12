import os
import re
import sys

# Configure output to support UTF-8 encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

directory = r"C:\Users\omris\TAU\hallucination_detection\papers\extracted"
files = [
    ("gradients-with-respect-to-semantics-preserving-embeddings.md", "gradients"),
    ("how-do-llms-compute-verbal-confidence.md", "verbal-confidence"),
    ("inference-time-conformal-reasoning.md", "conformal-reasoning"),
    ("haloprobe-bayesian-detection.md", "haloprobe"),
    ("revis-sparse-latent-steering.md", "revis"),
    ("adaptive-residual-update-steering.md", "rudder"),
    ("agentic-confidence-calibration.md", "agentic-calibration")
]

for filename, slug in files:
    path = os.path.join(directory, filename)
    print("="*60)
    print(f"SLUG: {slug}")
    if not os.path.exists(path):
        print("  Missing!")
        continue
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
        
    lines = text.splitlines()
    abstract = ""
    
    for i, l in enumerate(lines[:100]):
        if "Abstract" in l:
            abs_lines = []
            for j in range(i+1, i+40):
                if j < len(lines) and not lines[j].strip().startswith("1."):
                    abs_lines.append(lines[j])
                else:
                    break
            abstract = " ".join(abs_lines).strip()
            break
            
    print(f"Abstract preview:\n{abstract[:400]}")
    
    print("Evaluation tables/metrics:")
    table_lines = []
    for i, l in enumerate(lines):
        if "|" in l:
            table_lines.append(l)
    print(f"  Total table lines: {len(table_lines)}")
    # Print the most table-like part (up to 30 lines)
    for l in table_lines[:30]:
        print(f"    {l}")
    print("="*60)
