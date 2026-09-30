import os
import re
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

directory = r"C:\Users\omris\TAU\hallucination_detection\papers\extracted"
files = [
    "gradients-with-respect-to-semantics-preserving-embeddings.md",
    "how-do-llms-compute-verbal-confidence.md",
    "inference-time-conformal-reasoning.md",
    "haloprobe-bayesian-detection.md",
    "revis-sparse-latent-steering.md",
    "adaptive-residual-update-steering.md",
    "agentic-confidence-calibration.md"
]

output_path = r"C:\Users\omris\TAU\hallucination_detection\scratch\detailed_paper_tables.md"

with open(output_path, "w", encoding="utf-8") as out:
    for f in files:
        path = os.path.join(directory, f)
        out.write("="*80 + "\n")
        out.write(f"FILE: {f}\n")
        if not os.path.exists(path):
            out.write("  Missing!\n")
            continue
        with open(path, "r", encoding="utf-8", errors="ignore") as file:
            content = file.read()
            
        lines = content.splitlines()
        
        # Find where "Table" is mentioned (case insensitive)
        hits = []
        for i, l in enumerate(lines):
            if re.search(r"\bTable\s+\d+\b", l, re.IGNORECASE) or "Table " in l or "table " in l:
                hits.append(i)
                
        out.write(f"Found {len(hits)} occurrences of Table.\n")
        for idx in hits[:15]:
            out.write(f"--- Hit at line {idx+1}:\n")
            start = max(0, idx-5)
            end = min(len(lines), idx+25) # print 30 lines total
            for j in range(start, end):
                out.write(f"  {j+1}: {lines[j]}\n")
            out.write("\n")
        out.write("="*80 + "\n\n")

print("Wrote detailed tables to scratch/detailed_paper_tables.md")
