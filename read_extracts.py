import os
import re

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

output_path = r"C:\Users\omris\TAU\hallucination_detection\scratch\extracts_summary.md"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w", encoding="utf-8") as out:
    for f in files:
        path = os.path.join(directory, f)
        out.write("="*60 + "\n")
        out.write(f"FILE: {f}\n")
        if not os.path.exists(path):
            out.write("  Does not exist!\n")
            continue
            
        with open(path, "r", encoding="utf-8", errors="ignore") as file:
            content = file.read()
            
        # Print the first 80 lines
        lines = content.splitlines()[:80]
        out.write("FIRST 80 LINES:\n")
        for i, l in enumerate(lines):
            out.write(f"  {i+1}: {l}\n")
            
        # Search for Tables
        out.write("\nFOUND TABLES / METRICS (approx):\n")
        table_lines = [l for l in content.splitlines() if "|" in l]
        if table_lines:
            out.write(f"  Found {len(table_lines)} table-like lines. Sample of first 30:\n")
            for l in table_lines[:30]:
                out.write(f"    {l}\n")
        else:
            out.write("  No tables found.\n")
            
        # Also find section titles (lines with "#" or "##")
        out.write("\nFOUND HEADERS:\n")
        headers = [l for l in content.splitlines() if l.startswith("#")]
        for h in headers[:30]:
            out.write(f"  {h}\n")
            
        out.write("="*60 + "\n\n")

print("Wrote summary to scratch/extracts_summary.md")
