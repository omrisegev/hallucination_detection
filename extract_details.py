import os
import re

directory = r"C:\Users\omris\TAU\hallucination_detection\papers\extracted"
files = {
    "gradients-with-respect-to-semantics-preserving-embeddings.md": ["Abstract", "Introduction", "Experiments", "Experiment", "Dataset", "Model", "Baseline"],
    "how-do-llms-compute-verbal-confidence.md": ["Abstract", "Introduction", "Experiments", "Dataset", "Model", "Baseline"],
    "inference-time-conformal-reasoning.md": ["Abstract", "Introduction", "Experiments", "Dataset", "Model", "Baseline"],
    "haloprobe-bayesian-detection.md": ["Abstract", "Introduction", "Experiments", "Dataset", "Model", "Baseline"],
    "revis-sparse-latent-steering.md": ["Abstract", "Introduction", "Experiments", "Dataset", "Model", "Baseline"],
    "adaptive-residual-update-steering.md": ["Abstract", "Introduction", "Experiments", "Dataset", "Model", "Baseline"],
    "agentic-confidence-calibration.md": ["Abstract", "Introduction", "Experiments", "Dataset", "Model", "Baseline"]
}

output_path = r"C:\Users\omris\TAU\hallucination_detection\scratch\detailed_paper_info.md"

with open(output_path, "w", encoding="utf-8") as out:
    for f, keywords in files.items():
        path = os.path.join(directory, f)
        out.write("="*80 + "\n")
        out.write(f"FILE: {f}\n")
        if not os.path.exists(path):
            out.write("  Does not exist!\n")
            continue
            
        with open(path, "r", encoding="utf-8", errors="ignore") as file:
            content = file.read()
            
        # Find lines with tables or scores
        out.write("### TABLES & SCORES IN FILE:\n")
        lines = content.splitlines()
        table_lines = []
        in_table = False
        for i, l in enumerate(lines):
            if "|" in l:
                table_lines.append(f"{i+1}: {l}")
        
        if table_lines:
            out.write(f"Found {len(table_lines)} table lines.\n")
            # Write first 50 table lines
            for tl in table_lines[:50]:
                out.write(f"  {tl}\n")
        else:
            out.write("No table lines found.\n")
            
        # Let's search for some text around "datasets", "baselines", "AUC"
        out.write("### SEARCH HITS FOR DATASETS/MODELS/BASELINES/AUC:\n")
        for i, l in enumerate(lines):
            # check if line contains keywords (case insensitive)
            for kw in ["dataset", "dataset", "baselines", "baseline", "auc", "auroc", "evaluation", "result", "compare"]:
                if re.search(r"\b" + re.escape(kw) + r"\b", l, re.IGNORECASE):
                    # Write the line and surrounding 2 lines
                    start = max(0, i-2)
                    end = min(len(lines), i+3)
                    out.write(f"--- Hit for '{kw}' (Lines {start+1}-{end}):\n")
                    for j in range(start, end):
                        out.write(f"  {j+1}: {lines[j]}\n")
                    out.write("\n")
                    break # only print once per line
                    
        out.write("="*80 + "\n\n")

print("Wrote detailed info to scratch/detailed_paper_info.md")
