import os
import re

directory = r"C:\Users\omris\TAU\hallucination_detection"
keywords = [
    "Frequency-Aware Attention", "evidence drop", "reasoning manifold",
    "Gradients with Respect to Semantics", "SemGrad", "HybridGrad",
    "Verbal Confidence", "Conformal Reasoning", "Factuality Control",
    "GAUSS", "HaloProbe", "REVIS", "Residual-Update Steering",
    "Agentic Confidence Calibration"
]

results = {kw: [] for kw in keywords}

for root, dirs, files in os.walk(directory):
    if ".git" in dirs:
        dirs.remove(".git")
    for file in files:
        if file.endswith((".md", ".txt", ".py", ".ipynb")):
            path = os.path.join(root, file)
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    for kw in keywords:
                        if re.search(r"\b" + re.escape(kw) + r"\b", content, re.IGNORECASE):
                            results[kw].append(path)
            except Exception as e:
                pass

for kw, files in results.items():
    if files:
        print(f"Keyword: {kw}")
        for f in files:
            print(f"  - {f}")
