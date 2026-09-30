import subprocess
import os

files = [
    "Gradients_with_Respect_to_Semantics_Preserving_Embeddings.pdf",
    "How_do_LLMs_Compute_Verbal_Confidence.pdf",
    "Inference_Time_Conformal_Reasoning.pdf",
    "HaloProbe_Bayesian_Detection.pdf",
    "REVIS_Sparse_Latent_Steering.pdf",
    "Adaptive_Residual_Update_Steering.pdf",
    "Agentic_Confidence_Calibration.pdf"
]

papers_dir = r"C:\Users\omris\TAU\hallucination_detection\papers"

for f in files:
    path = os.path.join(papers_dir, f)
    print(f"Extracting: {f}")
    
    cmd = [
        "python", 
        r"C:\Users\omris\TAU\hallucination_detection\skills\paper-digest\scripts\extract_pdf_text.py",
        path
    ]
    
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            print(f"  Successfully extracted: {f}")
            print(f"  Output: {res.stdout.strip()}")
        else:
            print(f"  Failed: {res.stderr}\nOutput: {res.stdout}")
    except Exception as e:
        print(f"  Error running extraction: {e}")

print("Finished extracting batch.")
