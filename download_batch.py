import subprocess
import time
import os

ids = {
    "2605.04638": "Gradients_with_Respect_to_Semantics_Preserving_Embeddings.pdf",
    "2603.17839": "How_do_LLMs_Compute_Verbal_Confidence.pdf",
    "2606.08831": "Inference_Time_Conformal_Reasoning.pdf",
    "2604.06165": "HaloProbe_Bayesian_Detection.pdf",
    "2602.11824": "REVIS_Sparse_Latent_Steering.pdf",
    "2511.10292": "Adaptive_Residual_Update_Steering.pdf",
    "2601.15778": "Agentic_Confidence_Calibration.pdf"
}

output_dir = r"C:\Users\omris\TAU\hallucination_detection\papers"
os.makedirs(output_dir, exist_ok=True)

for arxiv_id, filename in ids.items():
    output_path = os.path.join(output_dir, filename)
    print(f"Downloading {arxiv_id} -> {output_path}")
    
    cmd = [
        r"C:\Users\DELL\.local\bin\uv.exe", "run", 
        r"C:\Users\DELL\.gemini\config\plugins\science\skills\literature_search_arxiv\scripts\download_paper.py",
        "--id", arxiv_id,
        "--format", "pdf",
        "--output", output_path
    ]
    
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            print(f"  Successfully downloaded: {filename}")
        else:
            print(f"  Failed: {res.stderr}\nOutput: {res.stdout}")
    except Exception as e:
        print(f"  Error running download: {e}")
        
    time.sleep(3) # Rate limit

print("Finished downloading batch.")
