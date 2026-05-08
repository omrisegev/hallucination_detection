# Research Workflow Patterns

## Pattern: The "Pre-emption Check"
Use this when checking if a research idea or method is already published.
1.  Search for core keywords on Google Scholar / arXiv.
2.  Identify the top 3-5 most similar papers.
3.  Fetch their abstracts and key methodology sections.
4.  Create a "Differentiation Table" comparing our method vs. theirs on:
    - Input data axis (e.g., generation time vs. layer depth)
    - Feature type (e.g., spectral vs. geometric)
    - Goal (e.g., unsupervised vs. supervised)

## Pattern: The "Benchmark Audit"
Use this to understand a new dataset structure.
1.  Fetch the dataset's README from GitHub/HuggingFace.
2.  Identify the JSON schema for samples.
3.  Look for "standard evaluation scripts" to match their metric implementation.
4.  Check for "ground truth" caveats (e.g., multiple correct answers, noisy labels).
