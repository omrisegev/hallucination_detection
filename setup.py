from setuptools import setup, find_packages

setup(
    name="spectral_utils",
    version="0.1.0",
    description="Shared utilities for the hallucination detection spectral pipeline",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
    ],
    extras_require={
        "benchmark": [
            "torch",
            "matplotlib",
        ],
        "inference": [
            "torch",
            "transformers>=4.40",
            "accelerate",
            "datasets",
            "bitsandbytes",
        ],
        # Pinned because the adapter relies on the 0.2.0 probability/alignment
        # API and the package is still alpha.  This extra is intentionally
        # separate from GPU inference: the dependency-fusion experiment can run
        # entirely from cached feature matrices.
        "dependency-experiment": [
            "deem==0.2.0",
        ],
    },
)
