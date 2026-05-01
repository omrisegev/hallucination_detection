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
        "inference": [
            "torch",
            "transformers>=4.40",
            "accelerate",
            "datasets",
            "bitsandbytes",
        ],
    },
)
