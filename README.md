# Hallucination Detection via L-SML

A modular Python framework for detecting hallucinations in LLM answers using:
- **MetaQA-style perturbations** for answer diversity
- **Embedding & clustering** to create weak binary classifiers
- **Latent Spectral Meta-Learner (L-SML)** for unsupervised ensemble meta-classification

## Architecture

The system follows a strict separation of concerns:

```
hallucination_detection/
├── core/               # Core detection modules
│   ├── metaqa.py      # Question perturbation generator
│   ├── answers.py     # LLM answer collector with caching
│   ├── embedding.py   # Embedding-based feature extraction
│   ├── weak_votes.py  # Unsupervised weak classifier generation
│   ├── ensemble_matrix.py  # Vote matrix Z storage
│   └── lsml.py        # Latent Spectral Meta-Learner (full math)
├── orchestrator/       # Pipeline orchestration
│   └── pipeline.py    # HallucinationDetectionPipeline
├── eval/              # Evaluation & visualization
│   ├── metrics.py
│   └── visualize.py
├── config/            # Configuration
│   └── defaults.yaml
└── examples/          # Usage examples
    └── demo_pipeline.py
```

## Key Features

### L-SML Mathematical Foundation

The Latent Spectral Meta-Learner extends the Dawid-Skene conditional independence model to handle **dependent classifiers** via latent binary variables {α_k}. 

- **Model**: Classifiers are grouped; within each group they're conditionally independent given α_k
- **Prediction**: Maximum likelihood under learned generative model (Eq. 18)
- **Complexity**: O(m^4) score matrix construction with optional sampling for efficiency

### Modular Design

Each module is:
- Independently importable and testable
- Type-annotated with clear interfaces
- Side-effect free at import time
- Configurable via dependency injection

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from hallucination_detection.orchestrator.pipeline import HallucinationDetectionPipeline
from hallucination_detection.core.metaqa import MetaQAGenerator
from hallucination_detection.core.answers import AnswerCollector
from hallucination_detection.core.embedding import EmbeddingAnalyzer
from hallucination_detection.core.weak_votes import WeakClassifierBuilder
from hallucination_detection.core.ensemble_matrix import EnsembleMatrix
from hallucination_detection.core.lsml import LatentSpectralMetaLearner

# Initialize pipeline
pipeline = HallucinationDetectionPipeline(
    metaqa=MetaQAGenerator(llm_client=your_llm, num_perturbations=12),
    collector=AnswerCollector(llm_client=your_llm, cache_path=".cache"),
    embedder=EmbeddingAnalyzer(embedder_name="sentence-transformers/all-mpnet-base-v2"),
    weak_builder=WeakClassifierBuilder(method="gaussian_mixture"),
    ensemble=EnsembleMatrix(num_classifiers=10),
    lsml=LatentSpectralMetaLearner()
)

# Calibration phase (unsupervised)
for question in calibration_questions:
    pipeline.process_question(question)

pipeline.train_meta()

# Inference
result = pipeline.predict("Who discovered oxygen and when?")
print(result)  # {label: 'faithful'/'hallucinated', score: float, explanation: dict}
```

## Requirements

- Python 3.10+
- numpy, scipy, scikit-learn
- sentence-transformers
- PyYAML

## References

This implementation follows the dependent classifiers with latent groups model, generalizing the Dawid-Skene conditional independence framework. Key mathematical components:

- Score matrix Ŝ from 2×2 covariance determinants
- Spectral clustering for group discovery
- Two-stage parameter estimation
- ML prediction via Eq. 18

## License

MIT

