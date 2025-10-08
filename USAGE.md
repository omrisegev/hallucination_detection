# Usage Guide

## Quick Start

### 1. Installation

```bash
cd hallucination_detection
pip install -e .
```

### 2. Basic Usage

```python
from hallucination_detection.orchestrator.pipeline import HallucinationDetectionPipeline
from hallucination_detection.core.metaqa import MetaQAGenerator
from hallucination_detection.core.answers import AnswerCollector
from hallucination_detection.core.embedding import EmbeddingAnalyzer
from hallucination_detection.core.weak_votes import WeakClassifierBuilder
from hallucination_detection.core.ensemble_matrix import EnsembleMatrix
from hallucination_detection.core.lsml import LatentSpectralMetaLearner

# Your LLM client (must have .complete(prompt, temperature) method)
llm_client = YourLLMClient()

# Initialize pipeline
pipeline = HallucinationDetectionPipeline(
    metaqa=MetaQAGenerator(llm_client, num_perturbations=12),
    collector=AnswerCollector(llm_client, cache_path=".cache"),
    embedder=EmbeddingAnalyzer(),
    weak_builder=WeakClassifierBuilder(method='gaussian_mixture'),
    ensemble=EnsembleMatrix(num_classifiers=10),
    lsml=LatentSpectralMetaLearner()
)

# Calibration
for question in calibration_questions:
    pipeline.process_question(question)

pipeline.train_meta()

# Inference
result = pipeline.predict("Your question here?")
print(result['label'])  # 'faithful' or 'hallucinated'
print(result['score'])  # confidence score
```

### 3. Run Demo

```bash
# With mock LLM (no API key needed)
python examples/demo_pipeline.py

# With OpenAI (requires OPENAI_API_KEY)
export OPENAI_API_KEY='your-key-here'
python examples/openai_example.py
```

## Architecture

### Core Modules

1. **metaqa.py**: Question perturbation generator
   - Template-based paraphrasing
   - Role-based prompting
   - Synonym substitution
   - Temperature sampling

2. **answers.py**: Answer collector with caching
   - SHA-based cache keys
   - Async batching support
   - Retry logic

3. **embedding.py**: Embedding analysis
   - Sentence-BERT embeddings
   - KMeans/Agglomerative clustering
   - 11 diversity features

4. **weak_votes.py**: Weak classifier builder
   - Otsu thresholding
   - Gaussian Mixture Models
   - Percentile-based thresholds
   - Binary votes {-1, +1}

5. **ensemble_matrix.py**: Vote matrix storage
   - Z ∈ {-1,+1}^(m×n)
   - Add/retrieve instances
   - Statistics computation

6. **lsml.py**: Latent Spectral Meta-Learner
   - Covariance computation
   - Score matrix from 2×2 determinants
   - Spectral clustering for K groups
   - Parameter estimation (ψ, η)
   - ML prediction (Eq. 18)

### Orchestrator

**pipeline.py**: End-to-end orchestration
- process_question(): Full calibration path
- train_meta(): Fit L-SML
- predict(): Inference on new questions

### Evaluation

**metrics.py**: Performance metrics
- Accuracy, precision, recall, F1
- ROC AUC
- Calibration curves
- Error analysis

**visualize.py**: Plotting utilities
- Calibration curves
- Confusion matrices
- Feature distributions
- Ensemble matrix heatmap

## Configuration

Edit `config/defaults.yaml` to customize:
- Number of perturbations
- Clustering parameters
- Weak classifier methods
- L-SML hyperparameters (K_max, sampling)

## LLM Client Interface

Your LLM client must implement:

```python
class YourLLMClient:
    def complete(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate completion for prompt."""
        # Your implementation
        return answer_string
```

For async support (optional):

```python
async def complete_async(self, prompt: str, temperature: float = 0.7) -> str:
    """Async completion."""
    return answer_string
```

## Advanced Usage

### Custom Feature Extraction

Extend `EmbeddingAnalyzer` to add custom features:

```python
class CustomEmbedder(EmbeddingAnalyzer):
    def analyze(self, answers):
        features = super().analyze(answers)
        # Add custom features
        features['my_feature'] = self._compute_my_feature(answers)
        return features
```

### Ensemble Pruning

Reduce computational cost by keeping only top classifiers:

```python
# After training
important_indices = [0, 1, 5, 8, 12]  # Based on your criteria
ensemble.prune_classifiers(important_indices)
```

### Batch Prediction

```python
questions = ["Q1?", "Q2?", "Q3?"]
results = pipeline.batch_predict(questions)
```

## Testing

```bash
pytest tests/ -v
```

## Troubleshooting

### Issue: "sentence-transformers not found"
```bash
pip install sentence-transformers
```

### Issue: Low accuracy
- Increase calibration data size (50+ questions recommended)
- Tune K_max parameter
- Try different clustering methods
- Add more perturbations per question

### Issue: Slow performance
- Enable L-SML sampling: `enable_sampling=True`
- Reduce number of perturbations
- Use smaller embedding model
- Enable answer caching

## Mathematical Details

### L-SML Algorithm

**Input**: Vote matrix Z ∈ {-1,+1}^(m×n)

**Output**: Predicted label for new votes

**Steps**:
1. Compute covariance Ĉ = (1/n) Z_centered @ Z_centered^T
2. Build score matrix Ŝ from 2×2 determinants
3. Spectral clustering on Ŝ → K groups
4. Estimate ψ_i^α (sensitivity) and η_i^α (specificity) per group
5. Estimate group priors Pr(α_k | Y)
6. Prediction: ŷ = argmax_y Π_k [Σ_α Pr(α_k=α|Y=y) · Π_{i∈G_k} Pr(f_i|α_k=α)]

### Feature → Vote Mapping

Each continuous feature is thresholded to produce binary votes:
- **+1**: Faithful (consistent answers)
- **-1**: Hallucinated (inconsistent answers)

Feature polarity:
- High intra-cluster similarity → +1
- High pairwise distance → -1
- Large cluster share → +1
- High entropy → -1

## Citation

If you use this framework, please cite the relevant papers on:
- Dawid-Skene model
- Spectral meta-learning
- Latent variable models for dependent classifiers

## Support

For issues or questions:
1. Check this guide
2. Review examples/
3. Run tests to verify installation
4. Check docstrings in code
