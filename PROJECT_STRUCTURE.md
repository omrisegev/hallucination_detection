# Project Structure

```
hallucination_detection/
│
├── README.md                      # Project overview and quick start
├── USAGE.md                       # Detailed usage guide
├── setup.py                       # Package installation configuration
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore patterns
│
├── config/
│   └── defaults.yaml              # Default configuration parameters
│
├── hallucination_detection/       # Main package
│   ├── __init__.py
│   │
│   ├── core/                      # Core detection modules
│   │   ├── __init__.py
│   │   ├── metaqa.py             # MetaQA question perturbation
│   │   ├── answers.py            # Answer collection with caching
│   │   ├── embedding.py          # Embedding analysis & clustering
│   │   ├── weak_votes.py         # Weak classifier generation
│   │   ├── ensemble_matrix.py    # Vote matrix Z storage
│   │   └── lsml.py               # Latent Spectral Meta-Learner
│   │
│   ├── orchestrator/              # Pipeline orchestration
│   │   ├── __init__.py
│   │   └── pipeline.py           # HallucinationDetectionPipeline
│   │
│   └── eval/                      # Evaluation & visualization
│       ├── __init__.py
│       ├── metrics.py            # Performance metrics
│       └── visualize.py          # Plotting utilities
│
├── examples/                      # Usage examples
│   ├── __init__.py
│   ├── demo_pipeline.py          # Full demo with mock LLM
│   └── openai_example.py         # OpenAI integration example
│
└── tests/                         # Unit tests
    ├── __init__.py
    └── test_all.py               # Test suite

```

## Module Hierarchy

```
User Code
    ↓
HallucinationDetectionPipeline (orchestrator/pipeline.py)
    ↓
    ├─→ MetaQAGenerator (core/metaqa.py)
    │       ↓
    │   LLM Client (user-provided)
    │
    ├─→ AnswerCollector (core/answers.py)
    │       ↓
    │   LLM Client + Cache
    │
    ├─→ EmbeddingAnalyzer (core/embedding.py)
    │       ↓
    │   sentence-transformers + clustering
    │
    ├─→ WeakClassifierBuilder (core/weak_votes.py)
    │       ↓
    │   Otsu/GMM/Percentile thresholding
    │
    ├─→ EnsembleMatrix (core/ensemble_matrix.py)
    │       ↓
    │   Vote matrix Z storage
    │
    └─→ LatentSpectralMetaLearner (core/lsml.py)
            ↓
        Covariance → Score Matrix → Spectral Clustering → ML Prediction
```

## Data Flow

### Calibration Phase
```
Question
  ↓
[MetaQAGenerator] → Perturbations (12 variants)
  ↓
[AnswerCollector] → Answers (12 responses)
  ↓
[EmbeddingAnalyzer] → Features (11 numeric values)
  ↓
[WeakClassifierBuilder] → Votes (m binary values ∈ {-1,+1})
  ↓
[EnsembleMatrix] → Store as column in Z matrix
  ↓
(Repeat for n questions)
  ↓
[L-SML.fit(Z)] → Learn K groups and parameters
```

### Inference Phase
```
New Question
  ↓
[MetaQAGenerator] → Perturbations
  ↓
[AnswerCollector] → Answers
  ↓
[EmbeddingAnalyzer] → Features
  ↓
[WeakClassifierBuilder] → Votes (m values)
  ↓
[L-SML.predict(votes)] → Label + Score
  ↓
Result: {'label': 'faithful'/'hallucinated', 'score': 0.0-1.0}
```

## Key Files by Purpose

### If you want to...

**Understand the algorithm:**
- `core/lsml.py` - Full L-SML implementation with math

**Generate question variants:**
- `core/metaqa.py` - Perturbation strategies

**Extract diversity features:**
- `core/embedding.py` - 11 embedding-based features

**Understand weak classifiers:**
- `core/weak_votes.py` - Unsupervised thresholding

**See end-to-end flow:**
- `orchestrator/pipeline.py` - Complete orchestration

**Run a demo:**
- `examples/demo_pipeline.py` - Runnable demo
- `examples/openai_example.py` - OpenAI integration

**Evaluate performance:**
- `eval/metrics.py` - Accuracy, calibration, etc.
- `eval/visualize.py` - Plots and charts

**Test the code:**
- `tests/test_all.py` - Unit tests

**Configure system:**
- `config/defaults.yaml` - All hyperparameters

## File Sizes (approximate)

- core/lsml.py: ~600 lines (most complex)
- orchestrator/pipeline.py: ~300 lines
- core/embedding.py: ~300 lines
- core/weak_votes.py: ~250 lines
- core/metaqa.py: ~200 lines
- core/answers.py: ~200 lines
- examples/demo_pipeline.py: ~250 lines

Total: ~3000 lines of well-documented Python code

## Dependencies

**Required:**
- numpy, scipy, scikit-learn (core ML)
- sentence-transformers (embeddings)
- PyYAML (config)

**Optional:**
- openai (for OpenAI integration)
- matplotlib (for visualization)
- pytest (for testing)

## Entry Points

1. **Command line demo:**
   ```bash
   python examples/demo_pipeline.py
   ```

2. **Programmatic usage:**
   ```python
   from hallucination_detection.orchestrator.pipeline import HallucinationDetectionPipeline
   pipeline = HallucinationDetectionPipeline(...)
   ```

3. **Tests:**
   ```bash
   pytest tests/
   ```
