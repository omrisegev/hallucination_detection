"""
Unit Tests for Hallucination Detection Framework

Run with: pytest tests/
"""

import pytest
import numpy as np


class TestMetaQAGenerator:
    """Tests for MetaQAGenerator."""
    
    def test_generate_perturbations(self):
        from hallucination_detection.core.metaqa import MetaQAGenerator
        
        generator = MetaQAGenerator(num_perturbations=5)
        perturbations = generator.generate("What is the capital of France?")
        
        assert len(perturbations) == 5
        assert all(isinstance(p, str) for p in perturbations)
        assert "What is the capital of France?" in perturbations  # Original included
    
    def test_temperature_schedule(self):
        from hallucination_detection.core.metaqa import MetaQAGenerator
        
        generator = MetaQAGenerator(num_perturbations=10, temperature_range=(0.3, 0.9))
        schedule = generator.get_temperature_schedule()
        
        assert len(schedule) == 10
        assert all(0.3 <= temp <= 0.9 for temp in schedule)


class TestAnswerCollector:
    """Tests for AnswerCollector."""
    
    def test_cache_key_generation(self):
        from hallucination_detection.core.answers import AnswerCollector
        
        class MockLLM:
            def complete(self, prompt, temperature=0.7):
                return "answer"
        
        collector = AnswerCollector(MockLLM(), cache_path=".cache/test")
        prompts = ["q1", "q2", "q3"]
        key1 = collector._compute_cache_key(prompts)
        key2 = collector._compute_cache_key(sorted(prompts))
        
        assert key1 == key2  # Order-independent


class TestEmbeddingAnalyzer:
    """Tests for EmbeddingAnalyzer."""
    
    def test_analyze_similar_answers(self):
        from hallucination_detection.core.embedding import EmbeddingAnalyzer
        
        analyzer = EmbeddingAnalyzer()
        answers = ["Paris", "Paris", "Paris is the capital"]
        features = analyzer.analyze(answers)
        
        assert 'intra_cluster_similarity' in features
        assert 'largest_cluster_share' in features
        assert features['intra_cluster_similarity'] > 0.7  # Similar answers
    
    def test_analyze_diverse_answers(self):
        from hallucination_detection.core.embedding import EmbeddingAnalyzer
        
        analyzer = EmbeddingAnalyzer()
        answers = ["Paris", "London", "Berlin", "Rome"]
        features = analyzer.analyze(answers)
        
        assert features['mean_pairwise_distance'] > 0  # Diverse answers


class TestWeakClassifierBuilder:
    """Tests for WeakClassifierBuilder."""
    
    def test_build_votes(self):
        from hallucination_detection.core.weak_votes import WeakClassifierBuilder
        
        builder = WeakClassifierBuilder(num_thresholds_per_feature=2)
        features = {
            'intra_cluster_similarity': 0.9,
            'mean_pairwise_distance': 0.2,
        }
        votes = builder.build_votes(features)
        
        assert isinstance(votes, np.ndarray)
        assert len(votes) > 0
        assert all(v in [-1, 1] for v in votes)
    
    def test_calibration(self):
        from hallucination_detection.core.weak_votes import WeakClassifierBuilder
        
        builder = WeakClassifierBuilder(method='percentile')
        feature_list = [
            {'feature_a': 0.1, 'feature_b': 0.8},
            {'feature_a': 0.5, 'feature_b': 0.6},
            {'feature_a': 0.9, 'feature_b': 0.4},
        ]
        builder.calibrate(feature_list)
        
        assert 'feature_a' in builder.thresholds
        assert 'feature_b' in builder.thresholds


class TestEnsembleMatrix:
    """Tests for EnsembleMatrix."""
    
    def test_add_instance(self):
        from hallucination_detection.core.ensemble_matrix import EnsembleMatrix
        
        ensemble = EnsembleMatrix(num_classifiers=10)
        votes = np.array([1, -1, 1, 1, -1, 1, -1, 1, 1, -1])
        
        ensemble.add_instance("q1", votes)
        
        assert ensemble.num_instances() == 1
        assert np.array_equal(ensemble.get_instance_votes("q1"), votes)
    
    def test_to_numpy(self):
        from hallucination_detection.core.ensemble_matrix import EnsembleMatrix
        
        ensemble = EnsembleMatrix(num_classifiers=5)
        ensemble.add_instance("q1", np.array([1, -1, 1, -1, 1]))
        ensemble.add_instance("q2", np.array([-1, 1, -1, 1, -1]))
        
        Z = ensemble.to_numpy()
        
        assert Z.shape == (5, 2)
        assert set(np.unique(Z)) <= {-1, 1}


class TestLatentSpectralMetaLearner:
    """Tests for L-SML."""
    
    def test_fit_and_predict(self):
        from hallucination_detection.core.lsml import LatentSpectralMetaLearner
        
        # Create synthetic vote matrix
        np.random.seed(42)
        m, n = 20, 50
        Z = np.random.choice([-1, 1], size=(m, n))
        
        lsml = LatentSpectralMetaLearner(k_max=5)
        lsml.fit(Z)
        
        assert lsml._is_fitted
        assert lsml.K >= 1
        assert len(lsml.c) == m
        
        # Predict
        f_vec = np.random.choice([-1, 1], size=m)
        result = lsml.predict(f_vec)
        
        assert 'label' in result
        assert result['label'] in ['faithful', 'hallucinated']
        assert 0 <= result['score'] <= 1
    
    def test_covariance_computation(self):
        from hallucination_detection.core.lsml import LatentSpectralMetaLearner
        
        Z = np.array([
            [1, 1, 1, -1],
            [1, 1, -1, -1],
            [-1, -1, 1, 1],
        ])
        
        lsml = LatentSpectralMetaLearner()
        C_hat = lsml._compute_covariance(Z)
        
        assert C_hat.shape == (3, 3)
        # Classifiers 0 and 1 should be positively correlated
        assert C_hat[0, 1] > 0


class TestHallucinationDetectionPipeline:
    """Tests for complete pipeline."""
    
    def test_pipeline_workflow(self):
        from hallucination_detection.orchestrator.pipeline import HallucinationDetectionPipeline
        from hallucination_detection.core.metaqa import MetaQAGenerator
        from hallucination_detection.core.answers import AnswerCollector
        from hallucination_detection.core.embedding import EmbeddingAnalyzer
        from hallucination_detection.core.weak_votes import WeakClassifierBuilder
        from hallucination_detection.core.ensemble_matrix import EnsembleMatrix
        from hallucination_detection.core.lsml import LatentSpectralMetaLearner
        
        # Mock LLM
        class MockLLM:
            def complete(self, prompt, temperature=0.7):
                return "Mock answer"
        
        # Initialize components
        metaqa = MetaQAGenerator(MockLLM(), num_perturbations=5)
        collector = AnswerCollector(MockLLM(), cache_path=".cache/test")
        embedder = EmbeddingAnalyzer()
        weak_builder = WeakClassifierBuilder()
        ensemble = EnsembleMatrix(weak_builder.get_num_classifiers())
        lsml = LatentSpectralMetaLearner(k_max=3)
        
        pipeline = HallucinationDetectionPipeline(
            metaqa, collector, embedder, weak_builder, ensemble, lsml, verbose=False
        )
        
        # Process calibration questions
        for i in range(5):
            pipeline.process_question(f"Question {i}")
        
        # Train
        pipeline.train_meta()
        
        # Predict
        result = pipeline.predict("New question")
        
        assert 'label' in result
        assert 'score' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
