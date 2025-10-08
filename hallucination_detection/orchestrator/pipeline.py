"""
Hallucination Detection Pipeline

Orchestrates the end-to-end hallucination detection workflow:
1. Question perturbation (MetaQA)
2. Answer collection
3. Embedding analysis
4. Weak classifier generation
5. Ensemble matrix accumulation
6. L-SML meta-learning
7. Prediction

This pipeline implements the full MetaQA → Embedding → L-SML framework
with clean separation between calibration and inference phases.
"""

from typing import Dict, List, Optional
import numpy as np

from ..core.metaqa import MetaQAGenerator
from ..core.answers import AnswerCollector
from ..core.embedding import EmbeddingAnalyzer
from ..core.weak_votes import WeakClassifierBuilder
from ..core.ensemble_matrix import EnsembleMatrix
from ..core.lsml import LatentSpectralMetaLearner


class HallucinationDetectionPipeline:
    """
    End-to-end hallucination detection pipeline.
    
    This class orchestrates all components of the detection framework,
    providing a unified interface for calibration and inference.
    
    Workflow:
        Calibration Phase:
            1. For each calibration question:
                - Generate perturbations (MetaQA)
                - Collect answers (AnswerCollector)
                - Analyze embeddings (EmbeddingAnalyzer)
                - Build weak votes (WeakClassifierBuilder)
                - Add to ensemble matrix
            2. Train meta-learner on ensemble matrix (L-SML)
        
        Inference Phase:
            1. Process new question through same pipeline
            2. Predict using trained L-SML
    
    Attributes:
        metaqa: Question perturbation generator
        collector: Answer collector with caching
        embedder: Embedding analyzer
        weak_builder: Weak classifier builder
        ensemble: Ensemble matrix
        lsml: L-SML meta-learner
        
    Example:
        >>> pipeline = HallucinationDetectionPipeline(...)
        >>> for q in calibration_questions:
        ...     pipeline.process_question(q)
        >>> pipeline.train_meta()
        >>> result = pipeline.predict("New question?")
        >>> print(result['label'])  # 'faithful' or 'hallucinated'
    """
    
    def __init__(
        self,
        metaqa: MetaQAGenerator,
        collector: AnswerCollector,
        embedder: EmbeddingAnalyzer,
        weak_builder: WeakClassifierBuilder,
        ensemble: EnsembleMatrix,
        lsml: LatentSpectralMetaLearner,
        verbose: bool = True,
    ):
        """
        Initialize HallucinationDetectionPipeline.
        
        Args:
            metaqa: Question perturbation generator
            collector: Answer collector
            embedder: Embedding analyzer
            weak_builder: Weak classifier builder
            ensemble: Ensemble matrix
            lsml: L-SML meta-learner
            verbose: Print progress messages
        """
        self.metaqa = metaqa
        self.collector = collector
        self.embedder = embedder
        self.weak_builder = weak_builder
        self.ensemble = ensemble
        self.lsml = lsml
        self.verbose = verbose
        
        # Track processed questions
        self.processed_questions: List[str] = []
        self.feature_history: List[Dict[str, float]] = []
        
        self._is_trained = False
    
    def process_question(
        self, 
        question: str, 
        question_id: Optional[str] = None
    ) -> Dict:
        """
        Process a question through the full pipeline.
        
        Args:
            question: Question text
            question_id: Optional unique identifier
            
        Returns:
            Dictionary with features and votes
            
        Pipeline:
            1. Generate perturbations
            2. Collect answers
            3. Analyze embeddings → features
            4. Build weak classifier votes
            5. Add to ensemble matrix
        """
        if question_id is None:
            question_id = f"q_{len(self.processed_questions)}"
        
        if self.verbose:
            print(f"\n[Pipeline] Processing: {question_id}")
        
        # Step 1: Generate perturbations
        if self.verbose:
            print("[Pipeline] Generating perturbations...")
        perturbations = self.metaqa.generate(question)
        
        # Step 2: Collect answers
        if self.verbose:
            print(f"[Pipeline] Collecting {len(perturbations)} answers...")
        answers = self.collector.collect(perturbations)
        
        # Step 3: Analyze embeddings
        if self.verbose:
            print("[Pipeline] Analyzing embeddings...")
        features = self.embedder.analyze(answers)
        
        # Step 4: Build weak classifier votes
        if self.verbose:
            print("[Pipeline] Building weak classifier votes...")
        votes = self.weak_builder.build_votes(features)
        
        # Step 5: Add to ensemble matrix
        self.ensemble.add_instance(question_id, votes)
        
        # Track
        self.processed_questions.append(question_id)
        self.feature_history.append(features)
        
        if self.verbose:
            print(f"[Pipeline] Processed {question_id} successfully")
            print(f"[Pipeline] Features: {list(features.keys())}")
            print(f"[Pipeline] Votes: {votes.tolist()}")
        
        return {
            'question_id': question_id,
            'perturbations': perturbations,
            'answers': answers,
            'features': features,
            'votes': votes.tolist(),
        }
    
    def train_meta(self):
        """
        Train the L-SML meta-learner on accumulated ensemble matrix.
        
        This method should be called after processing calibration questions.
        It performs two key steps:
            1. Calibrate weak classifier thresholds
            2. Fit L-SML on the ensemble matrix
        """
        if len(self.processed_questions) < 2:
            raise RuntimeError(
                "Need at least 2 processed questions for training. "
                f"Currently have {len(self.processed_questions)}."
            )
        
        if self.verbose:
            print(f"\n[Pipeline] Training meta-learner on {len(self.processed_questions)} questions...")
        
        # Step 1: Calibrate weak classifier thresholds
        if self.verbose:
            print("[Pipeline] Calibrating weak classifier thresholds...")
        self.weak_builder.calibrate(self.feature_history)
        
        # Step 2: Get ensemble matrix
        Z = self.ensemble.to_numpy()
        
        if self.verbose:
            print(f"[Pipeline] Ensemble matrix shape: {Z.shape}")
            stats = self.ensemble.get_statistics()
            print(f"[Pipeline] Ensemble stats: {stats}")
        
        # Step 3: Fit L-SML
        if self.verbose:
            print("[Pipeline] Fitting L-SML...")
        self.lsml.fit(Z)
        
        self._is_trained = True
        
        if self.verbose:
            print("[Pipeline] Training complete!")
            print(f"[Pipeline] Discovered {self.lsml.K} latent groups")
    
    def predict(self, question: str) -> Dict:
        """
        Predict hallucination for a new question.
        
        Args:
            question: Question text
            
        Returns:
            Dictionary with prediction results
            
        Pipeline:
            1. Process question (perturb → collect → embed → vote)
            2. Predict using trained L-SML
        """
        if not self._is_trained:
            raise RuntimeError(
                "Must call train_meta() before predict(). "
                "Process calibration questions and train first."
            )
        
        if self.verbose:
            print(f"\n[Pipeline] Predicting for: '{question}'")
        
        # Step 1: Generate perturbations
        perturbations = self.metaqa.generate(question)
        
        # Step 2: Collect answers
        answers = self.collector.collect(perturbations)
        
        # Step 3: Analyze embeddings
        features = self.embedder.analyze(answers)
        
        # Step 4: Build weak classifier votes
        votes = self.weak_builder.build_votes(features)
        
        # Step 5: Predict with L-SML
        prediction = self.lsml.predict(votes)
        
        # Add context
        prediction['question'] = question
        prediction['features'] = features
        prediction['votes'] = votes.tolist()
        prediction['answers'] = answers
        
        if self.verbose:
            print(f"[Pipeline] Prediction: {prediction['label']} (score: {prediction['score']:.3f})")
        
        return prediction
    
    def batch_predict(self, questions: List[str]) -> List[Dict]:
        """
        Predict hallucination for multiple questions.
        
        Args:
            questions: List of question texts
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for i, question in enumerate(questions):
            if self.verbose:
                print(f"\n[Pipeline] Batch prediction {i+1}/{len(questions)}")
            
            result = self.predict(question)
            results.append(result)
        
        return results
    
    def get_calibration_summary(self) -> Dict:
        """
        Get summary of calibration data.
        
        Returns:
            Dictionary with calibration statistics
        """
        if not self.processed_questions:
            return {'num_questions': 0}
        
        Z = self.ensemble.to_numpy()
        
        return {
            'num_questions': len(self.processed_questions),
            'num_classifiers': self.weak_builder.get_num_classifiers(),
            'ensemble_shape': Z.shape,
            'ensemble_stats': self.ensemble.get_statistics(),
            'is_trained': self._is_trained,
            'lsml_groups': self.lsml.K if self._is_trained else None,
        }
    
    def reset(self):
        """Reset pipeline to initial state (clears calibration data)."""
        self.ensemble.clear()
        self.processed_questions.clear()
        self.feature_history.clear()
        self._is_trained = False
        
        if self.verbose:
            print("[Pipeline] Reset complete")
