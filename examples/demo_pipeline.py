"""
Demo: Hallucination Detection Pipeline

This example demonstrates the full MetaQA → Embedding → L-SML pipeline
for detecting hallucinations in LLM answers.

Workflow:
1. Define a simple LLM client (or use a real one)
2. Initialize all pipeline components
3. Calibrate on a set of questions
4. Make predictions on new questions
5. Analyze results

Note: This demo uses a mock LLM for demonstration purposes.
      Replace MockLLMClient with your actual LLM client (OpenAI, etc.)
"""

import numpy as np
from pathlib import Path

# Import pipeline components
from hallucination_detection.core.metaqa import MetaQAGenerator
from hallucination_detection.core.answers import AnswerCollector
from hallucination_detection.core.embedding import EmbeddingAnalyzer
from hallucination_detection.core.weak_votes import WeakClassifierBuilder
from hallucination_detection.core.ensemble_matrix import EnsembleMatrix
from hallucination_detection.core.lsml import LatentSpectralMetaLearner
from hallucination_detection.orchestrator.pipeline import HallucinationDetectionPipeline


# ============================================================================
# Mock LLM Client (Replace with your actual LLM)
# ============================================================================

class MockLLMClient:
    """
    Mock LLM client for demonstration.
    
    This client generates synthetic answers:
    - For "consistent" questions: returns similar answers
    - For "hallucinated" questions: returns diverse/inconsistent answers
    
    Replace this with your actual LLM client (OpenAI, local model, etc.)
    """
    
    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)
        
        # Define consistent vs hallucinated behavior
        self.knowledge_base = {
            'paris': 'Paris is the capital of France.',
            'oxygen': 'Oxygen was discovered by Carl Wilhelm Scheele in 1772 and independently by Joseph Priestley in 1774.',
            'water': 'Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at sea level.',
            'einstein': 'Albert Einstein was born in 1879 in Ulm, Germany.',
            'photosynthesis': 'Photosynthesis is the process by which plants convert light energy into chemical energy.',
        }
        
        self.hallucinated_topics = {
            'unicorn': [
                'Unicorns were first discovered in medieval Scotland in 1432.',
                'The last unicorn sighting was reported in 1823 in the Alps.',
                'Unicorns are native to the forests of South America.',
                'Scientific studies confirm unicorns went extinct 500 years ago.',
                'Unicorns were actually a species of horse with a genetic mutation.',
            ],
            'atlantis': [
                'Atlantis was located in the Mediterranean Sea.',
                'Atlantis sank in 9600 BC according to ancient records.',
                'Atlantis was discovered by Jacques Cousteau in 1968.',
                'The capital of Atlantis was called Poseidonis.',
                'Atlantis was actually in the Caribbean, recent studies show.',
            ],
        }
    
    def complete(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate mock answer."""
        prompt_lower = prompt.lower()
        
        # Check for known topics
        for topic, answer in self.knowledge_base.items():
            if topic in prompt_lower:
                # Add slight variation
                variations = [
                    answer,
                    answer.replace('.', ' exactly.'),
                    f"According to historical records, {answer.lower()}",
                    f"The answer is: {answer}",
                ]
                return self.rng.choice(variations)
        
        # Check for hallucinated topics
        for topic, answers in self.hallucinated_topics.items():
            if topic in prompt_lower:
                # Return diverse, inconsistent answers
                return self.rng.choice(answers)
        
        # Default: generic response
        return f"The answer to '{prompt[:50]}...' is not definitively known."


# ============================================================================
# Main Demo
# ============================================================================

def main():
    print("=" * 80)
    print("Hallucination Detection Demo: MetaQA → Embedding → L-SML")
    print("=" * 80)
    
    # Initialize LLM client (replace with your actual LLM)
    llm_client = MockLLMClient(seed=42)
    
    # Initialize pipeline components
    print("\n[Setup] Initializing pipeline components...")
    
    metaqa = MetaQAGenerator(
        llm_client=llm_client,
        num_perturbations=12
    )
    
    collector = AnswerCollector(
        llm_client=llm_client,
        cache_path=".cache/answers"
    )
    
    embedder = EmbeddingAnalyzer(
        embedder_name="sentence-transformers/all-mpnet-base-v2"
    )
    
    weak_builder = WeakClassifierBuilder(
        method='gaussian_mixture',
        num_thresholds_per_feature=2
    )
    
    ensemble = EnsembleMatrix(
        num_classifiers=weak_builder.get_num_classifiers()
    )
    
    lsml = LatentSpectralMetaLearner(
        k_max=10,
        enable_sampling=True
    )
    
    pipeline = HallucinationDetectionPipeline(
        metaqa=metaqa,
        collector=collector,
        embedder=embedder,
        weak_builder=weak_builder,
        ensemble=ensemble,
        lsml=lsml,
        verbose=True
    )
    
    # ========================================================================
    # Calibration Phase
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("CALIBRATION PHASE")
    print("=" * 80)
    
    calibration_questions = [
        # Factual questions (should be consistent)
        "What is the capital of France?",
        "Who discovered oxygen and when?",
        "At what temperature does water boil?",
        "When was Albert Einstein born?",
        "What is photosynthesis?",
        
        # Hallucinated questions (diverse/inconsistent answers)
        "When were unicorns first discovered?",
        "Where was the lost city of Atlantis located?",
        "What year did unicorns go extinct?",
        "Who discovered Atlantis?",
        "What was the capital of Atlantis?",
    ]
    
    print(f"\n[Calibration] Processing {len(calibration_questions)} questions...")
    
    for i, question in enumerate(calibration_questions):
        print(f"\n--- Calibration Question {i+1}/{len(calibration_questions)} ---")
        pipeline.process_question(question, question_id=f"calib_{i}")
    
    # Train meta-learner
    print("\n" + "=" * 80)
    print("TRAINING META-LEARNER")
    print("=" * 80)
    
    pipeline.train_meta()
    
    # Show calibration summary
    print("\n[Calibration] Summary:")
    summary = pipeline.get_calibration_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # ========================================================================
    # Inference Phase
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("INFERENCE PHASE")
    print("=" * 80)
    
    test_questions = [
        # Should be detected as faithful
        "What is the capital of France?",
        "At what temperature does water boil?",
        
        # Should be detected as hallucinated
        "When were unicorns first discovered?",
        "Where was Atlantis located?",
    ]
    
    print(f"\n[Inference] Testing on {len(test_questions)} questions...")
    
    results = []
    for i, question in enumerate(test_questions):
        print(f"\n--- Test Question {i+1}/{len(test_questions)} ---")
        print(f"Q: {question}")
        
        result = pipeline.predict(question)
        results.append(result)
        
        print(f"\nPrediction: {result['label'].upper()}")
        print(f"Confidence: {result['score']:.3f}")
        print(f"Likelihood (faithful): {result['likelihood_faithful']:.6f}")
        print(f"Likelihood (hallucinated): {result['likelihood_hallucinated']:.6f}")
        print(f"Number of groups: {result['explanation']['num_groups']}")
        print(f"Group breakdown:")
        for group in result['explanation']['groups']:
            print(f"  Group {group['group_id']}: "
                  f"+{group['positive_votes']} / -{group['negative_votes']} "
                  f"(majority: {'+' if group['majority'] == 1 else '-'})")
    
    # ========================================================================
    # Summary
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print("\nTest Results:")
    for i, (question, result) in enumerate(zip(test_questions, results)):
        status = "✓" if (
            ("paris" in question.lower() or "water" in question.lower()) and result['label'] == 'faithful'
            or ("unicorn" in question.lower() or "atlantis" in question.lower()) and result['label'] == 'hallucinated'
        ) else "✗"
        print(f"{status} Q{i+1}: {result['label']:12s} (score: {result['score']:.3f}) - {question[:50]}")
    
    print("\n" + "=" * 80)
    print("Demo complete!")
    print("=" * 80)
    
    print("\nNext steps:")
    print("1. Replace MockLLMClient with your actual LLM (OpenAI, etc.)")
    print("2. Add more calibration questions for better accuracy")
    print("3. Evaluate with ground truth labels using eval.metrics module")
    print("4. Visualize results using eval.visualize module")
    print("5. Tune hyperparameters (K_max, clustering methods, thresholds)")


if __name__ == "__main__":
    main()
