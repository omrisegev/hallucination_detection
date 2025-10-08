"""
Example: Using OpenAI with Hallucination Detection

This example shows how to integrate the hallucination detector with OpenAI's API.
"""

import os
from openai import OpenAI

from hallucination_detection.core.metaqa import MetaQAGenerator
from hallucination_detection.core.answers import AnswerCollector
from hallucination_detection.core.embedding import EmbeddingAnalyzer
from hallucination_detection.core.weak_votes import WeakClassifierBuilder
from hallucination_detection.core.ensemble_matrix import EnsembleMatrix
from hallucination_detection.core.lsml import LatentSpectralMetaLearner
from hallucination_detection.orchestrator.pipeline import HallucinationDetectionPipeline


class OpenAILLMClient:
    """
    Wrapper for OpenAI API compatible with our framework.
    
    This implements the LLMClient protocol expected by MetaQAGenerator
    and AnswerCollector.
    """
    
    def __init__(
        self, 
        api_key: str = None,
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 200
    ):
        """
        Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: Model name (gpt-3.5-turbo, gpt-4, etc.)
            max_tokens: Maximum tokens in response
        """
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.max_tokens = max_tokens
    
    def complete(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Generate completion using OpenAI API.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=self.max_tokens
        )
        
        return response.choices[0].message.content.strip()


def main():
    """Run hallucination detection with OpenAI."""
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        return
    
    print("Initializing pipeline with OpenAI...")
    
    # Initialize OpenAI client
    llm_client = OpenAILLMClient(
        model="gpt-3.5-turbo",
        max_tokens=150
    )
    
    # Initialize pipeline components
    metaqa = MetaQAGenerator(
        llm_client=llm_client,
        num_perturbations=8  # Use fewer for cost savings
    )
    
    collector = AnswerCollector(
        llm_client=llm_client,
        cache_path=".cache/openai_answers",  # Cache to save API calls
        temperature=0.7
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
        k_max=8,
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
    
    # Calibration questions
    calibration_questions = [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "What is the speed of light?",
        "When did World War II end?",
        "What is the largest planet in our solar system?",
        "Who painted the Mona Lisa?",
        "What is the chemical symbol for gold?",
        "How many continents are there?",
    ]
    
    print(f"\nCalibrating on {len(calibration_questions)} questions...")
    print("(This will make API calls - using cache to minimize cost)")
    
    for i, question in enumerate(calibration_questions):
        print(f"\nProcessing calibration question {i+1}/{len(calibration_questions)}...")
        pipeline.process_question(question, question_id=f"calib_{i}")
    
    print("\nTraining meta-learner...")
    pipeline.train_meta()
    
    # Test questions
    test_questions = [
        "What is the capital of Germany?",  # Should be consistent/faithful
        "When was the internet invented?",   # Should be consistent/faithful
    ]
    
    print("\nTesting on new questions...")
    
    for question in test_questions:
        print(f"\nQuestion: {question}")
        result = pipeline.predict(question)
        
        print(f"Prediction: {result['label']}")
        print(f"Confidence: {result['score']:.3f}")
        print(f"Sample answers:")
        for i, answer in enumerate(result['answers'][:3], 1):
            print(f"  {i}. {answer}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
