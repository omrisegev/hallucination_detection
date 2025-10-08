#!/usr/bin/env python3
"""
Quick verification script to test the installation.

Run this after installing the package to ensure everything works.
"""

import sys

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from hallucination_detection.core.metaqa import MetaQAGenerator
        from hallucination_detection.core.answers import AnswerCollector
        from hallucination_detection.core.embedding import EmbeddingAnalyzer
        from hallucination_detection.core.weak_votes import WeakClassifierBuilder
        from hallucination_detection.core.ensemble_matrix import EnsembleMatrix
        from hallucination_detection.core.lsml import LatentSpectralMetaLearner
        from hallucination_detection.orchestrator.pipeline import HallucinationDetectionPipeline
        from hallucination_detection.eval.metrics import evaluate_predictions
        from hallucination_detection.eval.visualize import plot_calibration_curve
        
        print("✓ All imports successful!")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality of key components."""
    print("\nTesting basic functionality...")
    
    try:
        import numpy as np
        from hallucination_detection.core.ensemble_matrix import EnsembleMatrix
        from hallucination_detection.core.lsml import LatentSpectralMetaLearner
        
        # Test EnsembleMatrix
        ensemble = EnsembleMatrix(num_classifiers=5)
        votes = np.array([1, -1, 1, -1, 1])
        ensemble.add_instance("test", votes)
        assert ensemble.num_instances() == 1
        print("✓ EnsembleMatrix works!")
        
        # Test L-SML
        Z = np.random.choice([-1, 1], size=(10, 20))
        lsml = LatentSpectralMetaLearner(k_max=3)
        lsml.fit(Z)
        assert lsml._is_fitted
        
        f_vec = np.random.choice([-1, 1], size=10)
        result = lsml.predict(f_vec)
        assert 'label' in result
        assert result['label'] in ['faithful', 'hallucinated']
        print("✓ L-SML works!")
        
        return True
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        return False


def test_sentence_transformers():
    """Test sentence-transformers (optional but recommended)."""
    print("\nTesting sentence-transformers...")
    
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        embeddings = model.encode(["test sentence"])
        print("✓ sentence-transformers available!")
        return True
    except ImportError:
        print("✗ sentence-transformers not installed (required for full functionality)")
        print("  Install with: pip install sentence-transformers")
        return False
    except Exception as e:
        print(f"✗ sentence-transformers test failed: {e}")
        return False


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("Hallucination Detection Framework - Installation Verification")
    print("=" * 60)
    
    results = []
    
    # Test imports
    results.append(("Imports", test_imports()))
    
    # Test basic functionality
    results.append(("Basic Functionality", test_basic_functionality()))
    
    # Test sentence-transformers
    results.append(("Sentence Transformers", test_sentence_transformers()))
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:30s} {status}")
    
    all_passed = all(passed for _, passed in results[:-1])  # Exclude optional sentence-transformers
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ Installation verified successfully!")
        print("\nNext steps:")
        print("1. Run the demo: python examples/demo_pipeline.py")
        print("2. Read the docs: cat README.md")
        print("3. Check usage: cat USAGE.md")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        print("\nTo install dependencies:")
        print("  pip install -e .")
        print("  pip install sentence-transformers")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
