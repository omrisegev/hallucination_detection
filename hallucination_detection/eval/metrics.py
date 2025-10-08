"""
Evaluation Metrics for Hallucination Detection

Provides metrics for evaluating detection performance when ground truth is available.
"""

from typing import List, Dict, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)


def evaluate_predictions(
    predictions: List[Dict],
    ground_truth: List[str],
) -> Dict[str, float]:
    """
    Evaluate predictions against ground truth labels.
    
    Args:
        predictions: List of prediction dictionaries from pipeline
        ground_truth: List of true labels ('faithful' or 'hallucinated')
        
    Returns:
        Dictionary of evaluation metrics
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")
    
    # Extract predicted labels and scores
    y_pred = [p['label'] for p in predictions]
    y_scores = [p['score'] for p in predictions]
    y_true = ground_truth
    
    # Convert to binary (1 = faithful, 0 = hallucinated)
    y_pred_binary = [1 if label == 'faithful' else 0 for label in y_pred]
    y_true_binary = [1 if label == 'faithful' else 0 for label in y_true]
    
    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(y_true_binary, y_pred_binary),
        'precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
        'recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
        'f1': f1_score(y_true_binary, y_pred_binary, zero_division=0),
    }
    
    # ROC AUC (if we have both classes)
    if len(set(y_true_binary)) > 1:
        metrics['roc_auc'] = roc_auc_score(y_true_binary, y_scores)
    
    # Confusion matrix
    cm = confusion_matrix(y_true_binary, y_pred_binary)
    metrics['confusion_matrix'] = cm.tolist()
    
    if len(cm) == 2:
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
    
    # Classification report
    report = classification_report(
        y_true_binary, 
        y_pred_binary,
        target_names=['hallucinated', 'faithful'],
        output_dict=True,
        zero_division=0
    )
    metrics['classification_report'] = report
    
    return metrics


def compute_confidence_metrics(
    predictions: List[Dict],
    ground_truth: List[str],
    confidence_bins: int = 10
) -> Dict:
    """
    Compute calibration metrics for confidence scores.
    
    Args:
        predictions: List of prediction dictionaries
        ground_truth: List of true labels
        confidence_bins: Number of bins for calibration curve
        
    Returns:
        Dictionary with calibration metrics
    """
    y_scores = np.array([p['score'] for p in predictions])
    y_pred = [p['label'] for p in predictions]
    y_true = ground_truth
    
    # Convert to binary
    y_pred_binary = np.array([1 if label == 'faithful' else 0 for label in y_pred])
    y_true_binary = np.array([1 if label == 'faithful' else 0 for label in y_true])
    
    # Calibration curve
    bin_edges = np.linspace(0, 1, confidence_bins + 1)
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for i in range(confidence_bins):
        mask = (y_scores >= bin_edges[i]) & (y_scores < bin_edges[i + 1])
        
        if mask.sum() > 0:
            bin_accuracy = y_true_binary[mask].mean()
            bin_confidence = y_scores[mask].mean()
            bin_count = mask.sum()
            
            bin_accuracies.append(bin_accuracy)
            bin_confidences.append(bin_confidence)
            bin_counts.append(int(bin_count))
    
    # Expected Calibration Error (ECE)
    ece = 0.0
    total = len(y_scores)
    for acc, conf, count in zip(bin_accuracies, bin_confidences, bin_counts):
        ece += (count / total) * abs(acc - conf)
    
    return {
        'expected_calibration_error': float(ece),
        'bin_accuracies': bin_accuracies,
        'bin_confidences': bin_confidences,
        'bin_counts': bin_counts,
        'mean_confidence': float(y_scores.mean()),
        'confidence_std': float(y_scores.std()),
    }


def analyze_errors(
    predictions: List[Dict],
    ground_truth: List[str],
    questions: List[str]
) -> Dict:
    """
    Analyze prediction errors in detail.
    
    Args:
        predictions: List of prediction dictionaries
        ground_truth: List of true labels
        questions: List of question texts
        
    Returns:
        Dictionary with error analysis
    """
    errors = {
        'false_positives': [],  # Predicted faithful, actually hallucinated
        'false_negatives': [],  # Predicted hallucinated, actually faithful
    }
    
    for pred, true_label, question in zip(predictions, ground_truth, questions):
        if pred['label'] != true_label:
            error_info = {
                'question': question,
                'predicted': pred['label'],
                'true': true_label,
                'confidence': pred['score'],
                'features': pred.get('features', {}),
            }
            
            if pred['label'] == 'faithful' and true_label == 'hallucinated':
                errors['false_positives'].append(error_info)
            else:
                errors['false_negatives'].append(error_info)
    
    return errors


def feature_importance_analysis(
    predictions: List[Dict],
    ground_truth: List[str]
) -> Dict[str, float]:
    """
    Analyze which features correlate most with correct predictions.
    
    Args:
        predictions: List of prediction dictionaries
        ground_truth: List of true labels
        
    Returns:
        Dictionary of feature importance scores
    """
    # Extract features
    if not predictions or 'features' not in predictions[0]:
        return {}
    
    feature_names = list(predictions[0]['features'].keys())
    feature_values = np.array([
        [p['features'][fname] for fname in feature_names]
        for p in predictions
    ])
    
    # Correctness
    y_true_binary = np.array([1 if label == 'faithful' else 0 for label in ground_truth])
    y_pred_binary = np.array([1 if p['label'] == 'faithful' else 0 for p in predictions])
    correct = (y_true_binary == y_pred_binary).astype(int)
    
    # Compute correlations
    importance = {}
    for i, fname in enumerate(feature_names):
        correlation = np.corrcoef(feature_values[:, i], correct)[0, 1]
        importance[fname] = float(abs(correlation))  # Absolute correlation
    
    # Sort by importance
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    return importance
