"""
Visualization Utilities for Hallucination Detection

Provides plotting functions for analysis and debugging.
Requires matplotlib (optional dependency).
"""

from typing import List, Dict, Optional
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def plot_calibration_curve(
    calibration_metrics: Dict,
    save_path: Optional[str] = None
):
    """
    Plot calibration curve.
    
    Args:
        calibration_metrics: Output from compute_confidence_metrics
        save_path: Optional path to save figure
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available, skipping visualization")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bin_confidences = calibration_metrics['bin_confidences']
    bin_accuracies = calibration_metrics['bin_accuracies']
    
    # Plot calibration curve
    ax.plot(bin_confidences, bin_accuracies, 'o-', label='Calibration curve')
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    
    ax.set_xlabel('Mean Predicted Confidence')
    ax.set_ylabel('Actual Accuracy')
    ax.set_title('Calibration Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add ECE to plot
    ece = calibration_metrics['expected_calibration_error']
    ax.text(0.05, 0.95, f'ECE: {ece:.4f}', 
            transform=ax.transAxes, verticalalignment='top')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved calibration curve to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrix(
    metrics: Dict,
    save_path: Optional[str] = None
):
    """
    Plot confusion matrix.
    
    Args:
        metrics: Output from evaluate_predictions
        save_path: Optional path to save figure
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available, skipping visualization")
        return
    
    cm = np.array(metrics['confusion_matrix'])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    # Labels
    classes = ['Hallucinated', 'Faithful']
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes,
           yticklabels=classes,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')
    
    # Rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_feature_distributions(
    predictions: List[Dict],
    ground_truth: List[str],
    save_path: Optional[str] = None
):
    """
    Plot feature distributions for faithful vs hallucinated.
    
    Args:
        predictions: List of prediction dictionaries
        ground_truth: List of true labels
        save_path: Optional path to save figure
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available, skipping visualization")
        return
    
    if not predictions or 'features' not in predictions[0]:
        print("No features available for visualization")
        return
    
    feature_names = list(predictions[0]['features'].keys())
    
    # Extract features by true label
    features_faithful = []
    features_hallucinated = []
    
    for pred, true_label in zip(predictions, ground_truth):
        features = [pred['features'][fname] for fname in feature_names]
        if true_label == 'faithful':
            features_faithful.append(features)
        else:
            features_hallucinated.append(features)
    
    features_faithful = np.array(features_faithful)
    features_hallucinated = np.array(features_hallucinated)
    
    # Plot
    n_features = len(feature_names)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for i, fname in enumerate(feature_names):
        ax = axes[i]
        
        if len(features_faithful) > 0:
            ax.hist(features_faithful[:, i], bins=20, alpha=0.5, 
                   label='Faithful', color='green')
        
        if len(features_hallucinated) > 0:
            ax.hist(features_hallucinated[:, i], bins=20, alpha=0.5,
                   label='Hallucinated', color='red')
        
        ax.set_xlabel(fname)
        ax.set_ylabel('Count')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Feature Distributions by True Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved feature distributions to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_ensemble_matrix(
    Z: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Visualize ensemble vote matrix.
    
    Args:
        Z: Vote matrix (m, n)
        save_path: Optional path to save figure
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available, skipping visualization")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Use diverging colormap for {-1, +1}
    im = ax.imshow(Z, aspect='auto', cmap='RdYlGn', vmin=-1, vmax=1)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set_xlabel('Instance')
    ax.set_ylabel('Classifier')
    ax.set_title(f'Ensemble Vote Matrix ({Z.shape[0]} classifiers × {Z.shape[1]} instances)')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved ensemble matrix to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_lsml_groups(
    lsml,
    save_path: Optional[str] = None
):
    """
    Visualize L-SML group structure.
    
    Args:
        lsml: Fitted LatentSpectralMetaLearner
        save_path: Optional path to save figure
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available, skipping visualization")
        return
    
    if not lsml._is_fitted:
        print("L-SML not fitted, cannot visualize")
        return
    
    # Group sizes
    unique_groups, counts = np.unique(lsml.c, return_counts=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(unique_groups, counts, color='steelblue')
    ax.set_xlabel('Group ID')
    ax.set_ylabel('Number of Classifiers')
    ax.set_title(f'L-SML Group Structure (K={lsml.K} groups)')
    ax.grid(True, alpha=0.3, axis='y')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved L-SML groups to {save_path}")
    else:
        plt.show()
    
    plt.close()
