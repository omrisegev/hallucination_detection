"""
Weak Binary Classifier Builder via Unsupervised Thresholding

This module converts continuous features into binary votes {-1, +1} using
unsupervised thresholding methods:
- Otsu's method (histogram-based optimal threshold)
- Gaussian Mixture Models (EM-based bimodal separation)
- Percentile-based thresholds

Each feature can generate multiple weak classifiers with different thresholds,
yielding correlated votes that capture uncertainty.

References:
- Otsu's method for automatic thresholding
- GMM for bimodal distribution modeling
- Weak classifier ensembles in meta-learning
"""

import numpy as np
from typing import Dict, List, Literal, Optional
from sklearn.mixture import GaussianMixture
import warnings

warnings.filterwarnings('ignore')


class WeakClassifierBuilder:
    """
    Builds weak binary classifiers from continuous features.
    
    This class applies unsupervised thresholding to convert feature values
    into binary predictions {-1, +1}. Multiple thresholds per feature create
    an ensemble of correlated weak classifiers.
    
    Attributes:
        method: Thresholding method ('otsu', 'gaussian_mixture', 'percentile')
        num_thresholds_per_feature: Number of thresholds per feature
        percentile_thresholds: Percentile values for 'percentile' method
        
    Binary Vote Convention:
        +1 → Faithful (consistent, non-hallucinated)
        -1 → Hallucinated (inconsistent, diverse answers)
        
    Intuition:
        High intra-cluster similarity → consistent → faithful (+1)
        High pairwise distance → diverse → hallucinated (-1)
        Large cluster share → consistent → faithful (+1)
        High entropy → diverse → hallucinated (-1)
    
    Example:
        >>> builder = WeakClassifierBuilder(method='gaussian_mixture')
        >>> features = {'intra_cluster_similarity': 0.9, 'mean_pairwise_distance': 0.2}
        >>> votes = builder.build_votes(features)
        >>> votes.shape
        (10,)  # Multiple weak classifiers
    """
    
    def __init__(
        self,
        method: Literal['otsu', 'gaussian_mixture', 'percentile'] = 'gaussian_mixture',
        num_thresholds_per_feature: int = 2,
        percentile_thresholds: Optional[List[float]] = None,
    ):
        """
        Initialize WeakClassifierBuilder.
        
        Args:
            method: Thresholding method
            num_thresholds_per_feature: Thresholds per feature (for otsu/gmm)
            percentile_thresholds: Percentile values (for percentile method)
        """
        self.method = method
        self.num_thresholds_per_feature = num_thresholds_per_feature
        self.percentile_thresholds = percentile_thresholds or [25, 50, 75]
        
        # Feature polarity: does higher value indicate faithful (+1) or hallucinated (-1)?
        # +1: higher value → faithful; -1: higher value → hallucinated
        self.feature_polarity = {
            'intra_cluster_similarity': +1,      # High similarity → faithful
            'largest_cluster_share': +1,          # Large cluster → consistent → faithful
            'num_clusters': -1,                   # Many clusters → diverse → hallucinated
            'silhouette_score': +1,               # Good clustering → consistent → faithful
            'mean_pairwise_distance': -1,         # High distance → diverse → hallucinated
            'max_pairwise_distance': -1,          # High max distance → outliers → hallucinated
            'centroid_max_dev': -1,               # High deviation → diverse → hallucinated
            'centroid_mean_dev': -1,              # High mean deviation → diverse → hallucinated
            'cluster_variance': -1,               # High variance → uneven → hallucinated
            'embedding_entropy': -1,              # High entropy → diverse → hallucinated
            'normalized_entropy': -1,             # High entropy → diverse → hallucinated
        }
        
        # Learned thresholds from calibration data
        self.thresholds: Dict[str, List[float]] = {}
        self.feature_history: Dict[str, List[float]] = {}
    
    def calibrate(self, feature_dict_list: List[Dict[str, float]]):
        """
        Calibrate thresholds from a collection of feature dictionaries.
        
        Args:
            feature_dict_list: List of feature dicts from multiple instances
            
        This method learns optimal thresholds from calibration data using
        the specified unsupervised method.
        """
        # Collect feature values
        for feature_dict in feature_dict_list:
            for feature_name, value in feature_dict.items():
                if feature_name not in self.feature_history:
                    self.feature_history[feature_name] = []
                self.feature_history[feature_name].append(value)
        
        # Compute thresholds for each feature
        for feature_name, values in self.feature_history.items():
            values = np.array(values)
            
            if self.method == 'percentile':
                thresholds = [
                    np.percentile(values, p) 
                    for p in self.percentile_thresholds
                ]
            elif self.method == 'otsu':
                thresholds = self._otsu_thresholds(values)
            elif self.method == 'gaussian_mixture':
                thresholds = self._gmm_thresholds(values)
            else:
                # Fallback to median
                thresholds = [np.median(values)]
            
            self.thresholds[feature_name] = thresholds
    
    def build_votes(self, features: Dict[str, float]) -> np.ndarray:
        """
        Build binary votes from features.
        
        Args:
            features: Dictionary of feature name → value
            
        Returns:
            Array of binary votes in {-1, +1}
            
        Each feature generates multiple votes (one per threshold),
        resulting in m = num_features × num_thresholds total votes.
        """
        votes = []
        
        for feature_name, feature_value in features.items():
            # Get thresholds for this feature
            if feature_name in self.thresholds:
                thresholds = self.thresholds[feature_name]
            else:
                # Use default threshold (median of seen values or 0.5)
                if (feature_name in self.feature_history and 
                    len(self.feature_history[feature_name]) > 0):
                    thresholds = [np.median(self.feature_history[feature_name])]
                else:
                    thresholds = [0.5]
            
            # Get polarity for this feature
            polarity = self.feature_polarity.get(feature_name, +1)
            
            # Generate vote for each threshold
            for threshold in thresholds:
                if feature_value > threshold:
                    vote = +1 * polarity
                else:
                    vote = -1 * polarity
                
                votes.append(vote)
        
        return np.array(votes, dtype=int)
    
    def _otsu_thresholds(self, values: np.ndarray) -> List[float]:
        """
        Compute Otsu thresholds.
        
        Args:
            values: Array of feature values
            
        Returns:
            List of threshold values
            
        Otsu's method finds the threshold that minimizes intra-class variance
        (or equivalently, maximizes inter-class variance) for bimodal distributions.
        """
        if len(values) < 2:
            return [np.median(values)]
        
        # Normalize to [0, 1]
        values_norm = (values - values.min()) / (values.max() - values.min() + 1e-10)
        
        # Compute histogram
        hist, bin_edges = np.histogram(values_norm, bins=50)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Otsu's method
        best_threshold = 0.5
        best_variance = 0
        
        for i in range(1, len(hist) - 1):
            # Split into two classes
            w0 = hist[:i].sum()
            w1 = hist[i:].sum()
            
            if w0 == 0 or w1 == 0:
                continue
            
            # Class means
            mu0 = (hist[:i] * bin_centers[:i]).sum() / w0
            mu1 = (hist[i:] * bin_centers[i:]).sum() / w1
            
            # Inter-class variance
            variance = w0 * w1 * (mu0 - mu1) ** 2
            
            if variance > best_variance:
                best_variance = variance
                best_threshold = bin_centers[i]
        
        # Denormalize
        threshold = best_threshold * (values.max() - values.min()) + values.min()
        
        # Generate multiple thresholds by perturbing
        thresholds = [threshold]
        if self.num_thresholds_per_feature > 1:
            std = values.std()
            for i in range(1, self.num_thresholds_per_feature):
                offset = (i / self.num_thresholds_per_feature - 0.5) * std
                thresholds.append(threshold + offset)
        
        return thresholds
    
    def _gmm_thresholds(self, values: np.ndarray) -> List[float]:
        """
        Compute GMM-based thresholds.
        
        Args:
            values: Array of feature values
            
        Returns:
            List of threshold values
            
        Fits a Gaussian Mixture Model with 2 components and uses the
        intersection point(s) as thresholds. For multiple thresholds,
        uses quantiles of the component distributions.
        """
        if len(values) < 2:
            return [np.median(values)]
        
        values_reshaped = values.reshape(-1, 1)
        
        try:
            # Fit GMM with 2 components
            gmm = GaussianMixture(
                n_components=2, 
                random_state=42,
                covariance_type='full'
            )
            gmm.fit(values_reshaped)
            
            # Get component parameters
            means = gmm.means_.flatten()
            stds = np.sqrt(gmm.covariances_.flatten())
            weights = gmm.weights_
            
            # Sort by mean
            order = np.argsort(means)
            means = means[order]
            stds = stds[order]
            weights = weights[order]
            
            # Intersection point (approximate)
            # For Gaussians N(μ1, σ1²) and N(μ2, σ2²), intersection is complex
            # Use simple approximation: weighted average
            if self.num_thresholds_per_feature == 1:
                threshold = np.average(means, weights=weights)
                return [threshold]
            else:
                # Multiple thresholds: use quantiles from mixture
                thresholds = []
                for i in range(self.num_thresholds_per_feature):
                    quantile = (i + 1) / (self.num_thresholds_per_feature + 1)
                    
                    # Sample from mixture and compute quantile
                    samples = []
                    for j in range(len(means)):
                        component_samples = np.random.normal(
                            means[j], stds[j], 
                            size=int(weights[j] * 1000)
                        )
                        samples.extend(component_samples)
                    
                    threshold = np.percentile(samples, quantile * 100)
                    thresholds.append(threshold)
                
                return thresholds
                
        except Exception as e:
            # GMM fitting failed, fallback to percentiles
            return [
                np.percentile(values, p) 
                for p in np.linspace(25, 75, self.num_thresholds_per_feature)
            ]
    
    def get_num_classifiers(self) -> int:
        """
        Get total number of weak classifiers.
        
        Returns:
            Total number of weak classifiers (votes)
        """
        if not self.thresholds:
            # Before calibration, estimate based on typical features
            return len(self.feature_polarity) * self.num_thresholds_per_feature
        
        return sum(len(thresholds) for thresholds in self.thresholds.values())
