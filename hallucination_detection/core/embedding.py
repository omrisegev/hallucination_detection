"""
Embedding-Based Feature Extraction and Clustering Analysis

This module analyzes answer diversity through embeddings and clustering:
- Sentence embeddings via sentence-transformers
- Clustering (KMeans, Agglomerative) to group similar answers
- Feature extraction: cluster statistics, silhouette scores, distances

These features reveal answer consistency and are used as input to weak classifiers.

References:
- Sentence-BERT for semantic embeddings
- Clustering metrics for answer diversity assessment
"""

import numpy as np
from typing import List, Dict, Literal, Optional
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.preprocessing import normalize
import warnings

warnings.filterwarnings('ignore')


class EmbeddingAnalyzer:
    """
    Analyzes answer embeddings to extract diversity features.
    
    This class embeds answers using sentence transformers, clusters them,
    and computes various statistical features that indicate consistency
    or potential hallucination.
    
    Attributes:
        embedder_name: Name of sentence-transformers model
        clustering_methods: List of clustering algorithms to apply
        num_clusters_range: Range of cluster numbers to try
        
    Features Extracted:
        - intra_cluster_similarity: Average cosine similarity within largest cluster
        - largest_cluster_share: Fraction of answers in largest cluster
        - num_clusters: Optimal number of distinct semantic clusters
        - silhouette_score: Overall clustering quality metric
        - mean_pairwise_distance: Average distance between all answer pairs
        - centroid_max_dev: Maximum deviation from global centroid
        - cluster_variance: Variance in cluster sizes
        - embedding_entropy: Entropy of cluster distribution
    
    Example:
        >>> analyzer = EmbeddingAnalyzer(embedder_name="all-mpnet-base-v2")
        >>> answers = ["Paris", "Paris is the capital", "The capital is Paris"]
        >>> features = analyzer.analyze(answers)
        >>> features['largest_cluster_share'] > 0.5  # Answers are similar
        True
    """
    
    def __init__(
        self,
        embedder_name: str = "sentence-transformers/all-mpnet-base-v2",
        clustering_methods: Optional[List[str]] = None,
        num_clusters_range: tuple[int, int] = (2, 5),
    ):
        """
        Initialize EmbeddingAnalyzer.
        
        Args:
            embedder_name: Sentence-transformers model name
            clustering_methods: List of clustering methods ('kmeans', 'agglomerative')
            num_clusters_range: (min, max) number of clusters to try
        """
        self.embedder_name = embedder_name
        self.clustering_methods = clustering_methods or ['kmeans', 'agglomerative']
        self.num_clusters_range = num_clusters_range
        self._embedder = None
    
    @property
    def embedder(self):
        """Lazy-load sentence transformer model."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.embedder_name)
        return self._embedder
    
    def analyze(self, answers: List[str]) -> Dict[str, float]:
        """
        Analyze answer diversity through embeddings and clustering.
        
        Args:
            answers: List of answer strings
            
        Returns:
            Dictionary of numeric features
            
        Pipeline:
            1. Embed answers
            2. Cluster embeddings
            3. Compute cluster-based features
            4. Compute distance-based features
        """
        if len(answers) < 2:
            # Not enough answers to analyze
            return self._default_features()
        
        # 1. Embed answers
        embeddings = self._embed_answers(answers)
        
        # 2. Cluster embeddings
        best_labels, best_k = self._cluster_embeddings(embeddings)
        
        # 3. Compute features
        features = {}
        features.update(self._compute_cluster_features(embeddings, best_labels, best_k))
        features.update(self._compute_distance_features(embeddings))
        features.update(self._compute_distribution_features(embeddings, best_labels))
        
        return features
    
    def _embed_answers(self, answers: List[str]) -> np.ndarray:
        """
        Embed answers using sentence transformer.
        
        Args:
            answers: List of answer strings
            
        Returns:
            Embeddings array of shape (n_answers, embedding_dim)
        """
        embeddings = self.embedder.encode(
            answers, 
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        # Normalize for cosine similarity
        embeddings = normalize(embeddings, norm='l2')
        
        return embeddings
    
    def _cluster_embeddings(
        self, 
        embeddings: np.ndarray
    ) -> tuple[np.ndarray, int]:
        """
        Cluster embeddings and find optimal K.
        
        Args:
            embeddings: Embedding matrix
            
        Returns:
            Tuple of (cluster_labels, optimal_k)
            
        Strategy:
            Try different K values and clustering methods, select best by
            silhouette score (higher is better).
        """
        n_samples = len(embeddings)
        min_k, max_k = self.num_clusters_range
        
        # Adjust range based on sample size
        max_k = min(max_k, n_samples - 1)
        min_k = min(min_k, max_k)
        
        if max_k < 2:
            # Too few samples for clustering
            return np.zeros(n_samples, dtype=int), 1
        
        best_score = -1
        best_labels = None
        best_k = min_k
        
        for k in range(min_k, max_k + 1):
            for method in self.clustering_methods:
                try:
                    if method == 'kmeans':
                        clusterer = KMeans(n_clusters=k, random_state=42, n_init=10)
                    elif method == 'agglomerative':
                        clusterer = AgglomerativeClustering(n_clusters=k)
                    else:
                        continue
                    
                    labels = clusterer.fit_predict(embeddings)
                    
                    # Compute silhouette score
                    if len(np.unique(labels)) > 1:
                        score = silhouette_score(embeddings, labels, metric='cosine')
                        
                        if score > best_score:
                            best_score = score
                            best_labels = labels
                            best_k = k
                            
                except Exception as e:
                    # Clustering failed for this configuration
                    continue
        
        if best_labels is None:
            # Fallback: single cluster
            best_labels = np.zeros(n_samples, dtype=int)
            best_k = 1
        
        return best_labels, best_k
    
    def _compute_cluster_features(
        self, 
        embeddings: np.ndarray, 
        labels: np.ndarray,
        k: int
    ) -> Dict[str, float]:
        """
        Compute cluster-based features.
        
        Args:
            embeddings: Embedding matrix
            labels: Cluster labels
            k: Number of clusters
            
        Returns:
            Dictionary of cluster features
        """
        features = {}
        
        # Number of clusters
        features['num_clusters'] = float(k)
        
        # Cluster sizes
        unique_labels, counts = np.unique(labels, return_counts=True)
        cluster_sizes = dict(zip(unique_labels, counts))
        
        # Largest cluster share
        max_cluster_size = max(counts)
        features['largest_cluster_share'] = max_cluster_size / len(labels)
        
        # Intra-cluster similarity (for largest cluster)
        largest_cluster_label = unique_labels[np.argmax(counts)]
        cluster_mask = labels == largest_cluster_label
        cluster_embeddings = embeddings[cluster_mask]
        
        if len(cluster_embeddings) > 1:
            # Average pairwise cosine similarity within cluster
            similarities = cluster_embeddings @ cluster_embeddings.T
            # Exclude diagonal
            mask = ~np.eye(len(similarities), dtype=bool)
            features['intra_cluster_similarity'] = similarities[mask].mean()
        else:
            features['intra_cluster_similarity'] = 1.0
        
        # Silhouette score
        if k > 1:
            try:
                features['silhouette_score'] = silhouette_score(
                    embeddings, labels, metric='cosine'
                )
            except:
                features['silhouette_score'] = 0.0
        else:
            features['silhouette_score'] = 0.0
        
        # Cluster size variance (normalized)
        if k > 1:
            features['cluster_variance'] = float(np.var(counts) / np.mean(counts))
        else:
            features['cluster_variance'] = 0.0
        
        return features
    
    def _compute_distance_features(
        self, 
        embeddings: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute distance-based features.
        
        Args:
            embeddings: Embedding matrix
            
        Returns:
            Dictionary of distance features
        """
        features = {}
        
        # Mean pairwise distance (cosine)
        distances = pairwise_distances(embeddings, metric='cosine')
        # Exclude diagonal
        mask = ~np.eye(len(distances), dtype=bool)
        features['mean_pairwise_distance'] = distances[mask].mean()
        
        # Maximum pairwise distance
        features['max_pairwise_distance'] = distances[mask].max()
        
        # Centroid deviation
        centroid = embeddings.mean(axis=0, keepdims=True)
        centroid = normalize(centroid, norm='l2')
        deviations = pairwise_distances(embeddings, centroid, metric='cosine').flatten()
        features['centroid_max_dev'] = deviations.max()
        features['centroid_mean_dev'] = deviations.mean()
        
        return features
    
    def _compute_distribution_features(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute distribution-based features.
        
        Args:
            embeddings: Embedding matrix
            labels: Cluster labels
            
        Returns:
            Dictionary of distribution features
        """
        features = {}
        
        # Cluster distribution entropy
        unique_labels, counts = np.unique(labels, return_counts=True)
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        features['embedding_entropy'] = entropy
        
        # Normalized entropy (0 to 1)
        max_entropy = np.log(len(unique_labels)) if len(unique_labels) > 1 else 1.0
        features['normalized_entropy'] = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return features
    
    def _default_features(self) -> Dict[str, float]:
        """Return default features when analysis cannot be performed."""
        return {
            'intra_cluster_similarity': 1.0,
            'largest_cluster_share': 1.0,
            'num_clusters': 1.0,
            'silhouette_score': 0.0,
            'mean_pairwise_distance': 0.0,
            'max_pairwise_distance': 0.0,
            'centroid_max_dev': 0.0,
            'centroid_mean_dev': 0.0,
            'cluster_variance': 0.0,
            'embedding_entropy': 0.0,
            'normalized_entropy': 0.0,
        }
