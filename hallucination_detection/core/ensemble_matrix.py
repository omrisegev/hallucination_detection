"""
Ensemble Matrix for Storing Weak Classifier Votes

This module maintains the global vote matrix Z ∈ {-1, +1}^(m×n) where:
- m = number of weak classifiers
- n = number of instances (questions)

The matrix accumulates votes across calibration questions and is used
by L-SML to learn classifier dependencies and group structure.

References:
- Ensemble learning matrix representation
- Vote aggregation for meta-learning
"""

import numpy as np
from typing import Dict, Optional, List


class EnsembleMatrix:
    """
    Maintains the ensemble vote matrix Z.
    
    This class stores binary votes from weak classifiers across multiple
    instances. The matrix is used by L-SML to learn dependencies between
    classifiers and make meta-predictions.
    
    Attributes:
        num_classifiers: Expected number of weak classifiers (m)
        votes: Dictionary mapping question_id → vote array
        
    Matrix Convention:
        Z[i, j] = vote of classifier i on instance j
        Z ∈ {-1, +1}^(m×n)
        Rows = classifiers
        Columns = instances
    
    Example:
        >>> ensemble = EnsembleMatrix(num_classifiers=10)
        >>> votes = np.array([1, -1, 1, 1, -1, 1, -1, 1, 1, -1])
        >>> ensemble.add_instance("q1", votes)
        >>> Z = ensemble.to_numpy()
        >>> Z.shape
        (10, 1)
    """
    
    def __init__(self, num_classifiers: int):
        """
        Initialize EnsembleMatrix.
        
        Args:
            num_classifiers: Expected number of weak classifiers
        """
        self.num_classifiers = num_classifiers
        self.votes: Dict[str, np.ndarray] = {}
        self.question_ids: List[str] = []
    
    def add_instance(self, question_id: str, votes: np.ndarray):
        """
        Add votes for a single instance.
        
        Args:
            question_id: Unique identifier for the question
            votes: Array of votes, shape (num_classifiers,)
            
        Raises:
            ValueError: If votes array has wrong shape
        """
        votes = np.array(votes, dtype=int)
        
        if votes.shape[0] != self.num_classifiers:
            raise ValueError(
                f"Expected {self.num_classifiers} votes, got {votes.shape[0]}"
            )
        
        # Check values are in {-1, +1}
        unique_values = np.unique(votes)
        if not np.all(np.isin(unique_values, [-1, 1])):
            raise ValueError(
                f"Votes must be in {{-1, +1}}, got {unique_values}"
            )
        
        # Store votes
        self.votes[question_id] = votes
        if question_id not in self.question_ids:
            self.question_ids.append(question_id)
    
    def to_numpy(self) -> np.ndarray:
        """
        Convert to numpy matrix.
        
        Returns:
            Vote matrix Z of shape (m, n)
            
        The matrix is constructed by stacking vote vectors as columns.
        """
        if not self.votes:
            return np.zeros((self.num_classifiers, 0), dtype=int)
        
        # Stack votes as columns
        vote_list = [self.votes[qid] for qid in self.question_ids]
        Z = np.column_stack(vote_list)
        
        return Z
    
    def get_instance_votes(self, question_id: str) -> Optional[np.ndarray]:
        """
        Get votes for a specific instance.
        
        Args:
            question_id: Question identifier
            
        Returns:
            Vote array or None if not found
        """
        return self.votes.get(question_id)
    
    def num_instances(self) -> int:
        """Get number of instances (n)."""
        return len(self.question_ids)
    
    def clear(self):
        """Clear all stored votes."""
        self.votes.clear()
        self.question_ids.clear()
    
    def remove_instance(self, question_id: str):
        """
        Remove an instance.
        
        Args:
            question_id: Question identifier to remove
        """
        if question_id in self.votes:
            del self.votes[question_id]
            self.question_ids.remove(question_id)
    
    def get_classifier_votes(self, classifier_idx: int) -> np.ndarray:
        """
        Get all votes from a specific classifier.
        
        Args:
            classifier_idx: Classifier index (0 to m-1)
            
        Returns:
            Array of votes across all instances, shape (n,)
        """
        Z = self.to_numpy()
        return Z[classifier_idx, :]
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get statistics about the ensemble matrix.
        
        Returns:
            Dictionary of statistics
        """
        if not self.votes:
            return {
                'num_instances': 0,
                'num_classifiers': self.num_classifiers,
                'sparsity': 0.0,
                'positive_rate': 0.5,
            }
        
        Z = self.to_numpy()
        
        return {
            'num_instances': Z.shape[1],
            'num_classifiers': Z.shape[0],
            'positive_rate': (Z == 1).sum() / Z.size,
            'negative_rate': (Z == -1).sum() / Z.size,
            'mean_agreement': self._compute_mean_agreement(Z),
        }
    
    def _compute_mean_agreement(self, Z: np.ndarray) -> float:
        """
        Compute mean pairwise agreement between classifiers.
        
        Args:
            Z: Vote matrix
            
        Returns:
            Mean agreement rate (0 to 1)
        """
        if Z.shape[1] == 0:
            return 0.0
        
        # Pairwise agreement matrix
        m = Z.shape[0]
        agreements = []
        
        for i in range(m):
            for j in range(i + 1, m):
                # Agreement: fraction of instances where votes match
                agreement = (Z[i, :] == Z[j, :]).mean()
                agreements.append(agreement)
        
        return np.mean(agreements) if agreements else 0.0
    
    def prune_classifiers(self, keep_indices: List[int]):
        """
        Prune ensemble to keep only specified classifiers.
        
        Args:
            keep_indices: List of classifier indices to keep
            
        This is useful for ensemble pruning to reduce computational cost
        while maintaining diversity and accuracy.
        """
        # Update votes for each instance
        for qid in self.question_ids:
            self.votes[qid] = self.votes[qid][keep_indices]
        
        # Update num_classifiers
        self.num_classifiers = len(keep_indices)
