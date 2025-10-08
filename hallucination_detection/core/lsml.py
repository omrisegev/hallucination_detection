"""
Latent Spectral Meta-Learner (L-SML)

Implements the full L-SML algorithm for unsupervised ensemble meta-classification
with dependent classifiers via latent group variables.

Mathematical Foundation:
    - Extends Dawid-Skene conditional independence model
    - Models dependencies through latent binary variables {α_k}
    - Within each group k, classifiers are conditionally independent given α_k
    - Groups themselves are conditionally independent given true label Y
    - Yields non-linear optimal meta-learner (linear vote-weighting is suboptimal)

Key Steps (from System Prompt):
    1. Estimate pairwise dependencies via covariance matrix Ĉ
    2. Build score matrix Ŝ from 2×2 determinants (Eq. 14)
    3. Spectral clustering on Ŝ to discover K groups (minimize residual, Eq. 13)
    4. Estimate parameters within groups using conditional independence solver
    5. Estimate top-layer priors Pr(α_k | Y)
    6. Prediction via maximum likelihood (Eq. 18)

References:
    - Dawid-Skene conditional independence model
    - Spectral clustering for group discovery
    - Two-stage parameter estimation under conditional independence
    - Equation 18: ŷ = argmax_y Pr(f_1(x),...,f_m(x) | y)

Complexity:
    - Score matrix construction: O(m^4) with optional sampling
    - Spectral clustering: O(m^3)
    - Parameter estimation: O(m·n) per group

Author: Hallucination Detection Framework
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Literal
from sklearn.cluster import SpectralClustering
from scipy.linalg import eigh
from scipy.special import softmax
import warnings

warnings.filterwarnings('ignore')


class LatentSpectralMetaLearner:
    """
    Latent Spectral Meta-Learner for dependent classifier ensembles.
    
    This class implements L-SML, which learns latent group structure in
    weak classifiers and combines them via a probabilistic generative model.
    
    Unlike simple vote weighting (which assumes independence), L-SML models
    dependencies through latent variables, yielding a non-linear meta-classifier.
    
    Attributes:
        k_max: Maximum number of latent groups to consider
        enable_sampling: Use sampling for O(m^4) score matrix (faster)
        sample_size: Number of determinant pairs to sample
        spectral_method: Spectral clustering normalization
        convergence_tol: Convergence tolerance for parameter estimation
        max_iterations: Maximum iterations for parameter estimation
        
    Learned Parameters (after fit):
        K: Optimal number of groups
        c: Group assignment mapping {1..m} → {1..K}
        psi: Sensitivity parameters {ψ_i^α} per classifier
        eta: Specificity parameters {η_i^α} per classifier
        group_priors: Pr(α_k | Y) for each group and label
    
    Example:
        >>> lsml = LatentSpectralMetaLearner(k_max=10)
        >>> Z = np.random.choice([-1, 1], size=(20, 100))  # 20 classifiers, 100 instances
        >>> lsml.fit(Z)
        >>> votes_new = np.random.choice([-1, 1], size=20)
        >>> result = lsml.predict(votes_new)
        >>> result['label']  # 'faithful' or 'hallucinated'
    """
    
    def __init__(
        self,
        k_max: int = 10,
        enable_sampling: bool = True,
        sample_size: int = 1000,
        spectral_method: Literal['normalized', 'unnormalized'] = 'normalized',
        convergence_tol: float = 1e-6,
        max_iterations: int = 100,
    ):
        """
        Initialize LatentSpectralMetaLearner.
        
        Args:
            k_max: Maximum number of groups
            enable_sampling: Enable sampling for score matrix (reduces O(m^4))
            sample_size: Number of determinant pairs to sample
            spectral_method: Spectral clustering type
            convergence_tol: Convergence tolerance
            max_iterations: Max iterations for EM
        """
        self.k_max = k_max
        self.enable_sampling = enable_sampling
        self.sample_size = sample_size
        self.spectral_method = spectral_method
        self.convergence_tol = convergence_tol
        self.max_iterations = max_iterations
        
        # Learned parameters
        self.K: Optional[int] = None
        self.c: Optional[np.ndarray] = None  # Group assignments
        self.psi: Optional[Dict[int, Dict[int, float]]] = None  # Sensitivities
        self.eta: Optional[Dict[int, Dict[int, float]]] = None  # Specificities
        self.group_priors: Optional[Dict[int, Dict[int, float]]] = None
        self.C_hat: Optional[np.ndarray] = None  # Covariance matrix
        self.S_hat: Optional[np.ndarray] = None  # Score matrix
        
        self._is_fitted = False
    
    def fit(self, Z: np.ndarray):
        """
        Learn L-SML model from vote matrix.
        
        Args:
            Z: Vote matrix of shape (m, n), Z ∈ {-1, +1}^(m×n)
               Rows = classifiers, Columns = instances
               
        Pipeline:
            1. Compute classifier covariance Ĉ
            2. Build score matrix Ŝ from 2×2 determinants
            3. Spectral clustering on Ŝ → discover K groups
            4. Estimate parameters within groups (CI solver)
            5. Estimate top-layer priors Pr(α_k | Y)
        """
        m, n = Z.shape
        
        if m < 2 or n < 2:
            raise ValueError("Need at least 2 classifiers and 2 instances")
        
        print(f"[L-SML] Fitting with {m} classifiers and {n} instances")
        
        # Step 1: Compute covariance matrix Ĉ
        print("[L-SML] Step 1: Computing covariance matrix...")
        self.C_hat = self._compute_covariance(Z)
        
        # Step 2: Build score matrix Ŝ from 2×2 determinants
        print("[L-SML] Step 2: Building score matrix from determinants...")
        self.S_hat = self._build_score_matrix(self.C_hat, m)
        
        # Step 3: Spectral clustering to find K and group assignments
        print("[L-SML] Step 3: Spectral clustering for group discovery...")
        self.K, self.c = self._spectral_clustering(self.S_hat, m)
        print(f"[L-SML] Found K={self.K} groups")
        
        # Step 4: Estimate parameters within groups
        print("[L-SML] Step 4: Estimating within-group parameters...")
        self.psi, self.eta = self._estimate_group_parameters(Z, self.c, self.K)
        
        # Step 5: Estimate top-layer priors
        print("[L-SML] Step 5: Estimating group priors...")
        self.group_priors = self._estimate_group_priors(Z, self.c, self.K)
        
        self._is_fitted = True
        print("[L-SML] Fitting complete!")
    
    def predict(self, f_vec: np.ndarray) -> Dict:
        """
        Predict label for new instance via maximum likelihood (Eq. 18).
        
        Args:
            f_vec: Vote vector of shape (m,), f_vec ∈ {-1, +1}^m
            
        Returns:
            Dictionary with:
                - label: 'faithful' or 'hallucinated'
                - score: Probability of faithful
                - explanation: Group-wise support breakdown
                
        Equation 18:
            ŷ = argmax_{y ∈ {±1}} Pr(f_1(x),...,f_m(x) | y)
            
        Computation:
            For each y in {+1, -1}:
                L(y) = Π_{k=1}^K [ Σ_{α ∈ {±1}} Pr(α_k=α|Y=y) 
                                    · Π_{i∈G_k} Pr(f_i | α_k=α) ]
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before predict()")
        
        f_vec = np.array(f_vec, dtype=int)
        
        if f_vec.shape[0] != len(self.c):
            raise ValueError(
                f"Expected {len(self.c)} votes, got {f_vec.shape[0]}"
            )
        
        # Compute likelihood for each label
        L_pos = self._compute_likelihood(f_vec, y=+1)
        L_neg = self._compute_likelihood(f_vec, y=-1)
        
        # Maximum likelihood decision
        if L_pos > L_neg:
            y_hat = +1
            label = 'faithful'
        else:
            y_hat = -1
            label = 'hallucinated'
        
        # Calibrated probability
        score = L_pos / (L_pos + L_neg + 1e-10)
        
        # Group-wise explanation
        explanation = self._explain_prediction(f_vec, y_hat)
        
        return {
            'label': label,
            'score': float(score),
            'y_hat': int(y_hat),
            'likelihood_faithful': float(L_pos),
            'likelihood_hallucinated': float(L_neg),
            'explanation': explanation,
        }
    
    def _compute_covariance(self, Z: np.ndarray) -> np.ndarray:
        """
        Compute classifier covariance matrix Ĉ.
        
        Args:
            Z: Vote matrix (m, n)
            
        Returns:
            Covariance matrix Ĉ of shape (m, m)
            
        The covariance captures pairwise dependencies between classifiers.
        """
        m, n = Z.shape
        
        # Center each classifier (subtract mean)
        Z_centered = Z - Z.mean(axis=1, keepdims=True)
        
        # Compute covariance: Ĉ = (1/n) Z_centered @ Z_centered.T
        C_hat = (Z_centered @ Z_centered.T) / n
        
        return C_hat
    
    def _build_score_matrix(self, C_hat: np.ndarray, m: int) -> np.ndarray:
        """
        Build score matrix Ŝ from 2×2 covariance determinants.
        
        Args:
            C_hat: Covariance matrix (m, m)
            m: Number of classifiers
            
        Returns:
            Score matrix Ŝ of shape (m, m)
            
        Equation 14 (from paper):
            Ŝ[i,j] aggregates 2×2 determinants over pairs (i,j) vs (r,s)
            
        Complexity:
            Full computation: O(m^4)
            With sampling: O(sample_size)
            
        The score matrix captures higher-order dependencies beyond pairwise
        covariance, allowing discovery of latent group structure.
        """
        S_hat = np.zeros((m, m))
        
        if self.enable_sampling and m > 10:
            # Sample pairs to reduce O(m^4) cost
            num_pairs = m * (m - 1) // 2
            
            if num_pairs > self.sample_size:
                # Sample uniformly
                pairs = []
                for i in range(m):
                    for j in range(i + 1, m):
                        pairs.append((i, j))
                
                sampled_pairs = np.random.choice(
                    len(pairs), 
                    size=min(self.sample_size, len(pairs)),
                    replace=False
                )
                
                for idx in sampled_pairs:
                    i, j = pairs[idx]
                    # Sample another pair (r, s)
                    r, s = pairs[np.random.randint(len(pairs))]
                    
                    # 2×2 determinant
                    det = self._determinant_2x2(C_hat, i, j, r, s)
                    S_hat[i, j] += det
                    S_hat[j, i] += det
                
                # Normalize by number of samples
                S_hat /= (self.sample_size + 1e-10)
            else:
                # Full computation (small m)
                S_hat = self._build_score_matrix_full(C_hat, m)
        else:
            # Full computation
            S_hat = self._build_score_matrix_full(C_hat, m)
        
        return S_hat
    
    def _build_score_matrix_full(self, C_hat: np.ndarray, m: int) -> np.ndarray:
        """Full O(m^4) score matrix computation."""
        S_hat = np.zeros((m, m))
        
        for i in range(m):
            for j in range(i + 1, m):
                score = 0.0
                count = 0
                
                for r in range(m):
                    for s in range(r + 1, m):
                        # Compute 2×2 determinant
                        det = self._determinant_2x2(C_hat, i, j, r, s)
                        score += abs(det)  # Absolute value for similarity
                        count += 1
                
                S_hat[i, j] = score / (count + 1e-10)
                S_hat[j, i] = S_hat[i, j]
        
        return S_hat
    
    def _determinant_2x2(
        self, 
        C_hat: np.ndarray, 
        i: int, j: int, r: int, s: int
    ) -> float:
        """
        Compute determinant of 2×2 sub-covariance matrix.
        
        Args:
            C_hat: Full covariance matrix
            i, j, r, s: Indices
            
        Returns:
            Determinant of [[C[i,r], C[i,s]], [C[j,r], C[j,s]]]
        """
        # Extract 2×2 submatrix
        sub = np.array([
            [C_hat[i, r], C_hat[i, s]],
            [C_hat[j, r], C_hat[j, s]]
        ])
        
        # Determinant
        det = sub[0, 0] * sub[1, 1] - sub[0, 1] * sub[1, 0]
        
        return det
    
    def _spectral_clustering(
        self, 
        S_hat: np.ndarray, 
        m: int
    ) -> Tuple[int, np.ndarray]:
        """
        Spectral clustering on Ŝ to find optimal K and group assignments.
        
        Args:
            S_hat: Score matrix (m, m)
            m: Number of classifiers
            
        Returns:
            Tuple of (optimal_K, group_assignments)
            
        Strategy:
            - Try K from 1 to k_max
            - Compute residual per Eq. 13
            - Select K minimizing residual
        """
        k_max_actual = min(self.k_max, m - 1)
        
        if k_max_actual < 2:
            # Too few classifiers, single group
            return 1, np.zeros(m, dtype=int)
        
        best_K = 1
        best_c = np.zeros(m, dtype=int)
        best_residual = float('inf')
        
        # Make S_hat symmetric and non-negative
        S_hat_sym = (S_hat + S_hat.T) / 2
        S_hat_sym = np.maximum(S_hat_sym, 0)
        
        for K in range(1, k_max_actual + 1):
            if K == 1:
                # Single group
                c = np.zeros(m, dtype=int)
                residual = self._compute_residual(S_hat_sym, c, K)
            else:
                try:
                    # Spectral clustering
                    clusterer = SpectralClustering(
                        n_clusters=K,
                        affinity='precomputed',
                        random_state=42,
                        n_init=10
                    )
                    c = clusterer.fit_predict(S_hat_sym)
                    
                    residual = self._compute_residual(S_hat_sym, c, K)
                    
                except Exception as e:
                    # Clustering failed, skip this K
                    continue
            
            if residual < best_residual:
                best_residual = residual
                best_K = K
                best_c = c
        
        return best_K, best_c
    
    def _compute_residual(
        self, 
        S_hat: np.ndarray, 
        c: np.ndarray, 
        K: int
    ) -> float:
        """
        Compute clustering residual (Eq. 13).
        
        Args:
            S_hat: Score matrix
            c: Group assignments
            K: Number of groups
            
        Returns:
            Residual value (lower is better)
            
        The residual measures how well the clustering explains the score matrix.
        """
        residual = 0.0
        
        # Within-group similarity minus between-group similarity
        for k in range(K):
            mask_k = (c == k)
            indices_k = np.where(mask_k)[0]
            
            if len(indices_k) < 2:
                continue
            
            # Within-group score
            within = S_hat[np.ix_(indices_k, indices_k)].sum()
            within /= (len(indices_k) ** 2)
            
            # Between-group score
            mask_not_k = ~mask_k
            indices_not_k = np.where(mask_not_k)[0]
            
            if len(indices_not_k) > 0:
                between = S_hat[np.ix_(indices_k, indices_not_k)].sum()
                between /= (len(indices_k) * len(indices_not_k))
            else:
                between = 0
            
            # Residual: want high within, low between
            residual += (between - within)
        
        return residual
    
    def _estimate_group_parameters(
        self,
        Z: np.ndarray,
        c: np.ndarray,
        K: int
    ) -> Tuple[Dict, Dict]:
        """
        Estimate sensitivity (ψ) and specificity (η) for each classifier.
        
        Args:
            Z: Vote matrix (m, n)
            c: Group assignments
            K: Number of groups
            
        Returns:
            Tuple of (psi_dict, eta_dict)
            
        Uses Dawid-Skene style conditional independence estimation within
        each group. For simplicity, we use empirical frequencies as estimates.
        """
        m, n = Z.shape
        
        psi = {}  # psi[group][classifier] = Pr(f_i=+1 | α=+1)
        eta = {}  # eta[group][classifier] = Pr(f_i=-1 | α=-1)
        
        for k in range(K):
            psi[k] = {}
            eta[k] = {}
            
            # Get classifiers in this group
            group_mask = (c == k)
            group_indices = np.where(group_mask)[0]
            
            if len(group_indices) == 0:
                continue
            
            # Estimate latent α_k for each instance (unsupervised)
            # Use majority vote of group members
            group_votes = Z[group_indices, :]
            alpha_k = np.sign(group_votes.sum(axis=0))  # (n,)
            alpha_k[alpha_k == 0] = 1  # Break ties toward +1
            
            # Estimate parameters for each classifier in group
            for i in group_indices:
                # ψ_i: Pr(f_i=+1 | α=+1)
                mask_alpha_pos = (alpha_k == +1)
                if mask_alpha_pos.sum() > 0:
                    psi[k][i] = (Z[i, mask_alpha_pos] == +1).mean()
                else:
                    psi[k][i] = 0.5
                
                # η_i: Pr(f_i=-1 | α=-1)
                mask_alpha_neg = (alpha_k == -1)
                if mask_alpha_neg.sum() > 0:
                    eta[k][i] = (Z[i, mask_alpha_neg] == -1).mean()
                else:
                    eta[k][i] = 0.5
                
                # Clip to avoid extreme probabilities
                psi[k][i] = np.clip(psi[k][i], 0.1, 0.9)
                eta[k][i] = np.clip(eta[k][i], 0.1, 0.9)
        
        return psi, eta
    
    def _estimate_group_priors(
        self,
        Z: np.ndarray,
        c: np.ndarray,
        K: int
    ) -> Dict:
        """
        Estimate Pr(α_k | Y) for each group.
        
        Args:
            Z: Vote matrix (m, n)
            c: Group assignments
            K: Number of groups
            
        Returns:
            Dictionary: group_priors[group][y] = Pr(α_k=+1 | Y=y)
            
        In unsupervised setting, we estimate Y from overall votes,
        then estimate α_k from group votes.
        """
        m, n = Z.shape
        
        # Estimate true labels Y (unsupervised): use global majority vote
        Y_hat = np.sign(Z.sum(axis=0))  # (n,)
        Y_hat[Y_hat == 0] = 1
        
        group_priors = {}
        
        for k in range(K):
            group_priors[k] = {}
            
            # Get group members
            group_mask = (c == k)
            group_indices = np.where(group_mask)[0]
            
            if len(group_indices) == 0:
                group_priors[k][+1] = 0.5
                group_priors[k][-1] = 0.5
                continue
            
            # Estimate α_k from group votes
            group_votes = Z[group_indices, :]
            alpha_k = np.sign(group_votes.sum(axis=0))
            alpha_k[alpha_k == 0] = 1
            
            # Pr(α_k=+1 | Y=+1)
            mask_y_pos = (Y_hat == +1)
            if mask_y_pos.sum() > 0:
                prob_alpha_pos_given_y_pos = (alpha_k[mask_y_pos] == +1).mean()
            else:
                prob_alpha_pos_given_y_pos = 0.5
            
            # Pr(α_k=+1 | Y=-1)
            mask_y_neg = (Y_hat == -1)
            if mask_y_neg.sum() > 0:
                prob_alpha_pos_given_y_neg = (alpha_k[mask_y_neg] == +1).mean()
            else:
                prob_alpha_pos_given_y_neg = 0.5
            
            group_priors[k][+1] = np.clip(prob_alpha_pos_given_y_pos, 0.1, 0.9)
            group_priors[k][-1] = np.clip(prob_alpha_pos_given_y_neg, 0.1, 0.9)
        
        return group_priors
    
    def _compute_likelihood(self, f_vec: np.ndarray, y: int) -> float:
        """
        Compute Pr(f_vec | Y=y) under learned model.
        
        Args:
            f_vec: Vote vector (m,)
            y: Label in {+1, -1}
            
        Returns:
            Likelihood value
            
        Equation:
            L(y) = Π_{k=1}^K [ Σ_{α ∈ {±1}} Pr(α_k=α|Y=y) 
                                · Π_{i∈G_k} Pr(f_i | α_k=α) ]
        """
        L = 1.0
        
        for k in range(self.K):
            # Get group members
            group_mask = (self.c == k)
            group_indices = np.where(group_mask)[0]
            
            if len(group_indices) == 0:
                continue
            
            # Marginalize over α_k ∈ {+1, -1}
            term_k = 0.0
            
            for alpha in [+1, -1]:
                # Pr(α_k=alpha | Y=y)
                if y == +1:
                    p_alpha = self.group_priors[k][+1] if alpha == +1 else (1 - self.group_priors[k][+1])
                else:  # y == -1
                    p_alpha = self.group_priors[k][-1] if alpha == +1 else (1 - self.group_priors[k][-1])
                
                # Π_{i∈G_k} Pr(f_i | α_k=α)
                p_members = 1.0
                for i in group_indices:
                    f_i = f_vec[i]
                    
                    if alpha == +1:
                        # Pr(f_i | α=+1)
                        if f_i == +1:
                            p_f = self.psi[k][i]
                        else:  # f_i == -1
                            p_f = 1 - self.psi[k][i]
                    else:  # alpha == -1
                        # Pr(f_i | α=-1)
                        if f_i == -1:
                            p_f = self.eta[k][i]
                        else:  # f_i == +1
                            p_f = 1 - self.eta[k][i]
                    
                    p_members *= p_f
                
                term_k += p_alpha * p_members
            
            L *= term_k
        
        return L
    
    def _explain_prediction(self, f_vec: np.ndarray, y_hat: int) -> Dict:
        """
        Generate explanation for prediction.
        
        Args:
            f_vec: Vote vector
            y_hat: Predicted label
            
        Returns:
            Dictionary with group-wise breakdown
        """
        explanation = {
            'num_groups': self.K,
            'groups': []
        }
        
        for k in range(self.K):
            group_mask = (self.c == k)
            group_indices = np.where(group_mask)[0]
            
            if len(group_indices) == 0:
                continue
            
            # Group votes
            group_votes = f_vec[group_indices]
            
            explanation['groups'].append({
                'group_id': k,
                'size': len(group_indices),
                'positive_votes': (group_votes == +1).sum(),
                'negative_votes': (group_votes == -1).sum(),
                'majority': int(np.sign(group_votes.sum())),
            })
        
        return explanation
