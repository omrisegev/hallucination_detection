# Thesis Pivot Reference: Online State Estimation & Manifold Learning for LLM Hallucination Detection

This document acts as a comprehensive reference detailing the theoretical motivation, mathematical formulations, academic citations, and structural classifications for the proposed pivot directions. It is designed to be read by other autonomous coding agents (e.g., Claude) and shared with advisors to coordinate the next research phase.

---

## 1. Motivation: Shifting from Offline to Online Detection

The current thesis approach uses **unsupervised spectral fusion (L-SML)** of 16 features computed over the token entropy trace $H(n)$ of a completed answer.

On July 2, 2026, a **streaming pilot (Step 148)** was conducted. It revealed that prefix-based static feature extraction (re-calculating FFT/STFT/CUSUM at each token step $t$) fails to outperform simple sliding-window averages (DeepConf, arXiv:2508.15260) over most of the generation. It only showed a statistically significant edge in the **earliest 10%** of the trace (first 8-32 tokens) where sliding windows are starved of data.

### Key Structural Bottlenecks of Prefix L-SML:
1. **No Temporal Transition Model**: It treats prefix slices as independent static sequences, ignoring the Markovian "momentum" of hallucination states.
2. **Ambiguous Sign/Weighting**: Solving eigenvectors at each step $t$ causes erratic sign-flips and weight shifts.
3. **Delayed Detection**: It cannot locate the exact token where the model drifted from grounded reasoning into hallucination.

**The Solution**: Pivot to **Online State Estimation** and **Manifold Learning**, tracking the LLM's grounding state recursively as generation unfolds.

---

## 2. Classification Taxonomy Matrix

| Option | Method | Access Level | Supervision | Inference Cost | Adaptivity | Domain Target |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Option 1** | **KalmanNet (State Space Model)** | **Gray-box** | Unsupervised / Weakly | **1-pass** | **Adaptive** | Long-CoT Reasoning, Math, Coding |
| **Option 2** | **LOCA (Manifold Learning)** | **Gray-box** | Unsupervised | **1-pass** | **Static** (Mapping) | Math, QA, Variable-length CoT |
| **Option 3** | **Diverging Flows (Normalizing Flows)** | **White-box** | Unsupervised | **1-pass** | **Static** | Factual QA, RAG |
| **Option 4** | **PRAE (Robust Autoencoder)** | **White-box** | Unsupervised | **1-pass** | **Static** | Multi-hop QA, RAG |
| **Option 5** | **IMM (Interacting Multiple Model)** | **Gray-box** | Unsupervised | **1-pass** | **Adaptive** | Real-time Chat, Agent Loops |

### Taxonomy Definitions:
* **Access Level**:
  * *White-box*: Requires access to model internals (intermediate hidden states, weights, or attention maps).
  * *Gray-box*: Requires access only to output logprobabilities / logits.
  * *Black-box*: Requires only the generated text (API-only).
* **Supervision**:
  * *Supervised*: Requires step-level ground-truth labels for training.
  * *Unsupervised*: No labels used; trained on internal statistical signals.
  * *Weakly Supervised*: Trained using only sequence-level correctness labels (final answer correct/incorrect).
* **Inference Cost**:
  * *1-pass*: Evaluates a single greedy generation run (zero-extra compute).
  * *Sampling-based*: Requires generating $K > 1$ responses to measure consistency.
* **Adaptivity**:
  * *Adaptive*: Dynamically updates tracking parameters, gains, or states recursively at each step.
  * *Static*: Evaluates fixed thresholds or pre-learned static mappings.

---

## 3. Deep-Dive of Pivot Candidates

---

### Option 1: State Space Models (SSM) & Neural Filtering (KalmanNet)

#### Rationale & How it Works
We model the LLM generation process as a Discrete-Time State Space Model:
* **Hidden State ($x_t \in \mathbb{R}^d$)**: Represents the model's latent "groundedness" or "hallucination state."
* **Observation ($y_t \in \mathbb{R}^p$)**: Token-level features extracted from the vocabulary distribution at step $t$:
  $$y_t = [H(t), \Delta E(t), \text{gap}(t), \mathbf{P}_{\text{top-k}}(t)]^T$$
  where $H(t)$ is Shannon entropy and $\Delta E(t)$ is the negative log-probability of the sampled token (Spilled Energy).
* **System Equations**:
  $$x_t = \mathbf{F}x_{t-1} + w_t, \quad w_t \sim \mathcal{N}(0, Q_t)$$
  $$y_t = g(x_t) + v_t, \quad v_t \sim \mathcal{N}(0, R_t)$$

Instead of requiring explicit analytical forms for the complex non-linear functions $f, g$ and unknown noise covariances $Q_t, R_t$, **KalmanNet** uses a compact RNN/GRU to learn the Kalman Gain $K_t$ from data, preserving the recursive correction loop of the classical Kalman Filter:
$$\hat{x}_{t \mid t} = \hat{x}_{t \mid t-1} + K_t (y_t - \hat{y}_{t \mid t-1})$$

#### Unsupervised Training Formulation
To remain unsupervised, the GRU parameters $\theta$ are trained by minimizing the **Observation Prediction Loss (Innovation Loss)** over unlabeled traces:
$$\mathcal{L}_{\text{unsup}} = \sum_{t=1}^{T} \| y_t - \hat{y}_{t \mid t-1} \|^2_2$$
We can inject **weak supervision** by backpropagating a terminal Binary Cross Entropy loss using the final correctness label $Y \in \{0, 1\}$ through the filter's time steps:
$$\mathcal{L}_{\text{weak}} = \text{BCE}(\sigma(\mathbf{W}^T \hat{x}_T), Y)$$

#### Key Citations
* **KalmanNet**: G. Revach, N. Shlezinger, Y. C. Eldar, and R. J. G. van Sloun, *"KalmanNet: Neural Network Aided Kalman Filtering for Partially Known Dynamics,"* IEEE Transactions on Signal Processing, 2022.
* **Unsupervised KalmanNet**: T. Locher, G. Revach, N. Shlezinger, and R. J. G. van Sloun, *"Unsupervised Learned Kalman Filtering,"* arXiv preprint, 2022 (EURASIP 2024).

#### Competitors & Baselines
* **HALT (arXiv:2602.02888, Feb 2026)**: Uses top-20 logprobs as a time series with a supervised GRU. It lacks the model-based Kalman structure, requiring extensive labeled training data and failing to generalize across different LLMs.
* **DeepConf (arXiv:2508.15260, Aug 2025)**: Simple static threshold over sliding-window entropy.

---

### Option 2: Geometric Manifold Learning (LOCA & Diffusion Maps)

#### Rationale & How it Works
This approach focuses on the spatial/geometric geometry of the entropy trace. Rather than treating $H(t)$ as a scalar time series, we treat the sequence of windowed features (e.g., 16 spectral/time-domain features) as a trajectory on a low-dimensional manifold embedded in a higher-dimensional space.
* **LOCA (Local Conformal Autoencoder)** learns an embedding that is locally isometric to the manifold's latent variables, rectifying nonlinear deformations caused by prompt variations and model temperature dynamics.
* **Detection Mechanism**: Under the assumption of **local uniformity** on the manifold, a grounded reasoning trajectory forms a smooth, low-dimensional manifold. A hallucination corresponds to the trajectory **"escaping"** the grounded manifold into a higher-entropy parametric space. We detect this by tracking:
  1. The reconstruction error of a LOCA autoencoder trained on correct-answer traces.
  2. The intrinsic dimensionality of the local neighborhood of the trajectory at step $t$.

#### Key Citations
* **LOCA**: E. Peterfreund, O. Lindenbaum, F. Dietrich, T. Bertalan, M. Gavish, I. G. Kevrekidis, and R. R. Coifman, *"Local Conformal Autoencoder for Standardized Data Coordinates,"* Proceedings of the National Academy of Sciences (PNAS), 2020.
* **Diffusion Maps**: R. R. Coifman and S. Lafon, *"Diffusion maps,"* Applied and Computational Harmonic Analysis, 2006.

#### Competitors & Baselines
* **Standard Autoencoders (AE)**: Face regularization issues; cannot handle the diffeomorphism (deformations) of entropy scale across different tasks.
* **VSDE (Variance Stabilized Density Estimation)**: O. Lindenbaum et al. (2021) — anomaly detection via gradient/variance stability.

---

### Option 3: Normalizing Flows & Transport Cost (Diverging Flows)

#### Rationale & How it Works
Normalizing flows learn to map complex, high-dimensional target distributions (e.g., hidden states of the transformer during correct generation) to a simple base distribution (e.g., a standard Gaussian).
* **The Method**: Train a **Diverging Flow** on the intermediate hidden states of correct (grounded) generations. The flow is structurally regularized to enforce "inefficient transport" (divergence) for inputs that do not lie on the target manifold.
* **Detection Mechanism**: When the model begins to hallucinate, its hidden states drift off the knowledge manifold. The transport cost (or log-likelihood) of the flow diverges, generating an immediate anomaly trigger.

#### Key Citations
* **Diverging Flows**: B. Laufer-Goldshtein et al., *"Extrapolation Detection and Diverging Flows,"* 2025/2026. (Enforces inefficient transport to detect out-of-distribution inputs).
* **eMOSAIC**: A. Badkul, L. Xie, S. Zhang, and L. Xie, *"Embedding Mahalanobis Outlier Scoring and Anomaly Identification via Clustering,"* 2024. (Uses Mahalanobis distance in embedding space for uncertainty quantification).

#### Competitors & Baselines
* **Mahalanobis Distance**: Static distance metric; struggles to capture highly non-linear, multi-modal manifolds of transformer embeddings.
* **Monte Carlo Dropout (MC-Dropout)**: Highly expensive sampling-based UQ competitor.

---

### Option 4: Probabilistic Robust Autoencoder (PRAE) on Latent Manifolds

#### Rationale & How it Works
Standard autoencoders are heavily influenced by outliers (hallucinated segments) during training, which corrupts the learned latent manifold.
* **The Method**: **PRAE** incorporates a differentiable Bernoulli relaxation (a soft, continuous version of the RANSAC algorithm) to jointly train the autoencoder parameters $\Theta$ and an indicator vector $z$ that "selects" only the inliers (grounded states) to participate in reconstruction.
* **Detection Mechanism**: During inference, the learned robust autoencoder reconstructs the hidden states. Hallucinated tokens will suffer from high reconstruction error because the network's latent manifold was trained to exclude outlier behaviors.

#### Key Citations
* **PRAE**: O. Lindenbaum, Y. Aizenbud, and Y. Kluger, *"Probabilistic Robust Autoencoder for Outlier Detection,"* International Conference on Machine Learning (ICML), 2021.

#### Competitors & Baselines
* **Robust PCA (RPCA)**: Linear outlier extraction; fails to capture the deep non-linear dynamics of transformers.
* **Local Outlier Factor (LOF)**: Computational cost scales poorly with sequence length.

---

### Option 5: Interacting Multiple Model (IMM) Filter

#### Rationale & How it Works
The LLM is modeled as a hybrid system that switches between discrete "regimes" or "modes" during generation:
1. **Mode 1 (Grounded)**: Low-entropy drift, stable CUSUM, high permutation entropy.
2. **Mode 2 (Hallucinated)**: High-entropy drift, local variance spikes, decaying permutation entropy.

* **The Method**: Run a **bank of parallel filters** (e.g., two Kalman Filters, one tuned for the Grounded mode and one for the Hallucinated mode).
* **Detection Mechanism**: At each step $t$, the IMM algorithm calculates the probability of each mode based on the measurement residuals (innovation) of the respective filters and a pre-defined Markov transition matrix:
  $$\mathbf{P} = \begin{bmatrix}
  p_{gg} & p_{gh} \\
  p_{hg} & p_{hh}
  \end{bmatrix}$$
  The "hallucination score" is the posterior probability that the system has transitioned to Mode 2.

#### Key Citations
* **IMM**: H. A. P. Blom and Y. Bar-Shalom, *"The interacting multiple model algorithm for systems with Markovian switching coefficients,"* IEEE Transactions on Automatic Control, 1988.

#### Competitors & Baselines
* **Markov Chains / HMMs**: Cannot easily integrate continuous time-varying inputs like STFT values or entropy variations without heavy discretization.

---

## 4. Proposed Hybrid Framework: LOCA + KalmanNet (Online Manifold Tracking)

We can synthesize these options into a unified framework that combines spatial manifold learning with temporal state estimation:

```
                  +----------------------------------------------+
                  |   Greedy Forward Pass (Single generation)    |
                  +----------------------------------------------+
                                         |
                                         v
                  +----------------------------------------------+
                  |      Observation Vector y_t (Logprobs)       |
                  +----------------------------------------------+
                                         |
                                         v
                  +----------------------------------------------+
                  |   LOCA Encoder (Maps y_t -> Manifold dim)    |
                  +----------------------------------------------+
                                         |
                       Manifold Coordinates x_t_tilde
                                         v
                  +----------------------------------------------+
                  |   KalmanNet Filter (GRU Gain Estimation)     |
                  +----------------------------------------------+
                                         |
                            Latent State Estimate x_t
                                         v
                  +----------------------------------------------+
                  |   Conformal Calibration Gate (LTT / COIN)    |
                  +----------------------------------------------+
                                         |
                 Exit / Abort generation if Anomaly Flag = True
```

1. **Spatial Reduction (LOCA)**: The high-dimensional observation vector $y_t$ (e.g., top-20 logprobabilities) is mapped via a learned LOCA encoder to a low-dimensional manifold coordinate $\tilde{x}_t$. LOCA acts as a spatial normalizer, stripping out prompt-specific scaling issues.
2. **Temporal Tracking (KalmanNet)**: The manifold coordinate $\tilde{x}_t$ is fed into a KalmanNet filter. The KalmanNet tracks the state trajectory $x_t$, smoothing out localized noise and modeling the transition probability (regime switching) over time.
3. **Risk-Controlled Stop (LTT/COIN)**: The continuous state estimate $x_t$ is monitored. If it crosses a threshold calibrated via Conformal Prediction (LTT), the generator exits early, providing a formal mathematical guarantee on the false-negative rate (FNR).
