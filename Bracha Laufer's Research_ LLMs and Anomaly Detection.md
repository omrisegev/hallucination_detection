# **The Convergence of Statistical Reliability and Generative Artificial Intelligence: A Comprehensive Analysis of the Research of Bracha Laufer-Goldshtein**

The academic trajectory of Bracha Laufer-Goldshtein represents a sophisticated evolution from classical statistical signal processing to the frontier of reliable machine learning and large language model (LLM) optimization. While her foundational work was rooted in the geometry of acoustic environments and multi-microphone processing, her recent contributions have established her as a pivotal figure in the development of uncertainty quantification (UQ) and risk-control frameworks for natural language processing (NLP). The transition from analyzing the physics of sound waves to the statistical behavior of transformers underscores a broader shift in the field, where the demand for efficiency in large-scale models must be balanced against rigorous safety and reliability guarantees. This report examines her research across three primary pillars: acoustic manifold learning, risk-controlled natural language generation, and anomaly detection in high-dimensional distributions.

## **Foundations in Acoustic Manifold Learning and Source Localization**

The early research career of Bracha Laufer-Goldshtein was defined by a rigorous exploration of acoustic source localization (SSL) and speaker tracking, primarily utilizing manifold learning to characterize complex environments. Traditional localization techniques often rely on explicit physical models that struggle in the presence of heavy reverberation and non-line-of-sight conditions. In contrast, her work championed a data-driven approach, proposing that acoustic responses, though high-dimensional, lie on a low-dimensional manifold determined by the spatial coordinates of the source.1

The core of this paradigm is the use of Diffusion Maps and manifold regularization to uncover the intrinsic structure of the acoustic space. By viewing the Relative Transfer Function (RTF) as a feature vector, the research demonstrated that the geometric relationships between these vectors correspond to the physical distance between source locations in a room.1 This insight led to the development of semi-supervised localization methods, which require only a small amount of labeled data to map an entire environment.4 This was particularly influential in the context of distributed microphone networks, where asynchronous sensors and unknown geometries pose significant challenges to classical geometry-based methods.5

### **Manifold-Based Localization Frameworks**

The effectiveness of manifold learning in acoustics is best understood through the comparison of traditional signal processing techniques against data-driven manifolds. While techniques like Steered-Response Power with Phase Transform (SRP-PHAT) are computationally efficient, they are frequently compromised by environmental noise and reflections. The manifold-based approach, however, learns these reflections as part of the environment's unique acoustic signature.7

| Method Category | Core Mechanism | Environmental Adaptation | Reliability Indicator |
| :---- | :---- | :---- | :---- |
| **Classical (SRP-PHAT)** | Cross-correlation and phase transform | Poor; sensitive to reverberation | Peak likelihood value (Heuristic) 7 |
| **Manifold-Based** | Dimensionality reduction (Diffusion Maps) | High; learns room-specific reflections | Euclidean distance on the manifold 1 |
| **Deep Learning (SRP-DNN)** | Neural feature extraction and mapping | Moderate; requires extensive training data | Dirichlet distribution parameterization 7 |
| **Hybrid (TDOA \+ Data-Driven)** | Integration of physical TDOA and statistical models | High; combines prior physics with data | Conformal prediction regions 7 |

The research further extended these concepts into the domain of multi-source scenarios through simplex analysis. By modeling the probability of speaker activity across time, the proposed algorithms could detect frames containing a single active speaker, effectively identifying the vertices of a simplex in the feature space.1 This allowed for the separation of overlapping voices and the counting of unknown numbers of sources, a critical requirement for autonomous systems and smart assistants.1 The culmination of this work was the publication of the monograph "Multi-microphone speaker localization on manifolds," which serves as a foundational text for the application of manifold learning to acoustic signal processing.1

## **Evolution Toward Reliable Machine Learning and Uncertainty Quantification**

As machine learning models, particularly deep neural networks, became the standard for signal processing tasks, the focus of the research shifted toward the inherent "black-box" nature of these models. This led to a significant body of work on uncertainty quantification, ensuring that model predictions are not only accurate but also carry statistically valid confidence measures. This shift was catalyzed by the realization that while deep learning improved performance, it often did so at the cost of reliability, especially in out-of-distribution (OOD) scenarios.8

### **Conformal Prediction and the "Learn then Test" Framework**

The research heavily utilizes Conformal Prediction (CP), a distribution-free framework that provides finite-sample guarantees for model outputs.2 Unlike traditional Bayesian methods that require strong distributional assumptions, CP relies only on the exchangeability of data. The research contributed to the development of the "Learn then Test" (LTT) procedure, which reframes risk control as a multiple hypothesis testing problem.2

The LTT framework allows practitioners to tune hyperparameters—such as a threshold for a predictive model—to ensure that the expected loss remains below a user-specified level with high probability. This is mathematically expressed as finding a threshold ![][image1] such that:

![][image2]  
where ![][image3] is a loss function, ![][image4] is the tolerated risk, and ![][image5] is the confidence level.2 This framework has become central to her recent work in both signal processing and natural language generation, providing a rigorous mathematical foundation for "safety" in AI deployment.

## **Research in Natural Language Processing and Large Language Models**

The inquiry regarding research in LLMs and NLP is addressed by a significant and growing collection of recent publications. Since approximately 2021, the research has directly targeted the reliability and efficiency of transformer-based models. This work often involves collaboration with prominent NLP researchers at the Massachusetts Institute of Technology (MIT) and Google DeepMind, such as Regina Barzilay, Tommi Jaakkola, and Adam Fisch.4

### **Inference Acceleration via Early-Exiting and Speculative Decoding**

One of the primary challenges in the deployment of LLMs is the computational cost of inference. The research has addressed this through adaptive computation strategies that utilize risk control to determine when a model can "exit" early or rely on a faster, smaller "draft" model.16

In "Fast yet Safe: Early-Exiting with Risk Control," the research investigates Early-Exit Neural Networks (EENNs), which generate predictions at intermediate layers. The fundamental problem is determining when it is safe to exit without significantly degrading performance. The proposed solution involves tuning the exiting mechanism so that exits only occur when the output quality is guaranteed to meet a user-specified goal.16 This approach was validated across vision and language tasks, demonstrating that it can produce substantial computational savings while maintaining strict performance guarantees.16

Similarly, the research on speculative decoding applies these risk-control principles to the verification phase of LLM generation. By defining a token-level confidence threshold, the system can dynamically decide when to defer to a larger, more expensive model.16 This is particularly relevant for maintaining the "consistency" of the output relative to a full-sized model, as demonstrated in the "Consistent Adaptive Transformers" (CATs) framework.20

| NLP/LLM Application | Mechanism of Action | Role of Risk Control | Performance Impact |
| :---- | :---- | :---- | :---- |
| **Speculative Decoding** | Draft model generation with target model verification | Thresholds token-level uncertainty for model deferral | Accelerates generation while preserving output quality 16 |
| **Early-Exiting (EENNs)** | Prediction at intermediate network layers | Exits only when output quality is statistically guaranteed | Significant reduction in FLOPs/latency for standard tasks 16 |
| **Adaptive Transformers (CATs)** | Meta-consistency classifier for layer-wise termination | Guarantees consistency with the original model | Increases throughput in multi-layer transformer architectures 20 |
| **Generative Error Correction (GER)** | LLM-based post-processing of ASR hypotheses | Dynamically scales the N-best hypothesis set size | Minimizes Word Error Rate (WER) with optimal resource use 22 |

### **Hallucination Mitigation and Safety in Generation**

Beyond efficiency, the research addresses the critical issue of model "hallucinations" through uncertainty-guarding frameworks. The "COIN" framework (Uncertainty-guarding selection) provides a method for calibrating statistically valid uncertainty thresholds to filter generated responses.2 This is essential for high-stakes applications like medical report generation or fact verification, where provide "point estimates" is insufficient and potentially dangerous.

The research proposes move from single-point generation to set-valued prediction. In this paradigm, the LLM produces a set of candidate responses, and the calibration procedure ensures that the set contains at least one correct answer with high probability.2 This handles the "unbounded" nature of the output space in language generation by sampling diverse sequences and evaluating their likelihoods through a rigorous statistical lens.23

### **Generative Error Correction in Speech Recognition**

A notable convergence of the researcher’s acoustic background and her recent LLM work is found in the application of LLMs to Automatic Speech Recognition (ASR). Traditional ASR systems often produce errors due to acoustic variability. Modern approaches leverage LLMs to perform generative error correction (GER) by re-processing the top-N hypotheses from an ASR model.22

The research introduces an adaptive framework for GER that dynamically determines the optimal number of hypotheses for each input. By applying "Learn then Test" (LTT) to ASR confidence scores, the system controls the expected relative Word Error Rate (WER) degradation. This ensures that the system uses only as many resources as necessary for a given audio segment—relying on a small set of hypotheses for "easy" audio and a larger set for "difficult" audio—while providing theoretical guarantees on the final transcription accuracy.22

## **Research in Anomaly Detection and Out-of-Distribution Generalization**

The research demonstrates a sustained and deep involvement in anomaly detection, often framed as a core component of uncertainty quantification. This work addresses the challenge of identifying instances that do not conform to the patterns seen during training, which is critical for the safety and robustness of any machine learning system.11

### **The eMOSAIC Framework**

One of the most significant contributions to anomaly detection is the development of "eMOSAIC" (embedding Mahalanobis Outlier Scoring and Anomaly Identification via Clustering).11 This method was designed for multimodal individual uncertainty quantification and has been successfully applied to complex scientific problems such as drug discovery and ligand binding affinity prediction.

The eMOSAIC framework addresses three major challenges in deep learning for drug discovery: generalizing to out-of-distribution compounds, quantifying uncertainty when traditional assumptions fail, and scaling to large datasets. By using a structure-informed large protein language model and featuring the divergence between the multimodal representations of known cases and unseen instances, eMOSAIC quantifies prediction uncertainty on a compound-by-compound basis.11 The experimental results showed that eMOSAIC significantly outperformed state-of-the-art sequence-based and structure-based methods, as well as existing UQ approaches like Monte Carlo Dropout and standard Conformal Prediction.11

### **Extrapolation Detection and Diverging Flows**

Another key contribution to the field of anomaly detection is the "Diverging Flows" approach.11 This method enables a single model to simultaneously perform conditional generation and native extrapolation detection. It achieves this by structurally enforcing "inefficient transport" for off-manifold inputs. In simpler terms, when the model encounters data that is far from its training manifold, the "flow" of the model diverges, providing a natural signal that the input is an anomaly. This technique was evaluated on synthetic manifolds, cross-domain style transfer, and weather forecasting, demonstrating effective detection without compromising the model's primary predictive fidelity.11

### **Outlier Detection in Acoustic Systems**

Anomaly detection also remains a constant theme in the researcher’s signal processing work. In the context of 3D source localization, she proposed an outlier detection procedure based on hypercone fitting residuals.6 By identifying measurements that do not align with the expected geometric model—often caused by sensor errors or extreme multipath interference—the algorithm can ignore these contributions and refine the source location estimate. This integration of geometric reasoning with anomaly detection highlights her ability to solve practical engineering problems with mathematical rigor.6

## **Interdisciplinary Impact and Vocal Affect Dynamics**

The interdisciplinary nature of her work is further highlighted by her research into vocal affect dynamics during psychotherapy. This work utilized automatic vocal measures to capture intrapersonal and interpersonal emotional shifts in patients and therapists.1 The study demonstrated that these vocal dynamics are predictive of treatment outcomes, offering an objective tool for mental health clinicians.1

This application of signal processing to psychology required the development of robust feature extraction techniques that could operate in the uncontrolled acoustic environment of a therapy room. The success of this research provided empirical evidence for the advantages of using automatic measures over subjective annotations, and it demonstrated how complex interpersonal dynamics can be quantified through acoustic analysis.1

## **Theoretical Contributions to Risk-Controlling Model Selection**

A unifying theme in the most recent work is the development of multi-objective risk control. Most machine learning problems involve trade-offs between different types of errors or costs. The research "Efficiently controlling multiple risks with Pareto testing" and "Risk-Controlling Model Selection via Guided Bayesian Optimization" provides the mathematical tools to navigate these trade-offs.9

### **Pareto Testing for Multi-Objective Safety**

Pareto testing is a two-stage process that combines multi-objective optimization with multiple hypothesis testing.24 In the first stage, the algorithm identifies the set of models that are "Pareto optimal," meaning no other model can improve one risk without worsening another. In the second stage, it identifies a configuration that satisfies all user-specified risk constraints simultaneously. This is critical for LLMs, where a model must be optimized for accuracy, toxicity, and speed at the same time.22

The use of Bayesian Optimization (BO) in this context allows for an efficient search of the model space.15 Instead of exhaustively testing every possible model configuration, the BO procedure "guides" the search toward promising regions of the Pareto frontier. This reduces the amount of calibration data needed to find a safe and high-performing model, a significant advantage when data is scarce or expensive to collect.15

### **Mathematical Rigor in Conformal Selection**

The research also explores the theoretical limits of these methods. For instance, her work proved that the "TIB wealth process" remains a valid supermartingale under all source-target divergences, which has implications for the convergence of online calibration procedures.2 Such proofs are essential for ensuring that the statistical guarantees provided by these systems are robust and not merely empirical observations.

| Theoretical Framework | Key Contribution | Primary Mathematical Tool | Application Domain |
| :---- | :---- | :---- | :---- |
| **LTT (Learn then Test)** | Reframing risk control as multiple hypothesis testing | Fixed-sequence testing and p-value computation | Model calibration and hyperparameter tuning 14 |
| **Pareto Testing** | Simultaneous control of multiple risk functions | Multi-objective optimization \+ multiple testing | Safety-critical AI and multi-task learning 24 |
| **Guided Bayes Opt** | Efficient search for risk-controlled configurations | Gaussian Processes and acquisition functions | Model selection for resource-constrained LLMs 15 |
| **PPI (Prediction-Powered Inference)** | Leveraging unlabeled data to improve calibration | Semi-supervised inference and imputed labels | Sample-efficient risk control in classification/NLP 26 |

## **Future Outlook: The Intersection of Generative AI and Physical Signal Processing**

The trajectory of Bracha Laufer-Goldshtein’s research points toward an increasingly integrated future where the distinction between "signal processing" and "artificial intelligence" continues to blur. Her work on "Multi-dimensional conformal prediction" and "Neural Acoustic Fields" (where researchers use neural networks to model room impulse responses) suggests a convergence of her geometric acoustic roots with modern generative modeling.4

The future directions of LLM research, as outlined in surveys she has contributed to or cited, emphasize the need for real-time processing, sustainable modeling practices, and interdisciplinary collaboration.29 Her work specifically addresses these trends by providing the "principled" tools needed for efficiency (through early-exiting and speculative decoding) and reliability (through risk control and uncertainty quantification).30

Moreover, her expansion into drug discovery and medicine through the eMOSAIC framework illustrates the versatility of these statistical tools.11 The ability to quantify the "unknownness" of a molecule or the "uncertainty" of a medical report is fundamentally the same problem as identifying the "reverberation" in an acoustic signal—it is about uncovering the latent structure of complex, high-dimensional data and providing a rigorous mathematical boundary for what a model can and cannot reliably predict.

## **Conclusion**

Bracha Laufer-Goldshtein has established a comprehensive research portfolio that bridges the gap between the physical reality of acoustic signals and the statistical complexity of modern machine learning. Her recent work definitively answers the user’s query: she is actively involved in the fields of Large Language Models and Natural Language Processing, particularly in the areas of inference acceleration, generative error correction, and hallucination mitigation. Her work in anomaly detection, exemplified by the eMOSAIC and Diverging Flows frameworks, provides essential tools for maintaining model robustness in open-world scenarios.

Through the development of risk-control frameworks and the application of conformal prediction, she has provided a path toward safer and more efficient AI deployment. Whether through identifying a speaker on a manifold or calibrating a transformer's generation, her research emphasizes that the success of AI in high-stakes environments depends on its ability to "know when it does not know".31 This dedication to statistical rigor, combined with an understanding of complex data modalities, positions her work at the forefront of the next generation of reliable artificial intelligence.

#### **Works cited**

1. Bracha Laufer-Goldshtein's research works | Bar Ilan University and other places, accessed April 9, 2026, [https://www.researchgate.net/scientific-contributions/Bracha-Laufer-Goldshtein-2079515024](https://www.researchgate.net/scientific-contributions/Bracha-Laufer-Goldshtein-2079515024)  
2. Distribution-free, Risk-controlling Prediction Sets | Request PDF \- ResearchGate, accessed April 9, 2026, [https://www.researchgate.net/publication/357462027\_Distribution-free\_Risk-controlling\_Prediction\_Sets](https://www.researchgate.net/publication/357462027_Distribution-free_Risk-controlling_Prediction_Sets)  
3. Educational Activities | IEEE Signal Processing Society, accessed April 9, 2026, [https://signalprocessingsociety.org/community-involvement/audio-and-acoustic-signal-processing/educational-activities](https://signalprocessingsociety.org/community-involvement/audio-and-acoustic-signal-processing/educational-activities)  
4. ‪Bracha Laufer Goldshtein‬ \- ‪Google Scholar‬, accessed April 9, 2026, [https://scholar.google.com/citations?user=1YweTFwAAAAJ\&hl=iw](https://scholar.google.com/citations?user=1YweTFwAAAAJ&hl=iw)  
5. (PDF) Multilevel B-Splines-Based Learning Approach for Sound Source Localization, accessed April 9, 2026, [https://www.researchgate.net/publication/330718364\_Multilevel\_B-Splines-Based\_Learning\_Approach\_for\_Sound\_Source\_Localization](https://www.researchgate.net/publication/330718364_Multilevel_B-Splines-Based_Learning_Approach_for_Sound_Source_Localization)  
6. A Robust and Low-Complexity Source Localization Algorithm for Asynchronous Distributed Microphone Networks | Request PDF \- ResearchGate, accessed April 9, 2026, [https://www.researchgate.net/publication/278160203\_A\_Robust\_and\_Low-Complexity\_Source\_Localization\_Algorithm\_for\_Asynchronous\_Distributed\_Microphone\_Networks](https://www.researchgate.net/publication/278160203_A_Robust_and_Low-Complexity_Source_Localization_Algorithm_for_Asynchronous_Distributed_Microphone_Networks)  
7. Uncertainty Quantification and Risk Control for Multi-Speaker Sound Source Localization, accessed April 9, 2026, [https://arxiv.org/html/2603.17377v1](https://arxiv.org/html/2603.17377v1)  
8. Uncertainty Estimation for Sound Source Localization With Deep Learning \- ResearchGate, accessed April 9, 2026, [https://www.researchgate.net/publication/387420396\_Uncertainty\_Estimation\_for\_Sound\_Source\_Localization\_With\_Deep\_Learning](https://www.researchgate.net/publication/387420396_Uncertainty_Estimation_for_Sound_Source_Localization_With_Deep_Learning)  
9. Publications \- Bracha Laufer-Goldshtein \- Wix.com, accessed April 9, 2026, [https://brachalaufer.wixsite.com/home/publications](https://brachalaufer.wixsite.com/home/publications)  
10. Deep-Simplex Multichannel Speech Separation \- ResearchGate, accessed April 9, 2026, [https://www.researchgate.net/publication/396811305\_Deep-Simplex\_Multichannel\_Speech\_Separation](https://www.researchgate.net/publication/396811305_Deep-Simplex_Multichannel_Speech_Separation)  
11. (PDF) Learning by Transduction. \- ResearchGate, accessed April 9, 2026, [https://www.researchgate.net/publication/221404685\_Learning\_by\_Transduction](https://www.researchgate.net/publication/221404685_Learning_by_Transduction)  
12. Conformal Prediction: A Data Perspective \- arXiv, accessed April 9, 2026, [https://arxiv.org/pdf/2410.06494](https://arxiv.org/pdf/2410.06494)  
13. Conformal Prediction for Natural Language Processing: A Survey \- ACL Anthology, accessed April 9, 2026, [https://aclanthology.org/2024.tacl-1.82.pdf](https://aclanthology.org/2024.tacl-1.82.pdf)  
14. Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control, accessed April 9, 2026, [https://www.semanticscholar.org/paper/Learn-then-Test%3A-Calibrating-Predictive-Algorithms-Angelopoulos-Bates/b5437d881723437508de4173142ef4b21c81bfe2](https://www.semanticscholar.org/paper/Learn-then-Test%3A-Calibrating-Predictive-Algorithms-Angelopoulos-Bates/b5437d881723437508de4173142ef4b21c81bfe2)  
15. Risk-Controlling Model Selection via Guided Bayesian Optimization \- Semantic Scholar, accessed April 9, 2026, [https://www.semanticscholar.org/paper/Risk-Controlling-Model-Selection-via-Guided-Laufer-Goldshtein-Fisch/c88c4d8c2689e463afcf54f47320305e75b94000](https://www.semanticscholar.org/paper/Risk-Controlling-Model-Selection-via-Guided-Laufer-Goldshtein-Fisch/c88c4d8c2689e463afcf54f47320305e75b94000)  
16. Fast yet Safe: Early-Exiting with Risk Control \- arXiv, accessed April 9, 2026, [https://arxiv.org/html/2405.20915v2](https://arxiv.org/html/2405.20915v2)  
17. Fast yet Safe: Early-Exiting with Risk Control, accessed April 9, 2026, [https://proceedings.neurips.cc/paper\_files/paper/2024/file/ea5a63f7ddb82e58623693fd1f4933f7-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2024/file/ea5a63f7ddb82e58623693fd1f4933f7-Paper-Conference.pdf)  
18. Fast yet Safe: Early-Exiting with Risk Control \- arXiv, accessed April 9, 2026, [https://arxiv.org/html/2405.20915v1](https://arxiv.org/html/2405.20915v1)  
19. BUDGET-CONSTRAINED LEARNING TO DEFER FOR AU- TOREGRESSIVE MODELS \- OpenReview, accessed April 9, 2026, [https://openreview.net/pdf?id=iQOGhe3Cj9](https://openreview.net/pdf?id=iQOGhe3Cj9)  
20. Consistent Accelerated Inference via Confident Adaptive Transformers \- ACL Anthology, accessed April 9, 2026, [https://aclanthology.org/2021.emnlp-main.406/](https://aclanthology.org/2021.emnlp-main.406/)  
21. \[2104.08803\] Consistent Accelerated Inference via Confident Adaptive Transformers \- arXiv, accessed April 9, 2026, [https://arxiv.org/abs/2104.08803](https://arxiv.org/abs/2104.08803)  
22. CONFIDENT AND ADAPTIVE GENERATIVE SPEECH RECOGNITION VIA RISK CONTROL \- OpenReview, accessed April 9, 2026, [https://openreview.net/pdf/21e1d42402093770712980fc44f266ba4d866d85.pdf](https://openreview.net/pdf/21e1d42402093770712980fc44f266ba4d866d85.pdf)  
23. Conformal Language Modeling \- arXiv, accessed April 9, 2026, [https://arxiv.org/html/2306.10193v2](https://arxiv.org/html/2306.10193v2)  
24. ICLR-2023-Paper-Digests.pdf \- https://www.paperdigest.org, accessed April 9, 2026, [https://www.paperdigest.org/wp-content/uploads/2023/02/ICLR-2023-Paper-Digests.pdf](https://www.paperdigest.org/wp-content/uploads/2023/02/ICLR-2023-Paper-Digests.pdf)  
25. EFFICIENTLY CONTROLLING MULTIPLE RISKS WITH PARETO TESTING \- Fingerprint \- Tel Aviv University, accessed April 9, 2026, [https://cris.tau.ac.il/en/publications/efficiently-controlling-multiple-risks-with-pareto-testing/fingerprints/](https://cris.tau.ac.il/en/publications/efficiently-controlling-multiple-risks-with-pareto-testing/fingerprints/)  
26. Semi-Supervised Risk Control via Prediction-Powered Inference \- ResearchGate, accessed April 9, 2026, [https://www.researchgate.net/publication/394173862\_Semi-Supervised\_Risk\_Control\_via\_Prediction-Powered\_Inference](https://www.researchgate.net/publication/394173862_Semi-Supervised_Risk_Control_via_Prediction-Powered_Inference)  
27. Semi-Supervised Risk Control via Prediction-Powered Inference \- arXiv, accessed April 9, 2026, [https://arxiv.org/html/2412.11174v1](https://arxiv.org/html/2412.11174v1)  
28. Interspeech 2025 \- ISCA Archive, accessed April 9, 2026, [https://www.isca-archive.org/interspeech\_2025/](https://www.isca-archive.org/interspeech_2025/)  
29. Large Language Models for Forecasting and Anomaly Detection: A Systematic Literature Review \- arXiv, accessed April 9, 2026, [https://arxiv.org/html/2402.10350v1](https://arxiv.org/html/2402.10350v1)  
30. 1 Previous Research \- People | MIT CSAIL, accessed April 9, 2026, [https://people.csail.mit.edu/fisch/assets/pdf/research.pdf](https://people.csail.mit.edu/fisch/assets/pdf/research.pdf)  
31. Conformal Prediction for Natural Language Processing: A Survey \- MIT Press Direct, accessed April 9, 2026, [https://direct.mit.edu/tacl/article/doi/10.1162/tacl\_a\_00715/125278/Conformal-Prediction-for-Natural-Language](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00715/125278/Conformal-Prediction-for-Natural-Language)  
32. arXiv:2405.01976v1 \[cs.CL\] 3 May 2024, accessed April 9, 2026, [https://arxiv.org/pdf/2405.01976](https://arxiv.org/pdf/2405.01976)  
33. Conformal Prediction for Natural Language Processing: A Survey \- arXiv, accessed April 9, 2026, [https://arxiv.org/html/2405.01976v1](https://arxiv.org/html/2405.01976v1)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAZCAYAAADnstS2AAAAmElEQVR4XmNgGPrgExD/B+JD6BK4wAQGiAaiAUhxBLogLrAGiN+iC+ICigwQ0znRJXABkOIGdEFcAKT4N7ogLgBSDMLi6BLo4DQQezJAFC9Hk4MDTQaIAjMoHxQiID4HXAUUmEAlcpDEQGGNLsZgBRWchSwIBIxQ8Z8wAWGowC2YABo4wYDkUZDuABRpVACSD2GA2D4KyAMAZTEfzkts8dwAAAAASUVORK5CYII=>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmwAAAA4CAYAAABAFaTtAAAEQUlEQVR4Xu3dTah1UxgH8CUU+RooHxEykfIxUERGokykMKBMMDEzI6LeiZGSRGIiAxNJKUoY3FIoAxNSYkA+SiEDBsrH+ttnd/dd7z7n7nPP9b73nvP71dM591lnn732uZOntdZeuxQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA2Fhf1ri2TR5S79e4o00eR7/V+KfG9zVubtoAgA3yRI1fSlcY9PFT83dizAs1Xhv8fWKNx2t8WOOuGo/VeH32/oEav9b4u8a9/QEH0LxrPdbernF2jcvL4v8BALBBvi5dUXBhkz9hln+myUfyaW9tzV4fqXH+IB+nlq5oO6jurnFNm1xB//sNC9spvqlxbpsEADbbolGc78rRbWfVeKfJ9VJsxFjBFte3iQMmI4GrymjjpzX+ahsmerbGD20SANhcGVFKQZZpuDFp+6LJfV7jjCbXm1ew9fmpUvTk3A+1Df+znPPGNjlR1p1lSjkjiatKP+5skwDAZnq5dMXB2ML2TBGm7Zwm3464DY0VbA8P8su6qXTne75tWEGu9aUaR5p8/FHj1Ta5i/QtfUyRuR/OLNs3HexH8QcAHHK/l64wOKXJXzLLf9vkT5vl5xkWbBm9y4hTbkbYa8HWu6x0572qbVhC+pHveKV06/VO39n8n09KNw28m6xPe7es3qfWZzW+Ktvr3z7e2QwAbJq+KJgXY4VIRs2mFmxX13ivdEXQqgVbrx/Nur1t2EVG1HLcxW1DI8XclBsjbijd9z3VNqwg26QM1671/wcAYIP169dSUE3VHzPP2JToLYP8fsnatvTj/rZhjnx2ymhVCrZF19e6rnSff6NtWFJ+o3zP8O7QTM8u0xcAYA3l5oEUBNnva6q9FGzx1uD9lYP3e9Gv8cq5Tm7axvTTuJe2DSOWLdiGni7dsVl3t6wc92OT+3kkBwBsmL1MuS0zJTq2rUdMmXKcJ1tlZMuMZRb4Z61a+jy2Zq2Vu2Wzrm8Vy47+RT6fYrHNHWlyAMCGSUGQUZxlnFQWF2wfzF7nFWznlWmL+oey1i5TjovOu5scO28rkqFsIpzYD1lnl2nNjPDtJkXsWMGW3xsA2FD9PmdtkTDFosJpa/Z6W40rBvmc76Mab5bp22b0G9D+WbpHNa0ifc7zQofa7UoihdOTbXIFKTZfbJMjHiw7f9esjbMPGwBsqIvK9lToMLLn2lRbNS5octkmo/3OeXHr7JhF0p9sCTJldGqqFIx9HzLyNSZt7bUdK/eU7f7l2gEA9izTdO3TD9ZBrmvKnaQAAIdCRoHyTNF18lyZdmNC3FeOHjmcF3t9pigAwEqyvmq/FucfBP2dpAAAa+XR0j0ndB1kFGzdRgwBAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA4Hj4F7jv6/2fqHDCAAAAAElFTkSuQmCC>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA0AAAAZCAYAAADqrKTxAAAAhklEQVR4XmNgGP5AHIj/A/EXIH4EZcMwQRDEAFGogi6BDzxhINJ0ZEC0k2BAmgGiAWQb0aCcAaIJRBMNHjBANIFsJBpQ3T+WQJyOLriVAaLJBV0CCv6hC4AASBCkiRtdAggqgXg+uiAHA27/1DFAxEHJDAxigfgXVBAZP0Pj/4RpGAVDAwAAiD0poL4cvKoAAAAASUVORK5CYII=>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAZCAYAAAAMhW+1AAAAYUlEQVR4XmNgGAVkgRog/o+ELWESjED8F4jfQ9kY4DoDRAdWSREGhJGPkHASTIEvVNIGJoAOXBggCozRJWAAZC9IQQOaOAowYIAoSgZiViDuY4D4CgOYAHEIEHOgSwxzAACPgxTCgdar/QAAAABJRU5ErkJggg==>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAkAAAAbCAYAAACuj6WAAAAAtUlEQVR4XmNgGBTADYjF0AWRwT8gngXE/6H4Lqo0A4MxEKcj8a8AsScSHwxkgPgnuiA2ALKiDF0QHSxngCiUR5dABhMYEI7GAAYMEAmQu3ZA2TnICkBGgwSFoXxxKB/F+6CwOYEswABR9BDG0YQK2MClIQAkdhjGKYcKsMClIQAkBg9YmCJkIA0V44QJwBzJARMAggNAvAqJDwagCP3KAPF+JRC/R5VGAFYgDgFiBTTxUUAkAABHWyZPvpuxhgAAAABJRU5ErkJggg==>