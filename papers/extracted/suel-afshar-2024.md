# SUEL (Afshar, Chen, Han, Lin), arXiv 2408.07796, extracted text

1 
 
Ranking and Combining Latent Structured Predictive Scores without 
Labeled Data 
Shiva Afshara, Yinghan Chenb, Shizhong Hanc, Ying Lin a* 
aDepartment of Industrial Engineering, University of Houston, Houston, TX, 77204, USA; 
bDepartment of Mathematics and Statistics, University of Nevada, Reno, NV, 89557, USA; 
 cLieber Institute for Brain Development, Baltimore, MD, 21205, USA 
Abstract 
Combining multiple predictors obtained from distributed data sources to an accurate meta-learner is 
promising to achieve enhanced performance in lots of prediction problems. As the accuracy of each 
predictor is usually unknown, integrating the predictors to achieve better performance is challenging. 
Conventional ensemble learning methods assess the accuracy of predictors based on extensive labeled 
data. In practical applications, however, the acquisition of such labeled data can prove to be an arduous 
task. Furthermore, the predictors under consideration may exhibit high degrees of correlation, 
particularly when similar data sources or machine learning algorithms were employed during their 
model training. In response to these challenges, this paper introduces a novel structured unsupervised 
ensemble learning model (SUEL) to exploit the dependency between a set of predictors with continuous 
predictive scores, rank the predictors without labeled data and combine them to an ensembled score 
with weights. Two novel correlation-based decomposition algorithms are further proposed to estimate 
the SUEL model, constrained quadratic optimization (SUEL.CQO) and matrix-factorization-based 
(SUEL.MF) approaches. The efficacy of the proposed methods is rigorously assessed through both 
simulation studies and real-world application of risk genes discovery. The results compellingly 
demonstrate that the proposed methods can efficiently integrate the dependent predictors to an ensemble 
model without the need of ground truth data. 
 

2 
 
Keywords: unsupervised ensemble learning; dependent predictive scores; classification; risk genes 
discovery. 
1. Introduction  
Combining multiple predictors obtained from distributed data sources or various machine learning 
algorithms to an accurate meta-learner is promising to achieve enhanced and robust performance in a 
lots of classification problems (Erekat, Servis, Madathil, & Khasawneh, 2020; Song & Liu, 2018; Wang, 
Wang, & Tang, 2021; Wang, Wang, Tang, & Zhang, 2023). For instance, in genetics, researchers seek 
to understand disease-associated genes or variants by consolidating multiple association statistics 
originating from various genome-wide association studies conducted across diverse large-scale 
populations. The integration of these association statistics furnishes complementary evidence, thereby 
enhancing the discovery of risk (Brueggeman, Koomar, & Michaelson, 2020; Cao et al., 2022). 
Similarly, in the realm of anomaly detection, there is a need to combine predictions from distributed 
detectors or models, constructed from data captured by local sensors or diverse datasets. This enables 
the detection of more robust anomalies at a system-level (Araya, Grolinger, ElYamany, Capretz, & 
Bitsuamlak, 2017; Tama & Lim, 2021; Tsogbaatar et al., 2020; Vanerio & Casas, 2017; J. Zhang, Li, 
Nai, Gu, & Sallam, 2019). Furthermore, the ability to combine predictors obtained from distributed and 
local data holds significant promise in facilitating the development of federated or privacy-preserving 
machine learning models, all without the necessity of accessing or sharing local datasets (Chung & Al 
Kontar, 2024; Yue, Kontar, & Gómez, 2024). However, achieving the optimal combination of 
predictors necessitates prior knowledge regarding the reliability or accuracy of each model, which is 
often unavailable in practical applications. In genetics, for example, our current knowledge on the 
disease-gene associations is still very limited and the effectiveness of the existing genome-wide 
association studies remain unknown.  Consequently, there is a pressing need to develop computational 
algorithms that can effectively rank predictors with unknown accuracy and achieve their optimal 
combination.  

3 
 
Ensemble learning algorithms are most widely used approaches to combine multiple predictors 
to a meta-learner, which consistently outperforms individual predictors. These ensemble approaches 
are broadly categorized into supervised and unsupervised methods. In the supervised ensemble learning 
realm, the techniques are primarily centered around the ranking and combination of different predictors 
using ground truth information. These supervised ensemble learning methods can be further divided 
into four principal categories: stacking, blending, bagging, and boosting. Stacking and blending 
techniques operate by aggregating predictions from multiple machine learning algorithms, all trained 
on the same dataset.  Predictions generated by various algorithms are treated as new features and feed 
into a supervised learning model to predict the final ensembled scores (Kalia, 2018; Pavlyshenko, 2018; 
Wu, Zhang, Jiao, Guo, & Hamoud, 2021; Yao, Zhang, Luo, Liu, & Ren, 2022). Bagging-based 
ensemble models, such as random forest, enhance prediction performance by reducing the prediction 
variance using ensemble of machine learning models (such as unpruned decision trees) that are trained 
in different subsets of the data. Then, these models are combined using voting or averaging (Breiman, 
1996; Caltanissetta, Bertoli, & Colosimo, 2023; Qi, 2012). Boosting-based methods, such as Adaboost 
and Gradient Boosting, were developed to mitigate the bias of prediction. They do so by weighted 
combining sequentially estimated weak learners into a meta-learner, assigning higher weights to the 
more accurate weak learners (Chen & Guestrin, 2016; Gaw, Yousefi, & Gahrooei, 2022; Iranzad et al., 
2022; Marudi, Ben-Gal, & Singer, 2022; Schapire, 2013).  
However, it’s worth noting that estimating these supervised ensemble learning models typically 
require the access to the entire training dataset, which prohibit them to be used when the original training 
datasets are distributed and protected. Moreover, supervised ensemble learning methods heavily rely 
on a substantial amount of the labeled data to achieve robust prediction performance. In practice, 
however, real-world problems often suffer from a scarcity of labeled data, which can significantly 
undermine the effectiveness of supervised learning techniques. The lack of labeled genes is regarded as 
a critical challenge to build the effective risk gene discovery model for most complex diseases (Craig, 
2008). The Autism Spectrum Disorders (ASD), for example, is a neurodevelopmental condition that 
has a strong genetic basis (Devlin & Scherer, 2012). While the role of genetic factors has been firmly 

4 
 
established in ASD, the knowledge of individual gene’s contribution to ASD development is still far 
from complete (Weiner et al., 2017).  To address these issues, unsupervised ensemble learning 
approaches have been developed (Shaham et al., 2016; Valpola & Karhunen, 2002; Yang, 2016; J. 
Zhang et al., 2019). These approaches enable the combination of several predictors when the original 
training data and ground truth information are not readily available. A pivotal step in unsupervised 
ensemble learning methods involves estimating the reliability of each predictor without labeled data. 
Various approaches for estimating predictor reliability without labeled data have been proposed in the 
literature. For example, building upon the model presented in (Dawid & Skene, 1979; Jaffe, Nadler, & 
Kluger, 2015; Y. Zhang, Chen, Zhou, & Jordan, 2014), recent works have employed methods such as 
maximum likelihood estimation and the EM algorithm. These techniques utilize an iterative approach 
to solve the maximum likelihood estimation of parameters related to the latent probability distribution 
of predictors (Parisi, Strino, Nadler, & Kluger, 2014).  In (Traganitis, Pages-Zamora, & Giannakis, 
2018), a maximum posterior approach was introduced to estimate the confusion matrix of predictors in 
a multi-class ensemble classification problem. Additionally, (Li & Yu, 2014) and (Cabrera-Bean, Díaz-
Vilor, & Vidal, 2016) derived an upper bound for the error rate in EM-based and maximum posterior-
based approaches for determining the unknown accuracy of each predictor. However, it’s important to 
note that the EM-based approach, while conceptually straightforward, has limitations. It only guarantees 
convergence to a local optimal solution and can entail high computational costs. To mitigate this issue, 
a spectral-based approach was recently developed by spectrally decomposing the covariance matrix of 
predictors and inferring the model parameters from the covariance matrix (Ionita-Laza, McCallum, Xu, 
& Buxbaum, 2016; Jain & Oh, 2014). Nonetheless, one limitation of spectral-based approaches is their 
assumption of conditional independence among classifiers given the true label. This assumption may 
not hold in many real-world problems. Recent research has aimed to relax this conditional independence 
assumption by considering a latent tree structure capable of capturing dependencies between predictors 
(Jaffe, Fetaya, Nadler, Jiang, & Kluger, 2016; Kleindessner & Awasthi, 2018; Shaham et al., 2016). 
However, it’s essential to acknowledge that these studies primarily focused on integrating the 
categorical outcomes predicted from the models, which cannot be used to deal with the continuous 

5 
 
predictive scores often encountered in the real-world problems. The advantages of using continuous 
predictive scores compared with categorical outcomes include the better characterization of prediction 
uncertainty and the independence of cut-off threshold. 
Hence, there arises a critical necessity to develop new models capable of ranking and combining 
the dependent and continuous predictive scores without labeled data. To fill this gap, this paper 
introduces an innovative Structured Unsupervised Ensemble Learning model (SUEL) accompanied by 
two efficient learning algorithms to estimate the model (denoted as SUEL.CQO and SUEL.MF). The 
proposed SUEL model 1) provides an optimal ranking and combination schema of continuous 
predictive scores without labeled data; 2) explicitly characterizes the dependencies between predictive 
scores through a set of latent models; 3) efficiently uncovers the latent models and estimates model 
parameters by leveraging the spectral structure of the correlation coefficient matrix between predictors. 
The novelty and contributions of the proposed SUEL methods can be succinctly summarized as follows: 
• 
A novel structured unsupervised ensemble learning model is proposed to rank and combine 
the dependent and continuous predictive scores without labeled data. 
• 
Two novel learning algorithms based on the spectral structure of correlation coefficient 
matrix between predictive scores are developed to efficiently estimated the proposed model.  
• 
Properties of the proposed model and algorithms are analysed theoretically, including how 
to determine the optimal latent structure among dependent predictive scores and the optimal 
weights for combining the scores. 
• 
The proposed method is applied to a risk gene discovery for autism spectrum disorder 
problem, in which the proposed method optimally combines the existing disease-association 
evidence to an unsupervised ensemble score with enhanced prediction accuracy.       
The rest of the paper is organized as follows. Section 2 introduces the proposed SUEL model, 
elaborates its theoretical properties and presents two correlation matrix-based decomposition 
algorithms to solve it. Section 3 conducts a simulation study to evaluate the performance of the proposed 
method. A real-world case study on risk genes discovery is further presented in Section 4. The 
conclusions are drawn in Section 5. 

6 
 
2. Materials and methods 
2.1 Problem setup 
We consider the problem of ranking and combining 𝑀 predictors with unknown accuracy, denoted as 
{𝑓$}$&'
( , to predict a binary outcome. Each predictor provides continuous predictive scores 𝑓$(𝑥+) for a 
set of 𝑁 samples, denoted as 𝐷= {𝑥+}+&'
0
. It is notable that the feature information used in each 
predictor of 𝑁 samples can be different. The true outcomes of these samples are unknown and 
represented as a binary variable, i.e., {𝑦+}+&'
0 . Each predictor is trained on its local training dataset which 
is also unavailable to us. To rank and combine the predictors to a meta-learner, we adopt the ensemble 
learning approach which weighted aggregates the predictors to an ensemble model, denoted as 𝑔(𝑥+) =
∑
𝑤$𝑓$(𝑥+)
(
$&'
, 𝑡∈{1, … , 𝑁}, where 𝑤$ represents the ensemble weight assigned to 𝑖th predictor and a 
higher weight indicates the predictor is more reliable or accurate. The objective is to find a set of weights 
that optimally rank and combine the predictors to best predict the distribution of outcome variable 
without labeled data.  
As the scores predicted from 𝑀 predictors are continuous, a set of two-component Gaussian 
mixture distributions are used to represent the relationships between the predictors and outcome 
variable (Ionita-Laza et al., 2016). If the scores do not follow Gaussian distribution assumption, the 
Cox-Box transformation (Daimon, 2011) can be used to transform the scores to Gaussian distributions 
first and the proposed method can be applied on the transformed scores. The assumption is that the 
scores predicted from each predictor have a two-component Gaussian mixture distribution where each 
component represents the distribution of the predicted scores in one class. The Gaussian mixture 
distribution can be formulated as: 
                 P<𝑓$(𝑥+)= = 𝜋P(𝑓$(𝑥+)|𝑦+ = 1) + (1 −𝜋)P(𝑓$(𝑥+)|𝑦+ = 0)                             (1) 
where 𝜋 represents the prior probability of belonging to class 1. P(𝑓$(𝑥+)|𝑦+ = 1)  and 
P(𝑓$(𝑥+)|𝑦+ = 0) represent the component-specific distributions of the predicted score in two classes, 
respectively. The Gaussian distributions in two classes are described by component-specific means, 𝜇$' 
and 𝜇$D, and component-specific standard deviations, 𝜎$' and 𝜎$D. 

7 
 
Conventional ranking and combining methods assume the prediction scores are conditionally 
independent given the true label of outcome and the variances of two components are the same, i.e., 
𝜎$' = 𝜎$D = 𝜎$. Based on these assumptions, the maximum likelihood estimation of the ensemble 
weights can be explicitly characterized in Property 1. 
Property 1 – If the predictive scores from each predictor follow a two-component Gaussian mixture 
distribution with different component-specific means but equal variance, the maximum likelihood 
estimation of the ensemble weights equals to: 
     𝑤$ =
'
FGH
I (𝜇$' −𝜇$D).                                                         (2) 
Property 1 highlights that the scores exhibiting a larger discrepancy between component-specific 
means and lower variance are indicative of greater reliability and accuracy. The larger difference 
between component-specific means indicates a better discrimination between the two classes, while 
lower variance suggests that the predicted scores within each class are more closely clustered. Figure 1 
illustrates this concept with an example, showcasing how the difference between component-specific 
means and the variance of a Gaussian mixture distribution can impact predictor performance. As 
depicted, a larger difference in component-specific means and smaller component-specific standard 
deviation correspond to enhanced prediction performance of the score. It's worth noting that the 
maximum likelihood estimation of ensemble weights proposed in Property 1 holds an advantage over 
weights defined in conventional ranking and combining methods (Ionita-Laza et al., 2016). These 
conventional methods solely consider the difference between component-specific means and disregard 
the influence of component-specific variance on predictor performance and, subsequently, the 
estimation of ensemble weights.  

8 
 
 
Figure 1. The impact of means and variance of components in accuracy of predictors. 
2.2 The proposed structured unsupervised ensemble learning method for dependent scores 
Conventional ranking and combining methods often hinge on the assumption that predictors are 
independent, given the outcome variable. This assumption is known as the conditional independence 
assumption. However, in practical scenarios, predictors can exhibit dependencies. This dependence can 
arise when predictors are estimated using overlapping training data or similar algorithms, challenging 
the conditional independence assumption. To capture the dependency among 𝑀 predictors, a novel 
structured unsupervised ensemble learning model is proposed by assuming the predictors are correlated 
in 𝐾 (𝐾≤𝑀)  latent models (as shown in Figure 2). The predictors generated from the same latent 
model tend to exhibit high correlations, while those stemming from different latent models are 
conditionally independent given the true label of outcome. Conceptually, each latent model can be 
viewed as a distinct source of information or an algorithm, and the associated predictors are essentially 
different perturbations or variations of this model. The latent models are conditionally independent of 
each other when considering the true label of the outcome. Each latent model, denoted as ℎM, 𝑘=
1, … , 𝐾, is described by an underlining two-component Gaussian mixture distribution. Thus, the 
distribution of each latent model can be represented as: 
ℎM(∙|𝑦+ = 0)~𝑁<𝜇M
D,𝜎M
F=,    ℎM(∙|𝑦+ = 1)~𝑁<𝜇M
', 𝜎M
F=                              (3) 

9 
 
where 𝜇M
D and 𝜇M
' represent the component-specific means for the latent model distribution and 𝜎M
F 
represents the component-consistent variance of the distribution.  
 
Figure 2. The structure of the proposed method. 
The set of predictors that are associated with a latent model, ℎM, is represented as {𝑓$}$∈RS, 
where 𝐶M indicates the set of associated predictors. Each predictor is assumed to be a perturbation of 
the latent variable following the linear relationship below: 
          𝑓$(𝑥+) = 𝑏$ℎM(𝑥+) + 𝜀$, 𝜀$~𝑁<0, 𝜃$
F=                                               (4) 
where 𝑏$ represents the strength of association between the predictor and corresponding latent model 
and 𝜀$ represents a normally distributed random noise that is independent of the latent model. 𝜃$
F 
indicates the variance of the random noise for 𝑖th score.  
2.3 Property of the proposed model 
This session discusses the statistical properties of the proposed structured unsupervised ensemble 
learning (SUEL) model. These statistical properties encompass: 1) the optimal weights of standardized 
predictive scores under the latent group structure, which is provided in Property 2; 2) the covariance 
matrix of the standardized predictors, if arranged under certain order, has a block-wise structure with 
each block is a rank-one submatrix (Lemma 1); 3) the number of the latent models in the proposed 
SUEL model can be determined by the eigen decomposition of the covariance matrix between 
standardized predictors  (Lemma 2). 

10 
 
Property 2 – Given 𝑀 predictors that are correlated by the latent group structure described in the 
proposed SUEL model and have standardized predictions, the weight of 𝑖th predictor (𝑤$) can be 
estimated as: 𝑤$ = −
XHYS
Z
[\('\[)XH
IYS
ZI. 
Proof of Property 2 is provided in the Supplemental materials. 
It is notable that the numerator in the equation of Property 2 represents the difference in 
component-specific means of 𝑓$(𝑥)  and the denominator represents the total variance of 𝑓$(𝑥) . 
Compared to the accuracy of conditionally independent predictors obtained from the literature (𝑤$ =
−
YHZ
[ ) (Ionita-Laza et al., 2016), the accuracy of predictors in the proposed method depends on 1) the 
accuracy or reliability of its associated latent model (𝜇M
D), 2) the strength of association between that 
predictor with the latent model (𝑏$) and 3) the variance of the predictor ( 𝑏$
F𝜎M
F + 𝜃$
F). The variance of 
each predictor can also be estimated using the parameters 𝜇M
D and 𝑏$. Therefore, we need to estimate 𝜇M
D, 
𝑏$, and 𝜋 to compute 𝑤$ for each predictor.  
If the latent group structure is known and 𝐶M denotes the set of predictors that are associated to 
latent model 𝑘, the covariance matrix of 𝑀 standardized predictors has block-wise structure as 
illustrated in Lemma 1. 
Lemma 1 – Given the covariance matrix among 𝑀standardized predictors as ^𝑟$` = 𝐸b(𝑓$(𝑥) −
𝐸[𝑓$(𝑥)])<𝑓`(𝑥) −𝐸b𝑓`(𝑥)e=ef$,`&',..,(, the off-diagonal elements of covariance matrix has a block-
wise structure with each block is a rank-one submatrix and can be represented by the following 
piecewise function:  
𝑟$` = g
𝜈$
ij𝜈`
ij    𝑖𝑓 𝑖≠𝑗 𝑎𝑛𝑑, 𝑓$ 𝑎𝑛𝑑 𝑓` 𝑎𝑟𝑒 𝑎𝑠𝑠𝑜𝑐𝑖𝑎𝑡𝑒𝑑 𝑤𝑖𝑡ℎ 𝑡ℎ𝑒 𝑠𝑎𝑚𝑒 𝑙𝑎𝑡𝑒𝑛𝑡 𝑚𝑜𝑑𝑒𝑙 
𝜈$
ivv𝜈`
ivv                𝑖𝑓 𝑓$ 𝑎𝑛𝑑 𝑓` 𝑎𝑟𝑒 𝑎𝑠𝑠𝑜𝑐𝑖𝑎𝑡𝑒𝑑 𝑤𝑖𝑡ℎ 𝑑𝑖𝑓𝑓𝑒𝑟𝑒𝑛𝑡 𝑙𝑎𝑡𝑒𝑛𝑡 𝑚𝑜𝑑𝑒𝑙𝑠  
where 𝑣ij = [𝜎M𝑏$]M∈{',…,x},$∈RS  , 𝑣ivv = yz'\[
[ 𝜇M𝑏${
M∈{',…,x},$∈RS
. 
Proof of Lemma 1 is provided in the Supplemental materials. 
It can be observed from Lemma 1 that the covariance matrix is a combination of 2 rank-one 
matrices. The elements in diagonal blocks, i.e. correlations between predictors associated with the same 

11 
 
latent model, correspond to the rank-one matrix 𝑣ij(𝑣ij)|, which only depends on the strength of 
association between the predictors and their corresponding latent models, 𝒃M = {𝑏$}$∈RS , and the 
standard deviation of latent model (𝜎M).  The elements in the off-diagonal blocks, which represent the 
correlations between the conditional independent predictors, correspond to another rank-one matrix 
𝑣ivv<𝑣ivv=
| . It depends on the strength of association between the predictors and their corresponding 
latent models and the difference in the component-specific means of the associated latent model (𝜇M). 
Using a binary variable to indicate whether a pair of predictors are associated with the 𝑘th latent model 
or not, denoted as 𝐼M(𝑖, 𝑗), the covariance matrix in Lemma 1 can be rewritten as Equation 5 below: 
                                  𝑟$` = 𝐼M(𝑖, 𝑗)𝑣ij(𝑣ij)| + <1 −𝐼M(𝑖, 𝑗)=𝑣ivv<𝑣ivv=
|, ∀𝑖≠𝑗.                         (5) 
Lemma 1 also provides insight into the fact that if we have knowledge of the assignment of scores to 
the latent models and the optimal number of latent models, we can decompose the covariance matrix. 
This decomposition allows us to estimate the parameters in the proposed method based on the elements 
within the covariance matrix. Determining the optimal number of latent models is subsequently 
addressed in Lemma 2.  
Lemma 2 – The covariance matrix of 𝑀 predictors ( 𝑅) can be decomposed as  
𝑅= 𝐺+ 𝛩. 𝐺 is a rank-𝐾 matrix relating to the covariance of latent models and 𝛩 is a diagonal matrix 
represents the variances of random noise. If the variance of latent models is larger than the variance 
of random noise, the number of the latent models (𝐾) can be determined by the number of leading 
components in the eigen decomposition of the covariance matrix (𝑅).  
Proof of Lemma 2 is provided in the Supplemental materials. 
According to Lemma 2, to determine the number of the latent models in a set of standardized 
predictors, the sample covariance matrix needs to be first calculated. Then, the number of leading 
eigenvalues can be estimated by using the principal component analysis on the sample covariance 
matrix (Mishra et al., 2017).  

12 
 
2.4 Solving the structured unsupervised ensemble learning model via two novels correlation-
based algorithms 
The proposed structured unsupervised ensemble learning method can be characterized as a statistical 
model involving latent variables. While the Expectation-Maximization (EM) algorithm is a 
conventional method for estimating models with latent variables, it has its limitations. EM's 
performance is notably sensitive to the initial values, and it can be computationally expensive, 
particularly when dealing with a large number of parameters. To address these challenges and arrive at 
a unique optimal solution for the proposed model while maintaining computational efficiency, we 
introduce two innovative correlation-based decomposition algorithms.  
The proposed algorithms consist of two sequential steps: optimal assignment of predictors to 
latent groups and estimation of unknown parameters in the model. The initial steps of both algorithms 
are identical. They involve standardizing the scores obtained from predictors and determining the 
optimal latent structure of the predictors. This includes identifying the number of latent models and 
assigning predictors to these latent models. This is achieved through the eigen decomposition of the 
covariance matrix, as elucidated in Lemma 2, and the use of a quartet-based similarity matrix, as 
explained in Section 2.5. The quartet-based similarity matrix is derived from the covariance matrix and 
is demonstrated to be effective in accurately distinguishing between predictors associated with the same 
latent model and those associated with different latent models, as highlighted in Lemma 3.  
Using the optimal latent structure identified through the quartet-based similarity matrix, we can 
infer the parameters in the proposed method. These inferences rely on the relationships between model 
parameters and elements in the covariance matrix, as illustrated in Lemma 1. However, estimating 
these parameters directly from the covariance matrix poses challenges, as it constitutes a nonconvex 
problem for which no closed-form solution exists.  
To tackle this issue, we introduce two algorithms in the second step. The first algorithm, 
denoted as SUEL.CQO, approximates the covariance matrix using a set of exponential functions and 
subsequently estimates the parameters through the solution of a constrained quadratic optimization 
problem. This process is detailed in Section 2.6.  

13 
 
Recognizing that the exponential approximation of the covariance matrix may introduce 
significant bias, we have introduced an alternative algorithm, SUEL.MF, detailed in Section 2.7. This 
matrix-factorization-based approach is designed to achieve more accurate estimation of the model's 
parameters. It decomposes the covariance matrix, as demonstrated in Lemma 2, and estimates the 
model parameters using a coordinate descent algorithm. Ultimately, the weight assigned to each 
predictor is estimated based on Property 2. These weights are then employed to construct the ensemble 
model. 
2.5 Determining the optimal latent structure using quartet-based similarity matrix 
The first step in our proposed method involves determining the optimal number of latent models and 
assigning predictive scores to each latent model. To achieve this, we introduce a quartet-based similarity 
matrix, which facilitates the identification of the global optimal assignments of predictive scores to 
latent models with relatively low computational cost (as outlined in Algorithm 1). 
The quartet-based similarity matrix is constructed based on correlations between scores. Our 
theoretical analysis demonstrates that dependent predictors, those associated with the same latent model, 
tend to exhibit larger similarities in the quartet-based similarity matrix compared to conditionally 
independent predictors, particularly when the accuracy of predictors is sufficiently high. Consequently, 
we can derive the optimal assignments of predictors by employing hierarchical clustering on the quartet-
based similarity matrix, grouping similar objects into the same cluster. Specifically, the quartets of a 
covariance matrix are represented as the determinants of every 2×2 submatrix.  
Definition 1 – The quartet of four predictors, 𝑓$(𝑥), 𝑓`(𝑥), 𝑓M(𝑥) , 𝑓ƒ(𝑥) , is defined as 𝑈$`Mƒ =
det ˆ𝑟$`
𝑟$ƒ
𝑟M`
𝑟Mƒ‰.                                                                
Based on the low rank structure of covariance matrix described in Lemma 1, the quartets 
defined above are sparse, with the following property. 
Property 3 –  𝑈$`Mƒ = 0 if and only if three or more of the predictors (𝑖, 𝑗, 𝑘, 𝑎𝑛𝑑 𝑙) are associated with 
the same latent model or they belong to four different latent models.  
Proof of Property 3 is provided in the Supplemental materials. 

14 
 
To detect the predictors strongly violating the conditional independence assumption and fall 
into the same latent model, the quartet-based similarity between each pair of predictors, denoted as 𝑠$`, 
is defined by aggregating all quartets including this pair of predictors. 
Definition 2 – The quartet-based similarity between predictors 𝑓$(𝑥) and 𝑓`(𝑥) is defined as: 𝑠$` =
∑
𝑈$`Mƒ
MŠ$,`;ƒŠ$,`
. 
Intuitively, each element of the quartet-based similarity matrix represents the similarity 
between a pair of predictors. The large similarity indicates that the two predictors are more likely to be 
correlated by the same latent model. When 𝑓$(𝑥) and 𝑓`(𝑥) are associated with different latent models, 
the similarity between them is estimated by aggregating their associated quartets, most of which come 
from four different latent models and thus have values equal to zero (Property 3). This fact leads to the 
small value of similarity.  
When 𝑓$(𝑥) and 𝑓`(𝑥) are associated with the same latent model, on the other hand, their 
similarity is large because most of the associated quartets have non-zero and positive values. Therefore, 
the similarity between predictors associated with the same latent model is always higher than the 
similarity of conditional independent predictors if the accuracy of predictors is high enough, which is 
rigorously proved in Lemma 3.   
Assuming 1) the prior distribution of two classes is balanced, i.e. π = 0.5; 2) the absolute value 
of conditional covariance between dependent predictors are lower bounded by 𝜃, i.e. 
•𝑐𝑜𝑣<𝑓$(𝑥), 𝑓`(𝑥)•𝑦= 1=• ≥𝜃 and •𝑐𝑜𝑣<𝑓$(𝑥), 𝑓`(𝑥)•𝑦= 0=• ≥𝜃, and 3) the accuracy of predictors 
are lowered bounded by 𝛿, i.e. •𝑤`• ≥𝛿, the following lemma can be obtained: 
Lemma 3- Under the assumption described above if  𝑓$(𝑥) and 𝑓`(𝑥) are associated with the same 
latent model then 
𝑠$` > 1
4 𝜃𝛿F𝑀F ’1 −3
𝐾+ 2
𝐾F• 
and if  𝑓$(𝑥) 𝑎𝑛𝑑 𝑓`(𝑥) are associated with different latent models, then  
𝑠$` < 𝑀F
𝐾’5 −6
𝐾•. 
Proof of Lemma 3 is provided in the Supplemental materials. 

15 
 
From Lemma 3, we can draw the conclusion that if 𝛿F ≥ 
FD
™(x\F) , where 𝛿F represents a 
threshold of predictor accuracy,  the lower bound of the quartet-based similarity between two predictors 
associated with the same latent model exceeds the upper bound of the quartet-based similarity between 
two predictors associated with different latent models. Consequently, the similarities between 
conditionally independent predictors are consistently lower than the similarities between conditionally 
dependent predictors. The values in similarity matrix exhibit a clear gap that even a simple single-
linkage hierarchical clustering algorithm can recover the correct group structure of predictors. Therefore, 
the hierarchical clustering algorithm can be used. Utilizing hierarchical clustering on the predictors, we 
can identify groups of predictors that exhibit similar quartet-based similarity values, signifying that they 
belong to the same latent model. If the hierarchical clustering algorithm does not work, more robust 
clustering algorithms, such as the spectral clustering, can be used. Thus, the process of determining the 
number of latent models and assigning predictors to each latent model is succinctly outlined in 
Algorithm 1. 
2.6 Estimating the model parameters using a constrained quadratic optimization 
(SUEL.CQO) problem 
We are interested in estimating the model parameters, including the component-specific means and 
variance in the latent models (𝜇M
D, 𝑘= 1, … , 𝐾), strength of correlation between the predictors and the 
latent models (𝑏$,𝑖= 1, … , 𝑀) , and the prior distribution of outcome variable ( 𝜋 ), to infer the 
accuracy of each score. Given that the covariance matrix between standardized predictors can be 
expressed as a piecewise function concerning the model parameters, we employ a collection of linear 
Algorithm 1: Estimating the number of latent models (𝐊) and assignments of scores to latent 
models (𝐂𝐤𝐬) 
Input: Prediction scores from 𝑀 predictors on 𝑁 samples (𝑓$(𝑥+),𝑖∈{1, … , 𝑀}, 𝑡∈{1, … , 𝑁}) 
1. 
Standardize the prediction scores. 
2. 
Estimate the covariance matrix between standardized scores. 
3. 
Estimate the number of latent groups by computing the rank of estimated covariance matrix 
using the optimal number of components in principal component analysis (PCA). 
4. 
Obtain the quartet-based similarity matrix (𝑆Ÿ). 
5. 
Estimate 𝐶Ms  by performing hierarchical clustering on the quartet-based similarity matrix. 
Output: Number of latent groups (𝐾)  and the assignments (𝐶M, 𝑘∈{1, … , 𝐾}) 

16 
 
equations to approximate the logarithm of the covariance matrix. Subsequently, we address the 
estimation of these unknown parameters by resolving a constrained quadratic optimization problem. 
Specifically, the covariance matrix has a block-wise structure where the diagonal blocks 
represent the covariance among the scores that are grouped to the same latent model and off-diagonal 
blocks represent the covariance between the scores that are conditionally independent.  According to 
Lemma 1, elements in the covariance matrix can be represented by a piecewise function on the model 
parameters. Based on these relationships, the elements in the diagonal blocks of the covariance matrix 
are decomposed as •𝑟$`• = 𝑒 ¡. 𝑒¢H. 𝑒¢£, with 𝑒 ¡, 𝑒¢H, and 𝑒¢£ represent |𝑏$|,|𝑏`|, and 𝜎iF respectively, 
∀𝑖, 𝑗∈𝐶M, 𝑘= 1, … , 𝐾. In the off-diagonal blocks of the covariance matrix (i.e. 𝑖∈𝐶¤, 𝑗∈𝐶¥, 𝑣≠
𝑢, 𝑢, 𝑣∈{1, … , 𝐾}), the elements are decomposed as •𝑟$`• = 𝑒§. 𝑒¢H. 𝑒¢£. 𝑒ƒ¨.𝑒ƒ©, where 𝑒§, 𝑒¢H,  𝑒¢£ , 
𝑒ƒ¨, and 𝑒ƒ© present 
'\[
[ , |𝑏$|, |𝑏`|, |𝜇¤D|, and |𝜇¥D|, respectively. Each element in the covariance matrix 
provides a relationship or equation between the model parameters. Therefore, we obtain a system of  
(((\')
F
  linear equations that capture the relationships between 𝑀+ 2𝐾+ 1 model parameters. Since 
the population covariance matrix (𝑅) is unknown, sample covariance matrix (𝑅ª) is used as an unbiased 
estimation. Thus, the relationships between model parameters and elements in the covariance matrix is 
formulated as a multivariate linear equation problem with random noises in the following: 
                        g
  log•𝑟®𝑖𝑗• = 𝑠𝑜+ 𝑑𝑖+ 𝑑𝑗+ 𝜖                                    𝑖, 𝑗∈𝐶𝑡  
 log•𝑟®𝑖𝑗• = 𝑝+ 𝑑𝑖+ 𝑑𝑗+ 𝑙𝑣+ 𝑙𝑢+ 𝜖         𝑖∈𝐶𝑣, 𝑗∈𝐶𝑢,
                                  (6) 
where 𝜖 represents the random noise in sample covariance matrix. The problem can be reformulated as 
𝒛= 𝐴𝒈+ 𝝐, where 𝒛 is a 
(((\')
F
 dimensional vector representing log of the upper triangular part of the 
sample covariance matrix, 𝐴 is a 
(((\')
F
× (𝑀+ 2𝐾+ 1) index matrix provides the indexes of each 
parameter in each equation. 𝒈 is a  𝑀+ 2𝐾+ 1 dimensional vector including all unknown parameters. 
As the value of   log•𝑟¶·
¸• is always nonpositive, we require the estimated parameters to be less than or 
equal to 0. The constrained quadratic optimization problem is formulated as: 
                   minimize 
𝒈
‖𝐴𝒈−𝒛‖F
F , subject to  𝒈≤𝟎.                                                (7) 

17 
 
The solutions of Equation 5.7 can be efficiently obtained and transferred back to the model parameters 
using 𝜋=
'
'¿ÀÁ, |𝑏$| = 𝑒¢H, and |𝜇¤D| = 𝑒ƒ¨. The sign of the estimated accuracy for each score is 
determined by the sign of its correlation coefficient with other scores. Algorithm 2 summarizes the steps 
of SUEL.CQO. 
Algorithm 2: Estimating the model parameters required to compute ensemble weights 
using SUEL.CQO 
Input: The assignments (𝐶M, 𝑘∈{1, … , 𝐾}), sample covariance matrix( 𝑅ª), prediction scores 
(𝑓$(𝑥+), 𝑖∈{1, … , 𝑀}, 𝑡∈{1, … , 𝑁}), number of latent groups (𝐾) 
1. 
Formulate the linear system of equations as Equation 5.6. 
2. 
Solve the constrained quadratic optimization problem in Equation 5.7 to estimate the 
model parameters ( 𝜋®, 𝑏ª, 𝜇D
¸). 
3. 
Estimate the accuracy of each prediction score as 𝑤$ = −
XHYS
Z
[\('\[)XH
IYS
ZI.  
Output: Compute the ensemble scores as  𝑔(𝑥+) = ∑
𝑤$𝑓$(𝑥+)
(
$&'
. 
2.7 Estimating the model parameters using a matrix-factorization-based (SUEL.MF) 
approach 
While the SUEL.CQO algorithm offers a more cost-effective means of estimating model parameters in 
comparison to the EM algorithm, there are potential issues with the accuracy of the parameters 
estimated through this approach. This can be attributed to:1) the logarithm transformation of covariance 
matrix reduces the magnitude of estimation bias; 2) The presence of bias in the constrained quadratic 
optimization problem, particularly when the number of equations in the linear system of equations 
exceeds the number of variables. To address these issues, we further proposed a matrix-factorization-
based algorithm (SUEL.MF). This algorithm directly decomposes the covariance matrix to estimate 
unknown parameters and has the potential to provide more accurate estimations compared to 
SUEL.CQO. 
According to Lemma 2, the covariance matrix can be decomposed to three matrices, the 
coefficients between scores and latent models (𝐵), The covariance of latent models (Λ), and a diagonal 
matrix that represents the variance of random noise (Θ), as shown in Equation 8. Therefore, by 
estimating 𝐵, Λ, and Θ , we can estimate the parameters that are required to compute the reliability of 
each predictor (𝑤$ ). Representing the coefficients between scores and latent models as a sparse 

18 
 
coefficient matrix 𝐵, i.e. B = Æ
𝒃'
⋯
0
⋮
⋱
⋮
0
⋯
𝒃x
Ê
(×x
with 𝒃$ indicates the vector of coefficients associated 
with 𝑖th latent model, the covariance of latent models as Λ = Ë
𝜎'
F
⋯
'\[
[ 𝜇'
D 𝜇x
D
⋮
⋱
⋮
'\[
[ 𝜇'
D 𝜇x
D
⋯
𝜎x
F
Ì
x×x
and the 
variance of random noise as Θ = Í
𝜃'
F
⋯
0
⋮
⋱
⋮
0
⋯
𝜃(
F
Î
(×(
, the covariance matrix between predictors can be 
decomposed as: 
𝑅= 𝐵Λ𝐵| + Θ                                                            (8) 
The Λ matrix can be further decomposed to eigenvectors 𝑈 and eigenvalues 𝑉, where the eigenvectors 
are related to the unknown parameters 𝜇'
D, … , 𝜇x
D . To estimate the unknown parameters 𝐵, 𝑈, 𝑉, and Θ, 
the following optimization problem needs to be solved: 
minimize ‖𝑅−𝐵𝑈𝑉(𝐵𝑈)| + Θ‖Ð
F,  
subject to 𝑈𝑈| = 𝐼 and 𝐵 is sparse with off-diagonal blocks are 0.                 (9) 
To solve the optimization problem proposed formulated in Equation 9, a block coordinate descent 
(BCD)-based algorithm is introduced. The BCD algorithms break down a given matrix-based 
nonconvex optimization problem into convex subproblems (based on blocks of columns) which are 
easier to solve. To solve the problem presented in Equation 9, we adopt an iterative two-stage approach 
that estimate Θ and then estimate 𝐵 and 𝑈 alternatively.  
Estimation of 𝚯: When 𝐵 and 𝑈 are fixed, the diagonal matrix Θ can be estimated via Θ = 𝑑𝑖𝑎𝑔(𝑅−
𝐵𝑈𝑉(𝐵𝑈)|). 
Estimation of 𝑩 and 𝑼: When Θ is fixed, the estimation of 𝐵 and 𝑈 can be obtained by the following 
steps. First, we decompose the partial residual matrix, 𝑅−Θ, to its eigenvectors (𝑊) and eigenvalues 
(𝑉), where the columns of  𝑊 contain the first 𝐾 leading eigenvectors of the partial residual matrix. 
Then, the parameters 𝐵 and 𝑈 can be estimated by further decomposing the matrix of first 𝐾 leading 
eigenvectors to a sparse and structured matrix and an orthogonal matrix according to Equation 9. To 
obtain a sparse and structured estimation of 𝐵, we update each column in it alternatively. Specifically, 

19 
 
considering one column of 𝐵, 𝐵:$, as an unknown and treating the rest columns of 𝐵 and the whole 
of 𝑈 as constants, we reformulate an optimization problem in Equation 10 to estimate 𝐵:$: 
                                 minimize𝐹(𝐵:$) = ||𝑊∗−∑
𝐵:`
∗𝑈:`
∗| −𝐵:$𝑈:$
∗|
$Š`
||Ð
F.                     
subject to 𝐵M$ = 0 if 𝑘th score is not associated with 𝑖th latent model.                (10) 
Denote 𝑅∗= 𝑊∗−∑
𝐵:`
∗𝑈:`
∗|
$Š`
 , the solution of 𝐵:$ in Equation 10 can be obtained by assigning 𝑈:$
∗∗
𝑅M:
∗ to the non-zero elements in 𝐵:$, where * shows the summation of element-wise multiplication. Each 
column in the coefficient matrix can be updated alternatively using the same approach.  
The problem of estimating 𝑈 can be formulated as an orthogonal Procrustes problem (Gower, 
2004) as the following, we can be efficiently solved by the singular value decomposition: 
        minimize𝐹(𝑈) = ||𝑊∗−𝐵∗𝑈||Ð
F, subject to 𝑈𝑈| = 𝐼                         (11) 
This problem can be regarded as finding the nearest orthogonal matrix to a given matrix 𝑆= 𝐵∗|𝑊∗. 
The singular value decomposition algorithm can be used to solve it. Algorithm 3 summarizes the steps 
of estimating 𝐵, 𝑈, and 𝑉 using the proposed BCD-based algorithm.  
After estimating 𝐵 and 𝑈, the ensemble weights are computed using the algorithm presented in 
Algorithm 4. The values of  𝒃 are obtained directly from estimated matrix 𝐵 and  z'\[
[ 𝜇Ds are 
estimated as the elements of eigenvectors corresponding to the leading eigenvalue of Λ.  
Algorithm 3: Estimating 𝑩, 𝑼, and 𝑽 
Input: 𝑅 
1. Initialize ΘD, 𝑈D, and 𝐵D. 
2. For 𝑗 =1 to 𝑛_𝑖𝑡𝑒𝑟𝑎𝑡𝑖𝑜𝑛: 
3. 
Estimate 𝑊` as its columns are the first 𝐾 leading eigenvectors of 𝑅−Θ` and 𝑉` as a 
diagonal matrix which its diagonal elements are the first K leading eigenvalues of 𝑅−
Θ` . 
4. 
for 𝑖= 1 𝑡𝑜 𝐾: 
      Compute the nonzero elements of 𝐵`
∗
:$ as  𝐵`M$
∗= 𝑈`:$ ∗𝑅`
∗
M:. 
             End for 
5. 
Compute 𝑆= 𝐵`
|𝑊`. 
6. 
Compute singular value decomposition of 𝑆= 𝐷Σ𝐹|. 
7. 
Update 𝑈` as 𝑈` = 𝐷𝐹|. 
8. 
Update Θ` as 𝑑𝑖𝑎𝑔(Θ`) = 𝑑𝑖𝑎𝑔(𝑅−𝐵`𝑈`𝑉`<𝐵`𝑈`=
|). 
9. End for. 

20 
 
Output: Optimal  𝐵, 𝑈, 𝑉. 
 
Algorithm 4: Estimating model parameters required to compute ensemble weights using 
SUEL.MF 
Input: The assignments (𝐶M, 𝑘∈{1, … , 𝐾}), prediction scores (𝑓$(𝑥+),𝑖∈{1, … , 𝑀},𝑡∈
{1, … , 𝑁}), number of latent groups (𝐾), 𝐵, 𝑈, 𝑉. 
1. Compute the elements of 𝒃  from 𝐵. 
2. Compute z'\[
[ 𝜇M
Ds as the eigenvectors corresponding to the leading eigenvalue of 𝑈𝑉𝑈|. 
3. Estimate the accuracy of each predictive score as 𝑤$ = −
XHYS
Z
[\('\[)XH
IYS
ZI.  
Output: Compute the ensembled scores as  𝑔(𝑥+) = ∑
𝑤$𝑓$(𝑥+)
(
$&'
. 
3. Simulation 
In order to illustrate the effectiveness of our proposed method, we conducted a simulation study for 
model performance evaluation and compared the performance of the proposed method with other 
benchmark models. 
3.1 Simulation process 
The simulation study encompasses three distinct scenarios, each summarized as follows. In Scenario 1, 
eight conditionally independent predictors are simulated. The conditional independent predictors can 
be considered as a special case of the proposed method where the number of latent models equals to the 
number of predictors and each predictor is assigned to one latent model. Thus, the proposed method 
still can deal with it, but the advantages of the proposed method will not be reflected in this scenario.  
To demonstrate the advantages of the proposed method in capturing latent group structure among 
predictors and distinguishing their discriminability and stability, we further consider four latent models 
with different discriminability and stability in Scenario 2. The first latent model (C1) is the most 
accurate one as it has both high discriminability (i.e., difference of component- specific means) and 
stability (i.e., variance). The second latent model (C2) has low discriminability but high stability. The 
third latent model (C3) has high discriminability but low stability, and the fourth latent model (C4) is 
the least accurate one as it has both low discriminability and stability. 9, 6, 7, and 5 predictors are 
randomly simulated from each latent model with different strength of correlation. Scenario 3 simulates 

21 
 
two unbalanced latent models with one latent model (C1) associated with much larger number of 
predictors (20 predictors) than the other one (2 predictors in C2).  
400 samples are simulated in each scenario, whose binary outcomes (𝑦+ ) are randomly 
generated using a Bernoulli distribution with the prior probability of belonging to class 1 set at 0.8.  
Conditioning on the outcome of each sample, its scores from latent models, denoted as ℎM(𝑥+|𝑦+), are 
generated randomly using a two-component Gaussian distribution with pre-determined component-
specific means and variances as summarized in Table 1. The variances in the two components are equal. 
The strength of correlation between latent models and each predictor (𝑏$) is generated under constraints 
to ensure that the weights estimated using Property 2 are identical with their maximal likelihood 
estimations in Property 1. The random noises (𝜀$) are generated randomly using standard normal 
distribution, i.e. 𝜀$~𝑁(0,1). Therefore, the predictive score of sample 𝑡 from 𝑖th predictor is simulated 
by 𝑓$(𝑥+) = 𝑏$ℎM(𝑥+|𝑦+) + 𝜀$. Finally, the predictive scores of 400 samples from each predictor are 
standardized to have mean of 0 and variance of 1.  
Table 1. Component-specific means and variances for each scenario. 
Scenario 1 
Scenario 2 
Scenario 3 
𝜇M
D = [-5, -1, -5.2, -1.04, -5, -1, -
5.2, -1.04] 
𝜇M
' = [1.25, 0.25, 1.3, 0.26, 1.25, 
0.25, 1.3, 0.26] 
𝜎M =[4, 5, 6, 7, 8, 9, 10, 11] 
𝜇M
D = [-5, -1, -5.2, -1.04] 
𝜇M
' = [1.25, 0.25, 1.3, 0.26] 
𝜎M =[6, 6, 3, 3] 
𝜇M
D = [-5, -1] 
𝜇M
' = [1.25, 0.25] 
𝜎M =[5, 2] 
 
3.2 Benchmark models and evaluation metrics 
The prediction performance of SUEL.CQO and SUEL.MF approaches is compared with three 
benchmark models, Eigen model, the average-based unsupervised ensemble learning, and the ground 
truth. The Eigen model assumes that each predictor has a two-component Gaussian distribution and its 
ensemble weight is proportional solely to the difference of component-specific means. It neglects the 
influence of component-specific variance and considers only the covariance among pairs of predictors 
belonging to different blocks (Ionita-Laza et al., 2016). Notably, the proposed SUEL.MF and 
SUEL.CQO approaches explicitly leverage predictor dependencies and enhance accuracy estimation 

22 
 
by considering both distinguishability and stability. Therefore, they hold promise for outperforming the 
Eigen method, which does not account for these factors in the same manner. In addition, an average-
based unsupervised ensemble learning model is also compared, which simply takes the average of 
predictive scores. Furthermore, a ground truth model, which uses the ground truth accuracy of each 
predictor as the ensemble weight is compared. Lastly, the best individual predictor’s performance is 
incorporated for model comparison.  
The prediction accuracy of each ensemble model is evaluated through area under ROC (ROC-
AUC), area under Precision-Recall curve (PRC-AUC) and the decile analysis. In the decile analysis, 
the samples are segmented into 10 deciles based on the rank of their predicted ensemble scores. The 
distribution of positive samples across 10 deciles is visualized using the histogram. More positive 
samples allocating in the first decile indicates a more accurate prediction of the ensemble scores. To 
test the goodness-of-fit of the proposed method, the estimated weights are compared with the ground 
truth weights in each scenario.  
3.3 Results 
The optimal number of latent models and the assignment of predictors to different latent models are 
identified by the elbow method using the principal component analysis of sample covariance matrix 
and the hierarchical clustering on the quartet-based similarity matrix, respectively. The number of latent 
models and the assignment of scores are consistent with our pre-identified structures. Figure 3 shows 
the percentage of variance explained by different number of principal components in all scenarios. The 
knee of the curve represents the optimal number of latent variables which is equal to 8, 4, and 2 for 
Scenario 1, Scenario 2, and Scenario 3, respectively. Then, the number of latent models is set to the 
optimal number of latent models and optimal assignment of predictors to latent models are obtained 
using hierarchical clustering on the quartet-based similarity matrix. Figure 4 visualizes the structure of 
the quartet-based similarity matrix in all scenarios, clearly showing higher similarities among scores 
within each latent model and lower similarities between scores from different latent models. 
The prediction performance of our proposed models (SUEL.MF and SUEL.CQO) and four 
baseline models under different scenarios is summarized in Table 2. In all scenarios, SUEL.MF 

23 
 
outperforms the other ensemble approaches while SUEL.CQO is the second-best approach. The 
performance of the proposed approaches is also close to the ground truth model, which further 
demonstrates the effectiveness of the proposed algorithm. In addition, the estimated weights using 
SUEL.MF approach showcases high correlations with the true weights in all scenarios (Figure 5). 
Furthermore, in Scenario 2, we observe that predictors associated with the latent model C1, which has 
both high discriminability and stability, are ranked highest and the scores associated with the latent 
model C4, which has both low discriminability and stability, are ranked lowest in the proposed method. 
This observation is consistent with our prior knowledge of the simulation model. Figure 6-8 show the 
distribution of positive samples across different decile of prediction scores. In all scenarios, the 
proposed methods (MF and SUEL-CQO) have higher number of positive samples in the first few deciles 
of prediction scores compared to the benchmark models, which demonstrates that the scores predicted 
from the proposed methods can better distinguish the positive samples with negative ones. Moreover, 
the MF-based algorithm can achieve the same accuracy with the ground truth in the first decile. 
 
 
Figure 3. The percentage of explained variance versus the number of components. 
 
 
Figure 4. Heatmap of quartet-based similarity matrix 

24 
 
 
 
Figure 5. Estimated weights versus true weights in SUEL.MF under different scenarios. 
Table 2. The prediction performance of different approaches under three scenarios. 
Prediction 
models 
Scenario 1 
Scenario 2 
Scenario 3 
ROC 
PRC 
ROC 
PRC 
ROC 
PRC 
SUEL.MF 
0.92 
0.79 
0.93 
0.85 
0.80 
0.57 
SUEL.CQO 
0.92 
0.78 
0.93 
0.84 
0.80 
0.56 
Eigen 
0.89 
0.72 
0.90 
0.80 
0.73 
0.49 
Average 
0.85 
0.66 
0.84 
0.63 
0.76 
0.54 
Best score 
0.88 
0.70 
0.86 
0.67 
0.77 
0.37 
Ground truth 
0.93 
0.83 
0.93 
0.85 
0.80 
0.58 
 
 
Figure 6. The distribution of positive samples across different decile of prediction models in Scenario 
1. 

25 
 
 
Figure 7. The distribution of positive samples across different decile of prediction models in Scenario 
2. 
 
Figure 8. The distribution of positive samples across different decile of prediction models in Scenario 
3. 

26 
 
4. Case study 
4.1 Disease-associated risk genes discovery 
Autism Spectrum Disorder (ASD) is a complex neurodevelopmental disorder characterized by 
the deficits in social communication, impaired language development, and the presence of highly 
restricted interests and/or stereotyped repetitive behaviors (State & Levitt, 2011). The early diagnosis 
and treatment of ASD is still a challenge due to the very limited understanding of its underlying 
pathology. Recent advances in the Genome-Wide Association Studies (GWAS) of large-scale 
population have demonstrated the role of genetics and genomics in ASD development which are 
promising to facilitate the development of novel screening approaches and effective personalized 
medication for ASD (C Yuen et al., 2017; Hallmayer et al., 2011; Turner et al., 2017). However, the 
knowledge of risk genes associated with ASD is still very far from complete (Lin et al. 2020). Therefore, 
numerous computational models have been developed in the literature to rank and score the genes by 
their likelihood of association with ASD (C. Zhang & Shen, 2017); (Iossifov et al., 2015); (Krishnan et 
al., 2016); (Sanders et al., 2015). 
These scoring systems aim to predict the likelihood that each gene is linked to ASD. They 
incorporate different types of scores, such as statistical scores derived from extensive Genome-Wide 
Association Studies (GWAS), network-based scores that consider gene interactions, and scores 
generated using machine learning techniques. Despite these efforts, the performance of existing scoring 
systems remains below the desired level. Given that various predictive scores can offer complementary 
insights into disease-gene associations, the prospect of combining these existing predictive scores into 
a new ensemble score holds significant potential for enhancing the discovery of risk genes associated 
with ASD. This approach aims to create an ensemble score with improved robustness and accuracy, 
drawing upon the strengths of multiple predictive measures. 

27 
 
4.2 Data Description 
In this study, we aggregate 20 ranking scores developed for ASD risk gene discovery in the existing 
literature which were built under various approaches. These scores encompass 4 machine learning 
model-based gene ranking scores for ASD, including Zhang D (C. Zhang & Shen, 2017), IOSSIFOV 
(Iossifov et al., 2015), Krishnan (Krishnan et al., 2016), Sander (Sanders et al., 2015). We also consider  
9 gene-level constraint metrics and general gene features developed from the exome data of more than 
60,000 individuals from the ExAC to quantify the sensitivity of genes to variations (e.g.  Z scores for 
synonymous which shows whether the gene is more intolerant of variation). As risk genes are usually 
functionally correlated via gene networks, we further consider 7 gene network’s topological scores (e.g. 
the hubness of each gene in the network) obtained from prior research to rank the genes based on their 
functional connectivity (Lin, Afshar, Rajadhyaksha, Potash, & Han, 2020). The 20 ranking scores are 
curated for 14,642 genes in total. Among them 1,016 genes were previously implicated to be 
associated/independent with the ASD (115 are associated with ASD and 901 are independent with ASD). 
We use all 14,642 genes to train the proposed unsupervised ensemble learning model and 1,016 labeled 
genes to evaluate the model performance. Additionally, we conduct an external evaluation using a set 
of 214 independent ASD candidate genes. All scores are standardized with a mean of zero and a 
standard deviation of one. 
4.3 Evaluation metrics 
To evaluate the effectiveness of our proposed method, we employ various metrics including ROC-AUC 
and PRC-AUC, along with a downstream analysis. The downstream analysis involves computing the 
enrichment of ASD risk genes within each decile of the sorted predicted scores from different models. 
We anticipate that more accurate predictions will yield a higher fraction of ASD genes in the first decile. 
In our evaluation, we compare the prediction performance of the proposed approaches (SUEL.MF and 
SUEL.CQO)  to several other models: Eigen, the average-based ensemble score, and the best prediction 
performance obtained from individual scores. Furthermore, we conduct a sensitivity analysis to gauge 
how inaccurately estimating the number of latent models can impact the prediction performance of our 

28 
 
proposed models and the benchmark models. Specifically, we compare the prediction performance of 
SUEL.MF and Eigen models by varying the assignment of predictors to different numbers of latent 
models. 
4.4 Results 
To identify the optimal number of latent models, we employ the elbow method and principal 
component analysis on the covariance matrix of standardized scores. As illustrated in Figure 9, we 
determine that three latent models provide the best fit for our data. Subsequently, leveraging the quartet-
based similarity matrix, showcased in Figure 10, we employed hierarchical clustering to categorize the 
20 ranking scores into three latent models. It becomes evident that the first group predominantly consists 
of ASD ranking scores found in existing literature, the second group is comprised of gene-level 
constraint metrics, and the third group exclusively encompasses topological features.  
With the latent model structure identified, we proceeded to estimate model parameters using 
our two proposed algorithms. Our evaluation involved comparing the prediction performance of 
SUEL.CQO and SUEL.MF with three benchmark models, utilizing the test gene dataset. The results in 
Table 3 demonstrate that, SUEL.MF outperforms the benchmark models with respect to the ROC-AUC. 
However, the Sander score has higher PRC-AUC compared to the proposed method. It's worth 
mentioning that the Sander score was trained using a dataset that highly overlaps with our test data, 
encompassing around 50% of the test genes, possibly introducing bias. Figure 11 illustrates the 
estimated weights assigning to each ranking system. ExaC, IOSSIFOV and IOF_Z scores have the 
highest estimated weights among different ranking scores in the model. Thus, these ranking systems 
showcase the highest accuracy compared to other ranking scores in the ASD gene prediction. This 
finding can help us to assess the prediction performance of different ranking scores without using any 
labeled data. 
In Figure 12, our downstream analysis evaluates predictions from different methods using 
external independent data. The results indicate that ASD-associated genes were most enriched in the 
first decile of scores predicted by the proposed SUEL.MF method, demonstrating its superior 
performance in ranking ASD-associated risk genes compared to the other three models.  

29 
 
Furthermore, our sensitivity analysis reveals that while the SUEL.MF algorithm outperforms 
the Eigen method under the optimal number of latent models (three latent models), the prediction 
performance of SUEL.MF significantly declines in terms of ROC-AUC and PRC-AUC when latent 
models are inaccurately determined (see Figure 13). This is in contrast to the Eigen approach, which 
exhibited lower sensitivity to the number of latent models, as it does not exploit the latent structure in 
model estimation by assuming the scores are conditional independent. In the SUEL.MF approach, 
however, changing the number of latent models leads to the changes in model structure and number of 
parameters, which further leads to the significant drop in prediction performance of the model. 
Therefore, it is critical to find the optimal latent structure in the correlated scores using the proposed 
algorithm. 
 
 
Figure 9. The percentage of explained variance versus number of principal components. 
 

30 
 
 
Figure 10. The heatmap of quartet-based similarity matrix for ASD case study.  
 
 
 
Figure 11. The estimated weights of SUEL.MF approach for ranking systems. 
Table 3. The prediction performance of different approaches. 
 
SUEL.CQO 
Eigen 
SUEL.MF 
Average 
Best score 
ROC-AUC 
0.84 
0.82 
0.84 
0.66 
0.78 
PRC-AUC 
0.42 
0.42 
0.47 
0.20 
0.67 
 
 

31 
 
 
Figure 12. The distribution of ASD genes across different decile of prediction models. 
 
 
Figure 13. The impact of number of latent models in the accuracy of different models. 
5. Conclusion 
In conclusion, we have developed a novel Structured Unsupervised Ensemble Learning (SUEL) 
approach designed to seamlessly integrate dependent and continuous predictive scores in absence of 
labeled data. The proposed method estimates the unknown accuracy of each predictor by considering 
the latent structure among predictors, the distinguishability (mean difference) and the stability (variance) 
of each predictor. Two novel correlation-based decomposition algorithms, SUEL.CQO and SUEL.MF, 
have been proposed to find the optimal latent structure and estimate the model parameters. To evaluate 

32 
 
the proposed method, the prediction performance of SUEL.MF and SUEL.CQO are compared to other 
benchmark models using both simulated data and real-world data of ASD risk gene prediction. The 
results obtained from both simulated and real-world data demonstrate that SUEL.MF outperformed 
other benchmark models. Moreover, this approach can efficiently discover the dependency between 
predictors, accurately estimate the accuracy of each predictor, and optimally integrate the predictors to 
an ensemble model which leads to the most significant improvement in the prediction accuracy 
compared to the individual predictors.  
There are some limitations of this work. First, the proposed method can be applied to various 
unsupervised ensemble learning problems in the engineering systems. In the future, we will study and 
evaluate its effectiveness in the other domains, such as ranking and integrating various in-situ sensors 
for fault prognostics in the manufacturing system (Kolokas, Vafeiadis, Ioannidis, & Tzovaras, 2020) 
and ranking and integrating multiple clustering structures for consensus clustering (Ünlü & 
Xanthopoulos, 2019). Second, the proposed method is developed to rank and combine the scores from 
two balanced classes. In the future, we will extend the proposed model to predict more than two classes. 
Imbalanced classification problem is commonly observed in real applications, such as the abnormal 
events are much less than the normal ones in engineering systems. To solve these imbalanced 
classification problem, we will leverage the extreme event distributions to capture the latent models and 
their structure in the proposed method. Last but not least, the proposed method can be further extended 
to rank and combine scores coming from more complex latent structures, such as the tree structures.  
Acknowledgements 
This research was funded by NIMH, R01MH121394  
 
 
 
 

33 
 
References 
Araya, D. B., Grolinger, K., ElYamany, H. F., Capretz, M. A., & Bitsuamlak, G. (2017). An ensemble 
learning framework for anomaly detection in building energy consumption. Energy and 
Buildings, 144, 191-206.  
Breiman, L. (1996). Bagging predictors. Machine Learning, 24(2), 123-140.  
Brueggeman, L., Koomar, T., & Michaelson, J. J. (2020). Forecasting risk gene discovery in autism 
with machine learning and genome-scale data. Scientific Reports, 10(1), 1-11.  
C Yuen, R. K., Merico, D., Bookman, M., L Howe, J., Thiruvahindrapuram, B., Patel, R. V., . . . Wang, 
Z. (2017). Whole genome sequencing resource identifies 18 new candidate genes for autism 
spectrum disorder. Nature Neuroscience, 20(4), 602-611.  
Cabrera-Bean, M., Díaz-Vilor, C., & Vidal, J. (2016). Impact of noisy annotators' reliability in a 
crowdsourcing system performance. Paper presented at the 2016 24th European Signal 
Processing Conference (EUSIPCO). 
Caltanissetta, F., Bertoli, L., & Colosimo, B. M. (2023). In-situ Monitoring of Image Texturing via 
Random Forests and Clustering with applications to Additive Manufacturing. IISE 
Transactions(just-accepted), 1-30.  
Cao, Z., Huang, Y., Duan, R., Jin, P., Qin, Z. S., & Zhang, S. (2022). Disease category-specific 
annotation of variants using an ensemble learning framework. Briefings in Bioinformatics, 
23(1), bbab438.  
Chen, T., & Guestrin, C. (2016). Xgboost: A scalable tree boosting system. Paper presented at the 
Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data 
mining. 
Chung, S., & Al Kontar, R. (2024). Federated multi-output Gaussian processes. Technometrics, 66(1), 
90-103.  
Craig, J. (2008). Complex diseases: research and applications. Nature Education, 1(1), 184.  
Daimon, T. (2011). Box–cox transformation. In International Encyclopedia of Statistical Science (pp. 
176-178): Springer. 

34 
 
Dawid, A. P., & Skene, A. M. (1979). Maximum likelihood estimation of observer error-rates using the 
EM algorithm. Journal of the Royal Statistical Society: Series C (Applied Statistics), 28(1), 20-
28.  
Devlin, B., & Scherer, S. W. (2012). Genetic architecture in autism spectrum disorder. Current Opinion 
in Genetics & Development, 22(3), 229-237.  
Erekat, A., Servis, G., Madathil, S. C., & Khasawneh, M. T. (2020). Efficient operating room planning 
using an ensemble learning approach to predict surgery cancellations. IISE Transactions on 
Healthcare Systems Engineering, 10(1), 18-32.  
Gaw, N., Yousefi, S., & Gahrooei, M. R. (2022). Multimodal data fusion for systems improvement: A 
review. IISE Transactions, 54(11), 1098-1116.  
Gower, J. (2004). Dijksterhuis, GB: Procrustes problems. In: Oxford University Press. 
Hallmayer, J., Cleveland, S., Torres, A., Phillips, J., Cohen, B., Torigoe, T., . . . Smith, K. (2011). 
Genetic heritability and shared environmental factors among twin pairs with autism. Archives 
of General Psychiatry, 68(11), 1095-1102.  
Ionita-Laza, I., McCallum, K., Xu, B., & Buxbaum, J. D. (2016). A spectral approach integrating 
functional genomic annotations for coding and noncoding variants. Nature Genetics, 48(2), 
214-220.  
Iossifov, I., Levy, D., Allen, J., Ye, K., Ronemus, M., Lee, Y.-h., . . . Wigler, M. (2015). Low load for 
disruptive mutations in autism genes and their biased transmission. Proceedings of the National 
Academy of Sciences, 112(41), E5600-E5607.  
Iranzad, R., Liu, X., Chaovalitwongse, W. A., Hippe, D., Wang, S., Han, J., . . . Bowen, S. (2022). 
Gradient boosted trees for spatial data and its application to medical imaging data. IISE 
Transactions on Healthcare Systems Engineering, 12(3), 165-179.  
Jaffe, A., Fetaya, E., Nadler, B., Jiang, T., & Kluger, Y. (2016). Unsupervised ensemble learning with 
dependent classifiers. Paper presented at the Artificial Intelligence and Statistics. 
Jaffe, A., Nadler, B., & Kluger, Y. (2015). Estimating the accuracies of multiple classifiers without 
labeled data. Paper presented at the Artificial Intelligence and Statistics. 

35 
 
Jain, P., & Oh, S. (2014). Learning mixtures of discrete product distributions using spectral 
decompositions. Paper presented at the Conference on Learning Theory. 
Kalia, P. (2018). Stacking Supervised and Unsupervised Learning Models for Better Performance. 
International Research Journal of Engineering and Technology (IRJET) Volume, 5.  
Kleindessner, M., & Awasthi, P. (2018). Crowdsourcing with arbitrary adversaries. Paper presented at 
the International Conference on Machine Learning. 
Kolokas, N., Vafeiadis, T., Ioannidis, D., & Tzovaras, D. (2020). A generic fault prognostics algorithm 
for manufacturing industries using unsupervised machine learning classifiers. Simulation 
Modelling Practice and Theory, 103, 102109.  
Krishnan, A., Zhang, R., Yao, V., Theesfeld, C. L., Wong, A. K., Tadych, A., . . . Troyanskaya, O. G. 
(2016). Genome-wide prediction and functional characterization of the genetic basis of autism 
spectrum disorder. Nature Neuroscience, 19(11), 1454-1462.  
Li, H., & Yu, B. (2014). Error rate bounds and iterative weighted majority voting for crowdsourcing. 
arXiv preprint arXiv:1411.4086.  
Lin, Y., Afshar, S., Rajadhyaksha, A. M., Potash, J. B., & Han, S. (2020). A machine learning approach 
to predicting autism risk genes: Validation of known genes and discovery of new candidates. 
Frontiers in Genetics, 11, 500064.  
Marudi, M., Ben-Gal, I., & Singer, G. (2022). A decision tree-based method for ordinal classification 
problems. IISE Transactions, 1-15.  
Mishra, S. P., Sarkar, U., Taraphder, S., Datta, S., Swain, D., Saikhom, R., . . . Laishram, M. (2017). 
Multivariate statistical data analysis-principal component analysis (PCA). International 
Journal of Livestock Research, 7(5), 60-78.  
Parisi, F., Strino, F., Nadler, B., & Kluger, Y. (2014). Ranking and combining multiple predictors 
without labeled data. Proceedings of the National Academy of Sciences, 111(4), 1253-1258.  
Pavlyshenko, B. (2018). Using stacking approaches for machine learning models. Paper presented at 
the 2018 IEEE Second International Conference on Data Stream Mining & Processing (DSMP). 

36 
 
Qi, Y. (2012). Random forest for bioinformatics. In Ensemble Machine Learning (pp. 307-323): 
Springer. 
Sanders, S. J., He, X., Willsey, A. J., Ercan-Sencicek, A. G., Samocha, K. E., Cicek, A. E., . . . Dong, 
S. (2015). Insights into autism spectrum disorder genomic architecture and biology from 71 
risk loci. Neuron, 87(6), 1215-1233.  
Schapire, R. E. (2013). Explaining adaboost. In Empirical Inference (pp. 37-52): Springer. 
Shaham, U., Cheng, X., Dror, O., Jaffe, A., Nadler, B., Chang, J., & Kluger, Y. (2016). A deep learning 
approach to unsupervised ensemble learning. Paper presented at the International conference 
on machine learning. 
Song, C., & Liu, K. (2018). Statistical degradation modeling and prognostics of multiple sensor signals 
via data fusion: A composite health index approach. IISE Transactions, 50(10), 853-867.  
State, M. W., & Levitt, P. (2011). The conundrums of understanding genetic risks for autism spectrum 
disorders. Nature Neuroscience, 14(12), 1499-1506.  
Tama, B. A., & Lim, S. (2021). Ensemble learning for intrusion detection systems: A systematic 
mapping study and cross-benchmark evaluation. Computer Science Review, 39, 100357.  
Traganitis, P. A., Pages-Zamora, A., & Giannakis, G. B. (2018). Blind multiclass ensemble 
classification. IEEE Transactions on Signal Processing, 66(18), 4737-4752.  
Tsogbaatar, E., Bhuyan, M. H., Taenaka, Y., Fall, D., Gonchigsumlaa, K., Elmroth, E., & Kadobayashi, 
Y. (2020). SDN-enabled IoT anomaly detection using ensemble learning. Paper presented at 
the Artificial Intelligence Applications and Innovations: 16th IFIP WG 12.5 International 
Conference, AIAI 2020, Neos Marmaras, Greece, June 5–7, 2020, Proceedings, Part II 16. 
Turner, T. N., Coe, B. P., Dickel, D. E., Hoekzema, K., Nelson, B. J., Zody, M. C., . . . Pennacchio, L. 
A. (2017). Genomic patterns of de novo mutation in simplex autism. Cell, 171(3), 710-722. 
e712.  
Ünlü, R., & Xanthopoulos, P. (2019). A weighted framework for unsupervised ensemble learning based 
on internal quality measures. Annals of Operations Research, 276, 229-247.  

37 
 
Valpola, H., & Karhunen, J. (2002). An unsupervised ensemble learning method for nonlinear dynamic 
state-space models. Neural Computation, 14(11), 2647-2692.  
Vanerio, J., & Casas, P. (2017). Ensemble-learning approaches for network security and anomaly 
detection. Paper presented at the Proceedings of the Workshop on Big Data Analytics and 
Machine Learning for Data Communication Networks. 
Wang, X., Wang, Y., & Tang, L. (2021). Strip hardness prediction in continuous annealing using 
multiobjective sparse nonlinear ensemble learning with evolutionary feature selection. IEEE 
Transactions on Automation Science and Engineering, 19(3), 2397-2411.  
Wang, X., Wang, Y., Tang, L., & Zhang, Q. (2023). Multi-Objective Ensemble Learning with Multi-
Scale Data for Product Quality Prediction in Iron and Steel Industry. IEEE Transactions on 
Evolutionary Computation.  
Weiner, D. J., Wigdor, E. M., Ripke, S., Walters, R. K., Kosmicki, J. A., Grove, J., . . . Bybjerg-
Grauholm, J. (2017). Polygenic transmission disequilibrium confirms that common and rare 
variation act additively to create risk for autism spectrum disorders. Nature Genetics, 49(7), 
978-985.  
Wu, T., Zhang, W., Jiao, X., Guo, W., & Hamoud, Y. A. (2021). Evaluation of stacking and blending 
ensemble learning methods for estimating daily reference evapotranspiration. Computers and 
Electronics in Agriculture, 184, 106039.  
Yang, Y. (2016). Temporal data mining via unsupervised ensemble learning: Elsevier. 
Yao, J., Zhang, X., Luo, W., Liu, C., & Ren, L. (2022). Applications of Stacking/Blending ensemble 
learning approaches for evaluating flash flood susceptibility. International Journal of Applied 
Earth Observation and Geoinformation, 112, 102932.  
Yue, X., Kontar, R. A., & Gómez, A. M. E. (2024). Federated data analytics: A study on linear models. 
IISE Transactions, 56(1), 16-28.  
Zhang, C., & Shen, Y. (2017). A cell type-specific expression signature predicts haploinsufficient 
autism-susceptibility genes. Human Mutation, 38(2), 204-215.  

38 
 
Zhang, J., Li, Z., Nai, K., Gu, Y., & Sallam, A. (2019). DELR: A double-level ensemble learning 
method for unsupervised anomaly detection. Knowledge-Based Systems, 181, 104783.  
Zhang, Y., Chen, X., Zhou, D., & Jordan, M. I. (2014). Spectral methods meet EM: A provably optimal 
algorithm for crowdsourcing. Advances in Neural Information Processing Systems, 27.  
 
