# Reproducing "Knowledge Distillation Meets Self-Supervision": How Different Pretext Tasks Shape Student Learning

**Authors:** Yanzhe Xie, Chenyu Zhang, Shanghong Lin

**Date:** June 2026

**Repository:** [github.com/FN8211/SSKD](https://github.com/FN8211/SSKD)

**Paper:** Xu et al., "Knowledge Distillation Meets Self-Supervision," ECCV 2020. ([arXiv:2006.07114](https://arxiv.org/abs/2006.07114))

**Summary:** [TODO]

---

## 2. Introduction & Background

### 2.1 Knowledge Distillation

Knowledge distillation (KD) compresses a large, well-trained *teacher* network into a smaller *student* network by training the student to mimic the teacher's output distribution rather than only learning from hard one-hot labels (Hinton et al., 2015). The intuition is that the teacher's output probabilities over all classes — called *soft targets* — encode inter-class similarities that hard labels cannot express. For example, a teacher trained on CIFAR-100 might assign a small but nonzero probability to "bus" when the true label is "truck"; this relative probability carries structural information about semantic similarity that benefits the student.

To produce informative soft targets, a temperature parameter $\tau$ is applied to the logits before the softmax:

$$p^i(x;\tau) = \frac{\exp(s_i(x)/\tau)}{\sum_k \exp(s_k(x)/\tau)}$$

where $x$ is the input, $s_i(x)$ is the logit for class $i$, and $\tau$ controls how "soft" the distribution is. At $\tau = 1$ this is standard softmax; higher $\tau$ produces a smoother distribution that reveals more inter-class structure. The KD loss is the KL divergence between teacher and student soft targets, scaled by $\tau^2$:

$$\mathcal{L}_{kd} = -\tau^2 \sum_{x \sim \mathcal{D}_x} \sum_{i=1}^{C} p_t^i(x;\tau) \log p_s^i(x;\tau)$$

where $t$ and $s$ denote teacher and student, $C$ is the total number of classes, and $\mathcal{D}_x$ is the training set. The student's total loss combines the standard cross-entropy loss $\mathcal{L}_{ce}$ on hard labels with the KD loss: $\mathcal{L} = \lambda_1 \mathcal{L}_{ce} + \lambda_2 \mathcal{L}_{kd}$.

### 2.2 What SSKD Adds: Self-Supervision as an Auxiliary Distillation Channel

Conventional KD transfers knowledge through a single channel: the teacher's class predictions on normal training data. Xu et al. (ECCV 2020) argue that this single task captures only one facet of the knowledge embedded in a large teacher network. Their framework, *Self-Supervised Knowledge Distillation* (SSKD), introduces a second, complementary channel by appending a self-supervised (SS) pretext task to both teacher and student. The teacher's predictions on this auxiliary task — even when imperfect — encode additional structured knowledge about the composition of semantic and geometric information in the input, which is not captured by classification logits alone.

**Contrastive prediction as the main pretext task.** As illustrated in Figure 1, SSKD uses contrastive learning, inspired by SimCLR (Chen et al., 2020), as its primary SS task. Given a mini-batch of $N$ images $\{x_i\}_{i=1:N}$, each image is independently transformed by a function $t(\cdot)$ sampled from a pool of four transformations (color dropping, rotation by $\pm90°$ or $180°$, random cropping with resize, and color jitter) to produce $\{\tilde{x}_i\}_{i=1:N}$. Both $x_i$ and $\tilde{x}_i$ are fed through the network backbone $f(\cdot)$ to extract representations, which are then projected by a 2-layer MLP into a latent space where cosine similarities are computed. The pair $(\tilde{x}_i, x_i)$ is treated as a positive pair; all $(\tilde{x}_i, x_k)$ with $k \neq i$ are negative pairs. 

The contrastive loss encourages the network to identify the correct positive pair:

$$\mathcal{L}_{contrast} = -\sum_i \log \frac{\exp(\text{cosine}(\tilde{z}_i, z_i)/\tau_{ss})}{\sum_k \exp(\text{cosine}(\tilde{z}_i, z_k)/\tau_{ss})}$$

where $z_i = \text{MLP}(f(x_i))$ and $\tilde{z}_i = \text{MLP}(f(\tilde{x}_i))$ are the projected representations, and $\tau_{ss}$ is a temperature parameter separate from the KD temperature.

**The SSKD training pipeline** proceeds in three stages. First, the teacher backbone $f_t(\cdot)$ and classifier $p_t(\cdot)$ are trained on the classification task using standard cross-entropy. Second, with the teacher backbone frozen, a lightweight SS module $c_t(\cdot, \cdot)$ (the MLP projection head and similarity computation) is fine-tuned on the contrastive task. Third, the student is trained to simultaneously mimic the teacher on four objectives:

1. $\mathcal{L}_{ce}$: cross-entropy on hard labels (using normal data),
2. $\mathcal{L}_{kd}$: KL divergence on soft classification targets (using normal data),
3. $\mathcal{L}_T$: KL divergence on the teacher's classification output for *transformed* data $\tilde{x}$, encouraging the student to match the teacher's behavior on augmented inputs,
4. $\mathcal{L}_{ss}$: KL divergence between the teacher's and student's contrastive similarity matrices, transferring the SS module's structured knowledge.

The student's final loss combines all four terms:

$$\mathcal{L} = \lambda_1 \mathcal{L}_{ce} + \lambda_2 \mathcal{L}_{kd} + \lambda_3 \mathcal{L}_{ss} + \lambda_4 \mathcal{L}_T$$

![The SSKD training scheme: the teacher is first trained on classification, then its self-supervised module is fine-tuned on the contrastive task with the backbone frozen, and finally the student is distilled on all four objectives ($\mathcal{L}_{ce}$, $\mathcal{L}_{kd}$, $\mathcal{L}_T$, $\mathcal{L}_{ss}$) from both normal and transformed inputs.](Fig2_SSKD_train_scheme.png)

***Figure 1.** The three-stage SSKD training scheme (Figure 2 in Xu et al., ECCV 2020). Normal images $x$ and their transformed versions $\tilde{x}$ are passed through the backbone and the SS projection module; the student is trained to mimic the teacher's classification logits and contrastive similarity matrix on both branches.*

**Selective transfer.** The teacher's contrastive predictions are sometimes severely wrong (e.g., matching a transformed image to the wrong original). Xu et al. observe that extremely incorrect predictions can mislead the student. To handle this, SSKD ranks transformed samples by the teacher's prediction error level and only transfers the correct predictions plus the top-$k$% least-wrong incorrect predictions. The authors find that $k = 75$ gives the best trade-off across architectures.

### 2.3 Four Self-Supervised Methods

While contrastive prediction is SSKD's primary pretext task, the paper evaluates three additional SS methods to test whether the quality of the SS method influences the student's final accuracy. The four methods, ordered by their representation quality (measured by linear evaluation accuracy on ImageNet with ResNet-50, sourced from prior work), are:

- **Exemplar** (Dosovitskiy et al., 2014): treats each training instance as its own class and applies heavy transformations; the SS module is a classifier with as many classes as training samples (31.5% ImageNet linear eval).
- **Jigsaw** (Noroozi & Favaro, 2016): splits each image into a 2×2 grid of non-overlapping patches, shuffles them, and trains a 24-way classifier (4! permutations) to recognize the permutation (45.7%).
- **Rotation** (Gidaris et al., 2018): rotates images by 0°, ±90°, or 180° and trains a 4-way classifier to predict the rotation angle (48.9%).
- **Contrastive** (Chen et al., 2020 / SimCLR): the contrastive prediction task described above (69.3%).

For Rotation, Jigsaw, and Exemplar, knowledge is transferred to the student via the logits of the respective classification heads, replacing the cosine-similarity matching used in the contrastive variant.

### 2.4 The Paper's Central Claim

Table 2 of the SSKD paper reports student accuracies on CIFAR-100 (vgg13→vgg8) for each of the four SS methods. The results show a monotonic ranking — Exemplar (74.57%) < Jigsaw (74.85%) < Rotation (75.01%) < Contrastive (75.48%) — that mirrors the ImageNet linear evaluation ranking of the SS methods themselves. The paper concludes that student accuracy is *positively correlated* with the quality of the self-supervision method used.

### 2.5 Why Reproduce This Paper?

Three reasons motivate an independent reproduction. First, SSKD introduces a framework-level idea — combining self-supervised pretext tasks with knowledge distillation — whose generality rests on the ablation across four SS methods. If the correlation between SS quality and student accuracy does not hold, the framework's value as a principled approach (rather than a method-specific trick) is weakened.

Second, the original codebase ([xuguodong03/SSKD](https://github.com/xuguodong03/SSKD)) implements only the contrastive pretext task. The other three methods evaluated in Table 2 — Rotation, Jigsaw, and Exemplar — are described textually in the paper's Appendix (§6.3) but have no public code. This means Table 2's full claim is not independently verifiable without new implementation work, making reproduction particularly valuable.

Third, the claim that SS quality predicts student accuracy is tested on only one teacher-student pair (vgg13→vgg8) in Table 2. Whether this correlation generalizes across architectures is an open question that bears directly on the paper's "model-agnostic" framing.

---

## 3. Reproduction Scope & Plan

### 3.1 Primary Target

Our primary target is Table 2 of the SSKD paper ("Influence of Different Self-Supervision Tasks"), which evaluates four SS methods — Exemplar, Jigsaw, Rotation, and Contrastive — on the vgg13→vgg8 architecture pair on CIFAR-100. The table's central finding is that student accuracy increases monotonically with the quality of the SS method.

### 3.2 Extensions Beyond Table 2

We extend the reproduction in three directions:

**(a) Loss component analysis.** We evaluate two modifications to the standard KD loss $\mathcal{L}_{kd}$: Decoupled Knowledge Distillation (DKD), which separates target-class and non-target-class distillation signals, and Weighted Soft Labels Distillation (WSLD), which applies per-sample adaptive weights to prioritize informative training examples.

**(b) New dataset.** We train the vgg13→vgg8 pair on Tiny ImageNet (200 classes, 64×64 images) to test whether SSKD's advantage over standard KD transfers to a larger, more complex dataset. This evaluates the generality of the framework beyond the CIFAR-100 setting used in all of the paper's main experiments.

**(c) New architecture pair.** We run all four SS methods on resnet56→resnet20 on CIFAR-100 to test whether the SS-quality → student-accuracy correlation holds across architectures. The paper claims SSKD is "model-agnostic" because it transfers only output-level signals rather than architecture-specific intermediate features. If the correlation breaks down for a different architecture pair, this claim requires qualification.

### 3.3 Codebase Status

The original author repository ([xuguodong03/SSKD](https://github.com/xuguodong03/SSKD)) provides code for the contrastive pretext task only. Rotation, Jigsaw, and Exemplar are described in the paper's Appendix §6.3 but have no corresponding public implementation. Reproducing Table 2 in full — and running the resnet pair extension — therefore required us to implement these three SS heads ourselves, following the paper's textual descriptions:

- **Rotation:** a 4-way classification head predicting one of {0°, +90°, −90°, 180°}.
- **Jigsaw:** a 24-way classification head predicting the permutation of a 2×2 grid of image patches (4! = 24 possible permutations).
- **Exemplar:** an instance-classification head with one class per training sample (50,000 classes for CIFAR-100).

For Rotation, Jigsaw, and Exemplar, knowledge is transferred via the logits of these classification heads. The selective transfer strategy (top-$k$% filtering) used in the contrastive variant is not applied to these methods, as they use classification-based matching rather than similarity ranking.

### 3.4 Reproducibility Criteria

Our reproduction falls under "Reproduced" rather than "Replicated" in the course terminology: we evaluate existing author code (for the contrastive method) and supplement it with new implementations where needed, rather than re-implementing the entire framework from scratch. Each member is responsible for at least one distinct criterion, as summarized below:

| Experiment | Criterion | Owner |
|---|---|---|
| Reproduce Table 2 (four SS methods, vgg13→vgg8, CIFAR-100) | Reproduced | Yanzhe |
| KD loss modifications (DKD, WSLD) | New algorithm variant | Yanzhe |
| Train vgg13→vgg8 on Tiny ImageNet | New data | Chenyu |
| Run four SS methods on resnet56→resnet20, CIFAR-100 | New algorithm variant | Shanghong |

### 3.5 Training Setup

All experiments follow the hyperparameters from §6.4 of the SSKD paper unless otherwise noted. The temperatures are $\tau_{kd} = \tau_T = 4$ and $\tau_{ss} = 0.5$. The loss weights are $\lambda_1 = 0.1$, $\lambda_2 = 0.9$, $\lambda_3 = 2.7$, $\lambda_4 = 10.0$. All models are trained for 240 epochs with an initial learning rate of 0.05, decayed by a factor of 10 at epochs 150, 180, and 210. We use SGD with momentum 0.9 and weight decay $5 \times 10^{-4}$, with a batch size of 64. The original paper reports experiments on a TITAN-X-Pascal GPU; our experiments are run on consumer-grade GPUs (RTX 3060 / RTX 3090), which may introduce minor numerical differences due to floating-point nondeterminism across hardware.

---

### 4. Experimental Results

This is the heart of the blog (Content = 50%). Structure each experiment as a self-contained module with: **(1) Question → (2) Setup → (3) Results table/figure → (4) Analysis.**

---

### 4.1 Experiment 1: Table 2 Reproduction — SS Methods Comparison (vgg13 → vgg8, CIFAR-100)

**Question:**

Table 2 of the SSKD paper claims that student accuracy is positively correlated with the quality of the self-supervision (SS) method used, measured by linear evaluation accuracy on ImageNet, with the ranking Exemplar < Jigsaw < Rotation < Contrastive. Our question is: after re-running the original code and implementing the three missing SS methods, does this ranking still hold?

**Setup:**

The architecture pair is vgg13 (teacher) → vgg8 (student) on CIFAR-100. All four methods share the same hyperparameters from the paper: τ_kd = τ_T = 4, τ_ss = 0.5, λ1 = 0.1, λ2 = 0.9, λ3 = 2.7, λ4 = 10.0; 240 training epochs with an initial learning rate of 0.05, decayed by a factor of 10 at epochs 150, 180, and 210; batch size 64, SGD with momentum 0.9 and weight decay 5e-4. We use an RTX 3060 (6GB), whereas the original paper used a TITAN-X-Pascal.

For Contrastive, we used the implementation from the original authors' repository (`xuguodong03/SSKD`). Rotation, Jigsaw, and Exemplar were not provided in the original codebase and were implemented by us based on the descriptions in section 6.3. Rotation is a 4-way classification head over rotation angles (0°, ±90°, 180°); Jigsaw is a 24-way classification head over permutations of 2×2 image patches; Exemplar is an instance-classification head with a number of classes equal to the dataset size. In all three cases, knowledge is transferred to the student via the logits of these heads.

As reference points, we also trained the teacher (vgg13) and a vanilla student (vgg8) independently, using standard cross-entropy classification with no distillation.

**Results:**

**Main table (corresponding to Table 2 of the paper)**

| SS Method | SS Performance (ImageNet linear eval, paper) | Student Acc. (Paper) | Student Acc. (Reproduced) |
|---|---|---|---|
| Exemplar | 31.5 | 74.57 | 74.46 |
| Jigsaw | 45.7 | 74.85 | 74.34 |
| Rotation | 48.9 | 75.01 | 74.50 |
| Contrastive | 69.3 | 75.48 | 74.53 |

**Reference baselines**

| | Teacher Acc. | Vanilla Student Acc. |
|---|---|---|
| Paper | 75.38 | 70.68 |
| Reproduced | 74.49 | 70.73 |

**Analysis:**

Overall, Rotation and Contrastive still outperform Exemplar and Jigsaw, and Contrastive achieves the highest accuracy among the four (74.53). This finding broadly consistent with the paper's general direction that higher SS quality leads to higher student accuracy. All four SSKD variants also improve over the vanilla student, which scores 70.73, confirming that distillation with SS signals is beneficial regardless of the SS method used.

However, a local reversal occurs in the ranking: Exemplar (74.57) < Jigsaw (74.85) in the paper, but Exemplar (74.46) > Jigsaw (74.34) in our reproduction. The gap between the two is small in both cases, 0.28 points in the paper and 0.12 points in our reproduction, so this reversal likely falls within the variance of a single training run. It may also reflect implementation differences, since Jigsaw and Exemplar were both newly implemented by us based on textual descriptions rather than the original authors' code.

Contrastive directly reuses the original authors' code, so its result of 74.53 serves as a reference for how trustworthy our overall pipeline is. Compared with the paper's 75.48, this is a gap of about 0.95 points, similar in magnitude to the other three methods, which fall roughly 0.3 to 1.0 points below the paper. This suggests the overall downward shift may stem from pipeline-level factors such as random seed, single-run versus averaged results, or PyTorch version differences, rather than being specific to the newly implemented SS methods.

The reproduced Teacher Acc. of 74.49 is close to the paper's 75.38, a gap of about 0.89 points within a reasonable range, suggesting the teacher training pipeline itself is not a major source of discrepancy.

---

### 4.2 Loss Component Modifications

**Method:**

The SSKD framework achieves strong distillation performance, but its standard KD loss $L_{kd}$ has limitations in how it handles class-level and sample-level knowledge transfer. We therefore propose two complementary modifications, DKD and WSLD, to address these limitations.

##### [Decoupled Knowledge Distillation (DKD)](https://arxiv.org/abs/2203.08679)

For a training sample with ground-truth label $y$, the *target class* refers to class $y$ itself, while *non-target classes* refer to all other $C-1$ classes in the output space.

Standard KD couples the target class and non-target class distributions into a single KL divergence term, which may limit the flexibility of knowledge transfer. DKD decouples $L_{kd}$ into two components. Target Class Knowledge Distillation (TCKD) aligns the binary distribution between the target class and all non-target classes:

$$L_{TCKD} = \text{KL}\left(\left[p_t^y,\ 1-p_t^y\right] \,\|\, \left[p_s^y,\ 1-p_s^y\right]\right) := \sum_{k\in\{y,\neg y\}} b_t^k \log\frac{b_t^k}{b_s^k}$$

$$= p_t^y\log\frac{p_t^y}{p_s^y} + (1-p_t^y)\log\frac{1-p_t^y}{1-p_s^y}$$

where $p_t^y$ and $p_s^y$ are the teacher's and student's softmax probabilities on the target class $y$ at temperature $\tau$, where $\tau$ denotes the standard distillation temperature, identical in role to the temperature used in vanilla KD (Hinton et al., 2015), controlling the softness of the output distributions for both teacher and student:

$$p_t^y = \frac{\exp(z_t^y/\tau)}{\sum_{k=1}^{C}\exp(z_t^k/\tau)}, \qquad p_s^y = \frac{\exp(z_s^y/\tau)}{\sum_{k=1}^{C}\exp(z_s^k/\tau)}$$


Here $[p^y, 1-p^y]$ collapses the original $C$-way distribution into a binary one over {target class, all other classes}, capturing only whether the model assigns sufficient probability to the correct class, regardless of how probability is distributed among the remaining $C-1$ classes.

Non-target Class Knowledge Distillation (NCKD) aligns the distribution over non-target classes only:

$$L_{NCKD} = \text{KL}\left(\hat{p}_t^{\neg y} \,\|\, \hat{p}_s^{\neg y}\right)$$

$$\hat{p}_t^{\neg y,k} = \frac{p_t^k}{1-p_t^y}, \qquad \hat{p}_s^{\neg y,k} = \frac{p_s^k}{1-p_s^y} \qquad \text{for } k \neq y$$

where $\hat{p}^{\neg y}$ denotes the re-normalized distribution over non-target classes. The combined DKD loss replaces $L_{kd}$:

$$L_{kd}^{DKD} = \tau^2 \left(\alpha \cdot L_{TCKD} + \beta \cdot L_{NCKD}\right)$$

with $\alpha = 1.0$ and $\beta = 8.0$ in our experiments.

##### [Weighted Soft Labels Distillation (WSLD)](https://arxiv.org/abs/2102.00650)

Standard KD treats all samples equally, but samples that are already easy for the student carry less informative gradient signal. WSLD assigns a per-sample adaptive weight to the KD loss based on the ratio of student and teacher cross-entropy losses at temperature $\tau = 1$:

$$w_i = 1 - \exp\left(-\frac{L_{ce}^s(x_i)}{L_{ce}^t(x_i)}\right)$$

A higher weight is assigned when the student's loss is large relative to the teacher's. The weighted KD loss replaces $L_{kd}$:

$$L_{kd}^{WSLD} = \tau^2 \cdot \frac{1}{N}\sum_{i=1}^{N} w_i \cdot \text{KL}\left(p_s(x_i;\tau) \,\|\, p_t(x_i;\tau)\right)$$

**Setup:**

All three modifications are evaluated on vgg13→vgg8, CIFAR-100, using the same hyperparameters and training protocol as Section 4.1. The original SSKD with Contrastive is used as the baseline in all comparisons.

**Results:**

| Method | Student Acc. (%) |
|---|---|
| SSKD (Contrastive, baseline) | 74.53 |
| SSKD + DKD | 74.68 |
| SSKD + WSLD | 74.61 |

**Analysis:**

DKD and WSLD both outperform the baseline, achieving accuracies of 74.68% and 74.61%, respectively. This result shows that both class-level distillation and sample-level re-weighting improve performance. However, the gains are relatively small, suggesting that the SS auxiliary signal already recovers much of the information that standard KD fails to transfer.

DKD performs slightly better than WSLD, with a margin of 0.07 percentage points. This indicates that class-level decoupling is more effective than sample-level re-weighting in our setting. A possible reason is that CIFAR-100 contains many fine-grained classes. By separating target-class and non-target-class distillation signals, DKD can better exploit the teacher's dark knowledge.

Since DKD and WSLD address different aspects of knowledge distillation, they may provide complementary benefits. Exploring a combination of the two methods is an interesting direction for future work.

---

### 4.3 Experiment 3: New Dataset — vgg13 → vgg8 on Tiny ImageNet

*Owner: Chenyu · Criterion: New data*

**Question:** Does SSKD's advantage over standard KD transfer to a larger, more complex dataset (Tiny ImageNet) with the same vgg13→vgg8 architecture pair?

**Setup:** Train the vgg13→vgg8 pair on Tiny ImageNet (200 classes, 64×64 images). Describe any necessary adaptations from the CIFAR-100 setup: image resolution handling (resize/crop strategy), data augmentation adjustments, training schedule changes, and how the SS transformation pool was adapted for the different image size. State whether the teacher was trained from scratch on Tiny ImageNet or used pretrained weights.

**Results:** Table comparing vanilla student, standard KD, and SSKD (contrastive) on Tiny ImageNet. If multiple SS methods were run, include all. Also report teacher accuracy on Tiny ImageNet as a reference.

**Analysis:**

- Does SSKD still improve over standard KD on Tiny ImageNet? By how much, compared to the CIFAR-100 improvement margin?
- If accuracy is lower than on CIFAR-100 (expected for a harder dataset), is the *relative* improvement from SSKD preserved?
- Discuss any dataset-specific challenges (e.g., more classes → harder contrastive task, different image statistics → transformation pool may be less effective).

---

### 4.4 Experiment 4: New Architecture Variant — resnet56 → resnet20 on CIFAR-100

*Owner: Shanghong · Criterion: New algorithm variant*

**Question:** Does the SS-quality → student-accuracy correlation (Table 2's central claim) hold on a different teacher-student architecture pair (resnet56→resnet20)?

**Setup:** Run all four SS methods (Contrastive, Rotation, Jigsaw, Exemplar) on resnet56→resnet20, CIFAR-100, using the same hyperparameters as the vgg pair experiments (or note any necessary adjustments). For each SS method, describe how the SS head attaches to the resnet backbone.

**Results:**

- Table mirroring Table 2's format but for resnet56→resnet20: SS method, SS performance (from the paper), student accuracy (yours). Include teacher accuracy and vanilla student accuracy.
- Side-by-side comparison with the vgg13→vgg8 results from Experiment 1 (either in the same table or a paired figure).

**Analysis:**

- Does the ranking Exemplar < Jigsaw < Rotation < Contrastive hold for the resnet pair?
- The paper claims SSKD is "model-agnostic." Does your evidence support or challenge this claim?
- If the ranking differs from the vgg pair, discuss architectural reasons (e.g., residual connections may interact differently with SS tasks, different feature map sizes).

---

### 5. Discussion & Conclusion

**Purpose:** Step back from the numbers. Synthesize, reflect, and address limitations.

**Content to cover:**

- **Do our results uphold the paper's main conclusions?** (Required by the Exposition rubric.) The main claim is that SS method quality positively correlates with student accuracy, and that SSKD outperforms prior KD methods. State your verdict explicitly: "Our reproduction [supports / partially supports / does not support] the paper's central claim, because [...]." Draw on evidence from both the vgg pair (Exp 1) and the resnet pair (Exp 4) to assess the robustness of this conclusion.
- **Generalization to Tiny ImageNet.** Does SSKD's advantage hold on a new dataset? If the improvement margin shrinks or vanishes, what does this imply about the scope of the paper's claims?
- **Loss component contributions.** Do L_T and L_ss each contribute meaningfully, or does one dominate? What does this tell us about the SSKD framework's design?
- **Codebase completeness as a reproducibility obstacle.** The original codebase only implements contrastive. This means Table 2 as published is not independently verifiable without new implementation work. Discuss what this means for the paper's reproducibility.
- **Limitations of our reproduction:**
    - Limited compute → could not run all teacher-student pairs from Table 3/4.
    - Single-run results vs. multi-seed variance (if applicable).
    - Any hardware/framework differences.
    - Tiny ImageNet adaptations (resolution, augmentation) may introduce confounders that make direct comparison to CIFAR-100 results imperfect.
- **What we learned about reproducibility.** Reflect on the process itself: verifying what a codebase actually implements before planning, the gap between paper descriptions and code, absolute vs. relative trend reproduction.

**Writing notes:** This is where the Exposition grade is won or lost. Be specific, not generic. Don't write "reproducibility is important" — write about what *this* reproduction taught you.

---

### 6. Author Contributions

**Content:** A brief table or paragraph listing what each member did. Required by the submission guidelines. Example:

> **Yanzhe** reproduced Table 2 (four SS methods on vgg13→vgg8, CIFAR-100), explored the effect of loss components ($L_T$, $L_{ss}$) on student accuracy (Criterion: Reproduced). He also proposed two complementary loss modifications (DKD, WSLD) to address limitations in the standard KD loss (Criterion: New algorithm variant).
> **Chenyu** trained the vgg13→vgg8 pair on Tiny ImageNet, adapting the pipeline for the new dataset (Criterion: New data). 
> **Shanghong** ran all four SS methods on the resnet56→resnet20 pair on CIFAR-100 (Criterion: New algorithm variant). All members contributed to writing the blog post.
> 

---

### 7. References

Standard academic references. Cite at minimum: SSKD (Xu et al., ECCV 2020), Hinton et al. (2015) for KD, SimCLR (Chen et al., 2020) for contrastive learning, and the original papers for each SS method (Exemplar: Dosovitskiy et al., Jigsaw: Noroozi & Favaro, Rotation: Gidaris et al.). Also cite CRD (Tian et al.) since the SSKD codebase is based on it.