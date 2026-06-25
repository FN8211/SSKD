# Reproducing "Knowledge Distillation Meets Self-Supervision": How Different Pretext Tasks Shape Student Learning

**Authors:** Yanzhe Xie, Chenyu Zhang, Shanghong Lin

**Date:** June 2026

**Repository:** [github.com/FN8211/SSKD](https://github.com/FN8211/SSKD)

**Paper:** Xu et al., "Knowledge Distillation Meets Self-Supervision," ECCV 2020 [2]. ([arXiv:2006.07114](https://arxiv.org/abs/2006.07114))

**Summary:** We reproduce and extend Xu et al.'s SSKD (ECCV 2020), which augments knowledge distillation with a self-supervised pretext task and claims that higher-quality self-supervision monotonically improves student accuracy (Exemplar < Jigsaw < Rotation < Contrastive). We re-implement the three undocumented SS methods (Rotation, Jigsaw, Exemplar) from the paper's appendix, fix a learning-rate-scheduler bug in the original codebase, and extend evaluation to Tiny ImageNet and a resnet56→resnet20 architecture pair. Our results confirm that all four SS methods consistently outperform vanilla training across settings, but the monotonic quality–accuracy correlation breaks down: Rotation outperforms Contrastive on both the ResNet pair and Tiny ImageNet, revealing that the optimal SS method is architecture- and dataset-dependent. We additionally show that replacing the standard KD loss with DKD or WSLD yields modest but consistent accuracy gains over the SSKD baseline.

---

## 2. Introduction & Background

### 2.1 Knowledge Distillation

Knowledge distillation (KD) compresses a large, well-trained *teacher* network into a smaller *student* network by training the student to mimic the teacher's output distribution rather than only learning from hard one-hot labels (Hinton et al., 2015) [1]. The intuition is that the teacher's output probabilities over all classes — called *soft targets* — encode inter-class similarities that hard labels cannot express. For example, a teacher trained on CIFAR-100 might assign a small but nonzero probability to "bus" when the true label is "truck"; this relative probability carries structural information about semantic similarity that benefits the student.

To produce informative soft targets, a temperature parameter $\tau$ is applied to the logits before the softmax:

$$p^i(x;\tau) = \frac{\exp(s_i(x)/\tau)}{\sum_k \exp(s_k(x)/\tau)}$$

where $x$ is the input, $s_i(x)$ is the logit for class $i$, and $\tau$ controls how "soft" the distribution is. At $\tau = 1$ this is standard softmax; higher $\tau$ produces a smoother distribution that reveals more inter-class structure. The KD loss is the KL divergence between teacher and student soft targets, scaled by $\tau^2$:

$$\mathcal{L}_{kd} = -\tau^2 \sum_{x \sim \mathcal{D}_x} \sum_{i=1}^{C} p_t^i(x;\tau) \log p_s^i(x;\tau)$$

where $t$ and $s$ denote teacher and student, $C$ is the total number of classes, and $\mathcal{D}_x$ is the training set. The student's total loss combines the standard cross-entropy loss $\mathcal{L}_{ce}$ on hard labels with the KD loss: $\mathcal{L} = \lambda_1 \mathcal{L}_{ce} + \lambda_2 \mathcal{L}_{kd}$.

### 2.2 What SSKD Adds: Self-Supervision as an Auxiliary Distillation Channel

Conventional KD transfers knowledge through a single channel: the teacher's class predictions on normal training data. Xu et al. (ECCV 2020) [2] argue that this single task captures only one facet of the knowledge embedded in a large teacher network. Their framework, *Self-Supervised Knowledge Distillation* (SSKD), introduces a second, complementary channel by appending a self-supervised (SS) pretext task to both teacher and student. The teacher's predictions on this auxiliary task — even when imperfect — encode additional structured knowledge about the composition of semantic and geometric information in the input, which is not captured by classification logits alone.

**Contrastive prediction as the main pretext task.** As illustrated in Figure 1, SSKD uses contrastive learning, inspired by SimCLR (Chen et al., 2020) [3], as its primary SS task. Given a mini-batch of $N$ images $\{x_i\}_{i=1:N}$, each image is independently transformed by a function $t(\cdot)$ sampled from a pool of four transformations (color dropping, rotation by $\pm90°$ or $180°$, random cropping with resize, and color jitter) to produce $\{\tilde{x}_i\}_{i=1:N}$. Both $x_i$ and $\tilde{x}_i$ are fed through the network backbone $f(\cdot)$ to extract representations, which are then projected by a 2-layer MLP into a latent space where cosine similarities are computed. The pair $(\tilde{x}_i, x_i)$ is treated as a positive pair; all $(\tilde{x}_i, x_k)$ with $k \neq i$ are negative pairs. 

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

***Figure 1.** The three-stage SSKD training scheme (Figure 2 in Xu et al., ECCV 2020 [2]). Normal images $x$ and their transformed versions $\tilde{x}$ are passed through the backbone and the SS projection module; the student is trained to mimic the teacher's classification logits and contrastive similarity matrix on both branches.*

**Selective transfer.** The teacher's contrastive predictions are sometimes severely wrong (e.g., matching a transformed image to the wrong original). Xu et al. [2] observe that extremely incorrect predictions can mislead the student. To handle this, SSKD ranks transformed samples by the teacher's prediction error level and only transfers the correct predictions plus the top-$k$% least-wrong incorrect predictions. The authors find that $k = 75$ gives the best trade-off across architectures.

### 2.3 Four Self-Supervised Methods

While contrastive prediction is SSKD's primary pretext task, the paper evaluates three additional SS methods to test whether the quality of the SS method influences the student's final accuracy. The four methods, ordered by their representation quality (measured by linear evaluation accuracy on ImageNet with ResNet-50, sourced from prior work), are:

- **Exemplar** (Dosovitskiy et al., 2014) [4]: treats each training instance as its own class and applies heavy transformations; the SS module is a classifier with as many classes as training samples (31.5% ImageNet linear eval).
- **Jigsaw** (Noroozi & Favaro, 2016) [5]: splits each image into a 2×2 grid of non-overlapping patches, shuffles them, and trains a 24-way classifier (4! permutations) to recognize the permutation (45.7%).
- **Rotation** (Gidaris et al., 2018) [6]: rotates images by 0°, ±90°, or 180° and trains a 4-way classifier to predict the rotation angle (48.9%).
- **Contrastive** (Chen et al., 2020 / SimCLR) [3]: the contrastive prediction task described above (69.3%).

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

The SSKD repository's network architecture definitions (VGG, ResNet, WRN, ShuffleNet, and MobileNet) are borrowed from the CRD/RepDistiller repository [10], as acknowledged in the SSKD README. The training scripts and the SSKD-specific components (contrastive projection head, selective transfer, SS training pipeline) are original to the SSKD authors.

**Bug in the original teacher training code.** During setup, we discovered that the teacher training script (`teacher.py`) in the original codebase defines a `MultiStepLR` learning rate scheduler but never calls `scheduler.step()`, causing the learning rate to remain fixed at its initial value of 0.05 throughout all 240 training epochs. Without learning rate decay, the teacher fails to converge properly and reaches only ~60% validation accuracy — well below the paper's reported 75.38% and insufficient to provide meaningful soft targets for the student. We fixed this by adding the missing `scheduler.step()` call at the end of each training epoch. After the fix, the teacher achieves 74.49% validation accuracy, consistent with the paper's reported value. All teacher checkpoints used in our experiments are trained with this fix applied.

### 3.4 Reproducibility Criteria

Our reproduction falls under "Reproduced" rather than "Replicated" in the course terminology: we evaluate existing author code (for the contrastive method) and supplement it with new implementations where needed, rather than re-implementing the entire framework from scratch. Each member is responsible for at least one distinct criterion, as summarized below:

| Experiment | Criterion | Owner |
|---|---|---|
| Reproduce Table 2 (four SS methods, vgg13→vgg8, CIFAR-100) | Reproduced | Yanzhe |
| KD loss modifications (DKD, WSLD) | New algorithm variant | Yanzhe |
| Train vgg13→vgg8 on Tiny ImageNet | New data | Chenyu |
| Run four SS methods on resnet56→resnet20, CIFAR-100 | New algorithm variant | Shanghong |

### 3.5 Training Setup

All experiments follow the hyperparameters from §6.4 of the SSKD paper unless otherwise noted. The temperatures are $\tau_{kd} = \tau_T = 4$ and $\tau_{ss} = 0.5$. The loss weights are $\lambda_1 = 0.1$, $\lambda_2 = 0.9$, $\lambda_3 = 2.7$, $\lambda_4 = 10.0$. **Note:** the paper's Eq. 8 assigns $\lambda_3 = 2.7$ to $\mathcal{L}_{ss}$ and $\lambda_4 = 10.0$ to $\mathcal{L}_T$, giving the larger weight to the transformed-data KD term. However, the released codebase uses `ss-weight=10.0` and `tf-weight=2.7`, effectively swapping the two — giving the larger weight to $\mathcal{L}_{ss}$ instead. All our experiments follow the codebase values, since those determine what was actually executed. All models are trained for 240 epochs with an initial learning rate of 0.05, decayed by a factor of 10 at epochs 150, 180, and 210. We use SGD with momentum 0.9 and weight decay $5 \times 10^{-4}$, with a batch size of 64. The original paper reports experiments on a TITAN-X-Pascal GPU; our experiments are run on consumer-grade GPUs (RTX 3060 / RTX 3090), which may introduce minor numerical differences due to floating-point nondeterminism across hardware.

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

The reproduced Teacher Acc. of 74.49 is close to the paper's 75.38, a gap of about 0.89 points within a reasonable range, suggesting the teacher training pipeline itself is not a major source of discrepancy. Notably, this result was only achievable after fixing a bug in the original codebase (see Section 3.3): without the `scheduler.step()` fix, the teacher stalls at ~60%, making any meaningful comparison to the paper impossible.

---

### 4.2 Loss Component Modifications

**Method:**

The SSKD framework achieves strong distillation performance, but its standard KD loss $L_{kd}$ has limitations in how it handles class-level and sample-level knowledge transfer. We therefore propose two complementary modifications, DKD and WSLD, to address these limitations.

##### [Decoupled Knowledge Distillation (DKD)](https://arxiv.org/abs/2203.08679) [7]

For a training sample with ground-truth label $y$, the *target class* refers to class $y$ itself, while *non-target classes* refer to all other $C-1$ classes in the output space.

Standard KD couples the target class and non-target class distributions into a single KL divergence term, which may limit the flexibility of knowledge transfer. DKD decouples $L_{kd}$ into two components. Target Class Knowledge Distillation (TCKD) aligns the binary distribution between the target class and all non-target classes:

$$L_{TCKD} = \text{KL}\left(\left[p_t^y,\ 1-p_t^y\right] \,\|\, \left[p_s^y,\ 1-p_s^y\right]\right) := \sum_{k\in\{y,\neg y\}} b_t^k \log\frac{b_t^k}{b_s^k}$$

$$= p_t^y\log\frac{p_t^y}{p_s^y} + (1-p_t^y)\log\frac{1-p_t^y}{1-p_s^y}$$

where $p_t^y$ and $p_s^y$ are the teacher's and student's softmax probabilities on the target class $y$ at temperature $\tau$, where $\tau$ denotes the standard distillation temperature, identical in role to the temperature used in vanilla KD (Hinton et al., 2015) [1], controlling the softness of the output distributions for both teacher and student:

$$p_t^y = \frac{\exp(z_t^y/\tau)}{\sum_{k=1}^{C}\exp(z_t^k/\tau)}, \qquad p_s^y = \frac{\exp(z_s^y/\tau)}{\sum_{k=1}^{C}\exp(z_s^k/\tau)}$$


Here $[p^y, 1-p^y]$ collapses the original $C$-way distribution into a binary one over {target class, all other classes}, capturing only whether the model assigns sufficient probability to the correct class, regardless of how probability is distributed among the remaining $C-1$ classes.

Non-target Class Knowledge Distillation (NCKD) aligns the distribution over non-target classes only:

$$L_{NCKD} = \text{KL}\left(\hat{p}_t^{\neg y} \,\|\, \hat{p}_s^{\neg y}\right)$$

$$\hat{p}_t^{\neg y,k} = \frac{p_t^k}{1-p_t^y}, \qquad \hat{p}_s^{\neg y,k} = \frac{p_s^k}{1-p_s^y} \qquad \text{for } k \neq y$$

where $\hat{p}^{\neg y}$ denotes the re-normalized distribution over non-target classes. The combined DKD loss replaces $L_{kd}$:

$$L_{kd}^{DKD} = \tau^2 \left(\alpha \cdot L_{TCKD} + \beta \cdot L_{NCKD}\right)$$

with $\alpha = 1.0$ and $\beta = 8.0$ in our experiments.

##### [Weighted Soft Labels Distillation (WSLD)](https://arxiv.org/abs/2102.00650) [8]

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

**Question:** SSKD's original evaluation is entirely on CIFAR-100 (32×32, 100 classes). Does SSKD's advantage over standard knowledge distillation persist when the same vgg13→vgg8 architecture pair is trained on Tiny ImageNet (64×64, 200 classes), a dataset with higher resolution and a larger label space?

#### Setup

We train the vgg13→vgg8 pair on Tiny ImageNet (200 classes, 64×64 images) using all four SS methods: Contrastive, Rotation, Jigsaw, and Exemplar. The teacher (vgg13) is trained from scratch on Tiny ImageNet. We do not use ImageNet-pretrained weights, ensuring a clean comparison with the CIFAR-100 setting.

**Adaptations from the CIFAR-100 setup.** Several changes were necessary to adapt the pipeline to Tiny ImageNet:

| Hyperparameter | CIFAR-100 | Tiny ImageNet |
|---|---|---|
| Image size | 32×32 | 64×64 |
| # Classes | 100 | 200 |
| Total epochs | 240 | 140 |
| LR milestones | 150, 180, 210 | 60, 90, 110 |
| RandomCrop padding | 4 | 8 |
| Teacher augmentation | RandomCrop + Flip | + RandAugment(2, 9) + RandomErasing(p=0.25) |

The training schedule is compressed to 140 epochs with earlier LR decay milestones, because earlier convergence is observed on new dataset. The teacher uses additional augmentation (RandAugment and RandomErasing) to mitigate overfitting on the larger image size. Loss weights and temperatures are kept identical to the CIFAR-100 setup: τ_kd = τ_T = 4.0, τ_ss = 0.5, λ_CE = 0.1, λ_KD = 0.9, λ_TF = 2.7, λ_SS = 10.0.

**SS transformation pool.** Each SS method uses the same pretext task formulation as in the CIFAR-100 experiments, with one size-aware adjustment:

- **Contrastive:** Four augmented views via 90° rotation (0°, 90°, 180°, 270°) and the similarity matching head is unchanged.
- **Rotation:** Four rotation views (0°, 90°, 180°, 270°) and the 4-way classification head is unchanged.
- **Jigsaw:** 2×2 patch grid on 64×64 images yields 32×32 patches, compared to 16×16 on CIFAR-100. The `make_jigsaw` function is resolution-agnostic, so no code change was needed; the larger patches likely make the task somewhat easier due to richer per-patch content.
- **Exemplar:** Instance-discrimination head with 100,000 classes (one per Tiny ImageNet training sample), compared to 50,000 for CIFAR-100.

#### Results

Table 1 reports the best top-1 validation accuracy (%) for each SS method on Tiny ImageNet.

**Table 1: Student accuracy (%) on Tiny ImageNet for vgg13→vgg8 with all four SS methods.**

| Method | SS Quality (ImageNet linear eval, %) | Student Acc (TIN, %) |
|---|---|---|
| Teacher (vgg13) | — | 64.08 |
| **SSKD — Rotation** | 48.9 | **63.41** |
| **SSKD — Contrastive** | 69.3 | 63.22 |
| **SSKD — Exemplar** | 31.5 | 62.96 |
| **SSKD — Jigsaw** | 45.7 | 62.73 |

For reference, we also include the CIFAR-100 results from Section 4.1 to compare improvement margins.

**Table 2: CIFAR-100 vs. Tiny ImageNet results for vgg13→vgg8.**

| Method | Student Acc (CIFAR-100, %) | Student Acc (TIN, %) |
|---|---|---|
| Teacher | 74.49 | 64.08 |
| Vanilla student | 70.73 | 59.41 |
| SSKD — Contrastive | 74.53 | 63.22 |
| SSKD — Rotation | 74.50 | 63.41 |
| SSKD — Exemplar | 74.46 | 62.96 |
| SSKD — Jigsaw | 74.34 | 62.73 |

The vanilla student row reflects an experiment we ran on Tiny ImageNet (top-1 validation accuracy: 59.41%). A standard KD-only baseline (without self-supervision) was not run on TIN; the improvement figures below therefore compare SSKD against vanilla training, not against KD alone.

#### Analysis

**All SSKD methods underperform the teacher on TIN.** On CIFAR-100, the SSKD student (Contrastive, 74.53%) nearly matches — or marginally surpasses — the teacher (74.49%), closing the teacher–student gap almost entirely. On Tiny ImageNet, the gap is wider: the best SSKD variant (Rotation, 63.41%) sits 0.67 percentage points below the teacher (64.08%), while Jigsaw trails by 1.35 points. This is consistent with the expectation that harder datasets make knowledge distillation more challenging: the larger label space (200 vs. 100 classes) and higher resolution (64×64 vs. 32×32) both increase the complexity of what the student must learn.

**The ranking changes on TIN: Rotation overtakes Contrastive.** On CIFAR-100, the ranking is Contrastive (74.53%) > Rotation (74.50%) > Exemplar (74.46%) > Jigsaw (74.34%), broadly consistent with the paper's claim that SS quality correlates positively with student accuracy. On Tiny ImageNet the ranking becomes:

> Rotation (63.41%) > Contrastive (63.22%) > Exemplar (62.96%) > Jigsaw (62.73%)

Rotation overtakes Contrastive by 0.19 percentage points, reversing the paper's top-two ordering. The bottom two methods (Exemplar, Jigsaw) maintain their relative positions, and Jigsaw remains the weakest despite its higher SS quality score (45.7) compared to Exemplar (31.5) — a local reversal also observed on the CIFAR-100 results in Section 4.1.

The Rotation-over-Contrastive reversal is particularly notable on Tiny ImageNet. One structural explanation is the image size: on 64×64 images, a 90° rotation is a more visually salient and informationally rich transformation than on 32×32 images. The student may extract more complementary geometric structure from the rotation pretext task on higher-resolution images, making it a stronger auxiliary signal than similarity matching among four views. The contrastive task, by contrast, depends on learned embedding quality, which may not improve proportionally with image resolution given the fixed VGG head architecture.

**SSKD consistently outperforms the vanilla student on TIN.** We ran a vanilla vgg8 student on Tiny ImageNet and obtained a top-1 validation accuracy of 59.41%. With this baseline in hand, the improvement margins are:

| Method | Student Acc (TIN, %) | Δ vs. Vanilla (pp) |
|---|---|---|
| SSKD — Rotation | 63.41 | +4.00 |
| SSKD — Contrastive | 63.22 | +3.81 |
| SSKD — Exemplar | 62.96 | +3.55 |
| SSKD — Jigsaw | 62.73 | +3.32 |

The best method (Rotation, +4.00 pp) slightly exceeds the CIFAR-100 improvement margin (Contrastive, +3.80 pp: 74.53% − 70.73%), and all four methods clear a 3.32 pp improvement floor. This confirms that SSKD's advantage over standard training persists on Tiny ImageNet with improvement magnitudes broadly consistent with the CIFAR-100 results. A standard KD-only baseline (without self-supervision) was not run on TIN, so the contribution of the KD signal alone cannot be isolated on this dataset; the gains above reflect the full four-component SSKD loss (CE + KD + TF + SS) relative to vanilla cross-entropy training.

**The SS-quality to student-accuracy correlation weakens on TIN.** The paper's central claim rests on a monotone Exemplar < Jigsaw < Rotation < Contrastive ordering driven by SS quality measured on ImageNet. On TIN, the same ordering is partially disrupted: Rotation (SS quality 48.9) outperforms Contrastive (SS quality 69.3), and Exemplar (SS quality 31.5) outperforms Jigsaw (SS quality 45.7). These reversals parallel what we observe on the resnet56→resnet20 pair (Section 4.4), where Rotation also surpasses Contrastive. Taken together, the evidence from both the new dataset and the new architecture suggests that the SS-quality to student-accuracy correlation established in Table 2 is not fully robust across experimental conditions. The correlation holds at the extremes (Jigsaw and Exemplar at the bottom, Rotation and Contrastive at the top), but the specific top-two ordering is sensitive to both dataset and architecture.

---

### 4.4 Experiment 4: New Architecture Variant — resnet56 → resnet20 on CIFAR-100

**Question:** The SSKD paper evaluates all four SS methods only on the vgg13→vgg8 pair (Table 2) and applies only the contrastive method to other architecture pairs (Table 3). Does the positive correlation between SS method quality and student accuracy hold when all four SS methods are applied to a structurally different architecture pair, resnet56→resnet20?

**Setup:**

We train four separate SSKD models on CIFAR-100 using the resnet56→resnet20 pair, one per SS method. All four experiments use the hyperparameters from Section 3.5. Parameters not specified in the paper — teacher SS schedule and seed — are taken from the codebase defaults: the teacher's SS module is trained for 60 epochs with an initial learning rate of 0.05 decayed by 0.1 at epochs 30 and 45, and the seed is fixed at 0. For the contrastive method, selective transfer is applied to $\mathcal{L}_{ss}$ with ratio 0.75 (i.e., top-75% least-wrong incorrect predictions are kept); for Rotation, Jigsaw, and Exemplar, this filtering is not applied since these methods use classification-based matching rather than similarity ranking.

Each SS head attaches to `feats[-1]`, the backbone's final feature layer before the classifier. The head architectures are as defined in Sections 2.3 and 3.3: a two-layer MLP for Contrastive, and single linear layers for Rotation (4-way), Jigsaw (24-way), and Exemplar (50,000-way). A key architectural detail is that CIFAR-variant ResNets in this codebase use `num_filters = [16, 16, 32, 64]`, so both resnet56 and resnet20 produce 64-dimensional feature vectors after global average pooling — compared to 512-dimensional vectors for VGG13 and VGG8. This difference becomes relevant in the analysis below.

**Results:**

The following table reports the best validation accuracy (top-1, %) achieved during training for each SS method on the resnet56→resnet20 pair. SS Quality is the ImageNet linear evaluation accuracy (ResNet-50) reported in SSKD Table 2. The Paper Acc column shows the student accuracy from Table 2 on vgg13→vgg8 for comparison. The observed ranking on the resnet pair (Rotation > Contrastive > Exemplar ≈ Jigsaw) disrupts the paper's vgg-pair ordering (Contrastive > Rotation > Jigsaw > Exemplar).

| | SS Quality (%) | Paper Acc (vgg pair, %) | Student Acc (resnet pair, %) |
|---|---|---|---|
| Teacher | — | — | 73.44 |
| Vanilla student | — | — | 69.63 |
| **SSKD — Contrastive** | 69.3 | 75.48 | **70.87** |
| **SSKD — Rotation** | 48.9 | 75.01 | **71.31** |
| **SSKD — Jigsaw** | 45.7 | 74.85 | 70.68 |
| **SSKD — Exemplar** | 31.5 | 74.57 | 70.70 |

The following table provides additional context: the resnet56 teacher's SS pretext task accuracy on CIFAR-100 after the 60-epoch SS training phase. These values measure how well the teacher *solves* each SS task, which is distinct from the SS Quality column above (which measures representation quality via ImageNet linear evaluation).

| SS Method | Task Type | # Classes | Val SS Acc (%) | Train SS Acc (%) |
|---|---|---|---|---|
| Contrastive | Similarity matching | — | 78.62 | 74.81 |
| Rotation | Classification | 4 | 42.82 | 44.11 |
| Jigsaw | Classification | 24 | 28.20 | 29.27 |
| Exemplar | Classification | 50,000 | 0.00 | 9.72 |

**Analysis:**

**The ranking is partially disrupted.** The paper predicts the ordering Exemplar < Jigsaw < Rotation < Contrastive. Our resnet56→resnet20 results produce a different ranking:

> Jigsaw (70.68) ≤ Exemplar (70.70) < Contrastive (70.87) < **Rotation (71.31)**

Two deviations stand out. First, Rotation outperforms Contrastive by 0.44 percentage points, reversing the top-two ordering from Table 2. Second, Exemplar slightly outperforms Jigsaw (70.70 vs. 70.68), reversing their expected positions, although the 0.02 pp difference is within noise and effectively a tie.

**Rotation's advantage on ResNets.** Rotation achieves the highest student accuracy on the resnet pair, though the margin is small (0.44 pp) and based on a single seed, so we interpret this cautiously. Still, the reversal invites analysis of why Contrastive may benefit less from the ResNet backbone.

The most concrete architectural difference is feature dimensionality at the SS head attachment point. As noted above, both resnet56 and resnet20 produce 64-dimensional feature vectors, compared to 512 dimensions for VGG13/VGG8. The contrastive SS head projects features into an embedding space of the *same* dimensionality as the input (`nn.Linear(feat_dim, feat_dim) → ReLU → nn.Linear(feat_dim, feat_dim)`), so the contrastive task computes cosine similarities in a 64-dimensional space on the resnet pair versus a 512-dimensional space on the VGG pair — an 8× reduction in embedding capacity. Computing meaningful similarity rankings in such a low-dimensional space is inherently more constrained, potentially limiting how well the contrastive head can discriminate between views. Rotation prediction, by contrast, is a 4-way classification (`nn.Linear(64, 4)`) whose effectiveness does not depend on embedding richness, making it more robust to low feature dimensionality. This suggests that the contrastive method's relative advantage may scale with the backbone's feature dimension — a hypothesis that could be tested by widening the ResNet's final layer or using a larger projection head.

A second observation concerns teacher SS training quality. The resnet56 teacher achieves 42.82% validation accuracy on rotation prediction (chance level: 25%), while the contrastive head reaches 78.62%. Despite the contrastive head's higher raw accuracy, this does not translate to a student-accuracy advantage. One possible explanation is that what matters for distillation is not how well the teacher solves the SS task, but how well the SS-derived gradients *complement* the standard KD signal — and a 4-way geometric classification may provide more complementary training signal than similarity matching in a low-dimensional embedding space.

Confirming the feature-dimension hypothesis would require controlled ablations (e.g., varying projection head dimension while holding everything else fixed) and multi-seed runs. The broader question of how architecture interacts with SS method choice is studied by Kolesnikov et al. (2019) [9], who find that the optimal pretext task varies with architecture configuration across different ResNet widths and depths — consistent with the architecture-dependent ranking we observe, though their setting (self-supervised pre-training on ImageNet) differs from ours (SS-augmented KD on CIFAR-100).

**The bottom two methods are tightly clustered.** Jigsaw and Exemplar produce nearly identical accuracy (70.68 vs. 70.70), despite a substantial gap in SS Quality (45.7 vs. 31.5). This suggests that below a certain SS quality threshold, the marginal benefit to student accuracy levels off, and the positive correlation the paper observes at the top of the quality range does not extend to the bottom. The vgg13→vgg8 results in Section 4.1 show a similar but milder pattern: the Exemplar–Jigsaw gap is 0.12 pp in our reproduction (0.28 pp in the paper), much smaller than the Rotation–Contrastive gap.

**All four SSKD variants improve over the vanilla student.** Despite the ranking disruption, all four SS methods produce accuracy above 70.6%, a meaningful improvement over the vanilla resnet20 baseline (69.63%). The best-performing method (Rotation, 71.31%) closes 44% of the gap between the vanilla student and the teacher (73.44%). This supports the broader claim that integrating self-supervision into KD is beneficial regardless of the SS method used, even if the specific ranking is architecture-dependent.

**Implications for the "model-agnostic" claim.** The paper presents SSKD as a general framework where SS quality predicts student accuracy regardless of architecture. Our resnet pair results partially challenge this: while all four methods improve student accuracy (supporting generality), the specific ranking does not transfer from VGG to ResNet (challenging the quality–accuracy correlation as a universal law). The reversal of Contrastive and Rotation suggests that the optimal SS method may depend on architectural properties such as feature dimensionality. Practitioners applying SSKD to new architectures should not assume that the contrastive method will always be the best choice; evaluating multiple SS methods on the target architecture is advisable.

---

## 5. Discussion & Conclusion

We set out to test the central claim of SSKD (Xu et al., ECCV 2020) [2]: that the quality of a self-supervised pretext task positively correlates with student accuracy in knowledge distillation, and that integrating any SS method into KD improves over standard distillation. We reproduced Table 2 on the original vgg13→vgg8 pair (Section 4.1), proposed two KD loss modifications (Section 4.2), extended the evaluation to Tiny ImageNet (Section 4.3), and tested all four SS methods on a new architecture pair, resnet56→resnet20 (Section 4.4). Below we synthesize our findings, address limitations, and reflect on the reproducibility process.

### 5.1 Verdict on the Paper's Central Claim

Our reproduction partially supports the paper's central claim. The claim has two components: (1) all four SS methods improve student accuracy over vanilla training, and (2) student accuracy is positively correlated with SS method quality, producing the monotonic ranking Exemplar < Jigsaw < Rotation < Contrastive.

Component (1) is robustly supported. Across all three experimental settings — vgg13→vgg8 on CIFAR-100 (Section 4.1), vgg13→vgg8 on Tiny ImageNet (Section 4.3), and resnet56→resnet20 on CIFAR-100 (Section 4.4) — every SSKD variant outperforms the vanilla student by a meaningful margin. On CIFAR-100 with the vgg pair, the worst SSKD method (Jigsaw, 74.34%) still exceeds the vanilla student (70.73%) by 3.61 percentage points. On Tiny ImageNet the improvement floor is +3.32 pp (Jigsaw, 62.73% vs. vanilla 59.41%). On the resnet pair the improvement is smaller but consistent, with all four methods above 70.6% compared to the vanilla baseline of 69.63%. This finding holds regardless of dataset, architecture, or SS method, which is a strong endorsement of the framework's general utility.

Component (2), the monotonic quality–accuracy correlation, receives weaker support. On the original vgg13→vgg8 pair (Section 4.1), the coarse two-tier structure is preserved — Contrastive and Rotation form a top group, Jigsaw and Exemplar form a bottom group — but the fine-grained ordering within tiers is disrupted: Exemplar (74.46%) outperforms Jigsaw (74.34%) in our reproduction, reversing their paper-reported positions. More critically, both the resnet pair (Section 4.4) and Tiny ImageNet (Section 4.3) produce the ranking Rotation > Contrastive, directly contradicting the paper's top-two ordering. The reversal is consistent across two independent experimental axes (new architecture, new dataset), which makes it unlikely to be a one-off fluctuation.

In summary: SSKD as a *framework* — adding self-supervision to knowledge distillation — is well-supported by our results. But the specific claim that SS method quality, as measured by ImageNet linear evaluation, predicts student accuracy in a monotonic fashion is not robust across architectures and datasets. Practitioners should treat the quality–accuracy correlation as a useful heuristic for the original vgg setting rather than a universal law.

### 5.2 Generalization to Tiny ImageNet

The Tiny ImageNet experiments (Section 4.3) confirm that SSKD's advantage over vanilla training transfers to a new dataset. All four SS methods improve over the vanilla vgg8 student (59.41%) by at least 3.32 pp, with improvement magnitudes comparable to the CIFAR-100 results. This is a positive signal for SSKD's generality beyond the single dataset evaluated in the original paper.

Two differences emerge, however. First, the best SSKD variant (Rotation, 63.41%) falls 0.67 pp short of the teacher (64.08%), whereas on CIFAR-100 the best variant (Contrastive, 74.53%) essentially matches the teacher (74.49%). The larger teacher–student gap suggests that SSKD's effectiveness diminishes on harder tasks with more classes and higher resolution, consistent with the intuition that a larger output space makes it harder for a compact student to fully absorb the teacher's knowledge. Second, Rotation overtakes Contrastive on TIN (63.41% vs. 63.22%), paralleling the reversal observed on the resnet pair. One structural explanation is that 90° rotations become a more informative geometric signal on 64×64 images than on 32×32 images, while the contrastive head's effectiveness is bounded by the fixed VGG projection architecture. Since a standard KD-only baseline (without self-supervision) was not run on Tiny ImageNet, we cannot isolate the SS signal's contribution from the combined four-component loss. The gains above represent the full SSKD pipeline relative to vanilla training, which limits the precision of conclusions about the SS component specifically.

### 5.3 Generalization to a New Architecture Pair

The resnet56→resnet20 experiments (Section 4.4) provide the strongest evidence that the SS-quality → student-accuracy correlation is architecture-dependent. Rotation achieves the highest student accuracy (71.31%), outperforming Contrastive (70.87%) by 0.44 pp — a reversal that is consistent with the Tiny ImageNet result and that we traced to a concrete architectural difference: CIFAR-variant ResNets produce 64-dimensional feature vectors after global average pooling, compared to 512 dimensions for the VGG pair. The contrastive head projects features into an embedding space of the same dimensionality as the input, so on the resnet pair it computes cosine similarities in a 64-dimensional space — an 8× reduction compared to VGG. Rotation prediction, as a simple 4-way classification, is less sensitive to this bottleneck.

This finding has a practical implication: when applying SSKD to a new architecture, the best SS method cannot be assumed from prior benchmarks. The original paper acknowledges that SSKD is "model-agnostic" because it transfers only output-level signals, but our results show that *which* output-level signal works best depends on the backbone's feature geometry. Evaluating multiple SS methods on the target architecture is advisable.

### 5.4 Loss Component Contributions

The DKD and WSLD modifications (Section 4.2) each improve slightly over the SSKD baseline (Contrastive): DKD reaches 74.68% (+0.15 pp) and WSLD reaches 74.61% (+0.08 pp), compared to the baseline of 74.53%. Both gains are modest, which is itself informative: it suggests that the self-supervised auxiliary signal already captures much of the complementary knowledge that modifications to the standard KD loss aim to recover. In other words, the SS channel and improved KD losses address overlapping sources of dark knowledge, so their benefits do not stack additively. DKD's slightly stronger performance is consistent with its mechanism — decoupling target-class and non-target-class signals is well-suited to CIFAR-100's fine-grained label space (100 visually similar classes). Since DKD and WSLD operate on different axes (class-level structure vs. sample-level weighting), combining the two is a natural direction for future work, though the small individual gains temper expectations for the combination.

### 5.5 Limitations

Several factors constrain the scope and precision of our conclusions.

All results are single-run with a fixed seed (seed = 0). The observed differences between SS methods — particularly the Rotation–Contrastive gap of 0.44 pp on the resnet pair and 0.19 pp on Tiny ImageNet — fall within the range of typical single-run variance (0.3–0.5 pp on CIFAR-100). Multi-seed experiments with confidence intervals would be needed to confirm whether these reversals are statistically significant or within noise.

Our compute budget limited us to two architecture pairs (vgg13→vgg8 and resnet56→resnet20) and two datasets (CIFAR-100 and Tiny ImageNet). The original paper evaluates SSKD (with the contrastive method only) on six additional teacher–student pairs in Tables 3 and 4. We cannot assess whether the quality–accuracy ranking transfers to larger models such as ResNet-110→ResNet-32 or WRN-40-2→WRN-16-2.

The Tiny ImageNet pipeline required several adaptations beyond the CIFAR-100 setup: a compressed training schedule (140 vs. 240 epochs), earlier LR decay milestones, stronger teacher augmentation (RandAugment and RandomErasing), and larger random-crop padding. These changes were necessary for convergence but introduce confounders that make direct comparison to CIFAR-100 results imperfect. In particular, the stronger teacher augmentation may inflate the teacher's accuracy relative to a CIFAR-100-style training protocol, which could affect the teacher–student gap.

Our experiments were run on RTX 3060 and RTX 3090 GPUs, whereas the original paper used a TITAN-X-Pascal. Floating-point nondeterminism across hardware can produce small numerical differences, though this is unlikely to explain the systematic downward shift (~0.3–1.0 pp) observed across all methods in our vgg pair reproduction. A more probable explanation is the combination of single-seed variance and potential differences in library versions.

### 5.6 Lessons on Reproducibility

Three concrete lessons emerged from this reproduction that go beyond the general observation that reproducibility is valuable.

**Verify what the code implements, not what the paper describes.** The most impactful discovery was the discrepancy between the paper's loss weight assignment and the codebase's actual values. The paper's Eq. 8 assigns $\lambda_3 = 2.7$ to $\mathcal{L}_{ss}$ and $\lambda_4 = 10.0$ to $\mathcal{L}_T$, giving the larger weight to the transformed-data KD term. The released codebase uses `ss_weight=10.0` and `tf_weight=2.7`, effectively reversing the two — giving the larger weight to the self-supervised term instead. Since the paper's reported numbers were generated by the code, the codebase values are what was actually executed. We followed the codebase values in all our experiments. This kind of paper–code mismatch is easy to miss if one reads only the paper and treats the code as a black box, but it can change the interpretation of results.

**Incomplete codebases demand implementation work that blurs the line between reproduction and extension.** The original SSKD repository implements only the contrastive pretext task. Reproducing Table 2 in full — which compares all four SS methods — required us to implement Rotation, Jigsaw, and Exemplar from the paper's textual descriptions in the Appendix (§6.3). This means that Table 2 as published is not independently verifiable without new implementation work, and any discrepancy between our results and the paper's could stem from differences in our implementations rather than from issues with the original claim. The course distinguishes "Reproduced" (existing code evaluated) from "Replicated" (full re-implementation). Our project is technically "Reproduced" for the contrastive method and closer to "Replicated" for the other three, which required writing new SS heads from paper descriptions alone.

**Relative trends matter more than absolute numbers.** Our reproduced accuracies are systematically 0.3–1.0 pp below the paper's across all methods and both architecture pairs. This downward shift likely reflects pipeline-level factors (random seed, single run vs. averaged results, library versions) rather than method-specific issues, since it applies equally to the contrastive method — which reuses the original authors' code — and to our new implementations. The key question in a reproduction is not whether the absolute numbers match exactly, but whether the relative trends and conclusions hold. In our case, the broad two-tier ranking (Contrastive/Rotation at the top, Jigsaw/Exemplar at the bottom) and the consistent improvement over vanilla training both survive the reproduction, even though the fine-grained monotonic ordering and the absolute magnitudes do not match precisely. This distinction — between reproducing a *trend* and reproducing a *number* — is the most useful lens for evaluating reproduction outcomes.

---

## 6. Author Contributions

**Yanzhe** reproduced Table 2 (four SS methods on vgg13→vgg8, CIFAR-100) and proposed two complementary loss modifications (DKD and WSLD) to address limitations in the standard KD loss (Criteria: Reproduced, New algorithm variant). He implemented the Rotation, Jigsaw, and Exemplar SS heads based on the paper's Appendix descriptions.

**Chenyu** trained the vgg13→vgg8 pair on Tiny ImageNet, adapting the training schedule, augmentation pipeline, and data loader for the new dataset (Criterion: New data).

**Shanghong** ran all four SS methods on the resnet56→resnet20 pair on CIFAR-100, including debugging the CIFAR-variant ResNet architecture and analyzing the feature-dimensionality hypothesis (Criterion: New algorithm variant).

All members contributed to writing the blog post. The shared repository is available at [github.com/FN8211/SSKD](https://github.com/FN8211/SSKD).
**Writing notes:** This is where the Exposition grade is won or lost. Be specific, not generic. Don't write "reproducibility is important" — write about what *this* reproduction taught you.

---

### 6. Author Contributions

**Content:** A brief table or paragraph listing what each member did. Required by the submission guidelines. Example:

> **Yanzhe** reproduced Table 2 (four SS methods on vgg13→vgg8, CIFAR-100), explored the effect of loss components ($L_T$, $L_{ss}$) on student accuracy (Criterion: Reproduced). He also proposed two complementary loss modifications (DKD, WSLD) to address limitations in the standard KD loss (Criterion: New algorithm variant).
> **Chenyu** trained the vgg13→vgg8 pair on Tiny ImageNet, adapting the pipeline for the new dataset (Criterion: New data). During setup, he identified and fixed a bug in the original author's `teacher.py` where `scheduler.step()` was never called (Criterion: Reproduced).
> **Shanghong** ran all four SS methods on the resnet56→resnet20 pair on CIFAR-100 (Criterion: New algorithm variant). All members contributed to writing the blog post.
> 

---

### 7. References


[1] G. Hinton, O. Vinyals, and J. Dean, "Distilling the knowledge in a neural network," arXiv preprint arXiv:1503.02531, 2015.

[2] G. Xu, Z. Liu, X. Li, and C. C. Loy, "Knowledge distillation meets self-supervision," in *Proc. European Conf. Comput. Vis. (ECCV)*, 2020, pp. 588–604.

[3] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, "A simple framework for contrastive learning of visual representations," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2020, pp. 1597–1607.

[4] A. Dosovitskiy, J. T. Springenberg, M. Riedmiller, and T. Brox, "Discriminative unsupervised feature learning with exemplar convolutional neural networks," in *Proc. Advances in Neural Inf. Process. Syst. (NeurIPS)*, vol. 27, 2014, pp. 2017–2025.

[5] M. Noroozi and P. Favaro, "Unsupervised visual representation learning by solving jigsaw puzzles," in *Proc. European Conf. Comput. Vis. (ECCV)*, 2016, pp. 69–84.

[6] S. Gidaris, P. Singh, and N. Komodakis, "Unsupervised representation learning by predicting image rotations," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2018.

[7] B. Zhao, Q. Cui, R. Song, Y. Qiu, and J. Liang, "Decoupled knowledge distillation," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2022, pp. 11953–11962.

[8] H. Zhou, L. Song, J. Chen, Y. Zhou, G. Wang, J. Yuan, and Q. Zhang, "Rethinking soft labels for knowledge distillation: A bias-variance tradeoff perspective," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2021.

[9] A. Kolesnikov, X. Zhai, and L. Beyer, "Revisiting self-supervised visual representation learning," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2019, pp. 1920–1929.

[10] Y. Tian, D. Krishnan, and P. Isola, "Contrastive representation distillation," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2020.