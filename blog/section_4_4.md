### 4.4 Experiment 4: New Architecture Variant — resnet56 → resnet20 on CIFAR-100

*Owner: Shanghong · Criterion: New algorithm variant*

**Question:** The SSKD paper evaluates all four self-supervision methods only on the vgg13→vgg8 pair (Table 2), and applies only the contrastive method to other architecture pairs (Table 3). Does the positive correlation between SS method quality and student accuracy — Table 2's central claim — hold when all four SS methods are applied to a structurally different architecture pair, resnet56→resnet20?

#### Setup

We train four separate SSKD models on CIFAR-100 using the resnet56→resnet20 teacher-student pair, one for each SS method: Contrastive, Rotation, Jigsaw, and Exemplar. All four experiments use the same hyperparameters as the vgg pair experiments. Most values follow §6.4 of the paper; those not specified in the paper (teacher SS schedule, seed) are taken from the codebase defaults:

| Hyperparameter | Value |
|---|---|
| Total epochs | 240 |
| Teacher SS epochs | 60 |
| Batch size | 64 |
| Optimizer | SGD (momentum 0.9, weight decay 5×10⁻⁴) |
| LR schedule | 0.05, ×0.1 at epochs 150, 180, 210 |
| Teacher SS LR | 0.05, ×0.1 at epochs 30, 45 |
| τ_kd, τ_tf | 4.0 |
| τ_ss | 0.5 |
| λ_CE, λ_KD, λ_TF, λ_SS | 0.1, 0.9, 2.7, 10.0 † |
| Selective transfer ratio (TF / SS) | 1.0 / 0.75 (contrastive), 1.0 / — (others) |
| Seed | 0 |

† The paper's Eq. 8 writes the loss as L = λ₁L_ce + λ₂L_kd + λ₃L_ss + λ₄L_T with λ₃ = 2.7 and λ₄ = 10.0, assigning the larger weight to L_T. However, the released codebase uses `tf-weight=2.7` and `ss-weight=10.0`, assigning the larger weight to L_ss. We follow the codebase since it determines what was actually executed.

**SS head architectures.** Each SS method attaches a task-specific projection head to the backbone's last feature layer (the penultimate layer before the classifier). All four wrappers use `is_feat=True` to extract intermediate features from the ResNet backbone and attach the head to `feats[-1]`. The key difference lies in the head itself:

- **Contrastive:** Two-layer MLP projection head (Linear → ReLU → Linear), matching the original SSKD implementation. The head projects features into an embedding space where contrastive similarity is computed across transformed views.
- **Rotation:** Single linear layer mapping features to 4 classes (0°, 90°, 180°, 270°). Each image is rotated to produce four views; the SS task is to predict the rotation angle.
- **Jigsaw:** Single linear layer mapping features to 24 classes (4! permutations). Each image is split into a 2×2 grid and the patches are shuffled; the SS task is to predict which permutation was applied.
- **Exemplar:** Single linear layer mapping features to 50,000 classes (one per training sample). Each image is augmented with heavy color jitter, random resized cropping, and horizontal flipping; the SS task is instance discrimination.

Of the four methods, only Contrastive was implemented in the original SSKD codebase. We implemented Rotation, Jigsaw, and Exemplar following the descriptions in §6.3 of the paper. The Rotation and Jigsaw heads use cross-entropy loss against discrete class labels, replacing the contrastive cosine-similarity matching loss. The Exemplar head also uses cross-entropy but against a much larger label space (50,000 training instances). For Rotation, Jigsaw, and Exemplar, the selective transfer strategy for L_ss (ratio-ss = 0.75) is not applied, as these methods use classification-based matching rather than similarity ranking.

#### Results

Table 1 reports the best validation accuracy (top-1, %) achieved during training for each SS method on the resnet56→resnet20 pair.

**Table 1: Student accuracy (%) on CIFAR-100 for resnet56→resnet20 across four SS methods.** SS Quality is the linear evaluation accuracy on ImageNet (with ResNet50) as reported in SSKD Table 2 (sourced from [4,21,27]). The Paper Acc column shows the student accuracy reported in the paper's Table 2 on vgg13→vgg8 for comparison. The observed ranking on the resnet pair (Rotation > Contrastive > Exemplar ≈ Jigsaw) disrupts the paper's vgg-pair ordering (Contrastive > Rotation > Jigsaw > Exemplar), with Rotation surpassing Contrastive by 0.44 percentage points.

| | SS Quality (%) | Paper Acc (vgg pair, %) | Student Acc (resnet pair, %) |
|---|---|---|---|
| Teacher | — | — | 73.44 |
| Vanilla student | — | — | 69.63 |
| **SSKD — Contrastive** | 69.3 | 75.48 | **70.87** |
| **SSKD — Rotation** | 48.9 | 75.01 | **71.31** |
| **SSKD — Jigsaw** | 45.7 | 74.85 | 70.68 |
| **SSKD — Exemplar** | 31.5 | 74.57 | 70.70 |

To provide additional context, Table 2 reports the teacher's SS pretext task accuracy on CIFAR-100 after the 60-epoch SS training phase. These values reflect how well the resnet56 teacher learns each self-supervision task, and should be distinguished from the SS Quality column above (which measures representation quality via ImageNet linear evaluation).

**Table 2: Teacher SS pretext task accuracy (%) on CIFAR-100 after 60 epochs of SS module training.** The contrastive task achieves the highest teacher accuracy (78.62%), while the exemplar task remains near zero due to its 50,000-way instance-classification formulation. These values measure how well the teacher solves each SS task on CIFAR-100, not the quality of the learned representations.

| SS Method | Task Type | # Classes | Val SS Acc (%) | Train SS Acc (%) |
|---|---|---|---|---|
| Contrastive | Similarity matching | — | 78.62 | 74.81 |
| Rotation | Classification | 4 | 42.82 | 44.11 |
| Jigsaw | Classification | 24 | 28.20 | 29.27 |
| Exemplar | Classification | 50,000 | 0.00 | 9.72 |

#### Analysis

**The ranking is partially disrupted.** The paper predicts that SS method quality correlates positively with student accuracy, yielding the ordering Exemplar < Jigsaw < Rotation < Contrastive. Our resnet56→resnet20 results produce a different ranking:

> Jigsaw (70.68) ≤ Exemplar (70.70) < Contrastive (70.87) < **Rotation (71.31)**

Two deviations stand out. First, Rotation outperforms Contrastive by 0.44 percentage points, reversing the paper's top-two ordering. Second, Exemplar slightly outperforms Jigsaw (70.70 vs. 70.68), reversing their expected positions, although the 0.02 percentage point difference is within noise and effectively a tie.

**Rotation's advantage on ResNets.** The most striking finding is that Rotation — not Contrastive — achieves the highest student accuracy on the resnet pair, though the margin is small (0.44 pp) and based on a single seed, so we interpret this cautiously. Still, the reversal from the paper's predicted ordering invites analysis of why Contrastive may benefit less from the ResNet backbone.

The most concrete architectural difference is **feature dimensionality at the SS head attachment point**. The CIFAR-variant ResNets in this codebase use `num_filters = [16, 16, 32, 64]`, so both resnet56 and resnet20 produce 64-dimensional feature vectors after global average pooling. By contrast, VGG13 and VGG8 produce 512-dimensional vectors. The contrastive SS head (`wrapper.py`) is a two-layer MLP that projects features into an embedding space of the *same* dimensionality as the input: `nn.Linear(feat_dim, feat_dim) → ReLU → nn.Linear(feat_dim, feat_dim)`. This means the contrastive task computes cosine similarities in a 64-dimensional space on the resnet pair, compared to a 512-dimensional space on the VGG pair — an 8× reduction in embedding capacity. Computing meaningful similarity rankings in a 64-dimensional space is inherently more constrained, potentially limiting how much the contrastive head can discriminate between views. Rotation prediction, by contrast, is a 4-way classification (`nn.Linear(64, 4)`) that does not depend on embedding richness, making it more robust to low feature dimensionality. This suggests that the contrastive method's relative advantage may scale with the backbone's feature dimension — a hypothesis that could be tested by widening the ResNet's final layer or using a larger projection head.

A second observation concerns teacher SS training quality. The resnet56 teacher achieves 42.82% validation accuracy on rotation prediction (chance level: 25%), while the contrastive head reaches 78.62%. Despite the contrastive head's higher raw accuracy, this does not translate to a student-accuracy advantage. One possible explanation is that what matters for KD is not how well the teacher solves the SS task, but how well the SS-derived features *complement* the standard KD signal — and a 4-way geometric classification may produce more complementary gradients than similarity matching in a low-dimensional embedding space.

We note that confirming the feat_dim hypothesis would require controlled ablations (e.g., varying projection head dimension while holding everything else fixed) and multi-seed experiments. The broader question of how architecture interacts with SS method choice is studied by Kolesnikov et al. (2019), who compare Rotation, Exemplar, and Jigsaw across different ResNet widths and depths and find that the optimal pretext task varies with architecture configuration — consistent with the architecture-dependent ranking we observe here, though their setting (self-supervised pre-training on ImageNet) differs from ours (SS-augmented knowledge distillation on CIFAR-100).

**The bottom two methods are tightly clustered.** Jigsaw and Exemplar produce nearly identical student accuracy on the resnet pair (70.68 vs. 70.70), despite a substantial gap in SS Quality (45.7 vs. 31.5). This suggests that below a certain SS quality threshold, the marginal benefit to student accuracy levels off — and that the linear correlation the paper observes at the top of the quality range does not extend to the bottom. The paper's Table 2 shows a similar but milder pattern on vgg13→vgg8, where the gap between Exemplar (74.57) and Jigsaw (74.85) is 0.28 percentage points compared to the 0.47 pp gap between Rotation (75.01) and Contrastive (75.48).

**All four SSKD variants improve over the vanilla student.** Despite the ranking disruption at the top, all four SS methods produce student accuracy above 70.6%, representing a meaningful improvement over the vanilla resnet20 baseline (69.63%). The best-performing method (Rotation, 71.31%) closes 44% of the gap between the vanilla student and the teacher (73.44%). This supports the broader claim that integrating self-supervision into knowledge distillation is beneficial, even if the specific ranking of SS methods is architecture-dependent.

**Implications for the "model-agnostic" claim.** The paper presents SSKD as a general framework where SS quality predicts student accuracy regardless of architecture. Our resnet pair results partially challenge this claim: while all four methods do improve student accuracy (supporting generality), the specific ranking does not transfer from VGG to ResNet architectures (challenging the quality-accuracy correlation as a universal law). The reversal of Contrastive and Rotation rankings suggests that the optimal SS method may depend on architectural properties. This finding implies that practitioners applying SSKD to new architectures should not assume that the contrastive method will always be the best choice; evaluating multiple SS methods on the target architecture is advisable.
