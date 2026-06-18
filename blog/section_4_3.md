### 4.3 Experiment 3: New Dataset — vgg13 → vgg8 on Tiny ImageNet

*Owner: Chenyu · Criterion: New data*

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