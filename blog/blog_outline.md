### 1. Title & Metadata

**Content:**

- Title: e.g. *"Reproducing 'Knowledge Distillation Meets Self-Supervision': How Different Pretext Tasks Shape Student Learning"*
- Authors (all three group members)
- Date
- Link to repository: `github.com/FN8211/SSKD`
- One-line summary: which paper, what was reproduced, main finding

**Why:** Sets context immediately. Peer-understandable from the first line (WG4, WI1).

---

### 2. Introduction & Background

**Purpose:** Make the blog self-contained by first introducing the key concepts and the SSKD method/pipeline, then building on that foundation to motivate why this paper is worth reproducing and what this blog sets out to do. A peer who has not read the SSKD paper should be able to follow the rest of the blog after this section (WS1, WG2).

**Content to cover:**

*2a. Key concepts and method (establish the technical context first):*

- What is knowledge distillation? Soft targets, temperature scaling, KL divergence. Define all symbols (WM5). (1–2 paragraphs; define before using — WG4)
- What does SSKD add? Self-supervised pretext tasks as an auxiliary channel to extract richer "dark knowledge" from the teacher.
- Contrastive prediction as the main pretext task: transformation pool, projection head, contrastive loss. Explain in English first, then the equation (WM6, WM7).
- The SSKD training pipeline: teacher backbone is frozen; SS module is fine-tuned on transformed data; student simultaneously minimizes classification loss (L_ce), KD loss (L_kd), transformed-data KD loss (L_T), and SS matching loss (L_ss).
- Selective transfer strategy (top-k% noisy predictions).
- Brief description of the four SS methods evaluated in Table 2: Contrastive, Rotation, Jigsaw, Exemplar. Define each in 1–2 sentences.
- The paper's central empirical claim: SS method quality correlates positively with student accuracy (Table 2).

*2b. Reproduction motivation (build on the technical context):*

- Why is this paper worth reproducing? Now that the reader understands the pipeline and the claim, motivate: (a) SSKD introduces a framework-level idea (SS+KD), and its generality rests on the ablation across SS methods; (b) the original codebase implements only the contrastive pretext task, so the other three methods in Table 2 have no publicly verifiable code — independent reproduction is needed to verify the full claim. (c) the generality of the student performance correlated with ss method is worth exploring.
- What do we do in this blog? State your scope: (1) reproduce Table 2 on vgg13→vgg8; (2) explore loss component effects; (3) evaluate on Tiny ImageNet; (4) run all four SS methods on resnet56→resnet20.

**Writing notes:**

- The flow is: define KD → introduce SSKD's idea and pipeline → state the paper's claim → *then* argue why reproducing that claim matters → state what this blog covers. The motivation lands harder because the reader already understands what they are being asked to care about.
- Keep the method summary at the "logical narrative" level, not a full methods rewrite. Only define what you actually use later in your experiments (WM8).
- Use a pipeline figure if helpful (WI4 — visual abstract). The SSKD paper's Figure 1 is a good reference for style, but create your own version to avoid copyright issues.

**Rubric hit:** Content (self-contained technical context), Exposition (value of reproduction), Motivation (clear plan).

---

### 3. Reproduction Scope & Plan

**Purpose:** State exactly what you set out to do, how tasks are divided, and which reproducibility criteria each member owns. This is **explicitly required** by the submission guidelines.

**Content to cover:**

- **Primary target:** Table 2 of the SSKD paper (Influence of Different Self-Supervision Tasks): four SS methods × vgg13→vgg8 on CIFAR-100.
- **Extensions beyond Table 2:**
    - (a) Effects of new loss component [TO BE ADDED];
    - (b) New dataset — train the vgg13→vgg8 pair on Tiny ImageNet to test whether SSKD's advantage transfers beyond CIFAR-100;
    - (c) New architecture variant — train the resnet56→resnet20 pair on CIFAR-100 with all four SS methods to test whether the SS-quality → student-accuracy correlation holds across architectures.
- **Codebase status:** The original author repository (`xuguodong03/SSKD`) implements only the contrastive pretext task. Rotation, Jigsaw, and Exemplar are described in the paper's Appendix (§6.3) but have no public code. All four SS methods needed to be run for Table 2 reproduction and the resnet pair extension, requiring implementation of the missing heads.
- **Per-member criteria table:**

| Experiments | Criterion |
| --- | --- |
| Reproduce Table 2 (four SS methods, vgg13→vgg8, CIFAR-100) | Reproduced |
| Explore new loss component effects on student accuracy | New algorithm variant |
| Train vgg13→vgg8 on Tiny ImageNet | New data |
| Train resnet56→resnet20 on CIFAR-100 with all four SS methods (Contrastive, Rotation, Jigsaw, Exemplar) | New algorithm variant |
- **Infrastructure:** GPU server (RTX 3090, 24 GB), training details (epochs, LR schedule, batch size, optimizer — from §6.4 of the paper).

**Writing notes:**

- Be precise about what "Reproduced" means: existing code was evaluated. Distinguish it from "Replicated" (full re-implementation from scratch). The course overview makes these terms non-interchangeable.
- "New data" means evaluating on a different dataset to obtain similar results — make clear that Tiny ImageNet tests SSKD's generality beyond CIFAR-100.
- "New algorithm variant" means evaluating a slightly different variant — here, applying the four SS methods to a different teacher-student architecture pair to test the paper's model-agnostic claim.
- State assumptions and hyperparameter choices up front. For Tiny ImageNet, note any necessary adaptations (image resolution, data augmentation, training schedule) compared to the CIFAR-100 setup.

---

### 4. Experimental Results

This is the heart of the blog (Content = 50%). Structure each experiment as a self-contained module with: **(1) Question → (2) Setup → (3) Results table/figure → (4) Analysis.**

---

## 4.1 Experiment 1: Table 2 Reproduction — SS Methods Comparison (vgg13 → vgg8, CIFAR-100)

*Owner: Yanzhe · Criterion: Reproduced*

### Question

Table 2 of the SSKD paper claims that student accuracy is positively correlated with the quality of the self-supervision (SS) method used, measured by linear evaluation accuracy on ImageNet, with the ranking Exemplar < Jigsaw < Rotation < Contrastive. Our question is: after re-running the original code and implementing the three missing SS methods, does this ranking still hold?

### Setup

The architecture pair is vgg13 (teacher) → vgg8 (student) on CIFAR-100. All four methods share the same hyperparameters from the paper: τ_kd = τ_T = 4, τ_ss = 0.5, λ1 = 0.1, λ2 = 0.9, λ3 = 2.7, λ4 = 10.0; 240 training epochs with an initial learning rate of 0.05, decayed by a factor of 10 at epochs 150, 180, and 210; batch size 64, SGD with momentum 0.9 and weight decay 5e-4. We use an RTX 3060 (6GB), whereas the original paper used a TITAN-X-Pascal.

For Contrastive, we used the implementation from the original authors' repository (`xuguodong03/SSKD`). Rotation, Jigsaw, and Exemplar were not provided in the original codebase and were implemented by us based on the descriptions in section 6.3. Rotation is a 4-way classification head over rotation angles (0°, ±90°, 180°); Jigsaw is a 24-way classification head over permutations of 2×2 image patches; Exemplar is an instance-classification head with a number of classes equal to the dataset size. In all three cases, knowledge is transferred to the student via the logits of these heads.

As reference points, we also trained the teacher (vgg13) and a vanilla student (vgg8) independently, using standard cross-entropy classification with no distillation.

### Results

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

### Analysis

Overall, Rotation and Contrastive still outperform Exemplar and Jigsaw, and Contrastive achieves the highest accuracy among the four (74.53). This finding broadly consistent with the paper's general direction that higher SS quality leads to higher student accuracy. All four SSKD variants also improve over the vanilla student, which scores 70.73, confirming that distillation with SS signals is beneficial regardless of the SS method used.

However, a local reversal occurs in the ranking: Exemplar (74.57) < Jigsaw (74.85) in the paper, but Exemplar (74.46) > Jigsaw (74.34) in our reproduction. The gap between the two is small in both cases, 0.28 points in the paper and 0.12 points in our reproduction, so this reversal likely falls within the variance of a single training run. It may also reflect implementation differences, since Jigsaw and Exemplar were both newly implemented by us based on textual descriptions rather than the original authors' code.

Contrastive directly reuses the original authors' code, so its result of 74.53 serves as a reference for how trustworthy our overall pipeline is. Compared with the paper's 75.48, this is a gap of about 0.95 points, similar in magnitude to the other three methods, which fall roughly 0.3 to 1.0 points below the paper. This suggests the overall downward shift may stem from pipeline-level factors such as random seed, single-run versus averaged results, or PyTorch version differences, rather than being specific to the newly implemented SS methods.

The reproduced Teacher Acc. of 74.49 is close to the paper's 75.38, a gap of about 0.89 points within a reasonable range, suggesting the teacher training pipeline itself is not a major source of discrepancy.

---

### 4.2 Experiment 2: Loss Component Ablation (vgg13 → vgg8, CIFAR-100)

*Owner: Yanzhe · Criterion: Reproduced (extension)*

**Question:** How do the individual loss components (L_T and L_ss) contribute to the student's final accuracy? Is the improvement from SSKD driven primarily by the transformed-data KD loss, the SS matching loss, or their combination?

**Setup:** Compare student accuracy under different loss configurations on vgg13→vgg8. At minimum: (1) standard KD only (λ3=0, λ4=0), (2) KD + L_T (λ4=0), (3) full SSKD (KD + L_T + L_ss). This mirrors the paper's own ablation in Fig. 3(b). State which SS method is used (contrastive, unless multiple are tested).

**Results:** Bar chart or table with accuracy per loss configuration. If this ablation is run on multiple SS methods, present as a grouped bar chart.

**Analysis:** Does L_ss provide consistent additional gains over L_T alone, as the paper claims? Discuss whether the magnitude of the L_ss contribution matches the paper's reported improvement or differs.

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

## 4.5 Additional Experiments: Loss Component Modifications

### Method

The SSKD framework achieves strong distillation performance, but its standard KD loss $L_{kd}$ has limitations in how it handles class-level and sample-level knowledge transfer. We therefore propose two complementary modifications, DKD and WSLD, to address these limitations.

#### [Decoupled Knowledge Distillation (DKD)](https://arxiv.org/abs/2203.08679)

Standard KD couples the target class and non-target class distributions into a single KL divergence term, which may limit the flexibility of knowledge transfer. DKD decouples $L_{kd}$ into two components. TCKD aligns the binary distribution between the target class and all non-target classes:

$$L_{TCKD} = \text{KL}\left(\left[p_t^y,\ 1-p_t^y\right] \,\|\, \left[p_s^y,\ 1-p_s^y\right]\right)$$

where $p_t^y$ and $p_s^y$ are the teacher's and student's softmax probabilities on the target class $y$. 

NCKD aligns the distribution over non-target classes only:

$$L_{NCKD} = \text{KL}\left(\hat{p}_t^{\neg y} \,\|\, \hat{p}_s^{\neg y}\right)$$

where $\hat{p}^{\neg y}$ denotes the re-normalized distribution over non-target classes. The combined DKD loss replaces $L_{kd}$:

$$L_{kd}^{DKD} = \tau^2 \left(\alpha \cdot L_{TCKD} + \beta \cdot L_{NCKD}\right)$$

with $\alpha = 1.0$ and $\beta = 8.0$ in our experiments.

#### [Weighted Soft Labels Distillation (WSLD)](https://arxiv.org/abs/2102.00650)

Standard KD treats all samples equally, but samples that are already easy for the student carry less informative gradient signal. WSLD assigns a per-sample adaptive weight to the KD loss based on the ratio of student and teacher cross-entropy losses at temperature $\tau = 1$:

$$w_i = 1 - \exp\left(-\frac{L_{ce}^s(x_i)}{L_{ce}^t(x_i)}\right)$$

A higher weight is assigned when the student's loss is large relative to the teacher's. The weighted KD loss replaces $L_{kd}$:

$$L_{kd}^{WSLD} = \tau^2 \cdot \frac{1}{N}\sum_{i=1}^{N} w_i \cdot \text{KL}\left(p_s(x_i;\tau) \,\|\, p_t(x_i;\tau)\right)$$

### Setup

All three modifications are evaluated on vgg13→vgg8, CIFAR-100, using the same hyperparameters and training protocol as Section 4.1. The original SSKD with Contrastive is used as the baseline in all comparisons.

### Results

| Method | Student Acc. (%) |
|---|---|
| SSKD (Contrastive, baseline) | 74.53 |
| SSKD + DKD | 74.68 |
| SSKD + WSLD | 74.61 |

### Analysis

DKD and WSLD both outperform the baseline, achieving accuracies of 74.68% and 74.61%, respectively. This result shows that both class-level distillation and sample-level re-weighting improve performance. However, the gains are relatively small, suggesting that the SS auxiliary signal already recovers much of the information that standard KD fails to transfer.

DKD performs slightly better than WSLD, with a margin of 0.07 percentage points. This indicates that class-level decoupling is more effective than sample-level re-weighting in our setting. A possible reason is that CIFAR-100 contains many fine-grained classes. By separating target-class and non-target-class distillation signals, DKD can better exploit the teacher's dark knowledge.

Since DKD and WSLD address different aspects of knowledge distillation, they may provide complementary benefits. Exploring a combination of the two methods is an interesting direction for future work.

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

> **Yanzhe** reproduced Table 2 (four SS methods on vgg13→vgg8, CIFAR-100) and explored the effect of loss components (L_T, L_ss) on student accuracy (Criterion: Reproduced). **Chenyu** trained the vgg13→vgg8 pair on Tiny ImageNet, adapting the pipeline for the new dataset (Criterion: New data). **Shanghong** ran all four SS methods on the resnet56→resnet20 pair on CIFAR-100 (Criterion: New algorithm variant). All members contributed to writing the blog post.
> 

---

### 7. References

Standard academic references. Cite at minimum: SSKD (Xu et al., ECCV 2020), Hinton et al. (2015) for KD, SimCLR (Chen et al., 2020) for contrastive learning, and the original papers for each SS method (Exemplar: Dosovitskiy et al., Jigsaw: Noroozi & Favaro, Rotation: Gidaris et al.). Also cite CRD (Tian et al.) since the SSKD codebase is based on it.