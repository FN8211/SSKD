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

### 4.1 Experiment 1: Table 2 Reproduction — SS Methods Comparison (vgg13 → vgg8, CIFAR-100)

*Owner: Yanzhe · Criterion: Reproduced*

**Question:** Does student accuracy correlate positively with self-supervision method quality, as the paper claims in Table 2?

**Setup:** Four SS methods (Exemplar, Jigsaw, Rotation, Contrastive) on vgg13→vgg8, CIFAR-100. State exact hyperparameters used (from §6.4 of the paper: τ_kd=4, τ_ss=0.5, λ1=0.1, λ2=0.9, λ3=2.7, λ4=10.0, 240 epochs, SGD with momentum 0.9, weight decay 5e-4, batch size 64, LR schedule). State what was kept identical vs. what differed (hardware, random seed, PyTorch version). Describe each SS method's implementation briefly — for Contrastive, the original code was used; for Rotation, Jigsaw, and Exemplar, note whether they were newly implemented or adapted.

**Results:**

- **Main table:** Reproduce Table 2 format — columns for SS method, SS performance (linear evaluation on ImageNet, from the paper), paper-reported student accuracy, and reproduced student accuracy. Include teacher accuracy and vanilla student accuracy as reference points.
- Training curves (loss and accuracy over epochs) as a figure for at least the contrastive method.

**Analysis:**

- Do your numbers show the same positive correlation (Exemplar < Jigsaw < Rotation < Contrastive)? If the *ranking* is preserved even when absolute numbers differ, that's a meaningful positive finding.
- If absolute numbers differ from the paper, frame around whether **relative trends are preserved**. Discuss possible sources of discrepancy (random seed, hardware, undocumented preprocessing). This is where you show honesty and scientific maturity — the rubric says "results are inconsistent and not motivated" is a penalty, so motivate any gap.
- If ranking is disrupted, discuss what might explain it (implementation differences, hyperparameter sensitivity, training variance).

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

### 4.5 (Optional) Additional Experiments

If you ran any further experiments (hyperparameter sensitivity, effect of the selective transfer k parameter, wrn40-2→wrn16-2 as an additional pair), include them here as additional experiment modules. Each one follows the same Question → Setup → Results → Analysis structure.

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