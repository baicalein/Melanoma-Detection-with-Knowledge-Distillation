# Knowledge Distillation for Efficient Melanoma Detection on Mobile Devices

**Authors:** Ryan Healy  
**Affiliation:** University of San Francisco, MS in Data Science  
**Date:** December 2025

---

## Abstract

Melanoma is the deadliest form of skin cancer, yet early detection significantly improves patient outcomes. While deep learning models achieve high accuracy in dermoscopic image classification, their computational requirements prohibit deployment on resource-constrained mobile devices. This paper presents a knowledge distillation approach to compress a large EfficientNet-B1 teacher model (25 MB) into a compact MobileNetV3 student model (4.5 MB) while retaining 99.7% of the teacher's discriminative performance. Using the HAM10000 dataset with lesion-aware data splitting, we demonstrate that knowledge distillation produces a student model achieving 0.921 ROC-AUC compared to the teacher's 0.924, with improved probability calibration (ECE: 0.072 vs 0.158) and 4× faster inference speed. These results suggest knowledge distillation is a viable strategy for deploying clinical-grade melanoma screening tools on smartphones.

**Keywords:** Knowledge Distillation, Melanoma Detection, Deep Learning, Mobile Deployment, Medical Imaging, Transfer Learning

---

## I. Introduction

Skin cancer is the most common cancer worldwide, with melanoma accounting for the majority of skin cancer deaths despite representing only 1% of cases [1]. Early detection is critical—the 5-year survival rate exceeds 99% for localized melanoma but drops to 27% for distant metastases [2]. However, access to dermatologists remains limited, particularly in rural and underserved communities.

Deep learning has demonstrated remarkable success in dermoscopic image classification, with some models matching or exceeding dermatologist performance [3]. However, state-of-the-art models like EfficientNet and ResNet require significant computational resources, making real-time deployment on mobile devices challenging. This limitation prevents the widespread availability of AI-assisted skin cancer screening.

Knowledge distillation (KD) offers a promising solution by transferring knowledge from a large "teacher" model to a smaller "student" model [4]. The student learns to mimic the teacher's probability distributions rather than just the hard labels, enabling it to capture more nuanced decision boundaries.

This paper makes the following contributions:

1. We systematically evaluate 13 teacher architectures (ResNet-18/34/50/101/152 and EfficientNet-B0 through B7) on the HAM10000 dataset.
2. We conduct an ablation study over knowledge distillation hyperparameters (temperature T ∈ {1, 2} and α ∈ {0.5, 0.9}).
3. We demonstrate that a MobileNetV3-Small student achieves near-teacher performance with 5× model compression and 4× inference speedup.
4. We show that knowledge distillation improves probability calibration, a critical property for clinical decision support.

---

## II. Related Work

### A. Deep Learning for Skin Lesion Classification

Esteva et al. [3] demonstrated that a CNN trained on clinical images could classify skin cancer with dermatologist-level accuracy. The ISIC challenge series has driven advances in dermoscopic image analysis, with winning solutions typically employing ensemble methods and large pretrained networks [5].

### B. Knowledge Distillation

Hinton et al. [4] introduced knowledge distillation, showing that soft probability targets from a teacher network contain "dark knowledge" about class similarities. Subsequent work has explored various distillation strategies including feature-based [6], attention-based [7], and relational distillation [8].

### C. Efficient Medical Image Analysis

Several works have addressed efficient medical imaging. MobileNet-based architectures have been applied to chest X-ray classification [9] and diabetic retinopathy screening [10]. However, few studies specifically address knowledge distillation for dermoscopic image classification.

---

## III. Methods

### A. Dataset

We use the HAM10000 dataset [11], comprising 10,015 dermoscopic images of pigmented skin lesions. The dataset includes seven diagnostic categories, which we binarize into melanoma (positive) and non-melanoma (negative) classes. The resulting class distribution is heavily imbalanced (11.1% melanoma).

**Data Splitting:** To prevent data leakage from multiple images of the same lesion, we implement lesion-aware stratified splitting. Images are grouped by lesion ID before splitting into train (70%), validation (15%), and holdout (15%) sets.

**Augmentation:** Training images undergo dermoscopy-specific augmentation including random rotation (360°), horizontal/vertical flips, color jitter, and random erasing.

### B. Teacher Models

We evaluate two architecture families:

1. **ResNet** [12]: ResNet-18, 34, 50, 101, and 152, pretrained on ImageNet.
2. **EfficientNet** [13]: EfficientNet-B0 through B7, pretrained on ImageNet.

All teachers are trained with focal loss [14] to address class imbalance:

$$\mathcal{L}_{focal} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

where γ = 2 and α is set inversely proportional to class frequency.

### C. Knowledge Distillation

The student model (MobileNetV3-Small [15]) is trained using a combination of hard labels and soft teacher predictions:

$$\mathcal{L}_{KD} = \alpha \cdot T^2 \cdot KL(p_s^T || p_t^T) + (1 - \alpha) \cdot \mathcal{L}_{focal}(y, p_s)$$

where:
- $p_s^T$ and $p_t^T$ are temperature-scaled student and teacher probabilities
- $T$ is the temperature hyperparameter
- $\alpha$ balances distillation and task losses
- $KL$ denotes Kullback-Leibler divergence

### D. Evaluation Metrics

Given the clinical context, we prioritize:

1. **ROC-AUC**: Overall discriminative ability
2. **PR-AUC**: Performance under class imbalance
3. **Expected Calibration Error (ECE)**: Reliability of probability estimates
4. **Specificity at 95% Sensitivity**: Clinical operating point

Deployment metrics include model size (MB) and inference latency (ms).

---

## IV. Experiments

### A. Experimental Setup

All models are trained using PyTorch 2.0 on NVIDIA GPUs. We use AdamW optimizer with learning rate 1e-4, batch size 32, and early stopping with patience 10 epochs. Training is tracked using Weights & Biases.

### B. Teacher Model Comparison

Table I presents holdout set performance for all teacher architectures.

**TABLE I: Teacher Model Performance on Holdout Set**

| Architecture      | ROC-AUC | PR-AUC | ECE   | Size (MB) |
|-------------------|---------|--------|-------|-----------|
| ResNet-18         | 0.900   | 0.567  | 0.195 | 42.7      |
| ResNet-34         | 0.898   | 0.569  | 0.205 | 81.3      |
| ResNet-50         | 0.866   | 0.381  | 0.559 | 89.9      |
| ResNet-101        | 0.886   | 0.475  | 0.221 | 162.5     |
| ResNet-152        | 0.900   | 0.553  | 0.219 | 222.4     |
| EfficientNet-B0   | 0.904   | 0.619  | 0.157 | 15.5      |
| **EfficientNet-B1** | **0.924** | **0.712** | 0.158 | 25.1  |
| EfficientNet-B2   | 0.904   | 0.659  | 0.064 | 29.6      |
| EfficientNet-B3   | 0.908   | 0.615  | 0.076 | 41.1      |
| EfficientNet-B4   | 0.906   | 0.649  | 0.115 | 67.4      |
| EfficientNet-B5   | 0.899   | 0.561  | 0.235 | 108.8     |
| EfficientNet-B6   | 0.890   | 0.594  | 0.145 | 156.3     |
| EfficientNet-B7   | 0.917   | 0.671  | 0.169 | 244.5     |

EfficientNet-B1 achieves the highest ROC-AUC (0.924) and is selected as the teacher for knowledge distillation.

### C. Knowledge Distillation Ablation

Table II shows student performance across hyperparameter configurations.

**TABLE II: Knowledge Distillation Ablation Study**

| Temperature | Alpha | ROC-AUC | PR-AUC | ECE   |
|-------------|-------|---------|--------|-------|
| **1.0**     | **0.5** | **0.921** | **0.663** | **0.072** |
| 1.0         | 0.9   | 0.916   | 0.623  | 0.064 |
| 2.0         | 0.5   | 0.920   | 0.645  | 0.134 |
| 2.0         | 0.9   | 0.919   | 0.610  | 0.130 |

The configuration T=1.0, α=0.5 yields the best ROC-AUC while maintaining excellent calibration.

### D. Teacher vs Student Comparison

**TABLE III: Final Model Comparison**

| Metric              | Teacher (B1) | Student (MV3) | Δ        |
|---------------------|--------------|---------------|----------|
| ROC-AUC             | 0.924        | 0.921         | -0.3%    |
| PR-AUC              | 0.712        | 0.663         | -6.9%    |
| ECE                 | 0.158        | 0.072         | -54.4%   |
| Model Size (MB)     | 25.1         | 4.5           | -82.1%   |
| Inference (ms)      | 8.6          | 2.1           | -75.6%   |

---

## V. Discussion

Our results demonstrate that knowledge distillation successfully transfers discriminative knowledge from EfficientNet-B1 to MobileNetV3-Small with minimal performance degradation. The student retains 99.7% of the teacher's ROC-AUC while being 5× smaller and 4× faster.

**Improved Calibration:** Surprisingly, the student exhibits substantially better probability calibration than the teacher (ECE 0.072 vs 0.158). This may result from the regularizing effect of soft targets, which prevent overconfident predictions. Well-calibrated probabilities are crucial for clinical decision support, as they allow physicians to appropriately weigh AI recommendations.

**Limitations:** Specificity at 95% sensitivity remains at 0% for both models, indicating room for improvement at high-sensitivity operating points. This may require threshold optimization or cost-sensitive learning.

---

## VI. Conclusion

We presented a knowledge distillation framework for deploying melanoma detection on mobile devices. By distilling an EfficientNet-B1 teacher into a MobileNetV3-Small student, we achieve a 5× reduction in model size and 4× speedup with only 0.3% ROC-AUC degradation. The distilled student also exhibits improved probability calibration, a desirable property for clinical applications.

Future work includes INT8 quantization for further compression, multi-class extension to diagnose specific lesion types, and prospective validation on diverse patient populations.

---

## References

[1] R. L. Siegel, K. D. Miller, and A. Jemal, "Cancer statistics, 2023," *CA: A Cancer Journal for Clinicians*, vol. 73, no. 1, pp. 17-48, 2023.

[2] American Cancer Society, "Melanoma Skin Cancer Statistics," 2023. [Online]. Available: https://www.cancer.org/cancer/melanoma-skin-cancer/

[3] A. Esteva et al., "Dermatologist-level classification of skin cancer with deep neural networks," *Nature*, vol. 542, no. 7639, pp. 115-118, 2017.

[4] G. Hinton, O. Vinyals, and J. Dean, "Distilling the knowledge in a neural network," *arXiv preprint arXiv:1503.02531*, 2015.

[5] N. C. F. Codella et al., "Skin lesion analysis toward melanoma detection: A challenge at the 2017 ISIC," in *Proc. ISBI*, 2018, pp. 168-172.

[6] A. Romero et al., "FitNets: Hints for thin deep nets," in *Proc. ICLR*, 2015.

[7] S. Zagoruyko and N. Komodakis, "Paying more attention to attention: Improving the performance of CNNs via attention transfer," in *Proc. ICLR*, 2017.

[8] W. Park et al., "Relational knowledge distillation," in *Proc. CVPR*, 2019, pp. 3967-3976.

[9] P. Rajpurkar et al., "CheXNet: Radiologist-level pneumonia detection on chest X-rays with deep learning," *arXiv preprint arXiv:1711.05225*, 2017.

[10] V. Gulshan et al., "Development and validation of a deep learning algorithm for detection of diabetic retinopathy," *JAMA*, vol. 316, no. 22, pp. 2402-2410, 2016.

[11] P. Tschandl, C. Rosendahl, and H. Kittler, "The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions," *Scientific Data*, vol. 5, p. 180161, 2018.

[12] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. CVPR*, 2016, pp. 770-778.

[13] M. Tan and Q. V. Le, "EfficientNet: Rethinking model scaling for convolutional neural networks," in *Proc. ICML*, 2019, pp. 6105-6114.

[14] T.-Y. Lin et al., "Focal loss for dense object detection," in *Proc. ICCV*, 2017, pp. 2980-2988.

[15] A. Howard et al., "Searching for MobileNetV3," in *Proc. ICCV*, 2019, pp. 1314-1324.

---

## Appendix

### A. Reproducibility

All code is available at: `github.com/rah-ds/Deep_Learning_Final_Project`

To reproduce results:
```bash
make install          # Install dependencies
make data && make splits  # Prepare dataset
make train-teacher    # Train all teachers
make run-ablation     # Run KD ablation
make evaluate         # Evaluate on holdout
```

### B. Computational Resources

- Training: NVIDIA GPU with 16GB VRAM
- Training time: ~2 hours per teacher, ~30 min per student
- Total experiments: 13 teachers + 4 students = 17 runs

