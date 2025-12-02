# Melanoma Detection with Knowledge Distillation

## A Deep Learning Project for Mobile Deployment

---

# Slide 1: The Problem

## Skin Cancer Detection Matters

- **Melanoma** is the deadliest form of skin cancer
- Early detection saves lives
- Dermatologists are scarce in many areas
- **Goal**: Build an AI that runs on phones to help screen for melanoma

![](../imgs/00_eda/class_distribution.png)

---

# Slide 2: Our Dataset

## HAM10000 - Skin Lesion Images

- **10,015 dermoscopy images** from medical archives
- Binary classification: Melanoma vs Non-Melanoma
- Class imbalance: ~11% melanoma (minority class)
- Split data by **lesion ID** to prevent data leakage

| Split    | Samples | Melanoma % |
|----------|---------|------------|
| Train    | 7,009   | ~11%       |
| Val      | 1,502   | ~11%       |
| Holdout  | 1,513   | ~11%       |

---

# Slide 3: The Challenge

## Big Models vs Mobile Phones

- Best accuracy comes from **large neural networks** (ResNet, EfficientNet)
- Mobile phones have **limited memory and CPU**
- Large models are too slow for real-time use

### Our Solution: Knowledge Distillation

> Train a small "student" model to mimic a large "teacher" model

---

# Slide 4: How Knowledge Distillation Works

## Teacher → Student Transfer

1. **Train a Teacher**: Large EfficientNet-B1 model (best ROC-AUC: 0.92)
2. **Train a Student**: Small MobileNetV3 learns from teacher's predictions
3. **Fine-tune**: Balance between hard labels and soft teacher guidance

### Key Parameters
- **Temperature (T)**: How soft to make predictions (T=1.0 works best)
- **Alpha (α)**: Weight between teacher and true labels (α=0.5 works best)

---

# Slide 5: Results Summary

## Student Nearly Matches Teacher!

| Model              | ROC-AUC | Size (MB) | Speed    |
|--------------------|---------|-----------|----------|
| Teacher (EfficientNet-B1) | 0.924 | 25 MB  | ~9ms     |
| **Student (MobileNetV3)** | **0.921** | **4.5 MB** | **~2ms** |

### Key Findings
- Student is **5x smaller** than teacher
- Student is **4x faster** at inference
- Student has **better calibration** (ECE: 0.07 vs 0.16)
- Only **0.3% drop** in ROC-AUC!

---

# Slide 6: What We Learned

## Lessons from Knowledge Distillation

✅ **Works well** for melanoma detection
- Small student achieves 99.7% of teacher performance

✅ **Better calibrated** predictions
- Student predictions are more reliable probabilities

✅ **Temperature matters**
- T=1.0, α=0.5 gave best results in our ablation study

⚠️ **Class imbalance** remains challenging
- Specificity at 95% sensitivity still needs improvement

---

# Slide 7: Next Steps

## Future Improvements

1. **Quantization**: Compress student to INT8 for even smaller size
2. **Mobile deployment**: Package as iOS/Android app
3. **More data**: Add ISIC 2019/2020 datasets
4. **Multi-class**: Extend beyond binary to detect lesion types

### Project Links
- 📁 Code: `github.com/rah-ds/Deep_Learning_Final_Project`
- 📊 Notebooks: `notebooks/01_benchmarks.ipynb`
- 📈 W&B Dashboard: Training logs and metrics

---

# Thank You!

## Questions?

**Project**: Melanoma Detection with Knowledge Distillation  
**Dataset**: HAM10000 (10,015 images)  
**Best Result**: 0.921 ROC-AUC with 4.5MB student model

