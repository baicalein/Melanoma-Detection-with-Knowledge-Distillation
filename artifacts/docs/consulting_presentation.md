# Efficient AI for Medical Imaging at the Edge

## Knowledge Distillation for Scalable Melanoma Detection

**Prepared by**: Ryan Healy & Angie Yoon  
**Date**: December 2025  
**Engagement**: Deep Learning Final Project – University of Virginia

---

# Slide 1: Executive Summary

## Situation → Complication → Resolution

| | |
|---|---|
| **Situation** | Melanoma kills 8,000+ Americans annually; early detection improves 5-year survival from 30% to 99% |
| **Complication** | State-of-the-art AI models are too large (25-250 MB) and slow for mobile/edge deployment where screening is most needed |
| **Resolution** | Knowledge distillation compresses a 25 MB teacher to a 9 MB student with **minimal accuracy loss** and **3x faster inference** |

### Key Results

| Metric | Teacher | Student | Δ |
|--------|---------|---------|---|
| ROC-AUC | 0.924 | **0.921** | -0.3% |
| Model Size | 25.1 MB | 9.1 MB | **-64%** |
| Latency | 8.6 ms | 3.0 ms | **-65%** |
| Calibration (ECE) | 0.158 | 0.072 | **-54%** |

---

# Understanding the Metrics

## What Do These Numbers Mean?

### Classification Metrics

| Metric | What It Measures | Why It Matters | Good Value |
|--------|------------------|----------------|------------|
| **ROC-AUC** | How well the model separates melanoma from benign lesions across all thresholds | Higher = better at ranking risky lesions above safe ones | >0.90 |
| **PR-AUC** | Performance on the rare class (melanoma = 11% of data) | Critical for imbalanced datasets; ignores easy "benign" predictions | >0.50 |
| **F1 Score** | Balance between precision (avoiding false alarms) and recall (catching melanomas) | Single number summarizing the precision/recall tradeoff | >0.50 |
| **ECE** | How trustworthy the probability outputs are (e.g., "80% confident" should be right 80% of the time) | Lower = more reliable confidence scores for clinical decisions | <0.10 |

### Deployment Metrics

| Metric | What It Measures | Why It Matters | Target |
|--------|------------------|----------------|--------|
| **Model Size** | Storage required on device | Must fit in phone memory alongside other apps | <50 MB |
| **Latency** | Time to process one image | Must feel instant to user (<100ms perceived) | <10 ms |

### Visual Explanation: ROC-AUC

```
                    ROC Curve Interpretation
                    
   True     1.0 ┤        ●───────────────────  Perfect (AUC=1.0)
   Positive     │       ╱                    
   Rate    0.8 ┤     ╱   ← Our Model (AUC=0.92)
   (Recall)     │   ╱                         
           0.5 ┤  ╱  ╱── Random Guess (AUC=0.5)
               │ ╱ ╱                          
           0.0 ┼╱─────────────────────────────►
               0.0        0.5            1.0   
                    False Positive Rate        
                                               
   Higher AUC = Curve closer to top-left corner
```

### Visual Explanation: Calibration (ECE)

```
   ┌─────────────────────────────────────────────────────┐
   │                  CALIBRATION EXAMPLE                 │
   ├─────────────────────────────────────────────────────┤
   │                                                      │
   │  Model says "80% confident this is melanoma"        │
   │                                                      │
   │  Well-calibrated (low ECE):                         │
   │    → 80 out of 100 such predictions ARE melanoma    │
   │                                                      │
   │  Poorly-calibrated (high ECE):                      │
   │    → Only 50 out of 100 are actually melanoma       │
   │    → Model is overconfident!                        │
   │                                                      │
   │  Why it matters:                                     │
   │    Doctors need to trust the confidence scores      │
   │    to make informed referral decisions              │
   │                                                      │
   └─────────────────────────────────────────────────────┘
```

---

# Slide 2: The Business Case for Edge AI in Healthcare

## Market Context

```
┌─────────────────────────────────────────────────────────────────┐
│                      MARKET DRIVERS                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   📱 3.5B smartphone users globally                              │
│   🏥 Dermatologist shortage (1 per 30,000 in rural US)          │
│   💰 $1.2B mobile health imaging market (12% CAGR)              │
│   ⚖️  FDA cleared 500+ AI medical devices (2024)                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### The Problem We Solve

| Challenge | Current State | Our Solution |
|-----------|---------------|--------------|
| **Access** | Specialists concentrated in urban areas | AI screening on any smartphone |
| **Cost** | $150+ for dermatologist visit | Free/low-cost app-based triage |
| **Latency** | Cloud AI requires connectivity | On-device inference in <10ms |
| **Privacy** | Images sent to cloud servers | Data never leaves device |

---

# Slide 3: Technical Approach

## Knowledge Distillation Framework

```
┌──────────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│   PHASE 1: Teacher Training                                       │
│   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐      │
│   │  HAM10000   │ ───► │ EfficientNet│ ───► │  Teacher    │      │
│   │  Dataset    │      │ B0-B7       │      │  Checkpoint │      │
│   └─────────────┘      └─────────────┘      └─────────────┘      │
│                              │                                    │
│   PHASE 2: Student Distillation                                   │
│                              ▼                                    │
│   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐      │
│   │  Teacher    │ ───► │ MobileNetV3 │ ───► │  Student    │      │
│   │  Soft Labels│      │ Small       │      │  Checkpoint │      │
│   └─────────────┘      └─────────────┘      └─────────────┘      │
│                                                                   │
│   Loss = α × KL(student, teacher/T) + (1-α) × CE(student, label) │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Teacher Architecture | EfficientNet-B1 | Best accuracy/size tradeoff |
| Student Architecture | MobileNetV3-Small | Optimized for mobile inference |
| Temperature | T = 1.0 | Lower temp preserves discrimination |
| Alpha | α = 0.5 | Equal weight to hard/soft labels |
| Loss Function | Focal Loss | Handles 11% melanoma class imbalance |

---

# Slide 4: Data Strategy

## Rigorous Experimental Design

### Dataset: HAM10000

| Attribute | Value |
|-----------|-------|
| Total Images | 10,015 dermoscopy images |
| Classes | Binary (Melanoma vs. Benign) |
| Imbalance | 11% positive (melanoma) |
| Source | Vienna Medical University |

### Split Strategy: Lesion-Aware

```
┌─────────────────────────────────────────────────────────────────┐
│   ⚠️  CRITICAL: Images from same lesion NEVER cross splits     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   HAM10000 (10,015 images, 7,470 unique lesions)                │
│   │                                                              │
│   ├──► Train:    70% │ 7,010 images │ Hyperparameter tuning     │
│   ├──► Validate: 15% │ 1,502 images │ Early stopping, model selection│
│   └──► Holdout:  15% │ 1,503 images │ Final evaluation ONLY     │
│                                                                  │
│   ✓ Stratified by target (melanoma prevalence preserved)       │
│   ✓ Grouped by lesion_id (no data leakage)                     │
│   ✓ Reproducible (seed = 42)                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

# Slide 5: Model Comparison

## Comprehensive Architecture Benchmarking

### Teacher Models Evaluated (13 architectures)

| Family | Models | Best Performer | Holdout ROC-AUC |
|--------|--------|----------------|-----------------|
| **ResNet** | 18, 34, 50, 101, 152 | ResNet-18 | 0.902 |
| **EfficientNet** | B0, B1, B2, B3, B4, B5, B6, B7 | **B1** | **0.924** |

### Key Insight: Bigger ≠ Better

```
ROC-AUC vs Model Size (Holdout Set)
                                                    
  0.93 ┤                    ●B1                     
       │              ●B7                           
  0.91 ┤    ●B0  ●B3    ●B4                        
       │        ●B2  ●R18  ●R152                   
  0.90 ┤   ●R34    ●B5  ●B6                        
       │           ●R101                            
  0.88 ┤                                            
       │        ●R50                                
  0.86 ┼────────────────────────────────────────►  
       0    50   100   150   200   250  Size (MB)  
```

**Finding**: EfficientNet-B1 (25 MB) outperforms ResNet-152 (222 MB) by 2.4 points

---

# Slide 6: Knowledge Distillation Results

## Ablation Study: Temperature × Alpha

| Configuration | ROC-AUC | PR-AUC | ECE | F1 |
|---------------|---------|--------|-----|-----|
| T=1.0, α=0.5 | **0.921** | **0.663** | 0.072 | **0.642** |
| T=1.0, α=0.9 | 0.916 | 0.623 | **0.064** | 0.607 |
| T=2.0, α=0.5 | 0.920 | 0.645 | 0.134 | 0.560 |
| T=2.0, α=0.9 | 0.919 | 0.610 | 0.130 | 0.556 |

### Winner: T=1.0, α=0.5

**Why lower temperature wins:**

- Medical imaging requires **sharp decision boundaries**
- Higher temperature over-smooths predictions
- α=0.5 balances teacher knowledge with ground truth

---

# Slide 7: Deployment Readiness

## Production Metrics Comparison

```
┌────────────────────────────────────────────────────────────────┐
│              DEPLOYMENT READINESS SCORECARD                     │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Metric              Teacher        Student       Target   ✓/✗ │
│  ─────────────────────────────────────────────────────────────  │
│  ROC-AUC             0.924          0.921         >0.90    ✓   │
│  Model Size          25.1 MB        9.1 MB        <15 MB   ✓   │
│  Inference Latency   8.6 ms         3.0 ms        <10 ms   ✓   │
│  ECE (Calibration)   0.158          0.072         <0.10    ✓   │
│  ONNX Export         ✓              ✓             Required ✓   │
│  INT8 Quantization   N/A            Pending       Optional -   │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Student Advantages Over Teacher

| Advantage | Quantified Benefit |
|-----------|-------------------|
| **Smaller** | 64% size reduction (25→9 MB) |
| **Faster** | 65% latency reduction (8.6→3.0 ms) |
| **Better Calibrated** | 54% ECE improvement (0.16→0.07) |
| **Comparable Accuracy** | Only 0.3% ROC-AUC difference |

---

# Slide 8: Baseline Comparison

## Deep Learning vs. Traditional ML

### Sklearn Baselines (Handcrafted Features)

| Model | ROC-AUC | PR-AUC | Train Time |
|-------|---------|--------|------------|
| Random Forest | 0.853 | 0.392 | 1.3s |
| Gradient Boosting | 0.845 | 0.366 | 90s |
| SVM (RBF) | 0.824 | 0.336 | 16s |
| Logistic Regression | 0.797 | 0.289 | 10s |

### Deep Learning Advantage

```
                    ROC-AUC Comparison
                    
  Student (DL)  ████████████████████████████████  0.921
  Teacher (DL)  ███████████████████████████████▌  0.919
  Random Forest ██████████████████████████▌       0.853
  Grad Boost    █████████████████████████▌        0.845
  SVM           ████████████████████████          0.824
  Log Reg       ██████████████████████            0.797
                ├────────────────────────────────►
                0.0                            1.0
```

**Gap**: +6.8 points over best traditional ML (significant at p<0.01)

---

# Slide 9: Limitations & Risks

## Honest Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Single Dataset** | Medium | Validate on ISIC 2019, PH2 before deployment |
| **Binary Only** | Medium | Extend to 7-class (all lesion types) |
| **No Clinical Validation** | High | Partner with dermatology practice for prospective study |
| **Regulatory Path** | High | FDA 510(k) required for diagnostic claims |
| **Calibration at Extremes** | Low | Reliability diagrams show room for improvement |

### What This Model Is NOT

❌ A diagnostic tool (requires FDA approval)  
❌ A replacement for dermatologists  
❌ Validated on diverse skin tones (HAM10000 bias)  

### What This Model IS

✅ A proof-of-concept for efficient edge deployment  
✅ A screening/triage tool to prioritize specialist referrals  
✅ A foundation for further clinical validation  

---

# Slide 10: Recommendations & Next Steps

## Path to Production

### Immediate (0-3 months)

| Action | Owner | Deliverable |
|--------|-------|-------------|
| INT8 Quantization | Engineering | 2.5 MB model, <1ms inference |
| ONNX/CoreML Export | Engineering | iOS/Android ready packages |
| Cross-dataset Validation | Research | ISIC 2019/2020 benchmark results |

### Medium-term (3-6 months)

| Action | Owner | Deliverable |
|--------|-------|-------------|
| Multi-class Extension | Research | 7-class lesion classifier |
| Prospective Clinical Study | Clinical | 500+ patient validation |
| Fairness Audit | Ethics | Skin tone stratified performance |

### Long-term (6-12 months)

| Action | Owner | Deliverable |
|--------|-------|-------------|
| FDA 510(k) Submission | Regulatory | De novo or predicate pathway |
| Mobile App Launch | Product | iOS/Android app with edge inference |
| API Productization | Engineering | Cloud fallback for complex cases |

---

# Appendix A: Methodology Details

## Training Configuration

```python
# Teacher Training
TeacherConfig = {
    "architecture": "efficientnet_b1",
    "pretrained": True,  # ImageNet weights
    "epochs": 50,
    "batch_size": 32,
    "learning_rate": 1e-4,
    "loss": "focal",  # γ=2.0 for class imbalance
    "early_stopping": 10,  # patience
    "optimizer": "AdamW",
    "scheduler": "CosineAnnealingLR",
}

# Student Distillation
StudentConfig = {
    "architecture": "mobilenet_v3_small",
    "teacher_checkpoint": "teacher_efficientnet_b1_focal_best.pth",
    "temperature": 1.0,
    "alpha": 0.5,  # weight for KD loss
    "epochs": 50,
    "learning_rate": 1e-4,
}
```

## Evaluation Metrics

| Metric | Definition | Why It Matters |
|--------|------------|----------------|
| **ROC-AUC** | Area under ROC curve | Threshold-independent discrimination |
| **PR-AUC** | Area under Precision-Recall curve | Performance on minority class |
| **ECE** | Expected Calibration Error | Reliability of probability estimates |
| **F1** | Harmonic mean of precision & recall | Balance between false positives/negatives |

---

# Appendix B: Reproducibility

## Code & Artifacts

```bash
# Clone and setup
git clone https://github.com/rah-ds/Deep_Learning_Final_Project
cd Deep_Learning_Final_Project
make install

# Reproduce all experiments
make train-teacher           # Train all 13 teacher architectures
make train-student           # Train student with 4 KD configurations
make evaluate                # Generate all evaluation metrics
make sklearn-baselines       # Run traditional ML baselines

# Key artifacts
models/checkpoints/          # All trained model weights
artifacts/tbls/              # Evaluation results (CSV/JSON)
artifacts/imgs/              # All figures and plots
notebooks/01_benchmarks.ipynb # Interactive analysis
```

## Compute Requirements

| Resource | Specification |
|----------|---------------|
| GPU | NVIDIA RTX 3090 (24 GB) or A100 |
| Training Time | ~2 hours per teacher, ~1 hour per student |
| Total Compute | ~30 GPU-hours for full reproduction |

---

# Thank You

## Key Takeaways

1. **Knowledge distillation works**: 64% smaller, 65% faster, minimal accuracy loss
2. **Edge-ready**: 9 MB model with 3ms inference fits on any smartphone
3. **Calibration bonus**: Student probabilities are more trustworthy
4. **Strong baseline**: +6.8 points over traditional ML approaches

## Contact

**Ryan Healy** – rah5ff@virginia.edu  
**Angie Yoon** – aey4gf@virginia.edu

**Repository**: [github.com/rah-ds/Deep_Learning_Final_Project](https://github.com/rah-ds/Deep_Learning_Final_Project)
