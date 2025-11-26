# Experiment Summary Report

Generated: 2025-11-26 17:20:16

## Progress: 5/6 (83%)

### Teacher Experiments

| Experiment             | Type    | Completed   | Timestamp           |   ROC-AUC |   PR-AUC |       F1 |      ECE |   Spec@95%Sens |
|:-----------------------|:--------|:------------|:--------------------|----------:|---------:|---------:|---------:|---------------:|
| teacher_resnet34_focal | teacher | ✓           | 2025-11-26 12:45:23 |  0.899064 | 0.563196 | 0.490975 | 0.203014 |              0 |

### Student Experiments

| Experiment            | Type    | Completed   | Timestamp           |   ROC-AUC |   PR-AUC |       F1 |      ECE |   Spec@95%Sens |
|:----------------------|:--------|:------------|:--------------------|----------:|---------:|---------:|---------:|---------------:|
| student_T1.0_alpha0.5 | student | ✓           | 2025-11-26 14:50:37 |  0.917911 | 0.665378 | 0.489796 | 0.222017 |              0 |
| student_T1.0_alpha0.9 | student | ✓           | 2025-11-26 15:20:44 |  0.906578 | 0.606969 | 0.521401 | 0.197553 |              0 |
| student_T2.0_alpha0.5 | student | ✓           | 2025-11-26 13:31:55 |  0.920608 | 0.666288 | 0.539499 | 0.197179 |              0 |
| student_T2.0_alpha0.9 | student | ✓           | 2025-11-26 15:42:51 |  0.90121  | 0.592385 | 0.51341  | 0.206374 |              0 |

### Quantization Experiments

| Experiment        | Type         | Completed   | Timestamp   | ROC-AUC   | PR-AUC   | F1   | ECE   | Spec@95%Sens   |
|:------------------|:-------------|:------------|:------------|:----------|:---------|:-----|:------|:---------------|
| quantized_dynamic | quantization | ✗           |             |           |          |      |       |                |

## Missing Experiments

- [ ] quantized_dynamic (quantization)

## Best Results

**Best Student Configuration**: student_T2.0_alpha0.5
- ROC-AUC: 0.9206
- PR-AUC: 0.6663
- ECE: 0.1972
