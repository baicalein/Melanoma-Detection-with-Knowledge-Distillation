# Experiment Summary Report

Generated: 2025-12-01 15:54:14

## Progress: 9/10 (90%)

### Teacher Experiments

| Experiment              | Type    | Completed   | Timestamp           |   ROC-AUC |   PR-AUC |       F1 |      ECE |   Spec@95%Sens |
|:------------------------|:--------|:------------|:--------------------|----------:|---------:|---------:|---------:|---------------:|
| teacher_resnet18_focal  | teacher | ✓           | 2025-11-26 18:09:03 |  0.901666 | 0.563025 | 0.482639 | 0.193716 |              0 |
| teacher_resnet34_focal  | teacher | ✓           | 2025-11-26 12:45:23 |  0.899064 | 0.563196 | 0.490975 | 0.203014 |              0 |
| teacher_resnet50_focal  | teacher | ✓           | 2025-12-01 12:33:11 |  0.865837 | 0.381194 | 0.259091 | 0.559234 |              0 |
| teacher_resnet101_focal | teacher | ✓           | 2025-11-26 20:09:54 |  0.887025 | 0.473179 | 0.462295 | 0.219137 |              0 |
| teacher_resnet152_focal | teacher | ✓           | 2025-11-27 11:52:03 |  0.90046  | 0.547766 | 0.481728 | 0.217809 |              0 |

### Student Experiments

| Experiment            | Type    | Completed   | Timestamp           |   ROC-AUC |   PR-AUC |       F1 |       ECE |   Spec@95%Sens |
|:----------------------|:--------|:------------|:--------------------|----------:|---------:|---------:|----------:|---------------:|
| student_T1.0_alpha0.5 | student | ✓           | 2025-12-01 14:59:41 |  0.921353 | 0.663424 | 0.641791 | 0.0721372 |              0 |
| student_T1.0_alpha0.9 | student | ✓           | 2025-12-01 15:15:13 |  0.915791 | 0.622583 | 0.607143 | 0.0637395 |              0 |
| student_T2.0_alpha0.5 | student | ✓           | 2025-12-01 15:29:40 |  0.919962 | 0.645138 | 0.560461 | 0.134027  |              0 |
| student_T2.0_alpha0.9 | student | ✓           | 2025-12-01 15:44:25 |  0.918687 | 0.609702 | 0.555556 | 0.129928  |              0 |

### Quantization Experiments

| Experiment        | Type         | Completed   | Timestamp   | ROC-AUC   | PR-AUC   | F1   | ECE   | Spec@95%Sens   |
|:------------------|:-------------|:------------|:------------|:----------|:---------|:-----|:------|:---------------|
| quantized_dynamic | quantization | ✗           |             |           |          |      |       |                |

## Missing Experiments

- [ ] quantized_dynamic (quantization)

## Best Results

**Best Student Configuration**: student_T1.0_alpha0.5
- ROC-AUC: 0.9214
- PR-AUC: 0.6634
- ECE: 0.0721
