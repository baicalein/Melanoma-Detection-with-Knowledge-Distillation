# Experiment Summary Report

Generated: 2025-12-01 16:33:31

## Progress: 17/18 (94%)

### Teacher Experiments

| Experiment                    | Type    | Completed   | Timestamp           |   ROC-AUC |   PR-AUC |       F1 |       ECE |   Spec@95%Sens |
|:------------------------------|:--------|:------------|:--------------------|----------:|---------:|---------:|----------:|---------------:|
| teacher_resnet18_focal        | teacher | ✓           | 2025-11-26 18:09:03 |  0.901666 | 0.563025 | 0.482639 | 0.193716  |              0 |
| teacher_resnet34_focal        | teacher | ✓           | 2025-11-26 12:45:23 |  0.899064 | 0.563196 | 0.490975 | 0.203014  |              0 |
| teacher_resnet50_focal        | teacher | ✓           | 2025-12-01 12:33:11 |  0.865837 | 0.381194 | 0.259091 | 0.559234  |              0 |
| teacher_resnet101_focal       | teacher | ✓           | 2025-11-26 20:09:54 |  0.887025 | 0.473179 | 0.462295 | 0.219137  |              0 |
| teacher_resnet152_focal       | teacher | ✓           | 2025-11-27 11:52:03 |  0.90046  | 0.547766 | 0.481728 | 0.217809  |              0 |
| teacher_efficientnet_b0_focal | teacher | ✓           | 2025-11-26 22:08:35 |  0.9096   | 0.633915 | 0.54185  | 0.147887  |              0 |
| teacher_efficientnet_b1_focal | teacher | ✓           | 2025-11-26 22:37:03 |  0.924202 | 0.712442 | 0.617225 | 0.157777  |              0 |
| teacher_efficientnet_b2_focal | teacher | ✓           | 2025-11-26 23:26:11 |  0.904456 | 0.661197 | 0.594005 | 0.0538058 |              0 |
| teacher_efficientnet_b3_focal | teacher | ✓           | 2025-11-27 00:23:23 |  0.911451 | 0.633888 | 0.565934 | 0.0620439 |              0 |
| teacher_efficientnet_b4_focal | teacher | ✓           | 2025-11-27 01:40:59 |  0.904692 | 0.632535 | 0.587013 | 0.0959423 |              0 |
| teacher_efficientnet_b5_focal | teacher | ✓           | 2025-11-27 02:14:01 |  0.897981 | 0.557267 | 0.513966 | 0.229338  |              0 |
| teacher_efficientnet_b6_focal | teacher | ✓           | 2025-11-27 03:02:51 |  0.894399 | 0.59859  | 0.540404 | 0.141819  |              0 |
| teacher_efficientnet_b7_focal | teacher | ✓           | 2025-11-27 03:56:46 |  0.917985 | 0.673086 | 0.622449 | 0.176036  |              0 |

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
