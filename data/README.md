# Data Directory

This directory contains all data used in the project. Follow the organization structure below to maintain clarity and reproducibility.

## 📁 Directory Structure

```
data/
├── raw/              # Original, immutable data (never modify!)
├── interim/          # Intermediate data during processing
├── processed/        # Final, cleaned data ready for modeling
└── external/         # External reference data or supplementary datasets
```

## 📊 Data Organization Guidelines

### `raw/`
- **Purpose**: Store original data exactly as downloaded or received
- **Rules**:
  - NEVER modify files in this directory
  - Keep original file names for traceability
  - Document source and download date below
  - Add large files to `.gitignore`

### `interim/`
- **Purpose**: Temporary data from intermediate processing steps
- **Use cases**:
  - Partially cleaned data
  - Feature engineering intermediate results
  - Data during multi-step preprocessing
- **Note**: These files can be regenerated from raw data

### `processed/`
- **Purpose**: Final, analysis-ready datasets
- **Characteristics**:
  - Clean, validated data
  - Consistent formatting
  - Ready for model training/evaluation
  - Include train/val/test splits

### `external/`
- **Purpose**: External reference data or supplementary information
- **Examples**:
  - Pre-trained model weights
  - Reference datasets for comparison
  - Annotation files
  - Metadata

## 📝 Dataset Documentation

### Dataset 1: [Dataset Name]

**Source**: [URL or citation]  
**Download Date**: YYYY-MM-DD  
**License**: [License type]  
**Size**: [File size]  
**Format**: [CSV, JSON, images, etc.]

**Description**:
Brief description of the dataset, what it contains, and its purpose in your research.

**Files**:
- `raw/dataset_name.csv` - Original data file
- `processed/dataset_name_clean.csv` - Cleaned version

**Preprocessing Steps**:
1. Step 1 description
2. Step 2 description
3. ...

**Statistics**:
- Number of samples: X
- Number of features: Y
- Class distribution: ...
- Missing values: ...

**Known Issues**:
- Issue 1
- Issue 2

---

### Dataset 2: [Another Dataset]

[Follow the same format as above]

---

## 🔄 Data Processing Pipeline

Document your data processing pipeline here:

```bash
# Example pipeline
python src/deep_learning_final_project/data/preprocessing.py \
    --input data/raw/dataset.csv \
    --output data/processed/dataset_clean.csv
```

## ⚠️ Important Notes

### Version Control

- **Small datasets** (< 10MB): Can be committed to git
- **Medium datasets** (10MB - 100MB): Use Git LFS
- **Large datasets** (> 100MB): Store externally, document download instructions

### Privacy and Ethics

- [ ] Verify data usage complies with license terms
- [ ] Check for personally identifiable information (PII)
- [ ] Document any data anonymization steps
- [ ] Confirm institutional review board (IRB) approval if needed

### Reproducibility Checklist

- [ ] Document exact data source and version
- [ ] Save train/val/test split indices
- [ ] Record random seeds used in sampling
- [ ] Document preprocessing steps in code
- [ ] Include data statistics in results

## 📥 Download Instructions

If datasets must be downloaded separately:

```bash
# Example download commands
wget https://example.com/dataset.zip -O data/raw/dataset.zip
unzip data/raw/dataset.zip -d data/raw/
```

Or provide manual download instructions:

1. Visit [URL]
2. Download file [filename]
3. Place in `data/raw/`

## 🔗 References

List any papers or documentation related to the datasets:

1. Author et al., "Paper Title", Conference Year
2. Dataset documentation: [URL]

---

**Last Updated**: YYYY-MM-DD  
**Updated By**: Your Name
