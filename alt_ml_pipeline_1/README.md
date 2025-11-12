# Alt ML Pipeline 1 - Candida Albicans Morphology Detection

> Small-data optimized machine learning pipeline for detecting and classifying Candida albicans cell morphologies in time-lapse microscopy images.

## Overview

This pipeline addresses the challenge of achieving high detection accuracy (F1 > 0.80) with limited training data (205 annotated frames). It implements multiple advanced techniques:

- **Self-supervised learning** - Extract features from unlabeled data
- **Synthetic data generation** - Generate 10,000+ realistic training samples
- **Ensemble methods** - Combine multiple diverse models
- **Biological constraints** - Enforce domain-specific rules
- **Test-time augmentation** - Maximize inference performance

## Problem Statement

**Goal**: Detect, classify, and segment 7 different Candida albicans morphologies from microscopy images

**Challenge**: Only 205 labeled frames available (severe data limitation)

**Target Performance**: F1 > 0.80, mAP50 > 0.75

### 7 Cell Morphology Classes

1. **Planktonic** (35.4%) - Free-floating cells in biofilm matrix
2. **Single dispersed cell** (19.9%) - Individual cells dispersed from biofilm
3. **Hyphae** (17.5%) - Elongated filamentous structures
4. **Clump dispersed cell** (13.1%) - Clusters of dispersed cells
5. **Yeast form** (9.4%) - Round budding cells
6. **Biofilm** (2.5%) - Surface-attached cell communities
7. **Pseudohyphae** (2.1%) - Intermediate elongated forms

## Quick Start

### 1. Environment Setup

```bash
# Navigate to project directory
cd alt_ml_pipeline_1

# Run setup script (creates venv, installs dependencies, creates directories)
bash scripts/setup_environment.sh

# Activate environment
source ~/venvs/alt_pipeline_1/bin/activate
# OR use the helper script:
source ./activate_alt_pipeline.sh
```

### 2. Run Baseline Training

```bash
# Train baseline model on fold 0
python scripts/run_pipeline.py --phase foundation --fold 0 --visualize

# Or directly train baseline
python src/training/train_baseline.py --config configs/config.yaml --fold 0 --visualize
```

### 3. View Results

```bash
# Start MLflow UI to view experiments
mlflow ui --backend-store-uri ~/mlm_outputs/alt_pipeline_1/experiments/mlflow

# Open browser to: http://localhost:5000

# View visualizations
open ~/mlm_outputs/alt_pipeline_1/visualizations/
```

## Project Structure

```
alt_ml_pipeline_1/
├── configs/
│   └── config.yaml                    # Master configuration
├── src/
│   ├── data/
│   │   ├── data_loader.py            # Load YOLO dataset, create CV splits
│   │   ├── synthetic_generator.py    # [TODO] Generate synthetic frames
│   │   └── augmentation_pipeline.py  # [TODO] Advanced augmentations
│   ├── models/
│   │   ├── self_supervised.py        # [TODO] SSL pretraining
│   │   ├── few_shot_learning.py      # [TODO] Few-shot for rare classes
│   │   └── ensemble.py               # [TODO] Ensemble training
│   ├── training/
│   │   ├── train_baseline.py         # ✓ Baseline YOLOv8 training
│   │   ├── train_ssl.py              # [TODO] SSL training
│   │   └── train_ensemble.py         # [TODO] Ensemble training
│   ├── evaluation/
│   │   ├── visualize.py              # ✓ Visualization utilities
│   │   └── test_time_augmentation.py # [TODO] TTA inference
│   └── utils/
│       ├── biological_constraints.py # [TODO] Domain constraints
│       └── logger.py                 # [TODO] Custom logging
├── scripts/
│   ├── setup_environment.sh          # ✓ Environment setup
│   ├── run_pipeline.py               # ✓ Main orchestration
│   └── sagemaker_train.py            # [TODO] SageMaker training
├── notebooks/
│   ├── 01_data_exploration.ipynb     # [TODO] EDA
│   ├── 02_synthetic_validation.ipynb # [TODO] Validate synthetic data
│   └── 03_results_analysis.ipynb     # [TODO] Results analysis
├── requirements.txt                   # ✓ Python dependencies
└── README.md                          # ✓ This file

# Output directories (outside repo, not tracked by git)
~/mlm_outputs/alt_pipeline_1/
├── data/
│   ├── processed/                    # Preprocessed real data
│   ├── synthetic/                    # Synthetic frames
│   └── splits/                       # CV fold data
├── models/
│   ├── ssl_pretrained/               # Self-supervised encoders
│   ├── individual/                   # Single model checkpoints
│   └── ensemble/                     # Ensemble models
├── visualizations/
│   ├── data_quality/                 # Data quality checks
│   ├── training_progress/            # Training curves
│   ├── predictions/                  # Prediction visualizations
│   └── confusion_matrices/           # Confusion matrices
├── results/
│   ├── metrics.json                  # Performance metrics
│   └── predictions/                  # Final predictions
└── experiments/
    └── mlflow/                       # MLflow tracking
```

## Pipeline Phases

### Phase 1: Foundation ✓ IMPLEMENTED

**Status**: Complete
**Duration**: ~2 days

**Implemented**:
- ✓ Data loading from existing YOLO dataset
- ✓ Leave-one-sequence-out cross-validation (5 folds)
- ✓ Baseline YOLOv8 model training
- ✓ Visualization utilities (training curves, predictions, confusion matrices)
- ✓ MLflow experiment tracking

**Usage**:
```bash
python scripts/run_pipeline.py --phase foundation --fold 0 --visualize
```

**Expected Output**:
- Dataset summary with class distribution
- Baseline F1 score (benchmark to beat)
- Visualizations in `~/mlm_outputs/alt_pipeline_1/visualizations/`
- Trained model in `~/mlm_outputs/alt_pipeline_1/models/individual/`

---

### Phase 2: Data Augmentation 🚧 TODO

**Status**: Not yet implemented
**Duration**: ~2 days
**Priority**: HIGH IMPACT

**Will Implement**:
1. **Synthetic Data Generation**
   - Extract cell bank: crop all 7,462 annotated cells
   - Composite 10,000 synthetic frames
   - Microscopy-realistic augmentations (defocus blur, illumination)

2. **Advanced Augmentation Pipeline**
   - Biological augmentations (cell growth, division)
   - Temporal interpolation
   - CellMix augmentation

**Module**: `src/data/synthetic_generator.py`

---

### Phase 3: Self-Supervised Learning 🚧 TODO

**Status**: Not yet implemented
**Duration**: ~2 days
**Priority**: HIGH IMPACT

**Will Implement**:
1. **Temporal Pretext Tasks**
   - Frame ordering prediction
   - Future frame prediction
   - Speed variation detection

2. **Spatial Pretext Tasks**
   - Rotation prediction
   - Contrastive learning
   - Cell inpainting

**Module**: `src/models/self_supervised.py`

---

### Phase 4: Ensemble & Advanced Methods 🚧 TODO

**Status**: Not yet implemented
**Duration**: ~3 days
**Priority**: MEDIUM-HIGH IMPACT

**Will Implement**:
1. Train 10-15 diverse models
2. Snapshot ensemble from cosine annealing
3. Test-time augmentation (30+ augmentations)
4. Biological constraints filtering

**Modules**:
- `src/models/ensemble.py`
- `src/evaluation/test_time_augmentation.py`
- `src/utils/biological_constraints.py`

---

### Phase 5: Comprehensive Evaluation 🚧 TODO

**Status**: Not yet implemented
**Duration**: ~2 days
**Priority**: CRITICAL

**Will Implement**:
1. 5-fold cross-validation results aggregation
2. Per-class performance analysis
3. Failure analysis (which morphologies confused?)
4. Interactive HTML reports
5. Model selection and final submission

**Module**: `src/evaluation/evaluate.py`

## Configuration

All pipeline settings are in [`configs/config.yaml`](configs/config.yaml):

### Key Settings

```yaml
# Data
data:
  classes: [planktonic, single_dispersed, hyphae, ...]
  num_classes: 7
  split_strategy: "leave_one_sequence_out"

# Training
training:
  baseline:
    model: "yolov8n-seg.pt"
    epochs: 100
    batch_size: 8
    image_size: 640

  ssl:
    enabled: true
    epochs: 100

  ensemble:
    enabled: true
    n_models: 10

# Hardware
hardware:
  device: "auto"  # cuda, mps, or cpu
  use_amp: true   # Automatic mixed precision
```

## Key Features

### 1. Leave-One-Sequence-Out Cross-Validation

Ensures model generalizes to new sequences (not just new frames from seen sequences):

```python
from src.data.data_loader import YOLODataLoader

loader = YOLODataLoader('configs/config.yaml')
splits = loader.create_leave_one_sequence_out_splits()

# Each fold holds out one complete sequence for validation
# Fold 0: Train on sequences 1,2,3,4 → Val on sequence 0
# Fold 1: Train on sequences 0,2,3,4 → Val on sequence 1
# ...
```

### 2. Comprehensive Visualization

All training stages produce visual outputs:

- **Training curves**: Loss, F1, precision, recall over epochs (interactive Plotly)
- **Predictions**: Side-by-side ground truth vs predictions (16 samples)
- **Confusion matrices**: Per-class performance heatmaps
- **Class distribution**: Dataset imbalance visualization

### 3. MLflow Experiment Tracking

Every training run logged with:
- Hyperparameters (lr, batch size, model architecture)
- Metrics per epoch
- Model artifacts
- Visualizations

View experiments:
```bash
mlflow ui --backend-store-uri ~/mlm_outputs/alt_pipeline_1/experiments/mlflow
```

### 4. Modular Design

Each component is independent and testable:

```python
# Test data loader
python src/data/data_loader.py

# Test visualizer
python src/evaluation/visualize.py

# Train only baseline
python src/training/train_baseline.py --fold 0
```

## Training Strategies

### Current: Baseline YOLOv8

Simple YOLOv8-nano model to establish performance benchmark.

**Pros**:
- Fast training (~2-3 hours)
- Good starting point
- Pretrained COCO weights

**Cons**:
- Limited by small dataset (205 frames)
- May overfit

### Future: Advanced Techniques

1. **Self-Supervised Pretraining**: Extract maximum knowledge from 205 frames without labels
2. **Synthetic Data**: 10,000 generated frames to prevent overfitting
3. **Ensemble**: 10-15 diverse models for robustness
4. **Test-Time Augmentation**: 10-15% performance boost at inference

## Performance Targets

| Metric | Baseline | Target | Stretch Goal |
|--------|----------|--------|--------------|
| F1 Score | 0.40-0.50 | **0.80** | 0.85 |
| mAP50 | 0.50-0.60 | **0.75** | 0.80 |
| Precision | 0.50-0.60 | 0.80 | 0.85 |
| Recall | 0.40-0.50 | 0.80 | 0.85 |

### Per-Class Goals

Special attention to rare classes:
- **Pseudohyphae** (2.1%): F1 > 0.60
- **Biofilm** (2.5%): F1 > 0.60
- **Common classes**: F1 > 0.80

## AWS SageMaker Integration

For longer training runs, use SageMaker:

```yaml
# In configs/config.yaml
sagemaker:
  enabled: true
  instance_type: "ml.g4dn.2xlarge"  # ~$0.94/hour
  use_spot_instances: true          # Save 70%
```

```bash
# TODO: Implement sagemaker_train.py
python scripts/sagemaker_train.py --config configs/config.yaml
```

## Development Workflow

### 1. Data Exploration

```bash
# Load and visualize dataset
python src/data/data_loader.py

# Check output
open ~/mlm_outputs/alt_pipeline_1/visualizations/data_quality/
```

### 2. Baseline Training

```bash
# Train on fold 0
python src/training/train_baseline.py --config configs/config.yaml --fold 0 --visualize
```

### 3. Iterate & Improve

```bash
# Implement next phase (e.g., synthetic data)
# Edit src/data/synthetic_generator.py

# Test independently
python src/data/synthetic_generator.py

# Integrate into pipeline
python scripts/run_pipeline.py --phase augmentation
```

### 4. Cross-Validation

```bash
# Train on all 5 folds
for fold in {0..4}; do
    python src/training/train_baseline.py --fold $fold
done

# Aggregate results
python src/evaluation/evaluate.py --aggregate-folds
```

## Troubleshooting

### Issue: GPU not detected

```bash
# Check PyTorch GPU support
python -c "import torch; print(torch.cuda.is_available())"  # NVIDIA
python -c "import torch; print(torch.backends.mps.is_available())"  # Apple Silicon

# If False, training will use CPU (slower)
# Edit config.yaml: hardware.device = "cpu"
```

### Issue: Out of memory

```bash
# Reduce batch size in configs/config.yaml
training:
  baseline:
    batch_size: 4  # Reduce from 8
```

### Issue: Visualizations not generating

```bash
# Check output directory permissions
ls -la ~/mlm_outputs/alt_pipeline_1/visualizations/

# Recreate directories
bash scripts/setup_environment.sh
```

### Issue: MLflow UI not starting

```bash
# Check MLflow installation
pip install mlflow

# Start with explicit path
mlflow ui --backend-store-uri ~/mlm_outputs/alt_pipeline_1/experiments/mlflow --port 5001
```

## Contributing

This pipeline is designed for iterative development:

1. **Add new module**: Create file in appropriate `src/` subdirectory
2. **Update config**: Add parameters to `configs/config.yaml`
3. **Test independently**: Run module directly
4. **Integrate**: Update `scripts/run_pipeline.py`
5. **Document**: Update this README

## References

### Papers
- **YOLOv8**: Ultralytics YOLOv8 (2023)
- **Self-Supervised Learning**: SimCLR, MoCo, BYOL
- **Few-Shot Learning**: Prototypical Networks, MAML
- **Test-Time Augmentation**: Averaging multiple augmented predictions

### Datasets
- **LIVECell**: 1.6M cell annotations across 8 cell types
- **Cellpose**: Generalist cell segmentation model
- **DeepBacs**: Bacterial microscopy dataset

## License

This project is part of the UW Madison Machine Learning Marathon 2025.

## Authors

Brain Buddies Team - MLM 2025

---

**Last Updated**: 2025-11-11
**Status**: Phase 1 Complete, Phases 2-5 In Development
**Next Milestone**: Implement synthetic data generation (Phase 2)
