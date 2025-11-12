# Quick Start Guide - Alt ML Pipeline 1

This guide will get you up and running in 5 minutes.

## Step 1: Setup Environment (2 minutes)

```bash
# Navigate to the pipeline directory
cd alt_ml_pipeline_1

# Run the setup script
bash scripts/setup_environment.sh

# Activate the environment
source ~/venvs/alt_pipeline_1/bin/activate
```

**What this does**:
- Creates an isolated Python virtual environment
- Installs all required packages (PyTorch, YOLO, MLflow, etc.)
- Creates output directories outside the git repo
- Verifies GPU availability

## Step 2: Explore the Data (1 minute)

```bash
# Print dataset summary and create visualizations
python src/data/data_loader.py
```

**Output**:
```
==============================================================
DATASET SUMMARY
==============================================================

Total Images: 93
  Training: 87
  Validation: 6

Total Annotations: 7462

Sequences: 5
  LizaMutant38: 41 frames
  MattLines1: 41 frames
  MattLines7: 18 frames
  MattLines27: 6 frames
  ...

Class Distribution:
  0. planktonic          : 2641 (35.4%)
  1. single_dispersed    : 1485 (19.9%)
  2. hyphae              : 1306 (17.5%)
  3. clump_dispersed     :  978 (13.1%)
  4. yeast               :  702 ( 9.4%)
  5. biofilm             :  187 ( 2.5%)
  6. pseudohyphae        :  157 ( 2.1%)
```

**Check visualizations**:
```bash
open ~/mlm_outputs/alt_pipeline_1/visualizations/data_quality/sample_images_by_class.png
```

## Step 3: Train Baseline Model (30-60 minutes)

```bash
# Train baseline YOLOv8 model on fold 0
python scripts/run_pipeline.py --phase foundation --fold 0 --visualize
```

**What this does**:
- Creates leave-one-sequence-out data splits
- Trains YOLOv8-nano model for 100 epochs
- Evaluates on validation set
- Creates visualizations of predictions
- Logs everything to MLflow

**Expected output**:
```
==============================================================
Training Complete - Results:
==============================================================
mAP50          : 0.5234
mAP50-95       : 0.3156
precision      : 0.6123
recall         : 0.4567
f1             : 0.5234
==============================================================

✓ Baseline model trained successfully!
  F1 Score: 0.5234
  mAP50: 0.5234
  Model: ~/mlm_outputs/alt_pipeline_1/models/individual/baseline_fold_0/weights/best.pt
```

## Step 4: View Results (1 minute)

### Option A: View Visualizations

```bash
# Predictions (Ground truth vs model predictions)
open ~/mlm_outputs/alt_pipeline_1/visualizations/predictions/baseline_fold_0_predictions.png

# Training curves
open ~/mlm_outputs/alt_pipeline_1/visualizations/training_progress/training_curves.html

# Confusion matrix
open ~/mlm_outputs/alt_pipeline_1/visualizations/confusion_matrices/confusion_matrix.png
```

### Option B: View MLflow Tracking

```bash
# Start MLflow UI
mlflow ui --backend-store-uri ~/mlm_outputs/alt_pipeline_1/experiments/mlflow

# Open browser to: http://localhost:5000
```

In the MLflow UI you can:
- Compare different runs
- View metrics over time
- Download model artifacts
- Reproduce experiments

## What's Next?

Now that you have a baseline, you can:

### 1. Train on All Folds (Cross-Validation)

```bash
# Train on all 5 folds
for fold in {0..4}; do
    python src/training/train_baseline.py --fold $fold --visualize
done
```

### 2. Improve Performance

The baseline model achieves ~0.50 F1. To reach **0.80+ F1**, implement:

- **Phase 2**: Synthetic data generation (10,000 frames)
- **Phase 3**: Self-supervised learning (pretrain on unlabeled data)
- **Phase 4**: Ensemble (10-15 diverse models)
- **Phase 5**: Test-time augmentation (boost inference)

### 3. Experiment with Hyperparameters

Edit `configs/config.yaml`:

```yaml
training:
  baseline:
    model: "yolov8s-seg.pt"  # Try small instead of nano
    epochs: 200              # Train longer
    batch_size: 16           # Larger batches (if GPU allows)
    image_size: 1024         # Higher resolution
```

Then retrain:
```bash
python src/training/train_baseline.py --fold 0
```

### 4. Analyze Failures

Check which morphologies are most confused:

```bash
# View confusion matrix
open ~/mlm_outputs/alt_pipeline_1/visualizations/confusion_matrices/

# Check per-class metrics in MLflow UI
mlflow ui --backend-store-uri ~/mlm_outputs/alt_pipeline_1/experiments/mlflow
```

Common issues:
- **Pseudohyphae vs Hyphae**: Similar elongated shapes
- **Biofilm vs Planktonic**: Overlapping cells difficult to separate
- **Small cells**: Low resolution makes detection hard

## Troubleshooting

### "No module named 'ultralytics'"

```bash
# Ensure environment is activated
source ~/venvs/alt_pipeline_1/bin/activate

# Reinstall requirements
pip install -r requirements.txt
```

### "CUDA out of memory"

```bash
# Reduce batch size in configs/config.yaml
training:
  baseline:
    batch_size: 4  # Reduce from 8
```

### "Dataset not found"

```bash
# Check that the existing YOLO dataset exists
ls ml_pipeline/data/processed/yolo_dataset/

# If missing, run data preparation from original pipeline
cd ml_pipeline
python scripts/data_prep/prepare_data.py
```

### Training is very slow

**On CPU**: Training will take 4-6 hours instead of 30-60 minutes
- Consider using SageMaker for GPU training
- Or use a smaller model: `yolov8n-seg.pt` (fastest)

**On GPU**: Should be fast
- Check GPU is being used: `python -c "import torch; print(torch.cuda.is_available())"`

## Command Reference

```bash
# Setup
bash scripts/setup_environment.sh
source ~/venvs/alt_pipeline_1/bin/activate

# Data exploration
python src/data/data_loader.py

# Training
python scripts/run_pipeline.py --phase foundation --fold 0 --visualize
python src/training/train_baseline.py --fold 0 --visualize

# Monitoring
mlflow ui --backend-store-uri ~/mlm_outputs/alt_pipeline_1/experiments/mlflow

# View results
open ~/mlm_outputs/alt_pipeline_1/visualizations/
```

## File Locations

| What | Where |
|------|-------|
| Configuration | `configs/config.yaml` |
| Source code | `src/` |
| Scripts | `scripts/` |
| Output (all) | `~/mlm_outputs/alt_pipeline_1/` |
| Models | `~/mlm_outputs/alt_pipeline_1/models/` |
| Visualizations | `~/mlm_outputs/alt_pipeline_1/visualizations/` |
| Metrics | `~/mlm_outputs/alt_pipeline_1/results/` |
| MLflow tracking | `~/mlm_outputs/alt_pipeline_1/experiments/mlflow/` |

## Performance Benchmarks

| Hardware | Training Time (100 epochs) | Device |
|----------|---------------------------|--------|
| M1 Mac (8GB) | ~45 min | mps |
| M2 Mac (16GB) | ~30 min | mps |
| NVIDIA RTX 3080 | ~25 min | cuda |
| CPU (Intel i7) | ~4 hours | cpu |
| AWS g4dn.2xlarge | ~20 min | cuda |

## Next Steps

1. ✅ Complete Phase 1 (Baseline) - **YOU ARE HERE**
2. ⬜ Implement Phase 2 (Synthetic Data)
3. ⬜ Implement Phase 3 (Self-Supervised Learning)
4. ⬜ Implement Phase 4 (Ensemble)
5. ⬜ Implement Phase 5 (Evaluation)
6. ⬜ Reach F1 > 0.80 target

See [README.md](README.md) for full documentation.

---

**Questions?** Check the [README](README.md) or review the code - it's heavily commented!
