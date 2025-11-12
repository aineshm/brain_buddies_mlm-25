# SageMaker Notebook Training Guide

Complete guide for training your model in a SageMaker notebook instance or SageMaker Studio.

## Overview

```
┌──────────────────────────────────────────────────────────┐
│  1. Upload Data to S3 (one-time, from your local machine)│
│  2. Open SageMaker Notebook Instance                      │
│  3. Upload and run sagemaker_training.ipynb              │
│  4. Download results from S3                              │
└──────────────────────────────────────────────────────────┘
```

---

## Step 1: Upload Data to S3 (Local Machine)

### What Data to Upload

You need to upload the **entire YOLO dataset** to S3:

```
ml_pipeline/data/processed/yolo_dataset/
├── images/
│   ├── train/     # All training images (.tif files)
│   └── val/       # All validation images (.tif files)
└── labels/
    ├── train/     # Training labels (.txt files)
    └── val/       # Validation labels (.txt files)
```

### Upload Command (From Your Mac)

```bash
# Navigate to your project
cd /Users/aineshmohan/Library/Mobile\ Documents/com~apple~CloudDocs/College/MLM\ 25/brain_buddies_mlm-25

# Upload entire YOLO dataset to S3
aws s3 sync ml_pipeline/data/processed/yolo_dataset/ \
    s3://your-bucket-name/alt-pipeline-1/yolo_dataset/ \
    --exclude "*.DS_Store"

# This will upload:
# - All images (~93 .tif files)
# - All labels (~93 .txt files)
# - Maintaining the directory structure
```

**Replace `your-bucket-name` with your actual S3 bucket!**

### Verify Upload

```bash
# Check what was uploaded
aws s3 ls s3://your-bucket-name/alt-pipeline-1/yolo_dataset/ --recursive

# Should show:
# images/train/LizaMutant38_frame_0000.tif
# images/train/MattLines1_frame_0001.tif
# labels/train/LizaMutant38_frame_0000.txt
# ... etc
```

### Upload Size & Cost

- **Data size**: ~500-800 MB (93 images + labels)
- **Upload time**: 2-5 minutes (depending on internet speed)
- **S3 storage cost**: ~$0.023/month for 1 GB (basically free)

---

## Step 2: Create/Access SageMaker Notebook Instance

### Option A: Use Existing Notebook Instance

If you already have a SageMaker notebook instance:

1. Go to: https://console.aws.amazon.com/sagemaker/home#/notebook-instances
2. Find your instance
3. Click "Open JupyterLab" or "Open Jupyter"

### Option B: Create New Notebook Instance

1. **Go to SageMaker Console**
   - https://console.aws.amazon.com/sagemaker/home#/notebook-instances

2. **Click "Create notebook instance"**

3. **Configure instance:**
   ```
   Notebook instance name: alt-pipeline-training
   Notebook instance type: ml.g4dn.xlarge (GPU recommended)
   Platform: Amazon Linux 2, Jupyter Lab 3
   ```

4. **IAM role:**
   - Create new role OR use existing
   - Ensure it has:
     - `AmazonS3FullAccess` (to access your data)
     - `AmazonSageMakerFullAccess`

5. **Leave other settings as default**

6. **Click "Create notebook instance"**
   - Wait 5-10 minutes for instance to start
   - Status will change to "InService"

7. **Click "Open JupyterLab"**

---

## Step 3: Upload & Run Notebook

### Upload Notebook to SageMaker

1. **In JupyterLab**, click the upload button (⬆️ icon)

2. **Navigate to** and upload:
   ```
   alt_ml_pipeline_1/notebooks/sagemaker_training.ipynb
   ```

3. **Double-click** to open the notebook

### Configure the Notebook

**IMPORTANT**: Edit the configuration cell at the top:

```python
# ============================================================================
# CONFIGURATION - UPDATE THESE VALUES
# ============================================================================

# Your S3 bucket name (REQUIRED - update this!)
S3_BUCKET = "your-bucket-name"  # ← CHANGE THIS TO YOUR ACTUAL BUCKET

# S3 paths (should match where you uploaded data)
S3_DATA_PREFIX = "alt-pipeline-1/yolo_dataset"

# Which fold to train (0-4)
FOLD_IDX = 0  # Start with fold 0

# Training settings (can leave as default or adjust)
NUM_EPOCHS = 100
BATCH_SIZE = 16  # Increase to 32 if you have ml.g4dn.2xlarge
IMAGE_SIZE = 640
MODEL_NAME = "yolov8n-seg.pt"  # Nano is fastest
```

### Run the Notebook

**Option 1: Run All Cells**
- Click "Run" → "Run All Cells"
- Wait for training to complete (~30-60 minutes)

**Option 2: Run Cell by Cell**
- Click in first cell
- Press `Shift + Enter` to run and move to next
- Review output of each cell before continuing

### What Each Section Does

1. **Configuration & Setup** - Sets variables, installs packages
2. **Download Data from S3** - Downloads your YOLO dataset
3. **Analyze Dataset** - Shows class distribution
4. **Create CV Splits** - Creates 5-fold cross-validation splits
5. **Train Model** - Trains YOLO for specified epochs
6. **Evaluate Model** - Computes F1, mAP50, etc.
7. **Upload Results** - Saves model and metrics back to S3
8. **Summary** - Shows final results

---

## Step 4: Monitor Training

### In the Notebook

Watch the training progress in real-time:

```
Epoch 1/100:  100%|██████████| 10/10 [00:15<00:00,  0.65it/s]
      Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
        all         13        289      0.543      0.423      0.478      0.287

Epoch 2/100:  100%|██████████| 10/10 [00:14<00:00,  0.68it/s]
...
```

### Key Metrics to Watch

- **Box(P)** = Precision
- **R** = Recall
- **mAP50** = Mean Average Precision at IoU 0.5
- **mAP50-95** = mAP averaged over IoU 0.5-0.95

### Expected Training Time

| Instance Type | GPU | Time per Epoch | Total Time (100 epochs) |
|--------------|-----|----------------|------------------------|
| ml.g4dn.xlarge | NVIDIA T4 | ~20-30 sec | ~30-50 min |
| ml.g4dn.2xlarge | NVIDIA T4 | ~15-20 sec | ~25-35 min |
| ml.p3.2xlarge | NVIDIA V100 | ~10-15 sec | ~15-25 min |

---

## Step 5: Download Results

### From S3 to Local Machine

After training completes, download your results:

```bash
# Download all results for fold 0
aws s3 sync s3://your-bucket-name/alt-pipeline-1/training-output/fold_0/ \
    ~/mlm_outputs/alt_pipeline_1/sagemaker_results/fold_0/

# This downloads:
# - best.pt (best model checkpoint)
# - last.pt (last epoch checkpoint)
# - metrics.json (performance metrics)
# - *.png (training plots)
```

### View Results

```bash
# View metrics
cat ~/mlm_outputs/alt_pipeline_1/sagemaker_results/fold_0/*/metrics.json

# Example output:
{
  "fold": 0,
  "validation_sequence": "LizaMutant38",
  "mAP50": 0.5234,
  "mAP50-95": 0.3156,
  "precision": 0.6123,
  "recall": 0.4567,
  "f1": 0.5234,
  "timestamp": "2025-11-11T18:30:45"
}
```

---

## Training All 5 Folds

To train all folds for cross-validation:

### Method 1: Sequential (One at a Time)

1. **Train Fold 0**
   - Set `FOLD_IDX = 0` in notebook
   - Run all cells
   - Wait for completion

2. **Train Fold 1**
   - Change `FOLD_IDX = 1`
   - Run all cells again
   - Wait for completion

3. **Repeat** for folds 2, 3, 4

**Total time**: ~2.5-5 hours (depending on instance)

### Method 2: Parallel (Multiple Notebook Instances)

If you need faster results:

1. **Create 5 notebook instances** (one per fold)
2. **Upload notebook to each**
3. **Set different FOLD_IDX** in each (0-4)
4. **Run all simultaneously**

**Total time**: ~30-60 minutes (all folds in parallel)
**Cost**: 5x the single instance cost

---

## Cost Breakdown

### Notebook Instance Costs

| Instance Type | Price/Hour | Training Time | Cost per Fold | Cost for 5 Folds |
|--------------|-----------|---------------|--------------|-----------------|
| ml.g4dn.xlarge | $0.736 | ~50 min | **$0.61** | **$3.05** |
| ml.g4dn.2xlarge | $0.94 | ~35 min | **$0.55** | **$2.75** |
| ml.p3.2xlarge | $3.825 | ~20 min | **$1.27** | **$6.35** |

**Recommendation**: Use `ml.g4dn.xlarge` for best value.

### Additional Costs

- **S3 Storage**: ~$0.023/month for 1 GB (negligible)
- **S3 Data Transfer**: First 100 GB out is free
- **Total for 5 folds**: ~$3-4 using ml.g4dn.xlarge

---

## Troubleshooting

### "No module named 'ultralytics'"

The notebook installs packages automatically. If it fails:

```python
# In a notebook cell, run:
!pip install ultralytics==8.0.227 opencv-python-headless PyYAML tqdm
```

### "Access Denied" when downloading from S3

Your notebook instance role needs S3 permissions:

1. Go to IAM console
2. Find your notebook instance role
3. Attach policy: `AmazonS3FullAccess`
4. Restart notebook instance

### "CUDA out of memory"

Reduce batch size:

```python
# In configuration cell:
BATCH_SIZE = 8  # Reduce from 16
```

### Training is very slow

Check you're using GPU instance:

```python
# In notebook, run this cell:
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Should show:
# CUDA available: True
# GPU: Tesla T4
```

If False, you're using a CPU instance (not recommended).

### Files not found in S3

Verify data was uploaded correctly:

```bash
# Check S3 contents
aws s3 ls s3://your-bucket-name/alt-pipeline-1/yolo_dataset/ --recursive | head -20

# Should see images and labels
```

---

## Best Practices

### 1. Start with One Fold

Train fold 0 first to verify everything works:
- Check data downloads correctly
- Verify GPU is being used
- Confirm results upload to S3

### 2. Monitor Costs

Check your AWS billing dashboard regularly:
- https://console.aws.amazon.com/billing/home

Set up billing alerts:
- Create alarm for $10 threshold

### 3. Stop Instance When Done

**IMPORTANT**: Stop your notebook instance after training to avoid charges:

1. Go to: https://console.aws.amazon.com/sagemaker/home#/notebook-instances
2. Select your instance
3. Click "Actions" → "Stop"

Instance will continue to charge even when idle!

### 4. Use Checkpoints

The notebook saves checkpoints every 10 epochs:
- If training is interrupted, you can resume
- Both `best.pt` and `last.pt` are saved

### 5. Backup Results

Always download results from S3 to your local machine:
- S3 can be deleted accidentally
- Local backup ensures you don't lose trained models

---

## Quick Reference Commands

### Before Training (Local Machine)

```bash
# Upload data to S3
aws s3 sync ml_pipeline/data/processed/yolo_dataset/ \
    s3://your-bucket-name/alt-pipeline-1/yolo_dataset/

# Verify upload
aws s3 ls s3://your-bucket-name/alt-pipeline-1/yolo_dataset/ --recursive
```

### After Training (Local Machine)

```bash
# Download results
aws s3 sync s3://your-bucket-name/alt-pipeline-1/training-output/ \
    ~/mlm_outputs/alt_pipeline_1/sagemaker_results/

# View metrics
cat ~/mlm_outputs/alt_pipeline_1/sagemaker_results/fold_0/*/metrics.json
```

### In SageMaker Notebook

```python
# Check GPU
import torch
print(torch.cuda.is_available())

# Check S3 access
!aws s3 ls s3://your-bucket-name/

# Monitor S3 usage
!aws s3 ls s3://your-bucket-name/alt-pipeline-1/ --recursive --human-readable --summarize
```

---

## Next Steps After Training

1. **Compare folds**: Look at F1 scores across all 5 folds
2. **Ensemble**: Combine best models for better performance
3. **Analyze failures**: Which classes perform worst?
4. **Iterate**: Adjust hyperparameters and retrain

See main [README.md](../README.md) for next phases (synthetic data, ensemble, etc.)

---

**Ready to start?**

1. ✅ Upload data to S3
2. ✅ Open SageMaker notebook instance
3. ✅ Upload `sagemaker_training.ipynb`
4. ✅ Update `S3_BUCKET` in config cell
5. ✅ Run all cells!
