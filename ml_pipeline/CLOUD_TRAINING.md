# Cloud Training Guide - AWS SageMaker

**Run YOLOv8 training on AWS SageMaker with GPU acceleration**

---

## 🎯 Overview

This guide covers:
1. Setting up AWS SageMaker
2. Uploading your data and code
3. Running training on GPU instances
4. Downloading trained models

**Estimated Cost**: $2-5 for 20 hours of training (ml.g4dn.xlarge)

---

## 📋 Prerequisites

### 1. AWS Account
- Create free tier account at https://aws.amazon.com/
- You may be eligible for AWS credits (check with your university)

### 2. Local Setup
```bash
pip install boto3 sagemaker awscli
aws configure  # Enter your AWS credentials
```

---

## 🚀 Method 1: SageMaker Studio (Easiest)

### Step 1: Open SageMaker Studio

1. Go to AWS Console → SageMaker
2. Click "Studio" → "Open Studio"
3. Create domain if first time (takes 5 min)

### Step 2: Upload Code and Data

**Option A: Direct Upload**
```bash
# In SageMaker Studio terminal
git clone <your-repo-url>
cd brain_buddies_mlm-25/ml_pipeline

# Upload data (if not already in S3)
aws s3 cp /path/to/data s3://your-bucket/mlm-data/ --recursive
```

**Option B: From Local Machine**
```bash
# Package everything
cd brain_buddies_mlm-25
tar -czf ml_pipeline.tar.gz ml_pipeline/

# Upload to S3
aws s3 cp ml_pipeline.tar.gz s3://your-bucket/
aws s3 sync /Users/aineshmohan/Documents/mlm s3://your-bucket/mlm-data/
```

### Step 3: Create Notebook

In SageMaker Studio, create `train_on_sagemaker.ipynb`:

```python
# Cell 1: Setup
import sagemaker
from sagemaker.pytorch import PyTorch
import boto3

# Get SageMaker session
session = sagemaker.Session()
role = sagemaker.get_execution_role()
bucket = session.default_bucket()

print(f"SageMaker session region: {session.boto_region_name}")
print(f"Default S3 bucket: {bucket}")
print(f"Role: {role}")
```

```python
# Cell 2: Prepare data
# Your data should be in S3
data_location = f's3://{bucket}/mlm-data/'
print(f"Data location: {data_location}")

# Upload code if not already done
import os
if not os.path.exists('ml_pipeline'):
    !aws s3 cp s3://{bucket}/ml_pipeline.tar.gz .
    !tar -xzf ml_pipeline.tar.gz
```

```python
# Cell 3: Create training script
# This is the entry point for SageMaker
entry_script = """
import os
import sys
import subprocess

# Install additional dependencies
subprocess.check_call([sys.executable, "-m", "pip", "install",
                      "ultralytics", "mlflow", "albumentations"])

# Add ml_pipeline to path
sys.path.append('/opt/ml/code/ml_pipeline')

# Import training script
from scripts.training.train_yolo_segmentation import main

if __name__ == '__main__':
    # SageMaker passes arguments via command line
    main()
"""

# Save entry script
with open('train_entry.py', 'w') as f:
    f.write(entry_script)
```

```python
# Cell 4: Configure training job
estimator = PyTorch(
    entry_point='train_entry.py',
    source_dir='ml_pipeline',
    role=role,
    instance_type='ml.g4dn.xlarge',  # GPU instance (~$0.70/hr)
    instance_count=1,
    framework_version='2.0.0',
    py_version='py310',
    hyperparameters={
        'data': '/opt/ml/input/data/training/yolo_dataset/dataset.yaml',
        'model': 's',
        'img-size': 640,
        'batch-size': 8,
        'epochs': 150,
        'device': '0',
        'project': '/opt/ml/model',
        'name': 'candida_exp1'
    },
    output_path=f's3://{bucket}/sagemaker-output/',
    max_run=72000,  # 20 hours
    keep_alive_period_in_seconds=1800  # Keep instance warm for 30 min
)

print("Estimator configured!")
```

```python
# Cell 5: Start training
# This will launch a GPU instance and start training
estimator.fit({
    'training': data_location
})

print("Training job submitted!")
print(f"Job name: {estimator.latest_training_job.name}")
```

```python
# Cell 6: Monitor training (run in separate cell)
# Check CloudWatch logs
!aws logs tail /aws/sagemaker/TrainingJobs --follow
```

```python
# Cell 7: Download trained model
# After training completes
model_data = estimator.model_data
print(f"Model saved to: {model_data}")

# Download locally
!aws s3 cp {model_data} ./model.tar.gz
!tar -xzf model.tar.gz
```

---

## 🚀 Method 2: SageMaker Training Job (More Control)

### Step 1: Create Training Script for SageMaker

Create `ml_pipeline/scripts/training/train_sagemaker.py`:

```python
"""
SageMaker-compatible training script.
Handles SageMaker-specific paths and environment.
"""

import os
import sys
import argparse
from pathlib import Path

# SageMaker directories
SM_MODEL_DIR = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
SM_CHANNEL_TRAINING = os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training')
SM_OUTPUT_DATA_DIR = os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data')

def train():
    """Run training on SageMaker."""
    parser = argparse.ArgumentParser()

    # SageMaker parameters
    parser.add_argument('--model-dir', type=str, default=SM_MODEL_DIR)
    parser.add_argument('--data-dir', type=str, default=SM_CHANNEL_TRAINING)
    parser.add_argument('--output-dir', type=str, default=SM_OUTPUT_DATA_DIR)

    # Training parameters
    parser.add_argument('--model', type=str, default='s')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--device', type=str, default='0')

    args = parser.parse_args()

    print("="*80)
    print("TRAINING ON AWS SAGEMAKER")
    print("="*80)
    print(f"Model directory: {args.model_dir}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print("="*80)

    # Import after argument parsing
    from ultralytics import YOLO

    # Find dataset YAML
    dataset_yaml = Path(args.data_dir) / 'yolo_dataset' / 'dataset.yaml'
    if not dataset_yaml.exists():
        dataset_yaml = Path(args.data_dir) / 'dataset.yaml'

    print(f"Dataset YAML: {dataset_yaml}")

    # Initialize model
    model = YOLO(f'yolov8{args.model}-seg.pt')

    # Training configuration
    results = model.train(
        data=str(dataset_yaml),
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.img_size,
        device=args.device,
        project=args.model_dir,
        name='training',
        exist_ok=True,
        verbose=True,
        patience=50,
        save=True,
        plots=True
    )

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)

    # Copy best weights to model directory root for easy access
    import shutil
    best_weights = Path(args.model_dir) / 'training' / 'weights' / 'best.pt'
    if best_weights.exists():
        shutil.copy(best_weights, Path(args.model_dir) / 'best.pt')
        print(f"Best model saved to: {Path(args.model_dir) / 'best.pt'}")

    return results

if __name__ == '__main__':
    train()
```

### Step 2: Create Requirements for SageMaker

Create `ml_pipeline/requirements_sagemaker.txt`:

```
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
opencv-python>=4.8.0
scikit-image>=0.21.0
scipy>=1.10.0
Pillow>=10.0.0
tifffile>=2023.0.0
albumentations>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
pyyaml>=6.0
tqdm>=4.65.0
```

### Step 3: Create Launch Script

Create `ml_pipeline/launch_sagemaker.py`:

```python
#!/usr/bin/env python3
"""
Launch SageMaker training job from local machine.
"""

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from datetime import datetime
import argparse

def launch_training(
    data_s3_path: str,
    output_s3_path: str,
    instance_type: str = 'ml.g4dn.xlarge',
    model_size: str = 's',
    epochs: int = 150,
    batch_size: int = 8
):
    """Launch SageMaker training job."""

    # Initialize SageMaker session
    session = sagemaker.Session()
    role = sagemaker.get_execution_role()

    print("="*80)
    print("LAUNCHING SAGEMAKER TRAINING JOB")
    print("="*80)
    print(f"Instance type: {instance_type}")
    print(f"Model: YOLOv8{model_size}-seg")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Data: {data_s3_path}")
    print(f"Output: {output_s3_path}")
    print("="*80 + "\n")

    # Create estimator
    estimator = PyTorch(
        entry_point='train_sagemaker.py',
        source_dir='scripts/training',
        dependencies=['utils/', 'scripts/'],  # Include other modules
        role=role,
        instance_type=instance_type,
        instance_count=1,
        framework_version='2.0.0',
        py_version='py310',
        hyperparameters={
            'model': model_size,
            'epochs': epochs,
            'batch-size': batch_size,
            'img-size': 640,
            'device': '0'
        },
        output_path=output_s3_path,
        max_run=72000,  # 20 hours max
        keep_alive_period_in_seconds=1800,  # Keep warm for 30 min
        volume_size=50,  # GB
        checkpoint_s3_uri=output_s3_path + '/checkpoints',
        use_spot_instances=True,  # Save ~70% cost
        max_wait=86400  # 24 hours
    )

    # Start training
    print("Starting training job...")
    estimator.fit({
        'training': data_s3_path
    }, wait=False)

    job_name = estimator.latest_training_job.name

    print("\n" + "="*80)
    print("TRAINING JOB LAUNCHED")
    print("="*80)
    print(f"Job name: {job_name}")
    print(f"\nMonitor at:")
    print(f"  https://console.aws.amazon.com/sagemaker/home?region={session.boto_region_name}#/jobs/{job_name}")
    print(f"\nView logs:")
    print(f"  aws logs tail /aws/sagemaker/TrainingJobs/{job_name} --follow")
    print("="*80 + "\n")

    return estimator

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launch SageMaker training')
    parser.add_argument('--data-s3', required=True, help='S3 path to data')
    parser.add_argument('--output-s3', required=True, help='S3 path for output')
    parser.add_argument('--instance', default='ml.g4dn.xlarge', help='Instance type')
    parser.add_argument('--model', default='s', choices=['n', 's', 'm', 'l', 'x'])
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=8)

    args = parser.parse_args()

    estimator = launch_training(
        data_s3_path=args.data_s3,
        output_s3_path=args.output_s3,
        instance_type=args.instance,
        model_size=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
```

---

## 📤 Complete Workflow

### 1. Prepare Data Locally

```bash
cd ml_pipeline

# Convert annotations (if not done)
cd scripts/data_prep
python3 convert_all_annotations.py \
    --data-dir /Users/aineshmohan/Documents/mlm \
    --output-dir ../../data/processed/yolo_dataset

cd ../..
```

### 2. Upload to S3

```bash
# Set your bucket name
BUCKET="your-sagemaker-bucket"

# Create bucket if needed
aws s3 mb s3://$BUCKET

# Upload processed dataset
aws s3 sync data/processed/yolo_dataset s3://$BUCKET/mlm-data/yolo_dataset/

# Upload code (optional - can use git in SageMaker)
tar -czf ../ml_pipeline.tar.gz .
aws s3 cp ../ml_pipeline.tar.gz s3://$BUCKET/code/
```

### 3. Launch Training

**Option A: From SageMaker Studio**
- Open notebook `train_on_sagemaker.ipynb`
- Run cells sequentially

**Option B: From Local Machine**
```bash
cd ml_pipeline

python3 launch_sagemaker.py \
    --data-s3 s3://$BUCKET/mlm-data/ \
    --output-s3 s3://$BUCKET/sagemaker-output/ \
    --instance ml.g4dn.xlarge \
    --model s \
    --epochs 150 \
    --batch-size 8
```

### 4. Monitor Training

```bash
# Get job name from launch output
JOB_NAME="pytorch-training-2024-11-04-12-34-56-789"

# Watch logs
aws logs tail /aws/sagemaker/TrainingJobs/$JOB_NAME --follow

# Check status
aws sagemaker describe-training-job --training-job-name $JOB_NAME
```

### 5. Download Results

```bash
# After training completes, download model
aws s3 sync s3://$BUCKET/sagemaker-output/$JOB_NAME/output/ ./trained_models/

# Extract
cd trained_models
tar -xzf model.tar.gz

# The best.pt file is your trained model!
ls -lh best.pt
```

---

## 💰 Cost Estimation

### Instance Types & Pricing

| Instance | GPU | vCPU | RAM | Cost/hr | 20hr Cost |
|----------|-----|------|-----|---------|-----------|
| ml.g4dn.xlarge | T4 (16GB) | 4 | 16GB | $0.736 | $14.72 |
| ml.g4dn.2xlarge | T4 (16GB) | 8 | 32GB | $0.94 | $18.80 |
| ml.p3.2xlarge | V100 (16GB) | 8 | 61GB | $3.82 | $76.40 |

**Recommendation**: `ml.g4dn.xlarge` (best value)

### Cost Savings

- **Spot Instances**: Save ~70% (set `use_spot_instances=True`)
- **Early Stopping**: Saves compute if converging early
- **Smaller Model**: YOLOv8n trains 2x faster

**Estimated Total**: $5-15 depending on configuration

---

## 🐛 Troubleshooting

### Issue: "No module named 'ultralytics'"

**Solution**: Add to `requirements_sagemaker.txt` and rebuild

### Issue: "CUDA out of memory"

**Solution**: Reduce batch size
```python
hyperparameters={
    'batch-size': 4,  # Reduced from 8
    ...
}
```

### Issue: "Dataset YAML not found"

**Solution**: Check S3 path structure
```bash
aws s3 ls s3://$BUCKET/mlm-data/yolo_dataset/
# Should see: dataset.yaml, images/, labels/
```

### Issue: Training job fails immediately

**Solution**: Check CloudWatch logs
```bash
aws logs tail /aws/sagemaker/TrainingJobs/$JOB_NAME
```

---

## 📊 Monitoring Training Progress

### Option 1: CloudWatch Logs (Real-time)
```bash
aws logs tail /aws/sagemaker/TrainingJobs/$JOB_NAME --follow
```

### Option 2: SageMaker Console
1. Go to SageMaker → Training Jobs
2. Click on your job
3. View "Monitor" tab for metrics

### Option 3: Download Partial Results
```bash
# SageMaker saves checkpoints to S3
aws s3 sync s3://$BUCKET/sagemaker-output/$JOB_NAME/checkpoints/ ./checkpoints/
```

---

## ✅ Checklist

### Before Launch
- [ ] AWS account setup
- [ ] AWS CLI configured (`aws configure`)
- [ ] Data converted to YOLO format
- [ ] Data uploaded to S3
- [ ] S3 bucket created
- [ ] IAM role has SageMaker permissions

### During Training
- [ ] Training job launched successfully
- [ ] Logs showing progress
- [ ] Loss decreasing
- [ ] No errors in CloudWatch

### After Training
- [ ] Model downloaded from S3
- [ ] best.pt file exists
- [ ] Validation metrics acceptable
- [ ] Ready for inference

---

## 🚀 Quick Start Commands

```bash
# 1. Setup
pip install boto3 sagemaker awscli
aws configure

# 2. Upload data
BUCKET="your-bucket"
aws s3 sync data/processed/yolo_dataset s3://$BUCKET/mlm-data/yolo_dataset/

# 3. Launch
python3 launch_sagemaker.py \
    --data-s3 s3://$BUCKET/mlm-data/ \
    --output-s3 s3://$BUCKET/output/ \
    --instance ml.g4dn.xlarge \
    --model s \
    --epochs 150

# 4. Monitor
aws logs tail /aws/sagemaker/TrainingJobs/pytorch-training-* --follow

# 5. Download
aws s3 sync s3://$BUCKET/output/pytorch-training-*/output/ ./models/
```

---

## 📚 Additional Resources

- [SageMaker Python SDK](https://sagemaker.readthedocs.io/)
- [SageMaker Pricing](https://aws.amazon.com/sagemaker/pricing/)
- [PyTorch on SageMaker](https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/index.html)
- [AWS Free Tier](https://aws.amazon.com/free/)

---

**You're ready to train on the cloud! 🚀☁️**
