# SageMaker Training for Alt ML Pipeline 1

Train your model on AWS SageMaker with GPU acceleration for faster training.

## Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Your Local Machine                        │
│  1. Prepare data & upload to S3                             │
│  2. Launch training job                                      │
│  3. Monitor progress                                         │
│  4. Download trained model                                   │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                      AWS SageMaker                           │
│  • Spins up GPU instance (ml.g4dn.2xlarge)                  │
│  • Downloads data from S3                                    │
│  • Runs train_sagemaker.py                                   │
│  • Uploads trained model to S3                               │
│  • Shuts down instance (saves $$)                            │
└─────────────────────────────────────────────────────────────┘
```

## Why Use SageMaker?

**Pros:**
- ✅ **GPU acceleration** - 3-4x faster than M1 Mac
- ✅ **Spot instances** - Save 70% on costs
- ✅ **No local compute** - Frees up your machine
- ✅ **Scalable** - Train multiple folds in parallel

**Cons:**
- ❌ **Costs money** - ~$0.94/hour for GPU (but only $0.28 with spot)
- ❌ **Setup required** - AWS account, IAM role, etc.
- ❌ **Internet needed** - Upload/download data

**Cost Estimate:**
- Regular: ~$0.94/hour = ~$0.50 per training run (30 min)
- Spot: ~$0.28/hour = ~$0.15 per training run (30 min)
- 5 folds: ~$0.75 total with spot instances

## Prerequisites

### 1. AWS Account Setup

1. **Create AWS Account** (if you don't have one)
   - Go to https://aws.amazon.com
   - Sign up (free tier available)

2. **Install AWS CLI**
   ```bash
   # macOS
   brew install awscli

   # Or use pip
   pip install awscli
   ```

3. **Configure AWS Credentials**
   ```bash
   aws configure
   # Enter:
   #   AWS Access Key ID: [your key]
   #   AWS Secret Access Key: [your secret]
   #   Default region: us-east-1
   #   Default output format: json
   ```

### 2. Create SageMaker Execution Role

**Option A: AWS Console (Easiest)**

1. Go to https://console.aws.amazon.com/iam/
2. Click "Roles" → "Create role"
3. Select "SageMaker" as the service
4. Click "Next" through permissions
5. Name it: `SageMakerExecutionRole`
6. Copy the Role ARN (looks like: `arn:aws:iam::123456789:role/SageMakerExecutionRole`)

**Option B: AWS CLI**

```bash
# Create role
aws iam create-role \
    --role-name SageMakerExecutionRole \
    --assume-role-policy-document '{
      "Version": "2012-10-17",
      "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "sagemaker.amazonaws.com"},
        "Action": "sts:AssumeRole"
      }]
    }'

# Attach policies
aws iam attach-role-policy \
    --role-name SageMakerExecutionRole \
    --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

# Get role ARN
aws iam get-role --role-name SageMakerExecutionRole --query 'Role.Arn'
```

### 3. Set Environment Variable

```bash
# Add to ~/.zshrc or ~/.bashrc
export SAGEMAKER_ROLE="arn:aws:iam::YOUR_ACCOUNT_ID:role/SageMakerExecutionRole"

# Reload
source ~/.zshrc
```

## Quick Start

### Option 1: Launch Training (Recommended)

```bash
# Activate environment
cd alt_ml_pipeline_1
source ~/venvs/alt_pipeline_1/bin/activate

# Launch training for fold 0
python sagemaker/launch_sagemaker_training.py --fold 0

# Or specify role explicitly
python sagemaker/launch_sagemaker_training.py \
    --fold 0 \
    --role arn:aws:iam::123456789:role/SageMakerExecutionRole
```

This will:
1. Package your data (fold 0)
2. Upload to S3
3. Launch SageMaker training job
4. Wait for completion
5. Show results

### Option 2: Monitor Existing Job

```bash
# Check job status
aws sagemaker describe-training-job \
    --training-job-name alt-pipeline-fold-0-20231111-123456

# Download results from completed job
python sagemaker/launch_sagemaker_training.py \
    --download-only \
    --job-name alt-pipeline-fold-0-20231111-123456
```

## Training Process

### What Happens During Training

1. **Setup (1-2 minutes)**
   - SageMaker provisions GPU instance
   - Downloads Docker image with PyTorch
   - Downloads your data from S3

2. **Training (30-60 minutes)**
   - Runs `train_sagemaker.py` inside container
   - Trains YOLOv8 model for 100 epochs
   - Saves checkpoints every 10 epochs

3. **Cleanup**
   - Uploads model artifacts to S3
   - Saves metrics to output directory
   - Shuts down instance (stops billing)

### Monitor Training

**AWS Console:**
1. Go to https://console.aws.amazon.com/sagemaker/home#/jobs
2. Find your job (e.g., `alt-pipeline-fold-0-20231111-123456`)
3. View logs in CloudWatch

**Command Line:**
```bash
# List recent jobs
aws sagemaker list-training-jobs --max-results 5

# Get job details
aws sagemaker describe-training-job --training-job-name <job-name>

# View logs
aws logs tail /aws/sagemaker/TrainingJobs/<job-name> --follow
```

## Files in This Directory

```
sagemaker/
├── README.md                        # This file
├── train_sagemaker.py              # Training script (runs in SageMaker)
└── launch_sagemaker_training.py    # Launch script (runs locally)
```

## Configuration

Edit `configs/config.yaml` to customize SageMaker settings:

```yaml
sagemaker:
  enabled: true
  instance_type: "ml.g4dn.2xlarge"  # GPU instance
  volume_size: 50                    # GB storage
  max_run_time: 72000                # 20 hours max
  use_spot_instances: true           # Save 70%!
  spot_max_wait: 86400               # Wait up to 24h for spot
```

### Instance Types & Pricing

| Instance Type | GPUs | vCPUs | Memory | Price/Hour | Spot Price |
|--------------|------|-------|---------|-----------|------------|
| ml.g4dn.xlarge | 1 T4 | 4 | 16 GB | $0.736 | ~$0.22 |
| ml.g4dn.2xlarge | 1 T4 | 8 | 32 GB | $0.94 | ~$0.28 |
| ml.g4dn.4xlarge | 1 T4 | 16 | 64 GB | $1.505 | ~$0.45 |
| ml.p3.2xlarge | 1 V100 | 8 | 61 GB | $3.825 | ~$1.15 |

**Recommendation**: Use `ml.g4dn.2xlarge` with spot instances for best value.

## Spot Instances

**What are Spot Instances?**
- AWS sells unused capacity at 70% discount
- Can be interrupted with 2-minute warning
- SageMaker automatically handles interruptions

**Should I use them?**
- ✅ Yes for training (saves $$)
- ✅ Training jobs can resume from checkpoint
- ❌ No for production inference

**How to enable:**
```yaml
# In config.yaml
sagemaker:
  use_spot_instances: true
  max_wait: 86400  # Wait up to 24h if no spot available
```

## Training Multiple Folds

Train all 5 folds in parallel (saves time):

```bash
# Launch 5 separate jobs
for fold in {0..4}; do
    python sagemaker/launch_sagemaker_training.py --fold $fold &
done

# Wait for all to complete
wait

echo "All folds complete!"
```

**Cost**: 5 folds × $0.15 = $0.75 total with spot instances

## Download Results

After training completes:

```bash
# Results are automatically in S3
# Download with:
python sagemaker/launch_sagemaker_training.py \
    --download-only \
    --job-name alt-pipeline-fold-0-20231111-123456

# Results saved to:
# ~/mlm_outputs/alt_pipeline_1/sagemaker_results/
```

## Troubleshooting

### "No module named 'sagemaker'"

```bash
source ~/venvs/alt_pipeline_1/bin/activate
pip install sagemaker boto3
```

### "Could not find credentials"

```bash
aws configure
# Enter your AWS credentials
```

### "ResourceLimitExceeded"

You've hit AWS quotas. Either:
1. Use spot instances (separate quota)
2. Request quota increase in AWS Console
3. Try different instance type

### "Spot instance interrupted"

SageMaker will automatically retry. Your checkpoint is saved.

### Training takes too long

- Use larger instance (ml.g4dn.4xlarge)
- Reduce epochs in config.yaml
- Reduce batch size if OOM errors

### High costs

- ✅ Use spot instances (70% savings)
- ✅ Enable early stopping (patience parameter)
- ✅ Reduce max_run_time
- ❌ Don't use p3 instances unless necessary

## Comparison: Local vs SageMaker

| Metric | M1 Mac (Local) | ml.g4dn.2xlarge (SageMaker) |
|--------|----------------|----------------------------|
| **Training Time** | 45-60 min | 20-30 min |
| **Cost** | Free (electricity) | $0.15 with spot |
| **GPU** | Apple M1 Pro (MPS) | NVIDIA T4 (CUDA) |
| **Memory** | 16 GB | 32 GB |
| **Frees Local CPU** | ❌ No | ✅ Yes |
| **Internet Required** | ❌ No | ✅ Yes |

## Advanced: Custom Training Script

Want to modify training? Edit `train_sagemaker.py`:

```python
# Add custom callbacks
def custom_callback(trainer):
    print(f"Epoch {trainer.epoch}")

# In train_yolo_sagemaker():
model.add_callback('on_epoch_end', custom_callback)
```

## Best Practices

1. **Start with one fold** - Test before running all 5
2. **Use spot instances** - 70% cost savings
3. **Monitor first job** - Check logs for issues
4. **Set reasonable max_run_time** - Don't overpay for stuck jobs
5. **Clean up S3** - Delete old data after training

## Cleanup

After training, clean up S3 to avoid storage costs:

```bash
# List your S3 buckets
aws s3 ls

# Delete training data (after downloading results)
aws s3 rm s3://your-bucket/alt-pipeline-1/ --recursive
```

## Support

- **AWS Documentation**: https://docs.aws.amazon.com/sagemaker/
- **Pricing Calculator**: https://calculator.aws/
- **SageMaker Examples**: https://github.com/aws/amazon-sagemaker-examples

---

**Ready to train?**
```bash
python sagemaker/launch_sagemaker_training.py --fold 0
```

Estimated time: 30 minutes
Estimated cost: $0.15 (with spot)
