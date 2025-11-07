# SageMaker Quick Start - 3 Steps to Train on Cloud

**Train YOLOv8 on AWS SageMaker in under 10 minutes setup time!**

---

## 🚀 Super Quick Start (3 Commands)

```bash
# 1. Setup AWS (one-time)
pip install boto3 sagemaker awscli
aws configure

# 2. Upload data
./upload_to_s3.sh my-bucket-name

# 3. Launch training
python3 launch_sagemaker.py \
    --data-s3 s3://my-bucket-name/mlm-data/ \
    --output-s3 s3://my-bucket-name/output/ \
    --model s \
    --epochs 150
```

**That's it!** Training will run on cloud GPU. ☁️

---

## 📋 Detailed Steps

### Step 1: AWS Setup (5 min, one-time)

```bash
# Install AWS tools
pip install boto3 sagemaker awscli

# Configure credentials
aws configure
# Enter:
#   AWS Access Key ID: [your-key]
#   AWS Secret Access Key: [your-secret]
#   Region: us-east-1
#   Output format: json

# Verify
aws sts get-caller-identity
```

**Getting AWS Credentials:**
1. Go to AWS Console → IAM → Users
2. Click your username → Security Credentials
3. Create Access Key → Download credentials
4. Use in `aws configure`

### Step 2: Upload Data (5 min)

```bash
cd ml_pipeline

# Convert data (if not done) and upload to S3
./upload_to_s3.sh my-bucket-name

# With custom data directory
./upload_to_s3.sh my-bucket-name /path/to/mlm/data
```

**What this does:**
- ✅ Creates S3 bucket if needed
- ✅ Converts annotations to YOLO format (if needed)
- ✅ Uploads ~200 images + labels to S3
- ✅ Verifies upload successful

### Step 3: Launch Training (< 1 min)

```bash
# Basic launch (recommended)
python3 launch_sagemaker.py \
    --data-s3 s3://my-bucket-name/mlm-data/ \
    --output-s3 s3://my-bucket-name/output/ \
    --model s \
    --epochs 150

# Or with custom settings
python3 launch_sagemaker.py \
    --data-s3 s3://my-bucket-name/mlm-data/ \
    --output-s3 s3://my-bucket-name/output/ \
    --instance ml.g4dn.2xlarge \
    --model m \
    --epochs 200 \
    --batch-size 16 \
    --wait  # Wait and stream logs
```

**Output will show:**
- Job name (save this!)
- Monitoring URL
- Commands to check status

---

## 📊 Monitor Training

### Option 1: CloudWatch Logs (Real-time)

```bash
# Replace with your job name from launch output
JOB_NAME="pytorch-training-2024-11-04-12-34-56-789"

# Stream logs
aws logs tail /aws/sagemaker/TrainingJobs/$JOB_NAME --follow
```

### Option 2: AWS Console (Web UI)

1. Go to: https://console.aws.amazon.com/sagemaker/
2. Click "Training" → "Training jobs"
3. Find your job
4. View metrics, logs, status

### Option 3: Check Status (Command Line)

```bash
aws sagemaker describe-training-job --training-job-name $JOB_NAME | grep TrainingJobStatus
```

---

## 📥 Download Trained Model

### After Training Completes (~15-20 hours)

```bash
# Get job name from sagemaker_job_info.txt or launch output
JOB_NAME="pytorch-training-2024-11-04-12-34-56-789"

# Download model
aws s3 sync s3://my-bucket-name/output/$JOB_NAME/output/ ./trained_models/

# Extract
cd trained_models
tar -xzf model.tar.gz

# Your trained model!
ls -lh best.pt
```

### Use Model for Inference

```bash
cd ../scripts/inference

python3 inference_pipeline.py \
    --model ../../trained_models/best.pt \
    --input /path/to/sample.tif \
    --output ../../results/analysis
```

---

## 💰 Cost Estimate

| Instance | GPU | Cost/hr | 20hr | Spot (~70% off) |
|----------|-----|---------|------|-----------------|
| ml.g4dn.xlarge | T4 16GB | $0.736 | $14.72 | **$4.42** |
| ml.g4dn.2xlarge | T4 16GB | $0.94 | $18.80 | **$5.64** |
| ml.p3.2xlarge | V100 16GB | $3.82 | $76.40 | $22.92 |

**Recommended**: ml.g4dn.xlarge with spot = **~$5 total** 🎉

Our script uses spot instances by default!

---

## 🎯 What Model Size to Use?

| Model | Speed | Accuracy | GPU Memory | Recommended For |
|-------|-------|----------|------------|-----------------|
| **n** (nano) | Fastest | Good | 4GB | Quick testing |
| **s** (small) | Fast | Better | 8GB | **Production** ✓ |
| **m** (medium) | Medium | Best | 12GB | Max accuracy |
| **l/x** (large) | Slow | Excellent | 16GB+ | If s not enough |

**Start with 's'** - best balance of speed and accuracy.

---

## 🐛 Troubleshooting

### "NoSuchBucket" Error

```bash
# Create bucket manually
aws s3 mb s3://my-bucket-name
```

### "No module named 'sagemaker'"

```bash
pip install boto3 sagemaker awscli
```

### "Credentials could not be loaded"

```bash
aws configure
# Re-enter credentials
```

### Training Fails Immediately

```bash
# Check logs
aws logs tail /aws/sagemaker/TrainingJobs/$JOB_NAME

# Common issues:
# - Data not in S3 (run upload_to_s3.sh)
# - Wrong S3 path (check s3://bucket/mlm-data/yolo_dataset/)
# - Insufficient quota (request limit increase in AWS console)
```

### CUDA Out of Memory

```bash
# Use smaller batch size
python3 launch_sagemaker.py \
    --data-s3 ... \
    --output-s3 ... \
    --batch-size 4  # Reduced from 8
```

---

## ✅ Complete Workflow Checklist

### Initial Setup (One-time)
- [ ] AWS account created
- [ ] AWS CLI installed (`pip install awscli`)
- [ ] AWS credentials configured (`aws configure`)
- [ ] SageMaker libraries installed (`pip install sagemaker boto3`)

### Before Each Training Run
- [ ] Data converted to YOLO format
- [ ] Data uploaded to S3 (`./upload_to_s3.sh`)
- [ ] S3 paths verified

### Launch Training
- [ ] Run `launch_sagemaker.py` with correct parameters
- [ ] Note down job name
- [ ] Save `sagemaker_job_info.txt`

### During Training
- [ ] Monitor logs periodically
- [ ] Check training is progressing (loss decreasing)
- [ ] Estimate completion time

### After Training
- [ ] Download model from S3
- [ ] Extract `model.tar.gz`
- [ ] Verify `best.pt` exists
- [ ] Run inference to test

---

## 📚 File Reference

| File | Purpose |
|------|---------|
| `upload_to_s3.sh` | Upload data to S3 |
| `launch_sagemaker.py` | Launch training job |
| `scripts/training/train_sagemaker.py` | SageMaker training script |
| `sagemaker_job_info.txt` | Job info (auto-created) |
| `CLOUD_TRAINING.md` | Full cloud training guide |

---

## 💡 Pro Tips

1. **Use Spot Instances**: Save 70% (enabled by default in our script)
2. **Start Small**: Try 10 epochs first to verify everything works
3. **Monitor Early**: Check logs after 5 min to catch errors early
4. **Keep Bucket**: Don't delete S3 bucket between runs
5. **Checkpoint**: SageMaker auto-saves checkpoints every epoch

---

## 🎓 For Team Members

### Who Should Do What?

**Member with AWS access**:
- Configure AWS credentials
- Create S3 bucket
- Launch training job
- Monitor progress

**Everyone else**:
- Prepare data locally
- Help monitor training
- Download and test model
- Run inference

### Sharing Results

After training:
```bash
# Share model weights (not on git!)
# Option 1: Keep in S3
aws s3 cp trained_models/best.pt s3://my-bucket/models/best_model_v1.pt --acl public-read

# Option 2: Download and share via Google Drive
# Then teammates can:
# wget https://drive.google.com/... -O best.pt
```

---

## 📞 Need Help?

1. **Setup issues**: Check CLOUD_TRAINING.md
2. **AWS errors**: Check AWS CloudWatch logs
3. **Training issues**: Same as local (see README.md)
4. **Cost concerns**: Use spot instances + smaller model

---

## 🎉 Success!

Once training completes:

1. ✅ Download model from S3
2. ✅ Run inference on test data
3. ✅ Extract competition deliverables
4. ✅ Celebrate! 🎊

**Cost**: ~$5 for 20 hours of GPU training
**Performance**: Expected mAP50 > 0.70 (6x better than baseline!)

---

**Happy Cloud Training! ☁️🚀**
