#!/bin/bash
# Upload data and code to S3 for SageMaker training

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "================================================================"
echo "Upload Data to S3 for SageMaker Training"
echo "================================================================"
echo ""

# Check if bucket name provided
if [ -z "$1" ]; then
    echo -e "${RED}Error: S3 bucket name required${NC}"
    echo ""
    echo "Usage: ./upload_to_s3.sh <bucket-name> [data-directory]"
    echo ""
    echo "Example:"
    echo "  ./upload_to_s3.sh my-sagemaker-bucket"
    echo "  ./upload_to_s3.sh my-bucket /Users/aineshmohan/Documents/mlm"
    echo ""
    exit 1
fi

BUCKET=$1
DATA_DIR=${2:-"/Users/aineshmohan/Documents/mlm"}

echo "Configuration:"
echo "  S3 Bucket: s3://$BUCKET"
echo "  Data Directory: $DATA_DIR"
echo ""

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo -e "${RED}Error: AWS CLI not installed${NC}"
    echo "Install with: pip install awscli"
    exit 1
fi

# Check if AWS credentials configured
if ! aws sts get-caller-identity &> /dev/null; then
    echo -e "${RED}Error: AWS credentials not configured${NC}"
    echo "Run: aws configure"
    exit 1
fi

echo -e "${GREEN}✓ AWS CLI configured${NC}"
echo ""

# Create bucket if it doesn't exist
echo "[1/4] Checking S3 bucket..."
if aws s3 ls "s3://$BUCKET" 2>&1 | grep -q 'NoSuchBucket'; then
    echo "  Creating bucket..."
    aws s3 mb "s3://$BUCKET"
    echo -e "  ${GREEN}✓ Bucket created${NC}"
else
    echo -e "  ${GREEN}✓ Bucket exists${NC}"
fi
echo ""

# Convert data if not already done
echo "[2/4] Checking processed data..."
if [ ! -d "data/processed/yolo_dataset" ]; then
    echo -e "  ${YELLOW}⚠️  Processed data not found${NC}"
    echo "  Converting annotations..."

    cd scripts/data_prep
    python3 convert_all_annotations.py \
        --data-dir "$DATA_DIR" \
        --output-dir ../../data/processed/yolo_dataset
    cd ../..

    echo -e "  ${GREEN}✓ Data converted${NC}"
else
    echo -e "  ${GREEN}✓ Processed data exists${NC}"
fi
echo ""

# Upload processed dataset
echo "[3/4] Uploading dataset to S3..."
echo "  This may take a few minutes..."

aws s3 sync data/processed/yolo_dataset "s3://$BUCKET/mlm-data/yolo_dataset/" \
    --exclude "*.DS_Store" \
    --no-progress

echo -e "  ${GREEN}✓ Dataset uploaded${NC}"
echo ""

# Verify upload
echo "[4/4] Verifying upload..."
echo "  Checking dataset.yaml..."
if aws s3 ls "s3://$BUCKET/mlm-data/yolo_dataset/dataset.yaml" &> /dev/null; then
    echo -e "    ${GREEN}✓ dataset.yaml found${NC}"
else
    echo -e "    ${RED}✗ dataset.yaml not found${NC}"
    exit 1
fi

echo "  Checking images..."
IMAGE_COUNT=$(aws s3 ls "s3://$BUCKET/mlm-data/yolo_dataset/images/" --recursive | wc -l)
echo -e "    ${GREEN}✓ Found $IMAGE_COUNT image files${NC}"

echo "  Checking labels..."
LABEL_COUNT=$(aws s3 ls "s3://$BUCKET/mlm-data/yolo_dataset/labels/" --recursive | wc -l)
echo -e "    ${GREEN}✓ Found $LABEL_COUNT label files${NC}"

echo ""
echo "================================================================"
echo "UPLOAD COMPLETE"
echo "================================================================"
echo ""
echo "S3 Data Location: s3://$BUCKET/mlm-data/"
echo ""
echo "Next steps:"
echo ""
echo "1. Launch training:"
echo "   python3 launch_sagemaker.py \\"
echo "       --data-s3 s3://$BUCKET/mlm-data/ \\"
echo "       --output-s3 s3://$BUCKET/sagemaker-output/ \\"
echo "       --model s \\"
echo "       --epochs 150"
echo ""
echo "2. Monitor training:"
echo "   aws logs tail /aws/sagemaker/TrainingJobs/<job-name> --follow"
echo ""
echo "3. After completion, download model:"
echo "   aws s3 sync s3://$BUCKET/sagemaker-output/<job-name>/output/ ./trained_models/"
echo ""
echo "================================================================"
