#!/bin/bash
# Setup SageMaker for Alt ML Pipeline 1

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "=========================================="
echo "SageMaker Setup for Alt ML Pipeline 1"
echo "=========================================="
echo ""

# Check AWS CLI
echo "Checking AWS CLI..."
if ! command -v aws &> /dev/null; then
    echo -e "${RED}✗ AWS CLI not found${NC}"
    echo ""
    echo "Install with:"
    echo "  brew install awscli"
    echo "  # OR"
    echo "  pip install awscli"
    exit 1
fi
echo -e "${GREEN}✓ AWS CLI found: $(aws --version)${NC}"

# Check AWS credentials
echo ""
echo "Checking AWS credentials..."
if ! aws sts get-caller-identity &> /dev/null; then
    echo -e "${RED}✗ AWS credentials not configured${NC}"
    echo ""
    echo "Configure with:"
    echo "  aws configure"
    echo ""
    echo "You'll need:"
    echo "  - AWS Access Key ID"
    echo "  - AWS Secret Access Key"
    exit 1
fi

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo -e "${GREEN}✓ AWS credentials configured${NC}"
echo "  Account ID: $ACCOUNT_ID"

# Check for SageMaker role
echo ""
echo "Checking SageMaker execution role..."

ROLE_NAME="SageMakerExecutionRole"
ROLE_ARN=$(aws iam get-role --role-name $ROLE_NAME --query 'Role.Arn' --output text 2>/dev/null || echo "")

if [ -z "$ROLE_ARN" ]; then
    echo -e "${YELLOW}⚠ SageMaker role not found${NC}"
    echo ""
    read -p "Create SageMakerExecutionRole now? (y/n) " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Creating SageMaker execution role..."

        # Create role
        aws iam create-role \
            --role-name $ROLE_NAME \
            --assume-role-policy-document '{
              "Version": "2012-10-17",
              "Statement": [{
                "Effect": "Allow",
                "Principal": {"Service": "sagemaker.amazonaws.com"},
                "Action": "sts:AssumeRole"
              }]
            }' > /dev/null

        # Attach policies
        aws iam attach-role-policy \
            --role-name $ROLE_NAME \
            --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

        aws iam attach-role-policy \
            --role-name $ROLE_NAME \
            --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

        # Get role ARN
        ROLE_ARN=$(aws iam get-role --role-name $ROLE_NAME --query 'Role.Arn' --output text)

        echo -e "${GREEN}✓ Role created: $ROLE_ARN${NC}"
    else
        echo ""
        echo "Please create the role manually:"
        echo "  1. Go to https://console.aws.amazon.com/iam/"
        echo "  2. Create role for SageMaker"
        echo "  3. Name it: $ROLE_NAME"
        exit 1
    fi
else
    echo -e "${GREEN}✓ SageMaker role exists${NC}"
    echo "  Role ARN: $ROLE_ARN"
fi

# Add to environment
echo ""
echo "Adding SAGEMAKER_ROLE to environment..."

SHELL_RC="$HOME/.zshrc"
if [ ! -f "$SHELL_RC" ]; then
    SHELL_RC="$HOME/.bashrc"
fi

if ! grep -q "SAGEMAKER_ROLE" "$SHELL_RC" 2>/dev/null; then
    echo "" >> "$SHELL_RC"
    echo "# SageMaker execution role" >> "$SHELL_RC"
    echo "export SAGEMAKER_ROLE=\"$ROLE_ARN\"" >> "$SHELL_RC"
    echo -e "${GREEN}✓ Added to $SHELL_RC${NC}"
else
    echo -e "${YELLOW}⚠ Already in $SHELL_RC${NC}"
fi

# Export for current session
export SAGEMAKER_ROLE="$ROLE_ARN"

# Check Python packages
echo ""
echo "Checking Python packages..."
if ! python -c "import sagemaker" &> /dev/null; then
    echo -e "${YELLOW}⚠ sagemaker package not installed${NC}"
    echo ""
    read -p "Install sagemaker package now? (y/n) " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pip install sagemaker boto3
        echo -e "${GREEN}✓ Packages installed${NC}"
    fi
else
    echo -e "${GREEN}✓ sagemaker package installed${NC}"
fi

# Test SageMaker access
echo ""
echo "Testing SageMaker access..."
if aws sagemaker list-training-jobs --max-items 1 &> /dev/null; then
    echo -e "${GREEN}✓ SageMaker access confirmed${NC}"
else
    echo -e "${RED}✗ Cannot access SageMaker${NC}"
    echo "  Check IAM permissions"
    exit 1
fi

# Summary
echo ""
echo "=========================================="
echo -e "${GREEN}✓ SageMaker Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "Environment variables set:"
echo "  SAGEMAKER_ROLE=$ROLE_ARN"
echo ""
echo "Next steps:"
echo "  1. Reload shell: source $SHELL_RC"
echo "  2. Launch training: python sagemaker/launch_sagemaker_training.py --fold 0"
echo ""
echo "Cost estimate:"
echo "  ~$0.15 per training run (with spot instances)"
echo "  ~$0.75 for all 5 folds"
echo ""
