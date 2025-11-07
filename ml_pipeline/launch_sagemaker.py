#!/usr/bin/env python3
"""
Launch SageMaker training job from local machine.

Usage:
    python launch_sagemaker.py \
        --data-s3 s3://my-bucket/mlm-data/ \
        --output-s3 s3://my-bucket/output/ \
        --model s \
        --epochs 150
"""

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from datetime import datetime
import argparse
import sys


def check_aws_credentials():
    """Check if AWS credentials are configured."""
    try:
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print(f"✓ AWS credentials valid")
        print(f"  Account: {identity['Account']}")
        print(f"  User: {identity['Arn']}")
        return True
    except Exception as e:
        print(f"✗ AWS credentials not configured")
        print(f"  Run: aws configure")
        return False


def get_or_create_role(role_arn=None):
    """Get or create SageMaker execution role."""
    # If role ARN explicitly provided, use it
    if role_arn:
        print(f"✓ Using provided role: {role_arn}")
        return role_arn

    # Try to get execution role (works when running inside SageMaker)
    try:
        session = sagemaker.Session()
        role = sagemaker.get_execution_role()
        print(f"✓ Using existing role: {role}")
        return role
    except Exception:
        print(f"⚠️  No execution role found")
        print(f"  Please create a SageMaker execution role and pass it with --role")
        print(f"  Example: --role arn:aws:iam::032552343956:role/SageMakerExecutionRole")
        return None


def launch_training(
    data_s3_path: str,
    output_s3_path: str,
    instance_type: str = 'ml.g4dn.2xlarge',
    model_size: str = 's',
    epochs: int = 150,
    batch_size: int = 8,
    use_spot: bool = False,
    wait: bool = False,
    role_arn: str = None
):
    """Launch SageMaker training job."""

    # Check credentials
    if not check_aws_credentials():
        sys.exit(1)

    print("\n" + "="*80)
    print("LAUNCHING SAGEMAKER TRAINING JOB")
    print("="*80)

    # Initialize SageMaker session
    try:
        session = sagemaker.Session()
        region = session.boto_region_name
        print(f"Region: {region}")
    except Exception as e:
        print(f"✗ Failed to create SageMaker session: {e}")
        sys.exit(1)

    # Get execution role
    role = get_or_create_role(role_arn)
    if role is None:
        sys.exit(1)

    print("\nConfiguration:")
    print("-" * 80)
    print(f"  Instance type: {instance_type}")
    print(f"  Model: YOLOv8{model_size}-seg")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Use spot instances: {use_spot}")
    print(f"  Data: {data_s3_path}")
    print(f"  Output: {output_s3_path}")
    print("-" * 80)

    # Verify data exists in S3
    print("\nVerifying S3 data...")
    s3 = boto3.client('s3')
    try:
        # Parse S3 path
        if data_s3_path.startswith('s3://'):
            bucket = data_s3_path.split('/')[2]
            prefix = '/'.join(data_s3_path.split('/')[3:])
        else:
            print(f"✗ Invalid S3 path: {data_s3_path}")
            sys.exit(1)

        # Check if data exists
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=5)
        if 'Contents' in response:
            print(f"✓ Found data in S3")
            for obj in response['Contents'][:5]:
                print(f"    {obj['Key']}")
        else:
            print(f"⚠️  Warning: No files found at {data_s3_path}")
            print(f"  Make sure you've uploaded your data first")
    except Exception as e:
        print(f"⚠️  Could not verify S3 data: {e}")

    # Create estimator
    print("\nCreating PyTorch estimator...")

    # Determine max run time (in seconds)
    max_run = epochs * 300  # ~5 min per epoch estimate
    max_run = min(max_run, 72000)  # Cap at 20 hours

    estimator = PyTorch(
        entry_point='train_sagemaker.py',
        source_dir='scripts/training',
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
            'device': '0',
            'patience': 50
        },
        output_path=output_s3_path,
        code_location=output_s3_path,
        max_run=max_run,
        keep_alive_period_in_seconds=1800,  # Keep warm for 30 min
        volume_size=50,  # GB
        checkpoint_s3_uri=output_s3_path + '/checkpoints',
        use_spot_instances=use_spot,
        max_wait=max_run + 3600 if use_spot else None,  # Add 1hr buffer for spot
        enable_sagemaker_metrics=True,
        metric_definitions=[
            {'Name': 'train:loss', 'Regex': 'train/loss: ([0-9\\.]+)'},
            {'Name': 'val:loss', 'Regex': 'val/loss: ([0-9\\.]+)'},
            {'Name': 'metrics/mAP50', 'Regex': 'metrics/mAP50\\(B\\): ([0-9\\.]+)'},
            {'Name': 'metrics/mAP50-95', 'Regex': 'metrics/mAP50-95\\(B\\): ([0-9\\.]+)'},
        ]
    )

    # Start training
    print("\nStarting training job...")
    try:
        estimator.fit(
            inputs={
                'training': data_s3_path
            },
            wait=wait,
            logs='All' if wait else False
        )

        job_name = estimator.latest_training_job.name

        print("\n" + "="*80)
        print("TRAINING JOB LAUNCHED SUCCESSFULLY")
        print("="*80)
        print(f"Job name: {job_name}")
        print(f"\nMonitor at:")
        print(f"  https://console.aws.amazon.com/sagemaker/home?region={region}#/jobs/{job_name}")
        print(f"\nView logs:")
        print(f"  aws logs tail /aws/sagemaker/TrainingJobs/{job_name} --follow")
        print(f"\nCheck status:")
        print(f"  aws sagemaker describe-training-job --training-job-name {job_name}")
        print(f"\nAfter completion, download model:")
        print(f"  aws s3 sync {output_s3_path}/{job_name}/output/ ./trained_models/")
        print("="*80 + "\n")

        # Save job info
        with open('sagemaker_job_info.txt', 'w') as f:
            f.write(f"Job Name: {job_name}\n")
            f.write(f"Region: {region}\n")
            f.write(f"Output S3: {output_s3_path}/{job_name}/\n")
            f.write(f"\nCommands:\n")
            f.write(f"Monitor: aws logs tail /aws/sagemaker/TrainingJobs/{job_name} --follow\n")
            f.write(f"Status: aws sagemaker describe-training-job --training-job-name {job_name}\n")
            f.write(f"Download: aws s3 sync {output_s3_path}/{job_name}/output/ ./trained_models/\n")

        print("Job info saved to: sagemaker_job_info.txt\n")

        return estimator

    except Exception as e:
        print(f"\n✗ Failed to launch training job: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Launch YOLOv8 training on AWS SageMaker',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic launch
  python launch_sagemaker.py \\
      --data-s3 s3://my-bucket/mlm-data/ \\
      --output-s3 s3://my-bucket/output/

  # With custom settings
  python launch_sagemaker.py \\
      --data-s3 s3://my-bucket/mlm-data/ \\
      --output-s3 s3://my-bucket/output/ \\
      --instance ml.g4dn.2xlarge \\
      --model m \\
      --epochs 200 \\
      --batch-size 16

  # Wait for completion
  python launch_sagemaker.py \\
      --data-s3 s3://my-bucket/mlm-data/ \\
      --output-s3 s3://my-bucket/output/ \\
      --wait
        """
    )

    # Required arguments
    parser.add_argument(
        '--data-s3',
        required=True,
        help='S3 path to training data (e.g., s3://bucket/mlm-data/)'
    )
    parser.add_argument(
        '--output-s3',
        required=True,
        help='S3 path for output (e.g., s3://bucket/output/)'
    )

    # Optional arguments
    parser.add_argument(
        '--instance',
        default='ml.g4dn.2xlarge',
        help='SageMaker instance type (default: ml.g4dn.2xlarge, T4 16GB GPU, $0.94/hr)'
    )
    parser.add_argument(
        '--model',
        default='s',
        choices=['n', 's', 'm', 'l', 'x'],
        help='YOLOv8 model size (default: s)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=150,
        help='Number of training epochs (default: 150)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size (default: 8)'
    )
    parser.add_argument(
        '--spot',
        action='store_true',
        help='Enable spot instances (70%% cheaper but may be interrupted - requires quota)'
    )
    parser.add_argument(
        '--wait',
        action='store_true',
        help='Wait for training to complete (will stream logs)'
    )
    parser.add_argument(
        '--role',
        type=str,
        default=None,
        help='IAM role ARN for SageMaker execution (e.g., arn:aws:iam::123456789012:role/SageMakerExecutionRole)'
    )

    args = parser.parse_args()

    # Launch training
    estimator = launch_training(
        data_s3_path=args.data_s3,
        output_s3_path=args.output_s3,
        instance_type=args.instance,
        model_size=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_spot=args.spot,
        wait=args.wait,
        role_arn=args.role
    )

    if not args.wait:
        print("Tip: Run with --wait flag to stream logs and wait for completion\n")


if __name__ == '__main__':
    main()
