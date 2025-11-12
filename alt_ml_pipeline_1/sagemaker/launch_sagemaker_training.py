"""
Launch Training Job on AWS SageMaker
Run this script locally to start a SageMaker training job
"""

import os
import sys
import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from pathlib import Path
import yaml
import argparse
from datetime import datetime
import tarfile
import shutil


class SageMakerLauncher:
    """
    Launch training jobs on AWS SageMaker
    """

    def __init__(self, config_path: str, fold_idx: int = 0):
        """
        Initialize SageMaker launcher

        Args:
            config_path: Path to config.yaml
            fold_idx: Which fold to train (0-4)
        """
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.fold_idx = fold_idx

        # SageMaker config
        self.sm_config = self.config.get('sagemaker', {})
        self.instance_type = self.sm_config.get('instance_type', 'ml.g4dn.2xlarge')
        self.volume_size = self.sm_config.get('volume_size', 50)
        self.max_run_time = self.sm_config.get('max_run_time', 72000)  # 20 hours
        self.use_spot = self.sm_config.get('use_spot_instances', True)

        # AWS session
        self.session = sagemaker.Session()
        self.role = sagemaker.get_execution_role() if self._is_sagemaker_notebook() else self._get_role()
        self.bucket = self.session.default_bucket()

        print(f"SageMaker Launcher Initialized")
        print(f"  Role: {self.role}")
        print(f"  Bucket: {self.bucket}")
        print(f"  Instance: {self.instance_type}")
        print(f"  Spot instances: {self.use_spot}")
        print()

    def _is_sagemaker_notebook(self) -> bool:
        """Check if running inside SageMaker notebook"""
        return os.path.exists('/opt/ml/metadata/resource-metadata.json')

    def _get_role(self) -> str:
        """
        Get SageMaker execution role

        Returns:
            IAM role ARN
        """
        # Try to get from environment
        role = os.environ.get('SAGEMAKER_ROLE')
        if role:
            return role

        # Try to get default role
        try:
            iam = boto3.client('iam')
            roles = iam.list_roles()['Roles']

            # Look for SageMaker role
            for role_info in roles:
                if 'SageMaker' in role_info['RoleName']:
                    return role_info['Arn']

            raise ValueError("No SageMaker role found")

        except Exception as e:
            print(f"Error getting role: {e}")
            print("\nPlease either:")
            print("  1. Set SAGEMAKER_ROLE environment variable")
            print("  2. Create a SageMaker execution role in IAM console")
            print("  3. Provide role ARN via --role argument")
            raise

    def prepare_data(self, output_dir: Path) -> str:
        """
        Prepare and upload data to S3

        Args:
            output_dir: Local output directory

        Returns:
            S3 URI of uploaded data
        """
        print("Preparing data for SageMaker...")

        # Create temporary directory for data packaging
        temp_dir = output_dir / 'temp_sagemaker_data'
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Get fold data
        from src.data.data_loader import YOLODataLoader
        loader = YOLODataLoader(str(Path('configs/config.yaml')))
        splits = loader.create_leave_one_sequence_out_splits()

        # Prepare fold data
        dataset_yaml_path = loader.prepare_fold(self.fold_idx, splits)
        print(f"  Fold {self.fold_idx} prepared at: {dataset_yaml_path}")

        # Copy fold data to temp directory
        fold_dir = Path(dataset_yaml_path).parent
        dest_dir = temp_dir / 'data'

        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.copytree(fold_dir, dest_dir)

        print(f"  Data copied to: {dest_dir}")

        # Create tarball
        tarball_path = output_dir / f'fold_{self.fold_idx}_data.tar.gz'
        print(f"  Creating tarball: {tarball_path}")

        with tarfile.open(tarball_path, 'w:gz') as tar:
            tar.add(dest_dir, arcname='.')

        # Upload to S3
        s3_prefix = f'alt-pipeline-1/training-data/fold_{self.fold_idx}'
        s3_uri = f's3://{self.bucket}/{s3_prefix}/data.tar.gz'

        print(f"  Uploading to S3: {s3_uri}")
        self.session.upload_data(
            path=str(tarball_path),
            bucket=self.bucket,
            key_prefix=s3_prefix
        )

        # Cleanup
        shutil.rmtree(temp_dir)
        tarball_path.unlink()

        print(f"  ✓ Data uploaded successfully")
        return f's3://{self.bucket}/{s3_prefix}'

    def launch_training(self, data_s3_uri: str) -> None:
        """
        Launch SageMaker training job

        Args:
            data_s3_uri: S3 URI of training data
        """
        print("\nLaunching SageMaker training job...")

        # Training parameters from config
        train_config = self.config['training']['baseline']

        # Create PyTorch estimator
        estimator = PyTorch(
            entry_point='train_sagemaker.py',
            source_dir='sagemaker',
            role=self.role,
            instance_count=1,
            instance_type=self.instance_type,
            framework_version='2.1.0',
            py_version='py310',
            volume_size=self.volume_size,
            max_run=self.max_run_time,
            use_spot_instances=self.use_spot,
            max_wait=self.max_run_time + 3600 if self.use_spot else None,
            hyperparameters={
                'model': train_config['model'],
                'epochs': train_config['epochs'],
                'batch-size': train_config['batch_size'] * 2,  # Larger on GPU
                'image-size': train_config['image_size'],
                'patience': train_config['patience'],
                'optimizer': self.config['training']['optimizer']['name'],
                'learning-rate': self.config['training']['optimizer']['lr'],
                'weight-decay': self.config['training']['optimizer']['weight_decay'],
                'device': 'auto',
                'workers': 8
            },
            environment={
                'WANDB_DISABLED': 'true',
                'WANDB_MODE': 'disabled'
            },
            output_path=f's3://{self.bucket}/alt-pipeline-1/model-output',
            code_location=f's3://{self.bucket}/alt-pipeline-1/code',
            base_job_name=f'alt-pipeline-fold-{self.fold_idx}'
        )

        # Start training
        job_name = f'alt-pipeline-fold-{self.fold_idx}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'

        print(f"\nJob name: {job_name}")
        print(f"Instance: {self.instance_type}")
        print(f"Data: {data_s3_uri}")
        print("\nStarting training...")
        print("-" * 80)

        estimator.fit({'training': data_s3_uri}, job_name=job_name, wait=True)

        print("-" * 80)
        print("\n✓ Training complete!")
        print(f"\nModel artifacts: {estimator.model_data}")
        print(f"Training job: {estimator.latest_training_job.name}")

    def download_results(self, job_name: str, output_dir: Path) -> None:
        """
        Download results from S3

        Args:
            job_name: Training job name
            output_dir: Local directory to save results
        """
        print(f"\nDownloading results...")

        # Download model artifacts
        s3_client = boto3.client('s3')
        model_s3_uri = f's3://{self.bucket}/alt-pipeline-1/model-output/{job_name}/output/model.tar.gz'

        output_dir.mkdir(parents=True, exist_ok=True)
        local_path = output_dir / 'model.tar.gz'

        print(f"  Downloading: {model_s3_uri}")
        s3_client.download_file(
            self.bucket,
            f'alt-pipeline-1/model-output/{job_name}/output/model.tar.gz',
            str(local_path)
        )

        # Extract
        print(f"  Extracting to: {output_dir}")
        with tarfile.open(local_path, 'r:gz') as tar:
            tar.extractall(output_dir)

        local_path.unlink()
        print(f"  ✓ Results downloaded")


def main():
    """
    Main entry point
    """
    parser = argparse.ArgumentParser(description='Launch SageMaker training')

    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--fold', type=int, default=0,
                       help='Fold index (0-4)')
    parser.add_argument('--role', type=str, default=None,
                       help='SageMaker execution role ARN')
    parser.add_argument('--download-only', action='store_true',
                       help='Only download results from previous job')
    parser.add_argument('--job-name', type=str, default=None,
                       help='Job name for download')

    args = parser.parse_args()

    # Set role if provided
    if args.role:
        os.environ['SAGEMAKER_ROLE'] = args.role

    # Initialize launcher
    launcher = SageMakerLauncher(args.config, args.fold)

    if args.download_only:
        if not args.job_name:
            print("Error: --job-name required for download")
            sys.exit(1)

        output_dir = Path.home() / 'mlm_outputs' / 'alt_pipeline_1' / 'sagemaker_results'
        launcher.download_results(args.job_name, output_dir)

    else:
        # Prepare and upload data
        output_dir = Path.home() / 'mlm_outputs' / 'alt_pipeline_1'
        data_s3_uri = launcher.prepare_data(output_dir)

        # Launch training
        launcher.launch_training(data_s3_uri)

        print("\n" + "=" * 80)
        print("SageMaker Job Launched Successfully!")
        print("=" * 80)
        print("\nMonitor your job:")
        print("  1. AWS Console: https://console.aws.amazon.com/sagemaker/home#/jobs")
        print("  2. Or use: aws sagemaker describe-training-job --training-job-name <job-name>")
        print("\nEstimated cost: ~$0.94/hour for ml.g4dn.2xlarge")
        print("Expected duration: ~30-60 minutes")
        print()


if __name__ == '__main__':
    main()
