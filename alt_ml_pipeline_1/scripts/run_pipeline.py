"""
Alt ML Pipeline 1 - Main Orchestration Script
End-to-end pipeline execution
"""

import os
import sys
from pathlib import Path
import yaml
import argparse
from datetime import datetime
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def print_header(title):
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_stage(stage_name, stage_num, total_stages):
    """Print stage information"""
    print(f"\n{'─' * 80}")
    print(f"  Stage {stage_num}/{total_stages}: {stage_name}")
    print(f"{'─' * 80}\n")


def run_phase_1_foundation(config_path, args):
    """
    Phase 1: Foundation - Setup and baseline
    """
    print_header("PHASE 1: FOUNDATION")

    from src.data.data_loader import YOLODataLoader
    from src.training.train_baseline import BaselineTrainer

    # Stage 1: Data loading and exploration
    print_stage("Data Loading & Analysis", 1, 3)
    loader = YOLODataLoader(config_path)
    loader.print_dataset_summary()
    loader.visualize_sample_images(n_samples=3)

    # Stage 2: Create data splits
    print_stage("Creating Cross-Validation Splits", 2, 3)
    splits = loader.create_leave_one_sequence_out_splits()
    print(f"\nCreated {len(splits)} folds for cross-validation")
    for split in splits:
        print(f"  Fold {split['fold']}: Val sequence = {split['val_sequence']}, "
              f"Train images = {len(split['train_images'])}, "
              f"Val images = {len(split['val_images'])}")

    # Stage 3: Train baseline model
    if not args.skip_baseline:
        print_stage("Training Baseline Model", 3, 3)
        trainer = BaselineTrainer(config_path, fold_idx=args.fold)
        trainer.prepare_data()
        results, model_path = trainer.train()

        if args.visualize:
            trainer.visualize_results(model_path)

        print(f"\n✓ Baseline model trained successfully!")
        print(f"  F1 Score: {results['f1']:.4f}")
        print(f"  mAP50: {results['mAP50']:.4f}")
        print(f"  Model: {model_path}")

        return results
    else:
        print("\n⊗ Skipping baseline training (--skip-baseline flag)")
        return None


def run_phase_2_augmentation(config_path, args):
    """
    Phase 2: Data Augmentation - Synthetic data generation
    """
    print_header("PHASE 2: DATA AUGMENTATION")

    # TODO: Implement in future
    print("⚠ Synthetic data generation not yet implemented")
    print("  This phase will include:")
    print("    - Extract cell bank from annotations")
    print("    - Generate 10,000 synthetic frames")
    print("    - Implement microscopy-realistic augmentations")
    print("    - Validate synthetic vs real data distributions")


def run_phase_3_ssl(config_path, args):
    """
    Phase 3: Self-Supervised Learning
    """
    print_header("PHASE 3: SELF-SUPERVISED LEARNING")

    # TODO: Implement in future
    print("⚠ Self-supervised learning not yet implemented")
    print("  This phase will include:")
    print("    - Temporal pretext tasks (frame ordering, future prediction)")
    print("    - Spatial pretext tasks (rotation, contrastive learning)")
    print("    - Train encoder on all 205 frames without labels")
    print("    - Visualize learned embeddings")


def run_phase_4_ensemble(config_path, args):
    """
    Phase 4: Ensemble & Advanced Methods
    """
    print_header("PHASE 4: ENSEMBLE & ADVANCED METHODS")

    # TODO: Implement in future
    print("⚠ Ensemble training not yet implemented")
    print("  This phase will include:")
    print("    - Train 10-15 diverse models")
    print("    - Implement snapshot ensemble")
    print("    - Test-time augmentation")
    print("    - Biological constraints")


def run_phase_5_evaluation(config_path, args):
    """
    Phase 5: Comprehensive Evaluation
    """
    print_header("PHASE 5: COMPREHENSIVE EVALUATION")

    # TODO: Implement in future
    print("⚠ Comprehensive evaluation not yet implemented")
    print("  This phase will include:")
    print("    - 5-fold cross-validation")
    print("    - Per-class metrics analysis")
    print("    - Failure analysis")
    print("    - Generate final report")


def main():
    """
    Main pipeline orchestration
    """
    parser = argparse.ArgumentParser(description='Alt ML Pipeline 1 - Main orchestration script')

    # General arguments
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--phase', type=str, default='all',
                       choices=['all', 'foundation', 'augmentation', 'ssl', 'ensemble', 'evaluation'],
                       help='Which phase to run')

    # Phase-specific arguments
    parser.add_argument('--fold', type=int, default=0,
                       help='Fold index for training (0-4)')
    parser.add_argument('--skip-baseline', action='store_true',
                       help='Skip baseline training in foundation phase')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualizations')

    args = parser.parse_args()

    # Print pipeline header
    print_header("ALT ML PIPELINE 1 - CANDIDA ALBICANS MORPHOLOGY DETECTION")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config: {args.config}")
    print(f"Phase: {args.phase}")
    print(f"Fold: {args.fold}")

    # Load config
    config_path = args.config
    if not Path(config_path).exists():
        print(f"\n✗ Error: Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create output directory
    output_dir = Path(os.path.expandvars(config['project']['output_dir']))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run phases
    results = {}

    if args.phase == 'all' or args.phase == 'foundation':
        results['phase1'] = run_phase_1_foundation(config_path, args)

    if args.phase == 'all' or args.phase == 'augmentation':
        results['phase2'] = run_phase_2_augmentation(config_path, args)

    if args.phase == 'all' or args.phase == 'ssl':
        results['phase3'] = run_phase_3_ssl(config_path, args)

    if args.phase == 'all' or args.phase == 'ensemble':
        results['phase4'] = run_phase_4_ensemble(config_path, args)

    if args.phase == 'all' or args.phase == 'evaluation':
        results['phase5'] = run_phase_5_evaluation(config_path, args)

    # Save pipeline results
    results_path = output_dir / 'results' / 'pipeline_results.json'
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert results to JSON-serializable format
    json_results = {}
    for phase, result in results.items():
        if result is not None:
            if isinstance(result, dict):
                json_results[phase] = {k: float(v) if isinstance(v, (int, float)) else str(v)
                                      for k, v in result.items()}

    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)

    # Print final summary
    print_header("PIPELINE EXECUTION COMPLETE")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {results_path}")

    if 'phase1' in results and results['phase1'] is not None:
        print(f"\nBaseline Performance:")
        print(f"  F1 Score: {results['phase1']['f1']:.4f}")
        print(f"  mAP50: {results['phase1']['mAP50']:.4f}")

    print("\nNext steps:")
    print("  1. Review visualizations in: ~/mlm_outputs/alt_pipeline_1/visualizations/")
    print("  2. Check MLflow UI: mlflow ui --backend-store-uri ~/mlm_outputs/alt_pipeline_1/experiments/mlflow")
    print("  3. Implement Phase 2 (synthetic data generation) for data augmentation")
    print("  4. Iterate and improve based on baseline results\n")


if __name__ == "__main__":
    main()
