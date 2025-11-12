"""
Visualization utilities for Alt ML Pipeline 1
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from PIL import Image
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class Visualizer:
    """
    Visualization utilities for training, predictions, and analysis
    """

    def __init__(self, output_dir: Path, class_names: List[str]):
        """
        Initialize visualizer

        Args:
            output_dir: Output directory for visualizations
            class_names: List of class names
        """
        self.output_dir = Path(output_dir)
        self.class_names = class_names
        self.num_classes = len(class_names)

        # Create subdirectories
        self.vis_dirs = {
            'training': self.output_dir / 'visualizations' / 'training_progress',
            'predictions': self.output_dir / 'visualizations' / 'predictions',
            'confusion': self.output_dir / 'visualizations' / 'confusion_matrices',
            'data_quality': self.output_dir / 'visualizations' / 'data_quality'
        }

        for dir_path in self.vis_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

        # Color palette for classes
        self.colors = sns.color_palette("husl", self.num_classes)

    def plot_training_curves(self, metrics_history: Dict, epoch: int, save_name: str = "training_curves.png"):
        """
        Plot training and validation loss/metrics over time

        Args:
            metrics_history: Dictionary with metrics lists (train_loss, val_loss, etc.)
            epoch: Current epoch
            save_name: Filename for saving
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Loss', 'F1 Score', 'Precision', 'Recall'),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )

        epochs = list(range(1, epoch + 1))

        # Loss
        if 'train_loss' in metrics_history:
            fig.add_trace(
                go.Scatter(x=epochs, y=metrics_history['train_loss'], name='Train Loss',
                          line=dict(color='blue')),
                row=1, col=1
            )
        if 'val_loss' in metrics_history:
            fig.add_trace(
                go.Scatter(x=epochs, y=metrics_history['val_loss'], name='Val Loss',
                          line=dict(color='red')),
                row=1, col=1
            )

        # F1 Score
        if 'train_f1' in metrics_history:
            fig.add_trace(
                go.Scatter(x=epochs, y=metrics_history['train_f1'], name='Train F1',
                          line=dict(color='blue'), showlegend=False),
                row=1, col=2
            )
        if 'val_f1' in metrics_history:
            fig.add_trace(
                go.Scatter(x=epochs, y=metrics_history['val_f1'], name='Val F1',
                          line=dict(color='red'), showlegend=False),
                row=1, col=2
            )

        # Precision
        if 'train_precision' in metrics_history:
            fig.add_trace(
                go.Scatter(x=epochs, y=metrics_history['train_precision'], name='Train Precision',
                          line=dict(color='blue'), showlegend=False),
                row=2, col=1
            )
        if 'val_precision' in metrics_history:
            fig.add_trace(
                go.Scatter(x=epochs, y=metrics_history['val_precision'], name='Val Precision',
                          line=dict(color='red'), showlegend=False),
                row=2, col=1
            )

        # Recall
        if 'train_recall' in metrics_history:
            fig.add_trace(
                go.Scatter(x=epochs, y=metrics_history['train_recall'], name='Train Recall',
                          line=dict(color='blue'), showlegend=False),
                row=2, col=2
            )
        if 'val_recall' in metrics_history:
            fig.add_trace(
                go.Scatter(x=epochs, y=metrics_history['val_recall'], name='Val Recall',
                          line=dict(color='red'), showlegend=False),
                row=2, col=2
            )

        fig.update_xaxes(title_text="Epoch")
        fig.update_yaxes(title_text="Value")
        fig.update_layout(height=800, width=1200, title_text="Training Progress")

        output_path = self.vis_dirs['training'] / save_name.replace('.png', '.html')
        fig.write_html(str(output_path))
        print(f"Training curves saved to: {output_path}")

    def plot_confusion_matrix(self, confusion_matrix: np.ndarray, save_name: str = "confusion_matrix.png"):
        """
        Plot confusion matrix

        Args:
            confusion_matrix: Confusion matrix array (num_classes x num_classes)
            save_name: Filename for saving
        """
        fig, ax = plt.subplots(figsize=(12, 10))

        # Normalize to percentages
        cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)

        # Plot
        sns.heatmap(cm_normalized, annot=confusion_matrix, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   ax=ax, cbar_kws={'label': 'Percentage'})

        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('True', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        output_path = self.vis_dirs['confusion'] / save_name
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Confusion matrix saved to: {output_path}")

    def visualize_predictions(self, images: List[np.ndarray], ground_truths: List[Dict],
                             predictions: List[Dict], save_name: str = "predictions.png",
                             n_samples: int = 16):
        """
        Visualize predictions vs ground truth side by side

        Args:
            images: List of images (numpy arrays)
            ground_truths: List of ground truth dictionaries (boxes, labels, masks)
            predictions: List of prediction dictionaries (boxes, labels, scores, masks)
            save_name: Filename for saving
            n_samples: Number of samples to visualize
        """
        n_samples = min(n_samples, len(images))
        n_cols = 4
        n_rows = (n_samples + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for idx in range(n_samples):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            # Get image, GT, and predictions
            img = images[idx].copy()
            gt = ground_truths[idx] if idx < len(ground_truths) else {}
            pred = predictions[idx] if idx < len(predictions) else {}

            # Draw ground truth (green boxes)
            if 'boxes' in gt:
                for box_idx, box in enumerate(gt['boxes']):
                    x1, y1, x2, y2 = box
                    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                            linewidth=2, edgecolor='green',
                                            facecolor='none', linestyle='--')
                    ax.add_patch(rect)

                    # Add label
                    if 'labels' in gt and box_idx < len(gt['labels']):
                        label = self.class_names[gt['labels'][box_idx]]
                        ax.text(x1, y1 - 5, label, color='green',
                               fontsize=8, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

            # Draw predictions (red boxes)
            if 'boxes' in pred:
                for box_idx, box in enumerate(pred['boxes']):
                    x1, y1, x2, y2 = box
                    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                            linewidth=2, edgecolor='red', facecolor='none')
                    ax.add_patch(rect)

                    # Add label with confidence
                    if 'labels' in pred and box_idx < len(pred['labels']):
                        label = self.class_names[pred['labels'][box_idx]]
                        conf = pred['scores'][box_idx] if 'scores' in pred and box_idx < len(pred['scores']) else 0.0
                        ax.text(x2, y1 - 5, f'{label} {conf:.2f}', color='red',
                               fontsize=8, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f'Image {idx + 1}', fontsize=10)

        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='green', linestyle='--', linewidth=2, label='Ground Truth'),
            Line2D([0], [0], color='red', linewidth=2, label='Prediction')
        ]
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.98),
                  ncol=2, fontsize=12)

        plt.tight_layout()
        output_path = self.vis_dirs['predictions'] / save_name
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Predictions visualization saved to: {output_path}")

    def plot_per_class_metrics(self, per_class_metrics: Dict, save_name: str = "per_class_metrics.png"):
        """
        Plot per-class precision, recall, F1

        Args:
            per_class_metrics: Dictionary with per-class metrics
            save_name: Filename for saving
        """
        metrics = ['precision', 'recall', 'f1']
        data = {metric: [] for metric in metrics}

        for class_idx in range(self.num_classes):
            for metric in metrics:
                value = per_class_metrics.get(f'class_{class_idx}_{metric}', 0.0)
                data[metric].append(value)

        # Create bar plot
        x = np.arange(self.num_classes)
        width = 0.25

        fig, ax = plt.subplots(figsize=(14, 6))

        for i, metric in enumerate(metrics):
            offset = width * (i - 1)
            ax.bar(x + offset, data[metric], width, label=metric.capitalize())

        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        output_path = self.vis_dirs['training'] / save_name
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Per-class metrics saved to: {output_path}")

    def plot_class_distribution(self, class_counts: Dict, save_name: str = "class_distribution.png"):
        """
        Plot class distribution bar chart

        Args:
            class_counts: Dictionary mapping class names to counts
            save_name: Filename for saving
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        total = sum(counts)

        # Calculate percentages
        percentages = [count / total * 100 for count in counts]

        # Create bar plot
        bars = ax.bar(range(len(classes)), counts, color=self.colors[:len(classes)])

        # Add percentage labels
        for i, (bar, pct) in enumerate(zip(bars, percentages)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                   f'{counts[i]}\n({pct:.1f}%)',
                   ha='center', va='bottom', fontsize=10)

        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Class Distribution in Dataset', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        output_path = self.vis_dirs['data_quality'] / save_name
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Class distribution saved to: {output_path}")

    def create_comparison_report(self, baseline_metrics: Dict, new_metrics: Dict,
                                 save_name: str = "comparison_report.html"):
        """
        Create HTML report comparing baseline vs new model

        Args:
            baseline_metrics: Baseline model metrics
            new_metrics: New model metrics
            save_name: Filename for saving
        """
        metrics_to_compare = ['f1', 'precision', 'recall', 'mAP50']

        # Create comparison table
        fig = go.Figure(data=[go.Table(
            header=dict(values=['Metric', 'Baseline', 'New Model', 'Improvement'],
                       fill_color='paleturquoise',
                       align='left'),
            cells=dict(values=[
                [m.upper() for m in metrics_to_compare],
                [f"{baseline_metrics.get(m, 0.0):.4f}" for m in metrics_to_compare],
                [f"{new_metrics.get(m, 0.0):.4f}" for m in metrics_to_compare],
                [f"{(new_metrics.get(m, 0.0) - baseline_metrics.get(m, 0.0)):.4f} ({((new_metrics.get(m, 0.0) - baseline_metrics.get(m, 0.0)) / baseline_metrics.get(m, 0.01) * 100):.1f}%)"
                 for m in metrics_to_compare]
            ],
            fill_color='lavender',
            align='left'))
        ])

        fig.update_layout(title='Model Comparison Report', height=400)

        output_path = self.vis_dirs['training'] / save_name
        fig.write_html(str(output_path))
        print(f"Comparison report saved to: {output_path}")

    def save_metrics_json(self, metrics: Dict, save_name: str = "metrics.json"):
        """
        Save metrics to JSON file

        Args:
            metrics: Dictionary of metrics
            save_name: Filename for saving
        """
        output_path = self.output_dir / 'results' / save_name
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"Metrics saved to: {output_path}")


if __name__ == "__main__":
    # Test visualizer
    output_dir = Path(os.path.expanduser("~/mlm_outputs/alt_pipeline_1"))
    class_names = ["planktonic", "single_dispersed", "hyphae", "clump_dispersed",
                   "yeast", "biofilm", "pseudohyphae"]

    viz = Visualizer(output_dir, class_names)

    # Test training curves
    metrics_history = {
        'train_loss': [0.5 - 0.01 * i for i in range(50)],
        'val_loss': [0.55 - 0.008 * i for i in range(50)],
        'train_f1': [0.3 + 0.01 * i for i in range(50)],
        'val_f1': [0.25 + 0.009 * i for i in range(50)],
    }
    viz.plot_training_curves(metrics_history, epoch=50)

    # Test confusion matrix
    cm = np.random.randint(0, 100, size=(7, 7))
    viz.plot_confusion_matrix(cm)

    # Test class distribution
    class_counts = {name: np.random.randint(100, 1000) for name in class_names}
    viz.plot_class_distribution(class_counts)

    print("\nVisualization tests complete!")
