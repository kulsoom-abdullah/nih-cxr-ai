# src/nih_cxr_ai/utils/visualization/performance_viz.py
"""Model performance visualization utilities.

Tools for visualizing model metrics, including ROC curves, training progress,
and performance comparisons across different pathologies.
"""

# Standard library imports
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, roc_curve


class PerformanceVisualizer:
    """Visualizer for model performance metrics and training progress."""

    def __init__(
        self, save_dir: Optional[Path] = None, fig_size: Tuple[int, int] = (12, 8)
    ) -> None:
        """Initialize performance visualizer.

        Args:
            save_dir: Directory to save visualizations. Created if doesn't exist.
            fig_size: Default figure size for plots (width, height) in inches.
        """
        self.save_dir = Path(save_dir) if save_dir else Path("results/visualizations")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.fig_size = fig_size

        # Remove seaborn style setting and use a built-in style instead
        plt.style.use("default")  # Using default matplotlib style

        # Define default disease names - matches NIH dataset
        self.disease_names = [
            "Atelectasis",
            "Cardiomegaly",
            "Effusion",
            "Infiltration",
            "Mass",
            "Nodule",
            "Pneumonia",
            "Pneumothorax",
            "Consolidation",
            "Edema",
            "Emphysema",
            "Fibrosis",
            "Pleural_Thickening",
            "Hernia",
        ]

        # Set up color scheme
        self.colors = plt.cm.tab20(np.linspace(0, 1, len(self.disease_names)))

    def plot_roc_curves(
        self, y_true: np.ndarray, y_pred: np.ndarray, save_name: Optional[str] = None
    ) -> None:
        """Plot ROC curves for multi-label classification.

        Args:
            y_true: Ground truth labels, shape (n_samples, n_classes)
            y_pred: Predicted probabilities, shape (n_samples, n_classes)
            save_name: Optional filename to save the plot
        """
        plt.figure(figsize=self.fig_size)

        for i, disease_name in enumerate(self.disease_names):
            fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(
                fpr,
                tpr,
                label=f"{disease_name} (AUC = {roc_auc:.2f})",
                color=self.colors[i],
            )

        plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves by Disease Class")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)

        if save_name:
            plt.savefig(self.save_dir / f"{save_name}.png", bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def plot_training_curves(
        self, metrics_dict: Dict[str, List[float]], save_name: Optional[str] = None
    ) -> None:
        """Plot training and validation metrics over epochs.

        Args:
            metrics_dict: Dictionary mapping metric names to lists of values.
                        Keys should contain 'train' or 'val' to distinguish
                        between training and validation metrics.
            save_name: Optional filename to save the plot
        """
        plt.figure(figsize=self.fig_size)

        for metric_name, values in metrics_dict.items():
            if "train" in metric_name:
                plt.plot(values, label=metric_name, linestyle="-")
            elif "val" in metric_name:
                plt.plot(values, label=metric_name, linestyle="--")

        plt.xlabel("Epoch")
        plt.ylabel("Metric Value")
        plt.title("Training Progress")
        plt.legend()
        plt.grid(True)

        if save_name:
            plt.savefig(self.save_dir / f"{save_name}.png", bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def plot_confusion_matrices(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: float = 0.5,
        save_name: Optional[str] = None,
    ) -> None:
        """Plot confusion matrices for each disease class.

        Args:
            y_true: Ground truth labels, shape (n_samples, n_classes)
            y_pred: Predicted probabilities, shape (n_samples, n_classes)
            threshold: Classification threshold for predictions
            save_name: Optional filename to save the plot
        """
        n_classes = len(self.disease_names)
        fig, axes = plt.subplots(
            (n_classes + 3) // 4, 4, figsize=(20, 5 * ((n_classes + 3) // 4))
        )
        axes = axes.ravel()

        y_pred_binary = (y_pred >= threshold).astype(int)

        for idx, (disease_name, ax) in enumerate(zip(self.disease_names, axes)):
            cm = confusion_matrix(y_true[:, idx], y_pred_binary[:, idx])
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=ax,
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"],
            )
            ax.set_title(disease_name)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")

        plt.tight_layout()

        if save_name:
            plt.savefig(self.save_dir / f"{save_name}.png", bbox_inches="tight")
            plt.close()
        else:
            plt.show()
