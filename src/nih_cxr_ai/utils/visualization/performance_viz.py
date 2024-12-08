
# src/nih_cxr_ai/utils/visualization/performance_viz.py
"""Model performance visualization utilities.

Tools for visualizing model metrics, including ROC curves, training progress,
and performance comparisons across different pathologies.
"""

# Standard library imports
from typing import Dict, List, Tuple, Union

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

class PerformanceVisualizer:
    """Visualizer for model performance metrics and training progress."""
    
    def __init__(self, fig_size: Tuple[int, int] = (12, 8)) -> None:
        """Initialize performance visualizer.
        
        Args:
            fig_size: Default figure size for plots
        """
        self.fig_size = fig_size
        self.colors = plt.cm.tab20(np.linspace(0, 1, 20))

    def plot_roc_curves(self, 
                       y_true: np.ndarray, 
                       y_pred: np.ndarray, 
                       class_names: List[str]) -> plt.Figure:
        """Plot ROC curves for multi-label classification.
        
        Args:
            y_true: Ground truth labels, shape (n_samples, n_classes)
            y_pred: Predicted probabilities, shape (n_samples, n_classes)
            class_names: List of class names for legend
            
        Returns:
            matplotlib Figure object containing the plot
            
        Notes:
            - Plots individual ROC curves for each class
            - Includes AUC scores in legend
            - Shows diagonal reference line
        """
        plt.figure(figsize=self.fig_size)

        for i, class_name in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(
                fpr,
                tpr,
                color=self.colors[i],
                label=f"{class_name} (AUC = {roc_auc:.2f})",
            )

        plt.plot([0, 1], [0, 1], "k--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves per Class")
        plt.legend(loc="lower right", bbox_to_anchor=(1.7, 0.5))
        plt.grid(True)
        plt.tight_layout()
        return plt.gcf()

    def plot_training_curves(self, 
                           metrics_dict: Dict[str, List[float]]) -> plt.Figure:
        """Plot training and validation metrics over epochs.
        
        Args:
            metrics_dict: Dictionary mapping metric names to lists of values.
                        Keys should contain 'train' or 'val' to distinguish
                        between training and validation metrics.
                        
        Returns:
            matplotlib Figure object containing the plot
            
        Examples:
            >>> metrics = {
            ...     'train_loss': [0.5, 0.4, 0.3],
            ...     'val_loss': [0.6, 0.5, 0.4],
            ...     'train_acc': [0.8, 0.85, 0.9],
            ...     'val_acc': [0.75, 0.8, 0.85]
            ... }
            >>> visualizer.plot_training_curves(metrics)
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
        plt.tight_layout()
        return plt.gcf()