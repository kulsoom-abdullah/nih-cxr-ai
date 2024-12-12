"""Traditional ResNet-based model for chest X-ray classification.

This module implements a multi-label classification model for chest X-rays using 
a ResNet50 backbone pre-trained on ImageNet. It includes:
- Proper handling of multi-label classification metrics
- Balanced loss functions for handling class imbalance
- Comprehensive evaluation metrics appropriate for medical applications
- Robust error handling and logging
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.models as models
from torchmetrics import AUROC, F1Score, MetricCollection, Precision, Recall

print(f"Loading model file from: {os.path.abspath(__file__)}")


class TraditionalCXRModel(pl.LightningModule):
    """ResNet50-based model for multi-label chest X-ray classification."""

    def __init__(
        self,
        num_classes: int = 15,
        learning_rate: float = 1e-4,
        dropout_rate: float = 0.0,
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= dropout_rate <= 1:
            raise ValueError("dropout_rate must be between 0 and 1")

        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.class_weights = class_weights

        self.model = self._build_model()
        self.criterion = self._setup_criterion()

        self.train_metrics = self._setup_metrics("train")
        self.val_metrics = self._setup_metrics("val")
        self.test_metrics = self._setup_metrics("test")

    def _build_model(self) -> nn.Module:
        """Constructs ResNet50 model with custom final layer."""
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(model.fc.in_features, self.num_classes),
        )
        return model

    def _setup_criterion(self) -> nn.Module:
        """Configure loss function with optional class weighting."""
        return (
            nn.BCEWithLogitsLoss(pos_weight=self.class_weights)
            if self.class_weights is not None
            else nn.BCEWithLogitsLoss()
        )

    def _setup_metrics(self, prefix: str) -> MetricCollection:
        """Initialize evaluation metrics and move to correct device."""
        metrics = MetricCollection(
            {
                "auroc": AUROC(
                    task="multilabel", num_labels=self.num_classes, average="macro"
                ),
                "f1": F1Score(
                    task="multilabel", num_labels=self.num_classes, average="macro"
                ),
                "precision": Precision(
                    task="multilabel", num_labels=self.num_classes, average="macro"
                ),
                "recall": Recall(
                    task="multilabel", num_labels=self.num_classes, average="macro"
                ),
            },
            prefix=f"{prefix}_",
        )

        # Move metrics to same device as model parameters
        if next(self.parameters()).is_cuda:
            metrics.to("cuda")

        return metrics

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ResNet model."""
        return self.model(x)

    def _compute_step(
        self,
        batch: List[torch.Tensor],
        metrics: MetricCollection,
        prefix: str,
        sync_dist: bool = False,
    ) -> torch.Tensor:
        """Compute forward pass and metrics for a single step."""
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y.float())

        probs = torch.sigmoid(y_hat)
        metric_values = metrics(probs, y)

        self.log(f"{prefix}_loss", loss, prog_bar=True, sync_dist=sync_dist)
        self.log_dict(metric_values, prog_bar=True, sync_dist=sync_dist)

        return loss

    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Execute training step with metrics logging."""
        return self._compute_step(batch, self.train_metrics, "train")

    def validation_step(
        self, batch: List[torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Execute validation step with metrics logging."""
        return self._compute_step(batch, self.val_metrics, "val", sync_dist=True)

    def test_step(
        self, batch: List[torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Execute test step with metrics logging."""
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y.float())

        probs = torch.sigmoid(y_hat)
        self.test_metrics(probs, y)

        return {"loss": loss, "preds": probs, "targets": y}

    def on_test_epoch_end(self) -> Dict[str, float]:
        """Process and log test results."""
        try:
            metrics = self.test_metrics.compute()
            results = {}

            for metric_name in ["auroc", "f1", "precision", "recall"]:
                key = f"test_{metric_name}"
                if key in metrics:
                    try:
                        value = metrics[key]
                        results[f"avg_{metric_name}"] = (
                            value.item() if torch.is_tensor(value) else float(value)
                        )
                    except Exception as e:
                        self.print(f"Error processing {metric_name}: {str(e)}")
                        results[f"avg_{metric_name}"] = 0.0

            self.print("\nTest Results:")
            self.print("-" * 40)
            for key, value in sorted(results.items()):
                self.print(f"{key}: {value:.4f}")

            if self.logger:
                self.logger.log_metrics(results)

            return results

        except Exception as e:
            self.print(f"Error computing metrics: {str(e)}")
            return {
                "avg_auroc": 0.0,
                "avg_f1": 0.0,
                "avg_precision": 0.0,
                "avg_recall": 0.0,
            }

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
