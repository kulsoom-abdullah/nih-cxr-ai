# src/nih_cxr_ai/models/traditional.py
"""Traditional ResNet-based model for chest X-ray classification.

This module implements a multi-label classification model for chest X-rays using
a ResNet50 backbone pre-trained on ImageNet. It includes:
- Proper handling of multi-label classification metrics
- Balanced loss functions for handling class imbalance
- Comprehensive evaluation metrics appropriate for medical applications
- Robust error handling and logging
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.models as models
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import AUROC, F1Score, MetricCollection, Precision, Recall


class TraditionalCXRModel(pl.LightningModule):
    """
    CheXNet-style model for multi-label chest X-ray classification.
    Based on EfficientNet architecture with custom classification head.
    """

    def __init__(
        self,
        num_classes: int = 15,
        learning_rate: float = 1e-4,
        dropout_rate: float = 0.0,
        class_weights: Optional[torch.Tensor] = None,
        disease_names: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        # Log the incoming learning rate value and its type
        print(
            f"Initializing model with learning rate: {learning_rate} (type: {type(learning_rate)})"
        )
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.class_weights = class_weights
        try:
            self.learning_rate = float(learning_rate)
            print(f"Converted learning rate to float: {self.learning_rate}")
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Could not convert learning rate to float: {learning_rate}"
            ) from e

        self.save_hyperparameters()
        self.num_classes = num_classes
        self.disease_names = (
            disease_names
            if disease_names
            else [f"Disease_{i}" for i in range(num_classes)]
        )
        self.num_classes = num_classes

        self.model = self._build_model()
        self.criterion = self._setup_criterion()
        # Replace the old metric initialization with per-class metrics
        self.train_metrics = self._setup_metrics("train")
        self.val_metrics = self._setup_metrics("val")
        self.test_metrics = self._setup_metrics("test")

        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= dropout_rate <= 1:
            raise ValueError("dropout_rate must be between 0 and 1")

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
                    task="multilabel", num_labels=self.num_classes, average=None
                ),
                "f1": F1Score(
                    task="multilabel", num_labels=self.num_classes, average=None
                ),
                "precision": Precision(
                    task="multilabel", num_labels=self.num_classes, average=None
                ),
                "recall": Recall(
                    task="multilabel", num_labels=self.num_classes, average=None
                ),
            }
        )
        return metrics

        # Move metrics to same device as model parameters
        if next(self.parameters()).is_cuda:
            metrics.to("cuda")

        return metrics

    def _create_metrics(self, prefix: str) -> MetricCollection:
        """Create a collection of metrics for tracking model performance."""
        return MetricCollection(
            {
                "auroc": AUROC(
                    task="multilabel", num_labels=self.num_classes, average=None
                ),
                "f1": F1Score(
                    task="multilabel", num_labels=self.num_classes, average=None
                ),
                "precision": Precision(
                    task="multilabel", num_labels=self.num_classes, average=None
                ),
                "recall": Recall(
                    task="multilabel", num_labels=self.num_classes, average=None
                ),
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ResNet model."""
        return self.model(x)

    # def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
    #     """Training step processing a single batch."""
    #     x, y = batch
    #     y_hat = self(x)
    #     loss = self.criterion(y_hat, y.float())

    #     # Calculate and log metrics
    #     probs = torch.sigmoid(y_hat)
    #     metric_values = metrics(probs, y)

    #     self.log(f"{prefix}_loss", loss, prog_bar=True, sync_dist=sync_dist)

    #     for metric_name, values in metric_values.items():
    #         # Check if values is a vector
    #         if values.ndim > 0 and values.numel() > 1:
    #             # It's a vector of per-class metrics. Log mean and/or per-class.
    #             mean_val = values.mean()
    #             self.log(
    #                 f"{prefix}_{metric_name}_mean",
    #                 mean_val.item(),
    #                 prog_bar=True,
    #                 sync_dist=sync_dist,
    #             )

    #             # If you want per-class logging
    #             for i, d_name in enumerate(self.disease_names):
    #                 self.log(
    #                     f"{prefix}_{d_name}_{metric_name}",
    #                     values[i].item(),
    #                     prog_bar=False,
    #                     sync_dist=sync_dist,
    #                 )
    #         else:
    #             # It's a scalar
    #             self.log(
    #                 f"{prefix}_{metric_name}",
    #                 values.item(),
    #                 prog_bar=True,
    #                 sync_dist=sync_dist,
    #             )

    #     return loss

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

        for metric_name, values in metric_values.items():
            # Check if values is a vector
            if values.ndim > 0 and values.numel() > 1:
                # It's a vector of per-class metrics. Log mean and/or per-class.
                mean_val = values.mean()
                self.log(
                    f"{prefix}_{metric_name}_mean",
                    mean_val.item(),
                    prog_bar=True,
                    sync_dist=sync_dist,
                )

                # If you want per-class logging
                for i, d_name in enumerate(self.disease_names):
                    self.log(
                        f"{prefix}_{d_name}_{metric_name}",
                        values[i].item(),
                        prog_bar=False,
                        sync_dist=sync_dist,
                    )
            else:
                # It's a scalar
                self.log(
                    f"{prefix}_{metric_name}",
                    values.item(),
                    prog_bar=True,
                    sync_dist=sync_dist,
                )

        return loss

    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Execute training step with metrics logging."""
        return self._compute_step(batch, self.train_metrics, "train")

    def validation_step(
        self, batch: List[torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Execute validation step with metrics logging."""
        return self._compute_step(batch, self.val_metrics, "val", sync_dist=True)

    def test_step(self, batch, batch_idx):
        images, labels, ages, genders, positions = batch
        logits = self(images)
        preds = torch.sigmoid(logits)

        return {
            "preds": preds.detach(),
            "labels": labels.detach(),
            "ages": ages.detach() if torch.is_tensor(ages) else ages,
            "genders": genders,  # strings, just return as is
            "positions": positions,  # strings, just return as is
        }

    def test_epoch_end(self, outputs):
        """Process and log test results."""
        # 'outputs' is a list of dicts returned by test_step
        all_preds = torch.cat([x["preds"] for x in outputs], dim=0)
        all_labels = torch.cat([x["labels"] for x in outputs], dim=0)

        # ages/genders/positions might be lists of various types
        all_ages = torch.tensor(
            [x for out in outputs for x in out["ages"]], dtype=torch.int
        )
        all_genders = [x for out in outputs for x in out["genders"]]
        all_positions = [x for out in outputs for x in out["positions"]]

        # Compute overall per-disease metrics
        metrics = self.test_metrics(all_preds, all_labels)
        auroc_values = metrics["auroc"]
        f1_values = metrics["f1"]
        precision_values = metrics["precision"]
        recall_values = metrics["recall"]

        # Log mean values
        self.log("test_auroc_mean", metrics["test_auroc_mean"].item())
        self.log("test_f1_mean", metrics["test_f1_mean"].item())
        self.log("test_precision_mean", metrics["test_precision_mean"].item())
        self.log("test_recall_mean", metrics["test_recall_mean"].item())

        # Log per-disease metrics
        for i, d_name in enumerate(self.disease_names):
            self.log(f"test_{d_name}_auroc", auroc_values[i].item())
            self.log(f"test_{d_name}_f1", f1_values[i].item())
            self.log(f"test_{d_name}_precision", precision_values[i].item())
            self.log(f"test_{d_name}_recall", recall_values[i].item())

        # Subgroup analyses
        # Example: Age brackets
        young_mask = all_ages < 40
        middle_mask = (all_ages >= 40) & (all_ages < 60)
        old_mask = all_ages >= 60

        # Function to compute subgroup metrics given a mask
        def compute_subgroup_metrics(mask, prefix):
            if mask.sum() == 0:
                return  # no samples in this group
            sub_preds = all_preds[mask]
            sub_labels = all_labels[mask]
            sub_metrics = self.test_metrics(sub_preds, sub_labels)
            self.log(f"test_{prefix}_auroc_mean", sub_metrics["test_auroc_mean"].item())
            self.log(f"test_{prefix}_f1_mean", sub_metrics["test_f1_mean"].item())
            self.log(
                f"test_{prefix}_precision_mean",
                sub_metrics["test_precision_mean"].item(),
            )
            self.log(
                f"test_{prefix}_recall_mean", sub_metrics["test_recall_mean"].item()
            )
            # If you want per-disease per-subgroup metrics, loop again similarly:
            for i, d_name in enumerate(self.disease_names):
                self.log(
                    f"test_{prefix}_{d_name}_auroc", sub_metrics["auroc"][i].item()
                )

        # Compute subgroup metrics for each age bracket
        compute_subgroup_metrics(young_mask, "young")
        compute_subgroup_metrics(middle_mask, "middle_aged")
        compute_subgroup_metrics(old_mask, "old")

        # Similarly for gender
        all_genders = np.array(all_genders)
        male_mask = all_genders == "M"
        female_mask = all_genders == "F"

        compute_subgroup_metrics(torch.tensor(male_mask), "male")
        compute_subgroup_metrics(torch.tensor(female_mask), "female")

        # For positions
        all_positions = np.array(all_positions)
        pa_mask = all_positions == "PA"
        ap_mask = all_positions == "AP"

        compute_subgroup_metrics(torch.tensor(pa_mask), "PA_view")
        compute_subgroup_metrics(torch.tensor(ap_mask), "AP_view")

        # All logs go to W&B automatically since we're using Lightning + W&B logger

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(
            optimizer, mode="max", factor=0.1, patience=5, verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_auroc",
                "interval": "epoch",
            },
        }
