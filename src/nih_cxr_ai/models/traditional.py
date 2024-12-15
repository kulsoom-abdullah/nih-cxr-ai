from typing import Dict, Optional

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
        num_classes: int = 14,
        learning_rate: float = 1e-4,
        class_weights: Optional[torch.Tensor] = None,
        backbone: str = "efficientnet_b1",
        test_mode: bool = False,
    ):
        """Initialize model with specified backbone and training parameters.

        Args:
            num_classes: Number of disease classes to predict
            learning_rate: Initial learning rate
            class_weights: Optional tensor of class weights for loss function
            backbone: Model architecture to use ('efficientnet_b1' by default)
            test_mode: Whether to run in test mode with limited data
        """
        super().__init__()
        # Log the incoming learning rate value and its type
        print(
            f"Initializing model with learning rate: {learning_rate} (type: {type(learning_rate)})"
        )

        self.test_mode = test_mode
        try:
            self.learning_rate = float(learning_rate)
            print(f"Converted learning rate to float: {self.learning_rate}")
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Could not convert learning rate to float: {learning_rate}"
            ) from e

        self.save_hyperparameters()
        self.num_classes = num_classes

        # Load pretrained backbone
        if backbone == "efficientnet_b1":
            base_model = models.efficientnet_b1(weights="DEFAULT")
            # Get the number of features from the classifier layer
            num_features = base_model.classifier[1].in_features
            # Remove the original classifier
            self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Custom classifier head with dropout and batch normalization
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes),
        )

        # Loss function with class weights if provided
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)

        # Initialize metrics
        self.train_metrics = self._create_metrics("train_")
        self.val_metrics = self._create_metrics("val_")
        self.test_metrics = self._create_metrics("test_")

        # Per-class metrics for detailed performance tracking
        self.per_class_auroc = nn.ModuleList(
            [AUROC(task="binary") for _ in range(num_classes)]
        )

    def _create_metrics(self, prefix: str) -> MetricCollection:
        """Create a collection of metrics for tracking model performance."""
        return MetricCollection(
            {
                "auroc": AUROC(task="multilabel", num_labels=self.num_classes),
                "f1": F1Score(task="multilabel", num_labels=self.num_classes),
                "precision": Precision(task="multilabel", num_labels=self.num_classes),
                "recall": Recall(task="multilabel", num_labels=self.num_classes),
            },
            prefix=prefix,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Tensor of logits with shape (batch_size, num_classes)
        """
        features = self.feature_extractor(x)
        # Flatten the features
        features = features.flatten(start_dim=1)
        return self.classifier(features)

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Training step processing a single batch."""
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y.float())

        # Calculate and log metrics
        probs = torch.sigmoid(y_hat)
        self.train_metrics(probs, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log_dict(self.train_metrics, prog_bar=True)

        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Validation step processing a single batch."""
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y.float())

        # Calculate metrics
        probs = torch.sigmoid(y_hat)
        self.val_metrics(probs, y)

        # Calculate per-class performance
        for i, metric in enumerate(self.per_class_auroc):
            metric(probs[:, i], y[:, i])

        # Log metrics
        self.log("val_loss", loss, prog_bar=True)
        self.log_dict(self.val_metrics, prog_bar=True)

        return loss

    def configure_optimizers(self):
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
