# src/nih_cxr_ai/models/traditional.py
"""Traditional ResNet-based model for chest X-ray classification.

This module implements a multi-label classification model for chest X-rays using 
a ResNet50 backbone pre-trained on ImageNet. 
"""

# Standard library imports
from typing import Dict, Any, Optional, Union, List

# Third-party imports
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.models as models
from torchmetrics import MetricCollection, AUROC, Precision, Recall, F1Score

class TraditionalCXRModel(pl.LightningModule):
    """ResNet50-based model for multi-label chest X-ray classification."""
    
    def __init__(
        self, 
        num_classes: int = 15, 
        learning_rate: Union[float, str] = 1e-4,
        dropout_rate: Union[float, str] = 0.0
    ) -> None:
        """Initialize the model.
        
        Args:
            num_classes: Number of disease classes to predict
            learning_rate: Initial learning rate for optimization
            dropout_rate: Dropout rate for regularization
            
        Raises:
            ValueError: If parameters are invalid
        """
        super().__init__()
        
        # Convert and validate numeric parameters
        try:
            self.num_classes = int(num_classes)
            self.learning_rate = float(learning_rate)
            self.dropout_rate = float(dropout_rate)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Invalid parameter type. All parameters must be numeric. Error: {str(e)}"
            )
            
        # Validate parameter values
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= self.dropout_rate <= 1:
            raise ValueError("dropout_rate must be between 0 and 1")
            
        self.save_hyperparameters()
        
        # Model initialization
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.model.fc.in_features, self.num_classes)
        )
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Initialize metrics
        metrics = MetricCollection({
        'auroc': AUROC(task="multilabel", num_labels=num_classes),
        'precision': Precision(task="multilabel", num_labels=num_classes),
        'recall': Recall(task="multilabel", num_labels=num_classes),
        'f1': F1Score(task="multilabel", num_labels=num_classes)
    })

        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Model predictions before sigmoid activation
        """
        return self.model(x)

    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step.
        
        Args:
            batch: List containing [images, labels]
            batch_idx: Index of current batch
            
        Returns:
            Computed loss value
        """
        x, y = batch
        y_hat = self(x)
        
        # Calculate loss
        loss = self.criterion(y_hat, y.float())
        
        # Calculate metrics
        with torch.no_grad():
            probs = torch.sigmoid(y_hat)
            metrics = self.train_metrics(probs, y)
            
            # Log metrics
            self.log('train_loss', loss, prog_bar=True)
            for name, value in metrics.items():
                self.log(name, value, prog_bar=True)
        
        return loss

    def validation_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step.
        
        Args:
            batch: List containing [images, labels]
            batch_idx: Index of current batch
            
        Returns:
            Computed loss value
        """
        x, y = batch
        y_hat = self(x)
        
        # Calculate loss
        loss = self.criterion(y_hat, y.float())
        
        # Calculate metrics
        probs = torch.sigmoid(y_hat)
        metrics = self.val_metrics(probs, y)
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        for name, value in metrics.items():
            self.log(name, value, prog_bar=True, sync_dist=True)
        
        return loss

    def test_step(self, batch: List[torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Test step.
        
        Args:
            batch: List containing [images, labels]
            batch_idx: Index of current batch
            
        Returns:
            Dictionary containing loss, predictions and targets
        """
        x, y = batch
        y_hat = self(x)
        
        # Calculate loss
        loss = self.criterion(y_hat, y.float())
        
        # Get predictions
        probs = torch.sigmoid(y_hat)
        
        # Calculate metrics
        metrics = self.test_metrics(probs, y)
        self.log_dict(metrics)
        
        return {
            'loss': loss,
            'preds': probs,
            'targets': y
        }

    def on_test_epoch_end(self) -> Dict[str, float]:
        """Compute and log final test metrics.
        
        Returns:
            Dictionary containing final metrics
        """
        metrics = self.test_metrics.compute()
        
        # Calculate and log detailed metrics
        disease_names = [f"Class_{i}" for i in range(15)]  # Could be parametrized
        results = {}
        
        for i, disease in enumerate(disease_names):
            results[f"{disease}_auroc"] = metrics['test_auroc'][i].item()
            results[f"{disease}_f1"] = metrics['test_f1'][i].item()
        
        # Calculate averages
        results['avg_auroc'] = metrics['test_auroc'].mean().item()
        results['avg_f1'] = metrics['test_f1'].mean().item()
        
        # Print summary
        self.print("\nFinal Model Performance:")
        self.print("-----------------------")
        self.print(f"Average AUROC: {results['avg_auroc']:.3f}")
        self.print(f"Average F1: {results['avg_f1']:.3f}")
        
        # Log to whatever logger is configured
        self.logger.log_metrics(results)
        
        return results

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and learning rate scheduler.
        
        Returns:
            Dictionary containing optimizer and scheduler configuration
        """
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
# # src/models/traditional.py

# import torch
# import pytorch_lightning as pl
# import torch.nn as nn
# import torchvision.models as models
# from torchmetrics import MetricCollection, AUROC, Precision, Recall, F1Score


# class TraditionalCXRModel(pl.LightningModule):
#     def __init__(self, num_classes: int = 15, learning_rate: float = 1e-4):
#         super().__init__()
#         self.save_hyperparameters()
#         self.learning_rate = learning_rate
        
#         # Model initialization
#         self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
#         self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
#         # Loss function
#         self.criterion = nn.BCEWithLogitsLoss()
        
#         # Initialize metrics for both training and validation
#         metrics = MetricCollection({
#             'auroc': AUROC(task="multilabel", num_labels=num_classes),
#             'precision': Precision(task="multilabel", num_labels=num_classes),
#             'recall': Recall(task="multilabel", num_labels=num_classes),
#             'f1': F1Score(task="multilabel", num_labels=num_classes)
#         })
        
#         self.train_metrics = metrics.clone(prefix='train_')
#         self.val_metrics = metrics.clone(prefix='val_')
#         self.test_metrics = metrics.clone(prefix='test_')  

#     def forward(self, x):
#         return self.model(x)

#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self(x)
        
#         # Calculate loss
#         loss = self.criterion(y_hat, y.float())
        
#         # Calculate metrics
#         with torch.no_grad():
#             probs = torch.sigmoid(y_hat)
#             metrics = self.train_metrics(probs, y)
            
#             # Log metrics
#             self.log('train_loss', loss, prog_bar=True)
#             self.log('train_auroc', metrics['train_auroc'], prog_bar=True)
#             self.log('train_f1', metrics['train_f1'], prog_bar=True)
        
#         return loss

#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self(x)
        
#         # Calculate loss
#         loss = self.criterion(y_hat, y.float())
        
#         # Calculate metrics
#         probs = torch.sigmoid(y_hat)
#         metrics = self.val_metrics(probs, y)
        
#         # Log metrics
#         self.log('val_loss', loss, prog_bar=True)
#         self.log('val_auroc', metrics['val_auroc'], prog_bar=True)
#         self.log('val_f1', metrics['val_f1'], prog_bar=True)
        
#         return loss

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#             optimizer, mode="min", factor=0.1, patience=5, verbose=True
#         )
#         return {
#             "optimizer": optimizer,
#             "lr_scheduler": {
#                 "scheduler": scheduler,
#                 "monitor": "val_loss",
#                 "interval": "epoch",
#                 "frequency": 1,
#             },
#         }

#     def test_step(self, batch, batch_idx):
#         """Perform a single test step on a batch of data."""
#         x, y = batch
#         y_hat = self(x)
        
#         # Calculate loss
#         loss = self.criterion(y_hat, y.float())
        
#         # Get predictions
#         probs = torch.sigmoid(y_hat)
        
#         # Return predictions and targets for epoch end processing
#         return {
#             'loss': loss,
#             'preds': probs,
#             'targets': y
#         }

#     def on_test_epoch_end(self):
#         """Compute and log final test metrics."""
#         # Get the test results that Lightning has accumulated
#         metrics = self.test_metrics.compute()  # Assuming you've added test_metrics in __init__
        
#         # Calculate and log detailed metrics
#         disease_names = [f"Class_{i}" for i in range(15)]
#         results = {}
        
#         for i, disease in enumerate(disease_names):
#             results[f"{disease}_auroc"] = metrics['auroc'][i]
#             results[f"{disease}_f1"] = metrics['f1'][i]
        
#         # Calculate averages
#         results['avg_auroc'] = metrics['auroc'].mean()
#         results['avg_f1'] = metrics['f1'].mean()
        
#         # Print summary
#         print("\nFinal Model Performance:")
#         print("-----------------------")
#         print(f"Average AUROC: {results['avg_auroc']:.3f}")
#         print(f"Average F1: {results['avg_f1']:.3f}")
        
#         # Log to wandb
#         self.logger.log_metrics(results)
        
#         return results