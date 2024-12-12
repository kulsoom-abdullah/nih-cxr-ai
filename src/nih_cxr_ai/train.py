"""Training script for chest X-ray classification models.

This module provides the main training loop and utilities for training deep learning
models on the NIH Chest X-ray dataset using PyTorch Lightning. It handles configuration
loading, model instantiation, and training execution while providing proper logging
and error handling.
"""

import logging
from argparse import ArgumentParser
from importlib import import_module
from pathlib import Path
from typing import Any, Dict

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import Logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data.nih_dataset import NIHChestDataModule
from .models.traditional import TraditionalCXRModel
from .utils.visualization.image_viz import ImageVisualizer
from .utils.visualization.performance_viz import PerformanceVisualizer

# Configure logging with a more informative format
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def import_class(class_path: str) -> Any:
    """Dynamically import a class from a string path."""
    try:
        module_path, class_name = class_path.rsplit(".", 1)
        module = import_module(module_path)
        return getattr(module, class_name)
    except Exception as e:
        raise ImportError(f"Could not import {class_path}: {str(e)}")


class ChestXRayTrainer:
    """Manages the training process for chest X-ray classification models."""

    def __init__(self, config_path: str) -> None:
        """Initialize trainer with configuration."""
        self.config = self._load_config(config_path)
        self.device = self._setup_device()
        self.visualizer = PerformanceVisualizer(save_dir="results/model_evaluation")
        self.image_viz = ImageVisualizer(save_dir="results/sample_predictions")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load and validate configuration from YAML file."""
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        required_keys = {"model", "data", "trainer"}
        if not all(key in config for key in required_keys):
            raise ValueError(f"Configuration must contain {required_keys}")

        return config

    def _setup_device(self) -> torch.device:
        """Configure and return the training device."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("Using GPU for training")
        else:
            device = torch.device("cpu")
            logger.warning("GPU not available, falling back to CPU")
        return device

    def _instantiate_class(self, config: Dict[str, Any]) -> Any:
        """Instantiate a class from configuration."""
        class_ = import_class(config["class_path"])
        return class_(**config.get("init_args", {}))

    def _setup_trainer(self) -> pl.Trainer:
        """Initialize PyTorch Lightning trainer."""
        trainer_config = self.config["trainer"].copy()

        if "callbacks" in trainer_config:
            callbacks = [
                self._instantiate_class(callback_config)
                for callback_config in trainer_config["callbacks"]
            ]
            trainer_config["callbacks"] = callbacks

        if "logger" in trainer_config:
            trainer_config["logger"] = self._instantiate_class(trainer_config["logger"])

        return pl.Trainer(**trainer_config)

    def evaluate_model(
        self, model: pl.LightningModule, test_dataloader: DataLoader
    ) -> Dict[str, float]:
        """Evaluate model on test dataset.

        Returns:
            Dictionary with metric names and their float values:
            - test_auroc: Area under ROC curve
            - test_f1: F1 score
            - test_precision: Precision score
            - test_recall: Recall score
        """
        model.to(self.device)
        model.eval()

        all_preds = []
        all_labels = []

        # Collect predictions
        with torch.no_grad():
            for images, labels in tqdm(test_dataloader, desc="Evaluating"):
                # Move batch to GPU and get predictions
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = model(images)
                preds = torch.sigmoid(logits)

                # Keep everything on GPU
                all_preds.append(preds)
                all_labels.append(labels)

        # Concatenate predictions while still on GPU
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Compute metrics and convert to Python floats
        metrics = model.test_metrics(all_preds, all_labels)
        return {
            "test_auroc": metrics["test_auroc"].item(),
            "test_f1": metrics["test_f1"].item(),
            "test_precision": metrics["test_precision"].item(),
            "test_recall": metrics["test_recall"].item(),
        }

    def train(self) -> str:
        """Execute the training pipeline."""
        try:
            model = self._instantiate_class(self.config["model"])
            datamodule = self._instantiate_class(self.config["data"])

            # Setup data for all stages
            datamodule.setup(stage="fit")
            datamodule.setup(stage="test")

            trainer = self._setup_trainer()
            logger.info("Starting model training...")
            trainer.fit(model, datamodule)

            logger.info("Training completed. Running evaluation...")
            metrics = self.evaluate_model(model, datamodule.test_dataloader())

            logger.info("Test Results:")
            logger.info(f"Metrics: {metrics}")

            ckpt_path = trainer.checkpoint_callback.best_model_path
            logger.info(f"Best model saved to {ckpt_path}")
            return ckpt_path

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise RuntimeError(f"Training pipeline failed: {str(e)}") from e


def main():
    """Main entry point for training."""
    parser = ArgumentParser(description="Train chest X-ray classification model")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML configuration file"
    )
    args = parser.parse_args()

    try:
        trainer = ChestXRayTrainer(args.config)
        ckpt_path = trainer.train()
        logger.info(f"Training completed successfully. Model saved to: {ckpt_path}")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
# """Training script for chest X-ray classification models.

# This module provides the main training loop and utilities for training deep learning
# models on the NIH Chest X-ray dataset using PyTorch Lightning. It handles configuration
# loading, model instantiation, and training execution while providing proper logging
# and error handling.
# """

# import logging
# import os
# from argparse import ArgumentParser
# from importlib import import_module
# from pathlib import Path
# from typing import Any, Dict, Optional

# import pytorch_lightning as pl
# import torch
# import yaml
# from pytorch_lightning.callbacks import Callback
# from pytorch_lightning.loggers import Logger

# from .data.nih_dataset import NIHChestDataModule
# from .models.traditional import TraditionalCXRModel
# from utils.visualization.performance_viz import PerformanceVisualizer

# # Configure logging with a more informative format
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)


# class ChestXRayTrainer:
#     def train(self) -> str:
#         """Execute the training pipeline."""
#         try:
#             # Initialize model and data module
#             model = self._instantiate_model()
#             datamodule = self._instantiate_datamodule()

#             # Setup and run training
#             trainer = self._setup_trainer()
#             logger.info("Starting model training...")
#             trainer.fit(model, datamodule)

#             # Run evaluation
#             logger.info("Training completed. Running evaluation...")
#             test_metrics = self.evaluate_model(model, datamodule.test_dataloader())

#             # Log evaluation results
#             logger.info("Test Results:")
#             logger.info(f"Average AUROC: {test_metrics['avg_auroc']:.3f}")
#             logger.info(f"Average F1: {test_metrics['avg_f1']:.3f}")

#             # Return path to best checkpoint
#             ckpt_path = trainer.checkpoint_callback.best_model_path
#             logger.info(f"Best model saved to {ckpt_path}")
#             return ckpt_path

#         except Exception as e:
#             logger.error(f"Training failed: {str(e)}")
#             raise RuntimeError(f"Training pipeline failed: {str(e)}") from e

# # class ChestXRayTrainer:
# #     """Manages the training process for chest X-ray classification models."""

# #     def __init__(self, config_path: str) -> None:
# #         """Initialize the trainer with configuration parameters.

# #         Args:
# #             config_path: Path to YAML configuration file

# #         Raises:
# #             FileNotFoundError: If config file doesn't exist
# #             ValueError: If configuration is invalid
# #         """
# #         self.config = self._load_config(config_path)
# #         self.device = self._setup_device()

# #     def _load_config(self, config_path: str) -> Dict[str, Any]:
# #         """Load and validate configuration from YAML file.

# #         Args:
# #             config_path: Path to configuration file

#         # Returns:
#         #     Dict[str, Any]: Parsed configuration dictionary

#         # Raises:
#         #     FileNotFoundError: If config file doesn't exist
#         #     ValueError: If configuration is invalid
#         # """
#         # if not Path(config_path).exists():
#         #     raise FileNotFoundError(f"Configuration file not found: {config_path}")

#         # with open(config_path) as f:
#         #     config = yaml.safe_load(f)

#         # required_keys = {"model", "data", "trainer"}
#         # if not all(key in config for key in required_keys):
#         #     raise ValueError(f"Configuration must contain {required_keys}")

#         # return config

#     def _setup_device(self) -> torch.device:
#         """Configure and return the training device (CPU/GPU).

#         Returns:
#             torch.device: Selected training device
#         """
#         if torch.cuda.is_available():
#             device = torch.device("cuda")
#             logger.info("Using GPU for training")
#         else:
#             device = torch.device("cpu")
#             logger.warning("GPU not available, falling back to CPU")
#         return device

#     def _instantiate_class(self, config: Dict[str, Any]) -> Any:
#         """Instantiate a class from configuration.

#         Args:
#             config: Dictionary containing class_path and init_args keys

#         Returns:
#             Any: Instance of the specified class

#         Raises:
#             ImportError: If class cannot be imported
#         """
#         class_ = import_class(config["class_path"])
#         return class_(**config.get("init_args", {}))

#     def _setup_trainer(self) -> pl.Trainer:
#         """Initialize PyTorch Lightning trainer from configuration.

#         Returns:
#             pl.Trainer: Configured Lightning trainer instance

#         Notes:
#             Handles callback and logger instantiation from config
#         """
#         trainer_config = self.config["trainer"].copy()

#         # Handle callbacks if present
#         if "callbacks" in trainer_config:
#             callbacks = [
#                 self._instantiate_class(callback_config)
#                 for callback_config in trainer_config["callbacks"]
#             ]
#             trainer_config["callbacks"] = callbacks

#         # Handle logger if present
#         if "logger" in trainer_config:
#             trainer_config["logger"] = self._instantiate_class(
#                 trainer_config["logger"]
#             )

#         return pl.Trainer(**trainer_config)

#     def train(self) -> str:
#         """Execute the training pipeline.

#         Returns:
#             str: Path to saved model checkpoint

#         Raises:
#             RuntimeError: If training fails
#         """
#         try:
#             # Initialize model and data module
#             model = self._instantiate_class(self.config["model"])
#             datamodule = self._instantiate_class(self.config["data"])

#             # Setup and run training
#             trainer = self._setup_trainer()
#             logger.info("Starting model training...")
#             trainer.fit(model, datamodule)

#             # Run evaluation
#             logger.info("Training completed. Running evaluation...")
#             trainer.test(model, datamodule=datamodule)

#             # Return path to best checkpoint
#             ckpt_path = trainer.checkpoint_callback.best_model_path
#             logger.info(f"Best model saved to {ckpt_path}")
#             return ckpt_path

#         except Exception as e:
#             logger.error(f"Training failed: {str(e)}")
#             raise RuntimeError(f"Training pipeline failed: {str(e)}") from e

#     def evaluate_model(model: pl.LightningModule, test_dataloader: DataLoader) -> Dict[str, float]:
#         """Evaluate model performance and generate visualizations.

#         This function:
#         1. Runs model predictions on test data
#         2. Computes performance metrics
#         3. Generates visualization plots
#         4. Saves results to specified directories

#         Args:
#             model: Trained PyTorch Lightning model
#             test_dataloader: DataLoader containing test data

#         Returns:
#             Dictionary containing evaluation metrics
#         """
#         # Set up visualizers
#         perf_viz = PerformanceVisualizer(save_dir='results/model_evaluation')
#         img_viz = ImageVisualizer(save_dir='results/sample_predictions')

#         # Prepare for evaluation
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         model.to(device)
#         model.eval()

#         # Collect predictions and ground truth
#         all_preds = []
#         all_labels = []
#         example_images = []  # Store some examples for visualization

#         with torch.no_grad():
#             for batch_idx, (images, labels) in enumerate(tqdm(test_dataloader)):
#                 images = images.to(device)
#                 labels = labels.to(device)

#                 # Get model predictions
#                 logits = model(images)
#                 preds = torch.sigmoid(logits)

#                 # Store results
#                 all_preds.append(preds.cpu())
#                 all_labels.append(labels.cpu())

#                 # Store a few examples for visualization
#                 if batch_idx < 5:  # Save first 5 batches for example visualizations
#                     example_images.extend([
#                         (img, label, pred)
#                         for img, label, pred in zip(images, labels, preds)
#                     ])

#         # Concatenate all predictions and labels
#         all_preds = torch.cat(all_preds, dim=0).numpy()
#         all_labels = torch.cat(all_labels, dim=0).numpy()

#         # Generate performance visualizations
#         perf_viz.plot_roc_curves(
#             all_labels,
#             all_preds,
#             save_name='test_roc_curves'
#         )

#         perf_viz.plot_confusion_matrices(
#             all_labels,
#             all_preds,
#             save_name='test_confusion_matrices'
#         )

#         # Generate sample prediction visualizations
#         for idx, (image, label, pred) in enumerate(example_images[:5]):
#             img_viz.visualize_predictions(
#                 image.cpu(),
#                 label.cpu().numpy(),
#                 pred.cpu().numpy(),
#                 save_name=f'example_prediction_{idx}'
#             )

#         # Calculate and return metrics
#         metrics = {
#             'avg_auroc': roc_auc_score(all_labels, all_preds, average='macro'),
#             'avg_f1': f1_score(
#                 all_labels,
#                 (all_preds > 0.5).astype(int),
#                 average='macro'
#             ),
#             'per_class_auroc': roc_auc_score(
#                 all_labels,
#                 all_preds,
#                 average=None
#             ).tolist()
#         }

#         return metrics

# def main():
#     """Main entry point for training."""
#     parser = ArgumentParser(description="Train chest X-ray classification model")
#     parser.add_argument(
#         "--config",
#         type=str,
#         required=True,
#         help="Path to YAML configuration file"
#     )
#     args = parser.parse_args()

#     try:
#         trainer = ChestXRayTrainer(args.config)
#         ckpt_path = trainer.train()
#         logger.info(f"Training completed successfully. Model saved to: {ckpt_path}")
#     except Exception as e:
#         logger.error(f"Training failed: {str(e)}")
#         raise


# if __name__ == "__main__":
#     main()