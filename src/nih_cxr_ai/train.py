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
from torch.utils.data import DataLoader
from tqdm import tqdm

# from .utils.visualization.image_viz import ImageVisualizer
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
        # self.image_viz = ImageVisualizer(save_dir="results/sample_predictions")

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
        init_args = config.get("init_args", {})
        print(f"Instantiating {config['class_path']} with args: {init_args}")
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
            - test_auroc_mean: Area under ROC curve
            - test_f1_mean: F1 score
            - test_precision_mean: Precision score
            - test_recall_mean: Recall score
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

        disease_names = model.disease_names  # Ensure model.disease_names is accessible
        auroc_values = metrics["auroc"]  # This is a tensor [num_classes]
        f1_values = metrics["f1"]  # Also [num_classes]
        precision_values = metrics["precision"]
        recall_values = metrics["recall"]

        # Print or log per-class results
        for i, d_name in enumerate(disease_names):
            print(
                f"{d_name}: AUROC={auroc_values[i].item():.4f}, "
                f"F1={f1_values[i].item():.4f}, "
                f"Precision={precision_values[i].item():.4f}, "
                f"Recall={recall_values[i].item():.4f}"
            )

        test_auroc_mean = auroc_values.mean().item()
        test_f1_mean = f1_values.mean().item()
        test_precision_mean = precision_values.mean().item()
        test_recall_mean = recall_values.mean().item()

        return {
            "test_auroc_mean": test_auroc_mean,
            "test_f1_mean": test_f1_mean,
            "test_precision_mean": test_precision_mean,
            "test_recall_mean": test_recall_mean,
        }

    def train(self) -> str:
        """Execute the training pipeline."""
        try:
            # Instantiate the model
            model = self._instantiate_class(self.config["model"])

            # Instantiate dataloaders for train and val splits
            train_dataloader = self._instantiate_class(self.config["data"]["train"])
            val_dataloader = self._instantiate_class(self.config["data"]["val"])

            # Setup trainer
            trainer = self._setup_trainer()

            # Start training
            logger.info("Starting model training...")
            trainer.fit(
                model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
            )

            # Optionally run test set evaluation
            if "test" in self.config["data"]:
                logger.info("Running evaluation on test set...")
                test_dataloader = self._instantiate_class(self.config["data"]["test"])
                trainer.test(model, dataloaders=test_dataloader)

            return trainer.checkpoint_callback.best_model_path

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
