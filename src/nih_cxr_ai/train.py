# src/nih_cxr_ai/train.py
"""Training script for chest X-ray classification models.

This module provides the main training loop and utilities for training deep learning
models on the NIH Chest X-ray dataset using PyTorch Lightning.
"""

import os
import logging
from pathlib import Path
from typing import Optional
from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
import yaml
from pytorch_lightning.cli import LightningCLI
from importlib import import_module


from .models.traditional import TraditionalCXRModel
from .data.nih_dataset import NIHChestDataModule

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def import_class(class_path: str):
    """Dynamically import a class from a string path.
    
    Args:
        class_path: String path to class (e.g., 'src.models.traditional.TraditionalCXRModel')
        
    Returns:
        class: The imported class
        
    Raises:
        ImportError: If class cannot be imported
    """
    try:
        module_path, class_name = class_path.rsplit('.', 1)
        module = import_module(module_path)
        return getattr(module, class_name)
    except Exception as e:
        raise ImportError(f"Could not import {class_path}: {str(e)}")

class ChestXRayTrainer:
    """Manages the training process for chest X-ray classification models."""
    
    def __init__(self, config_path: str) -> None:
        """Initialize the trainer with configuration parameters.
        
        Args:
            config_path: Path to YAML configuration file
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If configuration is invalid
        """
        self.config = self._load_config(config_path)
        self.device = self._setup_device()
        
    def _load_config(self, config_path: str) -> dict:
        """Load and validate configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Parsed configuration dictionary
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If configuration is invalid
        """
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path) as f:
            config = yaml.safe_load(f)
            
        required_keys = {'model', 'data', 'trainer'}
        if not all(key in config for key in required_keys):
            raise ValueError(f"Configuration must contain {required_keys}")
            
        return config
    
    def _setup_device(self) -> torch.device:
        """Configure and return the training device (CPU/GPU).
        
        Returns:
            torch.device: Selected training device
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("Using GPU for training")
        else:
            device = torch.device("cpu")
            logger.warning("GPU not available, falling back to CPU")
        return device
    
    def _instantiate_class(self, config: dict):
        """Instantiate a class from configuration.
        
        Args:
            config: Dictionary containing class_path and init_args
            
        Returns:
            Instance of the specified class
        """
        class_ = import_class(config['class_path'])
        return class_(**config.get('init_args', {}))
    
    def _instantiate_model(self):
        """Instantiate model from configuration."""
        return self._instantiate_class(self.config['model'])
        
    def _instantiate_datamodule(self):
        """Instantiate datamodule from configuration."""
        return self._instantiate_class(self.config['data'])
    
    def _setup_trainer(self) -> pl.Trainer:
        """Initialize PyTorch Lightning trainer from configuration.
        
        Returns:
            pl.Trainer: Configured Lightning trainer
            
        Notes:
            Handles callback and logger instantiation from config
        """
        trainer_config = self.config['trainer'].copy()
        
        # Handle callbacks if present
        if 'callbacks' in trainer_config:
            callbacks = []
            for callback_config in trainer_config['callbacks']:
                callbacks.append(self._instantiate_class(callback_config))
            trainer_config['callbacks'] = callbacks
            
        # Handle logger if present
        if 'logger' in trainer_config:
            trainer_config['logger'] = self._instantiate_class(trainer_config['logger'])
            
        return pl.Trainer(**trainer_config)

    def train(self) -> str:
        """Execute the training pipeline.
        
        Returns:
            str: Path to saved model checkpoint
            
        Raises:
            RuntimeError: If training fails
        """
        try:
            # Initialize model and data module
            model = self._instantiate_model()
            datamodule = self._instantiate_datamodule()
            
            # Setup and run training
            trainer = self._setup_trainer()
            logger.info("Starting model training...")
            trainer.fit(model, datamodule)
            
            # Run evaluation
            logger.info("Training completed. Running evaluation...")
            trainer.test(model, datamodule=datamodule)
            
            # Return path to best checkpoint
            ckpt_path = trainer.checkpoint_callback.best_model_path
            logger.info(f"Best model saved to {ckpt_path}")
            return ckpt_path
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise RuntimeError(f"Training pipeline failed: {str(e)}") from e

def main():
    """Main entry point for training."""
    parser = ArgumentParser(description='Train chest X-ray classification model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML configuration file')
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
# models on the NIH Chest X-ray dataset using PyTorch Lightning.

# Typical usage:
#     python train.py --config configs/training_config.yaml
# """

# import os
# import logging
# from pathlib import Path
# from typing import Optional, Dict, Any

# import torch
# import pytorch_lightning as pl
# from pytorch_lightning.loggers import WandbLogger
# from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
# from pytorch_lightning.utilities.types import STEP_OUTPUT

# # Local imports
# from models.traditional import TraditionalCXRModel
# from data.nih_dataset import NIHChestDataModule
# from config import load_config  


# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class ChestXRayTrainer:
#     """Manages the training process for chest X-ray classification models."""
    
#     def __init__(
#         self, 
#         config_path: Optional[str] = None,
#         data_dir: Optional[Path] = None,
#         num_classes: int = 15,
#         batch_size: int = 32,
#         num_workers: int = 4,
#         image_size: tuple[int, int] = (224, 224)
#     ) -> None:
#         """Initialize the trainer with configuration parameters.
        
#         Args:
#             config_path: Path to YAML configuration file
#             data_dir: Directory containing the dataset
#             num_classes: Number of disease classes to predict
#             batch_size: Batch size for training
#             num_workers: Number of data loading workers
#             image_size: Target size for input images (height, width)
            
#         Raises:
#             FileNotFoundError: If config_path is provided but file doesn't exist
#             RuntimeError: If CUDA is requested but not available
#         """
#         self.config = load_config(config_path) if config_path else {}
#         self.data_dir = data_dir or Path(__file__).resolve().parent.parent / "data" / "nih_chest_xray"
#         self.num_classes = num_classes
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.image_size = image_size
        
#         # Set up device
#         self.device = self._setup_device()
        
#     def _setup_device(self) -> torch.device:
#         """Configure and return the training device (CPU/GPU).
        
#         Returns:
#             torch.device: Selected training device
            
#         Raises:
#             RuntimeError: If CUDA is requested but not available
#         """
#         if torch.cuda.is_available():
#             device = torch.device("cuda")
#             os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#             logger.info("Using GPU for training")
#         else:
#             device = torch.device("cpu")
#             logger.warning("GPU not available, falling back to CPU")
#         return device

#     def _setup_callbacks(self) -> list:
#         """Initialize training callbacks.
        
#         Returns:
#             list: List of PyTorch Lightning callbacks
#         """
#         return [
#             ModelCheckpoint(
#                 dirpath="checkpoints",
#                 filename="{epoch}-{val_loss:.2f}",
#                 save_top_k=3,
#                 monitor="val_loss",
#                 mode="min",
#             ),
#             EarlyStopping(
#                 monitor="val_loss",
#                 patience=10,
#                 mode="min",
#                 verbose=True
#             ),
#         ]

#     def _setup_trainer(self) -> pl.Trainer:
#         """Initialize PyTorch Lightning trainer.
        
#         Returns:
#             pl.Trainer: Configured Lightning trainer
#         """
#         return pl.Trainer(
#             max_epochs=self.config.get('max_epochs', 100),
#             accelerator="gpu" if self.device.type == "cuda" else "cpu",
#             devices=1,
#             precision="16-mixed" if self.device.type == "cuda" else "32",
#             callbacks=self._setup_callbacks(),
#             logger=WandbLogger(
#                 project=self.config.get('wandb_project', "chest-xray-comparison"),
#                 name=self.config.get('wandb_run_name', None)
#             ),
#             strategy="auto",
#         )

#     def train(self) -> Dict[str, Any]:
#         """Execute the training pipeline.
        
#         Returns:
#             Dict[str, Any]: Dictionary containing training metrics and results
            
#         Raises:
#             RuntimeError: If training fails
#         """
#         try:
#             # Initialize model and data module
#             model = TraditionalCXRModel(num_classes=self.num_classes)
#             datamodule = NIHChestDataModule(
#                 data_dir=str(self.data_dir),
#                 batch_size=self.batch_size,
#                 num_workers=self.num_workers,
#                 image_size=self.image_size,
#             )
            
#             # Setup and run training
#             trainer = self._setup_trainer()
#             logger.info("Starting model training...")
#             trainer.fit(model, datamodule)
            
#             # Run evaluation
#             logger.info("Training completed. Running evaluation...")
#             test_results = trainer.test(model, datamodule=datamodule)
            
#             # Save model
#             model_path = Path("checkpoints") / "final_model.ckpt"
#             trainer.save_checkpoint(str(model_path))
#             logger.info(f"Model saved to {model_path}")
            
#             return {
#                 "test_results": test_results,
#                 "model_path": str(model_path)
#             }
            
#         except Exception as e:
#             logger.error(f"Training failed: {str(e)}")
#             raise RuntimeError(f"Training pipeline failed: {str(e)}") from e

# def main() -> None:
#     """Main entry point for training."""
#     try:
#         trainer = ChestXRayTrainer()
#         results = trainer.train()
#         logger.info(f"Training completed successfully. Test results: {results['test_results']}")
#     except Exception as e:
#         logger.error(f"Training failed: {str(e)}")
#         raise

# if __name__ == "__main__":
#     main()