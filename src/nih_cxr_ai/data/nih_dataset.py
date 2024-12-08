# src/nih_cxr_ai/data/nih_dataset.py
"""NIH Chest X-ray Dataset Implementation.

This module provides dataset and datamodule implementations for the NIH Chest X-ray
dataset, supporting training, validation and test splits with appropriate transforms.
Handles multi-label classification data with configurable preprocessing.
"""

# Standard library imports
import logging
import math
from pathlib import Path
from typing import Optional, Tuple

# Third-party imports
import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from ..utils.transforms import get_train_transforms, get_val_transforms

logger = logging.getLogger(__name__)


class NIHChestDataset(Dataset):
    """Dataset class for NIH Chest X-ray dataset with multi-label support."""

    def __init__(self, 
                 data_dir: str,
                 csv_file: str,
                 transform: Optional[transforms.Compose] = None):
        """Initialize NIH Chest X-ray dataset.
        
        Args:
            data_dir: Root directory containing the dataset
            csv_file: Path to the CSV file containing image paths and labels
            transform: Optional transforms to be applied to images
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Load labels
        self.df = pd.read_csv(csv_file)
        
        # Convert string labels to multi-hot encoding if needed
        if isinstance(self.df['labels'].iloc[0], str):
            self.df['labels'] = self.df['labels'].apply(eval)
            
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample to get
            
        Returns:
            Tuple of (image, label_tensor)
        """
        # Get image path
        img_path = self.data_dir / self.df.iloc[idx]['image_file_path']
        
        # Load image
        image = Image.open(img_path).convert('RGB')
                
        # Apply transforms
        if self.transform:
            image = self.transform(image)
                
        # Convert labels to multi-hot encoding
        if isinstance(self.df.iloc[idx]['labels'], str):
            labels = eval(self.df.iloc[idx]['labels'])
        else:
            labels = self.df.iloc[idx]['labels']
            
        # Create multi-hot encoding
        label_tensor = torch.zeros(15, dtype=torch.long)
        label_tensor[labels] = 1
        return image, label_tensor


class NIHChestDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for NIH Chest X-ray dataset."""

    def __init__(
        self,
        data_dir: str = "data/nih_chest_xray",
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: Tuple[int, int] = (224, 224),
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15)
    ):
        """Initialize the DataModule.
        
        Args:
            data_dir: Root directory containing the dataset
            batch_size: Number of samples per batch 
            num_workers: Number of workers for data loading
            image_size: Target size for image transforms
            train_val_test_split: Ratios for train/val/test split
        """
        super().__init__()
        
        # Validate split ratios
        if not math.isclose(sum(train_val_test_split), 1.0, rel_tol=1e-9):
            raise ValueError("Train/val/test split ratios must sum to 1.0")
        if any(split < 0 for split in train_val_test_split):
            raise ValueError("Split ratios cannot be negative")
            
        self.save_hyperparameters()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.split_ratios = train_val_test_split
        
        # Initialize datasets to None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self) -> None:
        """Verify data exists and prepare if needed."""
        if not self.data_dir.exists():
            raise RuntimeError(f"Data directory {self.data_dir} does not exist")
            
        required_files = ['labels.csv']
        missing_files = [f for f in required_files 
                        if not (self.data_dir / f).exists()]
        if missing_files:
            raise RuntimeError(
                f"Missing required files in {self.data_dir}: {missing_files}"
            )

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up train, validation and test datasets.
        
        Args:
            stage: Optional stage ('fit', 'validate', 'test', None)
        """
        # Load full dataset
        df = pd.read_csv(self.data_dir / 'labels.csv')
        
        # Calculate split sizes
        train_size = int(len(df) * self.split_ratios[0])
        val_size = int(len(df) * self.split_ratios[1])
        
        # Create splits
        train_df = df[:train_size]
        val_df = df[train_size:train_size + val_size]
        test_df = df[train_size + val_size:]
        
        # Save split CSVs
        train_df.to_csv(self.data_dir / 'train_labels.csv', index=False)
        val_df.to_csv(self.data_dir / 'val_labels.csv', index=False)
        test_df.to_csv(self.data_dir / 'test_labels.csv', index=False)
        
        if stage == 'fit' or stage is None:
            self.train_dataset = NIHChestDataset(
                self.data_dir,
                self.data_dir / 'train_labels.csv',
                transform=get_train_transforms(self.image_size)
            )
            self.val_dataset = NIHChestDataset(
                self.data_dir,
                self.data_dir / 'val_labels.csv',
                transform=get_val_transforms(self.image_size)
            )
            logger.info(f"Train dataset size: {len(self.train_dataset)}")
            logger.info(f"Val dataset size: {len(self.val_dataset)}")
            
        if stage == 'test' or stage is None:
            self.test_dataset = NIHChestDataset(
                self.data_dir,
                self.data_dir / 'test_labels.csv',
                transform=get_val_transforms(self.image_size)
            )
            logger.info(f"Test dataset size: {len(self.test_dataset)}")

    def train_dataloader(self) -> DataLoader:
        """Create the training data loader."""
        if self.train_dataset is None:
            raise RuntimeError("Train dataset not initialized. Call setup() first")
            
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create the validation data loader."""
        if self.val_dataset is None:
            raise RuntimeError("Validation dataset not initialized. Call setup() first")
            
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
    def test_dataloader(self) -> DataLoader:
        """Create the test data loader."""
        if self.test_dataset is None:
            raise RuntimeError("Test dataset not initialized. Call setup() first")
            
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
# # src/data/nih_dataset.py

# import logging
# import os
# from pathlib import Path
# from typing import Optional, Tuple

# import pandas as pd
# import pytorch_lightning as pl
# import torch
# from PIL import Image
# from torch.utils.data import DataLoader, Dataset
# from torchvision import transforms

# from src.utils.transforms import get_train_transforms, get_val_transforms


# class NIHChestDataset(Dataset):
#     def __init__(self, 
#                  data_dir: str,
#                  csv_file: str,
#                  transform: Optional[transforms.Compose] = None):
#         """
#         Args:
#             data_dir: Root directory containing the dataset
#             csv_file: Path to the labels CSV file
#             transform: Optional transform to be applied to images
#         """
#         self.data_dir = data_dir
#         self.transform = transform
        
#         # Load labels
#         self.df = pd.read_csv(csv_file)
        
#         # Convert string labels to multi-hot encoding if needed
#         if isinstance(self.df['labels'].iloc[0], str):
#             self.df['labels'] = self.df['labels'].apply(eval)
            
#     def __len__(self) -> int:
#         return len(self.df)
    
#     def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
#         """Get a single sample from the dataset."""
#         # Get image path
#         img_path = os.path.join(self.data_dir, self.df.iloc[idx]['image_file_path'])
        
#         # Load image - we know it's 1024x1024
#         image = Image.open(img_path).convert('RGB')
                
#         # Apply transforms
#         if self.transform:
#             image = self.transform(image)
                
#         # Convert labels to multi-hot encoding
#         if isinstance(self.df.iloc[idx]['labels'], str):
#             labels = eval(self.df.iloc[idx]['labels'])
#         else:
#             labels = self.df.iloc[idx]['labels']
            
#         # Create multi-hot encoding
#         label_tensor = torch.zeros(15, dtype=torch.long)
#         label_tensor[labels] = 1
#         return image, label_tensor


# class NIHChestDataModule(pl.LightningDataModule):
#     def __init__(
#         self,
#         data_dir: str = "data/nih_chest_xray",
#         batch_size: int = 32,  # This will be our single source of truth for batch size
#         num_workers: int = 4,
#         image_size: Tuple[int, int] = (224, 224)
#     ):
#         super().__init__()
#         self.save_hyperparameters()  # This saves all init parameters for easy access
#         self.data_dir = data_dir
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.image_size = image_size
        
#     def setup(self, stage: Optional[str] = None):
#         if stage == 'fit' or stage is None:
#             # Load and split data
#             df = pd.read_csv(os.path.join(self.data_dir, 'labels.csv'))
#             train_size = int(0.8 * len(df))
            
#             train_df = df[:train_size]
#             val_df = df[train_size:]
            
#             # Create datasets
#             self.train_dataset = NIHChestDataset(
#                 self.data_dir,
#                 os.path.join(self.data_dir, 'train_labels.csv'),
#                 transform=get_train_transforms(self.image_size)
#             )
            
#             self.val_dataset = NIHChestDataset(
#                 self.data_dir,
#                 os.path.join(self.data_dir, 'val_labels.csv'),
#                 transform=get_val_transforms(self.image_size)
#             )
            
#             print(f"Train dataset size: {len(self.train_dataset)}")
#             print(f"Val dataset size: {len(self.val_dataset)}")
    
#     def train_dataloader(self):
#         return DataLoader(
#             self.train_dataset,
#             batch_size=self.batch_size,
#             shuffle=True,
#             num_workers=self.num_workers,
#             pin_memory=True
#         )
    
#     def val_dataloader(self):
#         return DataLoader(
#             self.val_dataset,
#             batch_size=self.batch_size,
#             shuffle=False,
#             num_workers=self.num_workers,
#             pin_memory=True
#         )
