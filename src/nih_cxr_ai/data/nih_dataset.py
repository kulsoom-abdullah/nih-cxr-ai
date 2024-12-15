"""NIH Chest X-ray Dataset Implementation.

This module provides dataset and datamodule implementations for the NIH Chest X-ray
dataset, supporting training, validation and test splits with appropriate transforms.
The implementation includes proper handling of multi-label classification data,
configurable preprocessing, and weighted sampling for class imbalance.
"""

import logging
import math
from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from ..utils.transforms import get_train_transforms, get_val_transforms

logger = logging.getLogger(__name__)


class NIHChestDataset(Dataset):
    """Dataset class for NIH Chest X-ray dataset with multi-label support.

    Handles loading and preprocessing of chest X-ray images along with their
    associated disease labels. Supports weighted sampling for handling class
    imbalance and proper image transformations.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        data_source: Union[str, Path, pd.DataFrame],
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        """Initialize NIH Chest X-ray dataset.

        Args:
            data_dir: Root directory containing the dataset
            data_source: Either a path to CSV file or a pandas DataFrame w/image paths and labels
            transform: Optional transforms to be applied to images

        Raises:
            FileNotFoundError: If data_dir or csv_file don't exist
            ValueError: If CSV file is missing required columns
        """
        self.data_dir = Path(data_dir)
        self.transform = transform

        # Validate data directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        # Load data from either DataFrame or CSV
        try:
            if isinstance(data_source, (str, Path)):
                self.df = pd.read_csv(data_source)
            elif isinstance(data_source, pd.DataFrame):
                self.df = data_source.copy()
            else:
                raise ValueError(
                    "data_source must be either a path to CSV or a DataFrame"
                )

            # Validate required columns
            required_columns = {"image_file_path", "labels"}
            if not all(col in self.df.columns for col in required_columns):
                raise ValueError(f"DataFrame must contain columns: {required_columns}")

        except Exception as e:
            raise ValueError(f"Error loading data: {str(e)}")

        # Convert string labels to list if needed
        if isinstance(self.df["labels"].iloc[0], str):
            self.df["labels"] = self.df["labels"].apply(eval)

        logger.info(f"Loaded {len(self.df)} samples")

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        This method is required by PyTorch's Dataset class and is used to
        determine how many samples are available for training/validation.
        """
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample from the dataset.

        This method is called by PyTorch's DataLoader to get each sample.
        It loads the image, applies any transforms, and returns the image
        tensor along with its labels in multi-hot encoding format.

        Args:
            idx: Index of the sample to get

        Returns:
            Tuple containing:
            - image_tensor: Transformed image tensor
            - label_tensor: Multi-hot encoded labels tensor
        """
        # Get image path and load image
        img_path = self.data_dir / self.df.iloc[idx]["image_file_path"]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_path}: {str(e)}")

        # Apply transforms if specified
        if self.transform:
            image = self.transform(image)

        # Process labels - convert to multi-hot encoding
        if isinstance(self.df.iloc[idx]["labels"], str):
            labels = eval(self.df.iloc[idx]["labels"])
        else:
            labels = self.df.iloc[idx]["labels"]

        # Filter out anything >= 14 if that happens to be "No Finding"
        labels = [lbl for lbl in labels if lbl < 14]
        # Create multi-hot encoding tensor
        label_tensor = torch.zeros(14, dtype=torch.long)
        label_tensor[labels] = 1

        return image, label_tensor


class NIHChestDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for NIH Chest X-ray dataset.

    Handles data splitting, loading, and preprocessing for training,
    validation, and testing phases.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: Tuple[int, int] = (224, 224),
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        subset_size: Optional[int] = None,
        debug: bool = False,
    ) -> None:
        """Initialize the DataModule."""
        super().__init__()

        # Validate split ratios
        if not math.isclose(sum(train_val_test_split), 1.0, rel_tol=1e-9):
            raise ValueError("Train/val/test split ratios must sum to 1.0")

        self.save_hyperparameters()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.split_ratios = train_val_test_split
        self.subset_size = subset_size
        self.debug = debug

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up train, validation and test datasets."""
        # Load full dataset
        try:
            full_df = pd.read_csv(self.data_dir / "labels.csv")
        except Exception as e:
            raise RuntimeError(f"Failed to load labels.csv: {str(e)}")

        # Apply subset if specified
        if self.subset_size is not None or self.debug:
            subset_size = self.subset_size if self.subset_size else 1000
            full_df = full_df.sample(n=min(subset_size, len(full_df)), random_state=42)
            logger.info(f"Using subset of {len(full_df)} samples for testing/debugging")

        # Calculate split sizes
        train_size = int(len(full_df) * self.split_ratios[0])
        val_size = int(len(full_df) * self.split_ratios[1])

        # Create splits using DataFrame indexing
        df_shuffled = full_df.sample(frac=1, random_state=42).reset_index(drop=True)
        train_df = df_shuffled.iloc[:train_size]
        val_df = df_shuffled.iloc[train_size : train_size + val_size]  # noqa: E203
        test_df = df_shuffled.iloc[train_size + val_size :]  # noqa: E203

        if stage == "fit" or stage is None:
            self.train_dataset = NIHChestDataset(
                self.data_dir, train_df, transform=get_train_transforms(self.image_size)
            )
            self.val_dataset = NIHChestDataset(
                self.data_dir, val_df, transform=get_val_transforms(self.image_size)
            )
            logger.info(f"Train dataset size: {len(self.train_dataset)}")
            logger.info(f"Val dataset size: {len(self.val_dataset)}")

        if stage == "test" or stage is None:
            self.test_dataset = NIHChestDataset(
                self.data_dir, test_df, transform=get_val_transforms(self.image_size)
            )
            logger.info(f"Test dataset size: {len(self.test_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )
