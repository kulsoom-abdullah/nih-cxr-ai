"""NIH Chest X-ray Dataset Implementation.

This module provides dataset and datamodule implementations for the NIH Chest X-ray
dataset, supporting training, validation and test splits with appropriate transforms.
The implementation includes proper handling of multi-label classification data,
configurable preprocessing, and weighted sampling for class imbalance.
"""

import logging
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
        num_classes: int = 14,
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
        self.num_classes = num_classes

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
        img_path = self.data_dir / self.df.iloc[idx]["image_file_path"]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        labels = self.df.iloc[idx]["labels"]

        label_tensor = torch.zeros(self.num_classes, dtype=torch.long)
        label_tensor[labels] = 1

        # age = self.df.iloc[idx]["Patient Age"]  # might be string like "058Y", so parse if needed
        # age = int(age.replace("Y",""))  # e.g., convert "058Y" to int(58)
        # gender = self.df.iloc[idx]["Patient Gender"] # "M" or "F"
        # position = self.df.iloc[idx]["View Position"] # "PA" or "AP" etc.

        return image, label_tensor  # , age, gender, position

        def get_sample_weights(self) -> torch.Tensor:
            """Calculate sample weights to handle class imbalance.

            Returns:
                torch.Tensor: Weights for each sample based on label frequencies
            """
            label_counts = self.df["labels"].value_counts()
            weights = 1.0 / label_counts[self.df["labels"]]
            return torch.FloatTensor(weights)


class NIHChestDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for NIH Chest X-ray dataset.

    Handles data splitting, loading, and preprocessing for training,
    validation, and testing phases.
    """

    def __init__(
        self,
        data_dir: Union[str, Path] = "src/data/nih_chest_xray",
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: Tuple[int, int] = (224, 224),
    ) -> None:
        """Initialize the DataModule.

        Args:
            data_dir: Root directory containing the dataset
            batch_size: Number of samples per batch
            num_workers: Number of workers for data loading
            image_size: Target size for image transforms

        Raises:
            ValueError: If split ratios don't sum to 1.0 or are negative
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        # print(f"DEBUG: data_dir set to {self.data_dir}")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self) -> None:
        """Verify data exists and prepare if needed."""
        required_files = [
            "labels.csv",
            "train_labels.csv",
            "val_labels.csv",
            "test_labels.csv",
        ]
        for f in required_files:
            if not (self.data_dir / f).exists():
                raise RuntimeError(f"Missing required file: {f} in {self.data_dir}")

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up train, validation and test datasets.

        Args:
            stage: Optional stage ('fit', 'validate', 'test', None)

        """
        if stage == "fit" or stage is None:
            self.train_dataset = NIHChestDataset(
                self.data_dir,
                self.data_dir / "train_labels.csv",
                transform=get_train_transforms(self.image_size),
            )
            self.val_dataset = NIHChestDataset(
                self.data_dir,
                self.data_dir / "val_labels.csv",
                transform=get_val_transforms(self.image_size),
            )
            logger.info(f"Train dataset size: {len(self.train_dataset)}")
            logger.info(f"Val dataset size: {len(self.val_dataset)}")

        if stage == "test" or stage is None:
            self.test_dataset = NIHChestDataset(
                self.data_dir,
                self.data_dir / "test_labels.csv",
                transform=get_val_transforms(self.image_size),
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
            pin_memory=True,
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
            pin_memory=True,
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
            pin_memory=True,
        )
