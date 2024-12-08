# src/nih_cxr_ai/utils/transforms.py
"""Image transformation utilities for chest X-rays.

Provides consistent image transformations for training and validation,
ensuring proper preprocessing of medical images while maintaining
clinically relevant features.
"""

# Standard library imports
from typing import Tuple

# Third-party imports
from torchvision import transforms

def get_train_transforms(image_size: Tuple[int, int] = (224, 224)):
    """
    Get training transforms for chest X-ray images.

    Args:
        image_size: Tuple of (height, width) for resizing images

    Returns:
        transforms.Compose: Composition of image transformations
    """
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(
                degrees=10, translate=(0.1, 0.1)
            ),  # Reinstating this
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_val_transforms(image_size: Tuple[int, int] = (224, 224)):
    """Get validation transforms for 1024x1024 input images."""
    return transforms.Compose(
        [
            transforms.Resize(image_size),  # From 1024x1024 to target size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


# Some medical imaging papers even suggest additional valid augmentations we could consider adding:

# Slight brightness/contrast variations (to handle different X-ray exposure settings)
# Minor scaling (to handle different chest sizes)
# Gaussian noise (to simulate different imaging equipment qualities)
