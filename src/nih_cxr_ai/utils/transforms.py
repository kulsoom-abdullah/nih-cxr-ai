# src/nih_cxr_ai/utils/transforms.py
"""Image transformation utilities for chest X-rays.

This module provides consistent image transformations for training and validation,
ensuring proper preprocessing of medical images while maintaining clinically 
relevant features.
"""

from typing import Tuple, Union

import torch
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode


def get_train_transforms(
    image_size: Tuple[int, int] = (224, 224), normalize: bool = True
) -> T.Compose:
    """Get training transforms for chest X-ray images.

    Applies a sequence of augmentations suitable for medical imaging:
    - Resize to target size
    - Random horizontal flip (anatomically valid)
    - Mild random rotation (±10 degrees)
    - Optional normalization using ImageNet statistics

    Args:
        image_size: Target (height, width) for resizing
        normalize: Whether to apply normalization

    Returns:
        Composition of transforms
    """
    transforms = [
        T.Resize(image_size, interpolation=InterpolationMode.BILINEAR),
        T.RandomHorizontalFlip(p=0.5),
        # Conservative rotation to preserve anatomical validity
        T.RandomRotation(degrees=10, interpolation=InterpolationMode.BILINEAR, fill=0),
        T.ToTensor(),
    ]

    if normalize:
        # ImageNet normalization - useful with pretrained models
        transforms.append(
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )

    return T.Compose(transforms)


def get_val_transforms(
    image_size: Tuple[int, int] = (224, 224), normalize: bool = True
) -> T.Compose:
    """Get validation/test transforms for chest X-ray images.

    Applies minimal transformations needed for evaluation:
    - Resize to target size
    - Optional normalization using ImageNet statistics

    Args:
        image_size: Target (height, width) for resizing
        normalize: Whether to apply normalization

    Returns:
        Composition of transforms
    """
    transforms = [
        T.Resize(image_size, interpolation=InterpolationMode.BILINEAR),
        T.ToTensor(),
    ]

    if normalize:
        transforms.append(
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )

    return T.Compose(transforms)


def get_gradient_visualization_transform(
    image_size: Tuple[int, int] = (224, 224)
) -> T.Compose:
    """Get transforms for gradient visualization.

    Similar to validation transforms but without normalization
    to maintain interpretable pixel values for visualization.

    Args:
        image_size: Target (height, width) for resizing

    Returns:
        Composition of transforms
    """
    return T.Compose(
        [
            T.Resize(image_size, interpolation=InterpolationMode.BILINEAR),
            T.ToTensor(),
        ]
    )
