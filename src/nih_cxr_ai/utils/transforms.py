# src/nih_cxr_ai/utils/transforms.py
"""Image transformation utilities for chest X-rays.

This module provides consistent image transformations for training and validation,
ensuring proper preprocessing of medical images while maintaining clinically
relevant features.
"""

from typing import Tuple

import torchvision.transforms as T
from torchvision.transforms import InterpolationMode


def get_train_transforms(
    image_size: Tuple[int, int] = (224, 224), normalize: bool = True
) -> T.Compose:
    """
    Get training transformations for chest X-ray images, inspired by the augmentation
    strategy described in the referenced research paper.

    The paper applied a combination of mild geometric and brightness augmentations
    to improve model generalization. This includes:

    - Resizing the image to a fixed size (e.g., 224x224) to maintain a consistent
      input dimension for the model.
    - Random horizontal flipping with a moderate probability (p=0.3), which can
      mimic the subtle variability in patient positioning. This is considered
      anatomically valid, as flipping a chest X-ray horizontally does not typically
      invalidate the anatomical relationships.
    - Random rotation of up to ±5 degrees to account for slight variations in patient
      orientation and X-ray acquisition angles.
    - Brightness adjustments of ±0.2 to account for differences in X-ray exposure
      levels and improve robustness to lighting conditions.
    - Normalization using ImageNet statistics, since the backbone model is often
      pretrained on ImageNet. This brings the X-ray intensity distribution closer
      to what the model's pretrained weights expect.

    Args:
        image_size: Target (height, width) for resizing. Defaults to (224, 224) as
                    commonly used for ImageNet-pretrained models.
        normalize: Whether to apply ImageNet normalization. Typically True for
                   pretrained models.

    Returns:
        Composition of transforms suitable for training a chest X-ray classification model.
    """
    transforms = [
        T.Resize(image_size, interpolation=InterpolationMode.BILINEAR),
        T.RandomHorizontalFlip(p=0.3),
        T.RandomRotation(degrees=5),
        T.ColorJitter(brightness=0.2),
        T.ToTensor(),
    ]

    if normalize:
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
