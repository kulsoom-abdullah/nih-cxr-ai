"""NIH Chest X-ray AI Package.

This package provides tools and models for analyzing chest X-rays using deep learning.
"""

__version__ = "0.1.0"
__author__ = "Kulsoom Abdullah"


from .data.nih_dataset import NIHChestDataModule
from .models.traditional import TraditionalCXRModel

__all__ = ["TraditionalCXRModel", "NIHChestDataModule"]
