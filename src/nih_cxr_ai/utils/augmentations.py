# src/nih_cxr_ai/utils/augmentations.py
"""Data augmentation configurations for chest X-rays.

Defines augmentation parameters and transformations suitable for 
medical imaging, with careful consideration for preserving diagnostic features.
"""

# Standard library imports
from dataclasses import dataclass
from typing import Any, Dict, Tuple

# Third-party imports
import numpy as np

@dataclass
class ChestXrayAugParams:
    """Configuration parameters for chest X-ray augmentations"""

    # Rotation
    rotation_degrees: Tuple[float, float] = (-7.0, 7.0)  # Conservative rotation range

    # Translation (as percentage of image size)
    translate_range: Tuple[float, float] = (-0.05, 0.05)  # ±5% translation

    # Intensity/Contrast
    intensity_window_center: float = 0.5
    intensity_window_width: float = 0.8
    contrast_range: Tuple[float, float] = (0.9, 1.1)  # Subtle contrast adjustment

    # Noise
    gaussian_noise_std: float = 0.01  # Very subtle noise

    # Horizontal flip probability
    flip_prob: float = 0.5

    @property
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary for easy access"""
        return {
            "rotation": self.rotation_degrees,
            "translate": self.translate_range,
            "intensity": {
                "center": self.intensity_window_center,
                "width": self.intensity_window_width,
            },
            "contrast": self.contrast_range,
            "noise_std": self.gaussian_noise_std,
            "flip_prob": self.flip_prob,
        }


# Default configuration
DEFAULT_AUGMENTATION_CONFIG = ChestXrayAugParams()
