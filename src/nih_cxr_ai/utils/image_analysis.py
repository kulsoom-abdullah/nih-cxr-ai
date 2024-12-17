# src/nih_cxr_ai/utils/image_analysis.py
"""Image analysis utilities for chest X-rays.

Provides tools for analyzing image characteristics, including size distributions,
intensity statistics, and quality metrics specific to medical imaging.
"""

# Standard library imports
from collections import Counter
from pathlib import Path
from typing import Dict

# Third-party imports
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def analyze_image_sizes(data_dir: str) -> Dict:
    """
    Analyzes image sizes in the dataset.

    Args:
        data_dir: Directory containing the images

    Returns:
        Dictionary containing:
            - size_distribution: Counter of image sizes
            - mean_size: Average size
            - std_size: Standard deviation of sizes
            - min_size: Minimum size
            - max_size: Maximum size
    """
    image_dir = Path(data_dir)
    sizes = []

    # Collect all image sizes
    for img_path in tqdm(list(image_dir.glob("*.png")), desc="Analyzing images"):
        try:
            with Image.open(img_path) as img:
                sizes.append(img.size)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    # Convert to numpy for statistics
    sizes_array = np.array(sizes)

    results = {
        "size_distribution": Counter(sizes),
        "mean_size": sizes_array.mean(axis=0),
        "std_size": sizes_array.std(axis=0),
        "min_size": sizes_array.min(axis=0),
        "max_size": sizes_array.max(axis=0),
        "total_images": len(sizes),
        "unique_sizes": len(set(sizes)),
    }

    # Create size distribution DataFrame for easy viewing
    df_sizes = pd.DataFrame.from_dict(
        results["size_distribution"], orient="index", columns=["count"]
    )
    df_sizes.index = df_sizes.index.map(lambda x: f"{x[0]}x{x[1]}")
    results["size_distribution_df"] = df_sizes

    return results
