# src/nih_cxr_ai/utils/visualization/image_viz.py
"""Image visualization utilities for chest X-rays.

Tools for visualizing individual X-ray images, sample sets by pathology,
and image characteristics like intensity distributions.
"""

from pathlib import Path

# Standard library imports
from typing import List, Optional, Union

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image


class ImageVisualizer:
    """Handles visualization of medical images and augmentations."""

    def __init__(
        self, data_dir: Optional[Path] = None, save_dir: Optional[Path] = None
    ) -> None:
        """Initialize image visualizer.

        Args:
            data_dir: Base directory containing the images folder
            save_dir: Directory to save generated visualizations
        """
        self.data_dir = data_dir
        self.save_dir = Path(save_dir) if save_dir else Path("results/visualizations")
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def load_image(self, image_path: Union[str, Path]) -> Image.Image:
        """Load and convert image to grayscale.

        Args:
            image_path: Path to the image file

        Returns:
            PIL Image object in grayscale mode

        Notes:
            If image_path is not absolute and data_dir is set,
            will look for image in data_dir/images/
        """
        img_path = Path(image_path)
        if not img_path.is_absolute() and self.data_dir:
            img_path = self.data_dir / "images" / img_path.name
        return Image.open(img_path).convert("L")

    def show_examples_by_label(
        self, df: pd.DataFrame, label_mapping: dict, num_per_label: int = 1
    ) -> None:
        """Display sample X-ray images for each disease label in a grid layout.

        Args:
            df: DataFrame containing image paths and labels
            label_mapping: Dictionary mapping label indices to disease names
            num_per_label: Number of examples to show per label

        Notes:
            - Uses a grid layout to display images compactly
            - Skips labels with no examples in the dataset
            - Labels are shown as figure titles
        """
        num_labels = len(label_mapping)
        num_cols = 4  # Optimize layout with 4 columns
        num_rows = (num_labels + num_cols - 1) // num_cols

        plt.figure(figsize=(15, 3 * num_rows))

        for idx, (label_idx, label_name) in enumerate(label_mapping.items()):
            # Find all images containing this label
            label_df = df[df["labels"].apply(lambda x: label_idx in eval(x))]
            if len(label_df) == 0:
                continue

            samples = label_df.sample(min(num_per_label, len(label_df)))

            for j, (_, row) in enumerate(samples.iterrows()):
                plt.subplot(num_rows, num_cols, idx + 1)
                img = self.load_image(row["image_file_path"])
                # image = np.clip(image, 0, 1)
                img = np.clip(np.array(img), 0, 1)
                plt.imshow(img, cmap="gray")
                plt.title(f"{label_name}", fontsize=8)
                plt.axis("off")

        plt.tight_layout()
        plt.show()

    def plot_intensity_distribution(self, sample_size: int = 10) -> None:
        """
        Plot the distribution of pixel intensities from a sample of images.

        Args:
            sample_size: Number of random images to sample for analysis.
                        Warning: A large sample_size (e.g., 1000+) combined with high-res images
                        can be resource-intensive. This may result in long processing times,
                        excessive memory usage, or even kernel instability, especially if running
                        on limited hardware.

        This function provides insight into the overall brightness and contrast characteristics
        of the dataset. By examining the frequency of pixel values, we can identify common intensity
        ranges and potential data quality issues (e.g., overly dark or bright images).

        If performance or stability is a concern, reduce the sample_size or consider preprocessing
        images (e.g., downsampling) to mitigate resource strain. For initial exploration, a small
        sample_size (like 10 or 20 images) typically suffices to get a general sense of intensity
        distribution without overwhelming the system.
        """
        if not self.data_dir:
            raise ValueError("data_dir must be set to analyze images")

        all_files = list((self.data_dir / "images").glob("*.png"))
        sample_files = np.random.choice(all_files, min(sample_size, len(all_files)))

        intensities = []
        for img_path in sample_files:
            img = self.load_image(img_path)  # Converts image to grayscale
            intensities.extend(np.array(img).ravel())

        plt.figure(figsize=(10, 6))
        plt.hist(intensities, bins=50, density=True)
        plt.title("Pixel Intensity Distribution")
        plt.xlabel("Pixel Value")
        plt.ylabel("Density")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def visualize_predictions(
        self,
        image: torch.Tensor,
        true_labels: np.ndarray,
        pred_probs: np.ndarray,
        disease_names: List[str],
        save_name: Optional[str] = None,
    ) -> None:
        """Display model predictions for a single image with disease names as x-axis ticks.

        A simplified visualization showing the image and a bar chart comparing
        true labels with predicted probabilities.

        Args:
            image: Input X-ray image tensor (C,H,W)
            true_labels: Ground truth binary labels
            pred_probs: Model's predicted probabilities
            save_name: Optional filename for saving the visualization
        """
        if save_name:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # Convert image tensor to numpy
        image = image.cpu().numpy()
        if image.shape[0] in (1, 3):  # CHW -> HWC
            image = image.transpose(1, 2, 0)
        if image.shape[-1] == 1:
            image = image.squeeze(-1)

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        image = np.clip(image, 0, 1)  # Ensure values are in [0,1]
        plt.imshow(image, cmap="gray")
        plt.title("X-ray Image")
        plt.axis("off")

        # Create bar plot of predictions and true labels
        plt.subplot(1, 2, 2)
        x = np.arange(len(disease_names))
        width = 0.4
        plt.bar(
            x - width / 2, true_labels, width, label="True", color="blue", alpha=0.5
        )
        plt.bar(
            x + width / 2, pred_probs, width, label="Predicted", color="red", alpha=0.5
        )
        plt.title("Predictions vs Ground Truth")
        plt.xlabel("Disease")
        plt.ylabel("Probability")
        plt.legend()

        # Set disease names as x ticks
        plt.xticks(x, disease_names, rotation=45, ha="right")

        plt.tight_layout()

        if save_name:
            plt.savefig(self.save_dir / f"{save_name}.png", bbox_inches="tight")
            plt.close()
        else:
            plt.show()
