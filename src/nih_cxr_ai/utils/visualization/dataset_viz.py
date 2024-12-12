# src/nih_cxr_ai/utils/visualization/dataset_viz.py
"""Dataset visualization utilities for chest X-rays.

Provides visualization tools for analyzing dataset statistics, label distributions,
and correlations between different pathologies.
"""

from pathlib import Path

# Standard library imports
from typing import Dict, List, Optional, Tuple

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image


class DatasetVisualizer:
    LABEL_MAPPING = {
        0: "Atelectasis",
        1: "Cardiomegaly",
        2: "Effusion",
        3: "Infiltration",
        4: "Mass",
        5: "Nodule",
        6: "Pneumonia",
        7: "Pneumothorax",
        8: "Consolidation",
        9: "Edema",
        10: "Emphysema",
        11: "Fibrosis",
        12: "Pleural_Thickening",
        13: "Hernia",
    }

    def __init__(
        self,
        df: pd.DataFrame,
        img_col: str = "image_file_path",
        label_col: str = "labels",
        data_dir: Optional[Path] = None,
    ):
        """Initialize dataset visualizer

        Args:
            df: DataFrame containing chest x-ray data
            img_col: Name of column containing image paths
            label_col: Name of column containing numeric labels
            data_dir: Base directory containing the images folder
        """
        self.df = df
        self.img_col = img_col
        self.label_col = label_col
        self.data_dir = (
            data_dir if data_dir is not None else Path("../data/nih_chest_xray")
        )

        # Process labels to compute frequencies
        self._process_labels()

    def _process_labels(self) -> None:
        """Process and calculate frequencies for all disease labels.

        Converts string representations of labels to lists and computes
        frequency counts for each disease type in the dataset.

        Note:
            Called during initialization to set up:
            - label_frequencies: Dict mapping disease names to counts
            - labels: List of unique disease names
        """
        # Convert string representation of list to actual list
        self.df["label_list"] = self.df[self.label_col].apply(eval)

        # Initialize frequencies dictionary
        self.label_frequencies = {}

        # Count frequencies for each disease type
        for label_idx in self.LABEL_MAPPING:
            count = sum(1 for labels in self.df["label_list"] if label_idx in labels)
            self.label_frequencies[self.LABEL_MAPPING[label_idx]] = count

        # Store unique labels for later use
        self.labels = list(self.LABEL_MAPPING.values())

    def _get_frequencies(self, df: pd.DataFrame) -> Dict[str, int]:
        """Calculate frequency of each disease label in a DataFrame.

        Args:
            df: DataFrame containing chest X-ray data with labels column

        Returns:
            Dictionary mapping disease names to their frequencies

        Note:
            Uses the class's label_col attribute to identify the labels column
        """
        freqs = {}
        df["label_list"] = df[self.label_col].apply(eval)
        for idx, name in self.LABEL_MAPPING.items():
            freqs[name] = sum(1 for labels in df["label_list"] if idx in labels)
        return freqs

    def plot_train_val_distribution(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        figsize: Tuple[int, int] = (12, 6),
    ) -> None:
        """Compare disease label distributions between train and validation sets.

        Creates a side-by-side bar chart comparing the frequency of each disease
        label in training and validation datasets.

        Args:
            train_df: DataFrame containing training set data
            val_df: DataFrame containing validation set data
            figsize: Width and height of the plot in inches

        Examples:
            >>> visualizer.plot_train_val_distribution(train_df, val_df)
        """
        # Process both datasets
        train_freqs = self._get_frequencies(train_df)
        val_freqs = self._get_frequencies(val_df)

        plt.figure(figsize=figsize)
        x = np.arange(len(self.LABEL_MAPPING))
        width = 0.35

        # Use distinct colors for better contrast
        train_bars = plt.bar(
            x - width / 2,
            list(train_freqs.values()),
            width,
            label="Train",
            color="#2ecc71",
        )
        val_bars = plt.bar(
            x + width / 2,
            list(val_freqs.values()),
            width,
            label="Validation",
            color="#e74c3c",
        )

        # Add value labels on bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                plt.text(
                    rect.get_x() + rect.get_width() / 2.0,
                    height,
                    f"{int(height):,}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        autolabel(train_bars)
        autolabel(val_bars)

        plt.xticks(x, list(self.LABEL_MAPPING.values()), rotation=45, ha="right")
        plt.title("Label Distribution in Train vs Validation Sets", pad=20)
        plt.xlabel("Disease Category")
        plt.ylabel("Number of Cases")
        plt.legend(loc="upper right")

        # Add grid for better readability
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()

    def plot_label_correlations(self, figsize: Tuple[int, int] = (10, 8)) -> None:
        """Visualize correlations between different disease labels.

        Creates a heatmap showing the correlation coefficients between pairs
        of disease labels, using a diverging color palette to highlight
        positive and negative correlations.

        Args:
            figsize: Width and height of the plot in inches

        Notes:
            Uses a mask to show only the lower triangle of the correlation matrix
            since the matrix is symmetric.

        Examples:
            >>> visualizer.plot_label_correlations()
        """
        # Create binary matrix for each label
        binary_df = pd.DataFrame()
        for idx, name in self.LABEL_MAPPING.items():
            binary_df[name] = self.df["label_list"].apply(
                lambda x: 1 if idx in x else 0
            )

        # Compute correlations
        corr_matrix = binary_df.corr()

        plt.figure(figsize=figsize)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        # Use a diverging colormap with better contrast
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            cmap=cmap,
            center=0,
            vmin=-1,
            vmax=1,
            fmt=".2f",
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.5},
        )

        plt.title("Disease Label Correlations in Chest X-Ray Dataset", pad=20)
        plt.tight_layout()
        plt.show()

    def plot_label_distribution(self, figsize: Tuple[int, int] = (12, 6)) -> None:
        """Visualize distribution of disease labels across the dataset.

        Creates a bar chart showing frequency of each disease label,
        with value annotations and enhanced styling for readability.

        Args:
            figsize: Width and height of the plot in inches.

        Examples:
            >>> visualizer = DatasetVisualizer(df)
            >>> visualizer.plot_label_distribution()
        """
        # Create enhanced bar plot with value annotations

        plt.figure(figsize=figsize)
        labels = list(self.label_frequencies.keys())
        values = list(self.label_frequencies.values())

        # Generate visually distinct colors for each disease category
        colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))
        bars = plt.bar(range(len(labels)), values, color=colors)

        # Add value labels on bars for quick reference
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height):,}",
                ha="center",
                va="bottom",
            )

        # Style and format plot
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
        plt.title("Distribution of Disease Labels in Chest X-Ray Dataset", pad=20)
        plt.xlabel("Disease Category")
        plt.ylabel("Number of Cases")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()

    def plot_multi_label_distribution(self, figsize: Tuple[int, int] = (8, 5)) -> None:
        """Visualize distribution of number of labels per image.

        Creates a histogram showing how many images have different
        numbers of disease labels, providing insight into the
        multi-label nature of the dataset.

        Args:
            figsize: Width and height of the plot in inches, defaults to (8, 5)

        Examples:
            >>> visualizer.plot_multi_label_distribution()
        """
        label_counts = self.df["label_list"].apply(len)
        plt.figure(figsize=figsize)
        plt.hist(label_counts, bins=range(max(label_counts) + 2), align="left")
        plt.title("Distribution of Labels per Image")
        plt.xlabel("Number of Labels")
        plt.ylabel("Count")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_sample_images(
        self,
        num_samples: int = 5,
        label: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 3),
    ) -> None:
        """Display a grid of sample X-ray images from the dataset.

        Creates a figure showing multiple X-ray images, optionally filtered
        by a specific disease label, with their associated disease labels
        as titles.

        Args:
            num_samples: Number of random samples to display
            label: Optional specific disease label to filter by
            figsize: Width and height of the plot in inches

        Notes:
            - Converts images to grayscale for consistent visualization
            - Handles missing images gracefully with error message
            - Wraps long disease label lists in titles

        Examples:
            >>> visualizer.plot_sample_images(num_samples=3)
            >>> visualizer.plot_sample_images(label="Pneumonia")
        """
        import textwrap

        from PIL import Image

        # Filter by label if specified
        if label:
            self.df["label_list"] = self.df[self.label_col].apply(eval)
            label_idx = {v: k for k, v in self.LABEL_MAPPING.items()}[label]
            sample_df = self.df[self.df["label_list"].apply(lambda x: label_idx in x)]
        else:
            sample_df = self.df

        # Sample rows
        samples = sample_df.sample(min(num_samples, len(sample_df)))

        # Create subplot
        fig, axes = plt.subplots(1, len(samples), figsize=figsize)
        if len(samples) == 1:
            axes = [axes]

        # Plot each image
        for ax, (_, row) in zip(axes, samples.iterrows()):
            try:
                # Construct proper image path
                img_filename = Path(row[self.img_col]).name
                img_path = self.data_dir / "images" / img_filename

                # Load and display image using PIL
                img = Image.open(img_path).convert("L")  # Convert to grayscale
                ax.imshow(img, cmap="gray")
                ax.axis("off")

                # Get labels for title
                if isinstance(row[self.label_col], str):
                    labels = eval(row[self.label_col])
                else:
                    labels = row[self.label_col]

                # Convert numeric labels to disease names
                disease_names = [self.LABEL_MAPPING[l] for l in labels]
                # Wrap long titles
                title = "\n".join(textwrap.wrap(", ".join(disease_names), 20))
                ax.set_title(title, fontsize=8)

            except Exception as e:
                print(f"Error plotting image {img_filename}: {e}")
                ax.text(0.5, 0.5, "Image\nNot Found", ha="center", va="center")
                ax.axis("off")

        plt.tight_layout()
        plt.show()

    def analyze_missing_data(self) -> None:
        """Visualize patterns of missing data in the dataset.

        Creates a bar chart showing counts of missing values in each column,
        if any missing values exist in the dataset.

        Examples:
            >>> visualizer.analyze_missing_data()
        """
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            plt.figure(figsize=(10, 6))
            missing[missing > 0].plot(kind="bar")
            plt.title("Missing Data by Column")
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
