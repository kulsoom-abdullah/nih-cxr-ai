# src/nih_cxr_ai/data/data_validation.py
"""Data validation module for the NIH Chest X-ray dataset.

This module provides validation and statistical analysis capabilities for verifying
dataset integrity and generating summary statistics. Includes functions for checking
image corruption, label consistency, and generating validation reports.
"""

# Standard library imports
import logging
import os
from pathlib import Path
from typing import Dict

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

logger = logging.getLogger(__name__)


class NIHDataValidator:
    """Validates and analyzes the NIH Chest X-ray dataset."""

    def __init__(self, data_dir: str):
        """Initialize the data validator.

        Args:
            data_dir: Root directory containing the dataset

        Raises:
            ValueError: If data directory doesn't exist
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {data_dir}")

        self.logger = logging.getLogger(__name__)
        self.stats_dir = self.data_dir / "statistics"
        os.makedirs(self.stats_dir, exist_ok=True)

        self.class_names = [
            "Atelectasis",
            "Cardiomegaly",
            "Effusion",
            "Infiltration",
            "Mass",
            "Nodule",
            "Pneumonia",
            "Pneumothorax",
            "Consolidation",
            "Edema",
            "Emphysema",
            "Fibrosis",
            "Pleural_Thickening",
            "Hernia",
        ]

    def _validate_data_stats(self) -> Dict:
        """Validate basic dataset statistics.

        Returns:
            Dictionary containing dataset statistics
        """
        try:
            labels_df = pd.read_csv(self.data_dir / "labels.csv")
            missing_files = []
            corrupted_files = []

            for _, row in labels_df.iterrows():
                img_path = self.data_dir / row["image_file_path"]
                if not img_path.exists():
                    missing_files.append(str(img_path))
                else:
                    try:
                        with Image.open(img_path) as img:
                            img.verify()
                    except Exception as e:
                        corrupted_files.append(str(img_path))

            return {
                "total_samples": len(labels_df),
                "missing_files": missing_files,
                "corrupted_files": corrupted_files,
            }
        except Exception as e:
            self.logger.error(f"Error validating data: {str(e)}")
            raise

    def _validate_labels(self) -> Dict:
        """Validate labels and compute statistics.

        Returns:
            Dictionary containing label statistics
        """
        df = pd.read_csv(self.data_dir / "labels.csv")

        # Convert string labels to lists if needed
        if isinstance(df["labels"].iloc[0], str):
            df["labels"] = df["labels"].apply(eval)

        # Compute label statistics
        label_counts = {label: 0 for label in self.class_names}
        multi_label_counts = []

        for labels in df["labels"]:
            multi_label_counts.append(len(labels))
            for label in labels:
                if isinstance(label, int):
                    label = self.class_names[label]
                label_counts[label] = label_counts.get(label, 0) + 1

        return {
            "label_counts": label_counts,
            "multi_label_stats": {
                "avg_labels_per_image": np.mean(multi_label_counts),
                "max_labels_per_image": max(multi_label_counts),
            },
        }

    def _generate_validation_report(self, results: Dict):
        """Generate validation report with visualizations.

        Args:
            results: Dictionary containing validation results
        """
        report_dir = self.data_dir / "validation_report"
        os.makedirs(report_dir, exist_ok=True)

        # Plot label distribution
        plot_data = pd.DataFrame(
            {
                "labels": list(results["label_stats"]["label_counts"].keys()),
                "values": list(results["label_stats"]["label_counts"].values()),
            }
        ).sort_values("values", ascending=False)

        plt.figure(figsize=(15, 5))
        sns.barplot(data=plot_data, x="labels", y="values")
        plt.xticks(rotation=45, ha="right")
        plt.title("Label Distribution")
        plt.tight_layout()
        plt.savefig(report_dir / "label_distribution.png")
        plt.close()

        # Generate text report
        with open(report_dir / "validation_report.txt", "w") as f:
            f.write("NIH Chest X-ray Dataset Validation Report\n")
            f.write("=====================================\n\n")

            f.write("Dataset Overview:\n")
            f.write(f"Total Samples: {results['data_stats']['total_samples']}\n")
            f.write(f"Missing Files: {len(results['data_stats']['missing_files'])}\n")
            f.write(
                f"Corrupted Files: {len(results['data_stats']['corrupted_files'])}\n\n"
            )

            f.write("Label Statistics:\n")
            for label, count in results["label_stats"]["label_counts"].items():
                percentage = (count / results["data_stats"]["total_samples"]) * 100
                f.write(f"{label}: {count} ({percentage:.1f}%)\n")

    def run_validation(self) -> Dict:
        """Run complete validation suite.

        Returns:
            Dictionary containing validation results
        """
        self.logger.info("Starting dataset validation...")
        results = {
            "data_stats": self._validate_data_stats(),
            "label_stats": self._validate_labels(),
        }
        self._generate_validation_report(results)
        return results
