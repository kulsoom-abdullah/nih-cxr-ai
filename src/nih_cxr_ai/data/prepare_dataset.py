# src/nih_cxr_ai/data/prepare_dataset.py
"""Dataset preparation utilities for NIH Chest X-ray dataset.

Handles downloading, processing, and organizing the NIH Chest X-ray dataset,
including creating proper train/validation/test splits and generating statistics.
"""

# Standard library imports
import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional

# Third-party imports
import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_nih_dataset(
    output_dir: str, full_dataset: bool = True, subset_size: Optional[int] = None
) -> Dict:
    """
    Prepare NIH dataset, either full or subset, with complete statistics.

    Args:
        output_dir: Directory to save the dataset
        full_dataset: Whether to process the full dataset
        subset_size: Number of images for subset if full_dataset is False

    Returns:
        Dict containing preparation and dataset statistics
    """
    # Create output directories
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    stats_dir = output_dir / "statistics"
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)

    stats = {
        "dataset_info": {
            "total_available": 0,
            "processed_images": 0,
            "errors": [],
            "class_distribution": {},
            "multi_label_stats": {},
            "image_stats": {},
        },
        "splits": {},
    }

    try:
        # Load full dataset first for statistics
        logger.info("Loading dataset from Hugging Face...")
        start_time = time.time()

        dataset = load_dataset(
            "alkzar90/NIH-Chest-X-ray-dataset",
            "image-classification",
            split="train",
            trust_remote_code=True,
        )

        stats["dataset_info"]["total_available"] = len(dataset)
        logger.info(f"Total available samples: {len(dataset)}")

        # Calculate full dataset statistics first
        logger.info("Calculating dataset statistics...")
        all_labels = []
        label_combinations = []
        for example in tqdm(dataset, desc="Analyzing labels"):
            all_labels.extend(example["labels"])
            label_combinations.append(tuple(sorted(example["labels"])))

        # Store full dataset statistics
        stats["dataset_info"]["class_distribution"] = (
            pd.Series(all_labels).value_counts().to_dict()
        )
        stats["dataset_info"]["multi_label_stats"] = {
            "avg_labels_per_image": np.mean(
                [len(labels) for labels in dataset["labels"]]
            ),
            "max_labels_per_image": max(len(labels) for labels in dataset["labels"]),
            "common_combinations": pd.Series(label_combinations)
            .value_counts()
            .head(10)
            .to_dict(),
        }

        # Save full statistics
        pd.DataFrame(
            {
                "class": list(stats["dataset_info"]["class_distribution"].keys()),
                "count": list(stats["dataset_info"]["class_distribution"].values()),
            }
        ).to_csv(stats_dir / "class_distribution.csv", index=False)

        # Determine processing size
        process_size = (
            len(dataset) if full_dataset else min(subset_size or 1000, len(dataset))
        )
        logger.info(f"Processing {process_size} images...")

        # Process images
        processing_subset = dataset.select(range(process_size))
        labels_data = []

        for idx in tqdm(range(len(processing_subset)), desc="Processing images"):
            try:
                example = processing_subset[idx]
                image_filename = f"image_{idx:06d}.png"
                image_path = images_dir / image_filename

                # Save image
                example["image"].save(image_path)

                # Store label information
                labels_data.append(
                    {
                        "image_file_path": f"images/{image_filename}",
                        "labels": str(example["labels"]),
                    }
                )

                stats["dataset_info"]["processed_images"] += 1

            except Exception as e:
                error_msg = f"Error processing image {idx}: {str(e)}"
                logger.error(error_msg)
                stats["dataset_info"]["errors"].append(error_msg)
                continue

        # Create and save splits
        labels_df = pd.DataFrame(labels_data)

        # Create stratified splits to maintain label distribution
        train_size = int(0.7 * len(labels_df))
        val_size = int(0.15 * len(labels_df))

        # Shuffle with fixed seed for reproducibility
        shuffled_df = labels_df.sample(frac=1, random_state=42).reset_index(drop=True)

        train_df = shuffled_df[:train_size]
        val_df = shuffled_df[train_size : train_size + val_size]
        test_df = shuffled_df[train_size + val_size :]

        # Save splits
        splits = {"train": train_df, "val": val_df, "test": test_df}

        for split_name, split_df in splits.items():
            split_df.to_csv(output_dir / f"{split_name}_labels.csv", index=False)
            stats["splits"][split_name] = len(split_df)

        # Save full labels
        labels_df.to_csv(output_dir / "labels.csv", index=False)

        # Save complete statistics
        pd.to_pickle(stats, stats_dir / "dataset_stats.pkl")

        logger.info("\nDataset preparation completed!")
        logger.info(
            f"Total processed: {stats['dataset_info']['processed_images']} images"
        )
        logger.info(f"Splits: {stats['splits']}")

    except Exception as e:
        logger.error(f"Error preparing dataset: {str(e)}")
        raise

    return stats


if __name__ == "__main__":
    try:
        # Process full dataset
        stats = prepare_nih_dataset(
            "src/data/nih_chest_xray",  # New directory for full dataset
            full_dataset=True,
        )
        logger.info("Full dataset preparation completed successfully!")

    except Exception as e:
        logger.error(f"Dataset preparation failed with error: {str(e)}")
