# src/nih_cxr_ai/data/prepare_dataset.py
"""Dataset preparation utilities for NIH Chest X-ray dataset.

Handles downloading, processing, and organizing the NIH Chest X-ray dataset,
including creating proper train/validation/test splits and generating statistics.
"""

# Standard library imports
import logging
import os
import shutil
from pathlib import Path
from typing import Dict

# Third-party imports
import numpy as np
import pandas as pd

# from datasets import load_dataset
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_if_needed(
    repo_id: str, filename: str, output_dir: Path, overwrite: bool
) -> str:
    """Download file from Hugging Face Hub if not present or overwrite is True."""
    local_path = (
        output_dir / Path(filename).name
    )  # Extract just the filename part to save locally
    if local_path.exists() and not overwrite:
        logger.info(f"File {local_path.name} already exists. Skipping download.")
        return str(local_path)
    logger.info(f"Downloading {filename}...")
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
        repo_type="dataset",
    )


def prepare_nih_dataset_from_hf(output_dir: str, overwrite: bool = False) -> Dict:
    """
    Prepare NIH dataset from Hugging Face by downloading original files, extracting them,
    creating splits, and merging metadata.

    Args:
        output_dir: Directory to save the dataset
        overwrite: If True, re-download files even if they exist.
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    stats_dir = output_dir / "statistics"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)

    print(f"Created directories:\n{output_dir}\n{images_dir}\n{stats_dir}\n")

    repo_id = "alkzar90/NIH-Chest-X-ray-dataset"

    # Correct remote filenames as per Hugging Face repo structure
    remote_metadata_file = "data/Data_Entry_2017_v2020.csv"
    remote_train_val_list = "data/train_val_list.txt"
    remote_test_list = "data/test_list.txt"
    image_archives = [f"data/images/images_{i:03d}.zip" for i in range(1, 13)]

    logger.info("Downloading metadata and split files if needed...")
    data_entry_path = download_if_needed(
        repo_id, remote_metadata_file, output_dir, overwrite
    )
    train_val_list_path = download_if_needed(
        repo_id, remote_train_val_list, output_dir, overwrite
    )
    test_list_path = download_if_needed(
        repo_id, remote_test_list, output_dir, overwrite
    )

    logger.info("Checking image zip files...")
    archive_paths = []
    for arch in image_archives:
        arch_path = download_if_needed(repo_id, arch, output_dir, overwrite)
        archive_paths.append(arch_path)

    if any(images_dir.iterdir()) and not overwrite:
        logger.info(
            "Images dir not empty. Assuming images are already extracted. Skipping extraction."
        )
    else:
        logger.info("Extracting images from zip files...")
        for arch_path in archive_paths:
            logger.info(f"Extracting {arch_path}...")
            shutil.unpack_archive(str(arch_path), extract_dir=str(output_dir))
            logger.info(f"Extraction of {arch_path} completed.")
            os.remove(arch_path)
            logger.info(f"Deleted archive {arch_path} after extraction.")

    # Load splits
    with open(test_list_path, "r") as f:
        test_files = set(line.strip() for line in f if line.strip())
    with open(train_val_list_path, "r") as f:
        train_val_files = set(line.strip() for line in f if line.strip())

    all_images = list(images_dir.glob("*.png"))
    all_filenames = [img.name for img in all_images]

    train_val_df = pd.DataFrame(
        {"Image Index": [f for f in all_filenames if f in train_val_files]}
    )
    test_df = pd.DataFrame(
        {"Image Index": [f for f in all_filenames if f in test_files]}
    )

    np.random.seed(2137)
    shuffled_train_val = train_val_df.sample(frac=1, random_state=2137).reset_index(
        drop=True
    )
    train_size = int(0.8 * len(shuffled_train_val))
    train_df = shuffled_train_val[:train_size]
    val_df = shuffled_train_val[train_size:]

    labels_df = pd.read_csv(
        f"{output_dir}/labels.csv"
    )  # This has image_file_path, labels, etc.

    labels_df["Image Index"] = labels_df["image_file_path"].apply(
        lambda x: x.replace("images/", "")
    )

    train_merged = pd.merge(train_df, labels_df, on="Image Index", how="left")
    val_merged = pd.merge(val_df, labels_df, on="Image Index", how="left")
    test_merged = pd.merge(test_df, labels_df, on="Image Index", how="left")

    # Now train_merged, val_merged, test_merged have the required columns
    train_merged.to_csv(f"{output_dir}/train_labels.csv", index=False)
    val_merged.to_csv(f"{output_dir}/val_labels.csv", index=False)
    test_merged.to_csv(f"{output_dir}/test_labels.csv", index=False)

    # Create directories for subsets
    train_dir = images_dir / "train"
    val_dir = images_dir / "val"
    test_dir = images_dir / "test"
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)

    # Function to copy images to their respective directory
    def copy_images(df: pd.DataFrame, subset_dir: Path):
        for img_name in df["Image Index"]:
            src = images_dir / img_name
            dest = subset_dir / img_name
            if not src.exists():
                # If an image is missing, raise an error or handle gracefully
                raise FileNotFoundError(f"Image {src} not found.")
            shutil.copy(src, dest)

    # Copy images for each subset
    copy_images(train_df, train_dir)
    copy_images(val_df, val_dir)
    copy_images(test_df, test_dir)

    for f in images_dir.glob("*.png"):
        f.unlink()  # remove original image

    # Merge metadata
    entry_df = pd.read_csv(data_entry_path)
    disease_names = [
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

    def parse_labels(label_str):
        if label_str == "No Finding":
            return []
        diseases = label_str.split("|")
        indices = []
        for d in diseases:
            if d in disease_names:
                indices.append(disease_names.index(d))
        return indices

    entry_df["labels"] = entry_df["Finding Labels"].apply(parse_labels)
    entry_df["image_file_path"] = "images/" + entry_df["Image Index"]

    merged_labels = entry_df[
        ["image_file_path", "labels", "Patient Age", "Patient Gender", "View Position"]
    ]
    merged_labels.to_csv(output_dir / "labels.csv", index=False)
    logger.info("Final labels.csv with metadata saved!")

    stats = {
        "splits": {"train": len(train_df), "val": len(val_df), "test": len(test_df)}
    }

    return stats


if __name__ == "__main__":
    try:
        stats = prepare_nih_dataset_from_hf("src/data/nih_chest_xray", overwrite=False)
        logger.info("Full dataset preparation completed successfully!")
    except Exception as e:
        logger.error(f"Dataset preparation failed with error: {str(e)}")
