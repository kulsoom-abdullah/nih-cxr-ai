# src/nih_cxr_ai/inference.py
"""Inference script for NIH Chest X-ray classification models.

This script:
- Loads a trained model checkpoint.
- Performs inference on a single image or a directory of images.
- Optionally compares predictions against ground truth if label_csv is provided.
- Optionally visualizes predictions if no --no-visualize flag is given.
- Optionally saves predictions to a CSV for later analysis if --output_csv is provided.

Usage:
    python -m nih_cxr_ai.inference --checkpoint path/to/checkpoint.ckpt --image_path path/to/images

Example with ground truth and output CSV:
    python -m nih_cxr_ai.inference --checkpoint path/to/checkpoint.ckpt \
      --image_path src/data/nih_chest_xray/images \
      --label_csv src/data/nih_chest_xray/test_labels.csv \
      --output_csv predictions.csv
"""


import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from nih_cxr_ai.models.traditional import TraditionalCXRModel
from nih_cxr_ai.utils.transforms import get_val_transforms
from nih_cxr_ai.utils.visualization.image_viz import ImageVisualizer

logger = logging.getLogger(__name__)

disease_labels = [
    "Atelectasis",
    "Consolidation",
    "Infiltration",
    "Pneumothorax",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Effusion",
    "Pneumonia",
    "Pleural_Thickening",
    "Cardiomegaly",
    "Nodule",
    "Mass",
    "Hernia",
]


def load_model(checkpoint_path: str, device: torch.device) -> TraditionalCXRModel:
    """Load the TraditionalCXRModel from a given checkpoint."""
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    model = TraditionalCXRModel.load_from_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()

    # Overwrite model disease_names with provided disease_labels
    if len(disease_labels) == model.num_classes:
        model.disease_names = disease_labels
    else:
        logger.warning(
            "Number of classes in model does not match provided disease_labels length."
        )

    return model


def load_images(image_path: Union[str, Path]) -> List[Path]:
    """Load image file paths from a given path.

    If image_path is a directory, load all .png/.jpg images inside it.
    If image_path is a single file, return just that file path.
    """
    image_path = Path(image_path)
    if image_path.is_dir():
        images = sorted(
            [
                p
                for p in image_path.iterdir()
                if p.suffix.lower() in [".png", ".jpg", ".jpeg"]
            ]
        )
        if not images:
            raise FileNotFoundError(f"No images found in directory: {image_path}")
        return images
    else:
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        return [image_path]


def load_labels(label_csv: Optional[str], num_classes: int) -> Dict[str, torch.Tensor]:
    """Load ground truth labels from a CSV file, if provided.

    The CSV must have columns: 'image_file_path' and 'labels'.
    'labels' should be a list of indices (e.g. [1, 2]) representing diseases.

    Args:
        label_csv: Path to a CSV file containing ground truth labels.
        num_classes: Number of classes to create multi-hot vectors.

    Returns:
        A dictionary mapping filename (e.g., '00000001_000.png') to
        a torch.Tensor of shape [num_classes].
    """
    if label_csv is None:
        return {}
    df = pd.read_csv(label_csv)
    if "image_file_path" not in df.columns or "labels" not in df.columns:
        raise ValueError("CSV must contain 'image_file_path' and 'labels' columns.")

    if isinstance(df["labels"].iloc[0], str):
        df["labels"] = df["labels"].apply(eval)

    label_map = {}
    for _, row in df.iterrows():
        filename = Path(row["image_file_path"]).name
        lbls = row["labels"]
        label_tensor = torch.zeros(num_classes, dtype=torch.long)
        label_tensor[lbls] = 1
        label_map[filename] = label_tensor
    return label_map


def run_inference(
    model: TraditionalCXRModel,
    image_paths: List[Path],
    device: torch.device,
    label_map: Dict[str, torch.Tensor],
    visualize: bool = True,
    output_dir: Path = Path("results/inference"),
    output_csv: Optional[str] = None,
) -> None:
    transforms = get_val_transforms(image_size=(224, 224))
    output_dir.mkdir(parents=True, exist_ok=True)

    disease_names = model.disease_names
    logger.info(f"Inference on {len(image_paths)} images...")

    visualizer = ImageVisualizer(save_dir=output_dir)

    results = []
    for idx, img_path in enumerate(
        tqdm(image_paths, desc="Running Inference", unit="image")
    ):
        img = Image.open(img_path).convert("RGB")
        img_tensor = transforms(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.sigmoid(logits).cpu().squeeze(0).numpy()

        filename = img_path.name
        true_labels = label_map.get(
            filename, torch.zeros(len(disease_names), dtype=torch.long)
        ).numpy()

        # Log if not quiet
        if logger.level <= logging.INFO:
            logger.info(f"Image {idx+1}: {filename}")
            for i, d_name in enumerate(disease_names):
                logger.info(f"  {d_name}: {probs[i]:.4f} (GT: {true_labels[i]})")

        # Visualize if requested
        if visualize:
            visualizer.visualize_predictions(
                image=img_tensor.squeeze(0).cpu(),
                true_labels=true_labels,
                pred_probs=probs,
                disease_names=disease_names,
                save_name=f"inference_{idx}",
            )
            if logger.level <= logging.INFO:
                logger.info(
                    f"Visualization saved to {output_dir / f'inference_{idx}.png'}"
                )

        # Save to results list for CSV
        row = {"filename": filename}
        for i, d_name in enumerate(disease_names):
            row[f"true_{d_name}"] = int(true_labels[i])
            row[f"pred_{d_name}"] = float(probs[i])
        results.append(row)

    # If output_csv is provided, save the predictions
    if output_csv:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        logger.info(f"Predictions saved to {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Inference script for NIH Chest X-ray model."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint (.ckpt)"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to an image file or directory",
    )
    parser.add_argument(
        "--label_csv",
        type=str,
        default=None,
        help="Optional CSV with ground truth labels",
    )
    parser.add_argument(
        "--no-visualize", action="store_true", help="Disable visualization output"
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce logging verbosity")
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Optional path to save predictions to a CSV file",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    if args.quiet:
        logger.setLevel(logging.WARNING)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    image_paths = load_images(args.image_path)
    label_map = load_labels(args.label_csv, num_classes=model.num_classes)

    run_inference(
        model=model,
        image_paths=image_paths,
        device=device,
        label_map=label_map,
        visualize=not args.no_visualize,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
