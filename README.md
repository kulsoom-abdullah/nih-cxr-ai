# NIH Chest X-ray Classification Project

A deep learning pipeline for multi-label chest X-ray disease classification using the NIH dataset. This implementation follows the baseline architecture and methodology from [Kufel et al. (2023)](https://doi.org/10.3390/jpm13101426) while providing a foundation for comparison with newer foundation models.

## Baseline Model

Our baseline implementation follows the approach described in "Multi-Label Classification of Chest X-ray Abnormalities Using Transfer Learning Techniques" (J. Pers. Med. 2023, 13, 1426). Key components include:

- **Architecture**: EfficientNet-B1 backbone with custom classification head
- **Classes**: 14 pathology classes (excluding "No Finding" as a separate class)
- **Data Augmentation**:
  - Random rotation ±5°
  - Random horizontal flip (p=0.3)
  - Random brightness adjustment (±0.2)
- **Training**:
  - Binary Cross Entropy Loss
  - Learning rate: 1e-4
  - Adam optimizer with ReduceLROnPlateau scheduling

The baseline achieved state-of-the-art performance with AUC-ROC scores ranging from 0.71 to 0.91 across different pathologies.

## Current Status

This project is in active development. Current features:
- Complete implementation of baseline model architecture and training pipeline
- Multi-label classification using PyTorch Lightning
- Data augmentation matching published methodology
- Experiment tracking with Weights & Biases
- Comprehensive evaluation metrics across all disease categories

## Installation

```bash
# Clone repository
git clone https://github.com/kulsoom-abdullah/nih-cxr-ai.git
cd nih-cxr-ai

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install package
pip install -e .
```

## Usage


## Data Preparation

The NIH Chest X-ray dataset is large and not included directly in this repo. Instead, scripts download and prepare the data:

1. **Download and Prepare Data:**
   ```bash
   python -m nih_cxr_ai.data.prepare_dataset

This will:

- Download `Data_Entry_2017_v2020.csv`,`train_val_list.txt`, and `test_list.txt` from Hugging Face.
- Download and extract all image zip files, preserving original NIH filenames.
- Merge metadata (age, gender, view position) into a final `labels.csv`.
- Create `train_labels.csv`, `val_labels.csv`, and `test_labels.csv` splits.
- The final `labels.csv` and splits will be located under `src/data/nih_chest_xray/`.

# Explore dataset characteristics and distribution
jupyter notebook notebooks/01_data_exploration.ipynb

Note: The images are large and not stored in this repository. You must run the preparation step to obtain them. For reference, see the official [Hugging Face](https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset) repository for the dataset.

2. Train model:
With the data prepared, you can train a baseline model:
```python
# For full training
python -m nih_cxr_ai.train --config configs/traditional_model.yaml
```
This trains a multi-label classification model using the prepared splits and logs metrics per disease. You can monitor training progress and metrics via Weights & Biases if configured.

## Project Structure

```
nih-cxr-ai/
├── src/nih_cxr_ai/       # Main package
│   ├── data/            # Data handling
│   ├── models/          # Model implementations
│   └── utils/           # Utilities
├── configs/             # Training configurations
├── notebooks/          # Analysis notebooks
└── tests/             # Test suite
```

## References

1. Kufel, J.; et al. Multi-Label Classification of Chest X-ray Abnormalities Using Transfer Learning Techniques. J. Pers. Med. 2023, 13, 1426. https://doi.org/10.3390/jpm13101426

## License

MIT License

## Acknowledgments

- NIH for the Chest X-ray dataset
- Lightning AI team for PyTorch Lightning
- Weights & Biases for experiment tracking

## Next Steps
- Integrate a foundation model and compare performance against the baseline.
- Add subgroup analyses (age, gender, view) to results.
- Log test evaluation metrics and possibly subgroup metrics to W&B.


