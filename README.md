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

1. Prepare dataset:
```python
python -m nih_cxr_ai.data.prepare_dataset
```

2. Train model:
```python
# For full training
python -m nih_cxr_ai.train --config configs/traditional_model.yaml

# For quick testing
python -m nih_cxr_ai.train --config configs/test_traditional_config.yaml
```

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