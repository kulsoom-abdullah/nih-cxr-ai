# NIH Chest X-ray Classification Project

A deep learning pipeline for multi-label chest X-ray disease classification using the NIH dataset, implemented with PyTorch Lightning.

## Current Status

This project is in active development. Current features:
- Data pipeline with validation and preprocessing
- Multi-label classification using PyTorch Lightning
- Basic visualization tools for dataset analysis
- Experiment tracking with Weights & Biases

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

1. Prepare and analyze dataset:
```python
# Prepare dataset
python -m nih_cxr_ai.data.prepare_dataset

# Explore dataset characteristics
jupyter notebook notebooks/01_data_exploration.ipynb


2. Train model:
```python
python -m nih_cxr_ai.train --config configs/traditional_model.yaml
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

## Development

Run tests:
```bash
python -m pytest tests/
```

Format code:
```bash
pre-commit run --all-files
```

## License

MIT License

## Acknowledgments

- NIH for the Chest X-ray dataset
- Lightning AI team for PyTorch Lightning
- Weights & Biases for experiment tracking
