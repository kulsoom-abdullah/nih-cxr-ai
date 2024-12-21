# NIH Chest X-ray Classification Project

## **Overview**
This repository demonstrates a modular, scalable deep learning pipeline for multi-label chest X-ray disease classification using the NIH dataset. The project highlights modern engineering practices and MLOps tools such as PyTorch Lightning, Hugging Face, and Weights & Biases (W&B). While leveraging NIH's limited metadata (age, gender, view position), the project provides a robust baseline for multi-label classification and explores subgroup performance.

## **Key Features**
- **Modular Design**:
  - Clear separation of data preparation, model training, evaluation, and visualization.
- **MLOps Integration**:
  - Experiment tracking with Weights & Biases (W&B).
  - Scalable and reproducible training with PyTorch Lightning.
  - Environment reproducibility managed via Lightning AI Studios.
- **Hugging Face Integration for Data**:
  - Used Hugging Face Hub for downloading NIH X-ray images and metadata.
- **Subgroup Analysis**:
  - Performance metrics by age, gender, and view position.
- **End-to-End Workflow**:
  - Dataset preparation, augmentation, training, inference, and visualization.

## **Installation**
### **Lightning AI Note**:
Lightning AI Studios simplifies workflows by abstracting traditional virtual environments like `venv` or `conda`. However, this project can be run outside Lightning Studios with a virtual environment:

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

## **Usage**

### **1. Dataset Preparation**
Prepare the NIH dataset for training:
```bash
python -m nih_cxr_ai.data.prepare_dataset
```
This script:
- Downloads NIH X-ray images and metadata from [Hugging Face](https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset).
- Creates `train_labels.csv`, `val_labels.csv`, and `test_labels.csv` splits.
- Outputs the final processed dataset under `src/data/nih_chest_xray`.

**Note**: The `-m` flag allows you to run Python modules as scripts. This setup aligns with the package configuration (`setup.py`) and ensures module paths are resolved correctly.

### **2. Model Training**
Train a multi-label classification model:
```bash
python -m nih_cxr_ai.train --config configs/traditional_model.yaml
```

### **3. Inference**
Run inference on test images:
```bash
python -m nih_cxr_ai.inference \
  --checkpoint path/to/checkpoint.ckpt \
  --image_path src/data/nih_chest_xray/images/test \
  --label_csv src/data/nih_chest_xray/test_labels.csv \
  --output_csv test_predictions.csv
```

### **4. Subgroup Analysis**
Explore model performance by subgroups (e.g., age, gender, view position):
- Use the `02_subgroup_analysis.ipynb` notebook to load predictions and compute subgroup metrics.

### **5. Visual EDA**
Conduct visual exploratory data analysis:
- Use the `01_data_exploration.ipynb` notebook for visualizing data distributions, label frequencies, and correlations.

## **Results**
The traditional model achieves AUROCs ranging from ~0.69 (Pneumothorax) to ~0.89 (Cardiomegaly) on the test set. Subgroup analyses highlight performance disparities across age groups, genders, and view positions, demonstrating the importance of richer datasets with diverse metadata.

## **Lessons Learned**
- The NIH dataset, while popular, lacks diversity and richer metadata (e.g., race, socioeconomic status, insurance coverage, geography).
- Subgroup analysis is constrained by the limited metadata, making it harder to fully evaluate biases or disparities.

## **Future Work**
- Extend the pipeline to publicly available X-ray datasets with greater diversity and richer metadata. This could allow deeper bias analyses and the evaluation of fairness and disparity across subgroups.
  - Note: MIMIC was not considered as it is part of the training data used in Google Foundation CXR. The original plan for this project was to experiment with whether a foundational model could help address bias and disparities.

- Expand test coverage beyond the current `test_imports.py` to include functional and integration tests.

## **Acknowledgments**
This project was completed independently, with the help of advanced AI tools such as OpenAI's ChatGPT and Anthropic's Claude Sonnet for brainstorming, debugging, and refining code.

## **References**
1. Wang, X.; et al. (2017). ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases. [http://arxiv.org/abs/1705.02315](http://arxiv.org/abs/1705.02315})
2. Kufel, J.; et al. Multi-Label Classification of Chest X-ray Abnormalities Using Transfer Learning Techniques. J. Pers. Med. 2023, 13, 1426. [https://doi.org/10.3390/jpm13101426](https://doi.org/10.3390/jpm13101426)

- This paper provided the latest NIH-specific baseline, inspiring:
- **Architecture**: EfficientNet-B1 backbone with a custom classification head.
- **Classes**: 14 pathology classes (excluding "No Finding" as a separate class).
- **Data Augmentation**:
  - Random rotation ±5°.
  - Random horizontal flip (p=0.3).
  - Random brightness adjustment (±0.2).
- **Training Setup**:
  - Binary Cross Entropy Loss.
  - Learning rate: 1e-4.
  - Adam optimizer with ReduceLROnPlateau scheduling.

## **Project Structure**
```
nih-cxr-ai/
├── src/nih_cxr_ai/       # Main package
│   ├── data/            # Data handling
│   ├── models/          # Model implementations
│   └── utils/           # Utilities
├── configs/             # Training configurations
├── notebooks/          # Analysis notebooks
├── tests/             # Test imports only (currently)
└── .pre-commit-config.yaml # Pre-commit hooks for clean code
```

## **Code Quality**
- The repository includes `.pre-commit-config.yaml` to enforce clean coding practices using tools like `black`, `isort`, and `flake8`.

---