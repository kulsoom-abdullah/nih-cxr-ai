# tests/test_imports.py


def test_core_imports():
    """Test importing core package components."""
    import nih_cxr_ai
    from nih_cxr_ai.data.nih_dataset import NIHChestDataModule
    from nih_cxr_ai.models.traditional import TraditionalCXRModel
    from nih_cxr_ai.utils.transforms import get_train_transforms

    # Simple assertions to verify objects are importable
    assert hasattr(TraditionalCXRModel, "__init__")
    assert hasattr(NIHChestDataModule, "__init__")
    assert callable(get_train_transforms)


def test_visualization_imports():
    """Test importing visualization components."""
    from nih_cxr_ai.utils.visualization.dataset_viz import DatasetVisualizer
    from nih_cxr_ai.utils.visualization.image_viz import ImageVisualizer

    assert hasattr(DatasetVisualizer, "__init__")
    assert hasattr(ImageVisualizer, "__init__")


def test_config_loading():
    """Test loading configuration files."""
    from pathlib import Path

    import yaml

    config_path = Path("configs/traditional_model.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Verify essential config sections exist
    assert "model" in config
    assert "data" in config
    assert "trainer" in config
