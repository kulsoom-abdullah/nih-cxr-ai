# src/config.py
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
from enum import Enum
import yaml
from pathlib import Path


class ModelType(Enum):
    TRADITIONAL = "traditional"
    FOUNDATION = "foundation"


@dataclass
class ModelConfig:
    """Model-specific configurations"""

    type: ModelType
    image_size: Tuple[int, int]

    @classmethod
    def get_config(cls, model_type: str) -> 'ModelConfig':
        """Factory method to get appropriate configuration.
        
        Args:
            model_type: String identifier for model type
            
        Returns:
            ModelConfig instance with appropriate settings
            
        Raises:
            ValueError: If unknown model type is provided
        """
        if model_type == ModelType.TRADITIONAL.value:
            return cls(
                type=ModelType.TRADITIONAL, 
                image_size=(224, 224)  # ResNet default
            )
        elif model_type == ModelType.FOUNDATION.value:
            return cls(
                type=ModelType.FOUNDATION,
                image_size=(1024, 1024)  # Foundation model default
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")


@dataclass
class TrainingConfig:
    """Complete training configuration"""

    model: ModelConfig
    batch_size: int = 32
    num_workers: int = 4
    learning_rate: float = 1e-4
    max_epochs: int = 100
    precision: str = "16-mixed"
    accelerator: str = "cuda"
    devices: int = 1
    wandb_project: str = "chest-xray-comparison"
    wandb_run_name: Optional[str] = None

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'TrainingConfig':
        """Create config from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            TrainingConfig instance
            
        Raises:
            FileNotFoundError: If YAML file doesn't exist
            ValueError: If YAML contains invalid configuration
        """
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        model_config = ModelConfig.get_config(config_dict.pop('model_type'))
        return cls(model=model_config, **config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary format.
        
        Returns:
            Dictionary containing all configuration parameters
        """
        return {
            'model_type': self.model.type.value,
            'image_size': self.model.image_size,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'learning_rate': self.learning_rate,
            'max_epochs': self.max_epochs,
            'precision': self.precision,
            'accelerator': self.accelerator,
            'devices': self.devices,
            'wandb_project': self.wandb_project,
            'wandb_run_name': self.wandb_run_name
        }


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from file or use defaults.
    
    Args:
        config_path: Optional path to YAML configuration file
        
    Returns:
        Dictionary containing configuration parameters
        
    Raises:
        FileNotFoundError: If config_path is provided but file doesn't exist
    """
    if config_path is None:
        # Return default configuration
        default_config = TrainingConfig(
            model=ModelConfig.get_config(ModelType.TRADITIONAL.value)
        )
        return default_config.to_dict()
    
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    return TrainingConfig.from_yaml(config_path).to_dict()

@dataclass
class TrainingConfig:
    """Complete training configuration"""

    model: ModelConfig
    batch_size: int = 32
    num_workers: int = 4
    learning_rate: float = 1e-4
    max_epochs: int = 100
    precision: str = "16-mixed"
    accelerator: str = "cuda"
    devices: int = 1
