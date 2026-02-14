"""
Model Registry - Manages model configurations and metadata.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import yaml


logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a registered model."""

    name: str
    path: str
    input_shapes: Dict[str, List[int]]
    dtype: str = "float32"
    warmup: int = 10
    iterations: int = 100
    description: str = ""
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


from dataclasses import dataclass


class ModelRegistry:
    """
    Registry of available models with their configurations.
    Supports loading from YAML config files.
    """

    DEFAULT_MODELS = {
        "resnet50": {
            "input_shapes": {"input": [1, 3, 224, 224]},
            "dtype": "float32",
            "description": "ResNet-50 image classification",
            "tags": ["vision", "classification"],
        },
        "bert-base": {
            "input_shapes": {
                "input_ids": [1, 128],
                "attention_mask": [1, 128],
            },
            "dtype": "int64",
            "description": "BERT-base encoder",
            "tags": ["nlp", "encoder"],
        },
        "vit-base": {
            "input_shapes": {"pixel_values": [1, 3, 224, 224]},
            "dtype": "float32",
            "description": "Vision Transformer base",
            "tags": ["vision", "transformer"],
        },
        "gpt2": {
            "input_shapes": {"input_ids": [1, 256]},
            "dtype": "int64",
            "description": "GPT-2 language model",
            "tags": ["nlp", "decoder", "llm"],
        },
    }

    def __init__(self, config_path: Optional[str] = None, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models: Dict[str, ModelConfig] = {}

        # Load default models
        self._load_defaults()

        # Load from config if provided
        if config_path:
            self.load_config(config_path)

    def _load_defaults(self) -> None:
        """Load default model configurations."""
        for name, config in self.DEFAULT_MODELS.items():
            model_path = self.models_dir / f"{name}.onnx"
            self.models[name] = ModelConfig(
                name=name,
                path=str(model_path),
                input_shapes=config["input_shapes"],
                dtype=config.get("dtype", "float32"),
                description=config.get("description", ""),
                tags=config.get("tags", []),
            )

    def load_config(self, config_path: str) -> None:
        """Load model configurations from YAML file."""
        path = Path(config_path)

        if not path.exists():
            logger.warning(f"Model config not found: {config_path}")
            return

        with open(path, "r") as f:
            config = yaml.safe_load(f)

        for name, model_cfg in config.get("models", {}).items():
            model_path = model_cfg.get("path", str(self.models_dir / f"{name}.onnx"))

            self.models[name] = ModelConfig(
                name=name,
                path=model_path,
                input_shapes=model_cfg.get("input_shapes", {}),
                dtype=model_cfg.get("dtype", "float32"),
                warmup=model_cfg.get("warmup", 10),
                iterations=model_cfg.get("iterations", 100),
                description=model_cfg.get("description", ""),
                tags=model_cfg.get("tags", []),
            )

        logger.info(f"Loaded {len(config.get('models', {}))} models from {config_path}")

    def get(self, name: str) -> Optional[ModelConfig]:
        """Get model configuration by name."""
        return self.models.get(name)

    def list_models(self, tag: Optional[str] = None) -> List[str]:
        """List available model names, optionally filtered by tag."""
        if tag:
            return [name for name, cfg in self.models.items() if tag in cfg.tags]
        return list(self.models.keys())

    def register(self, config: ModelConfig) -> None:
        """Register a new model configuration."""
        self.models[config.name] = config
        logger.info(f"Registered model: {config.name}")

    def validate(self, name: str) -> bool:
        """Check if model file exists."""
        config = self.get(name)
        if not config:
            return False
        return Path(config.path).exists()
