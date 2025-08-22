"""Configuration management utilities for the gender bias research pipeline."""

from pathlib import Path
from typing import Any

import yaml


class ConfigLoader:
    """Handles loading and accessing configuration parameters."""

    def __init__(self, config_path: str = None):
        """
        Initialize the configuration loader.

        Args:
            config_path: Path to the configuration YAML file.
                        If None, uses default config location.
        """
        if config_path is None:
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "config.yaml"

        self.config_path = Path(config_path)
        self._config = self._load_config()

    def _load_config(self) -> dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path) as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {e}")

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key_path: Dot-separated path to the configuration value (e.g., 'api.openai.model_name')
            default: Default value if key is not found

        Returns:
            Configuration value or default

        Example:
            config.get('api.openai.model_name')  # Returns 'gpt-4o-mini'
        """
        keys = key_path.split(".")
        value = self._config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def get_data_paths(self) -> dict[str, str]:
        """Get all data path configurations."""
        return self.get("data", {})

    def get_api_config(self, provider: str) -> dict[str, Any]:
        """Get API configuration for specific provider."""
        return self.get(f"api.{provider}", {})

    def get_model_config(self, model_type: str) -> dict[str, Any]:
        """Get model configuration for specific type."""
        return self.get(f"models.{model_type}", {})


# Global configuration instance
config = ConfigLoader()
