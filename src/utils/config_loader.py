"""Configuration loader utility."""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Dictionary containing configuration
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_all_configs(config_dir: str = "configs") -> Dict[str, Dict[str, Any]]:
    """
    Load all config files from a directory.
    
    Args:
        config_dir: Directory containing config files
        
    Returns:
        Dictionary with config names as keys and config dicts as values
    """
    config_dir_path = Path(config_dir)
    configs = {}
    
    for config_file in config_dir_path.glob("*.yaml"):
        config_name = config_file.stem
        configs[config_name] = load_config(str(config_file))
    
    return configs


def get_model_config(config_path: str, model_name: str) -> Dict[str, Any]:
    """
    Get specific model configuration.
    
    Args:
        config_path: Path to model_config.yaml
        model_name: Name of model (baseline, xgboost, lightgbm)
        
    Returns:
        Model configuration dictionary
    """
    config = load_config(config_path)
    return config.get(model_name, {})
