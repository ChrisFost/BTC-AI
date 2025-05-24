#!/usr/bin/env python
"""
Configuration management for the BTC-AI trading system.

This module provides a centralized configuration system with validation,
versioning, and backward compatibility support.
"""

import os
import json
import logging
import importlib
from typing import Dict, Any, List, Optional, Union
import copy
import datetime
from pathlib import Path
from src.utils.config_versioning import ConfigVersioning

# Try to import log manager
try:
    log_manager_module = importlib.import_module("src.utils.log_manager")
    LogManager = log_manager_module.LogManager
    logger = LogManager.get_logger('config')
except ImportError:
    # Fallback logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('config')
    print("Using fallback logging for TradeConfig")

class TradeConfig:
    """
    Manages trading system configuration with validation and versioning.
    
    This class provides a centralized configuration system for the BTC-AI trading system,
    handling configuration loading, saving, validation, and access. It implements a singleton
    pattern to ensure consistent configuration across the application.
    
    Note: This class replaces the deprecated ConfigManager class.
    Migration guide available in docs/config_system_migration.md
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """Ensure singleton pattern for TradeConfig."""
        if cls._instance is None:
            cls._instance = super(TradeConfig, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None, force_reload: bool = False):
        """
        Initialize the configuration system.
        
        Args:
            config_path: Optional path to configuration file
            force_reload: Whether to force reloading even if already initialized
        """
        # Skip if already initialized unless force_reload is True
        if hasattr(self, '_initialized') and self._initialized and not force_reload:
            return
            
        # Setup base attributes
        self._initialized = True
        
        # Determine project root directory and configuration paths
        self._determine_paths()
        
        # Convert config_path to Path if it's a string
        if isinstance(config_path, str):
            config_path = Path(config_path)
        
        # Set config_path
        self.config_path = config_path or self._get_default_config_path()
        
        # Load configuration with versioning support
        self.config = self._load_config()
        
        # Apply environment overrides if any
        self._apply_env_overrides()
        
        logger.info(f"Trade configuration loaded from {self.config_path}")
    
    def _determine_paths(self) -> None:
        """Determine project paths for configuration lookup."""
        # Find the module's location
        current_dir = os.path.dirname(os.path.abspath(__file__))  # src/utils
        src_dir = os.path.dirname(current_dir)                    # src
        self.base_dir = os.path.dirname(src_dir)                  # project root
        
        # Define standard configuration paths
        self.config_paths = [
            os.path.join(self.base_dir, "configs", "config.json"),
            os.path.join(self.base_dir, "config", "config.json"),
            os.path.join(self.base_dir, "config.json"),
            os.path.join(self.base_dir, "Scripts", "final_config.json")
        ]
    
    def _get_default_config_path(self) -> Path:
        """Get the default configuration path."""
        # Try standard paths
        for path in self.config_paths:
            if os.path.exists(path):
                return Path(path)
        
        # If no configuration found, use default location
        return Path(os.path.join(self.base_dir, "configs", "config.json"))
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration with versioning support.
        
        Returns:
            Dict[str, Any]: Loaded configuration
        """
        try:
            # If explicit path provided, check it first
            if self.config_path.exists():
                return ConfigVersioning.load_config_with_versioning(self.config_path)
            
            # Try standard paths if the explicit path doesn't exist
            for path in self.config_paths:
                path_obj = Path(path)
                if path_obj.exists():
                    self.config_path = path_obj
                    return ConfigVersioning.load_config_with_versioning(path_obj)
            
            # If no configuration found, use default
            logger.warning("No configuration file found. Using default configuration.")
            return ConfigVersioning.DEFAULT_CONFIG.copy()
            
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return ConfigVersioning.DEFAULT_CONFIG.copy()
    
    def _apply_env_overrides(self) -> None:
        """Apply environment-specific configuration overrides."""
        # Environment overrides (future feature)
        pass
    
    def save(self) -> bool:
        """
        Save the current configuration.
        
        Returns:
            bool: True if save was successful, False otherwise
        """
        # Ensure the directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        return ConfigVersioning.save_config_with_versioning(self.config, self.config_path)
    
    def update(self, new_config: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            new_config: Dictionary of new configuration values
        """
        self._deep_update(self.config, new_config)
    
    def _deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
        """
        Recursively update a dictionary with another dictionary.
        
        Args:
            base_dict: Base dictionary to update
            update_dict: Dictionary with new values
        """
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def as_dict(self) -> Dict[str, Any]:
        """
        Get the current configuration as a dictionary.
        
        Returns:
            Dict[str, Any]: Current configuration
        """
        return self.config.copy()
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        
        Args:
            key: Configuration key (can use dot notation for nested keys or legacy flat keys)
            default: Default value if key doesn't exist
            
        Returns:
            Any: Configuration value
        """
        try:
            # Handle dot notation
            if '.' in key:
                value = self.config
                for k in key.split('.'):
                    value = value[k]
                return value
            
            # Handle legacy flat keys with backward compatibility mapping
            legacy_key_mapping = {
                "BUCKET": "trading.bucket",
                "INITIAL_CAPITAL": "trading.initial_capital", 
                "MAX_POSITION_HOLDINGS": "trading.max_positions",
                "MAX_POSITIONS": "trading.max_positions",
                "WINDOW_SIZE": "trading.window_size",
                "LOOK_BACK_AMOUNT": "trading.look_back_amount",
                "LOOK_BACK_UNIT": "trading.look_back_unit",
                "RESUME_CHECKPOINT": "trading.resume_checkpoint",
                "CHECKPOINT_INTERVAL": "trading.checkpoint_interval",
                "HIDDEN_SIZE": "model.hidden_size",
                "LEARNING_RATE": "model.learning_rate",
                "BATCH_SIZE": "model.batch_size",
                "GAMMA": "model.gamma",
                "EPS_CLIP": "model.eps_clip",
                "ENTROPY_COEF": "model.entropy_coef",
                "VALUE_COEF": "model.value_coef",
                "MAX_GRAD_NORM": "model.max_grad_norm",
                "WEIGHT_DECAY": "model.weight_decay",
                "MAX_BTC_PER_POSITION": "risk.max_btc_per_position",
                "MAX_USD_PER_POSITION": "risk.max_usd_per_position", 
                "MAX_VOLUME_PERCENTAGE": "risk.max_volume_percentage",
                "STOP_LOSS": "risk.stop_loss",
                "TAKE_PROFIT": "risk.take_profit",
                "MAX_DRAWDOWN": "risk.max_drawdown",
                "RISK_MANAGEMENT": "risk.risk_management"
            }
            
            # Check if it's a legacy key
            if key in legacy_key_mapping:
                nested_key = legacy_key_mapping[key]
                return self.get(nested_key, default)
            
            # Check for special keys that need to be computed
            if key == "MODELS_DIR":
                # Return the standard Models directory path
                return os.path.join(self.base_dir, "Models")
            
            # Direct key access (for keys that might still be flat)
            return self.config.get(key, default)
            
        except (KeyError, TypeError):
            return default
    
    def __getitem__(self, key: str) -> Any:
        """
        Get a configuration value using dictionary-like access.
        
        Args:
            key: Configuration parameter key
            
        Returns:
            The configuration value
            
        Raises:
            KeyError: If the key is not found in the configuration
        """
        if key in self.config:
            return self.config[key]
        raise KeyError(f"Configuration key '{key}' not found")
    
    def __setitem__(self, key: str, value: Any) -> None:
        """
        Set a configuration value using dictionary-like access.
        
        Args:
            key: Configuration parameter key
            value: New value
        """
        self.config[key] = value
    
    def __contains__(self, key: str) -> bool:
        """
        Check if a key exists in the configuration.
        
        Args:
            key: Configuration parameter key
            
        Returns:
            True if the key exists, False otherwise
        """
        return key in self.config
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value by key.
        
        Args:
            key: Configuration key (can use dot notation for nested keys)
            value: Value to set
        """
        # Handle dot notation
        if '.' in key:
            keys = key.split('.')
            current = self.config
            for k in keys[:-1]:
                current = current.setdefault(k, {})
            current[keys[-1]] = value
        else:
            self.config[key] = value
    
    def get_path(self, key: str, create: bool = False) -> str:
        """
        Get a file or directory path from configuration.
        
        Args:
            key: Path configuration key (e.g., "MODELS_DIR")
            create: Whether to create the directory if it doesn't exist
            
        Returns:
            Absolute path
        """
        # Standard directory names
        standard_dirs = {
            "MODELS_DIR": "Models",
            "DATA_DIR": "data",
            "LOGS_DIR": "Logs",
            "CACHE_DIR": "Cache",
            "OUTPUTS_DIR": "outputs",
            "CONFIG_DIR": "configs",
            "SCRIPTS_DIR": "Scripts"
        }
        
        # Get the path from config or use standard directory
        rel_path = self.get(key)
        if not rel_path and key in standard_dirs:
            rel_path = standard_dirs[key]
        
        if not rel_path:
            raise ValueError(f"Configuration does not contain path for {key}")
        
        # Convert to absolute path
        abs_path = os.path.join(self.base_dir, rel_path)
        
        # Create directory if needed
        if create:
            os.makedirs(abs_path, exist_ok=True)
            logger.info(f"Created directory: {abs_path}")
        
        return abs_path
    
    def reload(self) -> None:
        """Reload configuration from the original source."""
        self.__init__(self.config_path, force_reload=True)
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get the complete configuration dictionary.
        
        Returns:
            Complete configuration dictionary (copy)
        """
        return copy.deepcopy(self.config)
    
    def get_section(self, section_name: str) -> Dict[str, Any]:
        """
        Get a configuration section.
        
        Args:
            section_name: Name of the configuration section
            
        Returns:
            Dictionary with section configuration or empty dict if not found
        """
        # Check for a dictionary section
        if section_name in self.config and isinstance(self.config[section_name], dict):
            return copy.deepcopy(self.config[section_name])
        
        # Check for prefix-based sections (e.g., "RISK_*")
        prefix = f"{section_name}_"
        result = {}
        
        for key, value in self.config.items():
            if key.startswith(prefix):
                short_key = key[len(prefix):]
                result[short_key] = value
        
        return result
    
    def from_ui_values(self, values: Dict[str, Any]) -> None:
        """
        Update configuration from UI values dictionary.
        
        Args:
            values: Dictionary of values from PySimpleGUI window
        """
        # Update config from UI values (key mapping)
        ui_keys = [
            "BUCKET", "WINDOW_SIZE", "INITIAL_CAPITAL", "MAX_POSITION_HOLDINGS",
            "LOOK_BACK_AMOUNT", "LOOK_BACK_UNIT", "HIDDEN_SIZE", "LEARNING_RATE",
            "PPO_EPOCHS", "BATCH_SIZE", "RESUME_CHECKPOINT", "USE_PROBABILISTIC",
            "USE_FUSION", "GAMMA", "EPS_CLIP", "ENTROPY_COEF", "PREDICTION_BONUS",
            "NOVELTY_BONUS_WEIGHT", "ES_INTERVAL", "ES_POPULATION", "ES_MUTATION_RATE",
            "MAX_BTC_PER_POSITION", "MAX_USD_PER_POSITION", "MAX_VOLUME_PERCENTAGE"
        ]
        
        # Add naturalistic learning keys
        ui_keys.extend([
            "USE_SURPRISE_REPLAY", "SURPRISE_THRESHOLD", "REPLAY_BUFFER_SIZE",
            "PRIORITY_ALPHA", "PRIORITY_BETA", "USE_META_LEARNING", "ADAPT_PARAM_FREQ",
            "USE_POST_TRADE_ANALYSIS", "LESSON_MEMORY_SIZE", "INITIAL_EXPLORATION_RATE",
            "MIN_EXPLORATION_RATE", "EXPLORATION_DECAY", "EXPLORATION_DECAY_METHOD",
            "USE_CONTEXTUAL_MEMORY", "MEMORY_CAPACITY", "SIMILARITY_THRESHOLD",
            "MEMORY_RECALL_COUNT", "USE_CROSS_BUCKET_TRANSFER", "WEIGHT_TRANSFER_ALPHA",
            "FEATURE_TRANSFER_ALPHA", "TRANSFER_COOLDOWN", "ENABLE_REVERSE_TRANSFER"
        ])
        
        # Add probabilistic prediction keys
        ui_keys.extend([
            "CONFIDENCE_THRESHOLD", "CALIBRATION_WEIGHT", "POSITION_SIZING_STRATEGY",
            "UNCERTAINTY_PENALTY", "MIN_CONFIDENCE_THRESHOLD", "MAX_UNCERTAINTY_THRESHOLD",
            "USE_DYNAMIC_HORIZONS", "MIN_HORIZON", "MAX_HORIZON", "HORIZON_DENSITY",
            "HORIZON_UPDATE_FREQ"
        ])
        
        # Update any matching keys
        for key in ui_keys:
            if key in values:
                self.config[key] = values[key]
        
        # Special handling for nested dictionaries
        if "withdrawal_simulation" in self.config and isinstance(self.config["withdrawal_simulation"], dict):
            for subkey in self.config["withdrawal_simulation"].keys():
                full_key = f"withdrawal_simulation_{subkey}"
                if full_key in values:
                    self.config["withdrawal_simulation"][subkey] = values[full_key]
        
        # Handle bucket-specific settings
        bucket = values.get("BUCKET", self.config.get("BUCKET", "Scalping"))
        bucket_prefix = f"{bucket.lower()}_"
        
        for key, value in values.items():
            if key.startswith(bucket_prefix):
                # Extract the setting name without the bucket prefix
                setting_name = key[len(bucket_prefix):]
                self.config[setting_name] = value
    
    def validate(self) -> bool:
        """
        Validate the current configuration.
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        # Validate version
        if not ConfigVersioning.validate_config_version(self.config):
            logger.error("Invalid configuration version")
            return False
        
        # Validate required sections
        required_sections = ["trading", "model", "risk"]
        for section in required_sections:
            if section not in self.config:
                logger.error(f"Missing required section: {section}")
                return False
        
        # Validate trading section
        trading = self.config["trading"]
        if not isinstance(trading.get("bucket"), str):
            logger.error("trading.bucket must be a string")
            return False
        if not isinstance(trading.get("initial_capital"), (int, float)):
            logger.error("trading.initial_capital must be a number")
            return False
        if not isinstance(trading.get("max_positions"), int):
            logger.error("trading.max_positions must be an integer")
            return False
        
        # Validate model section
        model = self.config["model"]
        if not isinstance(model.get("hidden_size"), int):
            logger.error("model.hidden_size must be an integer")
            return False
        if not isinstance(model.get("learning_rate"), (int, float)):
            logger.error("model.learning_rate must be a number")
            return False
        if not isinstance(model.get("batch_size"), int):
            logger.error("model.batch_size must be an integer")
            return False
        
        # Validate risk section
        risk = self.config["risk"]
        if not isinstance(risk.get("max_btc_per_position"), (int, float)):
            logger.error("risk.max_btc_per_position must be a number")
            return False
        if not isinstance(risk.get("max_usd_per_position"), (int, float)):
            logger.error("risk.max_usd_per_position must be a number")
            return False
        if not isinstance(risk.get("max_volume_percentage"), (int, float)):
            logger.error("risk.max_volume_percentage must be a number")
            return False
        
        return True
    
    def get_checkpoint_path(self, episode: int) -> str:
        """
        Generate checkpoint file path for a specific episode.
        
        Args:
            episode: Episode number
            
        Returns:
            Path to checkpoint file
        """
        models_dir = self.get_path("MODELS_DIR", create=True)
        bucket = self.get("trading.bucket") or self.get("BUCKET", "Scalping")
        bucket_dir = os.path.join(models_dir, bucket)
        os.makedirs(bucket_dir, exist_ok=True)
        return os.path.join(bucket_dir, f"checkpoint_{episode}.pth")
    
    def get_final_model_path(self) -> str:
        """
        Generate path for the final trained model.
        
        Returns:
            Path to final model file
        """
        models_dir = self.get_path("MODELS_DIR", create=True)
        bucket = self.get("trading.bucket") or self.get("BUCKET", "Scalping")
        bucket_dir = os.path.join(models_dir, bucket)
        os.makedirs(bucket_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        return os.path.join(bucket_dir, f"final_{bucket.lower()}_{timestamp}.pth")
    
    def get_preset_config(self, bucket: str) -> Dict[str, Any]:
        """
        Get preset configuration for a specific bucket.
        
        Args:
            bucket: Trading bucket type
            
        Returns:
            Dictionary with preset configuration
        """
        presets = {
            "Scalping": {
                "WINDOW_SIZE": 288,
                "INITIAL_CAPITAL": 100000.0,
                "MAX_POSITION_HOLDINGS": 50,
                "BUCKET": "Scalping",
                "MAX_BTC_PER_POSITION": 10.0,
                "MAX_USD_PER_POSITION": 1000000.0,
                "MAX_VOLUME_PERCENTAGE": 0.05,
                "monthly_target_min": 15.0,
                "monthly_target_max": 30.0,
                "yearly_target_min": 100.0,
                "yearly_target_max": 200.0,
                "min_gain_per_holding": 25.0,
                "max_gain_per_holding": 50.0,
                "bonus_multiplier": 1.1
            },
            "Short": {
                "WINDOW_SIZE": 576,
                "INITIAL_CAPITAL": 100000.0,
                "MAX_POSITION_HOLDINGS": 40,
                "BUCKET": "Short",
                "MAX_BTC_PER_POSITION": 8.0,
                "MAX_USD_PER_POSITION": 800000.0,
                "MAX_VOLUME_PERCENTAGE": 0.04,
                "monthly_target_min": 20.0,
                "monthly_target_max": 35.0,
                "yearly_target_min": 120.0,
                "yearly_target_max": 220.0,
                "min_gain_per_holding": 30.0,
                "max_gain_per_holding": 60.0,
                "bonus_multiplier": 1.15
            },
            "Medium": {
                "WINDOW_SIZE": 864,
                "INITIAL_CAPITAL": 100000.0,
                "MAX_POSITION_HOLDINGS": 30,
                "BUCKET": "Medium",
                "MAX_BTC_PER_POSITION": 6.0,
                "MAX_USD_PER_POSITION": 600000.0,
                "MAX_VOLUME_PERCENTAGE": 0.03,
                "monthly_target_min": 25.0,
                "monthly_target_max": 40.0,
                "yearly_target_min": 140.0,
                "yearly_target_max": 240.0,
                "min_gain_per_holding": 35.0,
                "max_gain_per_holding": 70.0,
                "bonus_multiplier": 1.2
            },
            "Long": {
                "WINDOW_SIZE": 1152,
                "INITIAL_CAPITAL": 100000.0,
                "MAX_POSITION_HOLDINGS": 20,
                "BUCKET": "Long",
                "MAX_BTC_PER_POSITION": 4.0,
                "MAX_USD_PER_POSITION": 400000.0,
                "MAX_VOLUME_PERCENTAGE": 0.02,
                "monthly_target_min": 30.0,
                "monthly_target_max": 45.0,
                "yearly_target_min": 160.0,
                "yearly_target_max": 260.0,
                "min_gain_per_holding": 40.0,
                "max_gain_per_holding": 80.0,
                "bonus_multiplier": 1.25
            }
        }
        return presets.get(bucket, presets["Scalping"])

# Create the singleton instance
_trade_config_instance = None

def get_trade_config(config_path: Optional[Union[str, Path]] = None) -> TradeConfig:
    """
    Get or create a TradeConfig instance.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        TradeConfig: Configuration instance
    """
    global _trade_config_instance
    if _trade_config_instance is None:
        _trade_config_instance = TradeConfig(config_path)
    return _trade_config_instance

# Create a default instance that can be imported directly
trade_config = get_trade_config() 