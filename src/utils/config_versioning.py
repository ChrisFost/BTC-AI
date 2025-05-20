"""
Configuration versioning system for BTC-AI trading system.

This module provides functionality for managing configuration versions,
including version detection, migration, and validation.
"""

import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import shutil
from datetime import datetime

logger = logging.getLogger(__name__)

class ConfigVersioning:
    """Handles configuration versioning and migration."""
    
    # Version mapping for different config formats
    VERSION_MAPPING = {
        "1.0": "legacy",
        "1.1": "legacy",
        "1.2": "legacy",
        "2.0": "current"
    }
    
    # Default configuration structure
    DEFAULT_CONFIG = {
        "version": "2.0",
        "environment": "development",
        "trading": {
            "bucket": "Scalping",
            "initial_capital": 100000.0,
            "max_positions": 50,
            "window_size": 288,
            "look_back_amount": 1,
            "look_back_unit": "day(s)",
            "resume_checkpoint": False,
            "checkpoint_interval": 10
        },
        "model": {
            "type": "ActorCritic",
            "hidden_size": 512,
            "num_layers": 2,
            "kernel_size": 3,
            "activation": "LeakyReLU",
            "dropout": 0.1,
            "use_fusion": True,
            "use_rnn": False,
            "rnn_type": "LSTM",
            "rnn_layers": 1,
            "rnn_hidden_size": 128,
            "attention_heads": 4,
            "learning_rate": 0.0003,
            "batch_size": 128,
            "buffer_size": 2048,
            "gamma": 0.99,
            "lambda": 0.95,
            "epsilon": 0.1,
            "eps_clip": 0.2,
            "entropy_coef": 0.01,
            "value_coef": 0.5,
            "max_grad_norm": 0.5,
            "weight_decay": 1e-5,
            "max_steps_per_episode": 500
        },
        "risk": {
            "max_btc_per_position": 10.0,
            "max_usd_per_position": 1000000.0,
            "max_volume_percentage": 0.05,
            "stop_loss": 0.05,
            "take_profit": 0.1,
            "trailing_stop": 0.02,
            "use_dynamic_sl": False,
            "max_drawdown": 0.15,
            "risk_management": True
        }
    }
    
    @staticmethod
    def detect_config_version(config: Dict[str, Any]) -> str:
        """
        Detect the version of a configuration dictionary.
        
        Args:
            config: Configuration dictionary to analyze
            
        Returns:
            str: Detected version string
        """
        version = config.get("version", "1.0")
        return str(version)
    
    @staticmethod
    def migrate_legacy_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Migrate a legacy configuration to the current format.
        
        Args:
            config: Legacy configuration dictionary
            
        Returns:
            Dict[str, Any]: Migrated configuration
        """
        # Create a copy of the default config as base
        migrated = ConfigVersioning.DEFAULT_CONFIG.copy()
        
        # Map legacy parameters to new structure
        legacy_mapping = {
            "BUCKET": ("trading", "bucket"),
            "INITIAL_CAPITAL": ("trading", "initial_capital"),
            "MAX_POSITIONS": ("trading", "max_positions"),
            "HIDDEN_SIZE": ("model", "hidden_size"),
            "LEARNING_RATE": ("model", "learning_rate"),
            "BATCH_SIZE": ("model", "batch_size"),
            "MAX_BTC_PER_POSITION": ("risk", "max_btc_per_position"),
            "MAX_USD_PER_POSITION": ("risk", "max_usd_per_position"),
            "MAX_VOLUME_PERCENTAGE": ("risk", "max_volume_percentage")
        }
        
        # Migrate parameters
        for legacy_key, (section, param) in legacy_mapping.items():
            if legacy_key in config:
                migrated[section][param] = config[legacy_key]
        
        # Add migration metadata
        migrated["_migration_metadata"] = {
            "original_version": config.get("version", "1.0"),
            "migration_date": datetime.now().isoformat(),
            "migrated_parameters": list(legacy_mapping.keys())
        }
        
        return migrated
    
    @staticmethod
    def load_config_with_versioning(config_path: Path) -> Dict[str, Any]:
        """
        Load a configuration file with version detection and migration.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Dict[str, Any]: Loaded and potentially migrated configuration
        """
        try:
            # Load the configuration file
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Detect version
            version = ConfigVersioning.detect_config_version(config)
            
            # Create backup if it's a legacy version
            if version in ["1.0", "1.1", "1.2"]:
                backup_path = config_path.with_suffix(f".{version}.backup")
                shutil.copy2(config_path, backup_path)
                logger.info(f"Created backup of legacy config at {backup_path}")
            
            # Migrate if needed
            if version in ["1.0", "1.1", "1.2"]:
                config = ConfigVersioning.migrate_legacy_config(config)
                logger.info(f"Migrated configuration from version {version} to 2.0")
            
            return config
            
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return ConfigVersioning.DEFAULT_CONFIG.copy()
    
    @staticmethod
    def save_config_with_versioning(config: Dict[str, Any], config_path: Path) -> bool:
        """
        Save a configuration with version tracking.
        
        Args:
            config: Configuration dictionary to save
            config_path: Path where to save the configuration
            
        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            # Ensure version is set
            if "version" not in config:
                config["version"] = "2.0"
            
            # Save the configuration
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            return False
    
    @staticmethod
    def validate_config_version(config: Dict[str, Any]) -> bool:
        """
        Validate that a configuration has a supported version.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            bool: True if version is supported, False otherwise
        """
        version = ConfigVersioning.detect_config_version(config)
        return version in ConfigVersioning.VERSION_MAPPING
    
    @staticmethod
    def get_supported_versions() -> List[str]:
        """
        Get a list of supported configuration versions.
        
        Returns:
            List[str]: List of supported version strings
        """
        return list(ConfigVersioning.VERSION_MAPPING.keys())
    
    @staticmethod
    def get_version_type(version: str) -> Optional[str]:
        """
        Get the type of a configuration version.
        
        Args:
            version: Version string to check
            
        Returns:
            Optional[str]: Version type ("legacy" or "current") or None if not supported
        """
        return ConfigVersioning.VERSION_MAPPING.get(version) 