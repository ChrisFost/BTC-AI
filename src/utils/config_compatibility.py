"""
Configuration Backward Compatibility Module

This module provides backward compatibility support for older configuration formats,
allowing smooth migration to the new TradeConfig system.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class ConfigCompatibility:
    """Handles backward compatibility for configuration files."""
    
    # Version mapping for different config formats
    VERSION_MAPPING = {
        "1.0": "legacy",  # Original format
        "1.1": "legacy",  # First update
        "1.2": "legacy",  # Second update
        "2.0": "current"  # New TradeConfig format
    }
    
    # Default configuration structure
    DEFAULT_CONFIG = {
        "version": "2.0",
        "environment": "development",
        "trading": {
            "bucket": "Scalping",
            "initial_capital": 100000.0,
            "max_positions": 50
        },
        "model": {
            "hidden_size": 512,
            "learning_rate": 0.0003,
            "batch_size": 128
        },
        "risk": {
            "max_btc_per_position": 10.0,
            "max_usd_per_position": 1000000.0,
            "max_volume_percentage": 0.05
        }
    }
    
    @staticmethod
    def detect_config_version(config: Dict[str, Any]) -> Tuple[str, str]:
        """
        Detect the version and format of a configuration file.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Tuple of (version, format_type)
        """
        version = config.get("version", "1.0")
        format_type = ConfigCompatibility.VERSION_MAPPING.get(version, "legacy")
        return version, format_type
    
    @staticmethod
    def migrate_legacy_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Migrate a legacy configuration to the new format.
        
        Args:
            config: Legacy configuration dictionary
            
        Returns:
            New format configuration dictionary
        """
        new_config = ConfigCompatibility.DEFAULT_CONFIG.copy()
        
        # Map legacy parameters to new structure
        if "BUCKET" in config:
            new_config["trading"]["bucket"] = config["BUCKET"]
            
        if "INITIAL_CAPITAL" in config:
            new_config["trading"]["initial_capital"] = float(config["INITIAL_CAPITAL"])
            
        if "MAX_POSITIONS" in config:
            new_config["trading"]["max_positions"] = int(config["MAX_POSITIONS"])
            
        if "HIDDEN_SIZE" in config:
            new_config["model"]["hidden_size"] = int(config["HIDDEN_SIZE"])
            
        if "LEARNING_RATE" in config:
            new_config["model"]["learning_rate"] = float(config["LEARNING_RATE"])
            
        if "BATCH_SIZE" in config:
            new_config["model"]["batch_size"] = int(config["BATCH_SIZE"])
            
        if "MAX_BTC_PER_POSITION" in config:
            new_config["risk"]["max_btc_per_position"] = float(config["MAX_BTC_PER_POSITION"])
            
        if "MAX_USD_PER_POSITION" in config:
            new_config["risk"]["max_usd_per_position"] = float(config["MAX_USD_PER_POSITION"])
            
        if "MAX_VOLUME_PERCENTAGE" in config:
            new_config["risk"]["max_volume_percentage"] = float(config["MAX_VOLUME_PERCENTAGE"])
        
        # Add migration metadata
        new_config["_migration_metadata"] = {
            "original_version": config.get("version", "1.0"),
            "migration_date": datetime.now().isoformat(),
            "migration_successful": True
        }
        
        return new_config
    
    @staticmethod
    def load_config_with_compatibility(file_path: str) -> Dict[str, Any]:
        """
        Load configuration with backward compatibility support.
        
        Args:
            file_path: Path to configuration file
            
        Returns:
            Configuration dictionary in new format
        """
        try:
            if not os.path.exists(file_path):
                logger.warning(f"Config file not found: {file_path}")
                return ConfigCompatibility.DEFAULT_CONFIG.copy()
            
            with open(file_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Detect version and format
            version, format_type = ConfigCompatibility.detect_config_version(config)
            
            if format_type == "legacy":
                logger.info(f"Detected legacy config version {version}, migrating to new format")
                new_config = ConfigCompatibility.migrate_legacy_config(config)
                
                # Save migrated config
                backup_path = f"{file_path}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak"
                os.rename(file_path, backup_path)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(new_config, f, indent=4)
                
                logger.info(f"Migrated config saved to {file_path}, original backed up to {backup_path}")
                return new_config
            else:
                logger.info(f"Using current config format version {version}")
                return config
                
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return ConfigCompatibility.DEFAULT_CONFIG.copy()
    
    @staticmethod
    def save_config_with_compatibility(config: Dict[str, Any], file_path: str) -> bool:
        """
        Save configuration with version tracking.
        
        Args:
            config: Configuration dictionary
            file_path: Path to save configuration
            
        Returns:
            bool: Success status
        """
        try:
            # Ensure version is set
            if "version" not in config:
                config["version"] = "2.0"
            
            # Create backup if file exists
            if os.path.exists(file_path):
                backup_path = f"{file_path}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak"
                os.rename(file_path, backup_path)
            
            # Save new config
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4)
            
            logger.info(f"Saved config to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            return False

    @staticmethod
    def adapt_state_version(state: Dict[str, Any], from_version: str, to_version: str) -> Dict[str, Any]:
        """
        Adapt a state object from one version to another.
        
        Args:
            state: The state dictionary to adapt
            from_version: The source version
            to_version: The target version
            
        Returns:
            The adapted state dictionary
        """
        logger.info(f"Adapting state from version {from_version} to {to_version}")
        
        # Create a copy of the state to avoid modifying the original
        adapted_state = state.copy()
        
        # Add version adaptation metadata
        if "_adaptation_metadata" not in adapted_state:
            adapted_state["_adaptation_metadata"] = []
            
        adapted_state["_adaptation_metadata"].append({
            "from_version": from_version,
            "to_version": to_version,
            "adaptation_date": datetime.now().isoformat()
        })
        
        # Update the version
        adapted_state["version"] = to_version
        
        # Handle specific version migrations
        if from_version == "0.0.1" and to_version.startswith("1."):
            # Migrate from initial version to 1.x
            if "config" in adapted_state:
                # Convert legacy keys to new format if needed
                if "BUCKET" in adapted_state["config"] and "bucket" not in adapted_state["config"]:
                    adapted_state["config"]["bucket"] = adapted_state["config"]["BUCKET"]
                
                # Ensure critical directories exist with defaults if missing
                critical_paths = ["MODELS_DIR", "DATA_DIR", "LOG_DIR"]
                for path_key in critical_paths:
                    if path_key not in adapted_state["config"]:
                        # Provide a default based on standard project structure
                        adapted_state["config"][path_key] = f"{path_key.split('_')[0].capitalize()}"
        
        elif from_version.startswith("1.") and to_version.startswith("1."):
            # Minor version changes within 1.x series
            # Apply any necessary changes between minor versions
            pass
            
        # Handle migration from any 1.x version to 2.0
        elif from_version.startswith("1.") and to_version.startswith("2."):
            # Major update to version 2.x
            if "config" in adapted_state:
                # Ensure new 2.x configuration keys exist
                if "USE_PROBABILISTIC" not in adapted_state["config"]:
                    adapted_state["config"]["USE_PROBABILISTIC"] = True
                
                # Add new resilience features from 2.x
                new_resilience_features = {
                    "AUTO_CHECKPOINT_FREQUENCY": 10,  # Minutes
                    "MAX_RETAINED_CHECKPOINTS": 5,
                    "ENABLE_STATE_VALIDATION": True,
                    "VALIDATION_STRICTNESS": "medium",  # Options: low, medium, high
                    "RECOVERY_STRATEGY": "restore-last-known-good" # Options: restore-default, restore-last-known-good
                }
                
                # Add these features if they don't exist
                for key, value in new_resilience_features.items():
                    if key not in adapted_state["config"]:
                        adapted_state["config"][key] = value
        
        logger.info(f"State adaptation from {from_version} to {to_version} completed")
        return adapted_state
    
    @staticmethod
    def check_config_compatibility(config: Dict[str, Any], current_version: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Check if a configuration is compatible with the current application version
        and fix it if possible.
        
        Args:
            config: The configuration to check
            current_version: The current application version
            
        Returns:
            Tuple of (is_compatible, fixed_config)
            - is_compatible: True if the config is compatible
            - fixed_config: The fixed configuration if incompatible, or None if can't be fixed
        """
        try:
            # Deep copy the config to avoid modifying the original
            fixed_config = json.loads(json.dumps(config))
            
            # List of required keys for the current version
            required_keys = ["BUCKET", "MODELS_DIR", "DATA_DIR", "LOG_DIR", "VERSION"]
            missing_keys = [key for key in required_keys if key not in fixed_config]
            
            # If any required keys are missing, add them with defaults
            if missing_keys:
                logger.warning(f"Missing required configuration keys: {missing_keys}")
                
                # Add defaults for missing keys
                for key in missing_keys:
                    if key == "BUCKET":
                        fixed_config["BUCKET"] = "Scalping"
                    elif key == "VERSION":
                        fixed_config["VERSION"] = current_version
                    elif key.endswith("_DIR"):
                        # Use standardized directory names
                        dir_name = key.replace("_DIR", "").capitalize()
                        fixed_config[key] = dir_name
            
            # Check for deprecated keys
            deprecated_keys = []
            for key in deprecated_keys:
                if key in fixed_config:
                    logger.warning(f"Deprecated configuration key found: {key}")
                    # Remove or remap deprecated keys
                    del fixed_config[key]
            
            # Check for invalid values
            if "BUCKET" in fixed_config and fixed_config["BUCKET"] not in ["Scalping", "Short", "Medium", "Long"]:
                logger.warning(f"Invalid BUCKET value: {fixed_config['BUCKET']}")
                fixed_config["BUCKET"] = "Scalping"
            
            # For numerical parameters, ensure they are within valid ranges
            numerical_constraints = {
                "LEARNING_RATE": (0.00001, 0.1),
                "BATCH_SIZE": (8, 512),
                "PPO_EPOCHS": (1, 20),
                "MEMORY_LIMIT_PERCENT": (0.1, 0.95)
            }
            
            for param, (min_val, max_val) in numerical_constraints.items():
                if param in fixed_config:
                    try:
                        value = float(fixed_config[param])
                        if value < min_val or value > max_val:
                            logger.warning(f"Parameter {param} out of range: {value}, constraining to [{min_val}, {max_val}]")
                            fixed_config[param] = min(max(value, min_val), max_val)
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid value for {param}: {fixed_config[param]}")
                        # Use a sensible default
                        if param == "LEARNING_RATE":
                            fixed_config[param] = 0.0003
                        elif param == "BATCH_SIZE":
                            fixed_config[param] = 128
                        elif param == "PPO_EPOCHS":
                            fixed_config[param] = 4
                        elif param == "MEMORY_LIMIT_PERCENT":
                            fixed_config[param] = 0.8
            
            # Check if fixed configuration differs from original
            is_compatible = (json.dumps(fixed_config, sort_keys=True) == json.dumps(config, sort_keys=True))
            
            return is_compatible, fixed_config
            
        except Exception as e:
            logger.error(f"Error checking configuration compatibility: {e}")
            return False, None 