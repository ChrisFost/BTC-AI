"""
Configuration Bridge Module

Connects preset management system with the centralized trade_config system.
This module provides functions to bridge between the preset system and the main configuration,
ensuring data flows consistently between them.
"""

import os
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path

# Try to set up proper logging
try:
    from src.utils.log_manager import LogManager
    logger = LogManager.get_logger('config_bridge')
except ImportError:
    # Fallback logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('config_bridge')

# Import trade_config
try:
    from src.utils.trade_config import get_trade_config, TradeConfig
except ImportError as e:
    logger.error(f"Failed to import trade_config: {e}")
    logger.error("Configuration bridge functionality will be limited")

# Import paths module
try:
    from src.utils.paths import get_absolute_path, get_common_paths
except ImportError as e:
    logger.warning(f"Failed to import paths module: {e}")
    logger.warning("Using fallback path handling")
    
    def get_absolute_path(path):
        """Fallback for getting absolute path"""
        if os.path.isabs(path):
            return path
        return os.path.abspath(path)
    
    def get_common_paths():
        """Fallback for common paths"""
        return {}


def load_preset_into_config(preset_path: str) -> bool:
    """
    Loads a preset file into the main configuration
    
    Args:
        preset_path: Path to the preset file
        
    Returns:
        bool: Success status
    """
    # Get singleton trade_config
    try:
        config = get_trade_config()
    except Exception as e:
        logger.error(f"Failed to get trade_config: {e}")
        return False
    
    # Try to load the preset
    try:
        from src.ui.preset_manager import load_preset
        preset_data = load_preset(preset_path)
        
        if preset_data and "params" in preset_data:
            # Update the main config with preset params
            config.update(preset_data["params"])
            logger.info(f"Loaded preset {preset_path} into config")
            return True
        else:
            logger.warning(f"Preset {preset_path} doesn't contain valid params")
            return False
    except Exception as e:
        logger.error(f"Error loading preset: {e}")
        return False

def save_config_as_preset(bucket: str, name: str, description: str = "", 
                         temporary: bool = False) -> str:
    """
    Saves current configuration as a preset
    
    Args:
        bucket: Bucket type (Scalping, Short, Medium, Long)
        name: Name for the preset
        description: Optional description
        temporary: Whether to save as a temporary preset
        
    Returns:
        Path to the saved preset or empty string on failure
    """
    # Get singleton trade_config
    try:
        config = get_trade_config()
    except Exception as e:
        logger.error(f"Failed to get trade_config: {e}")
        return ""
    
    # Extract bucket-specific parameters
    bucket_config = config.get_section(bucket.lower())
    
    # Add bucket to parameters
    params = bucket_config.copy()
    params["BUCKET"] = bucket
    
    try:
        from src.ui.preset_manager import save_preset
        preset_path = save_preset(bucket, name, params, description, temporary)
        if preset_path:
            logger.info(f"Saved config as preset: {preset_path}")
        return preset_path
    except Exception as e:
        logger.error(f"Error saving preset: {e}")
        return ""

def get_preset_default_config(bucket: str) -> Dict[str, Any]:
    """
    Get default configuration for a specific bucket from presets
    
    Args:
        bucket: Bucket type (Scalping, Short, Medium, Long)
        
    Returns:
        Dict with default configuration for the bucket
    """
    try:
        from src.ui.preset_manager import DEFAULT_PRESETS
        bucket_presets = DEFAULT_PRESETS.get(bucket, {})
        
        # Use the first preset for this bucket as default
        for preset_name, preset_data in bucket_presets.items():
            if "params" in preset_data:
                logger.info(f"Retrieved default config for {bucket} from preset {preset_name}")
                return preset_data["params"]
    except Exception as e:
        logger.error(f"Error getting default preset config: {e}")
    
    # Return empty dict if no presets found
    return {}

def sync_config_with_performance(preset_id: str, metrics: Dict[str, Any]) -> bool:
    """
    Update preset performance history with metrics from config
    
    Args:
        preset_id: ID of the preset
        metrics: Metrics from backtesting or training
        
    Returns:
        bool: Success status
    """
    try:
        from src.ui.preset_manager import update_preset_performance
        success = update_preset_performance(preset_id, metrics)
        if success:
            logger.info(f"Updated performance for preset {preset_id}")
        return success
    except Exception as e:
        logger.error(f"Error updating preset performance: {e}")
        return False

def get_preset_suggestions(bucket: str = None) -> Dict[str, Any]:
    """
    Get suggested presets based on performance
    
    Args:
        bucket: Optional bucket to filter by
        
    Returns:
        Dict of suggested presets
    """
    try:
        from src.ui.preset_manager import get_preset_suggestions
        return get_preset_suggestions(bucket or "Scalping", "overall")
    except Exception as e:
        logger.error(f"Error getting preset suggestions: {e}")
        return {} 