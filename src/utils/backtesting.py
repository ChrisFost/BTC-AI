# Configuration imports
try:
    from src.utils.trade_config import get_trade_config, TradeConfig
    trade_config = get_trade_config()
    logger.info("Successfully loaded trade_config for backtesting")
except ImportError as e:
    logger.warning(f"Could not import trade_config: {str(e)}")
    logger.warning("Attempting to load default preset configuration instead")
    
    def _load_default_preset(bucket: str = "Scalping") -> Dict[str, Any]:
        try:
            from src.ui import preset_manager
        except Exception as e:
            logger.error(f"Failed to import preset_manager: {e}")
            return {}

        bucket_presets = preset_manager.DEFAULT_PRESETS.get(bucket, {})
        for _name, data in bucket_presets.items():
            params = data.get("params")
            if params:
                return params
        return {}

    try:
        # Get default config for Scalping (most conservative)
        preset_config = _load_default_preset("Scalping")
        
        # Create a TradeConfig instance with preset data
        class PresetBasedConfig:
            def __init__(self, preset_data):
                self.config = preset_data or {"default": "configuration"}
                
            def get(self, key, default=None):
                return self.config.get(key, default)
                
            def as_dict(self):
                return self.config.copy()
                
            def __getitem__(self, key):
                return self.config.get(key)
                
            def get_section(self, section):
                return {k: v for k, v in self.config.items() if k.startswith(section.upper())}
        
        trade_config = PresetBasedConfig(preset_config)
        logger.warning("Using preset-based configuration as fallback")
    except ImportError as e:
        logger.error(f"Failed to load preset configuration: {str(e)}")
        logger.error("Backtesting will use minimal default values which may affect results")
        
        # Create very minimal config with essential defaults only
        class MinimalConfig:
            def __init__(self):
                self.config = {
                    "INITIAL_CAPITAL": 100000,
                    "RISK_LEVEL": "medium",
                    "MAX_POSITION_SIZE": 0.1,
                    "BUCKET": "Scalping"
                }
                
            def get(self, key, default=None):
                return self.config.get(key, default)
                
            def as_dict(self):
                return self.config.copy()
                
            def __getitem__(self, key):
                return self.config.get(key)
                
            def get_section(self, section):
                return {k: v for k, v in self.config.items() if k.startswith(section.upper())}
        
        trade_config = MinimalConfig()
        
# Legacy compatibility function with improved fallback behavior
def get_config():
    """Returns trade_config for backward compatibility."""
    return trade_config 