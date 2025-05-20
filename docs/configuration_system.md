# BTC-AI Configuration System

This document explains how configuration is managed in the BTC-AI system, including the interaction between different components and best practices for working with configuration.

## Overview

The BTC-AI system uses a centralized configuration system through `TradeConfig` in `src/utils/trade_config.py`. This system handles all trading parameters with versioning and compatibility support. The configuration system works alongside a preset management system that provides user-friendly preset handling for different trading strategies.

## Key Components

### 1. TradeConfig (`src/utils/trade_config.py`)

The central configuration manager that provides:
- Singleton pattern for consistent configuration across the application
- Configuration loading, saving, and versioning
- Default values and validation
- Environment-specific overrides
- Section-based access (by bucket type)

```python
from src.utils.trade_config import get_trade_config

config = get_trade_config()
risk_tolerance = config.get("risk_tolerance", "medium")
initial_capital = config.get("INITIAL_CAPITAL", 100000)
```

### 2. Preset System (`src/ui/preset_manager.py`)

The UI-friendly preset management system that handles:
- Default presets for different bucket types (Scalping, Short, Medium, Long)
- User-created presets
- Temporary presets with auto-cleanup
- Performance tracking for presets

```python
from src.ui.preset_manager import load_preset, save_preset

# Load a preset
preset_data = load_preset("presets/user/Scalping/My_Strategy.json")

# Save a preset
preset_path = save_preset("Scalping", "My Strategy", 
                         {"risk_tolerance": "high", "monthly_target": 30.0},
                         "My custom high-risk strategy")
```

### 3. Preset Integration (`src/ui/preset_manager.py`)

The preset manager provides functions to load and save strategy presets.
Modules can interact with it directly when the main configuration is unavailable.

```python
from src.ui.preset_manager import load_preset, save_preset

# Load a preset
preset_data = load_preset("presets/defaults/Scalping/Aggressive.json")

# Save current settings as a new preset
path = save_preset("Scalping", "My Strategy", {"risk_tolerance": "high"})
```

### 4. Path Management (`src/utils/paths.py`)

Provides consistent path handling for configuration files:
- Project root determination
- Cross-platform path normalization
- Standard paths for preset directories

```python
from src.utils.paths import get_common_paths

paths = get_common_paths()
preset_dir = paths["presets"]
```

## Configuration Flow

### 1. From UI to Configuration

1. User inputs parameters in the UI
2. Values are collected in `src/ui/app_state.py`
3. Values are saved via `app_state.save_all_state()`
4. `trade_config.update()` and `trade_config.save()` are called
5. Configuration is written to disk

### 2. From Configuration to Training

1. Training modules load config via `get_trade_config()`
2. Parameters are extracted and passed to the environment
3. Environment creates reward systems specific to the bucket type
4. Bucket-specific goals and parameters influence training

### 3. From Presets to Configuration

1. User selects a preset in the UI
2. The preset is loaded using `preset_manager.load_preset()`
3. Main configuration is updated with preset values
4. UI is updated to reflect loaded values

## Configuration Fallbacks

The system includes robust fallback mechanisms:

1. **TradeConfig Fallback**: If configuration file can't be loaded, default values are used
2. **Preset-Based Fallback in backtesting.py**: If trade_config can't be imported, the system attempts to:
   - Load a default preset from the preset system via `preset_manager`
   - If preset loading fails, a minimal but functional configuration is used
   - All fallbacks maintain the same interface as TradeConfig
3. **Path Fallback**: If paths.py can't be used, direct path determination is available

This multi-level fallback approach ensures the system continues to function even when configuration components are unavailable, while providing appropriate warnings to the user.

## Best Practices

### Using Configuration in New Modules

```python
# Get the trade_config instance
from src.utils.trade_config import get_trade_config

def my_function():
    config = get_trade_config()
    
    # Access parameters with defaults
    risk_level = config.get("risk_level", "medium")
    initial_capital = config.get("INITIAL_CAPITAL", 100000)
    
    # Access a section
    risk_params = config.get_section("risk")
```

### Saving User Preferences

```python
from src.utils.trade_config import get_trade_config

def save_user_preferences(params):
    config = get_trade_config()
    
    # Update with new parameters
    config.update(params)
    
    # Save to disk
    config.save()
```

### Working with Presets

```python
from src.ui.preset_manager import load_preset, save_preset

# Load a preset
preset_path = "presets/defaults/Scalping/Aggressive.json"
preset_data = load_preset(preset_path)

# After training/backtesting, save the current configuration
preset_path = save_preset("Scalping", "My Optimized Strategy", preset_data["params"])
```

## Error Handling

The configuration system includes comprehensive error handling:

1. **Import Errors**: All imports use try/except blocks to provide fallbacks
2. **Missing Files**: If configuration files are missing, defaults are used
3. **Invalid Configuration**: Validation ensures configuration is usable
4. **Version Compatibility**: Migration between versions is handled automatically

## File Storage Locations

- **Main Configuration**: `configs/config.json`
- **Default Presets**: `presets/defaults/{bucket}/{name}.json`
- **User Presets**: `presets/user/{bucket}/{name}.json`
- **Temporary Presets**: `presets/temp/{bucket}/{name}.json`
- **Performance History**: `presets/performance_history.json` 