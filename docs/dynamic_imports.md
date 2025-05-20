# Dynamic Import Architecture

## Overview

This document describes the dynamic import system implemented across the codebase to ensure flexible module loading and path handling. This architecture allows the application to:

1. Run from different directory contexts without hardcoded paths
2. Gracefully handle missing components with fallbacks
3. Support testing in isolation
4. Enable future extensibility

## Implementation Pattern

The standard pattern for dynamic imports follows this structure:

```python
import importlib

# Try to import dynamically
try:
    # Import module using absolute import path
    module = importlib.import_module("src.package.module")
    Function = module.Function
    Class = module.Class
except ImportError as e:
    # Define fallback components if import fails
    def Function(*args, **kwargs):
        print("Fallback implementation")
        
    class Class:
        def __init__(self, *args, **kwargs):
            print("Fallback implementation")
```

## Path Handling

Paths are determined dynamically using:

```python
# Dynamically determine base directory
current_dir = os.path.dirname(os.path.abspath(__file__))  # Current module dir
src_dir = os.path.dirname(current_dir)  # src
base_dir = os.path.dirname(src_dir)  # project root
```

## Key Examples

### Configuration Loading (src/utils/dataframe.py)

```python
# Dynamically determine base directory
current_dir = os.path.dirname(os.path.abspath(__file__))  # src/utils
src_dir = os.path.dirname(current_dir)  # src
base_dir = os.path.dirname(src_dir)  # project root

# Load configuration to align with agent script
CONFIG_PATHS = [
    os.path.join(base_dir, "Scripts", "final_config.json"),
    os.path.join(base_dir, "config", "config.json"),
    os.path.join(base_dir, "config.json")
]

# Try multiple locations
config = {}
for config_path in CONFIG_PATHS:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            print(f"Loaded configuration from {config_path}")
            break
```

### Dynamic Module Loading (src/agent/agent.py)

```python
# Import modules dynamically
try:
    # Import tensor utils
    tensor_utils_module = importlib.import_module("src.utils.tensor_utils")
    detect_market_regime = tensor_utils_module.detect_market_regime
    compute_fractal_dimension_tensor = tensor_utils_module.compute_fractal_dimension_tensor
    
    # Import models
    models_module = importlib.import_module("src.models.models")
    create_model = models_module.create_model
    
    # Import utils
    utils_module = importlib.import_module("src.utils.utils")
    log = utils_module.log
    optimize_memory = utils_module.optimize_memory
except Exception as e:
    print(f"Error importing modules: {e}")
    # Define fallback components
```

### Conditional Imports for Type Checking (src/environment/env_observation.py)

```python
# Use TYPE_CHECKING for imports used only for type checking
if TYPE_CHECKING:
    from src.environment.env_risk import RiskManager, RiskLevel, RiskEvent
    # This import is only used for type hints and not at runtime
    from typing import TypeVar
    Agent = TypeVar('Agent')
else:
    # For runtime, load risk manager dynamically when needed
    try:
        env_risk_module = importlib.import_module("src.environment.env_risk")
        RiskManager = env_risk_module.RiskManager
        RiskLevel = env_risk_module.RiskLevel
        RiskEvent = env_risk_module.RiskEvent
    except ImportError as e:
        # Define placeholder classes if the risk module cannot be imported
        class RiskManager: pass
        class RiskLevel: pass
        class RiskEvent: pass
```

## Testing Support

The dynamic import architecture supports testing by allowing:

1. Mock imports during testing
2. Isolation of components
3. Swapping implementations for testing

## Best Practices

When adding new modules or modifying existing ones:

1. Use absolute import paths starting with `src.` 
2. Always provide fallbacks for critical functionality
3. Use `importlib.import_module()` rather than `__import__()`
4. Implement graceful degradation through try/except blocks
5. Avoid circular dependencies by importing only what's needed
6. Use TYPE_CHECKING for type hints that might cause circular dependencies

## Configuration Handling

Configuration is loaded from multiple potential paths:

1. First check for legacy paths
2. Then check standard config locations
3. Load first found config file
4. Fall back to default values when config is missing

This approach ensures configuration can be found regardless of the execution context. 