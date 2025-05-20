# Path Management in BTC-AI

This document explains the path management system in the BTC-AI codebase, which ensures consistent path handling across different environments and platform-specific file system requirements.

## Overview

The BTC-AI codebase uses a centralized path management system through the `src/utils/paths.py` module. This module leverages the existing platform-specific functionality in `platform_utils.py` but provides a simpler, more consistent interface for common path operations.

## Key Components

1. **paths.py** - The main interface for path management
2. **platform_utils.py** - Handles platform-specific path concerns and provides core functionality

## Usage

### Basic Path Operations

```python
# Import path utilities
from src.utils.paths import (
    get_project_root, 
    get_absolute_path, 
    ensure_path_exists,
    get_common_paths
)

# Get the project root directory
project_root = get_project_root()

# Convert a relative path to absolute
config_path = get_absolute_path("configs/config.json")

# Ensure a directory exists (creates it if it doesn't)
log_dir = ensure_path_exists("Logs/training")

# Get a dictionary of common project paths
paths = get_common_paths()
models_dir = paths["models"]  # Gets the Models directory path
```

### Path Handling for Module Imports

```python
from src.utils.paths import add_project_to_path

# Add project root to sys.path for consistent imports
add_project_to_path()
```

## Best Practices

1. **Always use the paths module** instead of constructing paths manually:
   ```python
   # GOOD
   from src.utils.paths import get_absolute_path
   config_path = get_absolute_path("configs/config.json")
   
   # AVOID
   import os
   config_path = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "config.json")
   ```

2. **Use `get_common_paths()` for standard directories**:
   ```python
   # GOOD
   from src.utils.paths import get_common_paths
   paths = get_common_paths()
   models_dir = paths["models"]
   
   # AVOID
   import os
   models_dir = os.path.join(os.path.dirname(__file__), "..", "..", "Models")
   ```

3. **Add proper exception handling** when using path utilities:
   ```python
   try:
       from src.utils.paths import get_absolute_path
       config_path = get_absolute_path("configs/config.json")
   except Exception as e:
       logger.error(f"Error getting config path: {e}")
       # Provide reasonable fallback
   ```

## Implementing in New Modules

When creating new modules that need path handling:

1. Import the necessary functions from `src.utils.paths`
2. Use the provided functions for all path operations
3. Include fallback behavior if import fails

### Example Implementation

```python
import os
import logging

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to use the paths module
try:
    from src.utils.paths import get_project_root, get_absolute_path, add_project_to_path
    
    # Ensure the project is in the path for imports
    add_project_to_path()
    
    # Define paths using centralized system
    CONFIG_PATH = get_absolute_path("configs/config.json")
    
except ImportError:
    logger.warning("paths module not available, using fallback path handling")
    
    # Fallback path determination
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    CONFIG_PATH = os.path.join(project_root, "configs", "config.json")
```

## Path-Related Systems

The BTC-AI codebase has several path-related subsystems:

1. **Platform Detection** (in platform_utils.py) - Detects whether running as a script or executable
2. **Cross-Platform Compatibility** - Normalizes paths for different operating systems
3. **Dynamic Path Resolution** - Allows locating resources regardless of execution context

## Technical Details

### Project Root Determination

The project root is determined based on the execution context:

- When running as a script, project root is relative to the module location
- When running as an executable (PyInstaller), project root is based on the executable's location

### Data Root Separation

Some configurations separate the project root (code) from the data root (user data):

- In development, these are typically the same
- In production, data might be stored in user-specific locations 