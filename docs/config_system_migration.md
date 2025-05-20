# ConfigManager to TradeConfig Migration Guide

## Overview

This document outlines the step-by-step process to migrate from the deprecated `ConfigManager` system to the newer `TradeConfig` system. The `TradeConfig` system appears to be the evolution of the original `ConfigManager`, handling the same responsibilities including:

- Managing user-defined settings
- Distributing settings to the training system
- Preserving state during system crashes
- Supporting the 50-episode comparison in ES training

## Migration Steps

### 1. Analysis Phase

#### 1.1 Identify ConfigManager References

```bash
# Find all Python files referencing config_manager
grep -r "config_manager" --include="*.py" .

# Find all Python files referencing ConfigManager class
grep -r "ConfigManager" --include="*.py" .

# Find all import statements for config_manager
grep -r "from src.utils.config_manager import" --include="*.py" .
```

#### 1.2 Identify TradeConfig References

```bash
# Find all Python files referencing trade_config
grep -r "trade_config" --include="*.py" .

# Find all Python files referencing TradeConfig class
grep -r "TradeConfig" --include="*.py" .

# Find all import statements for trade_config
grep -r "from src.utils.trade_config import" --include="*.py" .
```

#### 1.3 Check Legacy Config References

```bash
# Find old config imports that might need updating
grep -r "from src.utils.config import" --include="*.py" .
```

#### 1.4 Document API Discrepancies

Document differences between the ConfigManager and TradeConfig APIs:

- Method naming
- Parameter formats
- Return value formats
- Singleton access patterns

### 2. Cleanup Phase

#### 2.1 Delete New ConfigManager Implementation

```bash
# Remove the new config_manager.py file if it conflicts
rm src/utils/config_manager.py
```

#### 2.2 Clean Python Cache Files

```bash
# Remove ConfigManager cache files
find . -name "__pycache__" -exec find {} -name "*config_manager*" -delete \;

# For thorough cleanup, you can also clear all __pycache__ files
find . -name "__pycache__" -exec rm -rf {} +
find . -name "*.pyc" -delete
```

#### 2.3 Review Backup Files

```bash
# Review the config_manager.txt file for any valuable code
less throw_away/maybe_config_manager.txt
```

### 3. Test Refactoring Phase

#### 3.1 Update Test File Names

```bash
# If needed, rename test files to match the modules they test
if [ -f "tests/unit/test_config_manager.py" ] && [ ! -f "tests/unit/test_trade_config.py" ]; then
    mv tests/unit/test_config_manager.py tests/unit/test_trade_config.py
fi
```

#### 3.2 Update Config Manager Tests

For each test file that references ConfigManager:

1. Open the file:
   ```bash
   nano tests/unit/test_config_manager.py
   ```

2. Replace imports:
   ```python
   # Replace
   from src.utils.config_manager import ConfigManager, config, get_config
   # With
   from src.utils.trade_config import TradeConfig, trade_config, get_trade_config
   ```

3. Update class references:
   ```python
   # Replace
   config = ConfigManager()
   # With
   config = TradeConfig()
   ```

4. Update method calls according to the TradeConfig API:
   - Update method names
   - Adjust parameters as needed
   - Update expected return values in assertions

#### 3.3 Remove Duplicate Test Files

```bash
# Identify duplicate test files
find tests -name "test_*config*.py"

# After reviewing, delete redundant files
rm tests/unit/duplicate_test_file.py
```

#### 3.4 Check UI/Setup Wizard Tests

Review and update tests related to the UI or setup wizard that might reference ConfigManager:

```bash
# Find all test files that might need updates
find tests -name "test_setup_wizard.py" -o -name "test_*ui*.py"
```

### 4. Code Refactoring Phase

#### 4.1 Update backtesting.py

The backtesting module appears to have legacy references:

```python
# In src/training/backtesting.py

# Replace
from src.utils.config import get_config
# With
from src.utils.trade_config import trade_config

# Replace function calls
config = get_config()
# With
config = trade_config
```

#### 4.2 Update All Other ConfigManager Imports

For each file identified in step 1.1:

1. Open the file
2. Replace ConfigManager imports:
   ```python
   # Replace
   from src.utils.config_manager import ConfigManager, config, get_config
   # With
   from src.utils.trade_config import TradeConfig, trade_config, get_trade_config
   ```

3. Update instantiation code:
   ```python
   # Replace
   config = ConfigManager()
   # With either
   config = TradeConfig()
   # Or
   config = trade_config
   ```

4. Update method calls according to the TradeConfig API

#### 4.3 Update Legacy Config Imports

For each file using the old config system:

```python
# Replace
from src.utils.config import get_config
# With
from src.utils.trade_config import trade_config as config
# Or
from src.utils.trade_config import get_trade_config as get_config
```

### 5. Documentation Updates

#### 5.1 Add Migration Notice to TradeConfig

```python
# In src/utils/trade_config.py
"""
...existing docstring...

Note: This class replaces the deprecated ConfigManager class.
Migration guide available in docs/config_system_migration.md
"""
```

#### 5.2 Update Module Documentation

Update any other documentation files that reference ConfigManager:

```bash
# Find documentation references
grep -r "ConfigManager" --include="*.md" docs/
```

### 6. Verification Phase

#### 6.1 Run Unit Tests

```bash
# Run all unit tests to verify the changes
python tests/run_tests.py --unit
```

#### 6.2 Run Integration Tests

```bash
# Run integration tests
python tests/run_tests.py --integration
```

#### 6.3 Run UI Tests

```bash
# Run UI tests
python tests/run_tests.py --ui
```

#### 6.4 Run Manual Verification

Start the application and verify it loads configurations correctly:

```bash
python src/ui/main.py
```

### 7. Backup Creation

#### 7.1 Create Backup

After all changes have been verified:

```bash
.\btc_backup.ps1 -Version "5.0.2" -Description "config_manager_cleanup"
```

## API Comparison Reference

### ConfigManager API (Deprecated)

```python
from src.utils.config_manager import ConfigManager, config, get_config

# Instantiation
config = ConfigManager(config_path=None, force_reload=False)

# Methods
value = config.get("key", default_value)
config.set("key", value)
config.save()
config["key"] = value  # Dictionary-style access
value = config["key"]
```

### TradeConfig API (Current)

```python
from src.utils.trade_config import TradeConfig, trade_config, get_trade_config

# Instantiation
config = TradeConfig(config_path=None, force_reload=False)
# Or use the singleton
config = trade_config

# Methods
value = config.get("key", default_value)
config.set("key", value)  # For nested keys with dot notation
config.update({"key": value})  # For multiple updates
config.save()
config["key"] = value  # Dictionary-style access
value = config["key"]
config_dict = config.as_dict()  # Get full config as dictionary
```

## Common Migration Patterns

### Basic Import Updates

```python
# Old code
from src.utils.config_manager import ConfigManager, config, get_config

# New code
from src.utils.trade_config import TradeConfig, trade_config, get_trade_config
```

### Constructor Updates

```python
# Old code
config = ConfigManager(config_path="path/to/config.json")

# New code
config = TradeConfig(config_path="path/to/config.json")
# Or preferably, use the singleton
from src.utils.trade_config import trade_config
```

### Method Call Updates

```python
# Old code
config.set_and_save("key", value)

# New code
config["key"] = value  # Or config.set("key", value)
config.save()
```

### Using Configuration in Other Modules

```python
# Old code
from src.utils.config_manager import config
my_value = config.get("my_setting", default_value)

# New code
from src.utils.trade_config import trade_config
my_value = trade_config.get("my_setting", default_value)
``` 