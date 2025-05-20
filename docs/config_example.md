# TradeConfig Usage Examples

This document provides examples of how to use the centralized `TradeConfig` in different contexts throughout the application.

## Basic Usage

### Importing the TradeConfig

```python
# Import the singleton instance (recommended)
from src.utils.trade_config import trade_config

# Or import the class if you need to create a new instance
from src.utils.trade_config import TradeConfig
```

### Getting Configuration Values

```python
# Using the singleton instance
from src.utils.trade_config import trade_config

# Get a value with default
batch_size = trade_config.get("BATCH_SIZE", 32)
learning_rate = trade_config.get("LEARNING_RATE", 0.001)

# Get a nested configuration value using dot notation
bucket_type = trade_config.get("trading.bucket", "Scalping")

# Dictionary-style access
model_params = trade_config["model"]
```

### Setting Configuration Values

```python
# Set a single value
trade_config["EPOCHS"] = 100

# Set a nested value using dot notation
trade_config.set("model.hidden_size", 128)

# Update multiple values at once
trade_config.update({
    "INITIAL_CAPITAL": 10000.0,
    "model": {
        "learning_rate": 0.001,
        "batch_size": 64
    }
})

# Save changes to the configuration file
trade_config.save()
```

## Module Initialization

Here's an example of initializing a module using the TradeConfig:

```python
from src.utils.trade_config import trade_config

class TrainingModule:
    def __init__(self):
        # Get configuration values with defaults
        self.learning_rate = trade_config.get("LEARNING_RATE", 0.001)
        self.epochs = trade_config.get("EPOCHS", 100)
        self.batch_size = trade_config.get("BATCH_SIZE", 32)
        
        # Get model parameters from a section
        model_params = trade_config.get("model", {})
        self.hidden_size = model_params.get("hidden_size", 256)
        
        # Initialize the model with the configured parameters
        self.model = self._create_model()
        
    def _create_model(self):
        # Use configuration values to create the model
        pass
```

## Testing with TradeConfig

For testing, you can manipulate the TradeConfig:

```python
from src.utils.trade_config import TradeConfig
import pytest
import tempfile
import json

def test_something_with_config():
    # Create a temporary configuration file for testing
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as temp:
        test_config = {
            "TEST_VALUE": 42,
            "LEARNING_RATE": 0.01
        }
        json.dump(test_config, temp)
        temp_path = temp.name
    
    # Create a TradeConfig instance with the test configuration
    test_config = TradeConfig(config_path=temp_path, force_reload=True)
    
    # Now use the test configuration
    assert test_config.get("TEST_VALUE") == 42
    assert test_config.get("LEARNING_RATE") == 0.01
    
    # Clean up
    import os
    os.unlink(temp_path)
```

## Environment-Specific Configuration

The TradeConfig system allows overriding configuration values for different environments:

```python
# In production code
from src.utils.trade_config import trade_config

# Configuration will automatically load the right values based on the environment
database_url = trade_config.get("database.url")
api_key = trade_config.get("api.key")
```

## Examples for Different Components

### Example: Data Loading

```python
from src.utils.trade_config import trade_config

def load_data():
    data_path = trade_config.get_path("DATA_DIR")
    file_pattern = trade_config.get("data.file_pattern", "*.csv")
    
    # Load data based on configuration
    import glob
    import pandas as pd
    
    files = glob.glob(f"{data_path}/{file_pattern}")
    return pd.concat([pd.read_csv(file) for file in files])
```

### Example: Model Training

```python
from src.utils.trade_config import trade_config

def train_model(data):
    # Get training parameters from configuration
    epochs = trade_config.get("EPOCHS", 100)
    learning_rate = trade_config.get("LEARNING_RATE", 0.001)
    batch_size = trade_config.get("BATCH_SIZE", 32)
    
    # Train the model with the configured parameters
    # ...
    
    # Save the model to the configured path
    model_path = trade_config.get_path("MODELS_DIR")
    # ...
```

## Migration Guide

Here's an example of how to migrate from direct configuration loading to using the TradeConfig:

### Before:

```python
import json

def load_config():
    with open('config.json', 'r') as f:
        return json.load(f)

def init_module():
    config = load_config()
    learning_rate = config.get('learning_rate', 0.001)
    # ...
```

### After:

```python
from src.utils.trade_config import trade_config

def init_module():
    learning_rate = trade_config.get('LEARNING_RATE', 0.001)
    # ...
``` 