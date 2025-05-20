# BTC AI - Training Recovery System

## Overview

The Training Recovery System provides robust error handling and automatic recovery capabilities for the BTC AI training process. It's designed to make the training process more resilient to errors, ensure that progress is not lost due to crashes, and automatically attempt to recover from common issues.

## Features

- **Automatic Error Recovery**: Intelligently attempts to recover from various types of errors during training
- **Configuration Validation**: Validates training configurations before starting to prevent errors
- **Configuration Repair**: Automatically repairs invalid configurations 
- **Emergency Checkpoints**: Creates emergency checkpoints when errors occur
- **Detailed Error Reports**: Generates comprehensive error reports for debugging
- **Performance History Tracking**: Maintains a history of training performance metrics
- **Recovery State Management**: Manages recovery state for resuming interrupted training
- **Integration with Existing Systems**: Works with the BTC AI error handling system and preset manager

## How to Use

### Basic Usage

```python
from src.training.training_recovery import TrainingRecoverySystem
from src.training.training import train_model
import pandas as pd

# Load data
df = pd.read_csv("your_data.csv")

# Create configuration
config = {
    "LEARNING_RATE": 0.001,
    "HIDDEN_SIZE": 512,
    "ES_POPULATION": 8,
    "BUCKET": "Scalping",
    "MAX_STEPS_PER_EPISODE": 500,
    # other parameters...
}

# Initialize recovery system
recovery_system = TrainingRecoverySystem(
    checkpoint_dir="checkpoints",
    max_retries=3,
    min_checkpoint_interval=5,
    enable_emergency_checkpoints=True
)

# Start training with recovery
model, metrics, best_reward, elapsed_time = recovery_system.start_training_with_recovery(
    training_func=train_model,
    config=config,
    df=df,
    save_path="checkpoints",
    recovery_state=None  # Optional recovery state from previous run
)

# Check result
if model:
    print(f"Training completed successfully in {elapsed_time:.2f} seconds")
    print(f"Best reward: {best_reward}")
else:
    print(f"Training failed: {metrics.get('error')}")
```

### Command Line Usage

The recovery system can also be used from the command line through the training.py script:

```bash
python src/training/training.py \
    --data_path data/btc_5m.csv \
    --bucket Scalping \
    --hidden_size 512 \
    --learning_rate 0.0003 \
    --es_population 8 \
    --max_retries 3 \
    --enable_emergency_checkpoints
```

### Testing the Recovery System

You can test the recovery system using the included test script:

```bash
python src/training/test_recovery.py
```

This will run tests with different error scenarios to demonstrate how the recovery system handles various issues.

## Configuration

The recovery system accepts the following parameters:

- **checkpoint_dir**: Directory to store checkpoints (default: "checkpoints")
- **max_retries**: Maximum number of retry attempts for recoverable errors (default: 3)
- **min_checkpoint_interval**: Minimum episodes between checkpoints (default: 5)
- **enable_emergency_checkpoints**: Whether to create emergency checkpoints on errors (default: True)

## Error Classification

The system classifies errors into recoverable and unrecoverable categories:

### Recoverable Errors

- `ValueError`: Can often be fixed by adjusting parameters
- `KeyError`: Missing configuration keys can be added
- `IndexError`: Likely recoverable with better bounds checking
- `AttributeError`: May be recoverable with fallbacks
- `RuntimeError`: May be recoverable depending on cause
- `DataValidationError`: Can retry with different constraints
- `JSONDecodeError`: Can try backup configs
- `ZeroDivisionError`: Can add checks to prevent this
- `NaNError`: Can restart with adjusted parameters
- `EnvironmentError`: Can recreate environment

### Unrecoverable Errors

- `ImportError`: Critical module missing
- `ModuleNotFoundError`: Critical module missing
- `FileNotFoundError`: Critical file missing
- `PermissionError`: System permission issue
- `MemoryError`: System resource issue
- `OSError`: System level error
- `GPUError`: Hardware issue

## Backup System Integration

The recovery system works with the BTC AI backup system (via btc_backup.ps1) to ensure you can recover from serious errors:

```powershell
.\btc_backup.ps1 -Version "1.4" -Description "post_training_stable"
```

This creates a backup named `BTC_AI_YYYY-MM-DD_v1.4_post_training_stable.zip` in the backup directory.

## Detailed Workflow

1. **Initialization**: The system validates the configuration and creates necessary directories
2. **Training Start**: Training begins with the validated configuration
3. **Error Detection**: If an error occurs, it's logged and classified
4. **Recovery Attempt**: For recoverable errors, the system tries to fix the configuration and restart
5. **Checkpoint Creation**: Regular checkpoints are created during training
6. **Error Reporting**: Detailed error reports are generated for debugging
7. **Final Result**: After max retries or successful completion, results are returned

## Files

- `training_recovery.py`: Main recovery system implementation
- `test_recovery.py`: Test script for the recovery system
- `README_RECOVERY.md`: This documentation
- Integration with `training.py`: Main training script with recovery support

## Requirements

- Python 3.6+
- PyTorch
- Pandas
- NumPy
- Access to BTC AI's error handling framework (optional)
- Access to BTC AI's preset manager (optional)
