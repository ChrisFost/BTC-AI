# Progressive Training

This document explains the progressive training functionality for the BTC-AI trading system, which allows for sequential training of different bucket types with knowledge transfer between them.

## Overview

Progressive training is an approach where trading buckets (Scalping, Short, Medium, Long) are trained in sequence rather than independently. This allows knowledge to flow from one bucket to another, creating a more integrated learning system.

## Features

### 1. Sequential Bucket Training

Buckets are trained in a predefined sequence (by default: Scalping → Short → Medium → Long) with knowledge transfer between them:

- Each bucket is trained to completion before moving to the next
- Training parameters are customized for each bucket type
- Hardware resources are managed efficiently between training sessions
- Memory is cleared between bucket training to prevent leaks

### 2. Cross-Bucket Knowledge Transfer

Knowledge is shared between buckets as they are trained:

- **Feature Importance**: Insights about which market features matter most
- **Prediction Horizons**: Effective time horizons for predictions
- **Model Weights**: Shared neural network components (feature extraction layers)

The transfer is hardware-aware, respecting GPU memory constraints and optimizing memory usage.

### 3. Bucket-Specific Configurations

Each bucket type has specialized configurations:

- **Scalping**: Shorter prediction horizons (1-72 bars), higher exploration
- **Short**: Medium prediction horizons (6-144 bars), balanced approach
- **Medium**: Longer prediction horizons (24-288 bars), lower exploration
- **Long**: Extended prediction horizons (72-576 bars), focus on major trends

### 4. Resource Management

The progressive trainer includes built-in resource management:

- Memory cleanup between bucket training sessions
- GPU memory monitoring with adaptive behavior
- Data caching with intelligent cleanup
- Error handling and recovery capabilities

## Usage

### Command Line Interface

The progressive trainer can be run from the command line:

```bash
# Train all buckets in sequence with default settings
python progressive_training.py

# Train a specific bucket
python progressive_training.py --bucket Scalping

# Train with a custom bucket sequence
python progressive_training.py --sequence "Scalping,Medium,Long"

# Resume training from a checkpoint
python progressive_training.py --resume

# Transfer knowledge from a specific bucket
python progressive_training.py --bucket Long --transfer Medium
```

### Python API

The trainer can also be used as a module:

```python
from progressive_training import ProgressiveTrainer

# Initialize trainer with progress callback
def progress_callback(msg):
    print(f"Progress: {msg}")

trainer = ProgressiveTrainer(progress_callback=progress_callback)

# Train a single bucket
model_path = trainer.train_bucket("Scalping", episodes=100)

# Train all buckets progressively
model_paths = trainer.train_progressively()

# Train with custom settings
model_paths = trainer.train_progressively(
    custom_sequence=["Scalping", "Medium"],
    episodes_per_bucket={"Scalping": 50, "Medium": 100}
)
```

You can also pass initialization details from the UI using the ``ui_params``
argument:

```python
ui_values = {
    "horizon_range": (12, 48),
    "frequency": "1h",
    "capital_allocation": 0.3,
}
model_paths = trainer.train_progressively(ui_params=ui_values)
```

## Integration with Naturalistic Learning

Progressive training works with the naturalistic learning features:

1. **Adaptive Exploration**: Each bucket adapts its exploration rate based on market conditions
2. **Experience Prioritization**: Important experiences are given higher priority in replay buffers
3. **Post-Trade Analysis**: Completed trades generate "lessons" that inform future decisions
4. **Contextual Memory**: Market situations are recognized and recalled when similar conditions occur
5. **Cross-Bucket Knowledge Transfer**: Insights flow between different time horizons

## Configuration

The progressive trainer reads configuration from the standard config.json file. Relevant configuration parameters include:

- `USE_CROSS_BUCKET_TRANSFER`: Enable/disable knowledge transfer between buckets
- `TRANSFER_COOLDOWN`: Minimum episodes between knowledge transfers
- `WEIGHT_TRANSFER_ALPHA`: How much weight to give source model parameters (0-1)
- `FEATURE_TRANSFER_ALPHA`: How much weight to give source feature importance (0-1)
- `ENABLE_REVERSE_TRANSFER`: Allow bidirectional knowledge flow between buckets
- `MEMORY_THRESHOLD`: Maximum GPU memory usage before limiting operations

## Implementation Details

### Training Pipeline

1. Each bucket is trained using the following process:
   - Load or create bucket-specific configuration
   - Load appropriate training data
   - Initialize model and optimizer
   - Transfer knowledge from previous bucket (if applicable)
   - Train for specified number of episodes
   - Save final model and performance metrics
   - Clear memory and prepare for next bucket

### Knowledge Transfer Process

The knowledge transfer occurs at two key points:

1. **Initial Transfer**: When a new bucket starts training, it receives knowledge from previously trained buckets
2. **Periodic Transfer**: During training, knowledge is periodically exchanged between buckets

The transfer process is hardware-aware and includes safety checks:
- Memory is monitored to prevent out-of-memory errors
- If full weight transfer isn't possible, fall back to lighter transfers (feature importance, horizons)
- Unexpected errors during transfer are handled gracefully

## Future Enhancements

Potential improvements to the progressive training system:

1. **Selective Bidirectional Transfer**: More sophisticated rules for when to allow reverse knowledge flow
2. **Parallel Training**: Training multiple buckets simultaneously on multi-GPU systems
3. **Transfer Curriculum**: Dynamically determine optimal transfer sequence based on performance
4. **Distributed Training**: Extend to distributed computing environments
5. **Automated Hyperparameter Search**: Optimize bucket-specific hyperparameters
6. **Adversarial Bucket Evaluation**: Buckets evaluate each other's strategies
7. **Market Regime Specialization**: Develop regime-specific models within each bucket 