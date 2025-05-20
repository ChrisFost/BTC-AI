# Progressive Training, Knowledge Transfer, and Monitoring

This guide provides comprehensive documentation for the progressive training pipeline, cross-bucket knowledge transfer, and monitoring/visualization features of the BTC-AI trading system.

## Table of Contents

1. [Introduction](#introduction)
2. [Progressive Training](#progressive-training)
3. [Cross-Bucket Knowledge Transfer](#cross-bucket-knowledge-transfer)
4. [Monitoring Dashboard](#monitoring-dashboard)
5. [Visualizations](#visualizations)
6. [Command-Line Interface](#command-line-interface)
7. [Configuration](#configuration)
8. [Update System](#update-system)
9. [Error Logging](#error-logging)
10. [Best Practices](#best-practices)
11. [Troubleshooting](#troubleshooting)

## Introduction

The BTC-AI trading system introduces naturalistic learning capabilities through progressive training and knowledge transfer between different time-frame buckets. These features allow the system to develop a more comprehensive understanding of market patterns across different trading horizons.

Key features include:
- **Progressive Training**: Sequential training of models from short to long time frames
- **Knowledge Transfer**: Sharing insights between models of different time horizons
- **Memory-Aware Operations**: Efficient resource utilization during training
- **Real-time Monitoring**: Visualizing the training progress and performance
- **Reporting**: Generating comprehensive training reports
- **Automated Updates**: Built-in update mechanism for easy software maintenance
- **Robust Error Logging**: Comprehensive logging system with rotation features

## Progressive Training

Progressive training enables the system to learn in a naturalistic sequence, similar to how human traders develop expertise.

### Concept

The progressive training pipeline trains bucket models in a sequence, typically:
1. Scalping (shortest time frame)
2. Short-term
3. Medium-term
4. Long-term (longest time frame)

This approach allows knowledge to flow from shorter to longer time frames, creating a foundation of micro-patterns that inform macro-pattern recognition.

### Usage

To start progressive training:

1. **From the Menu**: Navigate to the Training tab and select "Progressive Training"
2. **From Command Line**: Use the progressive_training.py script

```bash
python progressive_training.py --sequence Scalping Short Medium Long --episodes 100
```

### Benefits

- More efficient learning across time frames
- Better adaptation to market conditions
- Reduced overfitting through knowledge sharing
- More coherent strategy development

## Cross-Bucket Knowledge Transfer

Knowledge transfer enables sharing of insights between different bucket models.

### Types of Knowledge Transfer

1. **Feature Importance**: Sharing which features are most predictive
2. **Weights Transfer**: Initializing models with weights from previous training
3. **Prediction Horizons**: Adapting prediction timeframes between buckets

### When Knowledge Transfer Happens

Knowledge transfer occurs:
- After a bucket completes training
- When explicitly requested via the API or UI
- During progressive training sequences

### Memory Management During Transfer

The system includes safeguards to prevent memory issues during knowledge transfer:

- Memory monitoring before transfers
- Fallback to less memory-intensive operations
- Automatic cleanup after transfers
- Error handling for out-of-memory situations

## Monitoring Dashboard

The monitoring dashboard provides real-time insights into the training process.

### Launching the Dashboard

```bash
python monitor_training.py --models-dir Models
```

Or launch it from the main menu under "Monitoring" → "Launch Training Monitor"

### Dashboard Features

The dashboard includes multiple tabs:

1. **Overview**: Summary of all bucket training
2. **Training Progress**: Detailed metrics by bucket
3. **Knowledge Transfer**: Visualization of transfer events
4. **System Metrics**: GPU/CPU usage during training 
5. **Logs**: Training and transfer logs

### Controls

- **Start/Stop Watching**: Toggle real-time updates
- **Refresh Now**: Manually update the dashboard
- **Generate Report**: Create a comprehensive training report

## Visualizations

The system provides several visualizations to help understand the training process.

### Training Progress

Plots key metrics across time, including:
- Reward
- Loss
- Win rate
- Profit factor

### Knowledge Transfer Visualization

Visualizes knowledge flow between buckets:
- Direction of transfer
- Types of knowledge transferred
- Success/failure of transfers
- Performance impact

### Feature Importance

Shows the most important features for each bucket model.

### Memory Usage

Tracks GPU memory usage during training with:
- Optimal operating zones
- Warning thresholds
- Transfer event markers

### Dashboard

Comprehensive view combining multiple visualizations.

## Command-Line Interface

The system provides command-line tools for advanced users:

### Progressive Training

```bash
python progressive_training.py --help

# Basic usage
python progressive_training.py --sequence Scalping Short Medium Long --episodes 100

# Resume training
python progressive_training.py --resume --bucket Medium

# Disable knowledge transfer
python progressive_training.py --no-transfer
```

### Monitoring

```bash
python monitor_training.py --help

# Basic usage
python monitor_training.py --models-dir Models

# Set update interval
python monitor_training.py --update-interval 10
```

### Test Pipeline

```bash
python test_progressive_pipeline.py --help

# Basic usage
python test_progressive_pipeline.py --episodes 20

# Custom sequence
python test_progressive_pipeline.py --sequence Short Medium
```

## Configuration

The system is configured through a JSON configuration file.

### Example Configuration

```json
{
    "MODELS_DIR": "../Models",
    "DATA_DIR": "../Data",
    "LOG_DIR": "../Logs",
    
    "LEARNING_RATE": 0.0005,
    "GAMMA": 0.99,
    "BATCH_SIZE": 32,
    "BUFFER_SIZE": 10000,
    "MAX_EPISODES": 100,
    
    "USE_GPU": true,
    "MEMORY_LIMIT_PERCENT": 0.85,
    "MEMORY_WARNING_PERCENT": 0.65,
    
    "PROGRESSIVE_TRAINING": {
        "ENABLED": true,
        "SEQUENCE": ["Scalping", "Short", "Medium", "Long"],
        "TRANSFER_KNOWLEDGE": true,
        "EPISODES_PER_BUCKET": 20,
        "PAUSE_BETWEEN_BUCKETS": 3
    },
    
    "KNOWLEDGE_TRANSFER": {
        "ENABLED": true,
        "TRANSFER_WEIGHTS": true,
        "TRANSFER_FEATURES": true,
        "TRANSFER_HORIZONS": true,
        "MAX_MEMORY_USAGE": 0.8
    },
    
    "BUCKET_CONFIGS": {
        "Scalping": {
            "MIN_HORIZON": 1,
            "MAX_HORIZON": 72,
            "FEATURES": ["price", "volume", "macd", "rsi", "ema"]
        },
        "Short": {
            "MIN_HORIZON": 6,
            "MAX_HORIZON": 144,
            "FEATURES": ["price", "volume", "macd", "rsi", "ema", "bollinger"]
        },
        "Medium": {
            "MIN_HORIZON": 24,
            "MAX_HORIZON": 288,
            "FEATURES": ["price", "volume", "macd", "rsi", "ema", "bollinger", "ichimoku"]
        },
        "Long": {
            "MIN_HORIZON": 72,
            "MAX_HORIZON": 576,
            "FEATURES": ["price", "volume", "macd", "rsi", "ema", "bollinger", "ichimoku", "fibonacci"]
        }
    },
    
    "MONITORING": {
        "ENABLED": true,
        "UPDATE_INTERVAL": 5,
        "METRICS": ["reward", "loss", "win_rate", "profit_factor"],
        "GENERATE_REPORTS": true
    },
    
    "UPDATE_SYSTEM": {
        "ENABLED": true,
        "CHECK_ON_STARTUP": true,
        "AUTO_UPDATE": false,
        "UPDATE_SERVER": "https://updates.btc-ai.example.com/api/updates",
        "CHECK_INTERVAL_DAYS": 7,
        "BACKUP_BEFORE_UPDATE": true
    },
    
    "LOGGING": {
        "LEVEL": "INFO",
        "FILE_LOGGING": true,
        "CONSOLE_LOGGING": true,
        "MAX_FILE_SIZE_MB": 10,
        "BACKUP_COUNT": 5,
        "ROTATION_WHEN": "midnight",
        "LOG_FORMAT": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    }
}
```

### Configuration Options

#### General Settings
- `MODELS_DIR`: Directory for saving models
- `DATA_DIR`: Directory for training data
- `LOG_DIR`: Directory for log files

#### Training Parameters
- `LEARNING_RATE`: Learning rate for the optimizer
- `GAMMA`: Discount factor for future rewards
- `BATCH_SIZE`: Batch size for training
- `MAX_EPISODES`: Maximum episodes per bucket

#### Memory Management
- `USE_GPU`: Whether to use GPU acceleration
- `MEMORY_LIMIT_PERCENT`: Maximum GPU memory usage (0-1)
- `MEMORY_WARNING_PERCENT`: Warning threshold for memory usage

#### Progressive Training
- `ENABLED`: Enable/disable progressive training
- `SEQUENCE`: Order of bucket training
- `TRANSFER_KNOWLEDGE`: Enable/disable knowledge transfer
- `EPISODES_PER_BUCKET`: Episodes to train each bucket

#### Knowledge Transfer
- `ENABLED`: Enable/disable knowledge transfer
- `TRANSFER_WEIGHTS`: Enable weight transfer
- `TRANSFER_FEATURES`: Enable feature importance transfer
- `TRANSFER_HORIZONS`: Enable prediction horizon transfer

#### Bucket Configurations
Each bucket can be configured with:
- `MIN_HORIZON`: Minimum prediction horizon
- `MAX_HORIZON`: Maximum prediction horizon
- `FEATURES`: List of features to use

#### Monitoring
- `ENABLED`: Enable/disable monitoring
- `UPDATE_INTERVAL`: Seconds between dashboard updates
- `METRICS`: List of metrics to track
- `GENERATE_REPORTS`: Enable/disable automatic report generation

#### Update System
- `ENABLED`: Enable/disable the update system
- `CHECK_ON_STARTUP`: Check for updates when the application starts
- `AUTO_UPDATE`: Automatically download and install updates
- `UPDATE_SERVER`: URL of the update server
- `CHECK_INTERVAL_DAYS`: Days between update checks
- `BACKUP_BEFORE_UPDATE`: Create backup before applying updates

#### Logging
- `LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `FILE_LOGGING`: Enable/disable file logging
- `CONSOLE_LOGGING`: Enable/disable console logging
- `MAX_FILE_SIZE_MB`: Maximum log file size before rotation
- `BACKUP_COUNT`: Number of rotated log files to keep
- `ROTATION_WHEN`: When to rotate logs (midnight, h, d, w0-w6)
- `LOG_FORMAT`: Format string for log messages

## Update System

The BTC-AI application includes a comprehensive update system for keeping your software current with the latest features and bug fixes.

### How Updates Work

1. **Version Checking**: The application periodically connects to a configured update server to check for new versions
2. **Update Notification**: When a new version is available, you'll receive a notification
3. **Update Download**: Updates can be downloaded manually or automatically
4. **Update Installation**: The system applies updates with automatic backup for safety
5. **Rollback**: If an update fails, the system can restore from backup

### Update Settings

You can configure the update system in the Settings tab:

1. **Enable/Disable Update Checks**: Turn automatic update checking on or off
2. **Automatic Updates**: Choose whether to download and install updates automatically
3. **Check Frequency**: Set how often the system checks for updates
4. **Update Server**: Configure the update server URL (advanced users)

### Manual Update

To manually check for updates:

1. Open the BTC-AI application
2. Go to the Help menu
3. Select "Check for Updates"
4. If an update is available, follow the prompts to download and install

### Update Process

When installing an update:

1. The application creates a backup of the current version
2. Downloads the update package with progress reporting
3. Verifies the package checksum for security
4. Applies the update and restarts the application if necessary
5. If the update fails, automatically restores from backup

### Best Practices

1. **Regular Updates**: Keep your system updated for the latest features and security patches
2. **Review Changelogs**: Check what's new before updating
3. **Backup Data**: Although the system creates automatic backups, it's good practice to back up important data
4. **Stable Network**: Ensure you have a stable internet connection during updates

## Error Logging

The BTC-AI system includes a robust logging mechanism for tracking system operations and troubleshooting issues.

### Log Levels

The system uses standard Python logging levels:

1. **DEBUG**: Detailed information for diagnosing problems
2. **INFO**: Confirmation that things are working as expected
3. **WARNING**: Indication that something unexpected happened
4. **ERROR**: Due to a more serious problem, the software couldn't perform some function
5. **CRITICAL**: A serious error indicating the program may be unable to continue running

### Log Files

Logs are stored in the configured log directory with the following structure:

- `btc_ai.log`: Current log file
- `btc_ai.log.YYYY-MM-DD`: Rotated log files by date
- `error.log`: Critical errors and exceptions
- `training.log`: Training-specific information
- `update.log`: Update system activities

### Viewing Logs

You can access logs in several ways:

1. **Application Interface**: In the Help menu, select "View Logs"
2. **File System**: Navigate to the log directory and open with any text editor
3. **Command Line**: Use the `show_logs.py` script

```bash
python show_logs.py --level ERROR --days 7
```

### Log Rotation

The system automatically manages log file size through rotation:

1. **Size-based**: Creates a new log file when the current one reaches the configured size
2. **Time-based**: Rotates logs at configured intervals (daily, weekly, etc.)
3. **Cleanup**: Automatically removes old log files based on the backup count

### Using Logs for Troubleshooting

When troubleshooting issues:

1. Check the most recent logs first
2. Look for ERROR or CRITICAL entries
3. Note the timestamp of when the issue occurred
4. Examine logs around that time for context
5. Review stack traces for detailed error information

## Best Practices

### Training Sequence

1. **Start Small**: Begin with smaller time frames and progress to larger ones
2. **Adequate Episodes**: Ensure each bucket has enough training episodes (at least 50-100)
3. **Consistent Data**: Use consistent data across buckets for better knowledge transfer

### Hardware Considerations

1. **GPU Memory**: Ensure at least 6GB of GPU memory for optimal performance
2. **Monitor Usage**: Keep an eye on the memory usage visualization
3. **Close Applications**: Close other GPU-intensive applications during training

### Knowledge Transfer

1. **Adjacent Buckets**: Knowledge transfer works best between adjacent time frames
2. **Selective Transfer**: Sometimes transferring only selected aspects (e.g., just features) is more effective
3. **Monitor Impact**: Use the dashboard to verify the impact of knowledge transfer

## Troubleshooting

### Out of Memory Errors

1. Reduce batch size in the configuration
2. Disable some types of knowledge transfer
3. Train fewer buckets at once
4. Ensure no other GPU-intensive applications are running

### Poor Transfer Results

1. Verify that source bucket training was successful
2. Try transferring only specific knowledge types
3. Increase training episodes for the source bucket
4. Check for excessive differences in bucket configurations

### Dashboard Issues

1. Verify that training is generating proper log files
2. Check the permissions of the models directory
3. Restart the dashboard with debug logging
4. Check if files are being saved in the expected locations

---

For additional support, refer to the project repository or contact the development team. 