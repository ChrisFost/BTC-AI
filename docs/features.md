# Latest Features (Newest Stuff Directory)

This directory contains the latest features added to the BTC-AI trading system, focusing on naturalistic learning, progressive training, and monitoring capabilities.

## Main Components

### Training & Learning

- **progressive_training.py** - Orchestrates training across multiple buckets with knowledge transfer
- **mock_training.py** - Test implementation for the progressive training pipeline
- **bucket_goals.py** - Defines training goals for different bucket types
- **model_v2.py** - Enhanced model definitions with knowledge transfer capabilities

### Monitoring & Visualization

- **monitor_training.py** - Real-time dashboard for monitoring training progress
- **progressive_visualizer.py** - Visualization tools for training metrics and knowledge transfer
- **test_visualizer.py** - Test script for the visualization components
- **performance_optimizer.py** - Performance profiling, optimization, and resource monitoring

### Testing

- **test_progressive_pipeline.py** - End-to-end test for the progressive training pipeline

### Configuration

- **config.json** - Configuration settings for progressive training and monitoring

### Documentation

- **PROGRESSIVE_TRAINING.md** - Detailed documentation of the progressive training system
- **PROGRESSIVE_TRAINING_GUIDE.md** - Comprehensive user guide for all new features
- **PROGRESSIVE_QUICKSTART.md** - Step-by-step quickstart guide
- **NATURAL_LEARNING.md** - Documentation of naturalistic learning features
- **NEWEST_FEATURES.md** - This file, listing all new features

## Feature Overview

### Progressive Training

The progressive training system trains buckets in sequence (typically Scalping → Short → Medium → Long) and transfers knowledge between them. This approach simulates how human traders develop expertise by first mastering short-term patterns before moving to longer timeframes.

Key features:
- Sequential bucket training
- Cross-bucket knowledge transfer
- Memory-aware operations
- Adaptive training parameters

### Knowledge Transfer

Knowledge transfer allows insights learned in one bucket to be applied to others:

- **Weight Transfer**: Neural network weights from one model to another
- **Feature Importance**: Information about which features are most predictive
- **Prediction Horizons**: Adaptation of prediction timeframes

### Monitoring Dashboard

The monitoring dashboard provides real-time visualization of the training process:

- Training metrics by bucket
- Knowledge transfer events
- System resource utilization
- Log viewing and filtering

### Visualization Tools

The visualization tools include:
- Training progress plots
- Knowledge transfer visualizations 
- Feature importance comparisons
- Memory usage monitoring
- Comprehensive dashboard views
- Report generation

### Performance Optimization

The performance optimization system helps identify and fix bottlenecks:

- **Function-level profiling**: Identifies slow functions and bottlenecks
- **Memory usage optimization**: Reduces memory consumption during training and visualization
- **Smart caching strategies**: Optimizes data loading and retention
- **System resource monitoring**: Tracks CPU, memory, and GPU usage in real-time
- **Visualization optimization**: Optimizes visualizations for lower memory usage
- **Optimization suggestions**: Provides automatic suggestions for performance improvements

## Usage

1. Start by reading the `PROGRESSIVE_QUICKSTART.md` for a step-by-step introduction
2. Refer to `PROGRESSIVE_TRAINING_GUIDE.md` for comprehensive documentation
3. Use `test_progressive_pipeline.py` to verify your setup is working correctly

## Integrating into the Main System

These components integrate with the existing BTC-AI trading system:
1. Progressive training can be launched from the main menu
2. Models trained with progressive training can be used for backtesting and live trading
3. Knowledge gained through cross-bucket transfer enhances performance across timeframes 