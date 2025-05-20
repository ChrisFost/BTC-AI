# Quick Start Guide: Progressive Training

This quickstart guide will help you get up and running with progressive training, knowledge transfer, and monitoring features. Follow these steps to start using the naturalistic learning capabilities of the BTC-AI system.

## Prerequisites

- BTC-AI system installed and configured
- Trading data available for training
- CUDA-compatible GPU with at least 6GB memory (recommended)

## Step 1: Configure Your System

1. Create or edit the configuration file `config.json`:

```json
{
    "MODELS_DIR": "../Models",
    "DATA_DIR": "../Data",
    "PROGRESSIVE_TRAINING": {
        "ENABLED": true,
        "SEQUENCE": ["Scalping", "Short", "Medium", "Long"],
        "TRANSFER_KNOWLEDGE": true
    }
}
```

2. Adjust the bucket configurations if needed:

```json
"BUCKET_CONFIGS": {
    "Scalping": {
        "MIN_HORIZON": 1,
        "MAX_HORIZON": 72
    },
    "Short": {
        "MIN_HORIZON": 6,
        "MAX_HORIZON": 144
    },
    "Medium": {
        "MIN_HORIZON": 24,
        "MAX_HORIZON": 288
    },
    "Long": {
        "MIN_HORIZON": 72,
        "MAX_HORIZON": 576
    }
}
```

## Step 2: Launch the Monitor Dashboard

1. Open a terminal window and navigate to the scripts directory
2. Launch the monitor dashboard:

```bash
python monitor_training.py
```

3. The dashboard window should appear, ready to track training progress

## Step 3: Start Progressive Training

1. Open a new terminal window 
2. Start the progressive training process:

```bash
python progressive_training.py --episodes 50
```

3. Watch as the system trains each bucket sequentially and transfers knowledge between them

## Step 4: Monitor Training Progress

1. In the monitoring dashboard, click "Start Watching" to see real-time updates
2. Navigate between tabs to view:
   - Training progress plots
   - Knowledge transfer events
   - System resource usage
   - Training logs

## Step 5: Generate and View Reports

1. After training completes (or during training), click "Generate Report" in the dashboard
2. View the generated report in your browser
3. The report will include:
   - Training performance metrics
   - Visualization of knowledge transfer
   - Comparisons between buckets

## Step 6: Use Your Trained Models

After progressive training completes, your models will be ready for use in:
- Backtesting
- Live trading
- Further refinement

The models will be saved in the Models directory, organized by bucket type.

## Common Commands

### Launch Dashboard Only
```bash
python monitor_training.py --models-dir Models
```

### Run Specific Bucket Sequence
```bash
python progressive_training.py --sequence Short Medium Long
```

### Run with More Episodes
```bash
python progressive_training.py --episodes 100
```

### Disable Knowledge Transfer
```bash
python progressive_training.py --no-transfer
```

### Resume Interrupted Training
```bash
python progressive_training.py --resume
```

## Next Steps

- Refer to the comprehensive [Progressive Training Guide](PROGRESSIVE_TRAINING_GUIDE.md) for detailed information
- Explore advanced configuration options
- Experiment with different bucket sequences and knowledge transfer settings
- Analyze your training reports to optimize performance

For troubleshooting and advanced usage, refer to the full documentation. 