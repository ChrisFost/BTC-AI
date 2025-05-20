# BTC-AI User Guide

This guide provides step-by-step instructions for using the BTC-AI trading system, from installation through training and evaluation.

## Table of Contents

1. [Installation](#installation)
2. [System Overview](#system-overview)
3. [Getting Started](#getting-started)
4. [Configuration Options](#configuration-options)
5. [Training Models](#training-models)
6. [Trading Buckets](#trading-buckets)
7. [Monitoring & Visualization](#monitoring--visualization)
8. [Evaluating Performance](#evaluating-performance)
9. [Probabilistic Predictions](#probabilistic-predictions)
10. [Natural Learning Features](#natural-learning-features)
11. [Update System](#update-system)
12. [Error Logging](#error-logging)
13. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- Python 3.7 or higher
- CUDA-compatible GPU (recommended for faster training)
- 8GB+ RAM

### Option 1: Install from Source

#### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/BTC-AI.git
cd BTC-AI
```

#### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

#### Step 3: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
```

### Option 2: Standalone Executable

For users who prefer not to install Python and dependencies, we provide a standalone executable:

#### Step 1: Download the Executable

Download the latest release from our website or GitHub releases page:
- Windows: `BTC-AI-Setup.exe`
- macOS: `BTC-AI.dmg`
- Linux: `BTC-AI-Linux.AppImage`

#### Step 2: Install the Application

- **Windows**: Double-click the installer and follow the prompts
- **macOS**: Mount the DMG file and drag the application to your Applications folder
- **Linux**: Make the AppImage executable (`chmod +x BTC-AI-Linux.AppImage`) and run it

#### Step 3: Launch the Application

Open the installed application from your start menu, applications folder, or desktop shortcut.

## System Overview

The BTC-AI system uses reinforcement learning to develop trading strategies for Bitcoin markets. It features:

- **Multiple Trading Timeframes**: Four distinct "buckets" (Scalping, Short, Medium, Long)
- **Probabilistic Predictions**: Uncertainty-aware price movement forecasts
- **Naturalistic Learning**: Human-like learning patterns with experience prioritization
- **Cross-Bucket Knowledge Transfer**: Information sharing between timeframes
- **Automatic Updates**: Built-in update system for seamless software maintenance
- **Comprehensive Logging**: Detailed logs for monitoring and troubleshooting

The system is organized into several key components:

- **Agent**: The reinforcement learning algorithm making trading decisions
- **Environment**: Simulated trading environment with realistic market conditions
- **Models**: Neural network architectures for prediction and decision making
- **Interface**: User-friendly GUI for configuration and monitoring
- **Update System**: Keeps your software current with the latest features and fixes
- **Logging System**: Tracks system operations and helps diagnose issues

## Getting Started

### Launching the Interface

```bash
python src/ui/main.py
```

This will open the main interface with tabs for different settings and controls.

### Interface Tour

The interface consists of several tabs:

1. **Dashboard**: Quick overview and actions
2. **Strategy**: Core trading parameters
3. **Model**: Neural network architecture settings
4. **Reward**: Configure reward function behavior
5. **Probabilistic**: Set up uncertainty predictions
6. **Learning**: Configure naturalistic learning features
7. **Monitoring**: Training progress and visualization
8. **Withdrawal**: Cash management simulation

## Configuration Options

### Strategy Configuration

The **Strategy** tab lets you configure core trading parameters:

- **Trading Bucket**: Select from Scalping, Short, Medium, or Long
- **Initial Capital**: Starting capital for trading
- **Max Trades at Once**: Maximum concurrent positions
- **Look-Back Settings**: How far back in time the AI analyzes

#### Recommended Settings by Bucket

| Bucket | Look-Back | Max Trades | Initial Capital |
|--------|-----------|------------|-----------------|
| Scalping | 6 hours | 25 | 100,000 |
| Short | 1 day | 15 | 100,000 |
| Medium | 5 days | 8 | 100,000 |
| Long | 2 weeks | 4 | 100,000 |

### Model Architecture

The **Model** tab lets you configure the neural network:

- **Hidden Size**: Capacity of the neural network (larger = more complex)
- **Learning Rate**: How quickly the model adapts to new information
- **PPO Epochs**: How many times to reuse each experience
- **Batch Size**: How many experiences to process at once

#### Advanced Options

- **Fusion Architecture**: Enables attention mechanisms (recommended)
- **Entropy Coefficient**: Controls exploration vs. exploitation
- **Gamma (Discount)**: Value of future vs. immediate rewards

## Training Models

### Step 1: Select Bucket and Strategy

1. Go to the **Strategy** tab
2. Select the **Bucket** (e.g., "Scalping" for short-term trading)
3. Adjust **Look-Back Settings** according to your bucket
4. Set **Max Trades** according to your risk tolerance

### Step 2: Configure Learning Settings

1. Go to the **Model** tab
2. For beginners, use the default settings
3. For advanced users, adjust parameters based on your trading style

### Step 3: Start Training

1. Click **Start Training** on the control panel
2. Monitor progress in the **Monitoring** tab
3. Training can take several hours or days depending on settings

### Step 4: Save and Resume

- Training will automatically save checkpoints
- To resume training later, check **Resume from Last Checkpoint**
- Checkpoints are saved in the `Models/[Bucket]/checkpoints` directory

## Trading Buckets

The system supports four trading timeframes or "buckets":

### Scalping

- **Timeframe**: Minutes to hours
- **Look-Back**: 6-12 hours
- **Goal**: Monthly profit targets (15-30%)
- **Strategy**: Frequent trades with small profits
- **Best For**: High volatility, range-bound markets

### Short

- **Timeframe**: Hours to days
- **Look-Back**: 1 day
- **Goal**: Yearly profit targets (100-200%)
- **Strategy**: Selective entries with technical setups
- **Best For**: Short-term trends, daily patterns

### Medium

- **Timeframe**: Days to weeks
- **Look-Back**: 3-5 days
- **Goal**: Per-holding gain targets (25-50%)
- **Strategy**: Trend following with careful entries
- **Best For**: Medium-term trends, swing trading

### Long

- **Timeframe**: Weeks to months
- **Look-Back**: 1-2 weeks
- **Goal**: Per-holding gain targets (50-100%)
- **Strategy**: Major trend identification
- **Best For**: Major market cycles, position trading

## Monitoring & Visualization

### Live Training Monitor

The **Monitoring** tab provides real-time insights:

- **Training Progress**: Episodes completed and rewards
- **Performance Metrics**: Net profit, win rate, drawdown
- **CPU/RAM Usage**: System resource monitoring

### Dashboard Monitor

For advanced visualization, use the dashboard monitor:

1. Click **Launch Advanced Monitor** in the Monitoring tab
2. The monitor provides:
   - Real-time profit/loss graphs
   - Trade distribution visualizations
   - Model hyperparameter tracking
   - Comparison between training runs

### Logs & Notes

- **Live Log**: Shows real-time training events
- **Notes**: Area to document observations during training
- Use **Pop Out Log** for a dedicated log window
- Use **Save Log** to export the log for later analysis

## Evaluating Performance

### Key Performance Metrics

The system tracks several key metrics:

- **Net Profit**: Total profit/loss
- **Win Rate**: Percentage of profitable trades
- **Max Drawdown**: Largest decline from peak
- **Sharpe Ratio**: Risk-adjusted return
- **Profit Factor**: Gross profit divided by gross loss

### Running a Backtest

To evaluate model performance on historical data:

1. Ensure training has completed or reached a stable point
2. Click **Run Backtest** in the main interface
3. Review performance metrics in the results window
4. For detailed analysis, use the visualization tools

### Comparing Models

To compare different models or strategies:

1. Click **Run Comparison** in the main interface
2. Select the models/strategies to compare
3. Review side-by-side metrics and visualizations
4. Use the comparison to identify the best strategy

## Probabilistic Predictions

The system uses probabilistic predictions to quantify uncertainty:

### Configuring Predictions

In the **Probabilistic** tab:

- **Prediction Horizons**: Time points in the future to predict
- **Confidence Threshold**: Minimum confidence for trades
- **Position Sizing Strategy**: How position size scales with confidence

### Uncertainty Visualization

View uncertainty in predictions through:

- **Confidence Bands**: Shows range of likely price movements
- **Calibration Plots**: Shows reliability of uncertainty estimates
- **Uncertainty Metrics**: Quantitative measures of prediction quality

## Natural Learning Features

Configure naturalistic learning in the **Learning** tab:

### Adaptive Exploration

- **Initial Exploration Rate**: Starting exploration level
- **Minimum Exploration Rate**: Floor for exploration
- **Exploration Decay Method**: How exploration reduces over time

### Experience Prioritization

- **Surprise-Based Replay**: Prioritize surprising outcomes
- **Replay Buffer Size**: How many experiences to remember
- **Priority Alpha**: Weight for prioritization

### Contextual Memory

- **Memory Capacity**: How many market situations to remember
- **Similarity Threshold**: When to consider situations similar
- **Recall Count**: How many similar situations to recall

## Update System

The BTC-AI application includes a built-in update mechanism to keep your software current with the latest features, improvements, and bug fixes.

### Checking for Updates

The system can automatically check for updates:

1. At application startup (configurable)
2. On a periodic schedule (configurable)
3. Manually when you select "Check for Updates" from the Help menu

### Update Settings

Configure the update system in the Settings tab:

1. **Check for Updates Automatically**: Enable/disable automatic update checks
2. **Check Frequency**: How often to check for updates (daily, weekly, monthly)
3. **Automatic Download**: Whether to download updates automatically
4. **Automatic Installation**: Whether to install updates automatically
5. **Update Server**: The server URL for updates (advanced users only)

### Manual Update Process

To manually check for updates:

1. Click on **Help** in the menu bar
2. Select **Check for Updates**
3. If an update is available, you'll see details about the new version
4. Click **Download Update** to begin the download
5. After downloading, click **Install Update** to apply it

### Update Installation

When installing an update:

1. The system creates a backup of your current installation
2. Downloads and verifies the update package
3. Applies the update with a progress indicator
4. Restarts the application when complete
5. If the update fails, automatically restores from backup

### Update Notifications

When updates are available, you'll see:

1. A notification in the application's status bar
2. A badge on the Help menu
3. Details about the update including version and changes
4. Options to download, install, or dismiss the notification

## Error Logging

The BTC-AI system includes comprehensive error logging to help diagnose and resolve issues.

### Log Files

Log files are stored in the `logs` directory with the following organization:

- **btc_ai.log**: Main application log
- **training.log**: Training-specific information
- **error.log**: Critical errors and exceptions
- **update.log**: Update system activities

Logs are automatically rotated to prevent excessive disk usage:
- Size-based rotation (new log when size limit is reached)
- Time-based rotation (daily, weekly, or monthly)
- Limited number of backup logs retained

### Viewing Logs

You can access logs in several ways:

1. **From the Application**: Go to Help → View Logs
2. **File System**: Navigate to the logs directory and open with any text editor
3. **Log Viewer**: Use the integrated log viewer for filtering and searching

### Log Levels

Log messages are categorized by severity level:

- **DEBUG**: Detailed information for diagnosing problems
- **INFO**: Confirmation that things are working as expected
- **WARNING**: Something unexpected happened but the application continues
- **ERROR**: A serious problem that prevented a function from working
- **CRITICAL**: A serious error that may cause program failure

### Using Logs for Troubleshooting

When troubleshooting:

1. Look for ERROR or CRITICAL entries related to your issue
2. Note the timestamp and surrounding context
3. Check stack traces for detailed error information
4. Filter logs by component or severity for focused analysis
5. Include relevant log excerpts when seeking help

## Troubleshooting

### Common Issues and Solutions

#### Training Won't Start

- Check CUDA availability with `torch.cuda.is_available()`
- Ensure paths in `configs/config.json` are correct
- Check for file permission issues
- Check the application log for specific error messages

#### Out of Memory Errors

- Reduce **Batch Size** in the Model tab
- Reduce **Hidden Size** in the Model tab
- Close other GPU-intensive applications
- Monitor memory usage in the System tab

#### Training Crashes

- Check the error log in the `logs` directory
- Reduce **Max Envs Per Agent** in the Model tab
- Try disabling Fusion Architecture
- Check for GPU driver updates

#### Poor Performance

- Increase training time (more episodes)
- Try different bucket types for your timeframe
- Adjust reward function parameters
- Check for overfitting with cross-validation

#### Update Issues

- Ensure you have a stable internet connection
- Check firewall settings if updates can't be downloaded
- If update installation fails, try manually downloading from the website
- Verify adequate disk space for updates

#### Application Won't Launch

- Check the error.log file for startup issues
- Verify all required dependencies are installed
- Try reinstalling the application
- Check if antivirus software is blocking execution

### Getting Help

- Check error logs in the `logs` directory
- Review documentation in the `docs` directory
- Submit issues to the project repository
- Include log excerpts and system information when reporting issues 