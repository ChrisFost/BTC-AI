# Trading Agent with RL and Evolutionary Strategies

A sophisticated trading agent using reinforcement learning with proximal policy optimization (PPO) and evolutionary strategies (ES) for cryptocurrency trading.

## Features

- **Reinforcement Learning**: Uses PPO for learning optimal trading policies
- **Evolutionary Strategies**: Evolves a population of agents to find the best strategies
- **Multi-horizon Prediction**: Makes price predictions at multiple time horizons
- **Tensor-based Environment**: High-performance tensor calculations for rapid training
- **Adaptive Training**: Adjusts to hardware capabilities and memory constraints
- **Backtesting**: Comprehensive backtesting and performance metrics
- **Modular Design**: Clean, modular architecture for easy extension and customization

## Project Structure

```
trading_agent/
├── src/training/config.py         # Configuration management
├── utils.py          # Utility functions
├── tensor_utils.py   # Tensor-based utility functions
├── models.py         # Neural network model definitions
├── agents.py         # Reinforcement learning agent implementations
├── environment.py    # Trading environments
├── backtesting.py    # Backtesting and comparison functions
├── training.py       # Training loop with ES
└── main.py           # Main entry point
```

## Requirements

- Python 3.8+
- PyTorch 1.10+
- pandas
- numpy
- matplotlib
- seaborn

Optional dependencies:
- stable-baselines3 (for improved vectorized environments)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
python main.py --mode train --data your_data.csv --bucket Scalping
```

### Backtesting

```bash
python main.py --mode backtest --data your_data.csv --model path/to/model.pth --report backtest_report.txt
```

### Configuration Comparison

```bash
python main.py --mode compare --data your_data.csv --report comparison_report.txt
```

## Data Format

The agent expects a CSV file with OHLCV data and technical indicators. Required columns:

- `close`: Closing price
- `high`: High price
- `low`: Low price
- `volume`: Trading volume

Additional recommended columns include technical indicators such as:
- SMA9, SMA21, SMA50, SMA200 (Simple Moving Averages)
- RSI14 (Relative Strength Index)
- Bollinger Bands
- And more...

## Configuration

The agent can be configured through a JSON file:

```json
{
  "WINDOW_SIZE": 288,
  "HIDDEN_SIZE": 512,
  "LEARNING_RATE": 0.0003,
  "GAMMA": 0.99,
  "EPS_CLIP": 0.2,
  "INITIAL_CAPITAL": 100000.0,
  "MAX_POSITION_HOLDINGS": 50,
  "BUCKET": "Scalping",
  "ES_POPULATION": 5,
  "PREDICTION_BONUS": 0.03
}
```

## Trading Buckets

The agent supports four trading timeframes:

- **Scalping**: Very short-term trades (minutes to hours)
- **Short**: Short-term trades (hours to days)
- **Medium**: Medium-term trades (days to weeks)
- **Long**: Long-term trades (weeks to months)

Each bucket uses different prediction horizons and reward functions optimized for that timeframe.

## Performance Metrics

The agent tracks numerous performance metrics:

- Net Profit
- Win Rate
- Profit Factor
- Sharpe Ratio
- Maximum Drawdown
- And more...

## Command Line Options

```
usage: main.py [-h] --mode {train,backtest,compare} --data DATA [--config CONFIG]
               [--bucket {Scalping,Short,Medium,Long}] [--models-dir MODELS_DIR]
               [--resume] [--model MODEL] [--episodes EPISODES] [--report REPORT]
               [--cpu] [--mixed-precision]

Trading Agent with Evolutionary Strategies

options:
  -h, --help            show this help message and exit
  --mode {train,backtest,compare}
                        Operating mode: train, backtest, or compare
  --data DATA           Path to CSV data file with OHLCV and features
  --config CONFIG       Path to JSON configuration file
  --bucket {Scalping,Short,Medium,Long}
                        Trading timeframe bucket
  --models-dir MODELS_DIR
                        Directory to save/load models
  --resume              Resume training from latest checkpoint
  --model MODEL         Path to model file for backtesting
  --episodes EPISODES   Number of episodes for backtesting
  --report REPORT       Path to save backtest report
  --cpu                 Force CPU usage even if GPU is available
  --mixed-precision     Use mixed precision training if supported
```

## Development

### Adding New Models

New policy network architectures can be added in `models.py`. Implement your model as a subclass of `nn.Module` and add it to the `create_model` factory function.

### Adding New Environments

Custom environments can be implemented in `environment.py` by inheriting from `BaseTradingEnv`.

### Adding New Technical Indicators

Add new technical indicator tensors in `tensor_utils.py` and use them in the environment observation space.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Naturalistic Learning & Progressive Training

The system implements advanced naturalistic learning features that make the AI's learning process more human-like and adaptable:

### Naturalistic Learning Features

- **Adaptive Exploration Rates**: Context-sensitive exploration that adjusts based on market conditions
- **Experience Prioritization**: Focuses on surprising or important experiences through prioritized replay
- **Post-Trade Analysis**: Extracts lessons from completed trades through a LessonMemory system
- **Contextual Memory**: Recognizes and recalls similar market situations when making decisions
- **Adaptive Time Horizons**: Dynamically adjusts prediction timeframes based on market conditions
- **Simple Meta-Learning**: Optimizes its own hyperparameters during training

See `NATURAL_LEARNING.md` for detailed documentation of these features.

### Progressive Training

The system supports progressive training of buckets with knowledge transfer between them:

```bash
# Train all buckets in sequence with knowledge transfer
python progressive_training.py

# Train a specific bucket
python progressive_training.py --bucket Scalping

# Train with custom bucket sequence
python progressive_training.py --sequence "Scalping,Medium,Long"
```

Key features of progressive training:

1. **Sequential Bucket Training**: Trains buckets in sequence (Scalping → Short → Medium → Long)
2. **Cross-Bucket Knowledge Transfer**: Shares insights about feature importance, prediction horizons, and model weights
3. **Hardware-Aware**: Respects GPU memory constraints and optimizes memory usage
4. **Resource Management**: Intelligent memory cleanup and data caching between training sessions

See `PROGRESSIVE_TRAINING.md` for detailed documentation of the progressive training system.

### Monitoring and Visualization

The system includes a real-time monitoring dashboard and comprehensive visualization tools:

```bash
# Launch the monitoring dashboard
python monitor_training.py

# With custom models directory
python monitor_training.py --models-dir custom/models/path
```

Key monitoring and visualization features:

1. **Training Dashboard**: Real-time monitoring of training progress across all buckets
2. **Knowledge Transfer Visualization**: Visualizes transfer events and their impact
3. **Performance Metrics**: Tracks and visualizes key performance indicators
4. **Memory Usage Monitoring**: Tracks GPU/CPU memory usage during training
5. **Automated Reporting**: Generates comprehensive training reports with visualizations

The dashboard includes multiple tabs for different aspects of monitoring:
- Training progress plots
- Knowledge transfer events
- System resource utilization
- Training logs
- And more...

See `PROGRESSIVE_TRAINING_GUIDE.md` for detailed documentation of the monitoring and visualization system.

### End-to-End Testing

The system includes comprehensive end-to-end testing for the progressive training pipeline:

```bash
# Run the full end-to-end test
python test_progressive_pipeline.py

# Run test with custom parameters
python test_progressive_pipeline.py --episodes 30 --sequence Scalping Short
```

The end-to-end test:
1. Simulates progressive training across multiple buckets
2. Tests knowledge transfer between buckets
3. Verifies monitoring and visualization functionality
4. Generates test reports for validation

This ensures the entire pipeline functions correctly and helps diagnose any issues.

## Quick Start

For new users, see `PROGRESSIVE_QUICKSTART.md` for a step-by-step guide to getting started with the progressive training and monitoring system.
