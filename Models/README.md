# Probabilistic Trading System

A comprehensive framework for training and deploying probabilistic deep learning models for trading. This system provides uncertainty-aware predictions and confidence-based decision making.

## Features

- **Probabilistic Model Architectures**: Two main architectures (LSTM and CNN-LSTM) with probabilistic outputs
- **Uncertainty Quantification**: Predict means, standard deviations, and confidence scores
- **Multi-horizon Predictions**: Generate forecasts across multiple time horizons
- **Visualization Tools**: Interactive plots for uncertainty bands, calibration curves, and more
- **Backtesting Framework**: Test trading strategies with confidence-based position sizing
- **Model Evaluation**: Comprehensive metrics for both point and probabilistic predictions
- **Risk-Aware Trading**: Adjust position sizes based on prediction confidence
- **GUI Interface**: PySimpleGUI-based menu for easy interaction with the system

## Components

The system consists of the following main components:

- **probabilistic_model.py**: Model architectures with probabilistic outputs
- **train_probabilistic.py**: Training and validation logic
- **prepare_data.py**: Data preparation and preprocessing
- **trading_model.py**: Integration with trading environment
- **evaluate.py**: Model evaluation metrics
- **backtest.py**: Backtesting framework
- **visualize.py**: Visualization tools
- **main.py**: Main script for training and evaluating models
- **menu.py**: GUI interface for interacting with the system

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Ensure you have appropriate data in CSV format with timestamp and price columns.

## Usage

### Through the GUI

1. Launch the GUI:

```bash
python menu.py
```

2. Use the tabs to configure:
   - Data settings
   - Model architecture
   - Training parameters
   - Backtesting
   - Visualization

### From Command Line

1. Train a model:

```bash
python main.py --data_path your_data.csv --model_type lstm --num_epochs 100
```

2. Run backtest:

```bash
python main.py --backtest --model_path checkpoints/your_model.pt
```

3. Visualize predictions:

```bash
python main.py --visualize --model_path checkpoints/your_model.pt
```

## Model Architecture

The system provides two main model architectures:

1. **LSTM Model**: Simple recurrent model suitable for smaller datasets
2. **CNN-LSTM Model**: Convolutional feature extraction followed by LSTM, better for capturing patterns

Both architectures output probabilistic predictions:
- Mean values (expected predictions)
- Standard deviations (uncertainty estimates)
- Confidence scores (model certainty)
- Multi-horizon forecasts (predictions at different timeframes)

## Trading Strategy

The trading system uses a confidence-based approach:
- Position sizes are adjusted based on prediction confidence
- Risk is managed according to uncertainty levels
- Multiple time horizons inform entry and exit decisions
- Stop losses and take profits can be calibrated to prediction uncertainty

## Visualization

The visualization module provides:
- Price predictions with uncertainty bands
- Confidence metric trends
- Calibration curves to assess model accuracy
- Sample distribution plots
- Trading signal visualization

## Evaluation Metrics

The system evaluates models using:
- Standard metrics: MSE, RMSE, MAE, R²
- Probabilistic metrics: NLL, CRPS, calibration error
- Interval coverage analysis
- Trading performance metrics

## Integration

The probabilistic models can be integrated with the existing trading environment through:
- The `TradingModel` class in trading_model.py
- Direct integration with env_rewards.py for uncertainty-aware rewards
- Position sizing based on prediction confidence

## License

This project is licensed under the MIT License - see the LICENSE file for details. 