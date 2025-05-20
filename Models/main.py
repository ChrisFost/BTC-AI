"""
Main script for training and evaluating probabilistic trading models.
"""

import argparse
import torch
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from prepare_data import DataPreparator
from train_probabilistic import ProbabilisticTrainer, create_dataloaders
from probabilistic_model import ProbabilisticLSTMModel, ProbabilisticCNNLSTMModel
from trading_model import TradingModel
from src.training.prob_evaluator import ModelEvaluator
from backtest import Backtester
from visualize import PredictionVisualizer

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train and evaluate probabilistic trading models')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data file')
    parser.add_argument('--price_column', type=str, default='close', help='Name of price column')
    parser.add_argument('--timestamp_column', type=str, default='timestamp', help='Name of timestamp column')
    parser.add_argument('--seq_len', type=int, default=100, help='Sequence length for LSTM')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set proportion')
    parser.add_argument('--val_size', type=float, default=0.1, help='Validation set proportion')
    
    # Horizon parameters
    parser.add_argument('--horizon_names', type=str, nargs='+', 
                        default=['scalping', 'short', 'medium', 'long'],
                        help='Names of prediction horizons')
    parser.add_argument('--horizon_steps', type=int, nargs='+', default=[5, 20, 60, 120],
                        help='Number of steps for each horizon')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='lstm', choices=['lstm', 'cnn_lstm'],
                        help='Type of model to use')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size for LSTM')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--save_freq', type=int, default=5, help='Save frequency (epochs)')
    parser.add_argument('--early_stopping', type=int, default=10, help='Early stopping patience (epochs)')
    
    # Evaluation parameters
    parser.add_argument('--evaluate', action='store_true', help='Evaluate model after training')
    parser.add_argument('--backtest', action='store_true', help='Run backtest after training')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    parser.add_argument('--confidence_threshold', type=float, default=0.6, 
                        help='Confidence threshold for trading')
    parser.add_argument('--risk_factor', type=float, default=0.5, help='Risk factor for position sizing')
    
    # Other parameters
    parser.add_argument('--device', type=str, default=None, 
                        help='Device to use (e.g., "cuda" or "cpu"), defaults to cuda if available')
    parser.add_argument('--output_dir', type=str, default='output', 
                        help='Output directory for logs and results')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return args

def setup_logging(output_dir):
    """Set up logging configuration"""
    log_file = os.path.join(
        output_dir,
        f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def load_data(data_path):
    """Load data from file"""
    logging.info(f"Loading data from {data_path}")
    
    # Determine file type from extension
    file_ext = os.path.splitext(data_path)[1].lower()
    
    if file_ext == '.csv':
        data = pd.read_csv(data_path)
    elif file_ext in ['.xls', '.xlsx']:
        data = pd.read_excel(data_path)
    elif file_ext == '.parquet':
        data = pd.read_parquet(data_path)
    elif file_ext == '.h5':
        data = pd.read_hdf(data_path)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")
    
    logging.info(f"Data loaded, shape: {data.shape}")
    return data

def identify_feature_columns(data, exclude_columns):
    """Identify feature columns in the dataset"""
    feature_columns = [col for col in data.columns if col not in exclude_columns]
    
    logging.info(f"Identified {len(feature_columns)} feature columns")
    return feature_columns

def train_model(args, train_loader, val_loader, input_dim):
    """Train model"""
    logging.info(f"Initializing {args.model_type.upper()} model with {args.hidden_size} hidden units "
                f"and {args.num_layers} layers on {args.device}")
    
    # Initialize model
    if args.model_type == 'lstm':
        model = ProbabilisticLSTMModel(
            input_dim=input_dim,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            horizon_names=args.horizon_names
        )
    else:  # cnn_lstm
        model = ProbabilisticCNNLSTMModel(
            input_dim=input_dim,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            horizon_names=args.horizon_names
        )
    
    # Create trainer
    trainer = ProbabilisticTrainer(
        model=model,
        lr=args.learning_rate,
        device=args.device,
        save_dir=args.save_dir,
        save_freq=args.save_freq,
        patience=args.early_stopping
    )
    
    # Train model
    best_model, best_checkpoint_path = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs
    )
    
    return best_model, best_checkpoint_path

def evaluate_model(args, model, test_data, horizon_names):
    """Evaluate model performance"""
    logging.info("Evaluating model performance")
    
    # Create evaluator
    evaluator = ModelEvaluator(
        log_dir=os.path.join(args.output_dir, 'evaluation_logs'),
        confidence_levels=[0.5, 0.8, 0.9, 0.95, 0.99],
        save_plots=True
    )
    
    # Extract test features and targets
    features = test_data['features']
    price_targets = test_data['price_targets']
    
    # Move to device
    features = torch.FloatTensor(features).to(args.device)
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        outputs = model(features)
    
    # Extract predictions
    pred_means = {}
    pred_stds = {}
    samples = {}
    targets = {}
    
    # Process each horizon
    for i, horizon in enumerate(horizon_names):
        # Extract targets and predictions
        targets[horizon] = price_targets[:, i].cpu().numpy()
        pred_means[horizon] = outputs['price_means'][:, i].cpu().numpy()
        pred_stds[horizon] = outputs['price_stds'][:, i].cpu().numpy()
        
        # Generate samples
        samples[horizon] = np.random.normal(
            pred_means[horizon][:, np.newaxis],
            pred_stds[horizon][:, np.newaxis],
            (len(pred_means[horizon]), 100)
        )
    
    # Evaluate model
    metrics = evaluator.evaluate_model(
        targets=targets,
        pred_means=pred_means,
        pred_stds=pred_stds,
        samples=samples
    )
    
    return metrics, targets, pred_means, pred_stds, samples

def run_backtest(args, model, test_data, feature_columns, price_column, timestamp_column):
    """Run backtest on test data"""
    logging.info("Running backtest")
    
    # Create DataFrame from test data
    df = pd.DataFrame(test_data['raw_features'], columns=feature_columns)
    df[price_column] = test_data['prices']
    df[timestamp_column] = test_data['timestamps']
    
    # Create trading model
    trading_model = TradingModel(
        model_type=args.model_type,
        input_dim=len(feature_columns),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        device=args.device,
        horizon_names=args.horizon_names,
        confidence_threshold=args.confidence_threshold,
        risk_factor=args.risk_factor
    )
    
    # Set model weights
    trading_model.model = model
    
    # Create backtester
    backtester = Backtester(
        initial_capital=100000.0,
        position_sizing='confidence',
        max_position_size=0.2,
        stop_loss_pct=0.05,
        take_profit_pct=0.1,
        trading_fee_pct=0.001,
        slippage_pct=0.001,
        log_dir=os.path.join(args.output_dir, 'backtest_logs'),
        save_results=True,
        save_trades=True,
        save_equity_curve=True
    )
    
    # Run backtest
    results = backtester.run_backtest(
        data=df,
        model=trading_model,
        feature_columns=feature_columns,
        price_column=price_column,
        timestamp_column=timestamp_column
    )
    
    # Plot equity curve
    backtester.plot_equity_curve(
        equity_curve=results['equity_curve'],
        show_drawdowns=True,
        save_path=os.path.join(args.output_dir, 'backtest_logs', 'equity_curve.png')
    )
    
    # Plot trade analysis
    backtester.plot_trade_analysis(
        trades=results['trades'],
        save_path=os.path.join(args.output_dir, 'backtest_logs', 'trade_analysis.png')
    )
    
    return results

def create_visualizations(args, test_data, targets, pred_means, pred_stds, samples):
    """Create visualizations"""
    logging.info("Creating visualizations")
    
    # Create visualizer
    visualizer = PredictionVisualizer(
        output_dir=os.path.join(args.output_dir, 'visualization_output'),
        save_plots=True,
        confidence_levels=[0.5, 0.8, 0.9, 0.95, 0.99]
    )
    
    # Plot multi-horizon predictions
    timestamps = test_data['timestamps']
    prices = test_data['prices']
    horizon_predictions = {
        horizon: (pred_means[horizon], pred_stds[horizon])
        for horizon in args.horizon_names
    }
    
    visualizer.plot_multi_horizon_predictions(
        prices=prices,
        horizon_predictions=horizon_predictions,
        timestamps=timestamps,
        title='Multi-Horizon Price Predictions',
        show_plot=False
    )
    
    # Plot confidence metrics
    confidence_values = {
        horizon: np.clip(1.0 / pred_stds[horizon], 0, 1)  # Simple confidence metric
        for horizon in args.horizon_names
    }
    
    visualizer.plot_confidence_metrics(
        confidence_values=confidence_values,
        timestamps=timestamps,
        threshold=args.confidence_threshold,
        title='Prediction Confidence Over Time',
        show_plot=False
    )
    
    # Plot interval coverage
    visualizer.plot_interval_coverage(
        targets=targets,
        pred_means=pred_means,
        pred_stds=pred_stds,
        horizon_names=args.horizon_names,
        title='Prediction Interval Coverage',
        show_plot=False
    )
    
    logging.info(f"Visualizations saved to {os.path.join(args.output_dir, 'visualization_output')}")

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'evaluation_logs'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'backtest_logs'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'visualization_output'), exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set up logging
    setup_logging(args.output_dir)
    
    # Log arguments
    logging.info("Starting probabilistic trading model training")
    logging.info(f"Arguments: {args}")
    
    # Set random seed
    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_state)
    
    # Load data
    data = load_data(args.data_path)
    
    # Identify feature columns
    exclude_columns = [args.price_column, args.timestamp_column]
    feature_columns = identify_feature_columns(data, exclude_columns)
    
    # Prepare data
    logging.info("Preparing data")
    data_preparator = DataPreparator(
        seq_len=args.seq_len,
        horizon_names=args.horizon_names,
        horizon_steps=args.horizon_steps,
        test_size=args.test_size,
        validation_size=args.val_size,
        random_state=args.random_state
    )
    
    train_data, val_data, test_data = data_preparator.prepare_data(
        data,
        feature_columns=feature_columns,
        price_column=args.price_column,
        timestamp_column=args.timestamp_column
    )
    
    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        train_data=train_data,
        val_data=val_data,
        batch_size=args.batch_size
    )
    
    # Train model
    input_dim = train_data['features'].shape[2]
    best_model, best_checkpoint_path = train_model(args, train_loader, val_loader, input_dim)
    
    # Evaluate model
    if args.evaluate:
        metrics, targets, pred_means, pred_stds, samples = evaluate_model(
            args, best_model, test_data, args.horizon_names
        )
    
    # Run backtest
    if args.backtest:
        backtest_results = run_backtest(
            args, best_model, test_data, feature_columns, args.price_column, args.timestamp_column
        )
    
    # Create visualizations
    if args.visualize and args.evaluate:
        create_visualizations(args, test_data, targets, pred_means, pred_stds, samples)
    
    logging.info("Training and evaluation complete")

if __name__ == "__main__":
    main() 