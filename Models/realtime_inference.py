"""
Real-time inference script for probabilistic trading model.
This module provides tools for making predictions on streaming data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import torch
from datetime import datetime, timedelta
import logging
import os
import time
import json
from collections import deque
import threading
import argparse

from trading_model import TradingModel
from probabilistic_model import ProbabilisticLSTMModel, ProbabilisticCNNLSTMModel

class RealTimeInference:
    """Class for real-time inference with probabilistic models"""
    
    def __init__(
        self,
        model_path: str,
        model_type: str = 'lstm',
        input_dim: int = 20,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        device: str = None,
        horizon_names: List[str] = ['scalping', 'short', 'medium', 'long'],
        confidence_threshold: float = 0.6,
        risk_factor: float = 0.5,
        buffer_size: int = 100,
        log_dir: str = 'inference_logs',
        save_predictions: bool = True,
        prediction_interval: int = 60  # seconds
    ):
        self.model_path = model_path
        self.model_type = model_type
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.horizon_names = horizon_names
        self.confidence_threshold = confidence_threshold
        self.risk_factor = risk_factor
        self.buffer_size = buffer_size
        self.log_dir = log_dir
        self.save_predictions = save_predictions
        self.prediction_interval = prediction_interval
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize data buffer
        self.data_buffer = deque(maxlen=buffer_size)
        
        # Initialize prediction history
        self.prediction_history = []
        
        # Setup logging
        self.setup_logging()
        
        # Initialize model
        self.initialize_model()
        
        # Running flag
        self.running = False
    
    def setup_logging(self):
        """Set up logging configuration"""
        log_file = os.path.join(
            self.log_dir,
            f'inference_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def initialize_model(self):
        """Initialize the trading model"""
        logging.info(f"Initializing model of type {self.model_type} with device {self.device}")
        
        self.trading_model = TradingModel(
            model_type=self.model_type,
            input_dim=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            device=self.device,
            horizon_names=self.horizon_names,
            confidence_threshold=self.confidence_threshold,
            risk_factor=self.risk_factor
        )
        
        # Load checkpoint
        if os.path.exists(self.model_path):
            self.trading_model.load_checkpoint(self.model_path)
            logging.info(f"Model loaded from {self.model_path}")
        else:
            logging.warning(f"Model checkpoint not found at {self.model_path}")
    
    def update_data(self, new_data: Dict[str, Any]):
        """
        Update data buffer with new market data
        
        Args:
            new_data: Dictionary containing new market data
        """
        # Add timestamp if not present
        if 'timestamp' not in new_data:
            new_data['timestamp'] = datetime.now()
        
        # Add to buffer
        self.data_buffer.append(new_data)
        
        logging.debug(f"Data buffer updated, size: {len(self.data_buffer)}")
    
    def preprocess_data(self) -> torch.Tensor:
        """
        Preprocess data from buffer for model input
        
        Returns:
            Tensor of preprocessed data
        """
        if len(self.data_buffer) < self.buffer_size:
            logging.warning(f"Data buffer not filled yet ({len(self.data_buffer)}/{self.buffer_size})")
            return None
        
        # Convert buffer to DataFrame
        df = pd.DataFrame(list(self.data_buffer))
        
        # Extract features
        features = df.drop(columns=['timestamp']).values.astype(np.float32)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features).unsqueeze(0)  # Add batch dimension
        
        return features_tensor
    
    def make_prediction(self) -> Dict[str, Any]:
        """
        Generate prediction using current data
        
        Returns:
            Dictionary containing prediction results
        """
        # Preprocess data
        features = self.preprocess_data()
        
        if features is None:
            return None
        
        # Current price from latest data
        current_price = self.data_buffer[-1].get('close', 0.0)
        
        # Prepare observation (this will depend on your trading environment structure)
        observation = {
            'features': features,
            'current_price': current_price,
            'timestamp': datetime.now()
        }
        
        # Current position and balance (placeholder values)
        current_position = 0.0
        account_balance = 10000.0
        
        # Generate prediction
        prediction, orders = self.trading_model.predict(
            observation,
            account_balance,
            current_position
        )
        
        # Add timestamp
        prediction['timestamp'] = datetime.now()
        
        # Log prediction
        summary = self.summarize_prediction(prediction)
        logging.info(f"Prediction: {summary}")
        
        # Save prediction
        if self.save_predictions:
            self.prediction_history.append(prediction)
            self.save_prediction(prediction)
        
        return prediction
    
    def summarize_prediction(self, prediction: Dict[str, Any]) -> str:
        """
        Create a summary string from prediction
        
        Args:
            prediction: Dictionary containing prediction results
            
        Returns:
            Summary string
        """
        # Extract key elements for summary
        actions = prediction.get('actions', [0, 0, 0])
        action_probs = prediction.get('action_probs', [0, 0, 0])
        
        action_names = ['Buy', 'Hold', 'Sell']
        action_str = action_names[np.argmax(action_probs)]
        
        # Get confidence for each horizon
        confidences = prediction.get('confidences', {})
        conf_str = ", ".join([f"{h}: {c:.2f}" for h, c in confidences.items()])
        
        # Summarize price predictions
        price_means = prediction.get('price_means', {})
        price_stds = prediction.get('price_stds', {})
        
        price_str = ", ".join([
            f"{h}: {m:.4f}±{s:.4f}" 
            for h, (m, s) in zip(
                self.horizon_names, 
                zip(
                    [price_means.get(h, 0) for h in self.horizon_names],
                    [price_stds.get(h, 0) for h in self.horizon_names]
                )
            )
        ])
        
        return f"Action: {action_str} ({action_probs[np.argmax(action_probs)]:.2f}), Conf: [{conf_str}], Price: [{price_str}]"
    
    def save_prediction(self, prediction: Dict[str, Any]):
        """
        Save prediction to file
        
        Args:
            prediction: Dictionary containing prediction results
        """
        # Convert to serializable format
        serializable = {}
        
        for key, value in prediction.items():
            if isinstance(value, np.ndarray):
                serializable[key] = value.tolist()
            elif isinstance(value, torch.Tensor):
                serializable[key] = value.cpu().numpy().tolist()
            elif isinstance(value, datetime):
                serializable[key] = value.isoformat()
            elif isinstance(value, dict):
                serializable[key] = {
                    k: v.item() if isinstance(v, (np.ndarray, torch.Tensor)) else v
                    for k, v in value.items()
                }
            else:
                serializable[key] = value
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prediction_file = os.path.join(self.log_dir, f'prediction_{timestamp}.json')
        
        with open(prediction_file, 'w') as f:
            json.dump(serializable, f, indent=2)
    
    def run_inference_loop(self):
        """Run continuous inference loop"""
        self.running = True
        logging.info("Starting inference loop")
        
        last_prediction_time = datetime.now() - timedelta(seconds=self.prediction_interval)
        
        while self.running:
            now = datetime.now()
            
            # Check if it's time to make a prediction
            if (now - last_prediction_time).total_seconds() >= self.prediction_interval:
                # Make prediction
                prediction = self.make_prediction()
                
                if prediction is not None:
                    last_prediction_time = now
            
            # Sleep to avoid high CPU usage
            time.sleep(1)
    
    def start(self):
        """Start inference in a separate thread"""
        self.inference_thread = threading.Thread(target=self.run_inference_loop)
        self.inference_thread.daemon = True
        self.inference_thread.start()
        logging.info("Inference thread started")
    
    def stop(self):
        """Stop inference loop"""
        self.running = False
        if hasattr(self, 'inference_thread'):
            self.inference_thread.join(timeout=5.0)
        logging.info("Inference stopped")


def connect_to_data_source(source_type: str, **kwargs):
    """
    Connect to a data source and return a generator that yields data
    
    Args:
        source_type: Type of data source ('csv', 'api', 'random')
        **kwargs: Additional arguments for the data source
        
    Returns:
        Generator yielding data dictionaries
    """
    if source_type == 'csv':
        # Load data from CSV file
        file_path = kwargs.get('file_path')
        if not file_path:
            raise ValueError("file_path must be provided for csv data source")
        
        df = pd.read_csv(file_path)
        interval = kwargs.get('interval', 1.0)  # seconds
        
        for _, row in df.iterrows():
            yield row.to_dict()
            time.sleep(interval)
    
    elif source_type == 'api':
        # Connect to API (implement based on your specific API)
        import requests
        
        api_url = kwargs.get('api_url')
        api_key = kwargs.get('api_key')
        interval = kwargs.get('interval', 60.0)  # seconds
        
        headers = {'Authorization': f'Bearer {api_key}'} if api_key else {}
        
        while True:
            try:
                response = requests.get(api_url, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    yield data
                else:
                    logging.error(f"API request failed with status {response.status_code}")
            except Exception as e:
                logging.error(f"Error fetching data from API: {e}")
            
            time.sleep(interval)
    
    elif source_type == 'random':
        # Generate random data for testing
        interval = kwargs.get('interval', 1.0)  # seconds
        feature_count = kwargs.get('feature_count', 20)
        
        price = 100.0
        
        while True:
            # Generate random price movement
            price_change = np.random.normal(0, 1) * 0.01
            price *= (1 + price_change)
            
            # Generate random features
            features = np.random.normal(0, 1, feature_count)
            
            data = {
                'timestamp': datetime.now(),
                'close': price,
                'open': price * (1 - np.random.uniform(0, 0.005)),
                'high': price * (1 + np.random.uniform(0, 0.005)),
                'low': price * (1 - np.random.uniform(0, 0.005)),
                'volume': np.random.exponential(1000)
            }
            
            # Add features
            for i in range(feature_count):
                data[f'feature_{i}'] = features[i]
            
            yield data
            time.sleep(interval)
    
    else:
        raise ValueError(f"Unknown data source type: {source_type}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run real-time inference with probabilistic model')
    
    # Model parameters
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, default='lstm', choices=['lstm', 'cnn_lstm'], 
                        help='Type of model to use')
    parser.add_argument('--input_dim', type=int, default=20, help='Input dimension for model')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size for LSTM')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers')
    
    # Inference parameters
    parser.add_argument('--confidence_threshold', type=float, default=0.6, 
                        help='Confidence threshold for predictions')
    parser.add_argument('--risk_factor', type=float, default=0.5, help='Risk factor for position sizing')
    parser.add_argument('--buffer_size', type=int, default=100, help='Size of data buffer')
    parser.add_argument('--prediction_interval', type=int, default=60, 
                        help='Interval between predictions in seconds')
    
    # Data source parameters
    parser.add_argument('--data_source', type=str, default='random', 
                        choices=['csv', 'api', 'random'], help='Type of data source')
    parser.add_argument('--file_path', type=str, help='Path to CSV file (for csv data source)')
    parser.add_argument('--api_url', type=str, help='API URL (for api data source)')
    parser.add_argument('--api_key', type=str, help='API key (for api data source)')
    parser.add_argument('--data_interval', type=float, default=1.0, 
                        help='Interval between data points in seconds')
    
    # Other parameters
    parser.add_argument('--log_dir', type=str, default='inference_logs', 
                        help='Directory to save logs and predictions')
    parser.add_argument('--device', type=str, default=None, 
                        help='Device to use (e.g., "cuda" or "cpu"), defaults to cuda if available')
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return args


def main():
    """Main function"""
    args = parse_args()
    
    # Initialize inference engine
    inference = RealTimeInference(
        model_path=args.model_path,
        model_type=args.model_type,
        input_dim=args.input_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
        risk_factor=args.risk_factor,
        buffer_size=args.buffer_size,
        log_dir=args.log_dir,
        prediction_interval=args.prediction_interval
    )
    
    # Start inference thread
    inference.start()
    
    try:
        # Connect to data source
        data_source_kwargs = {
            'interval': args.data_interval,
            'file_path': args.file_path,
            'api_url': args.api_url,
            'api_key': args.api_key
        }
        
        data_generator = connect_to_data_source(args.data_source, **data_source_kwargs)
        
        # Process data
        logging.info(f"Starting data processing from source: {args.data_source}")
        
        for data in data_generator:
            # Update data buffer
            inference.update_data(data)
            
            # Check if user wants to quit
            if input("Press 'q' to quit (or any other key to continue): ").lower() == 'q':
                break
    
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    
    finally:
        # Stop inference
        inference.stop()
        logging.info("Inference stopped")


if __name__ == "__main__":
    main() 