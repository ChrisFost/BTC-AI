#!/usr/bin/env python
"""
Real-time Inference for Probabilistic Trading Models

This module provides functionality to connect to market data sources,
run probabilistic predictions in real-time, and log results.
"""

import time
import json
import argparse
import os
import sys
import logging
import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import threading
import queue
import requests
import websocket
import ccxt

from models import create_model
from src.utils.utils import log, setup_logging, configure_gpu
from src.utils.prediction_visualizer import PredictionVisualizer

class RealTimeInference:
    """
    Real-time inference engine for probabilistic trading models.
    
    Features:
    - Connects to various market data sources
    - Processes streaming data for model input
    - Generates probabilistic predictions with uncertainty
    - Logs predictions and confidence intervals
    - Optional visualization of real-time predictions
    """
    
    def __init__(self, model_path, config_path, log_dir="logs", visualize=False,
                 market_data_source="ccxt", symbol="BTC/USDT", interval="5m"):
        """
        Initialize real-time inference engine.
        
        Args:
            model_path (str): Path to trained model checkpoint.
            config_path (str): Path to configuration file.
            log_dir (str): Directory to save prediction logs.
            visualize (bool): Whether to visualize predictions in real-time.
            market_data_source (str): Source for market data ('ccxt', 'binance', 'coinbase', 'file').
            symbol (str): Trading symbol to monitor (e.g., 'BTC/USDT').
            interval (str): Time interval for data (e.g., '1m', '5m', '1h').
        """
        self.model_path = model_path
        self.config_path = config_path
        self.log_dir = log_dir
        self.visualize = visualize
        self.market_data_source = market_data_source
        self.symbol = symbol
        self.interval = interval
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize logging
        self.logger = setup_logging(log_dir, "realtime_inference")
        
        # Load configuration
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
                log(f"Loaded configuration from {config_path}")
        except Exception as e:
            log(f"Error loading configuration: {str(e)}")
            self.config = {
                "model_type": "actor_critic",
                "feature_columns": [],
                "sequence_length": 100,
                "prediction_horizons": [12, 36, 72, 144],
                "confidence_threshold": 0.6,
                "device": "cpu",
                "save_predictions": True,
                "plot_interval": 60,  # seconds
                "max_buffer_size": 1000,
                "feature_scaling": {
                    "enabled": True,
                    "method": "standard",  # 'standard', 'minmax', or 'robust'
                    "params": {}
                }
            }
        
        # Initialize model
        self.device = configure_gpu(self.config.get("device", "cpu"))
        self.model = self._load_model()
        
        # Data buffers
        self.raw_data_buffer = []
        self.processed_data_buffer = []
        self.prediction_buffer = []
        self.max_buffer_size = self.config.get("max_buffer_size", 1000)
        
        # Set up data processing parameters
        self.sequence_length = self.config.get("sequence_length", 100)
        self.prediction_horizons = self.config.get("prediction_horizons", [12, 36, 72, 144])
        self.feature_columns = self.config.get("feature_columns", [])
        
        # Visualization
        if visualize:
            self.visualizer = PredictionVisualizer()
            self.last_plot_time = time.time()
            self.plot_interval = self.config.get("plot_interval", 60)  # seconds
        
        # Exchange/API clients
        self.exchange = None
        self.websocket = None
        self.api_key = self.config.get("api_key", "")
        self.api_secret = self.config.get("api_secret", "")
        
        # Status
        self.running = False
        self.data_queue = queue.Queue()
        
        # Connect to data source
        self._setup_data_source()
        
        log(f"Initialized real-time inference engine for {symbol} on {interval} timeframe")
        
    def _load_model(self):
        """
        Load trained model from checkpoint.
        
        Returns:
            torch.nn.Module: Loaded model.
        """
        try:
            model_type = self.config.get("model_type", "actor_critic")
            model_kwargs = self.config.get("model_kwargs", {})
            
            input_dim = self.config.get("input_dim", 64)
            hidden_size = self.config.get("hidden_size", 128)
            
            # Create model
            model = create_model(
                model_type=model_type,
                input_dim=input_dim,
                hidden_size=hidden_size,
                horizons=self.prediction_horizons,
                device=self.device,
                **model_kwargs
            )
            
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            log(f"Successfully loaded model from {self.model_path}")
            return model
        except Exception as e:
            log(f"Error loading model: {str(e)}")
            sys.exit(1)
    
    def _setup_data_source(self):
        """Set up connection to market data source."""
        try:
            if self.market_data_source == "ccxt":
                # CCXT for various exchanges
                exchange_id = self.config.get("exchange", "binance")
                exchange_class = getattr(ccxt, exchange_id)
                self.exchange = exchange_class({
                    'apiKey': self.api_key,
                    'secret': self.api_secret,
                    'timeout': 30000,
                    'enableRateLimit': True,
                })
                log(f"Connected to {exchange_id} via ccxt")
                
            elif self.market_data_source == "binance":
                # Direct Binance WebSocket connection
                base_url = "wss://stream.binance.com:9443/ws/"
                symbol_lower = self.symbol.replace('/', '').lower()
                stream_url = f"{base_url}{symbol_lower}@kline_{self.interval}"
                
                self.websocket = websocket.WebSocketApp(
                    stream_url,
                    on_message=self._on_binance_message,
                    on_error=self._on_ws_error,
                    on_close=self._on_ws_close,
                    on_open=self._on_ws_open
                )
                log(f"Set up Binance WebSocket connection for {self.symbol}")
                
            elif self.market_data_source == "coinbase":
                # Coinbase Pro WebSocket
                self.websocket = websocket.WebSocketApp(
                    "wss://ws-feed.pro.coinbase.com",
                    on_message=self._on_coinbase_message,
                    on_error=self._on_ws_error,
                    on_close=self._on_ws_close,
                    on_open=self._on_coinbase_open
                )
                log(f"Set up Coinbase Pro WebSocket connection for {self.symbol}")
                
            elif self.market_data_source == "file":
                # Load historical data from file for testing/simulation
                data_file = self.config.get("data_file", "")
                if not data_file:
                    raise ValueError("data_file not specified for file data source")
                
                data = pd.read_csv(data_file)
                self.simulation_data = data
                self.simulation_index = 0
                log(f"Loaded {len(data)} data points from {data_file} for simulation")
                
            else:
                raise ValueError(f"Unsupported market data source: {self.market_data_source}")
                
        except Exception as e:
            log(f"Error setting up data source: {str(e)}")
            sys.exit(1)
    
    def _on_ws_open(self, ws):
        """WebSocket open callback."""
        log("WebSocket connection opened")
    
    def _on_ws_error(self, ws, error):
        """WebSocket error callback."""
        log(f"WebSocket error: {error}")
    
    def _on_ws_close(self, ws, close_status_code, close_msg):
        """WebSocket close callback."""
        log(f"WebSocket connection closed: {close_msg} ({close_status_code})")
        
        # Attempt to reconnect after a delay
        if self.running:
            log("Attempting to reconnect in 5 seconds...")
            time.sleep(5)
            self._setup_data_source()
            self._start_websocket()
    
    def _on_binance_message(self, ws, message):
        """
        Process messages from Binance WebSocket.
        
        Args:
            ws: WebSocket instance.
            message: Message received from WebSocket.
        """
        try:
            data = json.loads(message)
            
            # Filter for kline (candlestick) data
            if 'k' in data:
                kline = data['k']
                
                # Only process closed candles
                if kline['x']:  # 'x' is true when the candle is closed
                    bar_data = {
                        'timestamp': datetime.fromtimestamp(kline['T'] / 1000),  # Close time
                        'open': float(kline['o']),
                        'high': float(kline['h']),
                        'low': float(kline['l']),
                        'close': float(kline['c']),
                        'volume': float(kline['v']),
                        'symbol': self.symbol
                    }
                    
                    self.data_queue.put(bar_data)
                    log(f"Received closed candle: {bar_data['timestamp']} - {bar_data['close']}")
        except Exception as e:
            log(f"Error processing Binance message: {str(e)}")
    
    def _on_coinbase_open(self, ws):
        """Coinbase WebSocket open callback with subscription."""
        subscribe_message = {
            "type": "subscribe",
            "product_ids": [self.symbol],
            "channels": ["ticker"]
        }
        ws.send(json.dumps(subscribe_message))
        log(f"Subscribed to Coinbase {self.symbol} ticker")
    
    def _on_coinbase_message(self, ws, message):
        """
        Process messages from Coinbase WebSocket.
        
        Args:
            ws: WebSocket instance.
            message: Message received from WebSocket.
        """
        try:
            data = json.loads(message)
            
            # Filter for ticker data
            if data.get('type') == 'ticker':
                ticker_data = {
                    'timestamp': datetime.fromisoformat(data['time'].replace('Z', '+00:00')).astimezone(None),
                    'price': float(data['price']),
                    'volume_24h': float(data.get('volume_24h', 0)),
                    'symbol': data['product_id']
                }
                
                # Convert ticker to OHLC (simplified)
                bar_data = {
                    'timestamp': ticker_data['timestamp'],
                    'open': ticker_data['price'],
                    'high': ticker_data['price'],
                    'low': ticker_data['price'],
                    'close': ticker_data['price'],
                    'volume': 0,  # Can't determine from ticker
                    'symbol': ticker_data['symbol']
                }
                
                self.data_queue.put(bar_data)
        except Exception as e:
            log(f"Error processing Coinbase message: {str(e)}")
    
    def _fetch_historical_data(self):
        """
        Fetch historical data to initialize the buffer.
        
        Returns:
            pandas.DataFrame: Historical OHLCV data.
        """
        try:
            log(f"Fetching historical {self.interval} data for {self.symbol}...")
            
            if self.market_data_source == "ccxt":
                # Convert interval to milliseconds for CCXT
                timeframe = self.interval
                since = None  # could calculate based on sequence_length
                limit = max(self.sequence_length + max(self.prediction_horizons), 1000)
                
                ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe, since, limit)
                
                # Convert to DataFrame
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df['symbol'] = self.symbol
                
            elif self.market_data_source == "file":
                # Use simulation data
                df = self.simulation_data.copy()
                if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                
            else:
                # For WebSocket sources, we may need to fetch historical data separately
                # This is simplified - in practice, you'd make REST API calls to the respective exchange
                raise NotImplementedError(f"Historical data fetch not implemented for {self.market_data_source}")
            
            log(f"Fetched {len(df)} historical bars")
            return df
            
        except Exception as e:
            log(f"Error fetching historical data: {str(e)}")
            return pd.DataFrame()
    
    def _process_data(self, data):
        """
        Process raw market data into features for model input.
        
        Args:
            data (dict or pandas.DataFrame): Raw market data.
            
        Returns:
            numpy.ndarray: Processed features.
        """
        # If data is a single bar (dict), add it to the buffer
        if isinstance(data, dict):
            self.raw_data_buffer.append(data)
            
            # Trim buffer if it exceeds max size
            if len(self.raw_data_buffer) > self.max_buffer_size:
                self.raw_data_buffer = self.raw_data_buffer[-self.max_buffer_size:]
                
        # If data is a DataFrame, replace buffer
        elif isinstance(data, pd.DataFrame):
            # Convert DataFrame rows to list of dicts
            self.raw_data_buffer = data.to_dict('records')[-self.max_buffer_size:]
        
        # Not enough data
        if len(self.raw_data_buffer) < self.sequence_length:
            log(f"Not enough data: {len(self.raw_data_buffer)}/{self.sequence_length}")
            return None
            
        # Convert buffer to DataFrame
        df = pd.DataFrame(self.raw_data_buffer)
        
        # Apply feature engineering based on available columns
        # This is a placeholder - replace with your actual feature engineering
        engineered_df = self._engineer_features(df)
        
        # Select features for model input
        if self.feature_columns:
            # Use specified feature columns
            feature_cols = [col for col in self.feature_columns if col in engineered_df.columns]
        else:
            # Default to all numeric columns except timestamp
            feature_cols = engineered_df.select_dtypes(include=['number']).columns.tolist()
        
        # Scale features based on configuration
        if self.config.get("feature_scaling", {}).get("enabled", True):
            scaled_df = self._scale_features(engineered_df[feature_cols])
        else:
            scaled_df = engineered_df[feature_cols]
        
        # Get the last sequence_length rows for model input
        features = scaled_df.iloc[-self.sequence_length:].values
        
        # Store in processed buffer
        self.processed_data_buffer = features
        
        return features
    
    def _engineer_features(self, df):
        """
        Apply feature engineering to raw data.
        
        Args:
            df (pandas.DataFrame): Raw data.
            
        Returns:
            pandas.DataFrame: Data with engineered features.
        """
        # Simple features - replace/expand based on your model requirements
        try:
            df_copy = df.copy()
            
            # Ensure we have OHLCV columns
            essential_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df_copy.columns for col in essential_cols):
                log(f"Missing essential columns. Available: {df_copy.columns.tolist()}")
                return df_copy
            
            # Calculate returns
            df_copy['returns'] = df_copy['close'].pct_change()
            df_copy['log_returns'] = np.log(df_copy['close'] / df_copy['close'].shift(1))
            
            # Price differentials
            df_copy['hl_diff'] = (df_copy['high'] - df_copy['low']) / df_copy['low']
            df_copy['co_diff'] = (df_copy['close'] - df_copy['open']) / df_copy['open']
            
            # Volume features
            df_copy['volume_change'] = df_copy['volume'].pct_change()
            
            # Simple Moving Averages
            df_copy['sma5'] = df_copy['close'].rolling(5).mean()
            df_copy['sma20'] = df_copy['close'].rolling(20).mean()
            df_copy['sma50'] = df_copy['close'].rolling(50).mean()
            
            # Exponential Moving Averages
            df_copy['ema12'] = df_copy['close'].ewm(span=12).mean()
            df_copy['ema26'] = df_copy['close'].ewm(span=26).mean()
            
            # MACD
            df_copy['macd'] = df_copy['ema12'] - df_copy['ema26']
            df_copy['macd_signal'] = df_copy['macd'].ewm(span=9).mean()
            df_copy['macd_hist'] = df_copy['macd'] - df_copy['macd_signal']
            
            # Bollinger Bands
            df_copy['bb_middle'] = df_copy['close'].rolling(20).mean()
            df_copy['bb_std'] = df_copy['close'].rolling(20).std()
            df_copy['bb_upper'] = df_copy['bb_middle'] + 2 * df_copy['bb_std']
            df_copy['bb_lower'] = df_copy['bb_middle'] - 2 * df_copy['bb_std']
            df_copy['bb_width'] = (df_copy['bb_upper'] - df_copy['bb_lower']) / df_copy['bb_middle']
            
            # RSI (14-period)
            delta = df_copy['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rs = gain / loss
            df_copy['rsi14'] = 100 - (100 / (1 + rs))
            
            # Fill NaN values
            df_copy = df_copy.fillna(method='bfill').fillna(0)
            
            return df_copy
            
        except Exception as e:
            log(f"Error in feature engineering: {str(e)}")
            return df
    
    def _scale_features(self, df):
        """
        Scale features based on configured method.
        
        Args:
            df (pandas.DataFrame): Features to scale.
            
        Returns:
            pandas.DataFrame: Scaled features.
        """
        method = self.config.get("feature_scaling", {}).get("method", "standard")
        
        try:
            if method == "standard":
                # Standardization (zero mean, unit variance)
                for col in df.columns:
                    mean = df[col].mean()
                    std = df[col].std()
                    if std > 0:
                        df[col] = (df[col] - mean) / std
                        
            elif method == "minmax":
                # Min-Max scaling to [0, 1]
                for col in df.columns:
                    min_val = df[col].min()
                    max_val = df[col].max()
                    range_val = max_val - min_val
                    if range_val > 0:
                        df[col] = (df[col] - min_val) / range_val
                        
            elif method == "robust":
                # Robust scaling based on percentiles
                for col in df.columns:
                    median = df[col].median()
                    p25 = df[col].quantile(0.25)
                    p75 = df[col].quantile(0.75)
                    iqr = p75 - p25
                    if iqr > 0:
                        df[col] = (df[col] - median) / iqr
            
            return df
            
        except Exception as e:
            log(f"Error scaling features: {str(e)}")
            return df
    
    def _make_prediction(self, features):
        """
        Generate probabilistic predictions with the model.
        
        Args:
            features (numpy.ndarray): Processed features.
            
        Returns:
            dict: Prediction results with means, standard deviations, and confidence scores.
        """
        if features is None or len(features) < self.sequence_length:
            return None
            
        try:
            # Convert to tensor
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(x)
                
            # Extract predictions for each horizon
            predictions = {}
            current_time = datetime.now()
            
            # Get the latest close price from raw buffer
            last_close = self.raw_data_buffer[-1]['close']
            
            for i, horizon in enumerate(self.prediction_horizons):
                horizon_key = f"h{horizon}"
                
                # Get mean and std for this horizon
                mean = outputs['predictions'][horizon_key].cpu().numpy()[0][0]
                std = outputs['prediction_stds'][horizon_key].cpu().numpy()[0][0] 
                confidence = outputs['confidence'][horizon_key].cpu().numpy()[0][0]
                
                # Calculate target time for this horizon
                if self.interval == '1m':
                    target_time = current_time + timedelta(minutes=horizon)
                elif self.interval == '5m':
                    target_time = current_time + timedelta(minutes=5*horizon)
                elif self.interval == '15m':
                    target_time = current_time + timedelta(minutes=15*horizon)
                elif self.interval == '1h':
                    target_time = current_time + timedelta(hours=horizon)
                elif self.interval == '4h':
                    target_time = current_time + timedelta(hours=4*horizon)
                elif self.interval == '1d':
                    target_time = current_time + timedelta(days=horizon)
                else:
                    target_time = current_time + timedelta(minutes=5*horizon)  # Default assumption
                
                # Convert returns to price predictions
                # Assuming model outputs percent change in price
                predicted_price = last_close * (1 + mean)
                predicted_std_price = last_close * std
                
                # Store prediction
                predictions[horizon] = {
                    'target_time': target_time,
                    'mean': mean,
                    'mean_price': predicted_price,
                    'std': std,
                    'std_price': predicted_std_price,
                    'confidence': confidence,
                    'lower_bound_68': predicted_price - predicted_std_price,
                    'upper_bound_68': predicted_price + predicted_std_price,
                    'lower_bound_95': predicted_price - 2*predicted_std_price,
                    'upper_bound_95': predicted_price + 2*predicted_std_price
                }
            
            # Get overall model outputs
            trend_strength = outputs['trend_strength'].cpu().numpy()[0][0]
            
            # Store in prediction buffer
            prediction_entry = {
                'timestamp': current_time,
                'current_price': last_close,
                'predictions': predictions,
                'trend_strength': trend_strength
            }
            
            self.prediction_buffer.append(prediction_entry)
            
            # Trim buffer if needed
            if len(self.prediction_buffer) > self.max_buffer_size:
                self.prediction_buffer = self.prediction_buffer[-self.max_buffer_size:]
            
            return prediction_entry
            
        except Exception as e:
            log(f"Error making prediction: {str(e)}")
            return None
    
    def _log_prediction(self, prediction):
        """
        Log prediction to file.
        
        Args:
            prediction (dict): Prediction results.
        """
        if not prediction or not self.config.get("save_predictions", True):
            return
            
        try:
            # Create prediction log file with date
            date_str = datetime.now().strftime("%Y%m%d")
            log_file = os.path.join(self.log_dir, f"predictions_{date_str}.csv")
            
            # Check if file exists
            file_exists = os.path.isfile(log_file)
            
            # Extract data for logging
            current_time = prediction['timestamp']
            current_price = prediction['current_price']
            
            # Prepare rows for each horizon
            rows = []
            for horizon, pred in prediction['predictions'].items():
                row = {
                    'timestamp': current_time,
                    'price': current_price,
                    'horizon': horizon,
                    'target_time': pred['target_time'],
                    'predicted_mean': pred['mean'],
                    'predicted_price': pred['mean_price'],
                    'predicted_std': pred['std'],
                    'confidence': pred['confidence'],
                    'lower_bound_68': pred['lower_bound_68'],
                    'upper_bound_68': pred['upper_bound_68'],
                    'lower_bound_95': pred['lower_bound_95'],
                    'upper_bound_95': pred['upper_bound_95'],
                    'trend_strength': prediction['trend_strength']
                }
                rows.append(row)
            
            # Write to CSV
            df = pd.DataFrame(rows)
            df.to_csv(log_file, mode='a', header=not file_exists, index=False)
            
        except Exception as e:
            log(f"Error logging prediction: {str(e)}")
    
    def _update_visualization(self):
        """Update real-time visualization of predictions."""
        if not self.visualize or len(self.prediction_buffer) < 2:
            return
            
        # Check if it's time to update the plot
        current_time = time.time()
        if current_time - self.last_plot_time < self.plot_interval:
            return
            
        try:
            # Create new figure
            plt.figure(figsize=(12, 8))
            
            # Extract data for plotting
            timestamps = []
            prices = []
            pred_mean = []
            pred_std = []
            
            # Choose a specific horizon for visualization
            vis_horizon = self.prediction_horizons[0]  # Use first horizon by default
            
            for entry in self.prediction_buffer[-100:]:  # Last 100 entries
                timestamps.append(entry['timestamp'])
                prices.append(entry['current_price'])
                
                if vis_horizon in entry['predictions']:
                    pred = entry['predictions'][vis_horizon]
                    pred_mean.append(pred['mean_price'])
                    pred_std.append(pred['std_price'])
                else:
                    # Use previous values if not available
                    pred_mean.append(pred_mean[-1] if pred_mean else entry['current_price'])
                    pred_std.append(pred_std[-1] if pred_std else 0)
            
            # Convert to arrays
            prices = np.array(prices)
            pred_mean = np.array(pred_mean)
            pred_std = np.array(pred_std)
            
            # Plot actual prices
            plt.plot(timestamps, prices, label='Actual Price', color='black')
            
            # Plot predicted mean
            plt.plot(timestamps, pred_mean, label=f'Predicted (h={vis_horizon})', color='blue')
            
            # Plot confidence intervals
            plt.fill_between(timestamps, pred_mean - pred_std, pred_mean + pred_std, 
                            color='blue', alpha=0.2, label='68% CI')
            plt.fill_between(timestamps, pred_mean - 2*pred_std, pred_mean + 2*pred_std, 
                            color='blue', alpha=0.1, label='95% CI')
            
            # Format plot
            plt.title(f"Real-time {self.symbol} Price Prediction (Horizon: {vis_horizon})")
            plt.xlabel("Time")
            plt.ylabel("Price")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Format x-axis with dates
            plt.gcf().autofmt_xdate()
            
            # Show plot (non-blocking)
            plt.draw()
            plt.pause(0.001)
            
            # Update last plot time
            self.last_plot_time = current_time
            
        except Exception as e:
            log(f"Error updating visualization: {str(e)}")
    
    def _start_websocket(self):
        """Start WebSocket connection in a separate thread."""
        if self.websocket:
            websocket_thread = threading.Thread(target=self.websocket.run_forever)
            websocket_thread.daemon = True
            websocket_thread.start()
    
    def _simulate_data(self):
        """Simulate data stream from file for testing."""
        if not hasattr(self, 'simulation_data') or self.simulation_data is None:
            log("No simulation data available")
            return
            
        while self.running and self.simulation_index < len(self.simulation_data):
            # Get current row
            row = self.simulation_data.iloc[self.simulation_index]
            
            # Convert to dictionary
            data = row.to_dict()
            
            # Add to queue
            self.data_queue.put(data)
            
            # Increment index
            self.simulation_index += 1
            
            # Simulate real-time delay
            time.sleep(self.config.get("simulation_delay", 1))
    
    def _polling_loop(self):
        """Polling loop for exchanges that don't support WebSockets."""
        last_poll_time = 0
        poll_interval = self.config.get("poll_interval", 60)  # seconds
        
        while self.running:
            current_time = time.time()
            
            if current_time - last_poll_time >= poll_interval:
                try:
                    # Fetch latest data
                    if self.market_data_source == "ccxt":
                        ohlcv = self.exchange.fetch_ohlcv(
                            self.symbol, 
                            self.interval, 
                            limit=1
                        )
                        
                        if ohlcv and len(ohlcv) > 0:
                            # Convert to dictionary
                            data = {
                                'timestamp': datetime.fromtimestamp(ohlcv[0][0] / 1000),
                                'open': float(ohlcv[0][1]),
                                'high': float(ohlcv[0][2]),
                                'low': float(ohlcv[0][3]),
                                'close': float(ohlcv[0][4]),
                                'volume': float(ohlcv[0][5]),
                                'symbol': self.symbol
                            }
                            
                            self.data_queue.put(data)
                    
                    # Update last poll time
                    last_poll_time = current_time
                    
                except Exception as e:
                    log(f"Error polling data: {str(e)}")
            
            # Sleep to avoid busy waiting
            time.sleep(1)
    
    def _process_loop(self):
        """Main processing loop for incoming data."""
        while self.running:
            try:
                # Get data from queue with timeout
                try:
                    data = self.data_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                # Process data
                features = self._process_data(data)
                
                # Make predictions if we have enough data
                if features is not None and len(features) >= self.sequence_length:
                    prediction = self._make_prediction(features)
                    
                    if prediction:
                        # Log prediction
                        self._log_prediction(prediction)
                        
                        # Update visualization
                        if self.visualize:
                            self._update_visualization()
                        
                        # Log key prediction metrics
                        horizon_str = []
                        for h in sorted(prediction['predictions'].keys()):
                            pred = prediction['predictions'][h]
                            horizon_str.append(
                                f"h{h}: {pred['mean_price']:.2f} ±{pred['std_price']:.2f} "
                                f"({pred['confidence']:.2f} conf)"
                            )
                        
                        log(f"Prediction at {prediction['timestamp']} - "
                            f"Price: {prediction['current_price']:.2f}, "
                            f"Trend: {prediction['trend_strength']:.2f}, "
                            f"{', '.join(horizon_str)}")
                
            except Exception as e:
                log(f"Error in processing loop: {str(e)}")
    
    def start(self):
        """Start real-time inference."""
        if self.running:
            log("Already running")
            return
            
        self.running = True
        log("Starting real-time inference...")
        
        # Fetch historical data to initialize
        historical_data = self._fetch_historical_data()
        if not historical_data.empty:
            self._process_data(historical_data)
        
        # Start data source
        if self.market_data_source == "file":
            # Start simulation thread
            simulation_thread = threading.Thread(target=self._simulate_data)
            simulation_thread.daemon = True
            simulation_thread.start()
            
        elif self.websocket:
            # Start WebSocket thread
            self._start_websocket()
            
        else:
            # Start polling thread
            polling_thread = threading.Thread(target=self._polling_loop)
            polling_thread.daemon = True
            polling_thread.start()
        
        # Start processing thread
        processing_thread = threading.Thread(target=self._process_loop)
        processing_thread.daemon = True
        processing_thread.start()
        
        # Start visualization if enabled
        if self.visualize:
            plt.ion()  # Turn on interactive mode
        
        log("Real-time inference started")
        
        # Keep main thread alive
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            log("Keyboard interrupt detected")
            self.stop()
    
    def stop(self):
        """Stop real-time inference."""
        log("Stopping real-time inference...")
        self.running = False
        
        # Close WebSocket if applicable
        if self.websocket:
            self.websocket.close()
        
        # Close visualization
        if self.visualize:
            plt.close('all')
        
        log("Real-time inference stopped")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Real-time inference for probabilistic trading models")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--config_path", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to save prediction logs")
    parser.add_argument("--visualize", action="store_true", help="Visualize predictions in real-time")
    parser.add_argument("--market_data_source", type=str, default="ccxt", 
                       choices=["ccxt", "binance", "coinbase", "file"], 
                       help="Source for market data")
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Trading symbol to monitor")
    parser.add_argument("--interval", type=str, default="5m", help="Time interval for data")
    
    args = parser.parse_args()
    
    # Initialize real-time inference
    inference = RealTimeInference(
        model_path=args.model_path,
        config_path=args.config_path,
        log_dir=args.log_dir,
        visualize=args.visualize,
        market_data_source=args.market_data_source,
        symbol=args.symbol,
        interval=args.interval
    )
    
    # Start inference
    inference.start()

if __name__ == "__main__":
    main() 