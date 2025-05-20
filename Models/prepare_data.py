"""
Data preparation script for probabilistic prediction models.
This script handles data preprocessing and preparation for training.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'data_prep_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

class DataPreparator:
    """Class for preparing data for probabilistic prediction models"""
    
    def __init__(
        self,
        seq_len: int = 100,
        horizon_names: List[str] = ['scalping', 'short', 'medium', 'long'],
        horizon_steps: List[int] = [1, 5, 15, 60]  # Number of steps for each horizon
    ):
        self.seq_len = seq_len
        self.horizon_names = horizon_names
        self.horizon_steps = horizon_steps
        self.scalers: Dict[str, StandardScaler] = {}
        
    def prepare_features(
        self,
        data: pd.DataFrame,
        feature_columns: List[str]
    ) -> np.ndarray:
        """
        Prepare feature data
        
        Args:
            data: DataFrame containing raw data
            feature_columns: List of feature column names
            
        Returns:
            Array of prepared features
        """
        # Extract features
        features = data[feature_columns].values
        
        # Scale features
        scaled_features = np.zeros_like(features)
        for i, col in enumerate(feature_columns):
            if col not in self.scalers:
                self.scalers[col] = StandardScaler()
            scaled_features[:, i] = self.scalers[col].fit_transform(features[:, i].reshape(-1, 1)).flatten()
        
        return scaled_features
    
    def prepare_targets(
        self,
        data: pd.DataFrame,
        price_column: str = 'close'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare target data
        
        Args:
            data: DataFrame containing raw data
            price_column: Name of the price column
            
        Returns:
            Tuple containing:
                - Value targets (price changes)
                - Action targets (buy/hold/sell)
                - Price targets for each horizon
                - Mid-point targets for each horizon
        """
        prices = data[price_column].values
        
        # Calculate value targets (price changes)
        value_targets = np.diff(prices) / prices[:-1]
        value_targets = np.pad(value_targets, (1, 0), 'constant')
        
        # Calculate action targets based on price changes
        action_targets = np.zeros(len(prices))
        action_targets[value_targets > 0.001] = 0  # Buy
        action_targets[value_targets < -0.001] = 2  # Sell
        action_targets[(value_targets >= -0.001) & (value_targets <= 0.001)] = 1  # Hold
        
        # Calculate price targets for each horizon
        price_targets = np.zeros((len(prices), len(self.horizon_names)))
        for i, steps in enumerate(self.horizon_steps):
            future_prices = np.roll(prices, -steps)
            future_prices[-steps:] = prices[-1]  # Pad with last price
            price_targets[:, i] = (future_prices - prices) / prices
        
        # Calculate mid-point targets (average of current and future prices)
        mid_targets = np.zeros((len(prices), len(self.horizon_names)))
        for i, steps in enumerate(self.horizon_steps):
            future_prices = np.roll(prices, -steps)
            future_prices[-steps:] = prices[-1]  # Pad with last price
            mid_targets[:, i] = ((future_prices + prices) / 2 - prices) / prices
        
        return value_targets, action_targets, price_targets, mid_targets
    
    def create_sequences(
        self,
        features: np.ndarray,
        value_targets: np.ndarray,
        action_targets: np.ndarray,
        price_targets: np.ndarray,
        mid_targets: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create sequences for training
        
        Args:
            features: Feature array
            value_targets: Value target array
            action_targets: Action target array
            price_targets: Price target array
            mid_targets: Mid-point target array
            
        Returns:
            Tuple of sequences ready for training
        """
        num_sequences = len(features) - self.seq_len + 1
        
        # Create feature sequences
        x = np.zeros((num_sequences, self.seq_len, features.shape[1]))
        for i in range(num_sequences):
            x[i] = features[i:i+self.seq_len]
        
        # Create target sequences (using the last value of each sequence)
        y_value = value_targets[self.seq_len-1:]
        y_action = action_targets[self.seq_len-1:]
        y_price = price_targets[self.seq_len-1:]
        y_mid = mid_targets[self.seq_len-1:]
        
        return x, y_value, y_action, y_price, y_mid
    
    def prepare_data(
        self,
        data: pd.DataFrame,
        feature_columns: List[str],
        price_column: str = 'close',
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[torch.Tensor, ...]:
        """
        Prepare data for training
        
        Args:
            data: DataFrame containing raw data
            feature_columns: List of feature column names
            price_column: Name of the price column
            test_size: Proportion of data to use for testing
            random_state: Random state for reproducibility
            
        Returns:
            Tuple containing:
                - Training features
                - Training value targets
                - Training action targets
                - Training price targets
                - Training mid-point targets
                - Validation features
                - Validation value targets
                - Validation action targets
                - Validation price targets
                - Validation mid-point targets
        """
        # Prepare features
        features = self.prepare_features(data, feature_columns)
        
        # Prepare targets
        value_targets, action_targets, price_targets, mid_targets = self.prepare_targets(
            data, price_column
        )
        
        # Create sequences
        x, y_value, y_action, y_price, y_mid = self.create_sequences(
            features, value_targets, action_targets, price_targets, mid_targets
        )
        
        # Split data
        indices = np.arange(len(x))
        train_indices, val_indices = train_test_split(
            indices, test_size=test_size, random_state=random_state, shuffle=False
        )
        
        # Create training data
        train_data = (
            torch.FloatTensor(x[train_indices]),
            torch.FloatTensor(y_value[train_indices]),
            torch.LongTensor(y_action[train_indices]),
            torch.FloatTensor(y_price[train_indices]),
            torch.FloatTensor(y_mid[train_indices])
        )
        
        # Create validation data
        val_data = (
            torch.FloatTensor(x[val_indices]),
            torch.FloatTensor(y_value[val_indices]),
            torch.LongTensor(y_action[val_indices]),
            torch.FloatTensor(y_price[val_indices]),
            torch.FloatTensor(y_mid[val_indices])
        )
        
        return train_data + val_data

if __name__ == "__main__":
    # Example usage
    # Create dummy data
    n_samples = 1000
    n_features = 20
    
    # Create dummy DataFrame
    data = pd.DataFrame({
        'close': np.random.randn(n_samples).cumsum(),
        **{f'feature_{i}': np.random.randn(n_samples) for i in range(n_features)}
    })
    
    # Define feature columns
    feature_columns = [f'feature_{i}' for i in range(n_features)]
    
    # Initialize data preparator
    preparator = DataPreparator(
        seq_len=100,
        horizon_names=['scalping', 'short', 'medium', 'long'],
        horizon_steps=[1, 5, 15, 60]
    )
    
    # Prepare data
    train_data, val_data = preparator.prepare_data(
        data,
        feature_columns,
        price_column='close',
        test_size=0.2
    )
    
    # Print shapes
    logging.info("Training data shapes:")
    for i, tensor in enumerate(train_data):
        logging.info(f"Tensor {i}: {tensor.shape}")
    
    logging.info("\nValidation data shapes:")
    for i, tensor in enumerate(val_data):
        logging.info(f"Tensor {i}: {tensor.shape}") 