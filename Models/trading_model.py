"""
Trading model integration script.
This script integrates the probabilistic prediction model with the trading environment.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime

from probabilistic_model import ProbabilisticLSTMModel, ProbabilisticCNNLSTMModel
from src.environment.env_base.env_types import Position, Order, Trade

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'trading_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

class TradingModel:
    """Class for integrating probabilistic model with trading environment"""
    
    def __init__(
        self,
        model_type: str = 'lstm',  # 'lstm' or 'cnn_lstm'
        input_dim: int = 20,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        horizon_names: List[str] = ['scalping', 'short', 'medium', 'long'],
        confidence_threshold: float = 0.7,
        risk_factor: float = 0.02  # Maximum position size as fraction of account
    ):
        self.device = device
        self.horizon_names = horizon_names
        self.confidence_threshold = confidence_threshold
        self.risk_factor = risk_factor
        
        # Initialize model
        if model_type == 'lstm':
            self.model = ProbabilisticLSTMModel(
                input_dim=input_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                horizon_names=horizon_names
            )
        else:  # cnn_lstm
            self.model = ProbabilisticCNNLSTMModel(
                input_dim=input_dim,
                seq_len=100,  # Default sequence length
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                horizon_names=horizon_names
            )
        
        self.model = self.model.to(device)
        self.model.eval()  # Set to evaluation mode
        
        # Initialize hidden state
        self.hidden = None
    
    def load_checkpoint(self, path: str):
        """
        Load model checkpoint
        
        Args:
            path: Path to the checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"Loaded checkpoint from {path}")
    
    def predict(
        self,
        observation: np.ndarray,
        account_balance: float,
        current_position: Optional[Position] = None
    ) -> Tuple[Order, Dict[str, float]]:
        """
        Generate trading predictions and order
        
        Args:
            observation: Current market observation
            account_balance: Current account balance
            current_position: Current trading position
            
        Returns:
            Tuple containing:
                - Order to execute
                - Dictionary of prediction metrics
        """
        # Convert observation to tensor
        x = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            (
                value_mean, value_log_std, action_probs,
                price_means, price_stds, confidences,
                mid_means, mid_stds, trend_mean, trend_std,
                self.hidden
            ) = self.model(x, self.hidden)
        
        # Convert predictions to numpy
        action_probs = action_probs.cpu().numpy()[0]
        confidences = confidences.cpu().numpy()[0]
        price_means = price_means.cpu().numpy()[0]
        price_stds = price_stds.cpu().numpy()[0]
        trend_mean = trend_mean.cpu().numpy()[0]
        
        # Get action with highest probability
        action_idx = np.argmax(action_probs)
        action_confidence = action_probs[action_idx]
        
        # Only trade if confidence is above threshold
        if action_confidence < self.confidence_threshold:
            return None, {
                'action': 'hold',
                'confidence': action_confidence,
                'price_predictions': dict(zip(self.horizon_names, price_means)),
                'price_uncertainties': dict(zip(self.horizon_names, price_stds)),
                'trend_strength': trend_mean
            }
        
        # Calculate position size based on confidence and risk factor
        position_size = account_balance * self.risk_factor * action_confidence
        
        # Create order based on action
        if action_idx == 0:  # Buy
            order = Order(size=position_size, price=0.0)  # Price will be set by market
        elif action_idx == 2:  # Sell
            order = Order(size=-position_size, price=0.0)  # Price will be set by market
        else:  # Hold
            order = None
        
        return order, {
            'action': ['buy', 'hold', 'sell'][action_idx],
            'confidence': action_confidence,
            'price_predictions': dict(zip(self.horizon_names, price_means)),
            'price_uncertainties': dict(zip(self.horizon_names, price_stds)),
            'trend_strength': trend_mean
        }
    
    def update_position(
        self,
        current_position: Optional[Position],
        order: Optional[Order],
        executed_price: float
    ) -> Position:
        """
        Update position based on executed order
        
        Args:
            current_position: Current trading position
            order: Executed order
            executed_price: Price at which the order was executed
            
        Returns:
            Updated position
        """
        if order is None:
            return current_position
            
        if current_position is None:
            current_position = Position(size=0, entry_price=0.0)
        
        # Update position size
        new_size = current_position.size + order.size
        
        # Update entry price (weighted average)
        if new_size != 0:
            new_entry_price = (
                current_position.size * current_position.entry_price +
                order.size * executed_price
            ) / new_size
        else:
            new_entry_price = 0.0
        
        return Position(size=new_size, entry_price=new_entry_price)
    
    def get_position_metrics(
        self,
        position: Position,
        current_price: float,
        predictions: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate metrics for current position
        
        Args:
            position: Current trading position
            current_price: Current market price
            predictions: Model predictions
            
        Returns:
            Dictionary of position metrics
        """
        if position.size == 0:
            return {
                'position_value': 0.0,
                'unrealized_pnl': 0.0,
                'position_risk': 0.0,
                'expected_return': 0.0
            }
        
        # Calculate position value
        position_value = position.size * current_price
        
        # Calculate unrealized P&L
        unrealized_pnl = (current_price - position.entry_price) * position.size
        
        # Calculate position risk (using price uncertainty)
        avg_uncertainty = np.mean(list(predictions['price_uncertainties'].values()))
        position_risk = abs(position_value * avg_uncertainty)
        
        # Calculate expected return (using price predictions)
        avg_prediction = np.mean(list(predictions['price_predictions'].values()))
        expected_return = position_value * avg_prediction
        
        return {
            'position_value': position_value,
            'unrealized_pnl': unrealized_pnl,
            'position_risk': position_risk,
            'expected_return': expected_return
        }

if __name__ == "__main__":
    # Example usage
    # Create dummy observation
    seq_len = 100
    input_dim = 20
    observation = np.random.randn(seq_len, input_dim)
    
    # Initialize trading model
    model = TradingModel(
        model_type='lstm',
        input_dim=input_dim,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        confidence_threshold=0.7,
        risk_factor=0.02
    )
    
    # Generate predictions and order
    account_balance = 10000.0
    current_position = Position(size=0.1, entry_price=50000.0)
    
    order, predictions = model.predict(
        observation,
        account_balance,
        current_position
    )
    
    # Print results
    logging.info("Predictions:")
    for key, value in predictions.items():
        logging.info(f"{key}: {value}")
    
    if order is not None:
        logging.info(f"\nOrder: size={order.size}, price={order.price}")
    
    # Update position
    executed_price = 51000.0
    new_position = model.update_position(
        current_position,
        order,
        executed_price
    )
    
    logging.info(f"\nNew position: size={new_position.size}, entry_price={new_position.entry_price}")
    
    # Calculate position metrics
    metrics = model.get_position_metrics(
        new_position,
        executed_price,
        predictions
    )
    
    logging.info("\nPosition metrics:")
    for key, value in metrics.items():
        logging.info(f"{key}: {value}") 