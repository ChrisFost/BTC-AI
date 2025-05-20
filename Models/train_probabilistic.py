"""
Training script for probabilistic prediction models.
This script handles training and evaluation of the probabilistic LSTM and CNN-LSTM models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import os

from probabilistic_model import ProbabilisticLSTMModel, ProbabilisticCNNLSTMModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

class ProbabilisticTrainer:
    """Trainer class for probabilistic prediction models"""
    
    def __init__(
        self,
        model_type: str = 'lstm',  # 'lstm' or 'cnn_lstm'
        input_dim: int = 20,  # Number of input features
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        horizon_names: List[str] = ['scalping', 'short', 'medium', 'long']
    ):
        self.device = device
        self.batch_size = batch_size
        self.horizon_names = horizon_names
        
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
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Initialize loss functions
        self.value_criterion = nn.MSELoss()
        self.action_criterion = nn.CrossEntropyLoss()
        
    def train_step(
        self,
        batch: Tuple[torch.Tensor, ...]
    ) -> Dict[str, float]:
        """
        Perform a single training step
        
        Args:
            batch: Tuple containing (x, y_value, y_action, y_price, y_mid)
            
        Returns:
            Dict containing loss values
        """
        x, y_value, y_action, y_price, y_mid = [b.to(self.device) for b in batch]
        
        # Forward pass
        (
            value_mean, value_log_std, action_probs,
            price_means, price_stds, confidences,
            mid_means, mid_stds, trend_mean, trend_std,
            _
        ) = self.model(x)
        
        # Calculate losses
        # Value prediction loss (MSE)
        value_loss = self.value_criterion(value_mean, y_value)
        
        # Action prediction loss (Cross-entropy)
        action_loss = self.action_criterion(action_probs, y_action)
        
        # Price prediction loss (Negative log-likelihood)
        price_loss = self.model.nll_loss(price_means, price_stds, y_price)
        
        # Mid-point prediction loss (Negative log-likelihood)
        mid_loss = self.model.nll_loss(mid_means, mid_stds, y_mid)
        
        # Total loss (weighted sum)
        total_loss = (
            value_loss +
            action_loss +
            price_loss +
            mid_loss
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'value_loss': value_loss.item(),
            'action_loss': action_loss.item(),
            'price_loss': price_loss.item(),
            'mid_loss': mid_loss.item()
        }
    
    def evaluate(
        self,
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """
        Evaluate the model on a dataset
        
        Args:
            dataloader: DataLoader containing evaluation data
            
        Returns:
            Dict containing evaluation metrics
        """
        self.model.eval()
        total_loss = 0
        total_value_loss = 0
        total_action_loss = 0
        total_price_loss = 0
        total_mid_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                x, y_value, y_action, y_price, y_mid = [b.to(self.device) for b in batch]
                
                # Forward pass
                (
                    value_mean, value_log_std, action_probs,
                    price_means, price_stds, confidences,
                    mid_means, mid_stds, trend_mean, trend_std,
                    _
                ) = self.model(x)
                
                # Calculate losses
                value_loss = self.value_criterion(value_mean, y_value)
                action_loss = self.action_criterion(action_probs, y_action)
                price_loss = self.model.nll_loss(price_means, price_stds, y_price)
                mid_loss = self.model.nll_loss(mid_means, mid_stds, y_mid)
                
                # Total loss
                batch_loss = value_loss + action_loss + price_loss + mid_loss
                
                # Accumulate losses
                total_loss += batch_loss.item()
                total_value_loss += value_loss.item()
                total_action_loss += action_loss.item()
                total_price_loss += price_loss.item()
                total_mid_loss += mid_loss.item()
                num_batches += 1
        
        # Calculate average losses
        return {
            'total_loss': total_loss / num_batches,
            'value_loss': total_value_loss / num_batches,
            'action_loss': total_action_loss / num_batches,
            'price_loss': total_price_loss / num_batches,
            'mid_loss': total_mid_loss / num_batches
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 100,
        save_dir: str = 'checkpoints',
        save_freq: int = 5
    ):
        """
        Train the model
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: Number of training epochs
            save_dir: Directory to save model checkpoints
            save_freq: Frequency of saving checkpoints (in epochs)
        """
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            self.model.train()
            epoch_losses = []
            
            # Training loop
            for batch in train_loader:
                losses = self.train_step(batch)
                epoch_losses.append(losses)
            
            # Calculate average training losses
            avg_train_losses = {
                k: np.mean([l[k] for l in epoch_losses])
                for k in epoch_losses[0].keys()
            }
            
            # Evaluate on validation set
            val_losses = self.evaluate(val_loader)
            
            # Log metrics
            logging.info(f"Epoch {epoch+1}/{num_epochs}")
            logging.info(f"Train Loss: {avg_train_losses['total_loss']:.4f}")
            logging.info(f"Val Loss: {val_losses['total_loss']:.4f}")
            
            # Save best model
            if val_losses['total_loss'] < best_val_loss:
                best_val_loss = val_losses['total_loss']
                self.save_checkpoint(
                    os.path.join(save_dir, 'best_model.pt'),
                    epoch,
                    val_losses
                )
            
            # Save periodic checkpoint
            if (epoch + 1) % save_freq == 0:
                self.save_checkpoint(
                    os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt'),
                    epoch,
                    val_losses
                )
    
    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        val_losses: Dict[str, float]
    ):
        """
        Save a model checkpoint
        
        Args:
            path: Path to save the checkpoint
            epoch: Current epoch number
            val_losses: Validation losses
        """
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_losses': val_losses
        }, path)
        logging.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """
        Load a model checkpoint
        
        Args:
            path: Path to the checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logging.info(f"Loaded checkpoint from {path}")

def create_dataloaders(
    train_data: Tuple[torch.Tensor, ...],
    val_data: Tuple[torch.Tensor, ...],
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders for training and validation data
    
    Args:
        train_data: Tuple of training tensors (x, y_value, y_action, y_price, y_mid)
        val_data: Tuple of validation tensors (x, y_value, y_action, y_price, y_mid)
        batch_size: Batch size for the DataLoaders
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = TensorDataset(*train_data)
    val_dataset = TensorDataset(*val_data)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Example usage
    # Create dummy data for testing
    batch_size = 32
    seq_len = 100
    input_dim = 20
    num_horizons = 4
    
    # Create dummy tensors
    x = torch.randn(batch_size, seq_len, input_dim)
    y_value = torch.randn(batch_size, 1)
    y_action = torch.randint(0, 3, (batch_size,))
    y_price = torch.randn(batch_size, num_horizons)
    y_mid = torch.randn(batch_size, num_horizons)
    
    # Create train and validation data
    train_data = (x, y_value, y_action, y_price, y_mid)
    val_data = (x, y_value, y_action, y_price, y_mid)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_data,
        val_data,
        batch_size=batch_size
    )
    
    # Initialize trainer
    trainer = ProbabilisticTrainer(
        model_type='lstm',
        input_dim=input_dim,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        learning_rate=0.001,
        batch_size=batch_size,
        horizon_names=['scalping', 'short', 'medium', 'long']
    )
    
    # Train the model
    trainer.train(
        train_loader,
        val_loader,
        num_epochs=10,
        save_dir='checkpoints',
        save_freq=2
    ) 