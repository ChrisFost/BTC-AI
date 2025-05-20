"""
Probabilistic prediction model module.
This module implements neural network models with probabilistic output heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

class ProbabilisticLSTMModel(nn.Module):
    """
    LSTM-based model with probabilistic prediction heads
    """
    def __init__(
        self,
        input_dim: int = 10,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        horizon_names: List[str] = ['scalping', 'short', 'medium', 'long']
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.horizon_names = horizon_names
        self.num_horizons = len(horizon_names)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        # Value prediction (single mean and log_std for all horizons)
        self.value_head = nn.Linear(hidden_size, 2)  # Mean and log_std
        
        # Action prediction
        self.action_head = nn.Linear(hidden_size, 3)  # buy, hold, sell
        
        # Shared feature extractor for horizon-specific predictions
        self.shared_extractor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU()
        )
        
        # Probabilistic price prediction heads (output mean and std dev for each horizon)
        self.pred_heads = nn.ModuleDict({
            f"pred_{h}": nn.Linear(hidden_size // 2, 2)  # Mean and log_std
            for h in self.horizon_names
        })
        
        # Confidence heads (output probability between 0 and 1)
        self.conf_heads = nn.ModuleDict({
            f"conf_{h}": nn.Sequential(
                nn.Linear(hidden_size // 2, 1),
                nn.Sigmoid()
            )
            for h in self.horizon_names
        })
        
        # Probabilistic mid-point validation heads (output mean and std dev for each horizon)
        self.mid_heads = nn.ModuleDict({
            f"mid_{h}": nn.Linear(hidden_size // 2, 2)  # Mean and log_std
            for h in self.horizon_names
        })
        
        # Probabilistic trend strength estimator (-1 to 1)
        self.trend_strength = nn.Linear(hidden_size, 2)  # Mean and log_std
        
    def forward(
        self, 
        x: torch.Tensor, 
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple:
        """
        Forward pass through the model
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            hidden: Optional hidden state for LSTM
            
        Returns:
            Tuple containing:
                - Value prediction mean
                - Value prediction log_std
                - Action probabilities
                - Price predictions mean for each horizon
                - Price predictions std for each horizon
                - Confidence scores for each horizon
                - Mid-point predictions mean for each horizon
                - Mid-point predictions std for each horizon
                - Trend strength mean
                - Trend strength std
                - New hidden state
        """
        # Process through LSTM
        out, hidden = self.lstm(x, hidden)
        
        # Use only the last output for predictions
        last_out = out[:, -1]
        
        # Value prediction (mean and log_std)
        value_out = self.value_head(last_out)
        mean, log_std = torch.chunk(value_out, 2, dim=-1)
        
        # Action prediction
        act = F.softmax(self.action_head(last_out), dim=-1)
        
        # Shared features for horizon-specific heads
        shared_features = self.shared_extractor(last_out)
        
        # Multi-horizon predictions
        pred_means = []
        pred_stds = []
        confs = []
        mid_means = []
        mid_stds = []
        
        for h in self.horizon_names:
            # Price prediction (mean and log_std)
            pred_out = self.pred_heads[f"pred_{h}"](shared_features)
            p_mean, p_log_std = torch.chunk(pred_out, 2, dim=-1)
            pred_means.append(p_mean)
            pred_stds.append(torch.exp(p_log_std))
            
            # Confidence prediction
            conf = self.conf_heads[f"conf_{h}"](shared_features)
            confs.append(conf)
            
            # Mid-point prediction (mean and log_std)
            mid_out = self.mid_heads[f"mid_{h}"](shared_features)
            m_mean, m_log_std = torch.chunk(mid_out, 2, dim=-1)
            mid_means.append(m_mean)
            mid_stds.append(torch.exp(m_log_std))
        
        # Concatenate predictions across horizons
        pred_means = torch.cat(pred_means, dim=-1)
        pred_stds = torch.cat(pred_stds, dim=-1)
        confs = torch.cat(confs, dim=-1)
        mid_means = torch.cat(mid_means, dim=-1)
        mid_stds = torch.cat(mid_stds, dim=-1)
        
        # Trend strength prediction (mean and log_std)
        trend_out = self.trend_strength(last_out)
        trend_mean, trend_log_std = torch.chunk(trend_out, 2, dim=-1)
        trend_std = torch.exp(trend_log_std)
        
        # Apply tanh to the trend mean to constrain between -1 and 1
        trend_mean = torch.tanh(trend_mean)
        
        return (
            mean, log_std, act, 
            pred_means, pred_stds, 
            confs, 
            mid_means, mid_stds, 
            trend_mean, trend_std, 
            hidden
        )
    
    def sample_predictions(
        self, 
        pred_means: torch.Tensor, 
        pred_stds: torch.Tensor, 
        num_samples: int = 10
    ) -> torch.Tensor:
        """
        Sample from the predicted distributions
        
        Args:
            pred_means: Means of the predicted distributions
            pred_stds: Standard deviations of the predicted distributions
            num_samples: Number of samples to draw
            
        Returns:
            Samples from the predicted distributions
        """
        # Create normal distributions
        dist = torch.distributions.Normal(pred_means.unsqueeze(1), pred_stds.unsqueeze(1))
        
        # Sample from distributions
        samples = dist.sample((num_samples,))
        
        # Move sample dimension: (num_samples, batch_size, horizons) -> (batch_size, num_samples, horizons)
        samples = samples.permute(1, 0, 2)
        
        return samples
    
    def nll_loss(
        self, 
        pred_means: torch.Tensor, 
        pred_stds: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute negative log-likelihood loss for probabilistic predictions
        
        Args:
            pred_means: Means of the predicted distributions
            pred_stds: Standard deviations of the predicted distributions
            targets: Target values
            
        Returns:
            Negative log-likelihood loss
        """
        # Create normal distributions
        dist = torch.distributions.Normal(pred_means, pred_stds)
        
        # Compute log probabilities
        log_probs = dist.log_prob(targets)
        
        # Return negative log likelihood (mean over all dimensions)
        return -log_probs.mean()

class ProbabilisticCNNLSTMModel(nn.Module):
    """
    CNN-LSTM model with probabilistic prediction heads
    """
    def __init__(
        self,
        input_dim: int = 10,
        seq_len: int = 100,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        horizon_names: List[str] = ['scalping', 'short', 'medium', 'long']
    ):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.horizon_names = horizon_names
        self.num_horizons = len(horizon_names)
        
        # CNN feature extractor
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Calculate output size of CNN
        cnn_output_size = seq_len // 4  # After two max pooling layers with stride 2
        
        # LSTM layers
        self.lstm = nn.LSTM(
            128,  # Output channels from CNN
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        # Value prediction (mean and log_std)
        self.value_head = nn.Linear(hidden_size, 2)
        
        # Action prediction
        self.action_head = nn.Linear(hidden_size, 3)  # buy, hold, sell
        
        # Shared feature extractor for horizon-specific predictions
        self.shared_extractor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU()
        )
        
        # Probabilistic price prediction heads
        self.pred_heads = nn.ModuleDict({
            f"pred_{h}": nn.Linear(hidden_size // 2, 2)  # Mean and log_std
            for h in self.horizon_names
        })
        
        # Confidence heads
        self.conf_heads = nn.ModuleDict({
            f"conf_{h}": nn.Sequential(
                nn.Linear(hidden_size // 2, 1),
                nn.Sigmoid()
            )
            for h in self.horizon_names
        })
        
        # Probabilistic mid-point validation heads
        self.mid_heads = nn.ModuleDict({
            f"mid_{h}": nn.Linear(hidden_size // 2, 2)  # Mean and log_std
            for h in self.horizon_names
        })
        
        # Probabilistic trend strength estimator (-1 to 1)
        self.trend_strength = nn.Linear(hidden_size, 2)  # Mean and log_std
        
    def forward(
        self, 
        x: torch.Tensor, 
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple:
        """
        Forward pass through the model
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            hidden: Optional hidden state for LSTM
            
        Returns:
            Tuple containing model outputs (see ProbabilisticLSTMModel for details)
        """
        batch_size, seq_len, feat_dim = x.size()
        
        # Reshape for CNN: (batch_size, seq_len, feat_dim) -> (batch_size, feat_dim, seq_len)
        x_cnn = x.transpose(1, 2)
        
        # Apply CNN
        cnn_out = self.conv_layers(x_cnn)
        
        # Reshape for LSTM: (batch_size, channels, seq_len) -> (batch_size, seq_len, channels)
        lstm_in = cnn_out.transpose(1, 2)
        
        # Process through LSTM
        out, hidden = self.lstm(lstm_in, hidden)
        
        # Use only the last output for predictions
        last_out = out[:, -1]
        
        # Value prediction (mean and log_std)
        value_out = self.value_head(last_out)
        mean, log_std = torch.chunk(value_out, 2, dim=-1)
        
        # Action prediction
        act = F.softmax(self.action_head(last_out), dim=-1)
        
        # Shared features for horizon-specific heads
        shared_features = self.shared_extractor(last_out)
        
        # Multi-horizon predictions
        pred_means = []
        pred_stds = []
        confs = []
        mid_means = []
        mid_stds = []
        
        for h in self.horizon_names:
            # Price prediction (mean and log_std)
            pred_out = self.pred_heads[f"pred_{h}"](shared_features)
            p_mean, p_log_std = torch.chunk(pred_out, 2, dim=-1)
            pred_means.append(p_mean)
            pred_stds.append(torch.exp(p_log_std))
            
            # Confidence prediction
            conf = self.conf_heads[f"conf_{h}"](shared_features)
            confs.append(conf)
            
            # Mid-point prediction (mean and log_std)
            mid_out = self.mid_heads[f"mid_{h}"](shared_features)
            m_mean, m_log_std = torch.chunk(mid_out, 2, dim=-1)
            mid_means.append(m_mean)
            mid_stds.append(torch.exp(m_log_std))
        
        # Concatenate predictions across horizons
        pred_means = torch.cat(pred_means, dim=-1)
        pred_stds = torch.cat(pred_stds, dim=-1)
        confs = torch.cat(confs, dim=-1)
        mid_means = torch.cat(mid_means, dim=-1)
        mid_stds = torch.cat(mid_stds, dim=-1)
        
        # Trend strength prediction (mean and log_std)
        trend_out = self.trend_strength(last_out)
        trend_mean, trend_log_std = torch.chunk(trend_out, 2, dim=-1)
        trend_std = torch.exp(trend_log_std)
        
        # Apply tanh to the trend mean to constrain between -1 and 1
        trend_mean = torch.tanh(trend_mean)
        
        return (
            mean, log_std, act, 
            pred_means, pred_stds, 
            confs, 
            mid_means, mid_stds, 
            trend_mean, trend_std, 
            hidden
        )
    
    def nll_loss(
        self, 
        pred_means: torch.Tensor, 
        pred_stds: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute negative log-likelihood loss for probabilistic predictions
        
        Args:
            pred_means: Means of the predicted distributions
            pred_stds: Standard deviations of the predicted distributions
            targets: Target values
            
        Returns:
            Negative log-likelihood loss
        """
        # Create normal distributions
        dist = torch.distributions.Normal(pred_means, pred_stds)
        
        # Compute log probabilities
        log_probs = dist.log_prob(targets)
        
        # Return negative log likelihood (mean over all dimensions)
        return -log_probs.mean() 