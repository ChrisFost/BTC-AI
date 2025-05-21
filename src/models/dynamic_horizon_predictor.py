#!/usr/bin/env python
"""
Module containing the DynamicHorizonPredictor for probabilistic models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from typing import List, Dict, Optional


class DynamicHorizonPredictor(nn.Module):
    """
    Predicts a distribution over optimal prediction horizons and
    provides conditional outcome predictions for a requested horizon.
    """

    def __init__(self, feature_size: int, config: dict = None):
        """
        Initialize the predictor module.

        Args:
            feature_size (int): Size of the input feature vector from the base model.
            config (dict, optional): Configuration dictionary. Defaults to None.
        """
        super().__init__()
        self.config = config or {}
        self.feature_size = feature_size
        # Get device from config or default to cpu
        self.device = self.config.get("device", "cpu")

        # Configurable dimension for horizon embedding
        self.horizon_embedding_dim = self.config.get("HORIZON_EMBEDDING_DIM", 16)

        # --- Heads for Horizon Distribution Prediction ---
        self.horizon_mean_head = nn.Linear(self.feature_size, 1)
        self.horizon_log_std_head = nn.Linear(self.feature_size, 1)

        # --- Encoder for requested_horizon ---
        # Simple linear encoder for the single horizon value
        self.horizon_encoder = nn.Sequential(
            nn.Linear(1, self.horizon_embedding_dim), nn.ReLU()  # Add non-linearity
        )

        # --- Heads for Conditional Outcome Prediction ---
        # Input size combines input features and encoded horizon
        combined_feature_size = self.feature_size + self.horizon_embedding_dim

        self.outcome_mean_head = nn.Linear(combined_feature_size, 1)
        self.outcome_log_std_head = nn.Linear(combined_feature_size, 1)
        self.outcome_confidence_head = nn.Sequential(
            nn.Linear(combined_feature_size, 1),
            nn.Sigmoid(),  # Output confidence between 0 and 1
        )

        # Move layers to the specified device during initialization
        self.to(self.device)

    def forward(self, features: torch.Tensor, requested_horizon: float = None):
        """
        Forward pass.

        Args:
            features (torch.Tensor): Input features from the base model.
            requested_horizon (float, optional): Specific horizon to predict outcome for. Defaults to None.

        Returns:
            dict: Dictionary containing horizon distribution and conditional outcome predictions.
        """
        # Ensure features are on the correct device
        features = features.to(self.device)

        # --- Calculate Horizon Distribution (Always calculated) ---
        horizon_mean = self.horizon_mean_head(features)
        horizon_log_std = self.horizon_log_std_head(features)
        # Ensure std is positive and add small epsilon for stability
        horizon_std = torch.exp(horizon_log_std) + 1e-6

        # --- Calculate Conditional Outcome Prediction (If requested) ---
        outcome_mean = None
        outcome_std = None
        outcome_confidence = None

        if requested_horizon is not None:
            # Ensure requested_horizon is a tensor on the correct device
            if not isinstance(requested_horizon, torch.Tensor):
                # Assuming requested_horizon is a scalar or numpy array compatible value
                # Create a tensor matching the batch size
                batch_size = features.shape[0]
                # Use .item() if requested_horizon might be a 0-dim tensor
                horizon_value = (
                    float(requested_horizon.item())
                    if isinstance(requested_horizon, torch.Tensor)
                    and requested_horizon.ndim == 0
                    else float(requested_horizon)
                )
                horizon_tensor = torch.full(
                    (batch_size, 1),
                    horizon_value,
                    device=self.device,
                    dtype=torch.float32,
                )
            elif requested_horizon.ndim == 0:  # If it's a scalar tensor
                batch_size = features.shape[0]
                horizon_tensor = requested_horizon.expand(batch_size, 1).to(self.device)
            elif requested_horizon.ndim == 1:  # If it's a batch of scalars
                horizon_tensor = requested_horizon.unsqueeze(-1).to(self.device)
            else:  # Assume it's already [batch_size, 1]
                horizon_tensor = requested_horizon.to(self.device)

            # Encode the requested horizon
            encoded_horizon = self.horizon_encoder(horizon_tensor)

            # Combine features with encoded horizon
            combined_features = torch.cat([features, encoded_horizon], dim=-1)

            # Pass through outcome prediction heads
            outcome_mean = self.outcome_mean_head(combined_features)
            outcome_log_std = self.outcome_log_std_head(combined_features)
            outcome_std = torch.exp(outcome_log_std) + 1e-6  # Ensure positive
            outcome_confidence = self.outcome_confidence_head(combined_features)

        # --- Return Results ---
        return {
            "horizon_mean": horizon_mean,
            "horizon_std": horizon_std,
            "outcome_mean": outcome_mean,
            "outcome_std": outcome_std,
            "outcome_confidence": outcome_confidence,
        }

    def get_forecast(
        self,
        features: torch.Tensor,
        requested_horizon: int = None,
        confidence: float = 0.68,
        history_list: Optional[List[Dict]] = None,
    ) -> dict:
        """Return a simple forecast dictionary.

        Args:
            features: Input features tensor.
            requested_horizon: Desired horizon. If None, use predicted mean.
            confidence: Confidence level for interval bounds (0-1).
            history_list: Optional list to record forecast dictionaries.

        Returns:
            dict: {"mean", "low", "high", "horizon"}
        """
        outputs = self.forward(features, requested_horizon)

        # Determine horizon
        horizon_tensor = (
            outputs["horizon_mean"]
            if requested_horizon is None
            else torch.tensor([[float(requested_horizon)]], device=self.device)
        )
        horizon = int(torch.round(horizon_tensor).mean().item())

        mean = outputs.get("outcome_mean")
        std = outputs.get("outcome_std")
        if mean is None or std is None:
            forecast = {"mean": None, "low": None, "high": None, "horizon": horizon}
            if history_list is not None:
                history_list.append(forecast)
            return forecast

        z = stats.norm.ppf((1 + confidence) / 2)
        low = mean - z * std
        high = mean + z * std

        forecast = {
            "mean": mean.squeeze().item(),
            "low": low.squeeze().item(),
            "high": high.squeeze().item(),
            "horizon": horizon,
        }

        if history_list is not None:
            history_list.append(forecast)

        return forecast
