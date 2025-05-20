#!/usr/bin/env python
"""
Tensor-based Trading Environment optimized for GPU acceleration.

This module provides a tensor-optimized implementation of the BaseTradingEnv
for improved performance on GPUs.
"""

import time
import logging
import numpy as np
from collections import deque
import pandas as pd
import random
import torch
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional, Any, Union
import importlib

# Import interfaces and components using dynamic imports
try:
    # Import env_interfaces components
    env_interfaces_module = importlib.import_module("src.environment.env_interfaces")
    TradingEnvInterface = env_interfaces_module.TradingEnvInterface
    Position = env_interfaces_module.Position
    Order = env_interfaces_module.Order
    Trade = env_interfaces_module.Trade
    validate_action = env_interfaces_module.validate_action
    format_agent_action = env_interfaces_module.format_agent_action
    EnvRegistry = env_interfaces_module.EnvRegistry
    
    # Import env_utils components
    env_utils_module = importlib.import_module("src.environment.env_utils")
    log = env_utils_module.log
    calculate_sharpe_ratio = env_utils_module.calculate_sharpe_ratio
    calculate_sortino_ratio = env_utils_module.calculate_sortino_ratio
    calculate_drawdown = env_utils_module.calculate_drawdown
    
    # Import risk and reward components
    env_risk_module = importlib.import_module("src.environment.env_risk")
    create_risk_manager = env_risk_module.create_risk_manager
    
    env_rewards_module = importlib.import_module("src.environment.env_rewards")
    create_reward_system = env_rewards_module.create_reward_system
except ImportError as e:
    print(f"Error during dynamic import: {e}")

class TensorTradingEnv(TradingEnvInterface):
    """
    Tensor-based trading environment for faster computation.
    
    Implements the trading environment interface using PyTorch tensors
    for vectorized operations, significantly improving performance.
    """
    
    def __init__(self, df, window_size, initial_capital, max_positions, bucket, config, device="cpu"):
        """
        Initialize tensor trading environment.
        
        Args:
            df (pandas.DataFrame): DataFrame with OHLCV data and features.
            window_size (int): Number of time steps to use as observation.
            initial_capital (float): Starting capital in USD.
            max_positions (int): Maximum number of positions to hold simultaneously.
            bucket (str): Trading timeframe bucket (Scalping, Short, Medium, Long).
            config (dict): Configuration parameters.
            device (str): Device to use for tensor operations ("cpu" or "cuda").
        """
        # Initialize standard variables
        self.df = df
        self.window_size = window_size
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.bucket = bucket
        self.config = config
        self.device = device
        
        # Tensor-specific attributes
        self.tensor_cache = {}
        self.feature_indices = {}
        
        # History tracking
        self.liquidity_history = []
        self.spread_history = []
        
        # Available columns for features
        if df is not None and isinstance(df, pd.DataFrame):
            self.available_cols = list(df.columns)
        else:
            self.available_cols = ['close', 'high', 'low', 'volume']
        
        # Cache for expensive tensor computations
        self.volume_profile_last_update = -1
        self.volume_profile = None
        self.liquidity_zones = None
        self.pattern_recognition_last_update = -1
        self.detected_patterns = {}
        self.key_levels = {}
        
        # Initialize trading variables
        self.positions = []
        self.closed_trades = []
        self.profits = []
        self.losses = []
        self.returns = []
        self.rolling_trades = deque(maxlen=30*288)  # 30 days of 5-min bars
        self.pending_orders = []
        self.current_step = self.window_size
        self.capital = initial_capital
        self.done = False
        
        # Initialize risk management and reward system
        self.risk_manager = create_risk_manager(bucket, config)
        self.reward_system = create_reward_system(bucket, config)
        
        # Precompute tensor data for fast access
        self._precompute_tensors()
        
        # Set up advanced tensor features
        self._initialize_tensor_features()
        
        log(f"Initialized TensorTradingEnv on device '{device}' with {len(self.available_cols)} features")
    
    def _precompute_tensors(self):
        """Precompute tensors from DataFrame for faster access during simulation."""
        if self.df is None:
            log("Warning: No DataFrame provided, cannot precompute tensors")
            self.feature_matrix = None
            return
        
        start_time = time.time()
        
        try:
            # Extract OHLCV data as tensors
            self.close_tensor = torch.tensor(self.df['close'].values, dtype=torch.float32, device=self.device)
            self.high_tensor = torch.tensor(self.df['high'].values, dtype=torch.float32, device=self.device)
            self.low_tensor = torch.tensor(self.df['low'].values, dtype=torch.float32, device=self.device)
            self.volume_tensor = torch.tensor(self.df['volume'].values, dtype=torch.float32, device=self.device)
            
            # Try to get open prices if available
            if 'open' in self.df.columns:
                self.open_tensor = torch.tensor(self.df['open'].values, dtype=torch.float32, device=self.device)
            else:
                # Fallback: synthesize open prices as midpoint between previous close and current high/low
                prev_close = self.close_tensor[:-1]
                padded_prev_close = torch.cat([self.close_tensor[0].unsqueeze(0), prev_close])
                self.open_tensor = (padded_prev_close + self.high_tensor + self.low_tensor) / 3
                
            # Build combined OHLCV tensor for easier access
            self.ohlcv_tensor = torch.stack([
                self.close_tensor,
                self.open_tensor,
                self.high_tensor,
                self.low_tensor,
                self.volume_tensor
            ], dim=1)
            
            # Build additional feature tensors from available columns
            self.feature_tensors = {}
            
            # Create a mapping of feature name to index for fast lookup
            self.feature_indices = {name: i for i, name in enumerate(self.available_cols)}
            
            # Convert each feature column to tensor
            numeric_cols = []
            for col in self.available_cols:
                if col in ['close', 'high', 'low', 'volume']:
                    # Already precomputed as part of OHLCV
                    numeric_cols.append(col)
                    continue
                
                if col in self.df.columns:
                    try:
                        # Skip datetime columns and non-numeric data
                        if pd.api.types.is_numeric_dtype(self.df[col]):
                            self.feature_tensors[col] = torch.tensor(
                                self.df[col].values, 
                                dtype=torch.float32, 
                                device=self.device
                            )
                            numeric_cols.append(col)
                    except Exception as e:
                        log(f"Could not convert column {col} to tensor: {e}")
            
            # Create a combined feature tensor for all available numeric features
            feature_list = []
            for col in numeric_cols:
                if col in ['close', 'high', 'low', 'volume']:
                    # Use the values from the OHLCV tensor
                    col_idx = {'close': 0, 'high': 2, 'low': 3, 'volume': 4}.get(col, 0)
                    feature_list.append(self.ohlcv_tensor[:, col_idx])
                elif col in self.feature_tensors:
                    feature_list.append(self.feature_tensors[col])
                else:
                    # If feature is not available, use zeros
                    feature_list.append(torch.zeros_like(self.close_tensor))
            
            # Stack features into a combined tensor (time steps, features)
            if feature_list:
                self.feature_matrix = torch.stack(feature_list, dim=1)
                log(f"Precomputed tensors with shape {self.feature_matrix.shape} in {time.time() - start_time:.2f}s")
            else:
                # Handle case with no valid features
                self.feature_matrix = None
                log(f"No valid features to precompute tensors")
            
        except Exception as e:
            log(f"Error precomputing tensors: {e}")
            
            # Initialize minimal tensors if precomputation fails
            if not hasattr(self, 'close_tensor'):
                # Create default tensors with at least one element
                self.close_tensor = torch.ones(1, dtype=torch.float32, device=self.device)
                self.open_tensor = torch.ones(1, dtype=torch.float32, device=self.device)
                self.high_tensor = torch.ones(1, dtype=torch.float32, device=self.device) * 1.01
                self.low_tensor = torch.ones(1, dtype=torch.float32, device=self.device) * 0.99
                self.volume_tensor = torch.ones(1, dtype=torch.float32, device=self.device)
                
                # Combined OHLCV tensor
                self.ohlcv_tensor = torch.stack([
                    self.close_tensor,
                    self.open_tensor,
                    self.high_tensor,
                    self.low_tensor,
                    self.volume_tensor
                ], dim=1)
                
                # Create an empty feature matrix
                self.feature_matrix = torch.zeros((1, len(self.available_cols)), dtype=torch.float32, device=self.device)
    
    def _initialize_tensor_features(self):
        """Initialize additional tensor features for advanced analysis."""
        # Set up containers for derived features
        self.tensor_cache["hourly_volatility"] = deque(maxlen=24)  # Last 24 hours
        self.tensor_cache["daily_volatility"] = deque(maxlen=30)   # Last 30 days
        self.tensor_cache["volume_profile"] = None                 # Current volume profile
        self.tensor_cache["liquidity_zones"] = None                # Current liquidity zones
        self.tensor_cache["support_resistance"] = None             # Current support/resistance levels
        
        # Compute initial market condition metrics if data is available
        if len(self.close_tensor) > self.window_size:
            self._update_market_condition_metrics(self.window_size)
    
    def _update_market_condition_metrics(self, step_idx):
        """
        Update market condition metrics at the given step index.
        
        Args:
            step_idx (int): Current step index.
        """
        # Minimum lookback window
        lookback = min(288, step_idx)  # Max 1 day (288 5-min bars)
        
        # Skip if not enough data
        if step_idx < 12 or step_idx >= len(self.close_tensor):
            return
        
        # Calculate hourly volatility (12 bars = 1 hour with 5-min data)
        if step_idx >= 12:
            hourly_prices = self.close_tensor[step_idx-12:step_idx]
            hourly_returns = torch.log(hourly_prices[1:] / hourly_prices[:-1])
            hourly_vol = torch.std(hourly_returns).item()
            self.tensor_cache["hourly_volatility"].append(hourly_vol)
        
        # Calculate daily volatility (288 bars = 1 day with 5-min data)
        if step_idx >= 288:
            daily_prices = self.close_tensor[step_idx-288:step_idx]
            daily_returns = torch.log(daily_prices[1:] / daily_prices[:-1])
            daily_vol = torch.std(daily_returns).item()
            self.tensor_cache["daily_volatility"].append(daily_vol)
        
        # Estimate bid-ask spread based on recent volatility
        if len(self.tensor_cache["hourly_volatility"]) > 0:
            # Bid-ask spread often correlates with short-term volatility
            avg_hourly_vol = sum(self.tensor_cache["hourly_volatility"]) / len(self.tensor_cache["hourly_volatility"])
            estimated_spread = max(0.0001, min(0.001, avg_hourly_vol * 0.1))
            self.tensor_cache["bid_ask_spread"] = estimated_spread
            self.spread_history.append(estimated_spread)
        
        # Estimate market liquidity based on volume and volatility
        try:
            # Recent volume relative to longer-term average
            recent_vol = self.volume_tensor[max(0, step_idx-12):step_idx].mean()
            if step_idx >= 144:  # Compare to previous 12 hours
                avg_vol = self.volume_tensor[max(0, step_idx-144):step_idx-12].mean()
                vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1.0
            else:
                vol_ratio = 1.0
            
            # Volatility component (higher volatility often means lower liquidity)
            vol_component = 1.0
            if len(self.tensor_cache["hourly_volatility"]) > 0:
                vol_component = 1.0 / (1.0 + avg_hourly_vol * 10)
            
            # Combined liquidity score (0-1, higher is more liquid)
            tensor_value = torch.tensor([vol_ratio * vol_component * 2.0 - 0.5], device=self.device)
            liquidity_score = torch.sigmoid(tensor_value).item()
            
            self.tensor_cache["market_liquidity"] = liquidity_score
            self.liquidity_history.append(liquidity_score)
            
        except Exception as e:
            log(f"Error updating market metrics: {e}")
    
    def _tensor_get_observation(self):
        """
        Get observation using tensor operations for improved performance.
        
        Returns:
            torch.Tensor: Standardized observation tensor.
        """
        if self.feature_matrix is None or self.current_step < self.window_size:
            return None
            
        # Get window of data
        start_idx = self.current_step - self.window_size
        end_idx = self.current_step
        
        # Make sure indices are valid
        if start_idx < 0 or end_idx >= self.feature_matrix.shape[0]:
            return None
            
        # Extract observation using tensor slicing (much faster than pandas)
        raw_obs = self.feature_matrix[start_idx:end_idx]
        
        # Standardize observation
        return self._tensor_standardize_observation(raw_obs)
    
    def _tensor_standardize_observation(self, obs):
        """
        Normalize observation using tensor operations.
        
        Args:
            obs (torch.Tensor): Raw observation tensor.
            
        Returns:
            torch.Tensor: Standardized observation tensor.
        """
        if obs is None or obs.shape[0] == 0:
            return None
            
        # Fast tensor-based standardization
        mean = torch.mean(obs, dim=0, keepdim=True)
        std = torch.std(obs, dim=0, keepdim=True) + 1e-8
        return (obs - mean) / std
    
    def reset(self):
        """
        Reset the environment to initial state for new episode.
        
        Returns:
            torch.Tensor: Initial observation.
        """
        # Reset core state variables (from base class)
        self.current_step = self.window_size
        self.capital = self.initial_capital
        self.positions = []
        self.closed_trades = []
        self.profits = []
        self.losses = []
        self.returns = []
        self.rolling_trades.clear()
        self.pending_orders.clear()
        
        # Reset tensor-specific variables
        self.tensor_cache.clear()
        self.liquidity_history = []
        self.spread_history = []
        self.volume_profile_last_update = -1
        self.volume_profile = None
        self.liquidity_zones = None
        self.pattern_recognition_last_update = -1
        self.detected_patterns = {}
        self.key_levels = {}
        
        # Reinitialize tensor features
        self._initialize_tensor_features()
        
        self.done = False
        
        # Get tensor-based observation
        return self._tensor_get_observation()
    
    def _get_observation(self):
        """
        Override base _get_observation with tensor-optimized version.
        
        Returns:
            torch.Tensor: Standardized observation tensor.
        """
        return self._tensor_get_observation()
    
    def _calculate_portfolio_risk(self):
        """
        Calculate portfolio risk metrics using tensor operations for performance.
        
        Returns:
            dict: Risk metrics dictionary.
        """
        # Get current price from tensor
        if self.current_step < len(self.close_tensor):
            current_price = self.close_tensor[self.current_step].item()
        else:
            current_price = 0.0
            
            # Fallback to last position price if no current price
            if len(self.positions) > 0:
                # Try to extract entry price based on position type
                if isinstance(self.positions[-1], Position):
                    current_price = self.positions[-1].entry_price
                elif isinstance(self.positions[-1], dict) and 'entry_price' in self.positions[-1]:
                    current_price = self.positions[-1]["entry_price"]
        
        # Delegate to risk manager for calculation
        return self.risk_manager.calculate_risk_metrics(
            self.positions, 
            self.capital,
            current_price,
            self.returns,
            self.liquidity_history
        )
    
    def _compute_tensor_slippage(self, order_size, daily_volume):
        """
        Calculate price slippage using tensor operations.
        
        Args:
            order_size (float): Size of order in USD.
            daily_volume (float): Daily trading volume in USD.
            
        Returns:
            float: Estimated slippage as a fraction.
        """
        # Slippage increases nonlinearly with order size relative to daily volume
        volume_ratio = torch.tensor(order_size / max(daily_volume, 1e-8), device=self.device)
        base_slippage = 0.0001 + 0.1 * (volume_ratio ** 2)
        
        # Scale slippage based on market liquidity
        if self.liquidity_history:
            liquidity = torch.tensor(self.liquidity_history[-1], device=self.device)
            # Less liquidity = more slippage
            liquidity_factor = 1.5 - liquidity
            base_slippage = base_slippage * liquidity_factor
        
        # Cap at reasonable values
        return min(0.025, max(0.0001, base_slippage.item()))  # 0.01% to 2.5% slippage
    
    def _tensor_estimate_market_impact(self, size_btc, price, daily_volume):
        """
        Estimate market impact of a trade using tensor operations.
        
        Args:
            size_btc (float): Size of order in BTC.
            price (float): Current price.
            daily_volume (float): Daily trading volume in USD.
            
        Returns:
            tuple: (impact_percentage, impact_score) - Market impact estimates.
        """
        # Convert order size to USD
        order_size_usd = size_btc * price
        
        # Calculate order size as percentage of daily volume
        volume_percentage = torch.tensor(
            float(order_size_usd / max(daily_volume, 1e-10)),
            dtype=torch.float32,
            device=self.device
        )
        
        # Get recent volatility using tensor operations
        lookback = min(50, self.current_step)
        if lookback > 1 and self.current_step >= lookback:
            recent_prices = self.close_tensor[self.current_step-lookback:self.current_step]
            returns = torch.log(recent_prices[1:] / recent_prices[:-1])
            volatility = torch.std(returns).item()
        else:
            volatility = 0.01  # Default value
        
        # Get liquidity factor
        liquidity = 0.5  # Default mid-level liquidity
        if self.liquidity_history:
            liquidity = self.liquidity_history[-1]
        
        # Calculate impact using square-root formula
        # Impact = k * sigma * sqrt(size / ADV)
        k = 0.5 * (1.0 - 0.5 * liquidity)  # Adjust constant based on liquidity
        
        if volume_percentage > 0:
            impact_pct = k * volatility * torch.sqrt(volume_percentage).item()
        else:
            impact_pct = 0.0
        
        # Cap at reasonable limits
        impact_pct = min(0.1, max(0.0, impact_pct))  # Cap at 10% max impact
        
        # Calculate impact score (0-1)
        if volume_percentage < 0.01:
            # Linear scaling for small orders
            impact_score = volume_percentage.item() * 10
        else:
            # Non-linear scaling for larger orders
            impact_score = min(1.0, 0.1 + 0.9 * (volume_percentage.item() - 0.01) / 0.09)
        
        return impact_pct, impact_score


    def _process_tensor_pending_orders(self):
            """
            Process pending orders using tensor operations for improved performance.
        
            Returns:
                list: Completed orders.
            """
            if self.current_step >= len(self.close_tensor):
                return []
            
            current_step = self.current_step
            completed_orders = []
        
            # Get current market data from tensors
            price = self.close_tensor[current_step].item()
            daily_volume = (self.volume_tensor[current_step] * 288).item()  # 288 5-min bars per day
        
            # Get market conditions for order fill probability
            liquidity = 0.5  # Default
            if self.liquidity_history:
                liquidity = self.liquidity_history[-1]
            
            # Calculate bid-ask spread from tensor cache or recalculate
            if "bid_ask_spread" in self.tensor_cache:
                bid_ask_spread = self.tensor_cache["bid_ask_spread"]
            else:
                # Estimated based on recent price action
                if self.current_step >= 12:  # Need at least 1 hour of data
                    hourly_prices = self.close_tensor[self.current_step-12:self.current_step]
                    hourly_returns = torch.log(hourly_prices[1:] / hourly_prices[:-1])
                    hourly_vol = torch.std(hourly_returns).item()
                    bid_ask_spread = max(0.0001, min(0.001, hourly_vol * 0.1))
                    self.tensor_cache["bid_ask_spread"] = bid_ask_spread
                else:
                    bid_ask_spread = 0.0002  # Default spread (0.02%)
        
            # Process each pending order
            for order in self.pending_orders[:]:
                # Calculate fill probability
                time_elapsed = current_step - order.entry_step
                time_factor = min(1.0, time_elapsed / order.timeout) if order.timeout > 0 else 1.0
            
                # Adjust fill probability based on market conditions
                # 1. Higher liquidity increases fill probability
                liquidity_factor = 0.3 * liquidity
            
                # 2. Lower bid-ask spread increases fill probability
                spread_factor = 0.2 * (1.0 - min(1.0, bid_ask_spread * 2000))  # Normalize spread impact
            
                # 3. Base fill probability from time factor
                base_probability = 0.1 + 0.6 * time_factor
            
                # Combine factors
                fill_probability = min(1.0, base_probability + liquidity_factor + spread_factor)
            
                # Determine if order fills now
                should_fill = (random.random() < fill_probability) or (time_elapsed >= order.timeout)
            
                if should_fill:
                    # Apply current risk-adjusted size limits for buy orders
                    original_size_btc = order.size_btc
                
                    if order.direction > 0:  # Buy order
                        risk_metrics = self._calculate_portfolio_risk()
                        max_size_btc = self.risk_manager.calculate_risk_adjusted_size(
                            price, daily_volume, order.direction, risk_metrics, len(self.positions)
                        )
                        size_btc = min(original_size_btc, max_size_btc)
                    else:  # Sell order
                        size_btc = original_size_btc
                
                    # Calculate execution price with slippage
                    slippage = self._compute_tensor_slippage(size_btc * price, daily_volume)
                
                    # Direction-based price adjustment
                    if order.direction > 0:  # Buy
                        effective_price = price * (1 + slippage)
                    else:  # Sell
                        effective_price = price * (1 - slippage)
                
                    # Calculate trade value and fee
                    notional = size_btc * effective_price
                    self._update_rolling_volume(current_step, notional)
                    fee = notional * self._get_fee_rate(self._get_rolling_volume())
                
                    # Process buy or sell
                    if order.direction > 0:  # Buy
                        if len(self.positions) < self.max_positions:
                            # Create new position
                            new_position = Position(size_btc, effective_price, current_step)
                            self.positions.append(new_position)
                            self.capital -= min((size_btc * effective_price) + fee, self.capital)
                    else:  # Sell
                        # Get reference price for profit calculation
                        reference_price = order.reference_price if hasattr(order, 'reference_price') and order.reference_price is not None else effective_price
                        reference_step = order.reference_step if hasattr(order, 'reference_step') and order.reference_step is not None else current_step
                    
                        # Calculate profit (or loss)
                        profit = size_btc * (effective_price - reference_price) - fee
                        self.capital += profit
                    
                        # Record trade statistics
                        percentage_gain = (effective_price / reference_price - 1) * 100
                        hold_time = current_step - reference_step
                    
                        trade = Trade(profit, percentage_gain, hold_time, reference_step)
                        self.closed_trades.append(trade)
                    
                        if profit > 0:
                            self.profits.append(profit)
                        else:
                            self.losses.append(abs(profit))
                    
                        # Add to returns history for risk calculations
                        self.returns.append(profit / self.initial_capital)
                
                    # Add to completed orders and remove from pending
                    completed_orders.append(order)
                    self.pending_orders.remove(order)
        
            return completed_orders
    
        # pylint: disable=method-not-used
        # noqa: F811
    def _handle_tensor_buy(self, direction, fraction, price, daily_volume, risk_metrics):
            """
            Handle buy actions using tensor operations.
        
            Args:
                direction (float): Direction component of action (positive for buy).
                fraction (float): Fraction of capital to use.
                price (float): Current price.
                daily_volume (float): Daily trading volume.
                risk_metrics (dict): Current risk metrics.
            
            Returns:
                float: Reward adjustment from the action.
            """
            reward_adjustment = 0.0
        
            # Calculate maximum position size based on risk limits
            max_size_btc = self.risk_manager.calculate_risk_adjusted_size(
                price, daily_volume, direction, risk_metrics, len(self.positions)
            )
        
            # Calculate USD amount to use
            usd_amount = self.capital * fraction
        
            if usd_amount > 1e-8:
                # Calculate potential market impact before executing
                desired_size_btc = usd_amount / price
                impact_pct, impact_score = self._tensor_estimate_market_impact(desired_size_btc, price, daily_volume)
            
                # Apply market impact penalty to reward
                market_impact_penalty = -impact_score**2 * 0.1 * usd_amount if impact_score > 0.1 else 0.0
                reward_adjustment += market_impact_penalty
            
                # Calculate total slippage including impact
                base_slippage = self._compute_tensor_slippage(usd_amount, daily_volume)
                total_slippage = base_slippage + impact_pct
                effective_price = price * (1 + total_slippage)
            
                # Recalculate size after slippage
                actual_size_btc = usd_amount / effective_price
            
                # Apply size limit
                size_btc = torch.tensor(min(actual_size_btc, max_size_btc), device=self.device).item()
                actual_usd = size_btc * effective_price
            
                # Apply penalty if size was limited significantly
                if size_btc < desired_size_btc * 0.9:  # At least 10% reduction
                    limitation_ratio = size_btc / desired_size_btc
                    size_limitation_penalty = (1.0 - limitation_ratio) * 0.02 * usd_amount
                    reward_adjustment -= size_limitation_penalty
            
                # Check if order is too large for immediate fill based on liquidity
                current_liquidity = self.liquidity_history[-1] if self.liquidity_history else 0.5
                max_fill_ratio = 0.05 + 0.15 * current_liquidity  # 5-20% of daily volume
                max_fill = max_fill_ratio * daily_volume / effective_price
            
                if size_btc > max_fill:
                    # Split into immediate fill and pending order
                    filled_size = min(size_btc, max_fill)
                    pending_size = size_btc - filled_size
                
                    # Process immediate fill portion
                    immediate_notional = filled_size * effective_price
                    self._update_rolling_volume(self.current_step, immediate_notional)
                    immediate_fee = immediate_notional * self._get_fee_rate(self._get_rolling_volume())
                
                    if filled_size > 0:
                        # Create position for filled portion
                        position = Position(filled_size, effective_price, self.current_step)
                        self.positions.append(position)
                        self.capital -= min((filled_size * effective_price) + immediate_fee, self.capital)
                
                    # Queue the remainder as a pending order
                    if pending_size > 0:
                        order = Order(
                            size_btc=pending_size,
                            direction=1.0,
                            entry_step=self.current_step,
                            timeout=10  # 10 time steps before cancellation
                        )
                        self.pending_orders.append(order)
                else:
                    # Full immediate fill
                    if size_btc > 0:
                        self._update_rolling_volume(self.current_step, actual_usd)
                        fee = actual_usd * self._get_fee_rate(self._get_rolling_volume())
                    
                        # Create position
                        position = Position(size_btc, effective_price, self.current_step)
                        self.positions.append(position)
                        self.capital -= min((size_btc * effective_price) + fee, self.capital)
        
            return reward_adjustment
    
    def _handle_tensor_sell(self, direction, fraction, price, daily_volume, risk_metrics):
            """
            Handle sell actions using tensor operations.
        
            Args:
                direction (float): Direction component of action (negative for sell).
                fraction (float): Fraction of positions to sell.
                price (float): Current price.
                daily_volume (float): Daily trading volume.
                risk_metrics (dict): Current risk metrics.
            
            Returns:
                float: Reward adjustment from the action.
            """
            reward_adjustment = 0.0
        
            # Calculate total BTC to sell across all positions
            total_btc = 0.0
            for position in self.positions:
                if isinstance(position, Position):
                    total_btc += position.size_btc
                elif isinstance(position, dict) and 'size_btc' in position:
                    total_btc += position["size_btc"]
        
            # Early return if no positions
            if total_btc <= 1e-8:
                return 0.0
        
            # Calculate amount to sell
            btc_to_sell = total_btc * fraction
        
            if btc_to_sell > 1e-8:
                # Calculate market impact and slippage
                impact_pct, impact_score = self._tensor_estimate_market_impact(btc_to_sell, price, daily_volume)
                base_slippage = self._compute_tensor_slippage(btc_to_sell * price, daily_volume)
            
                # Total slippage includes market impact
                total_slippage = base_slippage + impact_pct
                effective_price = price * (1 - total_slippage)  # Negative for sell
            
                # Apply market impact penalty to reward
                if impact_score > 0.1:
                    impact_penalty = -impact_score**2 * 0.1 * (btc_to_sell * price)
                    reward_adjustment += impact_penalty
            
                # Process positions in reverse order of entry (LIFO)
                # This is often more tax-efficient and better for backtesting
                sorted_positions = sorted(
                    [(i, p) for i, p in enumerate(self.positions)],
                    key=lambda x: x[1].entry_step if isinstance(x[1], Position) else x[1]["entry_step"],
                    reverse=True
                )
            
                positions_to_remove = []
                remaining_to_sell = btc_to_sell
                raw_profit = 0.0
                trade_details = []
            
                # Sell from each position until target amount is reached
                for idx, position in sorted_positions:
                    if remaining_to_sell <= 1e-8:
                        break
                
                    # Extract position details based on type
                    if isinstance(position, Position):
                        position_size = position.size_btc
                        entry_price = position.entry_price
                        entry_step = position.entry_step
                    else:  # Dict format for backward compatibility
                        position_size = position["size_btc"]
                        entry_price = position["entry_price"]
                        entry_step = position["entry_step"]
                
                    # Calculate size to sell from this position
                    size_from_position = min(position_size, remaining_to_sell)
                    remaining_to_sell -= size_from_position
                
                    # Calculate profit/loss for this portion
                    position_pnl = size_from_position * (effective_price - entry_price)
                    raw_profit += position_pnl
                
                    # Update position size
                    if isinstance(position, Position):
                        position.size_btc -= size_from_position
                        remaining_size = position.size_btc
                    else:  # Dict format
                        position["size_btc"] -= size_from_position
                        remaining_size = position["size_btc"]
                
                    # Record trade details for completed positions
                    if remaining_size <= 1e-8:
                        positions_to_remove.append(idx)
                    
                        # Calculate percentage gain and hold time
                        percentage_gain = (effective_price / entry_price - 1) * 100
                        hold_time = self.current_step - entry_step
                    
                        # Store details for later trade creation
                        trade_details.append((position_pnl, percentage_gain, hold_time, entry_step))
            
                # Remove closed positions (in reverse order to maintain indices)
                for idx in sorted(positions_to_remove, reverse=True):
                    self.positions.pop(idx)
            
                # Calculate trade value and fee
                sold_btc = btc_to_sell - remaining_to_sell
                notional_value = sold_btc * effective_price
            
                # Update rolling volume for fee calculation
                self._update_rolling_volume(self.current_step, notional_value)
                fee = notional_value * self._get_fee_rate(self._get_rolling_volume())
            
                # Finalize profit and update capital
                net_profit = raw_profit - fee
                self.capital += net_profit
            
                # Add to return history for risk calculations
                self.returns.append(net_profit / self.initial_capital)
            
                # Create trade records for closed positions
                for pnl, gain, duration, entry_time in trade_details:
                    trade = Trade(pnl, gain, duration, entry_time)
                    self.closed_trades.append(trade)
                
                    # Track profits and losses separately
                    if pnl > 0:
                        self.profits.append(pnl)
                    else:
                        self.losses.append(abs(pnl))
            
                # Add profit to reward adjustment
                reward_adjustment += net_profit
        
            return reward_adjustment
    
    def _calculate_reward_from_trade_outcomes(self):
            """
            Calculate reward adjustment based on trade outcomes.
        
            Returns:
                float: Reward adjustment based on trade outcomes.
            """
            # Skip if no closed trades
            if not self.closed_trades:
                return 0.0
            
            # Calculate profit metrics
            reward_adjustment = 0.0
            net_profit = 0.0
        
            # Process recent trades (those that haven't been processed yet)
            for trade in self.closed_trades:
                if not hasattr(trade, 'processed') or not trade.processed:
                    # Mark as processed to avoid double-counting
                    trade.processed = True
                
                    # Calculate profit contribution
                    if trade.pnl > 0:
                        net_profit += trade.pnl
                    else:
                        net_profit += trade.pnl
                    
                        # Add to losses for risk tracking
                        self.losses.append(abs(trade.pnl))
            
                # Add profit to reward adjustment
                reward_adjustment += net_profit
        
            return reward_adjustment
    
    def _get_rolling_volume(self):
            """Get total trading volume in the past 30 days"""
            return sum(trade[1] for trade in self.rolling_trades)
    
    def _update_rolling_volume(self, timestamp, volume):
        """
        Update the rolling 30-day volume with a new trade.
        
        Args:
            timestamp (int): Timestamp of the trade (step index).
            volume (float): Volume of the trade in USD.
        """
        # Add new trade
        self.rolling_trades.append((timestamp, volume))
        
        # Remove trades older than 30 days (assuming 288 5-min bars per day)
        current_time = timestamp
        cutoff_time = current_time - (30 * 288)
        
        # No need to manually clean up since we're using a deque with maxlen
    
    def _get_fee_rate(self, rolling_volume=None):
            """
            Get trading fee rate based on rolling volume.
        
            Args:
                rolling_volume (float, optional): 30-day rolling volume for fee tiers.
            
            Returns:
                float: Fee rate as a fraction (e.g., 0.0040 for 0.40%).
            """
            if rolling_volume is None:
                rolling_volume = self._get_rolling_volume()
        
            # Kraken taker fee tiers (as of current schedule)
            if rolling_volume < 10000:
                return 0.0040  # 0.40% for volume $0+
            elif rolling_volume < 50000:
                return 0.0035  # 0.35% for volume $10,000+
            elif rolling_volume < 100000:
                return 0.0024  # 0.24% for volume $50,000+
            elif rolling_volume < 250000:
                return 0.0022  # 0.22% for volume $100,000+
            elif rolling_volume < 500000:
                return 0.0020  # 0.20% for volume $250,000+
            elif rolling_volume < 1000000:
                return 0.0018  # 0.18% for volume $500,000+
            elif rolling_volume < 2500000:
                return 0.0016  # 0.16% for volume $1,000,000+
            elif rolling_volume < 5000000:
                return 0.0014  # 0.14% for volume $2,500,000+
            elif rolling_volume < 10000000:
                return 0.0012  # 0.12% for volume $5,000,000+
            elif rolling_volume < 100000000:
                return 0.0010  # 0.10% for volume $10,000,000+
            else:
                return 0.0008  # 0.08% for volume $100,000,000+
    
    def _get_next_fee_tier_threshold(self, rolling_volume=None):
            """
            Get the next fee tier threshold based on current rolling volume.
        
            Args:
                rolling_volume (float, optional): Current 30-day rolling volume.
            
            Returns:
                tuple: (next_threshold, current_fee_rate, next_fee_rate) or None if at highest tier.
            """
            if rolling_volume is None:
                rolling_volume = self._get_rolling_volume()
        
            # Get current fee rate
            current_fee_rate = self._get_fee_rate(rolling_volume)
        
            # Determine next threshold based on Kraken taker fee schedule
            if rolling_volume < 10000:
                return 10000, current_fee_rate, 0.0035
            elif rolling_volume < 50000:
                return 50000, current_fee_rate, 0.0024
            elif rolling_volume < 100000:
                return 100000, current_fee_rate, 0.0022
            elif rolling_volume < 250000:
                return 250000, current_fee_rate, 0.0020
            elif rolling_volume < 500000:
                return 500000, current_fee_rate, 0.0018
            elif rolling_volume < 1000000:
                return 1000000, current_fee_rate, 0.0016
            elif rolling_volume < 2500000:
                return 2500000, current_fee_rate, 0.0014
            elif rolling_volume < 5000000:
                return 5000000, current_fee_rate, 0.0012
            elif rolling_volume < 10000000:
                return 10000000, current_fee_rate, 0.0010
            elif rolling_volume < 100000000:
                return 100000000, current_fee_rate, 0.0008
            else:
                # Already at highest tier
                return None, current_fee_rate, current_fee_rate
    
    def _analyze_fee_tier_threshold(self):
            """
            Analyze if it's worth taking trades to reach the next fee tier threshold.
        
            Returns:
                dict: Analysis results with threshold information and potential benefit.
            """
            # Get current rolling volume
            rolling_volume = self._get_rolling_volume()
        
            # Get next threshold info
            next_threshold_info = self._get_next_fee_tier_threshold(rolling_volume)
            if next_threshold_info is None or next_threshold_info[0] is None:
                # Already at highest tier
                return {
                    "at_highest_tier": True,
                    "worth_pursuing": False,
                    "volume_needed": 0,
                    "fee_savings_percentage": 0,
                    "max_acceptable_loss": 0
                }
        
            next_threshold, current_fee_rate, next_fee_rate = next_threshold_info
        
            # Calculate volume needed to reach next threshold
            volume_needed = next_threshold - rolling_volume
        
            # Calculate fee savings percentage
            fee_savings_percentage = (current_fee_rate - next_fee_rate) / current_fee_rate
        
            # Estimate future trading volume in next 30 days
            # Base this on recent trading activity (last ~1000 steps)
            look_back = min(1000, self.current_step)
            if look_back > 0 and hasattr(self, 'volume_tensor') and len(self.volume_tensor) > self.current_step:
                recent_volume = self.volume_tensor[self.current_step - look_back:self.current_step].sum().item()
                # Scale to estimate 30-day volume
                estimated_future_volume = (recent_volume / look_back) * 8640  # 30 days in 5-min bars
            else:
                # Conservative fallback
                estimated_future_volume = rolling_volume * 0.5  # Assume 50% of current rolling volume
        
            # Calculate potential fee savings on future trades
            potential_fee_savings = estimated_future_volume * (current_fee_rate - next_fee_rate)
        
            # Calculate the maximum acceptable loss to reach next tier
            # This is the loss that would be offset by future fee savings
            max_acceptable_loss = potential_fee_savings * 0.8  # 80% of savings as acceptable loss
        
            # Determine if it's worth pursuing next tier
            # Consider: 1) How close we are to threshold, 2) Potential savings
            proximity_ratio = volume_needed / next_threshold
            worth_pursuing = (
                proximity_ratio < 0.15 and  # Within 15% of next threshold
                potential_fee_savings > 50.0 and  # Meaningful savings (>$50)
                max_acceptable_loss > 0
            )
        
            return {
                "at_highest_tier": False,
                "worth_pursuing": worth_pursuing,
                "current_volume": rolling_volume,
                "next_threshold": next_threshold,
                "volume_needed": volume_needed,
                "proximity_ratio": proximity_ratio,
                "current_fee_rate": current_fee_rate,
                "next_fee_rate": next_fee_rate,
                "fee_savings_percentage": fee_savings_percentage,
                "estimated_future_volume": estimated_future_volume,
                "potential_fee_savings": potential_fee_savings,
                "max_acceptable_loss": max_acceptable_loss
            }
    
    def step(self, action):
            """
            Execute one step in the environment based on agent action.
        
            Args:
                action: Action to take in the environment.
            
            Returns:
                tuple: (observation, reward, done, info) - Step results.
            """
            if self.close_tensor is None:
                return None, 0.0, True, {}
            
            if self.done:
                return self._get_observation(), 0.0, True, {}

            # Make sure current_step is valid
            if self.current_step >= len(self.close_tensor):
                self.done = True
                return self._get_observation(), 0.0, True, {}

            # Process any pending orders first
            completed_orders = self._process_tensor_pending_orders()

            # Parse action
            try:
                direction, fraction = float(action[0]), float(action[1])
            except (TypeError, IndexError):
                log(f"Invalid action: {action}")
                direction, fraction = 0.0, 0.0

            reward = 0.0
        
            # Get current market data
            price = self.close_tensor[self.current_step].item()
            daily_volume = (self.volume_tensor[self.current_step] * 288).item()  # 288 5-min bars per day

            # Check for unrealized loss
            unrealized_loss = 0.0
            for position in self.positions:
                # Extract position details based on type
                if isinstance(position, Position):
                    if price < position.entry_price:
                        unrealized_loss += position.size_btc * (price - position.entry_price)
                elif isinstance(position, dict):  # For backward compatibility
                    if price < position["entry_price"]:
                        unrealized_loss += position["size_btc"] * (price - position["entry_price"])
        
            # Apply penalty for large unrealized losses
            if abs(unrealized_loss) > 0.05 * self.initial_capital:
                penalty = 0.05 * abs(unrealized_loss)
                reward -= penalty

            # Update market condition metrics
            self._update_market_condition_metrics(self.current_step)

            # Calculate portfolio risk metrics
            risk_metrics = self._calculate_portfolio_risk()
        
            # Apply risk-based penalty if overall risk is too high
            if risk_metrics["overall_risk_score"] > self.risk_manager.risk_score_threshold:
                risk_penalty = (risk_metrics["overall_risk_score"] - self.risk_manager.risk_score_threshold) * 0.2
                reward -= risk_penalty * abs(self.initial_capital * 0.01)  # Scale penalty with capital

            # Analyze fee tier threshold
            fee_tier_analysis = self._analyze_fee_tier_threshold()
        
            # Add fee tier information to info dictionary for transparency
            info = {
                "fee_tier_analysis": fee_tier_analysis
            }
        
            # Adjust reward based on fee tier threshold proximity
            if fee_tier_analysis["worth_pursuing"]:
                # Add a small bonus to encourage trades that help reach the next tier
                # The bonus is proportional to how close we are to the threshold
                tier_bonus = 0.1 * (1.0 - fee_tier_analysis["proximity_ratio"]) * fee_tier_analysis["potential_fee_savings"]
                reward += tier_bonus
            
                # Include in info
                info["fee_tier_bonus"] = tier_bonus

            # Handle buying (direction > 0.1)
            if direction > 0.1 and len(self.positions) < self.max_positions:
                buy_reward = self._handle_tensor_buy(direction, fraction, price, daily_volume, risk_metrics)
                reward += buy_reward

            # Handle selling (direction < -0.1)
            elif direction < -0.1 and self.positions:
                sell_reward = self._handle_tensor_sell(direction, fraction, price, daily_volume, risk_metrics)
                reward += sell_reward

            # Move to next step
            self.current_step += 1
        
            # Check if episode is done
            if self.current_step >= len(self.close_tensor) or (self.current_step - self.window_size) >= len(self.df) - 1:
                self.done = True
            
                # Calculate final reward with risk adjustment
                episode_days = (self.current_step - self.window_size) / 288.0  # 5-min bars, 288 per day
            
                # Convert closed_trades to expected format for reward system
                closed_trades_tuple = []
                for trade in self.closed_trades:
                    if isinstance(trade, Trade):
                        closed_trades_tuple.append(trade.to_tuple())
                    else:
                        closed_trades_tuple.append(trade)  # Already in tuple format
            
                # Get final risk metrics
                final_risk_metrics = self._calculate_portfolio_risk()
            
                # Use reward system to calculate final episode reward
                base_reward = sum(self.profits) - sum(self.losses)
                reward = self.reward_system.compute_reward(
                    base_reward, 
                    self.profits, 
                    self.losses, 
                    self.returns, 
                    closed_trades_tuple, 
                    episode_days, 
                    final_risk_metrics
                )

            return self._get_observation(), reward, self.done, {"risk_score": risk_metrics.get("overall_risk_score", 0.0)}


def _create_order_book_features(self, step_idx):
        """
        Create synthetic order book features using tensor operations.
        
        Args:
            step_idx (int): Current time step index.
            
        Returns:
            dict: Order book features dictionary.
        """
        features = {}
        
        # Skip if index is invalid
        if step_idx >= len(self.close_tensor):
            return features
        
        # Update volume profile every 5 bars or on first run
        if step_idx - self.volume_profile_last_update >= 5 or self.volume_profile is None:
            self.volume_profile, price_min, price_max = self._compute_volume_profile_tensor(step_idx)
            self.volume_profile_last_update = step_idx
            
            # Update liquidity zones
            self.liquidity_zones = self._identify_liquidity_zones_tensor(step_idx)
        
        # Update pattern recognition every 10 bars or on first run
        if step_idx - self.pattern_recognition_last_update >= 10 or not self.detected_patterns:
            # Detect patterns
            self.detected_patterns = self._detect_patterns_tensor(step_idx)
            
            # Identify key price levels
            self.key_levels = self._identify_key_levels_tensor(step_idx)
            
            # Analyze order flow
            order_flow = self._analyze_order_flow_tensor(step_idx)
            features.update(order_flow)
            
            self.pattern_recognition_last_update = step_idx
        
        # Add volume profile to features
        for i, level_vol in enumerate(self.volume_profile):
            features[f'vp_level_{i+1}'] = level_vol.item() if isinstance(level_vol, torch.Tensor) else level_vol
        
        # Calculate bid-ask spread
        bid_ask_spread = self._estimate_bid_ask_spread_tensor(step_idx)
        features['bid_ask_spread'] = bid_ask_spread
        self.spread_history.append(bid_ask_spread)
        
        # Calculate volume delta (buy/sell pressure)
        vol_delta, buy_vol, sell_vol = self._calculate_volume_delta_tensor(step_idx)
        features['volume_delta'] = vol_delta
        features['buy_volume_ratio'] = buy_vol / (buy_vol + sell_vol) if (buy_vol + sell_vol) > 0 else 0.5
        
        # Estimate market liquidity
        liquidity = self._estimate_market_liquidity_tensor(step_idx)
        features['market_liquidity'] = liquidity
        self.liquidity_history.append(liquidity)
        
        # Calculate distance to liquidity zones
        current_price = self.close_tensor[step_idx].item()
        for i, zone_price in enumerate(self.liquidity_zones):
            # Normalized distance to liquidity zone
            dist = (zone_price - current_price) / current_price if current_price > 0 else 0.0
            features[f'liq_zone_{i+1}_dist'] = dist
        
        # Add pattern recognition features
        features['pattern_strength'] = self.detected_patterns.get('strength', 0.0)
        features['pattern_trend'] = self.detected_patterns.get('trend', 0)
        
        # Add pattern confidence scores
        patterns_dict = self.detected_patterns.get('patterns', {})
        for pattern, confidence in patterns_dict.items():
            features[f'pattern_{pattern}'] = confidence
        
        # Add distances to nearest support and resistance levels
        if 'support' in self.key_levels and current_price > 0:
            support_levels = self.key_levels['support']
            if support_levels:
                support_prices = [s['price'] for s in support_levels]
                nearest_support_idx = torch.argmin(torch.abs(torch.tensor(
                    support_prices, device=self.device) - current_price)).item()
                nearest_support = support_prices[nearest_support_idx]
                features['support_distance'] = (nearest_support - current_price) / current_price
                features['support_strength'] = support_levels[nearest_support_idx]['strength']
            else:
                features['support_distance'] = -0.1
                features['support_strength'] = 0.0
            
        if 'resistance' in self.key_levels and current_price > 0:
            resistance_levels = self.key_levels['resistance']
            if resistance_levels:
                resistance_prices = [r['price'] for r in resistance_levels]
                nearest_resistance_idx = torch.argmin(torch.abs(torch.tensor(
                    resistance_prices, device=self.device) - current_price)).item()
                nearest_resistance = resistance_prices[nearest_resistance_idx]
                features['resistance_distance'] = (nearest_resistance - current_price) / current_price
                features['resistance_strength'] = resistance_levels[nearest_resistance_idx]['strength']
            else:
                features['resistance_distance'] = 0.1
                features['resistance_strength'] = 0.0
        
        return features
    
def _compute_volume_profile_tensor(self, current_step, lookback=144, num_levels=10):
        """
        Compute volume profile using tensor operations.
        
        Args:
            current_step (int): Current time step index.
            lookback (int, optional): Number of bars to look back. Defaults to 144.
            num_levels (int, optional): Number of price levels in volume profile. Defaults to 10.
            
        Returns:
            tuple: (profile, price_min, price_max) - Volume profile tensor and price range.
        """
        # Ensure indices are valid
        if self.ohlcv_tensor is None or len(self.ohlcv_tensor) <= current_step:
            return torch.zeros(num_levels, device=self.device), 0, 0
            
        start_idx = max(0, current_step - lookback)
        
        # Check if we have enough data
        if start_idx >= current_step:
            return torch.zeros(num_levels, device=self.device), 0, 0
            
        window = self.ohlcv_tensor[start_idx:current_step]
        
        if len(window) == 0:
            return torch.zeros(num_levels, device=self.device), 0, 0
        
        # Extract OHLCV data
        try:
            close = window[:, 0]
            open_price = window[:, 1]
            high = window[:, 2]
            low = window[:, 3]
            volume = window[:, 4]
        except IndexError:
            # Handle case where tensor doesn't have expected dimensions
            return torch.zeros(num_levels, device=self.device), 0, 0
        
        # Calculate price range for the window
        price_min = torch.min(low).item()
        price_max = torch.max(high).item()
            
        if price_max <= price_min:
            # Avoid division by zero
            price_max = price_min * 1.001
        
        # Create price levels
        level_size = (price_max - price_min) / num_levels
        levels = torch.linspace(price_min, price_max, num_levels + 1, device=self.device)
        
        # Initialize volume profile
        profile = torch.zeros(num_levels, device=self.device)
        
        # Assign volume to levels using vectorized operations
        typical_prices = (close + high + low + open_price) / 4
        
        # For each price level, find bars that fall within that level and sum their volumes
        for i in range(num_levels):
            level_min = levels[i]
            level_max = levels[i+1]
            
            # Find bars where typical price falls within this level
            in_level = (typical_prices >= level_min) & (typical_prices < level_max)
            
            # Sum volumes for bars in this level
            if torch.any(in_level):
                profile[i] = torch.sum(volume[in_level])
        
        # Normalize profile to be relative to total volume
        total_volume = torch.sum(volume)
        if total_volume > 0:
            profile = profile / total_volume
            
        return profile, price_min, price_max
    
def _identify_liquidity_zones_tensor(self, current_step, lookback=288, min_zone_distance=0.015):
        """
        Identify liquidity zones using tensor operations.
        
        Args:
            current_step (int): Current time step index.
            lookback (int, optional): Number of bars to look back. Defaults to 288.
            min_zone_distance (float, optional): Minimum distance between zones. Defaults to 0.015.
            
        Returns:
            list: List of price levels representing liquidity zones.
        """
        # Ensure we have valid data
        if self.ohlcv_tensor is None or len(self.ohlcv_tensor) <= current_step:
            return [0.0]  # Return a default value
            
        start_idx = max(0, current_step - lookback)
        window = self.ohlcv_tensor[start_idx:current_step]
        
        if len(window) < 2:
            return [self.ohlcv_tensor[current_step, 0].item()]
        
        # Extract OHLCV data
        try:
            close = window[:, 0]
            high = window[:, 2]
            low = window[:, 3] 
            volume = window[:, 4]
        except IndexError:
            return [self.ohlcv_tensor[current_step, 0].item()]
        
        # Current price for reference
        current_price = self.ohlcv_tensor[current_step, 0].item()
        
        # Calculate median volume for threshold
        median_volume = torch.median(volume)
        
        # Find swings (local minima and maxima)
        swings = []
        
        # Use a tensorized approach to detect swings
        for i in range(1, len(close) - 1):
            # Potential support (demand zone)
            if close[i] < close[i-1] and close[i] < close[i+1] and volume[i] > median_volume:
                swings.append((close[i].item(), volume[i].item(), 'support'))
            # Potential resistance (supply zone)
            elif close[i] > close[i-1] and close[i] > close[i+1] and volume[i] > median_volume:
                swings.append((close[i].item(), volume[i].item(), 'resistance'))
        
        # If no swings found, use simple price levels
        if not swings:
            # Use even divisions if no swings found
            price_min = torch.min(low).item()
            price_max = torch.max(high).item()
            range_size = price_max - price_min
            
            # Create 4 evenly spaced zones
            zones = [price_min + range_size * i / 4 for i in range(1, 4)]
            zones.append(current_price)
            return sorted(zones)
        
        # Sort by volume (importance)
        swings.sort(key=lambda x: x[1], reverse=True)
        
        # Filter clusters - remove zones that are too close to each other
        filtered_zones = []
        
        for price, vol, zone_type in swings:
            # Skip if too close to an existing zone
            if not any(abs(price - z) / current_price < min_zone_distance for z in filtered_zones):
                filtered_zones.append(price)
                if len(filtered_zones) >= 4:  # Limit number of zones
                    break
        
        # Always include current price
        if current_price not in filtered_zones:
            filtered_zones.append(current_price)
            
        return sorted(filtered_zones)
    
def _estimate_bid_ask_spread_tensor(self, current_step, window_size=12):
        """
        Estimate bid-ask spread using recent price volatility.
        
        Args:
            current_step (int): Current time step index.
            window_size (int, optional): Number of bars to consider. Defaults to 12.
            
        Returns:
            float: Estimated bid-ask spread as a fraction of price.
        """
        if self.ohlcv_tensor is None or len(self.ohlcv_tensor) <= current_step:
            return 0.0001  # Default small spread
            
        start_idx = max(0, current_step - window_size)
        window = self.ohlcv_tensor[start_idx:current_step]
        
        if len(window) == 0:
            return 0.0001  # Default small spread
        
        # Calculate typical intrabar volatility as % of price
        try:
            high = window[:, 2]
            low = window[:, 3]
            close = window[:, 0]
            
            # Ensure no division by zero
            valid_idx = close > 0
            if not torch.any(valid_idx):
                return 0.0001
                
            # Calculate high-low range as percentage of closing price
            volatility = torch.mean(torch.abs(high[valid_idx] - low[valid_idx]) / close[valid_idx])
        except Exception:
            return 0.0001
        
        # Use volatility to estimate spread, with bounds
        min_spread = 0.0001  # 0.01% minimum spread
        max_spread = 0.002   # 0.2% maximum spread
        spread = max(min_spread, min(max_spread, volatility.item() * 0.1))  # Spread is ~10% of short-term volatility
        
        return spread
    
def _calculate_volume_delta_tensor(self, current_step, window_size=6):
        """
        Calculate volume delta (buying vs selling pressure).
        
        Args:
            current_step (int): Current time step index.
            window_size (int, optional): Number of bars to consider. Defaults to 6.
            
        Returns:
            tuple: (volume_delta, buy_volume, sell_volume) - Volume statistics.
        """
        if self.ohlcv_tensor is None or len(self.ohlcv_tensor) <= current_step:
            return 0.0, 0.0, 0.0
            
        start_idx = max(0, current_step - window_size)
        window = self.ohlcv_tensor[start_idx:current_step]
        
        if len(window) == 0:
            return 0.0, 0.0, 0.0
        
        # Extract close, open, volume
        try:
            close = window[:, 0]
            open_price = window[:, 1]
            volume = window[:, 4]
            
            # Calculate price change
            price_change = close - open_price
            
            # Estimate buying and selling volume
            buy_mask = price_change > 0
            sell_mask = price_change < 0
            
            # Calculate volume for buying and selling pressure
            buy_volume = torch.sum(volume[buy_mask]) if torch.any(buy_mask) else torch.tensor(0.0, device=self.device)
            sell_volume = torch.sum(volume[sell_mask]) if torch.any(sell_mask) else torch.tensor(0.0, device=self.device)
            
            # Volume delta is the difference between buy and sell volume
            volume_delta = buy_volume - sell_volume
        except Exception:
            return 0.0, 0.0, 0.0
        
        # Convert to Python floats for return
        return volume_delta.item(), buy_volume.item(), sell_volume.item()
    
def _estimate_market_liquidity_tensor(self, current_step, window_size=72):
        """
        Estimate market liquidity based on volume and volatility.
        
        Args:
            current_step (int): Current time step index.
            window_size (int, optional): Number of bars to consider. Defaults to 72.
            
        Returns:
            float: Estimated market liquidity as a value between 0 and 1.
        """
        if self.ohlcv_tensor is None or len(self.ohlcv_tensor) <= current_step:
            return 0.5  # Default medium liquidity
            
        start_idx = max(0, current_step - window_size)
        window = self.ohlcv_tensor[start_idx:current_step]
        
        if len(window) == 0:
            return 0.5  # Default medium liquidity
        
        # Extract close, high, low, volume
        try:
            close = window[:, 0]
            volume = window[:, 4]
        except Exception:
            return 0.5
        
        # Calculate volume trend
        if len(volume) > 1:
            recent_window = min(12, len(volume))
            recent_volume = torch.mean(volume[-recent_window:])
            older_volume = torch.mean(volume[:-recent_window] if len(volume) > recent_window else volume)
            volume_trend = recent_volume / (older_volume + 1e-8)
        else:
            volume_trend = 1.0
        
        # Calculate volatility
        volatility = torch.std(close) / torch.mean(close) if len(close) > 1 else torch.tensor(0.01, device=self.device)
        
        # High volume and low volatility indicates high liquidity
        liquidity = (torch.mean(volume) / (volatility * 1000 + 1e-8))
        
        # Normalize to 0-1 range
        normalized_liquidity = torch.sigmoid(liquidity / 10).item()
        
        # Adjust for volume trend
        volume_trend_val = volume_trend.item() if isinstance(volume_trend, torch.Tensor) else volume_trend
        
        if volume_trend_val > 1.2:  # Volume increasing
            normalized_liquidity = min(1.0, normalized_liquidity * 1.2)
        elif volume_trend_val < 0.8:  # Volume decreasing
            normalized_liquidity = max(0.1, normalized_liquidity * 0.8)
        
        return normalized_liquidity
    
def _detect_patterns_tensor(self, current_step, lookback=100):
        """
        Detect common price patterns using tensor operations.
        
        Args:
            current_step (int): Current time step index.
            lookback (int, optional): Number of bars to look back. Defaults to 100.
            
        Returns:
            dict: Dictionary of detected patterns and their strengths.
        """
        if self.ohlcv_tensor is None or current_step < lookback:
            return {"patterns": {}, "strength": 0.0, "trend": 0}
        
        try:
            # Extract window of data
            start_idx = max(0, current_step - lookback)
            window = self.ohlcv_tensor[start_idx:current_step]
            
            if len(window) < 20:  # Need sufficient data for pattern detection
                return {"patterns": {}, "strength": 0.0, "trend": 0}
            
            # Extract OHLCV components
            close = window[:, 0]
            open_price = window[:, 1]
            high = window[:, 2]
            low = window[:, 3]
            volume = window[:, 4]
            
            # Calculate some key moving averages for trend context
            ma10 = torch.mean(close[-10:])
            ma20 = torch.mean(close[-20:])
            ma50 = torch.mean(close[-50:]) if len(close) >= 50 else ma20
            
            # Detect basic trend
            trend = 0  # 1 = uptrend, 0 = neutral, -1 = downtrend
            if ma10 > ma20 and ma20 > ma50:
                trend = 1
            elif ma10 < ma20 and ma20 < ma50:
                trend = -1
            
            # Normalize price series for pattern matching
            norm_close = (close - torch.min(close)) / (torch.max(close) - torch.min(close) + 1e-8)
            
            patterns = {}
            overall_strength = 0.0
            
            # Pattern detection logic for common chart patterns
            # Here we'll implement a simplified version focusing on the most reliable patterns
            
            # 1. Double Bottom pattern detection
            if trend == -1 and len(close) >= 30:
                # Look for two lows at similar price levels with confirmation
                min_idx1 = torch.argmin(close[-30:-15])
                min_idx2 = torch.argmin(close[-15:]) + 15
                
                min_price1 = close[-30:][min_idx1]
                min_price2 = close[-15:][min_idx2 - 15]
                
                # Check with tolerance and volume confirmation
                price_diff_pct = abs(min_price1 - min_price2) / min_price1
                volume_confirmation = volume[-15:][min_idx2 - 15] > torch.mean(volume[-15:])
                
                if price_diff_pct < 0.02 and close[-1] > min_price2:
                    # Calculate confirmation strength
                    confirmation = (close[-1] - min_price2) / min_price2
                    
                    # Pattern strength calculation
                    time_ratio = abs((min_idx2 - min_idx1) / lookback - 0.5) * 2  # Time symmetry
                    strength = min(1.0, max(0.1, (1.0 - price_diff_pct * 50) * 0.5 + confirmation * 0.3 + (1.0 - time_ratio) * 0.2))
                    
                    if volume_confirmation:
                        strength *= 1.2  # Boost for volume confirmation
                    
                    patterns["double_bottom"] = min(1.0, strength)
                    overall_strength += strength * 0.6  # Weighting factor
            
            # 2. Bullish Engulfing pattern detection
            if len(close) >= 2:
                # Previous day was bearish (close < open)
                prev_bearish = open_price[-2] > close[-2]
                # Current day is bullish (close > open)
                curr_bullish = close[-1] > open_price[-1]
                # Current day engulfs previous day
                engulfing = open_price[-1] <= close[-2] and close[-1] >= open_price[-2]
                
                if prev_bearish and curr_bullish and engulfing:
                    # Calculate strength based on multiple factors
                    candle_size_ratio = (close[-1] - open_price[-1]) / max(0.0001, (open_price[-2] - close[-2]))
                    price_location = (close[-1] - torch.min(low[-10:])) / (torch.max(high[-10:]) - torch.min(low[-10:]) + 1e-8)
                    volume_increase = volume[-1] > volume[-2] * 1.2  # 20% volume increase
                    
                    # Calculate pattern strength
                    strength = min(1.0, candle_size_ratio * 0.4 + price_location * 0.3 + 0.3)
                    
                    # Volume confirmation boost
                    if volume_increase:
                        strength *= 1.2  # Boost for volume confirmation
                    
                    patterns["bullish_engulfing"] = min(1.0, strength)
                    overall_strength += strength * 0.5
            
            # 3. Head and Shoulders pattern detection
            if len(close) >= 40:
                # Find local peaks (potential shoulders and head)
                peaks = []
                for i in range(2, len(close) - 2):
                    if all(close[i] > close[i-j] for j in range(1, 3)) and all(close[i] > close[i+j] for j in range(1, 3)):
                        peaks.append((i, close[i].item()))
                
                # Need at least 3 peaks for head and shoulders
                if len(peaks) >= 3:
                    # Find triplets that might form head and shoulders
                    for i in range(len(peaks) - 2):
                        left_idx, left_val = peaks[i]
                        head_idx, head_val = peaks[i + 1]
                        right_idx, right_val = peaks[i + 2]
                        
                        # Check if middle peak is higher (head)
                        if head_val > left_val and head_val > right_val:
                            # Check if shoulders are at similar height
                            shoulder_diff = abs(left_val - right_val) / max(left_val, right_val)
                            if shoulder_diff < 0.1:  # Shoulders within 10% of each other
                                # Verify pattern with neckline test
                                neckline = min(
                                    torch.min(close[left_idx:head_idx]).item(),
                                    torch.min(close[head_idx:right_idx]).item()
                                )
                                
                                # Strength calculation
                                height = head_val - neckline
                                pattern_width = right_idx - left_idx
                                
                                # More reliable if pattern is wider and taller
                                width_factor = min(1.0, pattern_width / 20)  # Normalize width
                                height_factor = min(1.0, height / (head_val * 0.05))  # Height relative to price
                                
                                # Pattern strength
                                strength = (width_factor * 0.4 + height_factor * 0.4 + (1.0 - shoulder_diff) * 0.2)
                                
                                patterns["head_and_shoulders"] = min(1.0, strength)
                                overall_strength += strength * 0.7
            
            # 4. Consolidation/Range detection
            if len(close) >= 20:
                # Calculate recent price range
                recent_high = torch.max(close[-20:])
                recent_low = torch.min(close[-20:])
                price_range = (recent_high - recent_low) / torch.mean(close[-20:])
                
                # Check if price has been consolidating (narrow range)
                if price_range < 0.03:  # Less than 3% range
                    # Calculate consecutive days in range
                    days_in_range = 0
                    mean_price = torch.mean(close[-20:])
                    
                    for i in range(1, min(20, len(close))):
                        if abs(close[-i] - mean_price) / mean_price < 0.02:
                            days_in_range += 1
                        else:
                            break
                    
                    # Strength based on how long the consolidation has lasted
                    strength = min(1.0, days_in_range / 15)  # Normalize to max of 15 days
                    
                    patterns["consolidation"] = strength
                    overall_strength += strength * 0.4
            
            # 5. Breakout detection
            if len(close) >= 30:
                # Calculate previous range
                range_start = max(0, len(close) - 30)
                range_end = max(0, len(close) - 5)
                
                if range_end > range_start:
                    range_high = torch.max(close[range_start:range_end])
                    range_low = torch.min(close[range_start:range_end])
                    
                    # Check for breakout in last 5 bars
                    latest_close = close[-1]
                    
                    # Bullish breakout
                    if latest_close > range_high:
                        # Calculate breakout strength
                        breakout_size = (latest_close - range_high) / range_high
                        range_size = (range_high - range_low) / range_low
                        
                        # More significant if breaking out of tight range
                        significance = 1.0 - min(1.0, range_size * 10)
                        
                        # Volume confirmation
                        volume_ratio = volume[-1] / torch.mean(volume[range_start:range_end])
                        volume_boost = min(1.0, volume_ratio / 2)  # Normalize volume ratio
                        
                        # Calculate pattern strength
                        strength = min(1.0, breakout_size * 20 * significance * (0.7 + 0.3 * volume_boost))
                        
                        patterns["bullish_breakout"] = strength
                        overall_strength += strength * 0.6
                    
                    # Bearish breakdown
                    elif latest_close < range_low:
                        # Calculate breakdown strength
                        breakdown_size = (range_low - latest_close) / range_low
                        range_size = (range_high - range_low) / range_low
                        
                        # More significant if breaking down from tight range
                        significance = 1.0 - min(1.0, range_size * 10)
                        
                        # Volume confirmation
                        volume_ratio = volume[-1] / torch.mean(volume[range_start:range_end])
                        volume_boost = min(1.0, volume_ratio / 2)  # Normalize volume ratio
                        
                        # Calculate pattern strength
                        strength = min(1.0, breakdown_size * 20 * significance * (0.7 + 0.3 * volume_boost))
                        
                        patterns["bearish_breakdown"] = strength
                        overall_strength += strength * 0.6
            
            # Cap overall strength at 1.0
            overall_strength = min(1.0, overall_strength)
            
            return {
                "patterns": patterns,
                "strength": overall_strength,
                "trend": trend
            }
        
        except Exception as e:
            log(f"Error in pattern detection: {e}")
            return {"patterns": {}, "strength": 0.0, "trend": 0}
    
def _identify_key_levels_tensor(self, current_step, lookback=500, num_levels=5):
        """
        Identify key support and resistance levels using tensor operations.
        
        Args:
            current_step (int): Current time step index.
            lookback (int, optional): Number of bars to look back. Defaults to 500.
            num_levels (int, optional): Maximum number of levels to identify. Defaults to 5.
            
        Returns:
            dict: Dictionary with support and resistance levels and their strengths.
        """
        try:
            if self.ohlcv_tensor is None or current_step < 50:
                return {"support": [], "resistance": []}
                
            # Extract window of data
            start_idx = max(0, current_step - lookback)
            window = self.ohlcv_tensor[start_idx:current_step]
            
            if len(window) < 50:  # Need sufficient data
                return {"support": [], "resistance": []}
            
            # Extract OHLCV components
            close = window[:, 0]
            high = window[:, 2]
            low = window[:, 3]
            volume = window[:, 4]
            
            current_price = close[-1].item()
            
            # Generate price bins (more granular near current price)
            price_range = torch.max(high) - torch.min(low)
            price_min = torch.min(low).item()
            price_max = torch.max(high).item()
            
            # Create bins with higher density near current price
            bin_edges = []
            
            # Bins below current price (more dense near current price)
            price_below_range = current_price - price_min
            if price_below_range > 0:
                # Dense region (50% of bins in 20% of range near current price)
                dense_region_min = max(price_min, current_price - 0.2 * price_below_range)
                
                # Sparse bins for lower region
                if dense_region_min > price_min:
                    sparse_bins_below = torch.linspace(price_min, dense_region_min, num_levels // 2, device=self.device)
                    bin_edges.extend(sparse_bins_below.tolist())
                
                # Dense bins near current price (below)
                dense_bins_below = torch.linspace(dense_region_min, current_price, num_levels, device=self.device)
                bin_edges.extend(dense_bins_below.tolist())
                
            # Bins above current price (more dense near current price)
            price_above_range = price_max - current_price
            if price_above_range > 0:
                # Dense region (50% of bins in 20% of range near current price)
                dense_region_max = min(price_max, current_price + 0.2 * price_above_range)
                
                # Dense bins near current price (above)
                dense_bins_above = torch.linspace(current_price, dense_region_max, num_levels, device=self.device)
                bin_edges.extend(dense_bins_above.tolist())
                
                # Sparse bins for upper region
                if dense_region_max < price_max:
                    sparse_bins_above = torch.linspace(dense_region_max, price_max, num_levels // 2, device=self.device)
                    bin_edges.extend(sparse_bins_above.tolist())
            
            # Ensure bins are unique and sorted
            bin_edges = sorted(list(set(bin_edges)))
            
            # Count touches at each price level, weighted by volume
            level_touches = {}
            level_volumes = {}
            
            for i in range(len(window)):
                # High & low for the bar
                h = high[i].item()
                l = low[i].item()
                vol = volume[i].item()
                
                # Find which levels this bar's range touched
                for level in bin_edges:
                    if l <= level <= h:
                        level_touches[level] = level_touches.get(level, 0) + 1
                        level_volumes[level] = level_volumes.get(level, 0) + vol
            
            # Calculate level strengths based on touch count and volume
            level_strengths = {}
            max_touches = max(level_touches.values()) if level_touches else 1
            max_volume = max(level_volumes.values()) if level_volumes else 1
            
            for level, touches in level_touches.items():
                # Strength based on touch count and volume
                touch_score = touches / max_touches
                volume_score = level_volumes.get(level, 0) / max_volume
                
                # Combine scores (70% weight to touches, 30% to volume)
                strength = (touch_score * 0.7) + (volume_score * 0.3)
                level_strengths[level] = strength
            
            # Filter and categorize levels as support and resistance
            support_levels = []
            resistance_levels = []
            
            # Only keep levels with significant strength
            significant_levels = [(level, strength) for level, strength in level_strengths.items() 
                                if strength > 0.3]
            
            # Sort by strength (strongest first)
            significant_levels.sort(key=lambda x: x[1], reverse=True)
            top_levels = significant_levels[:num_levels*2]  # Get twice as many for S/R separation
            
            # Separate into support (below price) and resistance (above price)
            for level, strength in top_levels:
                if level < current_price:
                    support_levels.append({"price": level, "strength": strength})
                else:
                    resistance_levels.append({"price": level, "strength": strength})
            
            # Sort by price (support: high to low, resistance: low to high)
            support_levels.sort(key=lambda x: x["price"], reverse=True)
            resistance_levels.sort(key=lambda x: x["price"])
            
            # Limit to requested number of levels
            support_levels = support_levels[:num_levels]
            resistance_levels = resistance_levels[:num_levels]
            
            return {
                "support": support_levels,
                "resistance": resistance_levels
            }
            
        except Exception as e:
            log(f"Error in identify_key_levels_tensor: {e}")
            return {"support": [], "resistance": []}
    
def _analyze_order_flow_tensor(self, current_step, window_size=30):
        """
        Analyze order flow and market imbalance using tensor operations.
        
        Args:
            current_step (int): Current time step index.
            window_size (int, optional): Number of bars to analyze. Defaults to 30.
            
        Returns:
            dict: Order flow analysis metrics.
        """
        try:
            if self.ohlcv_tensor is None or len(self.ohlcv_tensor) <= current_step:
                return {"imbalance": 0.0, "pressure": 0.0, "exhaustion": 0.0}
                
            start_idx = max(0, current_step - window_size)
            window = self.ohlcv_tensor[start_idx:current_step]
            
            if len(window) < 5:  # Need some data
                return {"imbalance": 0.0, "pressure": 0.0, "exhaustion": 0.0}
            
            # Extract data
            close = window[:, 0]
            open_price = window[:, 1]
            high = window[:, 2]
            low = window[:, 3]
            volume = window[:, 4]
            
            # Calculate bullish vs bearish volume
            price_change = close - open_price
            bullish_mask = price_change > 0
            bearish_mask = price_change < 0
            
            bullish_volume = torch.sum(volume[bullish_mask]) if torch.any(bullish_mask) else torch.tensor(0.0, device=self.device)
            bearish_volume = torch.sum(volume[bearish_mask]) if torch.any(bearish_mask) else torch.tensor(0.0, device=self.device)
            
            total_bullish = bullish_volume.item()
            total_bearish = bearish_volume.item()
            total_volume = total_bullish + total_bearish
            
            # Calculate buy/sell imbalance (-1.0 to 1.0)
            if total_volume > 0:
                imbalance = (total_bullish - total_bearish) / total_volume
            else:
                imbalance = 0.0
            
            # Calculate buying/selling pressure using price range
            # Higher volume with larger price movement indicates stronger pressure
            bar_ranges = high - low
            mean_range = torch.mean(bar_ranges).item()
            mean_volume = torch.mean(volume).item()
            
            pressure = 0.0
            
            if len(window) > 1:
                # Calculate pressure as combination of price movement and volume
                for i in range(len(window)):
                    range_factor = (bar_ranges[i] / mean_range).item() if mean_range > 0 else 1.0
                    vol_factor = (volume[i] / mean_volume).item() if mean_volume > 0 else 1.0
                    
                    # Direction of pressure (positive for bullish, negative for bearish)
                    if close[i] > open_price[i]:  # Bullish
                        pressure += range_factor * vol_factor
                    else:  # Bearish
                        pressure -= range_factor * vol_factor
                
                # Normalize pressure to -1 to 1 range
                pressure = torch.tanh(torch.tensor(pressure / len(window))).item()
            
            # Detect volume exhaustion
            # Volume increasing but price movement decreasing suggests exhaustion
            exhaustion = 0.0
            
            if len(window) > 10:
                recent_vol = volume[-5:]
                older_vol = volume[-10:-5]
                
                recent_range = torch.mean(high[-5:] - low[-5:])
                older_range = torch.mean(high[-10:-5] - low[-10:-5])
                
                recent_vol_mean = torch.mean(recent_vol).item()
                older_vol_mean = torch.mean(older_vol).item()
                
                vol_increasing = recent_vol_mean > older_vol_mean * 1.1  # 10% increase
                range_decreasing = recent_range < older_range * 0.9  # 10% decrease
                
                if vol_increasing and range_decreasing:
                    # Calculate exhaustion level
                    if older_vol_mean > 0:
                        vol_change = (recent_vol_mean / older_vol_mean - 1.0)
                        exhaustion = min(1.0, vol_change * 3)
            
            return {
                "imbalance": imbalance,
                "pressure": pressure,
                "exhaustion": exhaustion
            }
        
        except Exception as e:
            log(f"Error in analyze_order_flow_tensor: {e}")
            return {"imbalance": 0.0, "pressure": 0.0, "exhaustion": 0.0}
    
def _calculate_price_momentum_tensor(self, current_step, window_size=24):
        """
        Calculate price momentum using tensor operations.
        
        Args:
            current_step (int): Current time step index.
            window_size (int, optional): Number of bars to consider. Defaults to 24.
            
        Returns:
            float: Price momentum normalized to [-1, 1] range.
        """
        try:
            if self.ohlcv_tensor is None or len(self.ohlcv_tensor) <= current_step:
                return 0.0
                
            start_idx = max(0, current_step - window_size)
            window = self.ohlcv_tensor[start_idx:current_step]
            
            if len(window) < 2:
                return 0.0
            
            # Extract close prices
            close = window[:, 0]
            
            # Calculate rate of change
            price_change = (close[-1] - close[0]) / (close[0] + 1e-8)
            
            # Normalize to [-1, 1] range using tanh
            normalized_momentum = torch.tanh(price_change * 5).item()  # Scale factor of 5 for sensitivity
            
            return normalized_momentum
        
        except Exception as e:
            log(f"Error in calculate_price_momentum_tensor: {e}")
            return 0.0
    
def batch_process_features(self, current_indices, lookback=288):
        """
        Process multiple tensor features in batch for efficiency.
        
        Args:
            current_indices (list): List of indices to process.
            lookback (int, optional): Default lookback window. Defaults to 288.
            
        Returns:
            list: List of feature dictionaries for each index.
        """
        if self.ohlcv_tensor is None or not current_indices:
            return [{}] * len(current_indices)
        
        batch_features = []
        
        # Process each index
        for idx in current_indices:
            features = {}
            
            # Volume profile
            features['volume_profile'], features['price_min'], features['price_max'] = self._compute_volume_profile_tensor(
                idx, lookback=min(lookback, 144)
            )
            
            # Liquidity zones
            features['liquidity_zones'] = self._identify_liquidity_zones_tensor(
                idx, lookback=min(lookback, 288)
            )
            
            # Spread estimation
            features['bid_ask_spread'] = self._estimate_bid_ask_spread_tensor(
                idx, window_size=min(lookback, 12)
            )
            
            # Volume delta
            features['volume_delta'], features['buy_volume'], features['sell_volume'] = self._calculate_volume_delta_tensor(
                idx, window_size=min(lookback, 6)
            )
            
            # Liquidity estimation
            features['market_liquidity'] = self._estimate_market_liquidity_tensor(
                idx, window_size=min(lookback, 72)
            )
            
            # Price momentum
            features['momentum'] = self._calculate_price_momentum_tensor(
                idx, window_size=min(lookback, 24)
            )
            
            # Advanced pattern detection
            pattern_results = self._detect_patterns_tensor(
                idx, lookback=min(lookback, 100)
            )
            features['pattern_strength'] = pattern_results.get('strength', 0.0)
            features['pattern_trend'] = pattern_results.get('trend', 0)
            features['detected_patterns'] = pattern_results.get('patterns', {})
            
            # Support/Resistance
            features['key_levels'] = self._identify_key_levels_tensor(
                idx, lookback=min(lookback, 500)
            )
            
            # Order flow analysis
            order_flow = self._analyze_order_flow_tensor(
                idx, window_size=min(lookback, 30)
            )
            features.update(order_flow)
            
            batch_features.append(features)
        
        return batch_features
    
def get_tensor_metrics(self):
        """
        Get current market metrics based on tensor analysis.
        
        Returns:
            dict: Dictionary of current market metrics.
        """
        if self.current_step >= len(self.close_tensor):
            return {"error": "Current step out of range"}
        
        # Process features for current step
        features = self.batch_process_features([self.current_step])[0]
        
        # Add trend information
        pattern_results = features.get('detected_patterns', {})
        trend = features.get('pattern_trend', 0)
        trend_strength = 0.0
        
        if trend != 0:
            # Calculate trend strength
            if self.current_step >= 50:
                # Use linear regression on recent prices
                x = torch.arange(50, device=self.device).float()
                y = self.close_tensor[self.current_step-50:self.current_step]
                
                mean_x = torch.mean(x)
                mean_y = torch.mean(y)
                
                # Calculate slope using covariance and variance
                numerator = torch.sum((x - mean_x) * (y - mean_y))
                denominator = torch.sum((x - mean_x) ** 2)
                
                if denominator > 0:
                    slope = (numerator / denominator).item()
                    # Normalize slope to trend strength
                    avg_price = torch.mean(y).item()
                    trend_strength = min(1.0, abs(slope * 50 / avg_price) * 5)
                    
                    # Adjust direction based on slope
                    if slope > 0:
                        trend = 1
                    elif slope < 0:
                        trend = -1
        
        # Build comprehensive metrics
        return {
            "price": self.close_tensor[self.current_step].item(),
            "volume": self.volume_tensor[self.current_step].item(),
            "trend": trend,
            "trend_strength": trend_strength,
            "momentum": features.get('momentum', 0.0),
            "liquidity": features.get('market_liquidity', 0.5),
            "buy_sell_imbalance": features.get('imbalance', 0.0),
            "buying_pressure": features.get('pressure', 0.0),
            "volume_exhaustion": features.get('exhaustion', 0.0),
            "bid_ask_spread": features.get('bid_ask_spread', 0.0001),
            "patterns": features.get('detected_patterns', {}),
            "pattern_strength": features.get('pattern_strength', 0.0),
            "support_levels": [level["price"] for level in features.get('key_levels', {}).get('support', [])],
            "resistance_levels": [level["price"] for level in features.get('key_levels', {}).get('resistance', [])],
            "risk_score": self._calculate_portfolio_risk().get("overall_risk_score", 0.0)
        }


class TensorVectorEnv:
    """
    Vectorized tensor-based environment for parallel execution.
    
    Allows running multiple tensor environments simultaneously for faster 
    exploration and more stable learning.
    """
    
    def __init__(self, env_fns):
        """
        Initialize vectorized environment.
        
        Args:
            env_fns (list): List of functions that create tensor environments.
        """
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        log(f"Created TensorVectorEnv with {self.num_envs} environments")
        
        # For batch processing
        self.device = self.envs[0].device if self.envs else "cpu"
    
    def reset(self):
        """
        Reset all environments.
        
        Returns:
            list: List of observations from each environment.
        """
        return [env.reset() for env in self.envs]
    
    def get_current_steps(self):
        """
        Get current step from all environments.
        
        Returns:
            list: List of current step indices.
        """
        return [env.current_step for env in self.envs]
    
    def get_risk_metrics(self):
        """
        Get risk metrics from all environments.
        
        Returns:
            list: List of risk metrics dictionaries.
        """
        return [env._calculate_portfolio_risk() for env in self.envs]
    
    def get_market_metrics(self):
        """
        Get market metrics from all environments.
        
        Returns:
            list: List of market metrics dictionaries.
        """
        return [env.get_tensor_metrics() for env in self.envs]
    
    def get_states(self):
        """
        Get environment states for saving/checkpointing.
        
        Returns:
            list: List of environment states.
        """
        return [env.save_state() for env in self.envs]
    
    def set_states(self, states):
        """
        Set environment states from saved states.
        
        Args:
            states (list): List of environment states.
        """
        for env, state in zip(self.envs, states):
            env.load_state(state)
    
    def close(self):
        """Close all environments and release resources."""
        for env in self.envs:
            if hasattr(env, 'close'):
                env.close()
    
    def step(self, actions):
        """
        Step all environments with corresponding actions.
        
        Args:
            actions (list): List of actions for each environment.
            
        Returns:
            tuple: (observations, rewards, dones, infos) for all environments.
        """
        results = [env.step(action) for env, action in zip(self.envs, actions)]
        
        # Unpack results
        observations = [result[0] for result in results]
        rewards = [result[1] for result in results]
        dones = [result[2] for result in results]
        infos = [result[3] for result in results]
        
        return observations, rewards, dones, infos


def create_tensor_env(df, config, device="cpu"):
    """
    Create a tensor-based trading environment.
    
    Args:
        df (pandas.DataFrame): DataFrame with market data.
        config (dict): Configuration parameters.
        device (str, optional): Device to run on. Defaults to "cpu".
        
    Returns:
        TensorTradingEnv: Created environment.
    """
    # Extract key config parameters
    window_size = config.get("WINDOW_SIZE", 288)
    initial_capital = config.get("INITIAL_CAPITAL", 100000.0)
    max_positions = config.get("MAX_POSITION_HOLDINGS", 50)
    bucket = config.get("BUCKET", "Scalping")
    
    # Create environment
    env = TensorTradingEnv(
        df, 
        window_size, 
        initial_capital, 
        max_positions, 
        bucket,
        config,
        device=device
    )
    
    log(f"Created TensorTradingEnv with bucket={bucket}, device={device}")
    return env


def make_tensor_env_creator(df, config, device="cpu"):
    """
    Create a function that initializes a tensor environment.
    
    Useful for vectorized environments.
    
    Args:
        df (pandas.DataFrame): DataFrame with market data.
        config (dict): Configuration parameters.
        device (str, optional): Device to run on. Defaults to "cpu".
        
    Returns:
        function: Environment creator function.
    """
    def _create_env():
        return create_tensor_env(df, config, device)
    return _create_env


def create_tensor_vector_env(df, config, num_envs=4, device="cpu"):
    """
    Create a vectorized tensor-based trading environment.
    
    Args:
        df (pandas.DataFrame): DataFrame with market data.
        config (dict): Configuration parameters.
        num_envs (int, optional): Number of environments to create. Defaults to 4.
        device (str, optional): Device to run on. Defaults to "cpu".
        
    Returns:
        TensorVectorEnv: Vectorized environment.
    """
    env_fns = [make_tensor_env_creator(df, config, device) for _ in range(num_envs)]
    vec_env = TensorVectorEnv(env_fns)
    
    log(f"Created TensorVectorEnv with {num_envs} environments on device={device}")
    return vec_env


def detect_gpu_availablity():
    """
    Detect if GPU is available for tensor operations.
    
    Returns:
        str: "cuda" if GPU is available, otherwise "cpu".
    """
    if torch.cuda.is_available():
        # Check CUDA memory
        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory
            # If more than 2GB memory, use GPU
            if total_memory > 2 * 1024 * 1024 * 1024:
                return "cuda"
        except Exception as e:
            log(f"Error checking CUDA memory: {e}")
    
    return "cpu"


def benchmark_tensor_env(df, config, num_steps=1000):
    """
    Benchmark tensor environment performance.
    
    Args:
        df (pandas.DataFrame): DataFrame with market data.
        config (dict): Configuration parameters.
        num_steps (int, optional): Number of steps to benchmark. Defaults to 1000.
        
    Returns:
        dict: Benchmark results.
    """
    # Try both CPU and GPU if available
    cpu_device = "cpu"
    gpu_device = detect_gpu_availablity()
    
    results = {}
    
    # Benchmark CPU
    start_time = time.time()
    env = create_tensor_env(df, config, device=cpu_device)
    env.reset()
    
    for _ in range(min(num_steps, len(df) - env.window_size - 1)):
        action = [np.random.uniform(-1, 1), np.random.uniform(0, 1)]
        _, _, done, _ = env.step(action)
        if done:
            break
    
    cpu_time = time.time() - start_time
    results["cpu_time"] = cpu_time
    results["cpu_steps_per_second"] = min(num_steps, len(df) - env.window_size - 1) / cpu_time
    
    # Benchmark GPU if available and different from CPU
    if gpu_device != "cpu":
        start_time = time.time()
        env = create_tensor_env(df, config, device=gpu_device)
        env.reset()
        
        for _ in range(min(num_steps, len(df) - env.window_size - 1)):
            action = [np.random.uniform(-1, 1), np.random.uniform(0, 1)]
            _, _, done, _ = env.step(action)
            if done:
                break
        
        gpu_time = time.time() - start_time
        results["gpu_time"] = gpu_time
        results["gpu_steps_per_second"] = min(num_steps, len(df) - env.window_size - 1) / gpu_time
        results["speedup"] = cpu_time / gpu_time if gpu_time > 0 else float('inf')
    
    # Benchmark vectorized environment
    if gpu_device != "cpu":
        device = gpu_device
        num_envs = 4  # Use 4 environments for benchmark
    else:
        device = cpu_device
        num_envs = 2  # Use fewer environments on CPU
    
    start_time = time.time()
    vec_env = create_tensor_vector_env(df, config, num_envs=num_envs, device=device)
    observations = vec_env.reset()
    
    steps_per_env = min(num_steps // num_envs, len(df) - env.window_size - 1)
    for _ in range(steps_per_env):
        actions = [[np.random.uniform(-1, 1), np.random.uniform(0, 1)] for _ in range(num_envs)]
        observations, _, dones, _ = vec_env.step(actions)
        if all(dones):
            break
    
    vec_time = time.time() - start_time
    results["vec_time"] = vec_time
    results["vec_steps_per_second"] = steps_per_env * num_envs / vec_time
    results["vec_speedup"] = (cpu_time * num_envs) / vec_time if vec_time > 0 else float('inf')
    
    return results


# When imported as a module, register with EnvRegistry
try:
    # EnvRegistry should already be imported from the dynamic imports at the top
    # Don't need to import again: from src.environment.env_interfaces import EnvRegistry
    
    # Create factory function that matches the expected signature from create_environment
    def tensor_env_factory(df, config, device="cpu"):
        """Factory function for TensorTradingEnv"""
        window_size = config.get("window_size", 50)
        initial_capital = config.get("initial_capital", 10000.0)
        max_positions = config.get("max_positions", 5)
        bucket = config.get("bucket", "Short")
        
        return TensorTradingEnv(
            df=df,
            window_size=window_size,
            initial_capital=initial_capital,
            max_positions=max_positions,
            bucket=bucket,
            config=config,
            device=device
        )
    
    # Register with environment registry
    EnvRegistry.register("tensor", tensor_env_factory)
except ImportError:
    pass  # Registry not available


# When run as a script, demonstrate tensor environment usage
if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    
    log("Running TensorTradingEnv demonstration")
    
    # Create a synthetic dataset
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=5000, freq='5min')
    
    # Generate price data following a random walk with drift
    init_price = 50000
    returns = np.random.normal(0.0001, 0.002, len(dates)) # Slight positive drift
    prices = init_price * (1 + np.cumsum(returns))
    
    # Generate OHLCV data
    df = pd.DataFrame({
        'timestamp': dates,
        'close': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.002, len(dates)))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.002, len(dates)))),
        'volume': np.random.exponential(10, len(dates)) + 2  # Volume with occasional spikes
    })
    
    # Add some technical indicators
    df['SMA9'] = df['close'].rolling(9).mean()
    df['SMA21'] = df['close'].rolling(21).mean()
    df['RSI14'] = 50 + 15 * np.random.normal(0, 1, len(df))  # Simplified for demo
    
    # Setup configuration
    config = {
        "WINDOW_SIZE": 100,
        "INITIAL_CAPITAL": 100000.0,
        "MAX_POSITION_HOLDINGS": 5,
        "BUCKET": "Scalping",
        "RISK_SCORE_THRESHOLD": 0.7,
        "DRAWDOWN_LIMIT": 0.15
    }
    
    # Determine device (use CUDA if available)
    device = detect_gpu_availablity()
    log(f"Using device: {device}")
    
    # Create tensor environment
    env = create_tensor_env(df, config, device=device)
    
    # Benchmark tensor environment
    log("Benchmarking tensor environment...")
    benchmark_results = benchmark_tensor_env(df, config, num_steps=1000)
    
    for key, value in benchmark_results.items():
        if "time" in key:
            log(f"{key}: {value:.2f} seconds")
        elif "speedup" in key:
            log(f"{key}: {value:.2f}x")
        else:
            log(f"{key}: {value:.2f}")
    
    # Run basic test
    log("\nRunning basic test...")
    obs = env.reset()
    log(f"Observation shape: {obs.shape if obs is not None else None}")
    
    # Take 10 random actions
    total_reward = 0
    for i in range(10):
        action = [np.random.uniform(-1, 1), np.random.uniform(0, 1)]
        obs, reward, done, info = env.step(action)
        total_reward += reward
        log(f"Step {i+1}, Action: {action}, Reward: {reward:.4f}, Done: {done}, Risk: {info.get('risk_score', 0):.4f}")
        
        # Get market metrics
        if i == 5:
            metrics = env.get_tensor_metrics()
            log(f"Market metrics: {metrics}")
        
        if done:
            break
    
    log(f"Test complete, total reward: {total_reward:.4f}")
    
    # Demo tensorized feature calculation
    log("\nDemonstrating batch feature calculation...")
    features = env.batch_process_features([100, 200, 300])
    log(f"Processed features for 3 time steps in batch")
    
    # Cleanup
    log("\nTensor environment demonstration completed")
