#!/usr/bin/env python
"""
Base Trading Environment

This module implements the core trading environment functionality that is
common across different environment implementations.
"""

import numpy as np
from collections import deque
import pandas as pd
import random
import sys
import os
import uuid
import importlib
from datetime import datetime, timedelta
import logging

# Import modules dynamically
try:
    # Import env_utils module
    env_utils_module = importlib.import_module("src.environment.env_utils")
    log = env_utils_module.log
    optimize_memory = env_utils_module.optimize_memory
    calculate_drawdown = env_utils_module.calculate_drawdown
    calculate_sharpe_ratio = env_utils_module.calculate_sharpe_ratio
    calculate_sortino_ratio = env_utils_module.calculate_sortino_ratio
    calculate_win_rate = env_utils_module.calculate_win_rate
    calculate_profit_factor = env_utils_module.calculate_profit_factor
    calculate_expectancy = env_utils_module.calculate_expectancy
    calculate_trade_metrics = env_utils_module.calculate_trade_metrics
    
    # Import interfaces and components
    env_interfaces_module = importlib.import_module("src.environment.env_interfaces")
    TradingEnvInterface = env_interfaces_module.TradingEnvInterface
    Position = env_interfaces_module.Position
    Order = env_interfaces_module.Order
    Trade = env_interfaces_module.Trade
    EnvRegistry = env_interfaces_module.EnvRegistry
    Withdrawal = env_interfaces_module.Withdrawal
    WithdrawalType = env_interfaces_module.WithdrawalType
    WithdrawalStatus = env_interfaces_module.WithdrawalStatus
    format_agent_action = env_interfaces_module.format_agent_action
    
    # Import risk module
    env_risk_module = importlib.import_module("src.environment.env_risk")
    create_risk_manager = env_risk_module.create_risk_manager
    
    # Import rewards module
    env_rewards_module = importlib.import_module("src.environment.env_rewards")
    create_reward_system = env_rewards_module.create_reward_system
    
    # Import market module
    env_market_module = importlib.import_module("src.environment.env_market")
    compute_slippage = env_market_module.compute_slippage
    calculate_fee = env_market_module.calculate_fee
    estimate_execution = env_market_module.estimate_execution
    
    # Import observation module
    env_observation_module = importlib.import_module("src.environment.env_observation")
    ObservationSystem = env_observation_module.ObservationSystem
    extract_observation = env_observation_module.extract_observation
    standardize_observation = env_observation_module.standardize_observation

except ImportError as e:
    print(f"Error importing environment modules: {e}")
    # Define minimal fallback functions and classes if imports fail
    def log(message, level="info"):
        print(f"[{level.upper()}] {message}")
    
    def optimize_memory():
        import gc
        gc.collect()
    
    # Define minimal classes to allow module to load (won't be functional)
    class TradingEnvInterface:
        pass
    
    class Position:
        """Represents a trading position in the environment."""
        def __init__(self, size_btc, entry_price, entry_step):
            self.size_btc = size_btc
            self.entry_price = entry_price
            self.entry_step = entry_step
            
        @classmethod
        def from_dict(cls, data):
            """Create a Position instance from a dictionary."""
            return cls(
                size_btc=data.get('size_btc', 0.0),
                entry_price=data.get('entry_price', 0.0),
                entry_step=data.get('entry_step', 0)
            )
            
        def to_dict(self):
            """Convert the position to a dictionary."""
            return {
                'size_btc': self.size_btc,
                'entry_price': self.entry_price,
                'entry_step': self.entry_step
            }
    
    class Order:
        pass
    
    class Trade:
        pass
    
    class EnvRegistry:
        pass
    
    class Withdrawal:
        pass
    
    class WithdrawalType:
        STANDARD = "standard"
        EMERGENCY = "emergency"
    
    class WithdrawalStatus:
        PENDING = "pending"
        PROCESSING = "processing"
        COMPLETED = "completed"
        REJECTED = "rejected"

# Set up logger
logger = logging.getLogger(__name__)

class BaseTradingEnv(TradingEnvInterface):
    """
    Base trading environment with core functionality.
    
    Implements standard market mechanics like order execution,
    slippage, fees, and position management.
    """
    
    def __init__(self, df, window_size, initial_capital, max_positions, bucket, config):
        """
        Initialize base trading environment.
        
        Args:
            df (pandas.DataFrame): DataFrame with OHLCV and feature data.
            window_size (int): Number of bars for observation window.
            initial_capital (float): Starting capital in USD.
            max_positions (int): Maximum number of open positions.
            bucket (str): Trading timeframe bucket ('Scalping', 'Short', 'Medium', 'Long').
            config (dict): Configuration dictionary for dynamic settings.
        """
        self.df = df.reset_index(drop=True) if df is not None else None
        self.window_size = window_size
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.bucket = bucket
        self.config = config
        
        # Store prediction horizons based on bucket
        self.prediction_horizons = {
            "Scalping": [1, 6, 12, 24],
            "Short": [6, 24, 72, 144],
            "Medium": [24, 72, 144, 288],
            "Long": [72, 144, 288, 576]
        }.get(bucket, [12, 36, 72, 144])
        
        # Core state variables
        self.current_step = window_size
        self.capital = initial_capital
        self.positions = []  # List of Position objects
        self.closed_trades = []  # List of Trade objects
        self.profits = []
        self.losses = []
        self.returns = []
        self.done = False
        
        # Currency management - differentiate between USD and USDT
        # Capital is in USDT (trading currency), withdrawals are in USD
        self.usdt_balance = initial_capital  # Capital for trading (same as self.capital)
        self.usd_balance = 0.0  # Available USD for withdrawals
        self.usd_reserved = 0.0  # USD reserved for pending withdrawals
        
        # Withdrawal management
        self.withdrawals = []  # List of Withdrawal objects
        self.profit_reserve_ratio = config.get("profit_reserve_ratio", 0.3)  # Reserve 30% of profits by default
        self.deposit_conversion_fee = config.get("deposit_conversion_fee", 0.001)  # 0.1% fee for USD->USDT
        self.withdrawal_conversion_fee = config.get("withdrawal_conversion_fee", 0.001)  # 0.1% fee for USDT->USD
        
        # Order management
        self.rolling_trades = deque(maxlen=8640)  # ~30 days of 5-min bars
        self.pending_orders = []  # List of Order objects
        
        # Market liquidity tracking
        self.liquidity_history = []
        self.spread_history = []
        
        # Initialize bucket-specific components
        self.risk_manager = create_risk_manager(self.bucket, config)
        self.reward_system = create_reward_system(self.bucket, config)
        
        # Store feature columns - use columns specified in config if available
        if df is not None:
            # Get feature columns from config if available, otherwise use default approach
            if "feature_columns" in config and config["feature_columns"]:
                self.available_cols = config["feature_columns"]
                logger.info(f"Using {len(self.available_cols)} feature columns from config")
            else:
                # Fallback to standard features if none specified in config
                standard_cols = [
                    'close', 'high', 'low', 'open', 'volume',
                    'SMA9', 'SMA21', 'SMA50', 'SMA100', 'SMA200',
                    'RSI14', 'Stoch_K', 'Stoch_D', 'CCI', 'BB_upper20',
                    'BB_mid20', 'BB_lower20', 'ATR', 'MACD', 'MACD_signal'
                ]
                
                # Filter to only include columns that exist in the dataframe
                self.available_cols = [col for col in standard_cols if col in df.columns]
                
                # Make sure at least basic price columns are included
                if 'close' not in self.available_cols and 'close' in df.columns:
                    self.available_cols.append('close')
                
                logger.info(f"Using {len(self.available_cols)} default feature columns")
            
            logger.info(f"Feature columns: {self.available_cols}")
        else:
            self.available_cols = []
            logger.warning("No dataframe provided, available columns will be empty")

    def reset(self):
        """Reset the environment to initial state for new episode"""
        self.current_step = self.window_size
        self.capital = self.initial_capital
        self.usdt_balance = self.initial_capital
        self.usd_balance = 0.0
        self.usd_reserved = 0.0
        self.positions = []
        self.closed_trades = []
        self.profits = []
        self.losses = []
        self.returns = []
        self.withdrawals = []
        self.rolling_trades.clear()
        self.pending_orders.clear()
        self.liquidity_history = []
        self.spread_history = []
        self.done = False
        
        # Initialize random withdrawal/deposit simulation if enabled
        if self.config.get("simulate_withdrawals", False):
            self._setup_withdrawal_simulation()
            
        return self._get_observation()
        
    def _setup_withdrawal_simulation(self):
        """Configure random withdrawal and deposit events for this episode"""
        # Get simulation settings
        sim_config = self.config.get("withdrawal_simulation", {})
        
        # Schedule random withdrawals
        monthly_withdrawal_chance = sim_config.get("monthly_withdrawal_chance", 0.3)  # 30% chance each month
        emergency_withdrawal_chance = sim_config.get("emergency_withdrawal_chance", 0.05)  # 5% chance
        timed_withdrawal_chance = sim_config.get("timed_withdrawal_chance", 0.15)  # 15% chance
        
        # Schedule random deposits
        monthly_deposit_chance = sim_config.get("monthly_deposit_chance", 0.4)  # 40% chance each month
        
        # Set amount ranges (as percentage of capital)
        withdrawal_min_pct = sim_config.get("withdrawal_min_pct", 0.05)  # 5% of capital
        withdrawal_max_pct = sim_config.get("withdrawal_max_pct", 0.3)  # 30% of capital
        deposit_min_pct = sim_config.get("deposit_min_pct", 0.05)  # 5% of capital
        deposit_max_pct = sim_config.get("deposit_max_pct", 0.5)  # 50% of capital
        
        # Create schedule of events for this episode
        self._withdrawal_deposit_schedule = []
        
        # Simulate for up to a year
        max_days = min(365, (len(self.df) - self.current_step) / 288)
        current_day = 0
        
        while current_day < max_days:
            # Check for monthly withdrawal
            if random.random() < monthly_withdrawal_chance:
                # Determine type
                if random.random() < emergency_withdrawal_chance:
                    w_type = WithdrawalType.EMERGENCY
                    deadline = None
                elif random.random() < timed_withdrawal_chance:
                    w_type = WithdrawalType.TIMED
                    # Random deadline between 1-4 weeks
                    deadline_days = random.randint(7, 28)
                    deadline = datetime.now() + timedelta(days=deadline_days)
                else:
                    w_type = WithdrawalType.STANDARD
                    deadline = None
                    
                # Calculate amount
                amount = random.uniform(
                    withdrawal_min_pct * self.initial_capital,
                    withdrawal_max_pct * self.initial_capital
                )
                
                # Schedule the withdrawal
                day_step = self.current_step + int(current_day * 288)
                self._withdrawal_deposit_schedule.append(
                    (day_step, "withdrawal", w_type, amount, deadline)
                )
            
            # Check for monthly deposit
            if random.random() < monthly_deposit_chance:
                # Calculate amount
                amount = random.uniform(
                    deposit_min_pct * self.initial_capital,
                    deposit_max_pct * self.initial_capital
                )
                
                # Schedule the deposit
                day_step = self.current_step + int(current_day * 288)
                self._withdrawal_deposit_schedule.append(
                    (day_step, "deposit", None, amount, None)
                )
            
            # Move to next month
            current_day += 30
    
    def _process_scheduled_events(self):
        """Process any scheduled withdrawal or deposit events"""
        current_events = [event for event in self._withdrawal_deposit_schedule 
                         if event[0] == self.current_step]
        
        for event in current_events:
            _, event_type, w_type, amount, deadline = event
            
            if event_type == "withdrawal":
                self.add_withdrawal_request(amount, w_type, deadline)
            elif event_type == "deposit":
                self.process_deposit(amount)
            
            # Remove processed event
            self._withdrawal_deposit_schedule.remove(event)
    
    def _get_observation(self):
        """Get observation from dataframe using env_observation module."""
        if self.df is None or self.current_step < self.window_size:
            return None
            
        # Get window of data
        start_idx = self.current_step - self.window_size
        end_idx = self.current_step - 1
        
        # Make sure indices are valid
        if start_idx < 0 or end_idx >= len(self.df):
            return None
            
        # Extract observation using only the available columns defined during initialization
        raw_obs = extract_observation(self.df, start_idx, end_idx, self.available_cols)
        
        # Standardize observation
        return standardize_observation(raw_obs)
    
    def _update_rolling_volume(self, bar_idx, notional):
        """Update the 30-day rolling volume for fee calculation"""
        if notional > 1e-8:
            self.rolling_trades.append((bar_idx, notional))
        while self.rolling_trades and (bar_idx - self.rolling_trades[0][0]) > 8640:  # ~30 days
            self.rolling_trades.popleft()
    
    def _get_rolling_volume(self):
        """Get total trading volume in the past 30 days"""
        return sum(x[1] for x in self.rolling_trades)
    
    def _compute_slippage(self, order_size, daily_volume):
        """Calculate price slippage based on order size and market volume"""
        # Base slippage model: square of order size relative to daily volume
        # Larger orders have greater impact on market price
        return 0.001 * (order_size / (daily_volume + 1e-8)) ** 2
        
    def _calculate_fee(self, notional):
        """Calculate trading fee based on order size and rolling volume"""
        # Import from src.utils.utils or implement directly
        # Simplified version with tiered fee structure
        rolling_volume = self._get_rolling_volume()
        
        # Tiered fee structure based on 30-day volume (simplified Kraken-like)
        if rolling_volume > 1000000:  # > $1M
            fee_rate = 0.0010  # 0.10%
        elif rolling_volume > 500000:  # > $500k
            fee_rate = 0.0016  # 0.16%
        elif rolling_volume > 100000:  # > $100k
            fee_rate = 0.0020  # 0.20%
        else:
            fee_rate = 0.0026  # 0.26%
            
        return notional * fee_rate
    
    def _current_price(self):
        """Get the current price from the dataframe"""
        if self.df is None or self.current_step >= len(self.df):
            # If no dataframe or invalid index, try to use last position's entry price
            if len(self.positions) > 0:
                return self.positions[-1]["entry_price"]
            return 0.0
            
        try:
            return self.df.loc[self.current_step, "close"]
        except (KeyError, IndexError):
            # If "close" column doesn't exist or other error, try to use last position's entry price
            if len(self.positions) > 0:
                return self.positions[-1]["entry_price"]
            return 0.0
    
    def _calculate_portfolio_risk(self):
        """Retrieve portfolio risk metrics from the risk manager."""
        try:
            current_price = self.df.loc[self.current_step, "close"]
        except (AttributeError, KeyError, IndexError):
            current_price = 0.0
            
            if len(self.positions) > 0:
                # Use the latest entry price if we can't get current price
                if isinstance(self.positions[-1], Position):
                    current_price = self.positions[-1].entry_price
                elif isinstance(self.positions[-1], dict):
                    current_price = self.positions[-1]["entry_price"]
        
        # Calculate risk metrics using the risk manager
        return self.risk_manager.calculate_risk_metrics(
            self.positions, 
            self.capital,
            current_price,
            self.returns,
            self.liquidity_history
        )
    
    def _calculate_size_limits(self, price, daily_volume):
        """
        Calculate position size limits based on multiple constraints.
        
        Args:
            price (float): Current asset price.
            daily_volume (float): Estimated daily trading volume in USD.
            
        Returns:
            float: Maximum allowed position size in BTC.
        """
        # Get from risk manager
        risk_metrics = self._calculate_portfolio_risk()
        return self.risk_manager.calculate_position_size(price, daily_volume, risk_metrics)
    
    def _calculate_risk_adjusted_size(self, price, daily_volume, direction, risk_metrics, position_count):
        """
        Calculate position size with portfolio risk adjustments.
        
        Args:
            price (float): Current asset price.
            daily_volume (float): Estimated daily trading volume in USD.
            direction (float): Trade direction (positive = buy, negative = sell).
            risk_metrics (dict): Current risk metrics
            position_count (int): Current number of open positions
            
        Returns:
            float: Risk-adjusted maximum allowed position size in BTC.
        """
        # Get risk metrics
        risk_metrics = self._calculate_portfolio_risk()
        
        # Get from risk manager with direction information
        return self.risk_manager.calculate_risk_adjusted_size(
            price, 
            daily_volume, 
            direction, 
            risk_metrics, 
            position_count
        )
    
    def _estimate_market_impact(self, size_btc, price, daily_volume):
        """
        Estimate market impact of a trade based on order size relative to market liquidity.
        
        Args:
            size_btc (float): Size of order in BTC.
            price (float): Current price of BTC.
            daily_volume (float): Daily trading volume in USD.
            
        Returns:
            tuple: (impact_pct, impact_score) - Estimated price impact percentage and impact score (0-1)
        """
        # Convert order size to USD
        order_size_usd = size_btc * price
        
        # Calculate order size as percentage of daily volume
        volume_percentage = order_size_usd / daily_volume if daily_volume > 0 else 0
        
        # Calculate base impact using square-root formula (common in market impact models)
        # Impact = k * sigma * sqrt(size / ADV)
        # Where:
        # - k is a constant (typically 0.1-1.0 depending on market)
        # - sigma is asset volatility (we'll use a simplified approach)
        # - size is order size
        # - ADV is average daily volume
        
        # Get recent volatility (simplified)
        volatility = 0.01  # Default value
        if self.df is not None and self.current_step >= 50:
            recent_prices = self.df.loc[self.current_step-50:self.current_step, "close"].values
            if len(recent_prices) > 1:
                pct_changes = np.diff(recent_prices) / recent_prices[:-1]
                volatility = np.std(pct_changes)
        
        # Get additional factors from market state
        market_liquidity = 0.5  # Default mid-level liquidity
        if self.liquidity_history:
            market_liquidity = self.liquidity_history[-1]
        
        # Constants
        k = 0.5  # Base impact factor
        
        # Adjust k based on market liquidity
        adjusted_k = k * (1.0 - 0.5 * market_liquidity)  # More liquid = less impact
        
        # Calculate impact
        if volume_percentage > 0:
            impact_pct = adjusted_k * volatility * np.sqrt(volume_percentage)
        else:
            impact_pct = 0.0
        
        # Cap at reasonable limits
        impact_pct = min(0.1, max(0.0, impact_pct))  # Cap at 10% max impact
        
        # Calculate impact score (0-1)
        # Higher score = more negative impact on execution
        if volume_percentage < 0.01:  # Less than 1% of daily volume
            impact_score = volume_percentage * 10  # Linear scaling for small orders
        else:
            # Non-linear scaling for larger orders
            impact_score = min(1.0, 0.1 + 0.9 * (volume_percentage - 0.01) / 0.09)
        
        return impact_pct, impact_score
    
    def _calculate_market_impact_penalty(self, impact_score, absolute_size):
        """
        Calculate penalty for market impact to discourage excessively large orders.
        
        Args:
            impact_score (float): Market impact score (0-1).
            absolute_size (float): Absolute size of the order in USD.
            
        Returns:
            float: Penalty amount (negative value).
        """
        # Base penalty starts at zero for very small impact
        if impact_score < 0.1:
            return 0.0
        
        # Non-linear penalty that grows more severe with higher impact
        # Penalty is relative to position size
        relative_penalty = -impact_score**2 * 0.1  # Squared relationship for more aggressive penalty
        absolute_penalty = relative_penalty * absolute_size
        
        return absolute_penalty

    def _process_pending_orders(self):
        """
        Process any pending orders that are ready.
        
        Returns:
            list: List of completed orders.
        """
        if self.df is None or self.current_step >= len(self.df):
            return []
            
        current_step = self.current_step
        completed_orders = []
        
        # Get current market conditions for fill probability
        price = self.df.loc[current_step, "close"]
        daily_volume = self.df.loc[current_step, "volume"] * 288  # Estimate daily volume
        
        # Get market liquidity if available (simplified model)
        liquidity = 0.5  # Default mid-level liquidity
        if self.liquidity_history:
            liquidity = self.liquidity_history[-1]
        
        for order in self.pending_orders[:]:
            # Enforce position count limit for buy orders
            if order.direction > 0 and len(self.positions) >= self.max_positions:
                # Skip processing this buy order if we've reached maximum positions
                self.pending_orders.remove(order)
                continue
                
            # Calculate fill probability based on time elapsed
            time_factor = min(1.0, (current_step - order.entry_step) / order.timeout)
            
            # Higher probability in liquid markets
            liquidity_factor = 0.2 * liquidity
            
            # Combine factors for final fill probability
            fill_probability = 0.05 + time_factor * 0.7 + liquidity_factor
            
            # Try to fill order
            if random.random() < fill_probability or current_step - order.entry_step >= order.timeout:
                original_size_btc = order.size_btc
                
                # For buy orders, apply risk-adjusted position size limits
                if order.direction > 0:  # Buy
                    max_size_btc = self._calculate_risk_adjusted_size(price, daily_volume, 1.0, self._calculate_portfolio_risk(), len(self.positions))
                    size_btc = min(original_size_btc, max_size_btc)
                else:  # Sell
                    size_btc = original_size_btc
                
                # Calculate slippage
                slippage = self._compute_slippage(size_btc * price, daily_volume)
                
                # Direction-based price adjustment
                if order.direction > 0:  # Buy
                    effective_price = price * (1 + slippage)
                else:  # Sell
                    effective_price = price * (1 - slippage)
                    
                notional = size_btc * effective_price
                self._update_rolling_volume(current_step, notional)
                fee = self._calculate_fee(notional)
                
                # Process buy or sell
                if order.direction > 0:  # Buy
                    if len(self.positions) < self.max_positions:
                        new_position = Position(size_btc, effective_price, current_step)
                        self.positions.append(new_position)
                        self.capital -= min((size_btc * effective_price) + fee, self.capital)
                else:  # Sell
                    # Add profit to capital
                    reference_price = order.reference_price if order.reference_price is not None else effective_price
                    reference_step = order.reference_step if order.reference_step is not None else current_step
                    
                    profit = size_btc * (effective_price - reference_price) - fee
                    self.capital += profit
                    
                    # Record trade
                    hold_time = current_step - reference_step
                    percentage_gain = (effective_price / reference_price - 1) * 100
                    self.closed_trades.append((profit, percentage_gain, hold_time))
                    
                    # Track profit/loss for metrics
                    if profit > 0:
                        self.profits.append(profit)
                    else:
                        self.losses.append(abs(profit))
                
                # Mark as completed
                completed_orders.append(order)
                self.pending_orders.remove(order)
                
        return completed_orders
        
    def _handle_buy(self, direction, fraction, price, daily_volume, risk_metrics):
        """
        Handle buy action.
        
        Args:
            direction (float): Direction component of action (positive for buy)
            fraction (float): Fraction of capital to use
            price (float): Current price
            daily_volume (float): Daily trading volume
            risk_metrics (dict): Current risk metrics
            
        Returns:
            float: Reward adjustment from the action
        """
        reward_adjustment = 0.0
        
        # Check if we've reached maximum positions
        if len(self.positions) >= self.max_positions:
            # Apply a penalty for attempting to exceed position limit
            reward_adjustment -= 0.1 * self.capital * fraction
            return reward_adjustment
        
        # Calculate maximum size based on position and portfolio risk limits
        max_size_btc = self._calculate_risk_adjusted_size(price, daily_volume, direction, risk_metrics, len(self.positions))
        
        # Calculate volume-based limit
        max_btc_by_volume = (daily_volume * self.config.get("MAX_VOLUME_PERCENTAGE", 0.05)) / price if price > 0 else 0
        
        # Apply volume limit
        max_size_btc = min(max_size_btc, max_btc_by_volume)
        
        # Calculate USD-based limit
        max_usd = min(self.capital * fraction, self.config.get("MAX_USD_PER_POSITION", float('inf')))
        max_size_btc = min(max_size_btc, max_usd / price if price > 0 else 0)
        
        # Calculate BTC-based limit
        max_size_btc = min(max_size_btc, self.config.get("MAX_BTC_PER_POSITION", float('inf')))
        
        # Create order with the calculated size
        if max_size_btc > 0:
            order = Order(
                direction=1.0,  # Buy
                size_btc=max_size_btc,
                entry_step=self.current_step,
                timeout=12  # 1 hour timeout (12 5-min bars)
            )
            self.pending_orders.append(order)
            
        return reward_adjustment
    
    def _handle_sell(self, direction, fraction, price, daily_volume, risk_metrics):
        """
        Handle sell action.
        
        Args:
            direction (float): Direction component of action (negative for sell)
            fraction (float): Fraction of positions to sell
            price (float): Current price
            daily_volume (float): Daily trading volume
            risk_metrics (dict): Current risk metrics
            
        Returns:
            float: Reward adjustment from the action
        """
        reward_adjustment = 0.0
        
        # Calculate total BTC in positions
        total_btc = 0
        for position in self.positions:
            if isinstance(position, Position):
                total_btc += position.size_btc
            elif isinstance(position, dict):
                total_btc += position["size_btc"]
                    
        if total_btc > 1e-8:
            btc_to_sell = total_btc * fraction
            
            # Calculate market impact and slippage
            impact_pct, impact_score = self._estimate_market_impact(btc_to_sell, price, daily_volume)
            slippage = self._compute_slippage(btc_to_sell * price, daily_volume)
            
            # Total slippage includes market impact
            total_slippage = slippage + impact_pct
            effective_price = price * (1 - total_slippage)  # Negative for sell
            
            # Process each position for closing
            btc_remaining = btc_to_sell
            
            # Sort positions by LIFO order (last in, first out)
            sorted_positions = sorted(
                [(i, p) for i, p in enumerate(self.positions)],
                key=lambda x: x[1].entry_step if isinstance(x[1], Position) else x[1]["entry_step"],
                reverse=True
            )
            
            positions_to_remove = []
            raw_profit = 0.0
            
            for idx, position in sorted_positions:
                if btc_remaining <= 0:
                    break
                    
                # Get position details
                if isinstance(position, Position):
                    position_size = position.size_btc
                    entry_price = position.entry_price
                    entry_step = position.entry_step
                else:  # Dict format
                    position_size = position["size_btc"]
                    entry_price = position["entry_price"]
                    entry_step = position["entry_step"]
                
                # Calculate how much to sell from this position
                sell_size = min(position_size, btc_remaining)
                trade_profit = sell_size * (effective_price - entry_price)
                raw_profit += trade_profit
                
                # Update btc remaining to sell
                btc_remaining -= sell_size
                
                # Update position size
                if isinstance(position, Position):
                    position.size_btc -= sell_size
                else:  # Dict format
                    position["size_btc"] -= sell_size
                
                # If position fully closed, mark for removal
                if (isinstance(position, Position) and position.size_btc <= 1e-8) or \
                   (isinstance(position, dict) and position["size_btc"] <= 1e-8):
                    positions_to_remove.append(idx)
                    
                    # Record closed trade
                    hold_time = self.current_step - entry_step
                    percentage_gain = (effective_price / entry_price - 1) * 100
                    trade = Trade(trade_profit, percentage_gain, hold_time, entry_step)
                    self.closed_trades.append(trade)
                    
                    # Track profit/loss
                    if trade_profit > 0:
                        self.profits.append(trade_profit)
                    else:
                        self.losses.append(abs(trade_profit))
            
            # Remove fully closed positions (in reverse order to avoid index issues)
            for idx in sorted(positions_to_remove, reverse=True):
                self.positions.pop(idx)
            
            # Calculate fees and final profit
            notional = (total_btc * fraction - btc_remaining) * effective_price
            self._update_rolling_volume(self.current_step, notional)
            fee = self._calculate_fee(notional)
            net_profit = raw_profit - fee
            
            # Update capital and return tracking
            self.capital += net_profit
            self.returns.append(net_profit / self.initial_capital)
            
            # Add profit to reward adjustment
            reward_adjustment += net_profit
        
        return reward_adjustment

    def step(self, action):
        """Take an action and advance the environment by one step"""
        # Extract action info
        direction, fraction = self._extract_action(action)
        
        # Create and execute order
        self._execute_agent_order(direction, fraction)
        
        # Get current price and manage positions
        current_price = self._current_price()
        self._update_positions(current_price)
        self._manage_stop_losses(current_price)
        
        # Process any scheduled withdrawal/deposit events
        if hasattr(self, '_withdrawal_deposit_schedule'):
            self._process_scheduled_events()
        
        # Process any due withdrawals
        self.process_withdrawals()
        
        # Update state variables
        observation = self._get_observation()
        reward = self._calculate_reward()
        
        # Check if done
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            self.done = True
        
        info = self._get_info()
        return observation, reward, self.done, info

    def get_current_step(self):
        """Return the current time step in the environment"""
        return self.current_step
    
    def get_returns(self):
        """Return the returns list for external access"""
        return self.returns
    
    def get_closed_trades(self):
        """Return closed trades for external access"""
        return self.closed_trades
    
    def get_risk_metrics(self):
        """Return current risk metrics for external access"""
        return self._calculate_portfolio_risk()
    
    def save_state(self):
        """Save current environment state for later restoration"""
        # Convert positions to dict format
        positions_data = []
        for position in self.positions:
            if isinstance(position, Position):
                positions_data.append(position.to_dict())
            else:
                positions_data.append(position)  # Already a dict
        
        # Convert orders to dict format
        orders_data = []
        for order in self.pending_orders:
            if isinstance(order, Order):
                orders_data.append(order.to_dict())
            else:
                orders_data.append(order)  # Already a dict
        
        # Convert trades to tuple format
        trades_data = []
        for trade in self.closed_trades:
            if isinstance(trade, Trade):
                trades_data.append(trade.to_tuple())
            else:
                trades_data.append(trade)  # Already a tuple
        
        return {
            "current_step": self.current_step,
            "capital": self.capital,
            "usdt_balance": self.usdt_balance,
            "usd_balance": self.usd_balance,
            "usd_reserved": self.usd_reserved,
            "positions": positions_data,
            "closed_trades": trades_data,
            "profits": self.profits.copy(),
            "losses": self.losses.copy(),
            "returns": self.returns.copy(),
            "pending_orders": orders_data,
            "done": self.done,
            # Don't copy large objects like rolling_trades, liquidity_history, etc.
            # They'll be recalculated as needed
        }
    
    def load_state(self, state):
        """Restore environment to previously saved state"""
        self.current_step = state["current_step"]
        self.capital = state["capital"]
        self.usdt_balance = state["usdt_balance"]
        self.usd_balance = state["usd_balance"]
        self.usd_reserved = state["usd_reserved"]
        
        # Convert positions from dict format
        self.positions = []
        for position_data in state["positions"]:
            if isinstance(position_data, dict):
                self.positions.append(Position.from_dict(position_data))
            else:
                self.positions.append(position_data)  # Keep existing format
        
        # Convert orders from dict format
        self.pending_orders = []
        for order_data in state["pending_orders"]:
            if isinstance(order_data, dict):
                self.pending_orders.append(Order.from_dict(order_data))
            else:
                self.pending_orders.append(order_data)  # Keep existing format
        
        # Convert trades from tuple format
        self.closed_trades = []
        for trade_data in state["closed_trades"]:
            if isinstance(trade_data, tuple):
                self.closed_trades.append(Trade.from_tuple(trade_data))
            else:
                self.closed_trades.append(trade_data)  # Keep existing format
        
        self.profits = state["profits"].copy()
        self.losses = state["losses"].copy()
        self.returns = state["returns"].copy()
        self.done = state["done"]

    def to_dict(self):
        """Convert environment state to dictionary for serialization"""
        try:
            return {
                "id": self.id,
                "current_step": self.current_step,
                "capital": self.capital,
                "positions": len(self.positions),
                "closed_trades": len(self.closed_trades),
                "risk_metrics": self._calculate_portfolio_risk(),
                "metrics": self.get_metrics() if hasattr(self, "get_metrics") else {}
            }
        except Exception as e:
            return {"error": str(e)}

    # Withdrawal Management Functions
    
    def add_withdrawal_request(self, amount_usd, withdrawal_type, deadline=None):
        """
        Add a new withdrawal request to the system.
        
        Args:
            amount_usd (float): Amount in USD to withdraw
            withdrawal_type (WithdrawalType): Type of withdrawal
            deadline (datetime, optional): Deadline for timed withdrawals
        
        Returns:
            str: Withdrawal ID
        """
        withdrawal_id = str(uuid.uuid4())
        withdrawal = Withdrawal(
            withdrawal_id=withdrawal_id,
            amount_usd=amount_usd,
            withdrawal_type=withdrawal_type,
            deadline=deadline
        )
        
        # Handle emergency withdrawals immediately
        if withdrawal_type == WithdrawalType.EMERGENCY:
            self._process_emergency_withdrawal(withdrawal)
        
        self.withdrawals.append(withdrawal)
        return withdrawal_id
    
    def _process_emergency_withdrawal(self, withdrawal):
        """
        Process an emergency withdrawal immediately.
        
        Args:
            withdrawal (Withdrawal): Withdrawal request
        """
        # First, use any available USD balance
        if self.usd_balance > 0:
            fulfilled = min(self.usd_balance, withdrawal.amount_usd)
            withdrawal.fulfill(fulfilled)
            self.usd_balance -= fulfilled
            
        # If more is needed, convert USDT to USD
        remaining = withdrawal.get_remaining_amount()
        if remaining > 0 and self.usdt_balance > 0:
            # How much USDT we can convert
            usdt_available = self.usdt_balance
            
            # Apply conversion fee
            conversion_fee = usdt_available * self.withdrawal_conversion_fee
            usdt_after_fee = usdt_available - conversion_fee
            
            # Fulfill as much as possible
            fulfilled = min(usdt_after_fee, remaining)
            
            if fulfilled > 0:
                # Update balances
                self.usdt_balance -= (fulfilled + conversion_fee)
                self.capital = self.usdt_balance  # Keep capital in sync
                
                # Update withdrawal
                withdrawal.fulfill(fulfilled)
    
    def process_deposit(self, amount_usd):
        """
        Process a USD deposit, converting it to USDT for trading.
        
        Args:
            amount_usd (float): Deposit amount in USD
        
        Returns:
            float: Amount of USDT added to trading capital
        """
        if amount_usd <= 0:
            return 0
            
        # Apply conversion fee
        conversion_fee = amount_usd * self.deposit_conversion_fee
        usdt_amount = amount_usd - conversion_fee
        
        # Update balances
        self.usdt_balance += usdt_amount
        self.capital = self.usdt_balance  # Keep capital in sync
        
        return usdt_amount
    
    def reserve_profit_for_withdrawals(self, profit):
        """
        Reserve a portion of profit for pending withdrawals.
        
        Args:
            profit (float): Profit amount in USDT
            
        Returns:
            float: Amount reserved for withdrawals
        """
        if profit <= 0 or not self.withdrawals:
            return 0
            
        # Get active withdrawals that need funding
        active_withdrawals = [w for w in self.withdrawals 
                             if w.status == WithdrawalStatus.PENDING or 
                                w.status == WithdrawalStatus.PARTIAL]
        
        if not active_withdrawals:
            return 0
            
        # Calculate amount to reserve
        amount_to_reserve = profit * self.profit_reserve_ratio
        
        # Convert USDT to USD (apply conversion fee)
        conversion_fee = amount_to_reserve * self.withdrawal_conversion_fee
        usd_to_reserve = amount_to_reserve - conversion_fee
        
        if usd_to_reserve <= 0:
            return 0
            
        # Update USDT balance
        self.usdt_balance -= amount_to_reserve
        self.capital = self.usdt_balance  # Keep capital in sync
        
        # Update USD balance
        self.usd_balance += usd_to_reserve
        self.usd_reserved += usd_to_reserve
        
        # Allocate the reserved amount to withdrawals based on urgency
        self._allocate_reserved_funds(usd_to_reserve, active_withdrawals)
        
        return amount_to_reserve
    
    def _allocate_reserved_funds(self, amount_usd, withdrawals):
        """
        Allocate reserved funds to withdrawals based on urgency.
        
        Args:
            amount_usd (float): Amount to allocate
            withdrawals (list): List of Withdrawal objects
        """
        if amount_usd <= 0 or not withdrawals:
            return
            
        # Sort withdrawals by urgency
        withdrawals.sort(key=lambda w: w.get_urgency(), reverse=True)
        
        # Check if any withdrawal meets its full requirement
        total_needed = sum(w.get_remaining_amount() for w in withdrawals)
        
        if amount_usd >= total_needed:
            # We can fulfill all withdrawals
            for w in withdrawals:
                fulfilled = w.get_remaining_amount()
                w.fulfill(fulfilled)
                self.usd_reserved -= fulfilled
        else:
            # Allocate proportionally based on urgency and need
            total_urgency = sum(w.get_urgency() for w in withdrawals)
            
            for w in withdrawals:
                # Calculate share based on urgency
                urgency_share = w.get_urgency() / total_urgency if total_urgency > 0 else 1.0 / len(withdrawals)
                
                # Allocate funds
                allocated = min(w.get_remaining_amount(), amount_usd * urgency_share)
                
                if allocated > 0:
                    # Update withdrawal and reserved amount
                    w.update_reserved(allocated)
    
    def process_withdrawals(self):
        """
        Process withdrawals that have reached their deadlines.
        """
        # Get withdrawals that are ready to be processed
        current_time = datetime.now()
        
        for withdrawal in self.withdrawals:
            # Skip completed or canceled withdrawals
            if withdrawal.status in [WithdrawalStatus.COMPLETE, WithdrawalStatus.CANCELED]:
                continue
                
            # Check if it's time to process
            days_left = withdrawal.get_days_until_deadline()
            is_deadline_reached = days_left is not None and days_left <= 0
            
            if is_deadline_reached or withdrawal.withdrawal_type == WithdrawalType.EMERGENCY:
                # Fulfill as much as possible from reserved balance
                remaining = withdrawal.get_remaining_amount()
                
                if remaining > 0 and self.usd_balance > 0:
                    fulfilled = min(self.usd_balance, remaining)
                    withdrawal.fulfill(fulfilled)
                    self.usd_balance -= fulfilled
                    self.usd_reserved -= min(self.usd_reserved, fulfilled)
    
    def get_active_withdrawals(self):
        """
        Get list of active withdrawal requests.
        
        Returns:
            list: Active Withdrawal objects
        """
        return [w for w in self.withdrawals 
                if w.status in [WithdrawalStatus.PENDING, WithdrawalStatus.PARTIAL]]
    
    def get_withdrawal_status(self, withdrawal_id):
        """
        Get status of a specific withdrawal.
        
        Args:
            withdrawal_id (str): Withdrawal ID
            
        Returns:
            tuple: (status, remaining_amount, fulfilled_amount)
        """
        for w in self.withdrawals:
            if w.withdrawal_id == withdrawal_id:
                return (w.status, w.get_remaining_amount(), w.fulfilled_amount)
        return None
    
    def cancel_withdrawal(self, withdrawal_id):
        """
        Cancel a withdrawal request.
        
        Args:
            withdrawal_id (str): Withdrawal ID
            
        Returns:
            bool: Success status
        """
        for w in self.withdrawals:
            if w.withdrawal_id == withdrawal_id:
                reserved_to_return = w.cancel()
                # Return reserved amount back to USDT balance
                if reserved_to_return > 0:
                    # Apply conversion fee for USD->USDT
                    conversion_fee = reserved_to_return * self.deposit_conversion_fee
                    usdt_returned = reserved_to_return - conversion_fee
                    
                    self.usdt_balance += usdt_returned
                    self.capital = self.usdt_balance  # Keep capital in sync
                    self.usd_balance -= reserved_to_return
                    self.usd_reserved -= min(self.usd_reserved, reserved_to_return)
                return True
        return False

    def _update_capital(self, net_profit):
        """Update capital based on net profit and handle withdrawal reservations"""
        # Original capital update functionality
        self.capital += net_profit
        self.returns.append(net_profit / self.initial_capital)
        
        # Update USDT balance
        self.usdt_balance = self.capital
        
        # If we made a profit, reserve some for withdrawals
        if net_profit > 0:
            self.reserve_profit_for_withdrawals(net_profit)
            
        # Process any pending withdrawals
        self.process_withdrawals()

    def _get_info(self):
        """Get info dictionary with metrics about the current state"""
        risk_metrics = self._calculate_portfolio_risk()
        position_count = len(self.positions)
        
        return {
            "capital": self.capital,
            "position_count": position_count,
            "risk_score": risk_metrics.get("overall_risk_score", 0.0)
        }
        
    def _extract_action(self, action):
        """
        Extract direction and fraction components from action.
        
        Args:
            action: Action from agent, typically [direction, fraction]
            
        Returns:
            tuple: (direction, fraction) components of the action
        """
        # Use the format_agent_action function from env_interfaces that we imported at the top
        # This is safer than importing here which might fail
        direction, fraction = format_agent_action(action)
        return direction, fraction
        
    def _execute_agent_order(self, direction, fraction):
        """
        Execute an order based on the agent's action.
        
        Args:
            direction (float): Direction of the order (-1 to 1, negative for sell, positive for buy)
            fraction (float): Fraction of capital to use (0 to 1)
        """
        if self.df is None or self.current_step >= len(self.df):
            return
            
        price = self._current_price()
        daily_volume = self.df.loc[self.current_step, "volume"] * 288  # Estimate daily volume
        
        # Handle buying (direction > 0.1)
        if direction > 0.1:
            # Skip buy orders if we've reached maximum positions
            if len(self.positions) >= self.max_positions:
                return
                
            # Calculate maximum size based on position limits
            max_size_btc = self._calculate_size_limits(price, daily_volume)
            
            usd = self.capital * fraction
            if usd > 1e-8:
                slippage = self._compute_slippage(usd, daily_volume)
                effective_price = price * (1 + slippage)
                
                # Calculate desired size and apply position limits
                desired_size_btc = usd / effective_price
                size_btc = min(desired_size_btc, max_size_btc)
                
                # Double-check final position value stays within USD limit (handles floating point precision issues)
                while size_btc * effective_price > max_usd_limit and size_btc > 1e-8:
                    size_btc = size_btc * 0.999  # Reduce by 0.1% until we're within limit
                
                actual_usd = size_btc * effective_price
                
                # Execute the full order - only if we haven't reached position limit
                if len(self.positions) < self.max_positions:
                    self._update_rolling_volume(self.current_step, actual_usd)
                    fee = self._calculate_fee(actual_usd)
                    cost = actual_usd + fee
                    self.capital -= min(cost, self.capital)
                    self.positions.append({
                        "size_btc": size_btc, 
                        "entry_price": effective_price, 
                        "entry_step": self.current_step
                    })
        
        # Handle selling (direction < -0.1)
        elif direction < -0.1 and self.positions:
            total_btc = sum(p["size_btc"] for p in self.positions)
            if total_btc > 1e-8:
                btc_to_sell = total_btc * fraction
                slippage = self._compute_slippage(btc_to_sell * price, daily_volume)
                effective_price = price * (1 - slippage)  # Sell price lower due to slippage
                raw_profit = 0.0
                
                # Process each position for selling
                for p in self.positions[:]:
                    if btc_to_sell <= 0:
                        break
                    sell_size = min(p["size_btc"], btc_to_sell)
                    trade_profit = sell_size * (effective_price - p["entry_price"])
                    raw_profit += trade_profit
                    p["size_btc"] -= sell_size
                    btc_to_sell -= sell_size
                    
                    # If position fully closed, record it
                    if p["size_btc"] <= 1e-8:
                        hold_time = self.current_step - p["entry_step"]
                        self.positions.remove(p)
                        percentage_gain = (effective_price / p["entry_price"] - 1) * 100
                        self.closed_trades.append((trade_profit, percentage_gain, hold_time))
                        if trade_profit > 0:
                            self.profits.append(trade_profit)
                        else:
                            self.losses.append(abs(trade_profit))
                
                # Calculate fees and final profit
                notional = (total_btc * fraction - btc_to_sell) * effective_price
                self._update_rolling_volume(self.current_step, notional)
                fee = self._calculate_fee(notional)
                net_profit = raw_profit - fee
                self.capital += net_profit
                self.returns.append(net_profit / self.initial_capital)

    def _update_positions(self, current_price):
        """
        Update position values and apply any position management rules.
        
        Args:
            current_price (float): Current market price
        """
        if not self.positions:
            return
            
        # Apply any position management rules like trailing stops, take-profit, etc.
        # This is a simplified implementation
        positions_to_remove = []
        
        for idx, position in enumerate(self.positions):
            # Check for any conditions that would require position updates
            # For example, update trailing stops based on new prices
            pass  # In this simplified version, we don't update anything
    
    def _manage_stop_losses(self, current_price):
        """
        Check and execute stop losses for open positions.
        
        Args:
            current_price (float): Current market price
        """
        if not self.positions:
            return
            
        # Default stop loss percentage (e.g., 5%)
        stop_loss_pct = 0.05
        
        positions_to_remove = []
        for idx, position in enumerate(self.positions):
            entry_price = position["entry_price"]
            
            # Check if current price has fallen below stop loss level
            if current_price < entry_price * (1 - stop_loss_pct):
                # Execute stop loss
                size_btc = position["size_btc"]
                trade_profit = size_btc * (current_price - entry_price)
                self.capital += size_btc * current_price  # Add position value back to capital
                
                # Record closed trade
                hold_time = self.current_step - position["entry_step"]
                percentage_gain = (current_price / entry_price - 1) * 100
                self.closed_trades.append((trade_profit, percentage_gain, hold_time))
                if trade_profit > 0:
                    self.profits.append(trade_profit)
                else:
                    self.losses.append(abs(trade_profit))
                    
                positions_to_remove.append(idx)
        
        # Remove stopped out positions (in reverse order to maintain correct indices)
        for idx in sorted(positions_to_remove, reverse=True):
            del self.positions[idx]
            
    def _calculate_reward(self):
        """
        Calculate the reward for the current step.
        
        Returns:
            float: Calculated reward value
        """
        # Calculate base reward (profit/loss in this step)
        # For simplicity, we'll just return 0 reward if there were no closed trades in this step
        if not hasattr(self, 'last_closed_trades_count'):
            self.last_closed_trades_count = 0
            
        # Check if we have new closed trades
        new_trades_count = len(self.closed_trades) - self.last_closed_trades_count
        
        if new_trades_count <= 0:
            # No new trades closed, no immediate reward
            return 0.0
            
        # Get profits from newly closed trades
        base_reward = sum([profit for profit, _, _ in self.closed_trades[-new_trades_count:]])
        
        # Update the count of processed trades
        self.last_closed_trades_count = len(self.closed_trades)
        
        # If we have a reward system, use it for more sophisticated reward calculation
        if hasattr(self, 'reward_system'):
            # Calculate episode days (assuming 288 5-min bars per day)
            episode_days = (self.current_step - self.window_size) / 288.0
            
            # Get risk metrics
            risk_metrics = self._calculate_portfolio_risk()
            
            # No prediction metrics in this implementation
            prediction_metrics = {}
            
            # Use the reward system to calculate the reward
            return self.reward_system.compute_reward(
                base_reward,
                self.profits,
                self.losses,
                self.returns,
                self.closed_trades,
                episode_days,
                risk_metrics,
                prediction_metrics
            )
        
        # If no reward system, just return the base reward
        return base_reward


class VecEnvWrapper:
    """
    Simple vectorized environment wrapper for parallel environments.
    
    Used when the true multiprocessing SubprocVecEnv is not available.
    """
    
    def __init__(self, env_fns):
        """
        Initialize vectorized environment.
        
        Args:
            env_fns (list): List of functions that create environments.
        """
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        logger.info(f"Created VecEnvWrapper with {self.num_envs} environments")
    
    def reset(self):
        """
        Reset all environments.
        
        Returns:
            list: List of initial observations from all environments.
        """
        return [env.reset() for env in self.envs]
    
    def step(self, actions):
        """
        Step all environments with the given actions.
        
        Args:
            actions (list): List of actions for each environment.
            
        Returns:
            tuple: (observations, rewards, dones, infos) - Lists for each environment.
        """
        observations = []
        rewards = []
        dones = []
        infos = []
        
        for env, action in zip(self.envs, actions):
            obs, reward, done, info = env.step(action)
            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        
        return observations, rewards, dones, infos
    
    def close(self):
        """Close all environments and release resources."""
        for env in self.envs:
            if hasattr(env, 'close'):
                env.close()
    
    def get_attr(self, attr_name):
        """
        Get an attribute from each environment.
        
        Args:
            attr_name (str): Name of attribute to get.
            
        Returns:
            list: List of attribute values from each environment.
        """
        return [getattr(env, attr_name) for env in self.envs]
    
    def env_method(self, method_name, *args, indices=None, **kwargs):
        """
        Call a method on each environment.
        
        Args:
            method_name (str): Name of method to call.
            *args: Positional arguments to pass to the method.
            indices (list, optional): List of indices of environments to call method on.
            **kwargs: Keyword arguments to pass to the method.
            
        Returns:
            list: List of method return values from each environment.
        """
        if indices is None:
            indices = range(self.num_envs)
        
        return [getattr(self.envs[i], method_name)(*args, **kwargs) for i in indices]


def create_environment(df=None, config=None):
    """Factory function to create the appropriate environment type
    
    Args:
        df (pandas.DataFrame, optional): DataFrame with market data. Can also be passed in config.
        config (dict, optional): Configuration parameters.
        
    Returns:
        Environment instance based on configuration.
    """
    # Default configuration
    config = config or {
        "environment_type": "base",
        "timeframe": "5min",
        "symbol": "BTCUSDT",
        "window_size": 60,
        "initial_capital": 10000,
        "max_positions": 5,
        "trading_mode": "Medium",  # Scalping, Short, Medium, Long
        "risk_level": "moderate",  # conservative, moderate, aggressive
        "reward_type": "sharpe",  # sharpe, sortino, calmar, omega, custom
        "reward_scaling": 1.0,
        "use_tensor": False
    }
    
    # If df was passed as first argument, use it
    if df is not None:
        config["df"] = df
    
    # Get parameters from config
    df = config.get("df", None)
    window_size = config.get("window_size", 60)
    initial_capital = config.get("initial_capital", 10000)
    max_positions = config.get("max_positions", 5)
    bucket = config.get("trading_mode", "Medium")
    use_tensor = config.get("use_tensor", False)
    
    if use_tensor:
        # Use tensor-optimized environment if requested and available
        try:
            # Import dynamically
            env_tensor_module = importlib.import_module("src.environment.env_tensor")
            TensorTradingEnv = env_tensor_module.TensorTradingEnv
            detect_gpu_availablity = env_tensor_module.detect_gpu_availablity
            
            has_gpu = detect_gpu_availablity()
            if has_gpu:
                return TensorTradingEnv(df, window_size, initial_capital, max_positions, bucket, config)
            else:
                print("GPU not available. Using CPU-based environment.")
        except ImportError:
            print("Tensor environment not available. Using base environment.")
    
    # Default to base environment
    return BaseTradingEnv(df, window_size, initial_capital, max_positions, bucket, config)


def make_env_creator(df, config, device="cpu"):
    """
    Create a function that initializes an environment.
    
    Useful for vectorized environments and multiprocessing.
    
    Args:
        df (pandas.DataFrame): DataFrame with market data.
        config (dict): Configuration parameters.
        device (str, optional): Device to run on. Defaults to "cpu".
        
    Returns:
        function: Environment creator function.
    """
    def _create_env():
        return create_environment(df, config, device)
    return _create_env


def torch_available():
    """Check if PyTorch is available"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# Register environments with the EnvRegistry
# EnvRegistry is already imported at the top of the file

# Register the base trading environment
EnvRegistry.register("base", create_environment)


# If run as a script, demonstrate environment creation and basic usage
if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    
    # Create a simple synthetic dataset for testing
    dates = pd.date_range('2020-01-01', periods=1000, freq='5min')
    df = pd.DataFrame({
        'timestamp': dates,
        'close': np.random.normal(50000, 1000, 1000),  # Random price around 50k
        'high': np.random.normal(50500, 1000, 1000),   # Slightly higher
        'low': np.random.normal(49500, 1000, 1000),    # Slightly lower
        'volume': np.random.exponential(10, 1000),     # Random volume with positive skew
        'SMA9': np.random.normal(50000, 500, 1000),    # Simple moving average
        'RSI14': np.random.normal(50, 10, 1000)        # RSI indicator
    })
    
    # Ensure prices are properly ordered (high >= close >= low)
    for i in range(len(df)):
        row = df.iloc[i]
        high = max(row['close'], row['high'])
        low = min(row['close'], row['low'])
        df.at[i, 'high'] = high
        df.at[i, 'low'] = low
    
    # Create a configuration
    config = {
        "WINDOW_SIZE": 50,           # Smaller window for testing
        "INITIAL_CAPITAL": 100000.0,
        "MAX_POSITION_HOLDINGS": 5,
        "BUCKET": "Scalping",
        "TENSOR_BASED_ENV": False    # Force base environment for testing
    }
    
    # Create environment
    env = create_environment(df, config)
    
    # Test basic functionality
    obs = env.reset()
    print(f"Observation shape: {obs.shape if obs is not None else None}")
    
    # Take random actions
    for _ in range(10):
        action = [np.random.uniform(-1, 1), np.random.uniform(0, 1)]
        obs, reward, done, info = env.step(action)
        print(f"Action: {action}, Reward: {reward}, Done: {done}")
        
        if done:
            break
    
    # Check final metrics
    print(f"Positions: {len(env.positions)}")
    print(f"Closed trades: {len(env.closed_trades)}")
    print(f"Final capital: ${env.capital:.2f}")
    
    # Test vectorized environment
    print("\nTesting vectorized environment...")
    env_fns = [make_env_creator(df, config) for _ in range(3)]
    vec_env = VecEnvWrapper(env_fns)
    
    # Reset all environments
    obs_list = vec_env.reset()
    print(f"Vectorized observations: {len(obs_list)}")
    
    # Take random actions in all environments
    actions = [[np.random.uniform(-1, 1), np.random.uniform(0, 1)] for _ in range(3)]
    obs_list, rewards, dones, infos = vec_env.step(actions)
    print(f"Vectorized rewards: {rewards}")
    
    # Close environments
    vec_env.close()
    print("Test complete. Environment implementation verified.")
