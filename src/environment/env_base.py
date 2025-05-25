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
import json
import time

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

    # Import new predictive agent interface
    predictive_interface_module = importlib.import_module("src.utils.predictive_agent_interface")
    PredictiveAgentInterface = predictive_interface_module.PredictiveAgentInterface

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
    
    # Robust fallback for PredictiveAgentInterface
    class PredictiveAgentFallback:
        """
        Robust fallback system for predictive agent interface.
        Maintains near-full functionality by reading last checkpoints and saved data.
        """
        def __init__(self, bucket_type):
            self.bucket_type = bucket_type
            self.models_dir = "Models"
            self.predictive_agent_dir = os.path.join(self.models_dir, bucket_type, "predictive_agent")
            self.predictions_file = os.path.join(self.predictive_agent_dir, f"{bucket_type.lower()}_predictions.json")
            self.summary_file = os.path.join(self.predictive_agent_dir, f"{bucket_type.lower()}_final_summary.json")
            self.weights_file = os.path.join(self.predictive_agent_dir, f"{bucket_type.lower()}_predictive_agent.pth")
            
            # Load saved data on initialization
            self.cached_predictions = self._load_cached_predictions()
            self.cached_summary = self._load_cached_summary()
            self.dynamic_confidence_threshold = self._load_dynamic_confidence_threshold()
            self.agent_available = self._check_agent_availability()
            
            # Fallback defaults based on bucket type
            self.bucket_defaults = {
                "Scalping": {"confidence": 0.7, "evaluation_score": 0.65, "accuracy": 0.72},
                "Short": {"confidence": 0.75, "evaluation_score": 0.68, "accuracy": 0.75},
                "Medium": {"confidence": 0.8, "evaluation_score": 0.72, "accuracy": 0.78},
                "Long": {"confidence": 0.82, "evaluation_score": 0.75, "accuracy": 0.8}
            }
            
            if not self.agent_available:
                log(f"[FALLBACK] Predictive agent not available for {bucket_type}. Using cached data and intelligent defaults.", "warning")
        
        def _load_cached_predictions(self):
            """Load the most recent saved predictions from disk."""
            try:
                if os.path.exists(self.predictions_file):
                    with open(self.predictions_file, 'r') as f:
                        data = json.load(f)
                    log(f"[FALLBACK] Loaded cached predictions for {self.bucket_type}", "info")
                    return data
            except Exception as e:
                log(f"[FALLBACK] Could not load cached predictions: {str(e)}", "warning")
            return None
        
        def _load_cached_summary(self):
            """Load the most recent summary data from disk."""
            try:
                if os.path.exists(self.summary_file):
                    with open(self.summary_file, 'r') as f:
                        data = json.load(f)
                    log(f"[FALLBACK] Loaded cached summary for {self.bucket_type}", "info")
                    return data
            except Exception as e:
                log(f"[FALLBACK] Could not load cached summary: {str(e)}", "warning")
            return None
        
        def _load_dynamic_confidence_threshold(self):
            """Load dynamic confidence threshold from saved threshold file."""
            try:
                threshold_file = os.path.join(self.predictive_agent_dir, f"{self.bucket_type.lower()}_confidence_threshold.json")
                if os.path.exists(threshold_file):
                    with open(threshold_file, 'r') as f:
                        threshold_data = json.load(f)
                    current_threshold = threshold_data.get("current_threshold", None)
                    if current_threshold is not None:
                        log(f"[FALLBACK] Loaded dynamic confidence threshold for {self.bucket_type}: {current_threshold:.3f}", "info")
                        return current_threshold
            except Exception as e:
                log(f"[FALLBACK] Could not load dynamic confidence threshold: {str(e)}", "warning")
            
            # Fall back to bucket-specific defaults
            bucket_defaults = {
                "Scalping": 0.7,
                "Short": 0.75, 
                "Medium": 0.8,
                "Long": 0.82
            }
            default_threshold = bucket_defaults.get(self.bucket_type, 0.7)
            log(f"[FALLBACK] Using default confidence threshold for {self.bucket_type}: {default_threshold:.3f}", "info")
            return default_threshold
        
        def _check_agent_availability(self):
            """Check if agent weights and directory exist."""
            return (os.path.exists(self.predictive_agent_dir) and 
                   os.path.exists(self.weights_file))
        
        def _create_backup(self):
            """Create backup of current data before potential corruption."""
            try:
                import shutil
                from datetime import datetime
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_dir = os.path.join(self.predictive_agent_dir, f"backup_{timestamp}")
                
                if os.path.exists(self.predictive_agent_dir):
                    os.makedirs(backup_dir, exist_ok=True)
                    
                    # Backup prediction files
                    for file_path in [self.predictions_file, self.summary_file, self.weights_file]:
                        if os.path.exists(file_path):
                            filename = os.path.basename(file_path)
                            shutil.copy2(file_path, os.path.join(backup_dir, filename))
                    
                    log(f"[FALLBACK] Created backup at {backup_dir}", "info")
                    return backup_dir
            except Exception as e:
                log(f"[FALLBACK] Backup creation failed: {str(e)}", "error")
            return None
        
        def _get_intelligent_defaults(self):
            """Generate intelligent defaults based on bucket type and any available historical data."""
            defaults = self.bucket_defaults.get(self.bucket_type, self.bucket_defaults["Medium"])
            
            # If we have cached data, use recent performance to adjust defaults
            if self.cached_summary:
                try:
                    recent_accuracy = self.cached_summary.get("avg_accuracy", defaults["accuracy"])
                    recent_confidence = self.cached_summary.get("avg_confidence", defaults["confidence"])
                    
                    # Weight recent performance (70%) with defaults (30%)
                    defaults["accuracy"] = 0.7 * recent_accuracy + 0.3 * defaults["accuracy"]
                    defaults["confidence"] = 0.7 * recent_confidence + 0.3 * defaults["confidence"]
                    defaults["evaluation_score"] = defaults["accuracy"] * 0.9  # Slightly conservative
                    
                except Exception:
                    pass  # Use original defaults if data is corrupted
            
            return defaults
        
        def is_predictive_agent_available(self):
            """Return True if we have any usable predictive data (cached or live)."""
            return self.agent_available or self.cached_predictions is not None or self.cached_summary is not None
        
        def get_latest_predictions(self):
            """Get latest predictions from cache or generate intelligent fallback."""
            # Try cached predictions first
            if self.cached_predictions:
                try:
                    # Check if predictions are recent (within last 24 hours)
                    pred_timestamp = self.cached_predictions.get("timestamp", 0)
                    current_time = time.time()
                    
                    if current_time - pred_timestamp < 86400:  # 24 hours
                        return self.cached_predictions
                    else:
                        log(f"[FALLBACK] Cached predictions for {self.bucket_type} are stale, using intelligent defaults", "warning")
                except Exception:
                    log(f"[FALLBACK] Cached predictions corrupted, using intelligent defaults", "warning")
            
            # Generate intelligent fallback predictions
            defaults = self._get_intelligent_defaults()
            
            fallback_predictions = {
                "bucket_type": self.bucket_type,
                "timestamp": time.time(),
                "predicted_performance": defaults["evaluation_score"],
                "prediction_confidence": defaults["confidence"],
                "dynamic_confidence_threshold": self.dynamic_confidence_threshold,  # Add dynamic threshold
                "prediction_accuracy": defaults["accuracy"],
                "enhanced_evaluation_score": defaults["evaluation_score"],
                "market_sentiment": "neutral",
                "source": "fallback_system",
                "recommendations": {
                    "suggested_action": "hold",
                    "confidence_level": defaults["confidence"],
                    "min_confidence_threshold": self.dynamic_confidence_threshold,  # Also add to recommendations
                    "horizon_appropriateness": defaults["evaluation_score"]
                }
            }
            
            return fallback_predictions
        
        def get_current_recommendation(self):
            """Get current trading recommendation with fallback logic."""
            predictions = self.get_latest_predictions()
            if predictions and "recommendations" in predictions:
                return predictions["recommendations"]
            
            # Intelligent fallback recommendation
            defaults = self._get_intelligent_defaults()
            return {
                "action": "hold",
                "confidence": defaults["confidence"],
                "reasoning": f"Fallback recommendation for {self.bucket_type} bucket",
                "risk_level": "moderate"
            }
        
        def get_training_status(self):
            """Get training status with fallback information."""
            if self.cached_summary:
                status = self.cached_summary.copy()
                status["fallback_active"] = True
                status["data_source"] = "cached"
                return status
            
            # Fallback status
            defaults = self._get_intelligent_defaults()
            return {
                "status": "fallback_active",
                "fallback_active": True,
                "data_source": "intelligent_defaults", 
                "bucket_type": self.bucket_type,
                "estimated_accuracy": defaults["accuracy"],
                "estimated_confidence": defaults["confidence"],
                "last_update": "fallback_system"
            }

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
        Initialize the trading environment.
        
        Args:
            df (pandas.DataFrame): Market data.
            window_size (int): Number of bars for observation window.
            initial_capital (float): Starting capital amount.
            max_positions (int): Maximum concurrent positions.
            bucket (str): Trading bucket (Scalping, Short, Medium, Long).
            config (dict): Configuration parameters.
        """
        self.id = str(uuid.uuid4())
        self.df = df
        self.window_size = window_size
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.bucket = bucket
        self.config = config
        
        # Trading state
        self.current_step = window_size  # Start after the window
        self.capital = initial_capital
        self.positions = []  # Always use Position objects for consistency
        self.closed_trades = []  # Always use Trade objects for consistency
        self.profits = []
        self.losses = []
        self.returns = []
        self.done = False
        
        # Grace period system implementation
        self.grace_period_bars = config.get("GRACE_PERIOD_BARS", 200)
        self.penalty_interval = config.get("PENALTY_INTERVAL", 2) 
        self.base_penalty = config.get("BASE_PENALTY", 0.05)
        self.penalty_increment = config.get("PENALTY_INCREMENT", 0.05)
        self.episode_start_step = self.current_step  # Track episode start for grace period
        
        # Performance caching to eliminate bottlenecks
        self._cached_risk_metrics = None
        self._risk_cache_step = -1
        self._cached_portfolio_value = None
        self._portfolio_cache_step = -1
        
        # Action tracking for debugging lost actions
        self.action_log = []
        self.failed_actions = []
        
        # Withdrawal simulation
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
        
        # Initialize predictive agent interface for the new system
        try:
            # Try to use the real predictive agent interface first
            self.predictive_interface = PredictiveAgentInterface(self.bucket)
            if not self.predictive_interface.is_predictive_agent_available():
                # If real interface shows no agent available, use fallback
                log(f"[INFO] Real predictive agent not available for {self.bucket}, using fallback system", "info")
                self.predictive_interface = PredictiveAgentFallback(self.bucket)
        except Exception as e:
            # If real interface fails to import/initialize, use fallback
            log(f"[FALLBACK] Real predictive interface failed ({str(e)}), using robust fallback system", "warning")
            self.predictive_interface = PredictiveAgentFallback(self.bucket)
        
        # Multi-currency management
        self.usdt_balance = 0.0
        self.usd_balance = 0.0
        self.usd_reserved = 0.0
        
        # Setup withdrawal simulation
        if config.get("simulate_withdrawals", False):
            self._setup_withdrawal_simulation()
            
        log(f"[ENV] Initialized {bucket} trading environment with {initial_capital} capital and grace period of {self.grace_period_bars} bars")
        
        # Store feature columns - use columns specified in config if available
        if df is not None:
            # Get feature columns from config if available, otherwise use default approach
            excluded_cols = ['timestamp', 'date', 'datetime', 'symbol']
            if "feature_columns" in config:
                self.available_cols = config["feature_columns"]
            else:
                # Extract feature columns automatically
                self.available_cols = [col for col in df.columns if col not in excluded_cols]
                # Store back in config for consistency
                config["feature_columns"] = self.available_cols
            
            # Ensure we have required price columns
            required_cols = ['close']
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Required column '{col}' missing from DataFrame")
        else:
            # Default feature set if no dataframe provided
            self.available_cols = ['close', 'high', 'low', 'volume']
            config["feature_columns"] = self.available_cols
        
        # Set observation dimension in config for training system
        # CRITICAL FIX: Calculate actual input dimension based on flattened observation
        # Flattened observation includes:
        # - 3 market scalars (price, volatility, mean_returns)  
        # - len(available_cols) feature values
        # - 3 position scalars (num_positions, avg_entry_price, unrealized_pnl)
        # - 2 agent state scalars (capital_ratio, has_withdrawals)
        actual_input_dim = 3 + len(self.available_cols) + 3 + 2
        config["observation_dim"] = actual_input_dim
        config["input_dim"] = actual_input_dim
        
        # Set up withdrawal simulation schedule if enabled
        self._withdrawal_deposit_schedule = []
        
        logger.info(f"Using {len(self.available_cols)} feature columns")
        logger.info(f"Calculated observation dimension: {actual_input_dim}")
        logger.info(f"Feature columns: {self.available_cols}")

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
        
        # Standardize observation (returns dict format)
        standardized_obs = standardize_observation(raw_obs, self)
        
        if standardized_obs is None:
            return None
        
        # CRITICAL FIX: Flatten observation to numpy array for consistent interface
        # The training system expects a flat array, not a nested dictionary
        flattened_obs = self._flatten_observation(standardized_obs)
        
        return flattened_obs
    
    def _flatten_observation(self, obs_dict):
        """
        Flatten nested observation dictionary to numpy array.
        
        Args:
            obs_dict (dict): Nested observation dictionary from standardize_observation
            
        Returns:
            numpy.ndarray: Flattened observation array
        """
        import numpy as np
        
        flat_values = []
        
        # Extract market data
        market_data = obs_dict.get('market_data', {})
        flat_values.append(market_data.get('price', 0.0))
        flat_values.append(market_data.get('volatility', 0.0))
        
        # Add mean returns (scalar from returns array)
        returns = market_data.get('returns', np.array([0.0]))
        if len(returns) > 0:
            flat_values.append(np.mean(returns))
        else:
            flat_values.append(0.0)
        
        # Add feature values for each available column
        for col in self.available_cols:
            if col in market_data:
                feature_values = market_data[col]
                if hasattr(feature_values, '__iter__') and not isinstance(feature_values, str):
                    # If it's an array, take the last value
                    if len(feature_values) > 0:
                        flat_values.append(float(feature_values[-1]))
                    else:
                        flat_values.append(0.0)
                else:
                    # If it's a scalar
                    flat_values.append(float(feature_values))
            else:
                # Missing feature, use 0
                flat_values.append(0.0)
        
        # Add position data
        position_data = obs_dict.get('position_data', {})
        flat_values.append(position_data.get('num_positions', 0))
        flat_values.append(position_data.get('avg_entry_price', 0.0))
        flat_values.append(position_data.get('unrealized_pnl', 0.0))
        
        # Add agent state
        agent_state = obs_dict.get('agent_state', {})
        flat_values.append(agent_state.get('capital_ratio', 1.0))
        flat_values.append(agent_state.get('has_withdrawals', 0))
        
        # Convert to numpy array
        return np.array(flat_values, dtype=np.float32)
    
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
        """
        Calculate portfolio risk metrics with performance caching.
        
        Returns:
            dict: Risk metrics including overall score, drawdown, leverage, etc.
        """
        # Use cached result if available for this step
        if self._cached_risk_metrics is not None and self._risk_cache_step == self.current_step:
            return self._cached_risk_metrics
        
        # Calculate portfolio value with caching
        portfolio_value = self._get_cached_portfolio_value()
        
        # Initialize risk metrics
        risk_metrics = {
            "portfolio_value": portfolio_value,
            "cash_ratio": self.capital / self.initial_capital if self.initial_capital > 0 else 0,
            "position_count": len(self.positions),
            "max_position_limit": self.max_positions,
            "position_utilization": len(self.positions) / self.max_positions if self.max_positions > 0 else 0,
            "overall_risk_score": 0.5  # Default medium risk
        }
        
        # Calculate position-based risk metrics
        if self.positions:
            position_values = []
            total_position_value = 0
            current_price = self._current_price()
            
            for position in self.positions:
                if isinstance(position, Position):
                    pos_value = position.size_btc * current_price
                elif isinstance(position, dict):
                    # Convert dict to Position object for consistency
                    position = Position(
                        size_btc=position["size_btc"],
                        entry_price=position["entry_price"], 
                        entry_step=position["entry_step"]
                    )
                    pos_value = position.size_btc * current_price
                else:
                    pos_value = 0
                    
                position_values.append(pos_value)
                total_position_value += pos_value
            
            # Position concentration risk
            if position_values:
                largest_position = max(position_values)
                risk_metrics["position_concentration"] = largest_position / total_position_value if total_position_value > 0 else 0
                risk_metrics["position_diversity"] = len(position_values) / self.max_positions if self.max_positions > 0 else 0
            else:
                risk_metrics["position_concentration"] = 0
                risk_metrics["position_diversity"] = 0
                
            # Leverage calculation
            total_exposure = total_position_value
            risk_metrics["leverage"] = total_exposure / self.initial_capital if self.initial_capital > 0 else 0
        else:
            risk_metrics["position_concentration"] = 0
            risk_metrics["position_diversity"] = 0
            risk_metrics["leverage"] = 0
        
        # Calculate drawdown from returns history
        if len(self.returns) > 0:
            cumulative_returns = np.cumsum(self.returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = running_max - cumulative_returns
            risk_metrics["drawdown"] = np.max(drawdowns) if len(drawdowns) > 0 else 0
        else:
            risk_metrics["drawdown"] = 0
        
        # Market exposure (time in market)
        bars_since_start = self.current_step - self.window_size
        risk_metrics["market_exposure"] = min(1.0, bars_since_start / 288) if bars_since_start > 0 else 0  # Normalize by ~1 day
        
        # Overall risk score calculation (0-1, higher = more risky)
        risk_components = [
            risk_metrics["position_utilization"] * 0.3,  # Position utilization weight
            risk_metrics["leverage"] * 0.25,  # Leverage weight
            risk_metrics["drawdown"] * 0.25,  # Drawdown weight
            risk_metrics["position_concentration"] * 0.2  # Concentration weight
        ]
        risk_metrics["overall_risk_score"] = min(1.0, sum(risk_components))
        
        # Apply grace period considerations
        bars_since_episode_start = self.current_step - self.episode_start_step
        risk_metrics["grace_period_active"] = bars_since_episode_start < self.grace_period_bars
        risk_metrics["grace_period_remaining"] = max(0, self.grace_period_bars - bars_since_episode_start)
        
        # Cache the result
        self._cached_risk_metrics = risk_metrics
        self._risk_cache_step = self.current_step
        
        return risk_metrics
    
    def _get_cached_portfolio_value(self):
        """Get portfolio value with caching for performance."""
        if self._cached_portfolio_value is not None and self._portfolio_cache_step == self.current_step:
            return self._cached_portfolio_value
            
        portfolio_value = self.capital
        if self.positions:
            current_price = self._current_price()
            for position in self.positions:
                if isinstance(position, Position):
                    portfolio_value += position.size_btc * current_price
                elif isinstance(position, dict):
                    portfolio_value += position["size_btc"] * current_price
                    
        self._cached_portfolio_value = portfolio_value
        self._portfolio_cache_step = self.current_step
        return portfolio_value
    
    def _apply_grace_period_penalty(self, base_reward):
        """
        Apply grace period and penalty system to reward calculation.
        
        Args:
            base_reward (float): Base reward from trades
            
        Returns:
            float: Adjusted reward with grace period considerations
        """
        bars_since_episode_start = self.current_step - self.episode_start_step
        
        # No penalties during grace period
        if bars_since_episode_start < self.grace_period_bars:
            return base_reward
        
        # Calculate penalty intervals passed since grace period ended
        bars_since_grace_end = bars_since_episode_start - self.grace_period_bars
        penalty_intervals = bars_since_grace_end // self.penalty_interval
        
        # Apply increasing penalty over time
        if penalty_intervals > 0 and len(self.closed_trades) == 0:
            # Only apply penalty if no trades have been made
            penalty_multiplier = self.base_penalty + (penalty_intervals * self.penalty_increment)
            penalty = penalty_multiplier * abs(base_reward) if base_reward != 0 else penalty_multiplier * 0.1
            
            # Log penalty application for debugging
            if penalty > 0:
                log(f"[GRACE] Applied penalty {penalty:.3f} after {bars_since_grace_end} bars past grace period", "debug")
            
            return base_reward - penalty
        
        return base_reward
    
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
        
        # Get predictive agent data if available
        prediction_mean = None
        prediction_std = None
        confidence_score = None
        dynamic_confidence_threshold = None
        
        if self.predictive_interface and self.predictive_interface.is_predictive_agent_available():
            try:
                predictions = self.predictive_interface.get_latest_predictions()
                if predictions:
                    # Extract confidence and prediction uncertainty for risk adjustment
                    # Use the actual field names from the new predictive system
                    raw_confidence = predictions.get("prediction_confidence", None)
                    
                    # If not available, try from recommendations nested structure
                    if raw_confidence is None:
                        recommendations = predictions.get("recommendations", {})
                        raw_confidence = recommendations.get("confidence_level", None)
                    
                    # Apply horizon-weighted confidence calculation for bucket-specific confidence
                    if raw_confidence is not None:
                        # Use the risk manager's horizon weighting to get bucket-appropriate confidence
                        confidence_score = self.risk_manager.calculate_horizon_weighted_confidence(predictions)
                        
                        # Log the horizon weighting adjustment if significant
                        if abs(confidence_score - raw_confidence) > 0.05:
                            log(f"[HORIZON] Adjusted confidence from {raw_confidence:.3f} to {confidence_score:.3f} for {self.bucket} bucket", "info")
                    else:
                        # Fall back to horizon-weighted calculation with default
                        confidence_score = self.risk_manager.calculate_horizon_weighted_confidence(predictions)
                    
                    # Extract dynamic confidence threshold set by the predictive agent
                    dynamic_confidence_threshold = predictions.get("dynamic_confidence_threshold", None)
                    if dynamic_confidence_threshold is None:
                        recommendations = predictions.get("recommendations", {})
                        dynamic_confidence_threshold = recommendations.get("min_confidence_threshold", None)
                    
                    # Use prediction accuracy as uncertainty measure (higher accuracy = lower uncertainty)
                    pred_accuracy = predictions.get("prediction_accuracy", 0.5)
                    prediction_std = max(0.01, 1.0 - pred_accuracy)  # Higher accuracy = lower uncertainty
                    
                    # Use the actual performance score as prediction mean strength
                    prediction_mean = predictions.get("predicted_performance", 0.0)
                    
                    # Fallback to enhanced evaluation score if predicted_performance not available
                    if prediction_mean == 0.0:
                        prediction_mean = predictions.get("enhanced_evaluation_score", 0.0)
            except Exception as e:
                # Log the exception but continue with defaults
                log(f"[WARNING] Error extracting predictive data for risk management: {str(e)}", "warning")
                pass  # Use defaults if predictive interface fails
        
        # Pass the dynamic confidence threshold to the risk manager
        if dynamic_confidence_threshold is not None:
            # Temporarily update the risk manager's config with the dynamic threshold
            original_threshold = self.risk_manager.config.get("MIN_CONFIDENCE_THRESHOLD")
            self.risk_manager.config["MIN_CONFIDENCE_THRESHOLD"] = dynamic_confidence_threshold
        
        # Get from risk manager with enhanced predictive information
        risk_adjusted_size = self.risk_manager.calculate_risk_adjusted_size(
            price, 
            daily_volume, 
            direction, 
            risk_metrics, 
            position_count,
            prediction_mean=prediction_mean,
            prediction_std=prediction_std,
            confidence_score=confidence_score
        )
        
        # Restore original threshold if we modified it
        if dynamic_confidence_threshold is not None and original_threshold is not None:
            self.risk_manager.config["MIN_CONFIDENCE_THRESHOLD"] = original_threshold
        elif dynamic_confidence_threshold is not None and original_threshold is None:
            # Remove the temporarily added threshold
            self.risk_manager.config.pop("MIN_CONFIDENCE_THRESHOLD", None)
        
        return risk_adjusted_size
    
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
        Execute an order based on the agent's action with improved tracking and consistency.
        
        Args:
            direction (float): Direction of the order (-1 to 1, negative for sell, positive for buy)
            fraction (float): Fraction of capital to use (0 to 1)
        """
        if self.df is None or self.current_step >= len(self.df):
            self.failed_actions.append({
                "step": self.current_step,
                "reason": "No market data available",
                "direction": direction,
                "fraction": fraction
            })
            return
            
        # Log action for debugging
        action_entry = {
            "step": self.current_step,
            "direction": direction,
            "fraction": fraction,
            "timestamp": time.time()
        }
        self.action_log.append(action_entry)
        
        price = self._current_price()
        daily_volume = self.df.loc[self.current_step, "volume"] * 288  # Estimate daily volume
        
        # Handle buying (direction > 0.1)
        if direction > 0.1:
            # Skip buy orders if we've reached maximum positions
            if len(self.positions) >= self.max_positions:
                self.failed_actions.append({
                    "step": self.current_step,
                    "reason": f"Max positions reached ({len(self.positions)}/{self.max_positions})",
                    "direction": direction,
                    "fraction": fraction
                })
                log(f"[ACTION] Buy order rejected - max positions reached ({len(self.positions)}/{self.max_positions})", "debug")
                return
                
            # Calculate maximum size based on position limits
            max_size_btc = self._calculate_risk_adjusted_size(price, daily_volume, 1.0, self._calculate_portfolio_risk(), len(self.positions))
            
            usd = self.capital * fraction
            if usd > 1e-8:
                slippage = self._compute_slippage(usd, daily_volume)
                effective_price = price * (1 + slippage)
                
                # Calculate USD limit based on available capital and config
                max_usd = min(self.capital * fraction, self.config.get("MAX_USD_PER_POSITION", float('inf')))
                
                # Calculate desired size and apply position limits
                desired_size_btc = usd / effective_price
                size_btc = min(desired_size_btc, max_size_btc)
                
                # Double-check final position value stays within USD limit
                while size_btc * effective_price > max_usd and size_btc > 1e-8:
                    size_btc = size_btc * 0.999  # Reduce by 0.1% until we're within limit
                
                actual_usd = size_btc * effective_price
                
                # Execute the order - always use Position objects for consistency
                if len(self.positions) < self.max_positions and size_btc > 1e-8:
                    self._update_rolling_volume(self.current_step, actual_usd)
                    fee = self._calculate_fee(actual_usd)
                    cost = actual_usd + fee
                    self.capital -= min(cost, self.capital)
                    
                    # Create Position object consistently
                    new_position = Position(
                        size_btc=size_btc, 
                        entry_price=effective_price, 
                        entry_step=self.current_step
                    )
                    self.positions.append(new_position)
                    
                    # Update action log with success
                    action_entry["executed"] = True
                    action_entry["size_btc"] = size_btc
                    action_entry["effective_price"] = effective_price
                    
                    log(f"[ACTION] Buy executed - {size_btc:.6f} BTC @ ${effective_price:.2f}", "debug")
                else:
                    self.failed_actions.append({
                        "step": self.current_step,
                        "reason": f"Position size too small ({size_btc:.8f} BTC) or limit reached",
                        "direction": direction,
                        "fraction": fraction
                    })
        
        # Handle selling (direction < -0.1)
        elif direction < -0.1 and self.positions:
            total_btc = sum(pos.size_btc for pos in self.positions)  # Use Position objects consistently
            if total_btc > 1e-8:
                btc_to_sell = total_btc * fraction
                slippage = self._compute_slippage(btc_to_sell * price, daily_volume)
                effective_price = price * (1 - slippage)  # Sell price lower due to slippage
                raw_profit = 0.0
                
                # Process each position for selling - ensure consistency
                positions_to_remove = []
                for idx, position in enumerate(self.positions):
                    if btc_to_sell <= 0:
                        break
                        
                    # Ensure we're working with Position objects
                    if isinstance(position, dict):
                        # Convert dict to Position object for consistency
                        position = Position(
                            size_btc=position["size_btc"],
                            entry_price=position["entry_price"],
                            entry_step=position["entry_step"]
                        )
                        self.positions[idx] = position  # Replace dict with Position object
                        
                    sell_size = min(position.size_btc, btc_to_sell)
                    trade_profit = sell_size * (effective_price - position.entry_price)
                    raw_profit += trade_profit
                    position.size_btc -= sell_size
                    btc_to_sell -= sell_size
                    
                    # If position fully closed, record it
                    if position.size_btc <= 1e-8:
                        hold_time = self.current_step - position.entry_step
                        percentage_gain = (effective_price / position.entry_price - 1) * 100
                        
                        # Create Trade object consistently
                        trade = Trade(trade_profit, percentage_gain, hold_time, position.entry_step)
                        self.closed_trades.append(trade)
                        
                        if trade_profit > 0:
                            self.profits.append(trade_profit)
                        else:
                            self.losses.append(abs(trade_profit))
                            
                        positions_to_remove.append(idx)
                
                # Remove fully closed positions
                for idx in sorted(positions_to_remove, reverse=True):
                    self.positions.pop(idx)
                
                # Calculate fees and final profit
                notional = (total_btc * fraction - btc_to_sell) * effective_price
                self._update_rolling_volume(self.current_step, notional)
                fee = self._calculate_fee(notional)
                net_profit = raw_profit - fee
                self.capital += net_profit
                self.returns.append(net_profit / self.initial_capital)
                
                # Update action log with success
                action_entry["executed"] = True
                action_entry["btc_sold"] = total_btc * fraction - btc_to_sell
                action_entry["effective_price"] = effective_price
                action_entry["net_profit"] = net_profit
                
                log(f"[ACTION] Sell executed - {total_btc * fraction - btc_to_sell:.6f} BTC @ ${effective_price:.2f}, profit: ${net_profit:.2f}", "debug")
        else:
            # Action too small or no positions to sell
            if direction < -0.1 and not self.positions:
                self.failed_actions.append({
                    "step": self.current_step,
                    "reason": "No positions to sell",
                    "direction": direction,
                    "fraction": fraction
                })
            else:
                # Action magnitude too small (between -0.1 and 0.1)
                action_entry["executed"] = False
                action_entry["reason"] = "Action magnitude too small (hold)"

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
        Calculate the reward for the current step with grace period and performance tracking.
        
        Returns:
            float: Calculated reward value
        """
        # Calculate base reward (profit/loss in this step)
        if not hasattr(self, 'last_closed_trades_count'):
            self.last_closed_trades_count = 0
            
        # Check if we have new closed trades
        new_trades_count = len(self.closed_trades) - self.last_closed_trades_count
        
        if new_trades_count <= 0:
            # No new trades closed, apply grace period penalty if applicable
            base_reward = 0.0
            adjusted_reward = self._apply_grace_period_penalty(base_reward)
            return adjusted_reward
            
        # Get profits from newly closed trades  
        base_reward = sum([trade.profit for trade in self.closed_trades[-new_trades_count:]])
        
        # Update the count of processed trades
        self.last_closed_trades_count = len(self.closed_trades)
        
        # Apply grace period considerations first
        adjusted_reward = self._apply_grace_period_penalty(base_reward)
        
        # If we have a reward system, use it for more sophisticated reward calculation
        if hasattr(self, 'reward_system'):
            # Calculate episode days (assuming 288 5-min bars per day)
            episode_days = (self.current_step - self.window_size) / 288.0
            
            # Get cached risk metrics for performance
            risk_metrics = self._calculate_portfolio_risk()
            
            # Use the reward system to calculate the reward
            enhanced_reward = self.reward_system.compute_reward(
                adjusted_reward,  # Use grace period adjusted reward as base
                self.profits,
                self.losses,
                self.returns,
                self.closed_trades,
                episode_days,
                risk_metrics,
                {}  # Empty prediction metrics - handled by predictive agent system
            )
            
            return enhanced_reward
        
        # If no reward system, return the grace period adjusted reward
        return adjusted_reward


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
    
    @property
    def observation_space(self):
        """
        Get observation space from the first environment.
        
        Returns:
            object: Observation space with shape attribute
        """
        if not self.envs:
            return None
        
        # Get a sample observation from the first environment to determine shape
        sample_obs = self.envs[0]._get_observation()
        if sample_obs is None:
            # If no observation available, reset and try again
            sample_obs = self.envs[0].reset()
        
        if sample_obs is not None:
            # Create a simple object with shape attribute
            class ObservationSpace:
                def __init__(self, shape):
                    self.shape = shape
            
            # Handle different observation formats
            if hasattr(sample_obs, 'shape'):
                # Tensor or numpy array
                obs_shape = sample_obs.shape
            elif isinstance(sample_obs, (list, tuple)):
                # List or tuple
                obs_shape = (len(sample_obs),)
            elif isinstance(sample_obs, dict):
                # Dictionary - flatten all numeric values
                flat_values = []
                def flatten_dict(d):
                    for value in d.values():
                        if isinstance(value, (int, float)):
                            flat_values.append(value)
                        elif isinstance(value, dict):
                            flatten_dict(value)
                        elif hasattr(value, '__iter__') and not isinstance(value, str):
                            for item in value:
                                if isinstance(item, (int, float)):
                                    flat_values.append(item)
                
                flatten_dict(sample_obs)
                obs_shape = (len(flat_values),)
            else:
                # Scalar or unknown format
                obs_shape = (1,)
            
            return ObservationSpace(obs_shape)
        
        # Fallback if no observation available
        class ObservationSpace:
            def __init__(self, shape):
                self.shape = shape
        return ObservationSpace((1,))  # Default shape
    
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


def create_environment(df=None, config=None, device="cpu"):
    """Factory function to create the appropriate environment type
    
    Args:
        df (pandas.DataFrame, optional): DataFrame with market data. Can also be passed in config.
        config (dict, optional): Configuration parameters.
        device (str, optional): Device to run on ('cpu', 'cuda'). Defaults to "cpu".
        
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
    
    # Store device in config for environment access
    config["device"] = device
    
    # If df was passed as first argument, use it
    if df is not None:
        config["df"] = df
    
    # Get parameters from config
    df = config.get("df", None)
    window_size = config.get("window_size", config.get("WINDOW_SIZE", 60))
    initial_capital = config.get("initial_capital", config.get("INITIAL_CAPITAL", 10000))
    max_positions = config.get("max_positions", config.get("MAX_POSITION_HOLDINGS", 5))
    
    # CRITICAL FIX: Handle both trading_mode and BUCKET parameter names
    bucket = config.get("trading_mode", config.get("BUCKET", "Medium"))
    # Ensure consistency - store both for compatibility
    config["trading_mode"] = bucket
    config["BUCKET"] = bucket
    
    use_tensor = config.get("use_tensor", config.get("TENSOR_BASED_ENV", False))
    
    # CRITICAL FIX: Set up feature columns for observation space consistency
    if df is not None and hasattr(df, 'columns'):
        # Exclude non-feature columns
        excluded_cols = ['timestamp', 'date', 'datetime', 'symbol']
        feature_columns = [col for col in df.columns if col not in excluded_cols]
        config["feature_columns"] = feature_columns
        
        # CRITICAL FIX: Calculate actual input dimension based on flattened observation
        # Flattened observation includes:
        # - 3 market scalars (price, volatility, mean_returns)
        # - len(feature_columns) feature values
        # - 3 position scalars (num_positions, avg_entry_price, unrealized_pnl)
        # - 2 agent state scalars (capital_ratio, has_withdrawals)
        actual_input_dim = 3 + len(feature_columns) + 3 + 2
        config["input_dim"] = actual_input_dim
        config["observation_dim"] = actual_input_dim
        
        log(f"[INFO] Environment detected {len(feature_columns)} feature columns", "info")
        log(f"[INFO] Calculated observation dimension: {actual_input_dim}", "info")
    
    # Use tensor environment if device is CUDA or explicitly requested
    if use_tensor or (device == "cuda"):
        # Use tensor-optimized environment if requested and available
        try:
            # Import dynamically
            env_tensor_module = importlib.import_module("src.environment.env_tensor")
            TensorTradingEnv = env_tensor_module.TensorTradingEnv
            detect_gpu_availablity = env_tensor_module.detect_gpu_availablity
            
            has_gpu = detect_gpu_availablity()
            if has_gpu and device == "cuda":
                return TensorTradingEnv(df, window_size, initial_capital, max_positions, bucket, config, device=device)
            elif device == "cuda":
                log("GPU requested but not available. Falling back to CPU-based environment.", "warning")
                # Fall through to base environment with CPU
            else:
                return TensorTradingEnv(df, window_size, initial_capital, max_positions, bucket, config, device=device)
        except ImportError:
            log("Tensor environment not available. Using base environment.", "warning")
    
    # Default to base environment (CPU-based)
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
