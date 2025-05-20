#!/usr/bin/env python
"""
Backtesting Module

This module provides functionality for backtesting trading strategies.
"""

import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
import json
import csv
import importlib
from typing import Dict, List, Any, Tuple, Optional, Union, Callable
import gc
import time
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("backtesting")

# Add the project root to the Python path if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Core utility imports
try:
    from src.utils.utils import (
        log, 
        validate_dataframe, 
        calculate_metrics, 
        format_metrics,
        optimize_memory, 
        visualize_metrics
    )
except ImportError as e:
    logger.error(f"Critical error importing core utilities: {str(e)}")
    logger.error("Backtesting requires these utilities to function properly.")
    raise ImportError(f"Failed to import core utilities: {str(e)}")

# Configuration imports
try:
    from src.utils.trade_config import get_trade_config, TradeConfig
    trade_config = get_trade_config()
    logger.info("Successfully loaded trade_config for backtesting")
except ImportError as e:
    logger.warning(f"Could not import trade_config: {str(e)}")
    logger.warning("Attempting to load default preset configuration instead")
    
    def _load_default_preset(bucket: str = "Scalping") -> Dict[str, Any]:
        """Load the default preset parameters for the given bucket."""
        try:
            from src.ui import preset_manager
        except Exception as e:
            logger.error(f"Failed to import preset_manager: {e}")
            return {}

        bucket_presets = preset_manager.DEFAULT_PRESETS.get(bucket, {})
        for _name, data in bucket_presets.items():
            params = data.get("params")
            if params:
                return params
        return {}

    try:
        # Get default config for Scalping (most conservative)
        preset_config = _load_default_preset("Scalping")
        
        # Create a TradeConfig-like interface with preset data
        class PresetBasedConfig:
            def __init__(self, preset_data):
                self.config = preset_data or {
                    "INITIAL_CAPITAL": 100000,
                    "RISK_LEVEL": "medium",
                    "MAX_POSITION_SIZE": 0.1,
                    "BUCKET": "Scalping"
                }
                
            def get(self, key, default=None):
                return self.config.get(key, default)
                
            def as_dict(self):
                return self.config.copy()
                
            def __getitem__(self, key):
                return self.config.get(key)
                
            def get_section(self, section):
                return {k: v for k, v in self.config.items() if k.startswith(section.upper())}
        
        trade_config = PresetBasedConfig(preset_config)
        logger.warning("Using preset-based configuration as fallback")
    except ImportError as e:
        logger.error(f"Failed to load preset configuration: {str(e)}")
        logger.error("Backtesting will use minimal default values which may affect results")
        
        # Create very minimal config with essential defaults only
        class MinimalConfig:
            def __init__(self):
                self.config = {
                    "INITIAL_CAPITAL": 100000,
                    "RISK_LEVEL": "medium",
                    "MAX_POSITION_SIZE": 0.1,
                    "BUCKET": "Scalping"
                }
                
            def get(self, key, default=None):
                return self.config.get(key, default)
                
            def as_dict(self):
                return self.config.copy()
                
            def __getitem__(self, key):
                return self.config.get(key)
                
            def get_section(self, section):
                return {k: v for k, v in self.config.items() if k.startswith(section.upper())}
        
        trade_config = MinimalConfig()
        logger.error("Using absolute minimal configuration - results may be unreliable")

# Legacy compatibility function
def get_config():
    """Returns trade_config for backward compatibility with improved functionality."""
    return trade_config

# Environment imports
try:
    from src.environment.env_base import create_environment
except ImportError as e:
    logger.error(f"Critical error importing environment module: {str(e)}")
    logger.error("Backtesting requires the environment module to function.")
    raise ImportError(f"Failed to import environment module: {str(e)}")

# Tensor utilities imports
try:
    from src.utils.tensor_utils import (
        compute_fractal_dimension_tensor, 
        detect_elliott_wave_pattern_tensor,
        compute_market_fractals_tensor, 
        compute_timeframe_wavelet_features
    )
except ImportError as e:
    logger.error(f"Error importing tensor utilities: {str(e)}")
    logger.error("Advanced features requiring tensor utilities will be unavailable.")
    raise ImportError(f"Failed to import tensor utilities: {str(e)}")

# Agent imports
try:
    from src.agent.agent import PPOAgent
except ImportError as e:
    logger.error(f"Critical error importing agent module: {str(e)}")
    logger.error("Backtesting requires the agent module to function.")
    raise ImportError(f"Failed to import agent module: {str(e)}")

# Continue with the rest of the file...

def run_backtest(df, agent, config, episodes=1, log_callback=None):
    """
    Run backtest with a trained agent.
    
    Args:
        df (pandas.DataFrame): DataFrame with market data.
        agent (PPOAgent): Trained agent for decision making.
        config (dict): Configuration parameters.
        episodes (int, optional): Number of episodes to run. Defaults to 1.
        log_callback (function, optional): Callback for logging. Defaults to None.
        
    Returns:
        tuple: (all_metrics, equity_curves, trade_histories) - Backtest results.
    """
    def _log(msg):
        if log_callback:
            log_callback(msg)
        else:
            log(msg)
    
    # Add human-readable metrics logging function
    def log_human_metrics(metrics, prefix="BACKTEST"):
        """Log a compact version of metrics focused on what humans care about."""
        # Extract key metrics humans care about
        net_profit = metrics.get("net_profit", 0)
        profit_pct = (net_profit / config.get("INITIAL_CAPITAL", 100000)) * 100
        win_rate = metrics.get("win_rate", 0) * 100
        total_trades = metrics.get("total_trades", 0)
        winning_trades = metrics.get("winning_trades", 0)
        losing_trades = metrics.get("losing_trades", 0)
        max_dd = metrics.get("max_drawdown", 0) * 100
        profit_factor = metrics.get("profit_factor", 0)
        sharpe = metrics.get("sharpe_ratio", 0)
        
        # Create compact, readable output
        log_msg = (f"[{prefix}] "
                  f"Profit: ${net_profit:.2f} ({profit_pct:.1f}%) | "
                  f"Win Rate: {win_rate:.1f}% | "
                  f"Trades: {total_trades} (W:{winning_trades}/L:{losing_trades}) | "
                  f"Max DD: {max_dd:.1f}% | "
                  f"Sharpe: {sharpe:.2f} | "
                  f"PF: {profit_factor:.2f}")
        
        _log(log_msg)
        return log_msg
    
    # Validate dataframe
    valid, message = validate_dataframe(df)
    if not valid:
        _log(f"[ERROR] DataFrame validation failed: {message}")
        return [], [], []
    
    # Create device
    device = agent.device if hasattr(agent, 'device') else "cpu"
    
    # Create environment for backtesting
    # Force tensor-based environment for efficiency
    config_copy = config.copy()
    config_copy["TENSOR_BASED_ENV"] = True
    
    # Results storage
    all_metrics = []
    equity_curves = []
    trade_histories = []
    
    # Run multiple episodes
    for ep in range(episodes):
        _log(f"[BACKTEST] Starting episode {ep+1}/{episodes}")
        
        # Create fresh environment
        env = create_environment(df, config_copy, device)
        
        # Reset environment and get initial observation
        obs = env.reset()
        done = False
        hidden = None  # Hidden state for recurrent networks
        
        # Step counter
        step = 0
        
        # Track equity curve during episode
        equity_curve = [env.capital]
        
        # Set log intervals for showing progress
        log_interval = min(1000, max(100, int(len(df) / 10)))  # Log about 10 times during backtest
        last_log_step = 0
        
        # Start stepping through environment
        while not done:
            step += 1
            
            # Get action from agent
            action, log_prob, val, preds, confs, mids, trend, novelty, hidden = agent.select_action(obs, hidden)
            
            # Take action in environment
            obs, reward, done, info = env.step(action)
            
            # Record current equity
            equity_curve.append(env.capital)
            
            # Log progress periodically with human-readable metrics
            if step - last_log_step >= log_interval:
                positions = len(env.positions)
                profit_so_far = env.capital - env.initial_capital
                profit_pct = (profit_so_far / env.initial_capital) * 100
                trades_completed = len(env.closed_trades)
                
                # Create simple human-readable progress update
                _log(f"[BACKTEST] Step {step}/{len(df)}: Capital=${env.capital:.2f} | "
                     f"Profit=${profit_so_far:.2f} ({profit_pct:.1f}%) | "
                     f"Trades={trades_completed} | Positions={positions}")
                
                last_log_step = step
        
        # Calculate metrics at episode end
        metrics = calculate_metrics(env)
        all_metrics.append(metrics)
        
        # Store equity curve and trade history
        equity_curves.append(equity_curve)
        trade_histories.append(env.closed_trades.copy())
        
        # Log episode results with human-readable format
        _log(f"[BACKTEST] Episode {ep+1} complete - Steps: {step}, Final capital: ${env.capital:.2f}")
        log_human_metrics(metrics, prefix=f"BACKTEST EP{ep+1}")
        
        # Clean up
        del env
        optimize_memory()
    
    # Calculate average metrics across all episodes
    if all_metrics:
        avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
        _log("\n" + "="*50)
        _log("[BACKTEST] AVERAGE RESULTS")
        _log("="*50)
        log_human_metrics(avg_metrics, prefix="BACKTEST AVG")
        _log("="*50)
        _log(format_metrics(avg_metrics))
        _log("="*50)
    
    return all_metrics, equity_curves, trade_histories


def run_preset_comparison(df, preset_config, user_config, log_callback=None):
    """
    Run 50-episode comparison between preset and user configurations.
    
    Args:
        df (pandas.DataFrame): DataFrame with market data.
        preset_config (dict): Preset configuration parameters.
        user_config (dict): User configuration parameters.
        log_callback (function, optional): Callback for logging. Defaults to None.
        
    Returns:
        tuple: (preset_avg, user_avg) - Average metrics for preset and user configs.
    """
    def _log(msg):
        if log_callback:
            log_callback(msg)
        else:
            log(msg)
    
    # Add compact comparison function
    def log_comparison(preset_metrics, user_metrics):
        """Log a side-by-side comparison of preset and user metrics."""
        # Extract key metrics for comparison
        metrics_to_compare = [
            ("Profit", "net_profit", "${:.2f}"),
            ("Win Rate", "win_rate", "{:.1f}%", lambda x: x * 100),
            ("Trades", "total_trades", "{}"),
            ("Max DD", "max_drawdown", "{:.1f}%", lambda x: x * 100),
            ("Sharpe", "sharpe_ratio", "{:.2f}"),
            ("Profit Factor", "profit_factor", "{:.2f}")
        ]
        
        # Create comparison table
        _log("\n" + "="*60)
        _log("{:<15} {:<20} {:<20}".format("Metric", "Preset", "Your Config"))
        _log("="*60)
        
        for name, key, fmt, *transform in metrics_to_compare:
            # Apply transform function if provided
            transform_fn = transform[0] if transform else lambda x: x
            
            # Get values with defaults
            preset_val = preset_metrics.get(key, 0)
            user_val = user_metrics.get(key, 0)
            
            # Apply transform
            preset_val = transform_fn(preset_val)
            user_val = transform_fn(user_val)
            
            # Determine if user config is better
            is_better = False
            if key in ["net_profit", "win_rate", "sharpe_ratio", "profit_factor"]:
                is_better = user_val > preset_val
            elif key in ["max_drawdown"]:
                is_better = user_val < preset_val
            
            # Format values
            preset_str = fmt.format(preset_val)
            user_str = fmt.format(user_val)
            
            # Add indicator for better result
            if is_better:
                user_str += " ✓"
            
            _log("{:<15} {:<20} {:<20}".format(name, preset_str, user_str))
        
        _log("="*60)
    
    _log("[COMPARISON] Starting 50-episode comparison...")
    
    # Validate dataframe
    valid, message = validate_dataframe(df)
    if not valid:
        _log(f"[ERROR] DataFrame validation failed: {message}")
        return None, None
        
    # Get environment dimensions
    primary_cols = [col for col in df.columns if col in [
        'close', 'high', 'low', 'volume', 'SMA9', 'SMA21', 'SMA50', 'SMA100', 'SMA200', 'SMA400', 'ParabolicSAR',
        'RSI14', 'RSI28', 'Stoch_K', 'Stoch_D', 'CCI', 'MFI', 'ROC', 'BB_upper20', 'BB_mid20', 'BB_lower20',
        'BB_upper50', 'BB_mid50', 'BB_lower50', 'BB_upper100', 'BB_mid100', 'BB_lower100', 'ATR', 'Keltner_lower',
        'Donch_lower', 'Donch_upper', 'OBV', 'ChaikinMF', 'ForceIndex', 'VolumeOsc', 'MACD_hist_diff',
        'RSI_overbought', 'price_accel', 'pca_1', 'pca_2', 'pca_3', 'pca_4', 'pca_5', 'pca_6', 'pca_7', 'pca_8',
        'pca_9', 'pca_10', 'ae_1', 'ae_2', 'ae_3', 'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
        'day_of_month_sin', 'day_of_month_cos', 'day_of_year_sin', 'day_of_year_cos'
    ]]
    input_dim = len(primary_cols)
    
    # Adjust for order book features
    order_book_features = user_config.get("VOLUME_PROFILE_LEVELS", 10) + 4  # Volume profile levels + other features
    input_dim += order_book_features
    
    _log(f"[COMPARISON] Using {input_dim} input features")
    
    # Prediction horizons based on bucket
    prediction_horizons = {
        "Scalping": [1, 6, 12, 24],
        "Short": [6, 24, 72, 144],
        "Medium": [24, 72, 144, 288],
        "Long": [72, 144, 288, 576]
    }
    
    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    preset_metrics = []
    user_metrics = []

    # Run comparison for both preset and user config
    for config_name, conf in [("Preset", preset_config), ("User", user_config)]:
        _log(f"[COMPARISON] Running 50 episodes for {config_name} config...")
        
        # Determine prediction horizons based on bucket
        horizons = prediction_horizons.get(conf["BUCKET"], [12, 36, 72, 144])
        
        # Create environment with specified config
        env = create_environment(df, conf, device)
        
        # Create agent
        hidden_size = conf.get("HIDDEN_SIZE", 512)
        learning_rate = conf.get("LEARNING_RATE", 0.0003)
        use_mixed_precision = conf.get("USE_MIXED_PRECISION", False)
        
        agent = PPOAgent(
            input_dim, 
            hidden_size, 
            learning_rate, 
            horizons,
            use_mixed_precision=use_mixed_precision,
            device=device,
            config=conf
        )
        
        # Run 50 episodes
        total_reward = 0
        for ep in range(50):
            obs = env.reset()
            done = False
            ep_reward = 0
            hidden = agent.model.init_hidden()
            
            _log(f"[COMPARISON] {config_name} - Episode {ep+1}/50")
            
            step_count = 0
            while not done:
                step_count += 1
                action, log_prob, val, preds, confs, mids, trend, novelty, hidden = agent.select_action(obs, hidden)
                obs, reward, done, _ = env.step(action)
                ep_reward += reward
                
                # Status update every 100 steps
                if step_count % 100 == 0:
                    _log(f"[COMPARISON] {config_name} - Episode {ep+1}, Step {step_count}")
                
            total_reward += ep_reward
            metrics = calculate_metrics(env)
            
            if config_name == "Preset":
                preset_metrics.append(metrics)
            else:
                user_metrics.append(metrics)
                
            _log(f"[COMPARISON] {config_name} - Episode {ep+1} - Reward: {ep_reward:.2f}")
            
        _log(f"[COMPARISON] {config_name} 50-episode avg reward: {total_reward / 50:.2f}")
        
        # Clean up to free memory
        del env
        del agent
        optimize_memory()

    # Calculate average metrics
    if preset_metrics and user_metrics:
        preset_avg = {k: np.mean([m[k] for m in preset_metrics]) for k in preset_metrics[0]}
        user_avg = {k: np.mean([m[k] for m in user_metrics]) for k in user_metrics[0]}
        
        # Calculate percentage improvement
        improvements = {}
        for k in preset_avg:
            if preset_avg[k] != 0:
                if k in ['max_drawdown', 'avg_loss']:  # Lower is better
                    improvements[k] = ((preset_avg[k] - user_avg[k]) / preset_avg[k]) * 100
                else:  # Higher is better
                    improvements[k] = ((user_avg[k] - preset_avg[k]) / preset_avg[k]) * 100
            else:
                improvements[k] = 0

        # Log detailed comparison
        _log("\n" + "="*50)
        _log("COMPARISON RESULTS")
        _log("="*50)
        _log(f"{'Metric':<20} {'Preset':<12} {'User':<12} {'Change %':<12}")
        _log("-"*50)
        
        for k in sorted(preset_avg.keys()):
            # Format based on the type of metric
            if k in ['win_rate', 'max_drawdown']:
                preset_str = f"{preset_avg[k]*100:.2f}%"
                user_str = f"{user_avg[k]*100:.2f}%"
            elif k in ['net_profit', 'avg_win', 'avg_loss']:
                preset_str = f"${preset_avg[k]:.2f}"
                user_str = f"${user_avg[k]:.2f}"
            else:
                preset_str = f"{preset_avg[k]:.2f}"
                user_str = f"{user_avg[k]:.2f}"
                
            # Color code improvements
            if improvements[k] > 0 and k not in ['max_drawdown', 'avg_loss']:
                change_str = f"+{improvements[k]:.2f}%"
            elif improvements[k] < 0 and k in ['max_drawdown', 'avg_loss']:
                change_str = f"+{abs(improvements[k]):.2f}%"
            else:
                change_str = f"{improvements[k]:.2f}%"
                
            _log(f"{k:<20} {preset_str:<12} {user_str:<12} {change_str:<12}")
        
        _log("="*50)
        
        # Overall assessment
        key_metrics = {
            'net_profit': 4,   # Weight 4
            'win_rate': 3,     # Weight 3
            'sharpe': 2,       # Weight 2
            'profit_factor': 2 # Weight 2
        }
        
        total_weight = sum(key_metrics.values())
        weighted_improvement = sum(improvements.get(k, 0) * key_metrics[k] for k in key_metrics) / total_weight
        
        _log(f"\nOVERALL WEIGHTED IMPROVEMENT: {weighted_improvement:.2f}%")
        
        if weighted_improvement > 15:
            _log("ASSESSMENT: User configuration significantly outperforms preset.")
        elif weighted_improvement > 5:
            _log("ASSESSMENT: User configuration moderately improves over preset.")
        elif weighted_improvement > -5:
            _log("ASSESSMENT: User configuration performs similarly to preset.")
        else:
            _log("ASSESSMENT: Preset configuration performs better than user configuration.")
            
        return preset_avg, user_avg
    
    return None, None


def calculate_drawdowns(equity_curve):
    """
    Calculate drawdowns from an equity curve.
    
    Args:
        equity_curve (list): List of equity values over time.
        
    Returns:
        tuple: (drawdowns, max_drawdown, max_drawdown_duration) - Drawdown statistics.
    """
    if not equity_curve:
        return [], 0.0, 0
        
    # Calculate running maximum
    running_max = np.maximum.accumulate(equity_curve)
    
    # Calculate drawdown in dollars
    drawdowns = running_max - equity_curve
    
    # Calculate drawdown in percentage
    drawdown_pct = drawdowns / running_max
    
    # Find maximum drawdown
    max_drawdown = np.max(drawdown_pct)
    max_drawdown_idx = np.argmax(drawdown_pct)
    
    # Calculate drawdown duration
    if max_drawdown == 0:
        max_drawdown_duration = 0
    else:
        # Find the last time equity was at a peak before max drawdown
        peak_idx = max_drawdown_idx - np.argmax(equity_curve[max_drawdown_idx::-1])
        
        # Find the next time equity returned to that peak after max drawdown
        try:
            recovery_idx = max_drawdown_idx + np.argmax(equity_curve[max_drawdown_idx:] >= running_max[peak_idx])
            max_drawdown_duration = recovery_idx - peak_idx
        except ValueError:
            # No recovery within the equity curve
            max_drawdown_duration = len(equity_curve) - peak_idx
    
    return drawdown_pct.tolist(), max_drawdown, max_drawdown_duration


def analyze_trade_distribution(trade_history):
    """
    Analyze the distribution of trades for statistical properties.
    
    Args:
        trade_history (list): List of trade tuples (profit, percentage_gain, hold_time).
        
    Returns:
        dict: Dictionary of trade distribution statistics.
    """
    if not trade_history:
        return {
            "num_trades": 0,
            "win_rate": 0.0,
            "avg_profit": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "avg_hold_time": 0,
            "profit_distribution": [],
            "hold_time_distribution": []
        }
    
    # Extract profit and hold time data
    profits = [t[0] for t in trade_history]
    hold_times = [t[2] for t in trade_history]
    
    # Calculate basic statistics
    num_trades = len(trade_history)
    wins = [p for p in profits if p > 0]
    losses = [p for p in profits if p <= 0]
    win_rate = len(wins) / num_trades if num_trades > 0 else 0.0
    avg_profit = np.mean(wins) if wins else 0.0
    avg_loss = np.mean(np.abs(losses)) if losses else 0.0
    profit_factor = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float('inf')
    expectancy = (win_rate * avg_profit - (1 - win_rate) * avg_loss)
    avg_hold_time = np.mean(hold_times)
    
    # Create profit distribution
    profit_bins = np.linspace(min(profits), max(profits), 20) if profits else []
    profit_hist, _ = np.histogram(profits, bins=profit_bins)
    profit_distribution = [(profit_bins[i], profit_hist[i]) for i in range(len(profit_hist))]
    
    # Create hold time distribution
    hold_time_bins = np.linspace(min(hold_times), max(hold_times), 20) if hold_times else []
    hold_time_hist, _ = np.histogram(hold_times, bins=hold_time_bins)
    hold_time_distribution = [(hold_time_bins[i], hold_time_hist[i]) for i in range(len(hold_time_hist))]
    
    return {
        "num_trades": num_trades,
        "win_rate": win_rate,
        "avg_profit": avg_profit,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "avg_hold_time": avg_hold_time,
        "profit_distribution": profit_distribution,
        "hold_time_distribution": hold_time_distribution
    }


def analyze_market_conditions(df, trade_history):
    """
    Analyze which market conditions lead to profitable trades.
    
    Args:
        df (pandas.DataFrame): DataFrame with market data.
        trade_history (list): List of trade tuples (profit, percentage_gain, hold_time).
        
    Returns:
        dict: Dictionary of market condition analysis.
    """
    if not trade_history or df is None or df.empty:
        return {"market_conditions": []}
    
    # Check for required technical indicators
    required_indicators = ['RSI14', 'Stoch_K', 'SMA9', 'SMA50', 'SMA200']
    available_indicators = [col for col in required_indicators if col in df.columns]
    
    if not available_indicators:
        return {"market_conditions": []}
    
    # Group trades by profitability
    profitable_trades = [t for t in trade_history if t[0] > 0]
    losing_trades = [t for t in trade_history if t[0] <= 0]
    
    # Get entry points
    entry_points = []
    for trades, is_profitable in [(profitable_trades, True), (losing_trades, False)]:
        for trade in trades:
            _, _, hold_time = trade
            exit_step = getattr(trade, "exit_step", None)
            if exit_step is not None:
                entry_step = exit_step - hold_time
                entry_points.append((entry_step, is_profitable))
    
    # No valid entry points
    if not entry_points:
        return {"market_conditions": []}
    
    # Extract market conditions at entry
    market_conditions = []
    for entry_step, is_profitable in entry_points:
        if 0 <= entry_step < len(df):
            condition = {"profitable": is_profitable}
            
            # Add indicator values
            for indicator in available_indicators:
                if indicator in df.columns:
                    condition[indicator] = df.iloc[entry_step][indicator]
            
            # Add trend information if possible
            if 'SMA9' in df.columns and 'SMA50' in df.columns:
                # Short-term trend
                condition["short_trend"] = "up" if df.iloc[entry_step]['SMA9'] > df.iloc[entry_step]['SMA50'] else "down"
            
            if 'SMA50' in df.columns and 'SMA200' in df.columns:
                # Long-term trend
                condition["long_trend"] = "up" if df.iloc[entry_step]['SMA50'] > df.iloc[entry_step]['SMA200'] else "down"
            
            market_conditions.append(condition)
    
    # Analyze conditions for profitable vs losing trades
    analysis = {"market_conditions": market_conditions}
    
    # Analyze by indicator
    for indicator in available_indicators:
        profitable_values = [c[indicator] for c in market_conditions if c["profitable"] and indicator in c]
        losing_values = [c[indicator] for c in market_conditions if not c["profitable"] and indicator in c]
        
        if profitable_values and losing_values:
            analysis[f"{indicator}_profitable_avg"] = np.mean(profitable_values)
            analysis[f"{indicator}_losing_avg"] = np.mean(losing_values)
    
    # Analyze by trend
    if "short_trend" in market_conditions[0]:
        profitable_up = sum(1 for c in market_conditions if c["profitable"] and c["short_trend"] == "up")
        profitable_down = sum(1 for c in market_conditions if c["profitable"] and c["short_trend"] == "down")
        losing_up = sum(1 for c in market_conditions if not c["profitable"] and c["short_trend"] == "up")
        losing_down = sum(1 for c in market_conditions if not c["profitable"] and c["short_trend"] == "down")
        
        total_profitable = profitable_up + profitable_down
        if total_profitable > 0:
            analysis["short_trend_up_win_rate"] = profitable_up / (profitable_up + losing_up) if (profitable_up + losing_up) > 0 else 0
            analysis["short_trend_down_win_rate"] = profitable_down / (profitable_down + losing_down) if (profitable_down + losing_down) > 0 else 0
    
    return analysis


def plot_backtest_results(equity_curves, trade_histories, config, output_dir=None):
    """
    Create plots of backtest results.
    
    Args:
        equity_curves (list): List of equity curves.
        trade_histories (list): List of trade histories.
        config (dict): Configuration parameters.
        output_dir (str, optional): Directory to save plots. Defaults to None.
        
    Returns:
        list: List of created figure filenames.
    """
    if not equity_curves or not trade_histories:
        return []
    
    # Create output directory if it doesn't exist
    if output_dir is not None and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set up matplotlib
    plt.style.use('seaborn-v0_8-darkgrid')
    filenames = []
    
    # Plot 1: Equity Curve
    plt.figure(figsize=(12, 6))
    
    # If multiple equity curves, plot all of them
    if len(equity_curves) > 1:
        # Plot up to 5 individual curves
        for i, curve in enumerate(equity_curves[:5]):
            plt.plot(curve, alpha=0.3, label=f'Episode {i+1}')
        
        # Plot average curve
        avg_length = min(len(curve) for curve in equity_curves)
        avg_curve = np.mean([curve[:avg_length] for curve in equity_curves], axis=0)
        plt.plot(avg_curve, 'k-', linewidth=2, label='Average')
    else:
        # Just plot the single curve
        plt.plot(equity_curves[0], 'b-', label='Equity')
    
    plt.title(f'Equity Curve - {config.get("BUCKET", "Unknown")} Bucket')
    plt.xlabel('Steps')
    plt.ylabel('Capital ($)')
    plt.legend()
    plt.grid(True)
    
    # Add initial capital line
    init_capital = config.get("INITIAL_CAPITAL", 100000.0)
    plt.axhline(y=init_capital, color='r', linestyle='--', label='Initial Capital')
    
    # Save the figure
    if output_dir is not None:
        filename = os.path.join(output_dir, f'equity_curve_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        filenames.append(filename)
        plt.close()
    else:
        plt.show()
    
    # Plot 2: Drawdowns
    if len(equity_curves) > 0:
        plt.figure(figsize=(12, 6))
        
        # Calculate drawdowns for each equity curve
        all_drawdowns = [calculate_drawdowns(curve)[0] for curve in equity_curves]
        
        # Plot up to 5 individual drawdown curves
        for i, drawdown in enumerate(all_drawdowns[:5]):
            plt.plot(drawdown, alpha=0.3, label=f'Episode {i+1}')
        
        # Plot average drawdown
        if len(all_drawdowns) > 1:
            avg_length = min(len(d) for d in all_drawdowns)
            avg_drawdown = np.mean([d[:avg_length] for d in all_drawdowns], axis=0)
            plt.plot(avg_drawdown, 'k-', linewidth=2, label='Average')
        
        plt.title(f'Drawdowns - {config.get("BUCKET", "Unknown")} Bucket')
        plt.xlabel('Steps')
        plt.ylabel('Drawdown (%)')
        plt.legend()
        plt.grid(True)
        
        # Save the figure
        if output_dir is not None:
            filename = os.path.join(output_dir, f'drawdowns_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            filenames.append(filename)
            plt.close()
        else:
            plt.show()
    
    # Plot 3: Trade Profit Distribution
    if len(trade_histories) > 0:
        # Combine all trades from all histories
        all_trades = []
        for history in trade_histories:
            all_trades.extend(history)
        
        if all_trades:
            plt.figure(figsize=(12, 6))
            
            # Extract profits
            profits = [t[0] for t in all_trades]
            
            # Create histogram
            sns.histplot(profits, bins=30, kde=True)
            plt.axvline(x=0, color='r', linestyle='--')
            
            plt.title(f'Trade Profit Distribution - {config.get("BUCKET", "Unknown")} Bucket')
            plt.xlabel('Profit ($)')
            plt.ylabel('Frequency')
            plt.grid(True)
            
            # Save the figure
            if output_dir is not None:
                filename = os.path.join(output_dir, f'profit_dist_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                filenames.append(filename)
                plt.close()
            else:
                plt.show()
    
    # Plot 4: Hold Time Distribution
    if len(trade_histories) > 0 and all_trades:
        plt.figure(figsize=(12, 6))
        
        # Extract hold times
        hold_times = [t[2] for t in all_trades]
        
        # Convert to hours (assuming 5-minute bars, 12 bars per hour)
        hold_times_hours = [t / 12 for t in hold_times]
        
        # Create histogram
        sns.histplot(hold_times_hours, bins=30, kde=True)
        
        plt.title(f'Trade Hold Time Distribution - {config.get("BUCKET", "Unknown")} Bucket')
        plt.xlabel('Hold Time (hours)')
        plt.ylabel('Frequency')
        plt.grid(True)
        
        # Save the figure
        if output_dir is not None:
            filename = os.path.join(output_dir, f'hold_time_dist_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            filenames.append(filename)
            plt.close()
        else:
            plt.show()
    
    return filenames


def generate_backtest_report(metrics, equity_curves, trade_histories, config, filename=None):
    """
    Generate a comprehensive backtest report.
    
    Args:
        metrics (list): List of metrics dictionaries.
        equity_curves (list): List of equity curves.
        trade_histories (list): List of trade histories.
        config (dict): Configuration parameters.
        filename (str, optional): Path to save report. Defaults to None.
        
    Returns:
        str: Report content.
    """
    if not metrics:
        return "No metrics available for report generation."
    
    # Calculate average metrics
    avg_metrics = {k: np.mean([m[k] for m in metrics]) for k in metrics[0]}
    
    # Calculate drawdowns
    max_drawdowns = []
    max_drawdown_durations = []
    for curve in equity_curves:
        _, max_dd, max_dd_duration = calculate_drawdowns(curve)
        max_drawdowns.append(max_dd)
        max_drawdown_durations.append(max_dd_duration)
    
    avg_max_drawdown = np.mean(max_drawdowns) if max_drawdowns else 0.0
    avg_max_drawdown_duration = np.mean(max_drawdown_durations) if max_drawdown_durations else 0
    
    # Analyze trades
    all_trades = []
    for history in trade_histories:
        all_trades.extend(history)
    
    trade_analysis = analyze_trade_distribution(all_trades)
    
    # Generate report
    report = []
    report.append("="*80)
    report.append(f"BACKTEST REPORT - {config.get('BUCKET', 'Unknown')} BUCKET")
    report.append("="*80)
    report.append("")
    
    # Configuration section
    report.append("CONFIGURATION:")
    report.append("-"*80)
    for key, value in sorted(config.items()):
        report.append(f"{key}: {value}")
    report.append("")
    
    # Performance Metrics
    report.append("PERFORMANCE METRICS:")
    report.append("-"*80)
    report.append(f"Number of Episodes: {len(metrics)}")
    report.append(f"Initial Capital: ${config.get('INITIAL_CAPITAL', 100000.0):.2f}")
    report.append(f"Final Average Capital: ${avg_metrics['net_profit'] + config.get('INITIAL_CAPITAL', 100000.0):.2f}")
    report.append(f"Net Profit: ${avg_metrics['net_profit']:.2f}")
    report.append(f"Return on Investment: {(avg_metrics['net_profit'] / config.get('INITIAL_CAPITAL', 100000.0)) * 100:.2f}%")
    report.append(f"Sharpe Ratio: {avg_metrics['sharpe']:.2f}")
    report.append(f"Max Drawdown: {avg_max_drawdown*100:.2f}%")
    report.append(f"Max Drawdown Duration: {avg_max_drawdown_duration:.0f} steps")
    report.append("")
    
    # Trade Statistics
    report.append("TRADE STATISTICS:")
    report.append("-"*80)
    report.append(f"Total Trades: {avg_metrics['total_trades']:.1f}")
    report.append(f"Win Rate: {avg_metrics['win_rate']*100:.2f}%")
    report.append(f"Profit Factor: {avg_metrics['profit_factor']:.2f}")
    report.append(f"Average Win: ${avg_metrics['avg_win']:.2f}")
    report.append(f"Average Loss: ${avg_metrics['avg_loss']:.2f}")
    report.append(f"Average Hold Time: {avg_metrics['avg_hold']/12:.2f} hours")
    report.append(f"Longest Hold Time: {avg_metrics['longest_hold']/12:.2f} hours")
    report.append(f"Shortest Hold Time: {avg_metrics['shortest_hold']/12:.2f} hours")
    report.append(f"Trades Per Day: {avg_metrics['trades_per_day']:.2f}")
    report.append("")
    
    # Trade Distribution Analysis
    report.append("TRADE DISTRIBUTION ANALYSIS:")
    report.append("-"*80)
    report.append(f"Expectancy (Average Expected Profit per Trade): ${trade_analysis['expectancy']:.2f}")
    report.append(f"Top 5 Best Trades:")
    
    if all_trades:
        # Sort trades by profit
        sorted_trades = sorted(all_trades, key=lambda t: t[0], reverse=True)
        
        # Show top 5 best trades
        for i, trade in enumerate(sorted_trades[:5], 1):
            profit, pct_gain, hold_time = trade
            report.append(f"  {i}. Profit: ${profit:.2f}, Gain: {pct_gain:.2f}%, Hold Time: {hold_time/12:.2f} hours")
        
        report.append(f"Top 5 Worst Trades:")
        for i, trade in enumerate(sorted_trades[-5:], 1):
            profit, pct_gain, hold_time = trade
            report.append(f"  {i}. Profit: ${profit:.2f}, Gain: {pct_gain:.2f}%, Hold Time: {hold_time/12:.2f} hours")
    
    report.append("")
    report.append("="*80)
    report.append(f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("="*80)
    
    # Save report if filename is provided
    if filename:
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write("\n".join(report))
        except Exception as e:
            log(f"Error saving report: {e}")
    
    return "\n".join(report)


class BacktestingEngine:
    """
    Enhanced backtesting engine with support for:
    - Probabilistic predictions with uncertainty
    - Fractal pattern recognition
    - Regime detection
    - Explainable AI decision tracking
    """
    
    def __init__(self, df, agent, config, use_advanced_features=True):
        """
        Initialize backtesting engine with advanced features.
    
    Args:
            df (pd.DataFrame): DataFrame with market data
            agent (PPOAgent): Trained agent for decision making
            config (dict): Configuration parameters
            use_advanced_features (bool): Whether to use advanced features
        """
        self.df = df
        self.agent = agent
        self.config = config
        self.use_advanced_features = use_advanced_features
        
        # Advanced features tracking
        self.regime_history = []
        self.fractal_analysis = []
        self.explanation_history = []
        self.uncertainty_history = []
        self.feature_importance_history = []
        
        # Initialize metrics
        self.metrics = {}
        
        # Feature names for explainability
        self.feature_names = config.get("FEATURE_NAMES", None)
        
        log("Initialized enhanced backtesting engine")
        if self.use_advanced_features:
            log("Advanced features enabled: probabilistic predictions, fractal analysis, explainable AI")
    
    def run(self, episode_length=None, log_freq=10):
        """
        Run backtest with advanced feature analysis.
    
    Args:
            episode_length (int, optional): Length of episode. Defaults to None (full data).
            log_freq (int, optional): Logging frequency. Defaults to 10.
        
    Returns:
            dict: Backtest results and metrics
        """
        # Setup environment
        env = create_environment(self.df, self.config)
        obs = env.reset()
        
        # Track metrics
        rewards = []
        equities = []
        positions = []
        trades = []
        
        # Advanced feature tracking
        regimes = []
        explanations = []
        uncertainties = []
        fractal_data = []
        
        done = False
        step = 0
        
        # Set max steps
        max_steps = len(self.df) - 1 if episode_length is None else min(episode_length, len(self.df) - 1)
        
        log(f"Starting backtest for {max_steps} steps")
        
        # Main backtest loop
        while not done and step < max_steps:
            # Get current price data for advanced analysis
            current_idx = env.get_current_step()
            price_history = self.df['close'].values[max(0, current_idx-200):current_idx+1]
            price_tensor = torch.tensor(price_history, device=self.agent.device).float()
            
            # Perform advanced analyses if enabled
            if self.use_advanced_features:
                # Detect market regime
                regime = self.agent.detect_regime(price_tensor)
                regimes.append(regime)
                
                # Perform fractal analysis
                if len(price_history) > 50:
                    # Calculate Hurst exponent
                    hurst = compute_fractal_dimension_tensor(price_tensor).item()
                    
                    # Detect Elliott wave patterns
                    elliott = detect_elliott_wave_pattern_tensor(price_tensor)
                    
                    # Compute market fractals
                    high_tensor = torch.tensor(self.df['high'].values[max(0, current_idx-100):current_idx+1], device=self.agent.device).float()
                    low_tensor = torch.tensor(self.df['low'].values[max(0, current_idx-100):current_idx+1], device=self.agent.device).float()
                    bullish_fractals, bearish_fractals = compute_market_fractals_tensor(high_tensor, low_tensor)
                    
                    # Compute wavelet features
                    wavelet_features = compute_timeframe_wavelet_features(price_tensor)
                    
                    # Store fractal analysis
                    fractal_data.append({
                        'step': step,
                        'hurst': hurst,
                        'elliott_detected': elliott['detected'],
                        'elliott_confidence': elliott['confidence'],
                        'elliott_type': elliott['pattern_type'],
                        'bullish_fractal': bullish_fractals[-1].item() if len(bullish_fractals) > 0 else False,
                        'bearish_fractal': bearish_fractals[-1].item() if len(bearish_fractals) > 0 else False,
                        'trend_direction': wavelet_features['trend_direction'],
                        'fractal_dimension': wavelet_features['fractal_dimension']
                    })
            
            # Prepare observation tensor
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.agent.device)
                
                # Get action from agent
            with torch.no_grad():
                action, _ = self.agent.model.act(obs_tensor)
                action = action.cpu().numpy()[0]
                
                # Get probabilistic predictions and confidence
                if self.use_advanced_features:
                    # Get model outputs for explainability
                    model_outputs = self.agent.model(obs_tensor, explain=True)
                    
                    # Extract predictions and uncertainty
                    prediction_means = {}
                    prediction_stds = {}
                    confidence_scores = {}
                    
                    # Extract multi-horizon predictions
                    for horizon in self.agent.model.horizons:
                        prediction_means[horizon] = model_outputs['predictions'][horizon].item()
                        prediction_stds[horizon] = model_outputs['prediction_stds'][horizon].item()
                        confidence_scores[horizon] = model_outputs['confidence'][horizon].item()
                    
                    # Store uncertainty history
                    uncertainties.append({
                        'step': step,
                        'prediction_means': prediction_means,
                        'prediction_stds': prediction_stds,
                        'confidence_scores': confidence_scores,
                        'trend_strength': model_outputs['trend_strength'].item()
                    })
                    
                    # Get explanation for current decision
                    explanation = self.agent.model.get_decision_explanation(
                        obs_tensor, 
                        feature_names=self.feature_names
                    )
                    explanations.append(explanation)
            
            # Create action dictionary with probabilistic predictions
            action_dict = {'type': 'buy' if action[0] > 0 else 'sell', 'amount': abs(action[0])}
            
            # Add probabilistic predictions if available
            if self.use_advanced_features:
                action_dict.update({
                    'prediction_means': list(prediction_means.values()),
                    'prediction_stds': list(prediction_stds.values()),
                    'confidence_scores': list(confidence_scores.values()),
                    'trend_strength': model_outputs['trend_strength'].item()
                })
            
            # Take step in environment
            next_obs, reward, done, info = env.step(action_dict)
            
            # Store metrics
            rewards.append(reward)
            equities.append(info['total_value'])
            positions.append(len(info['positions']))
            
            # Store completed trades
            if 'completed_orders' in info and info['completed_orders']:
                for order in info['completed_orders']:
                    trades.append(order)
            
            # Update observation
            obs = next_obs
            step += 1
            
            # Log progress
            if step % log_freq == 0:
                log(f"Step {step}/{max_steps}: Equity = ${equities[-1]:.2f}, Reward = {reward:.4f}")
                if self.use_advanced_features and explanations:
                    log(f"Action explanation: {explanations[-1]['explanation']}")
        
        # Compute final metrics
        self.metrics = calculate_metrics(
            self.df, rewards, equities, trades, 
            self.config.get("INITIAL_CAPITAL", 100000)
        )
        
        # Store advanced feature histories
        if self.use_advanced_features:
            self.regime_history = regimes
            self.explanation_history = explanations
            self.uncertainty_history = uncertainties
            self.fractal_analysis = fractal_data
        
        # Return results
        results = {
            'metrics': self.metrics,
            'equity_curve': equities,
            'rewards': rewards,
            'positions': positions,
            'trades': trades
        }
        
        # Add advanced feature results if enabled
        if self.use_advanced_features:
            results.update({
                'regimes': regimes,
                'explanations': explanations,
                'uncertainties': uncertainties,
                'fractal_analysis': fractal_data
            })
        
        return results
    
    def generate_report(self, save_path=None):
        """
        Generate comprehensive backtest report with advanced feature insights.
    
    Args:
            save_path (str, optional): Path to save report. Defaults to None.
        
    Returns:
            dict: Report data
        """
        report = {
            'metrics': self.metrics,
            'advanced_features': self.use_advanced_features
        }
        
        # Add advanced feature insights if enabled
        if self.use_advanced_features:
            # Regime analysis
            regime_counts = {}
            for regime in self.regime_history:
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
            # Calculate regime performance
            regime_performance = {}
            for regime in set(self.regime_history):
                # Find all trades in this regime
                regime_trades = []
                for i, r in enumerate(self.regime_history):
                    if r == regime and i < len(self.uncertainty_history):
                        regime_trades.append({
                            'step': i,
                            'confidence': np.mean(list(self.uncertainty_history[i]['confidence_scores'].values())),
                            'reward': self.metrics['rewards'][i] if i < len(self.metrics['rewards']) else 0
                        })
                
                # Calculate average confidence and reward
                if regime_trades:
                    avg_confidence = np.mean([t['confidence'] for t in regime_trades])
                    avg_reward = np.mean([t['reward'] for t in regime_trades])
                    regime_performance[regime] = {
                        'count': len(regime_trades),
                        'avg_confidence': avg_confidence,
                        'avg_reward': avg_reward
                    }
            
            # Fractal pattern insights
            fractal_insights = {
                'hurst_exponent': np.mean([f['hurst'] for f in self.fractal_analysis]) if self.fractal_analysis else 0,
                'elliott_patterns': sum(1 for f in self.fractal_analysis if f['elliott_detected']),
                'bullish_fractals': sum(1 for f in self.fractal_analysis if f['bullish_fractal']),
                'bearish_fractals': sum(1 for f in self.fractal_analysis if f['bearish_fractal'])
            }
            
            # Decision explanation insights
            top_features = {}
            for exp in self.explanation_history:
                if 'action_features' in exp:
                    for feat in exp['action_features']:
                        if feat['name'] not in top_features:
                            top_features[feat['name']] = {
                                'count': 0,
                                'importance': 0
                            }
                        top_features[feat['name']]['count'] += 1
                        top_features[feat['name']]['importance'] += feat['importance']
            
            # Calculate average importance
            for feat, data in top_features.items():
                if data['count'] > 0:
                    data['avg_importance'] = data['importance'] / data['count']
            
            # Sort features by frequency
            top_features = {k: v for k, v in sorted(
                top_features.items(), 
                key=lambda item: item[1]['count'], 
                reverse=True
            )}
            
            # Add advanced insights to report
            report['regime_analysis'] = {
                'counts': regime_counts,
                'performance': regime_performance
            }
            report['fractal_insights'] = fractal_insights
            report['feature_importance'] = top_features
        
        # Save report if path provided
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=4)
        
        return report

# Add the following function at the end of the file, before the main section

def run_advanced_backtest(df, agent, config):
    """
    Run enhanced backtest with all advanced features.
    
    Args:
        df (pd.DataFrame): DataFrame with market data
        agent (PPOAgent): Trained agent for decision making
        config (dict): Configuration parameters
        
    Returns:
        dict: Backtest results and report
    """
    # Create backtesting engine with advanced features
    engine = BacktestingEngine(df, agent, config, use_advanced_features=True)
    
    # Run backtest
    results = engine.run()
    
    # Generate report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"backtest_report_{timestamp}.json"
    report = engine.generate_report(save_path=report_path)
    
    # Create visualization
    visualize_advanced_backtest(results, report, config)
    
    return {
        'results': results,
        'report': report,
        'report_path': report_path
    }

def visualize_advanced_backtest(results, report, config):
    """
    Create advanced visualizations for backtest results.
    
    Args:
        results (dict): Backtest results
        report (dict): Backtest report
        config (dict): Configuration parameters
    """
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"backtest_viz_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(results['equity_curve'])
    plt.title('Equity Curve')
    plt.xlabel('Steps')
    plt.ylabel('Equity')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'equity_curve.png'), dpi=300)
    plt.close()
    
    # Plot regime distribution
    if 'regimes' in results:
        regime_counts = {}
        for regime in results['regimes']:
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        plt.figure(figsize=(10, 6))
        regimes = list(regime_counts.keys())
        counts = list(regime_counts.values())
        
        plt.bar(regimes, counts)
        plt.title('Market Regime Distribution')
        plt.xlabel('Regime')
        plt.ylabel('Count')
        plt.savefig(os.path.join(output_dir, 'regime_distribution.png'), dpi=300)
        plt.close()
    
    # Plot confidence vs returns
    if 'uncertainties' in results:
        # Calculate returns from equity curve
        returns = []
        for i in range(1, len(results['equity_curve'])):
            returns.append((results['equity_curve'][i] / results['equity_curve'][i-1]) - 1)
        
        # Extract confidence scores
        confidences = []
        for u in results['uncertainties']:
            # Average confidence across horizons
            conf_values = list(u['confidence_scores'].values())
            confidences.append(np.mean(conf_values))
        
        # Truncate to same length
        min_len = min(len(returns), len(confidences))
        returns = returns[:min_len]
        confidences = confidences[:min_len]
        
        # Plot confidence vs returns
        plt.figure(figsize=(10, 6))
        plt.scatter(confidences, returns, alpha=0.5)
        plt.title('Decision Confidence vs. Returns')
        plt.xlabel('Confidence Score')
        plt.ylabel('Return')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'confidence_vs_returns.png'), dpi=300)
        plt.close()
    
    # Plot feature importance
    if 'feature_importance' in report:
        # Get top 10 features
        features = list(report['feature_importance'].keys())[:10]
        importances = [report['feature_importance'][f]['avg_importance'] for f in features]
        
        plt.figure(figsize=(12, 6))
        plt.barh(features, importances)
        plt.title('Top Features by Importance')
        plt.xlabel('Average Importance')
        plt.gca().invert_yaxis()  # Highest at top
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300)
        plt.close()
    
    # Plot fractal indicators if available
    if 'fractal_analysis' in results and results['fractal_analysis']:
        # Extract data
        steps = [f['step'] for f in results['fractal_analysis']]
        hurst = [f['hurst'] for f in results['fractal_analysis']]
        
        # Create figure
        plt.figure(figsize=(12, 6))
        plt.plot(steps, hurst, label='Hurst Exponent')
        
        # Add horizontal lines for reference
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random Walk (H=0.5)')
        plt.axhline(y=0.7, color='g', linestyle='--', alpha=0.5, label='Trend (H=0.7)')
        plt.axhline(y=0.3, color='b', linestyle='--', alpha=0.5, label='Mean Reversion (H=0.3)')
        
        plt.title('Fractal Analysis - Hurst Exponent')
        plt.xlabel('Steps')
        plt.ylabel('Hurst Exponent')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fractal_analysis.png'), dpi=300)
        plt.close()
    
    log(f"Advanced visualizations saved to {output_dir}")

def run_probabilistic_backtest(df, agent, config, output_dir="backtest_results"):
    """
    Run backtest with probabilistic model predictions and uncertainty metrics.
    
    Args:
        df (pandas.DataFrame): Market data for backtesting
        agent (object): Trained agent with probabilistic outputs
        config (dict): Configuration parameters
        output_dir (str): Directory to save results
        
    Returns:
        dict: Backtest results with probabilistic metrics
    """
    log("Starting probabilistic model backtest")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create environment for backtesting
    env = create_environment(df, config)
    obs = env.reset()
    
    # Track metrics
    rewards = []
    equities = []
    positions = []
    trades = []
    
    # Probabilistic metrics
    prediction_means = []
    prediction_stds = []
    confidence_scores = []
    prediction_hits = []  # Whether prediction was correct
    calibration_errors = []  # Difference between confidence and accuracy
    
    done = False
    step = 0
    
    # Determine max steps
    max_steps = min(config.get("MAX_BACKTEST_STEPS", 1000000), len(df) - 1)
    
    while not done and step < max_steps:
        # Get action from agent with probabilistic outputs
        action, prob_outputs = agent.act_with_uncertainty(obs)
        
        # Extract probabilistic predictions
        means = prob_outputs.get('prediction_means', [0.0])
        stds = prob_outputs.get('prediction_stds', [0.01])
        conf = prob_outputs.get('confidence_scores', [0.5])
        
        # Create action dictionary with probabilistic components
        action_dict = {
            'type': 'buy' if action[0] > 0 else 'sell',
            'amount': abs(action[0]),
            'prediction_means': means,
            'prediction_stds': stds, 
            'confidence_scores': conf,
            'trend_strength': prob_outputs.get('trend_strength', 0.0)
        }
        
        # Take step in environment
        next_obs, reward, done, info = env.step(action_dict)
        
        # Store metrics
        rewards.append(reward)
        equities.append(info.get('total_value', 0.0))
        positions.append(len(info.get('positions', [])))
        
        # Store probabilistic metrics
        prediction_means.append(means)
        prediction_stds.append(stds)
        confidence_scores.append(conf)
        
        # Calculate prediction accuracy
        prediction_correct = False
        for i, prefix in enumerate(['pred_horizon_0_accuracy', 'pred_horizon_1_accuracy']):
            if prefix in info and i < len(conf):
                prediction_correct = info[prefix]
                prediction_hits.append(prediction_correct)
                
                # Calculate calibration error (|confidence - accuracy|)
                calibration_errors.append(abs(conf[i] - float(prediction_correct)))
        
        # Store trades
        if 'completed_orders' in info:
            for order in info['completed_orders']:
                trades.append(order)
        
        # Update observation
        obs = next_obs
        step += 1
        
        # Log progress
        if step % 100 == 0:
            log(f"Step {step}/{max_steps}: Equity=${equities[-1]:.2f}, Reward={reward:.4f}")
    
    # Calculate metrics
    metrics = calculate_metrics(
        df, rewards, equities, trades, 
        config.get("INITIAL_CAPITAL", 100000.0)
    )
    
    # Add probabilistic metrics
    if prediction_hits:
        metrics['prediction_accuracy'] = sum(prediction_hits) / len(prediction_hits)
    if calibration_errors:
        metrics['calibration_error'] = sum(calibration_errors) / len(calibration_errors)
    
    # Calculate uncertainty-adjusted Sharpe ratio
    if len(equities) > 1:
        # Calculate returns
        returns = [(equities[i] / equities[i-1] - 1) for i in range(1, len(equities))]
        # Calculate Sharpe ratio
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)  # Annualized
        # Weight by average confidence
        avg_confidence = np.mean([np.mean(c) for c in confidence_scores])
        metrics['uncertainty_adjusted_sharpe'] = sharpe * avg_confidence
    
    # Save detailed results
    results = {
        'metrics': metrics,
        'equity_curve': equities,
        'rewards': rewards,
        'positions': positions,
        'trades': trades,
        'prediction_means': prediction_means,
        'prediction_stds': prediction_stds,
        'confidence_scores': confidence_scores,
        'prediction_hits': prediction_hits,
        'calibration_errors': calibration_errors
    }
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"prob_backtest_{timestamp}.json")
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for k, v in results.items():
        if isinstance(v, np.ndarray):
            serializable_results[k] = v.tolist()
        elif isinstance(v, list) and v and isinstance(v[0], np.ndarray):
            serializable_results[k] = [x.tolist() for x in v]
        else:
            serializable_results[k] = v
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    # Create visualization
    visualize_probabilistic_results(results, output_dir, timestamp)
    
    log(f"Probabilistic backtest completed. Results saved to {results_file}")
    return results

def visualize_probabilistic_results(results, output_dir, timestamp):
    """
    Create visualizations for probabilistic backtest results.
    
    Args:
        results (dict): Backtest results with probabilistic metrics
        output_dir (str): Directory to save visualizations
        timestamp (str): Timestamp for filenames
    """
    # Create visualization directory
    viz_dir = os.path.join(output_dir, f"prob_viz_{timestamp}")
    os.makedirs(viz_dir, exist_ok=True)
    
    # 1. Equity curve with confidence bands
    plt.figure(figsize=(12, 6))
    plt.plot(results['equity_curve'])
    plt.title('Equity Curve')
    plt.xlabel('Steps')
    plt.ylabel('Equity')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(viz_dir, 'equity_curve.png'), dpi=300)
    plt.close()
    
    # 2. Prediction confidence vs accuracy
    if results.get('prediction_hits') and results.get('confidence_scores'):
        # Calculate average confidence per step
        avg_confidences = [np.mean(conf) for conf in results['confidence_scores']]
        
        # Bin confidence scores and calculate accuracy per bin
        bins = np.linspace(0, 1, 11)  # 10 bins
        binned_accuracy = []
        binned_confidence = []
        
        for i in range(len(bins) - 1):
            lower, upper = bins[i], bins[i+1]
            mask = [(c >= lower and c < upper) for c in avg_confidences]
            
            if any(mask) and len(results['prediction_hits']) >= len(mask):
                # Get prediction hits for this bin
                bin_hits = [results['prediction_hits'][j] for j, m in enumerate(mask) if m and j < len(results['prediction_hits'])]
                if bin_hits:
                    accuracy = sum(bin_hits) / len(bin_hits)
                    confidence = (lower + upper) / 2
                    binned_accuracy.append(accuracy)
                    binned_confidence.append(confidence)
        
        # Plot calibration curve
        plt.figure(figsize=(8, 8))
        plt.plot(binned_confidence, binned_accuracy, 'o-', linewidth=2)
        plt.plot([0, 1], [0, 1], 'r--', linewidth=1)  # Perfect calibration line
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        plt.title('Model Calibration Curve')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(viz_dir, 'calibration_curve.png'), dpi=300)
        plt.close()
    
    # 3. Uncertainty vs returns
    if results.get('prediction_stds') and len(results['equity_curve']) > 1:
        # Calculate returns
        returns = [(results['equity_curve'][i] / results['equity_curve'][i-1] - 1) 
                  for i in range(1, len(results['equity_curve']))]
        
        # Get average uncertainty per step
        avg_uncertainty = [np.mean(std) for std in results['prediction_stds']]
        
        # Truncate to same length
        min_len = min(len(returns), len(avg_uncertainty))
        returns = returns[:min_len]
        uncertainty = avg_uncertainty[:min_len]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(uncertainty, returns, alpha=0.5)
        plt.title('Prediction Uncertainty vs Returns')
        plt.xlabel('Uncertainty (Average Std Dev)')
        plt.ylabel('Return')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(viz_dir, 'uncertainty_vs_returns.png'), dpi=300)
        plt.close()
    
    # 4. Trade sizing vs confidence
    if results.get('trades') and results.get('confidence_scores'):
        # Extract trade sizes
        trade_sizes = [trade.get('size_btc', 0) for trade in results.get('trades', [])]
        trade_steps = [trade.get('entry_step', 0) for trade in results.get('trades', [])]
        
        # Match trade confidence
        trade_confidences = []
        for step in trade_steps:
            if step < len(results['confidence_scores']):
                trade_confidences.append(np.mean(results['confidence_scores'][step]))
            else:
                trade_confidences.append(0.5)  # Default
        
        plt.figure(figsize=(10, 6))
        plt.scatter(trade_confidences, trade_sizes, alpha=0.6)
        plt.title('Trade Size vs Confidence')
        plt.xlabel('Confidence Score')
        plt.ylabel('Trade Size (BTC)')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(viz_dir, 'trade_size_vs_confidence.png'), dpi=300)
        plt.close()
    
    log(f"Probabilistic visualizations saved to {viz_dir}")

# Backtester class from the original backtest.py
class Backtester:
    """
    Backtester for evaluating probabilistic trading strategies.
    
    Features:
    - Position sizing based on prediction confidence
    - Multiple entry/exit strategies
    - Detailed performance metrics and visualizations
    - Risk-adjusted return calculations
    - Trade analysis and statistics
    """
    
    def __init__(self, initial_capital=100000, max_position_size=0.2, 
                 transaction_fee_pct=0.001, slippage_pct=0.001):
        """
        Initialize backtester with configuration parameters.
        
        Args:
            initial_capital (float): Starting capital in USD.
            max_position_size (float): Maximum position size as fraction of capital.
            transaction_fee_pct (float): Transaction fee as percentage of trade value.
            slippage_pct (float): Slippage as percentage of trade price.
        """
        self.initial_capital = initial_capital
        self.max_position_size = max_position_size
        self.transaction_fee_pct = transaction_fee_pct
        self.slippage_pct = slippage_pct
        
        # Performance metrics to track
        self.metrics = None
        
        # Results to track
        self.reset_results()
    
    def reset_results(self):
        """Reset all tracking variables for a new backtest run."""
        self.equity_curve = []
        self.positions = []
        self.trades = []
        self.capital = self.initial_capital
        self.current_position = 0.0
        self.entry_price = 0.0
        self.entry_time = None
        self.trade_count = 0
        self.profitable_trades = 0
        self.losing_trades = 0
        self.position_values = []
        self.total_fees = 0.0
        self.total_slippage = 0.0
        self.prediction_quality = []
        self.confidence_history = []
    
    def calculate_position_size(self, signal, confidence, price, volatility=None):
        """
        Calculate position size based on prediction confidence and market conditions.
        
        Args:
            signal (float): Trading signal strength (-1 to 1).
            confidence (float): Prediction confidence (0 to 1).
            price (float): Current asset price.
            volatility (float, optional): Current market volatility.
            
        Returns:
            float: Position size in units of the asset.
        """
        # Base position sizing on signal strength and confidence
        signal_strength = abs(signal)
        
        # Confidence factor (scales position based on prediction confidence)
        confidence_factor = 0.3 + (confidence * 0.7)  # Scale to 0.3-1.0 range
        
        # Volatility adjustment (optional)
        volatility_factor = 1.0
        if volatility is not None:
            # Reduce position size in high volatility
            volatility_factor = 1.0 / (1.0 + volatility)
        
        # Calculate position size as percentage of capital
        position_pct = signal_strength * confidence_factor * volatility_factor
        
        # Cap at maximum position size
        position_pct = min(position_pct, self.max_position_size)
        
        # Convert to asset units
        position_usd = self.capital * position_pct
        position_size = position_usd / price
        
        return position_size * np.sign(signal)  # Preserve direction
    
    def execute_trade(self, position_size, price, timestamp, confidence, slippage_override=None):
        """
        Execute a trade by updating the position and recording metrics.
        
        Args:
            position_size (float): Target position size (negative for short).
            price (float): Current price for the trade.
            timestamp: Timestamp of the trade.
            confidence (float): Confidence in the prediction.
            slippage_override (float, optional): Override default slippage.
            
        Returns:
            float: Executed price after slippage.
        """
        # Calculate position change
        position_change = position_size - self.current_position
        
        # If no change, skip
        if abs(position_change) < 1e-8:
            return price
        
        # Apply slippage
        slippage = slippage_override if slippage_override is not None else self.slippage_pct
        # Direction-aware slippage
        if position_change > 0:  # Buying
            executed_price = price * (1 + slippage)
        else:  # Selling
            executed_price = price * (1 - slippage)
        
        # Calculate trade value
        trade_value = abs(position_change) * executed_price
        
        # Calculate and deduct fees
        fee = trade_value * self.transaction_fee_pct
        self.total_fees += fee
        
        # Calculate slippage cost
        slippage_cost = abs(position_change) * abs(executed_price - price)
        self.total_slippage += slippage_cost
        
        # Update capital
        self.capital -= position_change * executed_price - fee
        
        # Check if closing a position or changing direction
        if self.current_position != 0 and (position_change * self.current_position <= 0 or position_size == 0):
            # Record completed trade
            trade_result = {
                'entry_time': self.entry_time,
                'exit_time': timestamp,
                'entry_price': self.entry_price,
                'exit_price': executed_price,
                'position': self.current_position,
                'pnl': (executed_price - self.entry_price) * self.current_position - fee,
                'pnl_pct': ((executed_price / self.entry_price) - 1) * 100 * np.sign(self.current_position),
                'fees': fee,
                'slippage': slippage_cost,
                'hold_time': (timestamp - self.entry_time).total_seconds() / 3600 if isinstance(timestamp, datetime) else None  # hours
            }
            
            self.trades.append(trade_result)
            
            # Update trade count
            self.trade_count += 1
            if trade_result['pnl'] > 0:
                self.profitable_trades += 1
            else:
                self.losing_trades += 1
        
        # If entering a new position or changing direction
        if self.current_position == 0 or (position_change * self.current_position < 0):
            self.entry_price = executed_price
            self.entry_time = timestamp
        
        # If partially modifying, calculate weighted average price
        elif position_change != 0 and position_change * self.current_position > 0:
            # Weighted average of entry prices
            self.entry_price = (self.entry_price * self.current_position + executed_price * position_change) / position_size
        
        # Update current position
        self.current_position = position_size
        
        # Record confidence
        self.confidence_history.append(confidence)
        
        return executed_price
    
    def update_metrics(self, price, timestamp, prediction=None, target=None):
        """
        Update performance metrics for each time step.
        
        Args:
            price (float): Current price.
            timestamp: Current timestamp.
            prediction (float, optional): Current price prediction.
            target (float, optional): Actual future price.
        """
        # Calculate portfolio value
        position_value = self.current_position * price
        total_value = self.capital + position_value
        
        # Record equity curve
        self.equity_curve.append({
            'timestamp': timestamp,
            'price': price,
            'position': self.current_position,
            'position_value': position_value,
            'capital': self.capital,
            'total_value': total_value
        })
        
        # Record position value
        self.position_values.append(position_value)
        
        # Record prediction quality if available
        if prediction is not None and target is not None:
            self.prediction_quality.append({
                'timestamp': timestamp,
                'prediction': prediction,
                'target': target,
                'error': prediction - target,
                'price': price
            })
    
    def run_backtest(self, data, signal_generator, strategy_params=None, verbose=True):
        """
        Run a backtest using the provided data and signal generator.
        
        Args:
            data (pandas.DataFrame): DataFrame with price data and features.
            signal_generator (callable): Function that generates signals from data.
            strategy_params (dict, optional): Parameters for the strategy.
            verbose (bool, optional): Whether to print progress updates.
            
        Returns:
            dict: Backtest results and metrics.
        """
        if strategy_params is None:
            strategy_params = {}
            
        # Reset results
        self.reset_results()
        
        # Process data chronologically
        timestamps = data.index
        for i, timestamp in enumerate(timestamps):
            if verbose and i % 100 == 0:
                log(f"Processing {i}/{len(timestamps)}")
                
            current_data = data.iloc[:i+1]
            current_price = current_data['close'].iloc[-1]
            
            # Generate signal and confidence
            signal, confidence = signal_generator(current_data, **strategy_params)
            
            # Determine position size
            if 'volatility' in current_data.columns:
                volatility = current_data['volatility'].iloc[-1]
                position_size = self.calculate_position_size(signal, confidence, current_price, volatility)
            else:
                position_size = self.calculate_position_size(signal, confidence, current_price)
                
            # Execute trade
            executed_price = self.execute_trade(position_size, current_price, timestamp, confidence)
            
            # Update metrics
            if i < len(timestamps) - 1:
                future_price = data['close'].iloc[i+1]
                self.update_metrics(current_price, timestamp, prediction=signal*current_price, target=future_price)
            else:
                self.update_metrics(current_price, timestamp)
                
        # Calculate performance metrics
        self.metrics = self.calculate_performance_metrics()
        
        if verbose:
            # Print summary
            log("\n--- Backtest Results ---")
            log(f"Total Capital: ${self.metrics['final_capital']:.2f} (Initial: ${self.initial_capital:.2f})")
            log(f"Return: {self.metrics['total_return']:.2f}%")
            log(f"Annualized Return: {self.metrics['annualized_return']:.2f}%")
            log(f"Sharpe Ratio: {self.metrics['sharpe_ratio']:.2f}")
            log(f"Max Drawdown: {self.metrics['max_drawdown']:.2f}%")
            log(f"Trades: {self.metrics['total_trades']} (Win Rate: {self.metrics['win_rate']:.2f}%)")
            log(f"Profit Factor: {self.metrics['profit_factor']:.2f}")
            
        return self.metrics
    
    def run_probabilistic_backtest(self, data, model, prediction_horizon=12, confidence_threshold=0.6, 
                                  entry_threshold=0.8, exit_threshold=0.2, max_hold_time=48, verbose=True):
        """
        Run a backtest using a probabilistic model for price prediction.
        
        Args:
            data (pandas.DataFrame): DataFrame with price data and features.
            model: Trained model that produces probabilistic predictions.
            prediction_horizon (int): Number of periods to look ahead for predictions.
            confidence_threshold (float): Minimum confidence required for trades.
            entry_threshold (float): Probability threshold for entries.
            exit_threshold (float): Probability threshold for exits.
            max_hold_time (int): Maximum holding time in periods.
            verbose (bool): Whether to print progress updates.
            
        Returns:
            dict: Backtest results and metrics.
        """
        # Define signal generator for probabilistic model
        def probabilistic_signal_generator(current_data, **params):
            # Prepare features for model input
            if hasattr(model, 'preprocess'):
                features = model.preprocess(current_data)
            else:
                # Assume data is already preprocessed
                features = current_data.iloc[-1].values.reshape(1, -1)
                
            # Get probabilistic prediction
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(features)[0]
                up_prob = probs[1]  # Probability of price going up
            else:
                # Try direct prediction if predict_proba not available
                pred = model.predict(features)[0]
                up_prob = 1.0 if pred > 0 else 0.0
                
            # Calculate signal and confidence
            # Signal: -1 (short) to +1 (long)
            # Confidence: 0 (low) to 1 (high)
            
            # Baseline: neutral position (no signal)
            signal = 0.0
            confidence = 0.0
            
            # Current position tracking
            current_position = 0.0
            if len(self.equity_curve) > 0 and 'position' in self.equity_curve[-1]:
                current_position = self.equity_curve[-1]['position']
            
            # Check hold time for existing position
            hold_time = 0
            if self.entry_time is not None and current_position != 0:
                current_time = current_data.index[-1]
                if isinstance(current_time, datetime) and isinstance(self.entry_time, datetime):
                    hold_time = (current_time - self.entry_time).total_seconds() / 3600
                else:
                    # Use number of bars if timestamps aren't datetime objects
                    # Find index of entry time in the data
                    try:
                        entry_idx = current_data.index.get_loc(self.entry_time)
                        current_idx = len(current_data) - 1
                        hold_time = current_idx - entry_idx
                    except:
                        hold_time = 0
            
            # Entry logic (if no position)
            if current_position == 0:
                if up_prob >= entry_threshold:
                    # Strong bullish signal
                    signal = 1.0
                    confidence = up_prob
                elif up_prob <= (1 - entry_threshold):
                    # Strong bearish signal
                    signal = -1.0
                    confidence = 1 - up_prob
            
            # Exit logic (if in position)
            elif hold_time >= max_hold_time:
                # Force exit if max hold time reached
                signal = 0.0
                confidence = 0.9  # High confidence for time-based exit
            elif (current_position > 0 and up_prob <= exit_threshold) or \
                 (current_position < 0 and up_prob >= (1 - exit_threshold)):
                # Exit if probability reverses beyond threshold
                signal = 0.0
                confidence = abs(0.5 - up_prob) * 2  # Scale to 0-1
            else:
                # Hold current position
                signal = np.sign(current_position)
                confidence = abs(up_prob - 0.5) * 2  # Scale to 0-1
            
            # Only trade if confidence is above threshold
            if confidence < confidence_threshold:
                signal = 0.0
                
            return signal, confidence
            
        # Run the backtest with our probabilistic signal generator
        return self.run_backtest(data, probabilistic_signal_generator, verbose=verbose)
    
    def calculate_performance_metrics(self):
        """
        Calculate comprehensive performance metrics.
        
        Returns:
            dict: Dictionary of performance metrics.
        """
        metrics = {}
        
        # Basic performance metrics
        if len(self.equity_curve) == 0:
            return {'error': 'No data available'}
            
        # Capital and returns
        initial_capital = self.initial_capital
        final_capital = self.equity_curve[-1]['total_value']
        total_return = ((final_capital / initial_capital) - 1) * 100
        
        # Convert equity curve to DataFrame for easier analysis
        if isinstance(self.equity_curve[0], dict):
            equity_df = pd.DataFrame(self.equity_curve)
        else:
            equity_df = pd.DataFrame(self.equity_curve, columns=['total_value'])
            
        # Basic metrics
        metrics['initial_capital'] = initial_capital
        metrics['final_capital'] = final_capital
        metrics['total_return'] = total_return
        metrics['total_trades'] = self.trade_count
        metrics['total_fees'] = self.total_fees
        metrics['total_slippage'] = self.total_slippage
        
        # Risk metrics
        if len(equity_df) > 1:
            # Daily returns
            if 'timestamp' in equity_df.columns:
                equity_df = equity_df.set_index('timestamp')
                equity_df = equity_df.resample('D').last().fillna(method='ffill')
                
            # Calculate returns
            returns = equity_df['total_value'].pct_change().dropna()
            
            # Annualized metrics (assuming daily data)
            n_periods = len(returns)
            years = n_periods / 252  # Trading days in a year
            if years > 0:
                metrics['annualized_return'] = ((1 + total_return/100) ** (1/years) - 1) * 100
            else:
                metrics['annualized_return'] = 0
            
            # Volatility
            if len(returns) > 1:
                volatility = returns.std() * np.sqrt(252)  # Annualized
                metrics['volatility'] = volatility * 100  # As percentage
                
                # Sharpe ratio (assuming risk-free rate of 0%)
                if volatility > 0:
                    sharpe_ratio = (metrics['annualized_return'] / 100) / volatility
                    metrics['sharpe_ratio'] = sharpe_ratio
                else:
                    metrics['sharpe_ratio'] = 0
                    
                # Sortino ratio (downside risk only)
                downside_returns = returns[returns < 0]
                if len(downside_returns) > 0:
                    downside_deviation = downside_returns.std() * np.sqrt(252)
                    if downside_deviation > 0:
                        sortino_ratio = (metrics['annualized_return'] / 100) / downside_deviation
                        metrics['sortino_ratio'] = sortino_ratio
                    else:
                        metrics['sortino_ratio'] = 0
                else:
                    metrics['sortino_ratio'] = float('inf') if metrics['annualized_return'] > 0 else 0
            else:
                metrics['volatility'] = 0
                metrics['sharpe_ratio'] = 0
                metrics['sortino_ratio'] = 0
                
            # Maximum drawdown
            if len(equity_df) > 1:
                equity_values = equity_df['total_value'].values
                max_dd = 0
                peak = equity_values[0]
                
                for value in equity_values:
                    if value > peak:
                        peak = value
                    dd = (peak - value) / peak
                    if dd > max_dd:
                        max_dd = dd
                        
                metrics['max_drawdown'] = max_dd * 100  # As percentage
            else:
                metrics['max_drawdown'] = 0
        else:
            # Default values if not enough data
            metrics['annualized_return'] = 0
            metrics['volatility'] = 0
            metrics['sharpe_ratio'] = 0
            metrics['sortino_ratio'] = 0
            metrics['max_drawdown'] = 0
            
        # Trade statistics
        if self.trade_count > 0:
            metrics['win_rate'] = (self.profitable_trades / self.trade_count) * 100
            
            # Extract PnL from trades
            pnl_values = [t['pnl'] for t in self.trades]
            winning_trades = [p for p in pnl_values if p > 0]
            losing_trades = [p for p in pnl_values if p <= 0]
            
            # Average trade metrics
            metrics['avg_trade_pnl'] = np.mean(pnl_values) if pnl_values else 0
            metrics['avg_win'] = np.mean(winning_trades) if winning_trades else 0
            metrics['avg_loss'] = np.mean(losing_trades) if losing_trades else 0
            metrics['largest_win'] = max(pnl_values) if pnl_values else 0
            metrics['largest_loss'] = min(pnl_values) if pnl_values else 0
            
            # Profit factor
            total_profit = sum(winning_trades) if winning_trades else 0
            total_loss = abs(sum(losing_trades)) if losing_trades else 0
            if total_loss > 0:
                metrics['profit_factor'] = total_profit / total_loss
            else:
                metrics['profit_factor'] = float('inf') if total_profit > 0 else 0
                
            # Recovery factor
            if metrics['max_drawdown'] > 0:
                metrics['recovery_factor'] = total_return / metrics['max_drawdown']
            else:
                metrics['recovery_factor'] = float('inf') if total_return > 0 else 0
                
            # Average hold time (if timestamps are datetime objects)
            hold_times = [t['hold_time'] for t in self.trades if 'hold_time' in t and t['hold_time'] is not None]
            metrics['avg_hold_time'] = np.mean(hold_times) if hold_times else 0  # In hours
        else:
            # Default values if no trades
            metrics['win_rate'] = 0
            metrics['avg_trade_pnl'] = 0
            metrics['avg_win'] = 0
            metrics['avg_loss'] = 0
            metrics['largest_win'] = 0
            metrics['largest_loss'] = 0
            metrics['profit_factor'] = 0
            metrics['recovery_factor'] = 0
            metrics['avg_hold_time'] = 0
            
        # Prediction quality metrics (if available)
        if self.prediction_quality:
            pq_df = pd.DataFrame(self.prediction_quality)
            metrics['prediction_accuracy'] = 1 - (np.mean(np.abs(pq_df['error'])) / np.mean(np.abs(pq_df['target'])))
            
            # Direction accuracy
            direction_correct = ((pq_df['prediction'] > pq_df['price']) & (pq_df['target'] > pq_df['price'])) | \
                               ((pq_df['prediction'] < pq_df['price']) & (pq_df['target'] < pq_df['price']))
            metrics['direction_accuracy'] = direction_correct.mean() * 100
            
        return metrics
    
    def plot_equity_curve(self, results=None, benchmark=None, title="Equity Curve", save_path=None):
        """
        Plot equity curve showing portfolio performance over time.
        
        Args:
            results (dict, optional): Results from backtest run.
            benchmark (pandas.Series, optional): Benchmark returns for comparison.
            title (str, optional): Plot title.
            save_path (str, optional): Path to save the plot.
            
        Returns:
            matplotlib.figure.Figure: The generated figure.
        """
        if results is None:
            results = {'equity_curve': self.equity_curve}
        
        if not results.get('equity_curve'):
            log("[WARNING] No equity curve data available for plotting.")
            return None
            
        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(results['equity_curve'])
        
        # Set up plot
        plt.figure(figsize=(12, 8))
        
        # Plot equity curve
        if 'timestamp' in equity_df.columns:
            equity_df = equity_df.set_index('timestamp')
        
        plt.subplot(3, 1, (1, 2))  # Top 2/3 of the figure
        plt.title(title, fontsize=14)
        
        # Plot portfolio value
        plt.plot(equity_df['total_value'], label="Portfolio", linewidth=2)
        
        # Plot buy & hold if price data is available
        if 'price' in equity_df.columns:
            initial_capital = results.get('initial_capital', self.initial_capital)
            price_series = equity_df['price']
            shares = initial_capital / price_series.iloc[0]
            buy_hold = price_series * shares
            plt.plot(buy_hold, label="Buy & Hold", linestyle="--", alpha=0.7)
            
        # Plot benchmark if provided
        if benchmark is not None:
            # Normalize benchmark to match starting value
            if 'timestamp' in equity_df.columns:
                benchmark = benchmark.reindex(equity_df.index, method='ffill')
            normalized_benchmark = benchmark / benchmark.iloc[0] * equity_df['total_value'].iloc[0]
            plt.plot(normalized_benchmark, label="Benchmark", linestyle="-.", alpha=0.7)
            
        plt.grid(True, alpha=0.3)
        plt.ylabel("Portfolio Value ($)")
        plt.legend()
        
        # Plot drawdowns
        plt.subplot(3, 1, 3)  # Bottom 1/3 of the figure
        
        if len(equity_df) > 1:
            # Calculate drawdown
            portfolio_value = equity_df['total_value'].values
            drawdowns = np.zeros_like(portfolio_value)
            
            # Calculate drawdown percentage
            peak = portfolio_value[0]
            for i in range(len(portfolio_value)):
                if portfolio_value[i] > peak:
                    peak = portfolio_value[i]
                drawdowns[i] = (peak - portfolio_value[i]) / peak * 100
                
            # Plot drawdowns
            plt.fill_between(equity_df.index, drawdowns, 0, color='red', alpha=0.3)
            plt.grid(True, alpha=0.3)
            plt.ylabel("Drawdown (%)")
            plt.xlabel("Date")
            
            # Format y-axis as percentage
            plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1f}%'.format(y)))
            
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path)
            
        return plt.gcf()
    
    def plot_trade_analysis(self, results=None, title="Trade Analysis", save_path=None):
        """
        Plot trade analysis charts.
        
        Args:
            results (dict, optional): Results from backtest run.
            title (str, optional): Plot title.
            save_path (str, optional): Path to save the plot.
            
        Returns:
            matplotlib.figure.Figure: The generated figure.
        """
        if results is None:
            results = {'trades': self.trades}
            
        trades = results.get('trades', [])
        if not trades:
            log("[WARNING] No trade data available for analysis.")
            return None
            
        # Create DataFrame from trades
        trade_df = pd.DataFrame(trades)
        
        # Set up plot
        plt.figure(figsize=(12, 10))
        
        # 1. Trade Outcomes (Profit/Loss Distribution)
        plt.subplot(2, 2, 1)
        plt.title("Trade Outcomes Distribution")
        
        if 'pnl' in trade_df.columns:
            sns.histplot(trade_df['pnl'], bins=20, kde=True)
            plt.axvline(x=0, color='r', linestyle='--', alpha=0.7)
            plt.grid(True, alpha=0.3)
            plt.xlabel("Profit/Loss ($)")
            
        # 2. Win/Loss Comparison
        plt.subplot(2, 2, 2)
        plt.title("Win/Loss Comparison")
        
        if 'pnl' in trade_df.columns:
            # Separate wins and losses
            wins = trade_df[trade_df['pnl'] > 0]['pnl']
            losses = trade_df[trade_df['pnl'] <= 0]['pnl'].abs()
            
            # Create bar data
            labels = ['Win Count', 'Loss Count', 'Avg Win', 'Avg Loss']
            values = [len(wins), len(losses), wins.mean() if len(wins) else 0, losses.mean() if len(losses) else 0]
            
            # Plot with different colors
            bars = plt.bar(labels, values, color=['green', 'red', 'green', 'red'])
            plt.grid(True, alpha=0.3)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        '{:.2f}'.format(height), ha='center', va='bottom')
                        
        # 3. Trade Metrics Over Time
        plt.subplot(2, 2, 3)
        plt.title("Running Profit/Loss")
        
        if 'pnl' in trade_df.columns and 'exit_time' in trade_df.columns:
            # Sort by exit time
            trade_df = trade_df.sort_values('exit_time')
            
            # Calculate cumulative P&L
            trade_df['cumulative_pnl'] = trade_df['pnl'].cumsum()
            
            # Plot
            plt.plot(range(len(trade_df)), trade_df['cumulative_pnl'], marker='o', markersize=3)
            plt.grid(True, alpha=0.3)
            plt.xlabel("Trade Number")
            plt.ylabel("Cumulative P&L ($)")
            
        # 4. Hold Time vs. Profit
        plt.subplot(2, 2, 4)
        plt.title("Hold Time vs. Profit")
        
        if 'hold_time' in trade_df.columns and 'pnl' in trade_df.columns:
            plt.scatter(trade_df['hold_time'], trade_df['pnl'], alpha=0.6, c=trade_df['pnl'] > 0, cmap='coolwarm')
            plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
            plt.grid(True, alpha=0.3)
            plt.xlabel("Hold Time (hours)")
            plt.ylabel("Profit/Loss ($)")
            
            # Add regression line
            if len(trade_df) > 1:
                try:
                    from scipy import stats
                    slope, intercept, r_value, p_value, std_err = stats.linregress(trade_df['hold_time'], trade_df['pnl'])
                    plt.plot(trade_df['hold_time'], intercept + slope * trade_df['hold_time'], 'r', alpha=0.3)
                    plt.text(0.05, 0.95, f'R²: {r_value**2:.2f}', transform=plt.gca().transAxes, verticalalignment='top')
                except:
                    pass
                
        plt.tight_layout()
        plt.suptitle(title, fontsize=16, y=1.02)
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path)
            
        return plt.gcf()
    
    def plot_prediction_analysis(self, results=None, title="Prediction Analysis", save_path=None):
        """
        Plot prediction analysis charts to evaluate model performance.
        
        Args:
            results (dict, optional): Results from backtest run.
            title (str, optional): Plot title.
            save_path (str, optional): Path to save the plot.
            
        Returns:
            matplotlib.figure.Figure: The generated figure.
        """
        if results is None:
            results = {'prediction_quality': self.prediction_quality}
            
        prediction_data = results.get('prediction_quality', [])
        if not prediction_data:
            log("[WARNING] No prediction data available for analysis.")
            return None
            
        # Create DataFrame from prediction data
        pred_df = pd.DataFrame(prediction_data)
        
        # Set up plot
        plt.figure(figsize=(12, 10))
        
        # 1. Prediction vs Actual
        plt.subplot(2, 2, 1)
        plt.title("Prediction vs Actual")
        
        if 'prediction' in pred_df.columns and 'target' in pred_df.columns:
            plt.scatter(pred_df['prediction'], pred_df['target'], alpha=0.6)
            
            # Add perfect prediction line
            min_val = min(pred_df['prediction'].min(), pred_df['target'].min())
            max_val = max(pred_df['prediction'].max(), pred_df['target'].max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
            
            plt.grid(True, alpha=0.3)
            plt.xlabel("Predicted Value")
            plt.ylabel("Actual Value")
            
            # Add R² value
            if len(pred_df) > 1:
                try:
                    from scipy import stats
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        pred_df['prediction'], pred_df['target']
                    )
                    plt.text(0.05, 0.95, f'R²: {r_value**2:.2f}', transform=plt.gca().transAxes, 
                            verticalalignment='top')
                except:
                    pass
                    
        # 2. Prediction Error Distribution
        plt.subplot(2, 2, 2)
        plt.title("Prediction Error Distribution")
        
        if 'error' in pred_df.columns:
            sns.histplot(pred_df['error'], bins=20, kde=True)
            plt.axvline(x=0, color='r', linestyle='--', alpha=0.7)
            plt.grid(True, alpha=0.3)
            plt.xlabel("Prediction Error")
            
            # Add mean and std
            mean_err = pred_df['error'].mean()
            std_err = pred_df['error'].std()
            plt.text(0.05, 0.95, f'Mean: {mean_err:.2f}\nStd: {std_err:.2f}', 
                    transform=plt.gca().transAxes, verticalalignment='top')
                    
        # 3. Prediction Error Over Time
        plt.subplot(2, 2, 3)
        plt.title("Prediction Error Over Time")
        
        if 'timestamp' in pred_df.columns and 'error' in pred_df.columns:
            plt.plot(pred_df['timestamp'], pred_df['error'], alpha=0.6)
            plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
            plt.grid(True, alpha=0.3)
            plt.xlabel("Time")
            plt.ylabel("Error")
            
            # Format dates on x-axis if timestamps are datetime objects
            if isinstance(pred_df['timestamp'].iloc[0], datetime):
                plt.gcf().autofmt_xdate()
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                
        # 4. Direction Accuracy
        plt.subplot(2, 2, 4)
        plt.title("Direction Prediction Accuracy")
        
        if 'prediction' in pred_df.columns and 'target' in pred_df.columns and 'price' in pred_df.columns:
            # Calculate direction accuracy
            pred_df['pred_direction'] = (pred_df['prediction'] > pred_df['price']).astype(int)
            pred_df['actual_direction'] = (pred_df['target'] > pred_df['price']).astype(int)
            pred_df['correct'] = (pred_df['pred_direction'] == pred_df['actual_direction']).astype(int)
            
            # Create confusion matrix
            cm = confusion_matrix(pred_df['actual_direction'], pred_df['pred_direction'])
            
            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
            plt.xlabel('Predicted Direction')
            plt.ylabel('Actual Direction')
            
            # Calculate accuracy
            accuracy = pred_df['correct'].mean() * 100
            plt.text(0.05, 0.05, f'Accuracy: {accuracy:.2f}%', 
                    transform=plt.gca().transAxes, verticalalignment='bottom')
            
        plt.tight_layout()
        plt.suptitle(title, fontsize=16, y=1.02)
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path)
            
        return plt.gcf()
    
    def save_report(self, results, report_dir='backtest_reports'):
        """
        Save a comprehensive backtest report with metrics and visualizations.
        
        Args:
            results (dict): Results from backtest run.
            report_dir (str, optional): Directory to save the report.
            
        Returns:
            str: Path to the saved report directory.
        """
        # Create unique timestamp for report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(report_dir, f"backtest_{timestamp}")
        
        # Create directory if it doesn't exist
        os.makedirs(report_path, exist_ok=True)
        
        # 1. Save metrics as JSON
        metrics = results.get('metrics', self.metrics)
        if metrics:
            metrics_path = os.path.join(report_path, "metrics.json")
            
            # Convert any non-serializable values to strings
            serializable_metrics = {}
            for k, v in metrics.items():
                if isinstance(v, (int, float, str, bool, list, dict)) and v is not None:
                    # Handle infinity and NaN
                    if isinstance(v, float) and (np.isinf(v) or np.isnan(v)):
                        serializable_metrics[k] = str(v)
                    else:
                        serializable_metrics[k] = v
                else:
                    serializable_metrics[k] = str(v)
                    
            with open(metrics_path, 'w') as f:
                json.dump(serializable_metrics, f, indent=4)
        
        # 2. Save equity curve plot
        equity_plot_path = os.path.join(report_path, "equity_curve.png")
        self.plot_equity_curve(results, title="Backtest Equity Curve", save_path=equity_plot_path)
        
        # 3. Save trade analysis plot
        trade_plot_path = os.path.join(report_path, "trade_analysis.png")
        self.plot_trade_analysis(results, title="Trade Analysis", save_path=trade_plot_path)
        
        # 4. Save prediction analysis plot if available
        if results.get('prediction_quality') or self.prediction_quality:
            pred_plot_path = os.path.join(report_path, "prediction_analysis.png")
            self.plot_prediction_analysis(results, title="Prediction Analysis", save_path=pred_plot_path)
        
        # 5. Save raw trade data as CSV
        trades = results.get('trades', self.trades)
        if trades:
            trades_path = os.path.join(report_path, "trades.csv")
            pd.DataFrame(trades).to_csv(trades_path, index=False)
            
        log(f"[INFO] Backtest report saved to: {report_path}")
        return report_path


if __name__ == "__main__":
    import pandas as pd
    from src.utils.config import get_config
    
    # Load configuration
    config = get_config()
    
    # Create synthetic data for testing
    dates = pd.date_range('2020-01-01', periods=10000, freq='5min')
    df = pd.DataFrame({
        'timestamp': dates,
        'close': np.random.random(10000) * 1000 + 40000,  # Random prices around 40-50k
        'open': np.random.random(10000) * 1000 + 40000,
        'high': np.random.random(10000) * 1000 + 41000,
        'low': np.random.random(10000) * 1000 + 39000,
        'volume': np.random.random(10000) * 100  # Random volume
    })
    
    # Add some technical indicators for testing
    df['SMA9'] = df['close'].rolling(9).mean()
    df['SMA21'] = df['close'].rolling(21).mean()
    df['SMA50'] = df['close'].rolling(50).mean()
    df['SMA200'] = df['close'].rolling(200).mean()
    df['RSI14'] = df['close'].pct_change().apply(lambda x: max(0, x)).rolling(14).mean() / df['close'].pct_change().abs().rolling(14).mean() * 100
    
    # Fill NaN values
    df = df.fillna(method='bfill')
    
    # Create random agent for testing
    class RandomAgent:
        def select_action(self, obs, hidden):
            direction = random.uniform(-1, 1)
            fraction = random.uniform(0, 1)
            return [direction, fraction], 0, 0, [], [], [], 0, 0, None
    
    # Test backtest function
    log("Testing backtest function...")
    agent = RandomAgent()
    metrics, equity_curves, trade_histories = run_backtest(df, agent, config.config, episodes=2)
    
    if metrics:
        log("Backtest completed successfully")
        log(f"Number of metrics: {len(metrics)}")
        log(f"Number of equity curves: {len(equity_curves)}")
        log(f"Number of trade histories: {len(trade_histories)}")
        
        # Test report generation
        report = generate_backtest_report(
            metrics, equity_curves, trade_histories, config.config
        )
        log("\nSample Report:")
        log(report)
        
        # Test plotting
        log("\nGenerating plots...")
        plot_backtest_results(
            equity_curves, trade_histories, config.config
        )
    else:
        log("Backtest failed")
