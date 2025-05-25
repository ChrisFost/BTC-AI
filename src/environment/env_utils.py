#!/usr/bin/env python
"""
Environment Utilities

This module provides utility functions for trading environments
that are common across different implementations.
"""

import os
import gc
import logging
import numpy as np
import torch
import time
import psutil
import importlib
from typing import Dict, List, Any, Union, Optional, Tuple, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_env.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("env_utils")

# Global settings
DEBUG = False


def log(message: str, level: str = "info") -> None:
    """
    Unified logging function to maintain consistent logging across modules.
    
    Args:
        message: Message to log
        level: Log level (debug, info, warning, error, critical)
    """
    # Convert level string to lowercase for case-insensitive comparison
    level = level.lower()
    
    if level == "debug":
        # Use standard logger level check instead of global DEBUG flag
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(message)
    elif level == "info":
        logger.info(message)
    elif level == "warning":
        logger.warning(message)
    elif level == "error":
        logger.error(message)
    elif level == "critical":
        logger.critical(message)
    else:
        # Default to info for unrecognized levels
        logger.info(message)


def set_debug(debug_mode: bool = True) -> None:
    """
    Set global debug mode.
    
    Args:
        debug_mode: Whether to enable debug mode
    """
    global DEBUG
    DEBUG = debug_mode
    log(f"Debug mode {'enabled' if debug_mode else 'disabled'}")


def optimize_memory() -> int:
    """
    Force garbage collection to free memory.
    
    Returns:
        int: Number of objects collected
    """
    # Clear PyTorch CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Force garbage collection
    collected = gc.collect()
    
    # Log memory usage
    if DEBUG:
        process = psutil.Process(os.getpid())
        memory_use = process.memory_info().rss / 1024 / 1024  # MB
        log(f"Memory usage after optimization: {memory_use:.2f} MB", "debug")
    
    return collected


def get_kraken_fee(rolling_volume: float) -> float:
    """
    Get Kraken trading fee based on 30-day rolling volume.
    
    Args:
        rolling_volume: 30-day trading volume in USD
        
    Returns:
        float: Fee as a fraction (e.g., 0.0026 for 0.26%)
    """
    # Kraken fee tiers (as of March 2023)
    if rolling_volume < 50000:
        return 0.0026  # 0.26% for < $50k
    elif rolling_volume < 100000:
        return 0.0024  # 0.24% for $50k-$100k
    elif rolling_volume < 250000:
        return 0.0022  # 0.22% for $100k-$250k
    elif rolling_volume < 500000:
        return 0.0020  # 0.20% for $250k-$500k
    elif rolling_volume < 1000000:
        return 0.0018  # 0.18% for $500k-$1M
    elif rolling_volume < 2500000:
        return 0.0016  # 0.16% for $1M-$2.5M
    elif rolling_volume < 5000000:
        return 0.0014  # 0.14% for $2.5M-$5M
    elif rolling_volume < 10000000:
        return 0.0012  # 0.12% for $5M-$10M
    else:
        return 0.0010  # 0.10% for $10M+


def compute_risk_adjusted_reward(
    base_reward: float,
    profits: List[float],
    losses: List[float],
    returns: List[float],
    bucket: str,
    closed_trades: List,
    episode_days: float,
    config: Dict[str, Any]
) -> float:
    """
    Legacy function for backward compatibility.
    Delegates to the dedicated reward module.
    
    Args:
        base_reward: Base reward from trading
        profits: List of profitable trade amounts
        losses: List of loss amounts
        returns: List of return percentages
        bucket: Trading timeframe bucket
        closed_trades: List of closed trades
        episode_days: Episode duration in days
        config: Configuration dictionary
        
    Returns:
        float: Calculated reward value
    """
    try:
        # Import dynamically from env_rewards
        rewards_module = importlib.import_module("src.environment.env_rewards")
        calc_reward = rewards_module.compute_risk_adjusted_reward
        
        return calc_reward(
            base_reward, profits, losses, returns, bucket, closed_trades, episode_days, config
        )
    except ImportError:
        # Fallback implementation if module cannot be imported
        log("Warning: Could not import rewards module, using fallback reward calculation", level="warning")
        # Simple fallback calculation - just use base reward
        return base_reward


def count_trainable_parameters(model) -> int:
    """
    Count trainable parameters in a neural network model.
    
    Args:
        model: PyTorch model
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_drawdown(returns: List[float]) -> float:
    """
    Calculate maximum drawdown from a list of returns.
    
    Args:
        returns: List of period returns
        
    Returns:
        float: Maximum drawdown as a fraction (0-1)
    """
    if not returns:
        return 0.0
    
    # Calculate cumulative returns
    cum_returns = np.cumprod(1 + np.array(returns))
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(cum_returns)
    
    # Calculate drawdowns
    drawdowns = (running_max - cum_returns) / running_max
    
    # Return maximum drawdown
    return float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0


def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0, annualization_factor: float = 252) -> float:
    """
    Calculate Sharpe ratio from a list of returns.
    
    Args:
        returns: List of period returns
        risk_free_rate: Risk-free rate
        annualization_factor: Factor to annualize returns (252 for daily, 12 for monthly, etc.)
        
    Returns:
        float: Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0
    
    returns_array = np.array(returns)
    excess_returns = returns_array - risk_free_rate
    
    # Calculate mean and standard deviation
    mean_return = np.mean(excess_returns)
    std_return = np.std(excess_returns, ddof=1)  # Using sample standard deviation
    
    if std_return == 0:
        return 0.0
    
    # Calculate and return annualized Sharpe ratio
    return mean_return / std_return * np.sqrt(annualization_factor)


def calculate_sortino_ratio(returns: List[float], risk_free_rate: float = 0.0, annualization_factor: float = 252) -> float:
    """
    Calculate Sortino ratio from a list of returns (penalizes only downside volatility).
    
    Args:
        returns: List of period returns
        risk_free_rate: Risk-free rate
        annualization_factor: Factor to annualize returns
        
    Returns:
        float: Sortino ratio
    """
    if len(returns) < 2:
        return 0.0
    
    # Convert to numpy array
    returns_array = np.array(returns)
    excess_returns = returns_array - risk_free_rate
    
    # Calculate mean return
    mean_return = np.mean(excess_returns)
    
    # Isolate downside returns
    downside_returns = np.minimum(excess_returns, 0)
    
    # Calculate downside deviation (standard deviation of negative returns)
    downside_dev = np.sqrt(np.mean(np.square(downside_returns)))
    
    if downside_dev == 0:
        return 0.0 if mean_return <= 0 else float('inf')
    
    # Calculate and return annualized Sortino ratio
    return mean_return / downside_dev * np.sqrt(annualization_factor)


def calculate_calmar_ratio(returns: List[float], annualization_factor: float = 252) -> float:
    """
    Calculate Calmar ratio (return divided by maximum drawdown).
    
    Args:
        returns: List of period returns
        annualization_factor: Factor to annualize returns
        
    Returns:
        float: Calmar ratio
    """
    if len(returns) < 2:
        return 0.0
    
    # Calculate annualized return
    total_return = np.prod(1 + np.array(returns)) - 1
    period_count = len(returns)
    annualized_return = (1 + total_return) ** (annualization_factor / period_count) - 1
    
    # Calculate maximum drawdown
    max_drawdown = calculate_drawdown(returns)
    
    if max_drawdown == 0:
        return 0.0 if annualized_return <= 0 else float('inf')
    
    # Calculate and return Calmar ratio
    return annualized_return / max_drawdown


def create_directory_if_needed(directory_path: str) -> bool:
    """
    Create a directory if it doesn't exist.
    
    Args:
        directory_path: Path to directory
        
    Returns:
        bool: True if directory exists or was created, False otherwise
    """
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
            log(f"Created directory: {directory_path}")
            return True
        except Exception as e:
            log(f"Error creating directory {directory_path}: {e}", "error")
            return False
    return True


def save_config(config: Dict[str, Any], file_path: str) -> bool:
    """
    Save configuration to a JSON file.
    
    Args:
        config: Configuration dictionary
        file_path: Path to save the file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import json
        
        # Ensure directory exists
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)
            
        log(f"Saved configuration to {file_path}")
        return True
    except Exception as e:
        log(f"Error saving configuration to {file_path}: {e}", "error")
        return False


def load_config(file_path: str, default_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load configuration from a JSON file, with fallback to default config.
    
    Args:
        file_path: Path to the configuration file
        default_config: Default configuration to use if file not found
        
    Returns:
        dict: Loaded configuration
    """
    if default_config is None:
        default_config = {}
        
    try:
        import json
        
        if not os.path.exists(file_path):
            log(f"Configuration file {file_path} not found, using default", "warning")
            return default_config.copy()
            
        with open(file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        # Fill in any missing parameters with defaults
        for key, value in default_config.items():
            if key not in config:
                config[key] = value
                
        log(f"Loaded configuration from {file_path}")
        return config
    except Exception as e:
        log(f"Error loading configuration from {file_path}: {e}", "error")
        return default_config.copy()


def time_function(func: Callable) -> Callable:
    """
    Decorator to time function execution.
    
    Args:
        func: Function to time
        
    Returns:
        callable: Wrapped function
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        log(f"Function {func.__name__} executed in {elapsed:.4f} seconds", "debug")
        return result
    return wrapper


def calculate_win_rate(profits: List[float], losses: List[float]) -> float:
    """
    Calculate win rate from profits and losses.
    
    Args:
        profits: List of profitable trades
        losses: List of losing trades
        
    Returns:
        float: Win rate as a fraction (0-1)
    """
    total_trades = len(profits) + len(losses)
    if total_trades == 0:
        return 0.0
    return len(profits) / total_trades


def calculate_profit_factor(profits: List[float], losses: List[float]) -> float:
    """
    Calculate profit factor (sum of profits / sum of losses).
    
    Args:
        profits: List of profitable trades
        losses: List of losing trades
        
    Returns:
        float: Profit factor (> 1 is profitable)
    """
    total_profit = sum(profits)
    total_loss = sum(losses)
    
    if total_loss == 0:
        return 2.0 if total_profit > 0 else 1.0  # Avoid division by zero
        
    return total_profit / total_loss


def calculate_expectancy(profits: List[float], losses: List[float]) -> float:
    """
    Calculate mathematical expectancy (average trade outcome).
    
    Args:
        profits: List of profitable trades
        losses: List of losing trades
        
    Returns:
        float: Expectancy value
    """
    win_rate = calculate_win_rate(profits, losses)
    
    if not profits:
        avg_win = 0
    else:
        avg_win = sum(profits) / len(profits)
        
    if not losses:
        avg_loss = 0
    else:
        avg_loss = sum(losses) / len(losses)
        
    # Expectancy formula: (Win Rate × Average Win) - (Loss Rate × Average Loss)
    return (win_rate * avg_win) - ((1 - win_rate) * avg_loss)


def calculate_average_hold_time(closed_trades: List) -> float:
    """
    Calculate average hold time for completed trades.
    
    Args:
        closed_trades: List of closed trades
        
    Returns:
        float: Average hold time in bars
    """
    if not closed_trades:
        return 0.0
        
    hold_times = []
    for trade in closed_trades:
        if isinstance(trade, tuple) and len(trade) > 2:
            hold_times.append(trade[2])
        elif hasattr(trade, 'hold_time'):
            hold_times.append(trade.hold_time)
            
    if not hold_times:
        return 0.0
        
    return sum(hold_times) / len(hold_times)


def bars_to_timeframe(bars: int) -> str:
    """
    Convert number of bars to human-readable timeframe.
    
    Args:
        bars: Number of 5-minute bars
        
    Returns:
        str: Human-readable timeframe
    """
    minutes = bars * 5
    
    if minutes < 60:
        return f"{minutes}min"
    elif minutes < 1440:  # Less than a day
        hours = minutes / 60
        return f"{hours:.1f}h"
    elif minutes < 10080:  # Less than a week
        days = minutes / 1440
        return f"{days:.1f}d"
    elif minutes < 43200:  # Less than a month
        weeks = minutes / 10080
        return f"{weeks:.1f}w"
    else:
        months = minutes / 43200
        return f"{months:.1f}mo"


def timeframe_to_bars(amount: float, unit: str) -> int:
    """
    Convert timeframe to number of 5-minute bars.
    
    Args:
        amount: Amount of time
        unit: Time unit (hour(s), day(s), week(s), month(s))
        
    Returns:
        int: Number of 5-minute bars
    """
    if unit == "minute(s)":
        return int(amount / 5)
    elif unit == "hour(s)":
        return int(amount * 12)
    elif unit == "day(s)":
        return int(amount * 288)
    elif unit == "week(s)":
        return int(amount * 7 * 288)
    elif unit == "month(s)":
        return int(amount * 30 * 288)
    return int(amount)  # Default to raw bars


def calculate_trade_metrics(closed_trades: List, initial_capital: float) -> Dict[str, float]:
    """
    Calculate comprehensive trading metrics.
    
    Args:
        closed_trades: List of closed trades
        initial_capital: Initial capital amount
        
    Returns:
        dict: Dictionary of trading metrics
    """
    if not closed_trades:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "net_profit": 0.0,
            "net_profit_percentage": 0.0,
            "average_profit": 0.0,
            "average_loss": 0.0,
            "largest_profit": 0.0,
            "largest_loss": 0.0,
            "average_hold_time": 0.0,
            "expectancy": 0.0
        }
    
    # Separate profits and losses
    profits = []
    losses = []
    
    for trade in closed_trades:
        if isinstance(trade, tuple) and len(trade) > 0:
            profit = trade[0]
        elif hasattr(trade, 'profit'):
            profit = trade.profit
        else:
            continue
            
        if profit > 0:
            profits.append(profit)
        else:
            losses.append(abs(profit))
    
    # Calculate basic metrics
    total_trades = len(profits) + len(losses)
    win_rate = calculate_win_rate(profits, losses)
    profit_factor = calculate_profit_factor(profits, losses)
    
    # Calculate profit statistics
    net_profit = sum(profits) - sum(losses)
    net_profit_percentage = (net_profit / initial_capital) * 100 if initial_capital > 0 else 0
    
    average_profit = sum(profits) / len(profits) if profits else 0
    average_loss = sum(losses) / len(losses) if losses else 0
    
    largest_profit = max(profits) if profits else 0
    largest_loss = max(losses) if losses else 0
    
    # Calculate hold time
    average_hold_time = calculate_average_hold_time(closed_trades)
    
    # Calculate expectancy
    expectancy = calculate_expectancy(profits, losses)
    
    return {
        "total_trades": total_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "net_profit": net_profit,
        "net_profit_percentage": net_profit_percentage,
        "average_profit": average_profit,
        "average_loss": average_loss,
        "largest_profit": largest_profit,
        "largest_loss": largest_loss,
        "average_hold_time": average_hold_time,
        "expectancy": expectancy
    }


def make_env_creator(df, config, device="cpu"):
    """
    Create a function that creates a fresh environment instance.
    
    This is needed for proper environment parallelization.
    
    Args:
        df (pandas.DataFrame): DataFrame with market data.
        config (dict): Configuration parameters.
        device (str, optional): Device to use for tensor operations. Defaults to "cpu".
        
    Returns:
        callable: Function that creates a new environment instance.
    """
    # CRITICAL FIX: Validate parameters and add signature compatibility checks
    if df is None:
        raise ValueError("DataFrame cannot be None for environment creation")
    
    if config is None:
        config = {}
    
    # Validate device parameter
    valid_devices = ["cpu", "cuda", "auto"]
    if device not in valid_devices:
        log(f"Invalid device '{device}'. Using 'cpu' as fallback.", "warning")
        device = "cpu"
    
    # Add environment creation validation
    def _creator():
        try:
            # Import dynamically with error handling
            env_base_module = importlib.import_module("src.environment.env_base")
            create_environment = env_base_module.create_environment
            
            # Test if create_environment accepts device parameter
            import inspect
            sig = inspect.signature(create_environment)
            if 'device' in sig.parameters:
                # New signature with device parameter
                env = create_environment(df, config, device)
            else:
                # Legacy signature without device parameter
                log("Environment creation function doesn't support device parameter. Using legacy signature.", "warning")
                env = create_environment(df, config)
                
            # Validate created environment
            if env is None:
                raise RuntimeError("Environment creation returned None")
                
            # Basic environment validation
            if not hasattr(env, 'reset') or not hasattr(env, 'step'):
                raise RuntimeError("Created environment missing required methods (reset, step)")
                
            return env
            
        except ImportError as e:
            log(f"Error importing environment module: {e}", level="error")
            raise RuntimeError(f"Failed to import environment creation function: {e}")
        except Exception as e:
            log(f"Error creating environment: {e}", level="error")
            raise RuntimeError(f"Environment creation failed: {e}")
    
    return _creator


def _test_debug_log(test_message: str = "Test debug message") -> bool:
    """
    Special function for testing debug logging.
    Temporarily enables DEBUG, logs a debug message, and returns True.
    This is only used for testing purposes.
    
    Args:
        test_message: Message to log for testing
        
    Returns:
        bool: Always True if executed
    """
    global DEBUG
    original_debug = DEBUG
    
    try:
        DEBUG = True
        log(test_message, level="DEBUG")
        return True
    finally:
        DEBUG = original_debug


if __name__ == "__main__":
    # Simple test code
    print("Environment utilities module")
    
    # Test logging
    log("This is a test message")
    
    # Test performance metrics
    returns = [0.01, -0.005, 0.02, -0.01, 0.015, 0.01, -0.02, 0.03]
    print(f"Max drawdown: {calculate_drawdown(returns):.4f}")
    print(f"Sharpe ratio: {calculate_sharpe_ratio(returns):.4f}")
    print(f"Sortino ratio: {calculate_sortino_ratio(returns):.4f}")
    
    # Test timeframe conversion
    bars = timeframe_to_bars(1, "day(s)")
    print(f"1 day = {bars} bars")
    print(f"{bars} bars = {bars_to_timeframe(bars)}")
