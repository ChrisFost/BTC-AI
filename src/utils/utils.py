#!/usr/bin/env python
"""
Utility functions for the Bitcoin AI project.

This module provides a variety of utility functions used across the project.
"""

import os
import time
import datetime
import gc
import sys
import random
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
import seaborn as sns
import importlib
from collections import deque
from matplotlib.colors import LinearSegmentedColormap

# Import bucket_goals module dynamically
try:
    bucket_goals_module = importlib.import_module("src.utils.bucket_goals")
    create_goal_provider = bucket_goals_module.create_goal_provider
except ImportError:
    print("Warning: Could not import bucket_goals module")
    def create_goal_provider(*args, **kwargs):
        print("Warning: create_goal_provider function not available")
        return None

def log(msg, log_file=None):
    """
    Consistent logging function with optional file output.
    
    Args:
        msg (str): Message to log.
        log_file (str, optional): Path to log file.
    """
    timestamp = datetime.datetime.now().strftime('%H:%M:%S')
    formatted_msg = f"[{timestamp}] {msg}"
    print(formatted_msg)
    
    if log_file:
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(formatted_msg + '\n')
        except Exception as e:
            print(f"[{timestamp}] Error writing to log file: {e}")

def optimize_memory():
    """
    Force garbage collection and CUDA memory cleanup to prevent memory leaks.
    """
    # Force garbage collection
    gc.collect()
    
    # CUDA memory cleanup if available
    if torch.cuda.is_available():
        # Empty cache
        torch.cuda.empty_cache()
        
        # Advanced: detect and clear leaked tensors (may have performance impact)
        # Only do this periodically, not every step
        if random.random() < 0.05:  # 5% chance to run full cleanup
            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj):
                        if obj.device.type == 'cuda':
                            # Check if tensor is not being used by Python
                            if sys.getrefcount(obj) <= 2:
                                del obj
                except Exception:
                    pass
            # Empty cache again after tensor cleanup
            torch.cuda.empty_cache()

def get_gradient_checkpointing_config(model_type="transformer", model_size=None):
    """
    Get recommended gradient checkpointing configuration for memory optimization
    during long training sessions.
    
    Gradient checkpointing trades computation for memory by not storing all
    intermediate activations and recomputing them during the backward pass.
    
    Args:
        model_type (str): Type of model ("transformer", "lstm", "gru", "cnn", etc.)
        model_size (int, optional): Model size in millions of parameters
        
    Returns:
        dict: Configuration for gradient checkpointing
    """
    # Default configuration
    config = {
        "use_checkpoint": True,
        "checkpoint_layers": True,
        "checkpoint_freq": 2,    # Apply to every 2nd layer
        "preserve_rng_state": True,
        "use_reentrant": False,  # More memory efficient but less compatible
    }
    
    # Adjust based on model type
    if model_type == "transformer":
        config["checkpoint_freq"] = 1  # Checkpoint every layer for transformers
    elif model_type in ["lstm", "gru", "rnn"]:
        config["checkpoint_freq"] = 3  # Less frequent for RNNs
    
    # Adjust based on model size
    if model_size:
        if model_size > 100:  # Large models > 100M parameters
            config["checkpoint_freq"] = 1
        elif model_size < 10:  # Small models < 10M parameters
            config["use_checkpoint"] = False  # Might not be necessary
    
    return config

def apply_gradient_checkpointing(model, config=None):
    """
    Apply gradient checkpointing to a model to reduce memory usage during training.
    
    Args:
        model (torch.nn.Module): PyTorch model
        config (dict, optional): Checkpointing configuration
        
    Returns:
        torch.nn.Module: Model with gradient checkpointing applied
    """
    if config is None:
        config = get_gradient_checkpointing_config()
    
    if not config.get("use_checkpoint", True):
        return model
    
    # Import here to avoid circular imports
    from torch.utils.checkpoint import checkpoint
    
    # For transformer models, apply checkpointing to transformer layers
    if hasattr(model, "transformer"):
        if hasattr(model.transformer, "layers"):
            # Apply to transformer layers with specified frequency
            freq = config.get("checkpoint_freq", 2)
            layers = model.transformer.layers
            
            # Replace forward methods with checkpointed versions
            for i, layer in enumerate(layers):
                if i % freq == 0:
                    # Store original forward
                    original_forward = layer.forward
                    
                    # Define checkpointed forward (closure to capture original_forward)
                    def make_checkpointed_forward(orig_forward):
                        def checkpointed_forward(*args, **kwargs):
                            return checkpoint(
                                orig_forward, 
                                *args, 
                                preserve_rng_state=config.get("preserve_rng_state", True),
                                use_reentrant=config.get("use_reentrant", False)
                            )
                        return checkpointed_forward
                    
                    # Replace the forward method
                    layer.forward = make_checkpointed_forward(original_forward)
    
    # For general sequential models
    elif hasattr(model, "sequential") or hasattr(model, "layers"):
        seq = getattr(model, "sequential", None) or getattr(model, "layers", None)
        if isinstance(seq, torch.nn.Sequential):
            freq = config.get("checkpoint_freq", 2)
            
            # Apply to layers with specified frequency
            for i, module in enumerate(seq):
                if i % freq == 0 and not isinstance(module, torch.nn.ReLU) and not isinstance(module, torch.nn.Dropout):
                    # Store original forward
                    original_forward = module.forward
                    
                    # Define checkpointed forward
                    def make_checkpointed_forward(orig_forward):
                        def checkpointed_forward(*args, **kwargs):
                            return checkpoint(
                                orig_forward, 
                                *args, 
                                preserve_rng_state=config.get("preserve_rng_state", True),
                                use_reentrant=config.get("use_reentrant", False)
                            )
                        return checkpointed_forward
                    
                    # Replace the forward method
                    module.forward = make_checkpointed_forward(original_forward)
    
    return model

def optimize_memory_for_long_training(model, optimizer, config):
    """
    Comprehensive memory optimization for long training sessions.
    
    This function applies multiple memory optimization techniques:
    1. Garbage collection and CUDA cache clearing
    2. Gradient checkpointing for large models
    3. Mixed precision training configuration
    4. Gradient accumulation setup
    5. Smart batching configuration
    
    Args:
        model (torch.nn.Module): PyTorch model
        optimizer (torch.optim.Optimizer): PyTorch optimizer
        config (dict): Training configuration
        
    Returns:
        tuple: (model, optimizer, updated_config)
    """
    # Basic memory optimization
    optimize_memory()
    
    # Apply gradient checkpointing if model is large enough
    if config.get("USE_GRADIENT_CHECKPOINTING", False):
        model_size = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6  # Millions of params
        ckpt_config = get_gradient_checkpointing_config(
            model_type=config.get("MODEL_TYPE", "transformer"),
            model_size=model_size
        )
        model = apply_gradient_checkpointing(model, ckpt_config)
        
    # Configure mixed precision if enabled
    if config.get("USE_MIXED_PRECISION", False) and torch.cuda.is_available():
        # Make sure amp is available in torch.cuda
        if not hasattr(torch.cuda, 'amp'):
            from torch.cuda import amp
        else:
            amp = torch.cuda.amp
            
        # Create GradScaler for mixed precision training
        if not hasattr(model, 'scaler'):
            model.scaler = amp.GradScaler()
            
        # Set mixed precision dtype based on GPU capability
        gpu_capability = torch.cuda.get_device_capability()
        if gpu_capability[0] >= 7:  # Volta or newer (tensor cores)
            config["MIXED_PRECISION_DTYPE"] = torch.float16
        else:
            config["MIXED_PRECISION_DTYPE"] = torch.bfloat16
    
    # Configure gradient accumulation for larger effective batch size
    if config.get("GRADIENT_ACCUMULATION_STEPS", 1) > 1:
        # Make sure optimizer zeroes gradients only after accumulation steps
        if not hasattr(optimizer, '_orig_zero_grad'):
            optimizer._orig_zero_grad = optimizer.zero_grad
            
            def _accumulation_zero_grad(set_to_none=False):
                # Only zero grads when needed (after update)
                if getattr(model, '_grad_accumulation_count', 0) == 0:
                    optimizer._orig_zero_grad(set_to_none=set_to_none)
                    
            optimizer.zero_grad = _accumulation_zero_grad
    
    # Configure smart batching if data doesn't fit in memory
    if config.get("USE_SMART_BATCHING", False):
        # Adjust batch size based on available memory
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory
            mem_utilization = torch.cuda.memory_allocated() / gpu_mem
            
            # Reduce batch size if memory is nearly full
            if mem_utilization > 0.8 and config.get("BATCH_SIZE", 32) > 8:
                config["BATCH_SIZE"] = max(8, config["BATCH_SIZE"] // 2)
                log(f"Reduced batch size to {config['BATCH_SIZE']} due to high memory utilization")
    
    return model, optimizer, config

def get_kraken_fee(rolling_30d_volume, order_type="taker"):
    """
    Calculate Kraken trading fee based on 30-day volume.
    
    Args:
        rolling_30d_volume (float): Trading volume in USD over past 30 days.
        order_type (str, optional): Order type ("taker" or "maker"). Defaults to "taker".
    
    Returns:
        float: Fee rate as a decimal (e.g., 0.0026 for 0.26%).
    """
    fee_tiers = [
        (0, 0.0040), (10000, 0.0035), (50000, 0.0024), (100000, 0.0022),
        (250000, 0.0020), (500000, 0.0018), (1000000, 0.0016), (2500000, 0.0014),
        (5000000, 0.0012), (10000000, 0.0010)
    ]
    
    # Find the appropriate fee tier
    for threshold, rate in fee_tiers:
        if rolling_30d_volume >= threshold:
            current_rate = rate
    
    # Apply maker discount if applicable
    if order_type.lower() == "maker":
        current_rate *= 0.9  # Maker orders typically get a 10% discount
    
    return current_rate

def compute_risk_adjusted_reward(base_reward, profits, losses, returns, bucket, closed_trades, episode_days, config):
    """
    Calculate risk-adjusted reward with bucket-specific bonuses and portfolio risk penalty.
    
    Enhanced to include comprehensive risk adjustment based on the portfolio risk model.
    Uses the BucketGoalProvider to manage bucket-specific goals.
    
    Args:
        base_reward (float): Base reward from trading.
        profits (list): List of profitable trade amounts.
        losses (list): List of loss amounts (positive values).
        returns (list): List of return percentages.
        bucket (str): Trading bucket ("Scalping", "Short", "Medium", or "Long").
        closed_trades (list): List of closed trades (profit, percentage_gain, hold_time).
        episode_days (float): Duration of the episode in days.
        config (dict): Configuration parameters.
    
    Returns:
        float: Risk-adjusted reward.
    """
    # Create a goal provider to manage bucket-specific goals
    goal_provider = create_goal_provider(config)
    
    # Calculate Sharpe ratio
    sharpe = np.mean(returns) / (np.std(returns) + 1e-8) if len(returns) > 1 else 0.0
    
    # Calculate profit factor
    profit_factor = sum(profits) / (sum(losses) + 1e-8) if losses else 1.0
    
    # Calculate total profit
    total_profit = sum(profits) - sum(losses)
    
    # Calculate Sortino ratio (downside risk only)
    negative_returns = [r for r in returns if r < 0]
    sortino = np.mean(returns) / (np.std(negative_returns) + 1e-8) if negative_returns else sharpe
    
    # Calculate max drawdown
    if closed_trades:
        cumulative_returns = np.cumsum([profit for profit, _, _ in closed_trades])
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (peak - cumulative_returns) / peak
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0
    else:
        max_drawdown = 0.0
    
    # Calculate Calmar ratio (return / max drawdown)
    calmar = total_profit / (max_drawdown * config.get("INITIAL_CAPITAL", 100000.0) + 1e-8)
    
    # Calculate win rate
    total_trades = len(closed_trades)
    winning_trades = sum(1 for profit, _, _ in closed_trades if profit > 0)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
    
    # Calculate average trade
    avg_trade = total_profit / total_trades if total_trades > 0 else 0.0
    
    # Calculate average win and average loss
    avg_win = np.mean(profits) if profits else 0.0
    avg_loss = np.mean(losses) if losses else 0.0
    
    # Calculate profit-to-loss ratio
    profit_loss_ratio = avg_win / (avg_loss + 1e-8) if avg_loss > 0 else 1.0
    
    # Calculate risk of ruin
    # Simple approximation using win rate and profit-loss ratio
    if win_rate > 0 and win_rate < 1 and profit_loss_ratio > 0:
        z = (profit_loss_ratio * win_rate - (1 - win_rate)) / (profit_loss_ratio * win_rate + (1 - win_rate))
        risk_of_ruin = ((1 - z) / (1 + z)) ** total_trades if z > 0 else 1.0
    else:
        risk_of_ruin = 0.0 if win_rate == 1.0 else 1.0
    
    # Prepare metrics dictionary for goal provider
    metrics = {
        "monthly_profit_estimate": total_profit / episode_days * 30 if episode_days > 0 else 0.0,
        "yearly_profit_estimate": total_profit / episode_days * 365 if episode_days > 0 else 0.0,
        "win_rate": win_rate,
        "total_trades": total_trades,
        "profit_factor": profit_factor,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "avg_profit_per_trade": avg_trade,
        "calmar": calmar,
    }
    
    # Calculate percentage of trades in the target gain range for Medium/Long buckets
    if bucket in ["Medium", "Long"]:
        num_good_trades = sum(1 for _, pg, _ in closed_trades 
                             if config.get("min_gain_per_holding", 25.0) <= pg <= config.get("max_gain_per_holding", 50.0))
        good_trades_pct = (num_good_trades / total_trades * 100) if total_trades > 0 else 0.0
        metrics["good_trades_pct"] = good_trades_pct
    
    # Get bonus from goal provider
    bonus = goal_provider.get_bonus_for_bucket(bucket, metrics, base_reward)
    
    # Risk adjustment factors
    risk_penalty = 0.0
    
    # Higher drawdown = higher penalty
    if max_drawdown > 0.1:  # More than 10% drawdown
        drawdown_penalty = (max_drawdown - 0.1) * 5  # Scales up quickly after 10%
        risk_penalty += drawdown_penalty * abs(base_reward) * 0.2
    
    # Higher risk of ruin = higher penalty
    if risk_of_ruin > 0.05:  # More than 5% risk of ruin
        ruin_penalty = risk_of_ruin * 2
        risk_penalty += ruin_penalty * abs(base_reward) * 0.1
    
    # Combine reward components with risk adjustment
    risk_adjusted_reward = (
        base_reward +                 # Base profit/loss
        0.5 * sharpe +                # Reward for risk-adjusted return
        0.3 * sortino +               # Reward for downside risk management
        0.2 * calmar +                # Reward for drawdown control
        0.3 * profit_factor +         # Reward for profit/loss ratio
        bonus -                       # Bucket-specific bonus from goal provider
        risk_penalty                  # Risk-based penalty
    )
    
    return risk_adjusted_reward

def calculate_portfolio_metrics(positions, current_price, capital, initial_capital, returns):
    """
    Calculate comprehensive portfolio performance and risk metrics.
    
    Args:
        positions (list): List of current positions.
        current_price (float): Current asset price.
        capital (float): Current capital.
        initial_capital (float): Initial capital.
        returns (list): Historical returns.
        
    Returns:
        dict: Dictionary of portfolio metrics.
    """
    # Calculate basic position metrics
    total_btc = sum(p["size_btc"] for p in positions)
    position_values = [p["size_btc"] * current_price for p in positions]
    total_position_value = sum(position_values)
    
    # Calculate portfolio value
    portfolio_value = capital + total_position_value
    
    # Calculate exposure metrics
    exposure_percentage = total_position_value / portfolio_value if portfolio_value > 0 else 0.0
    
    # Calculate return metrics
    total_return = (portfolio_value / initial_capital - 1) * 100 if initial_capital > 0 else 0.0
    
    # Calculate risk concentration (Herfindahl index)
    if total_position_value > 0 and len(position_values) > 1:
        proportions = [value / total_position_value for value in position_values]
        risk_concentration = sum(p * p for p in proportions)
        
        # Ideal concentration is 1/n (equal distribution)
        ideal_concentration = 1.0 / len(positions)
        concentration_risk = (risk_concentration - ideal_concentration) / (1.0 - ideal_concentration)
        concentration_risk = max(0.0, concentration_risk)  # Ensure non-negative
    else:
        risk_concentration = 1.0 if len(positions) > 0 else 0.0
        concentration_risk = 0.0
    
    # Calculate position diversity (inverse of concentration)
    position_diversity = 1.0 / max(risk_concentration, 1e-8) if risk_concentration > 0 else 0.0
    position_diversity = min(position_diversity, len(positions)) if len(positions) > 0 else 1.0
    position_diversity = position_diversity / max(len(positions), 1)  # Normalize to 0-1
    
    # Calculate historical volatility
    volatility = np.std(returns) * np.sqrt(252) if len(returns) >= 5 else 0.0  # Annualized
    
    # Calculate Sharpe ratio
    sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252) if len(returns) >= 5 else 0.0
    
    # Calculate max drawdown
    if returns:
        # Convert returns to cumulative returns
        cumulative_returns = np.cumprod(1 + np.array(returns)) - 1
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (peak - cumulative_returns) / (peak + 1)
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0
    else:
        max_drawdown = 0.0
    
    # Value at Risk (VaR) calculation - 95% confidence
    if len(returns) >= 10:
        # Sort returns in ascending order
        sorted_returns = sorted(returns)
        # Get the 5th percentile
        var_index = int(0.05 * len(sorted_returns))
        var_95 = sorted_returns[var_index] if var_index < len(sorted_returns) else sorted_returns[0]
        value_at_risk = -portfolio_value * var_95  # Negative return = loss
    else:
        # Default VaR estimation
        value_at_risk = portfolio_value * 0.05  # Assume 5% daily VaR
    
    # Conditional Value at Risk (CVaR) / Expected Shortfall - 95% confidence
    if len(returns) >= 10:
        var_index = int(0.05 * len(sorted_returns))
        worst_returns = sorted_returns[:var_index+1]
        cvar_95 = np.mean(worst_returns) if worst_returns else sorted_returns[0]
        conditional_var = -portfolio_value * cvar_95
    else:
        conditional_var = portfolio_value * 0.07  # Assume 7% CVaR (worse than VaR)
    
    # Overall risk score (0-1 scale)
    # Weighted combination of risk factors
    risk_weights = {
        "concentration": 0.2,
        "drawdown": 0.3,
        "var": 0.2,
        "volatility": 0.3
    }
    
    # Normalize components to 0-1 scale
    normalized_concentration = min(1.0, concentration_risk)
    normalized_drawdown = min(1.0, max_drawdown * 5)  # >20% drawdown is critical
    normalized_var = min(1.0, (value_at_risk / portfolio_value) * 10) if portfolio_value > 0 else 0.0  # >10% VaR is critical
    normalized_volatility = min(1.0, volatility * 2) if volatility > 0 else 0.0  # >50% volatility is critical
    
    overall_risk_score = (
        risk_weights["concentration"] * normalized_concentration +
        risk_weights["drawdown"] * normalized_drawdown +
        risk_weights["var"] * normalized_var +
        risk_weights["volatility"] * normalized_volatility
    )
    
    # Return comprehensive metrics dictionary
    return {
        "total_btc_exposure": total_btc,
        "total_position_value": total_position_value,
        "portfolio_value": portfolio_value,
        "exposure_percentage": exposure_percentage,
        "total_return_pct": total_return,
        "risk_concentration": risk_concentration,
        "concentration_risk": concentration_risk, 
        "position_diversity": position_diversity,
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "value_at_risk": value_at_risk,
        "var_pct": value_at_risk / portfolio_value if portfolio_value > 0 else 0.0,
        "conditional_var": conditional_var,
        "cvar_pct": conditional_var / portfolio_value if portfolio_value > 0 else 0.0,
        "overall_risk_score": overall_risk_score
    }

def calculate_metrics(env, rewards=None, equities=None, trades=None, initial_capital=None):
    """
    Calculate backtesting metrics for an episode.
    
    Has two calling conventions:
    1. Original: calculate_metrics(env) - Used by most modules
    2. Extended: calculate_metrics(df, rewards, equities, trades, initial_capital) - Used by backtesting
    
    Args:
        env: Trading environment instance or DataFrame with price data if using second convention
        rewards: List of rewards for each step (only for extended convention)
        equities: List of equity values for each step (only for extended convention)
        trades: List of completed trades (only for extended convention)
        initial_capital: Starting capital amount (only for extended convention)
        
    Returns:
        dict: Dictionary of trading metrics.
    """
    # Check if this is the extended call with multiple parameters
    if rewards is not None:
        # We have the expanded call with multiple parameters
        # Calculate metrics based on the provided data
        if not equities or len(equities) == 0:
            return {
                'total_profit': 0,
                'profit_factor': 0,
                'win_rate': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'total_trades': 0
            }
        
        # Calculate basic metrics
        final_equity = equities[-1]
        total_profit = final_equity - initial_capital
        profit_pct = (total_profit / initial_capital) * 100
        
        # Calculate drawdowns
        max_dd = 0
        peak = equities[0]
        for equity in equities:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak
            max_dd = max(max_dd, drawdown)
        
        # Calculate Sharpe ratio
        if len(equities) > 1:
            returns = [(equities[i] / equities[i-1]) - 1 for i in range(1, len(equities))]
            sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Trade metrics
        total_trades = len(trades) if trades else 0
        winning_trades = sum(1 for trade in trades if trade.get('profit', 0) > 0) if trades else 0
        losing_trades = total_trades - winning_trades
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate profit factor
        total_profits = sum(trade.get('profit', 0) for trade in trades if trade.get('profit', 0) > 0) if trades else 0
        total_losses = sum(abs(trade.get('profit', 0)) for trade in trades if trade.get('profit', 0) < 0) if trades else 0
        profit_factor = total_profits / total_losses if total_losses > 0 else (1 if total_profits > 0 else 0)
        
        # Return metrics dictionary
        return {
            'total_profit': total_profit,
            'profit_pct': profit_pct,
            'profit_factor': profit_factor,
            'win_rate': win_rate,
            'max_drawdown': max_dd,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'final_equity': final_equity
        }
    
    # Original implementation for single environment parameter
    # Basic profitability metrics
    net_profit = env.capital - env.initial_capital
    total_trades = len(env.closed_trades)
    
    # If no trades, return minimal metrics
    if total_trades == 0:
        return {
            'net_profit': 0,
            'profit_pct': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'avg_profit': 0,
            'avg_win': 0, 
            'avg_loss': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'risk_adjusted_return': 0
        }
    
    # Extract wins and losses
    wins = [t[0] for t in env.closed_trades if t[0] > 0]
    losses = [abs(t[0]) for t in env.closed_trades if t[0] <= 0]
    
    # Calculate win/loss metrics
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    win_rate = len(wins) / total_trades if total_trades > 0 else 0
    profit_factor = sum(wins) / (sum(losses) + 1e-8) if losses else float('inf')
    
    # Calculate Sharpe ratio
    sharpe = np.mean(env.returns) / (np.std(env.returns) + 1e-8) if env.returns else 0
    
    # Create equity curve and drawdown analysis
    equity_curve = [env.initial_capital] + [env.initial_capital + sum([t[0] for t in env.closed_trades[:i+1]]) 
                                          for i in range(total_trades)]
    
    # Calculate maximum drawdown
    peak_equity = env.initial_capital
    max_drawdown = 0
    drawdown_durations = []
    drawdown_start = None
    
    for i, equity in enumerate(equity_curve):
        if equity > peak_equity:
            peak_equity = equity
            # If we were in a drawdown and now we've recovered, record duration
            if drawdown_start is not None:
                drawdown_durations.append(i - drawdown_start)
                drawdown_start = None
        else:
            drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
            # If this is the start of a new drawdown, record it
            if drawdown_start is None and drawdown > 0:
                drawdown_start = i
    
    # Record final drawdown duration if still in drawdown at the end
    if drawdown_start is not None:
        drawdown_durations.append(len(equity_curve) - drawdown_start)
    
    # Calculate average drawdown duration
    avg_drawdown_duration = np.mean(drawdown_durations) if drawdown_durations else 0
    
    # Trade duration analysis
    hold_times = [t[2] for t in env.closed_trades]
    avg_hold = np.mean(hold_times) if hold_times else 0
    longest_hold = max(hold_times) if hold_times else 0
    shortest_hold = min(hold_times) if hold_times else 0
    
    # Trade frequency metrics
    trades_per_day = total_trades / (env.current_step / 288) if env.current_step > 0 else 0

# Bucket-specific metrics
    if hasattr(env, 'bucket'):
        bucket = env.bucket
        
        if bucket == "Scalping":
            monthly_profit = net_profit / (env.current_step / (288 * 30)) if env.current_step > 0 else 0
            target_min = getattr(env, 'monthly_target_min', 15.0) * 0.01 * env.initial_capital
            target_max = getattr(env, 'monthly_target_max', 30.0) * 0.01 * env.initial_capital
            within_target = target_min <= monthly_profit <= target_max
            
        elif bucket == "Short":
            yearly_profit = net_profit / (env.current_step / (288 * 365)) if env.current_step > 0 else 0
            target_min = getattr(env, 'yearly_target_min', 100.0) * 0.01 * env.initial_capital
            target_max = getattr(env, 'yearly_target_max', 200.0) * 0.01 * env.initial_capital
            within_target = target_min <= yearly_profit <= target_max
            
        else:  # Medium/Long
            # Percentage of trades within target gain
            target_min = getattr(env, 'min_gain_per_holding', 25.0)
            target_max = getattr(env, 'max_gain_per_holding', 50.0)
            good_trade_pct = sum(1 for _, pg, _ in env.closed_trades if target_min <= pg <= target_max) / total_trades if total_trades > 0 else 0
            within_target = good_trade_pct >= 0.6  # At least 60% of trades hit target
    else:
        within_target = False
    
    # Order book approximation metrics
    avg_liquidity = 0
    avg_spread = 0
    
    if hasattr(env, 'liquidity_history') and env.liquidity_history:
        avg_liquidity = np.mean(env.liquidity_history)
    
    if hasattr(env, 'spread_history') and env.spread_history:
        avg_spread = np.mean(env.spread_history)
    
    # Position size limits impact analysis
    size_limits_applied = 0
    avg_size_limitation = 0.0
    
    if hasattr(env, '_calculate_risk_adjusted_size'):
        # If the environment has risk-adjusted sizing, it likely tracked size limitations
        # In a more advanced implementation, the environment would track these metrics directly
        # For now, we'll estimate based on trade sizes
        if total_trades > 0:
            # Calculate coefficient of variation for trade sizes as a proxy for size limitation
            trade_sizes = [t[0] / t[1] * 100 for t in env.closed_trades if abs(t[1]) > 1e-8]  # Amount/percentage
            if trade_sizes:
                size_variation = np.std(trade_sizes) / (np.mean(trade_sizes) + 1e-8)
                # Higher variation suggests more size limitations were applied
                avg_size_limitation = min(1.0, size_variation * 2)
                # Estimate count of size-limited trades
                size_limits_applied = int(total_trades * avg_size_limitation * 0.5)  # Conservative estimate
    
    # Risk metrics from src.environment.env_base if available
    overall_risk_score = 0.0
    risk_metrics = {}
    
    if hasattr(env, 'get_risk_metrics'):
        try:
            risk_metrics = env.get_risk_metrics()
            overall_risk_score = risk_metrics.get("overall_risk_score", 0.0)
        except Exception as e:
            log(f"Error getting risk metrics: {e}")
    
    # Calculate additional risk metrics if not provided by environment
    if not risk_metrics and hasattr(env, 'positions') and hasattr(env, 'capital'):
        try:
            current_price = env.df.loc[env.current_step, "close"] if hasattr(env, 'df') else 0.0
            risk_metrics = calculate_portfolio_metrics(
                env.positions, 
                current_price, 
                env.capital, 
                env.initial_capital, 
                env.returns
            )
            overall_risk_score = risk_metrics.get("overall_risk_score", 0.0)
        except Exception as e:
            log(f"Error calculating portfolio metrics: {e}")
    
    # Return comprehensive metrics dictionary
    metrics = {
        "net_profit": net_profit,
        "total_trades": total_trades,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "avg_drawdown_duration": avg_drawdown_duration,
        "avg_hold": avg_hold,
        "longest_hold": longest_hold,
        "shortest_hold": shortest_hold,
        "trades_per_day": trades_per_day,
        "within_target": within_target,
        "avg_liquidity": avg_liquidity,
        "avg_spread": avg_spread,
        "size_limits_applied": size_limits_applied,
        "avg_size_limitation": avg_size_limitation,
        "overall_risk_score": overall_risk_score
    }
    
    # Add risk metrics if available
    if risk_metrics:
        for key, value in risk_metrics.items():
            if key not in metrics:
                metrics[key] = value
    
    return metrics

def calculate_env_metrics(env_idx, envs):
    """
    Safely extract and calculate metrics for a specific environment in a vectorized setup.
    
    Enhanced to include risk metrics when available.
    
    Args:
        env_idx (int): Environment index.
        envs: Vectorized environments object.
        
    Returns:
        dict: Dictionary of trading metrics.
    """
    try:
        # First approach: try to get closed_trades via get_attr
        closed_trades = envs.get_attr("closed_trades")[env_idx]
        capital = envs.get_attr("capital")[env_idx]
        initial_capital = envs.get_attr("initial_capital")[env_idx]
        current_step = envs.get_attr("current_step")[env_idx]
        window_size = envs.get_attr("window_size")[env_idx]
        
        # Create metrics structure similar to what calculate_metrics would return
        metrics = {}
        
        # Basic metrics
        metrics["net_profit"] = capital - initial_capital
        metrics["total_trades"] = len(closed_trades)
        
        # Profit/loss metrics
        wins = [t[0] for t in closed_trades if t[0] > 0]
        losses = [abs(t[0]) for t in closed_trades if t[0] <= 0]
        metrics["avg_win"] = np.mean(wins) if wins else 0
        metrics["avg_loss"] = np.mean(losses) if losses else 0
        metrics["win_rate"] = len(wins) / metrics["total_trades"] if metrics["total_trades"] > 0 else 0
        metrics["profit_factor"] = sum(wins) / (sum(losses) + 1e-8) if losses else float('inf')
        
        # Try to get returns from the environment
        try:
            returns = envs.env_method("get_returns", indices=[env_idx])[0]
            metrics["sharpe"] = np.mean(returns) / (np.std(returns) + 1e-8) if returns else 0
        except:
            # Fallback if method doesn't exist
            metrics["sharpe"] = 0.0
        
        # Try to get risk metrics if available
        try:
            risk_metrics = envs.env_method("get_risk_metrics", indices=[env_idx])[0]
            # Add risk metrics to the result
            for key, value in risk_metrics.items():
                metrics[key] = value
        except:
            # Fallback if method doesn't exist
            metrics["overall_risk_score"] = 0.0
        
        # More complex metrics that we could add later
        metrics["max_drawdown"] = 0.0  # Placeholder - would need equity curve
        metrics["avg_hold"] = np.mean([t[2] for t in closed_trades]) if closed_trades else 0
        metrics["longest_hold"] = max([t[2] for t in closed_trades]) if closed_trades else 0
        metrics["shortest_hold"] = min([t[2] for t in closed_trades]) if closed_trades else 0
        metrics["trades_per_day"] = metrics["total_trades"] / ((current_step - window_size) / 288) if current_step > window_size else 0
        
        # Size limit metrics (estimated)
        metrics["size_limits_applied"] = 0
        metrics["avg_size_limitation"] = 0.0
        
        # Default value for bucket-specific metrics
        metrics["within_target"] = True
        
        return metrics
    except Exception as e:
        log(f"[WARNING] Failed to calculate metrics for env {env_idx}: {e}")
        # Return a default metrics dict with zeros
        return {
            "net_profit": 0.0,
            "total_trades": 0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "avg_hold": 0.0,
            "longest_hold": 0.0,
            "shortest_hold": 0.0,
            "trades_per_day": 0.0,
            "within_target": False,
            "size_limits_applied": 0,
            "avg_size_limitation": 0.0,
            "overall_risk_score": 0.0
        }

def measure_gpu_usage():
    """
    Measure current GPU memory usage as a percentage.
    
    Returns:
        float: GPU memory usage as a fraction (0.0-1.0).
    """
    if not torch.cuda.is_available():
        return 0.0
    
    try:
        allocated = torch.cuda.memory_allocated(0)
        reserved = torch.cuda.memory_reserved(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory
        
        # Return the higher of allocated or reserved memory
        return max(allocated, reserved) / total_memory
    except Exception as e:
        log(f"[WARNING] Error measuring GPU usage: {e}")
        return 0.0

def get_optimal_gpu_targets():
    """
    Determine optimal GPU target parameters based on system capabilities.
    
    Returns:
        tuple: (low_target, high_target) - Optimal GPU usage targets (0.0-1.0).
    """
    if not torch.cuda.is_available():
        return 0.65, 0.85  # Default values when no GPU
        
    try:
        # Get GPU properties
        props = torch.cuda.get_device_properties(0)
        total_memory_gb = props.total_memory / (1024**3)
        
        # Adjust targets based on GPU memory
        if total_memory_gb >= 24:  # High-end GPUs (24GB+)
            return 0.7, 0.9
        elif total_memory_gb >= 16:  # Mid-range GPUs (16GB)
            return 0.65, 0.85
        elif total_memory_gb >= 8:  # Budget GPUs (8GB)
            return 0.6, 0.8
        else:  # Low memory GPUs
            return 0.5, 0.75
    except Exception as e:
        log(f"[WARNING] Error determining GPU targets: {e}")
        # Return default values if any error occurs
        return 0.65, 0.85

def check_multi_gpu():
    """
    Check for multiple GPUs and return their IDs if available.
    
    Returns:
        list: List of usable GPU IDs.
    """
    if not torch.cuda.is_available():
        return []
        
    num_gpus = torch.cuda.device_count()
    if num_gpus <= 1:
        return []
        
    # Check if GPUs have enough memory (> 4GB)
    usable_gpus = []
    for i in range(num_gpus):
        try:
            gpu_properties = torch.cuda.get_device_properties(i)
            # Convert bytes to GB
            memory_gb = gpu_properties.total_memory / (1024**3)
            if memory_gb >= 4.0:
                usable_gpus.append(i)
                log(f"Found usable GPU {i}: {gpu_properties.name} with {memory_gb:.1f}GB memory")
        except Exception as e:
            log(f"[WARNING] Error checking GPU {i}: {e}")
            
    return usable_gpus

def validate_dataframe(df):
    """
    Validate the dataframe before training to catch issues.
    
    Args:
        df: Pandas DataFrame with trading data.
    
    Returns:
        tuple: (is_valid, message) - Validation result and message.
    """
    if df is None:
        return False, "DataFrame is None"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    # Check for NaN values
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        return False, f"DataFrame contains {nan_count} NaN values"
    
    # Check for required columns
    required_cols = ['close', 'high', 'low', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return False, f"Missing required columns: {missing_cols}"
    
    # Check for minimum length (need at least window size + some extra)
    min_length = 388  # Default window size (288) + 100
    if len(df) < min_length:
        return False, f"DataFrame too short: {len(df)} rows, need at least {min_length}"
    
    # Check for monotonically increasing index
    if not df.index.is_monotonic_increasing:
        return False, "DataFrame index is not monotonically increasing"
    
    # Check for non-negative prices
    if (df['close'] <= 0).any() or (df['high'] <= 0).any() or (df['low'] <= 0).any():
        return False, "DataFrame contains non-positive price values"
    
    # Ensure high >= close >= low
    if (df['high'] < df['close']).any() or (df['close'] < df['low']).any() or (df['high'] < df['low']).any():
        return False, "DataFrame contains invalid price relationships (high < close or close < low or high < low)"
    
    return True, "DataFrame validation passed"

def cleanup_checkpoints(models_dir, prefix="checkpoint_", keep=5, keep_best=True):
    """
    Cleanup old checkpoints, keeping only the latest n and optionally the best checkpoint.
    
    Args:
        models_dir (str): Directory containing model checkpoints.
        prefix (str, optional): Prefix for checkpoint files to clean up (e.g., "checkpoint_", "periodic_").
        keep (int, optional): Number of latest checkpoints to keep.
        keep_best (bool, optional): Whether to always keep the best checkpoint.
    """
    if not os.path.exists(models_dir):
        log(f"Directory does not exist: {models_dir}")
        return

    # Find all checkpoints matching the prefix
    checkpoints = [f for f in os.listdir(models_dir) if f.startswith(prefix) and f.endswith(".pth")]
    
    if len(checkpoints) <= keep:
        return  # No cleanup needed
    
    # Always keep best_checkpoint.pth if it exists and keep_best is True
    best_checkpoint = "best_checkpoint.pth"
    best_checkpoint_path = os.path.join(models_dir, best_checkpoint)
    best_exists = os.path.exists(best_checkpoint_path) and keep_best
    
    # Sort checkpoints by episode number
    sorted_checkpoints = sorted(checkpoints, 
                               key=lambda x: int(x.split('_')[1].split('.pth')[0]), 
                               reverse=True)
    
    # Keep the latest 'keep' checkpoints
    to_keep = sorted_checkpoints[:keep]
    to_delete = sorted_checkpoints[keep:]
    
    # If best checkpoint exists and should be kept, don't delete it
    if best_exists and best_checkpoint in to_delete:
        to_delete.remove(best_checkpoint)
    
    # Delete old checkpoints
    for checkpoint in to_delete:
        checkpoint_path = os.path.join(models_dir, checkpoint)
        try:
            os.remove(checkpoint_path)
            log(f"Deleted old checkpoint: {checkpoint}")
        except Exception as e:
            log(f"Failed to delete checkpoint {checkpoint}: {e}")

def list_available_checkpoints(models_dir, include_metadata=True):
    """
    List all available checkpoints in the given directory with metadata.
    
    Args:
        models_dir (str): Directory containing model checkpoints
        include_metadata (bool): Whether to include metadata about each checkpoint
        
    Returns:
        list: List of dictionaries with checkpoint information
    """
    if not os.path.exists(models_dir):
        log(f"Directory does not exist: {models_dir}")
        return []
    
    # Find all checkpoint files
    checkpoint_files = [f for f in os.listdir(models_dir) if f.endswith(".pth")]
    
    # Store checkpoint info
    checkpoints = []
    
    for file in checkpoint_files:
        checkpoint_path = os.path.join(models_dir, file)
        checkpoint_info = {
            "filename": file,
            "path": checkpoint_path,
            "size_mb": os.path.getsize(checkpoint_path) / (1024 * 1024),
            "modified": datetime.datetime.fromtimestamp(os.path.getmtime(checkpoint_path)).strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Categorize the checkpoint
        if file.startswith("checkpoint_"):
            checkpoint_info["type"] = "regular"
            checkpoint_info["episode"] = int(file.split('_')[1].split('.pth')[0])
        elif file.startswith("emergency_"):
            checkpoint_info["type"] = "emergency"
            checkpoint_info["episode"] = int(file.split('_')[2].split('.pth')[0])
        elif file.startswith("periodic_"):
            checkpoint_info["type"] = "periodic"
            checkpoint_info["episode"] = int(file.split('_')[1].split('.pth')[0])
        elif file == "best_checkpoint.pth":
            checkpoint_info["type"] = "best"
        elif file == "last_interrupted.pth":
            checkpoint_info["type"] = "interrupted"
        elif file.startswith("final_"):
            checkpoint_info["type"] = "final"
        else:
            checkpoint_info["type"] = "unknown"
        
        # Extract metadata if requested
        if include_metadata:
            try:
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                checkpoint_info["metadata"] = {
                    "episode": checkpoint.get("episode", 0),
                    "best_reward": checkpoint.get("best_reward", None),
                    "timestamp": checkpoint.get("timestamp", None),
                    "is_emergency": checkpoint.get("is_emergency", False)
                }
            except Exception as e:
                checkpoint_info["metadata_error"] = str(e)
        
        checkpoints.append(checkpoint_info)
    
    # Sort checkpoints by episode if applicable
    checkpoints.sort(key=lambda x: (
        0 if x["type"] == "best" else
        1 if x["type"] == "final" else
        2 if x["type"] == "interrupted" else
        3,
        -x.get("episode", 0) if "episode" in x else 0
    ))
    
    return checkpoints

def format_metrics(metrics):
    """
    Format metrics dictionary for display.
    
    Args:
        metrics (dict): Dictionary of trading metrics.
        
    Returns:
        str: Formatted metrics string for display.
    """
    formatted = []
    for k, v in sorted(metrics.items()):
        if k in ['win_rate', 'max_drawdown']:
            formatted.append(f"{k}: {v*100:.2f}%")
        elif k in ['net_profit', 'avg_win', 'avg_loss', 'value_at_risk', 'conditional_var']:
            formatted.append(f"{k}: ${v:.2f}")
        elif k in ['within_target']:
            formatted.append(f"{k}: {'Yes' if v else 'No'}")
        elif k in ['size_limits_applied']:
            formatted.append(f"{k}: {int(v)}")
        else:
            formatted.append(f"{k}: {v:.2f}")
    
    # Group into 3 columns
    result = []
    for i in range(0, len(formatted), 3):
        row = formatted[i:i+3]
        result.append("   ".join(row))
    
    return "\n".join(result)

def count_trainable_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model.
    
    Returns:
        int: Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_model_from_checkpoint(checkpoint_path, model, optimizer=None, device="cpu"):
    """
    Load model weights from checkpoint.
    
    Args:
        checkpoint_path (str): Path to checkpoint file.
        model: PyTorch model to load weights into.
        optimizer (optional): Optimizer to load state into.
        device (str): Device to load model to.
        
    Returns:
        tuple: (episode, best_reward) - Episode number and best reward from checkpoint.
    """
    if not os.path.exists(checkpoint_path):
        log(f"[WARNING] Checkpoint not found: {checkpoint_path}")
        return 0, float('-inf')
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint["model_state"])
        
        # Load optimizer state if provided
        if optimizer is not None and "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        
        # Return episode number and best reward
        episode = checkpoint.get("episode", 0)
        best_reward = checkpoint.get("best_reward", float('-inf'))
        
        log(f"Loaded checkpoint from episode {episode} with best reward {best_reward:.2f}")
        return episode, best_reward
    
    except Exception as e:
        log(f"[ERROR] Failed to load checkpoint: {e}")
        return 0, float('-inf')

def save_checkpoint(model, optimizer, episode, best_reward, config, path, timestamp=None, is_emergency=False, 
                   performance_history=None, recent_rewards=None, horizons=None, additional_state=None):
    """
    Save model checkpoint with enhanced metadata for robust resumption.
    
    Args:
        model: PyTorch model to save.
        optimizer: Optimizer to save state.
        episode (int): Current episode number.
        best_reward (float): Best reward achieved.
        config (dict): Configuration parameters.
        path (str): Path to save checkpoint to.
        timestamp (str, optional): Timestamp for the checkpoint.
        is_emergency (bool, optional): Whether this is an emergency checkpoint.
        performance_history (list, optional): History of performance metrics.
        recent_rewards (list/deque, optional): Recent reward values.
        horizons (list, optional): Prediction horizons used in the model.
        additional_state (dict, optional): Any additional state to save.
        
    Returns:
        bool: True if checkpoint was saved successfully, False otherwise.
    """
    try:
        if timestamp is None:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            
        checkpoint = {
            "episode": episode,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_reward": best_reward,
            "config": config,
            "timestamp": timestamp,
            "is_emergency": is_emergency,
            "version": 3  # Increment version for enhanced checkpoints
        }
        
        # Add training state for resumption if provided
        if performance_history is not None:
            # Convert to list if it's another type
            checkpoint["performance_history"] = list(performance_history)
        
        if recent_rewards is not None:
            # Convert deque to list for serialization
            if isinstance(recent_rewards, deque):
                checkpoint["recent_rewards"] = list(recent_rewards)
            else:
                checkpoint["recent_rewards"] = recent_rewards
        
        if horizons is not None:
            checkpoint["horizons"] = horizons
        
        # Include any additional state
        if additional_state and isinstance(additional_state, dict):
            for key, value in additional_state.items():
                checkpoint[key] = value
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save(checkpoint, path)
        log(f"Saved {'enhanced ' if not is_emergency else 'emergency '}checkpoint to {path}")
        
        # Log detailed info about saved state
        details = []
        if performance_history is not None:
            details.append(f"performance history ({len(performance_history)} entries)")
        if recent_rewards is not None:
            details.append("recent rewards")
        if horizons is not None:
            details.append("prediction horizons")
        if additional_state:
            details.append(f"additional state ({len(additional_state)} items)")
            
        if details:
            log(f"Checkpoint includes: {', '.join(details)}")
            
        return True
    
    except Exception as e:
        log(f"[ERROR] Failed to save checkpoint: {e}")
        import traceback
        log(traceback.format_exc())
        return False

def write_metrics_history(metrics_history, filename="metrics_history.json"):
    """
    Write metrics history to JSON file.
    
    Args:
        metrics_history (list): List of metrics dictionaries.
        filename (str): Path to output file.
        
    Returns:
        bool: True if file was written successfully, False otherwise.
    """
    try:
        # Convert numpy types to Python native types for JSON serialization
        serializable_history = []
        for metrics in metrics_history:
            serializable_metrics = {}
            for k, v in metrics.items():
                if isinstance(v, (np.int32, np.int64)):
                    serializable_metrics[k] = int(v)
                elif isinstance(v, (np.float32, np.float64)):
                    serializable_metrics[k] = float(v)
                else:
                    serializable_metrics[k] = v
            serializable_history.append(serializable_metrics)
            
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_history, f, indent=2)
        
        log(f"Wrote metrics history to {filename}")
        return True
    
    except Exception as e:
        log(f"[ERROR] Failed to write metrics history: {e}")
        return False

def visualize_metrics(metrics_history, output_file=None):
    """
    Create visualization of key metrics over time.
    
    Enhanced with risk metrics visualization.
    
    Args:
        metrics_history (list): List of metrics dictionaries.
        output_file (str, optional): Path to save visualization to.
        
    Returns:
        bool: True if visualization was created successfully, False otherwise.
    """
    try:
        if not metrics_history:
            log("[WARNING] No metrics history to visualize")
            return False
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        
        # Extract key metrics
        episodes = list(range(len(metrics_history)))
        net_profits = [m.get("net_profit", 0) for m in metrics_history]
        win_rates = [m.get("win_rate", 0) * 100 for m in metrics_history]
        sharpes = [m.get("sharpe", 0) for m in metrics_history]
        drawdowns = [m.get("max_drawdown", 0) * 100 for m in metrics_history]
        
        # Extract risk metrics (if available)
        risk_scores = [m.get("overall_risk_score", 0) * 100 for m in metrics_history]
        
        # Extract position size limitation metrics (if available)
        size_limitations = [m.get("avg_size_limitation", 0) * 100 for m in metrics_history]
        
        # Plot net profit
        axes[0, 0].plot(episodes, net_profits, 'b-')
        axes[0, 0].set_title('Net Profit ($)')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Profit')
        axes[0, 0].grid(True)
        
        # Plot win rate
        axes[0, 1].plot(episodes, win_rates, 'g-')
        axes[0, 1].set_title('Win Rate (%)')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Win Rate')
        axes[0, 1].grid(True)
        
        # Plot Sharpe ratio
        axes[1, 0].plot(episodes, sharpes, 'r-')
        axes[1, 0].set_title('Sharpe Ratio')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Sharpe')
        axes[1, 0].grid(True)
        
        # Plot max drawdown
        axes[1, 1].plot(episodes, drawdowns, 'm-')
        axes[1, 1].set_title('Max Drawdown (%)')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Drawdown')
        axes[1, 1].grid(True)
        
        # Plot risk score
        axes[2, 0].plot(episodes, risk_scores, 'c-')
        axes[2, 0].set_title('Overall Risk Score (%)')
        axes[2, 0].set_xlabel('Episode')
        axes[2, 0].set_ylabel('Risk Score')
        axes[2, 0].grid(True)
        
        # Plot position size limitations
        axes[2, 1].plot(episodes, size_limitations, 'y-')
        axes[2, 1].set_title('Position Size Limitations (%)')
        axes[2, 1].set_xlabel('Episode')
        axes[2, 1].set_ylabel('Size Limitation')
        axes[2, 1].grid(True)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file)
            log(f"Saved metrics visualization to {output_file}")
        
        plt.close()
        return True
    
    except Exception as e:
        log(f"[ERROR] Failed to create metrics visualization: {e}")
        return False

def visualize_risk_metrics(risk_metrics_history, output_file=None):
    """
    Create detailed visualization of portfolio risk metrics.
    
    Args:
        risk_metrics_history (list): List of risk metrics dictionaries.
        output_file (str, optional): Path to save visualization to.
        
    Returns:
        bool: True if visualization was created successfully, False otherwise.
    """
    try:
        if not risk_metrics_history:
            log("[WARNING] No risk metrics history to visualize")
            return False
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        
        # Extract time steps
        time_steps = list(range(len(risk_metrics_history)))
        
        # Extract risk metrics
        exposure_pcts = [m.get("exposure_percentage", 0) * 100 for m in risk_metrics_history]
        concentration = [m.get("risk_concentration", 0) for m in risk_metrics_history]
        drawdowns = [m.get("drawdown", 0) * 100 for m in risk_metrics_history]
        var_pcts = [m.get("var_pct", 0) * 100 for m in risk_metrics_history]
        correlation_risks = [m.get("correlation_risk", 0) for m in risk_metrics_history]
        liquidity_risks = [m.get("liquidity_risk", 0) for m in risk_metrics_history]
        
        # Plot exposure percentage
        axes[0, 0].plot(time_steps, exposure_pcts, 'b-')
        axes[0, 0].set_title('Exposure Percentage (%)')
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('Exposure %')
        axes[0, 0].grid(True)
        
        # Plot risk concentration
        axes[0, 1].plot(time_steps, concentration, 'g-')
        axes[0, 1].set_title('Risk Concentration')
        axes[0, 1].set_xlabel('Time Step')
        axes[0, 1].set_ylabel('Concentration')
        axes[0, 1].grid(True)
        
        # Plot drawdown
        axes[1, 0].plot(time_steps, drawdowns, 'r-')
        axes[1, 0].set_title('Portfolio Drawdown (%)')
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('Drawdown %')
        axes[1, 0].grid(True)
        
        # Plot Value at Risk
        axes[1, 1].plot(time_steps, var_pcts, 'm-')
        axes[1, 1].set_title('Value at Risk (% of Portfolio)')
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('VaR %')
        axes[1, 1].grid(True)
        
        # Plot correlation risk
        axes[2, 0].plot(time_steps, correlation_risks, 'c-')
        axes[2, 0].set_title('Correlation Risk')
        axes[2, 0].set_xlabel('Time Step')
        axes[2, 0].set_ylabel('Risk Level')
        axes[2, 0].grid(True)
        
        # Plot liquidity risk
        axes[2, 1].plot(time_steps, liquidity_risks, 'y-')
        axes[2, 1].set_title('Liquidity Risk')
        axes[2, 1].set_xlabel('Time Step')
        axes[2, 1].set_ylabel('Risk Level')
        axes[2, 1].grid(True)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file)
            log(f"Saved risk metrics visualization to {output_file}")
        
        plt.close()
        return True
    
    except Exception as e:
        log(f"[ERROR] Failed to create risk metrics visualization: {e}")
        return False

def visualize_position_size_limits(trades_history, size_limits_history, output_file=None):
    """
    Visualize the impact of position size limits on trades.
    
    Args:
        trades_history (list): List of trade details including actual and desired sizes.
        size_limits_history (list): List of position size limit values applied.
        output_file (str, optional): Path to save visualization to.
        
    Returns:
        bool: True if visualization was created successfully, False otherwise.
    """
    try:
        if not trades_history or not size_limits_history:
            log("[WARNING] No trade or size limit history to visualize")
            return False
        
        # Extract data
        trade_indices = list(range(len(trades_history)))
        actual_sizes = [t.get("actual_size", 0) for t in trades_history]
        desired_sizes = [t.get("desired_size", actual_sizes[i]) for i, t in enumerate(trades_history)]
        
        # Calculate size reductions
        size_reductions = [max(0, desired - actual) for desired, actual in zip(desired_sizes, actual_sizes)]
        reduction_pcts = [100 * reduction / max(0.0001, desired) for reduction, desired in zip(size_reductions, desired_sizes)]
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot actual vs desired sizes
        axes[0].bar(trade_indices, desired_sizes, alpha=0.5, color='blue', label='Desired Size')
        axes[0].bar(trade_indices, actual_sizes, alpha=0.7, color='green', label='Actual Size')
        axes[0].set_title('Trade Sizes: Actual vs Desired')
        axes[0].set_xlabel('Trade Index')
        axes[0].set_ylabel('Size')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot reduction percentages
        axes[1].bar(trade_indices, reduction_pcts, color='red', alpha=0.7)
        axes[1].set_title('Size Reduction Percentage')
        axes[1].set_xlabel('Trade Index')
        axes[1].set_ylabel('Reduction %')
        axes[1].grid(True)
        
        # Highlight trades with significant reductions
        for i, pct in enumerate(reduction_pcts):
            if pct > 20:  # More than 20% reduction
                axes[1].annotate(f"{pct:.1f}%", 
                              xy=(i, pct),
                              xytext=(0, 10),
                              textcoords="offset points",
                              ha='center',
                              fontsize=8,
                              arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file)
            log(f"Saved position size limits visualization to {output_file}")
        
        plt.close()
        return True
    
    except Exception as e:
        log(f"[ERROR] Failed to create position size limits visualization: {e}")
        return False

def visualize_risk_heatmap(risk_components_history, output_file=None):
    """
    Create a heatmap visualization of risk components over time.
    
    Args:
        risk_components_history (list): List of risk component dictionaries.
        output_file (str, optional): Path to save visualization to.
        
    Returns:
        bool: True if visualization was created successfully, False otherwise.
    """
    try:
        if not risk_components_history:
            log("[WARNING] No risk components history to visualize")
            return False
        
        # Define risk components to extract
        risk_components = [
            'exposure_percentage',
            'risk_concentration',
            'drawdown',
            'correlation_risk',
            'liquidity_risk',
            'var_pct'
        ]
        
        # Extract time steps
        time_steps = list(range(len(risk_components_history)))
        
        # Extract risk components data
        risk_data = []
        for component in risk_components:
            values = [m.get(component, 0) for m in risk_components_history]
            risk_data.append(values)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Create custom colormap from green to red
        cmap = LinearSegmentedColormap.from_list('GrRd', ['green', 'yellow', 'red'])
        
        # Create heatmap
        sns.heatmap(risk_data, cmap=cmap, annot=False, 
                   yticklabels=risk_components, 
                   xticklabels=[t if t % 10 == 0 else "" for t in time_steps],
                   vmin=0, vmax=1)
        
        plt.title('Risk Components Heatmap Over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Risk Component')
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file)
            log(f"Saved risk heatmap visualization to {output_file}")
        
        plt.close()
        return True
    
    except Exception as e:
        log(f"[ERROR] Failed to create risk heatmap visualization: {e}")
        return False

def generate_dynamic_horizons(min_horizon=1, max_horizon=576, num_horizons=6, mode="mixed", seed=None):
    """
    Generate a set of prediction horizons based on specified parameters and mode.
    
    Args:
        min_horizon (int): Minimum prediction horizon (in bars)
        max_horizon (int): Maximum prediction horizon (in bars)
        num_horizons (int): Number of horizons to generate
        mode (str): Distribution mode for horizon generation:
            - "log": Logarithmic distribution (emphasis on shorter horizons)
            - "exp": Exponential distribution (emphasis on longer horizons)
            - "uniform": Uniform distribution across range
            - "mixed": Balanced mix of short and long horizons
        seed (int, optional): Random seed for reproducibility
    
    Returns:
        list: Generated prediction horizons
    """
    import numpy as np
    import random
    
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    horizons = []
    
    # Generate horizons based on mode
    if mode == "log":
        # Logarithmic distribution - emphasize shorter horizons
        # Take more samples from logarithmic space and filter
        log_min = np.log(max(1, min_horizon))
        log_max = np.log(max_horizon)
        
        # Generate more samples than needed and then select subset
        candidate_horizons = np.exp(np.linspace(log_min, log_max, num=num_horizons * 2)).astype(int)
        
        # Remove duplicates and ensure we have distinct values
        candidate_horizons = sorted(list(set(candidate_horizons)))
        
        # Select from candidates with emphasis on shorter horizons
        weights = [1.0 / (i + 1) for i in range(len(candidate_horizons))]
        cumulative_weights = np.cumsum(weights)
        cumulative_weights = cumulative_weights / cumulative_weights[-1]  # Normalize
        
        # Select horizons
        horizons = []
        for _ in range(min(num_horizons, len(candidate_horizons))):
            if not candidate_horizons:
                break
                
            # Sample based on weights
            r = random.random()
            idx = np.searchsorted(cumulative_weights, r)
            idx = min(idx, len(candidate_horizons) - 1)  # Safety check
            
            horizon = candidate_horizons.pop(idx)
            weights.pop(idx)
            
            if len(weights) > 0:
                # Update weights after removing an item
                cumulative_weights = np.cumsum(weights)
                if cumulative_weights[-1] > 0:  # Prevent division by zero
                    cumulative_weights = cumulative_weights / cumulative_weights[-1]
            
            horizons.append(horizon)
    
    elif mode == "exp":
        # Exponential distribution - emphasize longer horizons
        exp_min = min_horizon
        exp_max = max_horizon
        
        # Generate exponentially spaced horizons
        candidate_horizons = []
        for i in range(num_horizons * 2):
            # More emphasis on longer horizons
            p = (i + 1) / (num_horizons * 2 + 1)
            p = p ** 1.5  # Exponential emphasis
            horizon = int(exp_min + p * (exp_max - exp_min))
            candidate_horizons.append(horizon)
        
        # Remove duplicates
        candidate_horizons = sorted(list(set(candidate_horizons)))
        
        # Select with emphasis on longer horizons
        weights = [(i + 1) for i in range(len(candidate_horizons))]
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Cumulative weights for sampling
        cumulative_weights = np.cumsum(weights)
        
        # Select horizons
        horizons = []
        for _ in range(min(num_horizons, len(candidate_horizons))):
            if not candidate_horizons:
                break
                
            # Sample based on weights
            r = random.random()
            idx = np.searchsorted(cumulative_weights, r)
            idx = min(idx, len(candidate_horizons) - 1)  # Safety check
            
            horizon = candidate_horizons.pop(idx)
            weights.pop(idx)
            
            if len(weights) > 0:
                # Update weights after removing an item
                cumulative_weights = np.cumsum(weights)
                if cumulative_weights[-1] > 0:  # Prevent division by zero
                    cumulative_weights = cumulative_weights / cumulative_weights[-1]
            
            horizons.append(horizon)
    
    elif mode == "uniform":
        # Uniform distribution across range
        # Simple linear distribution
        horizons = np.linspace(min_horizon, max_horizon, num=num_horizons).astype(int)
        
        # Remove duplicates
        horizons = sorted(list(set(horizons)))
        
        # If we lost some horizons due to integer conversion duplicates, add more
        while len(horizons) < num_horizons:
            # Add horizons in gaps
            gaps = [(horizons[i+1] - horizons[i], i) for i in range(len(horizons)-1)]
            if not gaps:
                break
                
            # Find largest gap
            largest_gap, gap_idx = max(gaps)
            if largest_gap <= 1:
                break  # No more gaps to fill
                
            # Add horizon in middle of largest gap
            new_horizon = horizons[gap_idx] + largest_gap // 2
            horizons.append(new_horizon)
            horizons.sort()
    
    else:  # mode == "mixed" or fallback
        # Mixed approach - balanced between short and long horizons
        # Divide into segments
        num_segments = min(num_horizons, 3)  # Short, medium, long segments
        segment_size = num_horizons // num_segments
        remaining = num_horizons % num_segments
        
        # Generate horizons for each segment
        horizons = []
        
        # Short horizons (logarithmic distribution)
        short_max = min_horizon + (max_horizon - min_horizon) // 3
        short_count = segment_size + (1 if remaining > 0 else 0)
        
        log_min = np.log(max(1, min_horizon))
        log_max = np.log(short_max)
        short_horizons = np.exp(np.linspace(log_min, log_max, num=short_count)).astype(int)
        horizons.extend(short_horizons)
        
        # Medium horizons (uniform distribution)
        medium_min = short_max + 1
        medium_max = medium_min + (max_horizon - medium_min) // 2
        medium_count = segment_size + (1 if remaining > 1 else 0)
        
        if medium_count > 0:
            medium_horizons = np.linspace(medium_min, medium_max, num=medium_count).astype(int)
            horizons.extend(medium_horizons)
        
        # Long horizons (exponential distribution)
        long_min = medium_max + 1
        long_count = num_horizons - len(horizons)
        
        if long_count > 0 and long_min < max_horizon:
            # More emphasis on longer horizons
            long_horizons = []
            for i in range(long_count):
                p = (i + 1) / (long_count + 1)
                p = p ** 1.5  # Exponential emphasis
                horizon = int(long_min + p * (max_horizon - long_min))
                long_horizons.append(horizon)
            
            horizons.extend(long_horizons)
    
    # Ensure we have the right number of horizons
    horizons = sorted(list(set(horizons)))  # Remove any duplicates
    
    # If we have too many, remove some
    if len(horizons) > num_horizons:
        # Prioritize keeping endpoints and evenly distributed points
        if num_horizons >= 2:
            # Always keep min and max
            middle_horizons = horizons[1:-1]
            indices = np.linspace(0, len(middle_horizons) - 1, num_horizons - 2).astype(int)
            selected_horizons = [horizons[0]] + [middle_horizons[i] for i in indices] + [horizons[-1]]
            horizons = selected_horizons
        else:
            # Just keep first n horizons
            horizons = horizons[:num_horizons]
    
    # If we have too few (due to duplicates), add more
    while len(horizons) < num_horizons:
        # Find largest gap
        gaps = [(horizons[i+1] - horizons[i], i) for i in range(len(horizons)-1)]
        if not gaps:
            # Add one more at the end if possible
            if horizons[-1] < max_horizon:
                horizons.append(min(max_horizon, horizons[-1] + 1))
            break
            
        # Find largest gap
        largest_gap, gap_idx = max(gaps)
        if largest_gap <= 1:
            break  # No more gaps to fill
            
        # Add horizon in middle of largest gap
        new_horizon = horizons[gap_idx] + largest_gap // 2
        horizons.append(new_horizon)
        horizons.sort()
    
    # Convert to integers and sort
    horizons = sorted([int(h) for h in horizons])
    
    return horizons

# Dynamic horizon adaptation for active models
def adapt_prediction_horizons(model, market_data, config):
    """
    Adapt prediction horizons based on market conditions and model performance.
    
    Args:
        model: The model containing horizon performance information
        market_data: DataFrame containing recent market data
        config: Configuration dictionary with adaptation parameters
        
    Returns:
        list: Updated prediction horizons
    """
    import numpy as np
    
    # Get current horizons
    current_horizons = model.horizons if hasattr(model, 'horizons') else []
    
    # Calculate market volatility as a key metric for horizon adaptation
    if len(market_data) > 2:
        # Calculate normalized price volatility
        close_prices = market_data['close'].values
        returns = np.diff(close_prices) / close_prices[:-1]
        volatility = np.std(returns) * np.sqrt(len(returns))
    else:
        # Default volatility if insufficient data
        volatility = 1.0
    
    # Get horizon performance if available
    horizon_performance = {}
    if hasattr(model, 'horizon_performance'):
        horizon_performance = model.horizon_performance
    
    # Calculate optimal time horizons based on volatility regimes
    min_horizon = max(1, config.get("MIN_HORIZON", 1))
    max_horizon = config.get("MAX_HORIZON", 576)
    
    # Adjust horizon distribution based on volatility
    if volatility > 2.0:  # High volatility regime
        # Focus more on shorter timeframes
        mode = "log"
        weight_short = 0.7
    elif volatility < 0.5:  # Low volatility regime
        # Focus more on longer timeframes
        mode = "exp"
        weight_short = 0.3
    else:  # Medium volatility regime
        # Balanced approach
        mode = "mixed"
        weight_short = 0.5
    
    # Performance-based adjustment
    # If we have performance data, increase weight of well-performing horizons
    if horizon_performance:
        # Convert horizon names to integers for those that start with 'h'
        horizons_as_ints = {}
        for h_name, perf in horizon_performance.items():
            if h_name.startswith('h'):
                try:
                    h_value = int(h_name[1:])  # Extract number after 'h'
                    horizons_as_ints[h_value] = perf
                except ValueError:
                    # Skip if not a valid integer
                    continue
            
        # Sort horizons by performance
        sorted_horizons = sorted(horizons_as_ints.items(), key=lambda x: x[1], reverse=True)
        top_horizons = [h for h, _ in sorted_horizons[:max(1, len(sorted_horizons)//3)]]
        
        # Ensure top performing horizons are included
        # We'll use them to seed our new generation
        seed_horizons = top_horizons[:min(3, len(top_horizons))]
    else:
        seed_horizons = []
    
    # Generate new horizons but maintain some previous ones for stability
    num_horizons = len(current_horizons) if current_horizons else 6
    num_new = max(1, int(num_horizons * 0.3))  # 30% new horizons
    
    # Keep some old horizons for stability (prioritize best performing ones)
    keep_horizons = seed_horizons.copy()
    for h in current_horizons:
        if h not in keep_horizons:
            keep_horizons.append(h)
            if len(keep_horizons) >= (num_horizons - num_new):
                break
    
    # Generate new horizons to complement existing ones
    new_horizons = generate_dynamic_horizons(
        min_horizon=min_horizon,
        max_horizon=max_horizon,
        num_horizons=num_new,
        mode=mode,
        seed=None  # Allow randomness for exploration
    )
    
    # Combine keep_horizons and new_horizons
    updated_horizons = keep_horizons + [h for h in new_horizons if h not in keep_horizons]
    
    # Ensure we don't exceed the desired number of horizons
    if len(updated_horizons) > num_horizons:
        updated_horizons = updated_horizons[:num_horizons]
    
    # Ensure all horizons are integers before sorting
    updated_horizons = [int(h) for h in updated_horizons]
    
    # Sort horizons
    updated_horizons.sort()
    
    return updated_horizons

if __name__ == "__main__":
    # Simple tests if run directly
    print("Testing GPU functions...")
    print(f"GPU Usage: {measure_gpu_usage()}")
    print(f"Optimal GPU targets: {get_optimal_gpu_targets()}")
    print(f"Multi-GPU check: {check_multi_gpu()}")
    
    print("\nTesting metrics formatting...")
    test_metrics = {
        "net_profit": 1234.56,
        "total_trades": 42,
        "win_rate": 0.65,
        "profit_factor": 2.1,
        "sharpe": 1.75,
        "max_drawdown": 0.12,
        "overall_risk_score": 0.35,
        "within_target": True
    }
    print(format_metrics(test_metrics))
