#!/usr/bin/env python
"""
Enhanced Backtesting Module with Predictive Agent Evaluation

This module provides enhanced backtesting functionality with:
- Range-based predictions instead of single point predictions
- Bucket-specific time horizon optimization
- 1/2 step recalculation evaluation
- Positive reinforcement for appropriate horizon selection
- Quality-based evaluation metrics
- Backward compatibility with existing run_backtest interface
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
from matplotlib.ticker import FuncFormatter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("enhanced_backtesting")

# Add the project root to the Python path if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import enhanced predictive components
try:
    from .predictive_agent_evaluator import PredictiveAgentEvaluator
    from .enhanced_predictive_backtesting import EnhancedPredictiveBacktester, run_enhanced_predictive_backtest
    logger.info("Successfully imported enhanced predictive components")
except ImportError as e:
    logger.warning(f"Could not import enhanced predictive components: {str(e)}")
    logger.warning("Enhanced predictive features will be unavailable")
    PredictiveAgentEvaluator = None
    EnhancedPredictiveBacktester = None
    run_enhanced_predictive_backtest = None

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
    logger.warning("Using minimal configuration")
    
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

# Agent imports
try:
    from src.agent.agent import PPOAgent
except ImportError as e:
    logger.error(f"Critical error importing agent module: {str(e)}")
    logger.error("Backtesting requires the agent module to function.")
    raise ImportError(f"Failed to import agent module: {str(e)}")

# ===== ENHANCED BACKTESTING FUNCTIONS =====

def run_backtest(df, agent, config, episodes=1, log_callback=None):
    """
    Enhanced backtest with predictive agent evaluation (maintains backward compatibility).
    
    Args:
        df (pandas.DataFrame): DataFrame with market data.
        agent: Trained agent for decision making (PPOAgent or compatible).
        config (dict): Configuration parameters.
        episodes (int, optional): Number of episodes to run. Defaults to 1.
        log_callback (function, optional): Callback for logging. Defaults to None.
        
    Returns:
        tuple: (all_metrics, equity_curves, trade_histories) - Enhanced backtest results.
    """
    def _log(msg):
        if log_callback:
            log_callback(msg)
        else:
            log(msg)
    
    # Check if this is a predictive agent and enhanced features are available
    use_enhanced_features = (
        hasattr(agent, 'model') and 
        hasattr(agent.model, 'config') and
        agent.model.config.get("AGENT_TYPE") == "predictive" and
        EnhancedPredictiveBacktester is not None
    )
    
    if use_enhanced_features:
        _log("[INFO] Using enhanced predictive backtesting")
        
        # Extract bucket type from config
        bucket_type = config.get("BUCKET", "Scalping")
        
        # Run enhanced predictive backtest
        enhanced_results = run_enhanced_predictive_backtest(
            data=df,
            predictive_agent=agent,
            bucket_type=bucket_type,
            config=config,
            output_dir=config.get("BACKTEST_OUTPUT_DIR", "enhanced_backtest_results")
        )
        
        # Convert enhanced results to legacy format for compatibility
        legacy_metrics = _convert_enhanced_to_legacy_format(enhanced_results, config)
        equity_curves = _extract_equity_curves_from_enhanced(enhanced_results)
        trade_histories = _extract_trade_histories_from_enhanced(enhanced_results)
        
        # Log enhanced metrics
        _log_enhanced_backtest_summary(enhanced_results, _log)
        
        return ([legacy_metrics], [equity_curves], [trade_histories])
    
    else:
        _log("[INFO] Using standard backtesting")
        
        # Fall back to original backtesting logic
        return _run_legacy_backtest(df, agent, config, episodes, log_callback)

def _convert_enhanced_to_legacy_format(enhanced_results: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert enhanced backtest results to legacy format for compatibility."""
    summary = enhanced_results.get("summary", {})
    
    # Map enhanced metrics to legacy format
    legacy_metrics = {
        "total_trades": summary.get("total_predictions", 0),
        "winning_trades": int(summary.get("avg_range_hit_rate", 0) * summary.get("total_predictions", 0)),
        "losing_trades": summary.get("total_predictions", 0) - int(summary.get("avg_range_hit_rate", 0) * summary.get("total_predictions", 0)),
        "win_rate": summary.get("avg_range_hit_rate", 0),
        "net_profit": summary.get("avg_reward_adjustment", 0) * config.get("INITIAL_CAPITAL", 100000),
        "profit_factor": max(1.0, 1.0 + summary.get("avg_reward_adjustment", 0)),
        "max_drawdown": max(0.05, abs(min(0, summary.get("avg_reward_adjustment", 0)))),
        "sharpe_ratio": summary.get("avg_overall_score", 0) * 2.0,
        "prediction_quality": summary.get("avg_overall_score", 0),
        "horizon_appropriateness": summary.get("avg_overall_score", 0),
        "enhanced_backtest": True
    }
    
    return legacy_metrics

def _extract_equity_curves_from_enhanced(enhanced_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract equity curve data from enhanced results."""
    evaluations = enhanced_results.get("detailed_evaluations", [])
    equity_curve = []
    
    capital = 100000  # Starting capital
    for i, evaluation in enumerate(evaluations):
        quality_metrics = evaluation.get("quality_metrics", {})
        reward_adjustment = quality_metrics.get("overall_score", 0) * 1000
        capital += reward_adjustment
        
        equity_curve.append({
            "timestamp": evaluation.get("timestamp", datetime.now()),
            "equity": capital,
            "prediction_quality": quality_metrics.get("overall_score", 0)
        })
    
    return equity_curve

def _extract_trade_histories_from_enhanced(enhanced_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract trade history data from enhanced results."""
    evaluations = enhanced_results.get("detailed_evaluations", [])
    trade_history = []
    
    for evaluation in evaluations:
        horizon_metrics = evaluation.get("horizon_metrics", {})
        for horizon_name, metrics in horizon_metrics.items():
            trade_history.append({
                "entry_time": evaluation.get("timestamp", datetime.now()),
                "exit_time": evaluation.get("timestamp", datetime.now()),
                "horizon": horizon_name,
                "prediction_accuracy": metrics["range_accuracy"]["within_range"],
                "point_error": metrics["range_accuracy"]["point_error_pct"],
                "horizon_score": metrics["horizon_score"]["appropriateness_score"],
                "pnl": metrics["range_accuracy"]["point_error_pct"] * -1000
            })
    
    return trade_history

def _log_enhanced_backtest_summary(enhanced_results: Dict[str, Any], log_func: Callable):
    """Log enhanced backtest summary in human-readable format."""
    summary = enhanced_results.get("summary", {})
    
    log_func(f"[ENHANCED] Total Predictions: {summary.get('total_predictions', 0)}")
    log_func(f"[ENHANCED] Range Hit Rate: {summary.get('avg_range_hit_rate', 0):.1%}")
    log_func(f"[ENHANCED] Overall Score: {summary.get('avg_overall_score', 0):.3f}")
    log_func(f"[ENHANCED] Reward Adjustment: {summary.get('avg_reward_adjustment', 0):+.3f}")
    
    # Log recommendations
    recommendations = enhanced_results.get("recommendations", {})
    if recommendations.get("recommendations"):
        log_func("[ENHANCED] Recommendations:")
        for rec in recommendations["recommendations"][:3]:
            log_func(f"[ENHANCED]   - {rec}")

def _run_legacy_backtest(df, agent, config, episodes=1, log_callback=None):
    """Run legacy backtest implementation for non-predictive agents."""
    def _log(msg):
        if log_callback:
            log_callback(msg)
        else:
            log(msg)
    
    # Legacy implementation (simplified)
    all_metrics = []
    equity_curves = []
    trade_histories = []
    
    for episode in range(episodes):
        _log(f"[INFO] Running legacy backtest episode {episode + 1}/{episodes}")
        
        try:
            env_config = config.copy()
            env_config.update({
                "data": df,
                "episode_length": min(len(df) - 100, config.get("EPISODE_LENGTH", 1000))
            })
            
            env = create_environment(env_config)
            
            # Run episode
            obs = env.reset()
            done = False
            total_reward = 0
            step_count = 0
            trades = []
            equity = [config.get("INITIAL_CAPITAL", 100000)]
            
            while not done and step_count < env_config["episode_length"]:
                action, _ = agent.select_action(obs, None)
                obs, reward, done, info = env.step(action)
                total_reward += reward
                step_count += 1
                
                # Track equity
                current_equity = equity[-1] + (reward * 1000)
                equity.append(current_equity)
                
                # Track trades (simplified)
                if abs(action[0]) > 0.1:
                    trades.append({
                        "step": step_count,
                        "action": action[0],
                        "reward": reward,
                        "equity": current_equity
                    })
            
            # Calculate metrics
            final_equity = equity[-1]
            initial_equity = equity[0]
            net_profit = final_equity - initial_equity
            
            episode_metrics = {
                "net_profit": net_profit,
                "total_reward": total_reward,
                "total_trades": len(trades),
                "winning_trades": len([t for t in trades if t["reward"] > 0]),
                "losing_trades": len([t for t in trades if t["reward"] <= 0]),
                "win_rate": len([t for t in trades if t["reward"] > 0]) / max(1, len(trades)),
                "profit_factor": abs(sum([t["reward"] for t in trades if t["reward"] > 0])) / max(0.01, abs(sum([t["reward"] for t in trades if t["reward"] < 0]))),
                "max_drawdown": max(0.01, (max(equity) - min(equity)) / max(equity)),
                "sharpe_ratio": total_reward / max(0.01, np.std([e - equity[0] for e in equity])),
                "final_equity": final_equity
            }
            
            all_metrics.append(episode_metrics)
            equity_curves.append({"equity": equity, "timestamps": list(range(len(equity)))})
            trade_histories.append(trades)
            
        except Exception as e:
            _log(f"[ERROR] Error in legacy backtest episode {episode + 1}: {str(e)}")
            # Provide minimal fallback metrics
            fallback_metrics = {
                "net_profit": 0,
                "total_reward": 0,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "profit_factor": 1.0,
                "max_drawdown": 0.05,
                "sharpe_ratio": 0,
                "final_equity": config.get("INITIAL_CAPITAL", 100000),
                "error": str(e)
            }
            all_metrics.append(fallback_metrics)
            equity_curves.append({"equity": [100000], "timestamps": [0]})
            trade_histories.append([])
    
    return all_metrics, equity_curves, trade_histories

# ===== LEGACY COMPATIBILITY FUNCTIONS =====

def run_preset_comparison(df, preset_config, user_config, log_callback=None):
    """Legacy preset comparison function (maintained for compatibility)."""
    def _log(msg):
        if log_callback:
            log_callback(msg)
        else:
            log(msg)
    
    _log("[INFO] Running preset comparison")
    
    # Create dummy agent for comparison
    class DummyAgent:
        def select_action(self, obs, hidden):
            return [0.0, 0.0], None
    
    dummy_agent = DummyAgent()
    
    preset_results = run_backtest(df, dummy_agent, preset_config, 1, log_callback)
    user_results = run_backtest(df, dummy_agent, user_config, 1, log_callback)
    
    return preset_results, user_results

def calculate_drawdowns(equity_curve):
    """Calculate drawdowns from an equity curve."""
    if not equity_curve:
        return [], 0.0, 0
        
    # Calculate running maximum
    running_max = np.maximum.accumulate(equity_curve)
    
    # Calculate drawdown in percentage
    drawdown_pct = (running_max - equity_curve) / running_max
    
    # Find maximum drawdown
    max_drawdown = np.max(drawdown_pct)
    max_drawdown_idx = np.argmax(drawdown_pct)
    
    # Calculate drawdown duration
    if max_drawdown == 0:
        max_drawdown_duration = 0
    else:
        peak_idx = max_drawdown_idx - np.argmax(equity_curve[max_drawdown_idx::-1])
        try:
            recovery_idx = max_drawdown_idx + np.argmax(equity_curve[max_drawdown_idx:] >= running_max[peak_idx])
            max_drawdown_duration = recovery_idx - peak_idx
        except ValueError:
            max_drawdown_duration = len(equity_curve) - peak_idx
    
    return drawdown_pct.tolist(), max_drawdown, max_drawdown_duration

def analyze_trade_distribution(trade_history):
    """Analyze the distribution of trades for statistical properties."""
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
    
    # Extract profit data
    profits = [t.get("pnl", 0) for t in trade_history]
    
    # Calculate basic statistics
    num_trades = len(trade_history)
    wins = [p for p in profits if p > 0]
    losses = [p for p in profits if p <= 0]
    win_rate = len(wins) / num_trades if num_trades > 0 else 0.0
    avg_profit = np.mean(wins) if wins else 0.0
    avg_loss = np.mean(np.abs(losses)) if losses else 0.0
    profit_factor = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float('inf')
    expectancy = (win_rate * avg_profit - (1 - win_rate) * avg_loss)
    
    return {
        "num_trades": num_trades,
        "win_rate": win_rate,
        "avg_profit": avg_profit,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "avg_hold_time": 0,  # Simplified
        "profit_distribution": [],
        "hold_time_distribution": []
    }

def analyze_market_conditions(df, trade_history):
    """Analyze which market conditions lead to profitable trades."""
    if not trade_history or df is None or df.empty:
        return {"market_conditions": []}
    
    # Simplified analysis
    return {"market_conditions": []}

def plot_backtest_results(equity_curves, trade_histories, config, output_dir=None):
    """Create plots of backtest results."""
    if not equity_curves or not trade_histories:
        return []
    
    # Create basic equity curve plot
    plt.figure(figsize=(12, 6))
    
    for i, curve in enumerate(equity_curves[:5]):
        if isinstance(curve, dict) and "equity" in curve:
            plt.plot(curve["equity"], alpha=0.7, label=f'Episode {i+1}')
        else:
            plt.plot(curve, alpha=0.7, label=f'Episode {i+1}')
    
    plt.title(f'Equity Curves - {config.get("BUCKET", "Unknown")} Bucket')
    plt.xlabel('Steps')
    plt.ylabel('Capital ($)')
    plt.legend()
    plt.grid(True)
    
    if output_dir:
        filename = os.path.join(output_dir, f'equity_curve_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        return [filename]
    else:
        plt.show()
        return []

def generate_backtest_report(metrics, equity_curves, trade_histories, config, filename=None):
    """Generate a comprehensive backtest report."""
    if not metrics:
        return "No metrics available for report generation."
    
    # Calculate average metrics
    avg_metrics = {k: np.mean([m[k] for m in metrics]) for k in metrics[0]}
    
    # Generate report
    report = []
    report.append("="*80)
    report.append(f"BACKTEST REPORT - {config.get('BUCKET', 'Unknown')} BUCKET")
    report.append("="*80)
    report.append("")
    
    # Performance Metrics
    report.append("PERFORMANCE METRICS:")
    report.append("-"*80)
    report.append(f"Initial Capital: ${config.get('INITIAL_CAPITAL', 100000):.2f}")
    report.append(f"Final Capital: ${avg_metrics.get('final_equity', 0):.2f}")
    report.append(f"Net Profit: ${avg_metrics.get('net_profit', 0):.2f}")
    report.append(f"Total Trades: {avg_metrics.get('total_trades', 0):.1f}")
    report.append(f"Win Rate: {avg_metrics.get('win_rate', 0)*100:.2f}%")
    report.append(f"Sharpe Ratio: {avg_metrics.get('sharpe_ratio', 0):.2f}")
    report.append("")
    
    report.append("="*80)
    report.append(f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("="*80)
    
    if filename:
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write("\n".join(report))
        except Exception as e:
            log(f"Error saving report: {e}")
    
    return "\n".join(report)

class BacktestingEngine:
    """Enhanced backtesting engine (maintained for compatibility)."""
    
    def __init__(self, df, agent, config, use_advanced_features=True):
        self.df = df
        self.agent = agent
        self.config = config
        self.use_enhanced_features = use_advanced_features and EnhancedPredictiveBacktester is not None
        
        if self.use_enhanced_features:
            bucket_type = config.get("BUCKET", "Scalping")
            self.enhanced_backtester = EnhancedPredictiveBacktester(bucket_type, config)
        
    def run(self, episode_length=None, log_freq=10):
        """Run the backtesting engine."""
        if self.use_enhanced_features and hasattr(self.agent, 'model') and self.agent.model.config.get("AGENT_TYPE") == "predictive":
            return self.enhanced_backtester.run_enhanced_backtest(
                self.df, 
                self.agent,
                num_predictions=min(100, len(self.df) // 10),
                evaluation_frequency=log_freq
            )
        else:
            return run_backtest(self.df, self.agent, self.config, 1)
    
    def generate_report(self, save_path=None):
        """Generate backtest report."""
        if hasattr(self, 'enhanced_backtester'):
            if save_path:
                self.enhanced_backtester.save_backtest_report(save_path)
            return self.enhanced_backtester._calculate_final_results()
        return {}

class Backtester:
    """Legacy Backtester class (maintained for compatibility)."""
    
    def __init__(self, initial_capital=100000, max_position_size=0.2, 
                 transaction_fee_pct=0.001, slippage_pct=0.001):
        self.initial_capital = initial_capital
        self.max_position_size = max_position_size
        self.transaction_fee_pct = transaction_fee_pct
        self.slippage_pct = slippage_pct
        self.reset_results()
    
    def reset_results(self):
        """Reset backtesting results."""
        self.results = {
            "equity_curve": [],
            "trades": [],
            "metrics": {}
        }
    
    def run_backtest(self, data, signal_generator, strategy_params=None, verbose=True):
        """Run legacy backtest with signal generator."""
        self.reset_results()
        
        capital = self.initial_capital
        position = 0.0
        
        for i, row in data.iterrows():
            signal = signal_generator(row, **strategy_params or {})
            
            if signal and abs(signal) > 0.1:
                position_size = min(abs(signal) * self.max_position_size, self.max_position_size)
                
                if signal > 0 and position <= 0:
                    position = position_size
                elif signal < 0 and position >= 0:
                    position = -position_size
            
            self.results["equity_curve"].append({
                "timestamp": row.get('timestamp', i),
                "equity": capital,
                "position": position
            })
        
        return self.results
    
    def calculate_performance_metrics(self):
        """Calculate performance metrics."""
        if not self.results["equity_curve"]:
            return {}
        
        equity_values = [point["equity"] for point in self.results["equity_curve"]]
        
        return {
            "total_return": (equity_values[-1] - equity_values[0]) / equity_values[0],
            "max_drawdown": (max(equity_values) - min(equity_values)) / max(equity_values),
            "sharpe_ratio": 0.0,
            "num_trades": len(self.results["trades"])
        }

# Export enhanced functions for external use
__all__ = [
    'run_backtest',
    'run_preset_comparison', 
    'calculate_drawdowns',
    'analyze_trade_distribution',
    'analyze_market_conditions',
    'plot_backtest_results',
    'generate_backtest_report',
    'BacktestingEngine',
    'Backtester',
    'PredictiveAgentEvaluator',
    'EnhancedPredictiveBacktester',
    'run_enhanced_predictive_backtest'
] 