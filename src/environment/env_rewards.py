#!/usr/bin/env python
"""
Reward System for Trading Environments

This module implements bucket-specific reward systems for different trading
timeframes, focusing on appropriate performance metrics and behaviors
based on the trading strategy.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional, Union
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("env_rewards")

class BaseRewardSystem(ABC):
    """
    Base abstract class for reward systems across different trading buckets.
    
    Defines the interface and common methods for all reward systems, while
    allowing bucket-specific implementations to customize reward calculations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize base reward system with configuration parameters.
        
        Args:
            config: Configuration dictionary with reward parameters.
        """
        self.config = config
        
        # Common reward parameters
        self.initial_capital = config.get("INITIAL_CAPITAL", 100000.0)
        
        # Default reward weights (can be overridden by bucket-specific systems)
        self.profit_weight = config.get("PROFIT_WEIGHT", 0.5)
        self.consistency_weight = config.get("CONSISTENCY_WEIGHT", 0.3)
        self.risk_weight = config.get("RISK_WEIGHT", 0.5)
        self.drawdown_weight = config.get("DRAWDOWN_WEIGHT", 0.5)
        
        # Penalty caps
        self.max_risk_penalty = config.get("MAX_RISK_PENALTY", 1.0)
        self.max_drawdown_penalty = config.get("MAX_DRAWDOWN_PENALTY", 1.0)
        
        # Withdraw management weight (NEW)
        self.withdrawal_weight = config.get("WITHDRAWAL_WEIGHT", 0.3)
        
        # Confidence system parameters
        self.confidence_bonus_weight = config.get("CONFIDENCE_BONUS_WEIGHT", 0.1)
        self.uncertainty_penalty_weight = config.get("UNCERTAINTY_PENALTY_WEIGHT", 0.15)
        
        logger.info(f"Initialized BaseRewardSystem with initial capital: {self.initial_capital}")
    
    @abstractmethod
    def compute_reward(self, 
                     base_reward: float, 
                     profits: List[float], 
                     losses: List[float],
                     returns: List[float],
                     closed_trades: List[Tuple],
                     episode_days: float,
                     risk_metrics: Dict[str, Any],
                     prediction_metrics: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate reward based on trading performance.
        
        Args:
            base_reward: Base profit/loss reward from src.environment.env_base
            profits: List of profitable trade amounts
            losses: List of loss amounts (positive values)
            returns: List of return percentages
            closed_trades: List of closed trades as (profit, pct_gain, hold_time)
            episode_days: Episode duration in days
            risk_metrics: Dictionary of risk metrics
            prediction_metrics: Dictionary of prediction metrics including confidence scores,
                                prediction accuracy, and uncertainty metrics
            
        Returns:
            Calculated reward value
        """
        pass
    
    def _calculate_win_rate(self, profits: List[float], losses: List[float]) -> float:
        """
        Calculate win rate from profits and losses lists.
        
        Args:
            profits: List of profitable trades (positive values)
            losses: List of losing trades (positive values)
            
        Returns:
            Win rate as a fraction (0-1)
        """
        total_trades = len(profits) + len(losses)
        if total_trades == 0:
            return 0.0
        return len(profits) / total_trades
    
    def _calculate_profit_factor(self, profits: List[float], losses: List[float]) -> float:
        """
        Calculate profit factor as (sum of profits) / (sum of losses).
        
        Args:
            profits: List of profitable trades (positive values)
            losses: List of losing trades (positive values)
            
        Returns:
            Profit factor (> 1 is profitable)
        """
        total_profit = sum(profits)
        total_loss = sum(losses)
        
        if total_loss == 0:
            return 2.0 if total_profit > 0 else 1.0  # Avoid division by zero
            
        return total_profit / total_loss
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """
        Calculate Sharpe ratio from returns.
        
        Args:
            returns: List of trade returns
            
        Returns:
            Sharpe ratio value
        """
        if len(returns) < 2:
            return 0.0
            
        # Calculate with no risk-free rate for simplicity
        mean_return = np.mean(returns)
        std_return = np.std(returns) + 1e-8  # Avoid division by zero
        
        # Annualize (assuming daily returns data)
        annualized_sharpe = mean_return / std_return * np.sqrt(252)
        
        return annualized_sharpe
    
    def _calculate_sortino_ratio(self, returns: List[float]) -> float:
        """
        Calculate Sortino ratio from returns (penalizes only downside volatility).
        
        Args:
            returns: List of trade returns
            
        Returns:
            Sortino ratio value
        """
        if len(returns) < 2:
            return 0.0
            
        mean_return = np.mean(returns)
        
        # Calculate downside deviation (standard deviation of negative returns)
        negative_returns = [r for r in returns if r < 0]
        if not negative_returns:
            return mean_return * 10.0  # High value if no negative returns
            
        downside_dev = np.std(negative_returns) + 1e-8  # Avoid division by zero
        
        # Annualize (assuming daily returns data)
        annualized_sortino = mean_return / downside_dev * np.sqrt(252)
        
        return annualized_sortino
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """
        Calculate maximum drawdown from returns.
        
        Args:
            returns: List of trade returns
            
        Returns:
            Maximum drawdown as a positive fraction (0-1)
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
    
    def _calculate_profit_per_day(self, total_profit: float, episode_days: float) -> float:
        """
        Calculate average profit per day.
        
        Args:
            total_profit: Total profit amount
            episode_days: Episode duration in days
            
        Returns:
            Average daily profit
        """
        if episode_days <= 0:
            return 0.0
            
        return total_profit / episode_days
    
    def _calculate_trade_efficiency(self, closed_trades: List[Tuple]) -> float:
        """
        Calculate trade efficiency as profit relative to trade duration.
        
        Args:
            closed_trades: List of closed trades as (profit, pct_gain, hold_time)
            
        Returns:
            Trade efficiency score (higher is better)
        """
        if not closed_trades:
            return 0.0
            
        # Calculate profit per unit of hold time
        efficiencies = []
        for trade in closed_trades:
            profit, _, hold_time = trade[:3]
            
            # Skip trades with zero hold time
            if hold_time <= 0:
                continue
                
            # Efficiency is profit per unit of hold time
            efficiency = profit / hold_time
            efficiencies.append(efficiency)
        
        if not efficiencies:
            return 0.0
            
        # Average efficiency
        return np.mean(efficiencies)

    def _calculate_confidence_reward(self, prediction_metrics: Dict[str, Any]) -> float:
        """
        Calculate reward component based on prediction confidence and accuracy.
        
        Args:
            prediction_metrics: Dictionary containing confidence scores and accuracy
            
        Returns:
            Confidence-based reward component
        """
        if not prediction_metrics:
            return 0.0
            
        # Extract metrics
        confidence_scores = prediction_metrics.get("confidence_scores", [])
        prediction_accuracies = prediction_metrics.get("prediction_accuracies", [])
        
        if not confidence_scores or not prediction_accuracies:
            return 0.0
            
        # Average confidence score
        avg_confidence = np.mean(confidence_scores)
        
        # Average prediction accuracy
        avg_accuracy = np.mean(prediction_accuracies)
        
        # Calculate calibration error (difference between confidence and accuracy)
        calibration_error = abs(avg_confidence - avg_accuracy)
        
        # Well-calibrated predictions should have confidence ≈ accuracy
        # Overconfident: confidence > accuracy
        # Underconfident: confidence < accuracy
        
        # Reward for accuracy
        accuracy_reward = avg_accuracy * 0.5
        
        # Penalty for poor calibration
        calibration_penalty = calibration_error * 0.5
        
        # Overall confidence reward
        confidence_reward = accuracy_reward - calibration_penalty
        
        return confidence_reward
        
    def _calculate_uncertainty_reward(self, prediction_metrics: Dict[str, Any]) -> float:
        """
        Calculate reward component related to uncertainty handling.
        
        Args:
            prediction_metrics: Dictionary containing prediction metrics
            
        Returns:
            Reward/penalty amount
        """
        if not prediction_metrics:
            return 0.0
        
        # Default return if we don't have the necessary metrics
        if "uncertainty_scores" not in prediction_metrics:
            return 0.0
            
        # Extract uncertainty scores
        uncertainty_scores = prediction_metrics["uncertainty_scores"]
        if not uncertainty_scores:
            return 0.0
            
        # Calculate average uncertainty
        avg_uncertainty = sum(uncertainty_scores) / len(uncertainty_scores)
        
        # Calculate average confidence
        confidences = prediction_metrics.get("confidence_scores", [])
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        
        # Calculate uncertainty adjustment
        # Lower is better: well-calibrated models should have low uncertainty
        uncertainty_penalty = avg_uncertainty * self.uncertainty_penalty_weight
        
        # Scale by confidence - higher penalty when high confidence with high uncertainty
        scaled_penalty = uncertainty_penalty * (1.0 + avg_confidence)
        
        return -scaled_penalty  # Return as negative value (penalty)

    def _calculate_withdrawal_metrics(self, risk_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate metrics related to withdrawal management.
        
        Args:
            risk_metrics: Dictionary containing risk metrics, including withdrawal data
            
        Returns:
            Dictionary with withdrawal management metrics
        """
        # Default metrics
        withdrawal_metrics = {
            "has_withdrawals": False,
            "fulfilled_withdrawals": 0,
            "total_requested": 0,
            "fulfilled_amount": 0,
            "timed_withdrawal_count": 0,
            "timed_deadlines_met": 0,
            "emergency_withdrawal_count": 0,
            "emergency_requested": 0,
            "emergency_fulfilled": 0,
            "standard_withdrawal_count": 0,
            "fulfillment_consistency": 0.0,
            "usd_balance": 0.0,
            "usdt_balance": 0.0,
            "total_capital": self.initial_capital,
            "withdrawal_handling_score": 0.0,
        }
        
        # Return default if no withdrawal data is available
        if not risk_metrics or "withdrawal_data" not in risk_metrics:
            return withdrawal_metrics
            
        withdrawal_data = risk_metrics["withdrawal_data"]
        
        # Extract withdrawal statistics
        withdrawal_metrics["has_withdrawals"] = withdrawal_data.get("has_withdrawals", False)
        withdrawal_metrics["fulfilled_withdrawals"] = withdrawal_data.get("fulfilled_withdrawals", 0)
        withdrawal_metrics["total_requested"] = withdrawal_data.get("total_requested", 0)
        withdrawal_metrics["fulfilled_amount"] = withdrawal_data.get("fulfilled_amount", 0)
        
        # Type-specific metrics
        withdrawal_metrics["timed_withdrawal_count"] = withdrawal_data.get("timed_count", 0)
        withdrawal_metrics["timed_deadlines_met"] = withdrawal_data.get("timed_deadlines_met", 0)
        withdrawal_metrics["emergency_withdrawal_count"] = withdrawal_data.get("emergency_count", 0)
        withdrawal_metrics["emergency_requested"] = withdrawal_data.get("emergency_requested", 0)
        withdrawal_metrics["emergency_fulfilled"] = withdrawal_data.get("emergency_fulfilled", 0)
        withdrawal_metrics["standard_withdrawal_count"] = withdrawal_data.get("standard_count", 0)
        
        # Financial metrics
        withdrawal_metrics["usd_balance"] = withdrawal_data.get("usd_balance", 0.0)
        withdrawal_metrics["usdt_balance"] = withdrawal_data.get("usdt_balance", 0.0)
        withdrawal_metrics["total_capital"] = (withdrawal_metrics["usd_balance"] + 
                                              withdrawal_metrics["usdt_balance"])
        
        # Calculate overall withdrawal handling score (0.0-1.0)
        fulfillment_ratio = (withdrawal_metrics["fulfilled_amount"] / 
                            withdrawal_metrics["total_requested"]) if withdrawal_metrics["total_requested"] > 0 else 1.0
                            
        emergency_ratio = (withdrawal_metrics["emergency_fulfilled"] / 
                          withdrawal_metrics["emergency_requested"]) if withdrawal_metrics["emergency_requested"] > 0 else 1.0
                          
        timed_ratio = (withdrawal_metrics["timed_deadlines_met"] / 
                       withdrawal_metrics["timed_withdrawal_count"]) if withdrawal_metrics["timed_withdrawal_count"] > 0 else 1.0
        
        # Weight the different components
        withdrawal_metrics["withdrawal_handling_score"] = (
            fulfillment_ratio * 0.5 +  # General fulfillment
            emergency_ratio * 0.3 +    # Emergency handling
            timed_ratio * 0.2          # Meeting deadlines
        )
        
        return withdrawal_metrics


def compute_risk_adjusted_reward(
        base_reward: float,
        profits: List[float],
        losses: List[float],
        returns: List[float],
        bucket: str,
        closed_trades: List[Tuple],
        episode_days: float,
        config: Dict[str, Any],
        prediction_metrics: Optional[Dict[str, Any]] = None) -> float:
    """
    Legacy function for backwards compatibility with older environments.
    Creates appropriate reward system based on bucket and delegates computation.
    
    Args:
        base_reward: Base profit/loss from trading
        profits: List of profitable trade amounts
        losses: List of loss amounts
        returns: List of return percentages  
        bucket: Trading timeframe bucket
        closed_trades: List of closed trades
        episode_days: Episode duration in days
        config: Configuration dictionary
        prediction_metrics: Dictionary of prediction metrics including confidence 
                           scores, accuracy, and uncertainty bounds
        
    Returns:
        Calculated reward value
    """
    # Create appropriate reward system based on bucket
    if bucket.lower() == "scalping":
        reward_system = ScalpingRewardSystem(config)
    elif bucket.lower() == "short":
        reward_system = ShortRewardSystem(config)
    elif bucket.lower() == "medium":
        reward_system = MediumRewardSystem(config)
    elif bucket.lower() == "long":
        reward_system = LongRewardSystem(config)
    else:
        # Default to medium timeframe
        reward_system = MediumRewardSystem(config)
    
    # Create a basic risk metrics dictionary if none provided
    risk_metrics = {
        "overall_risk_score": 0.5,
        "drawdown": 0.1,
        "exposure_percentage": 0.5,
        "risk_concentration": 0.5,
        "position_diversity": 0.5
    }
    
    # Delegate to appropriate reward system
    return reward_system.compute_reward(
        base_reward, profits, losses, returns, closed_trades, episode_days, risk_metrics, prediction_metrics
    )

class ScalpingRewardSystem(BaseRewardSystem):
    """
    Reward system for Scalping trading strategy.
    
    Focuses on:
    - High trade frequency with small gains per trade
    - Very short holding periods
    - Tight risk management
    - Consistent execution
    - Low exposure to market
    
    Args:
        base_reward: Base profit/loss from trading
        profits: List of profitable trade amounts
        losses: List of loss amounts
        returns: List of return percentages
        closed_trades: List of closed trades as (profit, pct_gain, hold_time)
        episode_days: Episode duration in days
        risk_metrics: Dictionary of risk metrics
        prediction_metrics: Dictionary of prediction metrics
        
    Returns:
        Calculated reward value
    """
    def __init__(self, config: Dict[str, Any]):
        """Initialize scalping-specific reward system."""
        super().__init__(config)
        
        # Scalping-specific parameters
        self.monthly_target_min = config.get("monthly_target_min", 15.0)  # Minimum monthly return target (%)
        self.monthly_target_max = config.get("monthly_target_max", 30.0)  # Maximum monthly return target (%)
        
        # Expected trades and hold times
        self.expected_trades_per_day = config.get("SCALPING_EXPECTED_TRADES_PER_DAY", 20)  # Expected trades per day
        self.max_hold_time = config.get("SCALPING_MAX_HOLD_TIME", 48)  # Max hold time in bars
        self.ideal_hold_time = config.get("SCALPING_IDEAL_HOLD_TIME", 24)  # Ideal hold time in bars
        
        # Adjust weights for scalping strategy
        self.profit_weight = config.get("SCALPING_PROFIT_WEIGHT", 0.8)
        self.consistency_weight = config.get("SCALPING_CONSISTENCY_WEIGHT", 0.6)  # Higher consistency weight
        self.risk_weight = config.get("SCALPING_RISK_WEIGHT", 0.7)  # Higher risk sensitivity
        self.drawdown_weight = config.get("SCALPING_DRAWDOWN_WEIGHT", 0.7)  # More sensitive to drawdowns
        
        # Scalping-specific metrics
        self.trade_frequency_weight = config.get("TRADE_FREQUENCY_WEIGHT", 0.4)  # Reward frequent trading
        self.avg_trade_duration_target = config.get("AVG_TRADE_DURATION_TARGET", 24)  # Target trade duration in bars
        self.quick_profit_bonus = config.get("QUICK_PROFIT_BONUS", 0.3)  # Bonus for quick profits
        
        # Win rate expectation is higher for scalping
        self.min_win_rate = config.get("SCALPING_MIN_WIN_RATE", 0.5)  # Higher minimum win rate
        self.target_win_rate = config.get("SCALPING_TARGET_WIN_RATE", 0.65)  # Higher target win rate
        
        # Profit factor expectations for scalping
        self.min_profit_factor = config.get("SCALPING_MIN_PROFIT_FACTOR", 1.5)  # Minimum profit factor
        self.target_profit_factor = config.get("SCALPING_TARGET_PROFIT_FACTOR", 2.0)  # Target profit factor
        
        # Market exposure settings
        self.max_market_exposure = config.get("SCALPING_MAX_MARKET_EXPOSURE", 0.4)  # Low exposure for scalping
        
        logger.info(f"Initialized ScalpingRewardSystem with monthly targets: {self.monthly_target_min}%-{self.monthly_target_max}%")
    
    def compute_reward(self, 
                     base_reward: float, 
                     profits: List[float], 
                     losses: List[float],
                     returns: List[float],
                     closed_trades: List[Tuple],
                     episode_days: float,
                     risk_metrics: Dict[str, Any],
                     prediction_metrics: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate reward for scalping strategy.
        
        Focuses on:
        - High trade frequency with small gains per trade
        - Very short holding periods
        - Tight risk management
        - Consistent execution
        - Low exposure to market
        
        Args:
            base_reward: Base profit/loss from trading
            profits: List of profitable trade amounts
            losses: List of loss amounts
            returns: List of return percentages
            closed_trades: List of closed trades as (profit, pct_gain, hold_time)
            episode_days: Episode duration in days
            risk_metrics: Dictionary of risk metrics
            prediction_metrics: Dictionary of prediction metrics
            
        Returns:
            Calculated reward value
        """
        # Start with base reward (total P&L)
        reward = base_reward * self.profit_weight
        
        # Calculate total P&L
        total_profit = sum(profits)
        total_loss = sum(losses)
        net_pnl = total_profit - total_loss
        
        # Convert closed_trades to expected format if needed
        formatted_trades = []
        for trade in closed_trades:
            if isinstance(trade, tuple):
                formatted_trades.append(trade)
            elif hasattr(trade, 'to_tuple'):
                formatted_trades.append(trade.to_tuple())
            else:
                profit = getattr(trade, 'profit', 0.0)
                pct_gain = getattr(trade, 'percentage_gain', 0.0)
                hold_time = getattr(trade, 'hold_time', 0)
                formatted_trades.append((profit, pct_gain, hold_time))
        
        # Skip further calculations if no trades were made
        if not formatted_trades:
            return reward
        
        # Calculate monthly return for target comparison
        monthly_return_pct = 0.0
        if episode_days > 0:
            total_return_pct = (net_pnl / self.initial_capital) * 100
            monthly_return_pct = total_return_pct * (30.0 / episode_days)
        
        # Apply monthly target reward/penalty
        if monthly_return_pct < self.monthly_target_min:
            # Penalty for below minimum target
            target_penalty = (self.monthly_target_min - monthly_return_pct) / self.monthly_target_min
            target_penalty *= 0.5 * abs(base_reward)  # Scale by base reward
            reward -= target_penalty
        elif monthly_return_pct > self.monthly_target_max:
            # Penalty for exceeding maximum target (excessive risk)
            target_penalty = (monthly_return_pct - self.monthly_target_max) / self.monthly_target_max
            target_penalty *= 0.3 * abs(base_reward)  # Scale by base reward
            reward -= target_penalty
        else:
            # Bonus for hitting target range
            target_bonus = 0.3 * abs(base_reward)
            reward += target_bonus
        
        # Calculate trade frequency - scalping needs high frequency
        trade_count = len(formatted_trades)
        expected_trades_per_day = self.expected_trades_per_day
        expected_trades = max(1, expected_trades_per_day * episode_days)
        
        # Trade frequency component (critical for scalping)
        if trade_count < expected_trades * 0.5:
            # Significant penalty for too few trades
            trade_frequency_penalty = 0.4 * abs(base_reward) * (1 - (trade_count / (expected_trades * 0.5)))
            reward -= trade_frequency_penalty
        elif trade_count > expected_trades * 1.5:
            # Small penalty for excessive trading
            trade_frequency_penalty = 0.1 * abs(base_reward) * ((trade_count / (expected_trades * 1.5)) - 1)
            reward -= trade_frequency_penalty
        else:
            # Bonus for ideal trade frequency
            trade_frequency_bonus = 0.2 * abs(base_reward)
            reward += trade_frequency_bonus
        
        # Calculate key trading metrics
        win_rate = self._calculate_win_rate(profits, losses)
        profit_factor = self._calculate_profit_factor(profits, losses)
        
        # Win rate component (critical for scalping)
        if win_rate >= self.target_win_rate:
            # Bonus for high win rate
            win_rate_bonus = (win_rate - self.target_win_rate) * 0.5 * abs(base_reward)
            reward += win_rate_bonus * self.consistency_weight
        elif win_rate < self.min_win_rate:
            # Severe penalty for low win rate
            win_rate_penalty = (self.min_win_rate - win_rate) * 0.8 * abs(base_reward)
            reward -= win_rate_penalty * self.consistency_weight
        
        # Trade duration component (critical for scalping)
        if formatted_trades:
            avg_hold_time = sum(trade[2] for trade in formatted_trades if len(trade) > 2) / len(formatted_trades)
            
            if avg_hold_time > self.max_hold_time:
                # Severe penalty for holding too long
                hold_time_penalty = 0.5 * abs(base_reward) * (avg_hold_time / self.max_hold_time - 1)
                reward -= hold_time_penalty
            elif avg_hold_time < self.ideal_hold_time:
                # Small bonus for quick trades
                hold_time_bonus = 0.2 * abs(base_reward) * (1 - avg_hold_time / self.ideal_hold_time)
                reward += hold_time_bonus
        
        # Profit factor component
        if profit_factor >= self.target_profit_factor:
            # Bonus for high profit factor
            pf_bonus = min(1.0, (profit_factor - self.target_profit_factor) / self.target_profit_factor) * 0.3 * abs(base_reward)
            reward += pf_bonus
        elif profit_factor < self.min_profit_factor:
            # Penalty for low profit factor
            pf_penalty = (self.min_profit_factor - profit_factor) / self.min_profit_factor * 0.4 * abs(base_reward)
            reward -= pf_penalty
        
        # Risk metrics penalties
        # For scalping, we have very little tolerance for risk
        max_acceptable_drawdown = 0.08  # 8% max drawdown for scalping
        drawdown = risk_metrics.get("drawdown", self._calculate_max_drawdown(returns))
        
        if drawdown > max_acceptable_drawdown:
            drawdown_penalty = (drawdown - max_acceptable_drawdown) * self.max_drawdown_penalty * abs(base_reward)
            reward -= drawdown_penalty * self.drawdown_weight * 1.5  # Higher weight for drawdown in scalping
        
        # Calculate exposure time (percentage of time with open positions)
        exposure = risk_metrics.get("market_exposure", 0.5)
        
        # For scalping, we want low exposure time
        if exposure > self.max_market_exposure:
            # Penalty for excessive market exposure
            exposure_penalty = (exposure - self.max_market_exposure) / (1 - self.max_market_exposure) * 0.3 * abs(base_reward)
            reward -= exposure_penalty
        
        # Withdrawal management component (NEW)
        withdrawal_metrics = self._calculate_withdrawal_metrics(risk_metrics)
        
        if withdrawal_metrics["has_withdrawals"]:
            # Scalping strategies need to be especially liquid for withdrawals
            withdrawal_score = withdrawal_metrics["withdrawal_handling_score"]
            
            # Base withdrawal reward based on handling score
            withdrawal_reward = withdrawal_score * 0.4 * abs(base_reward)
            
            # For scalping, rapid response to emergency withdrawals is critical
            if withdrawal_metrics["emergency_withdrawal_count"] > 0:
                emergency_success = withdrawal_metrics["emergency_fulfilled"] / withdrawal_metrics["emergency_requested"]
                if emergency_success > 0.95:  # Higher standard for scalping (95%)
                    withdrawal_reward += 0.3 * abs(base_reward)
                elif emergency_success < 0.8:  # Harsher penalty for scalping
                    withdrawal_reward -= 0.3 * abs(base_reward)
            
            # Scalping should maintain higher liquidity reserves
            reserve_ratio = withdrawal_metrics["usd_balance"] / max(1.0, withdrawal_metrics["total_capital"])
            if reserve_ratio > 0.1:  # Scalping should maintain >10% reserves (higher than medium-term)
                withdrawal_reward += 0.2 * abs(base_reward)
            elif reserve_ratio < 0.05:  # Penalty for low reserves
                withdrawal_reward -= 0.2 * abs(base_reward)
            
            # Apply withdrawal reward with appropriate weight
            reward += withdrawal_reward * self.withdrawal_weight
        
        # Confidence and uncertainty rewards
        confidence_reward = self._calculate_confidence_reward(prediction_metrics)
        uncertainty_reward = self._calculate_uncertainty_reward(prediction_metrics)
        
        return reward + confidence_reward + uncertainty_reward


class ShortRewardSystem(BaseRewardSystem):
    """
    Reward system for Short-term trading strategy.
    
    Focuses on:
    - Medium-term gains with yearly performance targets
    - Moderate trade frequency
    - Balanced risk/reward approach
    - Strategic position sizing
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize short-term trading reward system."""
        super().__init__(config)
        
        # Short-term specific parameters
        self.yearly_target_min = config.get("yearly_target_min", 100.0)  # Minimum yearly return target (%)
        self.yearly_target_max = config.get("yearly_target_max", 200.0)  # Maximum yearly return target (%)
        
        # Adjust weights for short-term strategy
        self.profit_weight = config.get("SHORT_PROFIT_WEIGHT", 1.0)
        self.consistency_weight = config.get("SHORT_CONSISTENCY_WEIGHT", 0.4)
        self.risk_weight = config.get("SHORT_RISK_WEIGHT", 0.5)
        self.drawdown_weight = config.get("SHORT_DRAWDOWN_WEIGHT", 0.5)
        
        # Short-term specific metrics
        self.avg_trade_duration_target = config.get("SHORT_AVG_TRADE_DURATION", 72)  # Target in bars (longer than scalping)
        self.strategic_exit_bonus = config.get("STRATEGIC_EXIT_BONUS", 0.3)  # Bonus for well-timed exits
        
        # Win rate expectations for short-term are more balanced
        self.min_win_rate = config.get("SHORT_MIN_WIN_RATE", 0.45)
        self.target_win_rate = config.get("SHORT_TARGET_WIN_RATE", 0.55)
        
        # Profit factor expectations are higher for short-term
        self.min_profit_factor = config.get("SHORT_MIN_PROFIT_FACTOR", 1.3)
        self.target_profit_factor = config.get("SHORT_TARGET_PROFIT_FACTOR", 1.8)
        
        logger.info(f"Initialized ShortRewardSystem with yearly targets: {self.yearly_target_min}%-{self.yearly_target_max}%")
    
    def compute_reward(self, 
                     base_reward: float, 
                     profits: List[float], 
                     losses: List[float],
                     returns: List[float],
                     closed_trades: List[Tuple],
                     episode_days: float,
                     risk_metrics: Dict[str, Any],
                     prediction_metrics: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate reward for short-term trading strategy.
        
        Short-term focuses on balancing frequency and profit size with yearly targets.
        
        Args:
            base_reward: Base profit/loss from trading
            profits: List of profitable trade amounts
            losses: List of loss amounts
            returns: List of return percentages
            closed_trades: List of closed trades as (profit, pct_gain, hold_time)
            episode_days: Episode duration in days
            risk_metrics: Dictionary of risk metrics
            prediction_metrics: Dictionary of prediction metrics including confidence scores,
                                prediction accuracy, and uncertainty metrics
            
        Returns:
            Calculated reward value
        """
        # Start with base reward (total P&L)
        reward = base_reward * self.profit_weight
        
        # Calculate total P&L
        total_profit = sum(profits)
        total_loss = sum(losses)
        net_pnl = total_profit - total_loss
        
        # Convert closed_trades to expected format if needed
        formatted_trades = []
        for trade in closed_trades:
            if isinstance(trade, tuple):
                # If already a tuple, use as is
                formatted_trades.append(trade)
            elif hasattr(trade, 'to_tuple'):
                # If it has a to_tuple method, use that
                formatted_trades.append(trade.to_tuple())
            else:
                # Try to extract components
                profit = getattr(trade, 'profit', 0.0)
                pct_gain = getattr(trade, 'percentage_gain', 0.0)
                hold_time = getattr(trade, 'hold_time', 0)
                formatted_trades.append((profit, pct_gain, hold_time))
        
        # Skip further calculations if no trades were made
        if not formatted_trades:
            return reward
        
        # Calculate yearly performance (annualized)
        # For short-term strategy, yearly performance is the key metric
        yearly_return_pct = 0.0
        if episode_days > 0:
            # Convert net PnL to percentage of initial capital
            total_return_pct = (net_pnl / self.initial_capital) * 100
            # Annualize
            yearly_return_pct = total_return_pct * (365.0 / episode_days)
        
        # Apply yearly target reward/penalty
        if yearly_return_pct < self.yearly_target_min:
            # Penalty for below minimum target
            target_penalty = (self.yearly_target_min - yearly_return_pct) / self.yearly_target_min
            target_penalty *= 0.4 * abs(base_reward)  # Scale by base reward
            reward -= target_penalty
        elif yearly_return_pct > self.yearly_target_max:
            # Penalty for exceeding maximum target (excessive risk-taking)
            target_penalty = (yearly_return_pct - self.yearly_target_max) / self.yearly_target_max
            target_penalty *= 0.15 * abs(base_reward)  # Smaller penalty for exceeding
            reward -= target_penalty
        else:
            # Bonus for hitting target range
            target_bonus = 0.25 * abs(base_reward)
            reward += target_bonus
        
        # Calculate key trading metrics
        win_rate = self._calculate_win_rate(profits, losses)
        profit_factor = self._calculate_profit_factor(profits, losses)
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        max_drawdown = self._calculate_max_drawdown(returns)
        
        # Win rate component
        if win_rate >= self.target_win_rate:
            # Bonus for high win rate
            win_rate_bonus = (win_rate - self.target_win_rate) * 0.4 * abs(base_reward)
            reward += win_rate_bonus * self.consistency_weight
        elif win_rate < self.min_win_rate:
            # Penalty for low win rate
            win_rate_penalty = (self.min_win_rate - win_rate) * 0.5 * abs(base_reward)
            reward -= win_rate_penalty * self.consistency_weight
        
        # Profit factor component (more important for short-term)
        if profit_factor >= self.target_profit_factor:
            # Bonus for high profit factor
            pf_bonus = min(1.0, (profit_factor - self.target_profit_factor)) * 0.4 * abs(base_reward)
            reward += pf_bonus
        elif profit_factor < self.min_profit_factor:
            # Penalty for low profit factor
            pf_penalty = (self.min_profit_factor - profit_factor) * 0.5 * abs(base_reward)
            reward -= pf_penalty
        
        # Trade frequency component
        trade_count = len(formatted_trades)
        expected_trades_per_day = 2.5  # Short-term expects fewer trades than scalping
        expected_trades = max(1, expected_trades_per_day * episode_days)
        
        if trade_count < expected_trades * 0.5:
            # Penalty for too few trades
            trade_frequency_penalty = 0.2 * abs(base_reward) * (1 - (trade_count / (expected_trades * 0.5)))
            reward -= trade_frequency_penalty
        elif trade_count > expected_trades * 2.0:
            # Penalty for excessive trading
            trade_frequency_penalty = 0.2 * abs(base_reward) * ((trade_count / (expected_trades * 2.0)) - 1)
            reward -= trade_frequency_penalty
        
        # Trade timing component - evaluate exit quality
        if formatted_trades:
            # Calculate average percentage gain
            avg_pct_gain = sum(trade[1] for trade in formatted_trades if len(trade) > 1) / len(formatted_trades)
            expected_gain_per_trade = 1.0  # 1% per trade for short-term
            
            if avg_pct_gain >= expected_gain_per_trade:
                # Bonus for good exits
                exit_bonus = min(2.0, avg_pct_gain / expected_gain_per_trade) * 0.2 * abs(base_reward)
                reward += exit_bonus * self.strategic_exit_bonus
            else:
                # Small penalty for poor exits
                exit_penalty = (1.0 - avg_pct_gain / expected_gain_per_trade) * 0.15 * abs(base_reward)
                reward -= exit_penalty * self.strategic_exit_bonus
        
        # Average hold time component - short-term should have moderate duration
        if formatted_trades:
            avg_hold_time = sum(trade[2] for trade in formatted_trades if len(trade) > 2) / len(formatted_trades)
            
            # For short-term, we want holds in the target range
            if avg_hold_time < self.avg_trade_duration_target * 0.3:
                # Too short for short-term strategy
                duration_penalty = 0.2 * abs(base_reward) * (1 - (avg_hold_time / (self.avg_trade_duration_target * 0.3)))
                reward -= duration_penalty
            elif avg_hold_time > self.avg_trade_duration_target * 2.0:
                # Too long for short-term strategy
                duration_penalty = 0.2 * abs(base_reward) * ((avg_hold_time / (self.avg_trade_duration_target * 2.0)) - 1)
                reward -= duration_penalty
            else:
                # Bonus for ideal hold time
                duration_bonus = 0.15 * abs(base_reward)
                reward += duration_bonus
        
        # Risk metrics penalties
        risk_score = risk_metrics.get("overall_risk_score", 0.5)
        risk_threshold = 0.75  # Short-term has slightly higher risk tolerance than scalping
        
        if risk_score > risk_threshold:
            risk_penalty = (risk_score - risk_threshold) * self.max_risk_penalty * abs(base_reward)
            reward -= risk_penalty * self.risk_weight
        
        # Drawdown penalty
        drawdown = risk_metrics.get("drawdown", max_drawdown)
        max_acceptable_drawdown = 0.12  # 12% is reasonable for short-term
        
        if drawdown > max_acceptable_drawdown:
            drawdown_penalty = (drawdown - max_acceptable_drawdown) * self.max_drawdown_penalty * abs(base_reward)
            reward -= drawdown_penalty * self.drawdown_weight
        
        # Sharpe ratio bonus/penalty
        if sharpe_ratio > 1.5:
            sharpe_bonus = min(1.0, (sharpe_ratio - 1.5) / 1.5) * 0.3 * abs(base_reward)
            reward += sharpe_bonus
        elif sharpe_ratio < 0.8:
            sharpe_penalty = (0.8 - sharpe_ratio) / 0.8 * 0.3 * abs(base_reward)
            reward -= sharpe_penalty
        
        # Confidence and uncertainty rewards
        confidence_reward = self._calculate_confidence_reward(prediction_metrics)
        uncertainty_reward = self._calculate_uncertainty_reward(prediction_metrics)
        
        return reward + confidence_reward + uncertainty_reward


class MediumRewardSystem(BaseRewardSystem):
    """
    Reward system for medium-term trading.
    
    Focuses on:
    - Balanced risk and return
    - Moderate hold times
    - Profit consistency
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize medium-term specific reward system."""
        super().__init__(config)
        
        # Medium-term specific parameters
        self.monthly_target_min = config.get("monthly_target_min", 10.0)  # Minimum monthly return target (%)
        self.monthly_target_max = config.get("monthly_target_max", 20.0)  # Maximum monthly return target (%)
        self.yearly_target_min = config.get("yearly_target_min", 100.0)  # Minimum yearly return target (%)
        self.yearly_target_max = config.get("yearly_target_max", 200.0)  # Maximum yearly return target (%)
        
        # Adjust weights for medium-term strategy
        self.profit_weight = config.get("MEDIUM_PROFIT_WEIGHT", 0.5)
        self.consistency_weight = config.get("MEDIUM_CONSISTENCY_WEIGHT", 0.5)  # Balanced consistency weight
        self.risk_weight = config.get("MEDIUM_RISK_WEIGHT", 0.5)  # Moderate risk sensitivity
        self.drawdown_weight = config.get("MEDIUM_DRAWDOWN_WEIGHT", 0.4)  # Moderate drawdown sensitivity
        
        # Medium-term specific metrics
        self.ideal_hold_time = config.get("MEDIUM_IDEAL_HOLD_TIME", 288)  # ~1 day for 5-min bars
        self.min_win_rate = config.get("MEDIUM_MIN_WIN_RATE", 0.4)  # Realistic minimum win rate
        self.target_win_rate = config.get("MEDIUM_TARGET_WIN_RATE", 0.55)  # Target win rate
        self.expected_trades_per_day = config.get("MEDIUM_EXPECTED_TRADES_PER_DAY", 1)  # Expected trades per day
        
        # Special focus on risk-adjusted returns
        self.sharpe_weight = config.get("MEDIUM_SHARPE_WEIGHT", 0.4)  # Emphasize risk-adjusted returns
        
        # Withdrawal management weight (adjusted for medium-term)
        self.withdrawal_weight = config.get("MEDIUM_WITHDRAWAL_WEIGHT", 0.4)  # Medium-term strategy should handle withdrawals well
        
        logger.info(f"Initialized MediumRewardSystem with monthly targets: {self.monthly_target_min}%-{self.monthly_target_max}%")
    
    def compute_reward(self, 
                     base_reward: float, 
                     profits: List[float], 
                     losses: List[float],
                     returns: List[float],
                     closed_trades: List[Tuple],
                     episode_days: float,
                     risk_metrics: Dict[str, Any],
                     prediction_metrics: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate reward for medium-term strategy.
        
        Medium-term focuses on balanced risk and return with moderate hold times.
        Both monthly and yearly performance are evaluated, with an emphasis on
        consistency and capital preservation alongside growth.
        
        Args:
            base_reward: Base profit/loss from trading
            profits: List of profitable trade amounts
            losses: List of loss amounts
            returns: List of return percentages
            closed_trades: List of closed trades as (profit, pct_gain, hold_time)
            episode_days: Episode duration in days
            risk_metrics: Dictionary of risk metrics
            prediction_metrics: Dictionary of prediction metrics
            
        Returns:
            Calculated reward value
        """
        # Start with base reward (total P&L)
        reward = base_reward * self.profit_weight
        
        # Calculate total P&L
        total_profit = sum(profits)
        total_loss = sum(losses)
        net_pnl = total_profit - total_loss
        
        # Convert closed_trades to expected format if needed
        formatted_trades = []
        for trade in closed_trades:
            if isinstance(trade, tuple):
                # If already a tuple, use as is
                formatted_trades.append(trade)
            elif hasattr(trade, 'to_tuple'):
                # If it has a to_tuple method, use that
                formatted_trades.append(trade.to_tuple())
            else:
                # Try to extract components
                profit = getattr(trade, 'profit', 0.0)
                pct_gain = getattr(trade, 'percentage_gain', 0.0)
                hold_time = getattr(trade, 'hold_time', 0)
                formatted_trades.append((profit, pct_gain, hold_time))
        
        # Skip further calculations if no trades were made
        if not formatted_trades:
            return reward
        
        # Calculate performance metrics
        # For medium-term, we care about both monthly and yearly performance
        monthly_return_pct = 0.0
        yearly_return_pct = 0.0
        if episode_days > 0:
            # Convert net PnL to percentage of initial capital
            total_return_pct = (net_pnl / self.initial_capital) * 100
            # Calculate monthly and yearly returns
            monthly_return_pct = total_return_pct * (30.0 / episode_days)
            yearly_return_pct = total_return_pct * (365.0 / episode_days)
        
        # Apply monthly target reward/penalty
        if monthly_return_pct < self.monthly_target_min:
            # Penalty for below minimum target (but less severe than short-term)
            target_penalty = (self.monthly_target_min - monthly_return_pct) / self.monthly_target_min
            target_penalty *= 0.4 * abs(base_reward)  # Scale by base reward
            reward -= target_penalty
        elif monthly_return_pct > self.monthly_target_max:
            # Penalty for exceeding maximum target (excessive risk-taking)
            target_penalty = (monthly_return_pct - self.monthly_target_max) / self.monthly_target_max
            target_penalty *= 0.2 * abs(base_reward)  # Smaller penalty for exceeding
            reward -= target_penalty
        else:
            # Bonus for hitting target range
            target_bonus = 0.2 * abs(base_reward)
            reward += target_bonus
        
        # Apply yearly target reward/penalty
        if yearly_return_pct < self.yearly_target_min:
            # Penalty for below minimum yearly target
            target_penalty = (self.yearly_target_min - yearly_return_pct) / self.yearly_target_min
            target_penalty *= 0.3 * abs(base_reward)  # Scale by base reward
            reward -= target_penalty
        elif yearly_return_pct > self.yearly_target_max:
            # Penalty for exceeding maximum yearly target
            target_penalty = (yearly_return_pct - self.yearly_target_max) / self.yearly_target_max
            target_penalty *= 0.15 * abs(base_reward)  # Smaller penalty for exceeding
            reward -= target_penalty
        else:
            # Bonus for hitting yearly target range
            target_bonus = 0.25 * abs(base_reward)
            reward += target_bonus
        
        # Calculate key trading metrics
        win_rate = self._calculate_win_rate(profits, losses)
        profit_factor = self._calculate_profit_factor(profits, losses)
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        max_drawdown = self._calculate_max_drawdown(returns)
        
        # Win rate component
        if win_rate >= self.target_win_rate:
            # Bonus for high win rate
            win_rate_bonus = (win_rate - self.target_win_rate) * 0.4 * abs(base_reward)
            reward += win_rate_bonus * self.consistency_weight
        elif win_rate < self.min_win_rate:
            # Penalty for low win rate
            win_rate_penalty = (self.min_win_rate - win_rate) * 0.5 * abs(base_reward)
            reward -= win_rate_penalty * self.consistency_weight
        
        # Trade duration component - reward appropriate hold times for medium-term
        if formatted_trades:
            # Calculate average hold time
            avg_hold_time = sum(trade[2] for trade in formatted_trades if len(trade) > 2) / len(formatted_trades)
            
            # For medium-term, a reasonable range of hold times is desired
            # Too short might indicate poor execution, too long might miss opportunities
            if avg_hold_time < self.ideal_hold_time * 0.3:
                # Too short for medium-term
                hold_time_penalty = 0.2 * abs(base_reward) * (1 - (avg_hold_time / (self.ideal_hold_time * 0.3)))
                reward -= hold_time_penalty
            elif avg_hold_time > self.ideal_hold_time * 3.0:
                # Too long for medium-term
                hold_time_penalty = 0.2 * abs(base_reward) * ((avg_hold_time / (self.ideal_hold_time * 3.0)) - 1)
                reward -= hold_time_penalty
            else:
                # Ideal range - small bonus
                hold_time_bonus = 0.1 * abs(base_reward)
                reward += hold_time_bonus
        
        # Sharpe ratio component - important for medium-term risk-adjusted returns
        if sharpe_ratio > 1.0:
            # Bonus for good risk-adjusted returns
            sharpe_bonus = min(1.0, sharpe_ratio - 1.0) * 0.4 * abs(base_reward)
            reward += sharpe_bonus * self.sharpe_weight
        elif sharpe_ratio < 0.3:
            # Penalty for poor risk-adjusted returns
            sharpe_penalty = (0.3 - sharpe_ratio) * 0.4 * abs(base_reward)
            reward -= sharpe_penalty * self.sharpe_weight
        
        # Drawdown component
        drawdown = risk_metrics.get("drawdown", max_drawdown)
        max_acceptable_drawdown = 0.15  # 15% is reasonable for medium-term
        
        if drawdown > max_acceptable_drawdown:
            # Penalty for excessive drawdown
            drawdown_penalty = (drawdown - max_acceptable_drawdown) * self.max_drawdown_penalty * abs(base_reward)
            reward -= drawdown_penalty * self.drawdown_weight
        
        # Withdrawal management component (NEW)
        withdrawal_metrics = self._calculate_withdrawal_metrics(risk_metrics)
        
        if withdrawal_metrics["has_withdrawals"]:
            # Reward good withdrawal handling
            withdrawal_score = withdrawal_metrics["withdrawal_handling_score"]
            
            # Base withdrawal reward based on handling score
            withdrawal_reward = withdrawal_score * 0.3 * abs(base_reward)
            
            # Additional bonus for maintaining reserve balance
            reserve_ratio = withdrawal_metrics["usd_balance"] / max(1.0, withdrawal_metrics["total_capital"])
            if reserve_ratio > 0.05:  # If maintaining >5% reserve
                withdrawal_reward += 0.1 * abs(base_reward)
                
            # Extra reward for handling emergency withdrawals well
            if withdrawal_metrics["emergency_withdrawal_count"] > 0:
                emergency_success = withdrawal_metrics["emergency_fulfilled"] / withdrawal_metrics["emergency_requested"]
                if emergency_success > 0.9:  # If handled >90% of emergency requests
                    withdrawal_reward += 0.2 * abs(base_reward)
            
            # Apply withdrawal reward with appropriate weight
            reward += withdrawal_reward * self.withdrawal_weight
        
        # Confidence and uncertainty rewards
        confidence_reward = self._calculate_confidence_reward(prediction_metrics)
        uncertainty_reward = self._calculate_uncertainty_reward(prediction_metrics)
        
        return reward + confidence_reward + uncertainty_reward


class LongRewardSystem(BaseRewardSystem):
    """
    Reward system for Long-term trading strategy.
    
    Focuses on:
    - Significant per-position gains
    - Very low trade frequency
    - Strategic entries with high expected value
    - Maximum risk-adjusted return
    - Capital preservation alongside growth
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize long-term trading reward system."""
        super().__init__(config)
        
        # Long-term specific parameters
        self.min_gain_per_holding = config.get("min_gain_per_holding", 25.0)  # Min gain per position (%)
        self.max_gain_per_holding = config.get("max_gain_per_holding", 50.0)  # Max gain per position (%)
        self.bonus_multiplier = config.get("bonus_multiplier", 1.1)  # Multiplier for hitting targets
        
        # Adjust weights for long-term strategy
        self.profit_weight = config.get("LONG_PROFIT_WEIGHT", 1.0)
        self.consistency_weight = config.get("LONG_CONSISTENCY_WEIGHT", 0.2)  # Less emphasis on consistency
        self.risk_weight = config.get("LONG_RISK_WEIGHT", 0.3)  # Lower risk sensitivity
        self.drawdown_weight = config.get("LONG_DRAWDOWN_WEIGHT", 0.3)  # Greater drawdown tolerance
        
        # Long-term specific metrics
        self.avg_trade_duration_target = config.get("LONG_AVG_TRADE_DURATION", 720)  # ~2-3 days in 5-min bars
        self.entry_quality_weight = config.get("LONG_ENTRY_QUALITY_WEIGHT", 0.5)  # High emphasis on entry quality
        self.position_sizing_weight = config.get("LONG_POSITION_SIZING_WEIGHT", 0.5)  # High emphasis on position sizing
        
        # Expectation of fewer trades with higher quality
        self.win_rate_bonus_threshold = config.get("LONG_WIN_RATE_BONUS", 0.6)  # Bonus threshold for win rate
        
        # Lower profit factor expectations (fewer, bigger trades)
        self.min_profit_factor = config.get("LONG_MIN_PROFIT_FACTOR", 1.8)
        self.target_profit_factor = config.get("LONG_TARGET_PROFIT_FACTOR", 2.5)
        
        logger.info(f"Initialized LongRewardSystem with gain targets: {self.min_gain_per_holding}%-{self.max_gain_per_holding}%")
    
    def compute_reward(self, 
                     base_reward: float, 
                     profits: List[float], 
                     losses: List[float],
                     returns: List[float],
                     closed_trades: List[Tuple],
                     episode_days: float,
                     risk_metrics: Dict[str, Any],
                     prediction_metrics: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate reward for long-term trading strategy.
        
        Long-term focuses on significant per-position gains with low frequency
        and very strategic entry/exit decisions.
        
        Args:
            base_reward: Base profit/loss from trading
            profits: List of profitable trade amounts
            losses: List of loss amounts
            returns: List of return percentages
            closed_trades: List of closed trades as (profit, pct_gain, hold_time)
            episode_days: Episode duration in days
            risk_metrics: Dictionary of risk metrics
            prediction_metrics: Dictionary of prediction metrics including confidence scores,
                                prediction accuracy, and uncertainty metrics
            
        Returns:
            Calculated reward value
        """
        # Start with base reward (total P&L)
        reward = base_reward * self.profit_weight
        
        # Calculate total P&L
        total_profit = sum(profits) if profits else 0.0
        total_loss = sum(losses) if losses else 0.0
        net_pnl = total_profit - total_loss
        
        # Convert closed_trades to expected format if needed
        formatted_trades = []
        for trade in closed_trades:
            if isinstance(trade, tuple):
                formatted_trades.append(trade)
            elif hasattr(trade, 'to_tuple'):
                formatted_trades.append(trade.to_tuple())
            else:
                profit = getattr(trade, 'profit', 0.0)
                pct_gain = getattr(trade, 'percentage_gain', 0.0)
                hold_time = getattr(trade, 'hold_time', 0)
                formatted_trades.append((profit, pct_gain, hold_time))
        
        # Skip further calculations if no trades were made
        if not formatted_trades:
            return reward
        
        # For long-term, focus on per-position gains and risk-adjusted performance
        total_trades = len(formatted_trades)
        profitable_trades = sum(1 for trade in formatted_trades if trade[0] > 0)
        
        # Calculate average percentage gain
        if total_trades > 0:
            avg_pct_gain = sum(trade[1] for trade in formatted_trades if len(trade) > 1) / total_trades
        else:
            avg_pct_gain = 0.0
        
        # Per-position gain component (primary metric for long-term)
        if avg_pct_gain < self.min_gain_per_holding:
            # Significant penalty for below minimum target
            gain_penalty = (self.min_gain_per_holding - avg_pct_gain) / self.min_gain_per_holding
            gain_penalty *= 0.5 * abs(base_reward)  # Higher penalty than medium-term
            reward -= gain_penalty
        elif avg_pct_gain > self.max_gain_per_holding:
            # Minimal penalty for excessive gains (long-term benefits from outliers)
            gain_penalty = (avg_pct_gain - self.max_gain_per_holding) / self.max_gain_per_holding
            gain_penalty *= 0.05 * abs(base_reward)  # Very small penalty
            reward -= gain_penalty
        else:
            # Significant bonus for hitting target range
            target_ratio = (avg_pct_gain - self.min_gain_per_holding) / (self.max_gain_per_holding - self.min_gain_per_holding)
            target_bonus = 0.4 * abs(base_reward) * (0.5 + 0.5 * target_ratio)
            reward += target_bonus
            
            # Apply bonus multiplier for optimal performance
            if profitable_trades >= total_trades * 0.6:  # Higher threshold than medium-term
                reward *= self.bonus_multiplier
        
        # Calculate key trading metrics
        win_rate = self._calculate_win_rate(profits, losses)
        profit_factor = self._calculate_profit_factor(profits, losses)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        max_drawdown = self._calculate_max_drawdown(returns)
        
        # Trade frequency component - long-term expects much lower frequency
        trade_count = len(formatted_trades)
        expected_trades_per_day = 0.5  # Long-term expects much fewer trades
        expected_trades = max(1, expected_trades_per_day * episode_days)
        
        if trade_count < expected_trades * 0.2:
            # Small penalty for too few trades
            trade_frequency_penalty = 0.1 * abs(base_reward) * (1 - (trade_count / (expected_trades * 0.2)))
            reward -= trade_frequency_penalty
        elif trade_count > expected_trades * 1.2:
            # Large penalty for excessive trading (long-term should be very selective)
            trade_frequency_penalty = 0.3 * abs(base_reward) * ((trade_count / (expected_trades * 1.2)) - 1)
            reward -= trade_frequency_penalty
        
        # Position sizing is critical for long-term
        position_concentration = risk_metrics.get("risk_concentration", 0.5)
        position_diversity = risk_metrics.get("position_diversity", 0.5)
        
        # Long-term strategies often benefit from some concentration
        # But not excessive concentration
        if position_concentration > 0.5:  # More than 50% in one position
            # Penalty for excessive concentration
            concentration_penalty = (position_concentration - 0.5) * 0.4 * abs(base_reward)
            reward -= concentration_penalty * self.position_sizing_weight
        elif position_diversity < 0.3:  # Not enough diversity
            # Penalty for poor diversity
            diversity_penalty = (0.3 - position_diversity) * 0.3 * abs(base_reward)
            reward -= diversity_penalty * self.position_sizing_weight
        else:
            # Bonus for balanced position sizing
            position_bonus = 0.2 * abs(base_reward)
            reward += position_bonus * self.position_sizing_weight
        
        # Hold time component - long-term requires longer duration
        if formatted_trades:
            avg_hold_time = sum(trade[2] for trade in formatted_trades if len(trade) > 2) / len(formatted_trades)
            
            # For long-term, we want much longer holds
            if avg_hold_time < self.avg_trade_duration_target * 0.5:
                # Too short for long-term strategy
                duration_penalty = 0.3 * abs(base_reward) * (1 - (avg_hold_time / (self.avg_trade_duration_target * 0.5)))
                reward -= duration_penalty
            elif avg_hold_time > self.avg_trade_duration_target * 5.0:
                # Even long-term has limits on hold time
                duration_penalty = 0.1 * abs(base_reward) * ((avg_hold_time / (self.avg_trade_duration_target * 5.0)) - 1)
                reward -= duration_penalty
            else:
                # Bonus for ideal hold time
                target_ratio = min(1.0, avg_hold_time / self.avg_trade_duration_target)
                duration_bonus = 0.2 * abs(base_reward) * target_ratio
                reward += duration_bonus
        
        # Win rate bonus for long-term (quality over quantity)
        if win_rate >= self.win_rate_bonus_threshold:
            # Significant bonus for high win rate
            win_rate_bonus = (win_rate - self.win_rate_bonus_threshold) * 0.5 * abs(base_reward)
            reward += win_rate_bonus * self.consistency_weight
        
        # Profit factor component (extremely important for long-term)
        if profit_factor >= self.target_profit_factor:
            # Substantial bonus for high profit factor
            pf_bonus = min(2.0, (profit_factor - self.target_profit_factor) / self.target_profit_factor) * 0.5 * abs(base_reward)
            reward += pf_bonus
        elif profit_factor < self.min_profit_factor:
            # Severe penalty for low profit factor
            pf_penalty = (self.min_profit_factor - profit_factor) / self.min_profit_factor * 0.6 * abs(base_reward)
            reward -= pf_penalty
        
        # Risk metrics penalties
        risk_score = risk_metrics.get("overall_risk_score", 0.5)
        risk_threshold = 0.85  # Long-term has highest risk tolerance
        
        if risk_score > risk_threshold:
            risk_penalty = (risk_score - risk_threshold) / (1 - risk_threshold) * self.max_risk_penalty * abs(base_reward)
            reward -= risk_penalty * self.risk_weight
        
        # Drawdown penalty (long-term can tolerate larger drawdowns)
        drawdown = risk_metrics.get("drawdown", max_drawdown)
        max_acceptable_drawdown = 0.20  # 20% for long-term
        
        if drawdown > max_acceptable_drawdown:
            drawdown_penalty = (drawdown - max_acceptable_drawdown) / max_acceptable_drawdown * self.max_drawdown_penalty * abs(base_reward)
            reward -= drawdown_penalty * self.drawdown_weight
        
        # Sortino ratio is critical for long-term
        if sortino_ratio > 1.0:
            sortino_bonus = min(1.0, sortino_ratio - 1.0) * 0.4 * abs(base_reward)
            reward += sortino_bonus
        elif sortino_ratio < 0.5:
            sortino_penalty = (0.5 - sortino_ratio) / 0.5 * 0.4 * abs(base_reward)
            reward -= sortino_penalty
        
        # Confidence and uncertainty rewards
        confidence_reward = self._calculate_confidence_reward(prediction_metrics)
        uncertainty_reward = self._calculate_uncertainty_reward(prediction_metrics)
        
        return reward + confidence_reward + uncertainty_reward


def create_reward_system(bucket: str, config: Dict[str, Any]) -> BaseRewardSystem:
    """
    Factory function to create the appropriate reward system based on bucket.
    
    Args:
        bucket (str): Trading bucket name ('Scalping', 'Short', 'Medium', 'Long')
        config (Dict[str, Any]): Configuration dictionary with parameters
        
    Returns:
        BaseRewardSystem: Appropriate reward system for the specified bucket
    """
    bucket = bucket.lower() if isinstance(bucket, str) else "medium"
    
    if bucket == "scalping":
        return ScalpingRewardSystem(config)
    elif bucket == "short":
        return ShortRewardSystem(config)
    elif bucket == "medium":
        return MediumRewardSystem(config)
    elif bucket == "long":
        return LongRewardSystem(config)
    else:
        # Default to medium timeframe if unknown bucket
        logger.warning(f"Unknown bucket '{bucket}', defaulting to MediumRewardSystem")
        return MediumRewardSystem(config)


def visualize_reward_system(reward_system: BaseRewardSystem, 
                            base_reward: float = 1000.0,
                            num_samples: int = 100) -> Dict[str, List[float]]:
    """
    Generate data to visualize how the reward system responds to different performance metrics.
    
    Args:
        reward_system (BaseRewardSystem): Reward system to analyze
        base_reward (float): Base reward amount to use
        num_samples (int): Number of samples to generate
        
    Returns:
        Dict[str, List[float]]: Data for each metric's impact on reward
    """
    results = {}
    
    # Generate dummy data for a successful strategy
    profits = [100.0] * 10
    losses = [50.0] * 5
    returns = [0.01] * 15
    closed_trades = [(p, p/1000.0*100, 50) for p in profits] + [(-l, -l/1000.0*100, 50) for l in losses]
    episode_days = 10.0
    risk_metrics = {
        "overall_risk_score": 0.5,
        "drawdown": 0.1,
        "exposure_percentage": 0.5,
        "risk_concentration": 0.3,
        "position_diversity": 0.7
    }
    
    # Test win rate impact
    win_rates = []
    win_rate_rewards = []
    for i in range(num_samples + 1):
        win_rate = i / num_samples
        # Adjust profits/losses to achieve the target win rate
        total_trades = 20
        num_wins = int(win_rate * total_trades)
        num_losses = total_trades - num_wins
        test_profits = [100.0] * num_wins
        test_losses = [50.0] * num_losses
        test_closed_trades = [(p, p/1000.0*100, 50) for p in test_profits] + [(-l, -l/1000.0*100, 50) for l in test_losses]
        
        # Get reward
        reward = reward_system.compute_reward(
            base_reward, test_profits, test_losses, returns, test_closed_trades, episode_days, risk_metrics, None
        )
        win_rates.append(win_rate)
        win_rate_rewards.append(reward)
    
    results["win_rate"] = (win_rates, win_rate_rewards)
    
    # Test profit factor impact
    profit_factors = []
    pf_rewards = []
    for i in range(1, num_samples + 1):  # Start at 1 to avoid division by zero
        profit_factor = i / (num_samples / 5)  # Range from 0.05 to 5.0
        
        # Fixed losses, adjust profits to achieve the target profit factor
        test_losses = [50.0] * 5
        total_loss = sum(test_losses)
        total_profit_needed = total_loss * profit_factor
        num_wins = 10
        profit_per_win = total_profit_needed / num_wins
        test_profits = [profit_per_win] * num_wins
        
        test_returns = [0.01] * (num_wins + 5)
        test_closed_trades = [(p, p/1000.0*100, 50) for p in test_profits] + [(-l, -l/1000.0*100, 50) for l in test_losses]
        
        # Get reward
        reward = reward_system.compute_reward(
            base_reward, test_profits, test_losses, test_returns, test_closed_trades, episode_days, risk_metrics, None
        )
        profit_factors.append(profit_factor)
        pf_rewards.append(reward)
    
    results["profit_factor"] = (profit_factors, pf_rewards)
    
    # Test drawdown impact
    drawdowns = []
    dd_rewards = []
    for i in range(num_samples + 1):
        drawdown = i / num_samples
        test_risk_metrics = risk_metrics.copy()
        test_risk_metrics["drawdown"] = drawdown
        
        # Get reward
        reward = reward_system.compute_reward(
            base_reward, profits, losses, returns, closed_trades, episode_days, test_risk_metrics, None
        )
        drawdowns.append(drawdown)
        dd_rewards.append(reward)
    
    results["drawdown"] = (drawdowns, dd_rewards)
    
    # Test risk score impact
    risk_scores = []
    risk_rewards = []
    for i in range(num_samples + 1):
        risk_score = i / num_samples
        test_risk_metrics = risk_metrics.copy()
        test_risk_metrics["overall_risk_score"] = risk_score
        
        # Get reward
        reward = reward_system.compute_reward(
            base_reward, profits, losses, returns, closed_trades, episode_days, test_risk_metrics, None
        )
        risk_scores.append(risk_score)
        risk_rewards.append(reward)
    
    results["risk_score"] = (risk_scores, risk_rewards)
    
    return results


def analyze_reward_systems(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze and compare all reward systems with the same input data.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        
    Returns:
        Dict[str, Any]: Analysis results for each reward system
    """
    buckets = ["scalping", "short", "medium", "long"]
    reward_systems = {bucket: create_reward_system(bucket, config) for bucket in buckets}
    
    # Generate standard test case
    base_reward = 1000.0
    profits = [100.0] * 10
    losses = [50.0] * 5
    returns = [0.01] * 15
    closed_trades = [(p, p/1000.0*100, 50) for p in profits] + [(-l, -l/1000.0*100, 50) for l in losses]
    episode_days = 10.0
    
    risk_metrics = {
        "overall_risk_score": 0.5,
        "drawdown": 0.1,
        "exposure_percentage": 0.5,
        "risk_concentration": 0.3,
        "position_diversity": 0.7
    }
    
    # Run standard comparison
    results = {}
    for bucket, reward_system in reward_systems.items():
        reward = reward_system.compute_reward(
            base_reward, profits, losses, returns, closed_trades, episode_days, risk_metrics, None
        )
        results[bucket] = {"base_case_reward": reward}
    
    # Test variations for each bucket
    variations = {
        "win_rate_50": {"profits": [100.0] * 10, "losses": [50.0] * 10},  # 50% win rate
        "win_rate_80": {"profits": [100.0] * 16, "losses": [50.0] * 4},   # 80% win rate
        "high_drawdown": {"risk_metrics": {**risk_metrics, "drawdown": 0.2}},
        "high_risk": {"risk_metrics": {**risk_metrics, "overall_risk_score": 0.8}},
        "short_trades": {"closed_trades": [(p, p/1000.0*100, 10) for p in profits] + [(-l, -l/1000.0*100, 10) for l in losses]},
        "long_trades": {"closed_trades": [(p, p/1000.0*100, 500) for p in profits] + [(-l, -l/1000.0*100, 500) for l in losses]},
    }
    
    for variation_name, variation_data in variations.items():
        variation_results = {}
        for bucket, reward_system in reward_systems.items():
            # Apply variation data or use defaults
            test_profits = variation_data.get("profits", profits)
            test_losses = variation_data.get("losses", losses)
            test_returns = variation_data.get("returns", returns)
            test_closed_trades = variation_data.get("closed_trades", closed_trades)
            test_episode_days = variation_data.get("episode_days", episode_days)
            test_risk_metrics = variation_data.get("risk_metrics", risk_metrics)
            
            reward = reward_system.compute_reward(
                base_reward, test_profits, test_losses, test_returns, 
                test_closed_trades, test_episode_days, test_risk_metrics, None
            )
            variation_results[bucket] = reward
        
        # Store variation results
        for bucket in buckets:
            results[bucket][variation_name] = variation_results[bucket]
    
    return results


# Self-test when run as a script
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pprint import pprint
    
    # Test configuration
    test_config = {
        "INITIAL_CAPITAL": 100000.0,
        "monthly_target_min": 15.0,
        "monthly_target_max": 30.0,
        "yearly_target_min": 100.0,
        "yearly_target_max": 200.0,
        "min_gain_per_holding": 25.0,
        "max_gain_per_holding": 50.0,
        "bonus_multiplier": 1.1,
    }
    
    print("Testing reward systems...")
    print("-" * 50)
    
    # Create all reward systems
    scalping_rs = create_reward_system("scalping", test_config)
    short_rs = create_reward_system("short", test_config)
    medium_rs = create_reward_system("medium", test_config)
    long_rs = create_reward_system("long", test_config)
    
    # Create sample metrics for testing
    base_reward = 1000.0
    profits = [100.0] * 10  # 10 winning trades of $100 each
    losses = [50.0] * 5     # 5 losing trades of $50 each
    
    # Simple returns (1% per trade)
    returns = [0.01] * 15
    
    # Sample closed trades with profit, percentage gain, and hold time
    closed_trades = [
        (100.0, 1.0, 20),   # $100 profit, 1% gain, 20 bars hold time
        (150.0, 1.5, 30),
        (80.0, 0.8, 15),
        (120.0, 1.2, 25),
        (90.0, 0.9, 18),
        (110.0, 1.1, 22),
        (130.0, 1.3, 28),
        (95.0, 0.95, 19),
        (105.0, 1.05, 21),
        (115.0, 1.15, 24),
        (-50.0, -0.5, 10),  # $50 loss, 0.5% loss, 10 bars hold time
        (-40.0, -0.4, 8),
        (-60.0, -0.6, 12),
        (-45.0, -0.45, 9),
        (-55.0, -0.55, 11)
    ]
    
    episode_days = 10.0  # 10-day episode
    
    # Sample risk metrics
    risk_metrics = {
        "overall_risk_score": 0.5,
        "drawdown": 0.1,
        "exposure_percentage": 0.5,
        "risk_concentration": 0.3,
        "position_diversity": 0.7
    }
    
    # Test each reward system
    print("Testing with standard metrics:")
    print(f"Scalping reward: {scalping_rs.compute_reward(base_reward, profits, losses, returns, closed_trades, episode_days, risk_metrics, None):.2f}")
    print(f"Short-term reward: {short_rs.compute_reward(base_reward, profits, losses, returns, closed_trades, episode_days, risk_metrics, None):.2f}")
    print(f"Medium-term reward: {medium_rs.compute_reward(base_reward, profits, losses, returns, closed_trades, episode_days, risk_metrics, None):.2f}")
    print(f"Long-term reward: {long_rs.compute_reward(base_reward, profits, losses, returns, closed_trades, episode_days, risk_metrics, None):.2f}")
    
    print("\nAnalyzing reward systems...")
    analysis = analyze_reward_systems(test_config)
    pprint(analysis)
    
    # Optional: Visualize reward responses if matplotlib is available
    try:
        # Visualize win rate impact
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 2, 1)
        for bucket in ["scalping", "short", "medium", "long"]:
            rs = create_reward_system(bucket, test_config)
            results = visualize_reward_system(rs)
            win_rates, rewards = results["win_rate"]
            plt.plot(win_rates, rewards, label=bucket.capitalize())
        plt.title("Win Rate Impact on Reward")
        plt.xlabel("Win Rate")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        for bucket in ["scalping", "short", "medium", "long"]:
            rs = create_reward_system(bucket, test_config)
            results = visualize_reward_system(rs)
            profit_factors, rewards = results["profit_factor"]
            plt.plot(profit_factors, rewards, label=bucket.capitalize())
        plt.title("Profit Factor Impact on Reward")
        plt.xlabel("Profit Factor")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        for bucket in ["scalping", "short", "medium", "long"]:
            rs = create_reward_system(bucket, test_config)
            results = visualize_reward_system(rs)
            drawdowns, rewards = results["drawdown"]
            plt.plot(drawdowns, rewards, label=bucket.capitalize())
        plt.title("Drawdown Impact on Reward")
        plt.xlabel("Drawdown")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        for bucket in ["scalping", "short", "medium", "long"]:
            rs = create_reward_system(bucket, test_config)
            results = visualize_reward_system(rs)
            risk_scores, rewards = results["risk_score"]
            plt.plot(risk_scores, rewards, label=bucket.capitalize())
        plt.title("Risk Score Impact on Reward")
        plt.xlabel("Risk Score")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("reward_system_analysis.png")
        print("Visualization saved to reward_system_analysis.png")
    except ImportError:
        print("Matplotlib not available - skipping visualization")
