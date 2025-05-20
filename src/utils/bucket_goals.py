#!/usr/bin/env python
"""
Bucket Goal Provider Module

This module provides a unified interface for accessing and calculating performance goals
across different bucket types (Scalping, Short, Medium, Long).
"""

import os
import json
from typing import Dict, Any, Optional, Tuple, List, Union, Callable


class BucketGoalProvider:
    """
    Provides a unified interface for managing bucket-specific performance goals.
    
    This class abstracts away the different goal types for each trading bucket,
    allowing the reward calculation to be independent of the specific bucket implementation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the goal provider with configuration.
        
        Args:
            config: Dictionary containing all configuration parameters
        """
        self.config = config
        
        # Default goal parameters if not in config
        self._defaults = {
            # Scalping goals (monthly profit targets)
            "monthly_target_min": 15.0,  # Minimum monthly profit target (%)
            "monthly_target_max": 30.0,  # Maximum monthly profit target (%)
            
            # Short goals (yearly profit targets)
            "yearly_target_min": 100.0,  # Minimum yearly profit target (%)
            "yearly_target_max": 200.0,  # Maximum yearly profit target (%)
            
            # Medium/Long goals (per-trade gain targets)
            "min_gain_per_holding": 25.0,  # Minimum gain per trade (%)
            "max_gain_per_holding": 50.0,  # Maximum gain per trade (%)
            "bonus_multiplier": 1.1,      # Bonus multiplier for meeting targets
        }
        
        # Initialize goals from config, using defaults if not present
        self._setup_goals()
    
    def _setup_goals(self):
        """Set up goals from config, applying defaults where necessary."""
        # Copy all relevant parameters from config, using defaults as fallback
        for key, default_value in self._defaults.items():
            if key not in self.config:
                self.config[key] = default_value
    
    def get_goal_parameters(self, bucket_type: str) -> Dict[str, float]:
        """
        Get goal parameters for a specific bucket type.
        
        Args:
            bucket_type: The bucket type ('Scalping', 'Short', 'Medium', or 'Long')
            
        Returns:
            Dictionary of goal parameters relevant to the bucket type
        """
        if bucket_type == "Scalping":
            return {
                "monthly_target_min": self.config.get("monthly_target_min", self._defaults["monthly_target_min"]),
                "monthly_target_max": self.config.get("monthly_target_max", self._defaults["monthly_target_max"]),
            }
        elif bucket_type == "Short":
            return {
                "yearly_target_min": self.config.get("yearly_target_min", self._defaults["yearly_target_min"]),
                "yearly_target_max": self.config.get("yearly_target_max", self._defaults["yearly_target_max"]),
            }
        elif bucket_type in ["Medium", "Long"]:
            return {
                "min_gain_per_holding": self.config.get("min_gain_per_holding", self._defaults["min_gain_per_holding"]),
                "max_gain_per_holding": self.config.get("max_gain_per_holding", self._defaults["max_gain_per_holding"]),
                "bonus_multiplier": self.config.get("bonus_multiplier", self._defaults["bonus_multiplier"]),
            }
        else:
            # Return empty dict for unknown bucket type
            return {}
    
    def calculate_goal_achievement(self, bucket_type: str, metrics: Dict[str, Any]) -> Tuple[float, str]:
        """
        Calculate how well goals were achieved for the given bucket type.
        
        Args:
            bucket_type: The bucket type ('Scalping', 'Short', 'Medium', or 'Long')
            metrics: Dictionary of performance metrics
            
        Returns:
            Tuple of (achievement_score, reason)
            achievement_score: 0.0-1.0 indicating how well goals were met
            reason: String explaining the achievement calculation
        """
        if bucket_type == "Scalping":
            return self._calculate_scalping_achievement(metrics)
        elif bucket_type == "Short":
            return self._calculate_short_achievement(metrics)
        elif bucket_type in ["Medium", "Long"]:
            return self._calculate_medium_long_achievement(metrics)
        else:
            return 0.0, "Unknown bucket type"
    
    def _calculate_scalping_achievement(self, metrics: Dict[str, Any]) -> Tuple[float, str]:
        """Calculate goal achievement for Scalping bucket."""
        # Extract relevant metrics
        monthly_profit = metrics.get("monthly_profit_estimate", 0.0)
        
        # Get goal parameters
        monthly_min = self.config.get("monthly_target_min", self._defaults["monthly_target_min"])
        monthly_max = self.config.get("monthly_target_max", self._defaults["monthly_target_max"])
        
        # Calculate achievement
        if monthly_profit <= 0:
            return 0.0, f"No profit (goal: {monthly_min}% - {monthly_max}% monthly)"
        elif monthly_profit < monthly_min:
            # Partial achievement - proportional to distance from min target
            score = monthly_profit / monthly_min
            return score, f"Below target: {monthly_profit:.1f}% (goal: {monthly_min}% - {monthly_max}% monthly)"
        elif monthly_min <= monthly_profit <= monthly_max:
            # Full achievement - within target range
            # Calculate position within range (0.8-1.0 score range)
            position = (monthly_profit - monthly_min) / (monthly_max - monthly_min)
            score = 0.8 + (0.2 * position)
            return score, f"On target: {monthly_profit:.1f}% (goal: {monthly_min}% - {monthly_max}% monthly)"
        else:
            # Above maximum - diminishing returns
            excess = (monthly_profit - monthly_max) / monthly_max
            penalty = min(0.5, excess * 0.5)  # Maximum 50% penalty for being too far above target
            score = 1.0 - penalty
            return score, f"Above target: {monthly_profit:.1f}% (goal: {monthly_min}% - {monthly_max}% monthly)"
    
    def _calculate_short_achievement(self, metrics: Dict[str, Any]) -> Tuple[float, str]:
        """Calculate goal achievement for Short bucket."""
        # Extract relevant metrics
        yearly_profit = metrics.get("yearly_profit_estimate", 0.0)
        
        # Get goal parameters
        yearly_min = self.config.get("yearly_target_min", self._defaults["yearly_target_min"])
        yearly_max = self.config.get("yearly_target_max", self._defaults["yearly_target_max"])
        
        # Calculate achievement
        if yearly_profit <= 0:
            return 0.0, f"No profit (goal: {yearly_min}% - {yearly_max}% yearly)"
        elif yearly_profit < yearly_min:
            # Partial achievement - proportional to distance from min target
            score = yearly_profit / yearly_min
            return score, f"Below target: {yearly_profit:.1f}% (goal: {yearly_min}% - {yearly_max}% yearly)"
        elif yearly_min <= yearly_profit <= yearly_max:
            # Full achievement - within target range
            # Calculate position within range (0.8-1.0 score range)
            position = (yearly_profit - yearly_min) / (yearly_max - yearly_min)
            score = 0.8 + (0.2 * position)
            return score, f"On target: {yearly_profit:.1f}% (goal: {yearly_min}% - {yearly_max}% yearly)"
        else:
            # Above maximum - diminishing returns
            excess = (yearly_profit - yearly_max) / yearly_max
            penalty = min(0.5, excess * 0.5)  # Maximum 50% penalty for being too far above target
            score = 1.0 - penalty
            return score, f"Above target: {yearly_profit:.1f}% (goal: {yearly_min}% - {yearly_max}% yearly)"
    
    def _calculate_medium_long_achievement(self, metrics: Dict[str, Any]) -> Tuple[float, str]:
        """Calculate goal achievement for Medium/Long buckets."""
        # Extract relevant metrics
        good_trades_pct = metrics.get("good_trades_pct", 0.0)
        total_trades = metrics.get("total_trades", 0)
        
        # Get goal parameters
        min_gain = self.config.get("min_gain_per_holding", self._defaults["min_gain_per_holding"])
        max_gain = self.config.get("max_gain_per_holding", self._defaults["max_gain_per_holding"])
        
        # Need minimum number of trades for meaningful assessment
        if total_trades < 5:
            return 0.5, f"Insufficient trades: {total_trades} (need ≥5 for assessment)"
            
        # Calculate achievement
        if good_trades_pct <= 0:
            return 0.0, f"No trades in target range: {min_gain}% - {max_gain}%"
        elif good_trades_pct < 20:
            # Less than 20% of trades hitting targets - partial score
            score = good_trades_pct / 20
            return score, f"Few target trades: {good_trades_pct:.1f}% (goal: trades in {min_gain}% - {max_gain}% range)"
        elif 20 <= good_trades_pct <= 50:
            # 20-50% of trades hitting targets - good score
            position = (good_trades_pct - 20) / 30
            score = 0.6 + (0.4 * position)
            return score, f"Good trade ratio: {good_trades_pct:.1f}% (goal: trades in {min_gain}% - {max_gain}% range)"
        else:
            # More than 50% of trades hitting targets - excellent
            return 1.0, f"Excellent trade ratio: {good_trades_pct:.1f}% (goal: trades in {min_gain}% - {max_gain}% range)"
    
    def get_bonus_for_bucket(self, bucket_type: str, metrics: Dict[str, Any], base_reward: float) -> float:
        """
        Calculate bonus reward based on goal achievement for a specific bucket type.
        
        Args:
            bucket_type: The bucket type ('Scalping', 'Short', 'Medium', or 'Long')
            metrics: Dictionary of performance metrics
            base_reward: Base reward value to adjust
            
        Returns:
            Float value representing the additional bonus to add to reward
        """
        # Get goal achievement score
        achievement_score, _ = self.calculate_goal_achievement(bucket_type, metrics)
        
        # Calculate bonus based on bucket type and achievement
        if bucket_type == "Scalping":
            # Scalping gets higher bonus for hitting monthly targets
            return achievement_score * base_reward * 0.3
        elif bucket_type == "Short":
            # Short gets moderate bonus for hitting yearly targets
            return achievement_score * base_reward * 0.25
        elif bucket_type in ["Medium", "Long"]:
            # Medium/Long gets bonus from trade-specific achievements
            bonus_multiplier = self.config.get("bonus_multiplier", self._defaults["bonus_multiplier"])
            return achievement_score * base_reward * 0.2 * bonus_multiplier
        else:
            return 0.0
    
    def update_config(self, new_config: Dict[str, Any]):
        """
        Update the configuration with new values.
        
        Args:
            new_config: Dictionary containing updated configuration parameters
        """
        self.config.update(new_config)
        self._setup_goals()
    
    def get_bucket_goal_description(self, bucket_type: str) -> str:
        """
        Get a human-readable description of goals for a bucket type.
        
        Args:
            bucket_type: The bucket type ('Scalping', 'Short', 'Medium', or 'Long')
            
        Returns:
            String describing the goal parameters for the bucket type
        """
        if bucket_type == "Scalping":
            min_val = self.config.get("monthly_target_min", self._defaults["monthly_target_min"])
            max_val = self.config.get("monthly_target_max", self._defaults["monthly_target_max"])
            return f"Monthly profit target: {min_val}% - {max_val}%"
        elif bucket_type == "Short":
            min_val = self.config.get("yearly_target_min", self._defaults["yearly_target_min"])
            max_val = self.config.get("yearly_target_max", self._defaults["yearly_target_max"])
            return f"Yearly profit target: {min_val}% - {max_val}%"
        elif bucket_type in ["Medium", "Long"]:
            min_val = self.config.get("min_gain_per_holding", self._defaults["min_gain_per_holding"])
            max_val = self.config.get("max_gain_per_holding", self._defaults["max_gain_per_holding"])
            bonus = self.config.get("bonus_multiplier", self._defaults["bonus_multiplier"])
            return f"Target gain per trade: {min_val}% - {max_val}% (bonus: {bonus}x)"
        else:
            return "Unknown bucket type"


# Factory function to easily create a provider instance
def create_goal_provider(config: Dict[str, Any]) -> BucketGoalProvider:
    """
    Create a BucketGoalProvider instance from configuration.
    
    Args:
        config: Dictionary containing configuration parameters
        
    Returns:
        BucketGoalProvider instance
    """
    return BucketGoalProvider(config)


if __name__ == "__main__":
    # Simple test
    test_config = {
        "monthly_target_min": 10.0,
        "monthly_target_max": 25.0,
        "yearly_target_min": 120.0,
        "yearly_target_max": 240.0,
        "min_gain_per_holding": 20.0,
        "max_gain_per_holding": 40.0,
        "bonus_multiplier": 1.2
    }
    
    provider = create_goal_provider(test_config)
    
    # Test for Scalping
    scalping_metrics = {"monthly_profit_estimate": 15.0}
    score, reason = provider.calculate_goal_achievement("Scalping", scalping_metrics)
    print(f"Scalping: {score:.2f} - {reason}")
    
    # Test for Short
    short_metrics = {"yearly_profit_estimate": 200.0}
    score, reason = provider.calculate_goal_achievement("Short", short_metrics)
    print(f"Short: {score:.2f} - {reason}")
    
    # Test for Medium
    medium_metrics = {"good_trades_pct": 35.0, "total_trades": 10}
    score, reason = provider.calculate_goal_achievement("Medium", medium_metrics)
    print(f"Medium: {score:.2f} - {reason}") 