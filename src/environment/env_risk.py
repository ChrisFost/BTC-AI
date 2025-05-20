#!/usr/bin/env python
"""
Risk management module.
This module handles risk-related functionality for the trading environment.
"""

import numpy as np
from abc import ABC, abstractmethod
import importlib
from typing import Dict, Any, Optional, List, Tuple

# Import environment utilities dynamically
try:
    env_utils_module = importlib.import_module("src.environment.env_utils")
    log = env_utils_module.log
except ImportError as e:
    print(f"Error importing environment utilities in env_risk.py: {e}")
    # Define fallback logging function
    def log(msg, level="info"):
        print(f"[{level.upper()}] {msg}")

# Define enum-like classes for risk levels and events
class RiskLevel:
    """Risk levels for the trading environment"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RiskEvent:
    """Risk event types for the trading environment"""
    EXPOSURE_LIMIT = "exposure_limit"
    DRAWDOWN_LIMIT = "drawdown_limit"
    VOLATILITY_SPIKE = "volatility_spike"
    LIQUIDITY_ISSUE = "liquidity_issue"
    CORRELATION_WARNING = "correlation_warning"
    RISK_SCORE_THRESHOLD = "risk_score_threshold"

# For backward compatibility, create a wrapper function if needed
def local_log(msg, log_file=None):
    """Wrapper for the log function to maintain backward compatibility"""
    log(msg, level="info")

class BaseRiskManager(ABC):
    """
    Abstract base class for risk management.
    
    This class defines the interface for different risk management strategies.
    """
    
    def __init__(self, config):
        """
        Initialize base risk manager with configuration parameters.
        
        Args:
            config (dict): Configuration dictionary with risk parameters.
        """
        self.config = config
        
        # Common position size limits
        self.max_btc_per_position = config.get("MAX_BTC_PER_POSITION", 10.0)
        self.max_usd_per_position = config.get("MAX_USD_PER_POSITION", 1000000.0)
        self.max_volume_percentage = config.get("MAX_VOLUME_PERCENTAGE", 0.05)
        
        # Risk thresholds
        self.risk_score_threshold = config.get("RISK_SCORE_THRESHOLD", 0.7)
        self.drawdown_limit = config.get("DRAWDOWN_LIMIT", 0.15)  # 15% max drawdown
        self.concentration_limit = config.get("CONCENTRATION_LIMIT", 0.5)  # Position concentration
        self.var_limit = config.get("VAR_LIMIT", 0.1)  # Value at Risk limit (10% of portfolio)
        
        # Additional parameters for risk calculations
        self.exposure_weight = config.get("EXPOSURE_WEIGHT", 0.15)
        self.concentration_weight = config.get("CONCENTRATION_WEIGHT", 0.15)
        self.drawdown_weight = config.get("DRAWDOWN_WEIGHT", 0.20)
        self.var_weight = config.get("VAR_WEIGHT", 0.15)
        self.correlation_weight = config.get("CORRELATION_WEIGHT", 0.10)
        self.volatility_weight = config.get("VOLATILITY_WEIGHT", 0.15)
        self.liquidity_weight = config.get("LIQUIDITY_WEIGHT", 0.10)
        
        # Extract custom parameters from risk_management if present
        if isinstance(config, dict) and "risk_management" in config:
            risk_config = config["risk_management"]
            self.max_position_size = risk_config.get("max_position_size")
            self.max_drawdown = risk_config.get("max_drawdown")
    
    def calculate_risk_metrics(self, positions, capital, current_price, returns, liquidity_history=None):
        """
        Calculate comprehensive portfolio risk metrics.
        
        Args:
            positions (list): List of current positions.
            capital (float): Current capital.
            current_price (float): Current asset price.
            returns (list): List of return percentages.
            liquidity_history (list, optional): History of liquidity values.
            
        Returns:
            dict: Dictionary of risk metrics including:
                - total_exposure: Total BTC exposure
                - exposure_percentage: Percentage of capital in positions
                - risk_concentration: Concentration of positions (Herfindahl index)
                - drawdown: Current portfolio drawdown
                - value_at_risk: Value at Risk (95% confidence)
                - correlation_risk: Risk due to position correlation
                - position_diversity: Diversity of positions
                - volatility_exposure: Exposure to market volatility
                - liquidity_risk: Risk due to market liquidity
                - max_loss: Maximum potential loss
                - overall_risk_score: Aggregate risk score (0-1)
        """
        # Default risk metrics for empty portfolio
        if not positions:
            return {
                "total_exposure": 0.0,
                "exposure_percentage": 0.0,
                "risk_concentration": 0.0,
                "drawdown": 0.0,
                "value_at_risk": 0.0,
                "correlation_risk": 0.0,
                "position_diversity": 1.0,
                "volatility_exposure": 0.0,
                "liquidity_risk": 0.0,
                "max_loss": 0.0,
                "overall_risk_score": 0.0
            }
        
        # Calculate total exposure and position values
        total_btc = 0.0
        position_values = []
        
        for position in positions:
            # Extract position data based on type
            if hasattr(position, 'size_btc'):
                position_size = position.size_btc
            elif isinstance(position, dict) and 'size_btc' in position:
                position_size = position['size_btc']
            else:
                continue
                
            # Add to total BTC and position values
            total_btc += position_size
            position_values.append(position_size * current_price)
        
        total_value = sum(position_values)
        
        # Calculate exposure percentage
        current_portfolio_value = capital + total_value
        exposure_percentage = total_value / current_portfolio_value if current_portfolio_value > 0 else 0.0
        
        # Calculate risk concentration (Herfindahl index)
        # This measures how concentrated the portfolio is (1 = single position, 1/n = perfectly diversified)
        if total_value > 0 and len(position_values) > 0:
            proportions = [value / total_value for value in position_values]
            risk_concentration = sum(p * p for p in proportions)
            
            # Ideal concentration is 1/n (equal distribution)
            ideal_concentration = 1.0 / len(positions)
            # Avoid division by zero by checking if ideal_concentration is close to 1
            if ideal_concentration >= 0.999:  # Close to 1, would cause division by very small number or zero
                concentration_risk = 0.0  # No concentration risk with single position
            else:
                concentration_risk = max(0.0, (risk_concentration - ideal_concentration) / (1.0 - ideal_concentration))
        else:
            risk_concentration = 0.0
            concentration_risk = 0.0
        
        # Position diversity (inverse of concentration)
        position_diversity = 1.0 / max(risk_concentration, 1e-8)
        position_diversity = min(position_diversity, len(positions)) if len(positions) > 0 else 1.0
        position_diversity = position_diversity / max(len(positions), 1)  # Normalize to 0-1
        
        # Calculate current drawdown
        # Track peak capital
        peak_capital = max(current_portfolio_value, capital)  # Simplified estimate
        underwater = (peak_capital - current_portfolio_value) / peak_capital if peak_capital > 0 else 0.0
        
        # Calculate Value at Risk (VaR)
        value_at_risk = 0.0
        max_loss = 0.0
        
        # Get historical returns data if available
        if len(returns) > 5:
            # Calculate 95% VaR using historical returns
            sorted_returns = sorted(returns)
            var_index = int(0.05 * len(sorted_returns))  # 5th percentile for 95% confidence
            if var_index < len(sorted_returns):
                worst_return = sorted_returns[var_index]
                value_at_risk = -total_value * worst_return  # Negative return = loss
            
            # Maximum potential loss (worst historical return)
            if sorted_returns:
                max_loss = -total_value * sorted_returns[0]  # Worst case scenario
        else:
            # Default VaR if not enough history (conservative estimate)
            value_at_risk = total_value * 0.05  # Assume 5% VaR
            max_loss = total_value * 0.10  # Assume 10% max loss
        
        # Calculate correlation risk
        correlation_risk = 0.0
        
        if len(positions) > 1:
            # In a real implementation, we'd use actual correlations between assets
            # For this simplified version, we'll use entry timings and price patterns
            entry_times = []
            
            for position in positions:
                if hasattr(position, 'entry_step'):
                    entry_times.append(position.entry_step)
                elif isinstance(position, dict) and 'entry_step' in position:
                    entry_times.append(position['entry_step'])
            
            if entry_times:
                # Calculate time diversity
                entry_time_range = max(entry_times) - min(entry_times) + 1
                time_diversity_factor = min(1.0, entry_time_range / (100.0 + 1e-8))
                
                # Assume similar entry times indicate correlation
                correlation_risk = 1.0 - time_diversity_factor
        
        # Calculate volatility exposure
        volatility_exposure = 0.0
        volatility_risk = 0.0
        
        if len(returns) > 1:
            # Calculate historical volatility
            volatility = np.std(returns) * np.sqrt(288)  # Annualized (assuming 5-min bars, 288 per day)
            
            # Volatility exposure = portfolio value * volatility
            volatility_exposure = total_value * volatility
            
            # Normalize to risk score (0-1)
            volatility_risk = min(1.0, volatility * 2)  # 50%+ volatility is max risk
        else:
            # Default to moderate volatility
            volatility_risk = 0.5
        
        # Calculate liquidity risk
        liquidity_risk = 0.0
        
        if liquidity_history:
            # Use recent liquidity metrics
            recent_liquidity = sum(liquidity_history[-min(len(liquidity_history), 10):]) / min(10, len(liquidity_history))
            # Lower liquidity = higher risk
            liquidity_risk = (1.0 - recent_liquidity) * exposure_percentage
        else:
            # Default risk if no liquidity history
            liquidity_risk = 0.2 * exposure_percentage
        
        # Calculate overall risk score (0-1 scale)
        # Weight the components based on importance
        overall_risk_score = (
            self.exposure_weight * exposure_percentage +
            self.concentration_weight * concentration_risk +
            self.drawdown_weight * min(1.0, underwater * 2) +  # Scale drawdown (>50% is critical)
            self.var_weight * min(1.0, (value_at_risk / (current_portfolio_value + 1e-8)) * 5) +  # Scale VaR (>20% is critical)
            self.correlation_weight * correlation_risk +
            self.volatility_weight * volatility_risk +
            self.liquidity_weight * liquidity_risk
        )
        
        # Ensure score is in 0-1 range
        overall_risk_score = max(0.0, min(1.0, overall_risk_score))
        
        # Create and return risk metrics dictionary
        risk_metrics = {
            "total_exposure": total_btc,
            "exposure_percentage": exposure_percentage,
            "risk_concentration": risk_concentration,
            "concentration_risk": concentration_risk,
            "drawdown": underwater,
            "value_at_risk": value_at_risk,
            "var_percentage": value_at_risk / (current_portfolio_value + 1e-8),
            "correlation_risk": correlation_risk,
            "position_diversity": position_diversity,
            "volatility_exposure": volatility_exposure,
            "volatility_risk": volatility_risk,
            "liquidity_risk": liquidity_risk,
            "max_loss": max_loss,
            "overall_risk_score": overall_risk_score
        }
        
        return risk_metrics
    
    def calculate_position_size(self, price, daily_volume, risk_metrics):
        """
        Calculate maximum position size based on base constraints,
        without considering portfolio risk.
        
        Args:
            price (float): Current asset price.
            daily_volume (float): Daily trading volume in USD.
            risk_metrics (dict): Risk metrics from calculate_risk_metrics.
            
        Returns:
            float: Maximum allowed position size in BTC.
        """
        # Base size calculation without limits
        # The actual limits will be applied by the risk managers
        return self.max_btc_per_position
    
    def calculate_risk_adjusted_size(self, price, daily_volume, direction, risk_metrics, position_count,
                                   prediction_mean=None, prediction_std=None, confidence_score=None):
        """
        Calculate position size with basic risk adjustments.
        This is a fallback implementation for the base class.
        
        Args:
            price (float): Current price.
            daily_volume (float): Daily trading volume.
            direction (int): Trade direction (1 for long, -1 for short).
            risk_metrics (dict): Risk metrics dictionary.
            position_count (int): Current number of open positions.
            prediction_mean (float, optional): Mean of the price prediction. Defaults to None.
            prediction_std (float, optional): Standard deviation of the prediction. Defaults to None.
            confidence_score (float, optional): Confidence score from the model. Defaults to None.
            
        Returns:
            float: Position size in BTC.
        """
        # Get base size
        base_size = self.calculate_position_size(price, daily_volume, risk_metrics)
        
        # Apply basic risk adjustments based on risk score
        risk_score = risk_metrics.get("overall_risk_score", 0.5)
        risk_factor = max(0.2, 1.0 - risk_score)
        
        adjusted_size = base_size * risk_factor
        
        # Apply prediction adjustment if available
        if prediction_mean is not None and prediction_std is not None:
            adjusted_size = self.adjust_position_for_uncertainty(adjusted_size, prediction_mean, prediction_std, confidence_score)
            
        return adjusted_size

    def adjust_position_for_uncertainty(self, base_size, prediction_mean, prediction_std, confidence=None, trend_strength=None):
        """
        Adjust position size based on prediction uncertainty.
        
        Args:
            base_size (float): Base position size determined by risk rules.
            prediction_mean (float): Mean of the price prediction.
            prediction_std (float): Standard deviation of the price prediction.
            confidence (float, optional): Confidence score from the model (0-1). Defaults to None.
            trend_strength (float, optional): Trend strength indicator (-1 to 1). Defaults to None.
            
        Returns:
            float: Adjusted position size in BTC.
        """
        # Default adjustment factor
        adjustment_factor = 1.0
        
        # Calculate coefficient of variation (higher values indicate more uncertainty)
        if prediction_mean != 0 and prediction_std is not None:
            # CV = std/mean (but use absolute mean since mean can be negative)
            cv = prediction_std / abs(prediction_mean)
            
            # Define CV thresholds based on trading style
            if isinstance(self, ScalpingRiskManager):
                # Scalping requires higher precision
                max_cv = 0.1
            elif isinstance(self, ShortTermRiskManager):
                max_cv = 0.2
            elif isinstance(self, MediumTermRiskManager):
                max_cv = 0.3
            else:  # Long term is more tolerant to noise
                max_cv = 0.4
                
            # Scale position down as CV increases
            if cv > max_cv:
                # Linear reduction up to 80% reduction at 3x the max CV
                reduction_factor = min(0.8, 0.8 * (cv - max_cv) / (2 * max_cv))
                adjustment_factor *= (1.0 - reduction_factor)
                
                # Log high uncertainty adjustment
                if reduction_factor > 0.4:
                    log(f"High uncertainty detected (CV={cv:.3f}). Reducing position by {reduction_factor*100:.1f}%")
        
        # Adjust based on model confidence (if provided)
        if confidence is not None:
            # Minimum confidence threshold from config
            min_confidence = self.config.get("MIN_CONFIDENCE_THRESHOLD", 0.4)
            
            # Scale the position size linearly based on confidence
            # At min_confidence, factor = 0.4, at confidence=1.0, factor = 1.0
            if confidence < min_confidence:
                # Confidence below minimum threshold, reduce to 20% of base size
                confidence_factor = 0.2
            else:
                confidence_factor = 0.4 + 0.6 * ((confidence - min_confidence) / (1.0 - min_confidence))
                
            adjustment_factor *= confidence_factor
            
            # Log confidence-based adjustment
            if confidence_factor < 0.7:
                log(f"Low prediction confidence ({confidence:.2f}). Adjusted by factor: {confidence_factor:.2f}")
        
        # Adjust based on trend strength (if provided)
        if trend_strength is not None:
            # Weaker trends (closer to 0) get smaller positions
            # Strong trends (closer to -1 or 1) get larger positions
            trend_magnitude = abs(trend_strength)
            
            # Scale by trend magnitude with a minimum of 0.5 at zero trend
            trend_factor = 0.5 + 0.5 * trend_magnitude
            adjustment_factor *= trend_factor
            
            # Log trend-based adjustment
            if trend_magnitude < 0.3:
                log(f"Weak trend detected ({trend_strength:.2f}). Adjusted by factor: {trend_factor:.2f}")
        
        # For aggressive strategies, optionally boost position size for high-confidence predictions
        if isinstance(self, ScalpingRiskManager) and confidence is not None and confidence > 0.9:
            # Boost size for very high confidence scalping trades
            # But only if uncertainty is also low
            if prediction_std is not None and prediction_mean != 0:
                cv = prediction_std / abs(prediction_mean)
                if cv < 0.05:  # Very low uncertainty
                    boost_factor = 1.2  # Boost by 20%
                    adjustment_factor *= boost_factor
                    log(f"High confidence scalp with low uncertainty. Boosting position by {(boost_factor-1)*100:.0f}%")
        
        # Ensure we don't scale below a minimum threshold
        min_adjustment = 0.1  # Never reduce below 10% of base size
        adjustment_factor = max(min_adjustment, adjustment_factor)
        
        # Apply the adjustment
        adjusted_size = base_size * adjustment_factor
        
        # Log significant adjustments
        if adjustment_factor < 0.5 or adjustment_factor > 1.1:
            log(f"Position adjusted for uncertainty: {adjustment_factor:.2f}x (base={base_size:.4f}, adjusted={adjusted_size:.4f})")
            
        return adjusted_size

    def calculate_confidence_adjusted_risk(self, risk_metrics, prediction_means, prediction_stds, confidence_scores, trend_strength=None):
        """
        Adjust risk metrics based on prediction confidence and uncertainty.
        
        Args:
            risk_metrics (dict): Current risk metrics.
            prediction_means (dict): Dictionary of mean predictions for different horizons.
            prediction_stds (dict): Dictionary of standard deviations for predictions.
            confidence_scores (dict): Dictionary of confidence scores from the model.
            trend_strength (float, optional): Trend strength indicator (-1 to 1).
            
        Returns:
            dict: Adjusted risk metrics.
        """
        # Create a copy of the original metrics
        adjusted_metrics = risk_metrics.copy()
        
        # Skip if no prediction data available
        if not prediction_stds or len(prediction_stds) == 0:
            return adjusted_metrics
            
        # Extract all standard deviation values
        all_stds = []
        all_means = []
        all_conf_scores = []
        
        # Process dictionaries of predictions
        for horizon in prediction_stds:
            if horizon in prediction_means and horizon in confidence_scores:
                std_val = prediction_stds[horizon]
                mean_val = prediction_means[horizon]
                conf_val = confidence_scores[horizon]
                
                # Handle tensor or numpy array inputs
                if hasattr(std_val, 'item'):
                    std_val = std_val.item()
                elif isinstance(std_val, np.ndarray):
                    std_val = float(std_val.flatten()[0])
                    
                if hasattr(mean_val, 'item'):
                    mean_val = mean_val.item()
                elif isinstance(mean_val, np.ndarray):
                    mean_val = float(mean_val.flatten()[0])
                    
                if hasattr(conf_val, 'item'):
                    conf_val = conf_val.item()
                elif isinstance(conf_val, np.ndarray):
                    conf_val = float(conf_val.flatten()[0])
                
                all_stds.append(std_val)
                all_means.append(mean_val)
                all_conf_scores.append(conf_val)
        
        # If we couldn't extract any values, return original metrics
        if not all_stds:
            return adjusted_metrics
            
        # Calculate average uncertainty
        avg_std = sum(all_stds) / len(all_stds)
        
        # Calculate coefficient of variation for each prediction
        cvs = []
        for mean, std in zip(all_means, all_stds):
            if abs(mean) > 1e-8:
                cvs.append(abs(std / mean))
        else:
                cvs.append(1.0)
        
        avg_cv = sum(cvs) / len(cvs)
        
        # Calculate average confidence
        avg_confidence = sum(all_conf_scores) / len(all_conf_scores)
        
        # Combine trend strength if available
        if trend_strength is not None:
            # Stronger trends reduce uncertainty impact
            trend_modifier = 1.0 - (abs(trend_strength) * 0.3)
        else:
            trend_modifier = 1.0
            
        # Adjust risk metrics based on uncertainty, confidence, and trend
        # Higher uncertainty or lower confidence increases perceived risk
        uncertainty_factor = (avg_cv * (1.0 - avg_confidence) * trend_modifier)
        
        # Apply adjustments to risk components
        adjusted_metrics["value_at_risk"] *= (1.0 + uncertainty_factor)
        adjusted_metrics["max_loss"] *= (1.0 + uncertainty_factor)
        adjusted_metrics["volatility_exposure"] *= (1.0 + uncertainty_factor * 0.5)
        
        # Adjust overall risk score
        uncertainty_impact = uncertainty_factor / 3.0  # Scale down for overall score
        adjusted_metrics["overall_risk_score"] = min(1.0, adjusted_metrics["overall_risk_score"] + uncertainty_impact)
        
        # Add uncertainty metrics for tracking and debugging
        adjusted_metrics["prediction_uncertainty"] = avg_cv
        adjusted_metrics["prediction_confidence"] = avg_confidence
        adjusted_metrics["uncertainty_impact"] = uncertainty_impact
        adjusted_metrics["trend_strength"] = trend_strength if trend_strength is not None else 0.0
        
        return adjusted_metrics

# Define RiskManager as an alias for BaseRiskManager to fix import issues
RiskManager = BaseRiskManager

class ScalpingRiskManager(BaseRiskManager):
    """
    Risk manager for scalping strategies.
    
    Implements aggressive but short-term focused risk management,
    optimized for high-frequency trading with smaller position sizes.
    """
    
    def __init__(self, config):
        """Initialize scalping risk manager with scalping-specific parameters."""
        super().__init__(config)
        # Specific parameters for scalping
        self.max_position_percentage = config.get("SCALPING_MAX_POSITION_PCT", 0.05)  # 5% max per position
        self.target_profit_factor = config.get("SCALPING_PROFIT_FACTOR", 1.5)  # Profit-to-risk ratio
        self.slippage_buffer = config.get("SCALPING_SLIPPAGE_BUFFER", 1.2)  # Buffer for slippage in high-frequency
        
        # Extract custom parameters from risk_management if present
        if isinstance(config, dict) and "risk_management" in config:
            risk_config = config["risk_management"]
            self.max_position_size = risk_config.get("max_position_size")
            self.max_drawdown = risk_config.get("max_drawdown")
    
    def calculate_risk_adjusted_size(self, price, daily_volume, direction, risk_metrics, position_count,
                                   prediction_mean=None, prediction_std=None, confidence_score=None):
        """
        Calculate risk-adjusted position size for scalping strategy.
        
        Args:
            price (float): Current price.
            daily_volume (float): Daily trading volume.
            direction (int): Trade direction (1 for long, -1 for short).
            risk_metrics (dict): Risk metrics dictionary.
            position_count (int): Current number of open positions.
            prediction_mean (float, optional): Mean of the price prediction. Defaults to None.
            prediction_std (float, optional): Standard deviation of the prediction. Defaults to None.
            confidence_score (float, optional): Confidence score from the model. Defaults to None.
            
        Returns:
            float: Position size in BTC.
        """
        # Base calculation using parent class
        base_size = super().calculate_position_size(price, daily_volume, risk_metrics)
        
        # Apply position limits
        btc_limit = self.max_btc_per_position
        usd_limit = self.max_usd_per_position / price if price > 0 else btc_limit
        volume_limit = (daily_volume * self.max_volume_percentage) / price if price > 0 else btc_limit
        
        # Get most restrictive limit
        base_size = min(base_size, btc_limit, usd_limit, volume_limit)
        
        # Scalping-specific adjustments
        
        # Adjust for risk score
        risk_score = risk_metrics.get("overall_risk_score", 0.5)
        risk_factor = max(0.2, 1.0 - (risk_score * 2))  # Reduce size as risk increases
        
        # Adjust for capital
        if "exposure_percentage" in risk_metrics:
            exposure_pct = risk_metrics["exposure_percentage"]
            # Reduce size as exposure increases
            exposure_factor = max(0.2, 1.0 - (exposure_pct / self.max_position_percentage))
        else:
            exposure_factor = 1.0
        
        # Adjust for position concentration
        if "risk_concentration" in risk_metrics and position_count > 0:
            current_concentration = risk_metrics["risk_concentration"]
            ideal_concentration = 1.0 / max(1, position_count)
            
            if current_concentration > ideal_concentration + 0.2:
                # Too concentrated - reduce size
                concentration_factor = max(0.6, 1.0 - (current_concentration - (ideal_concentration + 0.2)) * 2)
            elif current_concentration < ideal_concentration - 0.1 and position_count > 0:
                # Too diversified - encourage concentration
                concentration_factor = 1.1  # Bonus for more concentration
            else:
                # Good concentration level
                concentration_factor = 1.0
        else:
            concentration_factor = 1.0
        
        # Adjust for market conditions (volatility and liquidity)
        if "volatility_exposure" in risk_metrics and "liquidity_risk" in risk_metrics:
            volatility = risk_metrics["volatility_exposure"]
            liquidity_risk = risk_metrics["liquidity_risk"]
            
            # Scalping strategies are more sensitive to volatility and liquidity
            market_factor = max(0.3, 1.0 - (volatility * 0.7 + liquidity_risk * 0.5))
        else:
            market_factor = 1.0
            
        # Adjust for uncertainty in prediction if available
        if prediction_mean is not None and prediction_std is not None:
            size_with_base_factors = base_size * risk_factor * exposure_factor * concentration_factor * market_factor
            final_size = self.adjust_position_for_uncertainty(size_with_base_factors, prediction_mean, prediction_std)
        else:
            final_size = base_size * risk_factor * exposure_factor * concentration_factor * market_factor
            
        # Apply additional confidence scaling if available
        if confidence_score is not None:
            confidence_factor = 0.5 + (confidence_score * 0.5)  # Scale from 0.5 to 1.0
            final_size *= confidence_factor
            
        # Ensure final size is within limits
        final_size = min(final_size, btc_limit, usd_limit, volume_limit)
        
        return final_size


class ShortRiskManager(BaseRiskManager):
    """
    Risk manager for short-term strategies.
    
    Focuses on moderate risk with medium-term hold periods,
    balancing exposure and profit potential.
    """
    
    def __init__(self, config):
        """Initialize short-term risk manager with specific parameters."""
        super().__init__(config)
        # Specific parameters for short-term trading
        self.max_position_percentage = config.get("SHORT_MAX_POSITION_PCT", 0.1)  # 10% max per position
        self.target_profit_factor = config.get("SHORT_PROFIT_FACTOR", 2.0)  # Profit-to-risk ratio
        self.dynamic_stop_factor = config.get("SHORT_DYNAMIC_STOP", 1.5)  # For dynamic stop loss calculation
    
    def calculate_risk_adjusted_size(self, price, daily_volume, direction, risk_metrics, position_count,
                                   prediction_mean=None, prediction_std=None, confidence_score=None):
        """
        Calculate risk-adjusted position size for short-term strategy.
        
        Args:
            price (float): Current price.
            daily_volume (float): Daily trading volume.
            direction (int): Trade direction (1 for long, -1 for short).
            risk_metrics (dict): Risk metrics dictionary.
            position_count (int): Current number of open positions.
            prediction_mean (float, optional): Mean of the price prediction. Defaults to None.
            prediction_std (float, optional): Standard deviation of the prediction. Defaults to None.
            confidence_score (float, optional): Confidence score from the model. Defaults to None.
            
        Returns:
            float: Position size in BTC.
        """
        # Base calculation using parent class
        base_size = super().calculate_position_size(price, daily_volume, risk_metrics)
        
        # Apply position limits
        btc_limit = self.max_btc_per_position
        usd_limit = self.max_usd_per_position / price if price > 0 else btc_limit
        volume_limit = (daily_volume * self.max_volume_percentage) / price if price > 0 else btc_limit
        
        # Get most restrictive limit
        base_size = min(base_size, btc_limit, usd_limit, volume_limit)
        
        # Short-term specific adjustments
        
        # Adjust for risk score
        risk_score = risk_metrics.get("overall_risk_score", 0.5)
        risk_factor = max(0.3, 1.0 - risk_score * 1.5)  # Moderate reduction as risk increases
        
        # Adjust for capital exposure (more conservative than scalping)
        if "exposure_percentage" in risk_metrics:
            exposure_pct = risk_metrics["exposure_percentage"]
            exposure_factor = max(0.4, 1.0 - (exposure_pct / self.max_position_percentage) * 0.8)
        else:
            exposure_factor = 1.0
        
        # Adjust for position concentration
        if "risk_concentration" in risk_metrics and position_count > 0:
            current_concentration = risk_metrics["risk_concentration"]
            ideal_concentration = 1.0 / max(1, position_count)
            
            if current_concentration > ideal_concentration + 0.15:
                # Too concentrated - reduce size
                concentration_factor = max(0.7, 1.0 - (current_concentration - (ideal_concentration + 0.15)) * 1.5)
            elif current_concentration < ideal_concentration - 0.1:
                # Too diversified - encourage concentration
                concentration_factor = 1.05  # Slight bonus for more concentration
            else:
                # Good concentration level - short-term strategies typically handle more equal weights
                concentration_factor = 1.0
        else:
            concentration_factor = 1.0
        
        # Adjust for market conditions
        if "volatility_exposure" in risk_metrics and "liquidity_risk" in risk_metrics:
            volatility = risk_metrics["volatility_exposure"]
            liquidity_risk = risk_metrics["liquidity_risk"]
            
            # Short-term strategies balance volatility vs opportunity
            market_factor = max(0.5, 1.0 - (volatility * 0.5 + liquidity_risk * 0.3))
        else:
            market_factor = 1.0
        
        # Factor in current drawdown if available
        if "drawdown" in risk_metrics:
            current_drawdown = risk_metrics["drawdown"]
            drawdown_factor = max(0.6, 1.0 - (current_drawdown / self.max_drawdown_tolerance) * 1.2)
        else:
            drawdown_factor = 1.0
        
        # Adjust for uncertainty in prediction if available
        if prediction_mean is not None and prediction_std is not None:
            size_with_base_factors = base_size * risk_factor * exposure_factor * concentration_factor * market_factor
            final_size = self.adjust_position_for_uncertainty(size_with_base_factors, prediction_mean, prediction_std)
        else:
            final_size = base_size * risk_factor * exposure_factor * concentration_factor * market_factor
        
        # Apply confidence factor if available
        if confidence_score is not None:
            confidence_factor = 0.4 + (confidence_score * 0.6)  # Scale from 0.4 to 1.0 (less aggressive than scalping)
            final_size *= confidence_factor
        
        # Ensure final size is within limits
        final_size = min(final_size, btc_limit, usd_limit, volume_limit)
        
        return final_size


class MediumRiskManager(BaseRiskManager):
    """
    Risk manager for medium-term strategies.
    
    Focuses on balanced risk with weeks-to-months hold periods,
    optimizing for sustainable gains and moderate drawdowns.
    """
    
    def __init__(self, config):
        """Initialize medium-term risk manager with specific parameters."""
        super().__init__(config)
        # Specific parameters for medium-term trading
        self.max_position_percentage = config.get("MEDIUM_MAX_POSITION_PCT", 0.15)  # 15% max per position
        self.target_profit_factor = config.get("MEDIUM_PROFIT_FACTOR", 2.5)  # Profit-to-risk ratio
        self.max_drawdown_tolerance = config.get("MEDIUM_MAX_DRAWDOWN", 0.12)  # 12% max drawdown tolerance
    
    def calculate_risk_adjusted_size(self, price, daily_volume, direction, risk_metrics, position_count,
                                   prediction_mean=None, prediction_std=None, confidence_score=None):
        """
        Calculate risk-adjusted position size for medium-term strategy.
        
        Args:
            price (float): Current price.
            daily_volume (float): Daily trading volume.
            direction (int): Trade direction (1 for long, -1 for short).
            risk_metrics (dict): Risk metrics dictionary.
            position_count (int): Current number of open positions.
            prediction_mean (float, optional): Mean of the price prediction. Defaults to None.
            prediction_std (float, optional): Standard deviation of the prediction. Defaults to None.
            confidence_score (float, optional): Confidence score from the model. Defaults to None.
            
        Returns:
            float: Position size in BTC.
        """
        # Base calculation using parent class
        base_size = super().calculate_position_size(price, daily_volume, risk_metrics)
        
        # Apply position limits
        btc_limit = self.max_btc_per_position
        usd_limit = self.max_usd_per_position / price if price > 0 else btc_limit
        volume_limit = (daily_volume * self.max_volume_percentage) / price if price > 0 else btc_limit
        
        # Get most restrictive limit
        base_size = min(base_size, btc_limit, usd_limit, volume_limit)
        
        # Medium-term specific adjustments
        
        # Adjust for risk score with medium risk tolerance
        risk_score = risk_metrics.get("overall_risk_score", 0.5)
        risk_factor = max(0.4, 1.0 - risk_score * 1.2)  # Balanced reduction as risk increases
        
        # Adjust for capital exposure (more tolerance than short-term)
        if "exposure_percentage" in risk_metrics:
            exposure_pct = risk_metrics["exposure_percentage"]
            exposure_factor = max(0.5, 1.0 - (exposure_pct / self.max_position_percentage) * 0.7)
        else:
            exposure_factor = 1.0
            
        # Adjust for drawdown (medium-term strategies care more about drawdown)
        if "drawdown" in risk_metrics:
            current_drawdown = risk_metrics["drawdown"]
            drawdown_factor = max(0.7, 1.0 - (current_drawdown / self.max_drawdown_tolerance) * 1.5)
        else:
            drawdown_factor = 1.0
            
        # Adjust for position concentration
        if "risk_concentration" in risk_metrics and position_count > 0:
            current_concentration = risk_metrics["risk_concentration"]
            ideal_concentration = 1.0 / max(1, position_count)
            
            if current_concentration > ideal_concentration + 0.1:
                # Too concentrated - reduce size
                concentration_factor = max(0.8, 1.0 - (current_concentration - (ideal_concentration + 0.1)) * 1.2)
            elif current_concentration < ideal_concentration - 0.05:
                # Too diversified - encourage concentration
                concentration_factor = 1.02  # Very slight bonus for more concentration
            else:
                # Good concentration level
                concentration_factor = 1.0
        else:
            concentration_factor = 1.0
            
        # Adjust for market conditions
        if "volatility_exposure" in risk_metrics and "liquidity_risk" in risk_metrics:
            volatility = risk_metrics["volatility_exposure"]
            liquidity_risk = risk_metrics["liquidity_risk"]
            
            # Medium-term strategies are more tolerant of volatility
            market_factor = max(0.6, 1.0 - (volatility * 0.4 + liquidity_risk * 0.2))
        else:
            market_factor = 1.0
            
        # Adjust for uncertainty in prediction if available
        if prediction_mean is not None and prediction_std is not None:
            size_with_base_factors = base_size * risk_factor * exposure_factor * concentration_factor * market_factor
            final_size = self.adjust_position_for_uncertainty(size_with_base_factors, prediction_mean, prediction_std)
        else:
            final_size = base_size * risk_factor * exposure_factor * concentration_factor * market_factor
            
        # Apply confidence factor if available
        if confidence_score is not None:
            confidence_factor = 0.3 + (confidence_score * 0.7)  # Scale from 0.3 to 1.0 (more conservative)
            final_size *= confidence_factor
            
        # Ensure final size is within limits
        final_size = min(final_size, btc_limit, usd_limit, volume_limit)
        
        return final_size


class LongRiskManager(BaseRiskManager):
    """
    Risk manager for long-term strategies.
    
    Focuses on conservative risk with months-to-years hold periods,
    prioritizing capital preservation and steady growth.
    """
    
    def __init__(self, config):
        """Initialize long-term risk manager with specific parameters."""
        super().__init__(config)
        # Specific parameters for long-term trading
        self.max_position_percentage = config.get("LONG_MAX_POSITION_PCT", 0.2)  # 20% max per position
        self.target_profit_factor = config.get("LONG_PROFIT_FACTOR", 3.0)  # Profit-to-risk ratio
        self.max_drawdown_tolerance = config.get("LONG_MAX_DRAWDOWN", 0.2)  # 20% max drawdown tolerance
        self.trend_factor_weight = config.get("LONG_TREND_WEIGHT", 0.6)  # Weight for trend factors
    
    def calculate_risk_adjusted_size(self, price, daily_volume, direction, risk_metrics, position_count,
                                   prediction_mean=None, prediction_std=None, confidence_score=None, trend_strength=None):
        """
        Calculate risk-adjusted position size for long-term strategy.
        
        Args:
            price (float): Current price.
            daily_volume (float): Daily trading volume.
            direction (int): Trade direction (1 for long, -1 for short).
            risk_metrics (dict): Risk metrics dictionary.
            position_count (int): Current number of open positions.
            prediction_mean (float, optional): Mean of the price prediction. Defaults to None.
            prediction_std (float, optional): Standard deviation of the prediction. Defaults to None.
            confidence_score (float, optional): Confidence score from the model. Defaults to None.
            trend_strength (float, optional): Trend strength indicator (-1 to 1). Defaults to None.
            
        Returns:
            float: Position size in BTC.
        """
        # Base calculation using parent class
        base_size = super().calculate_position_size(price, daily_volume, risk_metrics)
        
        # Apply position limits
        btc_limit = self.max_btc_per_position
        usd_limit = self.max_usd_per_position / price if price > 0 else btc_limit
        volume_limit = (daily_volume * self.max_volume_percentage) / price if price > 0 else btc_limit
        
        # Get most restrictive limit
        base_size = min(base_size, btc_limit, usd_limit, volume_limit)
        
        # Long-term specific adjustments
        
        # Adjust for risk score with high risk tolerance
        risk_score = risk_metrics.get("overall_risk_score", 0.5)
        risk_factor = max(0.5, 1.0 - risk_score)  # Less reduction as risk increases
        
        # Adjust for capital exposure (high tolerance for long-term)
        if "exposure_percentage" in risk_metrics:
            exposure_pct = risk_metrics["exposure_percentage"]
            exposure_factor = max(0.6, 1.0 - (exposure_pct / self.max_position_percentage) * 0.6)
        else:
            exposure_factor = 1.0
            
        # Adjust for drawdown (long-term strategies are more tolerant)
        if "drawdown" in risk_metrics:
            current_drawdown = risk_metrics["drawdown"]
            drawdown_factor = max(0.8, 1.0 - (current_drawdown / self.max_drawdown_tolerance) * 1.8)
        else:
            drawdown_factor = 1.0
            
        # Adjust for position concentration
        if "risk_concentration" in risk_metrics and position_count > 0:
            current_concentration = risk_metrics["risk_concentration"]
            ideal_concentration = 1.0 / max(1, position_count)
            
            if current_concentration > ideal_concentration + 0.05:
                # Too concentrated - reduce size
                concentration_factor = max(0.9, 1.0 - (current_concentration - (ideal_concentration + 0.05)) * 1.0)
            elif current_concentration < ideal_concentration - 0.02:
                # Too diversified - encourage concentration
                concentration_factor = 1.01  # Very slight bonus for more concentration
            else:
                # Good concentration level
                concentration_factor = 1.0
        else:
            concentration_factor = 1.0
            
        # Adjust for market conditions
        if "volatility_exposure" in risk_metrics and "liquidity_risk" in risk_metrics:
            volatility = risk_metrics["volatility_exposure"]
            liquidity_risk = risk_metrics["liquidity_risk"]
            
            # Long-term strategies are very tolerant of volatility
            market_factor = max(0.7, 1.0 - (volatility * 0.3 + liquidity_risk * 0.1))
        else:
            market_factor = 1.0
            
        # Adjust for uncertainty in prediction if available
        if prediction_mean is not None and prediction_std is not None:
            size_with_base_factors = base_size * risk_factor * exposure_factor * concentration_factor * market_factor
            final_size = self.adjust_position_for_uncertainty(size_with_base_factors, prediction_mean, prediction_std)
        else:
            final_size = base_size * risk_factor * exposure_factor * concentration_factor * market_factor
            
        # Apply confidence factor if available
        if confidence_score is not None:
            confidence_factor = 0.2 + (confidence_score * 0.8)  # Scale from 0.2 to 1.0 (very conservative)
            final_size *= confidence_factor
            
        # Adjust for trend strength if available
        if trend_strength is not None:
            # Long-term strategies care more about trend strength
            trend_factor = 0.5 + (abs(trend_strength) * 0.5)  # Scale from 0.5 to 1.0
            final_size *= trend_factor
            
        # Ensure final size is within limits
        final_size = min(final_size, btc_limit, usd_limit, volume_limit)
        
        return final_size


def create_risk_manager(bucket, config):
    """
    Factory function to create the appropriate risk manager for a trading bucket.
        
        Args:
        bucket (str): Trading bucket ('Scalping', 'Short', 'Medium', or 'Long').
        config (dict): Configuration dictionary with risk parameters.
            
        Returns:
        BaseRiskManager: Risk manager instance for the specified bucket.
    """
    # Convert bucket to title case to make it case-insensitive
    if bucket:
        bucket = bucket.title()
        
    if bucket == 'Scalping':
        return ScalpingRiskManager(config)
    elif bucket == 'Short':
        return ShortRiskManager(config)
    elif bucket == 'Medium':
        return MediumRiskManager(config)
    elif bucket == 'Long':
        return LongRiskManager(config)
    else:
        log(f"Unknown bucket: {bucket}. Using default risk manager.")
        # Use BaseRiskManager as fallback
        return BaseRiskManager(config)


# ======================================================================
# IMPORTANT: RiskAnalyzer is ONLY FOR TESTING PURPOSES
# This class is not intended for production use and only exists to satisfy
# test code requirements in the self-test section below
# ======================================================================
class RiskAnalyzer:
    """
    FOR TESTING ONLY: Risk analyzer for tracking and analyzing risk metrics over time.
    This class should not be used in production code.
    """
    
    def __init__(self, risk_manager):
        """Initialize risk analyzer with a risk manager."""
        self.risk_manager = risk_manager
        self.risk_snapshots = []
    
    def add_risk_snapshot(self, metrics):
        """Add a risk metrics snapshot to the history."""
        self.risk_snapshots.append(metrics.copy())
    
    def calculate_risk_trajectory(self):
        """Calculate the trajectory of risk over time."""
        if not self.risk_snapshots:
            return {"trend": "stable", "score": 0.0}
        
        # Extract overall risk scores
        scores = [snapshot.get("overall_risk_score", 0.0) for snapshot in self.risk_snapshots]
        
        # Calculate trend
        if len(scores) > 1:
            avg_change = (scores[-1] - scores[0]) / len(scores)
            if avg_change > 0.05:
                trend = "increasing"
            elif avg_change < -0.05:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "stable"
            
        return {
            "trend": trend,
            "score": scores[-1] if scores else 0.0,
            "change": avg_change if len(scores) > 1 else 0.0
        }
    
    def analyze_risk_components(self, metrics):
        """Analyze which components contribute most to the overall risk."""
        components = {}
        
        # Extract the key risk components
        risk_keys = [
            "exposure_percentage", "risk_concentration", "drawdown", 
            "correlation_risk", "volatility_risk", "liquidity_risk"
        ]
        
        # Calculate contribution of each component
        for key in risk_keys:
            if key in metrics:
                components[key] = metrics[key]
        
        return components
    
    def perform_stress_test(self, positions, capital, stress_factors):
        """Perform a stress test with given factors."""
        # Apply stress factors to positions and recalculate metrics
        price_change = stress_factors.get("price_decrease", 0.0)
        volatility_increase = stress_factors.get("volatility_increase", 0.0)
        
        # Clone positions with stress applied
        stressed_positions = []
        for position in positions:
            # Create a copy with reduced price
            if hasattr(position, 'entry_price'):
                position.entry_price *= (1.0 - price_change)
                stressed_positions.append(position)
        
        # Get base metrics
        current_price = 40000 * (1.0 - price_change)  # Apply price stress
        returns = [-0.01, 0.02, -0.03, 0.01, 0.02]
        
        # Apply volatility stress to returns
        stressed_returns = [r * (1.0 + volatility_increase) for r in returns]
        
        # Calculate risk metrics under stress
        stressed_metrics = self.risk_manager.calculate_risk_metrics(
            stressed_positions, capital, current_price, stressed_returns
        )
        
        return {
            "original_risk_score": metrics.get("overall_risk_score", 0.0),
            "stressed_risk_score": stressed_metrics.get("overall_risk_score", 0.0),
            "risk_increase": stressed_metrics.get("overall_risk_score", 0.0) - metrics.get("overall_risk_score", 0.0),
            "stressed_metrics": stressed_metrics
        }
    
    def get_risk_recommendations(self, metrics):
        """Get risk management recommendations based on metrics."""
        recommendations = []
        
        # Example recommendations based on metrics
        if metrics.get("overall_risk_score", 0.0) > 0.7:
            recommendations.append("Reduce overall exposure")
            
        if metrics.get("risk_concentration", 0.0) > 0.5:
            recommendations.append("Diversify positions to reduce concentration")
            
        if metrics.get("drawdown", 0.0) > 0.1:
            recommendations.append("Consider hedging strategies to limit drawdown")
            
        if metrics.get("volatility_risk", 0.0) > 0.6:
            recommendations.append("Reduce position sizes during high volatility")
            
        if metrics.get("liquidity_risk", 0.0) > 0.5:
            recommendations.append("Monitor liquidity conditions carefully")
            
        # Default recommendation
        if not recommendations:
            recommendations.append("Risk levels acceptable, maintain current strategy")
            
        return recommendations


# Self-test functionality when run as a script
if __name__ == "__main__":
    # Test risk manager functionality
    test_config = {
        "MAX_BTC_PER_POSITION": 10.0,
        "MAX_USD_PER_POSITION": 1000000.0,
        "MAX_VOLUME_PERCENTAGE": 0.05,
        "RISK_SCORE_THRESHOLD": 0.7,
        "DRAWDOWN_LIMIT": 0.15,
        "CONCENTRATION_LIMIT": 0.5,
        "VAR_LIMIT": 0.1,
        "EXPOSURE_WEIGHT": 0.15,
        "CONCENTRATION_WEIGHT": 0.15,
        "DRAWDOWN_WEIGHT": 0.20,
        "VAR_WEIGHT": 0.15,
        "CORRELATION_WEIGHT": 0.10,
        "VOLATILITY_WEIGHT": 0.15,
        "LIQUIDITY_WEIGHT": 0.10,
    }
    
    # Create a risk manager for each bucket
    scalping_rm = create_risk_manager('Scalping', test_config)
    short_rm = create_risk_manager('Short', test_config)
    medium_rm = create_risk_manager('Medium', test_config)
    long_rm = create_risk_manager('Long', test_config)
    
    print("Testing risk managers...")
    
    # Create test positions
    class TestPosition:
        def __init__(self, size_btc, entry_price, entry_step):
            self.size_btc = size_btc
            self.entry_price = entry_price
            self.entry_step = entry_step
    
    # Create test positions
    positions = [
        TestPosition(0.5, 40000, 100),
        TestPosition(0.3, 41000, 200),
        TestPosition(0.2, 42000, 300),
    ]
    
    # Create risk metrics
    risk_metrics = {
        "total_exposure": 1.0,
        "exposure_percentage": 0.4,
        "risk_concentration": 0.38,
        "drawdown": 0.1,
        "value_at_risk": 20000.0,
        "correlation_risk": 0.3,
        "position_diversity": 0.7,
        "volatility_exposure": 0.5,
        "liquidity_risk": 0.4,
        "max_loss": 30000.0,
        "overall_risk_score": 0.6
    }
    
    # Test probabilistic prediction parameters
    prediction_means = [0.02, 0.05, 0.08, 0.12]  # Predicted price changes for different horizons
    prediction_stds = [0.005, 0.01, 0.02, 0.04]  # Standard deviations for predictions
    confidence_scores = [0.9, 0.8, 0.7, 0.6]     # Confidence for each horizon
    
    # Test calculate_position_size
    price = 40000
    daily_volume = 1000000000  # $1B daily volume
    
    # Test functions across different risk managers
    for name, rm in [('Scalping', scalping_rm), ('Short', short_rm), 
                     ('Medium', medium_rm), ('Long', long_rm)]:
        print(f"\n--- {name} Risk Manager ---")
        
        # Calculate risk metrics
        metrics = rm.calculate_risk_metrics(positions, 100000, price, [-0.01, 0.02, -0.03, 0.01, 0.02])
        print(f"Risk metrics: {metrics}")
        
        # Calculate position size
        base_size = rm.calculate_position_size(price, daily_volume, metrics)
        print(f"Base position size: {base_size:.4f} BTC (${base_size * price:.2f})")
        
        # Calculate risk-adjusted size with probabilistic predictions
        risk_adj_size = rm.calculate_risk_adjusted_size(
            price, daily_volume, 1, metrics, len(positions),
            prediction_mean=prediction_means[0], 
            prediction_std=prediction_stds[0],
            confidence_score=confidence_scores[0]
        )
        print(f"Risk-adjusted size with uncertainty: {risk_adj_size:.4f} BTC (${risk_adj_size * price:.2f})")
        
        # For long-term, also test with trend strength
        if name == 'Long':
            trend_size = rm.calculate_risk_adjusted_size(
                price, daily_volume, 1, metrics, len(positions),
                prediction_mean=prediction_means[0], 
                prediction_std=prediction_stds[0],
                confidence_score=confidence_scores[0],
                trend_strength=0.7  # Strong positive trend
            )
            print(f"With trend strength 0.7: {trend_size:.4f} BTC (${trend_size * price:.2f})")
        
        # Test uncertainty-adjusted risk metrics
        adj_metrics = rm.calculate_confidence_adjusted_risk(
            metrics, prediction_means, prediction_stds, confidence_scores
        )
        print(f"Uncertainty-adjusted risk score: {adj_metrics['overall_risk_score']:.4f}")
        print(f"Prediction uncertainty: {adj_metrics.get('prediction_uncertainty', 'N/A')}")
        
        # Test position adjustment at different uncertainty levels
        base = 1.0  # 1 BTC base position
        for mean, std in [(0.05, 0.005), (0.05, 0.01), (0.05, 0.025), (0.05, 0.05)]:
            cv = std / abs(mean) if abs(mean) > 1e-8 else 1.0
            adj = rm.adjust_position_for_uncertainty(base, mean, std)
            print(f"CV={cv:.2f}: Base={base:.2f} → Adjusted={adj:.2f} BTC")
            
    # Test risk analyzer
    print("\n--- Risk Analyzer ---")
    risk_analyzer = RiskAnalyzer(scalping_rm)
    
    # Add risk snapshots
    for i in range(5):
        metrics = scalping_rm.calculate_risk_metrics(
            positions[:i+1], 100000, price, [-0.01, 0.02, -0.03, 0.01, 0.02]
        )
        risk_analyzer.add_risk_snapshot(metrics)
    
    # Calculate risk trajectory
    trajectory = risk_analyzer.calculate_risk_trajectory()
    print(f"Risk trajectory: {trajectory}")
    
    # Analyze risk components
    components = risk_analyzer.analyze_risk_components(metrics)
    print(f"Risk components: {components}")
    
    # Perform stress test
    stress_factors = {"price_decrease": 0.1, "volatility_increase": 0.3}
    stress_result = risk_analyzer.perform_stress_test(positions, 100000, stress_factors)
    print(f"Stress test result: {stress_result}")
    
    # Get risk recommendations
    recommendations = risk_analyzer.get_risk_recommendations(metrics)
    print(f"Risk recommendations:")
    for rec in recommendations:
        print(f"- {rec}")
