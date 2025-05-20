#!/usr/bin/env python
"""
Market Simulation Module

This module handles realistic market simulation including slippage,
fees, order execution, and liquidity modeling.
"""

import numpy as np
import torch
import random
import importlib
from typing import Dict, List, Any, Union, Optional, Tuple

# Import environment utilities dynamically
try:
    # Import utils module
    env_utils_module = importlib.import_module("src.environment.env_utils")
    log = env_utils_module.log
    get_kraken_fee = env_utils_module.get_kraken_fee
    
    # Import interfaces module
    env_interfaces_module = importlib.import_module("src.environment.env_interfaces")
    Trade = env_interfaces_module.Trade
except ImportError as e:
    print(f"Error importing environment modules in env_market.py: {e}")
    # Define fallback functions if imports fail
    def log(message, level="info"):
        print(f"[{level.upper()}] {message}")
    
    def get_kraken_fee(rolling_volume):
        return 0.0026  # Default fee of 0.26%
    
    # Define minimal classes
    class Trade:
        pass


def compute_slippage(order_size: float, 
                    daily_volume: float, 
                    liquidity: float = 0.5) -> float:
    """
    Calculate price slippage based on order size, market volume, and liquidity.
    
    Args:
        order_size: Size of order in USD
        daily_volume: Daily trading volume in USD
        liquidity: Market liquidity factor (0-1, higher is more liquid)
        
    Returns:
        float: Estimated slippage as a fraction of price
    """
    # Base slippage model: square of order size relative to daily volume
    # Larger orders have greater impact on market price
    if daily_volume <= 0:
        return 0.0025  # Default slippage for unknown volume
    
    volume_ratio = order_size / daily_volume
    base_slippage = 0.0005 + (volume_ratio ** 2) * 0.05
    
    # Adjust based on liquidity
    liquidity_multiplier = 2.0 - liquidity  # Ranges from 1.0 (high liquidity) to 2.0 (low liquidity)
    adjusted_slippage = base_slippage * liquidity_multiplier
    
    # Cap at reasonable limits
    return min(0.025, max(0.0001, adjusted_slippage))  # Between 0.01% and 2.5%


def calculate_fee(notional_value: float, rolling_volume: float = 0.0) -> float:
    """
    Calculate trading fee based on notional value and 30-day rolling volume.
    
    Args:
        notional_value: Notional value of the trade in USD
        rolling_volume: 30-day rolling volume for fee tiers
        
    Returns:
        float: Fee amount in USD
    """
    # Get fee rate based on volume
    fee_rate = get_kraken_fee(rolling_volume)
    
    # Calculate and return fee
    return notional_value * fee_rate


def estimate_execution(
    size_btc: float, 
    price: float, 
    market_conditions: Dict[str, Any] = None
) -> Tuple[float, float]:
    """
    Estimate realistic order execution size and fee.
    
    Args:
        size_btc: Requested size in BTC
        price: Current price
        market_conditions: Dictionary of market condition factors
        
    Returns:
        Tuple of (executed_size, fee)
    """
    if market_conditions is None:
        market_conditions = {}
    
    # Get market metrics
    liquidity = market_conditions.get("liquidity", 0.5)
    volatility = market_conditions.get("volatility", 0.01)
    bid_ask_spread = market_conditions.get("bid_ask_spread", 0.0005)
    
    # Calculate execution quality
    # Higher volatility or lower liquidity can reduce execution quality
    execution_quality = min(1.0, (1.25 - volatility * 10) * liquidity)
    
    # Partial fills more likely for larger orders in poor conditions
    if execution_quality < 0.8 and size_btc > 0.5:
        # Reduce size slightly based on conditions
        execution_size = size_btc * max(0.75, execution_quality)
    else:
        execution_size = size_btc
    
    # Calculate notional value
    notional = execution_size * price
    
    # Calculate fee (includes spread cost)
    spread_cost = notional * bid_ask_spread
    base_fee = calculate_fee(notional)
    total_fee = base_fee + spread_cost
    
    return execution_size, total_fee


def calculate_price_impact(
    size_btc: float,
    price: float,
    daily_volume: float,
    recent_volatility: float = 0.01,
    direction: int = 1  # 1 for buy, -1 for sell
) -> float:
    """
    Calculate estimated price impact of an order.
    
    Args:
        size_btc: Order size in BTC
        price: Current price
        daily_volume: Daily trading volume in USD
        recent_volatility: Recent price volatility
        direction: Order direction (1 for buy, -1 for sell)
        
    Returns:
        float: Estimated post-trade price
    """
    # Convert order size to USD
    order_size_usd = size_btc * price
    
    # Calculate order size as percentage of daily volume
    volume_percentage = order_size_usd / max(daily_volume, 1.0)
    
    # Square root model: impact scales with square root of order size
    # Kyle's lambda model: price impact = λ * σ * sqrt(V/ADV)
    # where λ is a constant, σ is volatility, V is order size, ADV is avg daily volume
    lambda_factor = 0.6  # Market impact factor
    impact_percentage = lambda_factor * recent_volatility * np.sqrt(volume_percentage)
    
    # Cap at reasonable limits
    impact_percentage = min(0.05, impact_percentage)  # Max 5% impact
    
    # Apply direction
    impact = price * impact_percentage * direction
    
    # Calculate and return post-trade price
    return price + impact


def estimate_daily_volume(
    recent_volumes: List[float],
    bar_size_minutes: int = 5
) -> float:
    """
    Estimate daily volume from recent bar volumes.
    
    Args:
        recent_volumes: List of recent bar volumes
        bar_size_minutes: Size of each bar in minutes
        
    Returns:
        float: Estimated daily volume in USD
    """
    if not recent_volumes:
        return 0.0
    
    # Calculate average bar volume
    avg_bar_volume = sum(recent_volumes) / len(recent_volumes)
    
    # Calculate bars per day
    bars_per_day = 24 * 60 / bar_size_minutes
    
    # Estimate daily volume
    return avg_bar_volume * bars_per_day


def simulate_market_hours_impact(
    timestamp,
    base_liquidity: float
) -> float:
    """
    Adjust liquidity based on market hours.
    
    Args:
        timestamp: Current timestamp (can be datetime or simple hour value)
        base_liquidity: Base liquidity value
        
    Returns:
        float: Adjusted liquidity value
    """
    # Extract hour information
    hour = timestamp.hour if hasattr(timestamp, 'hour') else timestamp % 24
    
    # Define market hours impact
    # Reduced liquidity during off-hours
    if 0 <= hour < 3:  # Late night (midnight to 3am UTC)
        hour_factor = 0.7
    elif 3 <= hour < 8:  # Early morning (3am to 8am UTC)
        hour_factor = 0.8
    elif 12 <= hour < 15:  # Mid-day peak (noon to 3pm UTC)
        hour_factor = 1.1
    elif 20 <= hour < 22:  # Evening peak (8pm to 10pm UTC)
        hour_factor = 1.05
    else:
        hour_factor = 1.0
    
    # Apply adjustment
    adjusted_liquidity = base_liquidity * hour_factor
    
    # Ensure within bounds
    return min(1.0, max(0.1, adjusted_liquidity))


def simulate_spread(
    price: float,
    volatility: float,
    liquidity: float
) -> Tuple[float, float]:
    """
    Simulate bid-ask spread based on market conditions.
    
    Args:
        price: Current price
        volatility: Current volatility
        liquidity: Current liquidity (0-1)
        
    Returns:
        Tuple of (bid_price, ask_price)
    """
    # Base spread is a function of volatility and inverse of liquidity
    base_spread_pct = (0.0002 + 0.05 * volatility) * (1.5 - liquidity)
    
    # Minimum spread based on price
    min_spread_pct = 0.0001
    
    # Apply spread
    spread_pct = max(min_spread_pct, base_spread_pct)
    spread_amount = price * spread_pct
    
    bid_price = price - spread_amount / 2
    ask_price = price + spread_amount / 2
    
    return bid_price, ask_price


def update_market_condition(
    current_step: int,
    ohlcv_data: Union[np.ndarray, torch.Tensor],
    lookback: int = 12,
    smooth_factor: float = 0.8
) -> Dict[str, float]:
    """
    Update market condition metrics based on recent data.
    
    Args:
        current_step: Current time step index
        ohlcv_data: Array/Tensor with OHLCV data
        lookback: Number of bars to consider
        smooth_factor: Exponential smoothing factor for updates
        
    Returns:
        dict: Updated market condition metrics
    """
    # Default conditions if not enough data
    if current_step < lookback:
        return {
            "volatility": 0.01,
            "liquidity": 0.5,
            "bid_ask_spread": 0.0005,
            "momentum": 0.0,
            "volume_imbalance": 0.0
        }
    
    # Get relevant data window
    start_idx = max(0, current_step - lookback)
    end_idx = current_step
    
    try:
        # Extract data based on type
        if isinstance(ohlcv_data, np.ndarray):
            window = ohlcv_data[start_idx:end_idx]
            close = window[:, 0] if window.shape[1] > 0 else np.ones(end_idx - start_idx)
            high = window[:, 2] if window.shape[1] > 2 else close * 1.001
            low = window[:, 3] if window.shape[1] > 3 else close * 0.999
            volume = window[:, 4] if window.shape[1] > 4 else np.ones(end_idx - start_idx)
        elif isinstance(ohlcv_data, torch.Tensor):
            window = ohlcv_data[start_idx:end_idx]
            close = window[:, 0].cpu().numpy() if window.shape[1] > 0 else np.ones(end_idx - start_idx)
            high = window[:, 2].cpu().numpy() if window.shape[1] > 2 else close * 1.001
            low = window[:, 3].cpu().numpy() if window.shape[1] > 3 else close * 0.999
            volume = window[:, 4].cpu().numpy() if window.shape[1] > 4 else np.ones(end_idx - start_idx)
        else:
            # Fallback for other data types
            return {
                "volatility": 0.01,
                "liquidity": 0.5,
                "bid_ask_spread": 0.0005,
                "momentum": 0.0,
                "volume_imbalance": 0.0
            }
        
        # Calculate volatility (standard deviation of returns)
        if len(close) > 1:
            returns = np.diff(close) / close[:-1]
            volatility = np.std(returns)
        else:
            volatility = 0.01
        
        # Calculate volume profile for liquidity estimation
        if len(volume) > 0:
            avg_volume = np.mean(volume)
            recent_volume = volume[-1]
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Higher volume generally means higher liquidity
            base_liquidity = min(1.0, 0.3 + 0.7 * min(volume_ratio, 3.0) / 3.0)
        else:
            base_liquidity = 0.5
            
        # Adjust liquidity based on price range
        if len(high) > 0 and len(low) > 0:
            avg_range = np.mean(high - low) / np.mean(close)
            range_factor = max(0.5, min(1.5, 1.0 / (avg_range * 100 + 0.5)))
            liquidity = base_liquidity * range_factor
        else:
            liquidity = base_liquidity
            
        # Estimate bid-ask spread
        if len(high) > 0 and len(low) > 0:
            # Spread often correlates with price range and inverse of volume
            avg_range_pct = np.mean((high - low) / close)
            spread_estimate = max(0.0001, min(0.002, avg_range_pct * 0.1))
        else:
            spread_estimate = 0.0005
            
        # Calculate momentum
        if len(close) > 1:
            momentum = (close[-1] / close[0] - 1.0) * (len(close) / lookback)
        else:
            momentum = 0.0
            
        # Calculate volume imbalance (buy vs sell pressure)
        if len(close) > 1 and len(volume) > 1:
            # Classify each bar as bullish or bearish
            up_volume = sum(volume[i] for i in range(1, len(close)) if close[i] > close[i-1])
            down_volume = sum(volume[i] for i in range(1, len(close)) if close[i] < close[i-1])
            total_volume = up_volume + down_volume
            
            if total_volume > 0:
                volume_imbalance = (up_volume - down_volume) / total_volume
            else:
                volume_imbalance = 0.0
        else:
            volume_imbalance = 0.0
        
        return {
            "volatility": float(volatility),
            "liquidity": float(liquidity),
            "bid_ask_spread": float(spread_estimate),
            "momentum": float(momentum),
            "volume_imbalance": float(volume_imbalance)
        }
        
    except Exception as e:
        log(f"Error updating market conditions: {e}", "error")
        return {
            "volatility": 0.01,
            "liquidity": 0.5,
            "bid_ask_spread": 0.0005,
            "momentum": 0.0,
            "volume_imbalance": 0.0
        }


def simulate_order_book(
    price: float,
    market_conditions: Dict[str, float],
    depth_levels: int = 5
) -> Dict[str, List[Tuple[float, float]]]:
    """
    Simulate a realistic order book based on current price and market conditions.
    
    Args:
        price: Current mid price
        market_conditions: Dictionary of market conditions
        depth_levels: Number of price levels to generate on each side
        
    Returns:
        dict: Order book with 'bids' and 'asks' lists of (price, size) tuples
    """
    # Extract market conditions
    volatility = market_conditions.get("volatility", 0.01)
    liquidity = market_conditions.get("liquidity", 0.5)
    bid_ask_spread = market_conditions.get("bid_ask_spread", 0.0005)
    volume_imbalance = market_conditions.get("volume_imbalance", 0.0)
    
    # Calculate bid and ask prices
    bid_price = price * (1 - bid_ask_spread / 2)
    ask_price = price * (1 + bid_ask_spread / 2)
    
    # Adjust for imbalance (shift prices slightly based on buy/sell pressure)
    imbalance_shift = price * volume_imbalance * 0.0002  # Subtle effect
    bid_price += imbalance_shift
    ask_price += imbalance_shift
    
    # Calculate step size between levels (larger for higher volatility)
    base_step = price * (0.0001 + volatility * 0.02)
    
    # Adjust liquidity factor (affects order sizes)
    liquidity_factor = liquidity * 2  # Scale for more intuitive sizes
    
    # Generate bids (buy orders)
    bids = []
    current_bid = bid_price
    
    for i in range(depth_levels):
        # Price decreases as we go down the book
        current_bid -= base_step * (1.0 + 0.2 * i)  # Increasing gaps
        
        # Size tends to increase deeper in the book
        # Add some randomness for realism
        size_factor = 1.0 + i * 0.5  # Larger sizes deeper in the book
        random_factor = 0.7 + 0.6 * random.random()  # 0.7-1.3x randomness
        
        # Calculate size (in BTC)
        size = 0.05 * liquidity_factor * size_factor * random_factor
        
        # Adjust for imbalance (more bids if positive imbalance/buying pressure)
        if volume_imbalance > 0:
            size *= (1.0 + volume_imbalance)
        
        bids.append((current_bid, size))
    
    # Generate asks (sell orders)
    asks = []
    current_ask = ask_price
    
    for i in range(depth_levels):
        # Price increases as we go up the book
        current_ask += base_step * (1.0 + 0.2 * i)  # Increasing gaps
        
        # Size calculation with randomness
        size_factor = 1.0 + i * 0.5
        random_factor = 0.7 + 0.6 * random.random()
        
        # Calculate size (in BTC)
        size = 0.05 * liquidity_factor * size_factor * random_factor
        
        # Adjust for imbalance (more asks if negative imbalance/selling pressure)
        if volume_imbalance < 0:
            size *= (1.0 - volume_imbalance)  # Note: imbalance is negative here
        
        asks.append((current_ask, size))
    
    return {
        "bids": bids,
        "asks": asks
    }


def simulate_execution_price(
    order_type: str,
    size_btc: float,
    current_price: float,
    order_book: Dict[str, List[Tuple[float, float]]],
    direction: int = 1  # 1 for buy, -1 for sell
) -> float:
    """
    Simulate execution price based on order type, size, and order book.
    
    Args:
        order_type: Type of order ('market', 'limit', 'stop', etc.)
        size_btc: Order size in BTC
        current_price: Current market price
        order_book: Simulated order book
        direction: Order direction (1 for buy, -1 for sell)
        
    Returns:
        float: Simulated execution price
    """
    if order_type.lower() == "market":
        # For market orders, walk the book
        if direction > 0:  # Buy order
            return _walk_order_book(size_btc, order_book["asks"])
        else:  # Sell order
            return _walk_order_book(size_btc, order_book["bids"])
    elif order_type.lower() == "limit":
        # For limit orders, assuming we're using the current price as limit price
        if direction > 0:  # Buy limit order
            # Can only execute if price is at or below limit
            if current_price <= current_price:  # Always true in this case
                return current_price
        else:  # Sell limit order
            # Can only execute if price is at or above limit
            if current_price >= current_price:  # Always true in this case
                return current_price
    elif order_type.lower() == "stop":
        # For stop orders, assuming current_price is used as stop price
        # Stop orders convert to market orders when triggered
        if direction > 0:  # Buy stop
            if current_price >= current_price:  # Stop is triggered
                return _walk_order_book(size_btc, order_book["asks"])
        else:  # Sell stop
            if current_price <= current_price:  # Stop is triggered
                return _walk_order_book(size_btc, order_book["bids"])
    
    # Default fallback
    return current_price


def _walk_order_book(size_btc: float, book_side: List[Tuple[float, float]]) -> float:
    """
    Walk through order book levels to calculate weighted average execution price.
    
    Args:
        size_btc: Order size to fill
        book_side: List of (price, size) tuples representing one side of the order book
        
    Returns:
        float: Weighted average execution price
    """
    if not book_side or size_btc <= 0:
        return 0.0
    
    remaining_size = size_btc
    total_cost = 0.0
    filled_size = 0.0
    
    for price, available_size in book_side:
        if remaining_size <= 0:
            break
            
        execute_size = min(remaining_size, available_size)
        total_cost += execute_size * price
        filled_size += execute_size
        remaining_size -= execute_size
    
    # If we couldn't fill the entire order, use the last price for the remainder
    if remaining_size > 0 and book_side:
        total_cost += remaining_size * book_side[-1][0]
        filled_size += remaining_size
    
    # Calculate weighted average price
    if filled_size > 0:
        return total_cost / filled_size
    else:
        return 0.0


def estimate_market_liquidity(
    recent_volumes: List[float],
    recent_ranges: List[float],
    avg_price: float
) -> float:
    """
    Estimate market liquidity based on recent volume and price ranges.
    
    Args:
        recent_volumes: List of recent bar volumes
        recent_ranges: List of recent high-low ranges
        avg_price: Average price over the period
        
    Returns:
        float: Estimated liquidity score (0-1)
    """
    if not recent_volumes or not recent_ranges or avg_price <= 0:
        return 0.5  # Default to medium liquidity
    
    # Calculate average volume
    avg_volume = sum(recent_volumes) / len(recent_volumes)
    
    # Calculate average range as percentage of price
    avg_range_pct = sum(r / avg_price for r in recent_ranges) / len(recent_ranges)
    
    # Liquidity is proportional to volume and inversely proportional to price range
    if avg_range_pct > 0:
        # Normalize volume (higher is better)
        volume_score = min(1.0, avg_volume / 10.0)  # Assuming 10.0 is high volume
        
        # Normalize range (lower is better)
        range_score = max(0.0, 1.0 - avg_range_pct * 100)  # Convert to percentage
        
        # Combined liquidity score (70% volume, 30% range)
        liquidity = 0.7 * volume_score + 0.3 * range_score
    else:
        liquidity = 0.5
    
    # Ensure within bounds
    return min(1.0, max(0.1, liquidity))


def calculate_fill_probability(
    order_type: str,
    price_difference: float,  # Difference between order price and market price
    market_conditions: Dict[str, float],
    time_in_force: int = 10  # Number of bars the order is valid for
) -> float:
    """
    Calculate probability of order fill based on order parameters and market conditions.
    
    Args:
        order_type: Type of order ('limit', 'stop', etc.)
        price_difference: Difference between order price and current price
        market_conditions: Dictionary of market condition metrics
        time_in_force: How long the order is valid for (in bars)
        
    Returns:
        float: Probability of fill (0-1)
    """
    # Extract market conditions
    volatility = market_conditions.get("volatility", 0.01)
    liquidity = market_conditions.get("liquidity", 0.5)
    
    # Base fill probability
    base_probability = 0.0
    
    if order_type.lower() == "limit":
        # For limit orders, probability depends on price difference
        # Negative price_difference means limit buy above market or limit sell below market
        # (i.e., immediately fillable)
        if price_difference <= 0:
            return 0.95  # Almost certain to fill
            
        # Otherwise, probability decreases with price difference
        # and increases with volatility and time in force
        
        # Normalize price difference by volatility
        normalized_diff = price_difference / (volatility * current_price)
        
        # Calculate probability based on normalized difference
        base_probability = np.exp(-normalized_diff)
        
        # Adjust for time in force (longer = higher probability)
        time_factor = min(1.0, time_in_force / 20.0)  # 20 bars for max effect
        
        # Adjust for liquidity (higher liquidity = higher fill probability)
        liquidity_factor = 0.5 + 0.5 * liquidity
        
        # Calculate final probability
        fill_probability = base_probability * (0.7 + 0.3 * time_factor) * liquidity_factor
        
    elif order_type.lower() == "stop":
        # For stop orders, similar logic but in reverse
        # Positive price_difference means stop price has been reached
        if price_difference >= 0:
            return 0.95  # Almost certain to fill
            
        # Otherwise, probability depends on how far from trigger
        normalized_diff = -price_difference / (volatility * current_price)
        base_probability = np.exp(-normalized_diff)
        
        # Apply time and liquidity factors
        time_factor = min(1.0, time_in_force / 20.0)
        liquidity_factor = 0.5 + 0.5 * liquidity
        
        fill_probability = base_probability * (0.7 + 0.3 * time_factor) * liquidity_factor
    else:
        # For other order types, default probability
        fill_probability = 0.5
    
    # Ensure within bounds
    return min(0.99, max(0.01, fill_probability))


def optimize_order_sizing(
    desired_size_btc: float,
    max_size_btc: float,
    price: float,
    market_conditions: Dict[str, float],
    risk_profile: str = "balanced"
) -> float:
    """
    Optimize order sizing based on market conditions and risk profile.
    
    Args:
        desired_size_btc: Initially requested order size
        max_size_btc: Maximum allowable size
        price: Current price
        market_conditions: Dictionary of market conditions
        risk_profile: Risk tolerance profile ('conservative', 'balanced', 'aggressive')
        
    Returns:
        float: Optimized order size
    """
    # Extract market conditions
    volatility = market_conditions.get("volatility", 0.01)
    liquidity = market_conditions.get("liquidity", 0.5)
    
    # Set risk factors based on profile
    if risk_profile.lower() == "conservative":
        volatility_factor = 2.0  # More sensitive to volatility
        liquidity_factor = 1.5  # More sensitive to liquidity
        base_scaling = 0.8  # Scale down from desired size
    elif risk_profile.lower() == "aggressive":
        volatility_factor = 1.0  # Less sensitive to volatility
        liquidity_factor = 0.8  # Less sensitive to liquidity
        base_scaling = 1.0  # Use full desired size
    else:  # balanced
        volatility_factor = 1.5
        liquidity_factor = 1.0
        base_scaling = 0.9
    
    # Start with base scaling of desired size
    optimal_size = desired_size_btc * base_scaling
    
    # Adjust for market conditions
    # Reduce size in high volatility
    volatility_adjustment = max(0.5, 1.0 - volatility * volatility_factor * 10)
    
    # Reduce size in low liquidity
    liquidity_adjustment = 0.5 + 0.5 * liquidity * liquidity_factor
    
    # Apply adjustments
    optimal_size *= volatility_adjustment * liquidity_adjustment
    
    # Ensure within bounds
    return min(max_size_btc, max(0.001, optimal_size))


if __name__ == "__main__":
    # Test market simulation functions
    print("Testing market simulation functions...")
    
    price = 50000.0
    daily_volume = 5000000.0
    order_size = 100000.0
    
    # Test slippage calculation
    slippage = compute_slippage(order_size, daily_volume, 0.7)
    print(f"Slippage for ${order_size} order: {slippage*100:.4f}%")
    
    # Test order execution
    market_conditions = {
        "volatility": 0.02,
        "liquidity": 0.6,
        "bid_ask_spread": 0.0008
    }
    
    executed_size, fee = estimate_execution(2.0, price, market_conditions)
    print(f"Executed size: {executed_size} BTC, Fee: ${fee:.2f}")
    
    # Test order book simulation
    order_book = simulate_order_book(price, market_conditions)
    print("\nSimulated Order Book:")
    print("Bids (Buy Orders):")
    for price, size in order_book["bids"]:
        print(f"  ${price:.2f}: {size:.4f} BTC")
    
    print("Asks (Sell Orders):")
    for price, size in order_book["asks"]:
        print(f"  ${price:.2f}: {size:.4f} BTC")
    
    # Test market impact
    impact_price = calculate_price_impact(5.0, price, daily_volume, 0.02, 1)
    print(f"\nPrice after market impact: ${impact_price:.2f} (Impact: ${impact_price-price:.2f})")
