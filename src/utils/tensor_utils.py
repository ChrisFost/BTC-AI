#!/usr/bin/env python
"""
Tensor Utility Functions for Trading Agent

This module provides tensor-based utility functions for market data analysis,
optimized for high-performance operation on GPU or CPU.
"""

import torch
import numpy as np
import importlib
from sklearn.cluster import KMeans

# Use dynamic imports
try:
    utils_module = importlib.import_module("src.utils.utils")
    log = utils_module.log
except ImportError:
    # Fallback logging function if import fails
    def log(message, level="INFO"):
        print(f"[{level}] {message}")

def compute_volume_profile_tensor(ohlcv_tensor, current_step, lookback=144, num_levels=10, device="cpu"):
    """
    Compute volume profile using tensor operations with robust error handling.
    
    Args:
        ohlcv_tensor (torch.Tensor): Tensor of OHLCV data (batch, 5).
        current_step (int): Current time step index.
        lookback (int, optional): Number of bars to look back.
        num_levels (int, optional): Number of price levels in volume profile.
        device (str, optional): Device to run calculations on.
    
    Returns:
        tuple: (profile, price_min, price_max) - Volume profile tensor and price range.
    """
    try:
        # Ensure indices are valid
        if ohlcv_tensor is None or len(ohlcv_tensor) <= current_step:
            return torch.zeros(num_levels, device=device), 0, 0
            
        start_idx = max(0, current_step - lookback)
        
        # Check if we have enough data
        if start_idx >= current_step:
            return torch.zeros(num_levels, device=device), 0, 0
            
        window = ohlcv_tensor[start_idx:current_step]
        
        if len(window) == 0:
            return torch.zeros(num_levels, device=device), 0, 0
        
        # Extract OHLCV data safely
        try:
            close, open_price, high, low, volume = window[:, 0], window[:, 1], window[:, 2], window[:, 3], window[:, 4]
        except IndexError:
            # Handle case where tensor doesn't have expected dimensions
            return torch.zeros(num_levels, device=device), 0, 0
        
        # Calculate price range for the window
        # Handle edge cases where high/low might be invalid
        if torch.isnan(high).any() or torch.isnan(low).any():
            valid_high = high[~torch.isnan(high)]
            valid_low = low[~torch.isnan(low)]
            
            price_min = torch.min(valid_low) if len(valid_low) > 0 else torch.tensor(0.0, device=device)
            price_max = torch.max(valid_high) if len(valid_high) > 0 else torch.tensor(1.0, device=device)
        else:
            price_min = torch.min(low)
            price_max = torch.max(high)
            
        if price_max == price_min:
            # Avoid division by zero
            price_max = price_min * 1.001
        
        # Create price levels
        level_size = (price_max - price_min) / num_levels
        levels = torch.linspace(price_min, price_max, num_levels + 1, device=device)
        
        # Initialize volume profile
        profile = torch.zeros(num_levels, device=device)
        
        # Assign volume to levels (simple approximation)
        # Use vectorized operations where possible
        for i in range(len(window)):
            try:
                # Use typical price (average of open, high, low, close)
                typical_price = (close[i] + high[i] + low[i] + open_price[i]) / 4
                vol = volume[i]
                
                # Handle NaN values
                if torch.isnan(typical_price) or torch.isnan(vol):
                    continue
                
                # Find which level this price belongs to
                level_idx = min(int((typical_price - price_min) / level_size), num_levels - 1)
                if level_idx < 0:  # Safeguard against numerical issues
                    level_idx = 0
                    
                profile[level_idx] += vol
            except Exception:
                # Skip this iteration if any error occurs
                continue
        
        # Normalize profile to be relative to total volume
        total_volume = torch.sum(volume)
        if total_volume > 0:
            profile = profile / total_volume
            
        return profile, price_min.item(), price_max.item()
    except Exception as e:
        # Log the error and return default values
        log(f"Error in compute_volume_profile_tensor: {e}")
        return torch.zeros(num_levels, device=device), 0, 0

def identify_liquidity_zones_tensor(ohlcv_tensor, current_step, lookback=288, min_zone_distance=0.015, device="cpu"):
    """
    Identify potential support and resistance zones based on price clusters.

    Args:
        ohlcv_tensor (torch.Tensor): Tensor of OHLCV data (batch, 5).
        current_step (int): Current time step index.
        lookback (int, optional): Number of bars to consider for clustering.
        min_zone_distance (float, optional): Minimum relative distance between zones (not currently used).
        device (str, optional): Device to run calculations on.

    Returns:
        tuple: (support_zones, resistance_zones) - Lists of price levels.
    """
    try:
        if ohlcv_tensor is None or len(ohlcv_tensor) <= current_step:
            log("Insufficient data for liquidity zones.")
            return [], [] # Return empty lists if not enough data

        # Ensure current_step allows fetching current price
        if current_step == 0:
             log("Cannot determine current price at step 0.")
             return [], []

        start_idx = max(0, current_step - lookback)
        window = ohlcv_tensor[start_idx:current_step]

        if len(window) < 10: # Need a minimum number of bars
             log("Window too small for liquidity zones.")
             return [], []

        # Get current price (using close price of the latest bar in the window)
        current_price = ohlcv_tensor[current_step - 1, 0].item() # Assuming column 0 is close price
        if not np.isfinite(current_price):
            log(f"Invalid current price ({current_price}) at step {current_step}.")
            return [], []

        # Detect price clusters using the existing function
        # This function returns a list of cluster centers (prices)
        cluster_centers = detect_price_clusters_tensor(
            ohlcv_tensor,
            current_step,
            window_size=lookback,
            num_clusters=5, # Using a fixed number for now, can be dynamic later
            device=device
        )

        if not cluster_centers: # If cluster detection failed or returned empty
            log("No clusters detected.")
            return [], []

        # Classify clusters into support and resistance based on current price
        support_zones = sorted([center for center in cluster_centers if center < current_price and np.isfinite(center)])
        resistance_zones = sorted([center for center in cluster_centers if center >= current_price and np.isfinite(center)])

        # Optional: Further refinement (e.g., filter weak zones, merge close zones)
        # For now, just return the sorted lists

        return support_zones, resistance_zones

    except Exception as e:
        # Log the specific error and traceback for debugging
        import traceback
        log(f"Error identifying liquidity zones: {e}\\n{traceback.format_exc()}", level="ERROR")
        return [], [] # Return empty lists on error

def estimate_bid_ask_spread_tensor(ohlcv_tensor, current_step, window_size=12, device="cpu"):
    """
    Estimate the bid-ask spread using high-low range as a proxy.
    
    Args:
        ohlcv_tensor (torch.Tensor): Tensor of OHLCV data (batch, 5).
        current_step (int): Current time step index.
        window_size (int, optional): Number of bars to consider for spread estimation.
        device (str, optional): Device to run calculations on.
    
    Returns:
        float: Estimated bid-ask spread as a fraction of price.
    """
    try:
        if ohlcv_tensor is None or len(ohlcv_tensor) <= current_step:
            return 0.0001  # Default small spread
            
        start_idx = max(0, current_step - window_size)
        window = ohlcv_tensor[start_idx:current_step]
        
        if len(window) == 0:
            return 0.0001  # Default small spread
        
        # Calculate typical intrabar volatility as % of price
        try:
            high = window[:, 2]
            low = window[:, 3]
            close = window[:, 0]
            
            # Ensure no division by zero
            valid_idx = close > 0
            if not torch.any(valid_idx):
                return 0.0001
                
            # Calculate high-low range as percentage of closing price
            volatility = torch.mean(torch.abs(high[valid_idx] - low[valid_idx]) / close[valid_idx])
        except Exception:
            return 0.0001
        
        # Use volatility to estimate spread, with bounds
        min_spread = 0.0001  # 0.01% minimum spread
        max_spread = 0.002   # 0.2% maximum spread
        spread = max(min_spread, min(max_spread, volatility.item() * 0.1))  # Spread is ~10% of short-term volatility
        
        return spread
    
    except Exception as e:
        log(f"Error in estimate_bid_ask_spread_tensor: {e}")
        return 0.0001  # Default in case of error

def calculate_volume_delta_tensor(ohlcv_tensor, current_step, window_size=6, device="cpu"):
    """
    Calculate volume delta (buying vs selling pressure).
    
    Args:
        ohlcv_tensor (torch.Tensor): Tensor of OHLCV data (batch, 5).
        current_step (int): Current time step index.
        window_size (int, optional): Number of bars to consider.
        device (str, optional): Device to run calculations on.
    
    Returns:
        tuple: (volume_delta, buy_volume, sell_volume) - Volume statistics.
    """
    try:
        if ohlcv_tensor is None or len(ohlcv_tensor) <= current_step:
            return 0.0, 0.0, 0.0
            
        start_idx = max(0, current_step - window_size)
        window = ohlcv_tensor[start_idx:current_step]
        
        if len(window) == 0:
            return 0.0, 0.0, 0.0
        
        # Extract close, open, volume
        try:
            close = window[:, 0]
            open_price = window[:, 1]
            volume = window[:, 4]
            
            # Calculate price change
            price_change = close - open_price
            
            # Estimate buying and selling volume
            buy_mask = price_change > 0
            sell_mask = price_change < 0
            
            # Calculate volume for buying and selling pressure
            buy_volume = torch.sum(volume[buy_mask]) if torch.any(buy_mask) else torch.tensor(0.0, device=device)
            sell_volume = torch.sum(volume[sell_mask]) if torch.any(sell_mask) else torch.tensor(0.0, device=device)
            
            # Volume delta is the difference between buy and sell volume
            volume_delta = buy_volume - sell_volume
        except Exception:
            return 0.0, 0.0, 0.0
        
        # Convert to Python floats for return
        return volume_delta.item(), buy_volume.item(), sell_volume.item()
    
    except Exception as e:
        log(f"Error in calculate_volume_delta_tensor: {e}")
        return 0.0, 0.0, 0.0

def estimate_market_liquidity_tensor(ohlcv_tensor, current_step, window_size=72, device="cpu"):
    """
    Estimate market liquidity based on volume and volatility.
    
    Args:
        ohlcv_tensor (torch.Tensor): Tensor of OHLCV data (batch, 5).
        current_step (int): Current time step index.
        window_size (int, optional): Number of bars to consider.
        device (str, optional): Device to run calculations on.
    
    Returns:
        float: Estimated market liquidity as a value between 0 and 1.
    """
    try:
        if ohlcv_tensor is None or len(ohlcv_tensor) <= current_step:
            return 0.5  # Default medium liquidity
            
        start_idx = max(0, current_step - window_size)
        window = ohlcv_tensor[start_idx:current_step]
        
        if len(window) == 0:
            return 0.5  # Default medium liquidity
        
        # Extract close, high, low, volume
        try:
            close = window[:, 0]
            high = window[:, 2]
            low = window[:, 3]
            volume = window[:, 4]
        except Exception:
            return 0.5
        
        # Calculate volume trend
        if len(volume) > 1:
            recent_window = min(12, len(volume))
            recent_volume = torch.mean(volume[-recent_window:])
            older_volume = torch.mean(volume[:-recent_window] if len(volume) > recent_window else volume)
            volume_trend = recent_volume / (older_volume + 1e-8)
        else:
            volume_trend = 1.0
        
        # Calculate volatility
        volatility = torch.std(close) / torch.mean(close) if len(close) > 1 else torch.tensor(0.01, device=device)
        
        # High volume and low volatility indicates high liquidity
        liquidity = (torch.mean(volume) / (volatility * 1000 + 1e-8))
        
        # Normalize to 0-1 range
        normalized_liquidity = torch.sigmoid(liquidity / 10).item()
        
        # Adjust for volume trend
        volume_trend_val = volume_trend.item() if isinstance(volume_trend, torch.Tensor) else volume_trend
        
        if volume_trend_val > 1.2:  # Volume increasing
            normalized_liquidity = min(1.0, normalized_liquidity * 1.2)
        elif volume_trend_val < 0.8:  # Volume decreasing
            normalized_liquidity = max(0.1, normalized_liquidity * 0.8)
        
        return normalized_liquidity
    
    except Exception as e:
        log(f"Error in estimate_market_liquidity_tensor: {e}")
        return 0.5  # Default in case of error

def calculate_price_momentum_tensor(ohlcv_tensor, current_step, window_size=24, device="cpu"):
    """
    Calculate price momentum using tensor operations.
    
    Args:
        ohlcv_tensor (torch.Tensor): Tensor of OHLCV data (batch, 5).
        current_step (int): Current time step index.
        window_size (int, optional): Number of bars to consider.
        device (str, optional): Device to run calculations on.
    
    Returns:
        float: Price momentum normalized to [-1, 1] range.
    """
    try:
        if ohlcv_tensor is None or len(ohlcv_tensor) <= current_step:
            return 0.0
            
        start_idx = max(0, current_step - window_size)
        window = ohlcv_tensor[start_idx:current_step]
        
        if len(window) < 2:
            return 0.0
        
        # Extract close prices
        close = window[:, 0]
        
        # Calculate rate of change
        price_change = (close[-1] - close[0]) / (close[0] + 1e-8)
        
        # Normalize to [-1, 1] range using tanh
        normalized_momentum = torch.tanh(price_change * 5).item()  # Scale factor of 5 provides good sensitivity
        
        return normalized_momentum
    
    except Exception as e:
        log(f"Error in calculate_price_momentum_tensor: {e}")
        return 0.0

def calculate_vwap_tensor(ohlcv_tensor, current_step, window_size=144, device="cpu"):
    """
    Calculate Volume-Weighted Average Price (VWAP).
    
    Args:
        ohlcv_tensor (torch.Tensor): Tensor of OHLCV data (batch, 5).
        current_step (int): Current time step index.
        window_size (int, optional): Number of bars to consider.
        device (str, optional): Device to run calculations on.
    
    Returns:
        float: VWAP price.
    """
    try:
        if ohlcv_tensor is None or len(ohlcv_tensor) <= current_step:
            return ohlcv_tensor[current_step, 0].item() if current_step < len(ohlcv_tensor) else 0.0
            
        start_idx = max(0, current_step - window_size)
        window = ohlcv_tensor[start_idx:current_step]
        
        if len(window) == 0:
            return ohlcv_tensor[current_step, 0].item() if current_step < len(ohlcv_tensor) else 0.0
        
        # Extract OHLCV data
        try:
            close = window[:, 0]
            open_price = window[:, 1]
            high = window[:, 2]
            low = window[:, 3]
            volume = window[:, 4]
            
            # Calculate typical price for each bar
            typical_price = (high + low + close) / 3
            
            # Calculate volume-weighted typical price
            vwap_numerator = torch.sum(typical_price * volume)
            vwap_denominator = torch.sum(volume)
            
            # Avoid division by zero
            if vwap_denominator > 0:
                vwap = vwap_numerator / vwap_denominator
            else:
                vwap = torch.mean(typical_price)
        except Exception:
            return ohlcv_tensor[current_step, 0].item() if current_step < len(ohlcv_tensor) else 0.0
        
        return vwap.item()
    
    except Exception as e:
        log(f"Error in calculate_vwap_tensor: {e}")
        return ohlcv_tensor[current_step, 0].item() if current_step < len(ohlcv_tensor) else 0.0

# ENHANCED PATTERN RECOGNITION SECTION
def detect_patterns_tensor(ohlcv_tensor, current_step, lookback=100, device="cpu"):
    """
    Detect common price patterns in market data using tensor operations.
    Enhanced with more sophisticated pattern recognition algorithms.
    
    Args:
        ohlcv_tensor (torch.Tensor): Tensor of OHLCV data (batch, 5).
        current_step (int): Current time step index.
        lookback (int, optional): Number of bars to look back for patterns.
        device (str, optional): Device to run calculations on.
    
    Returns:
        dict: Dictionary of detected patterns and their strengths.
    """
    if ohlcv_tensor is None or current_step < lookback:
        return {"patterns": {}, "strength": 0.0}
    
    try:
        # Extract window of data
        start_idx = max(0, current_step - lookback)
        window = ohlcv_tensor[start_idx:current_step]
        
        if len(window) < 20:  # Need sufficient data for pattern detection
            return {"patterns": {}, "strength": 0.0}
        
        # Extract OHLCV components
        close = window[:, 0]
        open_price = window[:, 1]
        high = window[:, 2]
        low = window[:, 3]
        volume = window[:, 4]
        
        # Calculate some key moving averages for trend context
        ma10 = torch.mean(close[-10:])
        ma20 = torch.mean(close[-20:])
        ma50 = torch.mean(close[-50:]) if len(close) >= 50 else ma20
        
        # Detect basic trend
        trend = 0  # 1 = uptrend, 0 = neutral, -1 = downtrend
        if ma10 > ma20 and ma20 > ma50:
            trend = 1
        elif ma10 < ma20 and ma20 < ma50:
            trend = -1
        
        # Enhanced with more powerful pattern detection
        # Normalize price series for better pattern matching
        norm_close = (close - torch.min(close)) / (torch.max(close) - torch.min(close) + 1e-8)
        
        patterns = {}
        overall_strength = 0.0
        
        # 1. Enhanced Double Bottom pattern with improved accuracy
        if trend == -1 and len(close) >= 30:
            # Look for two lows at similar price levels with advanced confirmation
            min_idx1 = torch.argmin(close[-30:-15])
            min_idx2 = torch.argmin(close[-15:]) + 15
            
            min_price1 = close[-30:][min_idx1]
            min_price2 = close[-15:][min_idx2 - 15]
            
            # Advanced check with better tolerance and volume confirmation
            price_diff_pct = abs(min_price1 - min_price2) / min_price1
            volume_confirmation = volume[-15:][min_idx2 - 15] > torch.mean(volume[-15:])
            
            if price_diff_pct < 0.02 and close[-1] > min_price2:
                # More sophisticated strength calculation
                confirmation = (close[-1] - min_price2) / min_price2
                
                # Calculate pattern strength (0-1) with multiple factors
                time_ratio = abs((min_idx2 - min_idx1) / lookback - 0.5) * 2  # Time symmetry (0=perfect, 1=worst)
                strength = min(1.0, max(0.1, (1.0 - price_diff_pct * 50) * 0.5 + confirmation * 0.3 + (1.0 - time_ratio) * 0.2))
                
                if volume_confirmation:
                    strength *= 1.2  # Boost for volume confirmation
                
                patterns["double_bottom"] = min(1.0, strength)
                overall_strength += strength * 0.6  # Weighting factor
        
        # 2. Enhanced Bullish Engulfing pattern with volume confirmation
        if len(close) >= 2:
            # Previous day was bearish (close < open)
            prev_bearish = open_price[-2] > close[-2]
            # Current day is bullish (close > open)
            curr_bullish = close[-1] > open_price[-1]
            # Current day engulfs previous day (stricter definition)
            engulfing = open_price[-1] <= close[-2] and close[-1] >= open_price[-2]
            
            # Enhanced with context awareness
            if prev_bearish and curr_bullish and engulfing:
                # Calculate strength based on multiple factors
                candle_size_ratio = (close[-1] - open_price[-1]) / max(0.0001, (open_price[-2] - close[-2]))
                price_location = (close[-1] - torch.min(low[-10:])) / (torch.max(high[-10:]) - torch.min(low[-10:]) + 1e-8)
                volume_increase = volume[-1] > volume[-2] * 1.2  # 20% volume increase
                
                # Contextual strength calculation
                strength = min(1.0, candle_size_ratio * 0.4 + price_location * 0.3 + 0.3)
                
                # Volume confirmation boost
                if volume_increase:
                    strength *= 1.2  # Boost for volume confirmation
                
                patterns["bullish_engulfing"] = min(1.0, strength)
                overall_strength += strength * 0.5
        
        # 3. HEAD AND SHOULDERS PATTERN (new sophisticated implementation)
        if len(close) >= 40:
            hs_confidence = detect_head_and_shoulders_pattern(norm_close)
            if hs_confidence > 0.6:  # Higher threshold for reliability
                patterns["head_and_shoulders"] = hs_confidence
                overall_strength += hs_confidence * 0.7
                
            # Inverse Head and Shoulders
            ihs_confidence = detect_head_and_shoulders_pattern(1.0 - norm_close, inverse=True)
            if ihs_confidence > 0.6:
                patterns["inverse_head_and_shoulders"] = ihs_confidence
                overall_strength += ihs_confidence * 0.7
        
        # 4. Triangle patterns (ascending, descending, symmetrical)
        if len(close) >= 30:
            triangle_patterns = detect_triangle_patterns(norm_close, high, low)
            for pattern_name, confidence in triangle_patterns.items():
                if confidence > 0.6:
                    patterns[pattern_name] = confidence
                    overall_strength += confidence * 0.5
        
        # 5. ADVANCED FLAG PATTERN (with slope and volume analysis)
        if len(close) >= 20:
            flag_patterns = detect_flag_patterns(norm_close, volume)
            for pattern_name, confidence in flag_patterns.items():
                if confidence > 0.55:
                    patterns[pattern_name] = confidence
                    overall_strength += confidence * 0.4
        
        # 6. ADVANCED HARMONIC PATTERNS (Gartley, Bat, Butterfly)
        if len(close) >= 40:
            harmonic_patterns = detect_harmonic_patterns(close, high, low)
            for pattern_name, confidence in harmonic_patterns.items():
                if confidence > 0.7:  # Higher threshold for complex patterns
                    patterns[pattern_name] = confidence
                    overall_strength += confidence * 0.8  # Higher weight for sophisticated patterns
        
        # Cap overall strength at 1.0
        overall_strength = min(1.0, overall_strength)
        
        return {
            "patterns": patterns,
            "strength": overall_strength,
            "trend": trend
        }
    
    except Exception as e:
        log(f"Error in enhanced pattern detection: {e}")
        return {"patterns": {}, "strength": 0.0, "trend": 0}

def detect_head_and_shoulders_pattern(prices, inverse=False):
    """
    Detect head and shoulders pattern with improved precision.
    
    Args:
        prices (torch.Tensor): Normalized price series.
        inverse (bool): Whether to detect inverse head and shoulders.
        
    Returns:
        float: Confidence score (0-1).
    """
    if len(prices) < 30:
        return 0.0
    
    try:
        # Find local peaks/troughs
        local_extrema = []
        extrema_values = []
        is_peak = not inverse  # For normal H&S we look for peaks, inverse for troughs
        
        min_gap = 3  # Minimum distance between extrema
        
        for i in range(min_gap, len(prices) - min_gap):
            if is_peak:
                # Find peaks
                if all(prices[i] > prices[i-j] for j in range(1, min_gap+1)) and \
                all(prices[i] > prices[i+j] for j in range(1, min_gap+1)):
                    local_extrema.append(i)
                    extrema_values.append(prices[i].item())
            else:
                # Find troughs
                if all(prices[i] < prices[i-j] for j in range(1, min_gap+1)) and \
                all(prices[i] < prices[i+j] for j in range(1, min_gap+1)):
                    local_extrema.append(i)
                    extrema_values.append(prices[i].item())
        
        # Need at least 5 extrema for a valid H&S
        if len(local_extrema) < 5:
            return 0.0
        
        max_confidence = 0.0
        
        # Try to find H&S pattern in all possible 5-point combinations
        for i in range(len(local_extrema) - 4):
            # Get 5 consecutive extrema
            idx1, idx2, idx3, idx4, idx5 = local_extrema[i:i+5]
            p1, p2, p3, p4, p5 = extrema_values[i:i+5]
            
            # Calculate pattern scores
            if not inverse:
                # H&S Pattern: p1=left shoulder, p3=head, p5=right shoulder
                # Head should be higher than shoulders
                if p3 > p1 and p3 > p5:
                    # Shoulders should be similar height
                    shoulder_similarity = 1.0 - abs(p1 - p5) / max(p1, p5, 0.0001)
                    
                    # Neckline (p2 and p4) should be similar level
                    neckline_similarity = 1.0 - abs(p2 - p4) / max(p2, p4, 0.0001)
                    
                    # Time symmetry (shoulders equidistant from head)
                    time_symmetry = 1.0 - abs((idx3 - idx1) - (idx5 - idx3)) / max(idx5 - idx1, 1)
                    
                    # Overall pattern confidence
                    confidence = (shoulder_similarity * 0.4 + neckline_similarity * 0.4 + time_symmetry * 0.2)
                    max_confidence = max(max_confidence, confidence)
            else:
                # Inverse H&S Pattern
                if p3 < p1 and p3 < p5:
                    # Shoulders should be similar height
                    shoulder_similarity = 1.0 - abs(p1 - p5) / max(p1, p5, 0.0001)
                    
                    # Neckline (p2 and p4) should be similar level
                    neckline_similarity = 1.0 - abs(p2 - p4) / max(p2, p4, 0.0001)
                    
                    # Time symmetry (shoulders equidistant from head)
                    time_symmetry = 1.0 - abs((idx3 - idx1) - (idx5 - idx3)) / max(idx5 - idx1, 1)
                    
                    # Overall pattern confidence
                    confidence = (shoulder_similarity * 0.4 + neckline_similarity * 0.4 + time_symmetry * 0.2)
                    max_confidence = max(max_confidence, confidence)
        
        return max_confidence
        
    except Exception as e:
        log(f"Error in head and shoulders detection: {e}")
        return 0.0

def detect_triangle_patterns(prices, highs, lows):
    """
    Detect ascending, descending, and symmetrical triangle patterns.
    
    Args:
        prices (torch.Tensor): Normalized price series.
        highs (torch.Tensor): High prices.
        lows (torch.Tensor): Low prices.
        
    Returns:
        dict: Dictionary of triangle patterns with their confidence scores.
    """
    if len(prices) < 20:
        return {}
    
    try:
        # Find peaks (for upper trendline) and troughs (for lower trendline)
        peaks = []
        troughs = []
        
        min_gap = 3
        
        for i in range(min_gap, len(prices) - min_gap):
            # Find peaks
            if all(prices[i] > prices[i-j] for j in range(1, min_gap+1)) and \
               all(prices[i] > prices[i+j] for j in range(1, min_gap+1)):
                peaks.append((i, highs[i].item()))
            
            # Find troughs
            if all(prices[i] < prices[i-j] for j in range(1, min_gap+1)) and \
               all(prices[i] < prices[i+j] for j in range(1, min_gap+1)):
                troughs.append((i, lows[i].item()))
        
        # Need at least 2 peaks and 2 troughs
        if len(peaks) < 2 or len(troughs) < 2:
            return {}
        
        # Calculate trendlines
        # Linear regression would be ideal, but we'll use a simpler approach
        # Find two most recent peaks and two most recent troughs
        if len(peaks) >= 2:
            peaks.sort(key=lambda x: x[0])  # Sort by index
            recent_peaks = peaks[-2:]
            
            # Calculate upper trendline slope
            x1, y1 = recent_peaks[0]
            x2, y2 = recent_peaks[1]
            upper_slope = (y2 - y1) / max(1, (x2 - x1))
        else:
            upper_slope = 0
        
        if len(troughs) >= 2:
            troughs.sort(key=lambda x: x[0])  # Sort by index
            recent_troughs = troughs[-2:]
            
            # Calculate lower trendline slope
            x1, y1 = recent_troughs[0]
            x2, y2 = recent_troughs[1]
            lower_slope = (y2 - y1) / max(1, (x2 - x1))
        else:
            lower_slope = 0
        
        # Detect triangle patterns based on slope directions
        patterns = {}
        
        # Ascending Triangle: flat upper, rising lower
        if abs(upper_slope) < 0.001 and lower_slope > 0.001:
            # Calculate confidence based on quality of the pattern
            # Stronger if we have more touchpoints
            touchpoints = min(len(peaks), len(troughs))
            confidence = min(1.0, 0.5 + touchpoints * 0.1)
            patterns["ascending_triangle"] = confidence
        
        # Descending Triangle: flat lower, falling upper
        if abs(lower_slope) < 0.001 and upper_slope < -0.001:
            touchpoints = min(len(peaks), len(troughs))
            confidence = min(1.0, 0.5 + touchpoints * 0.1)
            patterns["descending_triangle"] = confidence
        
        # Symmetrical Triangle: upper falling, lower rising
        if upper_slope < -0.001 and lower_slope > 0.001:
            # More confidence if slopes are similar in magnitude
            slope_ratio = min(abs(upper_slope), abs(lower_slope)) / max(abs(upper_slope), abs(lower_slope))
            touchpoints = min(len(peaks), len(troughs))
            confidence = min(1.0, 0.4 + touchpoints * 0.1 + slope_ratio * 0.3)
            patterns["symmetrical_triangle"] = confidence
        
        return patterns
        
    except Exception as e:
        log(f"Error in triangle pattern detection: {e}")
        return {}

def detect_flag_patterns(prices, volume):
    """
    Detect flag and pennant patterns with enhanced accuracy.
    
    Args:
        prices (torch.Tensor): Normalized price series.
        volume (torch.Tensor): Volume data.
        
    Returns:
        dict: Dictionary of flag patterns with their confidence scores.
    """
    if len(prices) < 15:
        return {}
    
    try:
        patterns = {}
        
        # For flags, we need:
        # 1. A sharp price move (flag pole)
        # 2. A consolidation period against the trend (flag)
        # 3. Usually volume decreases during consolidation
        
        # Check for bullish flag
        # Look for an upward sharp move followed by a slight downward consolidation
        first_half = prices[:len(prices)//2]
        second_half = prices[len(prices)//2:]
        
        # Check for flag pole (sharp move up)
        first_half_change = (first_half[-1] - first_half[0]) / max(0.0001, first_half[0])
        
        # Check for flag (consolidation)
        second_half_change = (second_half[-1] - second_half[0]) / max(0.0001, second_half[0])
        
        # Check volume profile
        first_volume = volume[:len(volume)//2]
        second_volume = volume[len(volume)//2:]
        volume_ratio = torch.mean(second_volume) / torch.mean(first_volume) if torch.mean(first_volume) > 0 else 1.0
        
        # Bullish Flag
        if first_half_change > 0.03 and -0.02 < second_half_change < 0.01 and volume_ratio < 0.9:
            # Calculate confidence
            pole_quality = min(1.0, first_half_change * 10)  # Strong pole gives higher confidence
            flag_quality = 1.0 - abs(second_half_change * 20)  # Flag should be relatively flat
            volume_quality = min(1.0, max(0.0, 1.2 - volume_ratio))  # Decreasing volume gives higher confidence
            
            confidence = (pole_quality * 0.4 + flag_quality * 0.4 + volume_quality * 0.2)
            patterns["bullish_flag"] = confidence
        
        # Bearish Flag (reverse logic)
        if first_half_change < -0.03 and 0.01 > second_half_change > -0.02 and volume_ratio < 0.9:
            pole_quality = min(1.0, abs(first_half_change) * 10)
            flag_quality = 1.0 - abs(second_half_change * 20)
            volume_quality = min(1.0, max(0.0, 1.2 - volume_ratio))
            
            confidence = (pole_quality * 0.4 + flag_quality * 0.4 + volume_quality * 0.2)
            patterns["bearish_flag"] = confidence
        
        # Pennant patterns are similar to flags but with converging trendlines
        # Check for converging price action in the second half
        if len(second_half) >= 10:
            # Calculate price volatility in the second half
            volatility = torch.std(second_half) / torch.mean(second_half)
            
            # Decreasing volatility suggests a pennant
            if (first_half_change > 0.03 or first_half_change < -0.03) and volatility < 0.02 and volume_ratio < 0.8:
                confidence = min(1.0, 0.5 + abs(first_half_change) * 5 + (1.0 - volatility * 30) * 0.3)
                
                if first_half_change > 0:
                    patterns["bullish_pennant"] = confidence
                else:
                    patterns["bearish_pennant"] = confidence
        
        return patterns
        
    except Exception as e:
        log(f"Error in flag pattern detection: {e}")
        return {}

def detect_harmonic_patterns(prices, highs, lows):
    """
    Detect harmonic patterns like Gartley, Butterfly, and Bat patterns.
    These patterns use specific Fibonacci ratios between points.
    
    Args:
        prices (torch.Tensor): Price series.
        highs (torch.Tensor): High prices.
        lows (torch.Tensor): Low prices.
        
    Returns:
        dict: Dictionary of harmonic patterns with their confidence scores.
    """
    if len(prices) < 30:
        return {}
    
    try:
        # Find significant pivots (swing highs and lows)
        pivots = find_significant_pivots(prices, highs, lows)
        
        # Need at least 4 pivots for harmonic patterns
        if len(pivots) < 4:
            return {}
        
        patterns = {}
        
        # Check the most recent 5 pivots for harmonic patterns
        recent_pivots = pivots[-5:]
        
        # Gartley Pattern: XA=AB*0.618, BC=AB*0.382-0.886, CD=BC*1.272-1.618
        # Bat Pattern: XA=AB*0.382-0.5, BC=AB*0.382-0.886, CD=BC*1.618-2.618
        # Butterfly Pattern: XA=AB*0.786, BC=AB*0.382-0.886, CD=BC*1.618-2.618
        
        # Extract pivot points
        if len(recent_pivots) >= 5:
            X, A, B, C, D = recent_pivots[-5:]
            
            # Calculate key ratios
            XA = abs(A - X)
            AB = abs(B - A)
            BC = abs(C - B)
            CD = abs(D - C)
            
            # Check for Gartley Pattern 
            # Gartley requires specific Fibonacci ratios
            # AB should be 0.618 of XA
            ab_xa_ratio = AB / XA if XA > 0 else 0
            # BC should be 0.382-0.886 of AB
            bc_ab_ratio = BC / AB if AB > 0 else 0
            # CD should be 1.272-1.618 of BC
            cd_bc_ratio = CD / BC if BC > 0 else 0
            
            # Check if the ratios match Gartley pattern
            if 0.568 < ab_xa_ratio < 0.668 and 0.332 < bc_ab_ratio < 0.936 and 1.222 < cd_bc_ratio < 1.668:
                # Calculate confidence based on how close the ratios are to the ideal
                gartley_confidence = calculate_harmonic_confidence(
                    ab_xa_ratio, 0.618, 0.05, 
                    bc_ab_ratio, 0.618, 0.3, 
                    cd_bc_ratio, 1.272, 0.2
                )
                if gartley_confidence > 0.6:
                    patterns["gartley"] = gartley_confidence
            
            # Check for Bat Pattern
            if 0.332 < ab_xa_ratio < 0.55 and 0.332 < bc_ab_ratio < 0.936 and 1.568 < cd_bc_ratio < 2.618:
                bat_confidence = calculate_harmonic_confidence(
                    ab_xa_ratio, 0.5, 0.1,
                    bc_ab_ratio, 0.618, 0.3,
                    cd_bc_ratio, 2.0, 0.5
                )
                if bat_confidence > 0.6:
                    patterns["bat"] = bat_confidence
            
            # Check for Butterfly Pattern
            if 0.736 < ab_xa_ratio < 0.836 and 0.332 < bc_ab_ratio < 0.936 and 1.568 < cd_bc_ratio < 2.618:
                butterfly_confidence = calculate_harmonic_confidence(
                    ab_xa_ratio, 0.786, 0.05,
                    bc_ab_ratio, 0.618, 0.3,
                    cd_bc_ratio, 1.618, 0.5
                )
                if butterfly_confidence > 0.6:
                    patterns["butterfly"] = butterfly_confidence
        
        return patterns
        
    except Exception as e:
        log(f"Error in harmonic pattern detection: {e}")
        return {}

def find_significant_pivots(prices, highs, lows, min_gap=5):
    """
    Find significant pivot points (swing highs and lows) in the price series.
    
    Args:
        prices (torch.Tensor): Price series.
        highs (torch.Tensor): High prices.
        lows (torch.Tensor): Low prices.
        min_gap (int): Minimum number of bars between pivots.
        
    Returns:
        list: List of significant pivot indices.
    """
    pivots = []
    
    for i in range(min_gap, len(prices) - min_gap):
        # Check for swing high
        if all(highs[i] > highs[i-j] for j in range(1, min_gap+1)) and \
           all(highs[i] > highs[i+j] for j in range(1, min_gap+1)):
            pivots.append(i)
        
        # Check for swing low
        elif all(lows[i] < lows[i-j] for j in range(1, min_gap+1)) and \
             all(lows[i] < lows[i+j] for j in range(1, min_gap+1)):
            pivots.append(i)
    
    # Sort pivots by time
    pivots.sort()
    
    return pivots

def calculate_harmonic_confidence(ratio1, ideal1, tol1, ratio2, ideal2, tol2, ratio3, ideal3, tol3):
    """
    Calculate confidence score for harmonic patterns based on how close the ratios
    are to the ideal Fibonacci values.
    
    Args:
        ratio1, ratio2, ratio3: Actual ratios
        ideal1, ideal2, ideal3: Ideal Fibonacci values
        tol1, tol2, tol3: Tolerance values for each ratio
        
    Returns:
        float: Confidence score (0-1)
    """
    # Calculate deviation from ideal values, normalized by tolerance
    dev1 = min(1.0, abs(ratio1 - ideal1) / tol1) if tol1 > 0 else 1.0
    dev2 = min(1.0, abs(ratio2 - ideal2) / tol2) if tol2 > 0 else 1.0
    dev3 = min(1.0, abs(ratio3 - ideal3) / tol3) if tol3 > 0 else 1.0
    
    # Convert deviations to confidences (1.0 = perfect match)
    conf1 = 1.0 - dev1
    conf2 = 1.0 - dev2
    conf3 = 1.0 - dev3
    
    # Weight the ratios (first and last points are more critical)
    weighted_conf = 0.4 * conf1 + 0.2 * conf2 + 0.4 * conf3
    
    return weighted_conf

def identify_key_levels_tensor(ohlcv_tensor, current_step, lookback=500, num_levels=5, device="cpu"):
    """
    Identify key support and resistance levels with improved precision.
    
    Args:
        ohlcv_tensor (torch.Tensor): Tensor of OHLCV data.
        current_step (int): Current time step index.
        lookback (int, optional): Number of bars to look back.
        num_levels (int, optional): Maximum number of levels to identify.
        device (str, optional): Computation device.
        
    Returns:
        dict: Dictionary with support and resistance levels and their strengths.
    """
    try:
        if ohlcv_tensor is None or current_step < 50:
            return {"support": [], "resistance": []}
            
        # Extract window of data
        start_idx = max(0, current_step - lookback)
        window = ohlcv_tensor[start_idx:current_step]
        
        if len(window) < 50:  # Need sufficient data
            return {"support": [], "resistance": []}
        
        # Extract OHLCV components
        close = window[:, 0]
        high = window[:, 2]
        low = window[:, 3]
        volume = window[:, 4]
        
        current_price = close[-1].item()
        
        # Generate price bins (more granular near current price)
        price_range = torch.max(high) - torch.min(low)
        price_min = torch.min(low).item()
        price_max = torch.max(high).item()
        
        # Create more bins around current price
        bin_edges = []
        
        # Bins below current price (more dense near current price)
        price_below_range = current_price - price_min
        if price_below_range > 0:
            # Dense region (50% of bins in 20% of range near current price)
            dense_region_min = max(price_min, current_price - 0.2 * price_below_range)
            
            # Sparse bins for lower region
            if dense_region_min > price_min:
                sparse_bins_below = torch.linspace(price_min, dense_region_min, num_levels // 2, device=device)
                bin_edges.extend(sparse_bins_below.tolist())
            
            # Dense bins near current price (below)
            dense_bins_below = torch.linspace(dense_region_min, current_price, num_levels, device=device)
            bin_edges.extend(dense_bins_below.tolist())
            
        # Bins above current price (more dense near current price)
        price_above_range = price_max - current_price
        if price_above_range > 0:
            # Dense region (50% of bins in 20% of range near current price)
            dense_region_max = min(price_max, current_price + 0.2 * price_above_range)
            
            # Dense bins near current price (above)
            dense_bins_above = torch.linspace(current_price, dense_region_max, num_levels, device=device)
            bin_edges.extend(dense_bins_above.tolist())
            
            # Sparse bins for upper region
            if dense_region_max < price_max:
                sparse_bins_above = torch.linspace(dense_region_max, price_max, num_levels // 2, device=device)
                bin_edges.extend(sparse_bins_above.tolist())
        
        # Ensure we have unique, sorted bins
        bin_edges = sorted(list(set(bin_edges)))
        
        # Count touches at each price level, weighted by volume
        level_touches = {}
        level_volumes = {}
        
        for i in range(len(window)):
            # High & low for the bar
            h = high[i].item()
            l = low[i].item()
            vol = volume[i].item()
            
            # Find which levels this bar's range touched
            for level in bin_edges:
                if l <= level <= h:
                    level_touches[level] = level_touches.get(level, 0) + 1
                    level_volumes[level] = level_volumes.get(level, 0) + vol
        
        # Calculate level strengths based on touch count and volume
        level_strengths = {}
        max_touches = max(level_touches.values()) if level_touches else 1
        max_volume = max(level_volumes.values()) if level_volumes else 1
        
        for level, touches in level_touches.items():
            # Strength based on touch count and volume
            touch_score = touches / max_touches
            volume_score = level_volumes.get(level, 0) / max_volume
            
            # Combine scores with 70% weight to touch count, 30% to volume
            strength = (touch_score * 0.7) + (volume_score * 0.3)
            level_strengths[level] = strength
        
        # Filter and categorize levels into support and resistance
        support_levels = []
        resistance_levels = []
        
        # Only keep levels with significant strength
        significant_levels = [(level, strength) for level, strength in level_strengths.items() 
                              if strength > 0.3]
        
        # Sort by strength, then take top levels
        significant_levels.sort(key=lambda x: x[1], reverse=True)
        top_levels = significant_levels[:num_levels*2]  # Get twice as many for S/R separation
        
        # Separate into support (below price) and resistance (above price)
        for level, strength in top_levels:
            if level < current_price:
                support_levels.append({"price": level, "strength": strength})
            else:
                resistance_levels.append({"price": level, "strength": strength})
        
        # Sort by price (support: high to low, resistance: low to high)
        support_levels.sort(key=lambda x: x["price"], reverse=True)
        resistance_levels.sort(key=lambda x: x["price"])
        
        # Limit to requested number of levels
        support_levels = support_levels[:num_levels]
        resistance_levels = resistance_levels[:num_levels]
        
        return {
            "support": support_levels,
            "resistance": resistance_levels
        }
        
    except Exception as e:
        log(f"Error in identify_key_levels_tensor: {e}")
        return {"support": [], "resistance": []}

def calculate_relative_volume_tensor(ohlcv_tensor, current_step, lookback=144, reference_window=1440, device="cpu"):
    """
    Calculate relative volume compared to historical average.
    
    Args:
        ohlcv_tensor (torch.Tensor): Tensor of OHLCV data (batch, 5).
        current_step (int): Current time step index.
        lookback (int, optional): Recent window to average.
        reference_window (int, optional): Historical window for baseline.
        device (str, optional): Device to run calculations on.
    
    Returns:
        float: Relative volume (1.0 = average, 2.0 = twice average, etc.)
    """
    try:
        if ohlcv_tensor is None or len(ohlcv_tensor) <= current_step:
            return 1.0
            
        # Get recent volume
        recent_start = max(0, current_step - lookback)
        recent_window = ohlcv_tensor[recent_start:current_step]
        
        if len(recent_window) == 0:
            return 1.0
        
        # Get historical volume for reference
        hist_start = max(0, current_step - reference_window)
        hist_end = max(0, current_step - lookback)
        hist_window = ohlcv_tensor[hist_start:hist_end]
        
        # If not enough historical data, use what we have
        if len(hist_window) < 10:
            hist_window = recent_window
        
        try:
            recent_volume = torch.mean(recent_window[:, 4])
            hist_volume = torch.mean(hist_window[:, 4])
            
            # Calculate relative volume
            if hist_volume > 0:
                rel_volume = recent_volume / hist_volume
            else:
                rel_volume = 1.0
        except Exception:
            return 1.0
        
        # Clamp to reasonable range
        rel_volume = max(0.1, min(10.0, rel_volume.item()))
        return rel_volume
    
    except Exception as e:
        log(f"Error in calculate_relative_volume_tensor: {e}")
        return 1.0

def estimate_volatility_tensor(ohlcv_tensor, current_step, lookback=72, device="cpu"):
    """
    Estimate price volatility using tensor operations.
    
    Args:
        ohlcv_tensor (torch.Tensor): Tensor of OHLCV data (batch, 5).
        current_step (int): Current time step index.
        lookback (int, optional): Number of bars to consider.
        device (str, optional): Device to run calculations on.
    
    Returns:
        float: Annualized volatility estimate.
    """
    try:
        if ohlcv_tensor is None or len(ohlcv_tensor) <= current_step:
            return 0.2  # Default medium volatility
            
        start_idx = max(0, current_step - lookback)
        window = ohlcv_tensor[start_idx:current_step]
        
        if len(window) < 2:
            return 0.2
        
        # Calculate log returns
        try:
            close_prices = window[:, 0]
            log_returns = torch.log(close_prices[1:] / close_prices[:-1])
            
            # Standard deviation of returns
            std_dev = torch.std(log_returns)
            
            # Annualize (assuming 5-minute bars, so 288 bars per day * 252 trading days)
            # sqrt(288 * 252) = ~269
            annualized_vol = std_dev * torch.sqrt(torch.tensor(269.0, device=device))
        except Exception:
            return 0.2
        
        # Convert to Python float and ensure reasonable range
        return max(0.05, min(2.0, annualized_vol.item()))
    
    except Exception as e:
        log(f"Error in estimate_volatility_tensor: {e}")
        return 0.2  # Default in case of error

def calculate_trend_metrics(close, high, low):
    """
    Calculate comprehensive trend metrics for market context.
    
    Args:
        close (torch.Tensor): Close prices.
        high (torch.Tensor): High prices.
        low (torch.Tensor): Low prices.
    
    Returns:
        dict: Dictionary of trend metrics.
    """
    try:
        if len(close) < 20:
            return {"trend": 0, "strength": 0.0, "volatility": 0.0}
        
        # Calculate moving averages
        ma10 = torch.mean(close[-10:])
        ma20 = torch.mean(close[-20:])
        if len(close) >= 50:
            ma50 = torch.mean(close[-50:])
        else:
            ma50 = ma20
        
        # Determine trend direction
        trend = 0  # neutral
        if ma10 > ma20 and ma20 > ma50:
            trend = 1  # uptrend
        elif ma10 < ma20 and ma20 < ma50:
            trend = -1  # downtrend
        
        # Calculate trend strength using multiple factors
        # 1. MA alignment strength
        ma_alignment = min(1.0, abs(ma10 - ma50) / (ma50 * 0.1 + 1e-8))
        
        # 2. Consistency of trend (percentage of bars in trend direction)
        if trend == 1:
            consistent_bars = sum(1 for i in range(1, min(20, len(close))) if close[-i] > close[-i-1])
        elif trend == -1:
            consistent_bars = sum(1 for i in range(1, min(20, len(close))) if close[-i] < close[-i-1])
        else:
            consistent_bars = 10  # neutral
        consistency = consistent_bars / min(20, len(close)-1)
        
        # 3. Volatility relative to trend
        returns = torch.tensor([(close[i] - close[i-1]) / close[i-1] for i in range(1, len(close))])
        volatility = torch.std(returns).item()
        
        # Combine factors for overall strength
        strength = (ma_alignment * 0.4 + consistency * 0.4 + (1.0 - min(1.0, volatility * 10)) * 0.2) * abs(trend)
        
        return {
            "trend": trend,
            "strength": strength,
            "volatility": volatility
        }
        
    except Exception as e:
        log(f"Error calculating trend metrics: {e}")
        return {"trend": 0, "strength": 0.0, "volatility": 0.0}

def batch_process_features(ohlcv_tensor, current_indices, lookback=288, device="cpu"):
    """
    Process multiple tensor features in batch for efficiency.
    
    Args:
        ohlcv_tensor (torch.Tensor): Tensor of OHLCV data (batch, 5).
        current_indices (list): List of indices to process.
        lookback (int, optional): Default lookback window.
        device (str, optional): Device to run calculations on.
    
    Returns:
        list: List of feature dictionaries for each index.
    """
    if ohlcv_tensor is None or not current_indices:
        return [{}] * len(current_indices)
    
    batch_features = []
    
    # Process each index
    for idx in current_indices:
        features = {}
        
        # Volume profile
        features['volume_profile'], features['price_min'], features['price_max'] = compute_volume_profile_tensor(
            ohlcv_tensor, idx, lookback=min(lookback, 144), device=device
        )
        
        # Liquidity zones
        features['liquidity_zones'] = identify_liquidity_zones_tensor(
            ohlcv_tensor, idx, lookback=min(lookback, 288), device=device
        )
        
        # Spread estimation
        features['bid_ask_spread'] = estimate_bid_ask_spread_tensor(
            ohlcv_tensor, idx, window_size=min(lookback, 12), device=device
        )
        
        # Volume delta
        features['volume_delta'], features['buy_volume'], features['sell_volume'] = calculate_volume_delta_tensor(
            ohlcv_tensor, idx, window_size=min(lookback, 6), device=device
        )
        
        # Liquidity estimation
        features['market_liquidity'] = estimate_market_liquidity_tensor(
            ohlcv_tensor, idx, window_size=min(lookback, 72), device=device
        )
        
        # Price momentum
        features['momentum'] = calculate_price_momentum_tensor(
            ohlcv_tensor, idx, window_size=min(lookback, 24), device=device
        )
        
        # VWAP
        features['vwap'] = calculate_vwap_tensor(
            ohlcv_tensor, idx, window_size=min(lookback, 144), device=device
        )
        
        # Advanced pattern detection (new enhanced version)
        pattern_results = detect_patterns_tensor(
            ohlcv_tensor, idx, lookback=min(lookback, 100), device=device
        )
        features['pattern_strength'] = pattern_results.get('strength', 0.0)
        features['pattern_trend'] = pattern_results.get('trend', 0)
        features['detected_patterns'] = pattern_results.get('patterns', {})
        
        # Support/Resistance
        features['support'], features['resistance'] = detect_support_resistance_tensor(
            ohlcv_tensor, idx, window_size=min(lookback, 576), device=device
        )
        
        # Price clusters
        features['price_clusters'] = detect_price_clusters_tensor(
            ohlcv_tensor, idx, window_size=min(lookback, 288), device=device
        )
        
        # Relative volume
        features['relative_volume'] = calculate_relative_volume_tensor(
            ohlcv_tensor, idx, lookback=min(lookback, 144), device=device
        )
        
        # Volatility
        features['volatility'] = estimate_volatility_tensor(
            ohlcv_tensor, idx, lookback=min(lookback, 72), device=device
        )
        
        # Key levels (enhanced version)
        features['key_levels'] = identify_key_levels_tensor(
            ohlcv_tensor, idx, lookback=min(lookback, 500), device=device
        )
        
        # Harmonic patterns (new advanced feature)
        if len(ohlcv_tensor) > 0 and idx > 40:
            window = ohlcv_tensor[max(0, idx-40):idx]
            if len(window) >= 40:
                try:
                    harmonic_patterns = detect_harmonic_patterns(
                        window[:, 0], window[:, 2], window[:, 3]
                    )
                    features['harmonic_patterns'] = harmonic_patterns
                except Exception as e:
                    log(f"Error calculating harmonic patterns: {e}")
                    features['harmonic_patterns'] = {}
        
        batch_features.append(features)
    
    return batch_features

def detect_support_resistance_tensor(ohlcv_tensor, current_step, window_size=576, num_levels=5, device="cpu"):
    """
    Detect support and resistance levels using tensor operations.
    
    Args:
        ohlcv_tensor (torch.Tensor): Tensor of OHLCV data (batch, 5).
        current_step (int): Current time step index.
        window_size (int, optional): Number of bars to consider.
        num_levels (int, optional): Maximum number of levels to detect.
        device (str, optional): Device to run calculations on.
    
    Returns:
        tuple: (support_levels, resistance_levels) - Lists of detected levels.
    """
    try:
        if ohlcv_tensor is None or len(ohlcv_tensor) <= current_step:
            return [], []
            
        start_idx = max(0, current_step - window_size)
        window = ohlcv_tensor[start_idx:current_step]
        
        if len(window) < 20:  # Need sufficient data
            return [], []
        
        # Extract OHLCV data
        try:
            close = window[:, 0]
            high = window[:, 2]
            low = window[:, 3]
            volume = window[:, 4]
        except Exception:
            return [], []
        
        # Current price for reference
        current_price = ohlcv_tensor[current_step, 0].item()
        
        # Use histogram-based approach
        # 1. Create price bins
        price_min = torch.min(low).item()
        price_max = torch.max(high).item()
        
        # Ensure min and max are different
        if price_max <= price_min:
            price_max = price_min * 1.01
        
        num_bins = min(100, max(20, int((price_max - price_min) / (price_min * 0.001))))
        bins = torch.linspace(price_min, price_max, num_bins, device=device)
        
        # 2. Count touches at each price level
        touches = torch.zeros(num_bins - 1, device=device)
        
        for i in range(len(window)):
            # Consider high and low for each bar
            high_idx = min(num_bins - 2, max(0, int((high[i].item() - price_min) / (price_max - price_min) * (num_bins - 1))))
            low_idx = min(num_bins - 2, max(0, int((low[i].item() - price_min) / (price_max - price_min) * (num_bins - 1))))
            
            # Weight by volume
            touches[high_idx] += volume[i].item() * 0.5
            touches[low_idx] += volume[i].item() * 0.5
        
        # 3. Find local maxima in the touch histogram
        touch_maxima = []
        
        for i in range(1, len(touches) - 1):
            if touches[i] > touches[i-1] and touches[i] > touches[i+1]:
                # Local maximum - this is a potential S/R level
                level_price = price_min + (i + 0.5) * (price_max - price_min) / num_bins
                touch_maxima.append((level_price, touches[i].item()))
        
        # 4. Sort by strength and filter
        touch_maxima.sort(key=lambda x: x[1], reverse=True)
        strongest_levels = [price for price, _ in touch_maxima[:num_levels*2]]  # Get twice as many for S/R separation
        
        # 5. Separate into support and resistance based on current price
        support_levels = sorted([level for level in strongest_levels if level < current_price])
        resistance_levels = sorted([level for level in strongest_levels if level > current_price])
        
        # Limit to requested number
        support_levels = support_levels[-num_levels:] if support_levels else []
        resistance_levels = resistance_levels[:num_levels] if resistance_levels else []
        
        return support_levels, resistance_levels
    
    except Exception as e:
        log(f"Error in detect_support_resistance_tensor: {e}")
        return [], []

def detect_price_clusters_tensor(ohlcv_tensor, current_step, window_size=288, num_clusters=3, device="cpu"):
    """
    Detect price clusters using k-means-inspired approach.
    
    Args:
        ohlcv_tensor (torch.Tensor): Tensor of OHLCV data (batch, 5).
        current_step (int): Current time step index.
        window_size (int, optional): Number of bars to consider.
        num_clusters (int, optional): Number of clusters to detect.
        device (str, optional): Device to run calculations on.
    
    Returns:
        list: List of cluster centers (prices).
    """
    try:
        if ohlcv_tensor is None or len(ohlcv_tensor) <= current_step:
            return []
            
        start_idx = max(0, current_step - window_size)
        window = ohlcv_tensor[start_idx:current_step]
        
        if len(window) < num_clusters * 3:  # Need sufficient data
            return []
        
        # Extract price data (we'll use typical prices)
        try:
            close = window[:, 0]
            high = window[:, 2]
            low = window[:, 3]
            typical_prices = (high + low + close) / 3
        except Exception:
            return []
        
        # Convert to numpy for easier clustering
        price_data = typical_prices.cpu().numpy()
        
        # Simple k-means approach (adapted for 1D data)
        # Initialize centers randomly
        min_price = np.min(price_data)
        max_price = np.max(price_data)
        centers = np.linspace(min_price, max_price, num_clusters)
        
        # Run simplified k-means
        max_iterations = 10
        for _ in range(max_iterations):
            # Assign points to nearest center
            clusters = [[] for _ in range(num_clusters)]
            
            for price in price_data:
                # Find closest center
                closest_idx = np.argmin([abs(price - c) for c in centers])
                clusters[closest_idx].append(price)
            
            # Update centers
            new_centers = []
            for cluster in clusters:
                if cluster:
                    new_centers.append(np.mean(cluster))
                else:
                    # If a cluster is empty, reinitialize randomly
                    new_centers.append(min_price + np.random.random() * (max_price - min_price))
            
            # Check for convergence
            if np.allclose(centers, new_centers, rtol=1e-4):
                break
            
            centers = new_centers
        
        # Sort centers for consistency
        centers.sort()
        return centers.tolist()
    
    except Exception as e:
        log(f"Error in detect_price_clusters_tensor: {e}")
        return []

def analyze_order_flow_tensor(ohlcv_tensor, current_step, window_size=30, device="cpu"):
    """
    Analyze order flow using volume delta and price action.
    
    Args:
        ohlcv_tensor (torch.Tensor): Tensor of OHLCV data (batch, 5).
        current_step (int): Current time step index.
        window_size (int): Number of bars to analyze.
        device (str): Device to run calculations on.
        
    Returns:
        dict: Order flow analysis metrics.
    """
    try:
        if ohlcv_tensor is None or len(ohlcv_tensor) <= current_step:
            return {"imbalance": 0.0, "pressure": 0.0, "exhaustion": 0.0}
            
        start_idx = max(0, current_step - window_size)
        window = ohlcv_tensor[start_idx:current_step]
        
        if len(window) < 5:  # Need some data
            return {"imbalance": 0.0, "pressure": 0.0, "exhaustion": 0.0}
        
        # Extract data
        close = window[:, 0]
        open_price = window[:, 1]
        high = window[:, 2]
        low = window[:, 3]
        volume = window[:, 4]
        
        # Calculate bullish vs bearish volume
        bullish_volume = torch.zeros_like(volume)
        bearish_volume = torch.zeros_like(volume)
        
        for i in range(len(window)):
            price_change = close[i] - open_price[i]
            if price_change > 0:
                bullish_volume[i] = volume[i]
            else:
                bearish_volume[i] = volume[i]
        
        total_bullish = torch.sum(bullish_volume).item()
        total_bearish = torch.sum(bearish_volume).item()
        total_volume = total_bullish + total_bearish
        
        # Calculate buy/sell imbalance (-1.0 to 1.0)
        if total_volume > 0:
            imbalance = (total_bullish - total_bearish) / total_volume
        else:
            imbalance = 0.0
        
        # Calculate buying/selling pressure using price range
        # Higher volume with larger price movement indicates stronger pressure
        bar_ranges = high - low
        pressure = 0.0
        
        if len(window) > 1:
            for i in range(len(window)):
                range_factor = (bar_ranges[i] / torch.mean(bar_ranges)).item()
                vol_factor = (volume[i] / torch.mean(volume)).item()
                
                if close[i] > open_price[i]:  # Bullish
                    pressure += range_factor * vol_factor
                else:  # Bearish
                    pressure -= range_factor * vol_factor
            
            # Normalize pressure
            pressure = torch.tanh(torch.tensor(pressure / len(window))).item()
        
        # Detect volume exhaustion
        # Volume increasing but price movement decreasing suggests exhaustion
        exhaustion = 0.0
        
        if len(window) > 10:
            recent_vol = volume[-5:]
            older_vol = volume[-10:-5]
            
            recent_range = torch.mean(high[-5:] - low[-5:])
            older_range = torch.mean(high[-10:-5] - low[-10:-5])
            
            vol_increasing = torch.mean(recent_vol) > torch.mean(older_vol) * 1.1  # 10% increase
            range_decreasing = recent_range < older_range * 0.9  # 10% decrease
            
            if vol_increasing and range_decreasing:
                exhaustion = min(1.0, (torch.mean(recent_vol) / torch.mean(older_vol) - 1.0).item() * 3)
        
        return {
            "imbalance": imbalance,
            "pressure": pressure,
            "exhaustion": exhaustion
        }
    
    except Exception as e:
        log(f"Error in analyze_order_flow_tensor: {e}")
        return {"imbalance": 0.0, "pressure": 0.0, "exhaustion": 0.0}

def detect_divergence_patterns(ohlcv_tensor, indicator_tensor, current_step, window_size=30, device="cpu"):
    """
    Detect divergence patterns between price and indicators (e.g., RSI).
    
    Args:
        ohlcv_tensor (torch.Tensor): Tensor of OHLCV data.
        indicator_tensor (torch.Tensor): Tensor of indicator values.
        current_step (int): Current time step index.
        window_size (int): Number of bars to analyze.
        device (str): Device to run calculations on.
        
    Returns:
        dict: Detected divergence patterns.
    """
    try:
        if ohlcv_tensor is None or indicator_tensor is None or len(ohlcv_tensor) <= current_step:
            return {"bullish": 0.0, "bearish": 0.0}
        
        start_idx = max(0, current_step - window_size)
        
        # Get price and indicator data
        price_window = ohlcv_tensor[start_idx:current_step, 0]  # Close prices
        indicator_window = indicator_tensor[start_idx:current_step]
        
        if len(price_window) < 10:  # Need enough data
            return {"bullish": 0.0, "bearish": 0.0}
        
        # Find local extrema in price
        price_highs = []
        price_lows = []
        
        # Find local extrema in indicator
        indicator_highs = []
        indicator_lows = []
        
        min_gap = 3  # Minimum distance between extrema
        
        # Find extrema in price
        for i in range(min_gap, len(price_window) - min_gap):
            # Price high
            if all(price_window[i] > price_window[i-j] for j in range(1, min_gap+1)) and \
               all(price_window[i] > price_window[i+j] for j in range(1, min_gap+1)):
                price_highs.append((i, price_window[i].item()))
            
            # Price low
            if all(price_window[i] < price_window[i-j] for j in range(1, min_gap+1)) and \
               all(price_window[i] < price_window[i+j] for j in range(1, min_gap+1)):
                price_lows.append((i, price_window[i].item()))
        
        # Find extrema in indicator
        for i in range(min_gap, len(indicator_window) - min_gap):
            # Indicator high
            if all(indicator_window[i] > indicator_window[i-j] for j in range(1, min_gap+1)) and \
               all(indicator_window[i] > indicator_window[i+j] for j in range(1, min_gap+1)):
                indicator_highs.append((i, indicator_window[i].item()))
            
            # Indicator low
            if all(indicator_window[i] < indicator_window[i-j] for j in range(1, min_gap+1)) and \
               all(indicator_window[i] < indicator_window[i+j] for j in range(1, min_gap+1)):
                indicator_lows.append((i, indicator_window[i].item()))
        
        # Need at least 2 extrema of each type
        if len(price_highs) < 2 or len(indicator_highs) < 2 or len(price_lows) < 2 or len(indicator_lows) < 2:
            return {"bullish": 0.0, "bearish": 0.0}
        
        # Bullish divergence: price makes lower lows, indicator makes higher lows
        bullish_divergence = 0.0
        
        if len(price_lows) >= 2 and len(indicator_lows) >= 2:
            # Sort by time
            price_lows.sort(key=lambda x: x[0])
            indicator_lows.sort(key=lambda x: x[0])
            
            # Get the two most recent lows
            recent_price_lows = price_lows[-2:]
            recent_ind_lows = indicator_lows[-2:]
            
            # Check if price makes lower low but indicator makes higher low
            if recent_price_lows[1][1] < recent_price_lows[0][1] and recent_ind_lows[1][1] > recent_ind_lows[0][1]:
                # Calculate how strong the divergence is
                price_decline = (recent_price_lows[0][1] - recent_price_lows[1][1]) / recent_price_lows[0][1]
                ind_improvement = (recent_ind_lows[1][1] - recent_ind_lows[0][1]) / max(0.0001, abs(recent_ind_lows[0][1]))
                
                # Combine for overall divergence strength
                bullish_divergence = min(1.0, (price_decline + ind_improvement) / 2 * 5)  # Scale up for visibility
        
        # Bearish divergence: price makes higher highs, indicator makes lower highs
        bearish_divergence = 0.0
        
        if len(price_highs) >= 2 and len(indicator_highs) >= 2:
            # Sort by time
            price_highs.sort(key=lambda x: x[0])
            indicator_highs.sort(key=lambda x: x[0])
            
            # Get the two most recent highs
            recent_price_highs = price_highs[-2:]
            recent_ind_highs = indicator_highs[-2:]
            
            # Check if price makes higher high but indicator makes lower high
            if recent_price_highs[1][1] > recent_price_highs[0][1] and recent_ind_highs[1][1] < recent_ind_highs[0][1]:
                # Calculate how strong the divergence is
                price_increase = (recent_price_highs[1][1] - recent_price_highs[0][1]) / recent_price_highs[0][1]
                ind_deterioration = (recent_ind_highs[0][1] - recent_ind_highs[1][1]) / max(0.0001, abs(recent_ind_highs[0][1]))
                
                # Combine for overall divergence strength
                bearish_divergence = min(1.0, (price_increase + ind_deterioration) / 2 * 5)  # Scale up for visibility
        
        return {
            "bullish": bullish_divergence,
            "bearish": bearish_divergence
        }
        
    except Exception as e:
        log(f"Error in detect_divergence_patterns: {e}")
        return {"bullish": 0.0, "bearish": 0.0}

# =====================================================================
# Fractal Pattern Recognition and Analysis Functions
# =====================================================================

def compute_fractal_dimension_tensor(price_tensor, window_size=30, step_size=1, device="cpu"):
    """
    Compute fractal dimension (Hurst exponent) of price series using tensor operations.
    
    The Hurst exponent measures the long-term memory of a time series:
    - H < 0.5: Anti-persistent (mean-reverting) behavior
    - H = 0.5: Random walk (Brownian motion)
    - H > 0.5: Persistent (trending) behavior
    
    Args:
        price_tensor (torch.Tensor): Tensor of price data [sequence_length]
        window_size (int): Maximum window size for calculation
        step_size (int): Step size for window
        device (str): Device to use for calculation
        
    Returns:
        torch.Tensor: Hurst exponent value (scalar tensor)
    """
    if price_tensor is None or len(price_tensor) < window_size + 1:
        return torch.tensor(0.5, device=device)  # Default to random walk assumption
    
    try:
        # Convert to returns
        returns = torch.log(price_tensor[1:] / price_tensor[:-1])
        n = len(returns)
        
        if n < 10:  # Need minimum data for meaningful calculation
            return torch.tensor(0.5, device=device)
        
        # Generate different window sizes
        window_sizes = torch.arange(2, min(window_size, n//4), step_size, device=device)
        
        # Prepare tensor for R/S values
        rs_values = torch.zeros_like(window_sizes, dtype=torch.float32)
        
        # Calculate R/S for each window size
        for i, w in enumerate(window_sizes):
            w = int(w.item())
            num_windows = n // w
            
            if num_windows < 1:
                continue
                
            # Process all windows of size w
            rs_w = torch.zeros(num_windows, device=device)
            
            for j in range(num_windows):
                window = returns[j*w:(j+1)*w]
                mean_w = torch.mean(window)
                
                # Calculate cumulative deviation
                cumdev = torch.cumsum(window - mean_w, dim=0)
                
                # Calculate R (max-min range of cumulative deviation)
                r = torch.max(cumdev) - torch.min(cumdev)
                
                # Calculate S (standard deviation)
                s = torch.std(window)
                
                # R/S value
                if s > 0:
                    rs_w[j] = r / s
                else:
                    rs_w[j] = 1.0  # Default when std is zero
            
            # Average R/S for this window size
            rs_values[i] = torch.mean(rs_w)
        
        # Filter out any zeros or NaNs
        valid_mask = (rs_values > 0) & ~torch.isnan(rs_values)
        window_sizes = window_sizes[valid_mask]
        rs_values = rs_values[valid_mask]
        
        if len(window_sizes) < 2:
            return torch.tensor(0.5, device=device)
        
        # Log-log regression to find H (Hurst exponent)
        log_window = torch.log(window_sizes.float())
        log_rs = torch.log(rs_values)
        
        # Calculate the slope (Hurst exponent)
        mean_log_w = torch.mean(log_window)
        mean_log_rs = torch.mean(log_rs)
        
        numerator = torch.sum((log_window - mean_log_w) * (log_rs - mean_log_rs))
        denominator = torch.sum((log_window - mean_log_w) ** 2)
        
        h = numerator / denominator if denominator > 0 else torch.tensor(0.5, device=device)
        
        # Clamp to valid range
        h = torch.clamp(h, 0.0, 1.0)
        
        return h
        
    except Exception as e:
        log(f"Error in Hurst exponent calculation: {str(e)}")
        return torch.tensor(0.5, device=device)

def detect_elliott_wave_pattern_tensor(price_tensor, window_size=144, min_pattern_size=20, max_pattern_size=100, device="cpu"):
    """
    Detect potential Elliott Wave patterns in price data using tensor operations.
    
    Args:
        price_tensor (torch.Tensor): Tensor of price data [sequence_length]
        window_size (int): Window size to analyze
        min_pattern_size (int): Minimum size for pattern detection
        max_pattern_size (int): Maximum size for pattern detection
        device (str): Device to use for calculation
        
    Returns:
        dict: Dictionary containing pattern information with keys:
            - 'detected': Boolean indicating if pattern was detected
            - 'confidence': Confidence score (0-1)
            - 'wave_points': Indices of wave points
            - 'pattern_type': Type of pattern detected ('impulse', 'correction', 'unknown')
    """
    if price_tensor is None or len(price_tensor) < window_size:
        return {
            'detected': False,
            'confidence': 0.0,
            'wave_points': [],
            'pattern_type': 'unknown'
        }
    
    try:
        # Get subset of data to analyze
        n = len(price_tensor)
        start_idx = max(0, n - window_size)
        prices = price_tensor[start_idx:n].clone().detach()
        
        # Find local extrema (peaks and troughs)
        diff = torch.diff(prices)
        zero_crossings = torch.where(diff[:-1] * diff[1:] <= 0)[0] + 1
        
        # Additional filter: ensure significant extrema (avoid noise)
        price_std = torch.std(prices)
        min_move = 0.1 * price_std  # Minimum price movement for significant extrema
        
        filtered_extrema = []
        prev_price = None
        for idx in zero_crossings:
            if prev_price is None or abs(prices[idx] - prev_price) >= min_move:
                filtered_extrema.append(idx.item())
                prev_price = prices[idx]
        
        # Need at least 9 points for a potential Elliott pattern
        if len(filtered_extrema) < 9:
            return {
                'detected': False,
                'confidence': 0.0,
                'wave_points': [],
                'pattern_type': 'unknown'
            }
        
        # Impulse wave criteria
        # - Wave 2 doesn't retrace more than 100% of Wave 1
        # - Wave 3 is typically the longest
        # - Wave 4 doesn't overlap with Wave 1
        # - Wave 5 is typically similar in length to Wave 1

        # Slide window over extrema points to find potential patterns
        best_confidence = 0.0
        best_pattern = {
            'detected': False,
            'confidence': 0.0,
            'wave_points': [],
            'pattern_type': 'unknown'
        }
        
        for i in range(len(filtered_extrema) - 8):
            # Extract potential 5-wave pattern (need 9 points for complete pattern)
            pattern_indices = filtered_extrema[i:i+9]
            
            # Skip if pattern size doesn't meet requirements
            pattern_size = pattern_indices[-1] - pattern_indices[0]
            if pattern_size < min_pattern_size or pattern_size > max_pattern_size:
                continue
            
            pattern_prices = prices[pattern_indices]
            
            # Check if this could be an impulse pattern
            # Wave directions (1,3,5 up in bull market, 2,4 down)
            is_bullish = pattern_prices[8] > pattern_prices[0]
            
            wave_directions = [1, -1, 1, -1, 1] if is_bullish else [-1, 1, -1, 1, -1]
            wave_points = []
            
            for j in range(5):
                start_idx = j * 2
                end_idx = j * 2 + 2
                
                # Get wave start, mid and end points
                wave_start = pattern_indices[start_idx]
                wave_end = pattern_indices[end_idx]
                
                # For each wave, see if the direction matches expected
                wave_move = pattern_prices[end_idx] - pattern_prices[start_idx]
                expected_dir = wave_directions[j]
                
                # Check if direction matches expectation
                actual_dir = 1 if wave_move > 0 else -1
                wave_points.append((wave_start, wave_end))
                
                # Direction mismatch reduces confidence
                if actual_dir != expected_dir:
                    wave_directions[j] = 0  # Mark this wave as invalid
            
            # Calculate confidence score based on pattern rules
            confidence = 0.0
            
            # Rule 1: Wave 2 doesn't retrace more than 100% of Wave 1
            wave1_size = abs(pattern_prices[2] - pattern_prices[0])
            wave2_size = abs(pattern_prices[4] - pattern_prices[2])
            if is_bullish and pattern_prices[4] >= pattern_prices[0] or not is_bullish and pattern_prices[4] <= pattern_prices[0]:
                rule1_score = 0.0  # Wave 2 fully retraced Wave 1 (invalid)
            else:
                retracement = min(1.0, wave2_size / wave1_size if wave1_size > 0 else 1.0)
                rule1_score = 1.0 - retracement  # Higher score for smaller retracement
            
            # Rule 2: Wave 3 is typically the longest
            wave3_size = abs(pattern_prices[6] - pattern_prices[4])
            wave_sizes = torch.tensor([wave1_size, wave3_size, abs(pattern_prices[8] - pattern_prices[6])])
            
            if wave3_size == torch.max(wave_sizes):
                rule2_score = 1.0
            else:
                # Calculate how close wave 3 is to being the longest
                max_size = torch.max(wave_sizes)
                rule2_score = wave3_size / max_size if max_size > 0 else 0.0
            
            # Rule 3: Wave 4 doesn't overlap with Wave 1
            if (is_bullish and torch.min(pattern_prices[6:8]) > torch.max(pattern_prices[0:2])) or \
               (not is_bullish and torch.max(pattern_prices[6:8]) < torch.min(pattern_prices[0:2])):
                rule3_score = 1.0
            else:
                rule3_score = 0.0  # Wave 4 overlapped Wave 1 (invalid for impulse)
            
            # Rule 4: Wave 5 should be proportional to Wave 1
            wave5_size = abs(pattern_prices[8] - pattern_prices[6])
            size_ratio = min(wave5_size / wave1_size if wave1_size > 0 else 0, 
                            wave1_size / wave5_size if wave5_size > 0 else 0)
            rule4_score = min(1.0, size_ratio)
            
            # Fibonacci relationships add to confidence
            fib_ratio1 = wave3_size / wave1_size if wave1_size > 0 else 0
            fib_ratio2 = wave5_size / wave3_size if wave3_size > 0 else 0
            
            # Check how close ratios are to Fibonacci numbers (0.618, 1.0, 1.618, 2.618)
            fib_targets = torch.tensor([0.618, 1.0, 1.618, 2.618], device=device)
            fib_score1 = torch.min(torch.abs(fib_targets - fib_ratio1)).item() / 2.618  # Normalize
            fib_score2 = torch.min(torch.abs(fib_targets - fib_ratio2)).item() / 2.618  # Normalize
            fib_score = 1.0 - (fib_score1 + fib_score2) / 2  # Higher score for closer to Fibonacci
            
            # Calculate overall confidence
            wave_scores = torch.tensor([rule1_score, rule2_score, rule3_score, rule4_score, fib_score])
            weights = torch.tensor([0.2, 0.3, 0.2, 0.15, 0.15])  # Weight by importance
            confidence = torch.sum(wave_scores * weights).item()
            
            # Determine pattern type
            if confidence > 0.5 and rule3_score > 0.0:
                pattern_type = 'impulse'
            else:
                pattern_type = 'correction'  # Might be a corrective pattern
                # Slightly adjust confidence for corrective pattern
                confidence *= 0.8
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_pattern = {
                    'detected': confidence > 0.6,  # Threshold for detection
                    'confidence': confidence,
                    'wave_points': [(start_idx + p[0], start_idx + p[1]) for p in wave_points],
                    'pattern_type': pattern_type
                }
        
        return best_pattern
        
    except Exception as e:
        log(f"Error in Elliott Wave detection: {str(e)}")
        return {
            'detected': False,
            'confidence': 0.0,
            'wave_points': [],
            'pattern_type': 'unknown'
        }

def compute_market_fractals_tensor(high_tensor, low_tensor, window_size=5, device="cpu"):
    """
    Identify Williams' fractals in market data using tensor operations.
    Williams' fractals identify potential reversal points in the market.
    
    Args:
        high_tensor (torch.Tensor): Tensor of high prices
        low_tensor (torch.Tensor): Tensor of low prices
        window_size (int): Window size for fractal identification (odd number)
        device (str): Device to use for calculation
        
    Returns:
        tuple: (bullish_fractals, bearish_fractals) - Boolean tensors indicating fractal points
    """
    if high_tensor is None or low_tensor is None:
        return None, None
        
    try:
        n = len(high_tensor)
        if n < window_size:
            return torch.zeros(n, dtype=torch.bool, device=device), torch.zeros(n, dtype=torch.bool, device=device)
        
        # Ensure window_size is odd
        if window_size % 2 == 0:
            window_size += 1
        
        half_window = window_size // 2
        
        # Initialize result tensors
        bullish_fractals = torch.zeros(n, dtype=torch.bool, device=device)
        bearish_fractals = torch.zeros(n, dtype=torch.bool, device=device)
        
        # Calculate fractals using tensor operations
        for i in range(half_window, n - half_window):
            # Bullish fractal: current low is lowest in the window
            window_lows = low_tensor[i-half_window:i+half_window+1]
            if low_tensor[i] == torch.min(window_lows):
                # Ensure it's a true fractal (exactly at the center)
                if all(low_tensor[i] < low_tensor[i-j] for j in range(1, half_window+1)) and \
                   all(low_tensor[i] < low_tensor[i+j] for j in range(1, half_window+1)):
                    bullish_fractals[i] = True
            
            # Bearish fractal: current high is highest in the window
            window_highs = high_tensor[i-half_window:i+half_window+1]
            if high_tensor[i] == torch.max(window_highs):
                # Ensure it's a true fractal (exactly at the center)
                if all(high_tensor[i] > high_tensor[i-j] for j in range(1, half_window+1)) and \
                   all(high_tensor[i] > high_tensor[i+j] for j in range(1, half_window+1)):
                    bearish_fractals[i] = True
        
        return bullish_fractals, bearish_fractals
        
    except Exception as e:
        log(f"Error in fractal calculation: {str(e)}")
        return torch.zeros(n, dtype=torch.bool, device=device), torch.zeros(n, dtype=torch.bool, device=device)

def compute_fractal_support_resistance_tensor(price_tensor, fractals_tensor, lookback=100, strength_threshold=2, device="cpu"):
    """
    Calculate fractal-based support and resistance levels with strength assessment.
    
    Args:
        price_tensor (torch.Tensor): Tensor of price data
        fractals_tensor (torch.Tensor): Boolean tensor indicating fractal points
        lookback (int): Maximum lookback period
        strength_threshold (int): Minimum level touches to consider strong S/R
        device (str): Device to use for calculation
        
    Returns:
        tuple: (support_levels, resistance_levels, strengths) - Tensors of levels and strengths
    """
    try:
        n = len(price_tensor)
        if n < lookback or not torch.any(fractals_tensor):
            return torch.tensor([], device=device), torch.tensor([], device=device), torch.tensor([], device=device)
        
        # Find all fractal points within lookback
        start_idx = max(0, n - lookback)
        fractal_indices = torch.where(fractals_tensor[start_idx:n])[0] + start_idx
        
        if len(fractal_indices) == 0:
            return torch.tensor([], device=device), torch.tensor([], device=device), torch.tensor([], device=device)
        
        # Extract prices at fractal points
        fractal_prices = price_tensor[fractal_indices]
        
        # Cluster similar price levels (within 0.5% of each other)
        clustered_levels = []
        clustered_indices = []
        cluster_touches = []
        
        for i, price in enumerate(fractal_prices):
            found_cluster = False
            for j, (cluster_price, indices, touches) in enumerate(zip(clustered_levels, clustered_indices, cluster_touches)):
                # If price is within 0.5% of a cluster, add to that cluster
                if abs(price - cluster_price) / cluster_price < 0.005:
                    clustered_indices[j].append(fractal_indices[i].item())
                    clustered_levels[j] = (cluster_price * len(indices) + price) / (len(indices) + 1)  # Update average
                    touches[0] += 1  # Increment touch count
                    found_cluster = True
                    break
            
            if not found_cluster:
                clustered_levels.append(price)
                clustered_indices.append([fractal_indices[i].item()])
                cluster_touches.append([1])  # Initialize touch count
        
        # Assess strength of each level by interactions with price
        for i, (level, indices) in enumerate(zip(clustered_levels, clustered_indices)):
            # Check how many times price approached within 0.5% but didn't cross
            bounces = 0
            for j in range(1, n - 1):
                if j not in indices:  # Don't count the fractal points themselves
                    # Check if price approached the level from below
                    approached_from_below = price_tensor[j-1] < level * 0.995 and price_tensor[j] > level * 0.995
                    
                    # Check if price approached the level from above
                    approached_from_above = price_tensor[j-1] > level * 1.005 and price_tensor[j] < level * 1.005
                    
                    # Check if price didn't cross the level
                    didnt_cross = (price_tensor[j+1] - level) * (price_tensor[j] - level) > 0
                    
                    if (approached_from_below or approached_from_above) and didnt_cross:
                        bounces += 1
            
            # Update strength based on bounces
            cluster_touches[i][0] += bounces
        
        # Convert to tensors
        levels = torch.tensor(clustered_levels, device=device)
        strengths = torch.tensor([touches[0] for touches in cluster_touches], device=device)
        
        # Sort levels
        sorted_indices = torch.argsort(levels)
        sorted_levels = levels[sorted_indices]
        sorted_strengths = strengths[sorted_indices]
        
        # Separate support and resistance based on current price
        current_price = price_tensor[-1]
        
        support_mask = sorted_levels < current_price
        resistance_mask = sorted_levels > current_price
        
        support_levels = sorted_levels[support_mask]
        support_strengths = sorted_strengths[support_mask]
        
        resistance_levels = sorted_levels[resistance_mask]
        resistance_strengths = sorted_strengths[resistance_mask]
        
        # Filter by strength threshold
        strong_support_mask = support_strengths >= strength_threshold
        strong_resistance_mask = resistance_strengths >= strength_threshold
        
        return support_levels[strong_support_mask], resistance_levels[strong_resistance_mask], \
               torch.cat([support_strengths[strong_support_mask], resistance_strengths[strong_resistance_mask]])
        
    except Exception as e:
        log(f"Error in fractal support/resistance calculation: {str(e)}")
        return torch.tensor([], device=device), torch.tensor([], device=device), torch.tensor([], device=device)

# =====================================================================
# Wavelet Analysis Functions
# =====================================================================

def morlet_wavelet_torch(length, scale, center, device="cpu"):
    """
    Generate a Morlet wavelet for wavelet transform.
    
    Args:
        length (int): Length of the wavelet
        scale (float): Scale parameter (1/frequency)
        center (int): Center position
        device (str): Device to use for computation
        
    Returns:
        torch.Tensor: Complex Morlet wavelet
    """
    try:
        # Create time array
        t = torch.arange(length, device=device).float() - center
        
        # Normalized time
        t = t / scale
        
        # Morlet wavelet (complex-valued)
        arg = -0.5 * t**2
        
        # Compute the wavelet
        norm = (torch.pi * scale)**(-0.25)  # Normalization
        realpart = norm * torch.exp(arg)
        imgpart = norm * torch.exp(arg) * torch.sin(5.0 * t)  # 5.0 is wavelet center frequency
        
        # Real and imaginary parts
        wavelet = torch.complex(realpart, imgpart)
        
        return wavelet
        
    except Exception as e:
        log(f"Error in morlet wavelet generation: {str(e)}")
        return torch.zeros(length, dtype=torch.complex64, device=device)

def continuous_wavelet_transform_torch(data, scales, wavelet=morlet_wavelet_torch, device="cpu"):
    """
    Perform continuous wavelet transform on data using PyTorch.
    
    Args:
        data (torch.Tensor): 1D tensor of data to transform
        scales (list or torch.Tensor): Scales to use for transform
        wavelet (function): Wavelet function (default: Morlet)
        device (str): Device to use for computation
        
    Returns:
        torch.Tensor: Wavelet coefficients (complex tensor)
    """
    try:
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, device=device).float()
        
        if isinstance(scales, list):
            scales = torch.tensor(scales, device=device).float()
            
        n = len(data)
        m = len(scales)
        
        # Prepare output array
        cwt = torch.zeros((m, n), dtype=torch.complex64, device=device)
        
        # FFT of data
        fft_data = torch.fft.rfft(data)
        
        # Frequency array
        freq = torch.fft.rfftfreq(n, 1.0)
        
        # Perform wavelet transform
        for i, scale in enumerate(scales):
            # Generate wavelet in frequency domain
            wavelet_freq = torch.zeros(len(freq), dtype=torch.complex64, device=device)
            
            for j, f in enumerate(freq):
                if f > 0:  # Skip zero frequency
                    # Morlet wavelet in frequency domain
                    arg = -0.5 * (2 * torch.pi * scale * f - 5.0)**2
                    wavelet_freq[j] = torch.exp(arg)
            
            # Convolve (multiply in frequency domain)
            convolution = fft_data * wavelet_freq
            
            # IFFT
            cwt[i] = torch.fft.irfft(convolution, n=n)
        
        return cwt
        
    except Exception as e:
        log(f"Error in wavelet transform: {str(e)}")
        return torch.zeros((len(scales), len(data)), dtype=torch.complex64, device=device)

def wavelet_power_spectrum_torch(cwt):
    """
    Compute wavelet power spectrum from wavelet coefficients.
    
    Args:
        cwt (torch.Tensor): Wavelet coefficients (complex tensor)
        
    Returns:
        torch.Tensor: Wavelet power spectrum (real tensor)
    """
    try:
        # Power is |W|²
        return torch.abs(cwt)**2
        
    except Exception as e:
        log(f"Error in power spectrum calculation: {str(e)}")
        return torch.zeros_like(cwt, dtype=torch.float32)

def wavelet_coherence_torch(x, y, scales, device="cpu"):
    """
    Compute wavelet coherence between two time series.
    
    Args:
        x (torch.Tensor): First time series
        y (torch.Tensor): Second time series
        scales (list or torch.Tensor): Scales for wavelet transform
        device (str): Device to use for computation
        
    Returns:
        torch.Tensor: Wavelet coherence (0-1)
    """
    try:
        # Convert inputs to tensors
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, device=device).float()
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, device=device).float()
        if isinstance(scales, list):
            scales = torch.tensor(scales, device=device).float()
            
        # Ensure same length
        min_len = min(len(x), len(y))
        x = x[:min_len]
        y = y[:min_len]
        
        # Compute CWT for both signals
        cwt_x = continuous_wavelet_transform_torch(x, scales, device=device)
        cwt_y = continuous_wavelet_transform_torch(y, scales, device=device)
        
        # Cross wavelet transform
        cross_wavelet = cwt_x * torch.conj(cwt_y)
        
        # Cross wavelet power
        cross_power = torch.abs(cross_wavelet)**2
        
        # Auto spectral density
        power_x = torch.abs(cwt_x)**2
        power_y = torch.abs(cwt_y)**2
        
        # Wavelet coherence
        coherence = cross_power / (power_x * power_y)
        
        # Clamp values to [0, 1]
        coherence = torch.clamp(coherence, 0.0, 1.0)
        
        return coherence
        
    except Exception as e:
        log(f"Error in wavelet coherence calculation: {str(e)}")
        return torch.zeros((len(scales), min(len(x), len(y))), device=device)

def wavelet_decompose_tensor(data, num_levels=4, device="cpu"):
    """
    Decompose time series into trend and different frequency components using wavelet transform.
    
    Args:
        data (torch.Tensor): 1D tensor time series data
        num_levels (int): Number of decomposition levels
        device (str): Device to use for computation
        
    Returns:
        dict: Dictionary containing decomposition components:
            - 'trend': Low frequency trend component
            - 'detail_1', 'detail_2', etc.: Detail components at different frequencies
            - 'reconstructed': Reconstructed signal (sum of all components)
    """
    try:
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, device=device).float()
            
        n = len(data)
        
        # Dyadic scales (powers of 2)
        scales = [2**j for j in range(1, num_levels+1)]
        
        # Compute CWT for the scales
        cwt = continuous_wavelet_transform_torch(data, scales, device=device)
        
        # Extract components using different scales
        components = {}
        
        # Initialize reconstruction
        reconstructed = torch.zeros_like(data)
        
        # Extract detail components
        for i, scale in enumerate(scales):
            # Get real part of CWT for this scale
            detail = cwt[i].real
            
            # Store detail component
            components[f'detail_{i+1}'] = detail
            
            # Add to reconstruction
            reconstructed += detail
        
        # Calculate trend component (residual)
        trend = data - reconstructed
        components['trend'] = trend
        
        # Update reconstructed with trend
        reconstructed = reconstructed + trend
        components['reconstructed'] = reconstructed
        
        return components
        
    except Exception as e:
        log(f"Error in wavelet decomposition: {str(e)}")
        return {'trend': data, 'reconstructed': data}

def detect_dominant_cycles_tensor(data, scales=None, num_cycles=3, device="cpu"):
    """
    Detect dominant cycles in time series data using wavelet analysis.
    
    Args:
        data (torch.Tensor): 1D tensor of time series data
        scales (list or None): Scales to use for transform, if None, uses wide range
        num_cycles (int): Number of dominant cycles to detect
        device (str): Device to use for computation
        
    Returns:
        tuple: (cycle_periods, cycle_powers, cycle_phases)
            - cycle_periods: Tensor of cycle periods (in data points)
            - cycle_powers: Tensor of cycle powers (strength)
            - cycle_phases: Tensor of cycle phases (radians)
    """
    try:
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, device=device).float()
            
        n = len(data)
        
        # Generate scales if not provided
        if scales is None:
            # Geometric spacing from 2 to n/4
            min_scale = 2
            max_scale = n // 4
            num_scales = 32  # Number of scales to use
            
            # Logarithmically spaced scales
            scales = torch.logspace(
                torch.log10(torch.tensor(min_scale, dtype=torch.float32)),
                torch.log10(torch.tensor(max_scale, dtype=torch.float32)),
                num_scales,
                device=device
            )
        
        # Compute CWT
        cwt = continuous_wavelet_transform_torch(data, scales, device=device)
        
        # Compute power spectrum
        power = wavelet_power_spectrum_torch(cwt)
        
        # Average power across time for each scale
        scale_power = torch.mean(power, dim=1)
        
        # Find indices of dominant cycles
        if num_cycles >= len(scale_power):
            num_cycles = len(scale_power)
            
        _, dominant_indices = torch.topk(scale_power, num_cycles)
        
        # Extract periods (scales) of dominant cycles
        cycle_periods = scales[dominant_indices]
        
        # Extract powers of dominant cycles
        cycle_powers = scale_power[dominant_indices]
        
        # Extract phases of dominant cycles at the end of the time series
        cycle_phases = torch.angle(cwt[dominant_indices, -1])
        
        return cycle_periods, cycle_powers, cycle_phases
        
    except Exception as e:
        log(f"Error in cycle detection: {str(e)}")
        return (
            torch.tensor([0.0] * num_cycles, device=device),
            torch.tensor([0.0] * num_cycles, device=device),
            torch.tensor([0.0] * num_cycles, device=device)
        )

def compute_timeframe_wavelet_features(price_tensor, lookback=200, device="cpu"):
    """
    Compute multi-timeframe features using wavelet decomposition.
    
    Args:
        price_tensor (torch.Tensor): Tensor of price data
        lookback (int): Lookback period for computation
        device (str): Device to use for computation
        
    Returns:
        dict: Dictionary of multi-timeframe wavelet features
    """
    try:
        n = len(price_tensor)
        if n < lookback:
            # Not enough data
            return {
                'trend_direction': 0.0,
                'cycle_strengths': torch.zeros(3, device=device),
                'momentum_aligned': False,
                'fractal_dimension': 0.5
            }
        
        # Extract window for analysis
        start_idx = max(0, n - lookback)
        window = price_tensor[start_idx:n].clone().detach()
        
        # Decompose price using wavelets
        components = wavelet_decompose_tensor(window, num_levels=5, device=device)
        
        # Extract trend component
        trend = components['trend']
        
        # Compute trend direction
        trend_diff = trend[-1] - trend[-20]
        trend_direction = torch.sign(trend_diff).item()
        
        # Detect dominant cycles
        cycle_periods, cycle_powers, cycle_phases = detect_dominant_cycles_tensor(
            window, num_cycles=3, device=device
        )
        
        # Normalize cycle powers
        if torch.sum(cycle_powers) > 0:
            cycle_strengths = cycle_powers / torch.sum(cycle_powers)
        else:
            cycle_strengths = torch.zeros(3, device=device)
        
        # Check if momentum is aligned with trend
        detail1 = components.get('detail_1', torch.zeros_like(window))
        momentum_aligned = torch.sign(detail1[-1] - detail1[-5]) == torch.sign(trend_diff)
        
        # Compute fractal dimension
        fractal_dim = compute_fractal_dimension_tensor(window, device=device)
        
        return {
            'trend_direction': trend_direction,
            'cycle_strengths': cycle_strengths,
            'momentum_aligned': momentum_aligned,
            'fractal_dimension': fractal_dim.item()
        }
        
    except Exception as e:
        log(f"Error in wavelet feature computation: {str(e)}")
        return {
            'trend_direction': 0.0,
            'cycle_strengths': torch.zeros(3, device=device),
            'momentum_aligned': False,
            'fractal_dimension': 0.5
        }

def advanced_time_series_hilbert_transform(data, device="cpu"):
    """
    Compute the Hilbert transform of a time series using FFT.
    Used for extracting instantaneous phase and amplitude information.
    
    Args:
        data (torch.Tensor): 1D tensor time series data
        device (str): Device to use for computation
        
    Returns:
        tuple: (analytic_signal, amplitude, instantaneous_phase)
            - analytic_signal: Complex tensor (original + i*hilbert)
            - amplitude: Instantaneous amplitude (envelope)
            - instantaneous_phase: Instantaneous phase in radians
    """
    try:
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, device=device).float()
            
        n = len(data)
        
        # FFT of the original signal
        fft_data = torch.fft.rfft(data)
        
        # Prepare for Hilbert transform
        h = torch.zeros(len(fft_data), device=device)
        
        # Create frequency domain filter for Hilbert transform
        # First and last elements remain unchanged
        # Double the amplitude for positive frequencies
        if len(h) > 2:
            h[1:-1] = 2
        
        # Apply filter
        fft_hilbert = fft_data * h
        
        # Inverse FFT to get Hilbert transform
        hilbert = torch.fft.irfft(fft_hilbert, n=n)
        
        # Analytical signal = original + i*hilbert
        analytic_signal = torch.complex(data, hilbert)
        
        # Calculate amplitude envelope and instantaneous phase
        amplitude = torch.abs(analytic_signal)
        phase = torch.angle(analytic_signal)
        
        return analytic_signal, amplitude, phase
        
    except Exception as e:
        log(f"Error in Hilbert transform: {str(e)}")
        zeros = torch.zeros_like(data)
        return torch.complex(data, zeros), torch.abs(data), zeros

def empirical_mode_decomposition_tensor(data, max_imfs=5, sift_threshold=0.05, device="cpu"):
    """
    Simplified Empirical Mode Decomposition (EMD) on time series data.
    Decomposes a signal into Intrinsic Mode Functions (IMFs).
    
    Args:
        data (torch.Tensor): 1D tensor time series data
        max_imfs (int): Maximum number of IMFs to extract
        sift_threshold (float): Stopping criterion for sifting process
        device (str): Device to use for computation
        
    Returns:
        tuple: (imfs, residue)
            - imfs: Tensor of IMFs [num_imfs, signal_length]
            - residue: Final residue after extracting IMFs
    """
    try:
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, device=device).float()
            
        n = len(data)
        imfs = []
        residue = data.clone()
        
        # For each IMF
        for _ in range(max_imfs):
            # Check if residue is too small
            if torch.std(residue) < 1e-6 * torch.std(data):
                break
                
            # Copy of residue for current IMF extraction
            h = residue.clone()
            
            # Sifting process
            for _ in range(10):  # Max 10 iterations for sifting
                # Find extrema using zero crossings of derivative
                diff = torch.diff(h)
                zero_crossings = torch.where(diff[:-1] * diff[1:] <= 0)[0] + 1
                
                if len(zero_crossings) < 4:  # Need at least 4 extrema
                    break
                
                # Simple moving average as envelope approximation
                window_size = min(11, len(h) // 10 * 2 + 1)  # Ensure odd window size
                padded = torch.nn.functional.pad(h, (window_size//2, window_size//2), mode='reflect')
                mean_env = torch.nn.functional.avg_pool1d(
                    padded.view(1, 1, -1), 
                    kernel_size=window_size, 
                    stride=1
                ).view(-1)
                
                # New estimate
                h_new = h - mean_env
                
                # Check convergence
                if torch.sum((h - h_new)**2) / torch.sum(h**2) < sift_threshold:
                    break
                    
                h = h_new
            
            # Add IMF to list
            imfs.append(h)
            
            # Update residue
            residue = residue - h
        
        # Convert to tensor
        if imfs:
            imfs_tensor = torch.stack(imfs)
        else:
            imfs_tensor = torch.zeros((0, n), device=device)
            
        return imfs_tensor, residue
    
    except Exception as e:
        log(f"Error in EMD: {str(e)}")
        return torch.zeros((0, len(data)), device=device), data.clone()

def phase_space_reconstruction_tensor(data, embedding_dim=3, time_delay=1, device="cpu"):
    """
    Perform phase space reconstruction of a time series using time delay embedding.
    Useful for analyzing nonlinear dynamical systems.
    
    Args:
        data (torch.Tensor): 1D tensor time series data
        embedding_dim (int): Embedding dimension
        time_delay (int): Time delay between dimensions
        device (str): Device to use for computation
        
    Returns:
        torch.Tensor: Reconstructed phase space with shape [num_points, embedding_dim]
    """
    try:
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, device=device).float()
            
        n = len(data)
        
        # Check if we have enough data
        if n < (embedding_dim - 1) * time_delay + 1:
            return torch.zeros((0, embedding_dim), device=device)
        
        # Number of points in reconstructed space
        num_points = n - (embedding_dim - 1) * time_delay
        
        # Initialize phase space tensor
        phase_space = torch.zeros((num_points, embedding_dim), device=device)
        
        # Fill in phase space
        for i in range(embedding_dim):
            start_idx = i * time_delay
            end_idx = start_idx + num_points
            phase_space[:, i] = data[start_idx:end_idx]
        
        return phase_space
        
    except Exception as e:
        log(f"Error in phase space reconstruction: {str(e)}")
        return torch.zeros((0, embedding_dim), device=device)

def compute_order_flow_imbalance(price_tensor, volume_tensor, window_size=20, device="cpu"):
    """
    Compute order flow imbalance based on price and volume data.
    
    Args:
        price_tensor (torch.Tensor): Tensor of price data
        volume_tensor (torch.Tensor): Tensor of volume data
        window_size (int): Size of sliding window for calculation
        device (str): Device to use for computation
        
    Returns:
        torch.Tensor: Order flow imbalance tensor
    """
    try:
        if price_tensor is None or volume_tensor is None:
            return torch.zeros(1, device=device)
            
        n = len(price_tensor)
        
        # Need at least window_size + 1 data points
        if n <= window_size:
            return torch.zeros(n, device=device)
        
        # Calculate price changes
        price_changes = torch.zeros(n, device=device)
        price_changes[1:] = price_tensor[1:] - price_tensor[:-1]
        
        # Calculate direction of price changes
        directions = torch.sign(price_changes)
        
        # Calculate order flow (volume * direction)
        order_flow = volume_tensor * directions
        
        # Calculate cumulative order flow
        cumulative_flow = torch.cumsum(order_flow, dim=0)
        
        # Calculate order flow imbalance using rolling window
        imbalance = torch.zeros(n, device=device)
        
        for i in range(window_size, n):
            # Volume-weighted order flow in window
            window_flow = cumulative_flow[i] - cumulative_flow[i - window_size]
            window_volume = torch.sum(volume_tensor[i-window_size:i])
            
            if window_volume > 0:
                imbalance[i] = window_flow / window_volume
        
        return imbalance
        
    except Exception as e:
        log(f"Error in order flow imbalance calculation: {str(e)}")
        return torch.zeros(len(price_tensor), device=device)

def detect_market_regime(price_tensor, volume_tensor=None, window_size=100, device="cpu"):
    """
    Detect market regime (trending, ranging, volatile) based on price patterns.
    
    Args:
        price_tensor (torch.Tensor): Tensor of price data
        volume_tensor (torch.Tensor, optional): Tensor of volume data
        window_size (int): Size of sliding window for calculation
        device (str): Device to use for computation
        
    Returns:
        dict: Dictionary of market regime indicators:
            - 'trend_strength': Strength of trend (0-1)
            - 'range_strength': Strength of ranging market (0-1)
            - 'volatility': Normalized volatility
            - 'regime_label': Categorical label ('trending', 'ranging', 'volatile', 'mixed')
    """
    try:
        if price_tensor is None:
            return {
                'trend_strength': 0.0,
                'range_strength': 0.0,
                'volatility': 0.0,
                'regime_label': 'unknown'
            }
            
        n = len(price_tensor)
        
        # Need at least window_size data points
        if n < window_size:
            return {
                'trend_strength': 0.0,
                'range_strength': 0.0,
                'volatility': 0.0,
                'regime_label': 'unknown'
            }
        
        # Extract window for analysis
        start_idx = max(0, n - window_size)
        window_prices = price_tensor[start_idx:n].clone().detach()
        window_volumes = volume_tensor[start_idx:n].clone().detach() if volume_tensor is not None else None
        
        # Calculate returns
        window_returns = torch.zeros_like(window_prices)
        window_returns[1:] = window_prices[1:] / window_prices[:-1] - 1
        
        # Compute volatility (standard deviation of returns)
        volatility = torch.std(window_returns)
        
        # Compute trend strength using Hurst exponent
        hurst = compute_fractal_dimension_tensor(window_prices, device=device)
        trend_strength = hurst.item() if hurst > 0.5 else 0.0
        
        # Compute range strength (inverse of trend strength, adjusted)
        range_strength = 0.0
        if hurst < 0.5:
            range_strength = 1.0 - (hurst.item() * 2)  # Ranges have H < 0.5
        
        # Determine overall regime
        if trend_strength > 0.65:
            regime_label = 'trending'
        elif range_strength > 0.65:
            regime_label = 'ranging'
        elif volatility > 0.02:  # Arbitrary threshold
            regime_label = 'volatile'
        else:
            regime_label = 'mixed'
        
        return {
            'trend_strength': float(trend_strength),
            'range_strength': float(range_strength),
            'volatility': float(volatility),
            'regime_label': regime_label
        }
        
    except Exception as e:
        log(f"Error in market regime detection: {str(e)}")
        return {
            'trend_strength': 0.0,
            'range_strength': 0.0,
            'volatility': 0.0,
            'regime_label': 'unknown'
        }

def compute_liquidity_metrics(price_tensor, volume_tensor, window_size=20, device="cpu"):
    """
    Compute market liquidity metrics.
    
    Args:
        price_tensor (torch.Tensor): Tensor of price data
        volume_tensor (torch.Tensor): Tensor of volume data
        window_size (int): Size of sliding window for calculation
        device (str): Device to use for computation
        
    Returns:
        dict: Dictionary of liquidity metrics:
            - 'amihud_illiquidity': Amihud illiquidity ratio
            - 'turnover_ratio': Volume/price ratio
            - 'market_depth': Estimated market depth
    """
    try:
        if price_tensor is None or volume_tensor is None:
            return {
                'amihud_illiquidity': 0.0,
                'turnover_ratio': 0.0,
                'market_depth': 0.0
            }
            
        n = len(price_tensor)
        
        # Need at least window_size data points
        if n < window_size:
            return {
                'amihud_illiquidity': 0.0,
                'turnover_ratio': 0.0,
                'market_depth': 0.0
            }
        
        # Calculate returns
        returns = torch.zeros(n, device=device)
        returns[1:] = torch.abs(price_tensor[1:] / price_tensor[:-1] - 1)
        
        # Extract window for analysis
        start_idx = max(0, n - window_size)
        window_returns = returns[start_idx:n]
        window_volumes = volume_tensor[start_idx:n]
        window_prices = price_tensor[start_idx:n]
        
        # Calculate Amihud illiquidity (|returns| / volume)
        window_illiquidity = window_returns / (window_volumes + 1e-10)
        amihud_illiquidity = torch.mean(window_illiquidity).item()
        
        # Calculate turnover ratio (volume/price)
        turnover_ratio = torch.mean(window_volumes / window_prices).item()
        
        # Estimate market depth (inverse of price impact)
        if amihud_illiquidity > 0:
            market_depth = 1.0 / amihud_illiquidity
        else:
            market_depth = 1000.0  # Arbitrary high value for very liquid markets
        
        return {
            'amihud_illiquidity': float(amihud_illiquidity),
            'turnover_ratio': float(turnover_ratio),
            'market_depth': float(market_depth)
        }
        
    except Exception as e:
        log(f"Error in liquidity metrics calculation: {str(e)}")
        return {
            'amihud_illiquidity': 0.0,
            'turnover_ratio': 0.0,
            'market_depth': 0.0
        }

if __name__ == "__main__":
    # Simple test for tensor utility functions
    import torch
    
    # Create a synthetic OHLCV tensor
    n_bars = 1000
    ohlcv = torch.zeros((n_bars, 5))
    
    # Generate realistic price data
    base_price = 50000.0
    price = base_price
    for i in range(n_bars):
        # Random walk
        change = torch.randn(1).item() * 100.0
        price += change
        price = max(10000, price)  # Ensure positive price
        
        # Generate OHLCV
        high = price + abs(torch.randn(1).item() * 50.0)
        low = price - abs(torch.randn(1).item() * 50.0)
        low = max(10000, low)  # Ensure positive price
        volume = torch.rand(1).item() * 10.0 + 1.0
        
        # Store
        ohlcv[i, 0] = price  # Close
        ohlcv[i, 1] = price - change  # Open (previous close)
        ohlcv[i, 2] = high  # High
        ohlcv[i, 3] = low  # Low
        ohlcv[i, 4] = volume  # Volume
    
    # Test various functions
    current_step = 500
    
    print("Testing enhanced tensor utility functions...")
    
    # Volume profile
    profile, price_min, price_max = compute_volume_profile_tensor(ohlcv, current_step)
    print(f"Volume profile: {len(profile)} levels, range: {price_min:.2f} - {price_max:.2f}")
    
    # Liquidity zones
    zones = identify_liquidity_zones_tensor(ohlcv, current_step)
    print(f"Liquidity zones: {zones}")
    
    # Spread estimation
    spread = estimate_bid_ask_spread_tensor(ohlcv, current_step)
    print(f"Estimated spread: {spread:.6f}")
    
    # Volume delta
    delta, buy, sell = calculate_volume_delta_tensor(ohlcv, current_step)
    print(f"Volume delta: {delta:.2f} (buy: {buy:.2f}, sell: {sell:.2f})")
    
    # Liquidity
    liquidity = estimate_market_liquidity_tensor(ohlcv, current_step)
    print(f"Market liquidity: {liquidity:.2f}")
    
    # Enhanced pattern detection
    patterns = detect_patterns_tensor(ohlcv, current_step)
    print(f"Detected patterns: {patterns}")
    
    # Key levels
    key_levels = identify_key_levels_tensor(ohlcv, current_step)
    print(f"Key levels: {key_levels}")
    
    # Order flow analysis
    order_flow = analyze_order_flow_tensor(ohlcv, current_step)
    print(f"Order flow analysis: {order_flow}")
    
    # Batch processing
    indices = [100, 200, 300, 400, 500]
    batch_features = batch_process_features(ohlcv, indices)
    print(f"Batch processed features for {len(indices)} indices")
    for i, features in enumerate(batch_features):
        print(f"Index {indices[i]}: {len(features)} features")
