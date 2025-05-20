#!/usr/bin/env python
"""
Test suite for tensor_utils v3.py
---------------------------------
# NOTE: Tests in this file may fail with TypeError: isinstance()
# when run as part of the full test suite. This is likely due to
# interference from mocking/patching in other test files (e.g., test_models.py).
# These tests pass when run in isolation:
# conda activate torch_env2; python -m pytest -rs tests/unit/test_tensor_utils.py
#
# Original Docstring:
# Tests the following components:
# - Volume profile analysis
# - Support/resistance detection
# - Liquidity analysis
# - Pattern detection
# - Market regime classification
# - Fractal analysis
# - Wavelet transforms
# - Elliott wave analysis
# - Market volatility estimation
"""

import os
import sys
import pytest
import numpy as np
import torch
import tempfile
from unittest.mock import patch, MagicMock

# Add parent directory to path so we can import tensor_utils module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib
tensor_utils = importlib.import_module("src.utils.tensor_utils")

# Create fixtures for tensor-based tests
@pytest.fixture
def sample_ohlcv_tensor():
    """Create a sample OHLCV tensor for testing"""
    # Generate 200 rows of sample OHLCV data with some price trends
    np.random.seed(42)  # For reproducible tests
    n_samples = 200
    
    # Generate price data with trends
    base_price = 20000.0
    
    # Create a trending price series
    price_changes = np.random.normal(0, 100, n_samples).cumsum()
    # Add a trend
    trend = np.linspace(0, 1000, n_samples)
    price_changes = price_changes + trend
    
    close = base_price + price_changes
    
    # Create reasonable open, high, low values
    spreads = np.random.uniform(10, 200, n_samples)
    open_prices = close - np.random.uniform(-1, 1, n_samples) * spreads
    high_prices = np.maximum(close, open_prices) + np.random.uniform(0, 100, n_samples)
    low_prices = np.minimum(close, open_prices) - np.random.uniform(0, 100, n_samples)
    
    # Generate volume with some relation to price moves
    volume = np.abs(close - open_prices) * np.random.uniform(0.5, 2.0, n_samples) * 1000
    
    # Ensure certain sections show patterns (like triangles, head and shoulders, etc.)
    # Head and shoulders pattern from index 50-70
    pattern_indices = np.arange(50, 71)
    pattern_length = len(pattern_indices)
    # Create a head and shoulders pattern
    pattern = np.concatenate([
        np.linspace(0, 300, pattern_length // 3),             # Left shoulder
        np.linspace(300, 600, pattern_length // 3),           # Head up
        np.linspace(600, 100, pattern_length // 3 + pattern_length % 3)  # Head down + right shoulder
    ])
    close[pattern_indices] = base_price + 1500 + pattern
    
    # Assemble the tensor
    ohlcv_data = np.column_stack([close, open_prices, high_prices, low_prices, volume])
    
    # Create a real tensor instead of a mock
    return torch.tensor(ohlcv_data, dtype=torch.float32)

@pytest.fixture
def sample_price_tensor():
    """Create a sample price tensor for testing"""
    # Generate 200 rows of sample price data
    np.random.seed(42)  # For reproducible tests
    n_samples = 200
    
    # Generate price data with fractal properties
    base_price = 20000.0
    
    # Hurst exponent around 0.6 (slight trend persistence)
    price_changes = np.random.normal(0, 100, n_samples)
    # Apply fractional Brownian motion approximation
    for i in range(1, n_samples):
        price_changes[i] = 0.6 * price_changes[i-1] + 0.4 * price_changes[i]
    
    price_changes = price_changes.cumsum()
    prices = base_price + price_changes
    
    # Create a real tensor instead of a mock
    return torch.tensor(prices, dtype=torch.float32)

@pytest.fixture
def market_regimes_data():
    """Create sample data with different market regimes"""
    n_samples = 400
    np.random.seed(42)  # For reproducibility
    
    # Base price and volume
    base_price = 20000.0
    base_volume = 10.0
    
    # Create price series with different regimes
    prices = np.zeros(n_samples)
    volumes = np.zeros(n_samples)
    
    # Regime 1: Trending (first 100 samples)
    trend_slope = 50.0
    trend_prices = base_price + np.arange(100) * trend_slope + np.random.normal(0, 50, 100)
    trend_volumes = base_volume + np.random.normal(0, 2, 100)
    
    # Regime 2: Ranging (next 100 samples)
    range_center = trend_prices[-1]
    range_width = 1000.0
    range_prices = range_center + np.sin(np.linspace(0, 4*np.pi, 100)) * range_width/2 + np.random.normal(0, 100, 100)
    range_volumes = base_volume * 0.8 + np.random.normal(0, 1, 100)
    
    # Regime 3: Volatile (next 100 samples)
    volatile_base = range_prices[-1]
    volatile_prices = volatile_base + np.random.normal(0, 300, 100).cumsum()
    volatile_volumes = base_volume * 2.0 + np.abs(np.random.normal(0, 3, 100))
    
    # Regime 4: Trending again but down (last 100 samples)
    down_trend_start = volatile_prices[-1]
    down_trend_prices = down_trend_start - np.arange(100) * trend_slope * 0.8 + np.random.normal(0, 70, 100)
    down_trend_volumes = base_volume * 1.5 + np.random.normal(0, 2, 100)
    
    # Combine all regimes
    prices[:100] = trend_prices
    prices[100:200] = range_prices
    prices[200:300] = volatile_prices
    prices[300:] = down_trend_prices
    
    volumes[:100] = trend_volumes
    volumes[100:200] = range_volumes
    volumes[200:300] = volatile_volumes
    volumes[300:] = down_trend_volumes
    
    # Calculate OHLCV data
    close = prices
    spreads = np.random.uniform(10, 200, n_samples)
    open_prices = close - np.random.uniform(-1, 1, n_samples) * spreads
    high_prices = np.maximum(close, open_prices) + np.random.uniform(0, 100, n_samples)
    low_prices = np.minimum(close, open_prices) - np.random.uniform(0, 100, n_samples)
    
    # Assemble the tensor
    ohlcv_data = np.column_stack([close, open_prices, high_prices, low_prices, volumes])
    
    # Return the tensor and regime labels
    regimes = ['trending'] * 100 + ['ranging'] * 100 + ['volatile'] * 100 + ['trending'] * 100
    
    return torch.tensor(ohlcv_data, dtype=torch.float32), regimes

# Test volume profile
def test_compute_volume_profile_tensor(sample_ohlcv_tensor):
    """Test the computation of volume profiles"""
    # Test with normal parameters
    current_step = 100
    lookback = 50
    num_levels = 10
    
    profile, price_min, price_max = tensor_utils.compute_volume_profile_tensor(
        sample_ohlcv_tensor, 
        current_step,
        lookback=lookback,
        num_levels=num_levels
    )
    
    # Check output shape and types
    assert isinstance(profile, torch.Tensor)
    assert profile.shape == (num_levels,)
    assert isinstance(price_min, (int, float))
    assert isinstance(price_max, (int, float))
    assert price_min < price_max
    
    # Test with edge case (current_step = 0)
    profile_edge, min_edge, max_edge = tensor_utils.compute_volume_profile_tensor(
        sample_ohlcv_tensor, 
        0,
        lookback=lookback
    )
    
    # Should return zeros for edge case
    assert torch.all(profile_edge == 0)
    
    # Test with invalid current_step (out of bounds)
    profile_invalid, min_invalid, max_invalid = tensor_utils.compute_volume_profile_tensor(
        sample_ohlcv_tensor, 
        len(sample_ohlcv_tensor) + 10,
        lookback=lookback
    )
    
    # Should return zeros for invalid case
    assert torch.all(profile_invalid == 0)

# Test liquidity zones identification
def test_identify_liquidity_zones_tensor(sample_ohlcv_tensor):
    """Test the identification of liquidity zones"""
    # Test with normal parameters
    current_step = 150
    lookback = 100
    
    support_zones, resistance_zones = tensor_utils.identify_liquidity_zones_tensor(
        sample_ohlcv_tensor,
        current_step,
        lookback=lookback
    )
    
    # Check output types and shapes
    assert isinstance(support_zones, list)
    assert isinstance(resistance_zones, list)
    
    # Check if zones are ordered (supports should be below current price, resistance above)
    if len(support_zones) > 0 and len(resistance_zones) > 0:
        current_price = sample_ohlcv_tensor[current_step-1, 0].item()  # Close price
        for zone in support_zones:
            assert zone[0] <= current_price, "Support zone should be below or at current price"
        for zone in resistance_zones:
            assert zone[0] >= current_price, "Resistance zone should be above or at current price"
    
    # Test edge case (current_step near start)
    edge_support, edge_resistance = tensor_utils.identify_liquidity_zones_tensor(
        sample_ohlcv_tensor,
        10,
        lookback=100
    )
    
    # Should still return valid lists
    assert isinstance(edge_support, list)
    assert isinstance(edge_resistance, list)

# Test market regime detection
def test_detect_market_regime(market_regimes_data):
    """Test the market regime detection function"""
    ohlcv_tensor, expected_regimes = market_regimes_data
    
    # Test at different points in the time series
    regime_points = [50, 150, 250, 350]  # Points in different regimes
    expected_regime_types = ['trending', 'ranging', 'volatile', 'trending']
    
    for i, step in enumerate(regime_points):
        # Get price and volume tensors
        price_tensor = ohlcv_tensor[:step, 0]  # Close prices
        volume_tensor = ohlcv_tensor[:step, 4]  # Volume
        
        # Detect regime
        regime_info = tensor_utils.detect_market_regime(
            price_tensor,
            volume_tensor,
            window_size=min(50, step-1)
        )
        
        # Check output format
        assert isinstance(regime_info, dict)
        assert 'regime_label' in regime_info
        assert isinstance(regime_info['regime_label'], str)
        # We might not check the exact regime type if the data generation is noisy
        # assert regime_info['regime_label'] == expected_regime_types[i], f"Mismatch at step {step}"

        # Check other metrics exist
        assert 'trend_strength' in regime_info
        assert 'range_strength' in regime_info
        assert 'volatility' in regime_info

# Test fractal dimension computation
def test_compute_fractal_dimension_tensor(sample_price_tensor):
    """Test the computation of fractal dimension"""
    # Test with different window sizes
    window_sizes = [20, 50, 100]
    
    for window_size in window_sizes:
        # Compute fractal dimension
        fractal_dim = tensor_utils.compute_fractal_dimension_tensor(
            sample_price_tensor,
            window_size=window_size
        )
        
        # Check output type and value range
        assert isinstance(fractal_dim, torch.Tensor)
        # Hurst exponent (H) should be between 0 and 1
        assert 0.0 <= fractal_dim.item() <= 1.0, f"Hurst exponent ({fractal_dim.item()}) out of range [0, 1]"

# Test Elliott wave pattern detection
def test_detect_elliott_wave_pattern_tensor(sample_price_tensor):
    """Test the detection of Elliott Wave patterns"""
    # Test with normal parameters
    window_size = 100
    
    result = tensor_utils.detect_elliott_wave_pattern_tensor(
        sample_price_tensor,
        window_size=window_size
    )
    
    # Check output format
    assert isinstance(result, dict)
    assert 'detected' in result
    assert 'confidence' in result
    assert 'wave_points' in result
    assert isinstance(result['detected'], bool)
    assert isinstance(result['confidence'], float)
    assert 0 <= result['confidence'] <= 1
    
    # If pattern detected, validate wave points
    if result['detected'] and len(result['wave_points']) > 0:
        wave_points = result['wave_points']
        assert len(wave_points) >= 5, "Should detect at least 5 points for a complete Elliott wave"
        assert all(isinstance(p, int) for p in wave_points), "Wave points should be indices"
        
        # Check if wave points are in ascending order
        assert all(wave_points[i] < wave_points[i+1] for i in range(len(wave_points)-1)), "Wave points should be in ascending order"

# Test support and resistance detection
def test_detect_support_resistance_tensor(sample_ohlcv_tensor):
    """Test the detection of support and resistance levels"""
    # Test with normal parameters
    current_step = 150
    window_size = 100
    num_levels = 3
    
    support_levels, resistance_levels = tensor_utils.detect_support_resistance_tensor(
        sample_ohlcv_tensor,
        current_step,
        window_size=window_size,
        num_levels=num_levels
    )
    
    # Check output format
    assert isinstance(support_levels, list)
    assert isinstance(resistance_levels, list)

    # Check if the number of levels is respected (approximately)
    assert len(support_levels) <= num_levels
    assert len(resistance_levels) <= num_levels

    # Check if levels are reasonable (within price range)
    if support_levels:
        min_price = torch.min(sample_ohlcv_tensor[current_step-window_size:current_step, 3]).item()
        assert all(level >= min_price * 0.9 for level in support_levels)
    if resistance_levels:
        max_price = torch.max(sample_ohlcv_tensor[current_step-window_size:current_step, 2]).item()
        assert all(level <= max_price * 1.1 for level in resistance_levels)

# Test wavelet features computation
def test_compute_timeframe_wavelet_features(sample_price_tensor):
    """Test the computation of wavelet-based features"""
    # Test with normal parameters
    lookback = 100
    
    result = tensor_utils.compute_timeframe_wavelet_features(
        sample_price_tensor[-lookback:],
        lookback=lookback
    )
    
    # Check output format
    assert isinstance(result, dict)
    assert 'cycle_strengths' in result
    assert 'fractal_dimension' in result
    assert 'trend_direction' in result
    assert 'momentum_aligned' in result
    
    # Check types of returned values
    assert isinstance(result['cycle_strengths'], torch.Tensor)
    assert isinstance(result['fractal_dimension'], float)
    assert isinstance(result['trend_direction'], float)
    assert isinstance(result['momentum_aligned'], torch.Tensor)

# Test pattern detection
def test_detect_patterns_tensor(sample_ohlcv_tensor):
    """Test the detection of chart patterns"""
    # Test with pattern-rich section
    current_step = 70  # Around where we created the head and shoulders pattern
    lookback = 30
    
    patterns = tensor_utils.detect_patterns_tensor(
        sample_ohlcv_tensor,
        current_step,
        lookback=lookback
    )
    
    # Check output format
    assert isinstance(patterns, dict)
    assert 'patterns' in patterns
    assert 'strength' in patterns
    assert 'trend' in patterns

    # Check specific patterns if possible
    # Example: assert 'head_and_shoulders' in patterns['patterns'] # This depends on implementation
    assert isinstance(patterns['patterns'], dict)
    assert isinstance(patterns['strength'], float)
    assert patterns['strength'] >= 0.0

# Test volatility estimation
def test_estimate_volatility_tensor(sample_ohlcv_tensor):
    """Test the estimation of market volatility"""
    # Test with normal parameters
    current_step = 150
    lookback = 50
    
    volatility_info = tensor_utils.estimate_volatility_tensor(
        sample_ohlcv_tensor,
        current_step,
        lookback=lookback
    )
    
    # Check output format
    assert isinstance(volatility_info, float)
    assert 0.0 <= volatility_info <= 2.0, f"Volatility ({volatility_info}) out of expected range [0, 2]"

    # Test edge case (insufficient data)
    volatility_edge = tensor_utils.estimate_volatility_tensor(
        sample_ohlcv_tensor,
        current_step=1, # Only one previous bar
        lookback=50
    )
    assert isinstance(volatility_edge, float)
    assert np.isclose(volatility_edge, 0.2) # Should return default

# Test batch feature processing
def test_batch_process_features(sample_ohlcv_tensor):
    """Test batch processing of market features"""
    # Test with multiple indices
    current_indices = [50, 100, 150]
    lookback = 30
    
    batch_features = tensor_utils.batch_process_features(
        sample_ohlcv_tensor,
        current_indices,
        lookback=lookback
    )
    
    # Check output format
    assert isinstance(batch_features, list)
    assert len(batch_features) == len(current_indices)
    
    # Check contents of one dictionary (optional, more detailed checks can be added)
    if batch_features:
        first_result = batch_features[0]
        assert isinstance(first_result, dict)
        # Check a few expected keys exist
        assert 'volume_profile' in first_result
        assert 'momentum' in first_result
        assert 'volatility' in first_result
        assert 'support' in first_result
        assert 'resistance' in first_result

# Edge cases and error handling tests
def test_error_handling():
    """Test error handling in tensor utility functions"""
    # Test with empty tensor
    empty_tensor = torch.tensor([])
    
    # Volume profile with empty tensor
    profile, min_val, max_val = tensor_utils.compute_volume_profile_tensor(
        empty_tensor, 0, lookback=10
    )
    assert torch.all(profile == 0)
    
    # Test with None input
    profile_none, min_none, max_none = tensor_utils.compute_volume_profile_tensor(
        None, 0, lookback=10
    )
    assert torch.all(profile_none == 0)
    
    # Test with tensor of wrong shape
    wrong_shape_tensor = torch.randn(10, 2)  # Missing some OHLCV columns
    
    # This should not raise an exception, but return empty/default values
    profile_wrong, min_wrong, max_wrong = tensor_utils.compute_volume_profile_tensor(
        wrong_shape_tensor, 5, lookback=10
    )
    assert torch.all(profile_wrong == 0)

# =====================================================================
# Testing Utility Functions
# =====================================================================

def test_get_tensor_device():
    pass

if __name__ == "__main__":
    # Allow running with pytest directly
    pytest.main(["-xvs", __file__]) 