#!/usr/bin/env python
"""
Test suite for dataframe.py
--------------------------
Tests the following components:
- Data loading
- Technical indicator calculations
- PCA feature extraction
- Autoencoder features
- Column verification
- Full dataframe building
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
import importlib
from unittest.mock import patch, MagicMock
import tempfile
import pickle

# Add parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Get the project root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
# Add project root to system path to ensure imports work
sys.path.insert(0, project_root)

# Import the dataframe module using dynamic imports
try:
    dataframe_module = importlib.import_module("src.utils.dataframe")
except ImportError as e:
    print(f"Error importing dataframe module: {e}")
    sys.exit(1)

# Create test data
@pytest.fixture
def sample_df():
    """Create a small sample dataframe with OHLCV data for testing"""
    # Generate 100 rows of sample price data
    np.random.seed(42)  # For reproducibility
    dates = pd.date_range(start='2023-01-01', periods=100, freq='5min')
    
    # Create a reasonable price series (starting at 20000 with random walk)
    close = 20000 + np.cumsum(np.random.normal(0, 100, 100))
    high = close + np.random.uniform(0, 200, 100)
    low = close - np.random.uniform(0, 200, 100)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': close - np.random.uniform(0, 50, 100),
        'high': high,
        'low': low,
        'close': close,
        'volume': np.random.uniform(5, 50, 100) * 1000
    })
    
    # Add hour, day_of_week for time features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_month'] = df['timestamp'].dt.day
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    
    # Add cyclic time features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_month_sin'] = np.sin(2 * np.pi * df['day_of_month'] / 31)
    df['day_of_month_cos'] = np.cos(2 * np.pi * df['day_of_month'] / 31)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 366)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 366)
    
    return df

# Test cache mechanism
def test_cached_operation():
    """Test the caching mechanism works correctly"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Mock the cache directory
        dataframe_module.CACHE_DIR = temp_dir
        
        # Create a function that counts calls
        call_counter = MagicMock(return_value=42)
        
        # First call - should execute the function
        result1 = dataframe_module.cached_operation(call_counter, "test_cache.pkl")
        
        # Second call - should use cache
        result2 = dataframe_module.cached_operation(call_counter, "test_cache.pkl")
        
        # Force recompute - should execute the function again
        result3 = dataframe_module.cached_operation(call_counter, "test_cache.pkl", force_recompute=True)
        
        # Verify
        assert call_counter.call_count == 2  # Called twice (initial + forced)
        assert result1 == result2 == result3 == 42  # Results are consistent
        assert os.path.exists(os.path.join(temp_dir, "test_cache.pkl"))  # Cache file created

# Test technical indicators
def test_compute_indicators(sample_df):
    """Test technical indicator computation"""
    # Mock the cache_operation to always compute
    with patch('src.utils.dataframe.cached_operation', side_effect=lambda f, *args, **kwargs: f()):
        result = dataframe_module.compute_indicators(sample_df)
        
        # Check key indicators are present
        expected_columns = [
            'SMA9', 'SMA21', 'RSI14', 'BB_upper20', 'BB_mid20', 'BB_lower20', 'ATR'
        ]
        
        for col in expected_columns:
            assert col in result.columns, f"Column {col} missing from result"
        
        # Check a few specific values
        assert not pd.isna(result['SMA9']).all(), "SMA9 contains only NaN values"
        assert not pd.isna(result['RSI14']).all(), "RSI14 contains only NaN values"

# Test PCA functionality
def test_dynamic_pca():
    """Test the dynamic PCA component selection"""
    # Create test data with known structure
    np.random.seed(42)
    test_data = np.random.rand(100, 20)
    
    # Make first 3 components explain most variance
    test_data[:, 0] = test_data[:, 0] * 10  # High variance
    test_data[:, 1] = test_data[:, 1] * 8   # Medium high variance
    test_data[:, 2] = test_data[:, 2] * 6   # Medium variance
    
    # Convert to dataframe
    df_test = pd.DataFrame(test_data)
    
    # Run PCA with our dynamic approach
    pca_results, n_components = dataframe_module.dynamic_pca(
        df_test, 
        variance_threshold=0.5,  # Lower threshold for testing
        min_components=2,
        max_components=10
    )
    
    # Verify results
    assert isinstance(pca_results, pd.DataFrame), "PCA should return DataFrame"
    assert n_components >= 2, "Should select at least min_components"
    assert n_components <= 10, "Should not exceed max_components"
    assert pca_results.shape[0] == 100, "Should return same number of rows"
    assert pca_results.shape[1] == n_components, "Should return n_components columns"

# Test the simple autoencoder
def test_autoencoder():
    """Test the autoencoder module creates proper latent representations"""
    import torch
    
    # Create simple test data
    np.random.seed(42)
    test_data = np.random.rand(50, 10)
    
    # Create model with small latent space
    input_dim = 10
    latent_dim = 3
    model = dataframe_module.SimpleAutoencoder(input_dim, latent_dim)
    
    # Forward pass
    x = torch.tensor(test_data, dtype=torch.float32)
    reconstructed, latent = model(x)
    
    # Verify shapes
    assert reconstructed.shape == x.shape, "Reconstruction shape should match input"
    assert latent.shape == (50, latent_dim), "Latent shape should be (batch_size, latent_dim)"

# Test verify_columns function
def test_verify_columns():
    """Test the column verification and addition of missing columns"""
    # Create test dataframe with some columns
    test_df = pd.DataFrame({
        'close': [1, 2, 3],
        'high': [2, 3, 4],
        'RSI14': [30, 40, 50]
    })
    
    # Define expected columns (including some missing ones)
    required = ['close', 'high', 'low', 'RSI14', 'SMA9']
    
    # Run verification
    result = dataframe_module.verify_columns(test_df, required_columns=required)
    
    # Check all required columns exist
    for col in required:
        assert col in result.columns, f"Column {col} should be added if missing"
    
    # Check missing columns are filled with zeros
    assert (result['low'] == 0).all(), "Missing columns should be filled with zeros"
    assert (result['SMA9'] == 0).all(), "Missing columns should be filled with zeros"

# Integration test for the full pipeline
@pytest.mark.slow  # Mark as slow to skip in quick test runs
def test_build_dataframe_integration(sample_df):
    """Integration test for the full dataframe building pipeline"""
    # Create a temporary file path (that doesn't exist yet)
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, f"test_dataframe_{os.getpid()}.csv")
    
    try:
        # Mock the load_5m_data to return our sample data
        with patch('src.utils.dataframe.load_5m_data', return_value=sample_df):
            # Mock the ML components that might fail with small datasets
            with patch('src.utils.dataframe.groupwise_pca', return_value=pd.DataFrame(
                    np.random.random((len(sample_df), 5)),
                    index=sample_df.index,
                    columns=[f'PCA{i}' for i in range(5)]
                )):
                with patch('src.utils.dataframe.build_autoencoder_features', return_value=pd.DataFrame(
                        np.random.random((len(sample_df), 5)),
                        index=sample_df.index,
                        columns=[f'AE{i}' for i in range(5)]
                    )):
                    # Also mock compute_additional_signals to provide expected columns
                    with patch('src.utils.dataframe.compute_additional_signals', return_value=pd.DataFrame(
                            np.random.random((len(sample_df), 3)),
                            index=sample_df.index,
                            columns=['signal1', 'signal2', 'signal3']
                        )):
                        # Also patch compute_indicators to prevent empty dataframe after dropna()
                        with patch('src.utils.dataframe.compute_indicators', return_value=sample_df):
                            # Set the output path
                            dataframe_module.OUTPUT_PATH = temp_file_path
                            
                            # Run the full pipeline with forced recompute
                            result = dataframe_module.build_dataframe(force_recompute=True)
                            
                            # Verify result
                            assert result is not None, "Should return a dataframe"
                            assert not result.empty, "Result should not be empty"
                            
                            # Check key column categories exist (we know they will because we're mocking them)
                            column_prefixes = ['pca_', 'ae_']
                            for prefix in column_prefixes:
                                assert any(col.startswith(prefix) for col in result.columns), f"Missing {prefix} columns"
                            
                            # Verify file was saved
                            assert os.path.exists(temp_file_path), "CSV file should be created"
                            
                            # Check loaded file matches result
                            loaded = pd.read_csv(temp_file_path)
                            assert loaded.shape == result.shape, "Saved file should have same shape"
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except (IOError, PermissionError):
                pass  # Ignore cleanup errors

if __name__ == "__main__":
    # Allow running with pytest or directly
    pytest.main(["-xvs", __file__]) 