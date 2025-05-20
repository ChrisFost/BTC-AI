#!/usr/bin/env python
"""
Test script for the Training Recovery System.

This script demonstrates how the TrainingRecoverySystem handles different error scenarios.
It can be used to verify the robustness of the training recovery system.
"""

import os
import sys
import pandas as pd
import numpy as np
import time
import torch
import logging
import random
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_recovery")

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(project_root)

# Import training recovery system
try:
    from src.training.training_recovery import TrainingRecoverySystem
    from src.training.training import train_model
except ImportError as e:
    logger.error(f"Could not import required modules: {e}")
    sys.exit(1)

def create_synthetic_data():
    """Create synthetic data for testing."""
    logger.info("Creating synthetic data for testing")
    dates = pd.date_range('2020-01-01', periods=5000, freq='5min')
    df = pd.DataFrame({
        'timestamp': dates,
        'close': np.random.random(5000) * 1000 + 40000,  # Random prices around 40-50k
        'open': np.random.random(5000) * 1000 + 40000,
        'high': np.random.random(5000) * 1000 + 41000,
        'low': np.random.random(5000) * 1000 + 39000,
        'volume': np.random.random(5000) * 100  # Random volume
    })
    
    # Add some technical indicators for testing
    df['SMA9'] = df['close'].rolling(9).mean()
    df['SMA21'] = df['close'].rolling(21).mean()
    df['SMA50'] = df['close'].rolling(50).mean()
    df['SMA200'] = df['close'].rolling(200).mean()
    
    # Calculate RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI14'] = 100 - (100 / (1 + rs))

    # Fill NaN values
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    logger.info(f"Created synthetic data with shape: {df.shape}")
    return df

# Test functions that simulate different error scenarios
def training_with_value_error(df, config, **kwargs):
    """Simulate a ValueError during training."""
    logger.info("Simulating a ValueError during training")
    raise ValueError("Invalid hyperparameter: learning_rate is too high")

def training_with_runtime_error(df, config, **kwargs):
    """Simulate a RuntimeError during training."""
    logger.info("Simulating a RuntimeError during training")
    # Process normally for a bit
    time.sleep(1)
    logger.info("Processing...")
    time.sleep(1)
    
    # Then raise an error
    raise RuntimeError("CUDA out of memory")

def training_with_key_error(df, config, **kwargs):
    """Simulate a KeyError during training."""
    logger.info("Simulating a KeyError during training")
    # Try to access a key that doesn't exist
    missing_key = config["NONEXISTENT_KEY"]
    return None

def training_with_nan_error(df, config, **kwargs):
    """Simulate a NaN error during training."""
    logger.info("Simulating a NaN error during training")
    # Process normally for a bit
    time.sleep(1)
    logger.info("Processing...")
    time.sleep(1)
    
    # Then raise an error
    raise RuntimeError("NaN values detected in gradient")

def successful_training(df, config, **kwargs):
    """Simulate a successful training run."""
    logger.info("Simulating a successful training run")
    # Process normally
    for i in range(3):
        logger.info(f"Training progress: {i+1}/3")
        time.sleep(1)
    
    # Create a dummy model
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 1)
            
        def forward(self, x):
            return self.linear(x)
    
    model = DummyModel()
    metrics = {
        "profit": 1000.0,
        "win_rate": 0.65,
        "sharpe_ratio": 1.8,
        "max_drawdown": 0.15
    }
    best_reward = 850.0
    
    return model, metrics, best_reward

def test_recovery_system():
    """Test the TrainingRecoverySystem with different error scenarios."""
    # Create a temp directory for checkpoints
    temp_dir = os.path.join(project_root, "temp_checkpoints")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create synthetic data
    df = create_synthetic_data()
    
    # Basic config for testing
    config = {
        "LEARNING_RATE": 0.001,
        "HIDDEN_SIZE": 64,
        "ES_POPULATION": 4,
        "BUCKET": "Scalping",
        "MAX_STEPS_PER_EPISODE": 100,
        "EPISODES": 5
    }
    
    # Create recovery system
    recovery_system = TrainingRecoverySystem(
        checkpoint_dir=temp_dir,
        max_retries=3,
        min_checkpoint_interval=1,
        enable_emergency_checkpoints=True
    )
    
    # Test different error scenarios
    error_tests = [
        ("ValueError", training_with_value_error),
        ("RuntimeError", training_with_runtime_error),
        ("KeyError", training_with_key_error),
        ("NaNError", training_with_nan_error),
        ("Successful training", successful_training)
    ]
    
    for test_name, test_func in error_tests:
        logger.info(f"\n========== TESTING: {test_name} ==========")
        
        try:
            # Run test function with recovery system
            result = recovery_system.start_training_with_recovery(
                training_func=test_func,
                config=config.copy(),  # Use a copy of config
                df=df,
                save_path=temp_dir
            )
            
            # Check result
            if result and result[0]:
                logger.info(f"Test {test_name} completed successfully")
                model, metrics, reward, elapsed_time = result
                logger.info(f"Result: reward={reward}, time={elapsed_time:.2f}s")
                logger.info(f"Metrics: {metrics}")
            else:
                logger.info(f"Test {test_name} failed as expected")
                logger.info(f"Result: {result}")
            
        except Exception as e:
            logger.error(f"Unexpected error during test {test_name}: {e}")
        
        # Small pause between tests
        time.sleep(1)
    
    logger.info("\n===== ALL TESTS COMPLETED =====")
    
    # Show recovery status
    logger.info("\nRecovery system status:")
    status = recovery_system.get_recovery_status()
    for key, value in status.items():
        logger.info(f"  {key}: {value}")

if __name__ == "__main__":
    logger.info("Starting Training Recovery System tests")
    test_recovery_system() 