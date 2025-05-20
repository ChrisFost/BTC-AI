"""
Environment-Configuration Integration Test

This test verifies that the environment can be properly initialized with preset-based
configuration during configuration fallback scenarios.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import with fallback handling to simulate how backtesting would handle this
try:
    from src.utils.config_bridge import get_preset_default_config
    from src.environment.env_base import create_environment
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Cannot run test without these modules")
    raise


class TestEnvironmentConfigIntegration(unittest.TestCase):
    """
    Tests that verify the environment and configuration integration,
    especially during fallback scenarios.
    """
    
    def setUp(self):
        """Set up test data"""
        # Create a simple test dataframe
        date_range = pd.date_range(start='2023-01-01', periods=100, freq='5min')
        self.test_df = pd.DataFrame({
            'timestamp': date_range,
            'open': np.random.normal(50000, 1000, 100),
            'high': np.random.normal(51000, 1000, 100),
            'low': np.random.normal(49000, 1000, 100),
            'close': np.random.normal(50500, 1000, 100),
            'volume': np.random.normal(10, 5, 100),
        })
        
        # Add a few common technical indicators for realism
        self.test_df['SMA9'] = self.test_df['close'].rolling(9).mean()
        self.test_df['SMA21'] = self.test_df['close'].rolling(21).mean()
        self.test_df['RSI14'] = np.random.normal(50, 15, 100)  # Simplified RSI
        
        # Fill NaN values
        self.test_df = self.test_df.fillna(method='bfill')
    
    def test_create_environment_with_direct_config(self):
        """Test that environment creates successfully with direct config"""
        # Basic configuration
        config = {
            "INITIAL_CAPITAL": 100000,
            "BUCKET": "Scalping",
            "RISK_LEVEL": "medium",
            "MAX_POSITION_SIZE": 0.1,
            "REWARD_TYPE": "sharpe"
        }
        
        # Create environment with direct config
        env = create_environment(self.test_df, config)
        
        # Verify the environment was created with proper parameters
        self.assertIsNotNone(env)
        self.assertEqual(env.initial_capital, 100000)
        self.assertEqual(env.bucket, "Scalping")
    
    def test_create_environment_with_preset_config(self):
        """Test environment creation using a preset-based configuration"""
        # Get a preset config
        preset_config = get_preset_default_config("Scalping")
        
        # Ensure we have a valid preset (or create a minimal one)
        if not preset_config:
            preset_config = {
                "BUCKET": "Scalping",
                "INITIAL_CAPITAL": 100000,
                "RISK_LEVEL": "medium"
            }
        
        # Create a PresetBasedConfig class similar to the one in backtesting.py
        class PresetBasedConfig:
            def __init__(self, preset_data):
                self.config = preset_data or {
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
        
        # Create config wrapper and convert to dict for environment
        config_wrapper = PresetBasedConfig(preset_config)
        config_dict = config_wrapper.as_dict()
        
        # Create environment using the preset-based config
        env = create_environment(self.test_df, config_dict)
        
        # Verify the environment was created with proper parameters
        self.assertIsNotNone(env)
        self.assertEqual(env.bucket, "Scalping")
        
        # Test that critical environment components are properly initialized
        self.assertIsNotNone(env.df)
        self.assertIsNotNone(env.risk_manager)
        self.assertIsNotNone(env.reward_function)
    
    def test_fallback_scenario_simulation(self):
        """
        Simulate a fallback scenario where main config fails to load
        and we use the preset system as fallback
        """
        # Create a minimal fallback config
        minimal_config = {
            "INITIAL_CAPITAL": 100000,
            "RISK_LEVEL": "medium",
            "MAX_POSITION_SIZE": 0.1,
            "BUCKET": "Scalping"
        }
        
        # Patch the get_preset_default_config to simulate failure
        with patch('src.utils.config_bridge.get_preset_default_config', 
                  side_effect=Exception("Preset system unavailable")):
            
            # Create environment with minimal config
            env = create_environment(self.test_df, minimal_config)
            
            # Verify environment still works
            self.assertIsNotNone(env)
            self.assertEqual(env.initial_capital, 100000)
    
    def test_essential_parameters_provided(self):
        """Test that essential parameters are always provided to the environment"""
        # Get a preset config
        preset_config = get_preset_default_config("Scalping")
        
        if preset_config:
            # Create environment with preset config
            env = create_environment(self.test_df, preset_config)
            
            # Verify essential parameters
            self.assertTrue(hasattr(env, 'initial_capital'))
            self.assertTrue(hasattr(env, 'bucket'))
            self.assertTrue(hasattr(env, 'reward_function'))
            self.assertTrue(hasattr(env, 'observation_space'))
            self.assertTrue(hasattr(env, 'action_space'))
            
            # Check for risk manager
            self.assertTrue(hasattr(env, 'risk_manager'))
            
            # Verify environment functionality still works
            obs = env.reset()
            self.assertIsNotNone(obs)
            
            # Try taking a simple action
            action = [0.1, 0.5]  # Simple action for testing
            obs, reward, done, info = env.step(action)
            self.assertIsNotNone(reward)
            self.assertIsInstance(info, dict)


if __name__ == '__main__':
    unittest.main() 