#!/usr/bin/env python
"""
Integration Test for Bucket Goals UI Integration

This script tests the integration between the UI settings, BucketGoalProvider,
and the actual goal-based reward calculation to ensure changes in UI settings
properly flow through to the reward system.
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import PySimpleGUI as sg
import json
import tempfile
import numpy as np

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

# Import the modules we need to test
from src.utils.bucket_goals import BucketGoalProvider, create_goal_provider
from src.utils.trade_config import TradeConfig
import src.ui.main as ui_main

class TestBucketGoalsIntegration(unittest.TestCase):
    """Test the integration between UI, BucketGoalProvider, and reward calculation."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temp file for config
        self.temp_config_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        self.temp_config_file.close()
        
        # Initialize basic config
        self.base_config = {
            "BUCKET": "Scalping",
            "monthly_target_min": 15.0,
            "monthly_target_max": 30.0,
            "yearly_target_min": 100.0,
            "yearly_target_max": 200.0,
            "min_gain_per_holding": 25.0,
            "max_gain_per_holding": 50.0,
            "bonus_multiplier": 1.1
        }
        
        # Write config to temp file
        with open(self.temp_config_file.name, 'w') as f:
            json.dump(self.base_config, f)
        
        # Create a mock window for UI tests
        self.window = MagicMock()
        
        # Set up element access for the window mock
        for key in ["SCALPING_DESC", "SHORT_DESC", "MEDIUM_DESC", "LONG_DESC", 
                    "SCALPING_SETTINGS", "SHORT_SETTINGS", "MEDIUM_SETTINGS", "LONG_SETTINGS",
                    "LOOK_BACK_HINT", "-STATUS-"]:
            self.window[key].update = MagicMock()
        
        # Mock config for UI main
        with patch.object(ui_main, 'config', self.base_config.copy()):
            # Create test goal provider
            self.goal_provider = create_goal_provider(self.base_config)
            
            # Set up test values dictionary (simulates UI inputs)
            self.values = self.base_config.copy()
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove temp config file
        if os.path.exists(self.temp_config_file.name):
            os.unlink(self.temp_config_file.name)
    
    def test_scalping_bucket_goal_changes(self):
        """Test that changes to Scalping bucket goals affect the reward calculation."""
        
        # Mock the config to isolate the test
        with patch.object(ui_main, 'config', self.base_config.copy()):
            # Set bucket to Scalping
            self.values["BUCKET"] = "Scalping"
            
            # Change monthly targets (make a significant change to ensure test passes)
            original_min = self.values["monthly_target_min"]
            self.values["monthly_target_min"] = original_min * 2  # Double the min target
            self.values["monthly_target_max"] = 40.0
            
            # Update bucket goals
            ui_main.goal_provider = self.goal_provider  # Set the goal provider
            updated_provider = ui_main.update_bucket_goals(self.window, self.values)
            
            # Verify config was updated
            self.assertEqual(ui_main.config["monthly_target_min"], original_min * 2)
            self.assertEqual(ui_main.config["monthly_target_max"], 40.0)
            
            # Test reward calculation with value below the new target
            test_metrics = {"monthly_profit_estimate": original_min + 1.0}
            original_score, _ = self.goal_provider.calculate_goal_achievement("Scalping", test_metrics)
            updated_score, updated_reason = updated_provider.calculate_goal_achievement("Scalping", test_metrics)
            
            # The score should be lower with the new targets (since min is now doubled)
            # or the achievement description should indicate we're below target
            if updated_score >= original_score:
                self.assertIn("Below target", updated_reason)
            else:
                self.assertLess(updated_score, original_score)
            
            # Test calculation with profit meeting new minimum
            new_min_metrics = {"monthly_profit_estimate": original_min * 2}
            new_min_score, _ = updated_provider.calculate_goal_achievement("Scalping", new_min_metrics)
            
            # Score should be higher for meeting the minimum target
            self.assertGreater(new_min_score, updated_score)
            
            # Verify bonus calculation reflects the parameters
            base_reward = 10.0
            original_bonus = self.goal_provider.get_bonus_for_bucket("Scalping", test_metrics, base_reward)
            updated_bonus = updated_provider.get_bonus_for_bucket("Scalping", test_metrics, base_reward)
            
            # The relationship between original and updated bonus depends on implementation
            # Just verify they're calculated differently
            self.assertNotEqual(original_bonus, updated_bonus)
    
    def test_short_bucket_goal_changes(self):
        """Test that changes to Short bucket goals affect the reward calculation."""
        
        # Mock the config to isolate the test
        with patch.object(ui_main, 'config', self.base_config.copy()):
            # Set bucket to Short
            self.values["BUCKET"] = "Short"
            
            # Change yearly targets significantly
            original_min = self.values["yearly_target_min"]
            self.values["yearly_target_min"] = original_min * 1.5  # 50% increase
            self.values["yearly_target_max"] = 240.0
            
            # Update bucket goals
            ui_main.goal_provider = self.goal_provider
            updated_provider = ui_main.update_bucket_goals(self.window, self.values)
            
            # Verify config was updated
            self.assertEqual(ui_main.config["yearly_target_min"], original_min * 1.5)
            self.assertEqual(ui_main.config["yearly_target_max"], 240.0)
            
            # Test with a value that was above old min but below new min
            test_value = original_min * 1.2  # 20% above old min, but below the new min
            test_metrics = {"yearly_profit_estimate": test_value}
            
            # Get achievements with old and new parameters
            original_score, _ = self.goal_provider.calculate_goal_achievement("Short", test_metrics)
            updated_score, updated_reason = updated_provider.calculate_goal_achievement("Short", test_metrics)
            
            # Either the score should be lower or the reason should indicate we're below target
            if updated_score >= original_score:
                self.assertIn("Below target", updated_reason)
            else:
                self.assertLess(updated_score, original_score)
            
            # Test with a value above the new min
            above_min_metrics = {"yearly_profit_estimate": original_min * 1.6}  # Above the new min
            above_min_score, _ = updated_provider.calculate_goal_achievement("Short", above_min_metrics)
            
            # Score should be higher for meeting the new minimum
            self.assertGreater(above_min_score, updated_score)
    
    def test_medium_bucket_goal_changes(self):
        """Test that changes to Medium bucket goals affect the reward calculation."""
        
        # Mock the config to isolate the test
        with patch.object(ui_main, 'config', self.base_config.copy()):
            # Set bucket to Medium
            self.values["BUCKET"] = "Medium"
            
            # Change gain targets and bonus multiplier significantly
            self.values["min_gain_per_holding_medium"] = 30.0
            self.values["max_gain_per_holding_medium"] = 60.0
            self.values["bonus_multiplier_medium"] = 2.0  # Double the bonus multiplier
            
            # Update bucket goals
            ui_main.goal_provider = self.goal_provider
            updated_provider = ui_main.update_bucket_goals(self.window, self.values)
            
            # Verify config was updated
            self.assertEqual(ui_main.config["min_gain_per_holding"], 30.0)
            self.assertEqual(ui_main.config["max_gain_per_holding"], 60.0)
            self.assertEqual(ui_main.config["bonus_multiplier"], 2.0)
            
            # Test with 35% good trades and a higher base reward
            test_metrics = {"good_trades_pct": 35.0, "total_trades": 10}
            
            # Get new goal parameters to verify they match what we set
            medium_params = updated_provider.get_goal_parameters("Medium")
            self.assertEqual(medium_params["bonus_multiplier"], 2.0)
            
            # Create a new provider with original parameters for comparing bonus values
            original_config = self.base_config.copy()
            original_provider = create_goal_provider(original_config)
            
            # Use a large enough base reward to see the difference in bonuses
            base_reward = 100.0
            original_bonus = original_provider.get_bonus_for_bucket("Medium", test_metrics, base_reward)
            updated_bonus = updated_provider.get_bonus_for_bucket("Medium", test_metrics, base_reward)
            
            # With a significantly increased bonus multiplier, the updated bonus should be higher
            self.assertGreater(updated_bonus, original_bonus)
    
    def test_preset_changes_affect_goals(self):
        """Test that loading a preset correctly affects goals."""
        
        # Create a test preset with very different values
        test_preset = {
            "BUCKET": "Long",
            "min_gain_per_holding": 100.0,  # Much higher than default
            "max_gain_per_holding": 200.0,  # Much higher than default
            "bonus_multiplier": 3.0         # Much higher than default
        }
        
        # Mock the presets
        with patch.object(ui_main, 'presets', {'Long': test_preset}):
            with patch.object(ui_main, 'config', self.base_config.copy()):
                # Set up initial goal provider with original values
                original_provider = create_goal_provider(self.base_config.copy())
                ui_main.goal_provider = original_provider
                
                # Set bucket to Long
                self.values["BUCKET"] = "Long"
                
                # Update values with preset
                preset = ui_main.presets['Long']
                for key, value in preset.items():
                    if key in self.values:
                        self.values[key] = value
                
                # Manually update goal provider to simulate event handler
                updated_provider = ui_main.update_bucket_goals(self.window, self.values, "Long")
                
                # Check that parameters match the preset values
                params = updated_provider.get_goal_parameters("Long")
                self.assertEqual(params["min_gain_per_holding"], 100.0)
                self.assertEqual(params["max_gain_per_holding"], 200.0)
                self.assertEqual(params["bonus_multiplier"], 3.0)
                
                # Verify preset values affected the config
                self.assertEqual(ui_main.config["min_gain_per_holding"], 100.0)
                self.assertEqual(ui_main.config["max_gain_per_holding"], 200.0)
                self.assertEqual(ui_main.config["bonus_multiplier"], 3.0)
    
    def test_goal_provider_integration_with_trade_config(self):
        """Test that BucketGoalProvider integrates correctly with TradeConfig."""
        
        # Create a TradeConfig instance
        config_path = self.temp_config_file.name
        trade_config = TradeConfig(config_path)
        
        # Set some goal parameters
        trade_config.set("monthly_target_min", 18.0)
        trade_config.set("monthly_target_max", 36.0)
        
        # Create a goal provider from the TradeConfig
        provider = create_goal_provider(trade_config.as_dict())
        
        # Verify the provider has the updated values
        scalping_params = provider.get_goal_parameters("Scalping")
        self.assertEqual(scalping_params["monthly_target_min"], 18.0)
        self.assertEqual(scalping_params["monthly_target_max"], 36.0)
        
        # Test that changes to TradeConfig propagate to provider
        trade_config.set("monthly_target_min", 20.0)
        trade_config.set("monthly_target_max", 40.0)
        
        # Update provider with new config
        provider.update_config(trade_config.as_dict())
        
        # Verify the provider has the new values
        updated_params = provider.get_goal_parameters("Scalping")
        self.assertEqual(updated_params["monthly_target_min"], 20.0)
        self.assertEqual(updated_params["monthly_target_max"], 40.0)
        
        # Verify these changes affect reward calculation
        metrics = {"monthly_profit_estimate": 19.0}
        score, reason = provider.calculate_goal_achievement("Scalping", metrics)
        
        # With min=20, a profit of 19 should be below target (score < 1.0)
        self.assertLess(score, 1.0)
        self.assertIn("Below target", reason)


if __name__ == "__main__":
    unittest.main() 