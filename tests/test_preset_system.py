#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for the parameter preset system functionality.
This script tests saving, loading, filtering, and performance tracking.
"""

import os
import sys
import unittest
import json
from datetime import datetime
import shutil

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.ui.preset_manager import (
        save_preset, 
        load_preset, 
        delete_preset, 
        update_preset_performance,
        get_preset_suggestions_with_metrics,
        PRESET_DIR
    )
except ImportError:
    print("Could not import preset_manager. Make sure you're running from the project root.")
    sys.exit(1)

class TestPresetSystem(unittest.TestCase):
    """Test cases for the parameter preset system"""
    
    def setUp(self):
        """Set up test environment"""
        # Create a test directory
        self.test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_presets")
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Sample test parameters
        self.test_params = {
            "LOOK_BACK_AMOUNT": 15,
            "LOOK_BACK_UNIT": "minutes",
            "TIME_HORIZON": 15, 
            "HORIZON_UNIT": "minutes",
            "ENTRY_THRESHOLD": 0.65,
            "EXIT_THRESHOLD": 0.55,
        }
        
        # Sample performance metrics
        self.test_metrics = {
            "net_profit": 1200.50,
            "win_rate": 0.65,
            "max_drawdown": 0.12,
            "sharpe_ratio": 1.35,
            "total_trades": 45,
            "profit_factor": 1.8
        }
        
        # Test preset IDs
        self.preset_ids = []
    
    def tearDown(self):
        """Clean up after tests"""
        # Delete test presets
        for preset_id in self.preset_ids:
            try:
                delete_preset(preset_id)
            except:
                pass
        
        # Remove test directory
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_save_and_load_preset(self):
        """Test saving and loading a preset"""
        # Save a test preset
        preset_id = save_preset(
            bucket="Scalping",
            name="Test Preset",
            params=self.test_params,
            description="Test preset for unit testing"
        )
        
        self.preset_ids.append(preset_id)
        self.assertIsNotNone(preset_id, "Preset should be saved successfully")
        
        # Load the preset
        preset_data = load_preset(preset_id)
        self.assertIsNotNone(preset_data, "Preset should be loaded successfully")
        self.assertEqual(preset_data.get("name"), "Test Preset", "Preset name should match")
        self.assertEqual(preset_data.get("bucket_type"), "Scalping", "Bucket type should match")
        
        # Verify parameters
        params = preset_data.get("parameters", {})
        for key, value in self.test_params.items():
            self.assertEqual(params.get(key), value, f"Parameter {key} should match")
    
    def test_performance_tracking(self):
        """Test tracking performance metrics for a preset"""
        # Save a test preset
        preset_id = save_preset(
            bucket="Scalping",
            name="Performance Test Preset",
            params=self.test_params
        )
        
        self.preset_ids.append(preset_id)
        
        # Update performance multiple times
        for i in range(3):
            # Modify metrics slightly for each update
            metrics = self.test_metrics.copy()
            metrics["net_profit"] += i * 100
            metrics["win_rate"] += i * 0.05
            
            # Update performance
            success = update_preset_performance(preset_id, metrics)
            self.assertTrue(success, "Performance update should succeed")
        
        # Get suggestions to verify performance is tracked
        suggestions = get_preset_suggestions_with_metrics("Scalping")
        
        # Find our test preset in suggestions
        found = False
        for suggestion in suggestions:
            if suggestion.get("id") == preset_id:
                found = True
                # Verify metrics are averaged correctly
                self.assertGreaterEqual(suggestion.get("net_profit", 0), self.test_metrics["net_profit"])
                self.assertGreaterEqual(suggestion.get("win_rate", 0), self.test_metrics["win_rate"])
                break
        
        self.assertTrue(found, "Test preset should be in suggestions with performance metrics")
    
    def test_filtering_suggestions(self):
        """Test filtering suggestions by different criteria"""
        # Create multiple presets with different performance profiles
        presets_data = [
            {
                "name": "High Profit Preset",
                "bucket": "Scalping",
                "params": self.test_params,
                "metrics": {
                    "net_profit": 2000.0,
                    "win_rate": 0.6,
                    "max_drawdown": 0.2,
                    "sharpe_ratio": 1.2,
                    "total_trades": 50
                }
            },
            {
                "name": "Low Risk Preset",
                "bucket": "Scalping",
                "params": self.test_params,
                "metrics": {
                    "net_profit": 800.0,
                    "win_rate": 0.7,
                    "max_drawdown": 0.05,
                    "sharpe_ratio": 1.8,
                    "total_trades": 40
                }
            },
            {
                "name": "Balanced Preset",
                "bucket": "Scalping",
                "params": self.test_params,
                "metrics": {
                    "net_profit": 1500.0,
                    "win_rate": 0.65,
                    "max_drawdown": 0.1,
                    "sharpe_ratio": 1.5,
                    "total_trades": 45
                }
            }
        ]
        
        # Save presets and add performance data
        for data in presets_data:
            preset_id = save_preset(
                bucket=data["bucket"],
                name=data["name"],
                params=data["params"]
            )
            self.preset_ids.append(preset_id)
            
            # Add performance metrics
            update_preset_performance(preset_id, data["metrics"])
        
        # Test profit filter
        profit_suggestions = get_preset_suggestions_with_metrics("Scalping", filter_type="profit")
        self.assertGreaterEqual(len(profit_suggestions), 1, "Should have at least one suggestion")
        
        # Verify the high profit preset is ranked highest
        self.assertEqual(profit_suggestions[0].get("name"), "High Profit Preset", 
                       "High Profit Preset should be top ranked with profit filter")
        
        # Test risk filter
        risk_suggestions = get_preset_suggestions_with_metrics("Scalping", filter_type="risk")
        self.assertGreaterEqual(len(risk_suggestions), 1, "Should have at least one suggestion")
        
        # Verify the low risk preset is ranked highest
        self.assertEqual(risk_suggestions[0].get("name"), "Low Risk Preset",
                       "Low Risk Preset should be top ranked with risk filter")

if __name__ == "__main__":
    unittest.main() 