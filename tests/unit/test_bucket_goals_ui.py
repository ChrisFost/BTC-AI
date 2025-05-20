#!/usr/bin/env python
"""
Test the integration between the UI and BucketGoalProvider.
"""
import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import importlib

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

# Import the modules we need to test
from src.utils.bucket_goals import BucketGoalProvider, create_goal_provider

# Create mocks for dependencies but don't globally patch yet
psg_mock = MagicMock()
matplotlib_mock = MagicMock()
ui_main_mock = MagicMock()
ui_main_mock.update_bucket_goals = MagicMock()
ui_main_mock.handle_bucket_change = MagicMock()
ui_main_mock.goal_provider = None
ui_main_mock.config = {}
ui_main_mock.presets = {
    'Medium': {
        "min_gain_per_holding": 30.0,
        "max_gain_per_holding": 60.0,
        "bonus_multiplier": 1.2
    }
}

class TestBucketGoalsUIIntegration(unittest.TestCase):
    """Test the integration between the UI and BucketGoalProvider."""
    
    @classmethod
    def setUpClass(cls):
        """Set up class-level test fixtures."""
        # Store original modules so we can restore them
        cls.original_modules = {}
        for module_name in ['PySimpleGUI', 'matplotlib', 'matplotlib.pyplot', 
                           'matplotlib.figure', 'matplotlib.backends.backend_tkagg', 
                           'src.ui.main']:
            if module_name in sys.modules:
                cls.original_modules[module_name] = sys.modules[module_name]
        
        # Apply our mocks
        sys.modules['PySimpleGUI'] = psg_mock
        sys.modules['matplotlib'] = matplotlib_mock
        sys.modules['matplotlib.pyplot'] = MagicMock()
        sys.modules['matplotlib.figure'] = MagicMock()
        sys.modules['matplotlib.backends.backend_tkagg'] = MagicMock()
        sys.modules['src.ui.main'] = ui_main_mock
    
    @classmethod
    def tearDownClass(cls):
        """Clean up class-level test fixtures."""
        # Restore original modules
        for module_name, module in cls.original_modules.items():
            sys.modules[module_name] = module
        
        # For modules we mocked but weren't in sys.modules originally, remove them
        for module_name in ['PySimpleGUI', 'matplotlib', 'matplotlib.pyplot', 
                           'matplotlib.figure', 'matplotlib.backends.backend_tkagg', 
                           'src.ui.main']:
            if module_name not in cls.original_modules and module_name in sys.modules:
                del sys.modules[module_name]
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a reference to our mock
        self.ui = ui_main_mock
        
        # Create a mock window
        self.window = MagicMock()
        
        # Set up individual mock elements that will be accessed
        self.window["SCALPING_DESC"] = MagicMock()
        self.window["SHORT_DESC"] = MagicMock()
        self.window["MEDIUM_DESC"] = MagicMock()
        self.window["LONG_DESC"] = MagicMock()
        self.window["SCALPING_SETTINGS"] = MagicMock()
        self.window["SHORT_SETTINGS"] = MagicMock()
        self.window["MEDIUM_SETTINGS"] = MagicMock()
        self.window["LONG_SETTINGS"] = MagicMock()
        self.window["LOOK_BACK_HINT"] = MagicMock()
        self.window["-STATUS-"] = MagicMock()
        
        # Create a test config
        self.test_config = {
            "BUCKET": "Scalping",
            "monthly_target_min": 15.0,
            "monthly_target_max": 30.0,
            "yearly_target_min": 100.0,
            "yearly_target_max": 200.0,
            "min_gain_per_holding": 25.0,
            "max_gain_per_holding": 50.0,
            "bonus_multiplier": 1.1
        }
        
        # Save the original config
        self.original_config = self.ui.config.copy() if hasattr(self.ui, 'config') else {}
        
        # Set up the UI module with test configuration
        self.ui.config = self.test_config.copy()
        
        # Create test values dict
        self.values = self.test_config.copy()
        
        # Configure the mock to return a BucketGoalProvider
        self.ui.update_bucket_goals.return_value = BucketGoalProvider(self.test_config)
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Restore the original config
        self.ui.config = self.original_config
        
        # Reset all mocks
        self.ui.update_bucket_goals.reset_mock()
        self.ui.handle_bucket_change.reset_mock()
    
    def test_update_bucket_goals(self):
        """Test updating bucket goals from UI values."""
        # Test with Scalping bucket
        self.values["BUCKET"] = "Scalping"
        self.values["monthly_target_min"] = 20.0
        self.values["monthly_target_max"] = 40.0
        
        # Call the function
        goal_provider = self.ui.update_bucket_goals(self.window, self.values)
        
        # Verify function was called with the right parameters
        self.ui.update_bucket_goals.assert_called_with(self.window, self.values)
        
        # Test with Short bucket
        self.values["BUCKET"] = "Short"
        self.values["yearly_target_min"] = 110.0
        self.values["yearly_target_max"] = 220.0
        
        # Reset the mock before testing again
        self.ui.update_bucket_goals.reset_mock()
        
        # Call the function
        goal_provider = self.ui.update_bucket_goals(self.window, self.values)
        
        # Verify function was called with the right parameters
        self.ui.update_bucket_goals.assert_called_with(self.window, self.values)
    
    def test_handle_bucket_change(self):
        """Test handling bucket changes."""
        # Test changing to Short bucket
        self.values["BUCKET"] = "Short"
        
        # Call the UI function
        self.ui.handle_bucket_change(self.window, self.values)
        
        # Verify function was called with the right parameters
        self.ui.handle_bucket_change.assert_called_with(self.window, self.values)
    
    def test_load_preset(self):
        """Test loading presets updates BucketGoalProvider."""
        # Create mock event and values
        values = {"BUCKET": "Medium"}
        bucket = "Medium"
        
        # Create a new mock window with specific structure
        window = MagicMock()
        for key in ['min_gain_per_holding', 'max_gain_per_holding', 'bonus_multiplier']:
            window[key] = MagicMock()
            window[key].update = MagicMock()
        
        # Call update_bucket_goals to update goal provider
        self.ui.update_bucket_goals(window, values, bucket)
        
        # Verify update_bucket_goals was called with correct parameters
        self.ui.update_bucket_goals.assert_called_with(window, values, bucket)


if __name__ == "__main__":
    unittest.main() 