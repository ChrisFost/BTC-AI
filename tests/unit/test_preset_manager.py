import unittest
import os
import sys
import shutil
import json
import tempfile
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import the modules to test
try:
    from src.ui.preset_manager import (
        ensure_preset_directories,
        initialize_default_presets,
        load_preset,
        save_preset,
        list_presets,
        delete_preset,
        cleanup_temp_presets,
        update_preset_performance,
        load_performance_history,
        get_preset_suggestions,
        calculate_average_metrics,
        format_preset_suggestions,
        get_preset_suggestions_with_metrics,
        PRESET_DIR,
        DEFAULT_PRESETS_DIR,
        USER_PRESETS_DIR,
        TEMP_PRESETS_DIR,
        PRESETS_PERFORMANCE_FILE
    )
except ImportError:
    print("Warning: Could not import preset_manager module. Tests will be skipped.")
    PRESET_MANAGER_AVAILABLE = False
else:
    PRESET_MANAGER_AVAILABLE = True

@unittest.skipIf(not PRESET_MANAGER_AVAILABLE, "preset_manager module not available")
class TestPresetManager(unittest.TestCase):
    """Test the preset manager module."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a temporary directory for presets
        self.temp_dir = tempfile.mkdtemp()
        
        # Patch the preset directories
        self.patcher1 = patch('src.ui.preset_manager.PRESET_DIR', self.temp_dir)
        self.patcher2 = patch('src.ui.preset_manager.DEFAULT_PRESETS_DIR', os.path.join(self.temp_dir, "defaults"))
        self.patcher3 = patch('src.ui.preset_manager.USER_PRESETS_DIR', os.path.join(self.temp_dir, "user"))
        self.patcher4 = patch('src.ui.preset_manager.TEMP_PRESETS_DIR', os.path.join(self.temp_dir, "temp"))
        self.patcher5 = patch('src.ui.preset_manager.PRESETS_PERFORMANCE_FILE', 
                             os.path.join(self.temp_dir, "performance_history.json"))
        
        # Start the patchers
        self.patcher1.start()
        self.patcher2.start()
        self.patcher3.start()
        self.patcher4.start()
        self.patcher5.start()
        
        # Ensure the preset directories exist
        ensure_preset_directories()
    
    def tearDown(self):
        """Clean up after the tests."""
        # Stop the patchers
        self.patcher1.stop()
        self.patcher2.stop()
        self.patcher3.stop()
        self.patcher4.stop()
        self.patcher5.stop()
        
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_ensure_preset_directories(self):
        """Test that preset directories are created."""
        # Ensure the preset directories exist
        ensure_preset_directories()
        
        # Check that the directories were created
        self.assertTrue(os.path.exists(self.temp_dir))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "defaults")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "user")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "temp")))
    
    def test_initialize_default_presets(self):
        """Test that default presets are initialized."""
        # Initialize the default presets
        initialize_default_presets()
        
        # Check that the default presets were created
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "defaults", "Scalping")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "defaults", "Short")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "defaults", "Medium")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "defaults", "Long")))
        
        # Check that at least one preset exists for each bucket
        scalping_presets = os.listdir(os.path.join(self.temp_dir, "defaults", "Scalping"))
        self.assertGreater(len(scalping_presets), 0)
        self.assertTrue(any(preset.endswith('.json') for preset in scalping_presets))
    
    def test_save_and_load_preset(self):
        """Test saving and loading a preset."""
        # Create a test preset
        preset_params = {
            "BUCKET": "Scalping",
            "monthly_target_min": 20.0,
            "monthly_target_max": 35.0,
            "use_advanced_features": True
        }
        
        # Save the preset
        preset_id = save_preset(
            bucket="Scalping",
            name="Test Preset",
            params=preset_params,
            description="Test preset description",
            is_temporary=False
        )
        
        # Check that the preset was saved
        self.assertTrue(os.path.exists(preset_id))
        
        # Load the preset
        loaded_preset = load_preset(preset_id)
        
        # Check that the loaded preset matches the saved preset
        self.assertEqual(loaded_preset["name"], "Test Preset")
        self.assertEqual(loaded_preset["description"], "Test preset description")
        self.assertEqual(loaded_preset["params"], preset_params)
        self.assertIn("created", loaded_preset)
        self.assertIn("modified", loaded_preset)
    
    def test_list_presets(self):
        """Test listing presets."""
        # Create some test presets
        save_preset("Scalping", "Test1", {"BUCKET": "Scalping"}, "Test 1", False)
        save_preset("Scalping", "Test2", {"BUCKET": "Scalping"}, "Test 2", True)
        save_preset("Short", "Test3", {"BUCKET": "Short"}, "Test 3", False)
        
        # Initialize default presets
        initialize_default_presets()
        
        # List all presets
        all_presets = list_presets()
        
        # Check that at least 5 presets were found (3 custom + at least 2 defaults)
        self.assertGreaterEqual(len(all_presets), 5)
        
        # List only Scalping presets
        scalping_presets = list_presets(bucket="Scalping")
        
        # Check that at least 3 Scalping presets were found (2 custom + at least 1 default)
        self.assertGreaterEqual(len(scalping_presets), 3)
        
        # List only user presets
        user_presets = list_presets(include_defaults=False, include_temp=False)
        
        # Check that 2 user presets were found
        self.assertEqual(len(user_presets), 2)
        
        # List only temp presets
        temp_presets = list_presets(include_defaults=False, include_user=False, include_temp=True)
        
        # Check that 1 temp preset was found
        self.assertEqual(len(temp_presets), 1)
    
    def test_delete_preset(self):
        """Test deleting a preset."""
        # Create a test preset
        preset_id = save_preset("Scalping", "TestDelete", {"BUCKET": "Scalping"}, "Test Delete", False)
        
        # Check that the preset was saved
        self.assertTrue(os.path.exists(preset_id))
        
        # Delete the preset
        success = delete_preset(preset_id)
        
        # Check that the deletion was successful
        self.assertTrue(success)
        
        # Check that the preset was deleted
        self.assertFalse(os.path.exists(preset_id))
    
    def test_cleanup_temp_presets(self):
        """Test cleaning up temporary presets."""
        # Create a temporary preset
        preset_id = save_preset("Scalping", "TestTemp", {"BUCKET": "Scalping"}, "Test Temp", True)
        
        # Modify the creation date to be 8 days old
        preset_data = load_preset(preset_id)
        preset_data["created"] = (datetime.now() - timedelta(days=8)).isoformat()
        
        # Save the modified preset
        with open(preset_id, 'w') as f:
            json.dump(preset_data, f)
        
        # Clean up temporary presets older than 7 days
        deleted_count = cleanup_temp_presets(7)
        
        # Check that 1 preset was deleted
        self.assertEqual(deleted_count, 1)
        
        # Check that the preset was deleted
        self.assertFalse(os.path.exists(preset_id))
    
    def test_performance_tracking(self):
        """Test preset performance tracking."""
        # Create a test preset
        preset_id = save_preset("Scalping", "TestPerf", {"BUCKET": "Scalping"}, "Test Perf", False)
        
        # Check that the performance history file doesn't exist yet
        performance_file = os.path.join(self.temp_dir, "performance_history.json")
        self.assertFalse(os.path.exists(performance_file))
        
        # Update the preset performance
        metrics = {
            "net_profit": 1000.0,
            "win_rate": 0.65,
            "total_trades": 50,
            "winning_trades": 32,
            "losing_trades": 18,
            "max_drawdown": 0.15,
            "profit_factor": 2.5,
            "sharpe_ratio": 1.8
        }
        
        success = update_preset_performance(preset_id, metrics)
        
        # Check that the update was successful
        self.assertTrue(success)
        
        # Check that the performance history file was created
        self.assertTrue(os.path.exists(performance_file))
        
        # Load the performance history
        history = load_performance_history()
        
        # Check that the performance history contains the preset
        self.assertIn(preset_id, history)
        
        # Check that the performance history contains the metrics
        self.assertEqual(len(history[preset_id]), 1)
        self.assertEqual(history[preset_id][0]["metrics"], metrics)
    
    def test_suggestions(self):
        """Test preset suggestions."""
        # Create test presets with performance history
        preset1_id = save_preset("Scalping", "HighProfit", {"BUCKET": "Scalping"}, "High Profit", False)
        preset2_id = save_preset("Scalping", "LowRisk", {"BUCKET": "Scalping"}, "Low Risk", False)
        preset3_id = save_preset("Scalping", "Balanced", {"BUCKET": "Scalping"}, "Balanced", False)
        
        # Add performance metrics
        update_preset_performance(preset1_id, {
            "net_profit": 2000.0,
            "win_rate": 0.6,
            "max_drawdown": 0.2,
            "sharpe_ratio": 1.5
        })
        
        update_preset_performance(preset2_id, {
            "net_profit": 1000.0,
            "win_rate": 0.75,
            "max_drawdown": 0.1,
            "sharpe_ratio": 1.8
        })
        
        update_preset_performance(preset3_id, {
            "net_profit": 1500.0,
            "win_rate": 0.7,
            "max_drawdown": 0.15,
            "sharpe_ratio": 2.0
        })
        
        # Get suggestions by profit
        profit_suggestions = get_preset_suggestions("Scalping", "profit")
        
        # Check that the high profit preset is first
        self.assertEqual(profit_suggestions[0]["preset"]["id"], preset1_id)
        
        # Get suggestions by risk
        risk_suggestions = get_preset_suggestions("Scalping", "risk")
        
        # Check that the low risk preset is first
        self.assertEqual(risk_suggestions[0]["preset"]["id"], preset2_id)
        
        # Get suggestions by overall score
        overall_suggestions = get_preset_suggestions("Scalping", "overall")
        
        # Check that the balanced preset has the highest overall score
        self.assertEqual(overall_suggestions[0]["preset"]["id"], preset3_id)
        
        # Test formatting suggestions
        formatted, ids = get_preset_suggestions_with_metrics("Scalping", "overall")
        
        # Check that the correct number of suggestions were returned
        self.assertEqual(len(formatted), 3)
        self.assertEqual(len(ids), 3)
        
        # Check that the first formatted suggestion contains the expected data
        self.assertIn("Balanced", formatted[0])
        self.assertIn("User", formatted[0])
        self.assertIn("Profit", formatted[0])
        self.assertIn("Win", formatted[0])
        self.assertIn("DD", formatted[0])
    
    def test_calculate_average_metrics(self):
        """Test calculating average metrics."""
        # Create test performance history with multiple records
        history = [
            {
                "timestamp": datetime.now().isoformat(),
                "metrics": {
                    "net_profit": 1000.0,
                    "win_rate": 0.6,
                    "max_drawdown": 0.15,
                    "sharpe_ratio": 1.5,
                    "profit_factor": 2.0
                }
            },
            {
                "timestamp": datetime.now().isoformat(),
                "metrics": {
                    "net_profit": 1500.0,
                    "win_rate": 0.7,
                    "max_drawdown": 0.1,
                    "sharpe_ratio": 1.8,
                    "profit_factor": 2.5
                }
            }
        ]
        
        # Calculate average metrics
        avg_metrics = calculate_average_metrics(history)
        
        # Check the average metrics
        self.assertEqual(avg_metrics["net_profit"], 1250.0)
        self.assertAlmostEqual(avg_metrics["win_rate"], 0.65, places=6)
        self.assertAlmostEqual(avg_metrics["max_drawdown"], 0.125, places=6)
        self.assertAlmostEqual(avg_metrics["sharpe_ratio"], 1.65, places=6)
        self.assertAlmostEqual(avg_metrics["profit_factor"], 2.25, places=6)
        
        # Check that scores were calculated
        self.assertIn("profit_score", avg_metrics)
        self.assertIn("risk_score", avg_metrics)
        self.assertIn("overall_score", avg_metrics)


if __name__ == "__main__":
    unittest.main() 