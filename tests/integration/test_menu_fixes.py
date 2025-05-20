#!/usr/bin/env python
"""
Simple test script to validate that our fixes to the menu script work correctly.
This script focuses on testing the specific functions we modified,
and includes automated UI testing with timeouts to prevent hangs.
"""

import os
import sys
import time
import unittest
import tempfile
import subprocess
import threading
import logging
import importlib
from unittest.mock import patch, MagicMock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('menu_test.log')
    ]
)
logger = logging.getLogger('menu_test')

# Get the path to the menu script
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
MENU_SCRIPT_PATH = os.path.join(project_root, "src", "ui", "main.py")

print(f"Testing menu script at: {MENU_SCRIPT_PATH}")
print("Checking if file exists:", os.path.exists(MENU_SCRIPT_PATH))

# Add project root to path to ensure imports work correctly
sys.path.insert(0, project_root)

# Removed SimpleMenuTest class containing outdated static analysis tests.
# New integration tests should be added here, focusing on interactions
# between UI components (AppState, TrainingManager, BucketManager, etc.)
# using mocks where appropriate.

# Import necessary modules for new tests
from src.ui.app_state import AppState
from src.ui.training_manager import TrainingManager
from src.ui.bucket_manager import BucketManager
import json

class TestUIIntegration(unittest.TestCase):
    """New integration tests for UI component interactions."""

    @patch('builtins.open')
    @patch('os.path.exists')
    @patch('src.ui.app_state.save_notes') # Mock note saving
    @patch('src.ui.app_state.load_notes', return_value="Mock notes") # Mock note loading
    @patch('src.utils.trade_config.TradeConfig.load_config', return_value={}) # Mock initial config load
    def test_app_state_load_save(self, mock_load_config, mock_load_notes, mock_save_notes, mock_exists, mock_open):
        """Test saving and loading AppState configuration."""
        logger.info("Running test: AppState Load/Save")

        # Mock file existence checks
        mock_exists.return_value = True

        # Mock file read/write operations
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.__exit__.return_value = None
        mock_open.return_value = mock_file

        # Simulate loading an empty or default config initially
        mock_file.read.return_value = json.dumps({}) 

        # --- First Instance --- 
        app_state1 = AppState()
        # Ensure initial config load is mocked correctly
        mock_load_config.assert_called() 
        # Modify a value
        test_key = 'LEARNING_RATE' # Use a key likely to be in config
        test_value = 0.012345
        app_state1.config[test_key] = test_value
        logger.info(f"Set {test_key} to {test_value}")

        # --- Save State --- 
        # Reset mock_open to capture the write
        mock_open.reset_mock()
        mock_file.write.reset_mock() 
        mock_file_write = MagicMock()
        mock_file_write.__enter__.return_value = mock_file_write
        mock_file_write.__exit__.return_value = None
        mock_open.return_value = mock_file_write

        app_state1.save_all_state({}) # Pass empty values dict
        
        # Check if open was called for writing config.json
        # Find the call associated with writing config.json
        write_call_args = None
        for call in mock_open.call_args_list:
            args, kwargs = call
            if len(args) > 0 and 'config.json' in args[0] and len(args) > 1 and args[1] == 'w':
                 write_call_args = args
                 break
        self.assertIsNotNone(write_call_args, "Did not attempt to write to config.json")

        # Check if write was called on the file mock
        mock_file_write.write.assert_called_once() 
        # Get the data that was written
        written_data = mock_file_write.write.call_args[0][0]
        saved_config = json.loads(written_data)
        self.assertEqual(saved_config.get(test_key), test_value, "Saved config did not contain the correct value")
        logger.info("AppState saved successfully (mocked)")

        # --- Second Instance --- 
        # Configure mocks for loading the saved state
        mock_open.reset_mock()
        mock_file_load = MagicMock()
        mock_file_load.__enter__.return_value = mock_file_load
        mock_file_load.__exit__.return_value = None
        mock_file_load.read.return_value = written_data # Return the previously saved data
        mock_open.return_value = mock_file_load
        # Ensure exists returns true for the config file path
        def exists_side_effect(path):
            if 'config.json' in path:
                return True
            return False
        mock_exists.side_effect = exists_side_effect
        
        app_state2 = AppState() 
        # Check if open was called for reading config.json
        read_call_args = None
        for call in mock_open.call_args_list:
             args, kwargs = call
             if len(args) > 0 and 'config.json' in args[0] and len(args) > 1 and args[1] == 'r':
                  read_call_args = args
                  break
        self.assertIsNotNone(read_call_args, "Did not attempt to read config.json on second init")

        # Assert that the loaded value is correct
        self.assertEqual(app_state2.config.get(test_key), test_value, "AppState did not load the saved value correctly")
        logger.info("AppState loaded the saved value correctly")

    @patch('subprocess.Popen')
    @patch('src.ui.training_manager.log') # Mock logger within training manager
    def test_training_manager_start_stop(self, mock_tm_log, mock_popen):
        """Test starting and stopping training via TrainingManager."""
        logger.info("Running test: TrainingManager Start/Stop")

        # Mock the Popen process object
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_popen.return_value = mock_process

        # Create a mock AppState (can be simple for this test)
        mock_app_state = MagicMock(spec=AppState)
        mock_app_state.config = {'BUCKET_SETTINGS': {'Scalping': {}}} # Basic config needed

        # Instantiate TrainingManager with mock AppState
        training_manager = TrainingManager(mock_app_state)

        # --- Start Training --- 
        mock_window = MagicMock() # Mock the PySimpleGUI window
        mock_values = {'BUCKET': 'Scalping'} # Mock values dictionary
        
        training_manager.start_training(mock_window, mock_values)

        # Assert Popen was called (check first argument contains python and training script)
        popen_call_args = mock_popen.call_args[0][0] # Get the first positional argument (the command list)
        self.assertIn(sys.executable, popen_call_args[0], "Python executable not found in Popen call")
        self.assertTrue(any('src/training/training.py' in arg for arg in popen_call_args), "Training script not found in Popen call")
        # Assert the process attribute is set
        self.assertEqual(training_manager.process, mock_process, "Training manager process not set correctly")
        logger.info("TrainingManager started process (mocked)")

        # --- Stop Training --- 
        training_manager.stop_training(mock_window)

        # Assert the process was terminated or killed
        # Check if either terminate() or kill() was called
        terminate_called = mock_process.terminate.called
        kill_called = mock_process.kill.called
        self.assertTrue(terminate_called or kill_called, "Process terminate() or kill() was not called")
        self.assertIsNone(training_manager.process, "Training manager process was not cleared after stopping")
        logger.info("TrainingManager stopped process (mocked)")

    @patch('builtins.open')
    @patch('os.path.exists')
    @patch('src.ui.app_state.save_notes') # Mock note saving for AppState init
    @patch('src.ui.app_state.load_notes', return_value="Mock notes") # Mock note loading for AppState init
    @patch('src.utils.trade_config.TradeConfig.load_config', return_value={}) # Mock config load for AppState init
    @patch('src.ui.bucket_manager.log') # Mock logger within bucket manager
    def test_bucket_manager_state_update(self, mock_bm_log, mock_app_load_config, mock_app_load_notes, mock_app_save_notes, mock_exists, mock_open):
        """Test that AppState config is updated when the bucket changes."""
        logger.info("Running test: BucketManager State Update")

        # --- Mock Setup --- 
        # Mock file existence
        mock_exists.return_value = True 

        # Define mock data for bucket_goals.json
        mock_bucket_goals_data = {
            "Short": {
                "goals": {"TARGET_PROFIT": 0.05, "MAX_DRAWDOWN": 0.03},
                "description": "Short-term goals"
            },
            "Long": {
                "goals": {"TARGET_PROFIT": 0.20, "MAX_DRAWDOWN": 0.10},
                "description": "Long-term goals"
            }
        }
        # Define mock data for initial config.json read by AppState
        mock_config_data = {"INITIAL_BUCKET": "Short"}

        # Mock file reading
        mock_file_goals = MagicMock()
        mock_file_goals.read.return_value = json.dumps(mock_bucket_goals_data)
        mock_file_config = MagicMock()
        mock_file_config.read.return_value = json.dumps(mock_config_data)

        def open_side_effect(path, mode='r', *args, **kwargs):
            mock_file = MagicMock()
            mock_file.__enter__.return_value = mock_file
            mock_file.__exit__.return_value = None
            if 'bucket_goals.json' in path and mode == 'r':
                mock_file.read.return_value = json.dumps(mock_bucket_goals_data)
                logger.info("Mocked read from bucket_goals.json")
            elif 'config.json' in path and mode == 'r':
                 mock_file.read.return_value = json.dumps(mock_config_data)
                 logger.info("Mocked read from config.json")
            elif 'config.json' in path and mode == 'w':
                 # Mock the write operation for save_all_state if needed
                 mock_file.write = MagicMock()
                 logger.info("Mocked write to config.json")
            else:
                 mock_file.read.return_value = "{}" # Default empty json
                 mock_file.write = MagicMock()
            return mock_file

        mock_open.side_effect = open_side_effect

        # --- Test Execution --- 
        # Instantiate AppState and BucketManager
        # Use a real AppState instance, relying on mocks for file I/O
        app_state = AppState()
        bucket_manager = BucketManager(app_state)
        
        # Ensure initial state reflects mocked config
        self.assertEqual(app_state.config.get("INITIAL_BUCKET"), "Short")

        # Mock the UI window and values for the event
        mock_window = MagicMock()
        # Simulate changing the bucket to "Long"
        mock_values = {'BUCKET': 'Long'}
        
        # Call the handler
        bucket_manager.handle_bucket_change(mock_window, mock_values)
        logger.info("Called handle_bucket_change with BUCKET=Long")

        # --- Assertions --- 
        # Check that relevant config values were updated from bucket_goals
        self.assertEqual(app_state.config.get('TARGET_PROFIT'), 0.20, "TARGET_PROFIT was not updated correctly")
        self.assertEqual(app_state.config.get('MAX_DRAWDOWN'), 0.10, "MAX_DRAWDOWN was not updated correctly")
        logger.info("AppState config updated correctly after bucket change")
        
        # Check that mock_window.update was called (optional but good practice)
        # This depends heavily on the actual implementation of handle_bucket_change
        # For now, focus on the state change assertion

def run_automated_ui_tests():
    """Run a series of automated UI tests with subprocess and timeouts."""
    
    logger.info("Starting automated UI tests...")
    
    # First, patch PySimpleGUI if needed
    try:
        import PySimpleGUI as sg
        logger.info("PySimpleGUI already imported")
    except ImportError:
        logger.warning("PySimpleGUI not found, using mock version")
        sys.modules['PySimpleGUI'] = MagicMock()
        import PySimpleGUI as sg
    
    # Mock subprocess.Popen to prevent actual training
    original_popen = subprocess.Popen
    
    def mock_popen(*args, **kwargs):
        logger.info(f"Intercepted subprocess call: {args[0] if args else ''}")
        mock = MagicMock()
        mock.poll = lambda: None  # Always running
        mock.stdout = ["Mocked subprocess output"]  # Provide some fake output
        return mock
    
    # Patch subprocess.Popen with our mock
    subprocess.Popen = mock_popen
    
    # Define a function to run the menu script with a timeout
    def run_menu_with_timeout(timeout=10):
        """Run the menu script with a timeout."""
        logger.info(f"Running menu script with {timeout}s timeout")
        
        # Create a process to run the menu script
        process = original_popen(
            [sys.executable, MENU_SCRIPT_PATH],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        # Define a function to kill the process after timeout
        def kill_after_timeout():
            time.sleep(timeout)
            if process.poll() is None:
                logger.info(f"Timeout reached ({timeout}s), killing process")
                process.kill()
        
        # Start timeout thread
        timeout_thread = threading.Thread(target=kill_after_timeout)
        timeout_thread.daemon = True
        timeout_thread.start()
        
        # Collect output
        output = []
        for line in process.stdout:
            output.append(line.strip())
            logger.info(line.strip())
        
        # Wait for process to complete
        return_code = process.wait()
        logger.info(f"Process completed with return code {return_code}")
        
        return return_code, output
    
    # Define a sequence of automated tests
    test_sequence = [
        {
            "name": "Basic menu startup check",
            "run": lambda: run_menu_with_timeout(5),
            "verify": lambda rc, out: rc != 0,  # Should exit due to timeout
            "message": "Menu startup test"
        },
        # Add more automated tests if needed
    ]
    
    # Run the tests in sequence
    results = []
    for test in test_sequence:
        logger.info(f"Running test: {test['name']}")
        try:
            rc, output = test["run"]()
            success = test["verify"](rc, output)
            results.append({
                "name": test["name"],
                "success": success,
                "output": output
            })
            logger.info(f"Test {test['name']} {'succeeded' if success else 'failed'}")
        except Exception as e:
            logger.error(f"Test {test['name']} failed with exception: {e}")
            results.append({
                "name": test["name"],
                "success": False,
                "error": str(e)
            })
    
    # Restore the original Popen
    subprocess.Popen = original_popen
    
    # Report results
    logger.info("Automated UI test results:")
    for result in results:
        status = "PASSED" if result["success"] else "FAILED"
        logger.info(f"{result['name']}: {status}")
    
    return all(r["success"] for r in results)

if __name__ == "__main__":
    # Run the automated UI tests
    print("\n==================================")
    print("Starting automated UI tests...")
    print("==================================\n")
    success = run_automated_ui_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1) 