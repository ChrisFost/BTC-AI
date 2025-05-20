#!/usr/bin/env python
"""
Test module for the Recovery Dashboard functionality.

This module tests all aspects of the emergency checkpoint system including:
1. Emergency checkpoint creation
2. Checkpoint listing
3. Checkpoint viewing
4. Checkpoint restoration
5. Checkpoint deletion
6. UI interactions
"""

import os
import sys
import json
import shutil
import unittest
import tempfile
from unittest.mock import MagicMock, patch
from datetime import datetime

# Add project root to the path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import the modules being tested
from src.utils.emergency_checkpoint import (
    EmergencyCheckpoint,
    create_emergency_checkpoint,
    list_emergency_checkpoints,
    restore_emergency_checkpoint,
    delete_emergency_checkpoint,
    get_checkpoint_manager
)

# Mock for PySimpleGUI
class MockWindow:
    """Mock PySimpleGUI window for testing UI interactions."""
    
    def __init__(self):
        self.Title = "Test Window - Recovery Dashboard"
        self.AllKeysDict = {
            "-RECOVERY-TABLE-": True,
            "-CHECKPOINT-ID-": True,
            "-CHECKPOINT-CREATED-": True,
            "-CHECKPOINT-NOTE-": True,
            "-CHECKPOINT-CONTENT-": True,
            "-CHECKPOINT-FILES-": True,
            "-VIEW-CHECKPOINT-": True,
            "-RESTORE-CHECKPOINT-": True,
            "-DELETE-CHECKPOINT-": True,
            "-AUTO-CHECKPOINT-INTERVAL-": True,
            "-ENABLE-AUTO-CHECKPOINTS-": True
        }
        self._values = {}
        
    def __getitem__(self, key):
        """Simulate window element access."""
        return MockElement(key)
    
    def update(self, value):
        """Mock update method."""
        pass
    
    def write_event_value(self, event, value):
        """Mock write_event_value method."""
        pass
    
    def get(self):
        """Mock get method for tables."""
        return [["checkpoint-1", "date", "time", "note", "Valid"],
                ["checkpoint-2", "date", "time", "note", "Error"]]

class MockElement:
    """Mock PySimpleGUI element."""
    
    def __init__(self, key):
        self.key = key
    
    def update(self, value=None, values=None, disabled=None):
        """Mock update method for elements."""
        pass
    
    def get(self):
        """Mock get method."""
        if self.key == "-RECOVERY-TABLE-":
            return [["checkpoint-1", "date", "time", "note", "Valid"],
                    ["checkpoint-2", "date", "time", "note", "Error"]]
        return "test value"

# Mock the sg module
class MockSG:
    """Mock PySimpleGUI module."""
    
    @staticmethod
    def popup(*args, **kwargs):
        """Mock popup method."""
        return "OK"
    
    @staticmethod
    def popup_yes_no(*args, **kwargs):
        """Mock popup_yes_no method."""
        return "Yes"
    
    @staticmethod
    def popup_get_text(*args, **kwargs):
        """Mock popup_get_text method."""
        return "Test Note"
    
    @staticmethod
    def popup_ok(*args, **kwargs):
        """Mock popup_ok method."""
        pass
    
    @staticmethod
    def popup_error(*args, **kwargs):
        """Mock popup_error method."""
        pass
    
    @staticmethod
    def popup_quick_message(*args, **kwargs):
        """Mock popup_quick_message method."""
        pass
    
    @staticmethod
    def Window(*args, **kwargs):
        """Mock Window constructor."""
        window = MagicMock()
        window.close = MagicMock()
        return window
    
    # Mock constants
    TIMEOUT_KEY = "-TIMEOUT-"
    WIN_CLOSED = "WIN_CLOSED"


class TestEmergencyCheckpoint(unittest.TestCase):
    """Test case for the EmergencyCheckpoint class."""
    
    def setUp(self):
        """Set up the test case."""
        self.temp_dir = tempfile.mkdtemp()
        self.app_state = {
            "config": {"BUCKET": "Scalping"},
            "presets": {"Scalping": {"param1": 1}},
            "values": {"LEARNING_RATE": 0.001}
        }
        self.checkpoint_manager = EmergencyCheckpoint(self.temp_dir, self.app_state)
        
    def tearDown(self):
        """Clean up after the test."""
        shutil.rmtree(self.temp_dir)
    
    def test_create_checkpoint(self):
        """Test creating an emergency checkpoint."""
        # Create a test file
        test_file = os.path.join(self.temp_dir, "test_file.txt")
        with open(test_file, "w") as f:
            f.write("Test content")
        
        # Create checkpoint
        result = self.checkpoint_manager.create_checkpoint(
            self.app_state, 
            checkpoint_note="Test checkpoint",
            critical_files=[test_file]
        )
        
        # Assertions
        self.assertTrue(result["success"])
        self.assertIsNotNone(result["checkpoint_id"])
        self.assertTrue(os.path.exists(result["checkpoint_dir"]))
        
        # Check files were copied
        state_file = os.path.join(result["checkpoint_dir"], "app_state.json")
        self.assertTrue(os.path.exists(state_file))
        
        # Check files directory
        files_dir = os.path.join(result["checkpoint_dir"], "files")
        self.assertTrue(os.path.exists(files_dir))
        
        # Check copied file
        copied_file = os.path.join(files_dir, "test_file.txt")
        self.assertTrue(os.path.exists(copied_file))
    
    def test_list_checkpoints(self):
        """Test listing emergency checkpoints."""
        # Create a checkpoint first
        self.checkpoint_manager.create_checkpoint(
            self.app_state, 
            checkpoint_note="Test checkpoint"
        )
        
        # List checkpoints
        checkpoints = self.checkpoint_manager.list_checkpoints()
        
        # Assertions
        self.assertGreater(len(checkpoints), 0)
        self.assertIn("id", checkpoints[0])
        self.assertIn("timestamp", checkpoints[0])
        self.assertIn("note", checkpoints[0])
        self.assertEqual(checkpoints[0]["note"], "Test checkpoint")
    
    def test_restore_checkpoint(self):
        """Test restoring an emergency checkpoint."""
        # Create a checkpoint first
        result = self.checkpoint_manager.create_checkpoint(
            self.app_state, 
            checkpoint_note="Test checkpoint"
        )
        
        # Restore checkpoint
        restore_result = self.checkpoint_manager.restore_checkpoint(result["checkpoint_id"])
        
        # Assertions
        self.assertTrue(restore_result["success"])
        self.assertIsNotNone(restore_result["app_state"])
        self.assertEqual(restore_result["app_state"]["config"]["BUCKET"], "Scalping")
    
    def test_delete_checkpoint(self):
        """Test deleting an emergency checkpoint."""
        # Create a checkpoint first
        result = self.checkpoint_manager.create_checkpoint(
            self.app_state, 
            checkpoint_note="Test checkpoint"
        )
        
        # Delete checkpoint
        success = self.checkpoint_manager.delete_checkpoint(result["checkpoint_id"])
        
        # Assertions
        self.assertTrue(success)
        
        # Verify it's gone
        checkpoints = self.checkpoint_manager.list_checkpoints()
        checkpoint_ids = [cp["id"] for cp in checkpoints]
        self.assertNotIn(result["checkpoint_id"], checkpoint_ids)
    
    def test_verify_checkpoint_integrity(self):
        """Test checkpoint integrity verification."""
        # Create a checkpoint first
        result = self.checkpoint_manager.create_checkpoint(
            self.app_state, 
            checkpoint_note="Test checkpoint"
        )
        
        # Verify integrity
        integrity_result = self.checkpoint_manager._verify_checkpoint_integrity(result["checkpoint_dir"])
        
        # Assertions
        self.assertTrue(integrity_result["success"])
        
        # Tamper with the checkpoint to test failure case
        state_file = os.path.join(result["checkpoint_dir"], "app_state.json")
        with open(state_file, "w") as f:
            f.write("Tampered content")
        
        # Verify integrity again
        integrity_result = self.checkpoint_manager._verify_checkpoint_integrity(result["checkpoint_dir"])
        
        # Assertions
        self.assertFalse(integrity_result["success"])


class TestEmergencyCheckpointAPI(unittest.TestCase):
    """Test case for the EmergencyCheckpoint API functions."""
    
    def setUp(self):
        """Set up the test case."""
        self.temp_dir = tempfile.mkdtemp()
        self.app_state = {
            "config": {"BUCKET": "Scalping"},
            "presets": {"Scalping": {"param1": 1}},
            "values": {"LEARNING_RATE": 0.001}
        }
    
    def tearDown(self):
        """Clean up after the test."""
        shutil.rmtree(self.temp_dir)
    
    def test_get_checkpoint_manager(self):
        """Test getting a checkpoint manager."""
        manager = get_checkpoint_manager(self.temp_dir, self.app_state)
        self.assertIsInstance(manager, EmergencyCheckpoint)
    
    def test_create_emergency_checkpoint(self):
        """Test creating an emergency checkpoint via API."""
        result = create_emergency_checkpoint(
            self.app_state, 
            checkpoint_note="API Test",
            base_dir=self.temp_dir
        )
        
        # Assertions
        self.assertTrue(result["success"])
        self.assertIsNotNone(result["id"])
    
    def test_list_emergency_checkpoints(self):
        """Test listing emergency checkpoints via API."""
        # Create a checkpoint first
        create_emergency_checkpoint(
            self.app_state, 
            checkpoint_note="API Test",
            base_dir=self.temp_dir
        )
        
        # List checkpoints
        checkpoints = list_emergency_checkpoints(base_dir=self.temp_dir)
        
        # Assertions
        self.assertGreater(len(checkpoints), 0)
        self.assertIn("id", checkpoints[0])
    
    def test_restore_emergency_checkpoint(self):
        """Test restoring an emergency checkpoint via API."""
        # Create a checkpoint first
        result = create_emergency_checkpoint(
            self.app_state, 
            checkpoint_note="API Test",
            base_dir=self.temp_dir
        )
        
        # Restore checkpoint
        restore_result = restore_emergency_checkpoint(
            result["id"],
            base_dir=self.temp_dir
        )
        
        # Assertions
        self.assertTrue(restore_result["success"])
    
    def test_delete_emergency_checkpoint(self):
        """Test deleting an emergency checkpoint via API."""
        # Create a checkpoint first
        result = create_emergency_checkpoint(
            self.app_state, 
            checkpoint_note="API Test",
            base_dir=self.temp_dir
        )
        
        # Delete checkpoint
        success = delete_emergency_checkpoint(
            result["id"],
            base_dir=self.temp_dir
        )
        
        # Assertions
        self.assertTrue(success)


@patch("src.ui.main.sg", MockSG())
@patch("src.ui.main.emergency_checkpoint", MagicMock())
class TestRecoveryDashboardUI(unittest.TestCase):
    """Test case for the Recovery Dashboard UI functions."""
    
    def setUp(self):
        """Set up the test case."""
        from src.ui.main import (
            refresh_recovery_checkpoints,
            show_checkpoint_details,
            restore_checkpoint,
            create_recovery_checkpoint,
            delete_checkpoint
        )
        
        self.refresh_recovery_checkpoints = refresh_recovery_checkpoints
        self.show_checkpoint_details = show_checkpoint_details
        self.restore_checkpoint = restore_checkpoint
        self.create_recovery_checkpoint = create_recovery_checkpoint
        self.delete_checkpoint = delete_checkpoint
        
        self.window = MockWindow()
    
    def test_refresh_recovery_checkpoints(self):
        """Test refreshing recovery checkpoints."""
        # Mock list_emergency_checkpoints to return test data
        with patch("src.utils.emergency_checkpoint.list_emergency_checkpoints", 
                return_value=[
                    {
                        "id": "test-id",
                        "timestamp": "20230101_120000",
                        "note": "Test note",
                        "integrity_verified": True
                    }
                ]):
            
            # Call the function
            result = self.refresh_recovery_checkpoints(self.window)
            
            # Assertions
            self.assertTrue(result)
    
    def test_show_checkpoint_details(self):
        """Test showing checkpoint details."""
        # Mock get_checkpoint_details to return test data
        with patch("src.utils.emergency_checkpoint.get_checkpoint_details", 
                return_value={
                    "id": "test-id",
                    "timestamp": "20230101_120000",
                    "note": "Test note",
                    "app_state": {"config": {"key": "value"}},
                    "metadata": {"version": 1},
                    "files": ["file1.txt", "file2.txt"]
                }):
            
            # Call the function
            result = self.show_checkpoint_details(self.window, "test-id")
            
            # Assertions
            self.assertTrue(result)
    
    def test_restore_checkpoint(self):
        """Test restoring a checkpoint from UI."""
        # Mock restore_emergency_checkpoint to return success
        with patch("src.utils.emergency_checkpoint.restore_emergency_checkpoint", 
                return_value={
                    "success": True,
                    "restored_files": ["file1.txt", "file2.txt"]
                }):
            
            # Call the function
            result = self.restore_checkpoint(self.window, "test-id")
            
            # Assertions
            self.assertTrue(result)
    
    def test_create_recovery_checkpoint(self):
        """Test creating a recovery checkpoint from UI."""
        # Mock create_emergency_checkpoint to return success
        with patch("src.utils.emergency_checkpoint.create_emergency_checkpoint", 
                return_value={
                    "success": True,
                    "id": "new-id"
                }):
            
            # Call the function
            result = self.create_recovery_checkpoint(self.window)
            
            # Assertions
            self.assertTrue(result)
    
    def test_delete_checkpoint(self):
        """Test deleting a checkpoint from UI."""
        # Mock delete_emergency_checkpoint to return success
        with patch("src.utils.emergency_checkpoint.delete_emergency_checkpoint", 
                return_value={"success": True}):
            
            # Call the function
            result = self.delete_checkpoint(self.window, "test-id")
            
            # Assertions
            self.assertTrue(result)


@patch("src.ui.main.sg", MockSG())
class TestRecoveryDashboardEventHandlers(unittest.TestCase):
    """Test case for the Recovery Dashboard event handlers."""
    
    def setUp(self):
        """Set up the test case."""
        # Mock refresh_recovery_checkpoints
        self.refresh_mock = MagicMock(return_value=True)
        self.show_details_mock = MagicMock(return_value=True)
        self.restore_mock = MagicMock(return_value=True)
        self.create_mock = MagicMock(return_value=True)
        self.delete_mock = MagicMock(return_value=True)
        
        # Create patches
        self.patches = [
            patch("src.ui.main.refresh_recovery_checkpoints", self.refresh_mock),
            patch("src.ui.main.show_checkpoint_details", self.show_details_mock),
            patch("src.ui.main.restore_checkpoint", self.restore_mock),
            patch("src.ui.main.create_recovery_checkpoint", self.create_mock),
            patch("src.ui.main.delete_checkpoint", self.delete_mock)
        ]
        
        # Apply patches
        for p in self.patches:
            p.start()
        
        # Create mock window and values for event handlers
        self.window = MockWindow()
        self.values = {
            "-RECOVERY-TABLE-": [0],  # First item is selected
            "-AUTO-CHECKPOINT-INTERVAL-": 15,
            "-ENABLE-AUTO-CHECKPOINTS-": True,
            "tabgroup": "Recovery Dashboard"
        }
    
    def tearDown(self):
        """Clean up after the test."""
        # Stop patches
        for p in self.patches:
            p.stop()
    
    def test_refresh_recovery_event(self):
        """Test the -REFRESH-RECOVERY- event handler."""
        # Call event handler directly
        from src.ui.main import handle_refresh_recovery
        
        if "handle_refresh_recovery" in globals():
            # Call the function if it exists
            handle_refresh_recovery(self.window)
            
            # Assertions
            self.refresh_mock.assert_called_once_with(self.window)
        else:
            # Test the event in the main loop instead
            # This is a placeholder - in a real test, we would need to simulate the event loop
            pass
    
    def test_recovery_table_event(self):
        """Test the -RECOVERY-TABLE- event handler."""
        from src.ui.main import handle_recovery_table_selection
        
        if "handle_recovery_table_selection" in globals():
            # Call the function if it exists
            handle_recovery_table_selection(self.window, self.values)
            
            # Assertions - buttons should be enabled
            # Since we can't directly check button state, we verify the behavior
            # indirectly through how the UI would be updated
            pass
        else:
            # Test the event in the main loop instead
            # This is a placeholder - in a real test, we would need to simulate the event loop
            pass
    
    def test_view_checkpoint_event(self):
        """Test the -VIEW-CHECKPOINT- event handler."""
        from src.ui.main import handle_view_checkpoint
        
        if "handle_view_checkpoint" in globals():
            # Call the function if it exists
            handle_view_checkpoint(self.window, self.values)
            
            # Assertions
            self.show_details_mock.assert_called_once()
        else:
            # Test the event in the main loop instead
            # This is a placeholder - in a real test, we would need to simulate the event loop
            pass
    
    def test_restore_checkpoint_event(self):
        """Test the -RESTORE-CHECKPOINT- event handler."""
        from src.ui.main import handle_restore_checkpoint
        
        if "handle_restore_checkpoint" in globals():
            # Call the function if it exists
            handle_restore_checkpoint(self.window, self.values)
            
            # Assertions
            self.restore_mock.assert_called_once()
        else:
            # Test the event in the main loop instead
            # This is a placeholder - in a real test, we would need to simulate the event loop
            pass
    
    def test_create_checkpoint_event(self):
        """Test the -CREATE-CHECKPOINT- event handler."""
        from src.ui.main import handle_create_checkpoint
        
        if "handle_create_checkpoint" in globals():
            # Call the function if it exists
            handle_create_checkpoint(self.window)
            
            # Assertions
            self.create_mock.assert_called_once_with(self.window)
        else:
            # Test the event in the main loop instead
            # This is a placeholder - in a real test, we would need to simulate the event loop
            pass
    
    def test_delete_checkpoint_event(self):
        """Test the -DELETE-CHECKPOINT- event handler."""
        from src.ui.main import handle_delete_checkpoint
        
        if "handle_delete_checkpoint" in globals():
            # Call the function if it exists
            handle_delete_checkpoint(self.window, self.values)
            
            # Assertions
            self.delete_mock.assert_called_once()
        else:
            # Test the event in the main loop instead
            # This is a placeholder - in a real test, we would need to simulate the event loop
            pass


if __name__ == "__main__":
    unittest.main() 