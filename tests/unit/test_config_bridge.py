import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add the project root to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.config_bridge import (
    load_preset_into_config,
    save_config_as_preset,
    get_preset_default_config
)


class TestConfigBridge(unittest.TestCase):
    """Test cases for the config_bridge module."""

    @patch('src.utils.config_bridge.get_trade_config')
    @patch('src.utils.config_bridge.load_preset')
    def test_load_preset_into_config(self, mock_load_preset, mock_get_trade_config):
        """Test loading a preset into the main configuration."""
        # Setup
        mock_config = MagicMock()
        mock_get_trade_config.return_value = mock_config
        mock_load_preset.return_value = {"params": {"test_param": "test_value"}}
        
        # Execute
        result = load_preset_into_config("presets/defaults/Scalping/Test.json")
        
        # Assert
        self.assertTrue(result)
        mock_load_preset.assert_called_once_with("presets/defaults/Scalping/Test.json")
        mock_config.update.assert_called_once_with({"test_param": "test_value"})

    @patch('src.utils.config_bridge.get_trade_config')
    @patch('src.utils.config_bridge.load_preset')
    def test_load_preset_into_config_failure(self, mock_load_preset, mock_get_trade_config):
        """Test loading a preset that doesn't exist or lacks params."""
        # Setup
        mock_config = MagicMock()
        mock_get_trade_config.return_value = mock_config
        mock_load_preset.return_value = {}  # Empty result
        
        # Execute
        result = load_preset_into_config("nonexistent_preset.json")
        
        # Assert
        self.assertFalse(result)
        mock_config.update.assert_not_called()

    @patch('src.utils.config_bridge.get_trade_config')
    def test_get_trade_config_failure(self, mock_get_trade_config):
        """Test behavior when get_trade_config raises an exception."""
        # Setup
        mock_get_trade_config.side_effect = Exception("Config not available")
        
        # Execute
        with patch('src.utils.config_bridge.logger') as mock_logger:
            result = load_preset_into_config("any_preset.json")
        
        # Assert
        self.assertFalse(result)
        mock_logger.error.assert_called()

    @patch('src.utils.config_bridge.DEFAULT_PRESETS')
    def test_get_preset_default_config(self, mock_default_presets):
        """Test getting default configuration for a specific bucket."""
        # Setup
        mock_default_presets.get.return_value = {
            "Conservative": {
                "params": {
                    "BUCKET": "Scalping",
                    "risk_tolerance": "low"
                }
            }
        }
        
        # Execute
        result = get_preset_default_config("Scalping")
        
        # Assert
        self.assertEqual(result, {"BUCKET": "Scalping", "risk_tolerance": "low"})

    @patch('src.utils.config_bridge.DEFAULT_PRESETS')
    def test_get_preset_default_config_empty(self, mock_default_presets):
        """Test getting default config when no presets exist for a bucket."""
        # Setup
        mock_default_presets.get.return_value = {}
        
        # Execute
        result = get_preset_default_config("Unknown")
        
        # Assert
        self.assertEqual(result, {})

    @patch('src.utils.config_bridge.save_preset')
    @patch('src.utils.config_bridge.get_trade_config')
    def test_save_config_as_preset(self, mock_get_trade_config, mock_save_preset):
        """Test saving current configuration as a preset."""
        # Setup
        mock_config = MagicMock()
        mock_config.get_section.return_value = {"RISK_LEVEL": "medium"}
        mock_get_trade_config.return_value = mock_config
        mock_save_preset.return_value = "path/to/preset.json"
        
        # Execute
        result = save_config_as_preset("Scalping", "Test Preset", "Description")
        
        # Assert
        self.assertEqual(result, "path/to/preset.json")
        mock_config.get_section.assert_called_with("scalping")
        mock_save_preset.assert_called_once()


if __name__ == '__main__':
    unittest.main() 