"""Unit tests for TradeConfig class."""

import pytest
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.utils.trade_config import TradeConfig, get_trade_config

@pytest.fixture
def temp_config_file():
    """Create a temporary configuration file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        test_config = {
            "version": "2.0",
            "environment": "test",
            "trading": {
                "bucket": "Scalping",
                "initial_capital": 10000.0,
                "max_positions": 10
            },
            "model": {
                "hidden_size": 256,
                "learning_rate": 0.001,
                "batch_size": 64
            },
            "risk": {
                "max_btc_per_position": 1.0,
                "max_usd_per_position": 10000.0,
                "max_volume_percentage": 0.01
            }
        }
        f.write(json.dumps(test_config).encode('utf-8'))
        config_path = f.name
    
    yield config_path
    
    # Clean up
    if os.path.exists(config_path):
        os.unlink(config_path)

@pytest.fixture
def reset_trade_config():
    """Reset the TradeConfig singleton between tests."""
    TradeConfig._instance = None
    import src.utils.trade_config
    src.utils.trade_config._trade_config_instance = None
    yield

def test_singleton_pattern(reset_trade_config):
    """Test that TradeConfig follows the singleton pattern."""
    config1 = TradeConfig()
    config2 = TradeConfig()
    assert config1 is config2
    
    # Test get_trade_config function
    config3 = get_trade_config()
    assert config1 is config3

def test_load_config(temp_config_file, reset_trade_config):
    """Test loading configuration from a file."""
    config = TradeConfig(temp_config_file)
    
    # Check that config was loaded correctly
    assert config.config["version"] == "2.0"
    assert config.config["trading"]["bucket"] == "Scalping"
    assert config.config["model"]["hidden_size"] == 256
    assert config.config["risk"]["max_btc_per_position"] == 1.0

def test_get_method(temp_config_file, reset_trade_config):
    """Test the get method for accessing configuration values."""
    config = TradeConfig(temp_config_file)
    
    # Test nested key access with dot notation
    assert config.get("trading.bucket") == "Scalping"
    assert config.get("model.hidden_size") == 256
    assert config.get("risk.max_btc_per_position") == 1.0
    
    # Test direct key access
    assert config.get("version") == "2.0"
    assert config.get("environment") == "test"
    
    # Test default value for non-existent key
    assert config.get("non_existent_key", "default") == "default"
    assert config.get("trading.non_existent", 100) == 100

def test_set_method(temp_config_file, reset_trade_config):
    """Test the set method for updating configuration values."""
    config = TradeConfig(temp_config_file)
    
    # Test setting values with dot notation
    config.set("trading.bucket", "Medium")
    assert config.get("trading.bucket") == "Medium"
    
    # Test setting direct keys
    config.set("new_key", "new_value")
    assert config.get("new_key") == "new_value"
    
    # Test creating nested structure
    config.set("new_section.nested_key", "nested_value")
    assert config.get("new_section.nested_key") == "nested_value"

def test_dictionary_access(temp_config_file, reset_trade_config):
    """Test dictionary-style access to configuration values."""
    config = TradeConfig(temp_config_file)
    
    # Test __getitem__
    assert config["trading"]["bucket"] == "Scalping"
    assert config["model"]["hidden_size"] == 256
    
    # Test __setitem__
    config["trading"]["bucket"] = "Long"
    assert config["trading"]["bucket"] == "Long"
    
    # Test __contains__
    assert "trading" in config
    assert "non_existent" not in config

def test_update_method(temp_config_file, reset_trade_config):
    """Test updating configuration with a dictionary."""
    config = TradeConfig(temp_config_file)
    
    # Test updating with a dictionary
    update_dict = {
        "trading": {
            "bucket": "Long",
            "initial_capital": 20000.0
        },
        "new_section": {
            "new_key": "new_value"
        }
    }
    
    config.update(update_dict)
    
    # Check that values were updated correctly
    assert config.get("trading.bucket") == "Long"
    assert config.get("trading.initial_capital") == 20000.0
    assert config.get("trading.max_positions") == 10  # This should not be changed
    assert config.get("new_section.new_key") == "new_value"

def test_save_and_reload(temp_config_file, reset_trade_config):
    """Test saving configuration to a file and reloading it."""
    config = TradeConfig(temp_config_file)
    
    # Modify configuration
    config.set("trading.bucket", "Medium")
    config.set("new_key", "new_value")
    
    # Save configuration
    assert config.save() is True
    
    # Create a new instance with the same file
    config2 = TradeConfig(temp_config_file)
    
    # Check that changes were saved and loaded correctly
    assert config2.get("trading.bucket") == "Medium"
    assert config2.get("new_key") == "new_value"

def test_validate_method(temp_config_file, reset_trade_config):
    """Test configuration validation."""
    config = TradeConfig(temp_config_file)
    
    # Valid configuration should pass validation
    assert config.validate() is True
    
    # Invalid configuration should fail validation
    config.config["trading"]["bucket"] = 123  # Should be a string
    assert config.validate() is False
    
    # Fix the issue and validate again
    config.config["trading"]["bucket"] = "Scalping"
    assert config.validate() is True
    
    # Test missing required section
    del config.config["risk"]
    assert config.validate() is False

def test_get_path(reset_trade_config):
    """Test getting paths from configuration."""
    with patch('os.path.dirname') as mock_dirname:
        mock_dirname.side_effect = ["src/utils", "src", "/project"]
        config = TradeConfig()
        
        # Mock path existence and make sure makedirs is called
        with patch('os.path.exists', return_value=True):
            # Test standard directory
            path = config.get_path("MODELS_DIR")
            assert os.path.join("/project", "Models") == path
            
            # Test with create=True
            mock_makedirs = MagicMock()
            with patch('os.makedirs', mock_makedirs):
                config.get_path("DATA_DIR", create=True)
                # Verify makedirs was called at least once
                assert mock_makedirs.call_count >= 1

def test_ui_values_update(temp_config_file, reset_trade_config):
    """Test updating configuration from UI values."""
    config = TradeConfig(temp_config_file)
    
    # Create UI values dictionary
    ui_values = {
        "BUCKET": "Medium",
        "INITIAL_CAPITAL": 20000.0,
        "HIDDEN_SIZE": 512,
        "LEARNING_RATE": 0.0005,
        "withdrawal_simulation_monthly_withdrawal_chance": 0.5,
        "medium_min_gain_per_holding": 40.0
    }
    
    # Setup withdrawal simulation section
    config.config["withdrawal_simulation"] = {
        "monthly_withdrawal_chance": 0.3,
        "emergency_withdrawal_chance": 0.05
    }
    
    # Update from UI values
    config.from_ui_values(ui_values)
    
    # Check that values were updated correctly
    assert config.get("BUCKET") == "Medium"
    assert config.get("INITIAL_CAPITAL") == 20000.0
    assert config.get("HIDDEN_SIZE") == 512
    assert config.get("LEARNING_RATE") == 0.0005
    assert config.get("withdrawal_simulation")["monthly_withdrawal_chance"] == 0.5
    assert config.get("min_gain_per_holding") == 40.0  # Bucket-specific setting

def test_get_section(temp_config_file, reset_trade_config):
    """Test getting a configuration section."""
    config = TradeConfig(temp_config_file)
    
    # Test getting a section
    trading_section = config.get_section("trading")
    assert trading_section["bucket"] == "Scalping"
    assert trading_section["initial_capital"] == 10000.0
    assert trading_section["max_positions"] == 10
    
    # Test getting a non-existent section
    empty_section = config.get_section("non_existent")
    assert empty_section == {}
    
    # Test prefix-based section
    config.config["TEST_param1"] = "value1"
    config.config["TEST_param2"] = "value2"
    test_section = config.get_section("TEST")
    assert test_section["param1"] == "value1"
    assert test_section["param2"] == "value2"

def test_preset_config(reset_trade_config):
    """Test getting preset configurations for different buckets."""
    config = TradeConfig()
    
    # Test getting preset for Scalping
    scalping_preset = config.get_preset_config("Scalping")
    assert scalping_preset["BUCKET"] == "Scalping"
    assert scalping_preset["WINDOW_SIZE"] == 288
    
    # Test getting preset for Long
    long_preset = config.get_preset_config("Long")
    assert long_preset["BUCKET"] == "Long"
    assert long_preset["MAX_POSITION_HOLDINGS"] == 20
    
    # Test getting preset for non-existent bucket (should return Scalping)
    default_preset = config.get_preset_config("NonExistent")
    assert default_preset["BUCKET"] == "Scalping"

def test_model_paths(reset_trade_config):
    """Test generating model paths."""
    with patch('os.path.dirname') as mock_dirname:
        mock_dirname.side_effect = ["src/utils", "src", "/project"]
        config = TradeConfig()
        
        # Set bucket 
        config.config["trading"] = {"bucket": "Scalping"}
        
        # Mock path existence and makedirs
        with patch('os.path.exists', return_value=True), patch('os.makedirs'):
            # Test checkpoint path
            checkpoint_path = config.get_checkpoint_path(10)
            # Convert to normalized path with forward slashes for consistent testing
            normalized_path = str(Path(checkpoint_path).as_posix())
            assert "Scalping/checkpoint_10.pth" in normalized_path
            
            # Test final model path
            with patch('datetime.datetime') as mock_datetime:
                mock_datetime.now.return_value.strftime.return_value = "20250101_120000"
                final_path = config.get_final_model_path()
                # Convert to normalized path with forward slashes for consistent testing
                normalized_final_path = str(Path(final_path).as_posix())
                assert "Scalping/final_scalping_20250101_120000.pth" in normalized_final_path 