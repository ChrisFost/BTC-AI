"""
Tests for configuration compatibility layer.
"""

import os
import json
import pytest
from datetime import datetime
from src.utils.config_compatibility import ConfigCompatibility

@pytest.fixture
def legacy_config():
    """Create a legacy configuration for testing."""
    return {
        "version": "1.0",
        "BUCKET": "Scalping",
        "INITIAL_CAPITAL": 100000.0,
        "MAX_POSITIONS": 50,
        "HIDDEN_SIZE": 512,
        "LEARNING_RATE": 0.0003,
        "BATCH_SIZE": 128,
        "MAX_BTC_PER_POSITION": 10.0,
        "MAX_USD_PER_POSITION": 1000000.0,
        "MAX_VOLUME_PERCENTAGE": 0.05
    }

@pytest.fixture
def current_config():
    """Create a current configuration for testing."""
    return {
        "version": "2.0",
        "environment": "development",
        "trading": {
            "bucket": "Scalping",
            "initial_capital": 100000.0,
            "max_positions": 50
        },
        "model": {
            "hidden_size": 512,
            "learning_rate": 0.0003,
            "batch_size": 128
        },
        "risk": {
            "max_btc_per_position": 10.0,
            "max_usd_per_position": 1000000.0,
            "max_volume_percentage": 0.05
        }
    }

@pytest.fixture
def config_file(tmp_path, legacy_config):
    """Create a temporary configuration file."""
    config_path = tmp_path / "config.json"
    with open(config_path, 'w') as f:
        json.dump(legacy_config, f)
    return config_path

def test_detect_config_version():
    """Test configuration version detection."""
    # Test legacy version
    legacy_config = {"version": "1.0"}
    version, format_type = ConfigCompatibility.detect_config_version(legacy_config)
    assert version == "1.0"
    assert format_type == "legacy"
    
    # Test current version
    current_config = {"version": "2.0"}
    version, format_type = ConfigCompatibility.detect_config_version(current_config)
    assert version == "2.0"
    assert format_type == "current"
    
    # Test missing version
    no_version_config = {}
    version, format_type = ConfigCompatibility.detect_config_version(no_version_config)
    assert version == "1.0"  # Default version
    assert format_type == "legacy"

def test_migrate_legacy_config(legacy_config):
    """Test migration of legacy configuration."""
    new_config = ConfigCompatibility.migrate_legacy_config(legacy_config)
    
    # Check version
    assert new_config["version"] == "2.0"
    
    # Check trading section
    assert new_config["trading"]["bucket"] == legacy_config["BUCKET"]
    assert new_config["trading"]["initial_capital"] == legacy_config["INITIAL_CAPITAL"]
    assert new_config["trading"]["max_positions"] == legacy_config["MAX_POSITIONS"]
    
    # Check model section
    assert new_config["model"]["hidden_size"] == legacy_config["HIDDEN_SIZE"]
    assert new_config["model"]["learning_rate"] == legacy_config["LEARNING_RATE"]
    assert new_config["model"]["batch_size"] == legacy_config["BATCH_SIZE"]
    
    # Check risk section
    assert new_config["risk"]["max_btc_per_position"] == legacy_config["MAX_BTC_PER_POSITION"]
    assert new_config["risk"]["max_usd_per_position"] == legacy_config["MAX_USD_PER_POSITION"]
    assert new_config["risk"]["max_volume_percentage"] == legacy_config["MAX_VOLUME_PERCENTAGE"]
    
    # Check migration metadata
    assert "_migration_metadata" in new_config
    assert new_config["_migration_metadata"]["original_version"] == "1.0"
    assert "migration_date" in new_config["_migration_metadata"]
    assert new_config["_migration_metadata"]["migration_successful"] is True

def test_load_config_with_compatibility(config_file):
    """Test loading configuration with compatibility support."""
    # Load config
    config = ConfigCompatibility.load_config_with_compatibility(str(config_file))
    
    # Check if config was migrated
    assert config["version"] == "2.0"
    assert "trading" in config
    assert "model" in config
    assert "risk" in config
    
    # Check if backup was created
    backup_files = list(config_file.parent.glob(f"{config_file.name}.*.bak"))
    assert len(backup_files) == 1

def test_save_config_with_compatibility(tmp_path, current_config):
    """Test saving configuration with version tracking."""
    config_path = tmp_path / "config.json"
    
    # Save config
    success = ConfigCompatibility.save_config_with_compatibility(current_config, str(config_path))
    assert success
    
    # Check if file was created
    assert config_path.exists()
    
    # Load and verify saved config
    with open(config_path) as f:
        saved_config = json.load(f)
    assert saved_config == current_config

def test_load_nonexistent_config():
    """Test loading nonexistent configuration."""
    config = ConfigCompatibility.load_config_with_compatibility("nonexistent.json")
    assert config == ConfigCompatibility.DEFAULT_CONFIG.copy()

def test_save_config_error(tmp_path):
    """Test saving configuration with error."""
    # Create a directory that can't be written to
    read_only_dir = tmp_path / "readonly"
    read_only_dir.mkdir()
    os.chmod(read_only_dir, 0o444)  # Read-only
    
    config_path = read_only_dir / "config.json"
    success = ConfigCompatibility.save_config_with_compatibility({}, str(config_path))
    assert not success 