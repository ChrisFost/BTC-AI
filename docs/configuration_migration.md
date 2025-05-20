# Configuration System Migration Guide

## Overview

This guide explains how to migrate your existing configuration to the new TradeConfig system. The new system provides better organization, validation, and backward compatibility support.

## Migration Process

### Automatic Migration

The system will automatically detect and migrate legacy configurations when you first use the new TradeConfig system. Here's what happens:

1. When loading a configuration file, the system checks its version
2. If a legacy version is detected (1.0, 1.1, or 1.2), it will:
   - Create a backup of your original configuration file
   - Migrate the configuration to the new format
   - Save the migrated configuration
   - Log the migration process

### Manual Migration

If you prefer to migrate manually, follow these steps:

1. **Backup Your Configuration**
   ```bash
   cp configs/config.json configs/config.json.backup
   ```

2. **Update Configuration Structure**
   Convert your legacy configuration to the new format:

   ```json
   // Legacy Format (version 1.x)
   {
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

   // New Format (version 2.0)
   {
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
   ```

3. **Parameter Mapping**
   Here's how legacy parameters map to the new structure:

   | Legacy Parameter | New Location | Type |
   |-----------------|--------------|------|
   | BUCKET | trading.bucket | string |
   | INITIAL_CAPITAL | trading.initial_capital | number |
   | MAX_POSITIONS | trading.max_positions | integer |
   | HIDDEN_SIZE | model.hidden_size | integer |
   | LEARNING_RATE | model.learning_rate | number |
   | BATCH_SIZE | model.batch_size | integer |
   | MAX_BTC_PER_POSITION | risk.max_btc_per_position | number |
   | MAX_USD_PER_POSITION | risk.max_usd_per_position | number |
   | MAX_VOLUME_PERCENTAGE | risk.max_volume_percentage | number |

4. **Validation**
   The new system includes validation for all parameters. Ensure your values meet these requirements:

   - `trading.initial_capital`: Must be a positive number
   - `trading.max_positions`: Must be a positive integer
   - `model.learning_rate`: Must be between 0 and 1
   - `model.batch_size`: Must be a positive integer
   - `risk.max_volume_percentage`: Must be between 0 and 1

### Using the New Configuration System

1. **Loading Configuration**
   ```python
   from src.utils.trade_config import get_trade_config
   
   # Get configuration instance
   config = get_trade_config()
   
   # Access configuration values
   bucket = config.get("trading.bucket")
   learning_rate = config.get("model.learning_rate")
   ```

2. **Updating Configuration**
   ```python
   # Update single value
   config.set("trading.initial_capital", 200000.0)
   
   # Update multiple values
   config.update({
       "trading.bucket": "Short",
       "model.learning_rate": 0.0005
   })
   ```

3. **Saving Configuration**
   ```python
   # Save changes
   config.save()
   ```

### Troubleshooting

1. **Migration Errors**
   - Check the logs for detailed error messages
   - Verify your configuration file is valid JSON
   - Ensure all required parameters are present
   - Check parameter types and ranges

2. **Common Issues**
   - Invalid parameter types: Convert values to correct types
   - Missing required parameters: Add missing parameters
   - Out-of-range values: Adjust values to valid ranges
   - File permission issues: Check file permissions

3. **Recovery**
   - Use the backup file created during migration
   - Restore from backup if needed:
     ```bash
     cp configs/config.json.backup configs/config.json
     ```

## Additional Resources

- [Configuration Schema Documentation](configuration_schema.md)
- [Configuration Examples](configuration_examples.md)
- [Troubleshooting Guide](configuration_troubleshooting.md)
- [API Documentation](configuration_api.md) 