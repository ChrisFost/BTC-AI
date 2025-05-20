# Configuration Schema Documentation

## Overview

This document describes the complete configuration schema for the BTC-AI trading system. The configuration is organized into logical sections for better maintainability and clarity.

## Schema Version

Current version: 2.0

## Configuration Structure

```json
{
    "version": "2.0",
    "environment": "development",
    "trading": {
        "bucket": "Scalping",
        "initial_capital": 100000.0,
        "max_positions": 50,
        "window_size": 288,
        "look_back_amount": 1,
        "look_back_unit": "day(s)",
        "resume_checkpoint": false,
        "checkpoint_interval": 10
    },
    "model": {
        "type": "ActorCritic",
        "hidden_size": 512,
        "num_layers": 2,
        "kernel_size": 3,
        "activation": "LeakyReLU",
        "dropout": 0.1,
        "use_fusion": true,
        "use_rnn": false,
        "rnn_type": "LSTM",
        "rnn_layers": 1,
        "rnn_hidden_size": 128,
        "attention_heads": 4,
        "learning_rate": 0.0003,
        "batch_size": 128,
        "buffer_size": 2048,
        "gamma": 0.99,
        "lambda": 0.95,
        "epsilon": 0.1,
        "eps_clip": 0.2,
        "entropy_coef": 0.01,
        "value_coef": 0.5,
        "max_grad_norm": 0.5,
        "weight_decay": 1e-5,
        "max_steps_per_episode": 500
    },
    "risk": {
        "max_btc_per_position": 10.0,
        "max_usd_per_position": 1000000.0,
        "max_volume_percentage": 0.05,
        "stop_loss": 0.05,
        "take_profit": 0.1,
        "trailing_stop": 0.02,
        "use_dynamic_sl": false,
        "max_drawdown": 0.15,
        "risk_management": true
    },
    "withdrawal_simulation": {
        "monthly_withdrawal_chance": 0.3,
        "emergency_withdrawal_chance": 0.05,
        "timed_withdrawal_chance": 0.15,
        "monthly_deposit_chance": 0.4,
        "withdrawal_min_pct": 0.05,
        "withdrawal_max_pct": 0.3,
        "deposit_min_pct": 0.05,
        "deposit_max_pct": 0.5
    },
    "naturalistic_learning": {
        "use_surprise_replay": true,
        "surprise_threshold": 0.8,
        "replay_buffer_size": 10000,
        "priority_alpha": 0.6,
        "priority_beta": 0.4,
        "use_meta_learning": true,
        "adapt_param_freq": 10,
        "use_post_trade_analysis": true,
        "lesson_memory_size": 100,
        "initial_exploration_rate": 1.0,
        "min_exploration_rate": 0.1,
        "exploration_decay": 0.995,
        "exploration_decay_method": "performance"
    },
    "contextual_memory": {
        "use_contextual_memory": true,
        "memory_capacity": 200,
        "similarity_threshold": 0.7,
        "memory_recall_count": 3
    },
    "cross_bucket_transfer": {
        "use_cross_bucket_transfer": true,
        "weight_transfer_alpha": 0.3,
        "feature_transfer_alpha": 0.5,
        "transfer_cooldown": 10,
        "enable_reverse_transfer": true
    },
    "probabilistic_prediction": {
        "use_probabilistic": true,
        "prediction_horizons": [12, 36, 72, 144],
        "confidence_threshold": 0.6,
        "calibration_weight": 0.3,
        "position_sizing_strategy": "confidence_adjusted",
        "uncertainty_penalty": 0.5,
        "min_confidence_threshold": 0.4,
        "max_uncertainty_threshold": 0.3,
        "use_dynamic_horizons": true,
        "min_horizon": 1,
        "max_horizon": 576,
        "horizon_density": "medium",
        "horizon_update_freq": 10
    }
}
```

## Parameter Descriptions

### Trading Section

| Parameter | Type | Description | Constraints |
|-----------|------|-------------|-------------|
| bucket | string | Trading bucket type | ["Scalping", "Short", "Medium", "Long"] |
| initial_capital | number | Starting capital | > 0 |
| max_positions | integer | Maximum number of positions | > 0 |
| window_size | integer | Number of time steps in window | > 0 |
| look_back_amount | integer | Amount to look back | > 0 |
| look_back_unit | string | Unit for look back | ["day(s)", "week(s)", "month(s)"] |
| resume_checkpoint | boolean | Whether to resume from checkpoint | true/false |
| checkpoint_interval | integer | Episodes between checkpoints | > 0 |

### Model Section

| Parameter | Type | Description | Constraints |
|-----------|------|-------------|-------------|
| type | string | Model architecture type | ["ActorCritic", "PPO", "A2C"] |
| hidden_size | integer | Size of hidden layers | > 0 |
| num_layers | integer | Number of layers | > 0 |
| kernel_size | integer | Size of convolution kernel | > 0 |
| activation | string | Activation function | ["ReLU", "LeakyReLU", "ELU"] |
| dropout | number | Dropout rate | [0, 1] |
| learning_rate | number | Learning rate | [0, 1] |
| batch_size | integer | Batch size | > 0 |
| gamma | number | Discount factor | [0, 1] |
| lambda | number | GAE parameter | [0, 1] |
| epsilon | number | Exploration rate | [0, 1] |
| eps_clip | number | PPO clip parameter | [0, 1] |

### Risk Section

| Parameter | Type | Description | Constraints |
|-----------|------|-------------|-------------|
| max_btc_per_position | number | Maximum BTC per position | > 0 |
| max_usd_per_position | number | Maximum USD per position | > 0 |
| max_volume_percentage | number | Maximum volume percentage | [0, 1] |
| stop_loss | number | Stop loss percentage | [0, 1] |
| take_profit | number | Take profit percentage | [0, 1] |
| trailing_stop | number | Trailing stop percentage | [0, 1] |
| max_drawdown | number | Maximum drawdown | [0, 1] |

### Withdrawal Simulation Section

| Parameter | Type | Description | Constraints |
|-----------|------|-------------|-------------|
| monthly_withdrawal_chance | number | Chance of monthly withdrawal | [0, 1] |
| emergency_withdrawal_chance | number | Chance of emergency withdrawal | [0, 1] |
| withdrawal_min_pct | number | Minimum withdrawal percentage | [0, 1] |
| withdrawal_max_pct | number | Maximum withdrawal percentage | [0, 1] |

### Naturalistic Learning Section

| Parameter | Type | Description | Constraints |
|-----------|------|-------------|-------------|
| surprise_threshold | number | Threshold for surprise detection | [0, 1] |
| replay_buffer_size | integer | Size of replay buffer | > 0 |
| priority_alpha | number | Priority replay alpha | [0, 1] |
| priority_beta | number | Priority replay beta | [0, 1] |
| initial_exploration_rate | number | Initial exploration rate | [0, 1] |
| min_exploration_rate | number | Minimum exploration rate | [0, 1] |

### Contextual Memory Section

| Parameter | Type | Description | Constraints |
|-----------|------|-------------|-------------|
| memory_capacity | integer | Maximum memory capacity | > 0 |
| similarity_threshold | number | Threshold for similarity | [0, 1] |
| memory_recall_count | integer | Number of memories to recall | > 0 |

### Cross-Bucket Transfer Section

| Parameter | Type | Description | Constraints |
|-----------|------|-------------|-------------|
| weight_transfer_alpha | number | Weight transfer coefficient | [0, 1] |
| feature_transfer_alpha | number | Feature transfer coefficient | [0, 1] |
| transfer_cooldown | integer | Episodes between transfers | > 0 |

### Probabilistic Prediction Section

| Parameter | Type | Description | Constraints |
|-----------|------|-------------|-------------|
| confidence_threshold | number | Minimum confidence threshold | [0, 1] |
| calibration_weight | number | Calibration weight | [0, 1] |
| uncertainty_penalty | number | Penalty for uncertainty | [0, 1] |
| min_horizon | integer | Minimum prediction horizon | > 0 |
| max_horizon | integer | Maximum prediction horizon | > 0 |

## Validation Rules

1. All numeric parameters must be within their specified ranges
2. All integer parameters must be positive
3. All percentage values must be between 0 and 1
4. The bucket type must be one of the allowed values
5. The model type must be one of the allowed values
6. The activation function must be one of the allowed values
7. The look_back_unit must be one of the allowed values
8. The prediction_horizons array must be sorted in ascending order
9. The max_horizon must be greater than or equal to min_horizon
10. The withdrawal_max_pct must be greater than or equal to withdrawal_min_pct

## Default Values

Default values are provided in the `DEFAULT_CONFIG` constant in the `ConfigCompatibility` class. These values are used when:
1. A configuration file is not found
2. A required parameter is missing
3. A parameter value is invalid

## Version History

- 1.0: Original configuration format
- 1.1: Added withdrawal simulation parameters
- 1.2: Added naturalistic learning parameters
- 2.0: Reorganized into logical sections with validation 