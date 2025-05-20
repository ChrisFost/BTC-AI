# BTC-AI Trading System Architecture

## System Overview

The BTC-AI Trading System is a sophisticated platform for training, evaluating, and deploying AI-powered trading agents for Bitcoin markets. The system employs reinforcement learning with a novel progressive training approach that allows agents to transfer knowledge across different trading timeframes.

![System Architecture](https://mermaid.ink/img/pako:eNqNkk1PwzAMhv9KlAskpB7YKIeeEOK0E0JomaO2uMvQEleSCrEy_juGtdBNrNqdEr-P7cRpBKtdggrUI8FpLTPxstGp2YgbVR5LIemMyRXJ1jkDV9cxlGSdZzFY2vVsVKMZ2G3UJk1jXdTNUEHLNtK5C-KR3SZFhFdX5PV0AjmRTnN5Yj9OE3qOJUYpWVyPQrJXxlY0oDrTwK6l1o4aBNTpQHiQmTmYtGYf5zXGD-uqUj1ChlU_XrIuJLEsv9FcG3JdFPTJ1-CIOxvLbvbOz9f4NzfAG2cDw3yk_Rj9-vO_nP9SfvYnp-vXNBJrZlm4H5x_cP5Eo6a_8L9GxA52-dGz3DQFjgpdQrDfYXNUJ2JGZnI_bCDI_R6Sggo-pBbkwAVdULUBmXOJZdRtVQ1JOz9Kx7vdF5YJu_g?type=png)

## Core Components

### 1. User Interface (`src/ui/main.py`, `src/ui/components/`)
The primary interface for managing the trading system, offering:
- Configuration of trading parameters
- Training management (start, stop, pause)
- Visualization of training progress
- Model performance analysis
- Withdrawal/deposit simulation
- Bucket-specific goal setting
- Probabilistic prediction configuration
- Help tabs with comprehensive documentation
- Software updates through the update manager

#### Key Files:
- `src/ui/main.py`: Main application window and UI initialization
- `src/ui/components/training_tab.py`: Training configuration and control
- `src/ui/components/visualization_tab.py`: Performance visualization tools
- `src/ui/components/settings_tab.py`: System-wide settings management
- `src/ui/components/help_tab.py`: In-application documentation
- `src/ui/components/update_tab.py`: Software update interface

### 2. Agent (`src/agent/agent.py`, `src/agent/reasoning.py`)
The reinforcement learning agent that makes trading decisions, featuring:
- PPO (Proximal Policy Optimization) for stable training
- Probabilistic predictions with uncertainty estimates
- Experience replay with surprise-based prioritization
- Meta-learning for hyperparameter optimization
- Contextual memory for recognizing market patterns
- Dynamic exploration rate adaptation
- Post-trade analysis system
- Cross-bucket knowledge transfer

#### Key Files:
- `src/agent/agent.py`: Core agent implementation with PPO algorithm
- `src/agent/reasoning.py`: Chain of reasoning and reflection capabilities
- `src/agent/memory.py`: Experience replay and prioritization
- `src/agent/exploration.py`: Adaptive exploration strategies
- `src/agent/transfer.py`: Cross-bucket knowledge transfer mechanisms

### 3. Environment (`src/env/env_base.py`, `src/env/env_risk.py`)
Simulates the trading environment with:
- Realistic market dynamics
- Risk management constraints
- Trading fees and slippage
- Position sizing based on confidence
- Withdrawal and deposit handling
- Bucket-specific reward calculation
- Liquidity-aware execution model
- Simulated market events and scenarios

#### Key Files:
- `src/env/env_base.py`: Base environment implementation
- `src/env/env_risk.py`: Enhanced environment with risk management
- `src/env/reward.py`: Reward function implementation for each bucket
- `src/env/market_simulation.py`: Realistic market simulation
- `src/env/events.py`: Market event simulation (news, volatility spikes)

### 4. Models (`src/models/models.py`, `src/models/fusion.py`)
Neural network architectures for the agent:
- Fusion architecture combining CNN, LSTM, and Transformer layers
- Actor-critic network structure
- Probabilistic output layers for multiple horizons
- Attention mechanisms for time series
- Dynamic prediction horizon adjustment
- Uncertainty calibration mechanisms
- Feature importance tracking

#### Key Files:
- `src/models/models.py`: Core model architecture definitions
- `src/models/fusion.py`: Fusion network implementation
- `src/models/calibration.py`: Uncertainty calibration methods
- `src/models/layers.py`: Custom neural network layers
- `src/models/feature_importance.py`: Feature importance tracking

### 5. Visualization (`src/utils/visualize.py`)
Tools for visualizing agent performance:
- Real-time training metrics
- Trade history visualization
- P&L charts
- Confidence intervals for predictions
- Uncertainty visualization for decision making
- Reasoning chain visualization
- Multi-horizon prediction displays
- Calibration curve analysis

#### Key Files:
- `src/utils/visualize.py`: Core visualization functions
- `src/utils/plot_trades.py`: Trade visualization
- `src/utils/plot_metrics.py`: Training metrics visualization
- `src/utils/plot_uncertainty.py`: Uncertainty visualization
- `src/utils/plot_reasoning.py`: Reasoning visualization

### 6. Analytical Tools (`src/utils/reasoning_analyzer.py`)
Analysis of agent decision-making:
- Post-trade analysis
- Chain of reasoning process
- Self-reflection capabilities
- Performance attribution
- Temporal consistency analysis
- Meta-parameter adjustment based on reasoning effectiveness
- Contrastive analysis between successful and unsuccessful trades

#### Key Files:
- `src/utils/reasoning_analyzer.py`: Core reasoning analysis
- `src/utils/performance_attribution.py`: Performance breakdown
- `src/utils/consistency_tracker.py`: Temporal consistency analysis
- `src/utils/contrastive_analyzer.py`: Success vs. failure analysis

### 7. Data Management (`src/data/`)
Handles data fetching, processing, and storage:
- Historical data acquisition
- Real-time data streaming
- Feature engineering
- Data normalization and preprocessing
- Dataset splitting for training/validation
- Data augmentation techniques

#### Key Files:
- `src/data/fetcher.py`: Data acquisition from exchanges
- `src/data/processor.py`: Data preprocessing and normalization
- `src/data/features.py`: Feature engineering
- `src/data/dataset.py`: Dataset creation for training
- `src/data/augmentation.py`: Data augmentation methods

### 8. Update System (`src/update/update_manager.py`)
Manages application updates with:
- Version checking against remote server
- Secure downloading with checksum verification
- Automatic and manual update options
- Backup creation before updates
- Update application with progress reporting
- Rollback capability for failed updates

#### Key Files:
- `src/update/update_manager.py`: Core update management
- `src/update/version_checker.py`: Version comparison logic
- `src/update/download.py`: Secure download implementation
- `src/update/apply.py`: Update application process
- `src/update/rollback.py`: Rollback functionality

### 9. Error Logging (`src/utils/logger.py`)
Comprehensive error logging system:
- Rotating log files with automatic archiving
- Different log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Rich formatting with timestamps and context
- Exception capture with stack traces
- Configurable verbosity
- Log file rotation to prevent excessive disk usage

#### Key Files:
- `src/utils/logger.py`: Core logging implementation
- `src/utils/error_handler.py`: Exception handling helpers
- `src/utils/rotate.py`: Log rotation functionality

## Trading Buckets Framework

The system employs four distinct trading timeframes, or "buckets":

### 1. Scalping
- **Purpose**: Very short-term trades (minutes to hours) with frequent entries/exits
- **Lookback Window**: 144 bars (12 hours at 5-min intervals)
- **Goal Structure**: Monthly profit targets (15-30%)
- **Priority Metrics**: Net profit and win rate
- **Max Positions**: 25 concurrent trades

### 2. Short
- **Purpose**: Short-term trades (hours to days) with selective entries
- **Lookback Window**: 288 bars (1 day at 5-min intervals)
- **Goal Structure**: Yearly profit targets (100-200%)
- **Priority Metrics**: Win rate and drawdown management
- **Max Positions**: 15 concurrent trades

### 3. Medium
- **Purpose**: Medium-term trades (days to weeks) targeting larger moves
- **Lookback Window**: 1440 bars (5 days at 5-min intervals)
- **Goal Structure**: Per-holding gain targets (25-50%)
- **Priority Metrics**: Net profit with bonus multiplier
- **Max Positions**: 8 concurrent trades

### 4. Long
- **Purpose**: Long-term trades (weeks to months) for significant market trends
- **Lookback Window**: 4032 bars (2 weeks at 5-min intervals)
- **Goal Structure**: Per-holding gain targets (50-100%)
- **Priority Metrics**: Drawdown management and per-trade gains
- **Max Positions**: 4 concurrent trades

## Naturalistic Learning Features

The system implements several mechanisms to create more human-like learning:

### 1. Adaptive Exploration
- Dynamically adjusts exploration rate based on market regime
- Reduces exploration in well-understood scenarios
- Increases exploration in unfamiliar market conditions
- Supports multiple decay methods (time-based, performance-based, adaptive)

### 2. Experience Prioritization
- Prioritizes surprising outcomes for faster learning
- Assigns higher importance to trades with unexpected results
- Maintains a prioritized replay buffer for more efficient learning
- Implements importance sampling with bias correction

### 3. Post-Trade Analysis
- Performs detailed analysis after completing trades
- Compares predicted vs. actual outcomes
- Extracts lessons from each trade
- Maintains a memory of trade lessons for future reference

### 4. Contextual Memory
- Stores embeddings of key market situations
- Recalls similar past scenarios when making decisions
- Uses similarity thresholds to identify relevant experiences
- Adjusts confidence based on historical outcomes in similar situations

### 5. Cross-Bucket Knowledge Transfer
- Shares knowledge between different trading timeframes
- Transfers neural network weights with configurable alpha parameters
- Shares feature importance across timeframes
- Adapts prediction horizons based on performance
- Supports bidirectional transfer (short→long and long→short)

## Probabilistic Prediction System

The system employs advanced probabilistic prediction capabilities:

### 1. Multi-Horizon Predictions
- Predicts price movements across multiple time horizons
- Dynamically adjusts horizons based on market conditions
- Provides both mean prediction and standard deviation (uncertainty)
- Calibrates confidence scores with historical accuracy

### 2. Position Sizing Strategies
- Confidence-adjusted: Larger positions for more confident predictions
- Uncertainty-adjusted: Smaller positions when uncertainty is high
- Full probabilistic: Uses complete probability distributions for sizing

### 3. Uncertainty Management
- Enforces maximum uncertainty thresholds for trade entry
- Applies uncertainty penalties to position sizing
- Visualizes prediction uncertainty with confidence intervals
- Tracks calibration metrics for prediction reliability

## Chain of Reasoning and Reflection

The system implements an explicit reasoning process:

### 1. Reasoning Components
- Market regime classification
- Technical pattern recognition
- Support/resistance identification
- Volatility assessment
- Liquidity analysis
- Entry/exit signal generation

### 2. Reflection Capabilities
- Performance attribution to reasoning components
- Contrastive analysis between successful/unsuccessful reasoning
- Temporal consistency tracking
- Meta-parameter adjustment for reasoning weights
- Visualization of reasoning effectiveness

## Data Flow

1. **Data Ingestion**: Historical BTC market data is preprocessed and fed into the system
2. **Environment Setup**: Trading parameters are configured through the UI
3. **Training Process**: 
   - Agent interacts with environment by taking actions (buy/sell/hold)
   - Environment provides rewards based on trading outcomes
   - Agent updates its policy to maximize rewards
4. **Knowledge Transfer**: Knowledge is shared between different trading timeframes
5. **Evaluation**: Performance metrics are calculated and visualized
6. **Deployment**: Trained models can be used for real-time trading

## Training Pipeline

The progressive training pipeline follows these steps:
1. Start with short timeframe (Scalping)
2. Train until convergence
3. Transfer knowledge to next timeframe (Short)
4. Continue process through Medium and Long timeframes
5. Periodically update all models with shared knowledge

## Update Mechanism

The application includes a robust update system:

1. **Version Checking**:
   - Periodically checks for updates from a configured server
   - Compares semantic versioning (major.minor.patch)
   - Supports minimum required version enforcement

2. **Update Process**:
   - Downloads updates securely with SSL/TLS
   - Verifies package integrity with checksums
   - Creates automatic backups before applying updates
   - Provides detailed changelogs to users

3. **Rollback Capability**:
   - Automatically restores from backup if update fails
   - Maintains version history for manual rollback
   - Logs all update actions for troubleshooting

4. **User Control**:
   - Manual update checking through UI
   - Options for automatic or manual updates
   - User notification of available updates
   - Progress reporting during download and installation

## Error Logging System

The application features a comprehensive logging system:

1. **Log Levels**:
   - DEBUG: Detailed debugging information
   - INFO: General information about system operation
   - WARNING: Potential issues that don't prevent operation
   - ERROR: Errors that prevent specific functionality
   - CRITICAL: Errors that may cause system failure

2. **Rotation Features**:
   - Size-based rotation (creates new file when size limit reached)
   - Time-based rotation (daily, weekly, monthly options)
   - Backup count limit to prevent excessive storage usage
   - Compression of archived logs

3. **Output Formats**:
   - Console output for immediate visibility
   - File output for persistent records
   - Optional email notifications for critical errors
   - JSON format option for automated processing

4. **Context Capture**:
   - Function and line number tracking
   - Timestamp with millisecond precision
   - Thread/process identification
   - Exception stack traces with context

## Key Technologies

- **Python**: Core programming language
- **PyTorch**: Deep learning framework
- **PySimpleGUI**: User interface library
- **Numpy/Pandas**: Data processing
- **Matplotlib/Plotly**: Visualization
- **Multiprocessing**: Parallel training
- **Requests**: HTTP client for updates
- **Logging**: Python's built-in logging framework
- **PyInstaller**: Application packaging

## System Requirements

- **Hardware**:
  - CPU: Quad-core or better
  - RAM: 8GB minimum, 16GB recommended
  - GPU: CUDA-compatible (optional but recommended)
  - Storage: 10GB minimum
  
- **Software**:
  - Python 3.7 or higher
  - PyTorch 1.8 or higher
  - CUDA toolkit (for GPU acceleration)
  - Required Python packages (see requirements.txt)

## Cross-Component Communication

Components communicate primarily through:
1. **File System**: Checkpoints, logs, and configuration files
2. **Function Calls**: Direct API calls between components
3. **Subprocess Communication**: For isolating training processes
4. **Event-driven Architecture**: For UI interactions

## Error Handling

The system includes robust error handling:
- Graceful termination of training processes
- Automatic checkpoint saving for recovery
- Comprehensive logging for debugging
- Input validation to prevent misconfiguration
- Timeout mechanisms to prevent stuck processes
- Memory optimization for long-running operations

## Security Considerations

- API keys and sensitive data are stored securely
- No direct network access from core components
- Validation of all input data
- Limited file system access
- Checksums for update verification
- HTTPS for secure update downloads

## Scalability

The system is designed to scale with:
- Support for multiple GPU training
- Dynamic adjustment of resource usage
- Parallel environment simulation
- Efficient memory management for large datasets
- Distributed training capability
- Modular architecture for easy extension 