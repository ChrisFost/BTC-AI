# BTC AI System Features Overview

## Purpose
This document provides a high-level overview of the main features and components of the BTC AI system, organized by module/area.

## Table of Contents
1. [Core Logic](#core-logic)
    1.1. [Models](#models)
    1.2. [Environment](#environment)
    1.3. [Training](#training)
    1.4. [Agent](#agent)
2. [User Interface (UI)](#ui)
3. [Supporting Systems](#supporting-systems)
    3.1. [Utilities (Utils)](#utils)
    3.2. [Configuration Management](#configuration-management)
    3.3. [Data Management](#data-management)
4. [Development & Operations](#development--operations)
    4.1. [Build, Deployment, and Installation](#build-deployment-and-installation)
    4.2. [Scripts and Tools](#scripts-and-tools)
    4.3. [Update Mechanism](#update-mechanism)
    4.4. [Documentation](#documentation)
5. [Operational Directories](#operational-directories)

## 1. Core Logic

### 1.1. Models
Core modeling components that handle predictions and trading strategies. This includes investigating approaches like fractal pattern recognition and Evolutionary Strategies (ES).

#### Key Files/Dirs:
- `src/models/models.py`
- `src/models/dynamic_horizon_predictor.py`
- `Models/Short/` (Subdirectory intended for short-term models, currently holds training history logs)
- `Models/Scalping/` (Subdirectory intended for scalping models, currently holds training logs)
- `Models/Medium/` (Subdirectory intended for medium-term models, currently holds training history logs)
- `Models/Long/` (Subdirectory intended for long-term models, currently holds training history logs)
- `Models/monitoring/` (Subdirectory intended for monitoring related models/scripts, currently holds monitoring logs like memory usage)
- `Models/knowledge_transfer/` (Subdirectory intended for knowledge transfer models/scripts, currently holds knowledge transfer history logs)
- `Models/menu.py` (Standalone PySimpleGUI control panel, functionally redundant with `src/ui/` system)
- `Models/backtest.py`
- `Models/evaluate.py`
- `Models/main.py`
- `Models/prepare_data.py`
- `Models/probabilistic_model.py`
- `Models/README.md`
- `Models/realtime_inference.py`
- `Models/requirements.txt`
- `Models/trading_model.py`
- `Models/train_probabilistic.py`
- `Models/visualize.py`


*Features to be documented:*
    *   ***Primary Advanced Model Architecture (`ActorCritic` in `src/models/models.py`):***
        *   Core: Transformer-based Actor-Critic framework.
        *   Key Capabilities:
            *   Advanced sequential data processing via Transformer Encoder.
            *   Sophisticated "Chain of Draft" Reasoning Pipeline for emulating human-like analytical process with focused note-taking for decision support. Includes:
                *   Market Regime Analysis & Classification.
                *   Technical Pattern Recognition (potentially including Fractal Pattern Recognition).
                *   Support/Resistance Level Detection.
                *   Regime-influenced Volatility Assessment.
                *   Market Liquidity Assessment.
                *   Pattern & S/R influenced Entry/Exit Point Scoring.
                *   Integration of above factors for nuanced decision making.
            *   Probabilistic Policy Head (outputs distributions for trading actions).
            *   Critic/Value Head (estimates state value for reinforcement learning).
            *   Dedicated Trend Strength Estimation Head.
            *   Uncertainty-Aware Risk Assessment Head (explicit risk module).
        *   **`DynamicHorizonPredictor` module (`src/models/dynamic_horizon_predictor.py` - Status: Keep, Key Component):**
            *   Provides sophisticated dynamic and probabilistic forecasting.
            *   Core Functionalities:
                *   Predicts a probability distribution over the *optimal prediction horizon* based on current input features.
                *   Conditionally predicts a probabilistic *outcome* (mean, std) and a *confidence score* for any specifically *requested horizon*.
                *   Employs horizon encoding to process the requested horizon value as a feature.
            *   Intended as the primary system for flexible multi-horizon forecasting, superseding static multi-head approaches.
            *   *Assessment:* Well-structured and conceptually sound module.
            *   *Potential Future Considerations:* Minor enhancements could include exploring more complex internal head architectures if performance demands, or adding more extensive "why" comments for long-term clarity and more robust config validation.
        *   Extensive Explainability & Interpretability features (internal activation maps, gradient-based feature attribution, input feature weighting, dedicated methods to explain predictions, decisions, and reasoning chain outputs).
        *   Probabilistic Output Processing (generation of prediction samples, calculation of confidence intervals, potential for "Surprise Factor" analysis from prediction likelihoods).
    *   ***Alternative/Legacy Model Architectures (`src/models/models.py` - Identified for Review/Deprecation):***
        *   `LSTMPolicyNetwork`:
            *   Original Core: LSTM-based policy and value estimation.
            *   *Assessment:* Its prior unique feature, multi-fixed-horizon prediction heads, is now superseded by the `DynamicHorizonPredictor` system. The core LSTM policy, while functional, is a simpler recurrent architecture. Considered a candidate for deprecation in favor of the more advanced `ActorCritic` Transformer-based model. Does not possess the explicit risk/decision sub-modules found in `ActorCritic`.
        *   `HybridPolicyNetwork`:
            *   Original Core: Integrated 1D CNN feature extractor followed by an LSTM for policy and value estimation.
            *   *Assessment:* Its multi-fixed-horizon prediction heads are superseded. The integrated CNN+LSTM architecture is a specific combination. Considered a candidate for deprecation. The CNN feature extraction capability is available via the standalone `CNNFeatureExtractor`, and its LSTM policy component is similar to `LSTMPolicyNetwork` (also a deprecation candidate). Does not possess the explicit risk/decision sub-modules found in `ActorCritic`.
    *   ***Dedicated Feature Extraction Module (`src/models/models.py`):***
        *   `CNNFeatureExtractor`: Provides a reusable 1D CNN for extracting features from time-series data (e.g., local temporal patterns). This remains a potentially valuable utility for preprocessing or specialized feature engineering.
    *   ***Supporting Model Infrastructure & Management (from `Models/` directory & `src/models/models.py`):***
        *   Organization for different model types/timeframes (e.g., `Models/Scalping/`, `Models/Short/`, `Models/Medium/`, `Models/Long/` - currently store logs/history related to these specializations).
        *   `Models/menu.py` Assessment: This script is a standalone PySimpleGUI control panel, found to be functionally redundant with the more comprehensive and modular `src/ui/` system. It primarily interacts with scripts and configurations within the `Models/` directory itself and is a candidate for removal to avoid parallel, potentially conflicting UI systems.
        *   Utilities for model-specific data preparation (`Models/prepare_data.py`), training (`Models/train_probabilistic.py`), evaluation (`Models/evaluate.py`), and visualization (`Models/visualize.py`) - *Note: These need to be assessed for redundancy or integration with the main `src/training` and `src/utils` systems.*
        *   Potential for Knowledge Transfer mechanisms between models (implied by `Models/knowledge_transfer/` directory and its logs).
        *   General Infrastructure: Reusable low-level model components (e.g., `CrossAttention`, `TransformerBlock` in `src/models/models.py`), model persistence (save/load), factory function (`create_model`) for flexible model instantiation, device management (CPU/GPU), and extensive configuration options via `config` dictionaries.

### 1.2. Environment
Trading environment implementation that simulates market conditions, handles risk management, and interactions.

#### Key Files/Dirs:
- `src/environment/env_base.py`
- `src/environment/env_tensor.py`
- `src/environment/env_risk.py` (Covers risk management functions)
- `src/environment/env_market.py`
- `src/environment/env_observation.py`
- `src/environment/env_rewards.py`

*Features to be documented (e.g., simulation accuracy, market data integration, risk controls, observation space, reward structure)*

### 1.3. Training
Components for training models, evaluating performance, comprehensive backtesting, and optimization. This includes mechanisms like early stopping.

#### Key Files/Dirs:
- `src/training/training.py`
- `src/training/backtesting.py` (Core backtesting engine)
- `src/training/prob_evaluator.py` (Attached file, evaluates probabilistic predictions)
- `src/training/progressive.py`
- `src/training/optimizer.py` (Potential location for ES, early stopping logic)
- `src/training/realtime_inference.py`

*Features to be documented (e.g., backtesting framework capabilities, performance metrics from `prob_evaluator.py`, optimization algorithms, early stopping criteria, progressive training logic)*

### 1.4. Agent
Agent implementation that handles decision making and strategy execution based on model outputs and environmental state.

#### Key Files/Dirs:
- `src/agent/agent.py` (Main agent logic)
- `src/agent/agent_base.py`

*Features to be documented (e.g., decision-making logic, trade execution interface, state management, interaction with environment and models)*

## 2. User Interface (UI)
User interface components for application interaction, including configuration, monitoring, and backtesting integration.

#### Key Files/Dirs:
- `src/ui/main.py` (Main application entry point for UI)
- `src/ui/layouts.py` (Defines overall UI structure and tab organization)
- `src/ui/tabs.py` (Defines content for individual tabs like Dashboard, Strategy, Model, Probabilistic, etc.)
- `src/ui/app_state.py`
- `src/ui/bucket_manager.py`
- `src/ui/comparison_manager.py`
- `src/ui/preset_manager.py`
- `src/ui/training_manager.py`
- `src/ui/backtesting_integration.py` (Connects UI to backtesting features)
- `src/ui/setup_wizard.py` (Part of user-facing installation/setup)
- `src/ui/menus.py` (Defines the main menu bar)

*Features to be documented (e.g., main application flow, tab functionalities, training job management, state persistence, preset handling, backtesting controls, setup process guided by `setup_wizard.py`, detailed breakdown of available settings and controls per tab based on `src/ui/tabs.py`)*

## 3. Supporting Systems

### 3.1. Utilities (Utils)
Utility functions and components for supporting the system, including data frame building and manipulation, tensor operations, and checkpointing.

#### Key Files/Dirs:
- `src/utils/utils.py`
- `src/utils/tensor_utils.py`
- `src/utils/dataframe.py` (Handles data frame building features, potentially autoencoding)
- `src/utils/emergency_checkpoint.py`
- `src/utils/prediction_visualizer.py`
- `src/utils/training_visualizer.py`
- `src/utils/reasoning.py`
- `src/utils/config_compatibility.py`
- `src/utils/config_versioning.py`
- `src/utils/persistent_logger.py`
- `src/utils/platform_utils.py`
- `src/utils/update_manager.py` (Backend logic for updates, complements `src/ui/update_handler.py`)

*Features to be documented (e.g., common data structures, mathematical functions, file I/O, logging, specific utilities for dataframes, tensors, visualization tools, configuration handling, emergency save/load, update management)*

### 3.2. Configuration Management
System configuration, presets, and settings management.

#### Key Files/Dirs:
- `configs/` (Primary configuration directory, e.g., `config.json`, `app_state.json` - for the `src/` application)
- `presets/` (Preset configurations for the `src/` application, e.g., `presets/defaults/`, `presets/user/`)
- `Models/config.json` (Separate config for the `Models/menu.py` standalone UI)

*Features to be documented (e.g., structure of main config files, preset system functionality, application state persistence, distinction from `Models/config.json`, versioning and compatibility of configs handled by utils)*

### 3.3. Data Management
Data handling, processing, and storage components, including raw data sources and potential pipelines.


#### Key Files/Dirs:
- `2020-2024_BTCUSDT_DATA/` (Example primary raw data source)
- `data/` (Other general data storage)

*Features to be documented (e.g., data ingestion methods, cleaning processes, feature engineering steps, data storage solutions, how `dataframe.py` is used in pipelines)*

## 4. Development & Operations

### 4.1. Build, Deployment, and Installation
Components and scripts related to building the application, deploying it, and the initial installation process.

#### Key Files/Dirs:
- `build/` (Output of build process)
- `dist/` (Distribution packages)
- `requirements.txt` (Python package dependencies)
- `build_app.py` (Script for building the application, likely using PyInstaller)
- `simple_spec.spec` (PyInstaller specification file)
- `src/ui/setup_wizard.py` (Guides user through setup)
- `setup_wizard_test.log` (Log for setup wizard tests)

*Features to be documented (e.g., build process, packaging, dependency management, installation steps for users, automated setup features)*

### 4.2. Scripts and Tools
Various scripts and tools supporting development, operations, and maintenance.

#### Key Files/Dirs:
- `Scripts/` (General utility scripts)
- `tools/` (Various development or operational tools)
- `btc_backup.ps1` (PowerShell script for backups)
- `update_rules.ps1` (PowerShell script for updating rules)
- `make_changes.py`
- `add_presets_tab.py`

*Features to be documented (e.g., purpose of each script/tool, backup strategy, rule update mechanism)*

### 4.3. Update Mechanism
Components responsible for application updates.

#### Key Files/Dirs:
- `update_server/` (Potential server-side components for updates)
- `src/ui/update_handler.py` (Client-side UI for updates)
- `src/ui/update_dialog.py`
- `src/utils/update_manager.py` (Backend logic for updates)

*Features to be documented (e.g., update process, client-server interaction for updates, version checking)*

### 4.4. Documentation
User and developer documentation.

#### Key Files/Dirs:
- `docs/` (General documentation directory)
- `README.md` files in various directories (e.g., `Models/README.md`)
- `src/ui/README_PRESETS.md`

*Features to be documented (e.g., availability and scope of user guides, developer guides, API documentation)*

## 5. Operational Directories
Directories used during runtime for logging, caching, and storing operational data. These are generally not user-configured features but are important for system operation and debugging.

- `logs/`: Application and system logs.
- `Cache/`: Caching data to improve performance.
- `emergency_checkpoints/`: Storing state during critical situations (linked to `src/utils/emergency_checkpoint.py`).
- `comparison_results/`: Storing outputs of model/strategy comparisons.
- `outputs/`: General-purpose directory for outputs from various processes.
- `temp_fixes/`: Temporary patches or fixes.

*Briefly describe the purpose of each directory and any related management utilities or configurations.* 