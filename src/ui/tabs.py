"""
Tab System for the RL Trader Parameter Tuner

This module contains all the tab creation functions for the UI.
Each function returns a tab layout that can be used in the main window.
"""

import PySimpleGUI as sg
import os
import sys
import json
from pathlib import Path

# Make sure we can import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import any needed dependencies
# The dynamic import system will be used by main.py when it imports this module
try:
    from src.utils.bucket_goals import create_goal_provider
    bucket_goals_available = True
except ImportError:
    bucket_goals_available = False
    
    def create_goal_provider(*args, **kwargs):
        return None

# Default configuration in case not provided
config = {
    "BUCKET": "Scalping",
    "monthly_target_min": 15.0,
    "monthly_target_max": 30.0,
    "yearly_target_min": 100.0,
    "yearly_target_max": 200.0,
    "min_gain_per_holding": 25.0,
    "max_gain_per_holding": 50.0,
    "bonus_multiplier": 1.1
}

def create_dashboard_tab():
    """Create the dashboard overview tab"""
    # System status section
    system_status = [
        [sg.Text("CPU Usage:"), sg.ProgressBar(100, orientation='h', size=(20, 20), key="-CPU-BAR-")],
        [sg.Text("Memory Usage:"), sg.ProgressBar(100, orientation='h', size=(20, 20), key="-MEM-BAR-")],
        [sg.Text("GPU Usage:"), sg.ProgressBar(100, orientation='h', size=(20, 20), key="-GPU-BAR-")],
        [sg.Text("Training Time:"), sg.Text("0:00:00", key="-TRAIN-TIME-DASH-")],
        [sg.Text("Episodes:"), sg.Text("0", key="-EPISODES-")]
    ]
    
    # Performance metrics section
    performance_metrics = [
        [sg.Text("Net Profit:"), sg.Text("$0.00", key="-NET-PROFIT-")],
        [sg.Text("Win Rate:"), sg.Text("0%", key="-WIN-RATE-")],
        [sg.Text("Sharpe Ratio:"), sg.Text("0.00", key="-SHARPE-")],
        [sg.Text("Max Drawdown:"), sg.Text("0%", key="-MAX-DRAWDOWN-")],
        [sg.Text("Total Trades:"), sg.Text("0", key="-TOTAL-TRADES-")]
    ]
    
    # Quick actions section
    quick_actions = [
        [sg.Button("Start Training", key="-QUICK-START-"),
         sg.Button("Stop Training", key="-QUICK-STOP-", disabled=True)],
        [sg.Button("Save Settings", key="-QUICK-SAVE-"),
         sg.Button("Load Settings", key="-QUICK-LOAD-")],
        [sg.Button("View Performance", key="-QUICK-PERF-"),
         sg.Button("View Log", key="-QUICK-LOG-")]
    ]
    
    # Alerts section
    alerts = [
        [sg.Multiline(size=(40, 5), key="-ALERTS-", disabled=True, autoscroll=True)]
    ]
    
    # Combine all sections into the dashboard tab
    dashboard_layout = [
        [sg.Column([
            [sg.Frame("System Status", system_status)],
            [sg.Frame("Performance Metrics", performance_metrics)]
        ]), sg.Column([
            [sg.Frame("Quick Actions", quick_actions)],
            [sg.Frame("Alerts & Notifications", alerts)]
        ])]
    ]
    
    return dashboard_layout 

def create_main_tab():
    """Create the main strategy tab with improved bucket settings"""
    
    # Create a provider for goal descriptions
    goal_provider = create_goal_provider(config)
    
    # --- Define Layout Lists for each Bucket Goal Section ---
    scalping_layout_def = [
        [sg.Text("For Scalping (Monthly Targets):", font=("Helvetica", 10, "bold"))],
        [sg.Text("Min (%):"), sg.InputText(str(config.get("monthly_target_min", 15.0)), key="monthly_target_min", size=(5,1)),
         sg.Text("Max (%):"), sg.InputText(str(config.get("monthly_target_max", 30.0)), key="monthly_target_max", size=(5,1))],
        [sg.Text(goal_provider.get_bucket_goal_description("Scalping") if goal_provider else "Monthly profit target: 15.0% - 30.0%", font=("Helvetica", 8, "italic"), text_color="gray", key="SCALPING_DESC")]
    ]

    short_layout_def = [
        [sg.Text("For Short (Yearly Targets):", font=("Helvetica", 10, "bold"))],
        [sg.Text("Min (%):"), sg.InputText(str(config.get("yearly_target_min", 100.0)), key="yearly_target_min", size=(5,1)),
         sg.Text("Max (%):"), sg.InputText(str(config.get("yearly_target_max", 200.0)), key="yearly_target_max", size=(5,1))],
        [sg.Text(goal_provider.get_bucket_goal_description("Short") if goal_provider else "Yearly profit target: 100.0% - 200.0%", font=("Helvetica", 8, "italic"), text_color="gray", key="SHORT_DESC")]
    ]

    medium_layout_def = [
        [sg.Text("For Medium (Per Holding Gain Targets):", font=("Helvetica", 10, "bold"))],
        [sg.Text("Min % Gain:"), sg.InputText(str(config.get("min_gain_per_holding", 25.0)), key="min_gain_per_holding_medium", size=(5,1)),
         sg.Text("Max % Gain:"), sg.InputText(str(config.get("max_gain_per_holding", 50.0)), key="max_gain_per_holding_medium", size=(5,1)),
         sg.Text("Bonus:"), sg.InputText(str(config.get("bonus_multiplier", 1.1)), key="bonus_multiplier_medium", size=(5,1))],
        [sg.Text(goal_provider.get_bucket_goal_description("Medium") if goal_provider else "Per holding gain target: 25.0% - 50.0%", font=("Helvetica", 8, "italic"), text_color="gray", key="MEDIUM_DESC")]
    ]
    
    long_layout_def = [
        [sg.Text("For Long (Per Holding Gain Targets):", font=("Helvetica", 10, "bold"))],
        [sg.Text("Min % Gain:"), sg.InputText(str(config.get("min_gain_per_holding", 25.0)), key="min_gain_per_holding_long", size=(5,1)),
         sg.Text("Max % Gain:"), sg.InputText(str(config.get("max_gain_per_holding", 50.0)), key="max_gain_per_holding_long", size=(5,1)),
         sg.Text("Bonus:"), sg.InputText(str(config.get("bonus_multiplier", 1.1)), key="bonus_multiplier_long", size=(5,1))],
        [sg.Text(goal_provider.get_bucket_goal_description("Long") if goal_provider else "Per holding gain target: 25.0% - 50.0%", font=("Helvetica", 8, "italic"), text_color="gray", key="LONG_DESC")]
    ]
    # --- End Layout Definitions ---

    # --- Create the TabGroup for Bucket Goals using the layout definitions ---
    bucket_goals_tabgroup = sg.TabGroup([
        [sg.Tab('Scalping', scalping_layout_def, key='-TAB-SCALPING-'),
         sg.Tab('Short', short_layout_def, key='-TAB-SHORT-'),
         sg.Tab('Medium', medium_layout_def, key='-TAB-MEDIUM-'),
         sg.Tab('Long', long_layout_def, key='-TAB-LONG-')]
    ], key='-GOALS-TABGROUP-', enable_events=True) 

    # Look-back period input with hint
    look_back_frame = sg.Frame("Look-Back Settings", [
        [sg.Text("Look-Back Amount:"), sg.InputText(str(config.get("LOOK_BACK_AMOUNT", 1)), key="LOOK_BACK_AMOUNT", size=(5,1)),
         sg.Combo(["hour(s)", "day(s)", "week(s)", "month(s)"], default_value=config.get("LOOK_BACK_UNIT", "day(s)"), key="LOOK_BACK_UNIT", size=(10,1)),
         sg.Text("(e.g., 1 hour for Scalping, 1 week for Long)", key="LOOK_BACK_HINT")]
    ])

    # Checkpoint recovery section
    checkpoint_frame = sg.Frame("Checkpoint Management", [
        [sg.Checkbox("Resume from Last Checkpoint", default=config.get("RESUME_CHECKPOINT", False), key="RESUME_CHECKPOINT"),
         sg.Button("Manage Checkpoints", key="MANAGE_CHECKPOINTS")]
    ])

    # Position size limits frame
    position_limits_frame = sg.Frame("Position Size Limits", [
        [sg.Text("Max BTC per position:"), 
         sg.InputText(str(config.get("MAX_BTC_PER_POSITION", 10.0)), key="MAX_BTC_PER_POSITION", size=(8,1)),
         sg.Text("BTC")],
        [sg.Text("Max USD per position:"), 
         sg.InputText(str(config.get("MAX_USD_PER_POSITION", 1000000.0)), key="MAX_USD_PER_POSITION", size=(10,1)),
         sg.Text("USD")],
        [sg.Text("Max daily volume %:"), 
         sg.InputText(str(config.get("MAX_VOLUME_PERCENTAGE", 0.05) * 100), key="MAX_VOLUME_PERCENTAGE_DISPLAY", size=(6,1)),
         sg.Text("%")]
    ])

    # Add probabilistic model checkbox
    probabilistic_option = sg.Checkbox("Use Probabilistic Predictions", default=config.get("USE_PROBABILISTIC", True), 
                                     key="USE_PROBABILISTIC", enable_events=True,
                                     tooltip="Enable probabilistic predictions with uncertainty estimates")

    # --- Define the Metrics Priorities frame --- 
    metrics_priorities_frame = sg.Frame("Metrics Priorities", [
        [sg.Text("Net Profit Weight:"), sg.Slider(range=(0, 10), default_value=config.get("priority_net_profit", 4.0), resolution=0.5, orientation='h', size=(15, 15), key="priority_net_profit")],
        [sg.Text("Win Rate Weight:"), sg.Slider(range=(0, 10), default_value=config.get("priority_win_rate", 3.0), resolution=0.5, orientation='h', size=(15, 15), key="priority_win_rate")],
        [sg.Text("Max Drawdown Weight:"), sg.Slider(range=(0, 10), default_value=config.get("priority_max_drawdown", 2.0), resolution=0.5, orientation='h', size=(15, 15), key="priority_max_drawdown")],
        [sg.Text("Profit Factor Weight:"), sg.Slider(range=(0, 10), default_value=config.get("priority_profit_factor", 1.0), resolution=0.5, orientation='h', size=(15, 15), key="priority_profit_factor")]
    ])

    # --- Define the Trading Strategy frame --- 
    trading_strategy_frame = sg.Frame("Trading Strategy", [
        [sg.Text("Trading Bucket:"), sg.Combo(["Scalping", "Short", "Medium", "Long"], 
                                             default_value=config.get("BUCKET", "Scalping"), 
                                             key="BUCKET", enable_events=True,
                                             tooltip="Select trading timeframe bucket")],
        [sg.Text("Initial Capital ($):"), sg.InputText(str(config.get("INITIAL_CAPITAL", 100000.0)), key="INITIAL_CAPITAL", size=(10,1))],
        [sg.Text("Max Trades at Once:"), sg.InputText(str(config.get("MAX_POSITION_HOLDINGS", 50)), key="MAX_POSITION_HOLDINGS", size=(5,1))],
        [look_back_frame],
        [checkpoint_frame]
    ])
    
    # --- Define the Bucket Goals frame (containing the TabGroup) ---
    bucket_goals_frame = sg.Frame("Bucket Goals", [[bucket_goals_tabgroup]])
    
    # --- Define the Prediction Settings frame --- 
    prediction_settings_frame = sg.Frame("Prediction Settings", [
        [probabilistic_option]
    ])

    # --- Define the two main columns ---
    left_column_layout = [
        [trading_strategy_frame],
        [position_limits_frame],
        [bucket_goals_frame],
        [prediction_settings_frame]
    ]
    
    right_column_layout = [
        [metrics_priorities_frame]
        # Add more frames here later if needed
    ]

    # --- Final strategy_layout combining the two columns ---
    strategy_layout = [
        [sg.Column(left_column_layout), sg.Column(right_column_layout)]
    ]
    
    return strategy_layout 

def create_advanced_tab(title="Model"):
    """Create the advanced model settings tab"""
    advanced_layout = [
        [sg.Frame("Model Architecture", [
            [sg.Text("Hidden Size:"), sg.InputText(str(config.get("HIDDEN_SIZE", 512)), key="HIDDEN_SIZE", size=(5,1), 
              tooltip="Size of hidden layers in neural network. Larger values = more capacity but slower training")],
            [sg.Checkbox("Use Fusion Architecture", default=config.get("USE_FUSION", True), key="USE_FUSION",
              tooltip="Enable attention mechanism fusion for better pattern recognition")]
        ])],
        [sg.Frame("Training Parameters", [
            [sg.Text("Learning Rate:"), sg.InputText(str(config.get("LEARNING_RATE", 0.0003)), key="LEARNING_RATE", size=(8,1),
              tooltip="Step size for optimization. Lower = more stable but slower learning")],
            [sg.Text("PPO Epochs:"), sg.InputText(str(config.get("PPO_EPOCHS", 4)), key="PPO_EPOCHS", size=(5,1),
              tooltip="Number of epochs to optimize policy per batch")],
            [sg.Text("Batch Size:"), sg.InputText(str(config.get("BATCH_SIZE", 128)), key="BATCH_SIZE", size=(5,1),
              tooltip="Number of samples per gradient update")],
            [sg.Text("Gamma (Discount):"), sg.InputText(str(config.get("GAMMA", 0.99)), key="GAMMA", size=(5,1),
              tooltip="Value between 0-1. Higher values prioritize long-term rewards")],
            [sg.Text("Epsilon Clip:"), sg.InputText(str(config.get("EPS_CLIP", 0.2)), key="EPS_CLIP", size=(5,1),
              tooltip="PPO clipping parameter to limit policy updates")],
            [sg.Text("Entropy Coefficient:"), sg.InputText(str(config.get("ENTROPY_COEF", 0.01)), key="ENTROPY_COEF", size=(5,1),
              tooltip="Encourages exploration. Higher = more random actions")]
        ])],
        [sg.Frame("Evolutionary Strategy", [
            [sg.Text("ES Interval:"), sg.InputText(str(config.get("ES_INTERVAL", 10)), key="ES_INTERVAL", size=(5,1),
              tooltip="Episodes between evolutionary updates")],
            [sg.Text("Population Size:"), sg.InputText(str(config.get("ES_POPULATION", 5)), key="ES_POPULATION", size=(5,1),
              tooltip="Number of agents in population")],
            [sg.Text("Mutation Rate:"), sg.InputText(str(config.get("ES_MUTATION_RATE", 0.1)), key="ES_MUTATION_RATE", size=(5,1),
              tooltip="Probability of mutating model parameters during evolution")]
        ])],
        [sg.Frame("Hardware Optimization", [
            [sg.Text("GPU Target Low:"), sg.InputText(str(config.get("GPU_TARGET_LOW", 0.65)), key="GPU_TARGET_LOW", size=(5,1),
              tooltip="Lower threshold for GPU utilization (0.0-1.0)")],
            [sg.Text("GPU Target High:"), sg.InputText(str(config.get("GPU_TARGET_HIGH", 0.85)), key="GPU_TARGET_HIGH", size=(5,1),
              tooltip="Upper threshold for GPU utilization (0.0-1.0)")],
            [sg.Text("Min Envs/Agent:"), sg.InputText(str(config.get("MIN_ENVS_PER_AGENT", 1)), key="MIN_ENVS_PER_AGENT", size=(5,1),
              tooltip="Minimum environments per agent")],
            [sg.Text("Max Envs/Agent:"), sg.InputText(str(config.get("MAX_ENVS_PER_AGENT", 4)), key="MAX_ENVS_PER_AGENT", size=(5,1),
              tooltip="Maximum environments per agent")]
        ])]
    ]
    
    return advanced_layout

def create_reward_tab(title="Reward"):
    """Create the reward settings tab"""
    reward_layout = [
        [sg.Frame("Performance Bonuses", [
            [sg.Text("Prediction Bonus:"), sg.InputText(str(config.get("PREDICTION_BONUS", 0.03)), key="PREDICTION_BONUS", size=(5,1),
              tooltip="Reward multiplier for correct price direction predictions")],
            [sg.Text("Novelty Bonus:"), sg.InputText(str(config.get("NOVELTY_BONUS_WEIGHT", 0.01)), key="NOVELTY_BONUS_WEIGHT", size=(5,1),
              tooltip="Reward for exploring new strategies")]
        ])],
        [sg.Frame("Risk Management", [
            [sg.Text("Grace Period (bars):"), sg.InputText(str(config.get("GRACE_PERIOD_BARS", 200)), key="GRACE_PERIOD_BARS", size=(5,1),
              tooltip="Bars before applying penalties")],
            [sg.Text("Penalty Interval:"), sg.InputText(str(config.get("PENALTY_INTERVAL", 2)), key="PENALTY_INTERVAL", size=(5,1),
              tooltip="How often to check for penalties")],
            [sg.Text("Base Penalty:"), sg.InputText(str(config.get("BASE_PENALTY", 0.05)), key="BASE_PENALTY", size=(5,1),
              tooltip="Initial penalty amount")],
            [sg.Text("Penalty Increment:"), sg.InputText(str(config.get("PENALTY_INCREMENT", 0.05)), key="PENALTY_INCREMENT", size=(5,1),
              tooltip="Additional penalty per interval")]
        ])],
        [sg.Frame("Early Stopping", [
            [sg.Text("Convergence Threshold:"), sg.InputText(str(config.get("CONVERGENCE_THRESHOLD", 0.001)), key="CONVERGENCE_THRESHOLD", size=(8,1),
              tooltip="Minimum improvement required to continue training")],
            [sg.Text("Patience (Episodes):"), sg.InputText(str(config.get("PATIENCE", 50)), key="PATIENCE", size=(5,1),
              tooltip="Episodes without improvement before stopping")]
        ])]
    ]
    
    return reward_layout 

def create_probabilistic_tab(title="Probabilistic"):
    """Create the probabilistic settings tab"""
    
    # Create list of horizon options
    horizons_text = ', '.join([str(h) for h in config.get("PREDICTION_HORIZONS", [12, 36, 72, 144])])
    
    # Create dynamic horizons frame
    dynamic_horizons_frame = [
        [sg.Checkbox("Use Dynamic Prediction Horizons", default=config.get("USE_DYNAMIC_HORIZONS", True), 
                   key="USE_DYNAMIC_HORIZONS", enable_events=True,
                   tooltip="Enable dynamic prediction horizons based on market conditions")],
        [sg.Text("Min Horizon:"), 
         sg.Spin([i for i in range(1, 101)], initial_value=config.get("MIN_HORIZON", 1), 
               key="MIN_HORIZON", size=(5, 1),
               tooltip="Minimum prediction horizon in bars",
               disabled=not config.get("USE_DYNAMIC_HORIZONS", True))],
        [sg.Text("Max Horizon:"), 
         sg.Spin([i for i in range(100, 1001)], initial_value=config.get("MAX_HORIZON", 576), 
               key="MAX_HORIZON", size=(5, 1),
               tooltip="Maximum prediction horizon in bars",
               disabled=not config.get("USE_DYNAMIC_HORIZONS", True))],
        [sg.Text("Horizon Density:"), 
         sg.Combo(["low", "medium", "high"], default_value=config.get("HORIZON_DENSITY", "medium"), 
                key="HORIZON_DENSITY",
                tooltip="Number of horizons to generate (low=4, medium=6, high=8)",
                disabled=not config.get("USE_DYNAMIC_HORIZONS", True))],
        [sg.Text("Update Frequency:"), 
         sg.Spin([i for i in range(1, 51)], initial_value=config.get("HORIZON_UPDATE_FREQ", 10), 
               key="HORIZON_UPDATE_FREQ", size=(5, 1),
               tooltip="Update horizons every N episodes",
               disabled=not config.get("USE_DYNAMIC_HORIZONS", True))]
    ]
    
    probabilistic_layout = [
        [sg.Frame("Prediction Horizons", [
            [sg.Text("Prediction Horizons (comma-separated list of bar counts):"), 
             sg.InputText(horizons_text, key="PREDICTION_HORIZONS_TEXT", size=(20,1),
                        tooltip="List of horizons to predict, e.g., 12, 36, 72, 144")],
            [sg.Text("Example: 12 = 1 hour at 5min bars, 144 = 12 hours, 288 = 1 day")]
        ])],
        [sg.Frame("Confidence Settings", [
            [sg.Text("Minimum Confidence Threshold:"), 
             sg.Slider(range=(0, 1), default_value=config.get("MIN_CONFIDENCE_THRESHOLD", 0.4), 
                      resolution=0.05, orientation='h', size=(20, 15), key="MIN_CONFIDENCE_THRESHOLD",
                      tooltip="Minimum confidence required to take a trade (0-1)")],
            [sg.Text("Maximum Uncertainty Threshold:"), 
             sg.Slider(range=(0, 1), default_value=config.get("MAX_UNCERTAINTY_THRESHOLD", 0.3), 
                      resolution=0.05, orientation='h', size=(20, 15), key="MAX_UNCERTAINTY_THRESHOLD",
                      tooltip="Maximum uncertainty allowed (coefficient of variation, 0-1)")],
            [sg.Text("Position Sizing Strategy:"), 
             sg.Combo(["fixed", "confidence_adjusted", "uncertainty_adjusted", "full_probabilistic"], 
                     default_value=config.get("POSITION_SIZING_STRATEGY", "confidence_adjusted"), 
                     key="POSITION_SIZING_STRATEGY", size=(20,1),
                     tooltip="Method to calculate position sizes")]
        ])],
        [sg.Frame("Uncertainty Management", [
            [sg.Text("Uncertainty Penalty Factor:"), 
             sg.Slider(range=(0, 1), default_value=config.get("UNCERTAINTY_PENALTY", 0.5), 
                      resolution=0.05, orientation='h', size=(20, 15), key="UNCERTAINTY_PENALTY",
                      tooltip="Higher values reduce position size more with uncertainty")],
            [sg.Text("Calibration Weight:"), 
             sg.Slider(range=(0, 1), default_value=config.get("CALIBRATION_WEIGHT", 0.3), 
                      resolution=0.05, orientation='h', size=(20, 15), key="CALIBRATION_WEIGHT",
                      tooltip="Weight for calibration in reward calculation")]
        ])],
        [sg.Frame("Visualization Options", [
            [sg.Checkbox("Show Confidence Intervals", default=config.get("SHOW_CONFIDENCE_INTERVALS", True), 
                        key="SHOW_CONFIDENCE_INTERVALS",
                        tooltip="Display confidence intervals on price predictions")],
            [sg.Checkbox("Show Calibration Plots", default=config.get("SHOW_CALIBRATION_PLOTS", True), 
                        key="SHOW_CALIBRATION_PLOTS",
                        tooltip="Generate calibration diagnostic plots")],
            [sg.Checkbox("Show Uncertainty Metrics", default=config.get("SHOW_UNCERTAINTY_METRICS", True), 
                        key="SHOW_UNCERTAINTY_METRICS",
                        tooltip="Display detailed uncertainty metrics")]
        ])],
        [sg.Frame("Live Prediction Interface", [
            [sg.Text("Live Prediction Dashboard", font=("Helvetica", 14, "bold"))],
            [sg.Text("Monitor real-time model predictions and confidence levels")],
            [sg.HorizontalSeparator()],
            [sg.Text("Prediction Status:"), sg.Text("Disabled", key="PREDICTION_STATUS", text_color="red")],
            [sg.Text("Last Prediction:"), sg.Text("N/A", key="LAST_PREDICTION")],
            [sg.Text("Confidence Level:"), sg.Text("N/A", key="PREDICTION_CONFIDENCE")],
            [sg.HorizontalSeparator()],
            [sg.Text("Note: Predictive agent runs automatically with main training", font=("Helvetica", 10, "italic"))]
        ])]
    ]
    
    return probabilistic_layout

def create_help_tab():
    """Create a help tab with comprehensive documentation for users"""
    
    help_layout = [
        [sg.TabGroup([[
            sg.Tab('Quick Start', [
                [sg.Multiline("""
# BTC-AI Trading System - Quick Start

## Overview
This AI-powered trading system uses reinforcement learning to develop trading strategies for Bitcoin markets. 
The system analyzes market data, predicts price movements, and executes trades based on its learned strategy.

## Getting Started
1. Choose a Trading Bucket (Scalping, Short, Medium, or Long)
2. Configure basic parameters in the Strategy tab
3. Press "Start Training" to begin

## Tips
- Start with default settings until you understand how the system works
- Monitor the Dashboard tab for real-time performance metrics
- Use Checkpoints to save promising models
- Tune parameters gradually, one at a time
                """, size=(60, 20), disabled=True)]
            ]),
            sg.Tab('Trading Buckets', [
                [sg.Multiline("""
# Trading Buckets Guide

The system supports four different trading timeframes, or "buckets":

## Scalping
- Very short-term trades (minutes to hours)
- Frequent entries/exits
- Aim for small, consistent profits
- Higher trading frequency
- Monthly profit targets: 15-30%

## Short
- Short-term trades (hours to days)
- More selective entries
- Moderate trading frequency
- Yearly profit targets: 100-200%

## Medium
- Medium-term trades (days to weeks)
- Focus on larger market moves
- Lower trading frequency
- Per-holding gain targets: 25-50%

## Long
- Long-term trades (weeks to months)
- Target significant market trends
- Lowest trading frequency
- Per-holding gain targets: 50-100%

Choose a bucket that matches your trading style or risk tolerance.
                """, size=(60, 20), disabled=True)]
            ]),
            sg.Tab('Parameters', [
                [sg.Multiline("""
# Parameter Reference

## Strategy Tab
- Trading Bucket: Timeframe strategy (Scalping, Short, Medium, Long)
- Initial Capital: Starting capital for the simulation
- Max Trades at Once: Maximum concurrent positions
- Look-Back Settings: How far back the AI analyzes data
- Position Size Limits: Controls maximum trade size

## Model Tab
- Hidden Size: Neural network complexity (larger = more capacity)
- Learning Rate: Speed of learning (smaller = more stable)
- Batch Size: Samples per learning update
- Gamma: Discount factor for future rewards (0-1)
- Epsilon Clip: Limits policy update size
- Entropy Coefficient: Encourages exploration

## Reward Tab
- Prediction Bonus: Extra reward for correct predictions
- Novelty Bonus: Reward for exploring new strategies
- Risk Management: Penalties for excessive risk
- Early Stopping: When to end training

## Probabilistic Tab
- Prediction Horizons: Future timepoints to predict
- Confidence Settings: Thresholds for taking trades
- Uncertainty Management: How to handle prediction uncertainty
                """, size=(60, 20), disabled=True)]
            ]),
            sg.Tab('Advanced Usage', [
                [sg.Multiline("""
# Advanced Usage Guide

## Checkpoint Management
- Create checkpoints regularly to save your progress
- Use the Checkpoints tab to manage saved models
- Compare checkpoint performance
- Resume training from promising checkpoints

## Parameter Presets
- Save successful parameter combinations as presets
- Load presets for different market conditions
- Use suggestion system to get performance-based recommendations

## Evolutionary Strategy
- Enables population-based training
- Mutation helps discover better strategies
- Configure in the Model tab
- Higher population sizes provide more diversity

## Probabilistic Models
- Enable uncertainty-aware predictions
- Use confidence/uncertainty thresholds to filter trades
- Adjust position sizing based on prediction confidence
- Generate visualizations to understand model behavior

## Recovery Dashboard
- Create emergency checkpoints before major changes
- Validate checkpoints regularly
- Use auto-checkpoints for safety
                """, size=(60, 20), disabled=True)]
            ]),
            sg.Tab('Troubleshooting', [
                [sg.Multiline("""
# Troubleshooting Guide

## Common Issues

### Training Crashes
- Reduce batch size
- Decrease model complexity (hidden size)
- Check GPU memory usage
- Ensure you have the latest dependencies

### Poor Performance
- Try different trading buckets
- Adjust learning rate
- Increase exploration (entropy coefficient)
- Review reward settings
- Start with default settings and modify gradually

### High GPU Usage
- Reduce population size
- Decrease environments per agent
- Lower GPU target settings

### Slow Training
- Reduce look-back period
- Decrease batch size
- Consider hardware upgrades
- Disable unnecessary visualizations

### Data Issues
- Ensure data sources are properly configured
- Check internet connection for live data
- Verify data file formats

### UI Responsiveness
- Close performance-intensive visualizations
- Use the monitoring tab sparingly
- Consider running on a dedicated machine
                """, size=(60, 20), disabled=True)]
            ]),
            sg.Tab('About', [
                [sg.Multiline("""
# BTC-AI Trading System

Version: 1.0.0

## Overview
This advanced trading system combines reinforcement learning with probabilistic modeling to develop optimal trading strategies for cryptocurrency markets.

## Key Features
- Multi-timeframe trading strategies
- Reinforcement learning with PPO algorithm
- Evolutionary strategy optimization
- Uncertainty-aware probabilistic predictions
- Automatic checkpoint management
- Performance analytics and visualization
- Natural language feedback integration

## Credits
Developed by the RL Trader team

## License
This software is proprietary. Unauthorized distribution is prohibited.

© 2025 RL Trader Team
                """, size=(60, 20), disabled=True)]
            ])
        ]], key='-HELP-TABS-', expand_x=True, expand_y=True)]
    ]
    
    return help_layout 

def create_monitoring_tab(title="Monitoring"):
    """Create the monitoring settings tab"""
    monitoring_layout = [
        [sg.Frame("Training Progress", [
            [sg.Multiline(size=(80, 20), key="-LOG-", autoscroll=True, disabled=True)]
        ])],
        [sg.Frame("Performance Metrics", [
            [sg.Multiline(size=(80, 10), key="-PERFORMANCE-", autoscroll=True, disabled=True)]
        ])],
        [sg.Button("Clear Log", key="CLEAR_LOG"),
         sg.Button("Save Log", key="SAVE_LOG"),
         sg.Button("Pop Out Log", key="POP_OUT_LOG"),
         sg.Button("Launch Advanced Monitor", key="DASHBOARD_MONITOR", button_color=('white', '#9932CC'))]
    ]
    
    return monitoring_layout

def create_natural_learning_tab(title="Natural Learning"):
    """Create the natural language learning tab"""
    # Natural feedback section
    feedback_frame = sg.Frame("Natural Language Feedback", [
        [sg.Text("Enable natural language feedback:"),
         sg.Checkbox("", default=config.get("ENABLE_NL_FEEDBACK", True), key="-ENABLE-NL-FEEDBACK-")],
        [sg.Text("Feedback frequency:"), 
         sg.InputText(str(config.get("FEEDBACK_FREQUENCY", 10)), key="-FEEDBACK-FREQUENCY-", size=(3,1)), 
         sg.Text("episodes")],
        [sg.Text("Feedback detail level:"), 
         sg.Combo(["Basic", "Detailed", "Advanced"], default_value=config.get("FEEDBACK_DETAIL", "Detailed"), 
                 key="-FEEDBACK-DETAIL-")]
    ])
    
    # Learning adaptation section
    adaptation_frame = sg.Frame("Learning Adaptation", [
        [sg.Text("Enable dynamic learning rate:"),
         sg.Checkbox("", default=config.get("ENABLE_DYNAMIC_LR", True), key="-ENABLE-DYNAMIC-LR-")],
        [sg.Text("Adaptation frequency:"), 
         sg.InputText(str(config.get("ADAPTATION_FREQUENCY", 5)), key="-ADAPTATION-FREQUENCY-", size=(3,1)), 
         sg.Text("episodes")],
        [sg.Text("Learning style:"), 
         sg.Combo(["Conservative", "Balanced", "Aggressive"], default_value=config.get("LEARNING_STYLE", "Balanced"), 
                 key="-LEARNING-STYLE-")]
    ])
    
    # Experience replay section
    replay_frame = sg.Frame("Experience Replay", [
        [sg.Text("Enable smart replay selection:"),
         sg.Checkbox("", default=config.get("ENABLE_SMART_REPLAY", True), key="-ENABLE-SMART-REPLAY-")],
        [sg.Text("Replay buffer size:"),
         sg.Spin([1000, 5000, 10000, 50000, 100000], initial_value=config.get("REPLAY_BUFFER_SIZE", 10000),
                 key="-REPLAY-BUFFER-SIZE-")],
        [sg.Text("Priority scale:"),
         sg.Slider(range=(0.0, 1.0), default_value=config.get("PRIORITY_SCALE", 0.6), resolution=0.1,
                  orientation="h", size=(20, 15), key="-PRIORITY-SCALE-")]
    ])
    
    # Combine all sections into the natural learning tab
    natural_learning_layout = [
        [sg.Column([
            [feedback_frame],
            [adaptation_frame],
            [replay_frame]
        ])]
    ]
    
    return natural_learning_layout

def create_checkpoint_management_tab(title="Checkpoints"):
    """Create the checkpoint management tab"""
    
    checkpoint_list_layout = [
        [sg.Table(
            values=[],
            headings=["Type", "Filename", "Episode", "Reward", "Size (MB)", "Modified"],
            auto_size_columns=False,
            col_widths=[10, 25, 8, 10, 10, 20],
            justification="center",
            num_rows=10,
            key="-CHECKPOINT-TABLE-",
            enable_events=True,
            select_mode=sg.TABLE_SELECT_MODE_BROWSE
        )],
        [sg.Text("Selected: "), sg.Text("", size=(50, 1), key="-CHECKPOINT-SELECTED-")],
        [sg.Button("Refresh", key="-REFRESH-CHECKPOINTS-"),
         sg.Button("Load", key="-LOAD-CHECKPOINT-", disabled=True),
         sg.Button("Resume Training", key="-RESUME-TRAINING-", disabled=True),
         sg.Button("View Metadata", key="-VIEW-METADATA-", disabled=True),
         sg.Button("Update Integrity", key="-UPDATE-CHECKPOINT-INTEGRITY-", disabled=True),
         sg.Button("Delete", key="-DELETE-CHECKPOINT-", disabled=True),
         sg.Button("Create Manual Checkpoint", key="-CREATE-CHECKPOINT-")],
        [sg.Text("Total Checkpoints: "), sg.Text("0", key="-TOTAL-CHECKPOINTS-"),
         sg.Text("   Disk Usage: "), sg.Text("0.00 MB", key="-DISK-USAGE-")]
    ]
    
    # Checkpoint information panel
    checkpoint_info_layout = [
        [sg.Multiline("Select a checkpoint to view details", size=(80, 10), key="-CHECKPOINT-INFO-", disabled=True)]
    ]
    
    # Main layout for checkpoint tab
    checkpoint_management_layout = [
        [sg.Text("Checkpoint Management", font=("Helvetica", 16))],
        [sg.Column(checkpoint_list_layout)],
        [sg.Column(checkpoint_info_layout)]
    ]
    
    return checkpoint_management_layout 

def create_withdrawal_tab(title="Withdrawal Management"):
    """Create the withdrawal management tab"""
    withdrawal_layout = [
        [sg.Frame("Withdrawal Settings", [
            [sg.Text("Profit Reserve Ratio:"), 
             sg.Slider(range=(0, 0.9), default_value=config.get("profit_reserve_ratio", 0.3), 
                      resolution=0.05, orientation='h', size=(20, 15), key="profit_reserve_ratio",
                      tooltip="Percentage of profits reserved for withdrawals (0-90%)")],
            [sg.Text("Deposit Conversion Fee:"), 
             sg.InputText(str(config.get("deposit_conversion_fee", 0.001)), key="deposit_conversion_fee", size=(5,1),
                         tooltip="Fee for USD->USDT conversion")],
            [sg.Text("Withdrawal Conversion Fee:"), 
             sg.InputText(str(config.get("withdrawal_conversion_fee", 0.001)), key="withdrawal_conversion_fee", size=(5,1),
                         tooltip="Fee for USDT->USD conversion")]
        ])],
        [sg.Frame("Withdrawal Simulation", [
            [sg.Checkbox("Simulate Withdrawals During Training", default=config.get("simulate_withdrawals", True), 
                       key="simulate_withdrawals", enable_events=True,
                       tooltip="Enable realistic withdrawal simulation during training")],
            [sg.Column([
                [sg.Text("Monthly Withdrawal Chance:"), 
                 sg.Slider(range=(0, 1), default_value=config.get("monthly_withdrawal_chance", 0.3), 
                          resolution=0.05, orientation='h', size=(15, 15), key="monthly_withdrawal_chance",
                          tooltip="Probability of regular monthly withdrawal")],
                [sg.Text("Emergency Withdrawal Chance:"), 
                 sg.Slider(range=(0, 0.5), default_value=config.get("emergency_withdrawal_chance", 0.05), 
                          resolution=0.01, orientation='h', size=(15, 15), key="emergency_withdrawal_chance",
                          tooltip="Probability of emergency withdrawal")]
            ], pad=((20, 0), (0, 0))),
            sg.Column([
                [sg.Text("Withdrawal Min %:"), 
                 sg.Slider(range=(0.01, 0.3), default_value=config.get("withdrawal_min_pct", 0.05), 
                          resolution=0.01, orientation='h', size=(15, 15), key="withdrawal_min_pct",
                          tooltip="Minimum withdrawal amount (% of capital)")],
                [sg.Text("Withdrawal Max %:"), 
                 sg.Slider(range=(0.1, 0.9), default_value=config.get("withdrawal_max_pct", 0.3), 
                          resolution=0.05, orientation='h', size=(15, 15), key="withdrawal_max_pct",
                          tooltip="Maximum withdrawal amount (% of capital)")]
            ], pad=((20, 0), (0, 0)))]
        ])],
        [sg.Frame("Deposit Simulation", [
            [sg.Checkbox("Simulate Deposits During Training", default=config.get("simulate_deposits", True), 
                         key="simulate_deposits", enable_events=True)],
            [sg.Text("Monthly Deposit Chance:"), 
             sg.Slider(range=(0, 1), default_value=config.get("monthly_deposit_chance", 0.4), 
                      resolution=0.05, orientation='h', size=(20, 15), key="monthly_deposit_chance")],
            [sg.Text("Deposit Min %:"), 
             sg.InputText(str(config.get("deposit_min_pct", 0.05) * 100), key="deposit_min_pct_display", size=(5,1)),
             sg.Text("%")],
            [sg.Text("Deposit Max %:"), 
             sg.InputText(str(config.get("deposit_max_pct", 0.5) * 100), key="deposit_max_pct_display", size=(5,1)),
             sg.Text("%")]
        ])],
        [sg.Frame("Current Withdrawal Status", [
            [sg.Text("Capital: $"), sg.Text("0.00", key="-CAPITAL-")],
            [sg.Text("Reserved for Withdrawal: $"), sg.Text("0.00", key="-WITHDRAWAL-RESERVE-")],
            [sg.Text("Available for Withdrawal: $"), sg.Text("0.00", key="-AVAILABLE-WITHDRAWAL-")],
            [sg.Button("Withdraw Funds", key="WITHDRAW_FUNDS"),
             sg.Button("Deposit Funds", key="DEPOSIT_FUNDS"),
             sg.Button("Refresh", key="REFRESH_WITHDRAWAL")]
        ])]
    ]
    
    return withdrawal_layout

def create_presets_tab(title="Parameter Presets"):
    """Create the Parameter Presets tab UI"""
    
    # Try to import preset_manager if available
    try:
        from src.ui.preset_handlers import handle_preset_tab_events
        preset_system_available = True
    except ImportError:
        preset_system_available = False
    
    if preset_system_available:
        try:
            # Use enhanced presets tab from preset_manager
            from src.ui.preset_manager import create_presets_tab as pm_create_presets_tab
            return pm_create_presets_tab()
        except ImportError:
            # Fall back to the original implementation
            pass
    
    # Default implementation
    current_bucket = config.get("BUCKET", "Scalping")
    
    preset_layout = [
        [sg.Text("Parameter Presets", font=("Helvetica", 16))],
        [sg.Text("This tab allows you to save, load, and manage parameter presets.")],
        [sg.Frame("Built-in Defaults", [
            [sg.Text("Load built-in defaults for bucket types:")],
            [sg.Button("Load Scalping Defaults", key="LOAD_DEFAULT_SCALPING"), 
             sg.Button("Load Short Defaults", key="LOAD_DEFAULT_SHORT")],
            [sg.Button("Load Medium Defaults", key="LOAD_DEFAULT_MEDIUM"), 
             sg.Button("Load Long Defaults", key="LOAD_DEFAULT_LONG")]
        ])],
        [sg.Frame("Custom Presets", [
            [sg.Column([
                [sg.Text("Save current configuration as a custom preset:")],
                [sg.Text("Preset Name:"), sg.InputText(key="-PRESET-NAME-", size=(30, 1))],
                [sg.Text("Description:"), sg.InputText(key="-PRESET-DESC-", size=(30, 1))],
                [sg.Button("Save Custom Preset", key="-SAVE-PRESET-")]
            ]), sg.VerticalSeparator(), sg.Column([
                [sg.Text("Load saved custom presets:")],
                [sg.Listbox(values=[], key="-PRESET-LIST-", size=(40, 6), enable_events=True)],
                [sg.Button("Load Selected", key="-LOAD-PRESET-", disabled=True),
                 sg.Button("Delete", key="-DELETE-PRESET-", disabled=True), 
                 sg.Button("Refresh", key="-REFRESH-PRESETS-")]
            ])]
        ])],
        [sg.Frame("Preset Suggestions", [
            [sg.Text("Performance-based suggestions for the current bucket type:")],
            [sg.Text("Prioritize:"), 
             sg.Radio("Best Overall", "SUGGESTION_TYPE", key="-SUGGESTION-TYPE-OVERALL-", default=True, enable_events=True),
             sg.Radio("Highest Profit", "SUGGESTION_TYPE", key="-SUGGESTION-TYPE-PROFIT-", enable_events=True),
             sg.Radio("Lowest Risk", "SUGGESTION_TYPE", key="-SUGGESTION-TYPE-RISK-", enable_events=True)],
            [sg.Listbox(values=[], key="-SUGGESTIONS-LIST-", size=(70, 4), enable_events=True)],
            [sg.Text("Suggestions are optional. Select one and click 'Load Suggestion' to apply.",
                    font=("Helvetica", 8), text_color="gray")],
            [sg.Button("Load Suggestion", key="-LOAD-SUGGESTION-", disabled=True),
             sg.Button("Clear Selection", key="-CLEAR-SUGGESTION-"),
             sg.Button("Refresh Suggestions", key="-REFRESH-SUGGESTIONS-")]
        ])]
    ]
    
    return preset_layout

def create_recovery_dashboard_tab(title="Recovery Dashboard"):
    """Create the recovery dashboard tab."""
    
    # Create a frame for emergency checkpoints
    emergency_checkpoints_frame = sg.Frame("Emergency Checkpoints", [
        [sg.Text("Manage application recovery checkpoints for disaster recovery")],
        [sg.Table(
            values=[],
            headings=["ID", "Date", "Time", "Note", "Status"],
            auto_size_columns=False,
            col_widths=[10, 10, 10, 30, 10],
            display_row_numbers=False,
            justification="left",
            num_rows=10,
            key="-RECOVERY-TABLE-",
            enable_events=True,
            tooltip="Select a checkpoint to view details or restore"
        )],
        [sg.Button("Refresh", key="-REFRESH-RECOVERY-"),
         sg.Button("Create Checkpoint", key="-CREATE-CHECKPOINT-"),
         sg.Button("View Details", key="-VIEW-CHECKPOINT-", disabled=True),
         sg.Button("Restore Checkpoint", key="-RESTORE-CHECKPOINT-", disabled=True),
         sg.Button("Delete Checkpoint", key="-DELETE-CHECKPOINT-", disabled=True)]
    ])
    
    # Create a frame for checkpoint details
    checkpoint_details_frame = sg.Frame("Checkpoint Details", [
        [sg.Text("Checkpoint ID:", size=(15, 1)), sg.Text("", key="-CHECKPOINT-ID-", size=(30, 1))],
        [sg.Text("Created:", size=(15, 1)), sg.Text("", key="-CHECKPOINT-CREATED-", size=(30, 1))],
        [sg.Text("Note:", size=(15, 1)), sg.Text("", key="-CHECKPOINT-NOTE-", size=(30, 1))],
        [sg.Text("Content:", size=(15, 1))],
        [sg.Multiline(
            "",
            key="-CHECKPOINT-CONTENT-",
            size=(60, 10),
            disabled=True,
            autoscroll=True
        )],
        [sg.Text("Included Files:", size=(15, 1))],
        [sg.Listbox(
            values=[],
            key="-CHECKPOINT-FILES-",
            size=(60, 5),
            enable_events=False
        )]
    ])
    
    # Create a frame for recovery operations
    recovery_operations_frame = sg.Frame("Recovery Operations", [
        [sg.Text("Auto-save Checkpoint Interval (minutes):"),
         sg.Slider(
             range=(5, 60),
             default_value=15,
             resolution=5,
             orientation="horizontal",
             size=(20, 15),
             key="-AUTO-CHECKPOINT-INTERVAL-"
         ),
         sg.Checkbox("Enable Auto Checkpoints", key="-ENABLE-AUTO-CHECKPOINTS-", default=True)],
        [sg.Text("Critical Operations:")],
        [sg.Checkbox("Create checkpoint before training", key="-CHECKPOINT-BEFORE-TRAINING-", default=True),
         sg.Checkbox("Create checkpoint before updating", key="-CHECKPOINT-BEFORE-UPDATE-", default=True)],
        [sg.Button("Validate All Checkpoints", key="-VALIDATE-CHECKPOINTS-"),
         sg.Button("Cleanup Old Checkpoints", key="-CLEANUP-CHECKPOINTS-")]
    ])
    
    # Arrange the layout with all frames
    recovery_layout = [
        [sg.Text("Application Recovery Dashboard", font=("Helvetica", 16))],
        [sg.Text("This dashboard helps you manage application recovery in case of crashes or data corruption.")],
        [emergency_checkpoints_frame],
        [sg.Column([[checkpoint_details_frame]], vertical_alignment='top'),
         sg.VSeparator(),
         sg.Column([[recovery_operations_frame]], vertical_alignment='top')],
        [sg.Text("Note: Creating regular checkpoints is recommended before making significant changes.", font=("Helvetica", 10, "italic"))]
    ]
    
    return recovery_layout

def create_tabs():
    """Create all tabs for the GUI"""
    return [
        [sg.Tab("Dashboard", create_dashboard_tab(), key="-TAB-DASHBOARD-")],
        [sg.Tab("Strategy", create_main_tab(), key="-TAB-STRATEGY-")],
        [sg.Tab("Model", create_advanced_tab(), key="-TAB-MODEL-")],
        [sg.Tab("Reward", create_reward_tab(), key="-TAB-REWARD-")],
        [sg.Tab("Probabilistic", create_probabilistic_tab(), key="-TAB-PROBABILISTIC-")],
        [sg.Tab("Monitoring", create_monitoring_tab(), key="-TAB-MONITORING-")],
        [sg.Tab("Natural Learning", create_natural_learning_tab(), key="-TAB-NATURAL-LEARNING-")],
        [sg.Tab("Checkpoints", create_checkpoint_management_tab(), key="-TAB-CHECKPOINTS-")],
        [sg.Tab("Presets", create_presets_tab(), key="-TAB-PRESETS-")],
        [sg.Tab("Withdrawal", create_withdrawal_tab(), key="-TAB-WITHDRAWAL-")],
        [sg.Tab("Recovery", create_recovery_dashboard_tab(), key="-TAB-RECOVERY-")],
        [sg.Tab("Help", create_help_tab(), key="-TAB-HELP-")]
    ] 