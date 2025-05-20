# Parameter Presets System

This module provides a complete parameter presets management system for the trading application, including preset storage, performance tracking, and suggestions based on historical performance.

## Files

- `preset_manager.py` - Core preset management functionality
- `preset_handlers.py` - Event handlers for the UI
- `backtesting_integration.py` - Integration with backtesting module

## Features

### Preset Management

- Store and load default and custom presets
- Categorize presets by bucket type (Scalping, Short, Medium, Long)
- Save current settings as a new preset
- Temporary presets that auto-cleanup after 7 days

### Performance Tracking

- Automatic tracking of backtest results for each preset
- Maintains performance history with timestamps
- View historical performance for each preset
- Automatic integration with training sessions

### Performance-Based Suggestions

- Get preset suggestions based on historical performance
- Filter suggestions by best overall, highest profit, or lowest risk
- Automatically updates suggestions when new performance data is available
- Quick-load suggested presets with a single click

## Integration Status

The preset system is now fully integrated with the main application. All the necessary connections have been implemented:

1. ✅ Initialization in main.py
2. ✅ Event handling in the main event loop
3. ✅ Bucket change handling
4. ✅ Training integration
5. ✅ Suggestion filtering
6. ✅ Performance history display

## User Guide

### Saving Presets

1. Configure your parameters in the main application
2. Click "Save Settings" to save the current state
3. When prompted, choose "Yes" to save as a preset
4. Enter a name and description for your preset
5. Click "Save Custom Preset"

### Loading Presets

1. Navigate to the "Parameter Presets" tab
2. Select a preset from the list or from suggestions
3. Click "Load Selected" or "Load Suggestion"
4. The parameters will be applied to your application

### Using Preset Suggestions

1. Go to the "Parameter Presets" tab
2. In the "Preset Suggestions" section, you'll see suggestions based on performance data
3. Choose the filter type:
   - **Best Overall**: Balanced performance across all metrics
   - **Highest Profit**: Optimized for maximum profit
   - **Lowest Risk**: Optimized for minimum drawdown
4. Select a suggestion from the list
5. Click "Load Suggestion" to apply it to your application

### Viewing Performance History

1. Select a preset from the list
2. Click "Quick View Performance" for a brief summary
3. For detailed history, click "View Full History" in the summary window
4. The history shows all recorded performance metrics with timestamps

### Using Temporary Presets

Training sessions automatically create temporary presets to track performance. These are cleaned up after 7 days unless:

1. Select a temporary preset from the list
2. Click "Keep Selected Preset" to make it permanent
3. The preset will then be saved to your permanent presets collection

## Directory Structure

The preset system creates the following directory structure:

```
presets/
├── defaults/          # Built-in default presets
│   ├── Scalping/
│   ├── Short/
│   ├── Medium/
│   └── Long/
├── user/              # User's custom presets
│   ├── Scalping/
│   ├── Short/
│   ├── Medium/
│   └── Long/
├── temp/              # Temporary presets (auto-cleanup)
│   ├── Scalping/
│   ├── Short/
│   ├── Medium/
│   └── Long/
└── performance_history.json  # Performance tracking data 
```

## Implementation Details

### Preset Filtering Logic

The preset suggestions are filtered using the following criteria:

- **Overall Score**: A balanced score combining profit, win rate, and drawdown
- **Profit Score**: Calculated as net_profit * (1 + win_rate), prioritizing high profit with good win rates
- **Risk Score**: Calculated as max_drawdown * (2 - sharpe_ratio), prioritizing low drawdown with good Sharpe ratios

The filtering system automatically sorts presets according to the selected criteria and presents the top 5 options to the user.

## Getting Started

To integrate the preset system with your application, follow these steps:

### Step 1: Import required modules in main.py

Add these imports at the top of your main.py file:

```python
# Import preset modules
try:
    from src.ui.preset_manager import create_presets_tab
    from src.ui.preset_handlers import handle_preset_tab_events, initialize_preset_handlers
    from src.ui.backtesting_integration import initialize_backtesting_integration
    preset_system_available = True
except ImportError:
    preset_system_available = False
    logger.warning("Preset system not available. Some features will be disabled.")
```

### Step 2: Add the presets tab to the layout

Find the `create_layout()` function in main.py and add the presets tab:

```python
def create_layout():
    # ... existing code ...
    
    # Create TabGroup with all tabs
    tab_group = [
        sg.TabGroup([
            [sg.Tab('Main', create_main_tab())],
            [sg.Tab('Training', create_training_tab())],
            [sg.Tab('Visualization', create_visualization_tab())],
            # Add the presets tab if available
            [sg.Tab('Parameter Presets', create_presets_tab())] if preset_system_available else [],
            # ... other tabs ...
        ], key='-TABGROUP-', enable_events=True)
    ]
    
    # ... rest of the function ...
```

### Step 3: Initialize the preset system

Add initialization in your main function:

```python
def main():
    # ... existing code ...
    
    # Initialize preset system if available
    if preset_system_available:
        initialize_preset_handlers()
        initialize_backtesting_integration()
    
    # ... rest of the function ...
```

### Step 4: Handle preset-related events

Add preset event handling to your main event loop:

```python
def main():
    # ... existing code ...
    
    # Main event loop
    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, "Exit"):
            break
            
        # ... existing event handling ...
        
        # Handle preset events if available
        if preset_system_available and handle_preset_tab_events(window, event, values):
            continue
        
        # ... rest of the event loop ...
```

### Step 5: Add Backtest Completion Hooks

Modify your backtest and comparison functions to use the integrated versions:

```python
# Replace direct calls to backtesting functions with these:
from src.ui.backtesting_integration import run_backtest_with_preset_tracking, run_preset_comparison_with_tracking

# Instead of:
# results = run_backtest(df, agent, config)
# Use:
results = run_backtest_with_preset_tracking(df, agent, config)

# Instead of:
# results = run_preset_comparison(df, preset_config, user_config)
# Use:
results = run_preset_comparison_with_tracking(df, preset_config, user_config)
```