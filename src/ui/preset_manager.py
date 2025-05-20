"""
Preset Manager Module

This module manages saved preset configurations for different trading buckets (Scalping, Short, Medium, Long).
It provides functionality to create, load, save, and manage presets with their performance history.

Key features:
- Default presets for each bucket type
- User-created presets
- Temporary presets with auto-cleanup
- Performance tracking and comparison
- Preset suggestions based on historical performance

Fallback System Role:
This module serves as a critical component of the BTC-AI fallback system. If the main configuration 
system (trade_config) fails to load, the backtesting and training modules will attempt to load default presets
directly from this module. This ensures that even if the main configuration is unavailable,
the system can continue to function with sensible default values.
"""

import os
import json
import logging
import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import importlib
import PySimpleGUI as sg

# Try to import from src.utils
try:
    from src.utils.utils import log as _log
    from src.utils.log_manager import LogManager
    logger = LogManager.get_logger("preset_manager")
except ImportError:
    # Fallback if imports fail
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("preset_manager")
    def _log(message, level="INFO"):
        logger.info(message)

# Get paths from the centralized paths module
try:
    from src.utils.paths import get_common_paths, get_project_root, add_project_to_path
    
    # Ensure project root is in path for imports
    add_project_to_path()
    
    # Get common paths
    paths = get_common_paths()
    
    # Constants for preset management using standardized paths
    PRESET_DIR = paths["presets"]
    DEFAULT_PRESETS_DIR = os.path.join(PRESET_DIR, "defaults")
    USER_PRESETS_DIR = os.path.join(PRESET_DIR, "user")
    TEMP_PRESETS_DIR = os.path.join(PRESET_DIR, "temp")
    PRESETS_PERFORMANCE_FILE = os.path.join(PRESET_DIR, "performance_history.json")
    
    logger.info(f"Using preset directories: {PRESET_DIR}")
except Exception as e:
    # Fallback if paths module fails
    logger.error(f"Error using paths module: {e}")
    logger.warning("Falling back to direct path determination")
    
    # Use direct path determination as fallback
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    
    # Ensure src is in the path
    src_path = os.path.join(project_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    # Set up preset directories
    PRESET_DIR = os.path.join(project_root, "presets")
    DEFAULT_PRESETS_DIR = os.path.join(PRESET_DIR, "defaults")
    USER_PRESETS_DIR = os.path.join(PRESET_DIR, "user")
    TEMP_PRESETS_DIR = os.path.join(PRESET_DIR, "temp")
    PRESETS_PERFORMANCE_FILE = os.path.join(PRESET_DIR, "performance_history.json")

# Default presets by bucket type
DEFAULT_PRESETS = {
    "Scalping": {
        "Aggressive": {
            "description": "Higher risk, higher reward settings for quick trades",
            "params": {
                "BUCKET": "Scalping",
                "monthly_target_min": 25.0,
                "monthly_target_max": 40.0,
                "use_advanced_features": True,
                "risk_tolerance": "high"
            }
        },
        "Conservative": {
            "description": "Lower risk settings with consistent results",
            "params": {
                "BUCKET": "Scalping",
                "monthly_target_min": 10.0,
                "monthly_target_max": 20.0,
                "use_advanced_features": False,
                "risk_tolerance": "low"
            }
        }
    },
    "Short": {
        "Balanced": {
            "description": "Balanced approach for short-term trading",
            "params": {
                "BUCKET": "Short",
                "yearly_target_min": 120.0,
                "yearly_target_max": 180.0,
                "use_advanced_features": True,
                "risk_tolerance": "medium"
            }
        }
    },
    "Medium": {
        "Growth": {
            "description": "Settings optimized for consistent medium-term growth",
            "params": {
                "BUCKET": "Medium",
                "min_gain_per_holding_medium": 30.0,
                "max_gain_per_holding_medium": 60.0,
                "bonus_multiplier_medium": 1.2,
                "use_advanced_features": True
            }
        }
    },
    "Long": {
        "Stability": {
            "description": "Long-term stability with moderate growth",
            "params": {
                "BUCKET": "Long",
                "min_gain_per_holding_long": 40.0,
                "max_gain_per_holding_long": 80.0,
                "bonus_multiplier_long": 1.3,
                "use_advanced_features": True
            }
        }
    }
}

def ensure_preset_directories():
    """Ensure all preset directories exist"""
    for directory in [PRESET_DIR, DEFAULT_PRESETS_DIR, USER_PRESETS_DIR, TEMP_PRESETS_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created preset directory: {directory}")

def initialize_default_presets():
    """Initialize default presets if they don't exist"""
    ensure_preset_directories()
    
    # Check if default presets exist
    for bucket, presets in DEFAULT_PRESETS.items():
        bucket_dir = os.path.join(DEFAULT_PRESETS_DIR, bucket)
        if not os.path.exists(bucket_dir):
            os.makedirs(bucket_dir)
        
        for preset_name, preset_data in presets.items():
            preset_file = os.path.join(bucket_dir, f"{preset_name}.json")
            if not os.path.exists(preset_file):
                with open(preset_file, 'w') as f:
                    json.dump(preset_data, f, indent=4)
                logger.info(f"Created default preset: {preset_file}")

def load_preset(preset_id: str) -> Dict[str, Any]:
    """
    Load a preset by ID
    
    Args:
        preset_id: The ID of the preset (path/filename)
        
    Returns:
        Dict containing the preset data
    """
    if not os.path.exists(preset_id):
        logger.error(f"Preset file not found: {preset_id}")
        return {}
    
    try:
        with open(preset_id, 'r') as f:
            preset_data = json.load(f)
        return preset_data
    except Exception as e:
        logger.error(f"Error loading preset {preset_id}: {e}")
        return {}

def save_preset(bucket: str, name: str, params: Dict[str, Any], 
                description: str = "", is_temporary: bool = False) -> str:
    """
    Save a preset
    
    Args:
        bucket: The bucket type (Scalping, Short, Medium, Long)
        name: Name of the preset
        params: Dictionary of parameters
        description: Optional description
        is_temporary: Whether this is a temporary preset
        
    Returns:
        Path to the saved preset file
    """
    ensure_preset_directories()
    
    # Determine where to save the preset
    if is_temporary:
        target_dir = os.path.join(TEMP_PRESETS_DIR, bucket)
    else:
        target_dir = os.path.join(USER_PRESETS_DIR, bucket)
    
    # Ensure target directory exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # Create preset data
    preset_data = {
        "name": name,
        "description": description,
        "params": params,
        "created": datetime.datetime.now().isoformat(),
        "modified": datetime.datetime.now().isoformat()
    }
    
    # Save to file
    filename = f"{name.replace(' ', '_')}.json"
    filepath = os.path.join(target_dir, filename)
    
    try:
        with open(filepath, 'w') as f:
            json.dump(preset_data, f, indent=4)
        logger.info(f"Saved preset to {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Error saving preset: {e}")
        return ""

def list_presets(bucket: str = None, include_defaults: bool = True, 
                include_user: bool = True, include_temp: bool = True) -> List[Dict[str, Any]]:
    """
    List available presets
    
    Args:
        bucket: Optional bucket filter
        include_defaults: Whether to include default presets
        include_user: Whether to include user presets
        include_temp: Whether to include temporary presets
        
    Returns:
        List of preset metadata dictionaries
    """
    ensure_preset_directories()
    presets = []
    
    dirs_to_check = []
    if include_defaults:
        dirs_to_check.append(DEFAULT_PRESETS_DIR)
    if include_user:
        dirs_to_check.append(USER_PRESETS_DIR)
    if include_temp:
        dirs_to_check.append(TEMP_PRESETS_DIR)
    
    for base_dir in dirs_to_check:
        # If bucket specified, only check that bucket's directory
        if bucket:
            bucket_dir = os.path.join(base_dir, bucket)
            if os.path.exists(bucket_dir):
                preset_files = [os.path.join(bucket_dir, f) for f in os.listdir(bucket_dir) 
                              if f.endswith('.json')]
                
                for preset_file in preset_files:
                    try:
                        preset_data = load_preset(preset_file)
                        preset_type = "default" if base_dir == DEFAULT_PRESETS_DIR else \
                                    "user" if base_dir == USER_PRESETS_DIR else "temp"
                        
                        presets.append({
                            "id": preset_file,
                            "name": preset_data.get("name", os.path.basename(preset_file).replace('.json', '')),
                            "description": preset_data.get("description", ""),
                            "bucket": bucket,
                            "type": preset_type,
                            "created": preset_data.get("created", ""),
                            "modified": preset_data.get("modified", "")
                        })
                    except Exception as e:
                        logger.error(f"Error processing preset {preset_file}: {e}")
        else:
            # Check all bucket directories
            if os.path.exists(base_dir):
                for bucket_name in os.listdir(base_dir):
                    bucket_dir = os.path.join(base_dir, bucket_name)
                    if os.path.isdir(bucket_dir):
                        preset_files = [os.path.join(bucket_dir, f) for f in os.listdir(bucket_dir) 
                                      if f.endswith('.json')]
                        
                        for preset_file in preset_files:
                            try:
                                preset_data = load_preset(preset_file)
                                preset_type = "default" if base_dir == DEFAULT_PRESETS_DIR else \
                                            "user" if base_dir == USER_PRESETS_DIR else "temp"
                                
                                presets.append({
                                    "id": preset_file,
                                    "name": preset_data.get("name", os.path.basename(preset_file).replace('.json', '')),
                                    "description": preset_data.get("description", ""),
                                    "bucket": bucket_name,
                                    "type": preset_type,
                                    "created": preset_data.get("created", ""),
                                    "modified": preset_data.get("modified", "")
                                })
                            except Exception as e:
                                logger.error(f"Error processing preset {preset_file}: {e}")
    
    return presets

def delete_preset(preset_id: str) -> bool:
    """
    Delete a preset
    
    Args:
        preset_id: ID of the preset to delete
        
    Returns:
        True if successful, False otherwise
    """
    if not os.path.exists(preset_id):
        logger.error(f"Preset file not found: {preset_id}")
        return False
    
    # Don't allow deletion of default presets
    if DEFAULT_PRESETS_DIR in preset_id:
        logger.error("Cannot delete default presets")
        return False
    
    try:
        os.remove(preset_id)
        logger.info(f"Deleted preset: {preset_id}")
        return True
    except Exception as e:
        logger.error(f"Error deleting preset {preset_id}: {e}")
        return False

def cleanup_temp_presets(days_old: int = 7) -> int:
    """
    Clean up temporary presets older than specified days
    
    Args:
        days_old: Age in days to delete
        
    Returns:
        Number of presets deleted
    """
    ensure_preset_directories()
    count = 0
    
    # Get the cutoff date
    cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days_old)
    
    # Get all temp presets
    temp_presets = list_presets(include_defaults=False, include_user=False, include_temp=True)
    
    for preset in temp_presets:
        try:
            # Parse the creation date
            if preset.get("created"):
                created_date = datetime.datetime.fromisoformat(preset["created"])
                if created_date < cutoff_date:
                    if delete_preset(preset["id"]):
                        count += 1
        except Exception as e:
            logger.error(f"Error processing preset during cleanup: {e}")
    
    return count

# Performance tracking functions
def load_performance_history() -> Dict[str, List[Dict[str, Any]]]:
    """
    Load performance history for all presets
    
    Returns:
        Dictionary mapping preset IDs to lists of performance records
    """
    if not os.path.exists(PRESETS_PERFORMANCE_FILE):
        return {}
    
    try:
        with open(PRESETS_PERFORMANCE_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading performance history: {e}")
        return {}

def update_preset_performance(preset_id: str, metrics: Dict[str, Any]) -> bool:
    """
    Update performance metrics for a preset
    
    Args:
        preset_id: ID of the preset
        metrics: Dictionary of performance metrics
        
    Returns:
        True if successful, False otherwise
    """
    ensure_preset_directories()
    
    # Get existing performance history
    performance_history = load_performance_history()
    
    # If this preset doesn't exist in history yet, create an entry
    if preset_id not in performance_history:
        performance_history[preset_id] = []
    
    # Add new performance record
    performance_record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "metrics": metrics
    }
    
    performance_history[preset_id].append(performance_record)
    
    # Limit to last 20 records per preset to prevent file growth
    if len(performance_history[preset_id]) > 20:
        performance_history[preset_id] = performance_history[preset_id][-20:]
    
    # Save updated history
    try:
        with open(PRESETS_PERFORMANCE_FILE, 'w') as f:
            json.dump(performance_history, f, indent=4)
        return True
    except Exception as e:
        logger.error(f"Error saving performance history: {e}")
        return False

def get_preset_suggestions(bucket: str, metric_filter: str = "overall") -> List[Dict[str, Any]]:
    """
    Get preset suggestions based on historical performance
    
    Args:
        bucket: The bucket type to filter by
        metric_filter: Filter by 'profit', 'risk', or 'overall'
        
    Returns:
        List of preset suggestions with performance metrics
    """
    # Load all performance history
    performance_history = load_performance_history()
    
    # Get all presets for this bucket
    bucket_presets = list_presets(bucket=bucket)
    
    suggestions = []
    
    for preset in bucket_presets:
        preset_id = preset["id"]
        preset_history = performance_history.get(preset_id, [])
        
        if not preset_history:
            continue
        
        # Calculate average metrics from history
        avg_metrics = calculate_average_metrics(preset_history)
        
        # Determine score based on filter type
        if metric_filter == "profit":
            score = avg_metrics.get("profit_score", 0)
        elif metric_filter == "risk":
            score = avg_metrics.get("risk_score", 0)
        else:  # overall
            score = avg_metrics.get("overall_score", 0)
        
        suggestions.append({
            "preset": preset,
            "metrics": avg_metrics,
            "score": score
        })
    
    # Sort by score (descending)
    suggestions.sort(key=lambda x: x["score"], reverse=True)
    
    return suggestions

def calculate_average_metrics(performance_history: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate average metrics from performance history
    
    Args:
        performance_history: List of performance records
        
    Returns:
        Dictionary of average metrics
    """
    if not performance_history:
        return {}
    
    # Initialize metric counters
    sum_metrics = {
        "net_profit": 0.0,
        "win_rate": 0.0,
        "max_drawdown": 0.0,
        "sharpe_ratio": 0.0,
        "profit_factor": 0.0
    }
    
    # Sum up metrics from history
    valid_records = 0
    for record in performance_history:
        metrics = record.get("metrics", {})
        if metrics:
            for key in sum_metrics:
                if key in metrics:
                    sum_metrics[key] += float(metrics[key])
            valid_records += 1
    
    # Calculate averages
    avg_metrics = {}
    if valid_records > 0:
        for key in sum_metrics:
            avg_metrics[key] = sum_metrics[key] / valid_records
    
    # Calculate scores
    # Profit score (based on net profit and profit factor)
    profit_score = (avg_metrics.get("net_profit", 0) / 10000) + avg_metrics.get("profit_factor", 0)
    
    # Risk score (inverted drawdown + win rate)
    risk_score = (1 - avg_metrics.get("max_drawdown", 0)) + avg_metrics.get("win_rate", 0)
    
    # Overall score (combination of profit, risk, and sharpe)
    overall_score = profit_score + risk_score + avg_metrics.get("sharpe_ratio", 0)
    
    # Add scores to metrics
    avg_metrics["profit_score"] = profit_score
    avg_metrics["risk_score"] = risk_score
    avg_metrics["overall_score"] = overall_score
    
    return avg_metrics

def format_preset_suggestions(suggestions: List[Dict[str, Any]]) -> List[str]:
    """
    Format preset suggestions for display in UI
    
    Args:
        suggestions: List of preset suggestions
        
    Returns:
        List of formatted strings for display
    """
    formatted_suggestions = []
    
    for suggestion in suggestions:
        preset = suggestion["preset"]
        metrics = suggestion["metrics"]
        score = suggestion["score"]
        
        # Format basic preset info
        preset_name = preset.get("name", "Unknown")
        preset_type = preset.get("type", "user").capitalize()
        
        # Format key metrics
        net_profit = metrics.get("net_profit", 0)
        win_rate = metrics.get("win_rate", 0) * 100 if metrics.get("win_rate", 0) <= 1 else metrics.get("win_rate", 0)
        max_dd = metrics.get("max_drawdown", 0) * 100 if metrics.get("max_drawdown", 0) <= 1 else metrics.get("max_drawdown", 0)
        
        # Create formatted string
        formatted = f"{preset_name} ({preset_type}) - Score: {score:.2f} | "
        formatted += f"Profit: ${net_profit:.2f} | Win: {win_rate:.1f}% | DD: {max_dd:.1f}%"
        
        formatted_suggestions.append(formatted)
    
    return formatted_suggestions

def get_preset_suggestions_with_metrics(bucket: str, metric_filter: str = "overall") -> Tuple[List[str], List[str]]:
    """
    Get formatted preset suggestions with their IDs
    
    Args:
        bucket: The bucket type to filter by
        metric_filter: Filter by 'profit', 'risk', or 'overall'
        
    Returns:
        Tuple of (formatted_suggestions, preset_ids)
    """
    suggestions = get_preset_suggestions(bucket, metric_filter)
    formatted_suggestions = format_preset_suggestions(suggestions)
    preset_ids = [suggestion["preset"]["id"] for suggestion in suggestions]
    
    return formatted_suggestions, preset_ids

def update_suggestion_list(window, bucket, filter_type="overall"):
    """
    Update the suggestions list in the UI
    
    Args:
        window: The PySimpleGUI window
        bucket: The current bucket type
        filter_type: Filter by 'profit', 'risk', or 'overall'
        
    Returns:
        Dictionary mapping suggestion indices to preset IDs
    """
    global suggestion_id_map
    
    # Get the suggestions with their IDs
    formatted_suggestions, preset_ids = get_preset_suggestions_with_metrics(bucket, filter_type)
    
    # Update the suggestions listbox
    if "-SUGGESTION-LIST-" in window.AllKeysDict:
        window["-SUGGESTION-LIST-"].update(formatted_suggestions)
    
    # Update the ID map
    suggestion_id_map = {i: preset_id for i, preset_id in enumerate(preset_ids)}
    
    return suggestion_id_map

# UI components
def create_presets_tab():
    """
    Create the Parameter Presets tab layout
    
    Returns:
        List representing the tab layout
    """
    # Initialize preset directories and defaults
    ensure_preset_directories()
    initialize_default_presets()
    
    # Create the layout
    layout = [
        [sg.Text("Parameter Presets", font=("Helvetica", 16))],
        [sg.HorizontalSeparator()],
        
        # Presets section
        [sg.Frame("Load Preset", [
            [sg.Text("Select bucket:"), 
             sg.Combo(["Scalping", "Short", "Medium", "Long"], default_value="Scalping", key="-PRESET-BUCKET-", enable_events=True)],
            [sg.Text("Default presets:"), 
             sg.Button("Load Default Scalping", key="-LOAD-DEFAULT-SCALPING-"),
             sg.Button("Load Default Short", key="-LOAD-DEFAULT-SHORT-"),
             sg.Button("Load Default Medium", key="-LOAD-DEFAULT-MEDIUM-"),
             sg.Button("Load Default Long", key="-LOAD-DEFAULT-LONG-")],
            [sg.Text("Custom presets:"), sg.Listbox(values=[], size=(60, 5), key="-PRESET-LIST-", enable_events=True)],
            [sg.Button("Load Selected", key="-LOAD-PRESET-", disabled=True), 
             sg.Button("Delete", key="-DELETE-PRESET-", disabled=True),
             sg.Button("Quick View", key="-QUICK-VIEW-PERFORMANCE-", disabled=True),
             sg.Button("Refresh List", key="-REFRESH-PRESETS-")]
        ])],
        
        # Save preset section
        [sg.Frame("Save Current Settings", [
            [sg.Text("Preset Name:"), sg.Input(key="-PRESET-NAME-", size=(30, 1))],
            [sg.Text("Description:"), sg.Input(key="-PRESET-DESC-", size=(40, 1))],
            [sg.Checkbox("Save as temporary preset", key="-TEMP-PRESET-", default=False, tooltip="Temporary presets will be automatically deleted after 7 days of inactivity")],
            [sg.Button("Save Preset", key="-SAVE-PRESET-")]
        ])],
        
        # Suggestions section
        [sg.Frame("Suggestions Based on Performance", [
            [sg.Text("Filter by:"),
             sg.Radio("Best Overall", "SUGGESTION_FILTER", default=True, key="-FILTER-OVERALL-", enable_events=True),
             sg.Radio("Highest Profit", "SUGGESTION_FILTER", key="-FILTER-PROFIT-", enable_events=True),
             sg.Radio("Lowest Risk", "SUGGESTION_FILTER", key="-FILTER-RISK-", enable_events=True)],
            [sg.Listbox(values=[], size=(75, 6), key="-SUGGESTION-LIST-", enable_events=True)],
            [sg.Button("Load Selected Suggestion", key="-LOAD-SUGGESTION-", disabled=True),
             sg.Button("Clear Selection", key="-CLEAR-SUGGESTION-"),
             sg.Button("Refresh Suggestions", key="-REFRESH-SUGGESTIONS-")]
        ])],
        
        # Temporary Presets Management
        [sg.Frame("Temporary Presets Management", [
            [sg.Text("Temporary presets are automatically deleted after 7 days of inactivity.")],
            [sg.Button("List Temporary Presets", key="-LIST-TEMP-PRESETS-"),
             sg.Button("Keep Selected Preset", key="-KEEP-PRESET-", disabled=True),
             sg.Button("Clean Old Presets Now", key="-CLEAN-TEMP-PRESETS-")]
        ])]
    ]
    
    return layout

def integrate_with_backtesting(preset_id=None, config=None):
    """
    Setup integration functions for backtesting
    
    Args:
        preset_id: ID of the current preset being tested
        config: Current configuration
        
    Returns:
        Dict containing callback functions for backtesting
    """
    # Store the current preset ID and config for later use
    current_preset = {
        "id": preset_id,
        "config": config,
        "metrics": None
    }
    
    def on_backtest_complete(metrics, equity_curves, trade_histories):
        """
        Callback for when a backtest completes
        
        Args:
            metrics: List of metrics dictionaries from each episode
            equity_curves: List of equity curves from each episode
            trade_histories: List of trade histories from each episode
        """
        if not current_preset["id"]:
            logger.info("No preset ID provided, skipping performance update")
            return
        
        # Use the average metrics across all episodes
        if metrics and len(metrics) > 0:
            # Calculate average metrics if multiple episodes
            if len(metrics) > 1:
                avg_metrics = {}
                for key in metrics[0]:
                    avg_metrics[key] = sum(m.get(key, 0) for m in metrics) / len(metrics)
            else:
                # Just use the single episode metrics
                avg_metrics = metrics[0]
            
            # Store metrics for later use
            current_preset["metrics"] = avg_metrics
            
            # Update the preset performance history
            success = update_preset_performance(current_preset["id"], avg_metrics)
            if success:
                logger.info(f"Updated performance history for preset: {current_preset['id']}")
            else:
                logger.error(f"Failed to update performance history for preset: {current_preset['id']}")
    
    def on_comparison_complete(preset_metrics, user_metrics):
        """
        Callback for when a comparison completes
        
        Args:
            preset_metrics: Metrics for the preset configuration
            user_metrics: Metrics for the user configuration
        """
        if not current_preset["id"]:
            logger.info("No preset ID provided, skipping comparison performance update")
            return
        
        # Update the preset performance history with the preset metrics
        success = update_preset_performance(current_preset["id"], preset_metrics)
        if success:
            logger.info(f"Updated performance history from comparison for preset: {current_preset['id']}")
        else:
            logger.error(f"Failed to update comparison performance history for preset: {current_preset['id']}")
    
    # Return the callbacks
    return {
        "on_backtest_complete": on_backtest_complete,
        "on_comparison_complete": on_comparison_complete
    }

def get_backtest_metrics_from_results(backtesting_results):
    """
    Extract key metrics from backtesting results
    
    Args:
        backtesting_results: Results from a backtesting run
        
    Returns:
        Dictionary of key metrics
    """
    # Backtesting results can be in different formats depending on the function used
    # Try to handle common formats
    
    # Check if it's a tuple of (metrics, equity_curves, trade_histories)
    if isinstance(backtesting_results, tuple) and len(backtesting_results) >= 1:
        metrics = backtesting_results[0]
        
        # If metrics is a list (multiple episodes), calculate average
        if isinstance(metrics, list) and metrics:
            if len(metrics) > 1:
                avg_metrics = {}
                for key in metrics[0]:
                    values = [m.get(key, 0) for m in metrics]
                    # Filter out None values
                    values = [v for v in values if v is not None]
                    if values:
                        avg_metrics[key] = sum(values) / len(values)
                return avg_metrics
            else:
                return metrics[0]
        else:
            return metrics
    
    # Check if it's a dictionary of metrics
    elif isinstance(backtesting_results, dict):
        return backtesting_results
    
    # If we couldn't identify the format, return empty dict
    logger.warning("Could not extract metrics from backtesting results - unknown format")
    return {}

# UI components for preset performance history
def create_performance_history_layout(preset_id):
    """
    Create a layout for displaying preset performance history
    
    Args:
        preset_id: ID of the preset to display history for
        
    Returns:
        List representing the layout
    """
    # Load performance history for this preset
    all_history = load_performance_history()
    preset_history = all_history.get(preset_id, [])
    
    # Create header row
    header = ["Date", "Net Profit", "Win Rate", "Trades", "Max DD", "Sharpe"]
    
    # Create data rows
    rows = []
    for record in preset_history:
        metrics = record.get("metrics", {})
        timestamp = record.get("timestamp", "")
        
        try:
            # Format the timestamp
            date_obj = datetime.datetime.fromisoformat(timestamp)
            date_str = date_obj.strftime("%Y-%m-%d %H:%M")
        except:
            date_str = timestamp
        
        # Format the metrics
        net_profit = metrics.get("net_profit", 0)
        win_rate = metrics.get("win_rate", 0) * 100 if metrics.get("win_rate", 0) <= 1 else metrics.get("win_rate", 0)
        total_trades = metrics.get("total_trades", 0)
        max_dd = metrics.get("max_drawdown", 0) * 100 if metrics.get("max_drawdown", 0) <= 1 else metrics.get("max_drawdown", 0)
        sharpe = metrics.get("sharpe_ratio", 0)
        
        # Add row
        row = [
            date_str,
            f"${net_profit:.2f}",
            f"{win_rate:.1f}%",
            str(total_trades),
            f"{max_dd:.1f}%",
            f"{sharpe:.2f}"
        ]
        rows.append(row)
    
    # Sort rows by date (newest first)
    rows.sort(reverse=True, key=lambda x: x[0])
    
    # Create table layout
    if rows:
        layout = [
            [sg.Text("Performance History", font=("Helvetica", 12))],
            [sg.Table(
                values=rows,
                headings=header,
                auto_size_columns=True,
                display_row_numbers=False,
                justification='right',
                num_rows=min(10, len(rows)),
                key="-PERFORMANCE-TABLE-",
                expand_x=True
            )]
        ]
    else:
        layout = [
            [sg.Text("Performance History", font=("Helvetica", 12))],
            [sg.Text("No performance data available for this preset")]
        ]
    
    return layout

def show_performance_history(preset_id):
    """
    Show a window with performance history for a preset
    
    Args:
        preset_id: ID of the preset to display history for
    """
    # Load the preset data
    preset_data = load_preset(preset_id)
    preset_name = preset_data.get("name", os.path.basename(preset_id).replace('.json', ''))
    
    # Create window layout
    layout = [
        [sg.Text(f"Performance History: {preset_name}", font=("Helvetica", 16))],
        [sg.HorizontalSeparator()],
        *create_performance_history_layout(preset_id),
        [sg.Button("Close")]
    ]
    
    # Create window
    window = sg.Window(f"Preset Performance History: {preset_name}", layout, modal=True, finalize=True)
    
    # Event loop
    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, "Close"):
            break
    
    window.close()

def show_simplified_performance_history(preset_id):
    """
    Show a simplified window with key performance metrics for a preset
    
    Args:
        preset_id: ID of the preset to display history for
    """
    # Load the preset data
    preset_data = load_preset(preset_id)
    preset_name = preset_data.get("name", os.path.basename(preset_id).replace('.json', ''))
    
    # Load performance history for this preset
    all_history = load_performance_history()
    preset_history = all_history.get(preset_id, [])
    
    if not preset_history:
        sg.popup_quick("No performance history available for this preset", 
                      title=f"{preset_name} - No History")
        return
    
    # Sort history by date (newest first)
    preset_history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    
    # Keep only the 5 most recent entries
    recent_history = preset_history[:5]
    
    # Create a simplified table with just 3-4 key metrics
    rows = []
    headers = ["Date", "Net Profit", "Win Rate", "Max Drawdown", "Trades"]
    
    for record in recent_history:
        metrics = record.get("metrics", {})
        timestamp = record.get("timestamp", "")
        
        try:
            # Format the timestamp
            date_obj = datetime.datetime.fromisoformat(timestamp)
            date_str = date_obj.strftime("%Y-%m-%d %H:%M")
        except:
            date_str = timestamp
        
        # Format the metrics - keep it simple with just key numbers
        net_profit = metrics.get("net_profit", 0)
        win_rate = metrics.get("win_rate", 0) * 100 if metrics.get("win_rate", 0) <= 1 else metrics.get("win_rate", 0)
        max_dd = metrics.get("max_drawdown", 0) * 100 if metrics.get("max_drawdown", 0) <= 1 else metrics.get("max_drawdown", 0)
        total_trades = metrics.get("total_trades", 0)
        
        # Add row
        row = [
            date_str,
            f"${net_profit:.2f}",
            f"{win_rate:.1f}%",
            f"{max_dd:.1f}%",
            str(total_trades)
        ]
        rows.append(row)
    
    # Create layout
    layout = [
        [sg.Text(f"Recent Performance: {preset_name}", font=("Helvetica", 14))],
        [sg.Table(
            values=rows,
            headings=headers,
            auto_size_columns=True,
            display_row_numbers=False,
            justification='right',
            num_rows=min(5, len(rows)),
            key="-SIMPLE-PERF-TABLE-"
        )],
        [sg.Button("Close"), sg.Button("View Full History", key="-VIEW-FULL-HISTORY-")]
    ]
    
    # Create window
    window = sg.Window(f"Performance Summary: {preset_name}", layout, modal=True, finalize=True)
    
    # Event loop
    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, "Close"):
            break
        elif event == "-VIEW-FULL-HISTORY-":
            window.close()
            show_performance_history(preset_id)
            return
    
    window.close() 