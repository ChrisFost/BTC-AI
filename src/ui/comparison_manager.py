"""
Comparison Manager Module

This module handles functionality related to comparing trading strategies,
including comparing current settings with default settings and displaying
comparison results.
"""

import os
import sys
import json
import time
import logging
import threading
import tempfile
import subprocess
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Callable

import PySimpleGUI as sg

# Try to import error handling
try:
    from src.ui.error_handler import handle_error, ErrorSeverity
    from src.utils.persistent_logger import log_persistent_error
    error_handling_available = True
except ImportError:
    error_handling_available = False
    # Define stub functions if error handling is not available
    def handle_error(error, context="", window=None, retry_func=None, additional_context=None):
        if isinstance(error, Exception):
            logging.error(f"Error in {context}: {str(error)}")
        return {"handled": False}
    
    class ErrorSeverity:
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        FATAL = "fatal"
    
    def log_persistent_error(error, context="", severity="medium", additional_info=None):
        pass

# Try to import bucket manager
try:
    from src.ui.bucket_manager import BucketManager
    bucket_manager_available = True
except ImportError:
    bucket_manager_available = False
    logging.warning("Bucket manager not available for comparison analysis")

# Removed top-level import check for preset handlers
# We will check availability when needed

# Set up logger
logger = logging.getLogger(__name__)

# Constants
DEFAULT_EPISODE_COUNT = 50
DEFAULT_COMPARISON_METRICS = [
    "net_profit",
    "win_rate",
    "sharpe_ratio",
    "max_drawdown",
    "total_trades",
    "avg_trade_duration",
    "profit_factor"
]

# Paths
try:
    # Use the new paths module for consistent path handling
    from src.utils.paths import get_project_root, get_common_paths, add_project_to_path
    
    # Add project root to path for consistent imports
    add_project_to_path()
    
    # Get common paths dictionary
    paths = get_common_paths()
    
    # Define paths using the common paths dictionary
    COMPARISON_DIR = paths["comparison_results"]
    TRAINING_SCRIPT_PATH = paths["training_script"]
    
    logger.info(f"Project root: {paths['project_root']}")
    logger.info(f"Comparison directory: {COMPARISON_DIR}")
    logger.info(f"Training script path: {TRAINING_SCRIPT_PATH}")
except Exception as e:
    # Fallback if there's an error with the paths module
    logger.error(f"Error using paths module: {e}")
    logger.warning("Falling back to direct path determination")
    
    # Get project root from the module's location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    COMPARISON_DIR = os.path.join(project_root, "comparison_results")
    TRAINING_SCRIPT_PATH = os.path.join(project_root, "src", "training", "training.py")

class ComparisonManager:
    """Class to manage trading strategy comparisons"""
    
    def __init__(self, app_state=None, bucket_manager=None):
        """
        Initialize the ComparisonManager.
        
        Args:
            app_state: Optional reference to the application state
            bucket_manager: Optional reference to a BucketManager instance
        """
        self.app_state = app_state
        self.bucket_manager = bucket_manager
        self.comparison_process = None
        self.last_comparison_results = None
        
        # Create comparison directory if it doesn't exist
        os.makedirs(COMPARISON_DIR, exist_ok=True)
        
        logger.info("ComparisonManager initialized")
    
    def run_comparison(self, window, values, episodes=DEFAULT_EPISODE_COUNT):
        """
        Run a comparison between user settings and default settings for the selected bucket.
        
        Args:
            window: The PySimpleGUI window
            values: The current values dictionary from the window
            episodes: Number of episodes to run for each configuration (default 50)
            
        Returns:
            subprocess.Popen: The comparison process if successful, None otherwise
        """
        try:
            logger.info(f"Starting {episodes}-episode comparison")
            
            # Get current bucket
            bucket = values.get("BUCKET", "Scalping")
            
            # Save current settings to a temporary file
            current_settings_file = os.path.join(tempfile.gettempdir(), "current_settings.json")
            with open(current_settings_file, "w", encoding="utf-8") as f:
                # Extract current settings from values
                current_settings = {}
                for key, value in values.items():
                    if isinstance(value, (str, int, float, bool)) and key in values:
                        current_settings[key] = value
                json.dump(current_settings, f, indent=4)
            
            # Get default settings for this bucket
            # In a real implementation, we would get these from preset system
            # For now, create some sample defaults
            default_settings = self._get_default_settings(bucket, current_settings)
            
            # Save default settings to a temporary file
            default_settings_file = os.path.join(tempfile.gettempdir(), "default_settings.json")
            with open(default_settings_file, "w", encoding="utf-8") as f:
                json.dump(default_settings, f, indent=4)
            
            # Create directory for comparison results
            os.makedirs(COMPARISON_DIR, exist_ok=True)
            
            # Path to comparison script
            comparison_script = TRAINING_SCRIPT_PATH
            
            # Show a pop-up dialog about the comparison
            sg.popup_no_wait(
                f"Starting {episodes}-episode comparison between your settings and default {bucket} settings.\n\n"
                f"This will run {episodes} episodes with each configuration and may take some time.\n\n"
                f"You'll be notified when the comparison is complete.",
                title="Comparison Started",
                non_blocking=True
            )
            
            # Update log
            window["-LOG-"].print(f"[{datetime.now()}] Starting {episodes}-episode comparison")
            window["-LOG-"].print(f"[{datetime.now()}] Comparing current settings vs. default {bucket} settings")
            
            # Construct command for running the comparison
            cmd = [
                sys.executable,
                comparison_script,
                "--compare",
                "--episodes", str(episodes),
                "--current-config", current_settings_file,
                "--default-config", default_settings_file,
                "--output-dir", COMPARISON_DIR,
                "--bucket", bucket,
                "--no-save-models",  # Flag to prevent saving models from comparison runs
                "--temp-run"  # Flag to indicate this is a temporary comparison run
            ]
            
            # Start the comparison process
            self.comparison_process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                text=True
            )
            
            # Start a thread to monitor the comparison process
            self._start_monitoring_thread(window)
            
            # Enable/disable buttons
            window["RUN_COMPARISON"].update(disabled=True)
            window["-STATUS-"].update("Status: Running comparison...")
            
            return self.comparison_process
            
        except Exception as e:
            if error_handling_available:
                handle_error(
                    e,
                    context="Starting comparison",
                    window=window
                )
                log_persistent_error(
                    e,
                    context="Starting comparison",
                    severity=ErrorSeverity.MEDIUM
                )
            logger.error(f"Error starting comparison: {e}")
            window["-LOG-"].print(f"[{datetime.now()}] Error starting comparison: {e}")
            sg.popup_error(f"Error starting comparison: {str(e)}")
            return None
    
    def _start_monitoring_thread(self, window):
        """
        Start a thread to monitor the comparison process.
        
        Args:
            window: The PySimpleGUI window
        """
        def monitor_comparison():
            if not self.comparison_process:
                return
                
            output = []
            for line in self.comparison_process.stdout:
                output.append(line.strip())
                window.write_event_value("-COMPARISON-UPDATE-", line.strip())
            
            # Process completed, check return code
            return_code = self.comparison_process.wait()
            
            if return_code == 0:
                # Success
                results_file = os.path.join(COMPARISON_DIR, "comparison_results.json")
                if os.path.exists(results_file):
                    try:
                        with open(results_file, "r", encoding="utf-8") as f:
                            results = json.load(f)
                        
                        # Store the results
                        self.last_comparison_results = results
                        
                        # Send the results to the UI
                        window.write_event_value("-COMPARISON-COMPLETE-", results)
                    except Exception as e:
                        # Use error handling system if available
                        if error_handling_available:
                            handle_error(
                                e,
                                context="Reading comparison results",
                                additional_context={"file_path": results_file}
                            )
                            log_persistent_error(
                                e,
                                context="Reading comparison results",
                                severity=ErrorSeverity.MEDIUM,
                                additional_info={"file_path": results_file}
                            )
                        # Always log the error
                        logger.error(f"Error reading comparison results: {str(e)}")
                        window.write_event_value("-COMPARISON-ERROR-", f"Error reading results: {str(e)}")
                else:
                    if error_handling_available:
                        # Log a file not found error
                        error = FileNotFoundError(f"Results file not found: {results_file}")
                        handle_error(
                            error,
                            context="Locating comparison results",
                            additional_context={"file_path": results_file}
                        )
                    logger.error(f"Comparison results file not found: {results_file}")
                    window.write_event_value("-COMPARISON-ERROR-", "Results file not found")
            else:
                # Error with process
                error_message = "Comparison failed with exit code " + str(return_code)
                if output:
                    error_message += "\n\nLast output:\n" + "\n".join(output[-10:])
                
                if error_handling_available:
                    # Log process error
                    error = RuntimeError(f"Comparison process failed with exit code {return_code}")
                    handle_error(
                        error,
                        context="Running comparison process",
                        additional_context={"exit_code": return_code, "output": "\n".join(output[-10:])}
                    )
                    log_persistent_error(
                        error,
                        context="Running comparison process",
                        severity=ErrorSeverity.MEDIUM,
                        additional_info={"exit_code": return_code}
                    )
                logger.error(f"Comparison process failed: {error_message}")
                window.write_event_value("-COMPARISON-ERROR-", error_message)
        
        # Start the monitoring thread
        threading.Thread(target=monitor_comparison, daemon=True).start()
    
    def stop_comparison(self):
        """
        Stop the current comparison process if running.
        
        Returns:
            bool: True if stopped successfully, False otherwise
        """
        if not self.comparison_process:
            logger.warning("No comparison process to stop")
            return False
            
        try:
            # Try to terminate the process
            self.comparison_process.terminate()
            try:
                # Wait for process to terminate cleanly
                self.comparison_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # If timeout, force kill
                self.comparison_process.kill()
            
            logger.info("Comparison process terminated")
            self.comparison_process = None
            return True
        except Exception as e:
            logger.error(f"Error stopping comparison process: {e}")
            if error_handling_available:
                log_persistent_error(
                    e,
                    context="Stopping comparison process",
                    severity=ErrorSeverity.MEDIUM
                )
            return False
    
    def _get_default_settings(self, bucket, current_settings):
        """
        Get default settings for a bucket.
        
        Args:
            bucket: The strategy bucket
            current_settings: The current settings as fallback
            
        Returns:
            dict: Default settings for the bucket
        """
        # In a real implementation, we would load these from a preset system
        # For now, just modify some values from current settings
        default_settings = current_settings.copy()
        
        # Apply some standard defaults based on bucket type
        if bucket == "Scalping":
            default_settings.update({
                "LEARNING_RATE": 0.001,
                "DROPOUT": 0.2,
                "BATCH_SIZE": 64,
                "SEQ_LEN": 60
            })
        elif bucket == "Short":
            default_settings.update({
                "LEARNING_RATE": 0.0005,
                "DROPOUT": 0.3,
                "BATCH_SIZE": 128,
                "SEQ_LEN": 120
            })
        elif bucket == "Medium":
            default_settings.update({
                "LEARNING_RATE": 0.0003,
                "DROPOUT": 0.4,
                "BATCH_SIZE": 256,
                "SEQ_LEN": 240
            })
        elif bucket == "Long":
            default_settings.update({
                "LEARNING_RATE": 0.0001,
                "DROPOUT": 0.5,
                "BATCH_SIZE": 512,
                "SEQ_LEN": 480
            })
        
        return default_settings
    
    def handle_comparison_complete(self, window, values, event):
        """
        Handle completion of the comparison process.
        
        Args:
            window: The PySimpleGUI window
            values: The current values dictionary from the window
            event: The event that triggered this handler
        """
        # Re-enable comparison button
        if "RUN_COMPARISON" in window.AllKeysDict:
            window["RUN_COMPARISON"].update(disabled=False)
            
        # Update status
        if "-STATUS-" in window.AllKeysDict:
            window["-STATUS-"].update("Status: Comparison complete")
            
        # Log completion
        logger.info("Comparison process completed successfully")
        
        # Try to update preset performance if available
        try:
            # Import preset functions here, just before use
            from src.ui.preset_handlers import get_current_preset_id, update_preset_performance
            preset_handlers_available_runtime = True
        except ImportError:
            preset_handlers_available_runtime = False
            logger.warning("Preset handlers could not be imported at runtime for comparison results.")
            
        if preset_handlers_available_runtime and values[event]:
            try:
                current_preset_id = get_current_preset_id()
                
                if current_preset_id:
                    # Extract metrics from comparison results (assuming results format)
                    results = values[event]
                    current_metrics = results.get("current", {})
                    default_metrics = results.get("default", {})
                    
                    # Decide which metrics to update with (e.g., default's performance)
                    metrics_to_update = default_metrics # Or current_metrics, depending on logic
                    
                    if metrics_to_update:
                        success = update_preset_performance(current_preset_id, metrics_to_update)
                        if success:
                            logger.info(f"Updated performance for preset {current_preset_id} from comparison results")
                        else:
                            logger.warning(f"Failed to update performance for preset {current_preset_id}")
                    else:
                        logger.info("No metrics found in comparison results to update preset performance.")
                else:
                    logger.info("No current preset loaded, skipping performance update.")
            except Exception as e:
                logger.error(f"Error updating preset performance after comparison: {e}")
                # Use error handling system if available
                if error_handling_available:
                    handle_error(
                        e,
                        context="Updating preset performance post-comparison",
                        window=window
                    )
        
        # Show results dialog
        if values[event]:
            self.show_comparison_results(values[event])
        else:
            logger.warning("Comparison completed but no results data received.")
            sg.popup("Comparison completed, but no results data was found.")
    
    def show_comparison_results(self, results):
        """
        Display the results of a comparison in a window.
        
        Args:
            results: Dictionary containing comparison results
            
        Returns:
            bool: True if the user wants to run another comparison
        """
        if not results:
            sg.popup_error("No comparison results to display")
            return False
        
        # Extract metrics for both configurations
        current_metrics = results.get("current", {})
        default_metrics = results.get("default", {})
        
        # Create a table of metrics
        table_data = []
        metrics = [
            "Net Profit", "Win Rate", "Sharpe Ratio", "Max Drawdown", 
            "Total Trades", "Avg Trade Duration", "Profit Factor"
        ]
        
        for metric in metrics:
            key = metric.lower().replace(" ", "_")
            current_value = current_metrics.get(key, "N/A")
            default_value = default_metrics.get(key, "N/A")
            
            # Format values
            if isinstance(current_value, float):
                if "profit" in key or "drawdown" in key:
                    current_value = f"${current_value:.2f}"
                elif "rate" in key or "factor" in key or "ratio" in key:
                    current_value = f"{current_value:.2f}"
            
            if isinstance(default_value, float):
                if "profit" in key or "drawdown" in key:
                    default_value = f"${default_value:.2f}"
                elif "rate" in key or "factor" in key or "ratio" in key:
                    default_value = f"{default_value:.2f}"
            
            # Determine which is better
            better = ""
            if isinstance(current_metrics.get(key), (int, float)) and isinstance(default_metrics.get(key), (int, float)):
                if "drawdown" in key:  # Lower is better
                    better = "Current" if current_metrics.get(key) < default_metrics.get(key) else "Default"
                else:  # Higher is better
                    better = "Current" if current_metrics.get(key) > default_metrics.get(key) else "Default"
            
            table_data.append([metric, current_value, default_value, better])
        
        # Generate recommendation
        recommendation = self._generate_recommendation(current_metrics, default_metrics)
        
        # Create layout
        layout = [
            [sg.Text("50-Episode Comparison Results", font=("Helvetica", 16, "bold"))],
            [sg.Text(f"Bucket: {results.get('bucket', 'Unknown')}", font=("Helvetica", 12))],
            [sg.Text("Comparing your current settings vs. default bucket settings", font=("Helvetica", 10))],
            [sg.Table(
                values=table_data,
                headings=["Metric", "Your Settings", "Default Settings", "Better Option"],
                auto_size_columns=False,
                col_widths=[15, 15, 15, 12],
                justification="center",
                num_rows=min(10, len(table_data)),
                key="-RESULTS-TABLE-"
            )],
            [sg.Text("Recommendation:", font=("Helvetica", 12, "bold"))],
            [sg.Text(recommendation, font=("Helvetica", 10))],
            [sg.Button("Keep My Settings", key="-KEEP-SETTINGS-"), 
             sg.Button("Use Default Settings", key="-USE-DEFAULT-"),
             sg.Button("Go Back & Adjust", key="-GO-BACK-"),
             sg.Button("Close", key="-CLOSE-RESULTS-")]
        ]
        
        window = sg.Window("Comparison Results", layout, modal=True, finalize=True)
        
        # Return value for the caller to know if the user wants to run another comparison
        run_another_comparison = False
        
        # Event loop for the results window
        while True:
            event, values = window.read()
            
            if event == sg.WIN_CLOSED or event == "-CLOSE-RESULTS-":
                break
                
            elif event == "-KEEP-SETTINGS-":
                # User wants to keep their settings
                logger.info("User chose to keep current settings")
                break
                
            elif event == "-USE-DEFAULT-":
                # User wants to use default settings
                logger.info("User chose to use default settings")
                # In a real implementation, we would apply the default settings here
                sg.popup("Default settings will be applied when you close this window.")
                break
                
            elif event == "-GO-BACK-":
                # User wants to make more adjustments
                logger.info("User chose to make more adjustments")
                run_another_comparison = True
                break
        
        window.close()
        return run_another_comparison
    
    def _generate_recommendation(self, current_metrics, default_metrics):
        """
        Generate a recommendation based on comparison metrics.
        
        Args:
            current_metrics: Metrics for current settings
            default_metrics: Metrics for default settings
            
        Returns:
            str: Recommendation text
        """
        # Count how many metrics are better for each configuration
        current_better_count = 0
        default_better_count = 0
        
        # List of metrics to consider
        metrics_to_check = [
            "net_profit", "win_rate", "sharpe_ratio", "profit_factor", "max_drawdown"
        ]
        
        for metric in metrics_to_check:
            if metric not in current_metrics or metric not in default_metrics:
                continue
                
            # For drawdown, lower is better
            if metric == "max_drawdown":
                if current_metrics[metric] < default_metrics[metric]:
                    current_better_count += 1
                else:
                    default_better_count += 1
            else:  # For other metrics, higher is better
                if current_metrics[metric] > default_metrics[metric]:
                    current_better_count += 1
                else:
                    default_better_count += 1
        
        # Generate recommendation based on comparison
        if current_better_count > default_better_count:
            percentage = (current_better_count / len(metrics_to_check)) * 100
            return (
                f"Your current settings outperform the default settings in {current_better_count} "
                f"out of {len(metrics_to_check)} key metrics ({percentage:.1f}%). It is recommended "
                f"to keep your current settings, as they appear to be better optimized for your strategy."
            )
        elif default_better_count > current_better_count:
            percentage = (default_better_count / len(metrics_to_check)) * 100
            return (
                f"The default settings outperform your current settings in {default_better_count} "
                f"out of {len(metrics_to_check)} key metrics ({percentage:.1f}%). Consider using the "
                f"default settings as a better foundation for your strategy."
            )
        else:
            # If they're equal, check profit specifically
            if "net_profit" in current_metrics and "net_profit" in default_metrics:
                if current_metrics["net_profit"] > default_metrics["net_profit"]:
                    return (
                        "Your current settings and the default settings perform similarly overall, but your settings "
                        "achieve higher profit. It's recommended to keep your current settings."
                    )
                elif default_metrics["net_profit"] > current_metrics["net_profit"]:
                    return (
                        "Your current settings and the default settings perform similarly overall, but the default settings "
                        "achieve higher profit. Consider switching to the default settings."
                    )
            
            # If everything is equal or can't be determined
            return (
                "Your current settings and the default settings perform similarly. You can choose either based on "
                "your preference or run more episodes for a more detailed comparison."
            )
    
    def get_last_comparison_results(self):
        """
        Get the results of the last comparison run.
        
        Returns:
            dict: Comparison results or None if no comparison has been run
        """
        return self.last_comparison_results

# For backwards compatibility
def run_comparison(window, values, episodes=DEFAULT_EPISODE_COUNT):
    """
    Legacy function to run a comparison.
    This is maintained for backward compatibility.
    
    Args:
        window: The PySimpleGUI window
        values: The current values dictionary from the window
        episodes: Number of episodes to run for each configuration (default 50)
        
    Returns:
        subprocess.Popen: The comparison process if successful, None otherwise
    """
    # Create a temporary ComparisonManager and run the comparison
    comparison_manager = ComparisonManager()
    return comparison_manager.run_comparison(window, values, episodes)

def show_comparison_results(results):
    """
    Legacy function to display comparison results.
    This is maintained for backward compatibility.
    
    Args:
        results: Dictionary containing comparison results
        
    Returns:
        bool: True if the user wants to run another comparison
    """
    # Create a temporary ComparisonManager and show the results
    comparison_manager = ComparisonManager()
    return comparison_manager.show_comparison_results(results) 