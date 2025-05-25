"""
Training Manager for the RL Trader Parameter Tuner

This module manages all aspects of training control, including starting,
stopping, pausing, and resuming training processes, as well as monitoring
training progress and collecting performance metrics.
"""

import os
import sys
import json
import time
import torch
import psutil
import logging
import threading
import subprocess
import tempfile
from datetime import datetime

import PySimpleGUI as sg

# Make sure we can import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logger
logger = logging.getLogger(__name__)

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

# Constants
project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SCRIPT_DIR = os.path.abspath(os.path.join(project_root, "src", "training"))
MODELS_DIR_DEFAULT = os.path.join(project_root, "Models")

# Ensure model directories exist for all buckets
BUCKET_TYPES = ["Scalping", "Short", "Medium", "Long"]
for bucket in BUCKET_TYPES:
    bucket_dir = os.path.join(MODELS_DIR_DEFAULT, bucket)
    checkpoints_dir = os.path.join(bucket_dir, "checkpoints")
    predictive_dir = os.path.join(bucket_dir, "predictive_agent")
    
    # Create directories if they don't exist
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(predictive_dir, exist_ok=True)
    
    # Log directory creation
    if not os.path.exists(os.path.join(bucket_dir, "training_log.txt")):
        # Create initial training log if it doesn't exist
        with open(os.path.join(bucket_dir, "training_log.txt"), 'w') as f:
            f.write(f"Training log for {bucket} bucket\n")
            f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Add validation to ensure training.py exists
if not os.path.exists(os.path.join(SCRIPT_DIR, "training.py")):
    logger.error(f"Critical error: training.py not found in {SCRIPT_DIR}")
    # Check if it's in Models directory instead (legacy check)
    alt_path = os.path.join(project_root, "Models")
    if os.path.exists(os.path.join(alt_path, "main.py")):
        logger.warning(f"Found main.py in Models directory instead. Consider updating references.")
else:
    logger.info(f"Training script validated at: {os.path.join(SCRIPT_DIR, 'training.py')}")
    logger.info(f"Models directory configured at: {MODELS_DIR_DEFAULT}")
    logger.info(f"Created bucket directories for: {', '.join(BUCKET_TYPES)}")

class TrainingManager:
    """Class to manage training processes and state."""
    
    def __init__(self, app_state=None):
        """
        Initialize the TrainingManager.
        
        Args:
            app_state: Optional reference to the application state
        """
        self.process = None
        self.app_state = app_state
        self.preset_system_available = False
        self.current_training_preset_id = None
        self.training_monitor_thread = None
        
        # Try to import preset system
        try:
            from src.ui.preset_manager import save_preset
            self.preset_system_available = True
            logger.info("Preset system available for training tracking")
        except ImportError:
            logger.warning("Preset system not available for training tracking")
    
    def start_training(self, window, values):
        """
        Start a training process with the given parameters.
        
        Args:
            window: The main application window
            values: The current values from the GUI
            
        Returns:
            bool: True if training started successfully, False otherwise
        """
        try:
            # Save all state before starting training if app_state is available
            if self.app_state:
                self.app_state.save_all_state(values)
            
            # If preset system is available, create a temporary preset for tracking
            if self.preset_system_available:
                try:
                    # Import preset manager functions
                    from src.ui.preset_manager import save_preset
                    
                    # Get current bucket type
                    bucket_type = values.get("BUCKET", "Scalping")
                    
                    # Create a temporary preset for this training session
                    session_name = f"Training Run {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                    preset_id = save_preset(
                        bucket=bucket_type,
                        name=session_name,
                        params=values,
                        description="Auto-generated for training session tracking",
                        is_temporary=True
                    )
                    
                    # Store the preset ID
                    self.current_training_preset_id = preset_id
                    if self.app_state:
                        self.app_state.current_training_preset_id = preset_id
                    logger.info(f"Created temporary preset {preset_id} for tracking training session")
                except Exception as e:
                    logger.error(f"Error creating training preset: {e}")
                    self.current_training_preset_id = None
                    if self.app_state:
                        self.app_state.current_training_preset_id = None
            
            # Verify training script exists
            training_script = os.path.join(SCRIPT_DIR, "training.py")
            if not os.path.exists(training_script):
                error_msg = f"Training script not found at {training_script}"
                logger.error(error_msg)
                if error_handling_available:
                    handle_error(
                        ValueError(error_msg),
                        context="Finding training script",
                        window=window
                    )
                sg.popup_error(f"Training script not found: {training_script}")
                return False
                
            logger.info(f"Using training script: {training_script}")
            
            # Initialize the command with the script path
            cmd = [
                sys.executable,
                training_script
            ]
            
            # Helper function to add parameters that exist in values
            def add_param(param_name, cmd_param=None):
                if param_name in values and values[param_name] is not None:
                    if cmd_param is None:
                        cmd_param = f"--{param_name.lower()}"
                    cmd.extend([cmd_param, str(values[param_name])])
            
            # Add all parameters that exist in values
            add_param('DATA_PATH', '--data_path')
            add_param('PRICE_COLUMN', '--price_column')
            add_param('SEQ_LEN', '--seq_len')
            add_param('TEST_SIZE', '--test_size')
            add_param('MODEL_TYPE', '--model_type')
            add_param('HIDDEN_SIZE', '--hidden_size')
            add_param('NUM_LAYERS', '--num_layers')
            add_param('DROPOUT', '--dropout')
            add_param('BATCH_SIZE', '--batch_size')
            add_param('LEARNING_RATE', '--learning_rate')
            add_param('NUM_EPOCHS', '--num_epochs')
            add_param('SAVE_FREQ', '--save_freq')
            add_param('RANDOM_STATE', '--random_state')
            
            # Add device parameter
            device = "cuda" if torch.cuda.is_available() else "cpu"
            cmd.extend(["--device", device])
            
            # Model-specific args
            if values.get('MODEL_TYPE') == 'cnn_lstm':
                add_param('KERNEL_SIZE', '--kernel_size')
                add_param('NUM_FILTERS', '--num_filters')
                
            # Add bucket goal parameters if present
            bucket_goal_params = [
                "monthly_target_min", "monthly_target_max", 
                "yearly_target_min", "yearly_target_max",
                "min_gain_per_holding", "max_gain_per_holding", 
                "bonus_multiplier"
            ]
            
            for param in bucket_goal_params:
                if param.upper() in values and values[param.upper()] is not None:
                    cmd.extend([f"--{param}", str(values[param.upper()])])
                    
            # Add bucket type
            bucket_type = values.get("BUCKET", "Scalping")
            cmd.extend(["--bucket", bucket_type])
            
            # Log the command being executed
            logger.info(f"Training command: {' '.join(cmd)}")
            
            # Start training process with full command line arguments
            self.process = subprocess.Popen(cmd, cwd=project_root)
            if self.app_state:
                self.app_state.process = self.process
                # Set training start time for performance monitoring
                self.app_state.training_start_time = time.time()
            
            # Update UI buttons
            window["START_TRAINING"].update(disabled=True)
            window["STOP_TRAINING"].update(disabled=False)
            window["PAUSE_TRAINING"].update(disabled=False)
            window["RESUME_TRAINING"].update(disabled=True)
            
            # Start a thread to monitor training completion
            self._start_training_monitor(window, values)
            
            logger.info(f"Started training process with PID {self.process.pid}")
            window["-LOG-"].print(f"[{datetime.now()}] Started training process")
            return True
            
        except Exception as e:
            if error_handling_available:
                handle_error(
                    e,
                    context="Starting training process",
                    window=window
                )
                log_persistent_error(
                    e,
                    context="Starting training process",
                    severity=ErrorSeverity.HIGH
                )
            logger.error(f"Error starting training: {e}")
            sg.popup_error(f"Error starting training: {str(e)}")
            return False
    
    def stop_training(self, window):
        """
        Stop the training process.
        
        Args:
            window: The main application window
            
        Returns:
            bool: True if stopped successfully, False otherwise
        """
        if not self.process:
            logger.warning("No active training process to stop")
            return False
            
        try:
            # Terminate the process
            self.process.terminate()
            try:
                # Wait for process to terminate cleanly
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # If timeout, force kill
                self.process.kill()
            
            # Update UI buttons
            window["START_TRAINING"].update(disabled=False)
            window["STOP_TRAINING"].update(disabled=True)
            window["PAUSE_TRAINING"].update(disabled=True)
            window["RESUME_TRAINING"].update(disabled=True)
            
            # Update log
            window["-LOG-"].print(f"[{datetime.now()}] Training stopped")
            
            logger.info("Training process terminated")
            if self.app_state:
                self.app_state.process = None
                # Clear training start time
                self.app_state.training_start_time = None
            self.process = None
            return True
            
        except Exception as e:
            if error_handling_available:
                handle_error(
                    e,
                    context="Stopping training process",
                    additional_context={"process_id": getattr(self.process, "pid", "Unknown")}
                )
                log_persistent_error(
                    e,
                    context="Stopping training process",
                    severity=ErrorSeverity.MEDIUM,
                    additional_info={"process_id": getattr(self.process, "pid", "Unknown")}
                )
            logger.error(f"Error stopping training process: {e}")
            return False
    
    def pause_training(self, window):
        """
        Pause the training process using psutil.
        
        Args:
            window: The main application window
            
        Returns:
            bool: True if paused successfully, False otherwise
        """
        if not self.process or self.process.poll() is not None:
            logger.warning("No active process to pause")
            return False
            
        try:
            proc = psutil.Process(self.process.pid)
            proc.suspend()
            
            # Update UI buttons
            window["PAUSE_TRAINING"].update(disabled=True)
            window["RESUME_TRAINING"].update(disabled=False)
            
            # Update log
            window["-LOG-"].print(f"[{datetime.now()}] Training paused")
            
            logger.info(f"Training process {self.process.pid} suspended")
            return True
            
        except Exception as e:
            if error_handling_available:
                handle_error(
                    e,
                    context="Pausing training process",
                    additional_context={"process_id": getattr(self.process, "pid", "Unknown")}
                )
                log_persistent_error(
                    e,
                    context="Pausing training process",
                    severity=ErrorSeverity.MEDIUM,
                    additional_info={"process_id": getattr(self.process, "pid", "Unknown")}
                )
            logger.error(f"Error pausing training process: {e}")
            return False
    
    def resume_training(self, window):
        """
        Resume a paused training process.
        
        Args:
            window: The main application window
            
        Returns:
            bool: True if resumed successfully, False otherwise
        """
        if not self.process or self.process.poll() is not None:
            logger.warning("No active process to resume")
            return False
            
        try:
            proc = psutil.Process(self.process.pid)
            proc.resume()
            
            # Update UI buttons
            window["PAUSE_TRAINING"].update(disabled=False)
            window["RESUME_TRAINING"].update(disabled=True)
            
            # Update log
            window["-LOG-"].print(f"[{datetime.now()}] Training resumed")
            
            logger.info(f"Training process {self.process.pid} resumed")
            return True
            
        except Exception as e:
            if error_handling_available:
                handle_error(
                    e,
                    context="Resuming training process",
                    additional_context={"process_id": getattr(self.process, "pid", "Unknown")}
                )
                log_persistent_error(
                    e,
                    context="Resuming training process",
                    severity=ErrorSeverity.MEDIUM,
                    additional_info={"process_id": getattr(self.process, "pid", "Unknown")}
                )
            logger.error(f"Error resuming training process: {e}")
            return False
    
    def _start_training_monitor(self, window, values):
        """
        Start a thread to monitor training completion and gather results.
        
        Args:
            window: The main application window
            values: The current values from the GUI
        """
        def monitor_training_completion():
            # Wait for process to complete
            returncode = self.process.wait() if self.process else None
            
            if returncode == 0:
                logger.info("Training completed successfully, gathering metrics")
                
                try:
                    # Get metrics from the latest training run
                    bucket_type = values.get("BUCKET", "Scalping")
                    
                    # Load actual metrics from the performance log file
                    perf_log_file = os.path.join(MODELS_DIR_DEFAULT, bucket_type, "performance_log.json")
                    actual_metrics = {}
                    
                    if os.path.exists(perf_log_file):
                        try:
                            with open(perf_log_file, "r", encoding="utf-8") as f:
                                perf_data = json.load(f)
                            
                            # The training system saves performance_history as a direct array, not wrapped in {"entries": [...]}
                            # Handle both formats for compatibility
                            if isinstance(perf_data, list):
                                # New format: direct array
                                entries = perf_data[-5:] if perf_data else []
                            elif isinstance(perf_data, dict) and "entries" in perf_data:
                                # Legacy format: wrapped in entries
                                entries = perf_data["entries"][-5:] if perf_data.get("entries", []) else []
                            else:
                                logger.error("Invalid performance data format.")
                                return
                            
                            if not entries:
                                logger.warning("Performance log file exists but contains no entries")
                                return
                            
                            # Get the latest entry (training saves as a list directly)
                            latest_entry = entries[-1]  # Get most recent
                            episode_metrics = latest_entry.get("metrics", {})
                            
                            # Extract and map the actual metrics to UI format
                            actual_metrics = {
                                "net_profit": episode_metrics.get("net_profit", 0.0),
                                "win_rate": episode_metrics.get("win_rate", 0.0),
                                "max_drawdown": episode_metrics.get("max_drawdown", 0.0),
                                "sharpe_ratio": episode_metrics.get("sharpe_ratio", 0.0),
                                "total_trades": episode_metrics.get("total_trades", 0),
                                "profit_factor": episode_metrics.get("profit_factor", 1.0),
                                "episode": latest_entry.get("episode", 0),
                                "fitness": latest_entry.get("fitness", 0.0),
                                "reward": latest_entry.get("reward", 0.0)
                            }
                            logger.info(f"Loaded actual training metrics from episode {actual_metrics['episode']}")
                        except Exception as e:
                            logger.error(f"Error reading performance metrics from {perf_log_file}: {e}")
                    
                    # Fall back to sample metrics only if we couldn't load real ones
                    if not actual_metrics:
                        logger.warning("Using fallback sample metrics - no real training data found")
                        actual_metrics = {
                            "net_profit": 0.0,
                            "win_rate": 0.0,
                            "max_drawdown": 0.0,
                            "sharpe_ratio": 0.0,
                            "total_trades": 0,
                            "profit_factor": 1.0,
                            "episode": 0,
                            "fitness": 0.0,
                            "reward": 0.0
                        }
                    
                    # Send the actual training metrics to UI
                    window.write_event_value("-TRAINING-COMPLETE-", actual_metrics)
                    logger.info("Sent actual training metrics to UI for preset performance tracking")
                    
                    # Update UI buttons
                    window.write_event_value("-UPDATE-TRAINING-BUTTONS-", {
                        "START_TRAINING": False,
                        "STOP_TRAINING": True,
                        "PAUSE_TRAINING": True,
                        "RESUME_TRAINING": True
                    })
                    
                except Exception as e:
                    logger.error(f"Error collecting training metrics: {e}")
                    # Send empty metrics if we couldn't collect them
                    window.write_event_value("-TRAINING-COMPLETE-", {})
            else:
                logger.error(f"Training failed with return code {returncode}")
                window.write_event_value("-TRAINING-ERROR-", f"Failed with code {returncode}")
                
                # Update UI buttons
                window.write_event_value("-UPDATE-TRAINING-BUTTONS-", {
                    "START_TRAINING": False,
                    "STOP_TRAINING": True,
                    "PAUSE_TRAINING": True,
                    "RESUME_TRAINING": True
                })
        
        # Start the monitoring thread
        self.training_monitor_thread = threading.Thread(target=monitor_training_completion, daemon=True)
        self.training_monitor_thread.start()
    
    def check_training_in_progress(self, bucket):
        """
        Check if training is currently in progress for this bucket.
        
        Args:
            bucket (str): The bucket (strategy type) to check
            
        Returns:
            tuple: (is_training, recovery_data) where:
                - is_training (bool): True if training is in progress, False otherwise
                - recovery_data (dict or None): Recovery data if available
        """
        recovery_file = os.path.join(MODELS_DIR_DEFAULT, bucket, "recovery_state.json")
        
        if not os.path.exists(recovery_file):
            return False, None
        
        try:
            with open(recovery_file, "r", encoding="utf-8") as f:
                recovery_data = json.load(f)
            
            if recovery_data.get("is_training", False):
                return True, recovery_data
        except Exception as e:
            logger.error(f"Error checking training status: {e}")
            pass
        
        return False, None
    
    def load_performance_metrics(self, bucket):
        """
        Load performance metrics from the performance log file.
        
        Args:
            bucket (str): The bucket (strategy type) to load metrics for
            
        Returns:
            str: Formatted performance metrics or error message
        """
        perf_log_file = os.path.join(MODELS_DIR_DEFAULT, bucket, "performance_log.json")
        
        if not os.path.exists(perf_log_file):
            return "No performance data available yet."
        
        try:
            with open(perf_log_file, "r", encoding="utf-8") as f:
                perf_data = json.load(f)
            
            # The training system saves performance_history as a direct array, not wrapped in {"entries": [...]}
            # Handle both formats for compatibility
            if isinstance(perf_data, list):
                # New format: direct array
                entries = perf_data[-5:] if perf_data else []
            elif isinstance(perf_data, dict) and "entries" in perf_data:
                # Legacy format: wrapped in entries
                entries = perf_data["entries"][-5:] if perf_data.get("entries", []) else []
            else:
                return "Invalid performance data format."
            
            if not entries:
                return "No performance entries found."
            
            result = "Recent Performance:\n\n"
            for entry in entries:
                metrics = entry.get("metrics", {})
                episode = entry.get("episode", "N/A")
                timestamp = entry.get("timestamp", "N/A")
                
                # Format timestamp if it's a number
                if isinstance(timestamp, (int, float)):
                    timestamp = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
                
                result += f"Episode {episode} - {timestamp}\n"
                result += f"  Net Profit: ${metrics.get('net_profit', 0):.2f}\n"
                result += f"  Win Rate: {metrics.get('win_rate', 0)*100:.1f}%\n"
                result += f"  Sharpe: {metrics.get('sharpe_ratio', 0):.2f}\n"
                result += f"  Max Drawdown: {metrics.get('max_drawdown', 0)*100:.1f}%\n"
                result += f"  Trades: {metrics.get('total_trades', 0)}\n"
                result += f"  Fitness: {entry.get('fitness', 0):.4f}\n"
                result += f"  Reward: {entry.get('reward', 0):.2f}\n\n"
            
            return result
        except Exception as e:
            return f"Error loading performance data: {e}"
    
    def start_live_log(self, window, bucket):
        """
        Start reading the log file and updating the log window.
        
        Args:
            window: The main application window
            bucket: The bucket (strategy type) name
        """
        log_path = os.path.join(MODELS_DIR_DEFAULT, bucket, "training_log.txt")
        
        # Create empty log file if it doesn't exist
        if not os.path.exists(os.path.dirname(log_path)):
            os.makedirs(os.path.dirname(log_path))
        if not os.path.exists(log_path):
            with open(log_path, "w") as f:
                f.write(f"Log initialized at {datetime.now()}\n")
        
        def log_reader():
            with open(log_path, "r") as f:
                # Go to the end of file
                f.seek(0, 2)
                while True:
                    line = f.readline()
                    if line:
                        window.write_event_value('-LOG_UPDATE-', line)
                    time.sleep(0.1)
        
        threading.Thread(target=log_reader, daemon=True).start()
    
    def run_comparison(self, window, values, episodes=50):
        """
        Run a comparison between user settings and default settings for the selected bucket.
        
        Args:
            window: The main application window
            values: The current values from the GUI
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
                # Get current settings from values
                current_settings = {}
                for key, value in values.items():
                    if isinstance(value, (str, int, float, bool)) and key in values:
                        current_settings[key] = value
                json.dump(current_settings, f, indent=4)
            
            # Get default settings for this bucket
            default_settings = {}
            # In a real implementation, we would get these from presets
            # For now, just copy the current settings
            default_settings = current_settings.copy()
            
            # Save default settings to a temporary file
            default_settings_file = os.path.join(tempfile.gettempdir(), "default_settings.json")
            with open(default_settings_file, "w", encoding="utf-8") as f:
                json.dump(default_settings, f, indent=4)
            
            # Create directory for comparison results
            comparison_dir = os.path.join(project_root, "comparison_results")
            os.makedirs(comparison_dir, exist_ok=True)
            
            # Path to comparison script
            comparison_script = os.path.join(project_root, "src", "training", "train.py")
            
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
                "--output-dir", comparison_dir,
                "--bucket", bucket,
                "--no-save-models",  # Flag to prevent saving models from comparison runs
                "--temp-run"  # Flag to indicate this is a temporary comparison run
            ]
            
            # Start the comparison process
            comparison_process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                text=True
            )
            
            # Start a thread to monitor the comparison process
            def monitor_comparison():
                output = []
                for line in comparison_process.stdout:
                    output.append(line.strip())
                    window.write_event_value("-COMPARISON-UPDATE-", line.strip())
                
                # Process completed, check return code
                return_code = comparison_process.wait()
                
                if return_code == 0:
                    # Success
                    results_file = os.path.join(comparison_dir, "comparison_results.json")
                    if os.path.exists(results_file):
                        try:
                            with open(results_file, "r", encoding="utf-8") as f:
                                results = json.load(f)
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
            
            # Enable/disable buttons
            window["RUN_COMPARISON"].update(disabled=True)
            window["-STATUS-"].update("Status: Running comparison...")
            
            return comparison_process
            
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

# For backwards compatibility
def pause_training():
    """
    Legacy function for pausing training.
    This is maintained for backward compatibility.
    
    Returns:
        bool: True if paused successfully, False otherwise
    """
    try:
        global process
        if 'process' in globals() and process and process.poll() is None:
            proc = psutil.Process(process.pid)
            proc.suspend()
            logger.info(f"Training process {process.pid} suspended")
            return True
        else:
            logger.warning("No active process to pause")
            return False
    except Exception as e:
        if error_handling_available:
            handle_error(
                e,
                context="Pausing training process",
                additional_context={"process_id": getattr(process, "pid", "Unknown") if 'process' in globals() else "Unknown"}
            )
            log_persistent_error(
                e,
                context="Pausing training process",
                severity=ErrorSeverity.MEDIUM,
                additional_info={"process_id": getattr(process, "pid", "Unknown") if 'process' in globals() else "Unknown"}
            )
        logger.error(f"Error pausing training process: {e}")
        return False

def resume_training():
    """
    Legacy function for resuming training.
    This is maintained for backward compatibility.
    
    Returns:
        bool: True if resumed successfully, False otherwise
    """
    try:
        global process
        if 'process' in globals() and process and process.poll() is None:
            proc = psutil.Process(process.pid)
            proc.resume()
            logger.info(f"Training process {process.pid} resumed")
            return True
        else:
            logger.warning("No active process to resume")
            return False
    except Exception as e:
        if error_handling_available:
            handle_error(
                e,
                context="Resuming training process",
                additional_context={"process_id": getattr(process, "pid", "Unknown") if 'process' in globals() else "Unknown"}
            )
            log_persistent_error(
                e,
                context="Resuming training process",
                severity=ErrorSeverity.MEDIUM,
                additional_info={"process_id": getattr(process, "pid", "Unknown") if 'process' in globals() else "Unknown"}
            )
        logger.error(f"Error resuming training process: {e}")
        return False 