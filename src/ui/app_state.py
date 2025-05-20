"""
Application State Management for RL Trader Parameter Tuner

This module manages the application state, including configuration loading/saving,
backup creation, and state recovery mechanisms.
"""

import os
import sys
import json
import time
import glob
import logging
import hashlib
import shutil
import subprocess
from datetime import datetime

import PySimpleGUI as sg

# Make sure we can import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logger
logger = logging.getLogger(__name__)

# Import layouts to use the create_layout function
from src.ui.layouts import create_layout, create_combined_tabs

# Try to import trade_config
try:
    from src.utils.trade_config import trade_config
    trade_config_available = True
except ImportError:
    trade_config_available = False
    logger.warning("trade_config module not available, using local configuration")

# Try to import config_compatibility
try:
    from src.utils.config_compatibility import ConfigCompatibility as config_compatibility
except ImportError:
    # Create a stub config_compatibility if not available
    class ConfigCompatibilityStub:
        @staticmethod
        def adapt_state_version(state, state_version, app_version):
            logger.warning(f"No version adaptation available for {state_version} to {app_version}")
            return state
            
        @staticmethod
        def check_config_compatibility(config, app_version):
            return True, config
    
    config_compatibility = ConfigCompatibilityStub()
    logger.warning("config_compatibility module not available, using stub implementation")

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

# Import notes manager for loading/saving notes
from src.ui.notes_manager import load_notes, save_notes, notes_content

# Constants
VERSION = "1.0.0"  # Application version
CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "config", "config.json")
MODELS_DIR_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "models")

# Try to import checkpoint validation if available
try:
    from src.utils.checkpoint_validation import verify_checkpoint_integrity, sandbox_load_checkpoint
    checkpoint_validation_available = True
except ImportError:
    checkpoint_validation_available = False

# --- Add Helper Function for Time Conversion ---
def _convert_time_to_bars(amount, unit, bar_size_minutes=5):
    """
    Convert time units to bar counts.
    Adapted from bucket_manager.py
    """
    try:
        amount = float(amount)
        minutes = 0
        if unit == 'hour(s)':
            minutes = amount * 60
        elif unit == 'day(s)':
            minutes = amount * 24 * 60
        elif unit == 'week(s)':
            minutes = amount * 7 * 24 * 60
        elif unit == 'month(s)':
            # Approximate - using 30 days per month
            minutes = amount * 30 * 24 * 60
        else: # Handle case where unit might be missing or invalid
             logger.warning(f"Invalid time unit '{unit}' provided for look-back.")
             # Attempt to treat amount as raw bars if possible, else default
             try: return max(1, int(amount))
             except: return 288 # Default to 1 day
        
        bars = int(minutes / bar_size_minutes)
        return max(1, bars)  # Ensure at least 1 bar
    except (ValueError, TypeError):
        logger.error(f"Error converting time to bars: invalid amount '{amount}' or unit '{unit}'")
        return 288  # Default to 1 day (288 5-min bars) 
# --- End Helper Function ---

class AppState:
    """Class to manage application state."""
    def __init__(self):
        self.window = None
        self.process = None
        self.last_run_time = None
        self.potentially_corrupted = False
        self.current_training_preset_id = None
        self.load_config()
        
        # Initialize notes
        global notes_content
        if not notes_content:
            notes_content = load_notes()
        
        logger.info("AppState initialized with configuration and notes")
    
    def load_config(self):
        """Load configuration from TradeConfig."""
        try:
            # Get configuration from TradeConfig
            if trade_config_available:
                self.config = trade_config.as_dict()
                logger.info("Loaded configuration from TradeConfig")
            else:
                # Use default configuration as fallback
                # This should be updated to use a real default config
                self.config = {}
                logger.warning("Using empty configuration - TradeConfig not available")
        except Exception as e:
            logger.error(f"Error loading configuration from TradeConfig: {e}")
            # Use default configuration as fallback on any error
            self.config = {}
            logger.warning("Using empty configuration due to error")
    
    def save_config(self):
        """Save current configuration to TradeConfig."""
        try:
            # Update TradeConfig with current values
            if trade_config_available:
                trade_config.update(self.config)
                # Save to file
                trade_config.save()
                logger.info("Saved configuration to TradeConfig")
            else:
                # Save to a local config file as fallback
                config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "config", "config.json")
                os.makedirs(os.path.dirname(config_file), exist_ok=True)
                with open(config_file, 'w') as f:
                    json.dump(self.config, f, indent=4)
                logger.info(f"Saved configuration to local file: {config_file}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def create_window(self):
        """Create and initialize the main window."""
        from src.ui.layouts import create_combined_tabs
        
        # Get tabs from layouts module
        tabs = create_combined_tabs()
        
        # Load notes content
        global notes_content
        if not notes_content:
            notes_content = load_notes()
        
        # Create the main window with tabs and notes content
        self.window = sg.Window("BTC AI Trading Agent - Training Interface", 
                              create_layout(tabs, notes_content), 
                              resizable=True, 
                              finalize=True)
        return self.window
    
    def stop_training(self):
        """Stop the training process."""
        if self.process:
            try:
                self.process.terminate()
                try:
                    # Wait for process to terminate cleanly
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # If timeout, force kill
                    self.process.kill()
                logger.info("Training process terminated")
            except Exception as e:
                if error_handling_available:
                    handle_error(
                        e, 
                        context="Stopping training process",
                        additional_context={"process_id": getattr(self.process, "pid", "Unknown")}
                    )
                logger.error(f"Error stopping training process: {e}")
            finally:
                self.process = None
            
    def save_all_state(self, values=None):
        """Save all application state, including notes and config."""
        if values:
            # --- Calculate and update window_size in config --- 
            try:
                look_back_amount = values.get('LOOK_BACK_AMOUNT', 1)
                look_back_unit = values.get('LOOK_BACK_UNIT', 'day(s)')
                calculated_window_size = _convert_time_to_bars(look_back_amount, look_back_unit)
                self.config['window_size'] = calculated_window_size
                logger.info(f"Calculated and updated window_size: {calculated_window_size} bars from {look_back_amount} {look_back_unit}")
            except Exception as calc_e:
                logger.error(f"Error calculating window_size from UI values: {calc_e}. Using existing or default.")
            
            # --- Calculate and update MAX_VOLUME_PERCENTAGE in config --- 
            try:
                display_pct = float(values.get('MAX_VOLUME_PERCENTAGE_DISPLAY', 5.0)) # Get UI value (e.g., 5.0)
                decimal_val = display_pct / 100.0 # Convert to decimal (e.g., 0.05)
                self.config['MAX_VOLUME_PERCENTAGE'] = decimal_val # Save under correct key
                logger.info(f"Converted and updated MAX_VOLUME_PERCENTAGE: {decimal_val} from display value {display_pct}")
            except (ValueError, TypeError, KeyError) as conv_e:
                logger.error(f"Error converting MAX_VOLUME_PERCENTAGE_DISPLAY: {conv_e}. Using existing or default.")
            # --- End MAX_VOLUME_PERCENTAGE update ---
            
            # Update self.config with other values from the UI
            # Ensure this happens if needed, or confirm it happens elsewhere
            # Example (if not handled by other events):
            # for key, value in values.items():
            #     if key in self.config: # Only update existing keys?
            #         # Add type conversion logic if necessary
            #         self.config[key] = value 
            pass # Assuming self.config is updated by other UI events

        # Save configuration (now including updated window_size and MAX_VOLUME_PERCENTAGE)
        self.save_config()
        
        # Save notes if values are provided
        if values and "-NOTES-" in values:
            save_notes(values["-NOTES-"])
        
        # Create an emergency backup in case of failure during next load
        self._create_emergency_state_backup()
        
        logger.info("Saved all application state")
    
    def _create_emergency_state_backup(self):
        """Create an emergency backup of essential application state."""
        try:
            # Try to import the emergency checkpoint module
            try:
                from src.utils.emergency_checkpoint import create_emergency_checkpoint
                emergency_module_available = True
            except ImportError:
                emergency_module_available = False
                logger.warning("Emergency checkpoint module not available, using basic backup")
            
            # Create state structure
            state = {
                "version": VERSION,
                "last_run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "emergency_backup": True,
                "config": self.config.copy()  # Include the full config
            }
            
            # Try to add notes content if available
            global notes_content
            if notes_content:
                state["notes"] = notes_content
            
            # List of critical files to include in the backup
            critical_files = [
                os.path.join(os.path.dirname(CONFIG_FILE), "app_state.json"),
                CONFIG_FILE
            ]
            
            # Add checkpoint files if they exist
            checkpoints_dir = os.path.join(self.config.get("MODELS_DIR", MODELS_DIR_DEFAULT), "checkpoints")
            if os.path.exists(checkpoints_dir):
                # Find the latest checkpoint file
                checkpoint_files = glob.glob(os.path.join(checkpoints_dir, "*.pt"))
                if checkpoint_files:
                    # Sort by modification time (newest first)
                    checkpoint_files.sort(key=os.path.getmtime, reverse=True)
                    # Add the latest checkpoint file
                    critical_files.append(checkpoint_files[0])
                    logger.info(f"Including latest checkpoint file in emergency backup: {checkpoint_files[0]}")
            
            # If enhanced emergency checkpoint module is available, use it
            if emergency_module_available:
                checkpoint_note = f"Auto-created before application state load at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                result = create_emergency_checkpoint(state, checkpoint_note, critical_files)
                
                if result["success"]:
                    logger.info(f"Created enhanced emergency checkpoint: {result['checkpoint_id']}")
                    return True
                else:
                    logger.error(f"Failed to create enhanced emergency checkpoint: {result.get('error', 'unknown error')}")
                    # Fall back to basic backup
            
            # Basic emergency backup (if module not available or failed)
            emergency_file = os.path.join(os.path.dirname(CONFIG_FILE), "app_state.emergency.json")
            
            # Create a timestamped backup as well
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            timestamped_file = os.path.join(os.path.dirname(CONFIG_FILE), f"app_state.emergency.{timestamp}.json")
            
            # Save to both files
            with open(emergency_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=4)
                
            with open(timestamped_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=4)
                
            logger.info(f"Created basic emergency state backup at {emergency_file} and {timestamped_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create emergency state backup: {e}")
            return False
            
    def create_checkpoint(self, note="", notify=True):
        """
        Create a checkpoint of the current application state for disaster recovery.
        
        Args:
            note (str): Note to attach to the checkpoint
            notify (bool): Whether to notify the user of checkpoint creation
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Try to import the emergency checkpoint module
            try:
                from src.utils.emergency_checkpoint import create_emergency_checkpoint
                emergency_module_available = True
            except ImportError:
                emergency_module_available = False
                logger.warning("Emergency checkpoint module not available, using basic checkpoint")
            
            # Create state structure with full application state
            state = {
                "version": VERSION,
                "last_run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "checkpoint_time": datetime.now().isoformat(),
                "manual_checkpoint": True,
                "note": note,
                "config": self.config.copy()
            }
            
            # Try to add notes content if available
            global notes_content
            if notes_content:
                state["notes"] = notes_content
            
            # List of critical files to include in the backup
            critical_files = [
                os.path.join(os.path.dirname(CONFIG_FILE), "app_state.json"),
                CONFIG_FILE
            ]
            
            # Add checkpoint files
            checkpoints_dir = os.path.join(self.config.get("MODELS_DIR", MODELS_DIR_DEFAULT), "checkpoints")
            if os.path.exists(checkpoints_dir):
                # Include all checkpoint files from the last 24 hours
                checkpoint_files = glob.glob(os.path.join(checkpoints_dir, "*.pt"))
                one_day_ago = time.time() - (24 * 60 * 60)
                recent_checkpoints = [f for f in checkpoint_files if os.path.getmtime(f) > one_day_ago]
                
                if recent_checkpoints:
                    critical_files.extend(recent_checkpoints)
                    logger.info(f"Including {len(recent_checkpoints)} recent checkpoint files in backup")
            
            # If enhanced emergency checkpoint module is available, use it
            if emergency_module_available:
                checkpoint_note = note or f"Manual checkpoint created at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                result = create_emergency_checkpoint(state, checkpoint_note, critical_files)
                
                if result["success"]:
                    logger.info(f"Created application checkpoint: {result['checkpoint_id']}")
                    
                    # Notify user if requested
                    if notify and self.window:
                        sg.popup_quick_message(
                            f"Application checkpoint created:\n{result['checkpoint_id']}",
                            auto_close_duration=3,
                            title="Checkpoint Created"
                        )
                    return True
                else:
                    logger.error(f"Failed to create application checkpoint: {result.get('error', 'unknown error')}")
                    
                    # Notify user if requested
                    if notify and self.window:
                        sg.popup_error(f"Failed to create checkpoint: {result.get('error', 'unknown error')}")
                    
                    # Fall back to basic backup
            
            # Basic checkpoint (if module not available or failed)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_dir = os.path.join(os.path.dirname(CONFIG_FILE), "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            checkpoint_file = os.path.join(checkpoint_dir, f"app_state.checkpoint.{timestamp}.json")
            
            # Add timestamp to state
            state["timestamp"] = timestamp
            
            # Save checkpoint
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=4)
                
            logger.info(f"Created basic application checkpoint at {checkpoint_file}")
            
            # Notify user if requested
            if notify and self.window:
                sg.popup_quick_message(
                    f"Basic application checkpoint created",
                    auto_close_duration=3,
                    title="Checkpoint Created"
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create application checkpoint: {e}")
            
            # Notify user if requested
            if notify and self.window:
                sg.popup_error(f"Failed to create checkpoint: {str(e)}")
                
            return False
    
    def _sandbox_load_state(self, state_file):
        """
        Load state in a sandbox to validate before committing to real application state.
        
        Args:
            state_file (str): Path to the state file
            
        Returns:
            dict: Result of sandbox loading with keys 'success', 'state', and 'error'
        """
        result = {
            "success": False,
            "state": None,
            "error": None,
            "validation_warnings": []
        }
        
        try:
            # First use the checkpoint validation module if available
            if checkpoint_validation_available:
                try:
                    # Verify file integrity using the enhanced module
                    integrity_result = verify_checkpoint_integrity(state_file)
                    if not integrity_result["success"]:
                        result["error"] = f"State file integrity check failed: {integrity_result['error']}"
                        result["validation_warnings"].append(integrity_result["error"])
                        logger.warning(f"State file integrity check failed: {integrity_result['error']}")
                        # We'll continue with traditional loading despite the integrity warning
                    
                    # Try sandbox loading with the validation module
                    sandbox_result = sandbox_load_checkpoint(state_file)
                    if sandbox_result["success"]:
                        result["state"] = sandbox_result["data"]
                        # Store warnings but continue
                        if "warnings" in sandbox_result and sandbox_result["warnings"]:
                            result["validation_warnings"].extend(sandbox_result["warnings"])
                            for warning in sandbox_result["warnings"]:
                                logger.warning(f"State validation warning: {warning}")
                        
                        # Basic validation still needs to be performed
                        # Continue to standard validation below...
                    else:
                        logger.warning(f"Enhanced sandbox loading failed: {sandbox_result.get('error', 'Unknown error')}")
                        # Fall back to traditional loading
                
                except Exception as e:
                    logger.warning(f"Error during enhanced state validation: {e}")
                    # Continue with traditional loading
            
            # If enhanced validation didn't succeed, try traditional loading
            if result["state"] is None:
                # Try to load and parse the state file
                with open(state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
                result["state"] = state
            
            # Always perform basic validation, even if we loaded via the enhanced module
            state = result["state"]
            
            # Basic validation
            required_keys = ["config"]  # Make "bucket" optional
            for key in required_keys:
                if key not in state:
                    warning = f"Missing required key: {key}"
                    result["validation_warnings"].append(warning)
                    logger.warning(warning)
                    # We don't fail immediately, collect all warnings
            
            # Verify config is a dictionary
            if not isinstance(state.get("config", None), dict):
                warning = "Config is not a dictionary"
                result["validation_warnings"].append(warning)
                logger.warning(warning)
                # This is more serious - we can't proceed without a valid config
                if not result["error"]:  # Only set if not already set by enhanced validation
                    result["error"] = warning
                return result
            
            # Check for critical configuration keys
            critical_keys = ["MODELS_DIR", "DATA_DIR", "LOG_DIR"]  # BUCKET is now optional
            missing_critical = []
            for key in critical_keys:
                if key not in state["config"]:
                    missing_critical.append(key)
            
            if missing_critical:
                warning = f"Missing critical configuration keys: {', '.join(missing_critical)}"
                result["validation_warnings"].append(warning)
                logger.warning(warning)
                # Not fatal - we'll use defaults for these
            
            # Validate version compatibility
            app_version = VERSION
            state_version = state.get("version", "0.0.1")  # Default to 0.0.1 if not specified
            
            if app_version != state_version:
                warning = f"State file version ({state_version}) differs from application version ({app_version})"
                result["validation_warnings"].append(warning)
                logger.warning(warning)
                # Not fatal - we'll try to adapt the version
            
            # Verify checksum if present
            if "checksum" in state and "config" in state:
                calculated_checksum = hashlib.md5(json.dumps(state["config"], sort_keys=True).encode()).hexdigest()
                if calculated_checksum != state["checksum"]:
                    warning = "State file checksum validation failed. Configuration may be corrupted."
                    result["validation_warnings"].append(warning)
                    logger.warning(warning)
            
            # Check if bucket is valid
            if "bucket" in state and state["bucket"] not in ["Scalping", "Short", "Medium", "Long"]:
                warning = f"Invalid bucket type: {state['bucket']}"
                result["validation_warnings"].append(warning)
                logger.warning(warning)
            
            # If we have only warnings but no errors, the load is still successful
            if not result["error"]:
                result["success"] = True
            
        except json.JSONDecodeError as e:
            result["error"] = f"JSON parsing error: {str(e)}"
            logger.error(result["error"])
        except Exception as e:
            result["error"] = f"Unexpected error: {str(e)}"
            logger.error(result["error"])
        
        return result
        
    def load_state(self):
        """Load application state from the app_state.json file."""
        try:
            # Construct path to app_state.json
            state_file = os.path.join(os.path.dirname(CONFIG_FILE), "app_state.json")
            
            # Check if the file exists
            if not os.path.exists(state_file):
                logger.warning(f"Application state file {state_file} does not exist. Starting with default state.")
                return False
                
            # First perform a sandbox load to validate the state
            sandbox_result = self._sandbox_load_state(state_file)
            if not sandbox_result["success"]:
                logger.error(f"Sandbox loading of state failed: {sandbox_result['error']}")
                
                # Try to load from emergency backup
                return self._load_emergency_state()
                
            # If sandbox loading successful, proceed with actual loading
            state = sandbox_result["state"]
            
            # Notify user of any validation warnings
            if "validation_warnings" in sandbox_result and sandbox_result["validation_warnings"]:
                self._notify_user_of_validation_issues(sandbox_result["validation_warnings"], self.window)
                
            # Get current application version
            app_version = VERSION
            state_version = state.get("version", "0.0.1")  # Default to 0.0.1 if not specified
            
            # Check if versions are compatible
            if app_version != state_version:
                logger.warning(f"State file version ({state_version}) differs from application version ({app_version})")
                
                # Create a backup of the state file before attempting to load it
                backup_file = f"{state_file}.{state_version}.bak"
                try:
                    shutil.copy2(state_file, backup_file)
                    logger.info(f"Created backup of state file at {backup_file}")
                except Exception as e:
                    logger.error(f"Failed to create backup of state file: {e}")
                
                # Attempt to adapt the state to current version
                if hasattr(config_compatibility, "adapt_state_version"):
                    try:
                        state = config_compatibility.adapt_state_version(state, state_version, app_version)
                        logger.info(f"Successfully adapted state from version {state_version} to {app_version}")
                    except Exception as e:
                        logger.error(f"Failed to adapt state version: {e}")
                        # Continue with loading what we can
            
            # Verify checksum if present
            if "checksum" in state and "config" in state:
                calculated_checksum = hashlib.md5(json.dumps(state["config"], sort_keys=True).encode()).hexdigest()
                if calculated_checksum != state["checksum"]:
                    logger.warning("State file checksum validation failed. Configuration may be corrupted.")
                    # Continue loading but set a flag that can be checked later
                    self.potentially_corrupted = True
                    # Notify user of potential corruption
                    if self.window:
                        sg.popup_warning("The configuration may be corrupted. Some settings may be incorrect.\n"
                                       "Please verify all configuration values.", title="Checksum Validation Failed")
            
            # Update config with saved state
            if "config" in state:
                # First check if the config is compatible with current version
                try:
                    # Use any available config compatibility utils
                    if hasattr(config_compatibility, "check_config_compatibility"):
                        saved_config = state["config"]
                        compatible, fixed_config = config_compatibility.check_config_compatibility(saved_config, app_version)
                        
                        if not compatible:
                            logger.warning("Saved configuration contains incompatible settings")
                            # Use the fixed version if available
                            if fixed_config:
                                saved_config = fixed_config
                                logger.info("Using compatibility-fixed configuration")
                                # Notify user of compatibility fixes
                                if self.window:
                                    sg.popup_quick_message("Some configuration settings were automatically fixed for compatibility.",
                                                        auto_close_duration=5, title="Configuration Adjusted")
                            
                        # Now merge with current config
                        current_config = self.config.copy()
                        
                        # For each key in saved_config, update the current_config
                        for key, value in saved_config.items():
                            current_config[key] = value
                        
                        # Update the local config
                        self.config = current_config
                        
                        # Update TradeConfig with the loaded configuration
                        if trade_config_available:
                            trade_config.update(self.config)
                    else:
                        # Fall back to simple merging if compatibility check isn't available
                        current_config = self.config.copy()
                        saved_config = state["config"]
                        
                        # Merge configs
                        for key, value in saved_config.items():
                            current_config[key] = value
                        
                        # Update configs
                        self.config = current_config
                        if trade_config_available:
                            trade_config.update(self.config)
                except Exception as e:
                    logger.error(f"Error updating configuration: {e}")
                    if self.window:
                        sg.popup_error(f"Error updating configuration: {e}\nUsing partial configuration.", title="Configuration Error")
                    # Continue with what we have
                
            # Update bucket selection
            if "bucket" in state:
                bucket = state["bucket"]
                # Validate bucket type
                valid_buckets = ["Scalping", "Short", "Medium", "Long"]
                if bucket in valid_buckets:
                    self.config["BUCKET"] = bucket
                else:
                    logger.warning(f"Invalid bucket type '{bucket}' in state file. Using default bucket.")
                    if self.window:
                        sg.popup_warning(f"Invalid bucket type '{bucket}' in configuration.\nUsing default bucket.", title="Invalid Bucket")
                
            # Update models directory
            if "models_dir" in state:
                models_dir = state["models_dir"]
                # Verify the directory exists
                if os.path.isdir(models_dir):
                    self.config["MODELS_DIR"] = models_dir
                else:
                    logger.warning(f"Models directory {models_dir} does not exist. Using default directory.")
                    # Try to create the directory
                    try:
                        os.makedirs(models_dir, exist_ok=True)
                        logger.info(f"Created missing models directory: {models_dir}")
                        self.config["MODELS_DIR"] = models_dir
                    except Exception as dir_error:
                        logger.error(f"Failed to create models directory: {dir_error}")
            
            # Store last run time for diagnostics
            if "last_run_time" in state:
                self.last_run_time = state["last_run_time"]
                logger.info(f"Last application run time: {self.last_run_time}")
            
            logger.info(f"Application state loaded from {state_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading application state: {e}")
            # Consider loading a safe default or emergency backup state
            try:
                self._load_emergency_state()
            except Exception as backup_error:
                logger.error(f"Failed to load emergency state: {backup_error}")
            return False
        
    def _load_emergency_state(self):
        """Attempt to load an emergency backup state file if present."""
        emergency_file = os.path.join(os.path.dirname(CONFIG_FILE), "app_state.emergency.json")
        if os.path.exists(emergency_file):
            try:
                with open(emergency_file, "r", encoding="utf-8") as f:
                    emergency_state = json.load(f)
                
                # Only load essential settings to restore basic functionality
                if "config" in emergency_state and isinstance(emergency_state["config"], dict):
                    # Only merge these essential keys
                    essential_keys = ["MODELS_DIR", "DATA_DIR", "LOG_DIR"]  # BUCKET is now optional
                    for key in essential_keys:
                        if key in emergency_state["config"]:
                            self.config[key] = emergency_state["config"][key]
                            
                    # Only add BUCKET if it exists in emergency state
                    if "BUCKET" in emergency_state["config"]:
                        self.config["BUCKET"] = emergency_state["config"]["BUCKET"]
                            
                logger.info("Loaded emergency state configuration")
                return True
            except Exception as e:
                logger.error(f"Failed to load emergency state file: {e}")
                return False
        else:
            logger.warning("No emergency state file found")
            return False

    def _notify_user_of_validation_issues(self, validation_warnings, window=None):
        """
        Notify the user of validation issues with the state.
        
        Args:
            validation_warnings (list): List of validation warnings
            window: PySimpleGUI window to use for notifications (optional)
        """
        if not validation_warnings:
            return
            
        # Log all warnings
        for warning in validation_warnings:
            logger.warning(f"State validation warning: {warning}")
        
        # Prepare a user-friendly message
        if len(validation_warnings) == 1:
            message = f"Warning: {validation_warnings[0]}"
        else:
            message = f"There were {len(validation_warnings)} warnings during state validation:\n\n"
            for i, warning in enumerate(validation_warnings, 1):
                message += f"{i}. {warning}\n"
            
            message += "\nThe application will use safe defaults where needed."
        
        # Show popup if window is available
        if window:
            sg.popup_warning(message, title="State Validation Warnings")
        else:
            # Just log if no window is available
            logger.warning(message) 