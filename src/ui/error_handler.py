import PySimpleGUI as sg
import logging
import os
import json
import traceback
from enum import Enum
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime

# Try to import logging utilities
try:
    from src.utils.log_manager import LogManager
    logger = LogManager.get_logger("error_handler")
except ImportError:
    # Fallback if imports fail
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("error_handler")

# Try to import persistent logger
try:
    from src.utils.persistent_logger import log_persistent_error
    persistent_logging_available = True
except ImportError:
    logger.warning("Persistent logging not available")
    persistent_logging_available = False
    # Create stub function
    def log_persistent_error(error, context="", severity="medium", additional_info=None):
        pass

class ErrorSeverity(Enum):
    """Enum for error severity levels"""
    LOW = "low"       # Minor issues, can continue
    MEDIUM = "medium" # Significant issues but can recover
    HIGH = "high"     # Critical issues requiring action
    FATAL = "fatal"   # Application cannot continue

# Common error types and user-friendly messages
ERROR_TEMPLATES = {
    "FileNotFoundError": {
        "message": "File not found: {file_path}",
        "suggestions": [
            "Check if the file exists at the specified location",
            "Verify you have the correct permissions to access the file",
            "Try restarting the application"
        ],
        "severity": ErrorSeverity.MEDIUM
    },
    "PermissionError": {
        "message": "Permission denied when accessing: {file_path}",
        "suggestions": [
            "Verify you have the correct permissions for this file/directory",
            "Try running the application as administrator",
            "Close any other applications that might be using the file"
        ],
        "severity": ErrorSeverity.MEDIUM
    },
    "ImportError": {
        "message": "Could not import required module: {module_name}",
        "suggestions": [
            "Check if the module is installed correctly",
            "Try reinstalling the application",
            "Verify your Python environment is set up correctly"
        ],
        "severity": ErrorSeverity.HIGH
    },
    "JSONDecodeError": {
        "message": "Invalid JSON format in file: {file_path}",
        "suggestions": [
            "The file is corrupted or in an invalid format",
            "Try restoring from a backup",
            "Delete the file and let the application recreate it"
        ],
        "severity": ErrorSeverity.MEDIUM
    },
    "TrainingError": {
        "message": "Error during training process: {error_msg}",
        "suggestions": [
            "Check the training log for details",
            "Verify your model configuration is valid",
            "Try a different set of parameters"
        ],
        "severity": ErrorSeverity.MEDIUM
    },
    "ConnectionError": {
        "message": "Connection error: {error_msg}",
        "suggestions": [
            "Check your internet connection",
            "Verify the server is accessible",
            "Try again later"
        ],
        "severity": ErrorSeverity.MEDIUM
    },
    "ValueError": {
        "message": "Invalid value: {error_msg}",
        "suggestions": [
            "Check the input values",
            "Verify the data format is correct",
            "Make sure the values are within acceptable ranges"
        ],
        "severity": ErrorSeverity.MEDIUM
    },
    "KeyError": {
        "message": "Missing key in dictionary: {error_msg}",
        "suggestions": [
            "Configuration may be incomplete",
            "Verify the required keys exist in the configuration",
            "Try resetting to default configuration"
        ],
        "severity": ErrorSeverity.MEDIUM
    },
    # Data loading specific errors
    "DataFileFormatError": {
        "message": "Invalid data file format: {file_path}",
        "suggestions": [
            "Check that the file is a valid CSV file",
            "Verify the file has not been corrupted",
            "Try downloading the data file again"
        ],
        "severity": ErrorSeverity.MEDIUM
    },
    "MissingColumnsError": {
        "message": "Data file missing required columns: {missing_columns}",
        "suggestions": [
            "Check that the data file contains all required columns",
            "Verify the data file format meets the application requirements",
            "Try using a different data source"
        ],
        "severity": ErrorSeverity.MEDIUM
    },
    "EmptyDatasetError": {
        "message": "No data available or empty dataset",
        "suggestions": [
            "Check that data files exist in the data directory",
            "Verify that data files contain valid data",
            "Try downloading new data files or specify a different data location"
        ],
        "severity": ErrorSeverity.HIGH
    },
    "DataValidationError": {
        "message": "Data validation failed: {error_msg}",
        "suggestions": [
            "Check that the data meets the required format and constraints",
            "Look for missing or invalid values in the data",
            "Try reprocessing the data or using a different dataset"
        ],
        "severity": ErrorSeverity.MEDIUM
    },
    "FeatureEngineeringError": {
        "message": "Error calculating features: {error_msg}",
        "suggestions": [
            "Check that required columns exist for feature calculation",
            "Verify that data values are valid for the calculation",
            "Try using a different feature engineering method"
        ],
        "severity": ErrorSeverity.MEDIUM
    },
    "TimestampConversionError": {
        "message": "Failed to convert timestamps: {error_msg}",
        "suggestions": [
            "Check that timestamp data is in a valid format",
            "Verify that the data doesn't contain malformed dates",
            "Try preprocessing the timestamp column before conversion"
        ],
        "severity": ErrorSeverity.MEDIUM
    },
    "MemoryError": {
        "message": "Not enough memory to process data: {error_msg}",
        "suggestions": [
            "Try processing a smaller subset of the data",
            "Close other applications to free up memory",
            "Increase system swap space or add more RAM",
            "Use a system with more memory resources"
        ],
        "severity": ErrorSeverity.HIGH
    },
    # Training specific errors
    "InvalidHyperparameterError": {
        "message": "Invalid hyperparameter: {parameter} = {value}",
        "suggestions": [
            "Check the hyperparameter value is within acceptable ranges",
            "Refer to the documentation for valid hyperparameter settings",
            "Try using default hyperparameter values instead"
        ],
        "severity": ErrorSeverity.MEDIUM
    },
    "ModelInitializationError": {
        "message": "Failed to initialize model: {error_msg}",
        "suggestions": [
            "Check input dimensions and hyperparameter configuration",
            "Verify you have enough GPU memory if using CUDA",
            "Try using a simpler model architecture"
        ],
        "severity": ErrorSeverity.HIGH
    },
    "GPUError": {
        "message": "GPU error: {error_msg}",
        "suggestions": [
            "Check your GPU drivers are up to date",
            "Try disabling mixed precision training",
            "Lower batch size or model size to reduce memory usage",
            "Switch to CPU training if persistent GPU issues occur"
        ],
        "severity": ErrorSeverity.HIGH
    },
    "NaNError": {
        "message": "NaN values detected during training: {error_msg}",
        "suggestions": [
            "Check for extreme values in your data",
            "Lower the learning rate",
            "Add gradient clipping to prevent exploding gradients",
            "Initialize model weights with a different strategy"
        ],
        "severity": ErrorSeverity.HIGH
    },
    "EnvironmentError": {
        "message": "Environment error: {error_msg}",
        "suggestions": [
            "Check the environment configuration",
            "Verify input data is formatted correctly for the environment",
            "Try using a different environment configuration"
        ],
        "severity": ErrorSeverity.MEDIUM
    },
    "CheckpointError": {
        "message": "Error saving or loading checkpoint: {error_msg}",
        "suggestions": [
            "Check disk space and write permissions",
            "Verify the checkpoint file is not corrupted",
            "Try saving to a different location"
        ],
        "severity": ErrorSeverity.MEDIUM
    },
    "KnowledgeTransferError": {
        "message": "Error during knowledge transfer: {error_msg}",
        "suggestions": [
            "Check model compatibility between buckets",
            "Verify both models have compatible architectures",
            "Try using a different transfer rate or method"
        ],
        "severity": ErrorSeverity.MEDIUM
    },
    "default": {
        "message": "An unexpected error occurred: {error_msg}",
        "suggestions": [
            "Check the application logs for more details",
            "Try restarting the application",
            "If the problem persists, try restoring from a backup"
        ],
        "severity": ErrorSeverity.MEDIUM
    }
}

# Recovery actions
RECOVERY_ACTIONS = {
    "retry": lambda ctx: ctx.get("retry_func", lambda: None)(),
    "restore_backup": lambda ctx: restore_from_backup(ctx.get("file_path")),
    "recreate_file": lambda ctx: recreate_file(ctx.get("file_path"), ctx.get("template")),
    "restart_app": lambda ctx: restart_application(),
}

# Error state tracking
error_history = []
last_error = None

def handle_error(error: Exception, context: str = "", window: Any = None, 
                retry_func: Callable = None, additional_context: Dict = None) -> Dict:
    """
    Central error handling function
    
    Args:
        error: The exception that occurred
        context: Description of where the error occurred
        window: PySimpleGUI window if available
        retry_func: Function to call if retry is requested
        additional_context: Extra information about the error
        
    Returns:
        Dict with error details and how it was handled
    """
    global last_error, error_history
    
    # Log the exception
    logger.error(f"Error in {context}: {str(error)}")
    logger.debug(traceback.format_exc())
    
    # Prepare error context
    error_type = error.__class__.__name__
    error_context = additional_context or {}
    error_context.update({
        "error_type": error_type,
        "error_msg": str(error),
        "timestamp": datetime.now().isoformat(),
        "context": context,
        "retry_func": retry_func
    })
    
    # Get template
    template = ERROR_TEMPLATES.get(error_type, ERROR_TEMPLATES["default"])
    
    # Format message
    try:
        message = template["message"].format(**error_context)
    except KeyError:
        message = f"{template['message']} - {error_context['error_msg']}"
    
    # Determine severity
    severity = template["severity"]
    
    # Create error details
    error_details = {
        "type": error_type,
        "message": message,
        "context": context,
        "severity": severity,
        "suggestions": template["suggestions"],
        "timestamp": error_context["timestamp"],
        "handled": False
    }
    
    # Add to history
    error_history.append(error_details)
    last_error = error_details
    
    # Log to persistent error log if available
    if persistent_logging_available:
        severity_str = severity.value if isinstance(severity, ErrorSeverity) else str(severity)
        log_persistent_error(
            error, 
            context, 
            severity_str,
            additional_info=error_context
        )
    
    # Show error to user if window is available
    if window:
        show_error_dialog(window, error_details, retry_func, error_context)
        error_details["handled"] = True
    
    return error_details

def show_error_dialog(window, error_details, retry_func=None, error_context=None):
    """
    Show error dialog to the user
    
    Args:
        window: The PySimpleGUI window
        error_details: Dictionary with error details
        retry_func: Function to call for retry action
        error_context: Additional context about the error
    """
    severity = error_details["severity"]
    if isinstance(severity, str):
        severity_value = severity
    else:
        severity_value = severity.value if hasattr(severity, "value") else str(severity)
    
    color = {
        "low": "#FFD700",     # Gold
        "medium": "#FFA500",  # Orange
        "high": "#FF4500",    # Red-Orange
        "fatal": "#FF0000"    # Red
    }.get(severity_value, "#FFA500")
    
    # Create layout
    layout = [
        [sg.Text(f"❌ Error: {error_details['message']}", 
                 text_color=color, font=("Helvetica", 12, "bold"))],
        [sg.HorizontalSeparator()],
        [sg.Text("Suggestions:", font=("Helvetica", 10, "bold"))],
    ]
    
    # Add suggestions
    for i, suggestion in enumerate(error_details["suggestions"]):
        layout.append([sg.Text(f"• {suggestion}")])
    
    layout.append([sg.HorizontalSeparator()])
    
    # Add recovery options based on severity
    buttons = []
    if severity != ErrorSeverity.FATAL:
        if retry_func:
            buttons.append(sg.Button("Retry", key="-RETRY-"))
        
        # Add relevant recovery options
        if error_context and "file_path" in error_context:
            if severity != ErrorSeverity.HIGH:
                buttons.append(sg.Button("Restore Backup", key="-RESTORE-"))
                buttons.append(sg.Button("Recreate File", key="-RECREATE-"))
    
    # Always add Close button
    buttons.append(sg.Button("Close", key="-CLOSE-"))
    
    # Add buttons to layout
    layout.append([sg.Column([buttons], justification='right')])
    
    # Create window
    error_window = sg.Window("Error", layout, modal=True, finalize=True)
    
    # Event loop
    while True:
        event, values = error_window.read()
        if event in (None, "-CLOSE-"):
            break
        elif event == "-RETRY-" and retry_func:
            error_window.close()
            retry_func()
            return
        elif event == "-RESTORE-" and error_context and "file_path" in error_context:
            error_window.close()
            restore_from_backup(error_context["file_path"])
            return
        elif event == "-RECREATE-" and error_context and "file_path" in error_context:
            error_window.close()
            recreate_file(error_context["file_path"], 
                         error_context.get("template"))
            return
    
    error_window.close()

def restore_from_backup(file_path):
    """
    Attempt to restore a file from backup
    
    Args:
        file_path: Path to the file to restore
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import shutil
        
        # Check for direct backup
        backup_path = f"{file_path}.backup"
        if os.path.exists(backup_path):
            shutil.copy(backup_path, file_path)
            logger.info(f"Restored {file_path} from backup")
            return True
        
        # Check for timestamped backups
        backup_dir = os.path.join(os.path.dirname(file_path), "backups")
        if os.path.exists(backup_dir):
            file_name = os.path.basename(file_path)
            # Get all backups for this file
            import glob
            backups = glob.glob(os.path.join(backup_dir, f"{file_name}.*.backup"))
            
            if backups:
                # Sort by timestamp (newest first)
                backups.sort(reverse=True)
                # Restore from newest backup
                shutil.copy(backups[0], file_path)
                logger.info(f"Restored {file_path} from {backups[0]}")
                return True
        
        logger.warning(f"No backup found for {file_path}")
        return False
    except Exception as e:
        logger.error(f"Failed to restore from backup: {e}")
        return False

def recreate_file(file_path, template=None):
    """
    Recreate a file with default template
    
    Args:
        file_path: Path to the file to recreate
        template: Optional template data for the file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # If it's a JSON file and we have a template
        if file_path.endswith('.json') and template:
            with open(file_path, 'w') as f:
                json.dump(template, f, indent=4)
        # Otherwise create empty file
        else:
            with open(file_path, 'w') as f:
                f.write('')
        
        logger.info(f"Recreated file: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to recreate file: {e}")
        return False

def restart_application():
    """
    Restart the application
    
    Returns:
        bool: True if successful, False otherwise
    """
    # This needs to be implemented based on how your app is structured
    logger.info("Application restart requested but not implemented")
    
    # In a real implementation, we would:
    # 1. Save current application state
    # 2. Launch a new process
    # 3. Exit the current process
    
    # Here's a placeholder that could be expanded:
    try:
        import sys
        import subprocess
        
        # Get the executable path and arguments
        python = sys.executable
        script = sys.argv[0]
        args = sys.argv[1:]
        
        # Prepare restart command
        restart_cmd = [python, script] + args
        
        # Log the restart attempt
        logger.info(f"Attempting to restart with: {restart_cmd}")
        
        # Start new process
        subprocess.Popen(restart_cmd)
        
        # Schedule exit
        import threading
        threading.Timer(1.0, lambda: sys.exit(0)).start()
        
        return True
    except Exception as e:
        logger.error(f"Failed to restart application: {e}")
        return False

def get_last_error():
    """Get the most recent error"""
    return last_error

def get_error_history():
    """Get the full error history"""
    return error_history

def clear_error_history():
    """Clear the error history"""
    global error_history, last_error
    error_history = []
    last_error = None

# Testing function
def test_error_handler():
    """Test the error handler with a sample error"""
    try:
        raise ValueError("This is a test error")
    except Exception as e:
        error_details = handle_error(e, "test function")
        print(f"Error handled: {error_details['message']}")
        return error_details 