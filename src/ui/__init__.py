"""
User Interface Package

This package provides GUI and visualization components for the BTC-AI system.
"""

# Import and re-export main components
from .main import main as run_main_ui
from src.ui.monitor_training import TrainingMonitor
from .create_preview import create_dashboard_preview, create_icon

# Import error handling (will be no-op if not available)
try:
    from .error_handler import (
        handle_error, 
        show_error_dialog, 
        ErrorSeverity, 
        get_last_error, 
        get_error_history
    )
    error_handling_available = True
except ImportError:
    import logging
    logging.getLogger(__name__).warning("Error handling module not available")
    error_handling_available = False

# Try to import persistent logging utilities
try:
    from src.utils.persistent_logger import (
        log_persistent_error,
        export_error_logs,
        get_log_directory_path,
        open_log_directory,
        create_log_locations_file
    )
    persistent_logging_available = True
except ImportError:
    import logging
    logging.getLogger(__name__).warning("Persistent logging module not available")
    persistent_logging_available = False

# Export all important items
__all__ = [
    'run_main_ui',
    'TrainingMonitor',
    'create_dashboard_preview',
    'create_icon',
    'handle_error',
    'show_error_dialog',
    'ErrorSeverity',
    'get_last_error',
    'get_error_history',
    'log_persistent_error',
    'export_error_logs',
    'get_log_directory_path',
    'open_log_directory',
    'create_log_locations_file',
    'error_handling_available',
    'persistent_logging_available'
]
