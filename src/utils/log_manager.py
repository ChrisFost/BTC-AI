"""
Centralized logging module with rotation capabilities for the BTC-AI application.

This module provides:
1. A unified logging configuration with file rotation
2. Default handlers that can be used across the application
3. Easy setup for both console and file logging
4. Automatic daily log rotation with backup management
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import traceback
import time
import inspect
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

# Get the base directory (project root)
PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Define default log directory
DEFAULT_LOG_DIR = os.path.join(PROJECT_ROOT, "Logs")

# Define log levels as constants
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

# Default logger format
DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Default logger name
DEFAULT_LOGGER_NAME = 'btc_ai'

# Dictionary to keep track of configured loggers
_configured_loggers = {}

class LogManager:
    """Centralized log management class for BTC-AI application"""
    
    @staticmethod
    def setup_logger(
        logger_name: str = DEFAULT_LOGGER_NAME,
        log_file: Optional[str] = None,
        level: int = logging.INFO,
        format_str: str = DEFAULT_FORMAT,
        console: bool = True,
        file_logging: bool = True,
        max_size_mb: int = 10,
        backup_count: int = 5,
        use_timed_rotation: bool = True,
        when: str = 'midnight',
    ) -> logging.Logger:
        """
        Set up a logger with console and file handlers, with optional rotation
        
        Args:
            logger_name: Name of the logger
            log_file: Path to the log file (if None, uses logger_name.log in DEFAULT_LOG_DIR)
            level: Logging level
            format_str: Format string for logging
            console: Whether to add a console handler
            file_logging: Whether to add a file handler
            max_size_mb: Maximum size of the log file in MB (for RotatingFileHandler)
            backup_count: Number of backup files to keep
            use_timed_rotation: Whether to use TimedRotatingFileHandler instead of RotatingFileHandler
            when: Rotation interval for TimedRotatingFileHandler ('S', 'M', 'H', 'D', 'midnight')
            
        Returns:
            Configured logger
        """
        # Check if logger already exists to avoid duplicate handlers
        if logger_name in _configured_loggers:
            return _configured_loggers[logger_name]
        
        # Create logger
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter(format_str)
        
        # Add console handler if requested
        if console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # Add file handler if requested
        if file_logging:
            # Ensure log directory exists
            os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)
            
            # Set default log file path if not provided
            if log_file is None:
                log_file = os.path.join(DEFAULT_LOG_DIR, f"{logger_name}.log")
            
            # Create file handler with rotation
            if use_timed_rotation:
                file_handler = TimedRotatingFileHandler(
                    log_file, 
                    when=when,
                    backupCount=backup_count
                )
            else:
                file_handler = RotatingFileHandler(
                    log_file,
                    maxBytes=max_size_mb * 1024 * 1024,  # Convert MB to bytes
                    backupCount=backup_count
                )
            
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        # Store logger in dictionary
        _configured_loggers[logger_name] = logger
        
        return logger

    @staticmethod
    def get_logger(logger_name: str = DEFAULT_LOGGER_NAME) -> logging.Logger:
        """
        Get an existing logger or create a new one with default settings
        
        Args:
            logger_name: Name of the logger
            
        Returns:
            Logger instance
        """
        if logger_name in _configured_loggers:
            return _configured_loggers[logger_name]
        else:
            return LogManager.setup_logger(logger_name=logger_name)

    @staticmethod
    def get_all_logs() -> List[str]:
        """
        Get a list of all log files in the log directory
        
        Returns:
            List of log file paths
        """
        os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)
        log_files = []
        
        for file in os.listdir(DEFAULT_LOG_DIR):
            if file.endswith('.log'):
                log_files.append(os.path.join(DEFAULT_LOG_DIR, file))
                
        return log_files

    @staticmethod
    def read_log(log_file: str, lines: int = 100) -> List[str]:
        """
        Read the last n lines from a log file
        
        Args:
            log_file: Path to the log file
            lines: Number of lines to read from the end
            
        Returns:
            List of log lines
        """
        if not os.path.exists(log_file):
            return [f"Log file {log_file} does not exist"]
        
        try:
            with open(log_file, 'r') as f:
                all_lines = f.readlines()
                return all_lines[-lines:] if len(all_lines) > lines else all_lines
        except Exception as e:
            return [f"Error reading log file: {str(e)}"]

    @staticmethod
    def clear_log(log_file: str) -> bool:
        """
        Clear the contents of a log file
        
        Args:
            log_file: Path to the log file
            
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(log_file):
            return False
        
        try:
            with open(log_file, 'w') as f:
                f.write("")
            return True
        except:
            return False

    @staticmethod
    def archive_old_logs(days: int = 30) -> int:
        """
        Archive log files older than specified days to a zip file
        
        Args:
            days: Number of days to keep logs before archiving
            
        Returns:
            Number of archived files
        """
        import zipfile
        from datetime import datetime, timedelta
        
        archive_dir = os.path.join(DEFAULT_LOG_DIR, "archives")
        os.makedirs(archive_dir, exist_ok=True)
        
        cutoff_date = datetime.now() - timedelta(days=days)
        archive_filename = os.path.join(
            archive_dir, 
            f"logs_before_{cutoff_date.strftime('%Y%m%d')}.zip"
        )
        
        count = 0
        with zipfile.ZipFile(archive_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for log_file in LogManager.get_all_logs():
                file_time = datetime.fromtimestamp(os.path.getmtime(log_file))
                if file_time < cutoff_date:
                    zipf.write(log_file, os.path.basename(log_file))
                    count += 1
                    # Optionally delete the file after archiving
                    os.remove(log_file)
        
        return count

    @staticmethod
    def log_exception(
        logger: Optional[logging.Logger] = None,
        level: int = logging.ERROR,
        exc_info: bool = True,
        stack_info: bool = False
    ) -> None:
        """
        Log an exception with full traceback
        
        Args:
            logger: Logger to use (if None, uses default logger)
            level: Logging level
            exc_info: Whether to include exception info
            stack_info: Whether to include stack info
        """
        if logger is None:
            logger = LogManager.get_logger()
        
        # Get the current exception info
        exc_type, exc_value, exc_tb = sys.exc_info()
        
        # Get the frame where the exception occurred
        frame = inspect.currentframe().f_back
        func_name = frame.f_code.co_name
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno
        
        # Format a detailed error message
        error_msg = f"Exception in {func_name} ({os.path.basename(filename)}:{lineno}): {exc_value}"
        
        # Log the exception
        logger.log(level, error_msg, exc_info=exc_info, stack_info=stack_info)

# Create a default logger instance for direct import
default_logger = LogManager.setup_logger()

# Helper functions for direct import
def get_logger(name=DEFAULT_LOGGER_NAME):
    return LogManager.get_logger(name)

def log_exception(logger=None, level=logging.ERROR):
    LogManager.log_exception(logger, level)

# Set up uncaught exception handler to log all unhandled exceptions
def handle_uncaught_exception(exctype, value, tb):
    if exctype == KeyboardInterrupt:
        # Don't log keyboard interrupts
        sys.__excepthook__(exctype, value, tb)
        return
        
    # Format the traceback
    traceback_details = ''.join(traceback.format_exception(exctype, value, tb))
    error_msg = f"Uncaught {exctype.__name__}: {value}\n{traceback_details}"
    
    # Log the uncaught exception
    default_logger.critical(error_msg)
    
    # Call the default exception handler
    sys.__excepthook__(exctype, value, tb)

# Install the exception handler
sys.excepthook = handle_uncaught_exception 