"""
Platform Utilities Module

This module provides platform-specific utilities for the BTC AI application,
including path handling, file checking, and executable detection.
"""

import os
import sys
import logging
import datetime
import glob
import json
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

# Set up logger
try:
    from src.utils.log_manager import get_logger
    logger = get_logger('platform_utils')
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('platform_utils')

# Define constants
# Dynamically determine PROJECT_ROOT and DATA_ROOT based on execution context
def get_roots():
    """Determine the project root and data root directories based on execution context."""
    if getattr(sys, 'frozen', False):
        # Running as executable
        project_root = os.path.dirname(sys.executable)
        # Data files are in the _internal directory relative to sys._MEIPASS
        # Fallback to project_root if _MEIPASS is not available for some reason
        data_root = getattr(sys, '_MEIPASS', project_root)
    else:
        # Running as script
        project_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        data_root = project_root  # Data files are relative to the project root
    return project_root, data_root

PROJECT_ROOT, DATA_ROOT = get_roots()
REQUIRED_DIRS = ["Models", "Logs", "configs", "data"] # These should be relative to PROJECT_ROOT
REQUIRED_FILES = ["version.json"] # These should be relative to DATA_ROOT


def normalize_path(path: str) -> str:
    """
    Normalize path for cross-platform compatibility.
    
    Args:
        path (str): The path to normalize
        
    Returns:
        str: Normalized path for the current platform
    """
    # Handle different path separators
    path = path.replace('\\', os.sep).replace('/', os.sep)
    
    # Make absolute paths if relative
    if not os.path.isabs(path):
        path = os.path.join(PROJECT_ROOT, path)
    
    # Normalize the path (resolve .. and such)
    normalized = os.path.normpath(path)
    
    # On Windows, ensure consistent drive letter casing
    if sys.platform == 'win32' and ':' in normalized:
        drive, rest = normalized.split(':', 1)
        normalized = drive.upper() + ':' + rest
    
    return normalized


def check_required_files() -> Tuple[bool, str]:
    """
    Check if all required files and directories exist.
    Adjusts paths based on whether running as an executable.
    
    Returns:
        tuple: (bool, str) - Success status and error message if any
    """
    # Directories are expected relative to the project root (where the .exe is or the script runs from)
    required_dirs_paths = [os.path.join(PROJECT_ROOT, d) for d in REQUIRED_DIRS]
    
    # Files are expected relative to the data root (_internal for exe, project root for script)
    required_files_paths = [os.path.join(DATA_ROOT, f) for f in REQUIRED_FILES]
    
    # Check directories
    missing_dirs = [d for d in required_dirs_paths if not os.path.isdir(d)]
    if missing_dirs:
        # Try creating them, maybe they weren't created by the build script correctly
        created_count = 0
        creation_errors = []
        for d in missing_dirs:
            try:
                os.makedirs(d, exist_ok=True)
                logger.info(f"Created missing required directory: {d}")
                created_count += 1
            except Exception as e:
                logger.error(f"Failed to create missing directory {d}: {e}")
                creation_errors.append(os.path.basename(d))
        
        if creation_errors:
            dirs_str = ', '.join(creation_errors)
            return False, f"Missing required directories (and failed to create them): {dirs_str}"
        elif created_count < len(missing_dirs):
             # If some were missing but created, re-check
             missing_dirs = [d for d in required_dirs_paths if not os.path.isdir(d)]
             if missing_dirs:
                 dirs_str = ', '.join([os.path.basename(d) for d in missing_dirs])
                 return False, f"Missing required directories: {dirs_str}"

    # Check files
    missing_files = [f for f in required_files_paths if not os.path.isfile(f)]
    if missing_files:
        files_str = ', '.join([os.path.basename(f) for f in missing_files])
        # Provide more context about where it looked
        looked_in = DATA_ROOT if is_executable() else PROJECT_ROOT
        logger.error(f"Looked for required files in: {looked_in}")
        return False, f"Missing required files: {files_str}"
    
    return True, ""


def is_executable() -> bool:
    """
    Check if running as a compiled executable (PyInstaller).
    
    Returns:
        bool: True if running as executable, False if running as script
    """
    return getattr(sys, 'frozen', False)


def initialize_executable_logging() -> bool:
    """
    Initialize enhanced logging configuration for executable version.
    
    This provides more detailed logging when running in executable mode,
    helping with debugging issues in the field.
    
    Returns:
        bool: True if executable mode, False otherwise
    """
    # Check if we're running as a PyInstaller executable
    executable_mode = is_executable()
    
    if executable_mode:
        # We're running as an executable, set up advanced logging
        logger.info("Running as executable, setting up enhanced logging")
        
        # Configure more detailed logging for executable mode
        try:
            # Create logs directory if it doesn't exist
            logs_dir = os.path.join(PROJECT_ROOT, "logs")
            os.makedirs(logs_dir, exist_ok=True)
            
            # Set up a rotating log file with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(logs_dir, f"app_{timestamp}.log")
            
            # Add file handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            
            # Add handler to root logger to catch all logs
            root_logger = logging.getLogger()
            root_logger.addHandler(file_handler)
            root_logger.setLevel(logging.DEBUG)
            
            # Log system information
            logger.info(f"System Platform: {sys.platform}")
            logger.info(f"Python Version: {sys.version}")
            logger.info(f"Executable Path: {sys.executable}")
            
            # Clean up old logs (keep last 10)
            try:
                log_files = sorted(glob.glob(os.path.join(logs_dir, "app_*.log")))
                if len(log_files) > 10:
                    for old_file in log_files[:-10]:
                        try:
                            os.remove(old_file)
                            logger.debug(f"Removed old log file: {old_file}")
                        except Exception as e:
                            logger.warning(f"Failed to remove old log file {old_file}: {e}")
            except Exception as e:
                logger.warning(f"Error cleaning up old log files: {e}")
                
        except Exception as e:
            logger.error(f"Error setting up enhanced logging: {e}")
    
    return executable_mode


def backup_config_files(backup_suffix: str = None) -> bool:
    """
    Create backups of important configuration files.
    
    Args:
        backup_suffix (str, optional): Suffix to append to backup files
                                      (defaults to timestamp)
    
    Returns:
        bool: True if backup was successful, False otherwise
    """
    if backup_suffix is None:
        backup_suffix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # Define configuration directories to backup
        config_dirs = [
            os.path.join(PROJECT_ROOT, "configs"),
        ]
        
        # Create backup directory if it doesn't exist
        backup_dir = os.path.join(PROJECT_ROOT, "backups", f"config_{backup_suffix}")
        os.makedirs(backup_dir, exist_ok=True)
        
        files_backed_up = 0
        
        # Backup each config file
        for config_dir in config_dirs:
            if os.path.isdir(config_dir):
                for file in os.listdir(config_dir):
                    if file.endswith('.json') or file.endswith('.yaml') or file.endswith('.yml'):
                        src = os.path.join(config_dir, file)
                        dst = os.path.join(backup_dir, file)
                        try:
                            import shutil
                            shutil.copy2(src, dst)
                            files_backed_up += 1
                            logger.debug(f"Backed up {src} to {dst}")
                        except Exception as e:
                            logger.warning(f"Failed to backup {src}: {e}")
        
        logger.info(f"Backed up {files_backed_up} configuration files to {backup_dir}")
        return files_backed_up > 0
    
    except Exception as e:
        logger.error(f"Error backing up configuration files: {e}")
        return False


def get_app_version() -> str:
    """
    Get the current application version from version.json.
    Adjusts path based on whether running as an executable.
    
    Returns:
        str: Version string, or "0.0.1" if not found
    """
    # version.json is a data file, look in DATA_ROOT
    version_file = os.path.join(DATA_ROOT, "version.json")
    logger.debug(f"Attempting to read version file from: {version_file}")
    try:
        if os.path.exists(version_file):
            with open(version_file, 'r') as f:
                version_data = json.load(f)
                if 'version' in version_data:
                    logger.info(f"Application version found: {version_data['version']}")
                    return version_data['version']
                else:
                    logger.warning("version.json found but does not contain a 'version' key.")
        else:
             logger.error(f"Version file not found at expected location: {version_file}")
             # Log directory contents for debugging
             try:
                 parent_dir_contents = os.listdir(os.path.dirname(version_file))
                 logger.error(f"Contents of {os.path.dirname(version_file)}: {parent_dir_contents}")
             except Exception as list_err:
                 logger.error(f"Could not list contents of {os.path.dirname(version_file)}: {list_err}")

    except Exception as e:
        logger.error(f"Error reading version file '{version_file}': {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    logger.warning("Returning default version '0.0.1'.")
    return "0.0.1"  # Default version if not found


# Run initialization if module is loaded
if __name__ != "__main__":
    # Log module initialization
    logger.debug("Platform utilities module initialized") 