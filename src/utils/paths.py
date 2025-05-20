"""
Path Management Module

This module provides a simple interface for consistent path handling across
the entire codebase. It leverages the platform_utils module for the core
functionality.

Usage:
    from src.utils.paths import get_project_root, get_absolute_path, ensure_path_exists
    
    project_root = get_project_root()
    config_path = get_absolute_path("configs/config.json")
    ensure_path_exists("Logs/training")
"""

import os
import sys
import logging
from pathlib import Path
from typing import Union, Optional, List

# Try to import from platform_utils (preferred)
try:
    from src.utils.platform_utils import (
        PROJECT_ROOT, DATA_ROOT, normalize_path, get_roots
    )
    
    # Use the platform_utils versions
    def get_project_root() -> str:
        """
        Get the project root directory.
        
        Returns:
            str: Absolute path to the project root
        """
        return PROJECT_ROOT
    
    def get_data_root() -> str:
        """
        Get the data root directory.
        
        Returns:
            str: Absolute path to the data root (same as project root in development)
        """
        return DATA_ROOT
        
except ImportError:
    # Fallback implementation if platform_utils is not available
    logging.warning("platform_utils not available, using fallback path management")
    
    def get_project_root() -> str:
        """
        Get the project root directory.
        
        Returns:
            str: Absolute path to the project root
        """
        # Get the location of the current module
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up two directories from src/utils to get to project root
        project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
        return project_root
    
    def get_data_root() -> str:
        """
        Get the data root directory.
        
        Returns:
            str: Absolute path to the data root (same as project root in development)
        """
        return get_project_root()
    
    # Simple normalize_path implementation for fallback
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
        
        # Make absolute if relative
        if not os.path.isabs(path):
            path = os.path.join(get_project_root(), path)
        
        # Normalize the path
        return os.path.normpath(path)

def get_absolute_path(relative_path: str, relative_to: str = None) -> str:
    """
    Convert a relative path to an absolute path.
    
    Args:
        relative_path (str): Path relative to the project root or specified base
        relative_to (str, optional): Base directory to resolve from
                                    (defaults to project root)
    
    Returns:
        str: The absolute path
    """
    if relative_to is None:
        relative_to = get_project_root()
    
    # If already absolute, just normalize it
    if os.path.isabs(relative_path):
        return normalize_path(relative_path)
    
    # Join with base path and normalize
    return normalize_path(os.path.join(relative_to, relative_path))

def ensure_path_exists(path: str, is_dir: bool = True) -> str:
    """
    Ensure a path exists, creating directories as needed.
    
    Args:
        path (str): Path to ensure exists
        is_dir (bool): Whether the path is a directory (True) or file (False)
    
    Returns:
        str: The absolute path that was ensured
    """
    abs_path = get_absolute_path(path)
    
    if is_dir:
        # Create directory and any parent directories
        os.makedirs(abs_path, exist_ok=True)
    else:
        # Create parent directories for a file path
        dir_path = os.path.dirname(abs_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
    
    return abs_path

def get_src_path() -> str:
    """
    Get the path to the src directory.
    
    Returns:
        str: Absolute path to the src directory
    """
    return os.path.join(get_project_root(), "src")

def add_project_to_path() -> None:
    """
    Add the project root to sys.path if not already present.
    This ensures that imports work correctly.
    """
    project_root = get_project_root()
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

def get_common_paths() -> dict:
    """
    Get a dictionary of commonly used paths in the project.
    
    Returns:
        dict: Dictionary of path names to absolute paths
    """
    project_root = get_project_root()
    return {
        "project_root": project_root,
        "src": os.path.join(project_root, "src"),
        "configs": os.path.join(project_root, "configs"),
        "models": os.path.join(project_root, "Models"),
        "logs": os.path.join(project_root, "Logs"),
        "data": os.path.join(project_root, "data"),
        "presets": os.path.join(project_root, "presets"),
        "scripts": os.path.join(project_root, "Scripts"),
        "comparison_results": os.path.join(project_root, "comparison_results"),
        "training_script": os.path.join(project_root, "src", "training", "training.py"),
        "backtesting_script": os.path.join(project_root, "src", "training", "backtesting.py"),
        "ui_main": os.path.join(project_root, "src", "ui", "main.py"),
    } 