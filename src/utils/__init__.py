"""
Utilities Package

This package provides utility functions and classes for the BTC-AI system.
"""

# Import and re-export utilities as needed
try:
    from .utils import log, optimize_memory
    core_utils_available = True
except ImportError as e:
    import logging
    logging.getLogger(__name__).warning(f"Core utils module (.utils) not available: {e}")
    core_utils_available = False

# Try to import platform utils
try:
    from .platform_utils import (
        normalize_path,
        check_required_files,
        is_executable,
        initialize_executable_logging,
        backup_config_files,
        get_app_version
    )
    platform_utils_available = True
except ImportError as e:
    import logging
    logging.getLogger(__name__).warning(f"Platform utilities module (.platform_utils) not available: {e}")
    platform_utils_available = False

# Try to import persistent logger
try:
    from .persistent_logger import (
        log_persistent_error,
        export_error_logs,
        get_log_directory_path,
        open_log_directory,
        create_log_locations_file,
        merge_log_files
    )
    persistent_logging_available = True
except ImportError as e:
    import logging
    logging.getLogger(__name__).warning(f"Persistent logger (.persistent_logger) not available: {e}")
    persistent_logging_available = False

# Try to import log manager
try:
    from .log_manager import LogManager, log_exception
    log_manager_available = True
except ImportError as e:
    import logging
    logging.getLogger(__name__).warning(f"Log manager (.log_manager) not available: {e}")
    log_manager_available = False

# Export important items
__all__ = []

if core_utils_available:
    __all__.extend([
        'log',
        'optimize_memory',
    ])

if platform_utils_available:
    __all__.extend([
        'normalize_path',
        'check_required_files',
        'is_executable',
        'initialize_executable_logging',
        'backup_config_files',
        'get_app_version',
        'platform_utils_available'
    ])

if persistent_logging_available:
    __all__.extend([
        'log_persistent_error',
        'export_error_logs',
        'get_log_directory_path',
        'open_log_directory',
        'create_log_locations_file',
        'merge_log_files',
        'persistent_logging_available',
    ])

if log_manager_available:
    __all__.extend([
        'LogManager',
        'log_exception',
        'log_manager_available'
    ])
