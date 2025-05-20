import logging
import os
import importlib
from typing import Dict, Any, List, Tuple, Optional, Callable

# Try to import preset management
try:
    from src.ui.preset_manager import update_preset_performance, get_backtest_metrics_from_results
    preset_manager_available = True
except ImportError:
    preset_manager_available = False
    logging.warning("Preset manager not available, performance tracking will be disabled")

# Try to import preset handlers
try:
    from src.ui.preset_handlers import get_current_preset_id
    preset_handlers_available = True
except ImportError:
    preset_handlers_available = False
    logging.warning("Preset handlers not available, preset tracking will be disabled")

# Try to import logging utilities
try:
    from src.utils.log_manager import LogManager
    logger = LogManager.get_logger("backtesting_integration")
except ImportError:
    # Fallback if imports fail
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("backtesting_integration")

def import_backtesting_module():
    """
    Import the backtesting module dynamically
    
    Returns:
        Tuple of (module, success_flag)
    """
    try:
        # Import from src.training.backtesting
        backtesting_module = importlib.import_module("src.training.backtesting")
        
        # Verify that the module has the necessary functions
        required_functions = ['run_backtest', 'run_preset_comparison']
        for func in required_functions:
            if not hasattr(backtesting_module, func):
                logger.error(f"Backtesting module missing required function: {func}")
                return None, False
                
        logger.info("Successfully imported backtesting module with all required functions")
        return backtesting_module, True
    except ImportError as e:
        logger.error(f"Failed to import backtesting module: {str(e)}")
        logger.error("Make sure the backtesting module is installed correctly")
        return None, False
    except Exception as e:
        logger.error(f"Unexpected error importing backtesting module: {str(e)}")
        return None, False

def run_backtest_with_preset_tracking(df, agent, config, episodes=1, log_callback=None):
    """
    Run backtest with preset performance tracking
    
    Args:
        df: DataFrame with market data
        agent: Trained agent for decision making
        config: Configuration parameters
        episodes: Number of episodes to run
        log_callback: Callback for logging
        
    Returns:
        Results from the backtest
    """
    # Import backtesting module
    backtesting_module, success = import_backtesting_module()
    if not success:
        return None
    
    # Check if preset tracking is available
    current_preset_id = None
    if preset_handlers_available:
        current_preset_id = get_current_preset_id()
    
    # Run backtest
    results = backtesting_module.run_backtest(df, agent, config, episodes, log_callback)
    
    # Track preset performance if available
    if preset_manager_available and current_preset_id:
        # Get metrics from results
        metrics = get_backtest_metrics_from_results(results)
        
        # Update preset performance
        success = update_preset_performance(current_preset_id, metrics)
        if success:
            logger.info(f"Updated performance history for preset: {current_preset_id}")
        else:
            logger.error(f"Failed to update performance history for preset: {current_preset_id}")
    
    return results

def run_preset_comparison_with_tracking(df, preset_config, user_config, log_callback=None):
    """
    Run comparison between preset and user configs with performance tracking
    
    Args:
        df: DataFrame with market data
        preset_config: Preset configuration
        user_config: User configuration
        log_callback: Callback for logging
        
    Returns:
        Results from the comparison
    """
    # Import backtesting module
    backtesting_module, success = import_backtesting_module()
    if not success:
        return None
    
    # Check if preset tracking is available
    current_preset_id = None
    if preset_handlers_available:
        current_preset_id = get_current_preset_id()
    
    # Run comparison
    results = backtesting_module.run_preset_comparison(df, preset_config, user_config, log_callback)
    
    # Track preset performance if available
    if preset_manager_available and current_preset_id and results:
        # Results should be a tuple of (preset_avg, user_avg)
        if isinstance(results, tuple) and len(results) >= 1:
            preset_metrics = results[0]
            
            # Update preset performance
            success = update_preset_performance(current_preset_id, preset_metrics)
            if success:
                logger.info(f"Updated performance history from comparison for preset: {current_preset_id}")
            else:
                logger.error(f"Failed to update comparison performance history for preset: {current_preset_id}")
    
    return results

# Hook function for main event loop
def register_backtest_event_handlers(window):
    """
    Register event handlers for backtest-related events
    
    Args:
        window: The PySimpleGUI window
    """
    # No specific registration needed here, but useful for modular design
    pass

# Monkey patch backtest functions to add preset tracking
def patch_backtesting_module():
    """
    Monkey patch backtesting module functions to add preset tracking
    """
    # Import backtesting module
    backtesting_module, success = import_backtesting_module()
    if not success:
        return
    
    # Store original functions
    original_run_backtest = backtesting_module.run_backtest
    original_run_preset_comparison = backtesting_module.run_preset_comparison
    
    # Define patched functions
    def patched_run_backtest(df, agent, config, episodes=1, log_callback=None):
        """Patched version of run_backtest that adds preset tracking"""
        results = original_run_backtest(df, agent, config, episodes, log_callback)
        
        # Check if preset tracking is available
        if preset_handlers_available and preset_manager_available:
            current_preset_id = get_current_preset_id()
            if current_preset_id:
                metrics = get_backtest_metrics_from_results(results)
                update_preset_performance(current_preset_id, metrics)
        
        return results
    
    def patched_run_preset_comparison(df, preset_config, user_config, log_callback=None):
        """Patched version of run_preset_comparison that adds preset tracking"""
        results = original_run_preset_comparison(df, preset_config, user_config, log_callback)
        
        # Check if preset tracking is available
        if preset_handlers_available and preset_manager_available:
            current_preset_id = get_current_preset_id()
            if current_preset_id and results and isinstance(results, tuple) and len(results) >= 1:
                preset_metrics = results[0]
                update_preset_performance(current_preset_id, preset_metrics)
        
        return results
    
    # Apply monkey patching
    backtesting_module.run_backtest = patched_run_backtest
    backtesting_module.run_preset_comparison = patched_run_preset_comparison
    
    logger.info("Successfully patched backtesting module for preset tracking")

# Initialize the integration
def initialize_backtesting_integration():
    """
    Initialize the backtesting integration
    
    Returns:
        bool: True if successful
    """
    if preset_manager_available and preset_handlers_available:
        # Patch backtesting module
        patch_backtesting_module()
        return True
    else:
        logger.warning("Preset management or handlers not available, backtesting integration disabled")
        return False 